"""Audited offline conversion for the ONNX-only CosyVoice 3 tokenizer."""

from __future__ import annotations

import hashlib
from importlib import import_module
from pathlib import Path

import torch

from voicehub.checkpointing.errors import CheckpointCompatibilityError, CheckpointIntegrityError
from voicehub.models.cosyvoice.native.checkpoint import (
    _audited_speech_tokenizer_state,
    export_cosyvoice_checkpoint,
    inspect_cosyvoice_checkpoint,
)
from voicehub.models.cosyvoice.native.metadata import COSYVOICE3_SPEECH_TOKENIZER_FILE
from voicehub.models.cosyvoice.native.speech_tokenizer import CosyVoiceSpeechTokenizer


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def convert_audited_cosyvoice_speech_tokenizer(
    source: str | Path,
    destination: str | Path,
) -> Path:
    """Convert the immutable ONNX parameter store to native Safetensors.

    This module is an explicit offline tool and is not imported by any
    native inference or training path. The source's filename, bytes,
    SHA-256, opset, graph I/O, node count, complete initializer
    inventory, mapped keys, and shapes must match the published audit.
    """
    expected = COSYVOICE3_SPEECH_TOKENIZER_FILE
    source = Path(source).expanduser().resolve()
    if source.name != expected["filename"]:
        raise CheckpointIntegrityError(
            f"Expected audited file {expected['filename']!r}, "
            f"found {source.name!r}.")
    if not source.is_file():
        raise FileNotFoundError(f"Speech-tokenizer ONNX was not found: {source}.")
    if source.stat().st_size != expected["size"]:
        raise CheckpointIntegrityError("Speech-tokenizer ONNX size differs from the audited artifact.")
    if _sha256(source) != expected["sha256"]:
        raise CheckpointIntegrityError("Speech-tokenizer ONNX SHA-256 differs from the audited artifact.")

    try:
        onnx = import_module("onnx")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "The explicit conversion tool requires the optional `onnx` "
            "parser. ONNX Runtime is neither required nor used.") from error
    state = _audited_speech_tokenizer_state(
        source,
        onnx=onnx,
    )
    with torch.device("meta"):
        model = CosyVoiceSpeechTokenizer()
    model_state = model.state_dict()
    if set(state) != set(model_state):
        raise CheckpointCompatibilityError("Mapped speech-tokenizer keys do not match the native graph.")
    mismatches = [(
        name,
        tuple(state[name].shape),
        tuple(model_state[name].shape),
    ) for name in model_state if tuple(state[name].shape) != tuple(model_state[name].shape)]
    if mismatches:
        raise CheckpointCompatibilityError(
            "Mapped speech-tokenizer shapes differ from the native graph: "
            f"{mismatches[:12]!r}.")
    output = export_cosyvoice_checkpoint(
        model,
        destination,
        component="speech_tokenizer",
        state_override=state,
    )
    report = inspect_cosyvoice_checkpoint(
        output,
        component="speech_tokenizer",
    )
    actual = (
        report.tensor_count,
        report.parameter_count,
        report.header_fingerprint,
    )
    required = (
        expected["initializer_count"],
        expected["parameter_count"],
        expected["native_header_fingerprint"],
    )
    if actual != required:
        output.unlink(missing_ok=True)
        raise CheckpointCompatibilityError("Converted speech-tokenizer inventory does not match the audit.")
    return output


__all__ = ["convert_audited_cosyvoice_speech_tokenizer"]
