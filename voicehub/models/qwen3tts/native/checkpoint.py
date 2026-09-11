"""Strict streaming Safetensors loading/export for native Qwen3-TTS."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from voicehub.checkpointing import SafeTensorReader, save_safetensors
from voicehub.checkpointing.errors import CheckpointCompatibilityError
from voicehub.models.qwen3tts.native.codec import Qwen3TTSSpeechDecoder, materialize_qwen3_tts_decoder_buffers
from voicehub.models.qwen3tts.native.encoder import Qwen3TTSSpeechEncoder, materialize_qwen3_tts_encoder_buffers
from voicehub.models.qwen3tts.native.metadata import QWEN3_TTS_CHECKPOINTS, QWEN3_TTS_SPEECH_TOKENIZER
from voicehub.models.qwen3tts.native.modeling import Qwen3TTSForConditionalGeneration, materialize_qwen3_tts_buffers

NATIVE_QWEN3_TTS_FORMAT = "voicehub-qwen3-tts-v1"


@dataclass(frozen=True, slots=True)
class Qwen3TTSCheckpointReport:
    path: Path
    tensor_count: int
    parameter_count: int
    header_fingerprint: str


def _fingerprint(inventory: Mapping[str, tuple[str, tuple[int, ...]]], ) -> str:
    rows = [
        f"{name}|{dtype}|{'x'.join(str(item) for item in shape)}"
        for name, (dtype, shape) in sorted(inventory.items())
    ]
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def inspect_qwen3_tts_checkpoint(
    path: str | Path,
    *,
    prefix: str = "",
) -> Qwen3TTSCheckpointReport:
    source = Path(path).expanduser().resolve()
    with SafeTensorReader(source) as reader:
        names = tuple(name for name in reader.keys() if not prefix or name.startswith(prefix))
        inventory = {name: (reader.record(name).dtype, reader.tensor_shape(name)) for name in names}
        parameters = sum(reader.record(name).number_of_elements for name in names)
    return Qwen3TTSCheckpointReport(
        path=source,
        tensor_count=len(inventory),
        parameter_count=parameters,
        header_fingerprint=_fingerprint(inventory),
    )


def _expected_shapes(module: nn.Module) -> dict[str, tuple[int, ...]]:
    return {name: tuple(value.shape) for name, value in module.state_dict(keep_vars=True).items()}


def _validate_namespace(
    module: nn.Module,
    reader: SafeTensorReader,
    *,
    source_prefix: str,
    permit_other_prefixes: bool,
) -> tuple[str, ...]:
    expected = _expected_shapes(module)
    source_names = {name[len(source_prefix):] for name in reader.keys() if name.startswith(source_prefix)}
    all_names = set(reader.keys())
    selected_full_names = {source_prefix + name for name in source_names}
    missing = sorted(set(expected) - source_names)
    unexpected = sorted(source_names - set(expected))
    foreign = sorted(all_names - selected_full_names)
    mismatched = sorted((
        name,
        reader.tensor_shape(source_prefix + name),
        expected[name],
    ) for name in set(expected) & source_names if reader.tensor_shape(source_prefix + name) != expected[name])
    if missing or unexpected or mismatched or (foreign and not permit_other_prefixes):
        raise CheckpointCompatibilityError(
            "Qwen3-TTS checkpoint is incompatible: "
            f"missing={missing!r}, unexpected={unexpected!r}, "
            f"shape_mismatches={mismatched!r}, "
            f"foreign={foreign[:8]!r}.")
    return tuple(sorted(expected))


def _assign(
    module: nn.Module,
    reader: SafeTensorReader,
    names: tuple[str, ...],
    *,
    source_prefix: str,
    device: str | torch.device,
    dtype: torch.dtype | None,
) -> None:
    with torch.no_grad():
        for name in names:
            value = reader.get_tensor(source_prefix + name)
            target_dtype = (dtype if dtype is not None and value.is_floating_point() else value.dtype)
            module.load_state_dict(
                {name: value.to(
                    device=device,
                    dtype=target_dtype,
                )},
                strict=False,
                assign=True,
            )
    remaining = [name for name, value in module.state_dict().items() if value.device.type == "meta"]
    if remaining:
        raise CheckpointCompatibilityError(
            "Qwen3-TTS streaming assignment left meta tensors: " + ", ".join(remaining[:8]))


def load_qwen3_tts_model_checkpoint(
    model: Qwen3TTSForConditionalGeneration,
    path: str | Path,
    *,
    device: str | torch.device,
    dtype: torch.dtype | None,
    source: str | None = None,
    revision: str | None = None,
) -> Qwen3TTSCheckpointReport:
    """Validate the entire model header before assigning any tensor."""
    report = inspect_qwen3_tts_checkpoint(path)
    known = QWEN3_TTS_CHECKPOINTS.get(source or "")
    if known is not None and revision == known["revision"]:
        expected = (
            known["tensors"],
            known["parameters"],
            known["header_fingerprint"],
        )
        actual = (
            report.tensor_count,
            report.parameter_count,
            report.header_fingerprint,
        )
        if actual != expected:
            raise CheckpointCompatibilityError(
                "Published Qwen3-TTS model inventory verification failed: "
                f"found={actual!r}, expected={expected!r}.")
    with SafeTensorReader(report.path) as reader:
        names = _validate_namespace(
            model,
            reader,
            source_prefix="",
            permit_other_prefixes=False,
        )
        _assign(
            model,
            reader,
            names,
            source_prefix="",
            device=device,
            dtype=dtype,
        )
    materialize_qwen3_tts_buffers(model, device=device)
    return report


def load_qwen3_tts_decoder_checkpoint(
    decoder: Qwen3TTSSpeechDecoder,
    path: str | Path,
    *,
    device: str | torch.device,
    dtype: torch.dtype | None,
    verify_official: bool,
) -> Qwen3TTSCheckpointReport:
    """Load only the audited decoder namespace from the tokenizer artifact."""
    full_report = inspect_qwen3_tts_checkpoint(path)
    decoder_report = inspect_qwen3_tts_checkpoint(path, prefix="decoder.")
    if verify_official:
        expected_full = (
            QWEN3_TTS_SPEECH_TOKENIZER["tensors"],
            QWEN3_TTS_SPEECH_TOKENIZER["parameters"],
            QWEN3_TTS_SPEECH_TOKENIZER["header_fingerprint"],
        )
        actual_full = (
            full_report.tensor_count,
            full_report.parameter_count,
            full_report.header_fingerprint,
        )
        if actual_full != expected_full:
            raise CheckpointCompatibilityError(
                "Published Qwen3-TTS speech-tokenizer inventory verification "
                f"failed: found={actual_full!r}, expected={expected_full!r}.")
        expected_decoder = (
            QWEN3_TTS_SPEECH_TOKENIZER["decoder_tensors"],
            QWEN3_TTS_SPEECH_TOKENIZER["decoder_parameters"],
            QWEN3_TTS_SPEECH_TOKENIZER["decoder_header_fingerprint"],
        )
        actual_decoder = (
            decoder_report.tensor_count,
            decoder_report.parameter_count,
            decoder_report.header_fingerprint,
        )
        if actual_decoder != expected_decoder:
            raise CheckpointCompatibilityError(
                "Qwen3-TTS decoder inventory verification failed: "
                f"found={actual_decoder!r}, expected={expected_decoder!r}.")
    with SafeTensorReader(full_report.path) as reader:
        names = _validate_namespace(
            decoder,
            reader,
            source_prefix="decoder.",
            permit_other_prefixes=True,
        )
        _assign(
            decoder,
            reader,
            names,
            source_prefix="decoder.",
            device=device,
            dtype=dtype,
        )
    materialize_qwen3_tts_decoder_buffers(decoder, device=device)
    return decoder_report


def load_qwen3_tts_encoder_checkpoint(
    encoder: Qwen3TTSSpeechEncoder,
    path: str | Path,
    *,
    device: str | torch.device,
    dtype: torch.dtype | None,
    verify_official: bool,
) -> Qwen3TTSCheckpointReport:
    """Load the exact 225-tensor encoder namespace, failing closed."""
    full_report = inspect_qwen3_tts_checkpoint(path)
    encoder_report = inspect_qwen3_tts_checkpoint(path, prefix="encoder.")
    if verify_official:
        expected_full = (
            QWEN3_TTS_SPEECH_TOKENIZER["tensors"],
            QWEN3_TTS_SPEECH_TOKENIZER["parameters"],
            QWEN3_TTS_SPEECH_TOKENIZER["header_fingerprint"],
        )
        actual_full = (
            full_report.tensor_count,
            full_report.parameter_count,
            full_report.header_fingerprint,
        )
        if actual_full != expected_full:
            raise CheckpointCompatibilityError(
                "Published Qwen3-TTS speech-tokenizer inventory verification "
                f"failed: found={actual_full!r}, expected={expected_full!r}.")
        expected_encoder = (
            QWEN3_TTS_SPEECH_TOKENIZER["encoder_tensors"],
            QWEN3_TTS_SPEECH_TOKENIZER["encoder_parameters"],
            QWEN3_TTS_SPEECH_TOKENIZER["encoder_header_fingerprint"],
        )
        actual_encoder = (
            encoder_report.tensor_count,
            encoder_report.parameter_count,
            encoder_report.header_fingerprint,
        )
        if actual_encoder != expected_encoder:
            raise CheckpointCompatibilityError(
                "Qwen3-TTS encoder inventory verification failed: "
                f"found={actual_encoder!r}, expected={expected_encoder!r}.")
    with SafeTensorReader(full_report.path) as reader:
        names = _validate_namespace(
            encoder,
            reader,
            source_prefix="encoder.",
            permit_other_prefixes=True,
        )
        _assign(
            encoder,
            reader,
            names,
            source_prefix="encoder.",
            device=device,
            dtype=dtype,
        )
    materialize_qwen3_tts_encoder_buffers(
        encoder,
        device=device,
    )
    return encoder_report


def export_qwen3_tts_model(
    model: Qwen3TTSForConditionalGeneration,
    path: str | Path,
    *,
    state_override: Mapping[str, Tensor] | None = None,
) -> Path:
    state = model.state_dict() if state_override is None else state_override
    expected = set(model.state_dict())
    if set(state) != expected:
        raise ValueError(
            "Qwen3-TTS export state is incomplete: "
            f"missing={sorted(expected - set(state))!r}, "
            f"unexpected={sorted(set(state) - expected)!r}.")
    return save_safetensors(
        {
            name: value.detach()
            for name, value in state.items()
        },
        path,
        metadata={
            "format": NATIVE_QWEN3_TTS_FORMAT
        },
    ).resolve()


def export_qwen3_tts_decoder(
    decoder: Qwen3TTSSpeechDecoder,
    path: str | Path,
) -> Path:
    return save_safetensors(
        {
            "decoder." + name: value.detach()
            for name, value in decoder.state_dict().items()
        },
        path,
        metadata={
            "format": NATIVE_QWEN3_TTS_FORMAT,
            "component": "speech-decoder",
        },
    ).resolve()


def export_qwen3_tts_speech_tokenizer(
    encoder: Qwen3TTSSpeechEncoder,
    decoder: Qwen3TTSSpeechDecoder,
    path: str | Path,
) -> Path:
    """Export both exact tokenizer namespaces in the official layout."""
    state = {"encoder." + name: value.detach() for name, value in encoder.state_dict().items()}
    state.update({"decoder." + name: value.detach() for name, value in decoder.state_dict().items()})
    return save_safetensors(
        state,
        path,
        metadata={
            "format": NATIVE_QWEN3_TTS_FORMAT,
            "component": "speech-tokenizer",
        },
    ).resolve()


__all__ = [
    "NATIVE_QWEN3_TTS_FORMAT",
    "Qwen3TTSCheckpointReport",
    "export_qwen3_tts_decoder",
    "export_qwen3_tts_model",
    "export_qwen3_tts_speech_tokenizer",
    "inspect_qwen3_tts_checkpoint",
    "load_qwen3_tts_decoder_checkpoint",
    "load_qwen3_tts_encoder_checkpoint",
    "load_qwen3_tts_model_checkpoint",
]
