"""Strict conversion and loading for the audited SenseVoiceSmall checkpoint."""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicehub.checkpointing import CheckpointAdapter, CopyTensor, SafeTensorReader, TensorPlan, save_safetensors
from voicehub.hub import write_json_file
from voicehub.models.asr_funasr.native.configuration import SenseVoiceSmallConfig
from voicehub.models.asr_funasr.native.metadata import (
    SENSEVOICE_CHECKPOINT_FILENAME,
    SENSEVOICE_CHECKPOINT_SHA256,
    SENSEVOICE_CHECKPOINT_SIZE,
    SENSEVOICE_CMVN_FILENAME,
    SENSEVOICE_CMVN_SHA256,
    SENSEVOICE_CMVN_SIZE,
    SENSEVOICE_MODEL_LICENSE,
    SENSEVOICE_REVISION,
    SENSEVOICE_STATE_VALUES,
    SENSEVOICE_TENSOR_COUNT,
    SENSEVOICE_TENSOR_FINGERPRINT,
    SENSEVOICE_TOKENIZER_FILENAME,
    SENSEVOICE_TOKENIZER_SHA256,
    SENSEVOICE_TOKENIZER_SIZE,
    SENSEVOICE_UPSTREAM_CONFIG_FILENAME,
    SENSEVOICE_UPSTREAM_CONFIG_SHA256,
    SENSEVOICE_UPSTREAM_CONFIG_SIZE,
)

NATIVE_SENSEVOICE_FORMAT = "voicehub-sensevoice-small-v1"
NATIVE_SENSEVOICE_FILENAME = "model.safetensors"
NATIVE_SENSEVOICE_TOKENIZER = "tokenizer.model"
NATIVE_SENSEVOICE_CMVN = "am.mvn"
_SOURCE_FILES = {
    SENSEVOICE_CHECKPOINT_FILENAME: (
        SENSEVOICE_CHECKPOINT_SHA256,
        SENSEVOICE_CHECKPOINT_SIZE,
    ),
    SENSEVOICE_TOKENIZER_FILENAME: (
        SENSEVOICE_TOKENIZER_SHA256,
        SENSEVOICE_TOKENIZER_SIZE,
    ),
    SENSEVOICE_CMVN_FILENAME: (
        SENSEVOICE_CMVN_SHA256,
        SENSEVOICE_CMVN_SIZE,
    ),
    SENSEVOICE_UPSTREAM_CONFIG_FILENAME: (
        SENSEVOICE_UPSTREAM_CONFIG_SHA256,
        SENSEVOICE_UPSTREAM_CONFIG_SIZE,
    ),
}
SenseVoiceConfigLike = SenseVoiceSmallConfig | Mapping[str, Any]


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_inventory_fingerprint(tensors: Mapping[str, Any]) -> str:
    rows = []
    for name, tensor in sorted(tensors.items()):
        dtype = {
            "torch.bfloat16": "BF16",
            "torch.bool": "BOOL",
            "torch.float16": "F16",
            "torch.float32": "F32",
            "torch.float64": "F64",
            "torch.int16": "I16",
            "torch.int32": "I32",
            "torch.int64": "I64",
            "torch.int8": "I8",
            "torch.uint8": "U8",
        }.get(str(tensor.dtype), str(tensor.dtype))
        shape = "x".join(str(item) for item in tensor.shape)
        rows.append(f"{name}|{dtype}|{shape}")
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def native_sensevoice_tensor_shapes(config: SenseVoiceConfigLike | None = None) -> dict[str, tuple[int, ...]]:
    import torch

    from voicehub.models.asr_funasr.native.modeling import SenseVoiceSmallForCTC

    with torch.device("meta"):
        model = SenseVoiceSmallForCTC(SenseVoiceSmallConfig.coerce(config))
    return {name: tuple(tensor.shape) for name, tensor in model.state_dict().items()}


class SenseVoiceSafeTensorsCheckpointAdapter(CheckpointAdapter):
    """Identity-map a validated native SenseVoice Safetensors artifact."""

    architecture_id = "sensevoice-small"
    adapter_id = "voicehub-sensevoice-small-safetensors"
    adapter_version = "1"

    def probe(
        self,
        files: tuple[Path, ...],
        config: Mapping[str, Any],
    ) -> bool:
        try:
            SenseVoiceSmallConfig.from_dict(config)
        except (TypeError, ValueError):
            return False
        return any(path.suffix == ".safetensors" for path in files)

    def tensor_plan(self, config: Mapping[str, Any]) -> TensorPlan:
        shapes = native_sensevoice_tensor_shapes(config)
        return TensorPlan(rules=tuple(CopyTensor(name, name) for name in sorted(shapes)))


def _verify_source_directory(root: Path) -> dict[str, Path]:
    files = {}
    for name, (expected_sha, expected_size) in _SOURCE_FILES.items():
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"SenseVoice source directory is missing {name!r}: {root}.")
        if path.stat().st_size != expected_size:
            raise ValueError(f"SenseVoice source file {name!r} has an unexpected size.")
        actual_sha = file_sha256(path)
        if actual_sha != expected_sha:
            raise ValueError(f"SenseVoice source file {name!r} digest mismatch: "
                             f"{actual_sha}.")
        files[name] = path
    return files


def _load_restricted_state(
    checkpoint: Path,
    *,
    trust_pickle_checkpoint: bool,
) -> Mapping[str, Any]:
    if trust_pickle_checkpoint is not True:
        raise ValueError(
            "The official SenseVoiceSmall checkpoint uses Python's pickle "
            "container. Review the pinned digest and model license, then pass "
            "`trust_pickle_checkpoint=True` for one-time conversion. Native "
            "inference and later reloads use Safetensors only.")
    import torch

    try:
        state = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=True,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError("Could not read the restricted SenseVoice tensor state.") from error
    if not isinstance(state, Mapping) or not state:
        raise TypeError("SenseVoice checkpoint must be a non-empty tensor mapping.")
    if any(not isinstance(name, str) or not isinstance(tensor, torch.Tensor)
           for name, tensor in state.items()):
        raise TypeError("SenseVoice checkpoint must map string names to tensors only.")
    return state


def _validate_released_state(state: Mapping[str, Any]) -> str:
    expected_shapes = native_sensevoice_tensor_shapes()
    expected_names = set(expected_shapes)
    source_names = set(state)
    if source_names != expected_names:
        raise ValueError(
            "SenseVoice tensor namespace does not match the audited graph "
            f"(missing={sorted(expected_names - source_names)}, "
            f"unexpected={sorted(source_names - expected_names)}).")
    mismatches = {
        name: (tuple(state[name].shape), expected_shapes[name])
        for name in expected_names if tuple(state[name].shape) != expected_shapes[name]
    }
    if mismatches:
        raise ValueError(f"SenseVoice tensor shape mismatch: {mismatches}.")
    if len(state) != SENSEVOICE_TENSOR_COUNT:
        raise ValueError("SenseVoice tensor count mismatch.")
    if (sum(tensor.numel() for tensor in state.values()) != SENSEVOICE_STATE_VALUES):
        raise ValueError("SenseVoice stored-value count mismatch.")
    fingerprint = tensor_inventory_fingerprint(state)
    if fingerprint != SENSEVOICE_TENSOR_FINGERPRINT:
        raise ValueError("SenseVoice tensor inventory fingerprint mismatch.")
    return fingerprint


def convert_sensevoice_small_checkpoint(
    source: str | Path,
    destination: str | Path,
    *,
    trust_pickle_checkpoint: bool = False,
) -> Path:
    """Convert only the exact hash-pinned public release to Safetensors."""
    root = Path(source).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(
            "SenseVoice conversion requires the complete source directory "
            "containing model.pt, tokenizer, CMVN, and config.yaml.")
    files = _verify_source_directory(root)
    state = _load_restricted_state(
        files[SENSEVOICE_CHECKPOINT_FILENAME],
        trust_pickle_checkpoint=trust_pickle_checkpoint,
    )
    fingerprint = _validate_released_state(state)
    output = Path(destination).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
            prefix="voicehub-sensevoice-convert-",
            dir=output.parent,
    ) as temporary:
        staging = Path(temporary)
        save_safetensors(
            state,
            staging / NATIVE_SENSEVOICE_FILENAME,
            metadata={
                "architecture": "sensevoice-small",
                "format": NATIVE_SENSEVOICE_FORMAT,
                "model_license": SENSEVOICE_MODEL_LICENSE,
                "source_revision": SENSEVOICE_REVISION,
                "tensor_fingerprint": fingerprint,
            },
        )
        shutil.copy2(
            files[SENSEVOICE_TOKENIZER_FILENAME],
            staging / NATIVE_SENSEVOICE_TOKENIZER,
        )
        shutil.copy2(
            files[SENSEVOICE_CMVN_FILENAME],
            staging / NATIVE_SENSEVOICE_CMVN,
        )
        values = SenseVoiceSmallConfig().to_dict()
        values.update({
            'architectures': [
                "FunASRForSpeechRecognition",
                "SenseVoiceSmallForCTC",
            ],
            "checkpoint_format": NATIVE_SENSEVOICE_FORMAT,
            "model_type": "asr_funasr",
            "source_artifact_revision": SENSEVOICE_REVISION,
            "source_model_name": "SenseVoiceSmall",
            "voicehub_provider": "asr_funasr",
        })
        write_json_file(staging / "config.json", values)
        for name in (
                NATIVE_SENSEVOICE_FILENAME,
                NATIVE_SENSEVOICE_TOKENIZER,
                NATIVE_SENSEVOICE_CMVN,
                "config.json",
        ):
            (staging / name).replace(output / name)
    return output


def load_native_sensevoice_model(
    checkpoint: str | Path,
    config: SenseVoiceSmallConfig | Mapping[str, Any],
    *,
    device: str = "cpu",
    dtype: Any = None,
):
    """Allocate and strictly reload one native SenseVoice graph."""
    import torch

    from voicehub.models.asr_funasr.native.modeling import SenseVoiceSmallForCTC

    resolved = SenseVoiceSmallConfig.coerce(config)
    model = SenseVoiceSmallForCTC(resolved)
    adapter = SenseVoiceSafeTensorsCheckpointAdapter()
    with SafeTensorReader(checkpoint) as reader:
        declared = reader.metadata.get("format")
        if declared is not None and declared != NATIVE_SENSEVOICE_FORMAT:
            raise ValueError("SenseVoice Safetensors declares unsupported format "
                             f"{declared!r}.")
        adapter.load_streaming(
            model,
            reader,
            resolved.to_dict(),
            strict=True,
        )
    target_dtype = torch.float32 if dtype is None else dtype
    return model.to(device=device, dtype=target_dtype)


__all__ = [
    "NATIVE_SENSEVOICE_CMVN",
    "NATIVE_SENSEVOICE_FILENAME",
    "NATIVE_SENSEVOICE_FORMAT",
    "NATIVE_SENSEVOICE_TOKENIZER",
    "SenseVoiceSafeTensorsCheckpointAdapter",
    "convert_sensevoice_small_checkpoint",
    "file_sha256",
    "load_native_sensevoice_model",
    "native_sensevoice_tensor_shapes",
    "tensor_inventory_fingerprint",
]
