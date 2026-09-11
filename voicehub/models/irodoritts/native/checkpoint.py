"""Strict Safetensors loading and portable export for Irodori-TTS."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from voicehub.checkpointing import SafeTensorReader, save_safetensors
from voicehub.checkpointing.errors import CheckpointCompatibilityError

from .configuration import IrodoriModelConfig
from .metadata import IRODORI_CHECKPOINTS
from .modeling import TextToLatentRFDiT

_FLOAT_DTYPES = frozenset({"F16", "BF16", "F32", "F64"})
_CONFIG_METADATA_KEY = "config_json"


def irodori_header_fingerprint(inventory: Mapping[str, tuple[str, tuple[int, ...]]], ) -> str:
    if not isinstance(inventory, Mapping) or not inventory:
        raise ValueError("Irodori tensor inventory must be a non-empty mapping.")
    rows = []
    for name, (dtype, shape) in sorted(inventory.items()):
        if not isinstance(name, str) or not name or not isinstance(dtype, str):
            raise ValueError("Irodori tensor inventory contains an invalid record.")
        dimensions = tuple(shape)
        if any(isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0
               for dimension in dimensions):
            raise ValueError("Irodori tensor inventory contains an invalid shape.")
        rows.append(f"{name}\t{dtype}\t" + ",".join(str(value) for value in dimensions))
    return hashlib.sha256(("\n".join(rows) + "\n").encode("utf-8")).hexdigest()


def irodori_reader_inventory(reader: SafeTensorReader, ) -> dict[str, tuple[str, tuple[int, ...]]]:
    return {name: (reader.record(name).dtype, reader.record(name).shape) for name in reader.keys()}


def _infer_variant(values: Mapping[str, Any]) -> str:
    explicit = values.get("variant")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip().lower().replace("_", "-")
    caption = bool(values.get("use_caption_condition", False))
    speaker_value = values.get("use_speaker_condition")
    speaker = not caption if speaker_value is None else bool(speaker_value)
    duration = bool(values.get("use_duration_predictor", False))
    if duration and caption and speaker:
        return "v3-voice-design"
    if duration and speaker and not caption:
        return "v3"
    if not duration and caption and not speaker:
        return "v2-voice-design"
    if not duration and speaker and not caption:
        return "v2"
    return "custom"


def read_irodori_config(reader: SafeTensorReader) -> IrodoriModelConfig:
    raw = reader.metadata.get(_CONFIG_METADATA_KEY)
    if raw is None:
        raise CheckpointCompatibilityError(
            "Irodori Safetensors checkpoint is missing required `config_json` metadata.")
    try:
        values = json.loads(raw)
    except json.JSONDecodeError as error:
        raise CheckpointCompatibilityError("Irodori `config_json` metadata is not valid JSON.") from error
    if not isinstance(values, Mapping):
        raise CheckpointCompatibilityError("Irodori `config_json` must decode to an object.")
    normalized = dict(values)
    normalized["variant"] = _infer_variant(values)
    version = values.get("version")
    if (not isinstance(version, bool) and version == 1) or (isinstance(version, str) and
                                                            version.strip().lower() == "v1"):
        raise CheckpointCompatibilityError("Irodori v1 checkpoints are not architecture-compatible.")
    try:
        return IrodoriModelConfig.from_dict(normalized)
    except (TypeError, ValueError) as error:
        raise CheckpointCompatibilityError(f"Irodori checkpoint configuration is invalid: {error}") from error


def native_irodori_tensor_shapes(
        config: IrodoriModelConfig | Mapping[str, Any]) -> dict[str, tuple[int, ...]]:
    resolved = (config if isinstance(config, IrodoriModelConfig) else IrodoriModelConfig.from_dict(config))
    with torch.device("meta"):
        model = TextToLatentRFDiT(resolved)
    return {name: tuple(tensor.shape) for name, tensor in model.state_dict().items()}


def validate_irodori_reader(
    reader: SafeTensorReader,
    config: IrodoriModelConfig,
    *,
    model_id: str | None = None,
    revision: str | None = None,
) -> None:
    inventory = irodori_reader_inventory(reader)
    try:
        expected_shapes = native_irodori_tensor_shapes(config)
    except (TypeError, ValueError, RuntimeError) as error:
        raise CheckpointCompatibilityError(
            f"Irodori checkpoint graph configuration is invalid: {error}") from error
    available = set(inventory)
    expected = set(expected_shapes)
    missing = sorted(expected - available)
    unexpected = sorted(available - expected)
    mismatches = sorted(name for name in expected & available if inventory[name][1] != expected_shapes[name])
    invalid_dtypes = sorted(name for name, (dtype, _) in inventory.items() if dtype not in _FLOAT_DTYPES)
    if missing or unexpected or mismatches or invalid_dtypes:
        details = []
        if missing:
            details.append("missing=" + ", ".join(missing[:5]))
        if unexpected:
            details.append("unexpected=" + ", ".join(unexpected[:5]))
        if mismatches:
            details.append("shape_mismatches=" + ", ".join(mismatches[:5]))
        if invalid_dtypes:
            details.append("non_floating=" + ", ".join(invalid_dtypes[:5]))
        raise CheckpointCompatibilityError(
            "Irodori checkpoint does not match the native graph: " + "; ".join(details))
    if model_id is None and revision is None:
        return
    published = next(
        (
            facts for facts in IRODORI_CHECKPOINTS.values()
            if facts["model_id"] == model_id and facts["revision"] == revision),
        None,
    )
    if published is None:
        raise CheckpointCompatibilityError(
            "No audited Irodori checkpoint matches the requested model ID and revision.")
    dtype_counts = dict(Counter(dtype for dtype, _ in inventory.values()))
    parameters = sum(math.prod(shape) for _, shape in inventory.values())
    tensor_data_bytes = sum(
        math.prod(shape) * {
            "F16": 2,
            "BF16": 2,
            "F32": 4,
            "F64": 8
        }[dtype] for dtype, shape in inventory.values())
    actual = {
        "file_bytes": reader.path.stat().st_size,
        "header_fingerprint": irodori_header_fingerprint(inventory),
        "parameters": parameters,
        "tensor_data_bytes": tensor_data_bytes,
        "tensors": len(inventory),
        "dtype_counts": dtype_counts,
    }
    differences = [
        f"{name}={value!r} (expected {published[name]!r})" for name, value in actual.items()
        if value != published[name]
    ]
    if differences:
        raise CheckpointCompatibilityError(
            "Published Irodori checkpoint verification failed: " + "; ".join(differences))


def load_irodori_safetensors(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
    model_id: str | None = None,
    revision: str | None = None,
) -> tuple[TextToLatentRFDiT, IrodoriModelConfig]:
    requested = Path(path).expanduser()
    if requested.suffix.lower() != ".safetensors":
        raise CheckpointCompatibilityError(
            "Irodori model weights must use Safetensors; pickle, GGUF, MLX, and "
            "quantized provider artifacts are rejected.")
    resolved = requested.resolve()
    if dtype is not None and dtype not in {
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
    }:
        raise ValueError("Irodori execution dtype must be floating-point.")
    with SafeTensorReader(resolved) as reader:
        config = read_irodori_config(reader)
        validate_irodori_reader(
            reader,
            config,
            model_id=model_id,
            revision=revision,
        )
        with torch.device("meta"):
            model = TextToLatentRFDiT(config)
        with torch.no_grad():
            for name in reader.keys():
                value = reader.get_tensor(name, device="cpu")
                target_dtype = dtype if dtype is not None else value.dtype
                model.load_state_dict(
                    {name: value.to(device=device, dtype=target_dtype)},
                    strict=False,
                    assign=True,
                )
    remaining = [name for name, tensor in model.state_dict().items() if tensor.device.type == "meta"]
    if remaining:
        raise CheckpointCompatibilityError(
            "Irodori checkpoint assignment left meta tensors: " + ", ".join(remaining[:5]))
    return model, config


def validate_irodori_export(
    model: TextToLatentRFDiT,
    config: IrodoriModelConfig,
) -> dict[str, torch.Tensor]:
    if not isinstance(model, TextToLatentRFDiT):
        raise TypeError("Irodori export requires the native TextToLatentRFDiT graph.")
    if not isinstance(config, IrodoriModelConfig):
        raise TypeError("Irodori export requires IrodoriModelConfig.")
    config.validate()
    if model.cfg.to_dict() != config.to_dict():
        raise CheckpointCompatibilityError(
            "Irodori export configuration does not match the model graph configuration.")
    state = dict(model.state_dict())
    expected = native_irodori_tensor_shapes(config)
    if set(state) != set(expected):
        raise CheckpointCompatibilityError("Irodori export state has missing or extra tensors.")
    for name, tensor in state.items():
        if tuple(tensor.shape) != expected[name]:
            raise CheckpointCompatibilityError(
                f"Irodori export tensor {name!r} has shape {tuple(tensor.shape)}, "
                f"expected {expected[name]}.")
        if not tensor.is_floating_point() or tensor.is_quantized or tensor.is_complex():
            raise CheckpointCompatibilityError(
                f"Irodori export tensor {name!r} is not a portable floating tensor.")
        if tensor.device.type == "meta":
            raise CheckpointCompatibilityError(f"Irodori export tensor {name!r} is not materialized.")
        if not torch.isfinite(tensor.detach()).all().item():
            raise CheckpointCompatibilityError(f"Irodori export tensor {name!r} contains non-finite values.")
    return state


def save_irodori_safetensors(
    model: TextToLatentRFDiT,
    config: IrodoriModelConfig,
    path: str | Path,
    *,
    metadata: Mapping[str, str] | None = None,
) -> Path:
    """Validate the complete portable artifact before opening its
    destination."""
    state = validate_irodori_export(model, config)
    checkpoint_metadata = dict(metadata or {})
    if any(not isinstance(name, str) or not isinstance(value, str)
           for name, value in checkpoint_metadata.items()):
        raise TypeError("Irodori export metadata must map strings to strings.")
    checkpoint_metadata.update({
        _CONFIG_METADATA_KEY:
        json.dumps(
            config.to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
        "format":
        "voicehub-native-irodoritts-v1",
    })
    return save_safetensors(
        state,
        path,
        metadata=checkpoint_metadata,
    )


class IrodoriCheckpointAdapter:
    """Small adapter façade used by architecture-neutral VoiceHub callers."""

    architecture_id = "irodoritts-rf-dit"
    adapter_id = "irodoritts-native-safetensors"
    adapter_version = "1"

    def load(
        self,
        path: str | Path,
        **options: Any,
    ) -> tuple[TextToLatentRFDiT, IrodoriModelConfig]:
        return load_irodori_safetensors(path, **options)


__all__ = [
    "IrodoriCheckpointAdapter",
    "irodori_header_fingerprint",
    "irodori_reader_inventory",
    "load_irodori_safetensors",
    "native_irodori_tensor_shapes",
    "read_irodori_config",
    "save_irodori_safetensors",
    "validate_irodori_export",
    "validate_irodori_reader",
]
