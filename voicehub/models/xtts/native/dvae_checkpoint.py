"""Strict artifact boundary for the standalone XTTS v2 DVAE."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor

from voicehub.checkpointing import SafeTensorReader, save_safetensors
from voicehub.checkpointing.errors import CheckpointCompatibilityError
from voicehub.models.xtts.native.dvae import XTTS2DVAE, XTTS2DVAEConfig, XTTS2TrainingAudioEncoder
from voicehub.models.xtts.native.metadata import XTTS2_DVAE_SHA256, XTTS2_MEL_STATS_SHA256

NATIVE_XTTS2_DVAE_FORMAT = "voicehub-native-xtts2-dvae-v1"
NATIVE_XTTS2_DVAE_MEL_STATS_FORMAT = "voicehub-native-xtts2-dvae-mel-stats-v1"
NATIVE_XTTS2_DVAE_FILENAME = "dvae.safetensors"
NATIVE_XTTS2_DVAE_MEL_STATS_FILENAME = "mel_stats.safetensors"
_CONFIG_METADATA_KEY = "voicehub.xtts2_dvae_config"


@dataclass(frozen=True, slots=True)
class XTTS2DVAECheckpointInventory:
    """Header-only inventory for one native DVAE artifact."""

    path: Path
    tensor_count: int
    stored_element_count: int
    header_fingerprint: str
    source_sha256: str | None


def _config_metadata(config: XTTS2DVAEConfig) -> str:
    config.validate()
    return json.dumps(
        asdict(config),
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_expected_digest(
    path: Path,
    expected_sha256: str | None,
) -> str:
    digest = _sha256(path)
    if expected_sha256 is None:
        return digest
    if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64 or
            any(character not in "0123456789abcdef" for character in expected_sha256)):
        raise ValueError("Expected XTTS v2 artifact SHA-256 must be 64 lowercase hex characters.")
    if digest != expected_sha256:
        raise CheckpointCompatibilityError(
            f"XTTS v2 artifact digest mismatch for {path.name}: expected "
            f"{expected_sha256}, found {digest}.")
    return digest


def inspect_xtts2_dvae_checkpoint(path: str | Path, ) -> XTTS2DVAECheckpointInventory:
    source = Path(path).expanduser().resolve()
    if source.suffix != ".safetensors":
        raise ValueError("Native XTTS v2 DVAE checkpoints must use `.safetensors`.")
    with SafeTensorReader(source) as reader:
        if reader.metadata.get("format") != NATIVE_XTTS2_DVAE_FORMAT:
            raise CheckpointCompatibilityError(
                "XTTS v2 DVAE Safetensors metadata does not declare "
                f"{NATIVE_XTTS2_DVAE_FORMAT!r}.")
        if _CONFIG_METADATA_KEY not in reader.metadata:
            raise CheckpointCompatibilityError(
                "XTTS v2 DVAE Safetensors metadata has no architecture configuration.")
        rows = []
        stored_elements = 0
        for name in reader.keys():
            record = reader.record(name)
            stored_elements += record.number_of_elements
            rows.append(f"{name}|{record.dtype}|{'x'.join(map(str, record.shape))}")
        source_sha256 = reader.metadata.get("source_sha256")
    return XTTS2DVAECheckpointInventory(
        path=source,
        tensor_count=len(rows),
        stored_element_count=stored_elements,
        header_fingerprint=hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest(),
        source_sha256=source_sha256,
    )


def _validate_reader_against_model(
    reader: SafeTensorReader,
    model: XTTS2DVAE,
) -> tuple[str, ...]:
    expected = {name: tuple(value.shape) for name, value in model.state_dict().items()}
    names = set(reader.keys())
    missing = sorted(set(expected) - names)
    unexpected = sorted(names - set(expected))
    mismatched = sorted((
        name,
        reader.tensor_shape(name),
        expected[name],
    ) for name in names & set(expected) if reader.tensor_shape(name) != expected[name])
    if missing or unexpected or mismatched:
        raise CheckpointCompatibilityError(
            "XTTS v2 DVAE checkpoint namespace is incompatible: "
            f"missing={missing[:8]!r}, unexpected={unexpected[:8]!r}, "
            f"shape_mismatches={mismatched[:8]!r}.")
    declared_config = reader.metadata.get(_CONFIG_METADATA_KEY)
    expected_config = _config_metadata(model.config)
    if declared_config != expected_config:
        raise CheckpointCompatibilityError(
            "XTTS v2 DVAE checkpoint architecture metadata does not match "
            "the requested graph.")
    return tuple(sorted(expected))


def load_xtts2_dvae_checkpoint(
    model: XTTS2DVAE,
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype | None = None,
) -> XTTS2DVAECheckpointInventory:
    """Strictly load a native DVAE without accepting legacy pickle."""
    if not isinstance(model, XTTS2DVAE):
        raise TypeError("XTTS v2 DVAE loading requires an XTTS2DVAE instance.")
    inventory = inspect_xtts2_dvae_checkpoint(path)
    state = {}
    with SafeTensorReader(inventory.path) as reader:
        names = _validate_reader_against_model(reader, model)
        for name in names:
            value = reader.get_tensor(name)
            if not value.is_floating_point() or value.is_quantized or value.is_complex():
                raise CheckpointCompatibilityError(
                    f"XTTS v2 DVAE tensor {name!r} is not a portable floating tensor.")
            if not bool(torch.isfinite(value).all()):
                raise CheckpointCompatibilityError(
                    f"XTTS v2 DVAE tensor {name!r} contains non-finite values.")
            state[name] = value.to(
                device=device,
                dtype=value.dtype if dtype is None else dtype,
            )
    model.load_state_dict(
        state,
        strict=True,
        assign=True,
    )
    remaining = [name for name, value in model.state_dict().items() if value.device.type == "meta"]
    if remaining:
        raise CheckpointCompatibilityError(
            "XTTS v2 DVAE assignment left meta tensors: " + ", ".join(remaining[:8]))
    return inventory


def _portable_state(model: XTTS2DVAE) -> dict[str, Tensor]:
    state = {}
    for name, value in model.state_dict().items():
        if value.device.type == "meta":
            raise CheckpointCompatibilityError(f"XTTS v2 DVAE tensor {name!r} is not materialized.")
        if not value.is_floating_point() or value.is_quantized or value.is_complex():
            raise CheckpointCompatibilityError(
                f"XTTS v2 DVAE tensor {name!r} is not a portable floating tensor.")
        if not bool(torch.isfinite(value.detach()).all()):
            raise CheckpointCompatibilityError(f"XTTS v2 DVAE tensor {name!r} contains non-finite values.")
        state[name] = value.detach().cpu().contiguous()
    return state


def save_xtts2_dvae_checkpoint(
    model: XTTS2DVAE,
    path: str | Path,
    *,
    source_sha256: str | None = None,
) -> Path:
    if not isinstance(model, XTTS2DVAE):
        raise TypeError("XTTS v2 DVAE export requires an XTTS2DVAE instance.")
    metadata = {
        "format": NATIVE_XTTS2_DVAE_FORMAT,
        _CONFIG_METADATA_KEY: _config_metadata(model.config),
        "artifact_boundary": "standalone-frozen-xtts2-dvae",
    }
    if source_sha256 is not None:
        metadata["source_sha256"] = source_sha256
    return save_safetensors(
        _portable_state(model),
        path,
        metadata=metadata,
    ).resolve()


def _validate_legacy_state(
    state: Mapping[str, object],
    config: XTTS2DVAEConfig,
) -> dict[str, Tensor]:
    with torch.device("meta"):
        graph = XTTS2DVAE(config)
    expected = {name: tuple(value.shape) for name, value in graph.state_dict().items()}
    names = set(state)
    missing = sorted(set(expected) - names)
    unexpected = sorted(names - set(expected))
    mismatched = []
    output = {}
    for name in names & set(expected):
        value = state[name]
        if not isinstance(value, Tensor):
            raise CheckpointCompatibilityError(f"Legacy XTTS v2 DVAE state item {name!r} is not a tensor.")
        if tuple(value.shape) != expected[name]:
            mismatched.append((name, tuple(value.shape), expected[name]))
            continue
        if not value.is_floating_point() or value.is_quantized or value.is_complex():
            raise CheckpointCompatibilityError(
                f"Legacy XTTS v2 DVAE tensor {name!r} is not portable floating-point.")
        if not bool(torch.isfinite(value).all()):
            raise CheckpointCompatibilityError(
                f"Legacy XTTS v2 DVAE tensor {name!r} contains non-finite values.")
        output[name] = value.detach().cpu().contiguous()
    if missing or unexpected or mismatched:
        raise CheckpointCompatibilityError(
            "Legacy XTTS v2 DVAE namespace is incompatible: "
            f"missing={missing[:8]!r}, unexpected={unexpected[:8]!r}, "
            f"shape_mismatches={sorted(mismatched)[:8]!r}.")
    return output


def convert_trusted_legacy_xtts2_dvae_checkpoint(
    source: str | Path,
    destination: str | Path,
    *,
    config: XTTS2DVAEConfig | None = None,
    trust_legacy_pickle: bool = False,
    expected_sha256: str | None = XTTS2_DVAE_SHA256,
) -> Path:
    """Convert Coqui's separately published ``dvae.pth`` exactly once."""
    if trust_legacy_pickle is not True:
        raise PermissionError(
            "XTTS v2's official dvae.pth is a legacy pickle container; set "
            "`trust_legacy_pickle=True` only for a reviewed one-time conversion.")
    source_path = Path(source).expanduser().resolve()
    source_digest = _validate_expected_digest(source_path, expected_sha256)
    try:
        payload = torch.load(
            source_path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError as error:
        raise RuntimeError(
            "This PyTorch version cannot restrict legacy XTTS DVAE deserialization.") from error
    if not isinstance(payload, Mapping):
        raise CheckpointCompatibilityError("Legacy XTTS v2 dvae.pth must contain a direct tensor mapping.")
    resolved_config = XTTS2DVAEConfig() if config is None else config
    resolved_config.validate()
    state = _validate_legacy_state(payload, resolved_config)
    return save_safetensors(
        state,
        destination,
        metadata={
            "format": NATIVE_XTTS2_DVAE_FORMAT,
            _CONFIG_METADATA_KEY: _config_metadata(resolved_config),
            "artifact_boundary": "trusted-legacy-coqui-dvae.pth",
            "source_sha256": source_digest,
        },
    ).resolve()


def load_xtts2_dvae_mel_stats(
    path: str | Path,
    *,
    mel_channels: int = 80,
    device: torch.device | str = "cpu",
    dtype: torch.dtype | None = None,
) -> Tensor:
    source = Path(path).expanduser().resolve()
    if source.suffix != ".safetensors":
        raise ValueError("Native XTTS v2 mel statistics must use `.safetensors`.")
    with SafeTensorReader(source) as reader:
        if reader.metadata.get("format") != NATIVE_XTTS2_DVAE_MEL_STATS_FORMAT:
            raise CheckpointCompatibilityError(
                "XTTS v2 mel-stat Safetensors metadata has an incompatible format.")
        if set(reader.keys()) != {"mel_stats"}:
            raise CheckpointCompatibilityError("XTTS v2 mel-stat Safetensors must contain only `mel_stats`.")
        if reader.tensor_shape("mel_stats") != (mel_channels, ):
            raise CheckpointCompatibilityError(
                "XTTS v2 mel-stat shape mismatch: expected "
                f"{(mel_channels,)}, found {reader.tensor_shape('mel_stats')}.")
        value = reader.get_tensor("mel_stats")
    if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise CheckpointCompatibilityError("XTTS v2 mel statistics must be finite floating-point values.")
    if bool((value == 0).any()):
        raise CheckpointCompatibilityError("XTTS v2 mel statistics cannot contain zero.")
    return value.to(
        device=device,
        dtype=value.dtype if dtype is None else dtype,
    )


def save_xtts2_dvae_mel_stats(
    mel_stats: Tensor,
    path: str | Path,
    *,
    source_sha256: str | None = None,
) -> Path:
    if (not isinstance(mel_stats, Tensor) or mel_stats.ndim != 1 or mel_stats.shape[0] <= 0):
        raise ValueError("XTTS v2 mel statistics must be a non-empty rank-one tensor.")
    if not mel_stats.is_floating_point() or not bool(torch.isfinite(mel_stats).all()):
        raise CheckpointCompatibilityError("XTTS v2 mel statistics must be finite floating-point values.")
    if bool((mel_stats == 0).any()):
        raise CheckpointCompatibilityError("XTTS v2 mel statistics cannot contain zero.")
    metadata = {
        "format": NATIVE_XTTS2_DVAE_MEL_STATS_FORMAT,
        "artifact_boundary": "standalone-frozen-xtts2-mel-stats",
    }
    if source_sha256 is not None:
        metadata["source_sha256"] = source_sha256
    return save_safetensors(
        {
            "mel_stats": mel_stats.detach().cpu().contiguous()
        },
        path,
        metadata=metadata,
    ).resolve()


def convert_trusted_legacy_xtts2_mel_stats(
    source: str | Path,
    destination: str | Path,
    *,
    trust_legacy_pickle: bool = False,
    expected_sha256: str | None = XTTS2_MEL_STATS_SHA256,
) -> Path:
    if trust_legacy_pickle is not True:
        raise PermissionError(
            "XTTS v2's official mel_stats.pth is a legacy pickle container; set "
            "`trust_legacy_pickle=True` only for a reviewed one-time conversion.")
    source_path = Path(source).expanduser().resolve()
    source_digest = _validate_expected_digest(source_path, expected_sha256)
    try:
        value = torch.load(
            source_path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError as error:
        raise RuntimeError(
            "This PyTorch version cannot restrict legacy XTTS mel-stat deserialization.") from error
    if not isinstance(value, Tensor):
        raise CheckpointCompatibilityError("Legacy XTTS v2 mel_stats.pth must contain one tensor.")
    return save_xtts2_dvae_mel_stats(
        value,
        destination,
        source_sha256=source_digest,
    )


def load_xtts2_training_audio_encoder(
    dvae_path: str | Path,
    mel_stats_path: str | Path,
    *,
    config: XTTS2DVAEConfig | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype | None = None,
) -> XTTS2TrainingAudioEncoder:
    """Build the optional waveform-to-code data boundary from safe
    artifacts."""
    resolved_config = XTTS2DVAEConfig() if config is None else config
    with torch.device("meta"):
        dvae = XTTS2DVAE(resolved_config)
    load_xtts2_dvae_checkpoint(
        dvae,
        dvae_path,
        device=device,
        dtype=dtype,
    )
    mel_stats = load_xtts2_dvae_mel_stats(
        mel_stats_path,
        mel_channels=resolved_config.mel_channels,
        device=device,
        dtype=dtype,
    )
    boundary = XTTS2TrainingAudioEncoder(dvae, mel_stats)
    boundary.to(device=device)
    if dtype is not None:
        boundary.to(dtype=dtype)
    boundary.requires_grad_(False)
    boundary.eval()
    return boundary


def save_xtts2_training_audio_encoder(
    encoder: XTTS2TrainingAudioEncoder,
    directory: str | Path,
) -> tuple[Path, Path]:
    if not isinstance(encoder, XTTS2TrainingAudioEncoder):
        raise TypeError("XTTS v2 training-audio export requires an XTTS2TrainingAudioEncoder.")
    destination = Path(directory).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    dvae_path = save_xtts2_dvae_checkpoint(
        encoder.dvae,
        destination / NATIVE_XTTS2_DVAE_FILENAME,
    )
    stats_path = save_xtts2_dvae_mel_stats(
        encoder.mel_processor.mel_stats,
        destination / NATIVE_XTTS2_DVAE_MEL_STATS_FILENAME,
    )
    return dvae_path, stats_path


__all__ = [
    "NATIVE_XTTS2_DVAE_FILENAME",
    "NATIVE_XTTS2_DVAE_FORMAT",
    "NATIVE_XTTS2_DVAE_MEL_STATS_FILENAME",
    "NATIVE_XTTS2_DVAE_MEL_STATS_FORMAT",
    "XTTS2DVAECheckpointInventory",
    "convert_trusted_legacy_xtts2_dvae_checkpoint",
    "convert_trusted_legacy_xtts2_mel_stats",
    "inspect_xtts2_dvae_checkpoint",
    "load_xtts2_dvae_checkpoint",
    "load_xtts2_dvae_mel_stats",
    "load_xtts2_training_audio_encoder",
    "save_xtts2_dvae_checkpoint",
    "save_xtts2_dvae_mel_stats",
    "save_xtts2_training_audio_encoder",
]
