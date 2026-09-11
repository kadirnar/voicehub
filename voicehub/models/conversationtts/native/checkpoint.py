"""Strict checkpoint loading and safe export for ConversationTTS."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from voicehub.checkpointing import SafeTensorReader, save_safetensors
from voicehub.checkpointing.errors import CheckpointCompatibilityError
from voicehub.models.conversationtts.native.metadata import NATIVE_CONVERSATIONTTS_FORMAT

_RUNTIME_STATE_MARKERS = (
    ".kv_cache.",
    "backbone_causal_mask",
    "decoder_causal_mask",
)


@dataclass(frozen=True, slots=True)
class ConversationTTSCheckpointReport:
    """Validated checkpoint inventory loaded into one model."""

    path: Path
    format: str
    tensor_count: int
    parameter_count: int


def _is_runtime_state(name: str) -> bool:
    return any(marker in name for marker in _RUNTIME_STATE_MARKERS)


def exportable_state_dict(model: nn.Module) -> dict[str, Tensor]:
    """Return only persistent model state, excluding serving KV caches."""
    if not isinstance(model, nn.Module):
        raise TypeError("`model` must be a torch.nn.Module.")
    state = {
        name: value.detach()
        for name, value in model.state_dict().items() if not _is_runtime_state(name)
    }
    if not state:
        raise ValueError("ConversationTTS model state is empty.")
    return state


def _normalize_legacy_state(state: Mapping[Any, Any]) -> dict[str, Tensor]:
    normalized: dict[str, Tensor] = {}
    for key, value in state.items():
        if not isinstance(key, str):
            raise TypeError("ConversationTTS checkpoint keys must be strings.")
        name = key.removeprefix("module.")
        if name in normalized:
            raise CheckpointCompatibilityError(
                "ConversationTTS checkpoint contains colliding keys after "
                f"removing the DataParallel prefix: {name!r}.")
        if not isinstance(value, Tensor):
            raise TypeError(f"ConversationTTS state value {name!r} is not a tensor.")
        normalized[name] = value
    return normalized


def _expected_state(model: nn.Module) -> dict[str, Tensor]:
    return exportable_state_dict(model)


def _validate_inventory(
    expected: Mapping[str, Tensor],
    available: Mapping[str, tuple[int, ...]],
    *,
    path: Path,
) -> None:
    expected_names = set(expected)
    available_names = set(available)
    missing = sorted(expected_names - available_names)
    unexpected = sorted(available_names - expected_names)
    mismatched = sorted((
        name,
        available[name],
        tuple(expected[name].shape),
    ) for name in expected_names & available_names if available[name] != tuple(expected[name].shape))
    if missing or unexpected or mismatched:
        raise CheckpointCompatibilityError(
            f"ConversationTTS checkpoint {path} is incompatible: "
            f"missing={missing!r}, unexpected={unexpected!r}, "
            f"shape_mismatches={mismatched!r}.")


def _copy_state(
    expected: Mapping[str, Tensor],
    state: Mapping[str, Tensor],
) -> None:
    with torch.no_grad():
        for name, target in expected.items():
            source = state[name]
            target.copy_(source.to(
                device=target.device,
                dtype=target.dtype,
            ))


def _load_safetensors(
    model: nn.Module,
    path: Path,
) -> ConversationTTSCheckpointReport:
    expected = _expected_state(model)
    with SafeTensorReader(path) as reader:
        available = {name: tuple(reader.tensor_shape(name)) for name in reader.keys()}
        _validate_inventory(expected, available, path=path)
        with torch.no_grad():
            for name, target in expected.items():
                target.copy_(reader.get_tensor(
                    name,
                    device=target.device,
                    dtype=target.dtype,
                ))
        parameter_count = sum(reader.record(name).number_of_elements for name in reader.keys())
        tensor_count = len(reader)
    return ConversationTTSCheckpointReport(
        path=path,
        format="safetensors",
        tensor_count=tensor_count,
        parameter_count=parameter_count,
    )


def _load_restricted_legacy(
    model: nn.Module,
    path: Path,
) -> ConversationTTSCheckpointReport:
    try:
        payload = torch.load(
            path,
            # The published archive is roughly 9.3 GB. Materializing it on an
            # accelerator before validating its inventory can needlessly
            # exhaust device memory. Keep the one-time restricted conversion
            # on CPU; `_copy_state` moves each validated tensor independently.
            map_location="cpu",
            weights_only=True,
        )
    except TypeError as exc:
        if "weights_only" not in str(exc):
            raise
        raise RuntimeError(
            "This PyTorch build cannot load the published ConversationTTS "
            "checkpoint safely. Upgrade PyTorch or convert the trusted "
            "artifact to Safetensors in a supported environment.") from exc
    if not isinstance(payload, Mapping):
        raise TypeError("ConversationTTS legacy checkpoint must contain a mapping.")
    raw_state = payload.get("model")
    if not isinstance(raw_state, Mapping):
        raise TypeError("ConversationTTS legacy checkpoint is missing its `model` "
                        "state dictionary.")
    state = _normalize_legacy_state(raw_state)
    expected = _expected_state(model)
    available = {name: tuple(value.shape) for name, value in state.items()}
    _validate_inventory(expected, available, path=path)
    _copy_state(expected, state)
    return ConversationTTSCheckpointReport(
        path=path,
        format="pytorch-weights-only",
        tensor_count=len(state),
        parameter_count=sum(value.numel() for value in state.values()),
    )


def load_conversationtts_checkpoint(
    model: nn.Module,
    checkpoint: str | Path,
    *,
    device: str | torch.device,
) -> ConversationTTSCheckpointReport:
    """Load a native Safetensors file or the restricted official archive."""
    torch.device(device)
    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"ConversationTTS checkpoint was not found: {path}.")
    if path.suffix.lower() == ".safetensors":
        return _load_safetensors(model, path)
    return _load_restricted_legacy(model, path)


def export_conversationtts_checkpoint(
    model: nn.Module,
    path: str | Path,
) -> Path:
    """Write a deterministic, inference-reloadable Safetensors artifact."""
    state = exportable_state_dict(model)
    return save_safetensors(
        state,
        path,
        metadata={
            "format": NATIVE_CONVERSATIONTTS_FORMAT,
            "model_type": "conversationtts",
            "license": "CC-BY-NC-4.0",
        },
    )


__all__ = [
    "ConversationTTSCheckpointReport",
    "export_conversationtts_checkpoint",
    "exportable_state_dict",
    "load_conversationtts_checkpoint",
]
