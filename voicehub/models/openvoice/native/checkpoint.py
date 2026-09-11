"""Strict legacy conversion and Safetensors I/O for OpenVoice V2."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from voicehub.checkpointing import SafeTensorReader, save_safetensors
from voicehub.checkpointing.errors import CheckpointCompatibilityError
from voicehub.models.openvoice.native.artifacts import OpenVoiceArtifacts


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_state(
    model: nn.Module,
    state: Mapping[str, Any],
) -> dict[str, Tensor]:
    if any(not isinstance(name, str) or not isinstance(tensor, Tensor) for name, tensor in state.items()):
        raise TypeError("OpenVoice checkpoint state must map tensor names to tensors.")
    expected = model.state_dict()
    actual_names = set(state)
    expected_names = set(expected)
    missing = tuple(sorted(expected_names - actual_names))
    unexpected = tuple(sorted(actual_names - expected_names))
    mismatches = []
    for name in sorted(actual_names & expected_names):
        actual_shape = tuple(state[name].shape)
        expected_shape = tuple(expected[name].shape)
        if actual_shape != expected_shape:
            mismatches.append((name, actual_shape, expected_shape))
    mismatches = tuple(mismatches)
    invalid_dtypes = tuple(
        name for name, tensor in state.items()
        if not tensor.is_floating_point() or tensor.dtype != torch.float32)
    if missing or unexpected or mismatches or invalid_dtypes:
        details = []
        if missing:
            details.append("missing=" + ", ".join(missing[:12]))
        if unexpected:
            details.append("unexpected=" + ", ".join(unexpected[:12]))
        if mismatches:
            details.append(
                "shape_mismatch=" +
                ", ".join(f"{name}:{actual}!={wanted}" for name, actual, wanted in mismatches[:8]))
        if invalid_dtypes:
            details.append("non_f32=" + ", ".join(invalid_dtypes[:12]))
        raise CheckpointCompatibilityError("OpenVoice V2 checkpoint is incompatible: " + "; ".join(details))
    return dict(state)


def read_openvoice_checkpoint(
    model: nn.Module,
    artifacts: OpenVoiceArtifacts,
    *,
    trust_pickle_checkpoint: bool = False,
) -> dict[str, Tensor]:
    """Read native Safetensors or the explicitly trusted official pickle."""
    if artifacts.legacy_pytorch:
        if trust_pickle_checkpoint is not True:
            raise ValueError(
                "The official OpenVoice V2 checkpoint uses a PyTorch pickle "
                "container. Review its origin and pass "
                "`trust_pickle_checkpoint=True` once, then export "
                "Safetensors for steady-state inference and training.")
        if artifacts.expected_checkpoint_sha256 is not None:
            actual = _sha256(artifacts.checkpoint_path)
            if actual != artifacts.expected_checkpoint_sha256:
                raise CheckpointCompatibilityError(
                    "Official OpenVoice checkpoint SHA-256 mismatch: "
                    f"{actual}.")
        try:
            payload = torch.load(
                artifacts.checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError as error:  # pragma: no cover - old PyTorch
            raise RuntimeError(
                "OpenVoice conversion requires PyTorch with "
                "`torch.load(..., weights_only=True)`.") from error
        if not isinstance(payload, Mapping) or set(payload) != {"model"}:
            raise CheckpointCompatibilityError("Official OpenVoice checkpoint must contain exactly `model`.")
        state = payload["model"]
        if not isinstance(state, Mapping):
            raise CheckpointCompatibilityError("OpenVoice `model` payload must be a mapping.")
        return _validate_state(model, state)
    with SafeTensorReader(
            artifacts.checkpoint_path,
            max_tensors=1_024,
    ) as reader:
        expected = model.state_dict()
        if set(reader.keys()) != set(expected):
            return _validate_state(model, reader.state_dict())
        for name, target in expected.items():
            if reader.tensor_shape(name) != tuple(target.shape):
                return _validate_state(model, reader.state_dict())
            if reader.record(name).dtype != "F32":
                raise CheckpointCompatibilityError(f"OpenVoice tensor {name!r} must use F32.")
        return reader.state_dict()


def load_openvoice_checkpoint(
    model: nn.Module,
    artifacts: OpenVoiceArtifacts,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
    trust_pickle_checkpoint: bool = False,
) -> None:
    """Validate the complete namespace before one strict assignment."""
    state = read_openvoice_checkpoint(
        model,
        artifacts,
        trust_pickle_checkpoint=trust_pickle_checkpoint,
    )
    state = _validate_state(model, state)
    if dtype is not None:
        state = {name: tensor.to(dtype=dtype) for name, tensor in state.items()}
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise CheckpointCompatibilityError("OpenVoice strict state assignment reported incompatible keys.")
    if dtype is None:
        model.to(device=device)
    else:
        model.to(device=device, dtype=dtype)


def save_openvoice_checkpoint(
    model: nn.Module,
    path: str | Path,
) -> Path:
    """Write a deterministic, inference-reloadable Safetensors checkpoint."""
    state = {name: tensor.detach().cpu().float().contiguous() for name, tensor in model.state_dict().items()}
    state = _validate_state(model, state)
    return save_safetensors(
        state,
        path,
        metadata={
            "architecture": "openvoice-v2-converter",
            "format": "voicehub-openvoice-v2-v1",
            "license": "MIT",
        },
    )


__all__ = [
    "load_openvoice_checkpoint",
    "read_openvoice_checkpoint",
    "save_openvoice_checkpoint",
]
