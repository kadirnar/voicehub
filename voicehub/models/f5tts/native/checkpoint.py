"""Strict F5-TTS and Vocos checkpoint import/export."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from voicehub.checkpointing import SafeTensorReader, save_safetensors
from voicehub.models.f5tts.native.metadata import F5TTS_NATIVE_FORMAT, VOCOS_NATIVE_FORMAT


@dataclass(frozen=True, slots=True)
class F5CheckpointLoadReport:
    path: Path
    tensor_count: int
    prefix: str
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def require_file_integrity(
    path: str | Path,
    *,
    sha256: str | None = None,
    size: int | None = None,
) -> Path:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Checkpoint was not found: {source}.")
    if size is not None and source.stat().st_size != size:
        raise ValueError(
            f"Checkpoint size mismatch for {source.name}: expected {size}, "
            f"found {source.stat().st_size}.")
    if sha256 is not None:
        actual = file_sha256(source)
        if actual.lower() != sha256.lower():
            raise ValueError(
                f"Checkpoint SHA-256 mismatch for {source.name}: expected "
                f"{sha256}, found {actual}.")
    return source


def _state_shapes(module: nn.Module) -> dict[str, tuple[int, ...]]:
    return {name: tuple(tensor.shape) for name, tensor in module.state_dict().items()}


def _resolve_prefix(
    reader: SafeTensorReader,
    expected: Mapping[str, tuple[int, ...]],
    *,
    preferred_prefix: str,
) -> str:
    candidates = (preferred_prefix, "", "model.", "ema_model.")
    checkpoint_names = reader.keys()
    for prefix in dict.fromkeys(candidates):
        names = {
            name.removeprefix(prefix)
            for name in checkpoint_names if not prefix or name.startswith(prefix)
        }
        if set(expected).issubset(names):
            return prefix
    raise ValueError("F5-TTS checkpoint does not contain a complete compatible model "
                     "namespace.")


def load_f5tts_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    use_ema: bool = True,
    strict: bool = True,
    device: torch.device | str = "cpu",
) -> F5CheckpointLoadReport:
    """Load a native or official F5 Safetensors checkpoint."""
    logical_source = Path(path).expanduser()
    if logical_source.suffix.lower() != ".safetensors":
        raise ValueError(
            "Native F5-TTS loads Safetensors only. Convert legacy weights "
            "once with `convert_legacy_f5tts_checkpoint`.")
    source = logical_source.resolve()
    targets = model.state_dict(keep_vars=True)
    expected = {name: tuple(tensor.shape) for name, tensor in targets.items()}
    with SafeTensorReader(source) as reader:
        prefix = _resolve_prefix(
            reader,
            expected,
            preferred_prefix="ema_model." if use_ema else "",
        )
        missing = []
        for name, shape in expected.items():
            checkpoint_name = f"{prefix}{name}"
            if checkpoint_name not in reader:
                missing.append(name)
                continue
            actual_shape = reader.tensor_shape(checkpoint_name)
            if actual_shape != shape:
                raise ValueError(
                    f"F5-TTS tensor {checkpoint_name!r} has shape "
                    f"{actual_shape}, expected {shape}.")
        selected = {f"{prefix}{name}" for name in expected}
        ignored = {"step", "initted"}
        checkpoint_names = reader.keys()
        unexpected = tuple(name for name in checkpoint_names if name not in selected and name not in ignored)
    if strict and (missing or unexpected):
        raise ValueError(
            "F5-TTS checkpoint namespace mismatch: "
            f"missing={missing!r}, unexpected={list(unexpected)!r}.")
    with torch.no_grad(), SafeTensorReader(source) as reader:
        for name, target in targets.items():
            checkpoint_name = f"{prefix}{name}"
            if checkpoint_name not in reader:
                continue
            tensor = reader.get_tensor(
                checkpoint_name,
                device=target.device,
                dtype=target.dtype,
            )
            target.copy_(tensor)
    return F5CheckpointLoadReport(
        path=source,
        tensor_count=len(expected) - len(missing),
        prefix=prefix,
        missing=tuple(missing),
        unexpected=unexpected,
    )


def _unwrap_legacy_state(value: Any) -> Mapping[str, torch.Tensor]:
    if not isinstance(value, Mapping):
        raise TypeError("Legacy checkpoint must contain a tensor mapping.")
    for wrapper in ("state_dict", "model_state_dict"):
        nested = value.get(wrapper)
        if isinstance(nested, Mapping):
            value = nested
            break
    if "ema_model" in value and isinstance(value["ema_model"], Mapping):
        value = {f"ema_model.{name}": tensor for name, tensor in value["ema_model"].items()}
    if not value or any(not isinstance(name, str) or not isinstance(tensor, torch.Tensor)
                        for name, tensor in value.items()):
        raise TypeError("Legacy checkpoint state must map tensor names to PyTorch tensors.")
    return value


def convert_legacy_f5tts_checkpoint(
    source: str | Path,
    destination: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Convert trusted legacy PyTorch weights to deterministic Safetensors."""
    source_path = Path(source).expanduser().resolve()
    destination_path = Path(destination).expanduser()
    if destination_path.exists() and not overwrite:
        return destination_path.resolve()
    try:
        payload = torch.load(
            source_path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError as error:
        raise RuntimeError(
            "Legacy F5-TTS conversion requires PyTorch with safe "
            "`weights_only` loading.") from error
    state = _unwrap_legacy_state(payload)
    return save_safetensors(
        state,
        destination_path,
        metadata={
            "format": F5TTS_NATIVE_FORMAT,
            "converted_from": source_path.name,
        },
    ).resolve()


def export_f5tts_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    prefix: str = "ema_model.",
    state_override: Mapping[str, torch.Tensor] | None = None,
) -> Path:
    state = model.state_dict() if state_override is None else state_override
    tensors = {f"{prefix}{name}": tensor.detach().cpu().contiguous() for name, tensor in state.items()}
    return save_safetensors(
        tensors,
        path,
        metadata={"format": F5TTS_NATIVE_FORMAT},
    )


def convert_legacy_vocos_checkpoint(
    source: str | Path,
    destination: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Convert the released Vocos ``.bin`` artifact exactly once."""
    source_path = Path(source).expanduser().resolve()
    destination_path = Path(destination).expanduser()
    if destination_path.exists() and not overwrite:
        return destination_path.resolve()
    try:
        payload = torch.load(
            source_path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError as error:
        raise RuntimeError(
            "Vocos conversion requires PyTorch with safe `weights_only` "
            "loading.") from error
    state = _unwrap_legacy_state(payload)
    return save_safetensors(
        state,
        destination_path,
        metadata={
            "format": VOCOS_NATIVE_FORMAT,
            "converted_from": source_path.name,
        },
    ).resolve()


def load_vocos_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> F5CheckpointLoadReport:
    logical_source = Path(path).expanduser()
    if logical_source.suffix.lower() != ".safetensors":
        raise ValueError("Native Vocos loads converted Safetensors only.")
    source = logical_source.resolve()
    targets = model.state_dict(keep_vars=True)
    expected = {name: tuple(tensor.shape) for name, tensor in targets.items()}
    with SafeTensorReader(source) as reader:
        missing = tuple(name for name in expected if name not in reader)
        checkpoint_names = reader.keys()
        unexpected = tuple(name for name in checkpoint_names if name not in expected)
        if missing or unexpected:
            raise ValueError(
                "Vocos checkpoint namespace mismatch: "
                f"missing={list(missing)!r}, unexpected={list(unexpected)!r}.")
        for name, shape in expected.items():
            actual_shape = reader.tensor_shape(name)
            if actual_shape != shape:
                raise ValueError(f"Vocos tensor {name!r} has shape {actual_shape}, "
                                 f"expected {shape}.")
    with torch.no_grad(), SafeTensorReader(source) as reader:
        for name, target in targets.items():
            tensor = reader.get_tensor(
                name,
                device=target.device,
                dtype=target.dtype,
            )
            target.copy_(tensor)
    return F5CheckpointLoadReport(
        path=source,
        tensor_count=len(expected),
        prefix="",
        missing=(),
        unexpected=(),
    )


__all__ = [
    "F5CheckpointLoadReport",
    "convert_legacy_f5tts_checkpoint",
    "convert_legacy_vocos_checkpoint",
    "export_f5tts_checkpoint",
    "file_sha256",
    "load_f5tts_checkpoint",
    "load_vocos_checkpoint",
    "require_file_integrity",
]
