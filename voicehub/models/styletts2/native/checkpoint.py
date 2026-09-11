"""Strict legacy import and Safetensors reload for StyleTTS 2."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from voicehub.checkpointing import SafeTensorReader, save_safetensors
from voicehub.checkpointing.errors import CheckpointCompatibilityError
from voicehub.hub import write_json_file
from voicehub.models.styletts2.native.configuration import StyleTTS2ArchitectureConfig
from voicehub.models.styletts2.native.metadata import STYLETTS2_NATIVE_FORMAT
from voicehub.models.styletts2.native.modeling import DEPLOYABLE_STYLETTS2_COMPONENTS


@dataclass(frozen=True, slots=True)
class StyleTTS2CheckpointReport:
    path: Path
    tensor_count: int
    parameter_count: int
    inventory_fingerprint: str
    legacy: bool


def _fingerprint(inventory: Mapping[str, tuple[str, tuple[int, ...]]], ) -> str:
    rows = (
        f"{name}|{dtype}|{','.join(str(item) for item in shape)}"
        for name, (dtype, shape) in sorted(inventory.items()))
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def _expected_state(model: nn.Module) -> Mapping[str, Tensor]:
    state = model.state_dict(keep_vars=True)
    if not state:
        raise ValueError("StyleTTS 2 model returned an empty state dict.")
    prefixes = {name.split(".", 1)[0] for name in state}
    expected = set(DEPLOYABLE_STYLETTS2_COMPONENTS)
    if prefixes != expected:
        raise ValueError(
            "StyleTTS 2 model component inventory is incomplete: "
            f"missing={sorted(expected - prefixes)!r}, "
            f"unexpected={sorted(prefixes - expected)!r}.")
    return state


def inspect_styletts2_checkpoint(path: str | Path, ) -> StyleTTS2CheckpointReport:
    source = Path(path).expanduser().resolve()
    if source.suffix.lower() != ".safetensors":
        raise ValueError("Checkpoint inspection accepts Safetensors only.")
    with SafeTensorReader(source) as reader:
        inventory = {name: (reader.record(name).dtype, reader.tensor_shape(name)) for name in reader.keys()}
        parameter_count = sum(reader.record(name).number_of_elements for name in reader.keys())
    return StyleTTS2CheckpointReport(
        path=source,
        tensor_count=len(inventory),
        parameter_count=parameter_count,
        inventory_fingerprint=_fingerprint(inventory),
        legacy=False,
    )


def _validate_safe_layout(
    model: nn.Module,
    reader: SafeTensorReader,
) -> tuple[str, ...]:
    expected = _expected_state(model)
    expected_names = set(expected)
    actual_names = set(reader.keys())
    missing = sorted(expected_names - actual_names)
    unexpected = sorted(actual_names - expected_names)
    shape_mismatches = sorted((
        name,
        reader.tensor_shape(name),
        tuple(expected[name].shape),
    ) for name in expected_names & actual_names if reader.tensor_shape(name) != tuple(expected[name].shape))
    if missing or unexpected or shape_mismatches:
        raise CheckpointCompatibilityError(
            "StyleTTS 2 Safetensors namespace mismatch: "
            f"missing={missing[:12]!r}, "
            f"unexpected={unexpected[:12]!r}, "
            f"shape_mismatches={shape_mismatches[:12]!r}.")
    return tuple(sorted(expected_names))


def load_styletts2_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype | None = None,
) -> StyleTTS2CheckpointReport:
    """Validate the complete file before assigning any tensor."""
    report = inspect_styletts2_checkpoint(path)
    targets = _expected_state(model)
    with SafeTensorReader(report.path) as reader:
        names = _validate_safe_layout(model, reader)
        values = {}
        for name in names:
            target = targets[name]
            value = reader.get_tensor(name, device=device)
            if target.is_floating_point():
                value = value.to(dtype=dtype or target.dtype)
            else:
                value = value.to(dtype=target.dtype)
            values[name] = value
    incompatible = model.load_state_dict(values, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise CheckpointCompatibilityError("StyleTTS 2 assignment failed strict validation.")
    return report


def _normalize_legacy_component(
    component: str,
    state: Any,
) -> dict[str, Tensor]:
    if not isinstance(state, Mapping) or not state:
        raise CheckpointCompatibilityError(f"Legacy StyleTTS 2 component {component!r} is not a state dict.")
    normalized: dict[str, Tensor] = {}
    for source_name, value in state.items():
        if not isinstance(source_name, str) or not source_name:
            raise CheckpointCompatibilityError("Legacy StyleTTS 2 tensor names must be non-empty strings.")
        if not isinstance(value, Tensor):
            raise CheckpointCompatibilityError(f"Legacy StyleTTS 2 value {source_name!r} is not a tensor.")
        name = (source_name[7:] if source_name.startswith("module.") else source_name)
        target_name = f"{component}.{name}"
        if target_name in normalized:
            raise CheckpointCompatibilityError(f"Legacy StyleTTS 2 mapping collides at {target_name!r}.")
        normalized[target_name] = value
    return normalized


def read_legacy_styletts2_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    trust_pickle_checkpoint: bool,
) -> OrderedDict[str, Tensor]:
    """Read one reviewed legacy file with PyTorch's restricted unpickler."""
    if trust_pickle_checkpoint is not True:
        raise ValueError(
            "Legacy StyleTTS 2 checkpoints use a PyTorch pickle container. "
            "Review the file and pass `trust_pickle_checkpoint=True` once, "
            "then export Safetensors.")
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"StyleTTS 2 checkpoint was not found: {source}.")
    if source.suffix.lower() not in {".pth", ".pt", ".t7"}:
        raise ValueError("Legacy StyleTTS 2 import accepts .pth, .pt, or .t7 only.")
    try:
        payload = torch.load(
            source,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError as error:  # pragma: no cover - unsupported old PyTorch
        raise RuntimeError("StyleTTS 2 conversion requires safe weights-only loading.") from error
    if not isinstance(payload, Mapping):
        raise CheckpointCompatibilityError("Legacy StyleTTS 2 checkpoint must be a mapping.")
    components = payload.get("net", payload)
    if not isinstance(components, Mapping):
        raise CheckpointCompatibilityError("Legacy StyleTTS 2 checkpoint `net` must be a mapping.")
    missing_components = sorted(set(DEPLOYABLE_STYLETTS2_COMPONENTS) - set(components))
    if missing_components:
        raise CheckpointCompatibilityError(
            "Legacy StyleTTS 2 checkpoint misses deployable components: " + ", ".join(missing_components) +
            ".")
    flattened: dict[str, Tensor] = {}
    for component in DEPLOYABLE_STYLETTS2_COMPONENTS:
        flattened.update(_normalize_legacy_component(component, components[component]))

    expected = _expected_state(model)
    expected_names = set(expected)
    actual_names = set(flattened)
    missing = sorted(expected_names - actual_names)
    unexpected = sorted(actual_names - expected_names)
    shape_mismatches = []
    for name in sorted(expected_names & actual_names):
        actual_shape = tuple(flattened[name].shape)
        expected_shape = tuple(expected[name].shape)
        if actual_shape != expected_shape:
            shape_mismatches.append((name, actual_shape, expected_shape), )
    if missing or unexpected or shape_mismatches:
        raise CheckpointCompatibilityError(
            "Legacy StyleTTS 2 checkpoint is incompatible: "
            f"missing={missing[:12]!r}, "
            f"unexpected={unexpected[:12]!r}, "
            f"shape_mismatches={shape_mismatches[:12]!r}.")
    return OrderedDict((name, flattened[name].detach().cpu().contiguous()) for name in sorted(flattened))


def convert_legacy_styletts2_checkpoint(
    model: nn.Module,
    source: str | Path,
    destination: str | Path,
    *,
    trust_pickle_checkpoint: bool,
) -> Path:
    state = read_legacy_styletts2_checkpoint(
        model,
        source,
        trust_pickle_checkpoint=trust_pickle_checkpoint,
    )
    return save_safetensors(
        state,
        destination,
        metadata={
            "format": STYLETTS2_NATIVE_FORMAT,
            "architecture": "styletts2",
            "converted_from": Path(source).name,
        },
    ).resolve()


def export_styletts2_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    state_override: Mapping[str, Tensor] | None = None,
) -> Path:
    expected = _expected_state(model)
    state = expected if state_override is None else dict(state_override)
    if set(state) != set(expected):
        raise ValueError(
            "StyleTTS 2 export state is incomplete: "
            f"missing={sorted(set(expected) - set(state))!r}, "
            f"unexpected={sorted(set(state) - set(expected))!r}.")
    for name, value in state.items():
        if not isinstance(value, Tensor):
            raise TypeError(f"StyleTTS 2 state value {name!r} is not a tensor.")
        if tuple(value.shape) != tuple(expected[name].shape):
            raise ValueError(f"StyleTTS 2 state value {name!r} has the wrong shape.")
    return save_safetensors(
        {
            name: value.detach().cpu().contiguous()
            for name, value in state.items()
        },
        path,
        metadata={
            "format": STYLETTS2_NATIVE_FORMAT,
            "architecture": "styletts2",
            "training_boundary": "preprocessed-teacher-forced",
        },
    ).resolve()


def save_styletts2_pretrained(
    model: nn.Module,
    config: StyleTTS2ArchitectureConfig,
    directory: str | Path,
) -> Path:
    if not isinstance(config, StyleTTS2ArchitectureConfig):
        raise TypeError("`config` must be a StyleTTS2ArchitectureConfig.")
    destination = Path(directory).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    export_styletts2_checkpoint(
        model,
        destination / "model.safetensors",
    )
    write_json_file(destination / "config.json", config.to_dict())
    return destination.resolve()


__all__ = [
    "StyleTTS2CheckpointReport",
    "convert_legacy_styletts2_checkpoint",
    "export_styletts2_checkpoint",
    "inspect_styletts2_checkpoint",
    "load_styletts2_checkpoint",
    "read_legacy_styletts2_checkpoint",
    "save_styletts2_pretrained",
]
