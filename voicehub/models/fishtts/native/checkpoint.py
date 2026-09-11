"""Strict Fish S2 Safetensors loading, export, and legacy codec conversion."""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import Tensor, nn

from voicehub.checkpointing import SafeTensorReader, ShardedSafeTensorReader, save_safetensors
from voicehub.checkpointing.errors import CheckpointCompatibilityError, CheckpointIntegrityError
from voicehub.hub import write_json_file
from voicehub.models.fishtts.native.codec import FishModifiedDAC
from voicehub.models.fishtts.native.configuration import FishCodecConfig
from voicehub.models.fishtts.native.metadata import (
    FISH_ATTRIBUTION,
    FISH_LICENSE_NOTICE,
    FISH_S2_LEGACY_CODEC_SHA256,
    FISH_S2_LEGACY_CODEC_SIZE,
    NATIVE_FISH_CODEC_FORMAT,
    NATIVE_FISH_SEMANTIC_FORMAT,
)
from voicehub.models.fishtts.native.modeling import FishS2ForConditionalGeneration


class _TensorReader(Protocol):

    def keys(self) -> tuple[str, ...]:
        ...

    def tensor_shape(self, name: str) -> tuple[int, ...]:
        ...

    def get_tensor(self, name: str) -> Tensor:
        ...


@dataclass(frozen=True, slots=True)
class FishCheckpointReport:
    path: Path
    tensor_count: int
    parameter_count: int
    header_fingerprint: str


def write_fish_license_files(directory: str | Path) -> Path:
    """Copy Fish's license and mandatory attribution into an export.

    Fish model derivatives, including fine-tuned checkpoints, remain
    subject to the Fish Audio Research License.  Keeping this helper
    public lets custom VoiceHub training recipes preserve those
    obligations too.
    """
    destination = Path(directory).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    license_source = Path(__file__).with_name("THIRD_PARTY_LICENSE")
    if not license_source.is_file():
        raise FileNotFoundError("The bundled Fish Audio Research License is missing.")
    shutil.copy2(
        license_source,
        destination / "THIRD_PARTY_LICENSE",
    )
    (destination / "NOTICE").write_text(
        f"{FISH_LICENSE_NOTICE}\n"
        f"{FISH_ATTRIBUTION}\n"
        "Modification notice: VoiceHub provides a native PyTorch graph, "
        "strict Safetensors loading, and fine-tuning integration.\n",
        encoding="utf-8",
    )
    return destination.resolve()


def verify_file_integrity(
    path: str | Path,
    *,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
    chunk_size: int = 8 * 1024 * 1024,
) -> None:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Fish artifact was not found: {source}.")
    if expected_size is not None and source.stat().st_size != expected_size:
        raise CheckpointIntegrityError(
            f"{source.name} has size {source.stat().st_size}; expected "
            f"{expected_size}.")
    if expected_sha256 is not None:
        digest = hashlib.sha256()
        with source.open("rb") as stream:
            while block := stream.read(chunk_size):
                digest.update(block)
        actual = digest.hexdigest()
        if actual != expected_sha256:
            raise CheckpointIntegrityError(
                f"{source.name} has SHA-256 {actual}; expected "
                f"{expected_sha256}.")


def _open_reader(path: Path):
    if path.name.endswith(".safetensors.index.json"):
        return ShardedSafeTensorReader(path)
    if path.suffix == ".safetensors":
        return SafeTensorReader(path)
    raise ValueError("Fish steady-state checkpoints must be Safetensors.")


def _tensor_record(reader: _TensorReader, name: str) -> Any:
    record = getattr(reader, "record", None)
    if callable(record):
        return record(name)
    if isinstance(reader, ShardedSafeTensorReader):
        shard = reader.index.shard_path(name)
        return reader._reader(shard).record(name)
    raise TypeError(f"{type(reader).__name__} does not expose Safetensors metadata.")


def _fingerprint(reader: _TensorReader) -> str:
    rows = []
    for name in sorted(reader.keys()):
        record = _tensor_record(reader, name)
        rows.append(f"{name}|{record.dtype}|" + "x".join(str(item) for item in reader.tensor_shape(name)))
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def inspect_fish_checkpoint(path: str | Path) -> FishCheckpointReport:
    source = Path(path).expanduser().resolve()
    with _open_reader(source) as reader:
        count = sum(_tensor_record(reader, name).number_of_elements for name in reader.keys())
        fingerprint = _fingerprint(reader)
        tensors = len(reader.keys())
    return FishCheckpointReport(
        path=source,
        tensor_count=tensors,
        parameter_count=count,
        header_fingerprint=fingerprint,
    )


def _source_to_native(name: str) -> str:
    if name.startswith("text_model.model."):
        return name[len("text_model.model."):]
    if name.startswith("audio_decoder."):
        suffix = name[len("audio_decoder."):]
        if suffix.startswith("codebook_embeddings."):
            return suffix
        return "fast_" + suffix
    return name


def _native_to_source(name: str) -> str:
    if name.startswith("codebook_embeddings."):
        return "audio_decoder." + name
    if name.startswith("fast_"):
        return "audio_decoder." + name[len("fast_"):]
    return "text_model.model." + name


def _expected_shapes(module: nn.Module) -> dict[str, tuple[int, ...]]:
    return {name: tuple(value.shape) for name, value in module.state_dict(keep_vars=True).items()}


def _semantic_mapping(
    model: FishS2ForConditionalGeneration,
    reader: _TensorReader,
) -> dict[str, str]:
    expected = _expected_shapes(model)
    source_names = set(reader.keys())
    official = any(name.startswith(("text_model.model.", "audio_decoder.")) for name in source_names)
    if official:
        mapping = {name: _source_to_native(name) for name in source_names}
    else:
        mapping = {name: name for name in source_names}
    if len(set(mapping.values())) != len(mapping):
        raise CheckpointCompatibilityError("Fish checkpoint mapping produces duplicate native tensors.")
    native_names = set(mapping.values())
    missing = sorted(set(expected) - native_names)
    unexpected = sorted(native_names - set(expected))
    mismatched = []
    for source, target in mapping.items():
        if (target in expected and reader.tensor_shape(source) != expected[target]):
            mismatched.append((
                source,
                reader.tensor_shape(source),
                expected[target],
            ))
    mismatched.sort()
    if missing or unexpected or mismatched:
        raise CheckpointCompatibilityError(
            "Fish semantic checkpoint is incompatible: "
            f"missing={missing[:12]!r}, "
            f"unexpected={unexpected[:12]!r}, "
            f"shape_mismatches={mismatched[:12]!r}.")
    return mapping


def load_fish_semantic_checkpoint(
    model: FishS2ForConditionalGeneration,
    path: str | Path,
    *,
    device: str | torch.device,
    dtype: torch.dtype | None = None,
) -> FishCheckpointReport:
    """Validate every tensor before assigning any checkpoint value."""
    if not isinstance(model, FishS2ForConditionalGeneration):
        raise TypeError("`model` must be a native Fish S2 model.")
    report = inspect_fish_checkpoint(path)
    with _open_reader(report.path) as reader:
        mapping = _semantic_mapping(model, reader)
        with torch.no_grad():
            for source, target in sorted(mapping.items()):
                value = reader.get_tensor(source)
                target_dtype = (dtype if dtype is not None and value.is_floating_point() else value.dtype)
                model.load_state_dict(
                    {target: value.to(
                        device=device,
                        dtype=target_dtype,
                    )},
                    strict=False,
                    assign=True,
                )
    remaining = [name for name, value in model.state_dict().items() if value.device.type == "meta"]
    if remaining:
        raise CheckpointCompatibilityError(
            "Fish semantic load left meta tensors: " + ", ".join(remaining[:12]) + ".")
    return report


def export_fish_semantic_checkpoint(
    model: FishS2ForConditionalGeneration,
    path: str | Path,
    *,
    state_override: Mapping[str, Tensor] | None = None,
) -> Path:
    state = (dict(model.state_dict()) if state_override is None else dict(state_override))
    expected = set(model.state_dict())
    if set(state) != expected:
        raise ValueError(
            "Fish semantic export is incomplete: "
            f"missing={sorted(expected - set(state))[:12]!r}, "
            f"unexpected={sorted(set(state) - expected)[:12]!r}.")
    published = {_native_to_source(name): value.detach() for name, value in state.items()}
    return save_safetensors(
        published,
        path,
        metadata={
            "format": NATIVE_FISH_SEMANTIC_FORMAT,
            "architecture": "fish_qwen3_omni",
            "producer": "voicehub",
        },
    ).resolve()


def save_fish_semantic_pretrained(
    model: FishS2ForConditionalGeneration,
    directory: str | Path,
    *,
    state_override: Mapping[str, Tensor] | None = None,
) -> Path:
    destination = Path(directory).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    export_fish_semantic_checkpoint(
        model,
        destination / "model.safetensors",
        state_override=state_override,
    )
    write_json_file(destination / "config.json", model.config.to_dict())
    write_fish_license_files(destination)
    return destination.resolve()


def _validate_codec_layout(
    codec: FishModifiedDAC,
    reader: _TensorReader,
) -> tuple[str, ...]:
    expected = _expected_shapes(codec)
    actual = set(reader.keys())
    missing = sorted(set(expected) - actual)
    unexpected = sorted(actual - set(expected))
    mismatched = sorted((
        name,
        reader.tensor_shape(name),
        expected[name],
    ) for name in set(expected) & actual if reader.tensor_shape(name) != expected[name])
    if missing or unexpected or mismatched:
        raise CheckpointCompatibilityError(
            "Fish ModifiedDAC checkpoint is incompatible: "
            f"missing={missing[:12]!r}, "
            f"unexpected={unexpected[:12]!r}, "
            f"shape_mismatches={mismatched[:12]!r}.")
    return tuple(sorted(expected))


def load_fish_codec_checkpoint(
    codec: FishModifiedDAC,
    path: str | Path,
    *,
    device: str | torch.device,
    dtype: torch.dtype | None = None,
) -> FishCheckpointReport:
    report = inspect_fish_checkpoint(path)
    with _open_reader(report.path) as reader:
        names = _validate_codec_layout(codec, reader)
        with torch.no_grad():
            for name in names:
                value = reader.get_tensor(name)
                target_dtype = (dtype if dtype is not None and value.is_floating_point() else value.dtype)
                codec.load_state_dict(
                    {name: value.to(
                        device=device,
                        dtype=target_dtype,
                    )},
                    strict=False,
                    assign=True,
                )
    remaining = [name for name, value in codec.state_dict().items() if value.device.type == "meta"]
    if remaining:
        raise CheckpointCompatibilityError(
            "Fish codec load left meta tensors: " + ", ".join(remaining[:12]) + ".")
    return report


def export_fish_codec_checkpoint(
    codec: FishModifiedDAC,
    path: str | Path,
    *,
    state_override: Mapping[str, Tensor] | None = None,
) -> Path:
    state = (dict(codec.state_dict()) if state_override is None else dict(state_override))
    expected = set(codec.state_dict())
    if set(state) != expected:
        raise ValueError(
            "Fish codec export is incomplete: "
            f"missing={sorted(expected - set(state))[:12]!r}, "
            f"unexpected={sorted(set(state) - expected)[:12]!r}.")
    return save_safetensors(
        {
            name: value.detach()
            for name, value in state.items()
        },
        path,
        metadata={
            "format": NATIVE_FISH_CODEC_FORMAT,
            "architecture": "fish_modified_dac",
            "producer": "voicehub",
        },
    ).resolve()


def save_fish_codec_pretrained(
    codec: FishModifiedDAC,
    directory: str | Path,
    *,
    state_override: Mapping[str, Tensor] | None = None,
) -> Path:
    destination = Path(directory).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    export_fish_codec_checkpoint(
        codec,
        destination / "model.safetensors",
        state_override=state_override,
    )
    write_json_file(destination / "config.json", codec.config.to_dict())
    write_fish_license_files(destination)
    return destination.resolve()


def _legacy_codec_state(value: Any) -> dict[str, Tensor]:
    if isinstance(value, Mapping) and "state_dict" in value:
        value = value["state_dict"]
    if not isinstance(value, Mapping):
        raise CheckpointCompatibilityError("Fish legacy codec must contain a tensor mapping.")
    items = list(value.items())
    has_generator = any(isinstance(name, str) and name.startswith("generator.") for name, _ in items)
    output: dict[str, Tensor] = {}
    for raw_name, tensor in items:
        if not isinstance(raw_name, str) or not isinstance(tensor, Tensor):
            raise CheckpointCompatibilityError("Fish legacy codec state must map string names to tensors.")
        if has_generator:
            if not raw_name.startswith("generator."):
                continue
            name = raw_name[len("generator."):]
        else:
            name = raw_name
        if name in output:
            raise CheckpointCompatibilityError(f"Fish legacy conversion produced duplicate tensor {name!r}.")
        output[name] = tensor
    return output


def convert_legacy_fish_codec(
    legacy_path: str | Path,
    output_directory: str | Path,
    *,
    trust_legacy_pickle: bool = False,
    verify_official_integrity: bool = True,
    config: FishCodecConfig | None = None,
) -> Path:
    """Convert the pinned official codec pickle exactly once.

    Conversion is never implicit.  The caller must acknowledge the
    pickle trust boundary and, by default, the byte-for-byte official
    hash is required before ``torch.load`` is reached.
    """
    if trust_legacy_pickle is not True:
        raise PermissionError(
            "Fish codec conversion reads a PyTorch pickle. Pass "
            "`trust_legacy_pickle=True` only for the pinned official "
            "artifact.")
    source = Path(legacy_path).expanduser().resolve()
    if verify_official_integrity:
        verify_file_integrity(
            source,
            expected_size=FISH_S2_LEGACY_CODEC_SIZE,
            expected_sha256=FISH_S2_LEGACY_CODEC_SHA256,
        )
    elif not source.is_file():
        raise FileNotFoundError(f"Fish codec pickle was not found: {source}.")
    try:
        payload = torch.load(
            source,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError as error:  # pragma: no cover - unsupported old PyTorch
        raise RuntimeError(
            "Fish legacy conversion requires "
            "`torch.load(..., weights_only=True)`.") from error
    state = _legacy_codec_state(payload)
    resolved_config = FishCodecConfig() if config is None else config
    with torch.device("meta"):
        codec = FishModifiedDAC(resolved_config, initialize=False)
    expected = _expected_shapes(codec)
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    mismatched = sorted((name, tuple(state[name].shape), expected[name])
                        for name in set(expected) & set(state) if tuple(state[name].shape) != expected[name])
    if missing or unexpected or mismatched:
        raise CheckpointCompatibilityError(
            "Fish legacy codec is incompatible: "
            f"missing={missing[:12]!r}, "
            f"unexpected={unexpected[:12]!r}, "
            f"shape_mismatches={mismatched[:12]!r}.")
    destination = Path(output_directory).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    export_path = save_safetensors(
        state,
        destination / "model.safetensors",
        metadata={
            "format": NATIVE_FISH_CODEC_FORMAT,
            "converted_from": "integrity-verified-pytorch-pickle",
        },
    )
    write_json_file(
        destination / "config.json",
        resolved_config.to_dict(),
    )
    write_fish_license_files(destination)
    return export_path.resolve()


def save_fish_runtime_pretrained(runtime: Any, directory: str | Path) -> Path:
    destination = Path(directory).expanduser()
    save_fish_semantic_pretrained(runtime.semantic_model, destination)
    runtime.tokenizer.save_pretrained(destination)
    save_fish_codec_pretrained(runtime.codec, destination / "codec")
    return destination.resolve()


__all__ = [
    "FishCheckpointReport",
    "convert_legacy_fish_codec",
    "export_fish_codec_checkpoint",
    "export_fish_semantic_checkpoint",
    "inspect_fish_checkpoint",
    "load_fish_codec_checkpoint",
    "load_fish_semantic_checkpoint",
    "save_fish_codec_pretrained",
    "save_fish_runtime_pretrained",
    "save_fish_semantic_pretrained",
    "verify_file_integrity",
    "write_fish_license_files",
]
