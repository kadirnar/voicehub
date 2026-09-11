"""Artifact resolution and strict loading for native Silero VAD."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from voicehub.checkpointing.adapters import CheckpointAdapter
from voicehub.checkpointing.safetensors import SafeTensorReader
from voicehub.checkpointing.transforms import CopyTensor, TensorPlan
from voicehub.hub import resolve_pretrained_file
from voicehub.models.vad_silero.native.checkpoint import (
    OfficialSileroVADSafeTensorsCheckpointAdapter,
    OfficialSileroVADTorchScriptCheckpointAdapter,
    native_silero_vad_tensor_names,
)
from voicehub.models.vad_silero.native.configuration import SileroVADConfig as NativeSileroVADConfig
from voicehub.path_utils import is_explicit_local_path

NATIVE_SILERO_VAD_FORMAT = "voicehub-native-silero-vad-v1"
NATIVE_SILERO_VAD_FILENAME = "model.safetensors"
DEFAULT_SILERO_VAD_REPOSITORY = "safestack/silero-vad"


@dataclass(frozen=True, slots=True)
class SileroVADArtifact:
    """One resolved immutable checkpoint and its provenance."""

    checkpoint: Path
    source: str
    revision: str | None
    checkpoint_format: str

    def __post_init__(self) -> None:
        checkpoint = Path(self.checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Silero VAD checkpoint was not found: {checkpoint}.")
        if self.checkpoint_format not in {"safetensors", "torchscript"}:
            raise ValueError("`checkpoint_format` must be 'safetensors' or 'torchscript'.")
        object.__setattr__(self, "checkpoint", checkpoint)


class NativeSileroVADCheckpointAdapter(CheckpointAdapter):
    """Strictly load a VoiceHub-exported 8 kHz or 16 kHz state dict."""

    architecture_id = "silero-vad"
    adapter_id = "voicehub-native-silero-vad"
    adapter_version = "1"

    def probe(
        self,
        files: tuple[Path, ...],
        config: Mapping[str, Any],
    ) -> bool:
        try:
            NativeSileroVADConfig.coerce(config)
        except (TypeError, ValueError):
            return False
        return any(path.suffix.lower() == ".safetensors" for path in files)

    def tensor_plan(self, config: Mapping[str, Any]) -> TensorPlan:
        resolved = NativeSileroVADConfig.coerce(config)
        return TensorPlan(
            rules=tuple(CopyTensor(name, name) for name in native_silero_vad_tensor_names(resolved)))


def _checkpoint_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".safetensors":
        return "safetensors"
    if suffix == ".jit":
        return "torchscript"
    raise ValueError("Native Silero checkpoints must be `.safetensors` or an official "
                     "`.jit` archive.")


def _local_candidates(
    source: Path,
    *,
    sample_rate: int,
    checkpoint_filename: str | None,
) -> tuple[Path, ...]:
    if checkpoint_filename is not None:
        relative = Path(checkpoint_filename)
        return (
            source / relative,
            source / "native_export" / relative,
        )
    native_names = (
        Path("native_export") / NATIVE_SILERO_VAD_FILENAME,
        Path(NATIVE_SILERO_VAD_FILENAME),
        Path("native_export") / f"silero_vad_{sample_rate // 1_000}k.safetensors",
        Path(f"silero_vad_{sample_rate // 1_000}k.safetensors"),
        Path("data") / f"silero_vad_{sample_rate // 1_000}k.safetensors",
    )
    if sample_rate == 16_000:
        native_names += (Path("data") / "silero_vad_16k.safetensors", )
    return tuple(
        source / relative for relative in (
            *native_names,
            Path("native_export") / "silero_vad.jit",
            Path("silero_vad.jit"),
            Path("data") / "silero_vad.jit",
        ))


def _remote_filename(
    *,
    sample_rate: int,
    checkpoint_filename: str | None,
) -> tuple[str, str]:
    value = checkpoint_filename
    if value is None:
        value = ("data/silero_vad_16k.safetensors" if sample_rate == 16_000 else "data/silero_vad.jit")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.name in {"", ".", ".."}:
        raise ValueError("`checkpoint_filename` must be a safe repository-relative path.")
    subfolder = "" if str(path.parent) == "." else path.parent.as_posix()
    return path.name, subfolder


def resolve_silero_vad_artifact(
    pretrained_model_name_or_path: str | Path,
    *,
    sample_rate: int,
    checkpoint_filename: str | None = None,
    cache_dir: str | None = None,
    revision: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
) -> SileroVADArtifact:
    """Resolve a local export, official mirror, or explicit checkpoint."""
    native_config = NativeSileroVADConfig(sampling_rate=sample_rate)
    del native_config
    source_value = str(pretrained_model_name_or_path)
    source = Path(pretrained_model_name_or_path).expanduser()
    if source.is_file():
        checkpoint = source
        resolved_revision = None
    elif source.is_dir():
        checkpoint = next(
            (
                candidate for candidate in _local_candidates(
                    source,
                    sample_rate=sample_rate,
                    checkpoint_filename=checkpoint_filename,
                ) if candidate.is_file()),
            None,
        )
        if checkpoint is None:
            expected = ", ".join(
                str(path.relative_to(source)) for path in _local_candidates(
                    source,
                    sample_rate=sample_rate,
                    checkpoint_filename=checkpoint_filename,
                ))
            raise FileNotFoundError(
                f"No native Silero checkpoint was found in {source}. "
                f"Checked: {expected}.")
        resolved_revision = None
    else:
        if is_explicit_local_path(pretrained_model_name_or_path):
            raise FileNotFoundError(f"Local Silero model path was not found: {source}.")
        filename, subfolder = _remote_filename(
            sample_rate=sample_rate,
            checkpoint_filename=checkpoint_filename,
        )
        checkpoint = resolve_pretrained_file(
            source_value,
            filename,
            subfolder=subfolder,
            cache_dir=cache_dir,
            revision=revision,
            token=token,
            local_files_only=local_files_only,
        )
        resolved_revision = revision

    return SileroVADArtifact(
        checkpoint=checkpoint,
        source=source_value,
        revision=resolved_revision,
        checkpoint_format=_checkpoint_format(checkpoint),
    )


def load_silero_vad_checkpoint(
    model: Any,
    artifact: SileroVADArtifact,
    config: NativeSileroVADConfig,
) -> tuple[str, str]:
    """Strictly load an artifact and return ``(format, adapter_id)``."""
    if not isinstance(artifact, SileroVADArtifact):
        raise TypeError("`artifact` must be a SileroVADArtifact.")
    config = NativeSileroVADConfig.coerce(config)
    config_values = config.to_dict()
    if artifact.checkpoint_format == "safetensors":
        with SafeTensorReader(artifact.checkpoint) as reader:
            declared_format = reader.metadata.get("format")
            declared_rate = reader.metadata.get("sample_rate")
            if declared_format is not None and declared_format != NATIVE_SILERO_VAD_FORMAT:
                raise ValueError(
                    "Silero Safetensors declares an unsupported VoiceHub "
                    f"format {declared_format!r}.")
            if declared_rate is not None and declared_rate != str(config.sampling_rate):
                raise ValueError(
                    "Silero checkpoint sample-rate mismatch: artifact declares "
                    f"{declared_rate}, model requires {config.sampling_rate}.")
            if declared_format == NATIVE_SILERO_VAD_FORMAT:
                adapter: CheckpointAdapter = NativeSileroVADCheckpointAdapter()
            else:
                adapter = OfficialSileroVADSafeTensorsCheckpointAdapter()
            report = adapter.load_streaming(
                model,
                reader,
                config_values,
                strict=True,
            )
        return "safetensors", report.adapter

    # TorchScript is accepted as a weight container for the official merged
    # 8/16 kHz archive. VoiceHub never invokes the upstream scripted graph.
    import torch

    scripted = torch.jit.load(
        str(artifact.checkpoint),
        map_location="cpu",
    )
    try:
        source_state = scripted.state_dict()
    finally:
        del scripted
    report = OfficialSileroVADTorchScriptCheckpointAdapter().load(
        model,
        source_state,
        config_values,
        strict=True,
    )
    return "torchscript-state-dict", report.adapter


__all__ = [
    "DEFAULT_SILERO_VAD_REPOSITORY",
    "NATIVE_SILERO_VAD_FILENAME",
    "NATIVE_SILERO_VAD_FORMAT",
    "NativeSileroVADCheckpointAdapter",
    "SileroVADArtifact",
    "load_silero_vad_checkpoint",
    "resolve_silero_vad_artifact",
]
