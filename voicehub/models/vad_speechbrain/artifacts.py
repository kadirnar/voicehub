"""Artifact resolution for VoiceHub-native SpeechBrain CRDNN VAD."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from voicehub.hub import resolve_pretrained_file
from voicehub.models.vad_speechbrain.native.checkpoint import (
    NATIVE_SPEECHBRAIN_VAD_FILENAME,
    convert_speechbrain_vad_checkpoint,
)
from voicehub.models.vad_speechbrain.native.metadata import (
    SPEECHBRAIN_VAD_HPARAMS_SHA256,
    SPEECHBRAIN_VAD_MODEL_SHA256,
    SPEECHBRAIN_VAD_REPOSITORY,
    SPEECHBRAIN_VAD_REVISION,
)
from voicehub.path_utils import is_explicit_local_path

_OFFICIAL_ALIASES = frozenset({
    "speechbrain-vad",
    "speechbrain/vad-crdnn-libriparty",
    "vad-crdnn-libriparty",
})


@dataclass(frozen=True, slots=True)
class SpeechBrainVADArtifacts:
    checkpoint: Path
    config: Path
    source: str
    revision: str | None
    converted_from_pickle: bool = False

    def __post_init__(self) -> None:
        for name in ("checkpoint", "config"):
            value = Path(getattr(self, name)).expanduser().resolve()
            if not value.is_file():
                raise FileNotFoundError(f"SpeechBrain VAD {name} file was not found: {value}.")
            object.__setattr__(self, name, value)
        try:
            values = json.loads(self.config.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"SpeechBrain VAD config is not valid JSON: {self.config}.") from error
        if not isinstance(values, dict):
            raise TypeError("SpeechBrain VAD config root must be a JSON object.")


def _local_artifacts(
    source: Path,
    *,
    trust_pickle_checkpoint: bool,
) -> SpeechBrainVADArtifacts:
    root = source if source.is_dir() else source.parent
    checkpoint = (
        source if source.is_file() and source.suffix == ".safetensors" else root /
        NATIVE_SPEECHBRAIN_VAD_FILENAME)
    config = root / "config.json"
    if checkpoint.is_file() and config.is_file():
        return SpeechBrainVADArtifacts(
            checkpoint=checkpoint,
            config=config,
            source=str(source),
            revision=None,
        )
    pickle_checkpoint = (source if source.is_file() and source.name == "model.ckpt" else root / "model.ckpt")
    hyperparams = root / "hyperparams.yaml"
    if pickle_checkpoint.is_file():
        destination = root / ".voicehub-native" / "speechbrain-crdnn-vad"
        converted_checkpoint = destination / NATIVE_SPEECHBRAIN_VAD_FILENAME
        converted_config = destination / "config.json"
        if converted_checkpoint.is_file() and converted_config.is_file():
            return SpeechBrainVADArtifacts(
                checkpoint=converted_checkpoint,
                config=converted_config,
                source=str(source),
                revision=None,
                converted_from_pickle=True,
            )
        if trust_pickle_checkpoint is not True:
            raise ValueError(
                "The local SpeechBrain artifact contains pickle-based "
                "`model.ckpt`. Review it and pass "
                "`trust_pickle_checkpoint=True` for one-time conversion.")
        convert_speechbrain_vad_checkpoint(
            pickle_checkpoint,
            destination,
            hyperparams_file=(hyperparams if hyperparams.is_file() else None),
            trust_pickle_checkpoint=True,
        )
        return SpeechBrainVADArtifacts(
            checkpoint=destination / NATIVE_SPEECHBRAIN_VAD_FILENAME,
            config=destination / "config.json",
            source=str(source),
            revision=None,
            converted_from_pickle=True,
        )
    raise FileNotFoundError(
        f"No complete native SpeechBrain VAD artifact was found in {root}. "
        "Expected `model.safetensors` plus `config.json`, or a reviewed "
        "`model.ckpt` for explicit conversion.")


def resolve_speechbrain_vad_artifacts(
    source: str | Path,
    *,
    revision: str | None,
    cache_dir: str | Path | None,
    token: str | bool | None,
    local_files_only: bool,
    trust_pickle_checkpoint: bool,
) -> SpeechBrainVADArtifacts:
    """Resolve native files or convert the pinned official artifact once."""
    source_path = Path(source).expanduser()
    if source_path.exists():
        return _local_artifacts(
            source_path.resolve(),
            trust_pickle_checkpoint=trust_pickle_checkpoint,
        )
    if is_explicit_local_path(source):
        raise FileNotFoundError(f"Local SpeechBrain VAD model path was not found: {source_path}.")

    source_name = str(source)
    if source_name not in _OFFICIAL_ALIASES:
        checkpoint = resolve_pretrained_file(
            source_name,
            NATIVE_SPEECHBRAIN_VAD_FILENAME,
            cache_dir=cache_dir,
            revision=revision,
            token=token,
            local_files_only=local_files_only,
        )
        config = resolve_pretrained_file(
            source_name,
            "config.json",
            cache_dir=cache_dir,
            revision=revision,
            token=token,
            local_files_only=local_files_only,
        )
        return SpeechBrainVADArtifacts(
            checkpoint=checkpoint,
            config=config,
            source=source_name,
            revision=revision,
        )

    if revision is not None and revision != SPEECHBRAIN_VAD_REVISION:
        raise ValueError(
            "The official SpeechBrain VAD alias is pinned to immutable "
            f"revision {SPEECHBRAIN_VAD_REVISION}; use a custom native "
            "repository for another revision.")
    if trust_pickle_checkpoint is not True:
        raise ValueError(
            "The official SpeechBrain VAD artifact publishes pickle-based "
            "`model.ckpt`, and its model card does not declare a checkpoint "
            "license. Review the artifact and pass "
            "`trust_pickle_checkpoint=True`; conversion is performed only "
            "when the snapshot-local Safetensors artifact is absent.")
    checkpoint = resolve_pretrained_file(
        SPEECHBRAIN_VAD_REPOSITORY,
        "model.ckpt",
        cache_dir=cache_dir,
        revision=SPEECHBRAIN_VAD_REVISION,
        token=token,
        local_files_only=local_files_only,
    )
    destination = checkpoint.parent / ".voicehub-native" / "speechbrain-crdnn-vad"
    safe_checkpoint = destination / NATIVE_SPEECHBRAIN_VAD_FILENAME
    config = destination / "config.json"
    if not safe_checkpoint.is_file() or not config.is_file():
        hyperparams = resolve_pretrained_file(
            SPEECHBRAIN_VAD_REPOSITORY,
            "hyperparams.yaml",
            cache_dir=cache_dir,
            revision=SPEECHBRAIN_VAD_REVISION,
            token=token,
            local_files_only=local_files_only,
        )
        convert_speechbrain_vad_checkpoint(
            checkpoint,
            destination,
            hyperparams_file=hyperparams,
            trust_pickle_checkpoint=True,
            expected_checkpoint_sha256=SPEECHBRAIN_VAD_MODEL_SHA256,
            expected_hyperparams_sha256=SPEECHBRAIN_VAD_HPARAMS_SHA256,
        )
    return SpeechBrainVADArtifacts(
        checkpoint=safe_checkpoint,
        config=config,
        source=source_name,
        revision=SPEECHBRAIN_VAD_REVISION,
        converted_from_pickle=True,
    )


__all__ = [
    "SpeechBrainVADArtifacts",
    "resolve_speechbrain_vad_artifacts",
]
