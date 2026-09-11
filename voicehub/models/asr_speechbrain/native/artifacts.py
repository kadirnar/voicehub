"""Artifact resolution for VoiceHub-native SpeechBrain CRDNN ASR."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from voicehub.hub import resolve_pretrained_file
from voicehub.models.asr_speechbrain.native.checkpoint import (
    NATIVE_SPEECHBRAIN_ASR_FILENAME,
    NATIVE_SPEECHBRAIN_ASR_TOKENIZER,
    convert_speechbrain_asr_checkpoints,
    official_speechbrain_asr_conversion_kwargs,
)
from voicehub.models.asr_speechbrain.native.metadata import SPEECHBRAIN_ASR_REPOSITORY, SPEECHBRAIN_ASR_REVISION
from voicehub.path_utils import is_explicit_local_path

_OFFICIAL_ALIASES = frozenset({
    "speechbrain-asr",
    "speechbrain-crdnn-asr",
    SPEECHBRAIN_ASR_REPOSITORY,
})


@dataclass(frozen=True, slots=True)
class SpeechBrainASRArtifacts:
    """Coherent native runtime assets."""

    checkpoint: Path
    config: Path
    tokenizer: Path
    source: str
    revision: str | None
    converted_from_pickle: bool = False

    def __post_init__(self) -> None:
        for name in ("checkpoint", "config", "tokenizer"):
            value = Path(getattr(self, name)).expanduser().resolve()
            if not value.is_file():
                raise FileNotFoundError(f"SpeechBrain ASR {name} file was not found: {value}.")
            object.__setattr__(self, name, value)
        try:
            values = json.loads(self.config.read_text(encoding="utf-8"), )
        except (
                OSError,
                UnicodeDecodeError,
                json.JSONDecodeError,
        ) as error:
            raise ValueError(f"SpeechBrain ASR config is not valid JSON: {self.config}.") from error
        if not isinstance(values, dict):
            raise TypeError("SpeechBrain ASR config root must be a JSON object.")


def _native_artifacts(
    root: Path,
    *,
    source: str,
    revision: str | None,
    converted_from_pickle: bool = False,
) -> SpeechBrainASRArtifacts:
    return SpeechBrainASRArtifacts(
        checkpoint=root / NATIVE_SPEECHBRAIN_ASR_FILENAME,
        config=root / "config.json",
        tokenizer=root / NATIVE_SPEECHBRAIN_ASR_TOKENIZER,
        source=source,
        revision=revision,
        converted_from_pickle=converted_from_pickle,
    )


def _is_complete_native_artifact(root: Path) -> bool:
    return ((root / NATIVE_SPEECHBRAIN_ASR_FILENAME).is_file() and (root / "config.json").is_file() and
            (root / NATIVE_SPEECHBRAIN_ASR_TOKENIZER).is_file())


def _local_artifacts(
    source: Path,
    *,
    trust_pickle_checkpoint: bool,
) -> SpeechBrainASRArtifacts:
    root = source if source.is_dir() else source.parent
    native_root = root
    if _is_complete_native_artifact(native_root):
        return _native_artifacts(
            native_root,
            source=str(source),
            revision=None,
        )
    legacy = {
        "asr": root / "asr.ckpt",
        "lm": root / "lm.ckpt",
        "normalizer": root / "normalizer.ckpt",
        "tokenizer": root / "tokenizer.ckpt",
    }
    if all(path.is_file() for path in legacy.values()):
        destination = (root / ".voicehub-native" / "speechbrain-crdnn-asr")
        if _is_complete_native_artifact(destination):
            return _native_artifacts(
                destination,
                source=str(source),
                revision=None,
                converted_from_pickle=True,
            )
        if trust_pickle_checkpoint is not True:
            raise ValueError(
                "The local SpeechBrain ASR artifact contains pickle-based "
                "checkpoints. Review them and pass "
                "`trust_pickle_checkpoint=True` for one-time conversion.")
        hyperparams = root / "hyperparams.yaml"
        convert_speechbrain_asr_checkpoints(
            asr_checkpoint=legacy["asr"],
            lm_checkpoint=legacy["lm"],
            normalizer_checkpoint=legacy["normalizer"],
            tokenizer_model=legacy["tokenizer"],
            destination=destination,
            hyperparams_file=(hyperparams if hyperparams.is_file() else None),
            trust_pickle_checkpoint=True,
        )
        return _native_artifacts(
            destination,
            source=str(source),
            revision=None,
            converted_from_pickle=True,
        )
    raise FileNotFoundError(
        f"No complete native SpeechBrain ASR artifact was found in {root}. "
        "Expected `model.safetensors`, `config.json`, and "
        "`tokenizer.model`, or all four reviewed upstream checkpoint files.")


def resolve_speechbrain_asr_artifacts(
    source: str | Path,
    *,
    revision: str | None,
    cache_dir: str | Path | None,
    token: str | bool | None,
    local_files_only: bool,
    trust_pickle_checkpoint: bool,
) -> SpeechBrainASRArtifacts:
    """Resolve a native artifact or convert the pinned public release once."""
    source_path = Path(source).expanduser()
    if source_path.exists():
        return _local_artifacts(
            source_path.resolve(),
            trust_pickle_checkpoint=trust_pickle_checkpoint,
        )
    if is_explicit_local_path(source):
        raise FileNotFoundError(f"Local SpeechBrain ASR model path was not found: {source_path}.")
    source_name = str(source)
    if source_name not in _OFFICIAL_ALIASES:
        root_file = resolve_pretrained_file(
            source_name,
            NATIVE_SPEECHBRAIN_ASR_FILENAME,
            cache_dir=cache_dir,
            revision=revision,
            token=token,
            local_files_only=local_files_only,
        )
        resolve_pretrained_file(
            source_name,
            "config.json",
            cache_dir=cache_dir,
            revision=revision,
            token=token,
            local_files_only=local_files_only,
        )
        resolve_pretrained_file(
            source_name,
            NATIVE_SPEECHBRAIN_ASR_TOKENIZER,
            cache_dir=cache_dir,
            revision=revision,
            token=token,
            local_files_only=local_files_only,
        )
        return _native_artifacts(
            root_file.parent,
            source=source_name,
            revision=revision,
        )
    if revision is not None and revision != SPEECHBRAIN_ASR_REVISION:
        raise ValueError(
            "The official SpeechBrain ASR alias is pinned to immutable "
            f"revision {SPEECHBRAIN_ASR_REVISION}; use a custom native "
            "repository for another revision.")
    if trust_pickle_checkpoint is not True:
        # Acknowledgement is needed only while deserializing the original
        # pickle states. A cached conversion is a normal native artifact and
        # must remain usable by later processes without weakening that gate.
        try:
            cached_asr = resolve_pretrained_file(
                SPEECHBRAIN_ASR_REPOSITORY,
                "asr.ckpt",
                cache_dir=cache_dir,
                revision=SPEECHBRAIN_ASR_REVISION,
                token=token,
                local_files_only=True,
            )
        except FileNotFoundError:
            cached_asr = None
        if cached_asr is not None:
            cached_destination = (cached_asr.parent / ".voicehub-native" / "speechbrain-crdnn-asr")
            if _is_complete_native_artifact(cached_destination):
                return _native_artifacts(
                    cached_destination,
                    source=source_name,
                    revision=SPEECHBRAIN_ASR_REVISION,
                    converted_from_pickle=True,
                )
        raise ValueError(
            "The official SpeechBrain ASR release publishes three "
            "pickle-based checkpoints. Review the Apache-2.0 artifact and "
            "pass `trust_pickle_checkpoint=True` for its one-time "
            "conversion; later loads use the cached Safetensors artifact.")
    files = {
        name:
        resolve_pretrained_file(
            SPEECHBRAIN_ASR_REPOSITORY,
            filename,
            cache_dir=cache_dir,
            revision=SPEECHBRAIN_ASR_REVISION,
            token=token,
            local_files_only=local_files_only,
        )
        for name, filename in (
            ("asr", "asr.ckpt"),
            ("lm", "lm.ckpt"),
            ("normalizer", "normalizer.ckpt"),
            ("tokenizer", "tokenizer.ckpt"),
            ("hyperparams", "hyperparams.yaml"),
        )
    }
    destination = (files["asr"].parent / ".voicehub-native" / "speechbrain-crdnn-asr")
    if not _is_complete_native_artifact(destination):
        convert_speechbrain_asr_checkpoints(
            asr_checkpoint=files["asr"],
            lm_checkpoint=files["lm"],
            normalizer_checkpoint=files["normalizer"],
            tokenizer_model=files["tokenizer"],
            destination=destination,
            hyperparams_file=files["hyperparams"],
            trust_pickle_checkpoint=True,
            **official_speechbrain_asr_conversion_kwargs(),
        )
    return _native_artifacts(
        destination,
        source=source_name,
        revision=SPEECHBRAIN_ASR_REVISION,
        converted_from_pickle=True,
    )


__all__ = [
    "SpeechBrainASRArtifacts",
    "resolve_speechbrain_asr_artifacts",
]
