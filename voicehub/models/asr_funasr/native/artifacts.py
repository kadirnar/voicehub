"""Artifact resolution for VoiceHub-native SenseVoiceSmall."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from voicehub.hub import resolve_pretrained_file
from voicehub.models.asr_funasr.native.checkpoint import (
    NATIVE_SENSEVOICE_CMVN,
    NATIVE_SENSEVOICE_FILENAME,
    NATIVE_SENSEVOICE_TOKENIZER,
    convert_sensevoice_small_checkpoint,
)
from voicehub.models.asr_funasr.native.metadata import (
    SENSEVOICE_CHECKPOINT_FILENAME,
    SENSEVOICE_CMVN_FILENAME,
    SENSEVOICE_REPOSITORY,
    SENSEVOICE_REVISION,
    SENSEVOICE_TOKENIZER_FILENAME,
    SENSEVOICE_UPSTREAM_CONFIG_FILENAME,
)
from voicehub.path_utils import is_explicit_local_path

_OFFICIAL_ALIASES = frozenset({
    "FunAudioLLM/SenseVoiceSmall",
    "iic/SenseVoiceSmall",
    "sensevoice-small",
})


@dataclass(frozen=True, slots=True)
class SenseVoiceArtifacts:
    """One coherent native checkpoint, tokenizer, CMVN, and configuration."""

    checkpoint: Path
    tokenizer: Path
    cmvn: Path
    config: Path
    source: str
    revision: str | None
    converted_from_pickle: bool = False

    def __post_init__(self) -> None:
        for name in ("checkpoint", "tokenizer", "cmvn", "config"):
            value = Path(getattr(self, name)).expanduser().resolve()
            if not value.is_file():
                raise FileNotFoundError(f"SenseVoice {name} file was not found: {value}.")
            object.__setattr__(self, name, value)
        try:
            values = json.loads(self.config.read_text(encoding="utf-8"))
        except (
                OSError,
                UnicodeDecodeError,
                json.JSONDecodeError,
        ) as error:
            raise ValueError(f"SenseVoice config is not valid JSON: {self.config}.") from error
        if not isinstance(values, dict):
            raise TypeError("SenseVoice config root must be a JSON object.")


def _is_complete_native(root: Path) -> bool:
    return all((root / name).is_file() for name in (
        NATIVE_SENSEVOICE_FILENAME,
        NATIVE_SENSEVOICE_TOKENIZER,
        NATIVE_SENSEVOICE_CMVN,
        "config.json",
    ))


def _native_artifacts(
    root: Path,
    *,
    source: str,
    revision: str | None,
    converted_from_pickle: bool = False,
) -> SenseVoiceArtifacts:
    return SenseVoiceArtifacts(
        checkpoint=root / NATIVE_SENSEVOICE_FILENAME,
        tokenizer=root / NATIVE_SENSEVOICE_TOKENIZER,
        cmvn=root / NATIVE_SENSEVOICE_CMVN,
        config=root / "config.json",
        source=source,
        revision=revision,
        converted_from_pickle=converted_from_pickle,
    )


def _is_complete_upstream(root: Path) -> bool:
    return all((root / name).is_file() for name in (
        SENSEVOICE_CHECKPOINT_FILENAME,
        SENSEVOICE_TOKENIZER_FILENAME,
        SENSEVOICE_CMVN_FILENAME,
        SENSEVOICE_UPSTREAM_CONFIG_FILENAME,
    ))


def _local_artifacts(
    source: Path,
    *,
    trust_pickle_checkpoint: bool,
) -> SenseVoiceArtifacts:
    root = source if source.is_dir() else source.parent
    if _is_complete_native(root):
        return _native_artifacts(
            root,
            source=str(source),
            revision=None,
        )
    if not _is_complete_upstream(root):
        raise FileNotFoundError(
            f"No complete SenseVoiceSmall artifact was found in {root}. "
            "Expected a native model.safetensors/config.json/tokenizer.model/"
            "am.mvn set or the exact four-file upstream release.")
    destination = root / ".voicehub-native" / "sensevoice-small"
    if not _is_complete_native(destination):
        convert_sensevoice_small_checkpoint(
            root,
            destination,
            trust_pickle_checkpoint=trust_pickle_checkpoint,
        )
    return _native_artifacts(
        destination,
        source=str(source),
        revision=None,
        converted_from_pickle=True,
    )


def resolve_sensevoice_artifacts(
    source: str | Path,
    *,
    revision: str | None,
    cache_dir: str | Path | None,
    token: str | bool | None,
    local_files_only: bool,
    trust_pickle_checkpoint: bool,
) -> SenseVoiceArtifacts:
    """Resolve a native artifact or convert the pinned public release once."""
    source_path = Path(source).expanduser()
    if source_path.exists():
        return _local_artifacts(
            source_path.resolve(),
            trust_pickle_checkpoint=trust_pickle_checkpoint,
        )
    if is_explicit_local_path(source):
        raise FileNotFoundError(f"Local SenseVoice model path was not found: {source_path}.")
    source_name = str(source)
    if source_name not in _OFFICIAL_ALIASES:
        checkpoint = resolve_pretrained_file(
            source_name,
            NATIVE_SENSEVOICE_FILENAME,
            cache_dir=cache_dir,
            revision=revision,
            token=token,
            local_files_only=local_files_only,
        )
        for filename in (
                NATIVE_SENSEVOICE_TOKENIZER,
                NATIVE_SENSEVOICE_CMVN,
                "config.json",
        ):
            resolve_pretrained_file(
                source_name,
                filename,
                cache_dir=cache_dir,
                revision=revision,
                token=token,
                local_files_only=local_files_only,
            )
        return _native_artifacts(
            checkpoint.parent,
            source=source_name,
            revision=revision,
        )
    if revision is not None and revision != SENSEVOICE_REVISION:
        raise ValueError(
            "The official SenseVoiceSmall alias is pinned to immutable "
            f"revision {SENSEVOICE_REVISION}; use a custom native repository "
            "for another checkpoint.")
    upstream_config = resolve_pretrained_file(
        SENSEVOICE_REPOSITORY,
        SENSEVOICE_UPSTREAM_CONFIG_FILENAME,
        cache_dir=cache_dir,
        revision=SENSEVOICE_REVISION,
        token=token,
        local_files_only=local_files_only,
    )
    destination = (upstream_config.parent / ".voicehub-native" / "sensevoice-small")
    if _is_complete_native(destination):
        return _native_artifacts(
            destination,
            source=source_name,
            revision=SENSEVOICE_REVISION,
            converted_from_pickle=True,
        )
    if trust_pickle_checkpoint is not True:
        raise ValueError(
            "The official SenseVoiceSmall release is a hash-pinned "
            "pickle-based PyTorch checkpoint. Review its model license and "
            "pass `trust_pickle_checkpoint=True` for one-time restricted "
            "conversion. A cached native Safetensors conversion needs no "
            "future trust override.")
    for filename in (
            SENSEVOICE_CHECKPOINT_FILENAME,
            SENSEVOICE_TOKENIZER_FILENAME,
            SENSEVOICE_CMVN_FILENAME,
    ):
        resolve_pretrained_file(
            SENSEVOICE_REPOSITORY,
            filename,
            cache_dir=cache_dir,
            revision=SENSEVOICE_REVISION,
            token=token,
            local_files_only=local_files_only,
        )
    convert_sensevoice_small_checkpoint(
        upstream_config.parent,
        destination,
        trust_pickle_checkpoint=True,
    )
    return _native_artifacts(
        destination,
        source=source_name,
        revision=SENSEVOICE_REVISION,
        converted_from_pickle=True,
    )


__all__ = [
    "SenseVoiceArtifacts",
    "resolve_sensevoice_artifacts",
]
