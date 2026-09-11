"""Artifact resolution for the VoiceHub-native ESPnet runtime."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from voicehub.hub import resolve_pretrained_file
from voicehub.models.asr_espnet.native.checkpoint import (
    NATIVE_ESPNET_FILENAME,
    NATIVE_ESPNET_LM_FILENAME,
    NATIVE_ESPNET_TOKENIZER,
    NATIVE_ESPNET_TOKENS,
    convert_espnet_librispeech_checkpoints,
    official_espnet_conversion_kwargs,
)
from voicehub.models.asr_espnet.native.metadata import (
    ESPNET_ASR_FILENAME,
    ESPNET_CONFIG_FILENAME,
    ESPNET_LEGACY_ALIAS,
    ESPNET_LM_FILENAME,
    ESPNET_REPOSITORY,
    ESPNET_REVISION,
    ESPNET_TOKENIZER_FILENAME,
)
from voicehub.path_utils import is_explicit_local_path

_OFFICIAL_ALIASES = frozenset({
    "espnet-asr",
    "espnet-librispeech-transformer-e18",
    ESPNET_LEGACY_ALIAS,
    ESPNET_REPOSITORY,
})


@dataclass(frozen=True, slots=True)
class ESPnetArtifacts:
    """One coherent native ASR, LM, tokenizer, vocabulary, and config set."""

    checkpoint: Path
    language_model_checkpoint: Path
    tokenizer: Path
    tokens: Path
    config: Path
    source: str
    revision: str | None
    converted_from_pickle: bool = False

    def __post_init__(self) -> None:
        for name in (
                "checkpoint",
                "language_model_checkpoint",
                "tokenizer",
                "tokens",
                "config",
        ):
            value = Path(getattr(self, name)).expanduser().resolve()
            if not value.is_file():
                raise FileNotFoundError(f"ESPnet {name} file was not found: {value}.")
            object.__setattr__(self, name, value)
        try:
            values = json.loads(self.config.read_text(encoding="utf-8"))
        except (
                OSError,
                UnicodeDecodeError,
                json.JSONDecodeError,
        ) as error:
            raise ValueError(f"ESPnet native config is not valid JSON: {self.config}.") from error
        if not isinstance(values, dict):
            raise TypeError("ESPnet native config root must be a JSON object.")


def _is_complete_native(root: Path) -> bool:
    return all((root / name).is_file() for name in (
        NATIVE_ESPNET_FILENAME,
        NATIVE_ESPNET_LM_FILENAME,
        NATIVE_ESPNET_TOKENIZER,
        NATIVE_ESPNET_TOKENS,
        "config.json",
    ))


def _native_artifacts(
    root: Path,
    *,
    source: str,
    revision: str | None,
    converted_from_pickle: bool = False,
) -> ESPnetArtifacts:
    return ESPnetArtifacts(
        checkpoint=root / NATIVE_ESPNET_FILENAME,
        language_model_checkpoint=root / NATIVE_ESPNET_LM_FILENAME,
        tokenizer=root / NATIVE_ESPNET_TOKENIZER,
        tokens=root / NATIVE_ESPNET_TOKENS,
        config=root / "config.json",
        source=source,
        revision=revision,
        converted_from_pickle=converted_from_pickle,
    )


def _upstream_files(root: Path) -> dict[str, Path] | None:
    nested = {
        "asr": root / ESPNET_ASR_FILENAME,
        "lm": root / ESPNET_LM_FILENAME,
        "tokenizer": root / ESPNET_TOKENIZER_FILENAME,
        "config": root / ESPNET_CONFIG_FILENAME,
    }
    if all(path.is_file() for path in nested.values()):
        return nested
    flat = {
        "asr": root / "54epoch.pth",
        "lm": root / "17epoch.pth",
        "tokenizer": root / "bpe.model",
        "config": root / "config.yaml",
    }
    return flat if all(path.is_file() for path in flat.values()) else None


def _local_artifacts(
    source: Path,
    *,
    trust_pickle_checkpoint: bool,
) -> ESPnetArtifacts:
    root = source if source.is_dir() else source.parent
    if _is_complete_native(root):
        return _native_artifacts(
            root,
            source=str(source),
            revision=None,
        )
    upstream = _upstream_files(root)
    if upstream is None:
        raise FileNotFoundError(
            f"No complete native or audited upstream ESPnet artifact was "
            f"found in {root}.")
    destination = root / ".voicehub-native" / ("espnet-librispeech-transformer-e18")
    if not _is_complete_native(destination):
        convert_espnet_librispeech_checkpoints(
            asr_checkpoint=upstream["asr"],
            language_model_checkpoint=upstream["lm"],
            tokenizer_model=upstream["tokenizer"],
            config_yaml=upstream["config"],
            destination=destination,
            trust_pickle_checkpoint=trust_pickle_checkpoint,
            **official_espnet_conversion_kwargs(),
        )
    return _native_artifacts(
        destination,
        source=str(source),
        revision=None,
        converted_from_pickle=True,
    )


def _snapshot_root(path: Path, filename: str) -> Path:
    root = path
    for _ in Path(filename).parts:
        root = root.parent
    return root


def resolve_espnet_artifacts(
    source: str | Path,
    *,
    revision: str | None,
    cache_dir: str | Path | None,
    token: str | bool | None,
    local_files_only: bool,
    trust_pickle_checkpoint: bool,
) -> ESPnetArtifacts:
    """Resolve a native artifact or convert the exact public release once."""
    source_path = Path(source).expanduser()
    if source_path.exists():
        return _local_artifacts(
            source_path.resolve(),
            trust_pickle_checkpoint=trust_pickle_checkpoint,
        )
    if is_explicit_local_path(source):
        raise FileNotFoundError(f"Local ESPnet model path was not found: {source_path}.")
    source_name = str(source)
    if source_name not in _OFFICIAL_ALIASES:
        checkpoint = resolve_pretrained_file(
            source_name,
            NATIVE_ESPNET_FILENAME,
            cache_dir=cache_dir,
            revision=revision,
            token=token,
            local_files_only=local_files_only,
        )
        for filename in (
                NATIVE_ESPNET_LM_FILENAME,
                NATIVE_ESPNET_TOKENIZER,
                NATIVE_ESPNET_TOKENS,
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
    if revision is not None and revision != ESPNET_REVISION:
        raise ValueError(
            "The official ESPnet alias is pinned to immutable revision "
            f"{ESPNET_REVISION}; use a custom native repository for another "
            "checkpoint.")
    config_path = resolve_pretrained_file(
        ESPNET_REPOSITORY,
        ESPNET_CONFIG_FILENAME,
        cache_dir=cache_dir,
        revision=ESPNET_REVISION,
        token=token,
        local_files_only=local_files_only,
    )
    snapshot = _snapshot_root(config_path, ESPNET_CONFIG_FILENAME)
    destination = snapshot / ".voicehub-native" / ("espnet-librispeech-transformer-e18")
    if _is_complete_native(destination):
        return _native_artifacts(
            destination,
            source=source_name,
            revision=ESPNET_REVISION,
            converted_from_pickle=True,
        )
    if trust_pickle_checkpoint is not True:
        raise ValueError(
            "The official ESPnet release publishes pickle-based ASR and LM "
            "checkpoints. Review the pinned CC-BY-4.0 artifact and pass "
            "`trust_pickle_checkpoint=True` for one-time restricted "
            "conversion. Cached Safetensors reloads require no trust flag.")
    upstream = {
        "config":
        config_path,
        "asr":
        resolve_pretrained_file(
            ESPNET_REPOSITORY,
            ESPNET_ASR_FILENAME,
            cache_dir=cache_dir,
            revision=ESPNET_REVISION,
            token=token,
            local_files_only=local_files_only,
        ),
        "lm":
        resolve_pretrained_file(
            ESPNET_REPOSITORY,
            ESPNET_LM_FILENAME,
            cache_dir=cache_dir,
            revision=ESPNET_REVISION,
            token=token,
            local_files_only=local_files_only,
        ),
        "tokenizer":
        resolve_pretrained_file(
            ESPNET_REPOSITORY,
            ESPNET_TOKENIZER_FILENAME,
            cache_dir=cache_dir,
            revision=ESPNET_REVISION,
            token=token,
            local_files_only=local_files_only,
        ),
    }
    convert_espnet_librispeech_checkpoints(
        asr_checkpoint=upstream["asr"],
        language_model_checkpoint=upstream["lm"],
        tokenizer_model=upstream["tokenizer"],
        config_yaml=upstream["config"],
        destination=destination,
        trust_pickle_checkpoint=True,
        **official_espnet_conversion_kwargs(),
    )
    return _native_artifacts(
        destination,
        source=source_name,
        revision=ESPNET_REVISION,
        converted_from_pickle=True,
    )


__all__ = [
    "ESPnetArtifacts",
    "resolve_espnet_artifacts",
]
