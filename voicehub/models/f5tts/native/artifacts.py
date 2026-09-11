"""Coherent artifact resolution for native F5-TTS."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from voicehub.hub import resolve_pretrained_file
from voicehub.models.f5tts.native.checkpoint import (
    convert_legacy_f5tts_checkpoint,
    convert_legacy_vocos_checkpoint,
    require_file_integrity,
)
from voicehub.models.f5tts.native.metadata import (
    F5TTS_CHECKPOINT_REPOSITORY,
    F5TTS_CHECKPOINT_REVISION,
    F5TTS_V1_BASE_CHECKPOINT,
    F5TTS_V1_BASE_SHA256,
    F5TTS_V1_BASE_SIZE,
    F5TTS_V1_BASE_VOCABULARY,
    VOCOS_CHECKPOINT_REVISION,
    VOCOS_LEGACY_CHECKPOINT,
    VOCOS_LEGACY_SHA256,
    VOCOS_LEGACY_SIZE,
    VOCOS_REPOSITORY,
)
from voicehub.path_utils import is_explicit_local_path

_OFFICIAL_MODEL_FILES = {
    "f5tts_v1_base": (
        F5TTS_V1_BASE_CHECKPOINT,
        F5TTS_V1_BASE_VOCABULARY,
    ),
}


@dataclass(frozen=True, slots=True)
class F5TTSArtifacts:
    source: str | Path
    revision: str | None
    checkpoint: Path
    vocabulary: Path
    vocoder: Path | None
    checkpoint_is_official: bool
    vocoder_is_official: bool


def _model_key(model_name: str) -> str:
    return model_name.strip().lower().replace("-", "_")


def _as_existing_file(value: str | Path, *, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"{label} was not found: {path}.")
    return path.resolve()


def _find_local_file(
        root: Path,
        relative_name: str,
        *,
        alternatives: tuple[str, ...] = (),
        label: str,
) -> Path:
    names = (relative_name, *alternatives)
    matches = [root / name for name in names if (root / name).is_file()]
    unique = tuple(dict.fromkeys(path.resolve() for path in matches))
    if len(unique) != 1:
        found = ", ".join(str(path.relative_to(root)) for path in unique) or "none"
        raise FileNotFoundError(
            f"F5-TTS local artifact directory must resolve exactly one "
            f"{label}; found {found}.")
    return unique[0]


def _native_checkpoint(path: Path) -> Path:
    if path.suffix.lower() == ".safetensors":
        return path
    if path.suffix.lower() not in {".pt", ".pth", ".ckpt", ".bin"}:
        raise ValueError(
            "F5-TTS checkpoints must be Safetensors or a supported legacy "
            "PyTorch weight file.")
    return convert_legacy_f5tts_checkpoint(
        path,
        path.with_suffix(".safetensors"),
    )


def _native_vocoder(path: Path) -> Path:
    if path.suffix.lower() == ".safetensors":
        return path
    if path.suffix.lower() not in {".pt", ".pth", ".bin"}:
        raise ValueError("Vocos checkpoints must be Safetensors or a legacy PyTorch "
                         "weight file.")
    return convert_legacy_vocos_checkpoint(
        path,
        path.with_suffix(".safetensors"),
    )


def resolve_f5tts_artifacts(
    source: str | Path,
    *,
    model_name: str = "F5TTS_v1_Base",
    checkpoint_path: str | Path | None = None,
    vocabulary_path: str | Path | None = None,
    vocoder_path: str | Path | None = None,
    include_vocoder: bool = True,
    revision: str | None = None,
    cache_dir: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
    verify_integrity: bool = False,
) -> F5TTSArtifacts:
    """Resolve all files without importing Hub or model framework clients."""
    if not isinstance(include_vocoder, bool):
        raise TypeError("`include_vocoder` must be a boolean.")
    key = _model_key(model_name)
    official_layout = _OFFICIAL_MODEL_FILES.get(key)
    source_path = Path(source).expanduser()
    source_string = str(source)
    alias_source = key in _OFFICIAL_MODEL_FILES and source_string == model_name

    if source_path.exists() or checkpoint_path is not None:
        if checkpoint_path is not None:
            checkpoint = _as_existing_file(
                checkpoint_path,
                label="F5-TTS checkpoint",
            )
            root = checkpoint.parent
        elif source_path.is_file():
            checkpoint = source_path.resolve()
            root = checkpoint.parent
        elif official_layout is not None:
            root = source_path.resolve()
            checkpoint = _find_local_file(
                root,
                official_layout[0],
                alternatives=(
                    Path(official_layout[0]).name,
                    "model.safetensors",
                    "model_ema.safetensors",
                ),
                label="checkpoint",
            )
        else:
            root = source_path.resolve()
            checkpoint = _find_local_file(
                root,
                "model.safetensors",
                alternatives=(
                    "model_ema.safetensors",
                    "model.pt",
                    "model.pth",
                    "model.ckpt",
                ),
                label="checkpoint",
            )
        checkpoint = _native_checkpoint(checkpoint)
        if vocabulary_path is not None:
            vocabulary = _as_existing_file(
                vocabulary_path,
                label="F5-TTS vocabulary",
            )
        elif official_layout is not None:
            vocabulary = _find_local_file(
                root,
                official_layout[1],
                alternatives=("vocab.txt", ),
                label="vocabulary",
            )
        else:
            vocabulary = _find_local_file(
                root,
                "vocab.txt",
                label="vocabulary",
            )
        resolved_revision = None
        checkpoint_official = False
    else:
        if is_explicit_local_path(source):
            raise FileNotFoundError(f"F5-TTS model path was not found: {source_path}.")
        if official_layout is None:
            raise ValueError(
                "Remote custom F5-TTS checkpoints require a local converted "
                "artifact directory; only pinned released layouts are inferred.")
        repo_id = F5TTS_CHECKPOINT_REPOSITORY if alias_source else source_string
        resolved_revision = (
            revision or (F5TTS_CHECKPOINT_REVISION if repo_id == F5TTS_CHECKPOINT_REPOSITORY else None))
        checkpoint_relative, vocabulary_relative = official_layout
        checkpoint = resolve_pretrained_file(
            repo_id,
            Path(checkpoint_relative).name,
            subfolder=str(Path(checkpoint_relative).parent),
            cache_dir=cache_dir,
            revision=resolved_revision,
            token=token,
            local_files_only=local_files_only,
        )
        vocabulary = resolve_pretrained_file(
            repo_id,
            Path(vocabulary_relative).name,
            subfolder=str(Path(vocabulary_relative).parent),
            cache_dir=cache_dir,
            revision=resolved_revision,
            token=token,
            local_files_only=local_files_only,
        )
        checkpoint_official = (
            repo_id == F5TTS_CHECKPOINT_REPOSITORY and resolved_revision == F5TTS_CHECKPOINT_REVISION and
            checkpoint_relative == F5TTS_V1_BASE_CHECKPOINT)
    if verify_integrity and checkpoint_official:
        require_file_integrity(
            checkpoint,
            sha256=F5TTS_V1_BASE_SHA256,
            size=F5TTS_V1_BASE_SIZE,
        )

    vocoder: Path | None = None
    vocoder_official = False
    if include_vocoder:
        if vocoder_path is not None:
            vocoder_source = _as_existing_file(
                vocoder_path,
                label="Vocos checkpoint",
            )
        else:
            vocoder_source = resolve_pretrained_file(
                VOCOS_REPOSITORY,
                VOCOS_LEGACY_CHECKPOINT,
                cache_dir=cache_dir,
                revision=VOCOS_CHECKPOINT_REVISION,
                token=token,
                local_files_only=local_files_only,
            )
            vocoder_official = True
        if verify_integrity and vocoder_official:
            require_file_integrity(
                vocoder_source,
                sha256=VOCOS_LEGACY_SHA256,
                size=VOCOS_LEGACY_SIZE,
            )
        vocoder = _native_vocoder(vocoder_source)

    return F5TTSArtifacts(
        source=source,
        revision=resolved_revision,
        checkpoint=checkpoint,
        vocabulary=vocabulary,
        vocoder=vocoder,
        checkpoint_is_official=checkpoint_official,
        vocoder_is_official=vocoder_official,
    )


__all__ = ["F5TTSArtifacts", "resolve_f5tts_artifacts"]
