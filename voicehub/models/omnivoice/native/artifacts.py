"""Immutable local/Hub artifact resolution for native OmniVoice."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from voicehub.hub import resolve_pretrained_file
from voicehub.models.omnivoice.native.metadata import (
    HIGGS_AUDIO_V2_CONFIG_SHA256,
    HIGGS_AUDIO_V2_CONFIG_SIZE,
    HIGGS_AUDIO_V2_MODEL_ID,
    HIGGS_AUDIO_V2_REVISION,
    HIGGS_AUDIO_V2_SAFETENSORS_SHA256,
    HIGGS_AUDIO_V2_SAFETENSORS_SIZE,
    OMNIVOICE_CONFIG_SHA256,
    OMNIVOICE_CONFIG_SIZE,
    OMNIVOICE_MODEL_ID,
    OMNIVOICE_MODEL_REVISION,
    OMNIVOICE_MODEL_SAFETENSORS_SHA256,
    OMNIVOICE_MODEL_SAFETENSORS_SIZE,
    OMNIVOICE_TOKENIZER_SHA256,
    OMNIVOICE_TOKENIZER_SIZE,
)
from voicehub.path_utils import is_explicit_local_path

MODEL_FILE = "model.safetensors"
CONFIG_FILE = "config.json"
TOKENIZER_FILE = "tokenizer.json"
CODEC_DIRECTORY = "audio_tokenizer"


@dataclass(frozen=True, slots=True)
class OmniVoiceArtifacts:
    source: str
    revision: str | None
    model_checkpoint: Path
    model_config: Path
    text_tokenizer: Path
    codec_checkpoint: Path
    codec_config: Path
    official_model: bool
    official_codec: bool


def _required(root: Path, filename: str) -> Path:
    path = root / filename
    if not path.is_file():
        raise FileNotFoundError(f"Native OmniVoice requires {filename!r} in {root}.")
    return path.resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _verify(path: Path, *, size: int, sha256: str) -> None:
    if path.stat().st_size != size:
        raise ValueError(f"OmniVoice artifact {path.name!r} has an unexpected size.")
    actual = _sha256(path)
    if actual != sha256:
        raise ValueError(f"OmniVoice artifact {path.name!r} has SHA-256 {actual}; "
                         f"expected {sha256}.")


def _resolve_hub(
    repository: str,
    filename: str,
    *,
    revision: str | None,
    cache_dir: str | None,
    token: str | bool | None,
    local_files_only: bool,
) -> Path:
    return resolve_pretrained_file(
        repository,
        filename,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    ).resolve()


def resolve_omnivoice_artifacts(
    source: str | Path,
    *,
    revision: str | None = None,
    codec_source: str | Path | None = None,
    codec_revision: str | None = None,
    cache_dir: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
    verify_integrity: bool = True,
    verify_checkpoint_integrity: bool = False,
) -> OmniVoiceArtifacts:
    if not isinstance(source, (str, Path)) or not str(source).strip():
        raise ValueError("OmniVoice source must be a non-empty path or Hub ID.")
    local = Path(source).expanduser()
    if local.exists():
        checkpoint = (local.resolve() if local.is_file() else _required(local, MODEL_FILE))
        if checkpoint.suffix.lower() != ".safetensors":
            raise ValueError("Native OmniVoice requires Safetensors.")
        root = checkpoint.parent
        model_config = _required(root, CONFIG_FILE)
        tokenizer = _required(root, TOKENIZER_FILE)
        source_name = str(root)
        resolved_revision = None
        official_model = False
    else:
        if is_explicit_local_path(source):
            raise FileNotFoundError(f"OmniVoice path was not found: {local}.")
        source_name = str(source)
        resolved_revision = revision or (
            OMNIVOICE_MODEL_REVISION if source_name == OMNIVOICE_MODEL_ID else None)
        checkpoint = _resolve_hub(
            source_name,
            MODEL_FILE,
            revision=resolved_revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        )
        model_config = _resolve_hub(
            source_name,
            CONFIG_FILE,
            revision=resolved_revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        )
        tokenizer = _resolve_hub(
            source_name,
            TOKENIZER_FILE,
            revision=resolved_revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        )
        official_model = (source_name == OMNIVOICE_MODEL_ID and resolved_revision == OMNIVOICE_MODEL_REVISION)

    if codec_source is None:
        if local.exists():
            codec_root = root / CODEC_DIRECTORY
            codec_checkpoint = _required(codec_root, MODEL_FILE)
            codec_config = _required(codec_root, CONFIG_FILE)
            official_codec = False
        else:
            codec_checkpoint = _resolve_hub(
                source_name,
                f"{CODEC_DIRECTORY}/{MODEL_FILE}",
                revision=resolved_revision,
                cache_dir=cache_dir,
                token=token,
                local_files_only=local_files_only,
            )
            codec_config = _resolve_hub(
                source_name,
                f"{CODEC_DIRECTORY}/{CONFIG_FILE}",
                revision=resolved_revision,
                cache_dir=cache_dir,
                token=token,
                local_files_only=local_files_only,
            )
            official_codec = official_model
    else:
        codec_local = Path(codec_source).expanduser()
        if codec_local.exists():
            codec_root = (codec_local.parent if codec_local.is_file() else codec_local)
            codec_checkpoint = (
                codec_local.resolve() if codec_local.is_file() else _required(codec_root, MODEL_FILE))
            codec_config = _required(codec_root, CONFIG_FILE)
            official_codec = False
        else:
            if is_explicit_local_path(codec_source):
                raise FileNotFoundError(f"Higgs tokenizer path was not found: {codec_local}.")
            codec_repository = str(codec_source)
            resolved_codec_revision = codec_revision or (
                HIGGS_AUDIO_V2_REVISION if codec_repository == HIGGS_AUDIO_V2_MODEL_ID else None)
            codec_checkpoint = _resolve_hub(
                codec_repository,
                MODEL_FILE,
                revision=resolved_codec_revision,
                cache_dir=cache_dir,
                token=token,
                local_files_only=local_files_only,
            )
            codec_config = _resolve_hub(
                codec_repository,
                CONFIG_FILE,
                revision=resolved_codec_revision,
                cache_dir=cache_dir,
                token=token,
                local_files_only=local_files_only,
            )
            official_codec = (
                codec_repository == HIGGS_AUDIO_V2_MODEL_ID and
                resolved_codec_revision == HIGGS_AUDIO_V2_REVISION)
    if codec_checkpoint.suffix.lower() != ".safetensors":
        raise ValueError("Native Higgs Audio V2 requires Safetensors.")

    if verify_integrity and official_model:
        _verify(
            model_config,
            size=OMNIVOICE_CONFIG_SIZE,
            sha256=OMNIVOICE_CONFIG_SHA256,
        )
        _verify(
            tokenizer,
            size=OMNIVOICE_TOKENIZER_SIZE,
            sha256=OMNIVOICE_TOKENIZER_SHA256,
        )
    if verify_integrity and official_codec:
        _verify(
            codec_config,
            size=HIGGS_AUDIO_V2_CONFIG_SIZE,
            sha256=HIGGS_AUDIO_V2_CONFIG_SHA256,
        )
    if verify_checkpoint_integrity and official_model:
        _verify(
            checkpoint,
            size=OMNIVOICE_MODEL_SAFETENSORS_SIZE,
            sha256=OMNIVOICE_MODEL_SAFETENSORS_SHA256,
        )
    if verify_checkpoint_integrity and official_codec:
        _verify(
            codec_checkpoint,
            size=HIGGS_AUDIO_V2_SAFETENSORS_SIZE,
            sha256=HIGGS_AUDIO_V2_SAFETENSORS_SHA256,
        )
    return OmniVoiceArtifacts(
        source=source_name,
        revision=resolved_revision,
        model_checkpoint=checkpoint.resolve(),
        model_config=model_config.resolve(),
        text_tokenizer=tokenizer.resolve(),
        codec_checkpoint=codec_checkpoint.resolve(),
        codec_config=codec_config.resolve(),
        official_model=official_model,
        official_codec=official_codec,
    )


__all__ = [
    "CODEC_DIRECTORY",
    "CONFIG_FILE",
    "MODEL_FILE",
    "TOKENIZER_FILE",
    "OmniVoiceArtifacts",
    "resolve_omnivoice_artifacts",
]
