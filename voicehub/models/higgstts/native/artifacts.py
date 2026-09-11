"""Coherent local/Hub artifact resolution for native Higgs Audio v2."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from voicehub.hub import resolve_pretrained_file
from voicehub.models.higgstts.native.metadata import (
    HIGGS_AUDIO_V2_CHAT_TEMPLATE_FILE,
    HIGGS_AUDIO_V2_CHAT_TEMPLATE_SHA256,
    HIGGS_AUDIO_V2_CHAT_TEMPLATE_SIZE,
    HIGGS_AUDIO_V2_CHECKPOINT_FILE,
    HIGGS_AUDIO_V2_CHECKPOINT_SHA256,
    HIGGS_AUDIO_V2_CHECKPOINT_SIZE,
    HIGGS_AUDIO_V2_CODEC_CHECKPOINT_FILE,
    HIGGS_AUDIO_V2_CODEC_CHECKPOINT_SHA256,
    HIGGS_AUDIO_V2_CODEC_CHECKPOINT_SIZE,
    HIGGS_AUDIO_V2_CODEC_CONFIG_FILE,
    HIGGS_AUDIO_V2_CODEC_CONFIG_SHA256,
    HIGGS_AUDIO_V2_CODEC_CONFIG_SIZE,
    HIGGS_AUDIO_V2_CODEC_PREPROCESSOR_FILE,
    HIGGS_AUDIO_V2_CODEC_PREPROCESSOR_SHA256,
    HIGGS_AUDIO_V2_CODEC_PREPROCESSOR_SIZE,
    HIGGS_AUDIO_V2_CONFIG_FILE,
    HIGGS_AUDIO_V2_CONFIG_SHA256,
    HIGGS_AUDIO_V2_CONFIG_SIZE,
    HIGGS_AUDIO_V2_REPOSITORY,
    HIGGS_AUDIO_V2_REVISION,
    HIGGS_AUDIO_V2_SPECIAL_TOKENS_FILE,
    HIGGS_AUDIO_V2_SPECIAL_TOKENS_SHA256,
    HIGGS_AUDIO_V2_SPECIAL_TOKENS_SIZE,
    HIGGS_AUDIO_V2_TOKENIZER_CONFIG_FILE,
    HIGGS_AUDIO_V2_TOKENIZER_CONFIG_SHA256,
    HIGGS_AUDIO_V2_TOKENIZER_CONFIG_SIZE,
    HIGGS_AUDIO_V2_TOKENIZER_FILE,
    HIGGS_AUDIO_V2_TOKENIZER_REPOSITORY,
    HIGGS_AUDIO_V2_TOKENIZER_REVISION,
    HIGGS_AUDIO_V2_TOKENIZER_SHA256,
    HIGGS_AUDIO_V2_TOKENIZER_SIZE,
)
from voicehub.path_utils import is_explicit_local_path

_CODEC_DIRECTORY = "audio_tokenizer"


@dataclass(frozen=True, slots=True)
class HiggsAudioV2Artifacts:
    source: str
    revision: str | None
    root: Path
    checkpoint: Path
    config: Path
    tokenizer: Path
    tokenizer_config: Path | None
    special_tokens_map: Path | None
    chat_template: Path | None
    codec_source: str
    codec_revision: str | None
    codec_root: Path
    codec_checkpoint: Path
    codec_config: Path
    codec_preprocessor: Path | None
    official: bool


def _required(root: Path, filename: str, *, component: str) -> Path:
    path = root / filename
    if not path.is_file():
        raise FileNotFoundError(f"Native Higgs {component} requires {filename!r} in {root}.")
    return path.resolve()


def _optional(root: Path, filename: str) -> Path | None:
    path = root / filename
    return path.resolve() if path.is_file() else None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _verify(path: Path, *, size: int, sha256: str) -> None:
    actual_size = path.stat().st_size
    if actual_size != size:
        raise ValueError(f"Higgs artifact {path.name!r} has size {actual_size}; "
                         f"expected {size}.")
    actual_sha = _file_sha256(path)
    if actual_sha != sha256:
        raise ValueError(f"Higgs artifact {path.name!r} has SHA-256 {actual_sha}; "
                         f"expected {sha256}.")


def _remote_optional(
    repository: str,
    filename: str,
    *,
    revision: str,
    cache_dir: str | None,
    token: str | bool | None,
    local_files_only: bool,
) -> Path | None:
    try:
        return resolve_pretrained_file(
            repository,
            filename,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        )
    except FileNotFoundError:
        return None


def _resolve_main(
    source: str | Path,
    *,
    revision: str | None,
    cache_dir: str | None,
    token: str | bool | None,
    local_files_only: bool,
) -> tuple[
        str,
        str | None,
        Path,
        Path,
        Path,
        Path,
        Path | None,
        Path | None,
        Path | None,
]:
    local = Path(source).expanduser()
    if local.exists():
        root = local.parent if local.is_file() else local
        root = root.resolve()
        checkpoint = _required(
            root,
            HIGGS_AUDIO_V2_CHECKPOINT_FILE,
            component="model",
        )
        if local.is_file() and local.resolve() != checkpoint:
            raise ValueError("A direct Higgs model source must be model.safetensors.")
        return (
            str(local.resolve()),
            None,
            root,
            checkpoint,
            _required(
                root,
                HIGGS_AUDIO_V2_CONFIG_FILE,
                component="model",
            ),
            _required(
                root,
                HIGGS_AUDIO_V2_TOKENIZER_FILE,
                component="text tokenizer",
            ),
            _optional(root, HIGGS_AUDIO_V2_TOKENIZER_CONFIG_FILE),
            _optional(root, HIGGS_AUDIO_V2_SPECIAL_TOKENS_FILE),
            _optional(root, HIGGS_AUDIO_V2_CHAT_TEMPLATE_FILE),
        )
    if is_explicit_local_path(source):
        raise FileNotFoundError(f"Higgs model path was not found: {local}.")
    repository = str(source)
    resolved_revision = (
        revision or (HIGGS_AUDIO_V2_REVISION if repository == HIGGS_AUDIO_V2_REPOSITORY else "main"))
    options = {
        "revision": resolved_revision,
        "cache_dir": cache_dir,
        "token": token,
        "local_files_only": local_files_only,
    }
    checkpoint = resolve_pretrained_file(
        repository,
        HIGGS_AUDIO_V2_CHECKPOINT_FILE,
        **options,
    )
    config = resolve_pretrained_file(
        repository,
        HIGGS_AUDIO_V2_CONFIG_FILE,
        **options,
    )
    tokenizer_path = resolve_pretrained_file(
        repository,
        HIGGS_AUDIO_V2_TOKENIZER_FILE,
        **options,
    )
    root = checkpoint.parent.resolve()
    if any(path.parent.resolve() != root for path in (config, tokenizer_path)):
        raise RuntimeError("Higgs model files crossed immutable snapshot roots.")
    tokenizer_config = _remote_optional(
        repository,
        HIGGS_AUDIO_V2_TOKENIZER_CONFIG_FILE,
        revision=resolved_revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    special_tokens = _remote_optional(
        repository,
        HIGGS_AUDIO_V2_SPECIAL_TOKENS_FILE,
        revision=resolved_revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    chat_template = _remote_optional(
        repository,
        HIGGS_AUDIO_V2_CHAT_TEMPLATE_FILE,
        revision=resolved_revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    for path in (tokenizer_config, special_tokens, chat_template):
        if path is not None and path.parent.resolve() != root:
            raise RuntimeError("Higgs tokenizer files crossed immutable snapshot roots.")
    return (
        repository,
        resolved_revision,
        root,
        checkpoint,
        config,
        tokenizer_path,
        tokenizer_config,
        special_tokens,
        chat_template,
    )


def _resolve_codec(
    source: str | Path,
    *,
    revision: str | None,
    cache_dir: str | None,
    token: str | bool | None,
    local_files_only: bool,
) -> tuple[str, str | None, Path, Path, Path, Path | None]:
    local = Path(source).expanduser()
    if local.exists():
        if local.is_file():
            raise ValueError("A local Higgs audio tokenizer source must be a directory.")
        root = local.resolve()
        return (
            str(root),
            None,
            root,
            _required(
                root,
                HIGGS_AUDIO_V2_CODEC_CHECKPOINT_FILE,
                component="audio tokenizer",
            ),
            _required(
                root,
                HIGGS_AUDIO_V2_CODEC_CONFIG_FILE,
                component="audio tokenizer",
            ),
            _optional(root, HIGGS_AUDIO_V2_CODEC_PREPROCESSOR_FILE),
        )
    if is_explicit_local_path(source):
        raise FileNotFoundError(f"Higgs audio tokenizer path was not found: {local}.")
    repository = str(source)
    resolved_revision = (
        revision or
        (HIGGS_AUDIO_V2_TOKENIZER_REVISION if repository == HIGGS_AUDIO_V2_TOKENIZER_REPOSITORY else "main"))
    options = {
        "revision": resolved_revision,
        "cache_dir": cache_dir,
        "token": token,
        "local_files_only": local_files_only,
    }
    checkpoint = resolve_pretrained_file(
        repository,
        HIGGS_AUDIO_V2_CODEC_CHECKPOINT_FILE,
        **options,
    )
    config = resolve_pretrained_file(
        repository,
        HIGGS_AUDIO_V2_CODEC_CONFIG_FILE,
        **options,
    )
    preprocessor = _remote_optional(
        repository,
        HIGGS_AUDIO_V2_CODEC_PREPROCESSOR_FILE,
        revision=resolved_revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    root = checkpoint.parent.resolve()
    if config.parent.resolve() != root or (preprocessor is not None and
                                           preprocessor.parent.resolve() != root):
        raise RuntimeError("Higgs audio tokenizer files crossed immutable snapshot roots.")
    return (
        repository,
        resolved_revision,
        root,
        checkpoint,
        config,
        preprocessor,
    )


def resolve_higgs_audio_v2_artifacts(
    source: str | Path,
    *,
    revision: str | None = None,
    codec_source: str | Path | None = None,
    codec_revision: str | None = None,
    cache_dir: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
    verify_integrity: bool = False,
    verify_checkpoint_integrity: bool = False,
) -> HiggsAudioV2Artifacts:
    """Resolve both immutable snapshots without importing model libraries."""
    main = _resolve_main(
        source,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    (
        source_name,
        resolved_revision,
        root,
        checkpoint,
        config,
        tokenizer_path,
        tokenizer_config,
        special_tokens,
        chat_template,
    ) = main
    if codec_source is None:
        local_codec = root / _CODEC_DIRECTORY
        codec_source = (local_codec if local_codec.is_dir() else HIGGS_AUDIO_V2_TOKENIZER_REPOSITORY)
    codec = _resolve_codec(
        codec_source,
        revision=codec_revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    (
        codec_source_name,
        resolved_codec_revision,
        codec_root,
        codec_checkpoint,
        codec_config,
        codec_preprocessor,
    ) = codec
    official = (
        source_name == HIGGS_AUDIO_V2_REPOSITORY and resolved_revision == HIGGS_AUDIO_V2_REVISION and
        codec_source_name == HIGGS_AUDIO_V2_TOKENIZER_REPOSITORY and
        resolved_codec_revision == HIGGS_AUDIO_V2_TOKENIZER_REVISION)
    if verify_integrity and official:
        for path, size, digest in (
            (
                config,
                HIGGS_AUDIO_V2_CONFIG_SIZE,
                HIGGS_AUDIO_V2_CONFIG_SHA256,
            ),
            (
                tokenizer_path,
                HIGGS_AUDIO_V2_TOKENIZER_SIZE,
                HIGGS_AUDIO_V2_TOKENIZER_SHA256,
            ),
            (
                tokenizer_config,
                HIGGS_AUDIO_V2_TOKENIZER_CONFIG_SIZE,
                HIGGS_AUDIO_V2_TOKENIZER_CONFIG_SHA256,
            ),
            (
                special_tokens,
                HIGGS_AUDIO_V2_SPECIAL_TOKENS_SIZE,
                HIGGS_AUDIO_V2_SPECIAL_TOKENS_SHA256,
            ),
            (
                chat_template,
                HIGGS_AUDIO_V2_CHAT_TEMPLATE_SIZE,
                HIGGS_AUDIO_V2_CHAT_TEMPLATE_SHA256,
            ),
            (
                codec_config,
                HIGGS_AUDIO_V2_CODEC_CONFIG_SIZE,
                HIGGS_AUDIO_V2_CODEC_CONFIG_SHA256,
            ),
            (
                codec_preprocessor,
                HIGGS_AUDIO_V2_CODEC_PREPROCESSOR_SIZE,
                HIGGS_AUDIO_V2_CODEC_PREPROCESSOR_SHA256,
            ),
        ):
            if path is None:
                raise FileNotFoundError("An integrity-pinned Higgs auxiliary artifact is missing.")
            _verify(path, size=size, sha256=digest)
    if verify_checkpoint_integrity and official:
        _verify(
            checkpoint,
            size=HIGGS_AUDIO_V2_CHECKPOINT_SIZE,
            sha256=HIGGS_AUDIO_V2_CHECKPOINT_SHA256,
        )
        _verify(
            codec_checkpoint,
            size=HIGGS_AUDIO_V2_CODEC_CHECKPOINT_SIZE,
            sha256=HIGGS_AUDIO_V2_CODEC_CHECKPOINT_SHA256,
        )
    return HiggsAudioV2Artifacts(
        source=source_name,
        revision=resolved_revision,
        root=root,
        checkpoint=checkpoint,
        config=config,
        tokenizer=tokenizer_path,
        tokenizer_config=tokenizer_config,
        special_tokens_map=special_tokens,
        chat_template=chat_template,
        codec_source=codec_source_name,
        codec_revision=resolved_codec_revision,
        codec_root=codec_root,
        codec_checkpoint=codec_checkpoint,
        codec_config=codec_config,
        codec_preprocessor=codec_preprocessor,
        official=official,
    )


__all__ = [
    "HiggsAudioV2Artifacts",
    "resolve_higgs_audio_v2_artifacts",
]
