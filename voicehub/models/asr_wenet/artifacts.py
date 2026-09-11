"""Artifact resolution for the native WeNet GigaSpeech U2++ provider."""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from voicehub.hub import resolve_pretrained_file
from voicehub.models.asr_wenet.native.checkpoint import (
    NATIVE_WENET_FILENAME,
    WENET_TOKENIZER_FILENAME,
    WENET_UNITS_FILENAME,
    convert_wenet_gigaspeech_checkpoint,
    file_sha256,
)
from voicehub.models.asr_wenet.native.metadata import (
    GIGASPEECH_ARCHIVE_FILENAME,
    GIGASPEECH_ARCHIVE_SHA256,
    GIGASPEECH_ARCHIVE_SIZE,
    GIGASPEECH_MODEL_URL,
    GIGASPEECH_MODEL_VERSION,
)
from voicehub.path_utils import is_explicit_local_path

_OFFICIAL_ALIASES = frozenset({
    "english",
    "gigaspeech",
    "gigaspeech-u2pp",
    "gigaspeech-u2pp-conformer",
    "wenet/gigaspeech-u2pp-conformer",
})


@dataclass(frozen=True, slots=True)
class WeNetU2PPArtifacts:
    checkpoint: Path
    config: Path
    tokenizer: Path
    units: Path
    source: str
    revision: str | None
    converted_from_pickle: bool = False

    def __post_init__(self) -> None:
        for name in ("checkpoint", "config", "tokenizer", "units"):
            path = Path(getattr(self, name)).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Native WeNet {name} was not found: {path}.")
            object.__setattr__(self, name, path)


def _cache_root(cache_dir: str | Path | None) -> Path:
    if cache_dir is not None:
        root = Path(cache_dir).expanduser()
    else:
        configured = os.environ.get("VOICEHUB_CACHE")
        root = (Path(configured).expanduser() if configured else Path.home() / ".cache" / "voicehub")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _official_archive_path(cache_dir: str | Path | None) -> Path:
    digest = hashlib.sha256(GIGASPEECH_MODEL_URL.encode()).hexdigest()[:20]
    directory = (
        _cache_root(cache_dir) / "wenet" / "gigaspeech-u2pp" / f"{GIGASPEECH_MODEL_VERSION}-{digest}")
    directory.mkdir(parents=True, exist_ok=True)
    return directory / GIGASPEECH_ARCHIVE_FILENAME


def _download_official_archive(
    *,
    cache_dir: str | Path | None,
    local_files_only: bool,
) -> Path:
    destination = _official_archive_path(cache_dir)
    if destination.is_file():
        if (destination.stat().st_size == GIGASPEECH_ARCHIVE_SIZE and
                file_sha256(destination) == GIGASPEECH_ARCHIVE_SHA256):
            return destination
        raise ValueError(f"Cached WeNet archive failed integrity validation: {destination}.")
    if local_files_only:
        raise FileNotFoundError(
            "The pinned WeNet GigaSpeech archive is not in the local cache: "
            f"{destination}.")
    request = Request(
        GIGASPEECH_MODEL_URL,
        headers={
            "Accept-Encoding": "identity",
            "User-Agent": "voicehub",
        },
    )
    temporary_path: Path | None = None
    try:
        with (
                urlopen(request, timeout=30.0) as response,
                tempfile.NamedTemporaryFile(
                    mode="wb",
                    prefix=f".{GIGASPEECH_ARCHIVE_FILENAME}.",
                    suffix=".partial",
                    dir=destination.parent,
                    delete=False,
                ) as temporary,
        ):
            temporary_path = Path(temporary.name)
            digest = hashlib.sha256()
            size = 0
            while chunk := response.read(1024 * 1024):
                size += len(chunk)
                if size > GIGASPEECH_ARCHIVE_SIZE:
                    raise OSError("WeNet returned more data than the pinned artifact.")
                digest.update(chunk)
                temporary.write(chunk)
            temporary.flush()
            os.fsync(temporary.fileno())
        if size != GIGASPEECH_ARCHIVE_SIZE:
            raise OSError(
                f"WeNet archive size mismatch: expected "
                f"{GIGASPEECH_ARCHIVE_SIZE}, found {size}.")
        actual_sha = digest.hexdigest()
        if actual_sha != GIGASPEECH_ARCHIVE_SHA256:
            raise OSError(
                "WeNet archive SHA-256 mismatch: expected "
                f"{GIGASPEECH_ARCHIVE_SHA256}, found {actual_sha}.")
        os.replace(temporary_path, destination)
        temporary_path = None
        return destination
    except (HTTPError, URLError) as error:
        raise OSError("Could not download the pinned WeNet GigaSpeech checkpoint.") from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _native_artifacts(
    root: Path,
    *,
    source: str,
    revision: str | None,
    converted: bool,
) -> WeNetU2PPArtifacts:
    return WeNetU2PPArtifacts(
        checkpoint=root / NATIVE_WENET_FILENAME,
        config=root / "config.json",
        tokenizer=root / WENET_TOKENIZER_FILENAME,
        units=root / WENET_UNITS_FILENAME,
        source=source,
        revision=revision,
        converted_from_pickle=converted,
    )


def _converted_root(destination_root: Path) -> Path:
    return destination_root / ".voicehub-native" / "wenet-u2pp"


def _has_complete_native_artifact(root: Path) -> bool:
    return all((root / filename).is_file() for filename in (
        NATIVE_WENET_FILENAME,
        "config.json",
        WENET_TOKENIZER_FILENAME,
        WENET_UNITS_FILENAME,
    ))


def _convert(
    source: Path,
    *,
    destination_root: Path,
    trust_pickle_checkpoint: bool,
) -> WeNetU2PPArtifacts:
    destination = _converted_root(destination_root)
    if not _has_complete_native_artifact(destination):
        convert_wenet_gigaspeech_checkpoint(
            source,
            destination,
            trust_pickle_checkpoint=trust_pickle_checkpoint,
        )
    return _native_artifacts(
        destination,
        source=str(source),
        revision=GIGASPEECH_MODEL_VERSION,
        converted=True,
    )


def resolve_wenet_u2pp_artifacts(
    source: str | Path,
    *,
    checkpoint_filename: str,
    tokenizer_filename: str,
    units_filename: str,
    revision: str | None,
    cache_dir: str | Path | None,
    token: str | bool | None,
    local_files_only: bool,
    trust_pickle_checkpoint: bool,
) -> WeNetU2PPArtifacts:
    source_path = Path(source).expanduser()
    if source_path.exists():
        resolved = source_path.resolve()
        root = resolved if resolved.is_dir() else resolved.parent
        is_source_archive = (resolved.is_file() and resolved.name.endswith((".tar.gz", ".tgz")))
        if is_source_archive:
            return _convert(
                resolved,
                destination_root=root,
                trust_pickle_checkpoint=trust_pickle_checkpoint,
            )
        if resolved.is_file() and resolved.suffix == ".pt":
            raise ValueError(
                "A bare WeNet pickle does not contain the tokenizer, CMVN, "
                "or recipe contract. Provide the complete official archive "
                "or extracted directory.")
        checkpoint = (
            resolved if resolved.is_file() and resolved.suffix == ".safetensors" else root /
            checkpoint_filename)
        native = (
            checkpoint,
            root / "config.json",
            root / tokenizer_filename,
            root / units_filename,
        )
        if all(path.is_file() for path in native):
            return WeNetU2PPArtifacts(
                checkpoint=native[0],
                config=native[1],
                tokenizer=native[2],
                units=native[3],
                source=str(resolved),
                revision=None,
            )
        is_extracted_source = (resolved.is_dir() and (resolved / "final.pt").is_file())
        if is_extracted_source:
            return _convert(
                resolved,
                destination_root=root,
                trust_pickle_checkpoint=trust_pickle_checkpoint,
            )
        raise FileNotFoundError(f"No complete native WeNet artifact was found in {root}.")
    if is_explicit_local_path(source):
        raise FileNotFoundError(f"Local WeNet path was not found: {source_path}.")

    source_name = str(source)
    if source_name in _OFFICIAL_ALIASES:
        if revision not in {None, GIGASPEECH_MODEL_VERSION}:
            raise ValueError("The official WeNet alias is pinned to version "
                             f"{GIGASPEECH_MODEL_VERSION}.")
        archive_path = _official_archive_path(cache_dir)
        converted = _converted_root(archive_path.parent)
        if _has_complete_native_artifact(converted):
            return _native_artifacts(
                converted,
                source=source_name,
                revision=GIGASPEECH_MODEL_VERSION,
                converted=True,
            )
        archive = _download_official_archive(
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        artifacts = _convert(
            archive,
            destination_root=archive.parent,
            trust_pickle_checkpoint=trust_pickle_checkpoint,
        )
        return WeNetU2PPArtifacts(
            checkpoint=artifacts.checkpoint,
            config=artifacts.config,
            tokenizer=artifacts.tokenizer,
            units=artifacts.units,
            source=source_name,
            revision=GIGASPEECH_MODEL_VERSION,
            converted_from_pickle=True,
        )

    checkpoint = resolve_pretrained_file(
        source_name,
        checkpoint_filename,
        cache_dir=None if cache_dir is None else str(cache_dir),
        revision=revision,
        token=token,
        local_files_only=local_files_only,
    )
    config = resolve_pretrained_file(
        source_name,
        "config.json",
        cache_dir=None if cache_dir is None else str(cache_dir),
        revision=revision,
        token=token,
        local_files_only=local_files_only,
    )
    tokenizer = resolve_pretrained_file(
        source_name,
        tokenizer_filename,
        cache_dir=None if cache_dir is None else str(cache_dir),
        revision=revision,
        token=token,
        local_files_only=local_files_only,
    )
    units = resolve_pretrained_file(
        source_name,
        units_filename,
        cache_dir=None if cache_dir is None else str(cache_dir),
        revision=revision,
        token=token,
        local_files_only=local_files_only,
    )
    return WeNetU2PPArtifacts(
        checkpoint=checkpoint,
        config=config,
        tokenizer=tokenizer,
        units=units,
        source=source_name,
        revision=revision,
    )


__all__ = ["WeNetU2PPArtifacts", "resolve_wenet_u2pp_artifacts"]
