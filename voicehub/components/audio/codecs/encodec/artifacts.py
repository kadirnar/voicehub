"""Dependency-free resolution of official and converted Encodec weights."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from voicehub.path_utils import voicehub_cache_root

from .metadata import EncodecRelease, encodec_release

_DOWNLOAD_CHUNK_SIZE = 1024 * 1024
_DOWNLOAD_TIMEOUT_SECONDS = 30
_USER_AGENT = "voicehub"
_OFFICIAL_HOST = "dl.fbaipublicfiles.com"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(_DOWNLOAD_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def verify_official_checkpoint(
    path: str | Path,
    release: EncodecRelease,
) -> str:
    """Validate the exact size and release hash prefix of a local `.th` file."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Encodec checkpoint was not found: {resolved}.")
    size = resolved.stat().st_size
    if size != release.size:
        raise ValueError(
            f"{release.model_name} checkpoint size mismatch: expected "
            f"{release.size}, found {size}.")
    digest = file_sha256(resolved)
    if not digest.startswith(release.sha256_prefix):
        raise ValueError(
            f"{release.model_name} checkpoint digest does not match the "
            f"published {release.sha256_prefix} release.")
    return digest


def _cache_root(cache_dir: str | Path | None) -> Path:
    return voicehub_cache_root(cache_dir).resolve()


def _repository_candidate(
    repository: str | Path,
    release: EncodecRelease,
) -> Path:
    source = Path(repository).expanduser()
    if source.is_file():
        return source.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Encodec repository was not found: {source}.")
    candidates = (
        source / f"{release.model_name}.safetensors",
        source / "model.safetensors",
        source / release.filename,
    )
    matches = tuple(path.resolve() for path in candidates if path.is_file())
    if not matches:
        raise FileNotFoundError(
            f"No native Safetensors or {release.filename!r} checkpoint "
            f"was found in {source}.")
    # Prefer safe native files over the legacy pickle container. Ambiguous
    # duplicate Safetensors names are rejected instead of guessed.
    safe_matches = tuple(path for path in matches if path.suffix == ".safetensors")
    if len(safe_matches) > 1:
        raise ValueError(
            f"Encodec repository contains multiple native checkpoints: "
            f"{[path.name for path in safe_matches]!r}.")
    return safe_matches[0] if safe_matches else matches[0]


def _download_official(
    release: EncodecRelease,
    destination: Path,
) -> Path:
    target = urlsplit(release.url)
    if target.scheme != "https" or target.hostname != _OFFICIAL_HOST:
        raise ValueError("Encodec release URL is outside the pinned HTTPS origin.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        request = Request(
            release.url,
            headers={"User-Agent": _USER_AGENT},
        )
        with (
            urlopen(request, timeout=_DOWNLOAD_TIMEOUT_SECONDS) as response,
            tempfile.NamedTemporaryFile(
                mode="w+b",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as output,
        ):
            temporary = Path(output.name)
            final = urlsplit(response.geturl())
            if final.scheme != "https" or final.hostname != _OFFICIAL_HOST:
                raise OSError("Encodec download redirected outside the pinned HTTPS origin.")
            declared_size = response.headers.get("Content-Length")
            if declared_size is not None and int(declared_size) != release.size:
                raise OSError(
                    f"Encodec server declared {declared_size} bytes; "
                    f"expected {release.size}.")
            digest = hashlib.sha256()
            received = 0
            while chunk := response.read(_DOWNLOAD_CHUNK_SIZE):
                received += len(chunk)
                if received > release.size:
                    raise OSError("Encodec response exceeded the published file size.")
                digest.update(chunk)
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        if received != release.size:
            raise OSError(
                f"Encodec download ended at {received} bytes; expected "
                f"{release.size}.")
        if not digest.hexdigest().startswith(release.sha256_prefix):
            raise OSError("Encodec download failed the published SHA-256 prefix check.")
        os.replace(temporary, destination)
        temporary = None
        return destination
    except (HTTPError, URLError) as error:
        raise OSError(
            f"Could not download official {release.model_name} weights: {error}.") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def resolve_encodec_checkpoint(
    model_name: str,
    *,
    repository: str | Path | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> Path:
    """Resolve a native Safetensors file or the exact official `.th` release."""
    release = encodec_release(model_name)
    if repository is not None:
        return _repository_candidate(repository, release)

    directory = _cache_root(cache_dir) / "encodec"
    native = directory / f"{release.model_name}.safetensors"
    if native.is_file():
        return native.resolve()
    legacy = directory / release.filename
    if legacy.is_file():
        verify_official_checkpoint(legacy, release)
        return legacy.resolve()
    if local_files_only:
        raise FileNotFoundError(
            f"No cached checkpoint is available for {release.model_name} "
            f"under {directory}.")
    return _download_official(release, legacy).resolve()


__all__ = [
    "file_sha256",
    "resolve_encodec_checkpoint",
    "verify_official_checkpoint",
]
