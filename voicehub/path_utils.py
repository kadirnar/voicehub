"""Path classification helpers shared by local and Hub loaders."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path


def is_explicit_local_path(value: str | Path) -> bool:
    """Return whether a model source unambiguously denotes a local path."""
    if isinstance(value, Path):
        return True
    raw_value = str(value)
    return (Path(raw_value).expanduser().is_absolute() or raw_value.startswith(("./", "../", "~")))


def normalize_model_source(value: str | Path) -> str:
    """Normalize an explicit local source while preserving Hub identifiers.

    Local paths are made absolute without following symlinks. Hub-cache
    snapshot entries are symlinks to suffix-less content blobs, so resolving
    them would lose the filename and the sibling files of the snapshot.
    Resolve explicitly where a containment or identity check needs it.
    """
    if not isinstance(value, (str, Path)):
        raise TypeError("A model source must be a string or pathlib.Path.")
    source = Path(value).expanduser()
    if not is_explicit_local_path(value):
        return str(value)
    if not source.exists():
        raise FileNotFoundError(f"Local model path was not found: {source}.")
    return str(source.absolute())


def voicehub_cache_root(cache_dir: str | os.PathLike[str] | None = None) -> Path:
    """Return VoiceHub's own writable cache root.

    Resolution order: ``cache_dir``, ``VOICEHUB_CACHE``, the legacy
    ``VOICEHUB_CACHE_DIR`` alias, ``$XDG_CACHE_HOME/voicehub`` and
    ``~/.cache/voicehub``. This root is separate from the Hugging Face
    download cache, which is content-addressed and may be read-only.
    """
    if cache_dir is not None:
        return Path(cache_dir).expanduser()
    for variable in ("VOICEHUB_CACHE", "VOICEHUB_CACHE_DIR"):
        configured = os.environ.get(variable)
        if configured:
            return Path(configured).expanduser()
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg_cache).expanduser() if xdg_cache else Path.home() / ".cache"
    return base / "voicehub"


def derived_artifact_path(
    source: str | os.PathLike[str],
    filename: str,
    *,
    namespace: str,
) -> Path:
    """Return where a file converted from ``source`` is cached.

    Converted artifacts are never written next to their source, which
    may live inside a (possibly read-only) Hub cache snapshot. The key
    covers the resolved source path, size and modification time, so a
    replaced source file is converted again.
    """
    if Path(filename).name != filename or filename in {"", ".", ".."}:
        raise ValueError(f"Derived artifact filename must be a plain name: {filename!r}.")
    parts = Path(namespace).parts
    if not parts or Path(namespace).is_absolute() or any(part in {".", ".."} for part in parts):
        raise ValueError(f"Invalid derived artifact namespace: {namespace!r}.")
    resolved = Path(source).expanduser().resolve()
    status = resolved.stat()
    fingerprint = b"\0".join((
        os.fsencode(resolved),
        str(status.st_size).encode(),
        str(status.st_mtime_ns).encode(),
    ))
    key = hashlib.sha256(fingerprint).hexdigest()[:32]
    return voicehub_cache_root() / "derived" / Path(*parts) / key / filename
