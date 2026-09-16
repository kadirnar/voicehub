"""Secure, dependency-free transport for Hugging Face model files.

The transport intentionally implements only the small HTTP and cache
surface needed by VoiceHub.  Model architecture code must not depend on
a third-party Hub runtime just to resolve a checkpoint file.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import socket
import tempfile
import time
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath
from threading import local
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from voicehub.json_utils import parse_json_object, parse_json_value

_HUGGING_FACE_ENDPOINT = "https://huggingface.co"
_DEFAULT_REVISION = "main"
_USER_AGENT = "voicehub"
_DOWNLOAD_TIMEOUT_SECONDS = 30.0
_LOCK_TIMEOUT_SECONDS = 300.0
_STALE_LOCK_SECONDS = 86_400.0
_COPY_CHUNK_SIZE = 1024 * 1024
_MAX_API_RESPONSE_BYTES = 32 * 1024 * 1024
_MAX_SNAPSHOT_FILES = 100_000
_MAX_PAGINATION_PAGES = 10_000
_MAX_PATTERNS = 1_000
_HEX_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_COMMIT_HASH = re.compile(r"^[0-9a-fA-F]{7,64}$")
_TRUE_ENV_VALUES = frozenset({"1", "on", "true", "yes"})
_TRUSTED_REDIRECT_HEADERS = (
    "X-Linked-ETag",
    "X-Linked-Size",
    "X-Repo-Commit",
)


class HubDownloadError(OSError):
    """Raised when a remote model file cannot be downloaded safely."""


class _SafeRedirectHandler(HTTPRedirectHandler):
    """Follow HTTPS redirects without leaking credentials across hosts."""

    def __init__(self) -> None:
        super().__init__()
        self._state = local()

    def reset_response_metadata(self) -> None:
        """Clear trusted metadata captured for the current thread."""
        self._state.response_metadata = {}

    def response_metadata(self) -> dict[str, str]:
        """Return metadata captured before a redirect to object storage."""
        return dict(getattr(self._state, "response_metadata", {}))

    def redirect_request(
        self,
        request: Request,
        file_pointer: Any,
        code: int,
        message: str,
        headers: Mapping[str, str],
        new_url: str,
    ) -> Request | None:
        old_target = urlsplit(request.full_url)
        hub_target = urlsplit(_HUGGING_FACE_ENDPOINT)
        if (
                old_target.scheme.lower(),
                old_target.hostname,
                old_target.port,
        ) == (
                hub_target.scheme.lower(),
                hub_target.hostname,
                hub_target.port,
        ):
            captured = self.response_metadata()
            for header_name in _TRUSTED_REDIRECT_HEADERS:
                value = _get_header(headers, header_name)
                if value is not None:
                    captured[header_name] = value
            self._state.response_metadata = captured

        redirected = super().redirect_request(
            request,
            file_pointer,
            code,
            message,
            headers,
            new_url,
        )
        if redirected is None:
            return None

        new_target = urlsplit(redirected.full_url)
        if new_target.scheme.lower() != "https":
            raise HTTPError(
                redirected.full_url,
                code,
                "VoiceHub refused an insecure Hub redirect.",
                headers,
                file_pointer,
            )
        if (
                old_target.scheme.lower(),
                old_target.hostname,
                old_target.port,
        ) != (
                new_target.scheme.lower(),
                new_target.hostname,
                new_target.port,
        ):
            _remove_request_header(redirected, "Authorization")
        return redirected


_REDIRECT_HANDLER = _SafeRedirectHandler()
_URL_OPENER = build_opener(_REDIRECT_HANDLER)


@dataclass(frozen=True)
class _CachedFile:
    path: Path
    etag: str | None
    size: int | None
    sha256: str | None


@dataclass(frozen=True)
class _RepoFile:
    path: PurePosixPath
    size: int | None


class _FileLock:
    """A small cross-platform inter-process lock based on atomic creation."""

    def __init__(
        self,
        path: Path,
        *,
        timeout: float = _LOCK_TIMEOUT_SECONDS,
        stale_after: float = _STALE_LOCK_SECONDS,
    ) -> None:
        self.path = path
        self.timeout = timeout
        self.stale_after = stale_after
        self._host = socket.gethostname()
        self._owner = f"{self._host}:{os.getpid()}:{uuid.uuid4().hex}"
        self._acquired = False

    def __enter__(self) -> _FileLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                descriptor = os.open(
                    self.path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o600,
                )
            except FileExistsError:
                self._discard_stale_lock()
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out waiting for model cache lock: {self.path}")
                time.sleep(0.05)
                continue

            try:
                os.write(descriptor, self._owner.encode("ascii"))
            finally:
                os.close(descriptor)
            self._acquired = True
            return self

    def __exit__(self, *_: object) -> None:
        if not self._acquired:
            return
        try:
            if self.path.read_text(encoding="ascii") == self._owner:
                self.path.unlink(missing_ok=True)
        except (FileNotFoundError, OSError, UnicodeError):
            pass
        finally:
            self._acquired = False

    def _discard_stale_lock(self) -> None:
        try:
            initial_stat = self.path.stat()
            owner = self.path.read_text(encoding="ascii")
            parts = owner.split(":")
            age = time.time() - initial_stat.st_mtime
            discard = age > self.stale_after
            if len(parts) == 3 and parts[0] == self._host and os.name != "nt":
                try:
                    pid = int(parts[1])
                    if pid <= 0:
                        return
                    os.kill(pid, 0)
                except ProcessLookupError:
                    discard = True
                except (PermissionError, ValueError):
                    return
                else:
                    # A live download must never lose its lock merely because
                    # a large checkpoint takes longer than the stale timeout.
                    return
            elif len(parts) == 3 and parts[0] != self._host:
                # A shared cache may be owned by another host. Its PID is not
                # meaningful here, so do not infer process death locally.
                return
            if discard:
                current_stat = self.path.stat()
                if ((current_stat.st_dev, current_stat.st_ino, current_stat.st_mtime_ns)
                        == (initial_stat.st_dev, initial_stat.st_ino, initial_stat.st_mtime_ns) and
                        self.path.read_text(encoding="ascii") == owner):
                    self.path.unlink(missing_ok=True)
        except (FileNotFoundError, UnicodeError):
            pass


def download_hugging_face_file(
    repo_id: str,
    filename: str,
    *,
    subfolder: str = "",
    cache_dir: str | os.PathLike[str] | None = None,
    revision: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
) -> Path:
    """Resolve one Hugging Face model file into VoiceHub's local cache.

    Downloads are atomic and serialized per repository revision and
    file.  A successful response is stored under an immutable commit
    snapshot whenever the server supplies ``X-Repo-Commit``.  No
    authentication value is ever included in cache metadata.
    """
    normalized_repo = _validate_repo_id(repo_id)
    normalized_revision = _validate_revision(_DEFAULT_REVISION if revision is None else revision)
    relative_file = _validate_repo_path(subfolder, filename)
    root = _cache_root(cache_dir)
    repository_cache = _safe_join(
        root,
        "voicehub",
        "repos",
        _stable_key(normalized_repo),
    )
    metadata_path = _metadata_path(
        repository_cache,
        normalized_revision,
        relative_file,
    )
    cached = _read_cached_file(
        repository_cache,
        metadata_path,
        normalized_repo,
        normalized_revision,
        relative_file,
    )
    legacy_cached = _find_legacy_cached_file(
        root,
        normalized_repo,
        normalized_revision,
        relative_file,
    )

    offline = local_files_only or _offline_mode_enabled()
    if offline:
        available = cached.path if cached is not None else legacy_cached
        if available is not None:
            return available
        raise FileNotFoundError(
            "The requested Hub file is not available in the local cache: "
            f"{normalized_repo}@{normalized_revision}/{relative_file.as_posix()} "
            f"(cache: {root}).")

    lock_identity = "\0".join((normalized_repo, normalized_revision, relative_file.as_posix()))
    lock_path = _safe_join(
        root,
        "voicehub",
        "locks",
        f"{_stable_key(lock_identity)}.lock",
    )
    with _FileLock(lock_path):
        cached = _read_cached_file(
            repository_cache,
            metadata_path,
            normalized_repo,
            normalized_revision,
            relative_file,
        )
        return _download_or_reuse(
            repository_cache=repository_cache,
            metadata_path=metadata_path,
            repo_id=normalized_repo,
            revision=normalized_revision,
            relative_file=relative_file,
            token=_resolve_token(token),
            cached=cached,
            fallback=cached.path if cached is not None else legacy_cached,
        )


def download_hugging_face_snapshot(
    repo_id: str,
    *,
    cache_dir: str | os.PathLike[str] | None = None,
    revision: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
    allow_patterns: str | Iterable[str] | None = None,
    ignore_patterns: str | Iterable[str] | None = None,
) -> Path:
    """Resolve a complete, immutable Hugging Face repository snapshot.

    The repository API is used only to resolve a requested revision to
    an immutable commit and enumerate its files.  Every file is then
    downloaded through :func:`download_hugging_face_file`, retaining its
    atomic writes, integrity checks, credential handling, and cache
    locks.

    A snapshot manifest is published only after every selected file is
    present.  That manifest makes later offline resolution deterministic
    and prevents a partially downloaded repository from appearing
    complete. Patterns use POSIX repository paths and shell-style
    wildcards.
    """
    normalized_repo = _validate_repo_id(repo_id)
    normalized_revision = _validate_revision(_DEFAULT_REVISION if revision is None else revision)
    normalized_allow = _normalize_patterns(allow_patterns, "allow_patterns")
    normalized_ignore = _normalize_patterns(ignore_patterns, "ignore_patterns")
    root = _cache_root(cache_dir)
    repository_cache = _safe_join(
        root,
        "voicehub",
        "repos",
        _stable_key(normalized_repo),
    )
    manifest_path = _snapshot_manifest_path(
        repository_cache,
        normalized_revision,
        normalized_allow,
        normalized_ignore,
    )
    cached = _read_cached_snapshot(
        repository_cache,
        manifest_path,
        repo_id=normalized_repo,
        revision=normalized_revision,
        allow_patterns=normalized_allow,
        ignore_patterns=normalized_ignore,
    )
    legacy_cached = _find_legacy_cached_snapshot(
        root,
        normalized_repo,
        normalized_revision,
        allow_patterns=normalized_allow,
        ignore_patterns=normalized_ignore,
    )

    offline = local_files_only or _offline_mode_enabled()
    if offline:
        available = cached if cached is not None else legacy_cached
        if available is not None:
            return available.resolve()
        raise FileNotFoundError(
            "The requested Hub snapshot is not available as a complete local "
            f"snapshot: {normalized_repo}@{normalized_revision} (cache: {root}).")

    # An immutable revision with a complete manifest never needs a network
    # round-trip. Moving references are refreshed before reuse.
    if cached is not None and _COMMIT_HASH.fullmatch(normalized_revision):
        return cached.resolve()

    resolved_token = _resolve_token(token)
    try:
        commit = _resolve_hugging_face_commit(
            normalized_repo,
            normalized_revision,
            token=resolved_token,
        )
        repo_files = _list_hugging_face_files(
            normalized_repo,
            commit,
            token=resolved_token,
        )
    except HubDownloadError:
        if cached is not None:
            return cached.resolve()
        raise

    selected_files = tuple(
        repo_file for repo_file in repo_files if _matches_snapshot_patterns(
            repo_file.path,
            allow_patterns=normalized_allow,
            ignore_patterns=normalized_ignore,
        ))
    lock_identity = json.dumps(
        {
            "allow": normalized_allow,
            "commit": commit,
            "ignore": normalized_ignore,
            "repo_id": normalized_repo,
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    lock_key = _stable_key("snapshot" + chr(0) + lock_identity)
    lock_path = _safe_join(
        root,
        "voicehub",
        "locks",
        f"{lock_key}.lock",
    )
    with _FileLock(lock_path):
        # Another process may have completed the exact snapshot while this
        # process was enumerating the repository.
        completed = _read_cached_snapshot(
            repository_cache,
            manifest_path,
            repo_id=normalized_repo,
            revision=normalized_revision,
            allow_patterns=normalized_allow,
            ignore_patterns=normalized_ignore,
            expected_commit=commit,
        )
        if completed is not None:
            return completed.resolve()

        snapshot_root: Path | None = None
        manifest_files: list[dict[str, object]] = []
        for repo_file in selected_files:
            downloaded = download_hugging_face_file(
                normalized_repo,
                repo_file.path.as_posix(),
                cache_dir=root,
                revision=commit,
                token=resolved_token,
            )
            candidate_root = _snapshot_root_for_file(downloaded, repo_file.path)
            if snapshot_root is None:
                snapshot_root = candidate_root
            elif candidate_root.resolve() != snapshot_root.resolve():
                raise HubDownloadError("Hub files resolved into different immutable snapshots.")

            actual_size = downloaded.stat().st_size
            if repo_file.size is not None and actual_size != repo_file.size:
                raise HubDownloadError(
                    "A downloaded Hub snapshot file has an unexpected size: "
                    f"{repo_file.path.as_posix()} expected {repo_file.size} "
                    f"bytes, received {actual_size} bytes.")
            manifest_files.append({
                "path": repo_file.path.as_posix(),
                "size": actual_size,
            })

        if snapshot_root is None:
            snapshot_root = _safe_join(
                repository_cache,
                "snapshots",
                _snapshot_key(commit, None),
            )
            snapshot_root.mkdir(parents=True, exist_ok=True)
        snapshot_key = snapshot_root.name
        if not _HEX_SHA256.fullmatch(snapshot_key):
            raise HubDownloadError("The native Hub cache produced an invalid snapshot key.")

        _atomic_write_json(
            manifest_path,
            {
                "allow_patterns": list(normalized_allow),
                "commit": commit,
                "files": manifest_files,
                "ignore_patterns": list(normalized_ignore),
                "repo_id": normalized_repo,
                "revision": normalized_revision,
                "snapshot_key": snapshot_key,
                "version": 1,
            },
        )
        return snapshot_root.resolve()


def get_cached_hugging_face_commit(
    repo_id: str,
    filename: str,
    *,
    subfolder: str = "",
    cache_dir: str | os.PathLike[str] | None = None,
    revision: str | None = None,
) -> str | None:
    """Return the verified commit recorded for a cached Hub file.

    Callers resolving several files can download the first one from a
    moving branch, read its immutable commit, and request every
    remaining asset from that commit.  ``None`` means the file is
    absent, legacy-cached, or the Hub did not provide trustworthy commit
    metadata.
    """
    normalized_repo = _validate_repo_id(repo_id)
    normalized_revision = _validate_revision(revision or _DEFAULT_REVISION)
    relative_file = _validate_repo_path(subfolder, filename)
    root = _cache_root(cache_dir)
    repository_cache = _safe_join(
        root,
        "voicehub",
        "repos",
        _stable_key(normalized_repo),
    )
    metadata_path = _metadata_path(
        repository_cache,
        normalized_revision,
        relative_file,
    )
    cached = _read_cached_file(
        repository_cache,
        metadata_path,
        normalized_repo,
        normalized_revision,
        relative_file,
    )
    if cached is None:
        return None
    metadata = _read_cached_json_object(metadata_path)
    if metadata is None:
        return None
    commit = metadata.get("commit")
    if not isinstance(commit, str) or not _COMMIT_HASH.fullmatch(commit):
        return None
    return commit


def _resolve_hugging_face_commit(
    repo_id: str,
    revision: str,
    *,
    token: str | None,
) -> str:
    encoded_repo = "/".join(quote(part, safe="") for part in repo_id.split("/"))
    api_path = f"/api/models/{encoded_repo}/revision/{quote(revision, safe='')}"
    payload, _ = _request_hub_json(
        f"{_HUGGING_FACE_ENDPOINT}{api_path}",
        token=token,
        context=f"{repo_id}@{revision}",
        expected_path=api_path,
    )
    if not isinstance(payload, dict):
        raise HubDownloadError("The Hub model-info response is not a JSON object.")
    commit = payload.get("sha")
    if not isinstance(commit, str) or not _COMMIT_HASH.fullmatch(commit):
        raise HubDownloadError("The Hub model-info response has no valid commit hash.")
    if (_COMMIT_HASH.fullmatch(revision) and not commit.lower().startswith(revision.lower())):
        raise HubDownloadError("The Hub resolved an immutable revision to a different commit.")
    return commit.lower()


def _list_hugging_face_files(
    repo_id: str,
    commit: str,
    *,
    token: str | None,
) -> tuple[_RepoFile, ...]:
    encoded_repo = "/".join(quote(part, safe="") for part in repo_id.split("/"))
    api_path = f"/api/models/{encoded_repo}/tree/{quote(commit, safe='')}"
    next_url: str | None = (f"{_HUGGING_FACE_ENDPOINT}{api_path}?recursive=true&expand=false")
    visited_urls: set[str] = set()
    files: dict[str, _RepoFile] = {}
    page_count = 0

    while next_url is not None:
        next_url = _validate_hub_api_url(next_url, expected_path=api_path)
        if next_url in visited_urls:
            raise HubDownloadError("The Hub repository tree contains a pagination cycle.")
        visited_urls.add(next_url)
        page_count += 1
        if page_count > _MAX_PAGINATION_PAGES:
            raise HubDownloadError("The Hub repository tree has too many pages.")

        payload, headers = _request_hub_json(
            next_url,
            token=token,
            context=f"{repo_id}@{commit} repository tree",
            expected_path=api_path,
        )
        if not isinstance(payload, list):
            raise HubDownloadError("The Hub repository-tree response is not a JSON list.")
        for entry in payload:
            if not isinstance(entry, dict):
                raise HubDownloadError("The Hub repository tree contains an invalid entry.")
            entry_type = entry.get("type")
            if entry_type == "directory":
                continue
            if entry_type != "file":
                raise HubDownloadError("The Hub repository tree contains an unsupported entry type.")
            raw_path = entry.get("path")
            if not isinstance(raw_path, str):
                raise HubDownloadError("The Hub repository tree contains a file without a path.")
            try:
                path = _validate_repo_path("", raw_path)
            except (TypeError, ValueError) as error:
                raise HubDownloadError("The Hub repository tree contains an unsafe file path.") from error
            normalized_path = path.as_posix()
            if normalized_path in files:
                raise HubDownloadError("The Hub repository tree contains a duplicate file path.")

            size = entry.get("size")
            if size is not None and (not isinstance(size, int) or isinstance(size, bool) or size < 0):
                raise HubDownloadError("The Hub repository tree contains an invalid file size.")
            files[normalized_path] = _RepoFile(path=path, size=size)
            if len(files) > _MAX_SNAPSHOT_FILES:
                raise HubDownloadError("The Hub repository contains too many files for a safe snapshot.")

        next_url = _next_pagination_url(
            headers,
            current_url=next_url,
            expected_path=api_path,
        )

    return tuple(files[path] for path in sorted(files))


def _request_hub_json(
    url: str,
    *,
    token: str | None,
    context: str,
    expected_path: str,
) -> tuple[object, Mapping[str, str]]:
    trusted_url = _validate_hub_api_url(url, expected_path=expected_path)
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "identity",
        "User-Agent": _USER_AGENT,
    }
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(trusted_url, headers=headers, method="GET")
    _REDIRECT_HANDLER.reset_response_metadata()
    try:
        response = _URL_OPENER.open(request, timeout=_DOWNLOAD_TIMEOUT_SECONDS)
    except HTTPError as error:
        if error.code == 404:
            raise FileNotFoundError(
                f"Could not find the requested Hub repository revision: {context}.") from None
        if error.code in (401, 403):
            raise PermissionError(
                f"Hugging Face denied access to {context} (HTTP {error.code}). "
                "Check the repository permissions and token.") from None
        raise HubDownloadError(
            f"Hugging Face returned HTTP {error.code} while resolving {context}.") from None
    except (TimeoutError, URLError) as error:
        reason = getattr(error, "reason", error)
        raise HubDownloadError(f"Could not reach Hugging Face while resolving {context}: {reason}") from error

    with response:
        status = getattr(response, "status", None)
        if status is None:
            status = response.getcode()
        if status is not None and not 200 <= status < 300:
            raise HubDownloadError(f"Hugging Face returned HTTP {status} while resolving {context}.")
        response_headers = {str(key): str(value) for key, value in response.headers.items()}
        raw_length = _get_header(response_headers, "Content-Length")
        if raw_length is not None:
            try:
                content_length = int(raw_length)
            except (TypeError, ValueError):
                raise HubDownloadError("The Hub API response contains an invalid Content-Length.") from None
            if content_length < 0 or content_length > _MAX_API_RESPONSE_BYTES:
                raise HubDownloadError("The Hub API response is too large.")
        content = response.read(_MAX_API_RESPONSE_BYTES + 1)
        if not isinstance(content, bytes):
            raise HubDownloadError("The Hub API returned non-binary response data.")
        if len(content) > _MAX_API_RESPONSE_BYTES:
            raise HubDownloadError("The Hub API response is too large.")

    try:
        payload = parse_json_value(
            content,
            source=f"Hub API response for {context}",
        )
    except ValueError as error:
        raise HubDownloadError(f"The Hub API returned invalid JSON for {context}: {error}") from None
    return payload, response_headers


def _next_pagination_url(
    headers: Mapping[str, str],
    *,
    current_url: str,
    expected_path: str,
) -> str | None:
    raw_link = _get_header(headers, "Link")
    if raw_link is None:
        return None
    if len(raw_link) > 16_384 or "\r" in raw_link or "\n" in raw_link:
        raise HubDownloadError("The Hub API returned an invalid Link header.")
    for match in re.finditer(
            r'<([^<>]+)>\s*;\s*rel\s*=\s*(?:"([^"]+)"|([^,;\s]+))',
            raw_link,
            flags=re.IGNORECASE,
    ):
        relations = (match.group(2) or match.group(3) or "").lower().split()
        if "next" not in relations:
            continue
        candidate = urljoin(current_url, match.group(1))
        return _validate_hub_api_url(candidate, expected_path=expected_path)
    return None


def _validate_hub_api_url(url: str, *, expected_path: str) -> str:
    if not isinstance(url, str) or len(url) > 16_384:
        raise HubDownloadError("The Hub API returned an invalid pagination URL.")
    target = urlsplit(url)
    hub = urlsplit(_HUGGING_FACE_ENDPOINT)
    if (target.scheme.lower() != "https" or target.hostname != hub.hostname or target.port != hub.port or
            target.username is not None or target.password is not None or target.path != expected_path or
            target.fragment):
        raise HubDownloadError("VoiceHub refused an untrusted Hub API pagination URL.")
    return url


def _normalize_patterns(
    patterns: str | Iterable[str] | None,
    name: str,
) -> tuple[str, ...]:
    if patterns is None:
        return ()
    if isinstance(patterns, str):
        candidates = (patterns, )
    else:
        try:
            candidates = tuple(patterns)
        except TypeError:
            raise TypeError(f"`{name}` must be a string, iterable of strings, or None.") from None
    if len(candidates) > _MAX_PATTERNS:
        raise ValueError(f"`{name}` contains too many patterns.")

    normalized: set[str] = set()
    for pattern in candidates:
        if not isinstance(pattern, str):
            raise TypeError(f"Every `{name}` entry must be a string.")
        if (not pattern or len(pattern) > 1_024 or pattern.startswith("/") or "\\" in pattern or
                "\x00" in pattern or any(ord(character) < 32 for character in pattern) or
                ".." in pattern.split("/")):
            raise ValueError(f"Invalid repository pattern in `{name}`: {pattern!r}")
        normalized.add(pattern)
    return tuple(sorted(normalized))


def _matches_snapshot_patterns(
    path: PurePosixPath,
    *,
    allow_patterns: tuple[str, ...],
    ignore_patterns: tuple[str, ...],
) -> bool:
    value = path.as_posix()
    if allow_patterns and not any(fnmatchcase(value, pattern) for pattern in allow_patterns):
        return False
    return not any(fnmatchcase(value, pattern) for pattern in ignore_patterns)


def _snapshot_manifest_path(
    repository_cache: Path,
    revision: str,
    allow_patterns: tuple[str, ...],
    ignore_patterns: tuple[str, ...],
) -> Path:
    identity = json.dumps(
        {
            "allow": allow_patterns,
            "ignore": ignore_patterns,
            "revision": revision,
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return _safe_join(
        repository_cache,
        "snapshot_refs",
        f"{_stable_key(identity)}.json",
    )


def _read_cached_snapshot(
    repository_cache: Path,
    manifest_path: Path,
    *,
    repo_id: str,
    revision: str,
    allow_patterns: tuple[str, ...],
    ignore_patterns: tuple[str, ...],
    expected_commit: str | None = None,
) -> Path | None:
    manifest = _read_cached_json_object(manifest_path)
    if manifest is None:
        return None
    if (manifest.get("version") != 1 or manifest.get("repo_id") != repo_id or
            manifest.get("revision") != revision or manifest.get("allow_patterns") != list(allow_patterns) or
            manifest.get("ignore_patterns") != list(ignore_patterns)):
        return None
    commit = manifest.get("commit")
    if not isinstance(commit, str) or not _COMMIT_HASH.fullmatch(commit):
        return None
    if expected_commit is not None and commit.lower() != expected_commit.lower():
        return None
    snapshot_key = manifest.get("snapshot_key")
    if not isinstance(snapshot_key, str) or not _HEX_SHA256.fullmatch(snapshot_key):
        return None
    snapshot_root = _safe_join(
        repository_cache,
        "snapshots",
        snapshot_key,
    )
    if not snapshot_root.is_dir():
        return None

    files = manifest.get("files")
    if not isinstance(files, list) or len(files) > _MAX_SNAPSHOT_FILES:
        return None
    seen: set[str] = set()
    for entry in files:
        if not isinstance(entry, dict):
            return None
        raw_path = entry.get("path")
        size = entry.get("size")
        if (not isinstance(raw_path, str) or not isinstance(size, int) or isinstance(size, bool) or size < 0):
            return None
        try:
            path = _validate_repo_path("", raw_path)
        except (TypeError, ValueError):
            return None
        normalized_path = path.as_posix()
        if normalized_path in seen or not _matches_snapshot_patterns(
                path,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
        ):
            return None
        seen.add(normalized_path)
        cached_file = _safe_join(snapshot_root, *path.parts)
        try:
            if not cached_file.is_file() or cached_file.stat().st_size != size:
                return None
        except OSError:
            return None
    return snapshot_root


def _find_legacy_cached_snapshot(
    root: Path,
    repo_id: str,
    revision: str,
    *,
    allow_patterns: tuple[str, ...],
    ignore_patterns: tuple[str, ...],
) -> Path | None:
    """Read a complete legacy snapshot without importing huggingface_hub.

    A legacy cache has no VoiceHub manifest, so pattern-filtered
    snapshots cannot be proven complete and are deliberately not
    accepted.
    """
    if allow_patterns or ignore_patterns:
        return None
    repository = _safe_join(root, f"models--{repo_id.replace('/', '--')}")
    commit: str | None = revision if _COMMIT_HASH.fullmatch(revision) else None
    if commit is None:
        try:
            ref_path = _safe_join(repository, "refs", *_revision_parts(revision))
            candidate = ref_path.read_text(encoding="utf-8").strip()
            if _COMMIT_HASH.fullmatch(candidate):
                commit = candidate
        except (FileNotFoundError, OSError, UnicodeError, ValueError):
            return None
    snapshot = _safe_join(repository, "snapshots", commit)
    return snapshot if snapshot.is_dir() else None


def _snapshot_root_for_file(path: Path, relative_file: PurePosixPath) -> Path:
    snapshot_root = path
    for _ in relative_file.parts:
        snapshot_root = snapshot_root.parent
    if not snapshot_root.is_dir():
        raise HubDownloadError("A downloaded Hub file has no valid snapshot root.")
    return snapshot_root


def _download_or_reuse(
    *,
    repository_cache: Path,
    metadata_path: Path,
    repo_id: str,
    revision: str,
    relative_file: PurePosixPath,
    token: str | None,
    cached: _CachedFile | None,
    fallback: Path | None,
) -> Path:
    headers = {
        "Accept-Encoding": "identity",
        "User-Agent": _USER_AGENT,
    }
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    if cached is not None and cached.etag is not None:
        headers["If-None-Match"] = cached.etag

    request = Request(
        _resolve_url(repo_id, revision, relative_file),
        headers=headers,
        method="GET",
    )
    _REDIRECT_HANDLER.reset_response_metadata()
    try:
        response = _URL_OPENER.open(request, timeout=_DOWNLOAD_TIMEOUT_SECONDS)
    except HTTPError as error:
        if error.code == 304 and cached is not None:
            return cached.path
        if fallback is not None and error.code >= 500:
            return fallback
        raise _friendly_http_error(error, repo_id, revision, relative_file) from None
    except (TimeoutError, URLError) as error:
        if fallback is not None:
            return fallback
        reason = getattr(error, "reason", error)
        raise HubDownloadError(
            "Could not reach Hugging Face while resolving "
            f"{repo_id}@{revision}/{relative_file.as_posix()}: {reason}") from error

    with response:
        status = getattr(response, "status", None)
        if status is None:
            status = response.getcode()
        if status == 304 and cached is not None:
            return cached.path
        if status is not None and not 200 <= status < 300:
            raise HubDownloadError(
                f"Hugging Face returned HTTP {status} for "
                f"{repo_id}@{revision}/{relative_file.as_posix()}.")

        response_headers = _merged_response_headers(
            response.headers,
            _REDIRECT_HANDLER.response_metadata(),
        )
        commit = _validated_response_header(
            _get_header(response_headers, "X-Repo-Commit"),
            "X-Repo-Commit",
        )
        if commit is not None and not _COMMIT_HASH.fullmatch(commit):
            raise HubDownloadError("The Hub response contains an invalid X-Repo-Commit header.")
        if (commit is not None and _COMMIT_HASH.fullmatch(revision) and
                not commit.lower().startswith(revision.lower())):
            raise HubDownloadError("The Hub returned a different commit for an immutable revision.")
        etag = _validated_response_header(
            _get_header(response_headers, "X-Linked-ETag") or _get_header(response_headers, "ETag"),
            "ETag",
        )
        expected_size = _expected_size(response_headers)
        expected_sha256 = _expected_sha256(response_headers)
        snapshot_key = _snapshot_key(revision, commit)
        destination = _safe_join(
            repository_cache,
            "snapshots",
            snapshot_key,
            *relative_file.parts,
        )
        actual_size, actual_sha256 = _download_atomic(
            response,
            destination,
            expected_size=expected_size,
            expected_sha256=expected_sha256,
            immutable=commit is not None,
        )

    metadata = {
        "commit": commit,
        "etag": etag,
        "relative_file": relative_file.as_posix(),
        "repo_id": repo_id,
        "revision": revision,
        "sha256": actual_sha256,
        "size": actual_size,
        "snapshot_key": snapshot_key,
        "version": 1,
    }
    _atomic_write_json(metadata_path, metadata)
    return destination


def _download_atomic(
    response: Any,
    destination: Path,
    *,
    expected_size: int | None,
    expected_sha256: str | None,
    immutable: bool,
) -> tuple[int, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".incomplete",
    )
    temporary_path = Path(temporary_name)
    digest = hashlib.sha256()
    size = 0
    try:
        with os.fdopen(descriptor, "wb") as handle:
            while True:
                chunk = response.read(_COPY_CHUNK_SIZE)
                if not chunk:
                    break
                if not isinstance(chunk, bytes):
                    raise HubDownloadError("The Hub response returned non-binary data.")
                handle.write(chunk)
                digest.update(chunk)
                size += len(chunk)
            handle.flush()
            os.fsync(handle.fileno())

        actual_sha256 = digest.hexdigest()
        if expected_size is not None and size != expected_size:
            raise HubDownloadError(
                "Downloaded Hub file has an unexpected size: "
                f"expected {expected_size} bytes, received {size} bytes.")
        if expected_sha256 is not None and actual_sha256 != expected_sha256:
            raise HubDownloadError("Downloaded Hub file failed its SHA-256 integrity check.")

        if immutable and destination.is_file():
            existing_sha256 = _sha256_file(destination)
            if (destination.stat().st_size != size or existing_sha256 != actual_sha256):
                raise HubDownloadError(
                    "An immutable Hub commit snapshot conflicts with the "
                    f"existing cached file: {destination}")
            temporary_path.unlink()
        else:
            os.replace(temporary_path, destination)
        return size, actual_sha256
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _read_cached_file(
    repository_cache: Path,
    metadata_path: Path,
    repo_id: str,
    revision: str,
    relative_file: PurePosixPath,
) -> _CachedFile | None:
    metadata = _read_cached_json_object(metadata_path)
    if metadata is None:
        return None
    if (metadata.get("version") != 1 or metadata.get("repo_id") != repo_id or
            metadata.get("revision") != revision or
            metadata.get("relative_file") != relative_file.as_posix()):
        return None

    snapshot_key = metadata.get("snapshot_key")
    if not isinstance(snapshot_key, str) or not _HEX_SHA256.fullmatch(snapshot_key):
        return None
    path = _safe_join(
        repository_cache,
        "snapshots",
        snapshot_key,
        *relative_file.parts,
    )
    if not path.is_file():
        return None

    size = metadata.get("size")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        return None
    try:
        if path.stat().st_size != size:
            return None
    except OSError:
        return None

    etag = metadata.get("etag")
    if etag is not None:
        try:
            etag = _validated_response_header(etag, "ETag")
        except HubDownloadError:
            return None
    sha256 = metadata.get("sha256")
    if not isinstance(sha256, str) or not _HEX_SHA256.fullmatch(sha256):
        sha256 = None
    return _CachedFile(path=path, etag=etag, size=size, sha256=sha256)


def _read_cached_json_object(path: Path) -> dict[str, Any] | None:
    try:
        return parse_json_object(path.read_bytes(), source=path)
    except (OSError, ValueError):
        return None


def _find_legacy_cached_file(
    root: Path,
    repo_id: str,
    revision: str,
    relative_file: PurePosixPath,
) -> Path | None:
    """Read snapshots created by ``huggingface_hub`` without importing it."""
    repository = _safe_join(root, f"models--{repo_id.replace('/', '--')}")
    commit: str | None = revision if _COMMIT_HASH.fullmatch(revision) else None
    if commit is None:
        try:
            ref_path = _safe_join(repository, "refs", *_revision_parts(revision))
            candidate = ref_path.read_text(encoding="utf-8").strip()
            if _COMMIT_HASH.fullmatch(candidate):
                commit = candidate
        except (FileNotFoundError, OSError, UnicodeError, ValueError):
            return None
    candidate_path = _safe_join(
        repository,
        "snapshots",
        commit,
        *relative_file.parts,
    )
    return candidate_path if candidate_path.is_file() else None


def _resolve_url(
    repo_id: str,
    revision: str,
    relative_file: PurePosixPath,
) -> str:
    encoded_repo = "/".join(quote(part, safe="") for part in repo_id.split("/"))
    encoded_file = "/".join(quote(part, safe="") for part in relative_file.parts)
    return (f"{_HUGGING_FACE_ENDPOINT}/{encoded_repo}/resolve/"
            f"{quote(revision, safe='')}/{encoded_file}")


def _cache_root(cache_dir: str | os.PathLike[str] | None) -> Path:
    if cache_dir is not None:
        root = Path(cache_dir).expanduser()
    elif os.environ.get("HF_HUB_CACHE"):
        root = Path(os.environ["HF_HUB_CACHE"]).expanduser()
    elif os.environ.get("HUGGINGFACE_HUB_CACHE"):
        root = Path(os.environ["HUGGINGFACE_HUB_CACHE"]).expanduser()
    elif os.environ.get("HF_HOME"):
        root = Path(os.environ["HF_HOME"]).expanduser() / "hub"
    elif os.environ.get("XDG_CACHE_HOME"):
        root = Path(os.environ["XDG_CACHE_HOME"]).expanduser() / "huggingface" / "hub"
    else:
        root = Path.home() / ".cache" / "huggingface" / "hub"
    if root.exists() and not root.is_dir():
        raise NotADirectoryError(f"The model cache path is not a directory: {root}")
    return root


def _validate_repo_id(repo_id: str) -> str:
    if not isinstance(repo_id, str):
        raise TypeError("`repo_id` must be a string.")
    if not repo_id or len(repo_id) > 256 or "\\" in repo_id or "\x00" in repo_id:
        raise ValueError(f"Invalid Hugging Face repository identifier: {repo_id!r}")
    parts = repo_id.split("/")
    if len(parts) not in (1, 2) or any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"Invalid Hugging Face repository identifier: {repo_id!r}")
    if any(any(ord(character) < 32 for character in part) for part in parts):
        raise ValueError(f"Invalid Hugging Face repository identifier: {repo_id!r}")
    return repo_id


def _validate_revision(revision: str) -> str:
    if not isinstance(revision, str):
        raise TypeError("`revision` must be a string or None.")
    if (not revision or len(revision) > 512 or "\\" in revision or "\x00" in revision):
        raise ValueError(f"Invalid Hugging Face revision: {revision!r}")
    _revision_parts(revision)
    return revision


def _revision_parts(revision: str) -> tuple[str, ...]:
    parts = tuple(revision.split("/"))
    if any(part in ("", ".", "..") or any(ord(character) < 32 for character in part) for part in parts):
        raise ValueError(f"Invalid Hugging Face revision: {revision!r}")
    return parts


def _validate_repo_path(subfolder: str, filename: str) -> PurePosixPath:
    values: list[str] = []
    if subfolder:
        values.append(_pathlike_to_posix(subfolder, "subfolder"))
    values.append(_pathlike_to_posix(filename, "filename"))
    combined = "/".join(values)
    if "\x00" in combined or "\\" in combined or combined.startswith("/"):
        raise ValueError(f"Invalid repository file path: {combined!r}")
    raw_parts = combined.split("/")
    path = PurePosixPath(combined)
    if (not combined or combined.endswith("/") or any(part in ("", ".", "..") for part in raw_parts)):
        raise ValueError(f"Invalid repository file path: {combined!r}")
    return path


def _pathlike_to_posix(value: str | os.PathLike[str], name: str) -> str:
    if not isinstance(value, (str, os.PathLike)):
        raise TypeError(f"`{name}` must be path-like.")
    if isinstance(value, Path):
        return value.as_posix()
    return os.fspath(value)


def _resolve_token(token: str | bool | None) -> str | None:
    if token is False:
        return None
    if isinstance(token, str):
        candidate = token
    elif token is True:
        candidate = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "")
        if not candidate:
            raise ValueError(
                "`token=True` requires `HF_TOKEN` or "
                "`HUGGING_FACE_HUB_TOKEN` in the current environment.")
    elif token is None:
        if _env_truthy("HF_HUB_DISABLE_IMPLICIT_TOKEN"):
            return None
        candidate = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "")
    else:
        raise TypeError("`token` must be a string, boolean, or None.")

    if not candidate:
        return None
    if "\r" in candidate or "\n" in candidate:
        raise ValueError("A Hub token cannot contain newline characters.")
    return candidate


def _offline_mode_enabled() -> bool:
    return _env_truthy("HF_HUB_OFFLINE") or _env_truthy("VOICEHUB_OFFLINE")


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUE_ENV_VALUES


def _metadata_path(
    repository_cache: Path,
    revision: str,
    relative_file: PurePosixPath,
) -> Path:
    return _safe_join(
        repository_cache,
        "refs",
        _stable_key(revision),
        f"{_stable_key(relative_file.as_posix())}.json",
    )


def _snapshot_key(revision: str, commit: str | None) -> str:
    if commit is not None:
        return _stable_key(f"commit:{commit}")
    return _stable_key(f"revision:{revision}")


def _stable_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _safe_join(root: Path, *parts: str) -> Path:
    candidate = root.joinpath(*parts)
    resolved_root = root.resolve(strict=False)
    resolved_candidate = candidate.resolve(strict=False)
    try:
        resolved_candidate.relative_to(resolved_root)
    except ValueError:
        raise ValueError(f"Cache path escapes its root: {candidate}") from None
    return candidate


def _expected_size(headers: Mapping[str, str]) -> int | None:
    raw_size = (_get_header(headers, "X-Linked-Size") or _get_header(headers, "Content-Length"))
    if raw_size is None:
        return None
    try:
        size = int(raw_size)
    except (TypeError, ValueError):
        raise HubDownloadError("The Hub response contains an invalid file size.") from None
    if size < 0:
        raise HubDownloadError("The Hub response contains a negative file size.")
    return size


def _expected_sha256(headers: Mapping[str, str]) -> str | None:
    explicit = _get_header(headers, "X-Checksum-Sha256")
    if explicit is not None:
        normalized = explicit.strip().lower()
        if not _HEX_SHA256.fullmatch(normalized):
            raise HubDownloadError("The Hub response contains an invalid SHA-256 checksum.")
        return normalized

    digest_header = _get_header(headers, "Digest")
    if digest_header:
        for item in digest_header.split(","):
            algorithm, separator, encoded = item.strip().partition("=")
            if separator and algorithm.lower() in ("sha-256", "sha256"):
                try:
                    raw_digest = base64.b64decode(encoded, validate=True)
                except (ValueError, TypeError):
                    raise HubDownloadError("The Hub response contains an invalid Digest header.") from None
                if len(raw_digest) != hashlib.sha256().digest_size:
                    raise HubDownloadError("The Hub response contains an invalid SHA-256 digest.")
                return raw_digest.hex()

    normalized_etag = _normalize_etag(_get_header(headers, "X-Linked-ETag"))
    if normalized_etag is not None and _HEX_SHA256.fullmatch(normalized_etag):
        return normalized_etag.lower()
    return None


def _normalize_etag(etag: str | None) -> str | None:
    if etag is None:
        return None
    normalized = etag.strip()
    if normalized.startswith("W/"):
        normalized = normalized[2:].strip()
    if len(normalized) >= 2 and normalized[0] == normalized[-1] == '"':
        normalized = normalized[1:-1]
    if normalized.lower().startswith("sha256:"):
        normalized = normalized[7:]
    return normalized


def _validated_response_header(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    if len(value) > 1024 or "\r" in value or "\n" in value:
        raise HubDownloadError(f"The Hub response contains an invalid {name} header.")
    return value


def _get_header(headers: Mapping[str, str], name: str) -> str | None:
    value = headers.get(name)
    if value is not None:
        return value
    lowered = name.lower()
    for key, candidate in headers.items():
        if key.lower() == lowered:
            return candidate
    return None


def _merged_response_headers(
    response_headers: Mapping[str, str],
    redirect_metadata: Mapping[str, str],
) -> dict[str, str]:
    """Merge final headers with trusted metadata from the Hub redirect."""
    merged = {str(key): str(value) for key, value in response_headers.items()}
    merged.update(redirect_metadata)
    return merged


def _friendly_http_error(
    error: HTTPError,
    repo_id: str,
    revision: str,
    relative_file: PurePosixPath,
) -> BaseException:
    location = f"{repo_id}@{revision}/{relative_file.as_posix()}"
    if error.code == 404:
        return FileNotFoundError(f"Could not find the requested Hub file: {location}.")
    if error.code in (401, 403):
        return PermissionError(
            f"Hugging Face denied access to {location} (HTTP {error.code}). "
            "Check the repository permissions and token.")
    return HubDownloadError(f"Hugging Face returned HTTP {error.code} while resolving {location}.")


def _atomic_write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_COPY_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _remove_request_header(request: Request, name: str) -> None:
    lowered = name.lower()
    for collection in (request.headers, request.unredirected_hdrs):
        for key in tuple(collection):
            if key.lower() == lowered:
                del collection[key]
