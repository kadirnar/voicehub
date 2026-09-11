"""Artifact resolution for VoiceHub-native NeMo QuartzNet CTC."""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from voicehub.hub import resolve_pretrained_file
from voicehub.models.asr_nemo.native.checkpoint import (
    NATIVE_NEMO_CTC_FILENAME,
    convert_nemo_quartznet_checkpoint,
    file_sha256,
)
from voicehub.models.asr_nemo.native.metadata import (
    QUARTZNET_FILENAME,
    QUARTZNET_REPOSITORY,
    QUARTZNET_SHA256,
    QUARTZNET_SIZE_BYTES,
    QUARTZNET_URL,
    QUARTZNET_VERSION,
)
from voicehub.path_utils import is_explicit_local_path

_OFFICIAL_ALIASES = frozenset({
    "nemo-quartznet",
    "nemo-quartznet15x5",
    "nvidia/stt_en_quartznet15x5",
    QUARTZNET_REPOSITORY,
    "stt_en_quartznet15x5",
})
_UNSUPPORTED_FAMILY_MARKERS = (
    "canary",
    "citrinet",
    "conformer",
    "fastconformer",
    "jasper",
    "nemotron",
    "parakeet",
    "rnnt",
    "tdt",
    "transducer",
)


@dataclass(frozen=True, slots=True)
class NeMoCTCArtifacts:
    checkpoint: Path
    config: Path
    source: str
    revision: str | None
    converted_from_nemo: bool = False

    def __post_init__(self) -> None:
        for name in ("checkpoint", "config"):
            path = Path(getattr(self, name)).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Native NeMo CTC {name} was not found: {path}.")
            object.__setattr__(self, name, path)


def _default_cache_root(cache_dir: str | Path | None) -> Path:
    if cache_dir is not None:
        root = Path(cache_dir).expanduser()
    else:
        configured = os.environ.get("VOICEHUB_CACHE")
        root = (Path(configured).expanduser() if configured else Path.home() / ".cache" / "voicehub")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _official_cache_path(cache_dir: str | Path | None) -> Path:
    digest = hashlib.sha256(QUARTZNET_URL.encode("utf-8")).hexdigest()[:20]
    directory = (
        _default_cache_root(cache_dir) / "ngc" / "nemo-quartznet-ctc" / f"{QUARTZNET_VERSION}-{digest}")
    directory.mkdir(parents=True, exist_ok=True)
    return directory / QUARTZNET_FILENAME


def _download_official_checkpoint(
    *,
    cache_dir: str | Path | None,
    local_files_only: bool,
) -> Path:
    destination = _official_cache_path(cache_dir)
    if destination.is_file():
        if (destination.stat().st_size == QUARTZNET_SIZE_BYTES and
                file_sha256(destination) == QUARTZNET_SHA256):
            return destination
        raise ValueError(f"Cached QuartzNet checkpoint failed integrity validation: {destination}.")
    if local_files_only:
        raise FileNotFoundError(
            "The pinned NVIDIA QuartzNet checkpoint is not present in the "
            f"local VoiceHub cache: {destination}.")

    request = Request(
        QUARTZNET_URL,
        headers={
            "Accept-Encoding": "identity",
            "User-Agent": "voicehub",
        },
        method="GET",
    )
    temporary_path: Path | None = None
    try:
        with (
                urlopen(request, timeout=30.0) as response,
                tempfile.NamedTemporaryFile(
                    mode="wb",
                    prefix=f".{QUARTZNET_FILENAME}.",
                    suffix=".partial",
                    dir=destination.parent,
                    delete=False,
                ) as temporary,
        ):
            final_url = response.geturl()
            if urlsplit(final_url).scheme.lower() != "https":
                raise OSError("VoiceHub refused an insecure NGC redirect.")
            temporary_path = Path(temporary.name)
            digest = hashlib.sha256()
            size = 0
            while chunk := response.read(1024 * 1024):
                size += len(chunk)
                if size > QUARTZNET_SIZE_BYTES:
                    raise OSError("NGC returned more data than the pinned QuartzNet artifact.")
                digest.update(chunk)
                temporary.write(chunk)
            temporary.flush()
            os.fsync(temporary.fileno())
        if size != QUARTZNET_SIZE_BYTES:
            raise OSError("NGC QuartzNet size mismatch: "
                          f"expected {QUARTZNET_SIZE_BYTES}, found {size}.")
        actual_sha = digest.hexdigest()
        if actual_sha != QUARTZNET_SHA256:
            raise OSError(
                "NGC QuartzNet SHA-256 mismatch: "
                f"expected {QUARTZNET_SHA256}, found {actual_sha}.")
        os.replace(temporary_path, destination)
        temporary_path = None
        return destination
    except (HTTPError, URLError) as error:
        raise OSError("Could not download the pinned NVIDIA QuartzNet checkpoint.") from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _converted_artifacts(
    source: Path,
    *,
    destination_root: Path,
) -> NeMoCTCArtifacts:
    if file_sha256(source) != QUARTZNET_SHA256:
        raise ValueError(
            "Native NeMo conversion supports only the hash-pinned NVIDIA "
            "QuartzNet15x5 archive. Parakeet, Canary, Citrinet, Conformer, "
            "RNN-T, and TDT checkpoints require their own exact architecture.")
    destination = destination_root / ".voicehub-native" / "nemo-quartznet-ctc"
    checkpoint = destination / NATIVE_NEMO_CTC_FILENAME
    config = destination / "config.json"
    if not checkpoint.is_file() or not config.is_file():
        convert_nemo_quartznet_checkpoint(
            source,
            destination,
            expected_sha256=QUARTZNET_SHA256,
        )
    return NeMoCTCArtifacts(
        checkpoint=checkpoint,
        config=config,
        source=str(source),
        revision=None,
        converted_from_nemo=True,
    )


def _local_artifacts(source: Path) -> NeMoCTCArtifacts:
    root = source if source.is_dir() else source.parent
    checkpoint = (
        source if source.is_file() and source.suffix.lower() == ".safetensors" else root /
        NATIVE_NEMO_CTC_FILENAME)
    config = root / "config.json"
    if checkpoint.is_file() and config.is_file():
        return NeMoCTCArtifacts(
            checkpoint=checkpoint,
            config=config,
            source=str(source),
            revision=None,
        )

    nemo_source = source if source.is_file() and source.suffix.lower() == ".nemo" else None
    if source.is_dir():
        candidates = sorted(
            path for path in source.iterdir() if path.is_file() and path.suffix.lower() == ".nemo")
        if len(candidates) > 1:
            raise ValueError(
                "A local NeMo CTC directory may contain at most one "
                "top-level `.nemo` source.")
        nemo_source = candidates[0] if candidates else None
    if nemo_source is not None:
        return _converted_artifacts(
            nemo_source,
            destination_root=root,
        )
    if source.is_file() and source.suffix.lower() == ".ckpt":
        raise ValueError(
            "Generic NeMo `.ckpt` files do not identify a verified model "
            "graph. Convert the audited QuartzNet `.nemo` archive or provide "
            "a native `model.safetensors` plus `config.json`.")
    raise FileNotFoundError(
        f"No complete native NeMo CTC artifact was found in {root}. Expected "
        "`model.safetensors` plus `config.json`, or the audited NVIDIA "
        "QuartzNet15x5 `.nemo` archive.")


def _reject_known_unsupported_family(source: str) -> None:
    normalized = source.lower().replace("_", "-")
    markers = tuple(marker for marker in _UNSUPPORTED_FAMILY_MARKERS if marker in normalized)
    if markers:
        raise ValueError(
            f"NeMo source {source!r} is not the verified QuartzNet15x5 "
            "character-CTC graph. Native `asr_nemo` rejects distinct "
            f"architecture family marker(s): {', '.join(markers)}.")


def resolve_nemo_ctc_artifacts(
    source: str | Path,
    *,
    checkpoint_filename: str,
    revision: str | None,
    cache_dir: str | Path | None,
    token: str | bool | None,
    local_files_only: bool,
) -> NeMoCTCArtifacts:
    """Resolve a native directory or convert the pinned NGC release once."""
    source_path = Path(source).expanduser()
    if source_path.exists():
        return _local_artifacts(source_path.resolve())
    if is_explicit_local_path(source):
        raise FileNotFoundError(f"Local NeMo CTC model path was not found: {source_path}.")

    source_name = str(source)
    if source_name in _OFFICIAL_ALIASES:
        if revision is not None and revision != QUARTZNET_VERSION:
            raise ValueError(
                "Official QuartzNet aliases are pinned to NGC version "
                f"{QUARTZNET_VERSION}; use a custom native repository for "
                "another revision.")
        source_checkpoint = _download_official_checkpoint(
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        artifacts = _converted_artifacts(
            source_checkpoint,
            destination_root=source_checkpoint.parent,
        )
        return NeMoCTCArtifacts(
            checkpoint=artifacts.checkpoint,
            config=artifacts.config,
            source=source_name,
            revision=QUARTZNET_VERSION,
            converted_from_nemo=True,
        )

    _reject_known_unsupported_family(source_name)
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
    return NeMoCTCArtifacts(
        checkpoint=checkpoint,
        config=config,
        source=source_name,
        revision=revision,
    )


__all__ = [
    "NeMoCTCArtifacts",
    "resolve_nemo_ctc_artifacts",
]
