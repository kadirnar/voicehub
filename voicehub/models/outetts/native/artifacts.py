"""Coherent immutable artifact resolution for VoiceHub-native OuteTTS."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from voicehub.checkpointing import MANIFEST_NAME, VoiceHubManifest
from voicehub.hub import read_json_file, resolve_pretrained_file
from voicehub.path_utils import is_explicit_local_path

from .metadata import OUTETTS_CHECKPOINTS, OUTETTS_DAC

_CONFIG = "config.json"
_TOKENIZER = "tokenizer.json"
_TOKENIZER_CONFIG = "tokenizer_config.json"
_CHECKPOINT = "model.safetensors"
_CHECKPOINT_INDEX = "model.safetensors.index.json"
_IMMUTABLE_REVISION = re.compile(r"^[0-9a-fA-F]{40,64}$")


@dataclass(frozen=True, slots=True)
class OuteTTSArtifacts:
    """Language model and tokenizer from one immutable snapshot."""

    source: str
    revision: str | None
    config: Path
    tokenizer: Path
    checkpoint: Path
    tokenizer_config: Path | None = None
    manifest: VoiceHubManifest | None = None

    @property
    def root(self) -> Path:
        return self.config.parent


@dataclass(frozen=True, slots=True)
class OuteTTSDacArtifacts:
    """Either a native safe DAC export or the pinned IBM conversion source."""

    source: str
    revision: str | None
    checkpoint: Path
    config: Path | None
    legacy: bool


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def verify_file(
    path: str | Path,
    *,
    expected_size: int,
    expected_sha256: str,
    owner: str,
) -> Path:
    source = Path(path).expanduser().resolve()
    actual_size = source.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"{owner} file-size verification failed: expected "
            f"{expected_size}, found {actual_size}.")
    actual_digest = file_sha256(source)
    if actual_digest != expected_sha256:
        raise ValueError(
            f"{owner} SHA-256 verification failed: expected "
            f"{expected_sha256}, found {actual_digest}.")
    return source


def _required(root: Path, filename: str, *, owner: str) -> Path:
    path = (root / filename).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Native {owner} requires {filename!r} in {root}.")
    return path


def _optional(root: Path, filename: str) -> Path | None:
    path = (root / filename).resolve()
    return path if path.is_file() else None


def _safe_shards(index: Path) -> tuple[str, ...]:
    document = read_json_file(index)
    weight_map = document.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("OuteTTS Safetensors index requires a non-empty `weight_map`.")
    names: set[str] = set()
    for tensor_name, shard_name in weight_map.items():
        if (not isinstance(tensor_name, str) or not tensor_name or not isinstance(shard_name, str) or
                not shard_name):
            raise ValueError("OuteTTS Safetensors index contains an invalid entry.")
        shard = PurePosixPath(shard_name)
        if (shard.is_absolute() or len(shard.parts) != 1 or ".." in shard.parts or
                not shard_name.endswith(".safetensors")):
            raise ValueError(f"Unsafe OuteTTS checkpoint shard {shard_name!r}.")
        names.add(shard_name)
    return tuple(sorted(names))


def _local_checkpoint(root: Path) -> Path:
    single = _optional(root, _CHECKPOINT)
    index = _optional(root, _CHECKPOINT_INDEX)
    if single is not None and index is not None:
        raise ValueError("OuteTTS artifact contains both single-file and sharded "
                         "checkpoints.")
    checkpoint = single or index
    if checkpoint is None:
        raise FileNotFoundError(
            f"Native OuteTTS requires {_CHECKPOINT!r} or "
            f"{_CHECKPOINT_INDEX!r} in {root}.")
    if checkpoint.name.endswith(".index.json"):
        for shard in _safe_shards(checkpoint):
            _required(root, shard, owner="OuteTTS")
    return checkpoint


def _remote_optional(
    source: str,
    filename: str,
    *,
    revision: str,
    cache_dir: str | None,
    token: str | bool | None,
    local_files_only: bool,
) -> Path | None:
    try:
        return resolve_pretrained_file(
            source,
            filename,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        )
    except FileNotFoundError:
        return None


def _remote_checkpoint(
    source: str,
    *,
    revision: str,
    cache_dir: str | None,
    token: str | bool | None,
    local_files_only: bool,
) -> Path:
    checkpoint = _remote_optional(
        source,
        _CHECKPOINT,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    if checkpoint is not None:
        return checkpoint
    index = resolve_pretrained_file(
        source,
        _CHECKPOINT_INDEX,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    for shard in _safe_shards(index):
        resolve_pretrained_file(
            source,
            shard,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        )
    return index


def _validate_reference_assets(artifacts: OuteTTSArtifacts) -> None:
    reference = OUTETTS_CHECKPOINTS.get(artifacts.source)
    if reference is None or artifacts.revision != reference["revision"]:
        return
    verify_file(
        artifacts.config,
        expected_size=reference["config_size"],
        expected_sha256=reference["config_sha256"],
        owner="Published OuteTTS config",
    )
    verify_file(
        artifacts.tokenizer,
        expected_size=reference["tokenizer_size"],
        expected_sha256=reference["tokenizer_sha256"],
        owner="Published OuteTTS tokenizer",
    )
    if artifacts.checkpoint.name == _CHECKPOINT:
        verify_file(
            artifacts.checkpoint,
            expected_size=reference["checkpoint_size"],
            expected_sha256=reference["checkpoint_sha256"],
            owner="Published OuteTTS checkpoint",
        )


def resolve_outetts_artifacts(
    source: str | Path,
    *,
    tokenizer_source: str | Path | None = None,
    revision: str | None = None,
    cache_dir: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
) -> OuteTTSArtifacts:
    """Resolve one local native export or immutable Hub checkpoint."""
    if not isinstance(source, (str, Path)) or not str(source).strip():
        raise ValueError("OuteTTS `source` must be a non-empty path or Hub ID.")
    source_path = Path(source).expanduser()
    if source_path.exists():
        if not source_path.is_dir():
            raise NotADirectoryError("Native OuteTTS requires an artifact directory.")
        root = source_path.resolve()
        manifest = None
        manifest_path = root / MANIFEST_NAME
        if manifest_path.is_file():
            manifest = VoiceHubManifest.load(manifest_path)
            if manifest.architecture != "outetts":
                raise ValueError("Local manifest does not describe an OuteTTS artifact.")
            manifest.verify(root)
        tokenizer_root = (root if tokenizer_source is None else Path(tokenizer_source).expanduser().resolve())
        if not tokenizer_root.is_dir():
            raise NotADirectoryError("A local OuteTTS tokenizer source must be a directory.")
        return OuteTTSArtifacts(
            source=str(root),
            revision=None,
            config=_required(root, _CONFIG, owner="OuteTTS"),
            tokenizer=_required(
                tokenizer_root,
                _TOKENIZER,
                owner="OuteTTS tokenizer",
            ),
            tokenizer_config=_optional(
                tokenizer_root,
                _TOKENIZER_CONFIG,
            ),
            checkpoint=_local_checkpoint(root),
            manifest=manifest,
        )
    if is_explicit_local_path(source):
        raise FileNotFoundError(f"OuteTTS model path was not found: {source_path}.")

    repo_id = str(source)
    reference = OUTETTS_CHECKPOINTS.get(repo_id)
    if revision is None and reference is None:
        raise ValueError("Unknown OuteTTS Hub repositories require an explicit immutable "
                         "`revision`.")
    resolved_revision = (
        str(reference["revision"]) if revision is None and reference is not None else str(revision))
    if _IMMUTABLE_REVISION.fullmatch(resolved_revision) is None:
        raise ValueError("Remote OuteTTS artifacts require an immutable commit-hash "
                         "`revision`.")
    tokenizer_repo = repo_id if tokenizer_source is None else str(tokenizer_source)
    if Path(tokenizer_repo).expanduser().exists():
        tokenizer_root = Path(tokenizer_repo).expanduser().resolve()
        tokenizer = _required(
            tokenizer_root,
            _TOKENIZER,
            owner="OuteTTS tokenizer",
        )
        tokenizer_config = _optional(tokenizer_root, _TOKENIZER_CONFIG)
    else:
        if is_explicit_local_path(tokenizer_repo):
            raise FileNotFoundError(f"OuteTTS tokenizer path was not found: {tokenizer_repo}.")
        tokenizer = resolve_pretrained_file(
            tokenizer_repo,
            _TOKENIZER,
            revision=resolved_revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        )
        tokenizer_config = _remote_optional(
            tokenizer_repo,
            _TOKENIZER_CONFIG,
            revision=resolved_revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        )
    artifacts = OuteTTSArtifacts(
        source=repo_id,
        revision=resolved_revision,
        config=resolve_pretrained_file(
            repo_id,
            _CONFIG,
            revision=resolved_revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        ),
        tokenizer=tokenizer,
        tokenizer_config=tokenizer_config,
        checkpoint=_remote_checkpoint(
            repo_id,
            revision=resolved_revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        ),
    )
    _validate_reference_assets(artifacts)
    return artifacts


def resolve_outetts_dac_artifacts(
    source: str | Path | None,
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
) -> OuteTTSDacArtifacts:
    """Resolve a native DAC directory or the exact pinned IBM checkpoint."""
    if source is None:
        source = OUTETTS_DAC["repository"]
    source_path = Path(source).expanduser()
    if source_path.exists():
        if source_path.is_dir():
            root = source_path.resolve()
            return OuteTTSDacArtifacts(
                source=str(root),
                revision=None,
                checkpoint=_required(
                    root,
                    _CHECKPOINT,
                    owner="OuteTTS DAC",
                ),
                config=_required(root, _CONFIG, owner="OuteTTS DAC"),
                legacy=False,
            )
        if source_path.suffix.lower() not in {".pth", ".pt"}:
            raise ValueError(
                "OuteTTS DAC files must be the pinned `.pth` conversion "
                "source; steady-state artifacts use a directory containing "
                "Safetensors.")
        verify_file(
            source_path,
            expected_size=OUTETTS_DAC["size"],
            expected_sha256=OUTETTS_DAC["sha256"],
            owner="Pinned OuteTTS DAC",
        )
        return OuteTTSDacArtifacts(
            source=str(source_path.resolve()),
            revision=None,
            checkpoint=source_path.resolve(),
            config=None,
            legacy=True,
        )
    if is_explicit_local_path(source):
        raise FileNotFoundError(f"OuteTTS DAC path was not found: {source_path}.")
    if str(source) != OUTETTS_DAC["repository"]:
        raise ValueError(
            "Native OuteTTS only accepts a local DAC Safetensors export or "
            f"the audited {OUTETTS_DAC['repository']!r} checkpoint.")
    resolved_revision = revision or OUTETTS_DAC["revision"]
    if resolved_revision != OUTETTS_DAC["revision"]:
        raise ValueError("OuteTTS DAC revision is not the audited immutable release.")
    checkpoint = resolve_pretrained_file(
        str(source),
        OUTETTS_DAC["filename"],
        revision=resolved_revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    verify_file(
        checkpoint,
        expected_size=OUTETTS_DAC["size"],
        expected_sha256=OUTETTS_DAC["sha256"],
        owner="Pinned OuteTTS DAC",
    )
    return OuteTTSDacArtifacts(
        source=str(source),
        revision=resolved_revision,
        checkpoint=checkpoint,
        config=None,
        legacy=True,
    )


__all__ = [
    "OuteTTSArtifacts",
    "OuteTTSDacArtifacts",
    "file_sha256",
    "resolve_outetts_artifacts",
    "resolve_outetts_dac_artifacts",
    "verify_file",
]
