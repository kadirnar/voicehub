"""Pinned, framework-free artifact resolution for Supertonic 3."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from voicehub.checkpointing import ONNXModel, onnx_semantic_fingerprint, read_onnx_model
from voicehub.hub import resolve_pretrained_file
from voicehub.models.supertonic.native.metadata import (
    SUPERTONIC_CHECKPOINT_REPOSITORY,
    SUPERTONIC_CHECKPOINT_REVISION,
    SUPERTONIC_GRAPH_FILES,
    SUPERTONIC_GRAPH_INTEGRITY,
    SUPERTONIC_PROCESSOR_INTEGRITY,
    SUPERTONIC_STYLE_INTEGRITY,
)
from voicehub.path_utils import is_explicit_local_path

_VOICE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@dataclass(frozen=True, slots=True)
class SupertonicArtifacts:
    """One coherent graph/processor/native-weight snapshot."""

    source: str | Path
    revision: str | None
    graphs: Mapping[str, Path]
    graph_models: Mapping[str, ONNXModel]
    architecture_config: Path
    unicode_indexer: Path
    native_weights: Mapping[str, Path]
    local_root: Path | None
    official_snapshot: bool

    def without_materialized_graphs(self) -> SupertonicArtifacts:
        """Drop parsed initializer bytes after native modules own the state."""
        return SupertonicArtifacts(
            source=self.source,
            revision=self.revision,
            graphs=self.graphs,
            graph_models=MappingProxyType({}),
            architecture_config=self.architecture_config,
            unicode_indexer=self.unicode_indexer,
            native_weights=self.native_weights,
            local_root=self.local_root,
            official_snapshot=self.official_snapshot,
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require_integrity(
    path: Path,
    *,
    size: int,
    sha256: str,
    label: str,
) -> None:
    actual_size = path.stat().st_size
    if actual_size != size:
        raise ValueError(f"{label} has {actual_size} bytes; expected {size}.")
    actual_sha = _sha256(path)
    if actual_sha != sha256:
        raise ValueError(f"{label} SHA-256 mismatch: expected {sha256}, "
                         f"found {actual_sha}.")


def _local_layout(source: Path) -> tuple[Path, Path]:
    candidates = (
        source,
        source / "native_export",
    )
    for root in candidates:
        onnx = root / "onnx"
        if all((onnx / filename).is_file() for filename in SUPERTONIC_GRAPH_FILES.values()):
            return root, onnx
        if all((root / filename).is_file() for filename in SUPERTONIC_GRAPH_FILES.values()):
            return root, root
    raise FileNotFoundError(
        "A local Supertonic artifact must contain all four reviewed ONNX "
        "graphs under `onnx/` (or directly in the artifact root).")


def _native_weight_paths(root: Path) -> dict[str, Path]:
    directories = (
        root / "native_weights",
        root,
    )
    result = {}
    for role in SUPERTONIC_GRAPH_FILES:
        filename = f"{role}.safetensors"
        for directory in directories:
            candidate = directory / filename
            if candidate.is_file():
                result[role] = candidate.resolve()
                break
    if result and set(result) != set(SUPERTONIC_GRAPH_FILES):
        missing = ", ".join(sorted(set(SUPERTONIC_GRAPH_FILES) - set(result)))
        raise FileNotFoundError(
            "A fine-tuned Supertonic artifact must contain a native "
            f"Safetensors file for every graph; missing: {missing}.")
    return result


def _review_graph(
    role: str,
    path: Path,
    *,
    require_byte_integrity: bool,
) -> ONNXModel:
    size, digest, semantic = SUPERTONIC_GRAPH_INTEGRITY[role]
    if require_byte_integrity:
        _require_integrity(
            path,
            size=size,
            sha256=digest,
            label=f"Supertonic {role} graph",
        )
    model = read_onnx_model(path)
    actual_semantic = onnx_semantic_fingerprint(model)
    if actual_semantic != semantic:
        raise ValueError(
            f"Supertonic {role} graph semantics differ from the reviewed "
            f"release: expected {semantic}, found {actual_semantic}.")
    return model


def resolve_supertonic_artifacts(
    source: str | Path,
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
    verify_integrity: bool = True,
) -> SupertonicArtifacts:
    """Resolve and structurally review all graphs before execution."""
    if not isinstance(verify_integrity, bool):
        raise TypeError("`verify_integrity` must be a boolean.")
    source_path = Path(source).expanduser()
    source_string = str(source)
    local_root: Path | None = None
    if source_path.is_dir():
        local_root, graph_root = _local_layout(source_path.resolve())
        graphs = {
            role: (graph_root / filename).resolve()
            for role, filename in SUPERTONIC_GRAPH_FILES.items()
        }
        architecture_config = graph_root / "tts.json"
        unicode_indexer = graph_root / "unicode_indexer.json"
        for path, label in (
            (architecture_config, "tts.json"),
            (unicode_indexer, "unicode_indexer.json"),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"Local Supertonic artifact is missing {label}: {path}.")
        native_weights = _native_weight_paths(local_root)
        resolved_revision = None
        official_snapshot = False
    else:
        if is_explicit_local_path(source):
            raise FileNotFoundError(f"Local Supertonic path was not found: {source_path}.")
        resolved_revision = (
            revision or
            (SUPERTONIC_CHECKPOINT_REVISION if source_string == SUPERTONIC_CHECKPOINT_REPOSITORY else None))
        graphs = {
            role:
            resolve_pretrained_file(
                source_string,
                filename,
                subfolder="onnx",
                revision=resolved_revision,
                cache_dir=cache_dir,
                token=token,
                local_files_only=local_files_only,
            )
            for role, filename in SUPERTONIC_GRAPH_FILES.items()
        }
        architecture_config = resolve_pretrained_file(
            source_string,
            "tts.json",
            subfolder="onnx",
            revision=resolved_revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        )
        unicode_indexer = resolve_pretrained_file(
            source_string,
            "unicode_indexer.json",
            subfolder="onnx",
            revision=resolved_revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        )
        native_weights = {}
        official_snapshot = (
            source_string == SUPERTONIC_CHECKPOINT_REPOSITORY and
            resolved_revision == SUPERTONIC_CHECKPOINT_REVISION)
    require_bytes = verify_integrity or official_snapshot
    if require_bytes:
        for filename, (size, digest) in (SUPERTONIC_PROCESSOR_INTEGRITY.items()):
            path = (architecture_config if filename == "tts.json" else unicode_indexer)
            _require_integrity(
                path,
                size=size,
                sha256=digest,
                label=f"Supertonic {filename}",
            )
    graph_models = {
        role: _review_graph(
            role,
            path,
            require_byte_integrity=require_bytes,
        )
        for role, path in graphs.items()
    }
    return SupertonicArtifacts(
        source=source,
        revision=resolved_revision,
        graphs=MappingProxyType(graphs),
        graph_models=MappingProxyType(graph_models),
        architecture_config=architecture_config.resolve(),
        unicode_indexer=unicode_indexer.resolve(),
        native_weights=MappingProxyType(native_weights),
        local_root=local_root,
        official_snapshot=official_snapshot,
    )


def resolve_supertonic_style(
    artifacts: SupertonicArtifacts,
    voice: str | Path,
    *,
    cache_dir: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
    verify_integrity: bool = True,
) -> Path:
    """Resolve one released voice ID or explicit style JSON."""
    candidate = Path(voice).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    voice_id = str(voice).strip()
    if not _VOICE_ID.fullmatch(voice_id):
        raise ValueError("`voice` must be a released ID or an existing style JSON path.")
    if artifacts.local_root is not None:
        roots = (
            artifacts.local_root / "voice_styles",
            artifacts.local_root.parent / "voice_styles",
        )
        matches = [(root / f"{voice_id}.json").resolve() for root in roots
                   if (root / f"{voice_id}.json").is_file()]
        if not matches:
            available = sorted({path.stem for root in roots if root.is_dir() for path in root.glob("*.json")})
            suffix = (f" Available voices: {', '.join(available)}." if available else "")
            raise ValueError(f"Unknown Supertonic voice {voice_id!r}.{suffix}")
        path = matches[0]
    else:
        path = resolve_pretrained_file(
            str(artifacts.source),
            f"{voice_id}.json",
            subfolder="voice_styles",
            revision=artifacts.revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        )
    expected = SUPERTONIC_STYLE_INTEGRITY.get(voice_id)
    if (expected is not None and (verify_integrity or artifacts.official_snapshot)):
        _require_integrity(
            path,
            size=expected[0],
            sha256=expected[1],
            label=f"Supertonic voice style {voice_id}",
        )
    return path


__all__ = [
    "SupertonicArtifacts",
    "resolve_supertonic_artifacts",
    "resolve_supertonic_style",
]
