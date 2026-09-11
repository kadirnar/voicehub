"""Native TEN VAD artifact resolution and one-time ONNX conversion."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from voicehub.hub import resolve_pretrained_file
from voicehub.models.vad_ten.native.checkpoint import NATIVE_TEN_VAD_FILENAME, convert_ten_vad_onnx_checkpoint
from voicehub.path_utils import is_explicit_local_path


@dataclass(frozen=True, slots=True)
class TENVADArtifacts:
    checkpoint: Path
    config: Path
    source: str
    revision: str | None
    converted_from_onnx: bool = False
    source_onnx: Path | None = None

    def __post_init__(self) -> None:
        for name in ("checkpoint", "config"):
            value = Path(getattr(self, name)).expanduser().resolve()
            if not value.is_file():
                raise FileNotFoundError(f"TEN VAD {name} file was not found: {value}.")
            object.__setattr__(self, name, value)
        if self.source_onnx is not None:
            source = Path(self.source_onnx).expanduser().resolve()
            if not source.is_file():
                raise FileNotFoundError(f"TEN VAD source ONNX was not found: {source}.")
            object.__setattr__(self, "source_onnx", source)
        try:
            values = json.loads(self.config.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"TEN VAD config is not valid JSON: {self.config}.") from error
        if not isinstance(values, dict):
            raise TypeError("TEN VAD config root must be a JSON object.")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative(value: str) -> PurePosixPath:
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts or path.name in {"", ".", ".."}:
        raise ValueError("TEN VAD model filename must be a safe relative path.")
    return path


def _native_pair(root: Path, direct: Path | None = None) -> tuple[Path, Path] | None:
    candidates = []
    if direct is not None and direct.suffix == ".safetensors":
        candidates.append((direct, direct.parent / "config.json"))
    candidates.extend((
        (root / NATIVE_TEN_VAD_FILENAME, root / "config.json"),
        (
            root / "native_export" / NATIVE_TEN_VAD_FILENAME,
            root / "native_export" / "config.json",
        ),
        (
            root / ".voicehub-native" / "ten-vad" / NATIVE_TEN_VAD_FILENAME,
            root / ".voicehub-native" / "ten-vad" / "config.json",
        ),
    ))
    return next(
        ((checkpoint, config)
         for checkpoint, config in candidates if checkpoint.is_file() and config.is_file()),
        None,
    )


def _converted_pair(
    source: Path,
    *,
    trust_onnx_checkpoint: bool,
    window_size: int,
) -> tuple[Path, Path]:
    digest = _sha256(source)
    destination = (source.parent / ".voicehub-native" / f"ten-vad-{digest[:12]}-w{window_size}")
    checkpoint = destination / NATIVE_TEN_VAD_FILENAME
    config = destination / "config.json"
    if checkpoint.is_file() and config.is_file():
        try:
            values = json.loads(config.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"Cached TEN conversion config is invalid: {config}.") from error
        if values.get("source_onnx_sha256") != digest:
            raise ValueError("Cached TEN conversion does not match its source ONNX digest.")
        if values.get("window_size") != window_size:
            raise ValueError("Cached TEN conversion uses a different frontend window size.")
        checkpoint_digest = values.get("checkpoint_sha256")
        if checkpoint_digest is None or _sha256(checkpoint) != checkpoint_digest:
            raise ValueError(
                "Cached TEN conversion checkpoint does not match its recorded "
                "SHA-256 digest.")
        return checkpoint, config
    convert_ten_vad_onnx_checkpoint(
        source,
        destination,
        trust_onnx_checkpoint=trust_onnx_checkpoint,
        expected_source_sha256=digest,
        window_size=window_size,
    )
    return checkpoint, config


def resolve_ten_vad_artifacts(
    pretrained_model_name_or_path: str | Path,
    *,
    model_filename: str = "ten-vad.onnx",
    subfolder: str = "",
    revision: str | None = None,
    cache_dir: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
    trust_onnx_checkpoint: bool = False,
    window_size: int = 256,
) -> TENVADArtifacts:
    """Resolve safe native files or explicitly convert a reviewed ONNX file."""
    source_value = str(pretrained_model_name_or_path)
    source = Path(pretrained_model_name_or_path).expanduser()
    relative = _safe_relative(model_filename)
    nested = Path(subfolder) / Path(*relative.parts)
    source_onnx: Path | None = None
    converted = False

    if source.is_file():
        if subfolder:
            raise ValueError("`subfolder` cannot be used with a direct TEN checkpoint.")
        if revision is not None:
            raise ValueError("`revision` cannot be used with a direct TEN checkpoint.")
        root = source.parent
        pair = _native_pair(root, source)
        if pair is None:
            if source.suffix.lower() != ".onnx":
                raise ValueError(
                    "A direct TEN checkpoint must be Safetensors with config.json "
                    "or a reviewed ONNX source.")
            source_onnx = source.resolve()
            pair = _converted_pair(
                source_onnx,
                trust_onnx_checkpoint=trust_onnx_checkpoint,
                window_size=window_size,
            )
            converted = True
        resolved_revision = None
    elif source.is_dir():
        root = source.resolve()
        direct = root / nested
        pair = _native_pair(root, direct)
        if pair is None:
            candidate = direct
            if not candidate.is_file():
                raise FileNotFoundError(f"No native TEN artifact or source {nested} was found in {root}.")
            if candidate.suffix.lower() != ".onnx":
                raise FileNotFoundError("TEN Safetensors require a sibling config.json.")
            source_onnx = candidate
            pair = _converted_pair(
                source_onnx,
                trust_onnx_checkpoint=trust_onnx_checkpoint,
                window_size=window_size,
            )
            converted = True
        resolved_revision = None
    else:
        if is_explicit_local_path(pretrained_model_name_or_path):
            raise FileNotFoundError(f"Local TEN model path was not found: {source}.")
        suffix = relative.suffix.lower()
        filename = relative.name
        parents = [part for part in relative.parent.parts if part != "."]
        resolved_subfolder = "/".join(part for part in (subfolder, *parents) if part)
        resolved = resolve_pretrained_file(
            source_value,
            filename,
            subfolder=resolved_subfolder,
            cache_dir=cache_dir,
            revision=revision,
            token=token,
            local_files_only=local_files_only,
        )
        if suffix == ".onnx":
            source_onnx = resolved
            pair = _converted_pair(
                resolved,
                trust_onnx_checkpoint=trust_onnx_checkpoint,
                window_size=window_size,
            )
            converted = True
        elif suffix == ".safetensors":
            config = resolve_pretrained_file(
                source_value,
                "config.json",
                subfolder=resolved_subfolder,
                cache_dir=cache_dir,
                revision=revision,
                token=token,
                local_files_only=local_files_only,
            )
            pair = (resolved, config)
        else:
            raise ValueError("Remote TEN artifacts must be ONNX or Safetensors.")
        resolved_revision = revision

    return TENVADArtifacts(
        checkpoint=pair[0],
        config=pair[1],
        source=source_value,
        revision=resolved_revision,
        converted_from_onnx=converted,
        source_onnx=source_onnx,
    )


__all__ = ["TENVADArtifacts", "resolve_ten_vad_artifacts"]
