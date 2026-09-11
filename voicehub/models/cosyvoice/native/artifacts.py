"""Coherent local/Hub artifact resolution for native CosyVoice."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from voicehub.hub import resolve_pretrained_file
from voicehub.models.cosyvoice.native.metadata import COSYVOICE3_MODEL_ID, COSYVOICE3_MODEL_REVISION
from voicehub.path_utils import is_explicit_local_path

_REQUIRED = (
    "cosyvoice_config.json",
    "llm.safetensors",
    "flow.safetensors",
    "hift.safetensors",
    "vocab.json",
    "merges.txt",
    "tokenizer_config.json",
)
_OPTIONAL_SPEECH_TOKENIZER = "speech_tokenizer.safetensors"
_OPTIONAL_SPEECH_TOKENIZER_CONFIG = "speech_tokenizer_config.json"


@dataclass(frozen=True, slots=True)
class CosyVoiceArtifacts:
    source: str
    revision: str | None
    root: Path | None
    config: Path
    llm: Path
    flow: Path
    hift: Path
    speech_tokenizer: Path | None
    speech_tokenizer_config: Path | None
    vocab: Path
    merges: Path
    tokenizer_config: Path


def _local(root: Path) -> CosyVoiceArtifacts:
    missing = [filename for filename in _REQUIRED if not (root / filename).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Native CosyVoice artifact {root} is incomplete; missing " + ", ".join(missing))
    return CosyVoiceArtifacts(
        source=str(root),
        revision=None,
        root=root,
        config=root / "cosyvoice_config.json",
        llm=root / "llm.safetensors",
        flow=root / "flow.safetensors",
        hift=root / "hift.safetensors",
        speech_tokenizer=(
            root / _OPTIONAL_SPEECH_TOKENIZER if (root / _OPTIONAL_SPEECH_TOKENIZER).is_file() else None),
        speech_tokenizer_config=(
            root / _OPTIONAL_SPEECH_TOKENIZER_CONFIG if
            (root / _OPTIONAL_SPEECH_TOKENIZER_CONFIG).is_file() else None),
        vocab=root / "vocab.json",
        merges=root / "merges.txt",
        tokenizer_config=root / "tokenizer_config.json",
    )


def resolve_cosyvoice_artifacts(
    source: str | Path,
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
) -> CosyVoiceArtifacts:
    """Resolve a pickle-free artifact from one immutable source."""
    if not isinstance(source, (str, Path)) or not str(source).strip():
        raise ValueError("CosyVoice source must be a non-empty path or Hub ID.")
    local = Path(source).expanduser()
    if local.is_dir():
        return _local(local.resolve())
    if local.exists() or is_explicit_local_path(source):
        raise FileNotFoundError(f"Native CosyVoice path was not found: {local}.")
    repo_id = str(source)
    resolved_revision = (
        revision or (COSYVOICE3_MODEL_REVISION if repo_id == COSYVOICE3_MODEL_ID else "main"))
    try:
        files = {
            filename:
            resolve_pretrained_file(
                repo_id,
                filename,
                revision=resolved_revision,
                cache_dir=cache_dir,
                token=token,
                local_files_only=local_files_only,
            )
            for filename in _REQUIRED
        }
    except FileNotFoundError as error:
        if repo_id == COSYVOICE3_MODEL_ID:
            raise FileNotFoundError(
                "The official CosyVoice3 snapshot publishes audited legacy "
                "llm.pt/flow.pt/hift.pt files, not native Safetensors. Run "
                "`convert_audited_cosyvoice_legacy_checkpoint` explicitly "
                "once, then load the resulting local artifact.") from error
        raise
    return CosyVoiceArtifacts(
        source=repo_id,
        revision=resolved_revision,
        root=None,
        config=files["cosyvoice_config.json"],
        llm=files["llm.safetensors"],
        flow=files["flow.safetensors"],
        hift=files["hift.safetensors"],
        speech_tokenizer=None,
        speech_tokenizer_config=None,
        vocab=files["vocab.json"],
        merges=files["merges.txt"],
        tokenizer_config=files["tokenizer_config.json"],
    )


__all__ = [
    "CosyVoiceArtifacts",
    "resolve_cosyvoice_artifacts",
]
