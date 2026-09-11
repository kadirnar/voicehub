"""Artifact resolution for the VoiceHub-native FSMN VAD provider."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from voicehub.hub import resolve_pretrained_file
from voicehub.models.vad_funasr.native.checkpoint import NATIVE_FSMN_VAD_FILENAME, convert_funasr_fsmn_checkpoint
from voicehub.models.vad_funasr.native.metadata import (
    FUNASR_CMVN_SHA256,
    FUNASR_HF_REPOSITORY,
    FUNASR_HF_REVISION,
    FUNASR_MODEL_SHA256,
    FUNASR_MODELSCOPE_REPOSITORY,
)
from voicehub.path_utils import is_explicit_local_path

_OFFICIAL_ALIASES = frozenset({
    "fsmn-vad",
    "funasr/fsmn-vad",
    "FunAudioLLM/FSMN-VAD",
    FUNASR_MODELSCOPE_REPOSITORY,
})


@dataclass(frozen=True, slots=True)
class FSMNVADArtifacts:
    """One complete safe artifact and its immutable provenance."""

    checkpoint: Path
    config: Path
    source: str
    revision: str | None
    converted_from_pickle: bool = False

    def __post_init__(self) -> None:
        for name in ("checkpoint", "config"):
            value = Path(getattr(self, name)).expanduser().resolve()
            if not value.is_file():
                raise FileNotFoundError(f"FSMN VAD {name} file was not found: {value}.")
            object.__setattr__(self, name, value)
        try:
            values = json.loads(self.config.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"FSMN VAD config is not valid JSON: {self.config}.") from error
        if not isinstance(values, dict):
            raise TypeError("FSMN VAD config root must be a JSON object.")


def _local_artifacts(
    source: Path,
    *,
    trust_pickle_checkpoint: bool,
) -> FSMNVADArtifacts:
    root = source if source.is_dir() else source.parent
    if source.is_file() and source.suffix == ".safetensors":
        safe_checkpoint = source
    else:
        safe_checkpoint = root / NATIVE_FSMN_VAD_FILENAME
    config = root / "config.json"
    if safe_checkpoint.is_file() and config.is_file():
        return FSMNVADArtifacts(
            checkpoint=safe_checkpoint,
            config=config,
            source=str(source),
            revision=None,
        )

    pickle_checkpoint = (source if source.is_file() and source.name == "model.pt" else root / "model.pt")
    cmvn = root / "am.mvn"
    if pickle_checkpoint.is_file() and cmvn.is_file():
        if trust_pickle_checkpoint is not True:
            raise ValueError(
                "The local FunASR artifact contains pickle-based `model.pt`. "
                "Review it and pass `trust_pickle_checkpoint=True` for "
                "one-time conversion.")
        destination = root / ".voicehub-native" / "fsmn-vad"
        convert_funasr_fsmn_checkpoint(
            pickle_checkpoint,
            cmvn,
            destination,
            trust_pickle_checkpoint=True,
            expected_checkpoint_sha256=FUNASR_MODEL_SHA256,
            expected_cmvn_sha256=FUNASR_CMVN_SHA256,
        )
        return FSMNVADArtifacts(
            checkpoint=destination / NATIVE_FSMN_VAD_FILENAME,
            config=destination / "config.json",
            source=str(source),
            revision=None,
            converted_from_pickle=True,
        )
    raise FileNotFoundError(
        f"No complete native FSMN VAD artifact was found in {root}. "
        "Expected `model.safetensors` plus `config.json`, or reviewed "
        "`model.pt` plus `am.mvn` for explicit conversion.")


def resolve_fsmn_vad_artifacts(
    source: str | Path,
    *,
    revision: str | None,
    cache_dir: str | None,
    token: str | bool | None,
    local_files_only: bool,
    trust_pickle_checkpoint: bool,
) -> FSMNVADArtifacts:
    """Resolve native files or convert the pinned official release once."""
    source_path = Path(source).expanduser()
    if source_path.exists():
        return _local_artifacts(
            source_path.resolve(),
            trust_pickle_checkpoint=trust_pickle_checkpoint,
        )
    if is_explicit_local_path(source):
        raise FileNotFoundError(f"Local FSMN VAD model path was not found: {source_path}.")

    source_name = str(source)
    if source_name not in _OFFICIAL_ALIASES:
        checkpoint = resolve_pretrained_file(
            source_name,
            NATIVE_FSMN_VAD_FILENAME,
            cache_dir=cache_dir,
            revision=revision,
            token=token,
            local_files_only=local_files_only,
        )
        config = resolve_pretrained_file(
            source_name,
            "config.json",
            cache_dir=cache_dir,
            revision=revision,
            token=token,
            local_files_only=local_files_only,
        )
        return FSMNVADArtifacts(
            checkpoint=checkpoint,
            config=config,
            source=source_name,
            revision=revision,
        )

    if revision is not None and revision != FUNASR_HF_REVISION:
        raise ValueError(
            "Official `fsmn-vad` aliases are pinned to immutable revision "
            f"{FUNASR_HF_REVISION}; use a custom native repository for "
            "another revision.")
    if trust_pickle_checkpoint is not True:
        raise ValueError(
            "The official FunASR FSMN VAD release publishes pickle-based "
            "`model.pt`. Review its Apache-2.0 artifact terms and pass "
            "`trust_pickle_checkpoint=True` once; all subsequent runtime "
            "loads use the snapshot-local Safetensors conversion.")
    checkpoint = resolve_pretrained_file(
        FUNASR_HF_REPOSITORY,
        "model.pt",
        cache_dir=cache_dir,
        revision=FUNASR_HF_REVISION,
        token=token,
        local_files_only=local_files_only,
    )
    cmvn = resolve_pretrained_file(
        FUNASR_HF_REPOSITORY,
        "am.mvn",
        cache_dir=cache_dir,
        revision=FUNASR_HF_REVISION,
        token=token,
        local_files_only=local_files_only,
    )
    destination = checkpoint.parent / ".voicehub-native" / "fsmn-vad"
    safe_checkpoint = destination / NATIVE_FSMN_VAD_FILENAME
    config = destination / "config.json"
    if not safe_checkpoint.is_file() or not config.is_file():
        convert_funasr_fsmn_checkpoint(
            checkpoint,
            cmvn,
            destination,
            trust_pickle_checkpoint=True,
            expected_checkpoint_sha256=FUNASR_MODEL_SHA256,
            expected_cmvn_sha256=FUNASR_CMVN_SHA256,
        )
    return FSMNVADArtifacts(
        checkpoint=safe_checkpoint,
        config=config,
        source=source_name,
        revision=FUNASR_HF_REVISION,
        converted_from_pickle=True,
    )


__all__ = [
    "FSMNVADArtifacts",
    "resolve_fsmn_vad_artifacts",
]
