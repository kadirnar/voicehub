"""Lazy public exports for VoiceHub's native XTTS v2 architecture."""

from __future__ import annotations

import importlib
from typing import Any

_PACKAGE = "voicehub.models.xtts.native."
_EXPORTS = {
    "NATIVE_XTTS2_DVAE_FILENAME": _PACKAGE + "dvae_checkpoint",
    "NATIVE_XTTS2_DVAE_FORMAT": _PACKAGE + "dvae_checkpoint",
    "NATIVE_XTTS2_DVAE_MEL_STATS_FILENAME": _PACKAGE + "dvae_checkpoint",
    "NATIVE_XTTS2_DVAE_MEL_STATS_FORMAT": _PACKAGE + "dvae_checkpoint",
    "XTTS2AudioConfig": _PACKAGE + "configuration",
    "XTTS2CheckpointInventory": _PACKAGE + "checkpoint",
    "XTTS2Config": _PACKAGE + "configuration",
    "XTTS2DVAE": _PACKAGE + "dvae",
    "XTTS2DVAEAutoencoderOutput": _PACKAGE + "dvae",
    "XTTS2DVAECheckpointInventory": _PACKAGE + "dvae_checkpoint",
    "XTTS2DVAEConfig": _PACKAGE + "dvae",
    "XTTS2DVAEEncoding": _PACKAGE + "dvae",
    "XTTS2DVAEMelProcessor": _PACKAGE + "dvae",
    "XTTS2GPT": _PACKAGE + "gpt",
    "XTTS2Model": _PACKAGE + "modeling",
    "XTTS2ModelArgs": _PACKAGE + "configuration",
    "XTTS2Tokenizer": _PACKAGE + "tokenizer",
    "XTTS2TrainingAudioEncoder": _PACKAGE + "dvae",
    "convert_trusted_legacy_xtts2_checkpoint": _PACKAGE + "checkpoint",
    "convert_trusted_legacy_xtts2_dvae_checkpoint": _PACKAGE + "dvae_checkpoint",
    "convert_trusted_legacy_xtts2_mel_stats": _PACKAGE + "dvae_checkpoint",
    "inspect_xtts2_checkpoint": _PACKAGE + "checkpoint",
    "inspect_xtts2_dvae_checkpoint": _PACKAGE + "dvae_checkpoint",
    "load_xtts2_checkpoint": _PACKAGE + "checkpoint",
    "load_xtts2_dvae_checkpoint": _PACKAGE + "dvae_checkpoint",
    "load_xtts2_dvae_mel_stats": _PACKAGE + "dvae_checkpoint",
    "load_xtts2_training_audio_encoder": _PACKAGE + "dvae_checkpoint",
    "save_xtts2_checkpoint": _PACKAGE + "checkpoint",
    "save_xtts2_dvae_checkpoint": _PACKAGE + "dvae_checkpoint",
    "save_xtts2_dvae_mel_stats": _PACKAGE + "dvae_checkpoint",
    "save_xtts2_training_audio_encoder": _PACKAGE + "dvae_checkpoint",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_EXPORTS))
