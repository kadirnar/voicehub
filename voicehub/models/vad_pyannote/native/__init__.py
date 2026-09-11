"""VoiceHub-native PyanNet architecture with lazy public components."""

from __future__ import annotations

import importlib
from typing import Any

_PACKAGE = "voicehub.models.vad_pyannote.native."
_EXPORTS = {
    "ASTEROID_FILTERBANKS_SOURCE_REVISION": _PACKAGE + "metadata",
    "BROUHAHA_REPOSITORY_CHECKPOINT_SHA256": _PACKAGE + "metadata",
    "BROUHAHA_SOURCE_REVISION": _PACKAGE + "metadata",
    "DEFAULT_PYANNET_ALIASES": _PACKAGE + "registration",
    "PYANNOTE_AUDIO_3_SOURCE_REVISION": _PACKAGE + "metadata",
    "PYANNOTE_BROUHAHA_REVISION": _PACKAGE + "metadata",
    "PYANNOTE_SEGMENTATION_3_REVISION": _PACKAGE + "metadata",
    "PYANNOTE_SEGMENTATION_3_SHA256": _PACKAGE + "metadata",
    "PYANNOTE_SEGMENTATION_REVISION": _PACKAGE + "metadata",
    "PYANNOTE_SEGMENTATION_SHA256": _PACKAGE + "metadata",
    "PYANNOTE_VAD_PIPELINE_REVISION": _PACKAGE + "metadata",
    "Powerset": _PACKAGE + "powerset",
    "PyanNet": _PACKAGE + "modeling",
    "PyanNetConfig": _PACKAGE + "configuration",
    "PyanNetFrameInference": _PACKAGE + "inference",
    "PyanNetFrameOutput": _PACKAGE + "inference",
    "PyanNetOutput": _PACKAGE + "modeling",
    "PyanNetSafeTensorsCheckpointAdapter": _PACKAGE + "checkpoint",
    "convert_pyannote_lightning_checkpoint": _PACKAGE + "checkpoint",
    "create_pyannet_architecture_spec": _PACKAGE + "registration",
    "pyannet_loss": _PACKAGE + "objective",
    "register_pyannet_architecture": _PACKAGE + "registration",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve public components only when requested."""
    try:
        module_name = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return stable results for interactive discovery."""
    return sorted((*globals(), *_EXPORTS))
