"""VoiceHub-owned TEN VAD graph with lazy public exports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_PACKAGE = "voicehub.models.vad_ten.native."
_EXPORTS = {
    "DEFAULT_TEN_VAD_ALIASES": _PACKAGE + "registration",
    "NATIVE_TEN_VAD_FILENAME": _PACKAGE + "checkpoint",
    "NATIVE_TEN_VAD_FORMAT": _PACKAGE + "checkpoint",
    "TENVADConfig": _PACKAGE + "configuration",
    "TENVADFrameOutput": _PACKAGE + "modeling",
    "TENVADFrontend": _PACKAGE + "frontend",
    "TENVADFrontendOutput": _PACKAGE + "frontend",
    "TENVADFrontendState": _PACKAGE + "frontend",
    "TENVADModel": _PACKAGE + "modeling",
    "TENVADOutput": _PACKAGE + "modeling",
    "TENVADRecurrentState": _PACKAGE + "modeling",
    "TENVADSafeTensorsCheckpointAdapter": _PACKAGE + "checkpoint",
    "TENVADState": _PACKAGE + "modeling",
    "convert_ten_vad_onnx_checkpoint": _PACKAGE + "checkpoint",
    "create_ten_vad_architecture_spec": _PACKAGE + "registration",
    "native_ten_vad_tensor_shapes": _PACKAGE + "checkpoint",
    "register_ten_vad_architecture": _PACKAGE + "registration",
    "sherpa_ten_vad_mel_filterbank": _PACKAGE + "frontend",
    "sherpa_ten_vad_window": _PACKAGE + "frontend",
    "ten_vad_binary_cross_entropy": _PACKAGE + "objective",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_EXPORTS))
