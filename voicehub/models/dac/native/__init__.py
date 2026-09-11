"""VoiceHub-native Descript DAC architecture with lazy public exports."""

from __future__ import annotations

import importlib
from typing import Any

_PACKAGE = "voicehub.models.dac.native."
_EXPORTS = {
    "DEFAULT_DAC_ALIASES": _PACKAGE + "registration",
    "DESCRIPT_DAC_44KHZ_HEADER_FINGERPRINT": _PACKAGE + "checkpoint",
    "DESCRIPT_DAC_44KHZ_REVISION": _PACKAGE + "checkpoint",
    "DacConfig": _PACKAGE + "configuration",
    "DacDecoderOutput": _PACKAGE + "modeling",
    "DacEncoderOutput": _PACKAGE + "modeling",
    "DacModel": _PACKAGE + "modeling",
    "HFDacCheckpointAdapter": _PACKAGE + "checkpoint",
    "HuggingFaceDacCheckpointAdapter": _PACKAGE + "checkpoint",
    "TRANSFORMERS_DAC_REVISION": _PACKAGE + "checkpoint",
    "WeightNormalizedTensor": _PACKAGE + "checkpoint",
    "create_dac_architecture_spec": _PACKAGE + "registration",
    "dac_tensor_inventory_fingerprint": _PACKAGE + "checkpoint",
    "huggingface_dac_tensor_names": _PACKAGE + "checkpoint",
    "huggingface_dac_tensor_shapes": _PACKAGE + "checkpoint",
    "native_dac_tensor_name": _PACKAGE + "checkpoint",
    "register_dac_architecture": _PACKAGE + "registration",
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
