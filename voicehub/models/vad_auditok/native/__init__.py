"""VoiceHub-native short-term-energy activity detection."""

from __future__ import annotations

import importlib
from typing import Any

_PACKAGE = "voicehub.models.vad_auditok.native."
_EXPORTS = {
    "AUDITOK_REFERENCE_REVISION": _PACKAGE + "registration",
    "DEFAULT_ENERGY_VAD_ALIASES": _PACKAGE + "registration",
    "EnergyDetection": _PACKAGE + "modeling",
    "EnergyRegion": _PACKAGE + "modeling",
    "EnergyVoiceActivityDetector": _PACKAGE + "modeling",
    "create_energy_vad_architecture_spec": _PACKAGE + "registration",
    "estimate_energy_threshold": _PACKAGE + "modeling",
    "register_energy_vad_architecture": _PACKAGE + "registration",
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
