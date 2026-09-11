"""Lazy public imports for GPT-SoVITS."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "GPTSoVITSConfig": (
        "voicehub.models.gptsovits.configuration",
        "GPTSoVITSConfig",
    ),
    "GPTSoVITSForTextToSpeech": (
        "voicehub.models.gptsovits.modeling",
        "GPTSoVITSForTextToSpeech",
    ),
    "GPTSoVITSTTS": (
        "voicehub.models.gptsovits.modeling",
        "GPTSoVITSTTS",
    ),
}


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


__all__ = sorted(_EXPORTS)
