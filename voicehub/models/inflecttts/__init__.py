"""Lazy public exports for the VoiceHub-native Inflect TTS family."""

from __future__ import annotations

from importlib import import_module

_EXPORT_MODULES = {
    "InflectTTSConfig": ("voicehub.models.inflecttts.configuration"),
    "InflectTTSForTextToSpeech": "voicehub.models.inflecttts.modeling",
    "InflectTTSModel": "voicehub.models.inflecttts.modeling",
}


def __getattr__(name: str):
    try:
        module_name = _EXPORT_MODULES[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_EXPORT_MODULES))


__all__ = [
    "InflectTTSConfig",
    "InflectTTSForTextToSpeech",
    "InflectTTSModel",
]
