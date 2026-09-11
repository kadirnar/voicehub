"""Lazy OuteTTS configuration and model exports."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "OuteTTSConfig": (
        "voicehub.models.outetts.configuration",
        "OuteTTSConfig",
    ),
    "OuteTTSForTextToSpeech": (
        "voicehub.models.outetts.inference",
        "OuteTTSForTextToSpeech",
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


def __dir__() -> list[str]:
    return sorted((*globals(), *_EXPORTS))


__all__ = sorted(_EXPORTS)
