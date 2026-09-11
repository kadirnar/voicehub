"""Lazy public exports for VoiceHub's native StyleTTS 2 family."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "StyleTTS2Config": ("voicehub.models.styletts2.configuration"),
    "StyleTTS2ForTextToSpeech": "voicehub.models.styletts2.modeling",
}


def __getattr__(name: str) -> Any:
    try:
        module_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_EXPORTS))


__all__ = ["StyleTTS2Config", "StyleTTS2ForTextToSpeech"]
