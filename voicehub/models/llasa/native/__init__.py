"""Lazy public imports for VoiceHub's native LLaSA architecture."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from voicehub.models.llasa.native.registration import (
        DEFAULT_LLASSA_ALIASES,
        create_llasa_architecture_spec,
        register_llasa_architecture,
    )

_EXPORTS = {
    "DEFAULT_LLASSA_ALIASES": (
        "voicehub.models.llasa.native.registration",
        "DEFAULT_LLASSA_ALIASES",
    ),
    "create_llasa_architecture_spec": (
        "voicehub.models.llasa.native.registration",
        "create_llasa_architecture_spec",
    ),
    "register_llasa_architecture": (
        "voicehub.models.llasa.native.registration",
        "register_llasa_architecture",
    ),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    from importlib import import_module

    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_EXPORTS))


__all__ = sorted(_EXPORTS)
