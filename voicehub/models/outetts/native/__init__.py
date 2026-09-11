"""Lazy public surface for the VoiceHub-native OuteTTS architecture."""

from __future__ import annotations

from importlib import import_module

_PACKAGE = "voicehub.models.outetts.native."
_EXPORTS = {
    "OuteTTSArtifacts": _PACKAGE + "artifacts",
    "OuteTTSForCausalLM": _PACKAGE + "modeling",
    "OuteTTSPromptProcessor": _PACKAGE + "prompting",
    "OuteTTSRuntime": _PACKAGE + "runtime",
    "OuteTTSTokenizer": _PACKAGE + "tokenization",
    "SpeakerProfile": _PACKAGE + "prompting",
    "create_outetts_architecture_spec": _PACKAGE + "registration",
    "register_outetts_architecture": _PACKAGE + "registration",
    "resolve_outetts_artifacts": _PACKAGE + "artifacts",
}


def __getattr__(name: str):
    try:
        module_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_EXPORTS))


__all__ = sorted(_EXPORTS)
