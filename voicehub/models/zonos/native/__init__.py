"""VoiceHub-owned Zonos v0.1 dense Transformer architecture."""

from __future__ import annotations

from importlib import import_module

_PACKAGE = "voicehub.models.zonos.native."
_EXPORTS = {
    "NativeZonosRuntime": _PACKAGE + "runtime",
    "PrecomputedPhonemeFrontend": _PACKAGE + "frontend",
    "ZonosArchitectureConfig": _PACKAGE + "configuration",
    "ZonosCheckpointReport": _PACKAGE + "checkpoint",
    "ZonosDACCodec": _PACKAGE + "codec",
    "ZonosForCausalLM": _PACKAGE + "modeling",
    "ZonosForCausalLMOutput": _PACKAGE + "modeling",
    "ZonosGeneration": _PACKAGE + "runtime",
    "ZonosPhonemeFrontend": _PACKAGE + "frontend",
    "ZonosSamplingOptions": _PACKAGE + "sampling",
    "export_zonos_checkpoint": _PACKAGE + "checkpoint",
    "generate_zonos_codes": _PACKAGE + "sampling",
    "load_zonos_checkpoint": _PACKAGE + "checkpoint",
    "make_condition_dict": _PACKAGE + "frontend",
    "resolve_zonos_artifacts": _PACKAGE + "artifacts",
    "save_zonos_pretrained": _PACKAGE + "checkpoint",
    "tokenize_phonemes": _PACKAGE + "frontend",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
