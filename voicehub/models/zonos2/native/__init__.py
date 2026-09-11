"""VoiceHub-owned ZONOS2 architecture.

Imports remain lazy so registry discovery does not import PyTorch.
"""

from importlib import import_module

_PACKAGE = "voicehub.models.zonos2.native."
_EXPORTS = {
    "Zonos2ArchitectureConfig": _PACKAGE + "configuration",
    "Zonos2ForCausalLM": _PACKAGE + "modeling",
    "Zonos2ForCausalLMOutput": _PACKAGE + "modeling",
    "Zonos2TrainingOutput": _PACKAGE + "objective",
    "build_zonos2_prompt": _PACKAGE + "prompting",
    "prepare_zonos2_training_batch": _PACKAGE + "prompting",
    "load_zonos2_checkpoint": _PACKAGE + "checkpoint",
    "export_zonos2_checkpoint": _PACKAGE + "checkpoint",
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
