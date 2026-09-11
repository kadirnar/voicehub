"""VoiceHub-native NeMo QuartzNet CTC provider."""

from importlib import import_module

_PACKAGE = "voicehub.models.asr_nemo."
_EXPORTS = {
    "NeMoASRConfig": _PACKAGE + "configuration",
    "NeMoASRForSpeechRecognition": _PACKAGE + "modeling",
    "NativeNeMoCTCTrainingAdapter": _PACKAGE + "training_asr_nemo",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
