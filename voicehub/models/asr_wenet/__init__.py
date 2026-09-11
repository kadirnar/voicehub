"""VoiceHub-native WeNet U2++ provider."""

from importlib import import_module

_PACKAGE = "voicehub.models.asr_wenet."
_EXPORTS = {
    "NativeWeNetU2PPTrainingAdapter": _PACKAGE + "training_asr_wenet",
    "WeNetASRConfig": _PACKAGE + "configuration",
    "WeNetASRForSpeechRecognition": _PACKAGE + "modeling",
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
