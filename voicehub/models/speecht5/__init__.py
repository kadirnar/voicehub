"""Lazy exports for the speecht5 model family."""

from importlib import import_module

_EXPORTS = {
    'SpeechT5Config': ('voicehub.models.speecht5.configuration', 'SpeechT5Config'),
    'SpeechT5ForTextToSpeech': ('voicehub.models.speecht5.modeling', 'SpeechT5ForTextToSpeech')
}
__all__ = sorted(_EXPORTS)


def __getattr__(name):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted((*globals(), *_EXPORTS))
