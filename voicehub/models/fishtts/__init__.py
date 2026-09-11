"""Lazy exports for the fishtts model family."""

from importlib import import_module

_EXPORTS = {
    'FishTTSConfig': ('voicehub.models.fishtts.configuration', 'FishTTSConfig'),
    'FishTTS': ('voicehub.models.fishtts.modeling', 'FishTTS'),
    'FishTTSForTextToSpeech': ('voicehub.models.fishtts.modeling', 'FishTTSForTextToSpeech')
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
