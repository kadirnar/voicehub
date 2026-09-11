"""Lazy exports for the chatterbox model family."""

from importlib import import_module

_EXPORTS = {
    'ChatterboxConfig': ('voicehub.models.chatterbox.modeling', 'ChatterboxConfig'),
    'ChatterboxForTextToSpeech': ('voicehub.models.chatterbox.modeling', 'ChatterboxForTextToSpeech'),
    'ChatterboxTTS': ('voicehub.models.chatterbox.tts', 'ChatterboxTTS'),
    'ChatterboxVC': ('voicehub.models.chatterbox.vc', 'ChatterboxVC')
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
