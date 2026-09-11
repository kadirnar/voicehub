"""Lazy exports for the vibevoice model family."""

from importlib import import_module

_EXPORTS = {
    'VibeVoiceConfig': ('voicehub.models.vibevoice.modeling', 'VibeVoiceConfig'),
    'VibeVoiceForTextToSpeech': ('voicehub.models.vibevoice.modeling', 'VibeVoiceForTextToSpeech'),
    'VibeVoiceTTS': ('voicehub.models.vibevoice.modeling', 'VibeVoiceTTS')
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
