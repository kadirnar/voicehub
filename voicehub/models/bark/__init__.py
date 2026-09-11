"""Lazy exports for the bark model family."""

from importlib import import_module

_EXPORTS = {
    'BarkConfig': ('voicehub.models.bark.configuration', 'BarkConfig'),
    'BarkForTextToSpeech': ('voicehub.models.bark.modeling', 'BarkForTextToSpeech')
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
