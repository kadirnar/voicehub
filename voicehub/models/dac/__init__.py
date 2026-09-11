"""Lazy exports for the dac model family."""

from importlib import import_module

_EXPORTS = {
    'DacCodecConfig': ('voicehub.models.dac.configuration', 'DacCodecConfig'),
    'DacForAudioCodec': ('voicehub.models.dac.modeling', 'DacForAudioCodec')
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
