"""Lazy exports for the vad_pyannote model family."""

from importlib import import_module

_EXPORTS = {
    'PyannoteVADConfig': ('voicehub.models.vad_pyannote.configuration', 'PyannoteVADConfig'),
    'PyannoteVADForVoiceActivityDetection':
    ('voicehub.models.vad_pyannote.modeling', 'PyannoteVADForVoiceActivityDetection')
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
