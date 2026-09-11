"""Lazy exports for the vad_nemo model family."""

from importlib import import_module

_EXPORTS = {
    'NeMoVADConfig': ('voicehub.models.vad_nemo.configuration', 'NeMoVADConfig'),
    'NeMoVADForVoiceActivityDetection':
    ('voicehub.models.vad_nemo.modeling', 'NeMoVADForVoiceActivityDetection'),
    'MarbleNetVADTrainingDataset':
    ('voicehub.models.vad_nemo.training_vad_nemo', 'MarbleNetVADTrainingDataset'),
    'NativeMarbleNetVADTrainingAdapter':
    ('voicehub.models.vad_nemo.training_vad_nemo', 'NativeMarbleNetVADTrainingAdapter')
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
