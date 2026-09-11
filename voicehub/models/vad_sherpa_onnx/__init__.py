"""Lazy exports for the vad_sherpa_onnx model family."""

from importlib import import_module

_EXPORTS = {
    'SherpaONNXVADConfig': ('voicehub.models.vad_sherpa_onnx.configuration', 'SherpaONNXVADConfig'),
    'SherpaNativeVADSession': ('voicehub.models.vad_sherpa_onnx.modeling', 'SherpaNativeVADSession'),
    'SherpaONNXVADForVoiceActivityDetection':
    ('voicehub.models.vad_sherpa_onnx.modeling', 'SherpaONNXVADForVoiceActivityDetection'),
    'SherpaONNXVADSession': ('voicehub.models.vad_sherpa_onnx.modeling', 'SherpaONNXVADSession')
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
