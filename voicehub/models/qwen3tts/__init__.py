"""Lazy exports for the qwen3tts model family."""

from importlib import import_module

_EXPORTS = {
    'Qwen3TTSConfig': ('voicehub.models.qwen3tts.modeling', 'Qwen3TTSConfig'),
    'Qwen3TTSForTextToSpeech': ('voicehub.models.qwen3tts.modeling', 'Qwen3TTSForTextToSpeech')
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
