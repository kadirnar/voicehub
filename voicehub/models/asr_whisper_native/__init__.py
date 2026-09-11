"""Public native Whisper ASR provider."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from voicehub.models.asr_whisper_native.configuration import WhisperASRConfig
    from voicehub.models.asr_whisper_native.modeling import WhisperForSpeechRecognition
    from voicehub.models.asr_whisper_native.training_asr_whisper_native import NativeWhisperTrainingAdapter

_PUBLIC_COMPONENTS = {
    "NativeWhisperTrainingAdapter": ("voicehub.models.asr_whisper_native.training_asr_whisper_native"),
    "WhisperASRConfig": ("voicehub.models.asr_whisper_native.configuration"),
    "WhisperForSpeechRecognition": ("voicehub.models.asr_whisper_native.modeling"),
}


def __getattr__(name: str) -> Any:
    """Resolve public components without loading model or trainer code
    early."""
    try:
        module_name = _PUBLIC_COMPONENTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


__all__ = [
    "NativeWhisperTrainingAdapter",
    "WhisperASRConfig",
    "WhisperForSpeechRecognition",
]
