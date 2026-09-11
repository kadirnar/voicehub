"""Compatibility exports for ASR presets now implemented by VoiceHub.

This historical namespace is intentionally lazy.  Importing it does not
load PyTorch or any model graph, and every public symbol resolves to the
dedicated VoiceHub-native provider that owns the corresponding
architecture.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "CohereASRConfig": (
        "voicehub.models.asr_cohere.configuration",
        "CohereASRConfig",
    ),
    "CohereForSpeechRecognition": (
        "voicehub.models.asr_cohere.modeling",
        "CohereForSpeechRecognition",
    ),
    "HubertASRConfig": (
        "voicehub.models.asr_hubert.configuration",
        "HubertASRConfig",
    ),
    "HubertForSpeechRecognition": (
        "voicehub.models.asr_hubert.modeling",
        "HubertForSpeechRecognition",
    ),
    "MedASRConfig": (
        "voicehub.models.asr_medasr.configuration",
        "MedASRConfig",
    ),
    "MedASRForSpeechRecognition": (
        "voicehub.models.asr_medasr.modeling",
        "MedASRForSpeechRecognition",
    ),
    "MoonshineASRConfig": (
        "voicehub.models.asr_moonshine.configuration",
        "MoonshineASRConfig",
    ),
    "MoonshineForSpeechRecognition": (
        "voicehub.models.asr_moonshine.modeling",
        "MoonshineForSpeechRecognition",
    ),
    "NemotronASRConfig": (
        "voicehub.models.asr_nemotron.configuration",
        "NemotronASRConfig",
    ),
    "NemotronForSpeechRecognition": (
        "voicehub.models.asr_nemotron.modeling",
        "NemotronForSpeechRecognition",
    ),
    "ParakeetTDTASRConfig": (
        "voicehub.models.asr_parakeet_tdt.configuration",
        "ParakeetTDTASRConfig",
    ),
    "ParakeetTDTForSpeechRecognition": (
        "voicehub.models.asr_parakeet_tdt.modeling",
        "ParakeetTDTForSpeechRecognition",
    ),
    "SeamlessM4Tv2ASRConfig": (
        "voicehub.models.asr_seamless_m4t_v2.configuration",
        "SeamlessM4Tv2ASRConfig",
    ),
    "SeamlessM4Tv2ForSpeechRecognition": (
        "voicehub.models.asr_seamless_m4t_v2.modeling",
        "SeamlessM4Tv2ForSpeechRecognition",
    ),
    "Wav2Vec2ASRConfig": (
        "voicehub.models.asr_wav2vec2.configuration",
        "Wav2Vec2ASRConfig",
    ),
    "Wav2Vec2ForSpeechRecognition": (
        "voicehub.models.asr_wav2vec2.modeling",
        "Wav2Vec2ForSpeechRecognition",
    ),
    "WavLMASRConfig": (
        "voicehub.models.asr_wavlm.configuration",
        "WavLMASRConfig",
    ),
    "WavLMForSpeechRecognition": (
        "voicehub.models.asr_wavlm.modeling",
        "WavLMForSpeechRecognition",
    ),
    "WhisperASRConfig": (
        "voicehub.models.asr_whisper_native.configuration",
        "WhisperASRConfig",
    ),
    "WhisperForSpeechRecognition": (
        "voicehub.models.asr_whisper_native.modeling",
        "WhisperForSpeechRecognition",
    ),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve a legacy symbol to its canonical native implementation."""
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_EXPORTS))
