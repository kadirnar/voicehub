"""Configuration for VoiceHub's native WavLM CTC provider."""

from __future__ import annotations

from voicehub.models.asr_wav2vec2.configuration import Wav2Vec2ASRConfig


class WavLMASRConfig(Wav2Vec2ASRConfig):
    """Configure native WavLM loading, decoding, and fine-tuning."""

    model_type = "asr_wavlm"
    runtime_name = "WavLM"


__all__ = ["WavLMASRConfig"]
