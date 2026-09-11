"""Configuration for VoiceHub's native HuBERT CTC provider."""

from __future__ import annotations

from voicehub.models.asr_wav2vec2.configuration import Wav2Vec2ASRConfig


class HubertASRConfig(Wav2Vec2ASRConfig):
    """Configure native HuBERT loading, decoding, and fine-tuning.

    HuBERT uses the same raw-waveform processor contract as Wav2Vec2, so
    public loader controls remain identical and unsupported external
    runtime options are rejected by the shared validated configuration.
    """

    model_type = "asr_hubert"
    runtime_name = "HuBERT"


__all__ = ["HubertASRConfig"]
