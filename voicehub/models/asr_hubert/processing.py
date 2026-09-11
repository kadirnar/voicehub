"""HuBERT's official Wav2Vec2-compatible audio and CTC processor."""

from voicehub.models.asr_wav2vec2.processing import Wav2Vec2Processor

HubertProcessor = Wav2Vec2Processor

__all__ = ["HubertProcessor"]
