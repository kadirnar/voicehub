"""VoiceHub-owned Wav2Vec2 CTC architecture with lazy public exports."""

from __future__ import annotations

import importlib
from typing import Any

_PACKAGE = "voicehub.models.asr_wav2vec2.native."
_EXPORTS = {
    "CTCCharacterOffset": _PACKAGE + "tokenization",
    "CTCWordOffset": _PACKAGE + "tokenization",
    "DEFAULT_WAV2VEC2_ALIASES": _PACKAGE + "registration",
    "FACEBOOK_WAV2VEC2_BASE_960H_HEADER_FINGERPRINT": _PACKAGE + "checkpoint",
    "FACEBOOK_WAV2VEC2_BASE_960H_REVISION": _PACKAGE + "checkpoint",
    "Float32LayerNorm": _PACKAGE + "modeling",
    "HFWav2Vec2CheckpointAdapter": _PACKAGE + "checkpoint",
    "HuggingFaceWav2Vec2ClassificationCheckpointAdapter": _PACKAGE + "checkpoint",
    "HuggingFaceWav2Vec2CheckpointAdapter": _PACKAGE + "checkpoint",
    "TRANSFORMERS_WAV2VEC2_REVISION": _PACKAGE + "registration",
    "Wav2Vec2Artifacts": _PACKAGE + "artifacts",
    "Wav2Vec2ClassificationArtifacts": _PACKAGE + "artifacts",
    "Wav2Vec2Attention": _PACKAGE + "modeling",
    "Wav2Vec2CTCDecodeOutput": _PACKAGE + "tokenization",
    "Wav2Vec2CTCOutput": _PACKAGE + "modeling",
    "Wav2Vec2CTCTokenizer": _PACKAGE + "tokenization",
    "Wav2Vec2Config": _PACKAGE + "configuration",
    "Wav2Vec2Encoder": _PACKAGE + "modeling",
    "Wav2Vec2EncoderLayer": _PACKAGE + "modeling",
    "Wav2Vec2EncoderLayerStableLayerNorm": _PACKAGE + "modeling",
    "Wav2Vec2EncoderOutput": _PACKAGE + "modeling",
    "Wav2Vec2FeatureConvLayer": _PACKAGE + "modeling",
    "Wav2Vec2FeatureEncoder": _PACKAGE + "modeling",
    "Wav2Vec2FeatureProjection": _PACKAGE + "modeling",
    "Wav2Vec2FeatureExtractor": _PACKAGE + "processing",
    "Wav2Vec2FeedForward": _PACKAGE + "modeling",
    "Wav2Vec2ForCTC": _PACKAGE + "modeling",
    "Wav2Vec2ForAudioFrameClassification": _PACKAGE + "modeling",
    "Wav2Vec2ForSequenceClassification": _PACKAGE + "modeling",
    "Wav2Vec2FrameClassifierOutput": _PACKAGE + "modeling",
    "Wav2Vec2Model": _PACKAGE + "modeling",
    "Wav2Vec2ModelOutput": _PACKAGE + "modeling",
    "Wav2Vec2PositionalConvEmbedding": _PACKAGE + "modeling",
    "Wav2Vec2SequenceClassifierOutput": _PACKAGE + "modeling",
    "WeightNormalizedConv1d": _PACKAGE + "modeling",
    "create_wav2vec2_architecture_spec": _PACKAGE + "registration",
    "downsample_wav2vec2_lengths": _PACKAGE + "modeling",
    "feature_attention_mask": _PACKAGE + "modeling",
    "huggingface_wav2vec2_tensor_mapping": _PACKAGE + "checkpoint",
    "huggingface_wav2vec2_tensor_shapes": _PACKAGE + "checkpoint",
    "native_wav2vec2_tensor_names": _PACKAGE + "checkpoint",
    "native_wav2vec2_tensor_shapes": _PACKAGE + "checkpoint",
    "native_wav2vec2_frame_classification_tensor_shapes": _PACKAGE + "checkpoint",
    "native_wav2vec2_sequence_classification_tensor_shapes": _PACKAGE + "checkpoint",
    "register_wav2vec2_architecture": _PACKAGE + "registration",
    "resolve_wav2vec2_artifacts": _PACKAGE + "artifacts",
    "resolve_wav2vec2_classification_artifacts": _PACKAGE + "artifacts",
    "safetensors_header_fingerprint": _PACKAGE + "checkpoint",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve public components only when a caller requests one."""
    try:
        module_name = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return stable results for interactive discovery."""
    return sorted((*globals(), *_EXPORTS))
