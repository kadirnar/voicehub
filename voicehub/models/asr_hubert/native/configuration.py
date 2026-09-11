"""Validated configuration for VoiceHub's native HuBERT CTC family.

Field and graph semantics were reviewed against Hugging Face
Transformers revision ``ebea912f0bb6f9e28ad2df04acd9b4df035933a9`` and
the immutable ``facebook/hubert-large-ls960-ft`` configuration at
revision ``ece5fabbf034c1073acae96d5401b25be96709d8``.  The
implementation reuses the Wav2Vec2 convolutional/encoder contract only
where HuBERT is structurally identical.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from voicehub.models.asr_wav2vec2.native.configuration import Wav2Vec2Config


@dataclass(frozen=True, slots=True)
class HubertConfig(Wav2Vec2Config):
    """Executable configuration for HuBERT CTC encoders.

    HuBERT and Wav2Vec2 share the acoustic convolutional frontend and
    Transformer encoder. HuBERT additionally makes feature-projection
    normalization explicit and stores a learned SpecAugment time-mask
    embedding.
    """

    feat_proj_layer_norm: bool = True
    conv_pos_batch_norm: bool = False

    def __post_init__(self) -> None:
        # ``slots=True`` dataclasses return a replacement class object, which
        # makes zero-argument ``super()`` unreliable on supported Python
        # versions. Call the validated base contract directly.
        Wav2Vec2Config.__post_init__(self)
        for name in ("feat_proj_layer_norm", "conv_pos_batch_norm"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"`{name}` must be a boolean.")
        if self.conv_pos_batch_norm:
            raise ValueError(
                "Native HuBERT does not support `conv_pos_batch_norm=True`; "
                "the official CTC checkpoint uses weight-normalized "
                "positional convolution.")

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> HubertConfig:
        """Parse and validate a Hugging Face-compatible HuBERT mapping."""
        if not isinstance(values, Mapping):
            raise TypeError("HuBERT configuration values must be a mapping.")
        if values.get("gradient_checkpointing", False):
            raise ValueError(
                "Native HuBERT does not silently enable serialized gradient "
                "checkpointing. Configure training memory strategies through "
                "the VoiceHub trainer.")
        if values.get("use_weighted_layer_sum", False):
            raise ValueError(
                "`use_weighted_layer_sum=True` is a sequence-classification "
                "graph and is unsupported by native HuBERT CTC.")
        feat_extract_dropout = values.get("feat_extract_dropout", 0.0)
        if feat_extract_dropout not in (0, 0.0):
            raise ValueError(
                "Native HuBERT does not approximate the non-standard "
                "`feat_extract_dropout` option.")
        return Wav2Vec2Config.from_dict.__func__(cls, values)

    def to_dict(self) -> dict[str, Any]:
        """Return a detached HuBERT-compatible configuration mapping."""
        result = Wav2Vec2Config.to_dict(self)
        result["model_type"] = "hubert"
        result['architectures'] = ["HubertForCTC"]
        return result


__all__ = ["HubertConfig"]
