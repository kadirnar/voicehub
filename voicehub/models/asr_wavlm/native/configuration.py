"""Validated configuration for VoiceHub's native WavLM CTC family.

The graph contract was reviewed against Microsoft's official WavLM
source at revision ``833df7e7832e5064a281131ee64a481afa8e5b95`` and
Hugging Face Transformers revision
``ebea912f0bb6f9e28ad2df04acd9b4df035933a9``.  Neither runtime is
imported or executed by VoiceHub.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from voicehub.models.asr_wav2vec2.native.configuration import Wav2Vec2Config


@dataclass(frozen=True, slots=True)
class WavLMConfig(Wav2Vec2Config):
    """Executable WavLM CTC configuration.

    WavLM shares its raw-waveform convolutional frontend and feed-
    forward blocks with Wav2Vec2.  Its bucketed relative-position bias
    and content-dependent GRU-style gate are first-class fields because
    they change every encoder attention layer.
    """

    num_buckets: int = 320
    max_bucket_distance: int = 800

    def __post_init__(self) -> None:
        # ``slots=True`` dataclasses return a replacement class, so an
        # explicit base call is reliable across every supported Python.
        Wav2Vec2Config.__post_init__(self)
        for name in ("num_buckets", "max_bucket_distance"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"`{name}` must be an integer.")
            if value < 1:
                raise ValueError(f"`{name}` must be positive.")
        if self.num_buckets < 4 or self.num_buckets % 2:
            raise ValueError("`num_buckets` must be an even integer of at least four.")
        maximum_exact_distance = self.num_buckets // 4
        if self.max_bucket_distance <= maximum_exact_distance:
            raise ValueError(
                "`max_bucket_distance` must be greater than "
                f"`num_buckets // 4` ({maximum_exact_distance}).")

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> WavLMConfig:
        """Parse a WavLM CTC configuration and reject other graph variants."""
        if not isinstance(values, Mapping):
            raise TypeError("WavLM configuration values must be a mapping.")
        if values.get("add_adapter", False):
            raise ValueError(
                "Native WavLM CTC does not support the optional convolutional "
                "language-adapter graph declared by `add_adapter=True`.")
        if values.get("adapter_attn_dim") is not None:
            raise ValueError("Native WavLM CTC does not support attention-adapter checkpoints.")
        if values.get("gradient_checkpointing", False):
            raise ValueError(
                "Native WavLM does not silently enable serialized gradient "
                "checkpointing. Configure memory strategies through the "
                "VoiceHub trainer.")
        if values.get("use_weighted_layer_sum", False):
            raise ValueError(
                "`use_weighted_layer_sum=True` belongs to WavLM classification "
                "heads and is unsupported by the native CTC graph.")
        for name in ("mask_time_selection", "mask_channel_selection"):
            selection = values.get(name, "static")
            if selection != "static":
                raise ValueError(
                    f"Native WavLM supports only static SpecAugment spans; "
                    f"`{name}={selection!r}` is unsupported.")
        for name in ("no_mask_time_overlap", "no_mask_channel_overlap"):
            if values.get(name, False):
                raise ValueError(f"Native WavLM does not approximate `{name}=True`.")
        for name in ("mask_time_other", "mask_channel_other"):
            if values.get(name, 0.0) not in (0, 0.0):
                raise ValueError(
                    f"`{name}` is valid only for unsupported non-static "
                    "SpecAugment distributions.")
        if values.get("mask_channel_prob", 0.0) not in (0, 0.0):
            raise ValueError(
                "Legacy `mask_channel_prob` checkpoints are unsupported; use "
                "the checkpoint's canonical `mask_feature_prob` field.")
        if values.get("feat_extract_dropout", 0.0) not in (0, 0.0):
            raise ValueError(
                "Native WavLM does not approximate non-standard "
                "`feat_extract_dropout` behavior.")
        return Wav2Vec2Config.from_dict.__func__(cls, values)

    def to_dict(self) -> dict[str, Any]:
        """Return a detached WavLM-compatible configuration mapping."""
        result = Wav2Vec2Config.to_dict(self)
        result["model_type"] = "wavlm"
        result['architectures'] = ["WavLMForCTC"]
        return result


__all__ = ["WavLMConfig"]
