"""Serializable configuration for the qwen3tts model family."""

from __future__ import annotations

import math
from numbers import Real

from voicehub.configuration import VoiceHubConfig
from voicehub.models.qwen3tts.lora_config import QWEN3_TTS_LORA_TARGETS, normalize_qwen3_tts_lora_targets


class Qwen3TTSConfig(VoiceHubConfig):
    """Configuration for Qwen3-TTS Base, CustomVoice, and VoiceDesign."""

    model_type = "qwen3tts"

    def __init__(
        self,
        *,
        torch_dtype: str = "bfloat16",
        attention_implementation: str | None = None,
        training_speaker_name: str = "voicehub",
        training_speaker_id: int = 3000,
        sub_talker_loss_weight: float = 0.3,
        training_lora_rank: int | None = None,
        training_lora_alpha: float = 16.0,
        training_lora_dropout: float = 0.0,
        training_lora_target_modules: tuple[str, ...] = QWEN3_TTS_LORA_TARGETS,
        training_lora_seed: int = 0,
        training_lora_export_adapter: bool = True,
        sample_rate: int = 24000,
        **kwargs,
    ):
        if not isinstance(torch_dtype, str) or not torch_dtype.strip():
            raise ValueError("`torch_dtype` must be a non-empty string.")
        if attention_implementation not in {None, "eager"}:
            raise ValueError(
                "Native Qwen3-TTS supports only eager attention; "
                "`attention_implementation` must be None or 'eager'.")
        if (not isinstance(training_speaker_name, str) or not training_speaker_name.strip()):
            raise ValueError("`training_speaker_name` must be non-empty.")
        if (isinstance(training_speaker_id, bool) or not isinstance(training_speaker_id, int) or
                training_speaker_id < 0):
            raise ValueError("`training_speaker_id` must be a non-negative integer.")
        if (isinstance(sub_talker_loss_weight, bool) or not isinstance(sub_talker_loss_weight, Real) or
                not math.isfinite(sub_talker_loss_weight) or sub_talker_loss_weight < 0):
            raise ValueError("`sub_talker_loss_weight` must be finite and non-negative.")
        if training_lora_rank is not None and (isinstance(training_lora_rank, bool) or
                                               not isinstance(training_lora_rank, int) or
                                               training_lora_rank <= 0):
            raise ValueError("`training_lora_rank` must be a positive integer or None.")
        if (isinstance(training_lora_alpha, bool) or not isinstance(training_lora_alpha, Real) or
                not math.isfinite(training_lora_alpha) or training_lora_alpha <= 0):
            raise ValueError("`training_lora_alpha` must be finite and positive.")
        if (isinstance(training_lora_dropout, bool) or not isinstance(training_lora_dropout, Real) or
                not math.isfinite(training_lora_dropout) or not 0 <= training_lora_dropout < 1):
            raise ValueError("`training_lora_dropout` must be in [0, 1).")
        training_lora_target_modules = normalize_qwen3_tts_lora_targets(training_lora_target_modules, )
        if isinstance(training_lora_seed, bool) or not isinstance(training_lora_seed, int):
            raise TypeError("`training_lora_seed` must be an integer.")
        if not isinstance(training_lora_export_adapter, bool):
            raise TypeError("`training_lora_export_adapter` must be a boolean.")
        if sample_rate != 24_000:
            raise ValueError("Published Qwen3-TTS 12 Hz checkpoints output 24 kHz audio.")
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.torch_dtype = torch_dtype.strip()
        self.attention_implementation = attention_implementation
        self.training_speaker_name = training_speaker_name.strip()
        self.training_speaker_id = training_speaker_id
        self.sub_talker_loss_weight = float(sub_talker_loss_weight)
        self.training_lora_rank = training_lora_rank
        self.training_lora_alpha = float(training_lora_alpha)
        self.training_lora_dropout = float(training_lora_dropout)
        self.training_lora_target_modules = training_lora_target_modules
        self.training_lora_seed = training_lora_seed
        self.training_lora_export_adapter = training_lora_export_adapter


__all__ = ['Qwen3TTSConfig']
