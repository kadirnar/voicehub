"""Serializable configuration for the chatterbox model family."""

from __future__ import annotations

from voicehub.configuration import VoiceHubConfig


class ChatterboxConfig(VoiceHubConfig):
    """Configuration for the original Chatterbox architecture."""

    model_type = "chatterbox"

    def __init__(
        self,
        *,
        sample_rate: int = 24000,
        training_component: str = "language_model",
        training_text_loss_weight: float = 1.0,
        training_speech_loss_weight: float = 1.0,
        training_mask_prompt_loss: bool = True,
        training_text_vocab_size: int | None = None,
        training_lora_rank: int | None = None,
        training_lora_alpha: float = 16.0,
        training_lora_dropout: float = 0.0,
        training_lora_target_modules: tuple[str, ...] = (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "spkr_enc",
        ),
        training_lora_modules_to_train: tuple[str, ...] = (
            "text_emb",
            "text_head",
        ),
        training_lora_seed: int = 0,
        training_prompt_duration: float = 3.0,
        training_conditioning_dropout: float = 0.2,
        training_max_text_tokens: int = 256,
        training_max_speech_tokens: int = 850,
        checkpoint_revision: str | None = None,
        **kwargs,
    ):
        super().__init__(
            sample_rate=sample_rate,
            training_component=training_component,
            training_text_loss_weight=training_text_loss_weight,
            training_speech_loss_weight=training_speech_loss_weight,
            training_mask_prompt_loss=training_mask_prompt_loss,
            training_text_vocab_size=training_text_vocab_size,
            training_lora_rank=training_lora_rank,
            training_lora_alpha=training_lora_alpha,
            training_lora_dropout=training_lora_dropout,
            training_lora_target_modules=tuple(training_lora_target_modules),
            training_lora_modules_to_train=tuple(training_lora_modules_to_train),
            training_lora_seed=training_lora_seed,
            training_prompt_duration=training_prompt_duration,
            training_conditioning_dropout=training_conditioning_dropout,
            training_max_text_tokens=training_max_text_tokens,
            training_max_speech_tokens=training_max_speech_tokens,
            checkpoint_revision=checkpoint_revision,
            **kwargs,
        )


__all__ = ['ChatterboxConfig']
