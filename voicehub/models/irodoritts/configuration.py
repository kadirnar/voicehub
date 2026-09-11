"""Serializable configuration for the irodoritts model family."""

from __future__ import annotations

import math
from pathlib import Path

from voicehub.configuration import VoiceHubConfig


class IrodoriTTSConfig(VoiceHubConfig):
    """Configuration for Irodori-TTS v2/v3 and VoiceDesign checkpoints."""

    model_type = "irodoritts"

    def __init__(
        self,
        *,
        codec_name_or_path: str = "Aratako/Semantic-DACVAE-Japanese-32dim",
        tokenizer_name_or_path: str = "llm-jp/llm-jp-3-150m",
        codec_revision: str = "47376ee24834d7a05a48ebabfe3cde29b3c5e214",
        tokenizer_revision: str = "b112feef602fff752e4dac4c30af6a2c2fa41c7a",
        model_revision: str | None = None,
        model_precision: str = "fp32",
        codec_precision: str = "fp32",
        compile_model: bool = False,
        checkpoint_filename: str | None = None,
        sample_rate: int = 48000,
        training_objective: str = "joint",
        training_rf_loss_mode: str = "utterance_mean",
        training_duration_loss_weight: float = 0.1,
        training_duration_huber_delta: float = 0.1,
        training_gradient_checkpointing: bool = True,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.codec_name_or_path = codec_name_or_path
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.codec_revision = codec_revision
        self.tokenizer_revision = tokenizer_revision
        self.model_revision = model_revision
        self.model_precision = model_precision
        self.codec_precision = codec_precision
        self.compile_model = compile_model
        self.checkpoint_filename = checkpoint_filename
        self.training_objective = training_objective
        self.training_rf_loss_mode = training_rf_loss_mode
        self.training_duration_loss_weight = training_duration_loss_weight
        self.training_duration_huber_delta = training_duration_huber_delta
        self.training_gradient_checkpointing = training_gradient_checkpointing
        self.validate()

    def validate(self) -> None:
        for name in ("model_precision", "codec_precision"):
            value = str(getattr(self, name)).strip().lower()
            if value not in {"fp32", "bf16"}:
                raise ValueError(f"`{name}` must be 'fp32' or 'bf16'.")
            setattr(self, name, value)
        if not isinstance(self.compile_model, bool):
            raise TypeError("`compile_model` must be a boolean.")
        if not isinstance(self.training_gradient_checkpointing, bool):
            raise TypeError("`training_gradient_checkpointing` must be a boolean.")
        self.training_objective = str(self.training_objective).strip().lower()
        if self.training_objective not in {"flow", "duration", "joint"}:
            raise ValueError("`training_objective` must be flow, duration, or joint.")
        self.training_rf_loss_mode = str(self.training_rf_loss_mode).strip().lower()
        if self.training_rf_loss_mode not in {"echo", "utterance_mean"}:
            raise ValueError("`training_rf_loss_mode` must be echo or utterance_mean.")
        for name in ("training_duration_loss_weight", "training_duration_huber_delta"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                    not math.isfinite(float(value)) or value <= 0):
                raise ValueError(f"`{name}` must be finite and positive.")
        if self.checkpoint_filename is not None:
            filename = Path(self.checkpoint_filename)
            if filename.name != self.checkpoint_filename:
                raise ValueError("`checkpoint_filename` must be a filename without directories.")


__all__ = ['IrodoriTTSConfig']
