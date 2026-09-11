"""Configuration for the VoiceHub-native Inflect Micro/Nano v2 runtime."""

from __future__ import annotations

import math
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING, Any

from voicehub.configuration import VoiceHubConfig, reject_serialized_secrets

if TYPE_CHECKING:
    from voicehub.models.inflecttts.native.training import InflectLossWeights


class InflectTTSConfig(VoiceHubConfig):
    """Configure audited Inflect v2 loading and preprocessed fine-tuning."""

    model_type = "inflecttts"

    def __init__(
        self,
        *,
        checkpoint_filename: str | None = None,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        trust_pickle_checkpoint: bool = False,
        enable_native_finetuning: bool = False,
        training_enable_discriminator: bool = True,
        training_mel_loss_weight: float = 45.0,
        training_kl_loss_weight: float = 1.0,
        training_duration_loss_weight: float = 1.0,
        training_adversarial_loss_weight: float = 1.0,
        training_feature_matching_loss_weight: float = 1.0,
        training_waveform_loss_weight: float = 0.0,
        sample_rate: int = 24_000,
        **kwargs: Any,
    ) -> None:
        reject_serialized_secrets(kwargs, owner=self.__class__.__name__)
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.checkpoint_filename = checkpoint_filename
        self.revision = revision
        self.cache_dir = (None if cache_dir is None else str(Path(cache_dir).expanduser()))
        self.local_files_only = local_files_only
        self.trust_pickle_checkpoint = trust_pickle_checkpoint
        self.enable_native_finetuning = enable_native_finetuning
        self.training_enable_discriminator = training_enable_discriminator
        self.training_mel_loss_weight = training_mel_loss_weight
        self.training_kl_loss_weight = training_kl_loss_weight
        self.training_duration_loss_weight = training_duration_loss_weight
        self.training_adversarial_loss_weight = training_adversarial_loss_weight
        self.training_feature_matching_loss_weight = (training_feature_matching_loss_weight)
        self.training_waveform_loss_weight = training_waveform_loss_weight
        self._validate()

    @property
    def _loss_weight_values(self) -> tuple[tuple[str, object], ...]:
        return (
            ("mel", self.training_mel_loss_weight),
            ("kl", self.training_kl_loss_weight),
            ("duration", self.training_duration_loss_weight),
            ("adversarial", self.training_adversarial_loss_weight),
            ("feature_matching", self.training_feature_matching_loss_weight),
            ("waveform", self.training_waveform_loss_weight),
        )

    def _validate(self) -> None:
        if self.checkpoint_filename is not None:
            if (not isinstance(self.checkpoint_filename, str) or not self.checkpoint_filename.strip() or
                    Path(self.checkpoint_filename).name != self.checkpoint_filename):
                raise ValueError("`checkpoint_filename` must be one non-empty root filename.")
            if not self.checkpoint_filename.endswith((".safetensors", ".pth")):
                raise ValueError("Inflect checkpoints use .safetensors or the released "
                                 ".pth format.")
        if self.revision is not None:
            if (not isinstance(self.revision, str) or not self.revision.strip()):
                raise ValueError("`revision` must be non-empty or None.")
            self.revision = self.revision.strip()
        for name in (
                "local_files_only",
                "trust_pickle_checkpoint",
                "enable_native_finetuning",
                "training_enable_discriminator",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"`{name}` must be a boolean.")

        normalized_weights = []
        for name, value in self._loss_weight_values:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"`{name}` must be a real number.")
            normalized = float(value)
            if not math.isfinite(normalized) or normalized < 0:
                raise ValueError(f"`{name}` must be finite and non-negative.")
            normalized_weights.append(normalized)
        if not any(value > 0 for value in normalized_weights):
            raise ValueError("At least one Inflect loss weight must be positive.")
        if self.sample_rate != 24_000:
            raise ValueError("Inflect Micro/Nano v2 checkpoints synthesize at 24 kHz.")

    @property
    def training_loss_weights(self) -> InflectLossWeights:
        """Build the native objective weights only when training is loaded."""
        from voicehub.models.inflecttts.native.training import InflectLossWeights

        return InflectLossWeights(
            mel=self.training_mel_loss_weight,
            kl=self.training_kl_loss_weight,
            duration=self.training_duration_loss_weight,
            adversarial=self.training_adversarial_loss_weight,
            feature_matching=self.training_feature_matching_loss_weight,
            waveform=self.training_waveform_loss_weight,
        )


__all__ = ["InflectTTSConfig"]
