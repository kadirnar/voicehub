"""Serializable configuration for the vibevoice model family."""

from __future__ import annotations

import math
from pathlib import Path

from voicehub.configuration import VoiceHubConfig


class VibeVoiceConfig(VoiceHubConfig):
    """Configuration for the VibeVoice realtime streaming architecture."""

    model_type = "vibevoice"

    def __init__(
        self,
        *,
        torch_dtype: str = "bfloat16",
        attention_implementation: str = "sdpa",
        diffusion_steps: int = 5,
        training_ce_loss_weight: float = 1.0,
        training_diffusion_loss_weight: float = 1.0,
        training_ddpm_batch_mul: int = 1,
        sample_rate: int = 24000,
        revision: str | None = None,
        cache_dir: str | None = None,
        local_files_only: bool = False,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.torch_dtype = torch_dtype
        self.attention_implementation = attention_implementation
        self.diffusion_steps = diffusion_steps
        self.training_ce_loss_weight = training_ce_loss_weight
        self.training_diffusion_loss_weight = training_diffusion_loss_weight
        self.training_ddpm_batch_mul = training_ddpm_batch_mul
        self.revision = revision
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self.validate()

    def validate(self) -> None:
        if self.sample_rate != 24_000:
            raise ValueError("Published VibeVoice checkpoints require 24 kHz.")
        if not isinstance(self.torch_dtype, str) or not self.torch_dtype.strip():
            raise TypeError("`torch_dtype` must be a non-empty string.")
        if self.attention_implementation != "sdpa":
            raise ValueError(
                "Native VibeVoice currently implements the audited SDPA "
                "attention path only.")
        if (isinstance(self.diffusion_steps, bool) or not isinstance(self.diffusion_steps, int) or
                self.diffusion_steps <= 0):
            raise ValueError("`diffusion_steps` must be a positive integer.")
        for name in (
                "training_ce_loss_weight",
                "training_diffusion_loss_weight",
        ):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                    not math.isfinite(float(value)) or value < 0):
                raise ValueError(f"`{name}` must be finite and non-negative.")
        if (isinstance(self.training_ddpm_batch_mul, bool) or
                not isinstance(self.training_ddpm_batch_mul, int) or self.training_ddpm_batch_mul <= 0):
            raise ValueError("`training_ddpm_batch_mul` must be a positive integer.")
        if self.revision is not None and (not isinstance(self.revision, str) or not self.revision.strip()):
            raise ValueError("`revision` must be a non-empty string or None.")
        if self.cache_dir is not None:
            if not isinstance(self.cache_dir, (str, Path)):
                raise TypeError("`cache_dir` must be path-like or None.")
            self.cache_dir = str(Path(self.cache_dir).expanduser())
        if not isinstance(self.local_files_only, bool):
            raise TypeError("`local_files_only` must be a boolean.")


__all__ = ['VibeVoiceConfig']
