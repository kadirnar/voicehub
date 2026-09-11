"""Public configuration for VoiceHub-native XTTS v2."""

from __future__ import annotations

from pathlib import Path

from voicehub.configuration import VoiceHubConfig, reject_serialized_secrets
from voicehub.models.xtts.native.metadata import XTTS2_CHECKPOINT_REVISION


class XTTSConfig(VoiceHubConfig):
    model_type = "xtts"

    def __init__(
        self,
        *,
        language: str = "en",
        revision: str | None = XTTS2_CHECKPOINT_REVISION,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        torch_dtype: str = "float32",
        training_text_loss_weight: float = 0.01,
        training_mel_loss_weight: float = 1.0,
        training_dvae_checkpoint: str | Path | None = None,
        training_mel_stats_checkpoint: str | Path | None = None,
        sample_rate: int = 24_000,
        **kwargs,
    ) -> None:
        reject_serialized_secrets(kwargs, owner=self.__class__.__name__)
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.language = language
        self.revision = revision
        self.cache_dir = None if cache_dir is None else str(Path(cache_dir).expanduser())
        self.local_files_only = local_files_only
        self.torch_dtype = torch_dtype
        self.training_text_loss_weight = training_text_loss_weight
        self.training_mel_loss_weight = training_mel_loss_weight
        self.training_dvae_checkpoint = (
            None if training_dvae_checkpoint is None else str(Path(training_dvae_checkpoint).expanduser()))
        self.training_mel_stats_checkpoint = (
            None if training_mel_stats_checkpoint is None else str(
                Path(training_mel_stats_checkpoint).expanduser()))
        self.validate()

    def validate(self) -> None:
        reject_serialized_secrets(self.__dict__, owner=self.__class__.__name__)
        if not isinstance(self.language, str) or not self.language.strip():
            raise ValueError("XTTS `language` must be a non-empty language code.")
        self.language = self.language.strip().lower()
        if self.revision is not None and (not isinstance(self.revision, str) or not self.revision.strip()):
            raise ValueError("XTTS `revision` must be a non-empty string or None.")
        if self.revision is not None:
            self.revision = self.revision.strip()
        if not isinstance(self.local_files_only, bool):
            raise TypeError("XTTS `local_files_only` must be a boolean.")
        if self.sample_rate != 24_000:
            raise ValueError("XTTS v2 produces audio at 24,000 Hz.")
        if self.torch_dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("XTTS dtype must be float32, float16, or bfloat16.")
        if ((self.training_dvae_checkpoint is None) != (self.training_mel_stats_checkpoint is None)):
            raise ValueError(
                "XTTS raw-waveform fine-tuning requires both "
                "`training_dvae_checkpoint` and `training_mel_stats_checkpoint`.")
        for name in (
                "training_dvae_checkpoint",
                "training_mel_stats_checkpoint",
        ):
            value = getattr(self, name)
            if value is not None and Path(value).suffix != ".safetensors":
                raise ValueError(f"XTTS `{name}` must point to a `.safetensors` file.")
        weights = (
            self.training_text_loss_weight,
            self.training_mel_loss_weight,
        )
        if any(isinstance(item, bool) or not isinstance(item, (int, float)) or item < 0
               for item in weights) or sum(weights) <= 0:
            raise ValueError("XTTS training loss weights must be non-negative and non-zero.")


__all__ = ["XTTSConfig"]
