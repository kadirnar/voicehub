"""Configuration for VoiceHub's native Silero VAD provider."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from numbers import Real
from pathlib import Path
from typing import Any

from voicehub.configuration import VoiceHubConfig, reject_serialized_secrets
from voicehub.inference_configuration import VADInferenceConfig

DEFAULT_SILERO_VAD_REVISION = ("8a63e2e86cf654d7ba19fbedbccce5ff55de3c60")


class SileroVADConfig(VoiceHubConfig):
    """Configure native Silero checkpoint loading, inference, and tuning.

    ``use_onnx`` and ``force_reload`` remain accepted so existing
    serialized configurations fail with an actionable error instead of
    being silently reinterpreted. VoiceHub's native provider executes
    its own PyTorch graph; callers that need ONNX should select the
    dedicated Sherpa-ONNX provider.
    """

    model_type = "vad_silero"
    architecture_family = "frame-classification"

    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        revision: str | None = DEFAULT_SILERO_VAD_REVISION,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        checkpoint_filename: str | None = None,
        use_onnx: bool = False,
        force_reload: bool = False,
        training_train_encoder: bool = False,
        training_noise_loss: float = 0.5,
        training_max_duration_s: float = 8.0,
        training_label_threshold: float = 0.5,
        inference_config: VADInferenceConfig | Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        reject_serialized_secrets(
            {
                "inference_config": inference_config,
                **kwargs,
            },
            owner=self.__class__.__name__,
        )
        if isinstance(inference_config, VADInferenceConfig):
            inference_values = inference_config.to_dict()
        elif inference_config is None:
            inference_values = {}
        elif isinstance(inference_config, Mapping):
            inference_values = dict(inference_config)
        else:
            raise TypeError("`inference_config` must be a VADInferenceConfig, mapping, "
                            "or None.")
        for name in VADInferenceConfig._COMMON_FIELDS:
            if name in kwargs:
                inference_values[name] = kwargs.pop(name)

        super().__init__(
            sample_rate=sample_rate,
            inference_config=VADInferenceConfig.from_dict(inference_values).to_dict(),
            **kwargs,
        )
        self.revision = revision
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self.checkpoint_filename = checkpoint_filename
        self.use_onnx = use_onnx
        self.force_reload = force_reload
        self.training_train_encoder = training_train_encoder
        self.training_noise_loss = training_noise_loss
        self.training_max_duration_s = training_max_duration_s
        self.training_label_threshold = training_label_threshold
        self.validate()

    def validate(self) -> None:
        """Reject settings outside the released native graph contract."""
        reject_serialized_secrets(
            self.__dict__,
            owner=self.__class__.__name__,
        )
        if self.sample_rate not in (8_000, 16_000):
            raise ValueError("Silero VAD supports 8 kHz or 16 kHz audio.")
        for name in (
                "local_files_only",
                "use_onnx",
                "force_reload",
                "training_train_encoder",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"`{name}` must be a boolean.")
        if self.use_onnx:
            raise ValueError(
                "The native Silero provider does not execute ONNX graphs. "
                "Use `model_type='vad_sherpa_onnx'` for ONNX inference.")
        if self.force_reload:
            raise ValueError(
                "The native checkpoint resolver uses immutable, verified "
                "cache entries and does not support `force_reload=True`.")
        for name in (
                "training_noise_loss",
                "training_label_threshold",
        ):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value) or
                    not 0.0 <= value <= 1.0):
                raise ValueError(f"`{name}` must be finite and between 0 and 1.")
            setattr(self, name, float(value))
        maximum = self.training_max_duration_s
        if (isinstance(maximum, bool) or not isinstance(maximum, Real) or not isfinite(maximum) or
                maximum <= 0.0):
            raise ValueError("`training_max_duration_s` must be finite and positive.")
        self.training_max_duration_s = float(maximum)
        if self.revision is not None:
            if not isinstance(self.revision, str) or not self.revision.strip():
                raise ValueError("`revision` must be a non-empty string or None.")
            self.revision = self.revision.strip()
        if self.cache_dir is not None:
            if not isinstance(self.cache_dir, (str, Path)):
                raise TypeError("`cache_dir` must be path-like or None.")
            self.cache_dir = str(Path(self.cache_dir).expanduser())
        if self.checkpoint_filename is not None:
            if (not isinstance(self.checkpoint_filename, str) or not self.checkpoint_filename.strip()):
                raise ValueError("`checkpoint_filename` must be a non-empty string or None.")
            self.checkpoint_filename = self.checkpoint_filename.strip()
            suffix = Path(self.checkpoint_filename).suffix.lower()
            if suffix not in {".safetensors", ".jit"}:
                raise ValueError(
                    "Native Silero checkpoints must be Safetensors or an "
                    "official `.jit` archive used only for strict weight "
                    "conversion.")
        values = getattr(self, "inference_config", {})
        if isinstance(values, VADInferenceConfig):
            values = values.to_dict()
        if not isinstance(values, Mapping):
            raise TypeError("`inference_config` must serialize to a mapping.")
        self.inference_config = VADInferenceConfig.from_dict(dict(values)).to_dict()

    def to_dict(self) -> dict[str, Any]:
        """Validate mutable fields before serialization."""
        self.validate()
        return super().to_dict()


__all__ = [
    "DEFAULT_SILERO_VAD_REVISION",
    "SileroVADConfig",
]
