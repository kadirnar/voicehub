"""Dependency-light configuration for VoiceHub-native F5-TTS."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from voicehub.configuration import VoiceHubConfig
from voicehub.models.f5tts.native.configuration import F5TTSArchitectureConfig, f5tts_architecture_config


class F5TTSConfig(VoiceHubConfig):
    """Configuration for the dependency-free F5-TTS runtime."""

    model_type = "f5tts"

    def __init__(
        self,
        *,
        model_name: str = "F5TTS_v1_Base",
        checkpoint_path: str = "",
        vocabulary_path: str = "",
        architecture: Mapping[str, Any] | None = None,
        ode_method: str = "euler",
        torch_dtype: str = "float32",
        allow_unvalidated_reduced_precision_inference: bool = False,
        use_ema: bool = True,
        ema_decay: float = 0.9999,
        ema_update_after_step: int = 0,
        ema_update_every: int = 1,
        vocoder_path: str | None = None,
        cache_dir: str | None = None,
        revision: str | None = None,
        token: str | bool | None = None,
        local_files_only: bool = False,
        verify_artifacts: bool = False,
        sample_rate: int = 24_000,
        **kwargs: Any,
    ) -> None:
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.vocabulary_path = vocabulary_path
        self.architecture = dict(architecture or {})
        self.ode_method = ode_method
        self.torch_dtype = torch_dtype
        self.allow_unvalidated_reduced_precision_inference = (allow_unvalidated_reduced_precision_inference)
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.ema_update_after_step = ema_update_after_step
        self.ema_update_every = ema_update_every
        self.vocoder_path = vocoder_path
        self.cache_dir = cache_dir
        self.revision = revision
        # Runtime credentials are accepted but rejected by config
        # serialization through VoiceHubConfig's normal secret policy.
        self.token = token
        self.local_files_only = local_files_only
        self.verify_artifacts = verify_artifacts
        self.validate()

    def validate(self) -> None:
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("`model_name` must be a non-empty string.")
        if self.ode_method not in {"euler", "midpoint"}:
            raise ValueError("`ode_method` must be 'euler' or 'midpoint'.")
        if not isinstance(self.torch_dtype, str) or not self.torch_dtype.strip():
            raise ValueError("`torch_dtype` must be a non-empty string.")
        if not isinstance(self.architecture, Mapping):
            raise TypeError("`architecture` must be a mapping.")
        if not 0 < float(self.ema_decay) <= 1:
            raise ValueError("`ema_decay` must be in the interval (0, 1].")
        for name in ("ema_update_after_step", "ema_update_every"):
            value = getattr(self, name)
            minimum = 0 if name == "ema_update_after_step" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"`{name}` must be an integer greater than or equal to "
                                 f"{minimum}.")
        for name in (
                "allow_unvalidated_reduced_precision_inference",
                "use_ema",
                "local_files_only",
                "verify_artifacts",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"`{name}` must be a boolean.")

    def architecture_config(self) -> F5TTSArchitectureConfig:
        if self.architecture:
            values = dict(self.architecture)
            values.setdefault("model_name", self.model_name)
            values.setdefault("sample_rate", self.sample_rate)
            return F5TTSArchitectureConfig.from_mapping(values)
        return f5tts_architecture_config(self.model_name)

    def to_dict(self) -> dict[str, Any]:
        values = super().to_dict()
        values.pop("token", None)
        return values


__all__ = ["F5TTSConfig"]
