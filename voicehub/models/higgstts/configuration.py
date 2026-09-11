"""Public configuration for VoiceHub's native Higgs Audio v2 runtime."""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
from pathlib import Path
from typing import Any

from voicehub.configuration import VoiceHubConfig, reject_serialized_secrets
from voicehub.models.higgstts.native.metadata import HIGGS_AUDIO_V2_TOKENIZER_REPOSITORY

DEFAULT_SYSTEM_PROMPT = "Generate audio following instruction."
DEFAULT_SCENE_PROMPT = "Audio is recorded from a quiet room."

_DTYPE_ALIASES = {
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "float": "float32",
    "float16": "float16",
    "float32": "float32",
    "fp16": "float16",
    "fp32": "float32",
    "half": "float16",
}


def _optional_text(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"`{name}` must be a non-empty string or None.")
    return value.strip()


def _source(value: object, *, name: str) -> str:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"`{name}` must be a non-empty path or Hub ID.")
    return str(value)


def _optional_path(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"`{name}` must be a non-empty path or None.")
    return str(Path(value).expanduser())


def _loss_weight(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"`{name}` must be a real number.")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"`{name}` must be finite and non-negative.")
    return normalized


class HiggsTTSConfig(VoiceHubConfig):
    """Configure native Higgs Audio v2 loading, generation, and SFT.

    The serialized configuration contains no credentials and never
    enables repository code execution. The public wrapper accepts a Hub
    token only as a constructor-time runtime value.
    """

    model_type = "higgstts"

    def __init__(
        self,
        *,
        audio_tokenizer_name_or_path: str | Path = (HIGGS_AUDIO_V2_TOKENIZER_REPOSITORY),
        revision: str | None = None,
        codec_revision: str | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        torch_dtype: str = "bfloat16",
        use_safetensors: bool | None = None,
        trust_remote_code: bool = False,
        verify_integrity: bool = True,
        verify_checkpoint_integrity: bool = False,
        training_text_loss_weight: float = 1.0,
        training_audio_loss_weight: float = 1.0,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        scene_prompt: str | None = DEFAULT_SCENE_PROMPT,
        generation_config: Mapping[str, Any] | None = None,
        sample_rate: int = 24_000,
        **kwargs: Any,
    ) -> None:
        reject_serialized_secrets(kwargs, owner=self.__class__.__name__)
        generation_defaults: dict[str, Any] = {
            "force_audio_gen": True,
            "max_new_tokens": 1_024,
            "ras_win_len": 7,
            "ras_win_max_num_repeat": 2,
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 0.95,
        }
        if generation_config is not None:
            if not isinstance(generation_config, Mapping):
                raise TypeError("`generation_config` must be a mapping or None.")
            generation_defaults.update(dict(generation_config))
        super().__init__(
            sample_rate=sample_rate,
            generation_config=generation_defaults,
            **kwargs,
        )
        self.audio_tokenizer_name_or_path = audio_tokenizer_name_or_path
        self.revision = revision
        self.codec_revision = codec_revision
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self.torch_dtype = torch_dtype
        self.use_safetensors = use_safetensors
        self.trust_remote_code = trust_remote_code
        self.verify_integrity = verify_integrity
        self.verify_checkpoint_integrity = verify_checkpoint_integrity
        self.training_text_loss_weight = training_text_loss_weight
        self.training_audio_loss_weight = training_audio_loss_weight
        self.system_prompt = system_prompt
        self.scene_prompt = scene_prompt
        self.validate()

    def validate(self) -> None:
        """Validate native-runtime options without importing PyTorch."""
        reject_serialized_secrets(self.__dict__, owner=self.__class__.__name__)
        if self.sample_rate != 24_000:
            raise ValueError("Higgs Audio v2's native tokenizer produces 24,000 Hz audio.")
        self.audio_tokenizer_name_or_path = _source(
            self.audio_tokenizer_name_or_path,
            name="audio_tokenizer_name_or_path",
        )
        self.revision = _optional_text(self.revision, name="revision")
        self.codec_revision = _optional_text(
            self.codec_revision,
            name="codec_revision",
        )
        self.cache_dir = _optional_path(self.cache_dir, name="cache_dir")
        self.system_prompt = _optional_text(
            self.system_prompt,
            name="system_prompt",
        )
        if self.system_prompt is None:
            raise ValueError("`system_prompt` cannot be None.")
        self.scene_prompt = _optional_text(
            self.scene_prompt,
            name="scene_prompt",
        )
        for name in (
                "local_files_only",
                "trust_remote_code",
                "verify_integrity",
                "verify_checkpoint_integrity",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"`{name}` must be a boolean.")
        if self.trust_remote_code:
            raise ValueError(
                "Native Higgs never executes repository code; "
                "`trust_remote_code=True` is unsupported.")
        if self.use_safetensors is False:
            raise ValueError("Native Higgs Audio v2 requires Safetensors checkpoints.")
        if self.use_safetensors not in (None, True):
            raise TypeError("`use_safetensors` must be a boolean or None.")
        if not isinstance(self.torch_dtype, str):
            raise TypeError("`torch_dtype` must be a string.")
        normalized = _DTYPE_ALIASES.get(self.torch_dtype.strip().lower().removeprefix("torch."), )
        if normalized is None:
            supported = ", ".join(sorted(set(_DTYPE_ALIASES.values())))
            raise ValueError(
                f"Unsupported Higgs dtype {self.torch_dtype!r}; "
                f"expected one of: {supported}.")
        self.torch_dtype = normalized
        self.training_text_loss_weight = _loss_weight(
            self.training_text_loss_weight,
            name="training_text_loss_weight",
        )
        self.training_audio_loss_weight = _loss_weight(
            self.training_audio_loss_weight,
            name="training_audio_loss_weight",
        )
        if (self.training_text_loss_weight == 0.0 and self.training_audio_loss_weight == 0.0):
            raise ValueError("At least one Higgs training loss weight must be positive.")


__all__ = ["HiggsTTSConfig"]
