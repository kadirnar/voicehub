"""Configuration for VoiceHub's fully native OmniVoice provider."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicehub.configuration import VoiceHubConfig, reject_serialized_secrets
from voicehub.path_utils import is_explicit_local_path

_DTYPES = {
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "float": "float32",
    "float16": "float16",
    "float32": "float32",
    "fp16": "float16",
    "fp32": "float32",
    "half": "float16",
}
_GENERATION_KEYS = frozenset({
    "class_temperature",
    "denoise",
    "audio_chunk_duration",
    "audio_chunk_threshold",
    "fade_duration",
    "guidance_scale",
    "layer_penalty_factor",
    "num_steps",
    "pad_duration",
    "position_temperature",
    "postprocess_output",
    "time_shift",
})
_MASKING_KEYS = frozenset({
    "drop_cond_ratio",
    "instruct_ratio",
    "language_ratio",
    "mask_ratio_range",
    "normalize_raw_audio",
    "only_instruct_ratio",
    "prompt_ratio_range",
    "use_pinyin_ratio",
})


class OmniVoiceConfig(VoiceHubConfig):
    """Configure checkpoint loading, decoding, and masked-token fine-tuning."""

    model_type = "omnivoice"

    def __init__(
        self,
        *,
        revision: str | None = None,
        codec_source: str | Path | None = None,
        codec_revision: str | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        torch_dtype: str = "float32",
        use_safetensors: bool | None = None,
        trust_remote_code: bool = False,
        verify_integrity: bool = True,
        verify_checkpoint_integrity: bool = False,
        training_masking_config: Mapping[str, Any] | None = None,
        training_packing_tokens: int | None = None,
        generation_config: Mapping[str, Any] | None = None,
        sample_rate: int = 24_000,
        **kwargs: Any,
    ) -> None:
        reject_serialized_secrets(kwargs, owner=self.__class__.__name__)
        defaults = {
            "audio_chunk_duration": 15.0,
            "audio_chunk_threshold": 30.0,
            "class_temperature": 0.0,
            "denoise": True,
            "fade_duration": 0.1,
            "guidance_scale": 2.0,
            "layer_penalty_factor": 5.0,
            "num_steps": 32,
            "pad_duration": 0.1,
            "position_temperature": 5.0,
            "postprocess_output": True,
            "time_shift": 0.1,
        }
        if generation_config is not None:
            if not isinstance(generation_config, Mapping):
                raise TypeError("`generation_config` must be a mapping.")
            defaults.update(dict(generation_config))
        super().__init__(
            sample_rate=sample_rate,
            generation_config=defaults,
            **kwargs,
        )
        self.revision = revision
        self.codec_source = codec_source
        self.codec_revision = codec_revision
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self.torch_dtype = torch_dtype
        self.use_safetensors = use_safetensors
        self.trust_remote_code = trust_remote_code
        self.verify_integrity = verify_integrity
        self.verify_checkpoint_integrity = verify_checkpoint_integrity
        self.training_masking_config = (
            None if training_masking_config is None else dict(training_masking_config))
        self.training_packing_tokens = training_packing_tokens
        self.validate()

    @staticmethod
    def _optional_text(value: object, *, name: str) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"`{name}` must be a non-empty string or None.")
        return value.strip()

    def validate(self) -> None:
        reject_serialized_secrets(self.__dict__, owner=self.__class__.__name__)
        if self.sample_rate != 24_000:
            raise ValueError("Higgs Audio V2 produces 24,000 Hz audio.")
        self.revision = self._optional_text(self.revision, name="revision")
        self.codec_revision = self._optional_text(
            self.codec_revision,
            name="codec_revision",
        )
        if self.codec_source is not None:
            if (not isinstance(self.codec_source, (str, Path)) or not str(self.codec_source).strip()):
                raise ValueError("`codec_source` must be a non-empty path/ID.")
            if is_explicit_local_path(self.codec_source):
                self.codec_source = str(Path(self.codec_source).expanduser())
            else:
                self.codec_source = str(self.codec_source).strip()
        if self.cache_dir is not None:
            if (not isinstance(self.cache_dir, (str, Path)) or not str(self.cache_dir).strip()):
                raise ValueError("`cache_dir` must be a non-empty path.")
            self.cache_dir = str(Path(self.cache_dir).expanduser())
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
                "Native OmniVoice never executes repository code; "
                "`trust_remote_code=True` is unsupported.")
        if self.use_safetensors is False:
            raise ValueError("Native OmniVoice requires Safetensors.")
        if self.use_safetensors not in (None, True):
            raise TypeError("`use_safetensors` must be a boolean or None.")
        if not isinstance(self.torch_dtype, str):
            raise TypeError("`torch_dtype` must be a string.")
        normalized = _DTYPES.get(self.torch_dtype.lower().removeprefix("torch."))
        if normalized is None:
            raise ValueError(f"Unsupported OmniVoice dtype {self.torch_dtype!r}.")
        self.torch_dtype = normalized
        if self.training_packing_tokens is not None and (isinstance(self.training_packing_tokens, bool) or
                                                         not isinstance(self.training_packing_tokens, int) or
                                                         self.training_packing_tokens <= 0):
            raise ValueError("`training_packing_tokens` must be a positive integer or None.")
        masking = self.training_masking_config
        if masking is not None:
            unknown = sorted(set(masking) - _MASKING_KEYS)
            if unknown:
                raise ValueError("Unsupported OmniVoice masking option(s): " + ", ".join(unknown))
        generation = self.generation_config
        if not isinstance(generation, Mapping):
            raise TypeError("`generation_config` must be a mapping.")
        unknown = sorted(set(generation) - _GENERATION_KEYS)
        if unknown:
            raise ValueError("Unsupported OmniVoice generation option(s): " + ", ".join(unknown))
        if (isinstance(generation["num_steps"], bool) or not isinstance(generation["num_steps"], int) or
                generation["num_steps"] <= 0):
            raise ValueError("`generation_config.num_steps` must be positive.")
        for name, value in generation.items():
            if name in {"denoise", "postprocess_output", "num_steps"}:
                continue
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                    not math.isfinite(float(value)) or float(value) < 0):
                raise ValueError(f"`generation_config.{name}` must be finite and non-negative.")
        if generation["time_shift"] <= 0:
            raise ValueError("`generation_config.time_shift` must be positive.")
        self.generation_config = dict(generation)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return super().to_dict()


__all__ = ["OmniVoiceConfig"]
