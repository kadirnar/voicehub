"""Configuration for VoiceHub's dependency-free Whisper ASR provider."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicehub.configuration import VoiceHubConfig, reject_serialized_secrets
from voicehub.inference_configuration import ASRInferenceConfig

_DTYPE_ALIASES = {
    "auto": "auto",
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "float": "float32",
    "float16": "float16",
    "float32": "float32",
    "fp16": "float16",
    "fp32": "float32",
    "half": "float16",
}


class WhisperASRConfig(VoiceHubConfig):
    """Configure native Whisper loading, decoding, and fine-tuning.

    Unknown official checkpoint fields are retained by
    :class:`VoiceHubConfig`; the internal :class:`WhisperConfig` consumes the
    architecture dimensions when the model is allocated.
    """

    model_type = "asr_whisper"
    architecture_family = "speech-seq2seq"

    def __init__(
        self,
        *,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        checkpoint_filename: str | None = None,
        tokenizer_filename: str = "tokenizer.json",
        torch_dtype: str = "auto",
        sample_rate: int = 16_000,
        inference_config: ASRInferenceConfig | Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        reject_serialized_secrets(
            {
                "inference_config": inference_config,
                **kwargs,
            },
            owner=self.__class__.__name__,
        )
        if isinstance(inference_config, ASRInferenceConfig):
            inference_values = inference_config.to_dict()
        elif inference_config is None:
            inference_values = {}
        elif isinstance(inference_config, Mapping):
            inference_values = dict(inference_config)
        else:
            raise TypeError("`inference_config` must be an ASRInferenceConfig, mapping, "
                            "or None.")
        for name in ASRInferenceConfig._COMMON_FIELDS:
            if name in kwargs:
                inference_values[name] = kwargs.pop(name)

        super().__init__(
            sample_rate=sample_rate,
            inference_config=ASRInferenceConfig.from_dict(inference_values).to_dict(),
            **kwargs,
        )
        self.revision = revision
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self.checkpoint_filename = checkpoint_filename
        self.tokenizer_filename = tokenizer_filename
        self.torch_dtype = torch_dtype
        self.validate()

    def validate(self) -> None:
        """Reject settings that change Whisper's trained input semantics."""
        reject_serialized_secrets(
            self.__dict__,
            owner=self.__class__.__name__,
        )
        if self.sample_rate != 16_000:
            raise ValueError(
                "Whisper checkpoints require `sample_rate=16000`; VoiceHub "
                "resamples public inputs at the processor boundary.")
        if self.revision is not None and (not isinstance(self.revision, str) or not self.revision.strip()):
            raise ValueError("`revision` must be a non-empty string or None.")
        if self.revision is not None:
            self.revision = self.revision.strip()
        if self.cache_dir is not None:
            if not isinstance(self.cache_dir, (str, Path)):
                raise TypeError("`cache_dir` must be path-like or None.")
            self.cache_dir = str(Path(self.cache_dir).expanduser())
        if not isinstance(self.local_files_only, bool):
            raise TypeError("`local_files_only` must be a boolean.")
        for name in ("checkpoint_filename", "tokenizer_filename"):
            value = getattr(self, name)
            if value is None and name == "checkpoint_filename":
                continue
            if not isinstance(value, str) or not value.strip():
                suffix = " or None" if name == "checkpoint_filename" else ""
                raise ValueError(f"`{name}` must be a non-empty string{suffix}.")
            setattr(self, name, value.strip())
        if not isinstance(self.torch_dtype, str):
            raise TypeError("`torch_dtype` must be a string.")
        normalized_dtype = self.torch_dtype.strip().lower()
        try:
            self.torch_dtype = _DTYPE_ALIASES[normalized_dtype]
        except KeyError as error:
            choices = ", ".join(sorted(set(_DTYPE_ALIASES.values())))
            raise ValueError(f"`torch_dtype` must be one of: {choices}.") from error
        values = getattr(self, "inference_config", {})
        if isinstance(values, ASRInferenceConfig):
            values = values.to_dict()
        if not isinstance(values, Mapping):
            raise TypeError("`inference_config` must serialize to a mapping.")
        self.inference_config = ASRInferenceConfig.from_dict(dict(values)).to_dict()

    def to_dict(self) -> dict[str, Any]:
        """Validate mutable fields before serialization."""
        self.validate()
        return super().to_dict()


__all__ = ["WhisperASRConfig"]
