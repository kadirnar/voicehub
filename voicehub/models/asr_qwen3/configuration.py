"""Configuration for VoiceHub's native Qwen3-ASR provider."""

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


class Qwen3ASRConfig(VoiceHubConfig):
    """Configure native checkpoint loading, decoding, and fine-tuning.

    Compatibility options from the former Transformers wrapper are
    accepted only so callers receive explicit migration errors.
    Architecture code, tokenization, feature extraction, generation,
    loss, and checkpoint I/O are all implemented by VoiceHub.
    """

    model_type = "asr_qwen3"
    architecture_family = "speech-seq2seq"

    def __init__(
        self,
        *,
        architecture_family: str | None = None,
        config_name_or_path: str | Path | None = None,
        processor_name_or_path: str | Path | None = None,
        trust_remote_code: bool = False,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        use_safetensors: bool | None = None,
        model_kwargs: Mapping[str, Any] | None = None,
        processor_kwargs: Mapping[str, Any] | None = None,
        pipeline_kwargs: Mapping[str, Any] | None = None,
        torch_dtype: str = "auto",
        training_language: str | None = "English",
        sample_rate: int = 16_000,
        inference_config: ASRInferenceConfig | Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        reject_serialized_secrets(
            {
                "model_kwargs": model_kwargs,
                "processor_kwargs": processor_kwargs,
                "pipeline_kwargs": pipeline_kwargs,
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
        self.architecture_family = (
            self.__class__.architecture_family if architecture_family is None else architecture_family)
        self.config_name_or_path = config_name_or_path
        self.processor_name_or_path = processor_name_or_path
        self.trust_remote_code = trust_remote_code
        self.revision = revision
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self.use_safetensors = use_safetensors
        self.model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
        self.processor_kwargs = ({} if processor_kwargs is None else dict(processor_kwargs))
        self.pipeline_kwargs = ({} if pipeline_kwargs is None else dict(pipeline_kwargs))
        self.torch_dtype = torch_dtype
        self.training_language = training_language
        self.validate()

    def validate(self) -> None:
        reject_serialized_secrets(self.__dict__, owner=self.__class__.__name__)
        if not isinstance(self.architecture_family, str):
            raise TypeError("`architecture_family` must be a string.")
        family = self.architecture_family.strip().lower().replace("_", "-")
        if family != self.__class__.architecture_family:
            raise ValueError("Qwen3ASRConfig requires "
                             "`architecture_family='speech-seq2seq'`.")
        self.architecture_family = family
        if self.sample_rate != 16_000:
            raise ValueError(
                "Qwen3-ASR requires `sample_rate=16000`; VoiceHub resamples "
                "inputs at the native processor boundary.")
        if self.config_name_or_path is not None:
            raise ValueError(
                "Native Qwen3-ASR resolves `config.json` from the coherent "
                "checkpoint root; `config_name_or_path` is unsupported.")
        if self.processor_name_or_path is not None:
            raise ValueError(
                "Native Qwen3-ASR resolves all processor assets from the "
                "checkpoint root; `processor_name_or_path` is unsupported.")
        if not isinstance(self.trust_remote_code, bool):
            raise TypeError("`trust_remote_code` must be a boolean.")
        if self.trust_remote_code:
            raise ValueError(
                "Native Qwen3-ASR never executes repository code; "
                "`trust_remote_code=True` is unsupported.")
        if self.revision is not None:
            if not isinstance(self.revision, str) or not self.revision.strip():
                raise ValueError("`revision` must be a non-empty string or None.")
            self.revision = self.revision.strip()
        if self.cache_dir is not None:
            if not isinstance(self.cache_dir, (str, Path)):
                raise TypeError("`cache_dir` must be path-like or None.")
            self.cache_dir = str(Path(self.cache_dir).expanduser())
        if not isinstance(self.local_files_only, bool):
            raise TypeError("`local_files_only` must be a boolean.")
        if self.use_safetensors is False:
            raise ValueError(
                "Native Qwen3-ASR accepts Safetensors only; "
                "`use_safetensors=False` is unsupported.")
        if self.use_safetensors not in (None, True):
            raise TypeError("`use_safetensors` must be a boolean or None.")
        for name in ("model_kwargs", "processor_kwargs", "pipeline_kwargs"):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise TypeError(f"`{name}` must be a mapping.")
            value = dict(value)
            if value:
                options = ", ".join(sorted(str(key) for key in value))
                raise ValueError(
                    f"Native Qwen3-ASR does not delegate `{name}`; "
                    f"unsupported option(s): {options}.")
            setattr(self, name, value)
        if not isinstance(self.torch_dtype, str):
            raise TypeError("`torch_dtype` must be a string.")
        try:
            self.torch_dtype = _DTYPE_ALIASES[self.torch_dtype.strip().lower()]
        except KeyError as error:
            choices = ", ".join(sorted(set(_DTYPE_ALIASES.values())))
            raise ValueError(f"`torch_dtype` must be one of: {choices}.") from error
        if self.training_language is not None:
            from voicehub.models.asr_qwen3.native.languages import normalize_qwen3_asr_language

            self.training_language = normalize_qwen3_asr_language(self.training_language)
        inference_values = self.inference_config
        if isinstance(inference_values, ASRInferenceConfig):
            inference_values = inference_values.to_dict()
        if not isinstance(inference_values, Mapping):
            raise TypeError("`inference_config` must serialize to a mapping.")
        self.inference_config = ASRInferenceConfig.from_dict(dict(inference_values)).to_dict()

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return super().to_dict()


__all__ = ["Qwen3ASRConfig"]
