"""Public configuration for VoiceHub-native Nemotron 3.5 ASR."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicehub.configuration import VoiceHubConfig, reject_serialized_secrets
from voicehub.inference_configuration import ASRInferenceConfig

_DTYPES = {
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
_PUBLISHED_LOOKAHEADS = frozenset({0, 3, 6, 13})


class NemotronASRConfig(VoiceHubConfig):
    """Configure safe native loading, prompt conditioning, and training."""

    model_type = "asr_nemotron"
    architecture_family = "rnnt"

    def __init__(
        self,
        *,
        target_language: str = "auto",
        num_lookahead_tokens: int | None = None,
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
        training_language: str | None = None,
        sample_rate: int = 16_000,
        inference_config: (ASRInferenceConfig
                           | Mapping[str, Any]
                           | None) = None,
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
            raise TypeError("`inference_config` must be an ASRInferenceConfig, "
                            "mapping, or None.")
        for name in ASRInferenceConfig._COMMON_FIELDS:
            if name in kwargs:
                inference_values[name] = kwargs.pop(name)
        super().__init__(
            sample_rate=sample_rate,
            inference_config=ASRInferenceConfig.from_dict(inference_values, ).to_dict(),
            **kwargs,
        )
        self.target_language = target_language
        self.num_lookahead_tokens = num_lookahead_tokens
        self.architecture_family = (
            self.__class__.architecture_family if architecture_family is None else architecture_family)
        self.config_name_or_path = config_name_or_path
        self.processor_name_or_path = processor_name_or_path
        self.trust_remote_code = trust_remote_code
        self.revision = revision
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self.use_safetensors = use_safetensors
        self.model_kwargs = ({} if model_kwargs is None else dict(model_kwargs))
        self.processor_kwargs = ({} if processor_kwargs is None else dict(processor_kwargs))
        self.pipeline_kwargs = ({} if pipeline_kwargs is None else dict(pipeline_kwargs))
        self.torch_dtype = torch_dtype
        self.training_language = training_language
        self.validate()

    def validate(self) -> None:
        reject_serialized_secrets(
            self.__dict__,
            owner=self.__class__.__name__,
        )
        if (not isinstance(self.target_language, str) or not self.target_language.strip()):
            raise ValueError("`target_language` must be a non-empty Nemotron prompt.")
        self.target_language = self.target_language.strip()
        if self.num_lookahead_tokens is not None and (isinstance(self.num_lookahead_tokens, bool) or
                                                      not isinstance(self.num_lookahead_tokens, int) or
                                                      self.num_lookahead_tokens not in _PUBLISHED_LOOKAHEADS):
            raise ValueError("`num_lookahead_tokens` must be one of 0, 3, 6, 13, "
                             "or None.")
        if not isinstance(self.architecture_family, str):
            raise TypeError("`architecture_family` must be a string.")
        family = (self.architecture_family.strip().lower().replace("_", "-"))
        if family not in {"rnnt", "rnn-t"}:
            raise ValueError("NemotronASRConfig requires "
                             "`architecture_family='rnnt'`.")
        self.architecture_family = "rnnt"
        if self.sample_rate != 16_000:
            raise ValueError("Nemotron 3.5 ASR requires `sample_rate=16000`.")
        for name in (
                "config_name_or_path",
                "processor_name_or_path",
        ):
            if getattr(self, name) is not None:
                raise ValueError(
                    "Native Nemotron resolves one coherent artifact root; "
                    f"`{name}` is unsupported.")
        if not isinstance(self.trust_remote_code, bool):
            raise TypeError("`trust_remote_code` must be a boolean.")
        if self.trust_remote_code:
            raise ValueError("Native Nemotron never executes repository code.")
        if self.revision is not None:
            if (not isinstance(self.revision, str) or not self.revision.strip()):
                raise ValueError("`revision` must be a non-empty string or None.")
            self.revision = self.revision.strip()
        if self.cache_dir is not None:
            if not isinstance(self.cache_dir, (str, Path)):
                raise TypeError("`cache_dir` must be path-like or None.")
            self.cache_dir = str(Path(self.cache_dir).expanduser())
        if not isinstance(self.local_files_only, bool):
            raise TypeError("`local_files_only` must be a boolean.")
        if self.use_safetensors is False:
            raise ValueError("Native Nemotron accepts Safetensors only.")
        if self.use_safetensors not in (None, True):
            raise TypeError("`use_safetensors` must be a boolean or None.")
        for name in (
                "model_kwargs",
                "processor_kwargs",
                "pipeline_kwargs",
        ):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise TypeError(f"`{name}` must be a mapping.")
            normalized = dict(value)
            if normalized:
                options = ", ".join(sorted(str(key) for key in normalized), )
                raise ValueError(
                    f"Native Nemotron does not delegate `{name}`; "
                    f"unsupported options: {options}.")
            setattr(self, name, normalized)
        if not isinstance(self.torch_dtype, str):
            raise TypeError("`torch_dtype` must be a string.")
        try:
            self.torch_dtype = _DTYPES[self.torch_dtype.strip().lower()]
        except KeyError as error:
            choices = ", ".join(sorted(set(_DTYPES.values())))
            raise ValueError(f"`torch_dtype` must be one of: {choices}.") from error
        if self.training_language is not None:
            if (not isinstance(self.training_language, str) or not self.training_language.strip()):
                raise ValueError("`training_language` must be a non-empty prompt or None.")
            self.training_language = self.training_language.strip()
        values = self.inference_config
        if isinstance(values, ASRInferenceConfig):
            values = values.to_dict()
        if not isinstance(values, Mapping):
            raise TypeError("`inference_config` must serialize to a mapping.")
        self.inference_config = ASRInferenceConfig.from_dict(dict(values), ).to_dict()

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return super().to_dict()


__all__ = ["NemotronASRConfig"]
