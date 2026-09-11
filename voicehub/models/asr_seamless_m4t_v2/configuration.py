"""Public configuration for VoiceHub-native SeamlessM4T-v2 ASR."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicehub.configuration import VoiceHubConfig, reject_serialized_secrets
from voicehub.inference_configuration import ASRInferenceConfig
from voicehub.models.asr_seamless_m4t_v2.native.languages import SEAMLESS_M4T_V2_LANGUAGE_TO_ID

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


class SeamlessM4Tv2ASRConfig(VoiceHubConfig):
    """Configure multilingual speech recognition and full-model fine-tuning."""

    model_type = "asr_seamless_m4t_v2"
    architecture_family = "speech-seq2seq"

    def __init__(
        self,
        *,
        architecture_family: str | None = None,
        target_language: str = "eng",
        config_name_or_path: str | Path | None = None,
        processor_name_or_path: str | Path | None = None,
        trust_remote_code: bool = False,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        use_safetensors: bool | None = None,
        torch_dtype: str = "auto",
        sample_rate: int = 16_000,
        model_kwargs: Mapping[str, Any] | None = None,
        processor_kwargs: Mapping[str, Any] | None = None,
        pipeline_kwargs: Mapping[str, Any] | None = None,
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
            raise TypeError("`inference_config` must be ASRInferenceConfig, a mapping, "
                            "or None.")
        for name in ASRInferenceConfig._COMMON_FIELDS:
            if name in kwargs:
                inference_values[name] = kwargs.pop(name)
        super().__init__(
            sample_rate=sample_rate,
            inference_config=ASRInferenceConfig.from_dict(inference_values).to_dict(),
            **kwargs,
        )
        self.architecture_family = ("speech-seq2seq" if architecture_family is None else architecture_family)
        self.target_language = target_language
        self.config_name_or_path = config_name_or_path
        self.processor_name_or_path = processor_name_or_path
        self.trust_remote_code = trust_remote_code
        self.revision = revision
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self.use_safetensors = use_safetensors
        self.torch_dtype = torch_dtype
        self.model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
        self.processor_kwargs = ({} if processor_kwargs is None else dict(processor_kwargs))
        self.pipeline_kwargs = ({} if pipeline_kwargs is None else dict(pipeline_kwargs))
        self.validate()

    def validate(self) -> None:
        reject_serialized_secrets(
            self.__dict__,
            owner=self.__class__.__name__,
        )
        if not isinstance(self.architecture_family, str):
            raise TypeError("`architecture_family` must be a string.")
        if (self.architecture_family.strip().lower().replace("_", "-") != "speech-seq2seq"):
            raise ValueError("Native SeamlessM4T-v2 requires "
                             "`architecture_family='speech-seq2seq'`.")
        self.architecture_family = "speech-seq2seq"
        if self.sample_rate != 16_000:
            raise ValueError("The published SeamlessM4T-v2 frontend requires 16 kHz audio.")
        if not isinstance(self.target_language, str) or not self.target_language.strip():
            raise ValueError("`target_language` must be a non-empty code.")
        language = self.target_language.strip()
        if language.startswith("__") and language.endswith("__"):
            language = language[2:-2]
        language = ("cmn_Hant" if language.lower() == "cmn_hant" else language.lower())
        if language not in SEAMLESS_M4T_V2_LANGUAGE_TO_ID:
            raise ValueError(f"Unsupported SeamlessM4T-v2 target language {language!r}.")
        self.target_language = language
        for name in ("config_name_or_path", "processor_name_or_path"):
            if getattr(self, name) is not None:
                raise ValueError(
                    "Native SeamlessM4T-v2 resolves one coherent artifact "
                    f"root; `{name}` is unsupported.")
        if not isinstance(self.trust_remote_code, bool):
            raise TypeError("`trust_remote_code` must be a boolean.")
        if self.trust_remote_code:
            raise ValueError("Native SeamlessM4T-v2 never executes remote code.")
        if self.revision is not None:
            if not isinstance(self.revision, str) or not self.revision.strip():
                raise ValueError("`revision` must be non-empty or None.")
            self.revision = self.revision.strip()
        if self.cache_dir is not None:
            if not isinstance(self.cache_dir, (str, Path)):
                raise TypeError("`cache_dir` must be path-like or None.")
            self.cache_dir = str(Path(self.cache_dir).expanduser())
        if not isinstance(self.local_files_only, bool):
            raise TypeError("`local_files_only` must be a boolean.")
        if self.use_safetensors is False:
            raise ValueError("Native SeamlessM4T-v2 accepts Safetensors only.")
        if self.use_safetensors not in (None, True):
            raise TypeError("`use_safetensors` must be a boolean or None.")
        if not isinstance(self.torch_dtype, str):
            raise TypeError("`torch_dtype` must be a string.")
        try:
            self.torch_dtype = _DTYPES[self.torch_dtype.strip().lower()]
        except KeyError as error:
            raise ValueError("`torch_dtype` must be auto, float32, float16, or bfloat16.") from error
        for name in (
                "model_kwargs",
                "processor_kwargs",
                "pipeline_kwargs",
        ):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise TypeError(f"`{name}` must be a mapping.")
            if value:
                raise ValueError(
                    f"Native SeamlessM4T-v2 received unsupported `{name}` "
                    f"option(s): {', '.join(sorted(value))}.")
            setattr(self, name, {})
        values = self.inference_config
        if isinstance(values, ASRInferenceConfig):
            values = values.to_dict()
        if not isinstance(values, Mapping):
            raise TypeError("`inference_config` must serialize to a mapping.")
        self.inference_config = ASRInferenceConfig.from_dict(dict(values)).to_dict()

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return super().to_dict()


__all__ = ["SeamlessM4Tv2ASRConfig"]
