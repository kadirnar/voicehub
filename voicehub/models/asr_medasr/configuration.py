"""Configuration for VoiceHub's native Google MedASR provider."""

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


class MedASRASRConfig(VoiceHubConfig):
    """Configure gated artifact loading and native LASR CTC execution."""

    model_type = "asr_medasr"
    architecture_family = "ctc"

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
        checkpoint_filename: str | None = None,
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
            raise TypeError("`inference_config` must be an ASRInferenceConfig, mapping, "
                            "or None.")
        for name in ASRInferenceConfig._COMMON_FIELDS:
            if name in kwargs:
                inference_values[name] = kwargs.pop(name)
        super().__init__(
            sample_rate=sample_rate,
            inference_config=ASRInferenceConfig.from_dict(inference_values, ).to_dict(),
            **kwargs,
        )
        self.architecture_family = ("ctc" if architecture_family is None else architecture_family)
        self.config_name_or_path = config_name_or_path
        self.processor_name_or_path = processor_name_or_path
        self.trust_remote_code = trust_remote_code
        self.revision = revision
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self.use_safetensors = use_safetensors
        self.checkpoint_filename = checkpoint_filename
        self.torch_dtype = torch_dtype
        self.model_kwargs = ({} if model_kwargs is None else dict(model_kwargs))
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
        if (self.architecture_family.strip().lower().replace("_", "-") != "ctc"):
            raise ValueError("Native MedASR requires `architecture_family='ctc'`.")
        self.architecture_family = "ctc"
        if self.sample_rate != 16_000:
            raise ValueError("The released MedASR frontend requires 16 kHz audio.")
        for name in (
                "config_name_or_path",
                "processor_name_or_path",
        ):
            if getattr(self, name) is not None:
                raise ValueError(
                    "Native MedASR resolves one coherent artifact root; "
                    f"`{name}` is unsupported.")
        if not isinstance(self.trust_remote_code, bool):
            raise TypeError("`trust_remote_code` must be a boolean.")
        if self.trust_remote_code:
            raise ValueError("Native MedASR never executes repository code.")
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
            raise ValueError("Native MedASR accepts Safetensors only.")
        if self.use_safetensors not in (None, True):
            raise TypeError("`use_safetensors` must be a boolean or None.")
        if self.checkpoint_filename is not None:
            value = self.checkpoint_filename
            if (not isinstance(value, str) or not value or Path(value).name != value or
                    not value.endswith(".safetensors")):
                raise ValueError("`checkpoint_filename` must be a plain Safetensors "
                                 "filename or None.")
        if not isinstance(self.torch_dtype, str):
            raise TypeError("`torch_dtype` must be a string.")
        try:
            self.torch_dtype = _DTYPES[self.torch_dtype.strip().lower()]
        except KeyError as error:
            choices = ", ".join(sorted(set(_DTYPES.values())))
            raise ValueError(f"`torch_dtype` must be one of: {choices}.") from error
        for name in (
                "model_kwargs",
                "processor_kwargs",
                "pipeline_kwargs",
        ):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise TypeError(f"`{name}` must be a mapping.")
            if value:
                options = ", ".join(sorted(str(key) for key in value))
                raise ValueError(f"Native MedASR received unsupported `{name}` "
                                 f"option(s): {options}.")
            setattr(self, name, {})
        values = self.inference_config
        if isinstance(values, ASRInferenceConfig):
            values = values.to_dict()
        if not isinstance(values, Mapping):
            raise TypeError("`inference_config` must serialize to a mapping.")
        self.inference_config = ASRInferenceConfig.from_dict(dict(values), ).to_dict()

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return super().to_dict()


# Preserve the provider's established public class name while keeping the
# explicit ``ASR`` suffix available to distinguish it from the executable
# architecture configuration.
MedASRConfig = MedASRASRConfig

__all__ = [
    "MedASRASRConfig",
    "MedASRConfig",
]
