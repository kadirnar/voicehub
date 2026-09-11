"""Configuration for VoiceHub's native checkpoint-dispatching ASR provider."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path, PurePosixPath
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


class TransformersASRConfig(VoiceHubConfig):
    """Configure native ASR dispatch for standard checkpoint layouts.

    The historical model key is retained for API compatibility.  Despite
    its name, this provider does not import or execute Transformers.  It
    reads the checkpoint's declarative ``config.json`` and selects one
    of VoiceHub's verified native Whisper, Wav2Vec2, HuBERT, WavLM, or
    Moonshine runtimes.

    Historical delegation controls remain in the signature so existing
    configuration files fail with precise migration errors instead of
    being silently ignored. The public ``asr_transformers`` key always
    enforces the native-only contract.
    """

    model_type = "asr_transformers"
    supported_architecture_families = frozenset({
        "auto",
        "ctc",
        "speech-seq2seq",
    })

    def __init__(
        self,
        *,
        architecture_family: str = "auto",
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
        checkpoint_filename: str | None = None,
        tokenizer_filename: str = "tokenizer.json",
        vocabulary_filename: str = "vocab.json",
        target_language: str | None = None,
        torch_dtype: str = "auto",
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
        self.architecture_family = architecture_family
        self.config_name_or_path = config_name_or_path
        self.processor_name_or_path = processor_name_or_path
        self.trust_remote_code = trust_remote_code
        self.revision = revision
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self.use_safetensors = use_safetensors
        self.model_kwargs = self._copy_mapping(
            model_kwargs,
            name="model_kwargs",
        )
        self.processor_kwargs = self._copy_mapping(
            processor_kwargs,
            name="processor_kwargs",
        )
        self.pipeline_kwargs = self._copy_mapping(
            pipeline_kwargs,
            name="pipeline_kwargs",
        )
        self.checkpoint_filename = checkpoint_filename
        self.tokenizer_filename = tokenizer_filename
        self.vocabulary_filename = vocabulary_filename
        if target_language is not None or not hasattr(self, "target_language"):
            self.target_language = target_language
        self.torch_dtype = torch_dtype
        self.validate()

    @staticmethod
    def _copy_mapping(
        value: Mapping[str, Any] | None,
        *,
        name: str,
    ) -> dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise TypeError(f"`{name}` must be a mapping or None.")
        return dict(value)

    @staticmethod
    def _safe_filename(
        value: str | None,
        *,
        name: str,
        optional: bool,
    ) -> str | None:
        if value is None and optional:
            return None
        if not isinstance(value, str) or not value.strip():
            suffix = " or None" if optional else ""
            raise ValueError(f"`{name}` must be a non-empty string{suffix}.")
        normalized = value.strip()
        path = PurePosixPath(normalized)
        if ("\\" in normalized or path.is_absolute() or len(path.parts) != 1 or ".." in path.parts):
            raise ValueError(f"`{name}` must be one safe checkpoint-root filename.")
        return normalized

    def _validate_native_dispatch_controls(self) -> None:
        if self.config_name_or_path is not None:
            raise ValueError(
                "Native ASR dispatch resolves `config.json` from the coherent "
                "checkpoint root; `config_name_or_path` is unsupported.")
        if self.processor_name_or_path is not None:
            raise ValueError(
                "Native ASR dispatch resolves processor assets from the "
                "coherent checkpoint root; `processor_name_or_path` is "
                "unsupported.")
        if self.trust_remote_code:
            raise ValueError(
                "Native ASR dispatch never executes repository code; "
                "`trust_remote_code=True` is unsupported.")
        if self.use_safetensors is False:
            raise ValueError(
                "Native ASR dispatch accepts Safetensors only; "
                "`use_safetensors=False` is unsupported.")
        for name in ("model_kwargs", "processor_kwargs", "pipeline_kwargs"):
            value = getattr(self, name)
            if value:
                options = ", ".join(sorted(str(key) for key in value))
                raise ValueError(
                    "Native ASR dispatch does not delegate "
                    f"`{name}`; unsupported option(s): {options}.")

    def validate(self) -> None:
        """Validate the native provider contract and serialized controls."""
        reject_serialized_secrets(
            self.__dict__,
            owner=self.__class__.__name__,
        )
        if not isinstance(self.architecture_family, str):
            raise TypeError("`architecture_family` must be a string.")
        self.architecture_family = (self.architecture_family.strip().lower().replace("_", "-"))
        allowed_families = self.supported_architecture_families
        if self.architecture_family == "audio-text-to-text":
            raise ValueError(
                "`audio-text-to-text` checkpoints require a dedicated "
                "VoiceHub provider with the model's prompt and causal-label "
                "contract.")
        if self.architecture_family not in allowed_families:
            supported = ", ".join(sorted(allowed_families))
            raise ValueError(
                "`architecture_family` must be one of: "
                f"{supported}; received {self.architecture_family!r}.")
        for option_name in ("config_name_or_path", "processor_name_or_path"):
            value = getattr(self, option_name)
            if value is None:
                continue
            if not isinstance(value, (str, Path)) or not str(value).strip():
                raise ValueError(f"`{option_name}` must be a non-empty path or Hub ID.")
            setattr(self, option_name, str(value))
        if not isinstance(self.trust_remote_code, bool):
            raise TypeError("`trust_remote_code` must be a boolean.")
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
        if self.use_safetensors not in (None, True, False):
            raise TypeError("`use_safetensors` must be a boolean or None.")
        if (isinstance(self.sample_rate, bool) or not isinstance(self.sample_rate, int) or
                self.sample_rate <= 0):
            raise ValueError("`sample_rate` must be a positive integer.")
        if self.sample_rate != 16_000:
            raise ValueError(
                "The currently verified native ASR families require "
                "`sample_rate=16000`; public inputs are resampled at the "
                "processor boundary.")

        self.model_kwargs = self._copy_mapping(
            self.model_kwargs,
            name="model_kwargs",
        )
        self.processor_kwargs = self._copy_mapping(
            self.processor_kwargs,
            name="processor_kwargs",
        )
        self.pipeline_kwargs = self._copy_mapping(
            self.pipeline_kwargs,
            name="pipeline_kwargs",
        )
        reserved_model_options = {"config", "state_dict", "trust_remote_code"}
        conflicts = reserved_model_options.intersection(self.model_kwargs)
        if conflicts:
            names = ", ".join(sorted(conflicts))
            raise ValueError("`model_kwargs` cannot override provider-owned option(s): "
                             f"{names}.")
        if "trust_remote_code" in self.processor_kwargs:
            raise ValueError("`processor_kwargs` cannot override `trust_remote_code`.")
        reserved_pipeline_options = {
            "device",
            "feature_extractor",
            "model",
            "processor",
            "task",
            "tokenizer",
        }
        conflicts = reserved_pipeline_options.intersection(self.pipeline_kwargs)
        if conflicts:
            names = ", ".join(sorted(conflicts))
            raise ValueError("`pipeline_kwargs` cannot override provider-owned option(s): "
                             f"{names}.")

        self.checkpoint_filename = self._safe_filename(
            self.checkpoint_filename,
            name="checkpoint_filename",
            optional=True,
        )
        self.tokenizer_filename = self._safe_filename(
            self.tokenizer_filename,
            name="tokenizer_filename",
            optional=False,
        )
        self.vocabulary_filename = self._safe_filename(
            self.vocabulary_filename,
            name="vocabulary_filename",
            optional=False,
        )
        if self.target_language is not None:
            if (not isinstance(self.target_language, str) or not self.target_language.strip()):
                raise ValueError("`target_language` must be a non-empty string or None.")
            self.target_language = self.target_language.strip()
        if not isinstance(self.torch_dtype, str):
            raise TypeError("`torch_dtype` must be a string.")
        try:
            self.torch_dtype = _DTYPE_ALIASES[self.torch_dtype.strip().lower()]
        except KeyError as error:
            choices = ", ".join(sorted(set(_DTYPE_ALIASES.values())))
            raise ValueError(f"`torch_dtype` must be one of: {choices}.") from error

        inference_values = self.inference_config
        if isinstance(inference_values, ASRInferenceConfig):
            inference_values = inference_values.to_dict()
        if not isinstance(inference_values, Mapping):
            raise TypeError("`inference_config` must serialize to a mapping.")
        self.inference_config = ASRInferenceConfig.from_dict(dict(inference_values)).to_dict()
        self._validate_native_dispatch_controls()

    def to_dict(self) -> dict[str, Any]:
        """Validate mutable overrides before serialization."""
        self.validate()
        return super().to_dict()


__all__ = ["TransformersASRConfig"]
