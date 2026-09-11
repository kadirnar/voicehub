"""Serializable configuration for the speecht5 model family."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
from pathlib import Path
from typing import Any

from voicehub.configuration import VoiceHubConfig, reject_serialized_secrets
from voicehub.models.speecht5.metadata import SPEECHT5_HIFIGAN_REPOSITORY


class SpeechT5Config(VoiceHubConfig):
    """Serializable loading controls for the native SpeechT5 runtime."""

    model_type = "speecht5"

    def __init__(
        self,
        *,
        config_name_or_path: str | Path | None = None,
        processor_name_or_path: str | Path | None = None,
        trust_remote_code: bool = False,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        use_safetensors: bool | None = None,
        torch_dtype: str | None = None,
        model_kwargs: Mapping[str, Any] | None = None,
        processor_kwargs: Mapping[str, Any] | None = None,
        vocoder_name_or_path: str | Path = SPEECHT5_HIFIGAN_REPOSITORY,
        vocoder_revision: str | None = None,
        vocoder_kwargs: Mapping[str, Any] | None = None,
        default_speaker_embedding_path: str | Path | None = None,
        verify_official_integrity: bool = True,
        native_model_config: Mapping[str, Any] | None = None,
        native_vocoder_config: Mapping[str, Any] | None = None,
        sample_rate: int = 16_000,
        **kwargs: Any,
    ) -> None:
        reject_serialized_secrets(
            {
                "model_kwargs": model_kwargs,
                "processor_kwargs": processor_kwargs,
                "vocoder_kwargs": vocoder_kwargs,
                **kwargs,
            },
            owner=self.__class__.__name__,
        )
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.config_name_or_path = config_name_or_path
        self.processor_name_or_path = processor_name_or_path
        self.trust_remote_code = trust_remote_code
        self.revision = revision
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self.use_safetensors = use_safetensors
        self.torch_dtype = torch_dtype
        self.model_kwargs = self._copy_mapping(
            model_kwargs,
            name="model_kwargs",
        )
        self.processor_kwargs = self._copy_mapping(
            processor_kwargs,
            name="processor_kwargs",
        )
        self.vocoder_name_or_path = vocoder_name_or_path
        self.vocoder_revision = vocoder_revision
        self.vocoder_kwargs = self._copy_mapping(
            vocoder_kwargs,
            name="vocoder_kwargs",
        )
        self.default_speaker_embedding_path = default_speaker_embedding_path
        self.verify_official_integrity = verify_official_integrity
        self.native_model_config = self._copy_optional_mapping(
            native_model_config,
            name="native_model_config",
        )
        self.native_vocoder_config = self._copy_optional_mapping(
            native_vocoder_config,
            name="native_vocoder_config",
        )
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
    def _copy_optional_mapping(
        value: Mapping[str, Any] | None,
        *,
        name: str,
    ) -> dict[str, Any] | None:
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise TypeError(f"`{name}` must be a mapping or None.")
        return dict(value)

    @staticmethod
    def _optional_string(value: Any, *, name: str) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"`{name}` must be a non-empty string or None.")
        return value.strip()

    def validate(self) -> None:
        reject_serialized_secrets(self.__dict__, owner=self.__class__.__name__)
        if self.trust_remote_code is not False:
            raise ValueError(
                "Native SpeechT5 never executes remote architecture code; "
                "`trust_remote_code` must be False.")
        for option in ("config_name_or_path", "processor_name_or_path"):
            value = getattr(self, option)
            if value is not None:
                if not isinstance(value, (str, Path)) or not str(value).strip():
                    raise ValueError(f"`{option}` must be a non-empty path or None.")
                setattr(self, option, str(value))
        # Separate config/processor sources allowed the previous provider
        # runtime to mix revisions. Native resolution treats the model bundle
        # as one atomic artifact and therefore rejects that ambiguity.
        if self.config_name_or_path is not None:
            raise ValueError(
                "Native SpeechT5 resolves config and weights atomically; "
                "`config_name_or_path` must be None.")
        if self.processor_name_or_path is not None:
            raise ValueError(
                "Native SpeechT5 resolves processor assets atomically; "
                "`processor_name_or_path` must be None.")
        self.revision = self._optional_string(self.revision, name="revision")
        self.vocoder_revision = self._optional_string(
            self.vocoder_revision,
            name="vocoder_revision",
        )
        if self.cache_dir is not None:
            if not isinstance(self.cache_dir, (str, Path)):
                raise TypeError("`cache_dir` must be path-like or None.")
            self.cache_dir = str(self.cache_dir)
        if not isinstance(self.local_files_only, bool):
            raise TypeError("`local_files_only` must be a boolean.")
        if self.use_safetensors is not None and not isinstance(
                self.use_safetensors,
                bool,
        ):
            raise TypeError("`use_safetensors` must be a boolean or None.")
        self.torch_dtype = self._optional_string(
            self.torch_dtype,
            name="torch_dtype",
        )
        for name in ("model_kwargs", "processor_kwargs", "vocoder_kwargs"):
            values = self._copy_mapping(getattr(self, name), name=name)
            if values:
                raise ValueError(
                    f"`{name}` cannot inject provider-runtime options into "
                    "VoiceHub-native SpeechT5.")
            setattr(self, name, values)
        source = self.vocoder_name_or_path
        if not isinstance(source, (str, Path)) or not str(source).strip():
            raise ValueError("`vocoder_name_or_path` must be a non-empty path or Hub ID.")
        self.vocoder_name_or_path = str(source)
        speaker_path = self.default_speaker_embedding_path
        if speaker_path is not None:
            if (not isinstance(speaker_path, (str, Path)) or not str(speaker_path).strip()):
                raise ValueError("`default_speaker_embedding_path` must be a non-empty "
                                 "path or None.")
            self.default_speaker_embedding_path = str(speaker_path)
        if not isinstance(self.verify_official_integrity, bool):
            raise TypeError("`verify_official_integrity` must be a boolean.")
        if (isinstance(self.sample_rate, bool) or not isinstance(self.sample_rate, Integral) or
                self.sample_rate <= 0):
            raise ValueError("`sample_rate` must be a positive integer.")
        self.sample_rate = int(self.sample_rate)
        self.native_model_config = self._copy_optional_mapping(
            self.native_model_config,
            name="native_model_config",
        )
        self.native_vocoder_config = self._copy_optional_mapping(
            self.native_vocoder_config,
            name="native_vocoder_config",
        )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return super().to_dict()


__all__ = ['SpeechT5Config']
