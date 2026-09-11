"""Serializable configuration for the bark model family."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
from pathlib import Path
from typing import Any

from voicehub.configuration import VoiceHubConfig, reject_serialized_secrets


class BarkConfig(VoiceHubConfig):
    """Serializable controls for the provider-free Bark runtime."""

    model_type = "bark"

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
        verify_official_integrity: bool = True,
        native_model_config: Mapping[str, Any] | None = None,
        native_generation_config: Mapping[str, Any] | None = None,
        sample_rate: int = 24_000,
        **kwargs: Any,
    ) -> None:
        reject_serialized_secrets(
            {
                "model_kwargs": model_kwargs,
                "processor_kwargs": processor_kwargs,
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
        self.model_kwargs = self._mapping(model_kwargs, name="model_kwargs")
        self.processor_kwargs = self._mapping(
            processor_kwargs,
            name="processor_kwargs",
        )
        self.verify_official_integrity = verify_official_integrity
        self.native_model_config = self._optional_mapping(
            native_model_config,
            name="native_model_config",
        )
        self.native_generation_config = self._optional_mapping(
            native_generation_config,
            name="native_generation_config",
        )
        self.validate()

    @staticmethod
    def _mapping(
        value: Mapping[str, Any] | None,
        *,
        name: str,
    ) -> dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise TypeError(f"`{name}` must be a mapping or None.")
        return dict(value)

    @classmethod
    def _optional_mapping(
        cls,
        value: Mapping[str, Any] | None,
        *,
        name: str,
    ) -> dict[str, Any] | None:
        return None if value is None else cls._mapping(value, name=name)

    def validate(self) -> None:
        reject_serialized_secrets(self.__dict__, owner=self.__class__.__name__)
        if self.trust_remote_code is not False:
            raise ValueError("Native Bark never executes remote code; "
                             "`trust_remote_code` must be False.")
        for name in ("config_name_or_path", "processor_name_or_path"):
            value = getattr(self, name)
            if value is not None:
                if not isinstance(value, (str, Path)) or not str(value).strip():
                    raise ValueError(f"`{name}` must be a non-empty path or None.")
                setattr(self, name, str(value))
        if self.revision is not None:
            if not isinstance(self.revision, str) or not self.revision.strip():
                raise ValueError("`revision` must be a non-empty string or None.")
            self.revision = self.revision.strip()
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
        if self.torch_dtype is not None:
            if not isinstance(self.torch_dtype, str) or not self.torch_dtype.strip():
                raise ValueError("`torch_dtype` must be a non-empty string or None.")
            self.torch_dtype = self.torch_dtype.strip()
        self.model_kwargs = self._mapping(
            self.model_kwargs,
            name="model_kwargs",
        )
        self.processor_kwargs = self._mapping(
            self.processor_kwargs,
            name="processor_kwargs",
        )
        if self.model_kwargs:
            names = ", ".join(sorted(self.model_kwargs))
            raise ValueError("Native Bark rejects provider-owned `model_kwargs`; "
                             f"received: {names}.")
        if self.processor_kwargs:
            names = ", ".join(sorted(self.processor_kwargs))
            raise ValueError("Native Bark rejects provider-owned `processor_kwargs`; "
                             f"received: {names}.")
        if not isinstance(self.verify_official_integrity, bool):
            raise TypeError("`verify_official_integrity` must be a boolean.")
        if (isinstance(self.sample_rate, bool) or not isinstance(self.sample_rate, Integral) or
                self.sample_rate <= 0):
            raise ValueError("`sample_rate` must be greater than zero.")
        self.sample_rate = int(self.sample_rate)
        self.native_model_config = self._optional_mapping(
            self.native_model_config,
            name="native_model_config",
        )
        self.native_generation_config = self._optional_mapping(
            self.native_generation_config,
            name="native_generation_config",
        )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return super().to_dict()


__all__ = ['BarkConfig']
