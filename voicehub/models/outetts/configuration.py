"""Configuration for the VoiceHub-native OuteTTS 1.0 runtime."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicehub.configuration import VoiceHubConfig


class OuteTTSConfig(VoiceHubConfig):
    """Serializable model, codec, and immutable artifact settings."""

    model_type = "outetts"

    def __init__(
        self,
        *,
        tokenizer_path: str | Path | None = None,
        backend: str = "HF",
        interface_version: str = "V3",
        max_seq_length: int | None = None,
        additional_model_config: Mapping[str, Any] | None = None,
        revision: str | None = None,
        codec_name_or_path: str | Path | None = None,
        codec_revision: str | None = None,
        cache_dir: str | None = None,
        local_files_only: bool = False,
        torch_dtype: str = "auto",
        sample_rate: int = 24_000,
        **kwargs,
    ) -> None:
        if tokenizer_path is not None and (not isinstance(tokenizer_path,
                                                          (str, Path)) or not str(tokenizer_path).strip()):
            raise ValueError("`tokenizer_path` must be a non-empty path/Hub ID or None.")
        if not isinstance(backend, str) or not backend.strip():
            raise ValueError("`backend` must be a non-empty string.")
        if (not isinstance(interface_version, str) or not interface_version.strip()):
            raise ValueError("`interface_version` must be a non-empty string.")
        if max_seq_length is not None and (isinstance(max_seq_length, bool) or
                                           not isinstance(max_seq_length, int) or max_seq_length < 2):
            raise ValueError("`max_seq_length` must be an integer of at least two or None.")
        if (additional_model_config is not None and not isinstance(additional_model_config, Mapping)):
            raise TypeError("`additional_model_config` must be a mapping.")
        for name, value in (
            ("revision", revision),
            ("codec_revision", codec_revision),
            ("cache_dir", cache_dir),
            ("torch_dtype", torch_dtype),
        ):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"`{name}` must be a non-empty string or None.")
        if codec_name_or_path is not None and (not isinstance(codec_name_or_path, (str, Path)) or
                                               not str(codec_name_or_path).strip()):
            raise ValueError("`codec_name_or_path` must be a non-empty path/Hub ID or None.")
        if not isinstance(local_files_only, bool):
            raise TypeError("`local_files_only` must be a boolean.")
        if sample_rate != 24_000:
            raise ValueError("OuteTTS V3 uses the audited 24 kHz DAC; "
                             "`sample_rate` must be 24000.")
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.tokenizer_path = (None if tokenizer_path is None else str(tokenizer_path))
        self.backend = backend
        self.interface_version = interface_version
        self.max_seq_length = max_seq_length
        self.additional_model_config = dict(additional_model_config or {})
        self.revision = revision
        self.codec_name_or_path = (None if codec_name_or_path is None else str(codec_name_or_path))
        self.codec_revision = codec_revision
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self.torch_dtype = torch_dtype


__all__ = ["OuteTTSConfig"]
