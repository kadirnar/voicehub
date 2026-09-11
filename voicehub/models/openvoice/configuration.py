"""Public configuration for VoiceHub-native OpenVoice V2."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from voicehub.configuration import VoiceHubConfig


class OpenVoiceConfig(VoiceHubConfig):
    """Configure the converter, native base speech, and fine-tuning
    boundary."""

    model_type = "openvoice"
    architecture_family = "tone-color-converter"

    def __init__(
        self,
        *,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        trust_pickle_checkpoint: bool = False,
        dtype: str = "float32",
        base_model_name_or_path: str | Path | None = None,
        reference_segment_seconds: float = 10.0,
        watermark: str | None = None,
        enable_reconstructed_finetuning: bool = False,
        sample_rate: int = 22_050,
        **kwargs: Any,
    ) -> None:
        if revision is not None and (not isinstance(revision, str) or not revision.strip()):
            raise ValueError("`revision` must be non-empty or None.")
        if cache_dir is not None and not isinstance(cache_dir, (str, Path)):
            raise TypeError("`cache_dir` must be path-like or None.")
        if not isinstance(local_files_only, bool):
            raise TypeError("`local_files_only` must be a boolean.")
        if not isinstance(trust_pickle_checkpoint, bool):
            raise TypeError("`trust_pickle_checkpoint` must be a boolean.")
        if not isinstance(dtype, str) or not dtype.strip():
            raise ValueError("`dtype` must be a non-empty string.")
        if base_model_name_or_path is not None and (not isinstance(base_model_name_or_path, (str, Path)) or
                                                    not str(base_model_name_or_path).strip()):
            raise ValueError("`base_model_name_or_path` must be a non-empty path/ID or None.")
        if (isinstance(reference_segment_seconds, bool) or not isinstance(reference_segment_seconds,
                                                                          (int, float)) or
                not math.isfinite(float(reference_segment_seconds)) or reference_segment_seconds <= 0):
            raise ValueError("`reference_segment_seconds` must be finite and positive.")
        if watermark is not None and (not isinstance(watermark, str) or not watermark):
            raise ValueError("`watermark` must be non-empty or None.")
        if not isinstance(enable_reconstructed_finetuning, bool):
            raise TypeError("`enable_reconstructed_finetuning` must be a boolean.")
        if sample_rate != 22_050:
            raise ValueError("OpenVoice V2 converter audio must use 22,050 Hz.")
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.revision = None if revision is None else revision.strip()
        self.cache_dir = (None if cache_dir is None else str(Path(cache_dir).expanduser()))
        self.local_files_only = local_files_only
        self.trust_pickle_checkpoint = trust_pickle_checkpoint
        self.dtype = dtype.strip().lower()
        self.base_model_name_or_path = (
            None if base_model_name_or_path is None else str(base_model_name_or_path))
        self.reference_segment_seconds = float(reference_segment_seconds)
        self.watermark = watermark
        self.enable_reconstructed_finetuning = (enable_reconstructed_finetuning)


__all__ = ["OpenVoiceConfig"]
