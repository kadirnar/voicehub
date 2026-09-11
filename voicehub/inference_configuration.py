"""Serializable inference controls for audio-input speech tasks."""

from __future__ import annotations

from copy import deepcopy
from math import isfinite
from numbers import Integral, Real
from pathlib import Path
from typing import Any

from voicehub.configuration import reject_serialized_secrets
from voicehub.hub import read_json_file, resolve_pretrained_file, write_json_file
from voicehub.serialization_utils import serialize_paths

ASR_CONFIG_NAME = "transcription_config.json"
VAD_CONFIG_NAME = "vad_config.json"


class SpeechInferenceConfig:
    """Base for task-specific, extensible inference configuration."""

    config_name = "inference_config.json"
    _COMMON_FIELDS: frozenset[str] = frozenset()

    def __init__(self, **kwargs):
        reject_serialized_secrets(
            kwargs,
            owner=self.__class__.__name__,
        )
        for key, value in kwargs.items():
            if value is not None:
                setattr(self, key, value)
        self.validate()

    def validate(self) -> None:
        """Validate common values in task-specific subclasses."""

    def to_dict(self) -> dict[str, Any]:
        reject_serialized_secrets(
            self.__dict__,
            owner=self.__class__.__name__,
        )
        return serialize_paths(deepcopy(self.__dict__))

    @classmethod
    def from_dict(cls, values: dict[str, Any], **kwargs):
        merged = dict(values)
        merged.update(kwargs)
        return cls(**merged)

    @classmethod
    def from_model_config(cls, config):
        values = getattr(config, "inference_config", {})
        if isinstance(values, cls):
            return cls.from_dict(values.to_dict())
        if not isinstance(values, dict):
            raise TypeError("`inference_config` must be a mapping.")
        return cls.from_dict(values)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        subfolder: str = "",
        **kwargs,
    ):
        source = Path(pretrained_model_name_or_path).expanduser()
        if source.is_file() and source.name == cls.config_name:
            config_path = source
        else:
            config_path = resolve_pretrained_file(
                pretrained_model_name_or_path,
                cls.config_name,
                subfolder=subfolder,
                cache_dir=kwargs.pop("cache_dir", None),
                revision=kwargs.pop("revision", None),
                token=kwargs.pop("token", None),
                local_files_only=kwargs.pop("local_files_only", False),
            )
        return cls.from_dict(read_json_file(config_path), **kwargs)

    def save_pretrained(self, save_directory: str | Path) -> Path:
        output_path = Path(save_directory).expanduser() / self.config_name
        values = self.to_dict()
        reject_serialized_secrets(
            values,
            owner=self.__class__.__name__,
        )
        write_json_file(output_path, values)
        return output_path

    def __repr__(self) -> str:
        values = self.to_dict()
        reject_serialized_secrets(
            values,
            owner=self.__class__.__name__,
        )
        fields = ", ".join(f"{key}={value!r}" for key, value in sorted(values.items()))
        return f"{self.__class__.__name__}({fields})"


class ASRInferenceConfig(SpeechInferenceConfig):
    """Decoding and long-audio controls shared by ASR providers."""

    config_name = ASR_CONFIG_NAME
    _COMMON_FIELDS = frozenset({
        "language",
        "task",
        "return_timestamps",
        "chunk_length_s",
        "stride_length_s",
        "batch_size",
        "num_beams",
        "max_new_tokens",
        "hotwords",
    })

    def __init__(
        self,
        *,
        language: str | None = None,
        task: str = "transcribe",
        return_timestamps: bool | str = False,
        chunk_length_s: float | None = None,
        stride_length_s: float | tuple[float, float] | None = None,
        batch_size: int | None = None,
        num_beams: int | None = None,
        max_new_tokens: int | None = None,
        hotwords: str | tuple[str, ...] | list[str] | None = None,
        **kwargs,
    ):
        super().__init__(
            language=language,
            task=task,
            return_timestamps=return_timestamps,
            chunk_length_s=chunk_length_s,
            stride_length_s=stride_length_s,
            batch_size=batch_size,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            hotwords=hotwords,
            **kwargs,
        )

    def validate(self) -> None:
        language = getattr(self, "language", None)
        if language is not None and (not isinstance(language, str) or not language.strip()):
            raise ValueError("`language` must be a non-empty string or None.")
        if language is not None:
            self.language = language.strip()
        task = getattr(self, "task", "transcribe")
        if task not in ("transcribe", "translate"):
            raise ValueError("ASR `task` must be 'transcribe' or 'translate'.")
        timestamps = getattr(self, "return_timestamps", False)
        if not (isinstance(timestamps, bool) or
                isinstance(timestamps, str) and timestamps in {"segment", "word"}):
            raise ValueError("`return_timestamps` must be a boolean, 'segment', or 'word'.")
        for name in ("chunk_length_s", ):
            value = getattr(self, name, None)
            if value is not None and (isinstance(value, bool) or not isinstance(value, Real) or
                                      not isfinite(value) or value <= 0):
                raise ValueError(f"`{name}` must be finite and greater than zero.")
        stride = getattr(self, "stride_length_s", None)
        if isinstance(stride, (tuple, list)):
            if len(stride) != 2:
                raise ValueError("`stride_length_s` sequences must contain left and right values.")
            strides = tuple(stride)
            self.stride_length_s = strides
        else:
            strides = (stride, )
        for value in strides:
            if value is not None and (isinstance(value, bool) or not isinstance(value, Real) or
                                      not isfinite(value) or value < 0):
                raise ValueError("`stride_length_s` values must be finite and non-negative.")
        for name in ("batch_size", "num_beams", "max_new_tokens"):
            value = getattr(self, name, None)
            if value is not None and (isinstance(value, bool) or not isinstance(value, Integral) or
                                      value <= 0):
                raise ValueError(f"`{name}` must be a positive integer or None.")
        hotwords = getattr(self, "hotwords", None)
        if hotwords is not None:
            if isinstance(hotwords, str):
                words = (hotwords, )
            elif isinstance(hotwords, (tuple, list)):
                words = tuple(hotwords)
            else:
                raise TypeError("`hotwords` must be a string, a sequence of strings, or None.")
            if any(not isinstance(word, str) or not word.strip() for word in words):
                raise ValueError("`hotwords` must contain non-empty strings.")
            normalized = tuple(word.strip() for word in words)
            self.hotwords = (normalized[0] if isinstance(hotwords, str) else normalized)


class VADInferenceConfig(SpeechInferenceConfig):
    """Thresholding and segmentation controls shared by VAD providers."""

    config_name = VAD_CONFIG_NAME
    _COMMON_FIELDS = frozenset({
        "threshold",
        "onset",
        "offset",
        "min_speech_duration_ms",
        "min_silence_duration_ms",
        "speech_pad_ms",
        "max_speech_duration_s",
        "window_size_samples",
        "return_frames",
    })

    def __init__(
        self,
        *,
        threshold: float = 0.5,
        onset: float | None = None,
        offset: float | None = None,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30,
        max_speech_duration_s: float | None = None,
        window_size_samples: int | None = None,
        return_frames: bool = False,
        **kwargs,
    ):
        super().__init__(
            threshold=threshold,
            onset=onset,
            offset=offset,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
            max_speech_duration_s=max_speech_duration_s,
            window_size_samples=window_size_samples,
            return_frames=return_frames,
            **kwargs,
        )

    def validate(self) -> None:
        for name in ("threshold", "onset", "offset"):
            value = getattr(self, name, None)
            if value is not None and (isinstance(value, bool) or not isinstance(value, Real) or
                                      not isfinite(value) or not 0.0 <= value <= 1.0):
                raise ValueError(f"`{name}` must be between 0 and 1.")
        for name in (
                "min_speech_duration_ms",
                "min_silence_duration_ms",
                "speech_pad_ms",
        ):
            value = getattr(self, name, 0)
            if (isinstance(value, bool) or not isinstance(value, Integral) or value < 0):
                raise ValueError(f"`{name}` must be a non-negative integer.")
        maximum = getattr(self, "max_speech_duration_s", None)
        if maximum is not None and (isinstance(maximum, bool) or not isinstance(maximum, Real) or
                                    not isfinite(maximum) or maximum <= 0):
            raise ValueError("`max_speech_duration_s` must be finite and greater than zero.")
        window = getattr(self, "window_size_samples", None)
        if window is not None and (isinstance(window, bool) or not isinstance(window, Integral) or
                                   window <= 0):
            raise ValueError("`window_size_samples` must be a positive integer.")
        if not isinstance(getattr(self, "return_frames", False), bool):
            raise TypeError("`return_frames` must be a boolean.")
