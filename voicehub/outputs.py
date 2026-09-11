"""Typed outputs returned by VoiceHub speech models."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Iterator


class _OutputMixin:
    """Small mapping/tuple protocol shared by public output dataclasses."""

    _tuple_fields: tuple[str, ...] = ()
    _optional_fields: tuple[str, ...] = ()

    def to_tuple(self) -> tuple[Any, ...]:
        return tuple(getattr(self, key) for key in self._tuple_fields)

    def __iter__(self) -> Iterator[Any]:
        return iter(self.to_tuple())

    def __getitem__(self, key: str | int):
        if isinstance(key, str):
            if key not in self.keys():
                raise KeyError(key)
            return getattr(self, key)
        return self.to_tuple()[key]

    def keys(self) -> tuple[str, ...]:
        populated = list(self._tuple_fields)
        for name in self._optional_fields:
            value = getattr(self, name)
            is_empty_container = isinstance(value, (tuple, list, dict)) and not value
            if value is not None and not is_empty_container:
                populated.append(name)
        return tuple(populated)

    def to_dict(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in self.keys()}


def _validate_time(value: float | None, *, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"`{name}` must be a real number or None.")
    value = float(value)
    if not isfinite(value) or value < 0:
        raise ValueError(f"`{name}` must be finite and non-negative.")
    return value


def _validate_score(value: float | None, *, name: str = "score") -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"`{name}` must be a real number or None.")
    value = float(value)
    if not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"`{name}` must be between 0 and 1.")
    return value


def _validate_optional_text(
    value: str | None,
    *,
    name: str,
) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"`{name}` must be a non-empty string or None.")
    return value.strip()


@dataclass
class AudioOutput(_OutputMixin):
    """Audio output with sampling metadata and optional backend details."""

    audio: Any
    sample_rate: int
    file_path: str | Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    _tuple_fields = ("audio", "sample_rate")
    _optional_fields = ("file_path", "metadata")

    def __post_init__(self) -> None:
        if (isinstance(self.sample_rate, bool) or not isinstance(self.sample_rate, Integral) or
                self.sample_rate <= 0):
            raise ValueError("`sample_rate` must be a positive integer.")
        self.sample_rate = int(self.sample_rate)
        if not isinstance(self.metadata, dict):
            raise TypeError("`metadata` must be a dictionary.")
        from voicehub.models.base import BaseSpeechModel

        BaseSpeechModel.validate_audio(self.audio)

    def save(self, file_path: str | Path) -> str:
        """Write this output through VoiceHub's normalized audio writer."""
        from voicehub.models.base import BaseSpeechModel

        self.file_path = BaseSpeechModel.save_audio(file_path, self.audio, self.sample_rate)
        return self.file_path

    @property
    def path(self) -> Path | None:
        """Return ``file_path`` as a ``Path`` when present."""
        return Path(self.file_path) if self.file_path else None

    def to_tuple(self) -> tuple[Any, int]:
        """Return ``(audio, sample_rate)`` for interoperability."""
        return self.audio, self.sample_rate


@dataclass
class TTSOutput(AudioOutput):
    """Synthesized audio, with the same waveform contract as codec output."""


@dataclass
class CodecOutput(_OutputMixin):
    """Encoded audio with the information required for faithful decoding.

    ``codes`` is owned by the codec: it may contain discrete tensors,
    continuous latents, ragged codebooks, or segmented codes and scales.
    ``length`` counts original samples at ``sample_rate``, before
    padding.
    """

    codes: Any
    sample_rate: int
    length: int
    model_type: str
    metadata: dict[str, Any] = field(default_factory=dict)
    _tuple_fields = ("codes", "sample_rate", "length", "model_type")
    _optional_fields = ("metadata", )

    def __post_init__(self) -> None:
        for name in ("sample_rate", "length"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"`{name}` must be a positive integer.")
        if self.codes is None:
            raise ValueError("Codec output must contain codes or latents.")
        if not isinstance(self.model_type, str) or not self.model_type.strip():
            raise ValueError("Codec output must identify its model_type.")
        if not isinstance(self.metadata, dict):
            raise TypeError("`metadata` must be a dictionary.")


@dataclass(frozen=True)
class ASRWord:
    """One recognized word with optional timing and confidence metadata."""

    text: str
    start: float | None = None
    end: float | None = None
    confidence: float | None = None
    speaker: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("ASR word text must be a non-empty string.")
        object.__setattr__(self, "text", self.text.strip())
        object.__setattr__(self, "start", _validate_time(self.start, name="start"))
        object.__setattr__(self, "end", _validate_time(self.end, name="end"))
        if self.start is not None and self.end is not None and self.end < self.start:
            raise ValueError("ASR word `end` cannot be earlier than `start`.")
        object.__setattr__(
            self,
            "confidence",
            _validate_score(self.confidence, name="confidence"),
        )
        object.__setattr__(
            self,
            "speaker",
            _validate_optional_text(self.speaker, name="speaker"),
        )


@dataclass(frozen=True)
class ASRSegment:
    """A timestamped transcription segment."""

    text: str
    start: float | None = None
    end: float | None = None
    confidence: float | None = None
    language: str | None = None
    speaker: str | None = None
    words: tuple[ASRWord, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("ASR segment `text` must be a string.")
        object.__setattr__(self, "text", self.text.strip())
        object.__setattr__(self, "start", _validate_time(self.start, name="start"))
        object.__setattr__(self, "end", _validate_time(self.end, name="end"))
        if self.start is not None and self.end is not None and self.end < self.start:
            raise ValueError("ASR segment `end` cannot be earlier than `start`.")
        object.__setattr__(
            self,
            "confidence",
            _validate_score(self.confidence, name="confidence"),
        )
        object.__setattr__(
            self,
            "language",
            _validate_optional_text(self.language, name="language"),
        )
        object.__setattr__(
            self,
            "speaker",
            _validate_optional_text(self.speaker, name="speaker"),
        )
        words = tuple(self.words)
        if any(not isinstance(word, ASRWord) for word in words):
            raise TypeError("ASR segment `words` must contain ASRWord instances.")
        object.__setattr__(self, "words", words)
        if not isinstance(self.metadata, dict):
            raise TypeError("ASR segment `metadata` must be a dictionary.")


@dataclass
class ASROutput(_OutputMixin):
    """Normalized transcription produced by any ASR backend."""

    text: str
    segments: tuple[ASRSegment, ...] = ()
    language: str | None = None
    duration: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    _tuple_fields = ("text", "segments")
    _optional_fields = ("language", "duration", "metadata")

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("ASR output `text` must be a string.")
        self.text = self.text.strip()
        self.segments = tuple(self.segments)
        if any(not isinstance(segment, ASRSegment) for segment in self.segments):
            raise TypeError("ASR output `segments` must contain ASRSegment instances.")
        self.duration = _validate_time(self.duration, name="duration")
        self.language = _validate_optional_text(
            self.language,
            name="language",
        )
        if not isinstance(self.metadata, dict):
            raise TypeError("ASR output `metadata` must be a dictionary.")


@dataclass(frozen=True)
class SpeechSegment:
    """One detected interval of speech or non-speech."""

    start: float
    end: float
    score: float | None = None
    label: str = "speech"
    channel: int | str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", _validate_time(self.start, name="start"))
        object.__setattr__(self, "end", _validate_time(self.end, name="end"))
        if self.end <= self.start:
            raise ValueError("Speech segment `end` must be greater than `start`.")
        object.__setattr__(self, "score", _validate_score(self.score))
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("Speech segment `label` must be a non-empty string.")
        object.__setattr__(self, "label", self.label.strip())
        if not isinstance(self.metadata, dict):
            raise TypeError("Speech segment `metadata` must be a dictionary.")

    def sample_bounds(self, sample_rate: int) -> tuple[int, int]:
        """Return rounded ``(start_sample, end_sample)`` indices."""
        if (isinstance(sample_rate, bool) or not isinstance(sample_rate, Integral) or sample_rate <= 0):
            raise ValueError("`sample_rate` must be a positive integer.")
        return round(self.start * sample_rate), round(self.end * sample_rate)


VADSegment = SpeechSegment


@dataclass
class VADOutput(_OutputMixin):
    """Normalized speech regions produced by any VAD backend."""

    segments: tuple[SpeechSegment, ...]
    duration: float | None = None
    sample_rate: int | None = None
    probabilities: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    _tuple_fields = ("segments", )
    _optional_fields = ("duration", "sample_rate", "probabilities", "metadata")

    def __post_init__(self) -> None:
        self.segments = tuple(self.segments)
        if any(not isinstance(segment, SpeechSegment) for segment in self.segments):
            raise TypeError("VAD output `segments` must contain SpeechSegment instances.")
        previous_start = -1.0
        previous_end = -1.0
        for segment in self.segments:
            if segment.start < previous_start:
                raise ValueError("VAD output segments must be ordered by start time.")
            if segment.start < previous_end:
                raise ValueError("VAD output speech segments must not overlap.")
            previous_start = segment.start
            previous_end = segment.end
        self.duration = _validate_time(self.duration, name="duration")
        if self.sample_rate is not None:
            if (isinstance(self.sample_rate, bool) or not isinstance(self.sample_rate, Integral) or
                    self.sample_rate <= 0):
                raise ValueError("VAD output `sample_rate` must be a positive integer or None.")
            self.sample_rate = int(self.sample_rate)
        if not isinstance(self.metadata, dict):
            raise TypeError("VAD output `metadata` must be a dictionary.")
        if self.duration is not None and self.segments:
            if self.segments[-1].end > self.duration:
                raise ValueError("VAD segment end cannot exceed output duration.")

    @property
    def speech_duration(self) -> float:
        """Return the summed duration of all detected speech regions."""
        return sum(segment.end - segment.start for segment in self.segments)

    def contains(self, timestamp: float) -> bool:
        """Return whether *timestamp* falls inside a detected region."""
        value = _validate_time(timestamp, name="timestamp")
        return any(segment.start <= value < segment.end for segment in self.segments)


@dataclass
class SpeechTrainingOutput(_OutputMixin):
    """Task-neutral differentiable output consumed by
    :class:`voicehub.Trainer`.

    ``loss`` deliberately comes first, matching Transformers model
    outputs. Architecture-specific training implementations can leave
    unused fields empty and place additional values in ``metadata``.
    """

    loss: Any | None = None
    logits: Any | None = None
    predictions: Any | None = None
    audio_values: Any | None = None
    hidden_states: Any | None = None
    attentions: Any | None = None
    training_phase: str | None = None
    optimizer_names: tuple[str, ...] = ()
    losses: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def phase(self) -> str | None:
        """Backward-friendly short alias for the executed training phase."""
        return self.training_phase

    def to_tuple(self) -> tuple[Any, ...]:
        """Return populated fields in declaration order."""
        return tuple(getattr(self, key) for key in self.keys())

    def __iter__(self) -> Iterator[Any]:
        return iter(self.to_tuple())

    def __getitem__(self, key: str | int):
        if isinstance(key, str):
            if key not in self.keys():
                raise KeyError(key)
            return getattr(self, key)
        return self.to_tuple()[key]

    def keys(self) -> tuple[str, ...]:
        """Return fields that carry a value."""
        names = (
            "loss",
            "logits",
            "predictions",
            "audio_values",
            "hidden_states",
            "attentions",
            "training_phase",
        )
        populated = [name for name in names if getattr(self, name) is not None]
        if self.optimizer_names:
            populated.append("optimizer_names")
        if self.losses:
            populated.append("losses")
        if self.metadata:
            populated.append("metadata")
        return tuple(populated)

    def to_dict(self) -> dict[str, Any]:
        """Return populated fields as a mapping."""
        return {key: getattr(self, key) for key in self.keys()}


@dataclass
class TTSTrainingOutput(SpeechTrainingOutput):
    """Backward-compatible name for the shared speech training output."""
