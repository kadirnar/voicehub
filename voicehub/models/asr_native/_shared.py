"""Normalization helpers shared by native ASR provider wrappers."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from inspect import Parameter, signature
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from voicehub.audio import load_audio
from voicehub.models.base import BaseSpeechModel
from voicehub.models.native_utils import resolve_cpu_cuda_device
from voicehub.outputs import ASROutput, ASRSegment, ASRWord

_DEFAULT_INFERENCE_OPTIONS = {
    "task": "transcribe",
    "return_timestamps": False,
    "chunk_length_s": None,
    "stride_length_s": None,
    "batch_size": None,
    "num_beams": None,
    "max_new_tokens": None,
    "hotwords": None,
}


def supported_kwargs(callable_object, values: dict[str, Any]) -> dict[str, Any]:
    """Keep options accepted by a versioned third-party callable."""
    try:
        parameters = signature(callable_object).parameters
    except (TypeError, ValueError):
        return {key: value for key, value in values.items() if value is not None}
    if any(parameter.kind is Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return {key: value for key, value in values.items() if value is not None}
    return {key: value for key, value in values.items() if value is not None and key in parameters}


def require_supported_kwargs(
        callable_object,
        values: dict[str, Any],
        *,
        provider: str,
        required: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Filter versioned options and reject requested unsupported controls.

    ``supported_kwargs`` is appropriate for optional loader hints. Inference
    controls are different: silently dropping an explicitly requested option
    makes the returned transcript misleading. This helper preserves
    compatibility with older provider releases while enforcing that contract.
    """
    options = supported_kwargs(callable_object, values)
    missing = sorted(name for name in required if name not in options)
    if missing:
        formatted = ", ".join(f"`{name}`" for name in missing)
        raise ValueError(
            f"The installed {provider} runtime does not support the requested "
            f"inference option(s): {formatted}.")
    return options


def reject_unsupported_options(provider: str, **options: Any) -> None:
    """Reject non-default common inference options a provider cannot honor."""
    unsupported = sorted(
        name for name, value in options.items() if value != _DEFAULT_INFERENCE_OPTIONS.get(name))
    if not unsupported:
        return
    formatted = ", ".join(f"`{name}`" for name in unsupported)
    raise ValueError(f"{provider} does not support the requested inference option(s): "
                     f"{formatted}.")


def validate_ctranslate2_precision(
    compute_type: str,
    *,
    device: str,
    provider: str,
) -> None:
    """Reject CTranslate2 precision/device pairs known to be invalid."""
    if device == "cpu" and compute_type.strip().lower() in {
            "float16",
            "float16_float32",
    }:
        raise ValueError(
            f"{provider} cannot use `compute_type={compute_type!r}` on CPU. "
            "Use `default`, `float32`, or an integer quantization mode.")


def preferred_keyword(
    callable_object,
    names: tuple[str, ...],
    *,
    fallback: str,
) -> str:
    """Choose the first explicit keyword supported by a versioned callable."""
    try:
        parameters = signature(callable_object).parameters
    except (TypeError, ValueError):
        return fallback
    for name in names:
        if name in parameters:
            return name
    return fallback


def _confidence(value: Any) -> float | None:
    if hasattr(value, "item"):
        value = value.item()
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if 0.0 <= value <= 1.0 else None


def _scaled_time(value: Any, scale: float) -> float | None:
    if value is None:
        return None
    # Provider timestamps are frequently integer milliseconds. Rounding the
    # scaled result avoids exposing artifacts such as 0.7000000000000001 while
    # retaining sub-microsecond precision.
    return round(float(value) * scale, 12)


def _word(value: Any, *, timestamp_scale: float) -> ASRWord | None:
    get = (
        value.get if isinstance(value, Mapping) else lambda key, default=None: getattr(value, key, default))
    text = get("word", get("text", ""))
    text = "" if text is None else str(text).strip()
    if not text:
        return None
    start = get("start")
    end = get("end")
    return ASRWord(
        text=text,
        start=_scaled_time(start, timestamp_scale),
        end=_scaled_time(end, timestamp_scale),
        confidence=_confidence(get("probability", get("confidence", get("score")))),
        speaker=get("speaker"),
    )


def normalize_asr_result(
    result: Any,
    *,
    backend: str,
    duration: float | None = None,
    timestamp_scale: float = 1.0,
    language: str | None = None,
) -> ASROutput:
    """Convert Whisper-like dictionaries/objects to :class:`ASROutput`."""
    get = (
        result.get
        if isinstance(result, Mapping) else lambda key, default=None: getattr(result, key, default))
    text = get("text", "")
    language = get("language", language)
    raw_segments = get("segments", ())
    if raw_segments is None:
        raw_segments = ()
    segments = []
    for raw in raw_segments:
        segment_get = (
            raw.get if isinstance(raw, Mapping) else lambda key, default=None: getattr(raw, key, default))
        start = segment_get("start")
        end = segment_get("end", segment_get("stop"))
        raw_words = segment_get("words", ())
        if raw_words is None:
            raw_words = ()
        words = tuple(
            normalized for word in raw_words
            if (normalized := _word(
                word,
                timestamp_scale=timestamp_scale,
            )) is not None)
        segment_text_value = segment_get("text", "")
        segment_text = ("" if segment_text_value is None else str(segment_text_value).strip())
        if not segment_text and words:
            segment_text = " ".join(word.text for word in words)
        metadata = {}
        for name in ("id", "seek", "tokens", "temperature", "avg_logprob", "no_speech_prob"):
            value = segment_get(name)
            if value is not None:
                metadata[name] = value
        segments.append(
            ASRSegment(
                text=segment_text,
                start=_scaled_time(start, timestamp_scale),
                end=_scaled_time(end, timestamp_scale),
                confidence=_confidence(
                    segment_get(
                        "confidence",
                        segment_get("score", segment_get("probability")),
                    )),
                language=segment_get("language", language),
                speaker=segment_get("speaker"),
                words=words,
                metadata=metadata,
            ))
    if text is None:
        text = ""
    if not text and segments:
        text = " ".join(segment.text for segment in segments).strip()
    if isinstance(language, str):
        language = language.strip() or None
    return ASROutput(
        text=str(text),
        segments=tuple(segments),
        language=language,
        duration=duration,
        metadata={"backend": backend},
    )


@contextmanager
def materialized_audio_file_for_provider(
    audio: Any,
    *,
    sampling_rate: int | None,
    target_sampling_rate: int,
):
    """Yield ``(path, AudioInput)`` with provider-compatible PCM audio."""
    materialized = load_audio(
        audio,
        sampling_rate=sampling_rate,
        target_sampling_rate=target_sampling_rate,
    )
    temporary_path = None
    try:
        with NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            temporary_path = Path(handle.name)
        BaseSpeechModel.save_audio(
            temporary_path,
            materialized.waveform,
            materialized.sampling_rate,
        )
        yield str(temporary_path), materialized
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


@contextmanager
def audio_file_for_provider(
    audio: Any,
    *,
    sampling_rate: int | None,
    target_sampling_rate: int,
):
    """Yield an existing compatible file or one short-lived WAV file."""
    with materialized_audio_file_for_provider(
            audio,
            sampling_rate=sampling_rate,
            target_sampling_rate=target_sampling_rate,
    ) as (audio_path, _):
        yield audio_path
