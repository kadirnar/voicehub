"""Compatibility helpers for legacy Whisper runtime provider names."""

from __future__ import annotations

from pathlib import Path

WHISPER_MODEL_ALIASES = {
    "tiny": "openai/whisper-tiny",
    "tiny.en": "openai/whisper-tiny.en",
    "base": "openai/whisper-base",
    "base.en": "openai/whisper-base.en",
    "small": "openai/whisper-small",
    "small.en": "openai/whisper-small.en",
    "medium": "openai/whisper-medium",
    "medium.en": "openai/whisper-medium.en",
    # openai-whisper and faster-whisper map ``large`` to large-v3; the
    # original large-v1 release is published as ``openai/whisper-large``.
    "large": "openai/whisper-large-v3",
    "large-v1": "openai/whisper-large",
    "large-v2": "openai/whisper-large-v2",
    "large-v3": "openai/whisper-large-v3",
    "large-v3-turbo": "openai/whisper-large-v3-turbo",
    "turbo": "openai/whisper-large-v3-turbo",
}


def normalize_whisper_source(value: str | Path) -> str | Path:
    """Resolve a legacy size alias while preserving explicit local paths."""
    if isinstance(value, Path):
        return value
    if not isinstance(value, str):
        raise TypeError("Whisper model sources must be strings or Paths.")
    normalized = value.strip()
    if not normalized:
        return value
    path = Path(normalized).expanduser()
    if path.exists():
        return value
    return WHISPER_MODEL_ALIASES.get(normalized.lower(), value)


__all__ = ["WHISPER_MODEL_ALIASES", "normalize_whisper_source"]
