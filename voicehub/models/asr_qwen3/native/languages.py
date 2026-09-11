"""Dependency-free language metadata for Qwen3-ASR."""

from __future__ import annotations

from collections.abc import Sequence

LANGUAGE_CODES = {
    "ar": "Arabic",
    "yue": "Cantonese",
    "zh": "Chinese",
    "cs": "Czech",
    "da": "Danish",
    "nl": "Dutch",
    "en": "English",
    "fil": "Filipino",
    "fi": "Finnish",
    "fr": "French",
    "de": "German",
    "el": "Greek",
    "hi": "Hindi",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "mk": "Macedonian",
    "ms": "Malay",
    "fa": "Persian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "es": "Spanish",
    "sv": "Swedish",
    "th": "Thai",
    "tr": "Turkish",
    "vi": "Vietnamese",
}


def normalize_qwen3_asr_language(
        language: str | None,
        *,
        supported_languages: Sequence[str] = (),
) -> str | None:
    """Normalize a code/name and enforce checkpoint-declared languages."""
    if language is None:
        return None
    if not isinstance(language, str) or not language.strip():
        raise ValueError("`language` must be a non-empty string or None.")
    value = language.strip()
    canonical = LANGUAGE_CODES.get(value.lower())
    declared = tuple(supported_languages)
    if canonical is None:
        by_name = {name.lower(): name for name in (declared or tuple(LANGUAGE_CODES.values()))}
        canonical = by_name.get(value.lower())
    if canonical is None:
        choices = ", ".join(declared or tuple(LANGUAGE_CODES.values()))
        raise ValueError(f"Unsupported Qwen3-ASR language {language!r}. Supported: "
                         f"{choices}.")
    if declared and canonical not in declared:
        raise ValueError(f"Language {canonical!r} is not declared by this checkpoint.")
    return canonical


__all__ = ["LANGUAGE_CODES", "normalize_qwen3_asr_language"]
