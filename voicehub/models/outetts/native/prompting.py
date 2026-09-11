"""Validated OuteTTS V3 prompting and speaker-profile protocol."""

from __future__ import annotations

import copy
import json
import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from voicehub.tokenization.assets import read_bounded_asset

_CONTROL_CHARACTERS = re.compile(r"[\x00-\x1F\x7F-\x9F\u00AD\u200B-\u200D\uFEFF]")
_WHITESPACE = re.compile(r"\s+")
_PUNCTUATION_GAP = re.compile(r"\s+([,.?!:;])")
_PUNCTUATION_JOIN = re.compile(r"([,.?!:;])(?=[^\s,.?!:;])")


@dataclass(frozen=True, slots=True)
class SpeakerWord:
    word: str
    duration: float
    c1: tuple[int, ...]
    c2: tuple[int, ...]
    energy: int
    spectral_centroid: int
    pitch: int


@dataclass(frozen=True, slots=True)
class SpeakerProfile:
    text: str
    words: tuple[SpeakerWord, ...]
    energy: int
    spectral_centroid: int
    pitch: int
    interface_version: int = 3

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> SpeakerProfile:
        if not isinstance(value, Mapping):
            raise TypeError("OuteTTS speaker profile must be a mapping.")
        text = value.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("OuteTTS speaker profile requires non-empty `text`.")
        interface_version = value.get("interface_version", 3)
        if interface_version != 3:
            raise ValueError("Native OuteTTS accepts V3 speaker profiles only.")
        raw_words = value.get("words")
        if (isinstance(raw_words, (str, bytes)) or not isinstance(raw_words, Sequence) or not raw_words):
            raise ValueError("OuteTTS speaker profile requires a non-empty `words` list.")
        words = tuple(_speaker_word(item, index=index) for index, item in enumerate(raw_words))
        global_features = _features(
            value.get("global_features"),
            owner="global_features",
        )
        return cls(
            text=text.strip(),
            words=words,
            energy=global_features["energy"],
            spectral_centroid=global_features["spectral_centroid"],
            pitch=global_features["pitch"],
            interface_version=3,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "text":
            self.text,
            "words": [{
                "word": word.word,
                "duration": word.duration,
                "c1": list(word.c1),
                "c2": list(word.c2),
                "features": {
                    "energy": word.energy,
                    "spectral_centroid": word.spectral_centroid,
                    "pitch": word.pitch,
                },
            } for word in self.words],
            "global_features": {
                "energy": self.energy,
                "spectral_centroid": self.spectral_centroid,
                "pitch": self.pitch,
            },
            "interface_version":
            3,
        }


def _feature_value(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"OuteTTS feature `{name}` must be an integer.")
    if not 0 <= value <= 100:
        raise ValueError(f"OuteTTS feature `{name}` must be in [0, 100].")
    return value


def _features(value: Any, *, owner: str) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise TypeError(f"OuteTTS `{owner}` must be a mapping.")
    required = ("energy", "spectral_centroid", "pitch")
    missing = [name for name in required if name not in value]
    if missing:
        raise ValueError(f"OuteTTS `{owner}` is missing: {', '.join(missing)}.")
    return {name: _feature_value(value[name], name=f"{owner}.{name}") for name in required}


def _codes(value: Any, *, name: str) -> tuple[int, ...]:
    if (isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value):
        raise ValueError(f"OuteTTS `{name}` must be a non-empty sequence.")
    result = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError(f"OuteTTS `{name}` must contain integers.")
        if not 0 <= item <= 1_024:
            raise ValueError(f"OuteTTS `{name}` values must be in [0, 1024].")
        result.append(item)
    return tuple(result)


def _speaker_word(value: Any, *, index: int) -> SpeakerWord:
    if not isinstance(value, Mapping):
        raise TypeError(f"OuteTTS speaker word {index} must be a mapping.")
    word = value.get("word")
    if not isinstance(word, str) or not word.strip():
        raise ValueError(f"OuteTTS speaker word {index} requires non-empty `word`.")
    duration = value.get("duration")
    if (isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or
            duration <= 0):
        raise ValueError(f"OuteTTS speaker word {index} duration must be finite and positive.")
    c1 = _codes(value.get("c1"), name=f"words[{index}].c1")
    c2 = _codes(value.get("c2"), name=f"words[{index}].c2")
    if len(c1) != len(c2):
        raise ValueError(f"OuteTTS speaker word {index} codebooks have different lengths.")
    features = _features(
        value.get("features"),
        owner=f"words[{index}].features",
    )
    return SpeakerWord(
        word=word.strip(),
        duration=float(duration),
        c1=c1,
        c2=c2,
        energy=features["energy"],
        spectral_centroid=features["spectral_centroid"],
        pitch=features["pitch"],
    )


def normalize_outetts_text(text: str) -> str:
    """Apply V3 normalization for valid Unicode using only the stdlib.

    The upstream optional ``ftfy`` pass heuristically repairs already
    corrupted/mojibake input. VoiceHub does not guess an alternate byte
    encoding; callers should decode source text correctly before
    training.
    """
    if not isinstance(text, str):
        raise TypeError("OuteTTS text must be a string.")
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("…", "...")
    normalized = re.sub(r"\.{2,}", "...", normalized)
    normalized = re.sub(r'[“”„‟«»]', '"', normalized)
    normalized = re.sub(r"[‘’‛‹›`´]", "'", normalized)
    normalized = re.sub(r"[–—―−‐]", "-", normalized)
    normalized = re.sub(r"-{2,}", "-", normalized)
    normalized = _CONTROL_CHARACTERS.sub("", normalized)
    normalized = _WHITESPACE.sub(" ", normalized)
    normalized = _PUNCTUATION_GAP.sub(r"\1", normalized)
    normalized = _PUNCTUATION_JOIN.sub(r"\1 ", normalized)
    normalized = re.sub(r"(\w)\s+'\s*(\w)", r"\1'\2", normalized)
    normalized = re.sub(
        r"(\w)\s+'\s*([,.?!:;\s]|$)",
        r"\1'\2",
        normalized,
    )
    normalized = re.sub(
        r"(\w)\s*'\s*([tsdmre])\b",
        r"\1'\2",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(r"([?!])\1+", r"\1", normalized)
    return _WHITESPACE.sub(" ", normalized).replace('"', "").strip()


class OuteTTSPromptProcessor:
    """Construct and decode the author-published OuteTTS V3 protocol."""

    BOS = "<|im_start|>"
    EOS = "<|im_end|>"
    TEXT_START = "<|text_start|>"
    TEXT_END = "<|text_end|>"
    AUDIO_START = "<|audio_start|>"
    AUDIO_END = "<|audio_end|>"
    WORD_START = "<|word_start|>"
    WORD_END = "<|word_end|>"
    FEATURES = "<|features|>"
    CODE = "<|code|>"
    GLOBAL_START = "<|global_features_start|>"
    GLOBAL_END = "<|global_features_end|>"

    def __init__(self, tokenizer) -> None:
        required = ("encode", "convert_tokens_to_ids", "audio_codes_from_ids")
        if any(not callable(getattr(tokenizer, name, None)) for name in required):
            raise TypeError("OuteTTS prompt processing requires its native tokenizer.")
        self.tokenizer = tokenizer
        for spelling in (
                self.BOS,
                self.EOS,
                self.TEXT_START,
                self.TEXT_END,
                self.AUDIO_START,
                self.AUDIO_END,
                self.WORD_START,
                self.WORD_END,
                self.FEATURES,
                self.CODE,
                self.GLOBAL_START,
                self.GLOBAL_END,
        ):
            tokenizer.convert_tokens_to_ids(spelling)

    @staticmethod
    def load_speaker(path: str | Path) -> SpeakerProfile:
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"OuteTTS speaker profile was not found: {source}.")
        try:
            value = json.loads(read_bounded_asset(source, max_bytes=8 * 1024 * 1024).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid OuteTTS speaker profile JSON: {error}.") from error
        return SpeakerProfile.from_mapping(value)

    @staticmethod
    def _feature_tokens(
        energy: int,
        spectral_centroid: int,
        pitch: int,
    ) -> str:
        return (f"<|energy_{energy}|>"
                f"<|spectral_centroid_{spectral_centroid}|>"
                f"<|pitch_{pitch}|>")

    @classmethod
    def _word_prompt(cls, word: SpeakerWord) -> str:
        pairs = "".join(
            f"<|c1_{first}|><|c2_{second}|>" for first, second in zip(word.c1, word.c2, strict=True))
        return (
            cls.WORD_START + word.word + cls.FEATURES + f"<|t_{word.duration:.2f}|>" + cls._feature_tokens(
                word.energy,
                word.spectral_centroid,
                word.pitch,
            ) + cls.CODE + pairs + cls.WORD_END)

    @staticmethod
    def _separator(text: str) -> str:
        if any("\u3040" <= character <= "\u30ff" or "\u4e00" <= character <= "\u9fff" for character in text):
            return "。"
        return ". "

    @classmethod
    def _merge_text(
        cls,
        requested: str,
        speaker_text: str,
    ) -> tuple[str, str]:
        reference = speaker_text.strip()
        separator = cls._separator(reference)
        suffix = ""
        if reference:
            allowed = (("。", "？", "！", "?", "!") if separator == "。" else (".", "?", "!"))
            if not reference.endswith(allowed):
                suffix = separator
            elif separator != "。":
                suffix = " "
        return reference + suffix + requested.strip(), suffix.strip()

    @classmethod
    def _initial_prompt(cls, text: str) -> str:
        return (f"{cls.BOS}\n"
                f"{cls.TEXT_START}{text}{cls.TEXT_END}\n"
                f"{cls.AUDIO_START}\n")

    def completion_prompt(
        self,
        text: str,
        speaker: SpeakerProfile | Mapping[str, Any] | None,
    ) -> str:
        normalized = normalize_outetts_text(text)
        if not normalized:
            raise ValueError("OuteTTS generation text cannot be empty.")
        if speaker is None:
            return self._initial_prompt(normalized)
        profile = (speaker if isinstance(speaker, SpeakerProfile) else SpeakerProfile.from_mapping(speaker))
        merged, separator = self._merge_text(normalized, profile.text)
        words = list(profile.words)
        if separator:
            final = words[-1]
            words[-1] = SpeakerWord(
                word=final.word + separator,
                duration=final.duration,
                c1=final.c1,
                c2=final.c2,
                energy=final.energy,
                spectral_centroid=final.spectral_centroid,
                pitch=final.pitch,
            )
        codes = "\n".join(self._word_prompt(word) for word in words)
        return self._initial_prompt(merged) + codes + "\n" + self.WORD_START

    def training_prompt(
        self,
        profile: SpeakerProfile | Mapping[str, Any],
    ) -> str:
        resolved = (profile if isinstance(profile, SpeakerProfile) else SpeakerProfile.from_mapping(profile))
        text = normalize_outetts_text(resolved.text)
        global_features = (
            self.GLOBAL_START + self._feature_tokens(
                resolved.energy,
                resolved.spectral_centroid,
                resolved.pitch,
            ) + self.GLOBAL_END + "\n")
        words = "\n".join(self._word_prompt(word) for word in resolved.words)
        return (
            self._initial_prompt(text) + global_features + words + "\n" + self.AUDIO_END + "\n" + self.EOS +
            "\n")

    def training_prefix(
        self,
        profile: SpeakerProfile | Mapping[str, Any],
        *,
        prompt_word_count: int,
    ) -> str:
        resolved = (profile if isinstance(profile, SpeakerProfile) else SpeakerProfile.from_mapping(profile))
        if (isinstance(prompt_word_count, bool) or not isinstance(prompt_word_count, int) or
                not 0 <= prompt_word_count < len(resolved.words)):
            raise ValueError(
                "`prompt_word_count` must be between zero and one fewer "
                "than the profile word count.")
        text = normalize_outetts_text(resolved.text)
        global_features = (
            self.GLOBAL_START + self._feature_tokens(
                resolved.energy,
                resolved.spectral_centroid,
                resolved.pitch,
            ) + self.GLOBAL_END + "\n")
        if prompt_word_count == 0:
            return self._initial_prompt(text) + global_features
        words = "\n".join(self._word_prompt(word) for word in resolved.words[:prompt_word_count])
        return self._initial_prompt(text) + global_features + words + "\n"

    def encode(self, prompt: str) -> list[int]:
        return list(self.tokenizer.encode(
            prompt,
            add_special_tokens=False,
        ).input_ids)

    def extract_audio_codes(
        self,
        token_ids,
    ) -> tuple[list[int], list[int]]:
        return self.tokenizer.audio_codes_from_ids(token_ids)


def load_default_speaker() -> SpeakerProfile:
    return OuteTTSPromptProcessor.load_speaker(Path(__file__).with_name("default_speaker.json"))


__all__ = [
    "OuteTTSPromptProcessor",
    "SpeakerProfile",
    "SpeakerWord",
    "load_default_speaker",
    "normalize_outetts_text",
]
