"""Source-derived multilingual duration estimation for OmniVoice."""

from __future__ import annotations

import bisect
import unicodedata
from functools import lru_cache
from typing import ClassVar


class RuleDurationEstimator:
    """Estimate codec frames from a reference text/audio-token pair.

    The weights and Unicode block policy follow the estimator published
    by OmniVoice.  The returned unit is the unit supplied for
    ``reference_duration``; inference passes reference codec frames.
    """

    _WEIGHTS: ClassVar[dict[str, float]] = {
        "cjk": 3.0,
        "hangul": 2.5,
        "kana": 2.2,
        "ethiopic": 3.0,
        "yi": 3.0,
        "indic": 1.8,
        "thai_lao": 1.5,
        "khmer_myanmar": 1.8,
        "arabic": 1.5,
        "hebrew": 1.5,
        "latin": 1.0,
        "cyrillic": 1.0,
        "greek": 1.0,
        "armenian": 1.0,
        "georgian": 1.0,
        "punctuation": 0.5,
        "space": 0.2,
        "digit": 3.5,
        "mark": 0.0,
        "default": 1.0,
    }
    _RANGES = (
        (0x02AF, "latin"),
        (0x03FF, "greek"),
        (0x052F, "cyrillic"),
        (0x058F, "armenian"),
        (0x05FF, "hebrew"),
        (0x077F, "arabic"),
        (0x089F, "arabic"),
        (0x08FF, "arabic"),
        (0x0DFF, "indic"),
        (0x0EFF, "thai_lao"),
        (0x0FFF, "indic"),
        (0x109F, "khmer_myanmar"),
        (0x10FF, "georgian"),
        (0x11FF, "hangul"),
        (0x139F, "ethiopic"),
        (0x13FF, "default"),
        (0x177F, "default"),
        (0x17FF, "khmer_myanmar"),
        (0x18FF, "default"),
        (0x194F, "indic"),
        (0x19DF, "indic"),
        (0x19FF, "khmer_myanmar"),
        (0x1C7F, "indic"),
        (0x1C8F, "cyrillic"),
        (0x1CBF, "georgian"),
        (0x1CFF, "indic"),
        (0x1EFF, "latin"),
        (0x309F, "kana"),
        (0x30FF, "kana"),
        (0x312F, "cjk"),
        (0x318F, "hangul"),
        (0x9FFF, "cjk"),
        (0xA4CF, "yi"),
        (0xA4FF, "default"),
        (0xA63F, "default"),
        (0xA69F, "cyrillic"),
        (0xA6FF, "default"),
        (0xA7FF, "latin"),
        (0xA82F, "indic"),
        (0xA87F, "default"),
        (0xA8FF, "indic"),
        (0xA92F, "indic"),
        (0xA95F, "indic"),
        (0xA97F, "hangul"),
        (0xA9DF, "indic"),
        (0xA9FF, "khmer_myanmar"),
        (0xAA5F, "indic"),
        (0xAA7F, "khmer_myanmar"),
        (0xAAFF, "indic"),
        (0xAB2F, "ethiopic"),
        (0xAB6F, "latin"),
        (0xABBF, "default"),
        (0xABFF, "indic"),
        (0xD7AF, "hangul"),
        (0xFAFF, "cjk"),
        (0xFDFF, "arabic"),
        (0xFE6F, "default"),
        (0xFEFF, "arabic"),
        (0xFFEF, "latin"),
    )
    _BREAKPOINTS = tuple(end for end, _ in _RANGES)

    @classmethod
    @lru_cache(maxsize=4096)
    def _character_weight(cls, character: str) -> float:
        codepoint = ord(character)
        if 65 <= codepoint <= 90 or 97 <= codepoint <= 122:
            return cls._WEIGHTS["latin"]
        if codepoint == 32:
            return cls._WEIGHTS["space"]
        if codepoint == 0x0640:
            return cls._WEIGHTS["mark"]
        category = unicodedata.category(character)
        if category.startswith("M"):
            return cls._WEIGHTS["mark"]
        if category.startswith(("P", "S")):
            return cls._WEIGHTS["punctuation"]
        if category.startswith("Z"):
            return cls._WEIGHTS["space"]
        if category.startswith("N"):
            return cls._WEIGHTS["digit"]
        index = bisect.bisect_left(cls._BREAKPOINTS, codepoint)
        if index < len(cls._RANGES):
            return cls._WEIGHTS[cls._RANGES[index][1]]
        if codepoint > 0x20000:
            return cls._WEIGHTS["cjk"]
        return cls._WEIGHTS["default"]

    def calculate_total_weight(self, text: str) -> float:
        if not isinstance(text, str):
            raise TypeError("Duration-estimator text must be a string.")
        return sum(self._character_weight(character) for character in text)

    def estimate_duration(
        self,
        target_text: str,
        reference_text: str,
        reference_duration: float,
        *,
        low_threshold: float | None = 50.0,
        boost_strength: float = 3.0,
    ) -> float:
        if reference_duration <= 0 or not reference_text:
            return 0.0
        reference_weight = self.calculate_total_weight(reference_text)
        if reference_weight == 0:
            return 0.0
        estimate = (self.calculate_total_weight(target_text) / (reference_weight / reference_duration))
        if low_threshold is not None and estimate < low_threshold:
            return low_threshold * (estimate / low_threshold)**(1.0 / boost_strength)
        return estimate


__all__ = ["RuleDurationEstimator"]
