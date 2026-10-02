"""Voice-design instruction validation for OmniVoice.

Port of upstream ``omnivoice.utils.voice_design`` and ``_resolve_instruct``
(revision 468e927). The checkpoint was trained on lower-case English items
joined by ``", "`` or Chinese items joined by ``"，"``; upstream normalizes,
validates, and translates every instruction before tokenization, so an
unnormalized instruction would reach the model off-distribution.
"""

from __future__ import annotations

import difflib
import re

CHINESE_CHARACTER = re.compile(r"[一-鿿]")

_CATEGORIES: tuple[dict[str, str] | frozenset[str], ...] = (
    {
        "male": "男",
        "female": "女"
    },
    {
        "child": "儿童",
        "teenager": "少年",
        "young adult": "青年",
        "middle-aged": "中年",
        "elderly": "老年",
    },
    {
        "very low pitch": "极低音调",
        "low pitch": "低音调",
        "moderate pitch": "中音调",
        "high pitch": "高音调",
        "very high pitch": "极高音调",
    },
    {
        "whisper": "耳语"
    },
    frozenset({
        "american accent",
        "british accent",
        "australian accent",
        "chinese accent",
        "canadian accent",
        "indian accent",
        "korean accent",
        "portuguese accent",
        "russian accent",
        "japanese accent",
    }),
    frozenset({
        "河南话",
        "陕西话",
        "四川话",
        "贵州话",
        "云南话",
        "桂林话",
        "济南话",
        "石家庄话",
        "甘肃话",
        "宁夏话",
        "青岛话",
        "东北话",
    }),
)

_ENGLISH_TO_CHINESE: dict[str, str] = {}
_CHINESE_TO_ENGLISH: dict[str, str] = {}
_MUTUALLY_EXCLUSIVE: list[frozenset[str]] = []
for _category in _CATEGORIES:
    if isinstance(_category, dict):
        _ENGLISH_TO_CHINESE.update(_category)
        _CHINESE_TO_ENGLISH.update({value: key for key, value in _category.items()})
        _MUTUALLY_EXCLUSIVE.append(frozenset(_category) | frozenset(_category.values()))
    else:
        _MUTUALLY_EXCLUSIVE.append(_category)
_ALL_VALID = frozenset(_ENGLISH_TO_CHINESE) | frozenset(
    _CHINESE_TO_ENGLISH) | _CATEGORIES[-2] | _CATEGORIES[-1]
_VALID_ENGLISH = frozenset(item for item in _ALL_VALID if not CHINESE_CHARACTER.search(item))
_VALID_CHINESE = frozenset(item for item in _ALL_VALID if CHINESE_CHARACTER.search(item))


def resolve_instruction(instruction: str | None, *, use_chinese: bool = False) -> str | None:
    """Validate and normalize a voice-design instruction like upstream.

    ``use_chinese`` is true when the synthesis text contains Chinese
    characters; a dialect forces Chinese items and an accent forces English
    items. Unsupported, misspelled, mixed, or conflicting items raise
    ``ValueError`` exactly where upstream does.
    """
    if instruction is None:
        return None
    text = instruction.strip()
    if not text:
        return None
    items = [item for item in re.split(r"\s*[,，]\s*", text) if item]
    unknown = []
    normalized = []
    for raw in items:
        item = raw.strip().lower()
        if item in _ALL_VALID:
            normalized.append(item)
        else:
            suggestion = difflib.get_close_matches(item, _ALL_VALID, n=1, cutoff=0.6)
            unknown.append((raw, item, suggestion[0] if suggestion else None))
    if unknown:
        lines = [(
            f"  '{raw}' -> '{item}' (unsupported; did you mean '{suggestion}'?)"
            if suggestion else f"  '{raw}' -> '{item}' (unsupported)") for raw, item, suggestion in unknown]
        raise ValueError(
            f"Unsupported instruct items found in {text}:\n" + "\n".join(lines) +
            "\n\nValid English items: " + ", ".join(sorted(_VALID_ENGLISH)) + "\nValid Chinese items: " +
            "，".join(sorted(_VALID_CHINESE)) +
            "\n\nTip: Use only English or only Chinese instructs. English instructs "
            "should use comma + space (e.g. 'male, indian accent'),\nChinese instructs "
            "should use full-width comma (e.g. '男，河南话').")

    has_dialect = any(item.endswith("话") for item in normalized)
    has_accent = any(" accent" in item for item in normalized)
    if has_dialect and has_accent:
        raise ValueError(
            "Cannot mix Chinese dialect and English accent in a single instruct. "
            "Dialects are for Chinese speech, accents for English speech.")
    if has_dialect:
        use_chinese = True
    elif has_accent:
        use_chinese = False
    if use_chinese:
        normalized = [_ENGLISH_TO_CHINESE.get(item, item) for item in normalized]
    else:
        normalized = [_CHINESE_TO_ENGLISH.get(item, item) for item in normalized]

    conflicts = []
    for category in _MUTUALLY_EXCLUSIVE:
        hits = [item for item in normalized if item in category]
        if len(hits) > 1:
            conflicts.append(hits)
    if conflicts:
        raise ValueError(
            "Conflicting instruct items within the same category: " +
            "; ".join(" vs ".join(f"'{item}'" for item in group) for group in conflicts) +
            ". Each category (gender, age, pitch, style, accent, dialect) "
            "allows at most one item.")
    has_chinese = any(any("一" <= character <= "鿿" for character in item) for item in normalized)
    return ("，" if has_chinese else ", ").join(normalized)


__all__ = ["CHINESE_CHARACTER", "resolve_instruction"]
