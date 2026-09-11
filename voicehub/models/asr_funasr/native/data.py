"""Framework-free SenseVoice source-record normalization."""

from __future__ import annotations

from typing import Any

_CONTROL_WRAPPERS = ("<|", "|>")


def _control_name(value: dict[str, Any], name: str) -> None:
    item = value.get(name)
    if item is None or not isinstance(item, str):
        return
    normalized = item.strip()
    prefix, suffix = _CONTROL_WRAPPERS
    if normalized.startswith(prefix) and normalized.endswith(suffix):
        normalized = normalized[len(prefix):-len(suffix)]
    normalized = normalized.strip().lower()
    aliases = {
        "emo_unknown": "unknown",
        "event_unk": "unknown",
    }
    value[name] = aliases.get(normalized, normalized)


def normalize_record(
    record: dict[str, Any],
    *,
    index: int,
) -> dict[str, Any]:
    """Normalize official SenseVoice control-token spellings."""
    value = dict(record)
    for field_name in ("language", "emotion", "event"):
        _control_name(value, field_name)

    use_itn = value.get("use_itn")
    if not isinstance(use_itn, str):
        return value
    normalized_itn = use_itn.strip()
    prefix, suffix = _CONTROL_WRAPPERS
    if normalized_itn.startswith(prefix) and normalized_itn.endswith(suffix):
        normalized_itn = normalized_itn[len(prefix):-len(suffix)]
    normalized_itn = normalized_itn.strip().lower().replace("_", "")
    if normalized_itn in {"withitn", "true", "1", "yes"}:
        value["use_itn"] = True
    elif normalized_itn in {"woitn", "withoutitn", "false", "0", "no"}:
        value["use_itn"] = False
    else:
        raise ValueError(
            f"ASR record {index} field 'use_itn' must be a boolean or a "
            "SenseVoice <|withitn|>/<|woitn|> control token.")
    return value


__all__ = ["normalize_record"]
