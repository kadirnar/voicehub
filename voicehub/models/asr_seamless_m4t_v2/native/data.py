"""Framework-free SeamlessM4T-v2 source-record normalization."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _field_present(record: Mapping[str, Any], name: str) -> bool:
    if name not in record or record[name] is None:
        return False
    value = record[name]
    return not isinstance(value, (str, bytes, bytearray)) or bool(value.strip())


def normalize_record(
    record: dict[str, Any],
    *,
    index: int,
) -> dict[str, Any]:
    """Flatten the official SeamlessM4T source/target manifest shape."""
    value = dict(record)
    source = value.get("source")
    target = value.get("target")
    if not isinstance(source, Mapping) and not isinstance(target, Mapping):
        return value
    if not isinstance(source, Mapping) or not isinstance(target, Mapping):
        raise TypeError(f"ASR record {index} Seamless `source` and `target` must both "
                        "be mappings.")
    extracted = {
        "audio":
        next(
            (
                source[name] for name in (
                    "audio_local_path",
                    "audio",
                    "audio_path",
                    "audio_filepath",
                ) if _field_present(source, name)),
            None,
        ),
        "sampling_rate":
        source.get("sampling_rate", source.get("sample_rate")),
        "source_language":
        source.get("lang", source.get("language")),
        "target_language":
        target.get("lang", target.get("language")),
        "text":
        target.get("text"),
    }
    flattened = {name: item for name, item in extracted.items() if item is not None}
    for name in flattened:
        if name in value:
            raise ValueError(
                f"ASR record {index} contains both the official Seamless "
                f"nested value and canonical field {name!r}.")
    value.update(flattened)
    value.pop("source")
    value.pop("target")
    return value


__all__ = ["normalize_record"]
