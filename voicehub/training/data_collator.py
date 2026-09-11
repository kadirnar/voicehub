"""Data collators compatible with speech, text, and tensor-valued examples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from voicehub.dependencies import import_optional


def default_data_collator(
    features: list[Any],
    return_tensors: str = "pt",
) -> dict[str, Any]:
    """Collate mapping-like samples while preserving strings and metadata."""
    if return_tensors != "pt":
        raise ValueError("VoiceHub's default collator currently supports `pt` tensors.")
    if not features:
        return {}

    torch = import_optional(
        "torch",
        model_type="Trainer",
        install_extra="training",
    )
    first = features[0]
    if not isinstance(first, dict):
        if hasattr(first, "__dict__"):
            features = [vars(feature) for feature in features]
            first = features[0]
        else:
            raise TypeError("Dataset samples must be mappings or dataclass-like objects.")

    batch: dict[str, Any] = {}
    if "label" in first and first["label"] is not None:
        label = first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor(
            [feature["label"] for feature in features],
            dtype=dtype,
        )
    elif "label_ids" in first and first["label_ids"] is not None:
        label_ids = first["label_ids"]
        if isinstance(label_ids, torch.Tensor):
            batch["labels"] = torch.stack([feature["label_ids"] for feature in features])
        else:
            batch["labels"] = torch.tensor([feature["label_ids"] for feature in features])

    for key, value in first.items():
        if key in ("label", "label_ids") or value is None:
            continue
        values = [feature[key] for feature in features]
        if isinstance(value, torch.Tensor):
            batch[key] = torch.stack(values)
        elif hasattr(value, "dtype") and hasattr(value, "shape"):
            try:
                batch[key] = torch.stack(tuple(torch.as_tensor(item) for item in values))
            except (TypeError, ValueError, RuntimeError):
                batch[key] = values
        elif isinstance(value, (int, float, bool, list, tuple)):
            try:
                batch[key] = torch.tensor(values)
            except (TypeError, ValueError):
                batch[key] = values
        else:
            batch[key] = values
    return batch


@dataclass
class DefaultDataCollator:
    """Callable object form of :func:`default_data_collator`."""

    return_tensors: str = "pt"

    def __call__(self, features: list[Any]) -> dict[str, Any]:
        return default_data_collator(features, return_tensors=self.return_tensors)
