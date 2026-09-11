"""Torch-only powerset conversion used by pyannote segmentation checkpoints."""

from __future__ import annotations

from itertools import combinations

import torch
from torch import Tensor, nn


class Powerset(nn.Module):
    """Convert categorical powerset scores to hard multi-label activity."""

    def __init__(self, num_classes: int, max_set_size: int) -> None:
        super().__init__()
        if (isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 1):
            raise ValueError("`num_classes` must be a positive integer.")
        if (isinstance(max_set_size, bool) or not isinstance(max_set_size, int) or
                not 0 <= max_set_size <= num_classes):
            raise ValueError("`max_set_size` must be between zero and `num_classes`.")
        rows = []
        for set_size in range(max_set_size + 1):
            for current_set in combinations(range(num_classes), set_size):
                row = [0.0] * num_classes
                for index in current_set:
                    row[index] = 1.0
                rows.append(row)
        self.num_classes = num_classes
        self.max_set_size = max_set_size
        self.register_buffer(
            "mapping",
            torch.tensor(rows, dtype=torch.float32),
            persistent=False,
        )

    @property
    def num_powerset_classes(self) -> int:
        return int(self.mapping.shape[0])

    def to_multilabel(self, powerset: Tensor) -> Tensor:
        if not isinstance(powerset, Tensor):
            raise TypeError("`powerset` must be a PyTorch tensor.")
        if powerset.ndim < 1 or powerset.shape[-1] != self.num_powerset_classes:
            raise ValueError("The final powerset dimension does not match the mapping.")
        hard = torch.nn.functional.one_hot(
            torch.argmax(powerset, dim=-1),
            num_classes=self.num_powerset_classes,
        ).to(dtype=powerset.dtype)
        return hard @ self.mapping.to(dtype=powerset.dtype)

    def forward(self, powerset: Tensor) -> Tensor:
        return self.to_multilabel(powerset)

    def to_speech(self, powerset: Tensor) -> Tensor:
        """Return hard speech activity after upstream-compatible conversion."""
        return self.to_multilabel(powerset).amax(dim=-1)


__all__ = ["Powerset"]
