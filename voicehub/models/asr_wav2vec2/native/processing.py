"""Native raw-waveform processing for the Wav2Vec2 encoder family."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from voicehub.hub import read_json_file, write_json_file


class Wav2Vec2FeatureExtractor:
    """Normalize and right-pad raw mono waveforms for native encoders."""

    def __init__(
        self,
        *,
        sampling_rate: int = 16_000,
        do_normalize: bool = True,
        padding_value: float = 0.0,
        return_attention_mask: bool = False,
    ) -> None:
        if (isinstance(sampling_rate, bool) or not isinstance(sampling_rate, int) or sampling_rate <= 0):
            raise ValueError("`sampling_rate` must be a positive integer.")
        if not isinstance(do_normalize, bool):
            raise TypeError("`do_normalize` must be a boolean.")
        if (isinstance(padding_value, bool) or not isinstance(padding_value, (int, float)) or
                not math.isfinite(float(padding_value))):
            raise ValueError("`padding_value` must be a finite real number.")
        if not isinstance(return_attention_mask, bool):
            raise TypeError("`return_attention_mask` must be a boolean.")
        self.sampling_rate = sampling_rate
        self.do_normalize = do_normalize
        self.padding_value = float(padding_value)
        self.return_attention_mask = return_attention_mask

    @classmethod
    def from_preprocessor_config(
        cls,
        path: str | Path | None,
        *,
        default_sampling_rate: int = 16_000,
    ) -> Wav2Vec2FeatureExtractor:
        """Build from an official preprocessor JSON file or defaults."""
        values = {} if path is None else read_json_file(path)
        feature_size = values.get("feature_size", 1)
        if feature_size != 1:
            raise ValueError("Native Wav2Vec2 consumes raw mono waveforms and requires "
                             "`feature_size=1`.")
        if values.get("padding_side", "right") != "right":
            raise ValueError("Native Wav2Vec2 requires right-padded waveform batches.")
        return cls(
            sampling_rate=values.get(
                "sampling_rate",
                default_sampling_rate,
            ),
            do_normalize=values.get("do_normalize", True),
            padding_value=values.get("padding_value", 0.0),
            return_attention_mask=values.get(
                "return_attention_mask",
                False,
            ),
        )

    def normalize(self, waveform: Any) -> Any:
        """Apply official zero-mean/unit-variance normalization."""
        import torch

        if not isinstance(waveform, torch.Tensor) or waveform.ndim != 1:
            raise ValueError("Wav2Vec2 waveform must be a rank-one tensor.")
        if not waveform.is_floating_point():
            raise TypeError("Wav2Vec2 waveform must use a floating-point dtype.")
        if waveform.numel() == 0:
            raise ValueError("Wav2Vec2 waveform cannot be empty.")
        values = waveform.float()
        if not self.do_normalize:
            return values
        mean = values.mean()
        variance = values.var(unbiased=False)
        return (values - mean) / torch.sqrt(variance + 1e-7)

    def prepare_audio_batch(
        self,
        waveforms: Sequence[Any],
    ) -> dict[str, Any]:
        """Normalize and right-pad a non-empty waveform collection."""
        import torch

        values = tuple(waveforms)
        if not values:
            raise ValueError("Wav2Vec2 audio batch cannot be empty.")
        normalized = tuple(self.normalize(waveform) for waveform in values)
        maximum = max(waveform.numel() for waveform in normalized)
        input_values = normalized[0].new_full(
            (len(normalized), maximum),
            self.padding_value,
        )
        attention_mask = torch.zeros(
            (len(normalized), maximum),
            dtype=torch.bool,
            device=input_values.device,
        )
        for index, waveform in enumerate(normalized):
            length = waveform.numel()
            input_values[index, :length] = waveform
            attention_mask[index, :length] = True
        return {
            "input_values": input_values,
            "attention_mask": attention_mask,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return official feature-extractor fields."""
        return {
            "do_normalize": self.do_normalize,
            "feature_size": 1,
            "padding_side": "right",
            "padding_value": self.padding_value,
            "return_attention_mask": self.return_attention_mask,
            "sampling_rate": self.sampling_rate,
            "processor_class": "Wav2Vec2FeatureExtractor",
        }

    def save_pretrained(self, directory: str | Path) -> None:
        """Write the native preprocessing contract."""
        destination = Path(directory).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        write_json_file(
            destination / "preprocessor_config.json",
            self.to_dict(),
        )


__all__ = ["Wav2Vec2FeatureExtractor"]
