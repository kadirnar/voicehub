"""Native waveform frontend and joint audio/text processor for Parakeet TDT."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from voicehub.models.asr_parakeet_tdt.native.tokenization import ParakeetTokenizer
from voicehub.processing.audio import mel_filter_bank

LOG_ZERO_GUARD = 2**-24
NORMALIZATION_EPSILON = 1e-5


class ParakeetFeatureExtractor:
    """Official 16 kHz log-mel frontend implemented with PyTorch only."""

    model_input_names = ("input_features", "attention_mask")

    def __init__(
        self,
        *,
        feature_size: int = 128,
        sampling_rate: int = 16_000,
        hop_length: int = 160,
        n_fft: int = 512,
        win_length: int = 400,
        preemphasis: float = 0.97,
        padding_value: float = 0.0,
    ) -> None:
        for name, value in (
            ("feature_size", feature_size),
            ("sampling_rate", sampling_rate),
            ("hop_length", hop_length),
            ("n_fft", n_fft),
            ("win_length", win_length),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"`{name}` must be a positive integer.")
        if win_length > n_fft:
            raise ValueError("`win_length` cannot exceed `n_fft`.")
        if (isinstance(preemphasis, bool) or not isinstance(preemphasis, (int, float)) or
                not math.isfinite(preemphasis) or not 0.0 <= preemphasis < 1.0):
            raise ValueError("`preemphasis` must be finite and in [0, 1).")
        self.feature_size = feature_size
        self.sampling_rate = sampling_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.win_length = win_length
        self.preemphasis = float(preemphasis)
        self.padding_value = float(padding_value)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> ParakeetFeatureExtractor:
        if not isinstance(values, Mapping):
            raise TypeError("Parakeet feature extractor config must be a mapping.")
        expected_type = values.get(
            "feature_extractor_type",
            "ParakeetFeatureExtractor",
        )
        if expected_type not in {
                "ParakeetFeatureExtractor",
                "VoiceHubParakeetFeatureExtractor",
        }:
            raise ValueError("Native Parakeet TDT requires a Parakeet feature extractor.")
        allowed = {
            "feature_size",
            "sampling_rate",
            "hop_length",
            "n_fft",
            "win_length",
            "preemphasis",
            "padding_value",
        }
        return cls(**{name: values[name] for name in allowed if name in values})

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_extractor_type": "VoiceHubParakeetFeatureExtractor",
            "feature_size": self.feature_size,
            "sampling_rate": self.sampling_rate,
            "hop_length": self.hop_length,
            "n_fft": self.n_fft,
            "win_length": self.win_length,
            "preemphasis": self.preemphasis,
            "padding_value": self.padding_value,
            "padding_side": "right",
            "return_attention_mask": True,
        }

    @staticmethod
    def _waveforms(
        audio: Any,
        *,
        device: torch.device | str | None,
    ) -> tuple[torch.Tensor, ...]:
        if isinstance(audio, torch.Tensor):
            if audio.ndim == 1:
                values = (audio, )
            elif audio.ndim == 2:
                values = tuple(audio[index] for index in range(audio.shape[0]))
            else:
                raise ValueError("Parakeet audio tensor must have shape [time] or [batch, time].")
        elif isinstance(audio, Sequence) and not isinstance(audio, (str, bytes)):
            if not audio:
                raise ValueError("Parakeet audio cannot be empty.")
            first = audio[0]
            if isinstance(first, (int, float)):
                values = (torch.as_tensor(audio), )
            else:
                values = tuple(torch.as_tensor(value) for value in audio)
        else:
            values = (torch.as_tensor(audio), )
        normalized = []
        for waveform in values:
            if waveform.ndim != 1:
                raise ValueError(
                    "Parakeet processor accepts mono waveforms only; downmix at "
                    "the audio-loading boundary.")
            if waveform.numel() < 2:
                raise ValueError("Parakeet audio must contain at least two samples.")
            if not torch.isfinite(waveform).all():
                raise ValueError("Parakeet audio contains NaN or infinity.")
            normalized.append(waveform.to(device=device, dtype=torch.float32))
        return tuple(normalized)

    def __call__(
        self,
        audio: Any,
        *,
        sampling_rate: int,
        device: torch.device | str | None = None,
    ) -> dict[str, torch.Tensor]:
        if sampling_rate != self.sampling_rate:
            raise ValueError(
                f"Parakeet expects {self.sampling_rate} Hz audio; received "
                f"{sampling_rate} Hz.")
        waveforms = self._waveforms(audio, device=device)
        lengths = torch.tensor(
            [value.numel() for value in waveforms],
            device=waveforms[0].device,
            dtype=torch.long,
        )
        feature_lengths = torch.div(
            lengths,
            self.hop_length,
            rounding_mode="floor",
        )
        if torch.any(feature_lengths < 2):
            raise ValueError("Parakeet audio must produce at least two valid feature frames.")
        maximum = int(lengths.max())
        padded = torch.full(
            (len(waveforms), maximum),
            self.padding_value,
            device=waveforms[0].device,
            dtype=torch.float32,
        )
        for index, waveform in enumerate(waveforms):
            padded[index, :waveform.numel()] = waveform
        sample_mask = (torch.arange(maximum, device=padded.device)[None, :] < lengths[:, None])
        emphasized = torch.cat(
            (
                padded[:, :1],
                padded[:, 1:] - self.preemphasis * padded[:, :-1],
            ),
            dim=1,
        )
        emphasized = emphasized.masked_fill(~sample_mask, 0.0)
        window = torch.hann_window(
            self.win_length,
            periodic=False,
            device=padded.device,
            dtype=torch.float32,
        )
        spectrum = torch.stft(
            emphasized,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=True,
            pad_mode="constant",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        power = spectrum.abs().square()
        filters = mel_filter_bank(
            sample_rate=self.sampling_rate,
            n_fft=self.n_fft,
            n_mels=self.feature_size,
            # librosa constructs the published Slaney bank in double
            # precision and stores float32. Mirroring that order avoids
            # accumulating frontend drift without depending on NumPy/librosa.
            dtype=torch.float64,
            device=padded.device,
        ).to(torch.float32)
        features = torch.matmul(filters, power)
        features = torch.log(features + LOG_ZERO_GUARD).transpose(1, 2)
        frame_mask = (
            torch.arange(features.shape[1], device=features.device)[None, :] < feature_lengths[:, None])
        expanded_mask = frame_mask.unsqueeze(-1)
        masked = features * expanded_mask
        mean = masked.sum(dim=1) / feature_lengths.unsqueeze(-1)
        variance = ((masked - mean.unsqueeze(1)).square() * expanded_mask).sum(dim=1) / (feature_lengths -
                                                                                         1).unsqueeze(-1)
        standard_deviation = torch.sqrt(variance)
        features = (features - mean.unsqueeze(1)) / (standard_deviation.unsqueeze(1) + NORMALIZATION_EPSILON)
        features = features * expanded_mask
        return {
            "input_features": features,
            "attention_mask": frame_mask,
        }


class ParakeetProcessor:
    """Compose the native feature extractor and tokenizer."""

    def __init__(
        self,
        feature_extractor: ParakeetFeatureExtractor,
        tokenizer: ParakeetTokenizer,
        *,
        blank_token: str = "<blank>",
        decoder_type: str = "tdt",
        subsampling_factor: int = 8,
    ) -> None:
        if not isinstance(feature_extractor, ParakeetFeatureExtractor):
            raise TypeError("Invalid Parakeet feature extractor.")
        if not isinstance(tokenizer, ParakeetTokenizer):
            raise TypeError("Invalid Parakeet tokenizer.")
        if blank_token != "<blank>":
            raise ValueError("Native Parakeet currently requires blank token '<blank>'.")
        if tokenizer.token_piece(tokenizer.blank_token_id) != blank_token:
            raise ValueError("Parakeet tokenizer blank-token contract disagrees.")
        if decoder_type != "tdt":
            raise ValueError("Native Parakeet processor supports TDT decoding only.")
        if (isinstance(subsampling_factor, bool) or not isinstance(subsampling_factor, int) or
                subsampling_factor < 1):
            raise ValueError("`subsampling_factor` must be a positive integer.")
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.blank_token = blank_token
        self.blank_token_id = tokenizer.blank_token_id
        self.decoder_type = decoder_type
        self.subsampling_factor = subsampling_factor

    @classmethod
    def from_files(
        cls,
        processor_config_path: str | Path,
        tokenizer_path: str | Path,
        tokenizer_config_path: str | Path,
    ) -> ParakeetProcessor:
        path = Path(processor_config_path)
        values = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError("Parakeet processor config must be an object.")
        if values.get("processor_class") not in {
                "ParakeetProcessor",
                "VoiceHubParakeetProcessor",
        }:
            raise ValueError("Unsupported Parakeet processor class.")
        feature_values = values.get("feature_extractor")
        if not isinstance(feature_values, dict):
            raise ValueError("Parakeet processor config requires `feature_extractor`.")
        blank_token = values.get("blank_token", "<blank>")
        decoder_type = values.get("decoder_type", "tdt")
        return cls(
            ParakeetFeatureExtractor.from_dict(feature_values),
            ParakeetTokenizer.from_files(
                tokenizer_path,
                tokenizer_config_path,
                blank_token=blank_token,
            ),
            blank_token=blank_token,
            decoder_type=decoder_type or "tdt",
            subsampling_factor=int(values.get("subsampling_factor", 8)),
        )

    @property
    def frame_seconds(self) -> float:
        return (
            self.feature_extractor.hop_length / self.feature_extractor.sampling_rate *
            self.subsampling_factor)

    def __call__(
        self,
        audio: Any | None = None,
        *,
        text: str | Sequence[str] | None = None,
        sampling_rate: int = 16_000,
        device: torch.device | str | None = None,
    ) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {}
        if audio is not None:
            result.update(self.feature_extractor(
                audio,
                sampling_rate=sampling_rate,
                device=device,
            ))
        if text is None:
            if audio is None:
                raise ValueError("Parakeet processor requires audio, text, or both.")
            return result
        texts = (text, ) if isinstance(text, str) else tuple(text)
        if not texts or any(not isinstance(value, str) for value in texts):
            raise TypeError("Parakeet training text must be a string or sequence.")
        encoded = [self.tokenizer.encode(value).input_ids for value in texts]
        if any(not value for value in encoded):
            raise ValueError("Parakeet TDT training transcripts cannot be empty.")
        forbidden_targets = {
            self.tokenizer.pad_token_id,
            self.tokenizer.blank_token_id,
        }
        if any(token_id in forbidden_targets for row in encoded for token_id in row):
            raise ValueError("Parakeet TDT transcripts cannot contain literal pad or blank "
                             "tokens.")
        maximum = max(len(value) for value in encoded)
        labels = torch.full(
            (len(encoded), maximum),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        decoder_ids = torch.full(
            (len(encoded), maximum + 1),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        decoder_ids[:, 0] = self.blank_token_id
        for index, values in enumerate(encoded):
            row = torch.tensor(values, dtype=torch.long, device=device)
            labels[index, :len(values)] = row
            decoder_ids[index, 1:len(values) + 1] = row
        if audio is not None and result["input_features"].shape[0] != len(texts):
            raise ValueError("Parakeet audio and transcript batches must have equal length.")
        result["labels"] = labels
        result["decoder_input_ids"] = decoder_ids
        return result

    def save_pretrained(self, directory: str | Path) -> Path:
        destination = Path(directory).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(destination)
        processor_path = destination / "processor_config.json"
        processor_path.write_text(
            json.dumps(
                {
                    "blank_token": self.blank_token,
                    "decoder_type": "tdt",
                    "feature_extractor": self.feature_extractor.to_dict(),
                    "processor_class": "VoiceHubParakeetProcessor",
                    "subsampling_factor": self.subsampling_factor,
                },
                indent=2,
                sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )
        return processor_path


__all__ = ["ParakeetFeatureExtractor", "ParakeetProcessor"]
