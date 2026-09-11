"""Canonical PyTorch-native audio loading for ASR and VAD backends."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

from voicehub.models.base import BaseSpeechModel


@dataclass(frozen=True)
class AudioInput:
    """Materialized mono waveform with an explicit sampling rate."""

    waveform: Any
    sampling_rate: int
    path: Path | None = None

    def __post_init__(self) -> None:
        if (isinstance(self.sampling_rate, bool) or not isinstance(self.sampling_rate, Integral) or
                self.sampling_rate <= 0):
            raise ValueError("AudioInput `sampling_rate` must be a positive integer.")
        BaseSpeechModel.validate_audio(self.waveform)

    @property
    def duration(self) -> float:
        """Return duration in seconds."""
        shape = getattr(self.waveform, "shape", None)
        sample_count = int(shape[-1]) if shape else len(self.waveform)
        return sample_count / int(self.sampling_rate)


def load_audio(
    audio: AudioInput | Mapping[str, Any] | str | Path | Any,
    *,
    sampling_rate: int | None = None,
    target_sampling_rate: int | None = None,
) -> AudioInput:
    """Load, downmix, and optionally resample one waveform.

    Array and tensor inputs must carry an explicit sampling rate. File
    input uses VoiceHub's standard-library PCM WAVE decoder. Other
    containers must be decoded explicitly before this boundary, which
    keeps every ASR and VAD provider independent from NumPy, SoundFile,
    librosa, and torchaudio.
    """
    # Keep importing the top-level VoiceHub package lightweight. PyTorch is
    # required only when audio is actually materialized, not when users inspect
    # a configuration or registry entry.
    from voicehub.processing.waveform import load_native_audio

    source_path: Path | None = None
    if isinstance(audio, AudioInput):
        if (sampling_rate is not None and int(sampling_rate) != int(audio.sampling_rate)):
            raise ValueError("`sampling_rate` conflicts with the rate stored in "
                             "AudioInput.")
        source_path = audio.path
        native = load_native_audio(
            audio.waveform,
            sampling_rate=int(audio.sampling_rate),
            target_sampling_rate=target_sampling_rate,
        )
    else:
        if (sampling_rate is None and not isinstance(audio, (Mapping, str, Path)) and
                not hasattr(audio, "sampling_rate")):
            raise ValueError("Array and tensor audio inputs require a positive "
                             "`sampling_rate`.")
        native = load_native_audio(
            audio,
            sampling_rate=sampling_rate,
            target_sampling_rate=target_sampling_rate,
        )
    return AudioInput(
        waveform=native.waveform,
        sampling_rate=native.sampling_rate,
        path=source_path or native.path,
    )


__all__ = ["AudioInput", "load_audio"]
