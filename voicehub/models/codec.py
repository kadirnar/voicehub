"""Audio codec task contract, independent of the encoded representation."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
from numbers import Integral
from pathlib import Path
from typing import Any

from voicehub.audio import AudioInput
from voicehub.model import PreTrainedSpeechModel
from voicehub.outputs import AudioOutput, CodecOutput
from voicehub.processing.processor import AudioProcessor
from voicehub.tasks import SpeechTask
from voicehub.training.utils import NATIVE_EXPORT_DIR


def _audio_batch(audio: Any, sampling_rate: int | None, target_rate: int | None = None):
    """Preserve batch and channels; resample each channel independently."""
    import torch

    from voicehub.processing.waveform import load_pcm_wave, normalize_waveform, resample_waveform

    stored_rate = None
    if isinstance(audio, AudioInput):
        stored_rate, audio = audio.sampling_rate, audio.waveform
    elif isinstance(audio, Mapping):
        stored_rate = audio.get("sampling_rate", audio.get("sample_rate"))
        audio = next((audio[key] for key in ("array", "waveform", "audio") if key in audio), None)
    elif isinstance(audio, (str, Path)):
        audio, stored_rate = load_pcm_wave(audio, preserve_channels=True)
    if stored_rate is not None:
        if sampling_rate is not None and sampling_rate != stored_rate:
            raise ValueError("`sampling_rate` conflicts with the rate stored in the audio input.")
        sampling_rate = stored_rate
    if isinstance(sampling_rate, bool) or not isinstance(sampling_rate, Integral) or sampling_rate <= 0:
        raise ValueError("Codec audio requires a positive `sampling_rate`.")
    if audio is None:
        raise ValueError("Audio input cannot be None.")
    waveform = torch.as_tensor(audio)
    if waveform.ndim == 1:
        waveform = waveform[None, None, :]
    elif waveform.ndim == 2:
        waveform = waveform[None, :, :]
    if waveform.ndim != 3 or any(size == 0 for size in waveform.shape):
        raise ValueError(
            "Codec audio must have shape [samples], [channels, samples], or [batch, channels, samples].")
    batch, channels, _ = waveform.shape
    rows = [
        resample_waveform(normalize_waveform(channel), sampling_rate, target_rate or sampling_rate)
        for channel in waveform.reshape(-1, waveform.shape[-1])
    ]
    return torch.stack(rows).reshape(batch, channels, -1), int(target_rate or sampling_rate)


class PreTrainedCodecModel(PreTrainedSpeechModel):
    """Implement ``_encode`` and ``_decode`` for discrete or continuous codecs.

    Public inference preserves segmented scales, ragged codebooks, and
    latent tensors inside ``CodecOutput.codes``. Neural training uses
    the loaded runtime directly, so inference-mode tensors cannot enter
    an autograd graph.
    """

    task = SpeechTask.AUDIO_CODEC
    processor_class = AudioProcessor
    main_input_name = "audio"

    @property
    def checkpoint_source(self) -> str:
        source = Path(self.config.name_or_path).expanduser()
        native = source / NATIVE_EXPORT_DIR
        return str(native) if native.is_dir() else self.config.name_or_path

    def encode(self, audio: Any, *, sampling_rate: int | None = None, **kwargs) -> CodecOutput:
        import torch

        with self._lifecycle_lock, torch.inference_mode():
            # Materialize and validate before allocating checkpoint weights.
            waveform, input_rate = _audio_batch(audio, sampling_rate)
            self.load()
            if self.sample_rate != input_rate:
                waveform, _ = _audio_batch(waveform, input_rate, self.sample_rate)
            parameter = next(iter(self.model.parameters()), None)
            waveform = waveform.to(
                device=self.device,
                dtype=parameter.dtype if parameter is not None else torch.float32,
            )
            codes = self._encode(waveform, **kwargs)
            return CodecOutput(codes, self.sample_rate, waveform.shape[-1], self.config.model_type)

    def decode(self, encoded: CodecOutput, **kwargs) -> AudioOutput:
        import torch

        if not isinstance(encoded, CodecOutput):
            raise TypeError("`encoded` must be a CodecOutput returned by encode().")
        if encoded.model_type != self.config.model_type:
            raise ValueError("Encoded audio belongs to a different codec model type.")
        with self._lifecycle_lock, torch.inference_mode():
            self.load()
            if encoded.sample_rate != self.sample_rate:
                raise ValueError("Encoded audio and the codec have different sample rates.")
            waveform = self._decode(encoded.codes, **kwargs)
            if waveform.shape[-1] < encoded.length:
                raise ValueError("The decoder returned fewer samples than the encoded length.")
            return AudioOutput(waveform[..., :encoded.length], self.sample_rate)

    def __call__(self, audio: Any, *, sampling_rate: int | None = None, **kwargs) -> AudioOutput:
        """Reconstruct audio; use encode() to obtain the intermediate codes."""
        return self.decode(self.encode(audio, sampling_rate=sampling_rate, **kwargs))

    @abstractmethod
    def _encode(self, waveform, **kwargs):
        """Encode a floating-point [batch, channels, samples] tensor."""

    @abstractmethod
    def _decode(self, codes, **kwargs):
        """Decode the codec-owned payload to [batch, channels, samples]."""
