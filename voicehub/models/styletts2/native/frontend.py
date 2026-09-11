"""Explicit native phoneme and audio frontend for StyleTTS 2."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional

from voicehub.audio import AudioInput, load_audio
from voicehub.processing.audio import htk_mel_filter_bank

_PAD = "$"
_PUNCTUATION = ';:,.!?¡¿—…"«»“” '
_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_IPA = (
    "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘ"
    "ɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ")
STYLETTS2_SYMBOLS = tuple(_PAD + _PUNCTUATION + _LETTERS + _IPA)
STYLETTS2_SYMBOL_TO_ID = {symbol: index for index, symbol in enumerate(STYLETTS2_SYMBOLS)}


class NativeStyleTTS2Frontend:
    """Tokenize caller-supplied upstream-compatible phoneme strings.

    VoiceHub intentionally does not guess phonemes from orthographic
    text. The released frontend relies on eSpeak and NLTK behavior that
    is not part of the checkpoint. Callers must run their chosen
    licensed G2P explicitly and pass the resulting, already word-
    separated phoneme string.
    """

    vocabulary_size = len(STYLETTS2_SYMBOLS)
    bos_token_id = 0

    def encode_phonemes(
        self,
        phonemes: str,
        *,
        explicit: bool,
        device: torch.device | str | None = None,
    ) -> Tensor:
        if explicit is not True:
            raise ValueError(
                "StyleTTS 2 raw-text phonemization is deliberately disabled. "
                "Pass the eSpeak-compatible phoneme sequence and set "
                "`text_is_phonemes=True`.")
        if not isinstance(phonemes, str) or not phonemes.strip():
            raise ValueError("`phonemes` must be a non-empty string.")
        unknown = tuple(
            dict.fromkeys(character for character in phonemes if character not in STYLETTS2_SYMBOL_TO_ID))
        if unknown:
            rendered = ", ".join(repr(item) for item in unknown[:12])
            raise ValueError(
                "StyleTTS 2 phonemes contain symbols outside the released "
                f"178-token inventory: {rendered}.")
        token_ids = [
            self.bos_token_id,
            *(STYLETTS2_SYMBOL_TO_ID[character] for character in phonemes),
        ]
        return torch.tensor(
            token_ids,
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)

    def normalize_input_ids(
        self,
        input_ids: Tensor | Sequence[int] | Sequence[Sequence[int]],
        *,
        device: torch.device | str | None = None,
    ) -> Tensor:
        tensor = (input_ids if isinstance(input_ids, Tensor) else torch.as_tensor(input_ids))
        if tensor.dtype == torch.bool or tensor.is_floating_point():
            raise TypeError("StyleTTS 2 `input_ids` must use an integer dtype.")
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2 or tensor.shape[0] != 1 or tensor.shape[1] < 2:
            raise ValueError(
                "StyleTTS 2 inference IDs must have shape [1, text] with at "
                "least the BOS and one phoneme.")
        tensor = tensor.to(device=device, dtype=torch.long)
        if bool(((tensor < 0) | (tensor >= self.vocabulary_size)).any()):
            raise ValueError("StyleTTS 2 input IDs are outside its vocabulary.")
        if int(tensor[0, 0]) != self.bos_token_id:
            raise ValueError("StyleTTS 2 input IDs must begin with BOS ID 0.")
        return tensor


class StyleTTS2MelSpectrogram(nn.Module):
    """Torch-only equivalent of the released torchaudio mel transform."""

    def __init__(
        self,
        *,
        sample_rate: int = 24_000,
        n_fft: int = 2_048,
        win_length: int = 1_200,
        hop_length: int = 300,
        n_mels: int = 80,
    ) -> None:
        super().__init__()
        for name, value in (
            ("sample_rate", sample_rate),
            ("n_fft", n_fft),
            ("win_length", win_length),
            ("hop_length", hop_length),
            ("n_mels", n_mels),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"`{name}` must be a positive integer.")
        if not hop_length <= win_length <= n_fft:
            raise ValueError("Mel dimensions must satisfy hop_length <= win_length <= n_fft.")
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.register_buffer(
            "window",
            torch.hann_window(win_length),
            persistent=False,
        )
        self.register_buffer(
            "mel_filters",
            htk_mel_filter_bank(
                sample_rate=sample_rate,
                n_fft=n_fft,
                n_mels=n_mels,
                dtype=torch.float32,
            ),
            persistent=False,
        )

    def forward(self, waveform: Tensor) -> Tensor:
        if not isinstance(waveform, Tensor):
            raise TypeError("`waveform` must be a PyTorch tensor.")
        if waveform.ndim not in (1, 2):
            raise ValueError("`waveform` must have shape [time] or [batch, time].")
        if not waveform.is_floating_point():
            raise TypeError("`waveform` must use a floating-point dtype.")
        if waveform.shape[-1] < 2:
            raise ValueError("`waveform` is too short for mel extraction.")
        computation_dtype = (
            torch.float32 if waveform.dtype in {torch.float16, torch.bfloat16} else waveform.dtype)
        source = waveform.to(dtype=computation_dtype)
        window = self.window.to(
            device=source.device,
            dtype=computation_dtype,
        )
        spectrum = torch.stft(
            source,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        ).abs().square()
        filters = self.mel_filters.to(
            device=source.device,
            dtype=computation_dtype,
        )
        return torch.matmul(filters.transpose(0, 1), spectrum)


def trim_reference_silence(
    waveform: Tensor,
    *,
    top_db: float = 30.0,
    frame_length: int = 2_048,
    hop_length: int = 512,
) -> Tensor:
    """Frame-RMS silence trimming matching the released 30 dB policy."""
    if (not isinstance(waveform, Tensor) or waveform.ndim != 1 or not waveform.is_floating_point()):
        raise TypeError("Reference waveform must be a rank-one float tensor.")
    if waveform.numel() == 0:
        raise ValueError("Reference waveform cannot be empty.")
    if (isinstance(top_db, bool) or not isinstance(top_db, (int, float)) or
            not math.isfinite(float(top_db)) or top_db < 0):
        raise ValueError("`top_db` must be finite and non-negative.")
    padding = frame_length // 2
    padded = functional.pad(
        waveform,
        (padding, padding),
        mode="constant",
        value=0.0,
    )
    if padded.numel() < frame_length:
        padded = functional.pad(
            padded,
            (0, frame_length - padded.numel()),
        )
    frames = padded.unfold(0, frame_length, hop_length)
    rms = frames.square().mean(dim=-1).sqrt()
    peak = rms.max()
    if not bool(peak > 0):
        raise ValueError("StyleTTS 2 reference audio is silent.")
    threshold = peak * (10.0**(-float(top_db) / 20.0))
    active = torch.nonzero(rms > threshold, as_tuple=False).flatten()
    if active.numel() == 0:
        raise ValueError("StyleTTS 2 reference audio has no samples above the trim threshold.")
    start = max(0, int(active[0]) * hop_length)
    stop = min(
        waveform.numel(),
        (int(active[-1]) + 1) * hop_length,
    )
    trimmed = waveform[start:stop]
    if trimmed.numel() == 0:
        raise ValueError("StyleTTS 2 reference audio contains no samples after trimming.")
    return trimmed


def load_style_reference(
    audio: str | Path | AudioInput | Any,
    *,
    sample_rate: int,
) -> Tensor:
    loaded = load_audio(audio, target_sampling_rate=sample_rate)
    return trim_reference_silence(loaded.waveform.float())


__all__ = [
    "NativeStyleTTS2Frontend",
    "STYLETTS2_SYMBOLS",
    "STYLETTS2_SYMBOL_TO_ID",
    "StyleTTS2MelSpectrogram",
    "load_style_reference",
    "trim_reference_silence",
]
