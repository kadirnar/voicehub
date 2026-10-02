"""VoiceHub-native ECAPA speaker conditioning for ZONOS2."""

from __future__ import annotations

from math import ceil, gcd, pi
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from voicehub.architectures.qwen3_tts.configuration import Qwen3TTSSpeakerEncoderConfig
from voicehub.architectures.qwen3_tts.modeling import Qwen3TTSSpeakerEncoder
from voicehub.architectures.zonos2.artifacts import Zonos2SpeakerArtifacts, resolve_zonos2_speaker_artifacts
from voicehub.checkpointing import SafeTensorReader
from voicehub.checkpointing.errors import CheckpointCompatibilityError
from voicehub.hub import read_json_file
from voicehub.processing import load_native_audio, mel_filter_bank

ZONOS2_SPEAKER_SAMPLE_RATE = 24_000


def resample_zonos2_speaker_audio(
    waveform: Tensor,
    source_rate: int,
    target_rate: int = ZONOS2_SPEAKER_SAMPLE_RATE,
    *,
    lowpass_filter_width: int = 6,
    rolloff: float = 0.99,
) -> Tensor:
    """Resample like upstream's ``torchaudio.transforms.Resample`` default.

    Upstream ZONOS2 resamples reference audio with a Hann-windowed sinc
    polyphase filter (``sinc_interp_hann``) whose kernel is built in
    float64 and applied in float32. The speaker embedding is sensitive to
    the interpolation filter, so this mirrors that kernel exactly.
    """
    if not isinstance(waveform, Tensor) or waveform.ndim != 1:
        raise ValueError("ZONOS2 speaker waveform must be a rank-one tensor.")
    source_rate = int(source_rate)
    target_rate = int(target_rate)
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("ZONOS2 speaker sampling rates must be positive.")
    waveform = waveform.float()
    if source_rate == target_rate:
        return waveform
    divisor = gcd(source_rate, target_rate)
    original = source_rate // divisor
    target = target_rate // divisor
    base = min(original, target) * rolloff
    width = ceil(lowpass_filter_width * original / base)
    indices = torch.arange(
        -width,
        width + original,
        dtype=torch.float64,
        device=waveform.device,
    )[None, None] / original
    times = torch.arange(
        0,
        -target,
        -1,
        dtype=torch.float64,
        device=waveform.device,
    )[:, None, None] / target + indices
    times = (times * base).clamp(-lowpass_filter_width, lowpass_filter_width)
    window = torch.cos(times * pi / lowpass_filter_width / 2)**2
    times = times * pi
    kernel = torch.where(
        times == 0,
        torch.ones_like(times),
        times.sin() / times,
    ) * window * (base / original)
    length = waveform.shape[-1]
    padded = torch.nn.functional.pad(
        waveform.view(1, 1, length),
        (width, width + original),
    )
    resampled = torch.nn.functional.conv1d(
        padded,
        kernel.to(torch.float32),
        stride=original,
    ).transpose(1, 2).reshape(-1)
    return resampled[:ceil(target * length / original)].contiguous()


def zonos2_speaker_mel(waveform: Tensor) -> Tensor:
    """Compute the encoder's exact 24-kHz, 128-bin log-mel frontend."""
    if not isinstance(waveform, Tensor) or waveform.ndim != 1:
        raise ValueError("ZONOS2 speaker waveform must be a rank-one tensor.")
    if waveform.numel() < 769:
        raise ValueError("ZONOS2 reference audio is too short for the 1024-point "
                         "reflection-padded STFT.")
    waveform = waveform.float().unsqueeze(0)
    padding = (1_024 - 256) // 2
    waveform = torch.nn.functional.pad(
        waveform.unsqueeze(1),
        (padding, padding),
        mode="reflect",
    ).squeeze(1)
    spectrum = torch.stft(
        waveform,
        n_fft=1_024,
        hop_length=256,
        win_length=1_024,
        window=torch.hann_window(
            1_024,
            device=waveform.device,
            dtype=waveform.dtype,
        ),
        center=False,
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    magnitude = spectrum.abs()
    filters = mel_filter_bank(
        sample_rate=24_000,
        n_fft=1_024,
        n_mels=128,
        minimum_frequency=0.0,
        maximum_frequency=12_000.0,
        dtype=magnitude.dtype,
        device=magnitude.device,
    )
    return torch.log(torch.matmul(filters, magnitude).clamp_min(1e-5)).transpose(1, 2)


def _load_speaker_checkpoint(
    model: Qwen3TTSSpeakerEncoder,
    checkpoint: str | Path,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> None:
    expected = {name: tuple(value.shape) for name, value in model.state_dict(keep_vars=True).items()}
    with SafeTensorReader(checkpoint) as reader:
        actual = set(reader.keys())
        missing = sorted(set(expected) - actual)
        unexpected = sorted(actual - set(expected))
        mismatched = sorted((
            name,
            reader.tensor_shape(name),
            expected[name],
        ) for name in set(expected) & actual if reader.tensor_shape(name) != expected[name])
        if missing or unexpected or mismatched:
            raise CheckpointCompatibilityError(
                "ZONOS2 speaker checkpoint is incompatible: "
                f"missing={missing[:8]!r}, unexpected={unexpected[:8]!r}, "
                f"shape_mismatches={mismatched[:8]!r}.")
        with torch.no_grad():
            for name in sorted(expected):
                value = reader.get_tensor(name)
                target_dtype = dtype if value.is_floating_point() else value.dtype
                model.load_state_dict(
                    {name: value.to(
                        device=device,
                        dtype=target_dtype,
                    )},
                    strict=False,
                    assign=True,
                )
    remaining = [name for name, value in model.state_dict().items() if value.device.type == "meta"]
    if remaining:
        raise CheckpointCompatibilityError(
            "ZONOS2 speaker load left meta tensors: " + ", ".join(remaining[:8]))


def load_zonos2_speaker_encoder(
    *,
    artifacts: Zonos2SpeakerArtifacts | None = None,
    source: str | None = None,
    revision: str | None = None,
    cache_dir: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
    verify_integrity: bool = False,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[Qwen3TTSSpeakerEncoder, Zonos2SpeakerArtifacts]:
    """Load the pinned standalone ECAPA graph without Transformers."""
    if artifacts is None:
        keyword = {} if source is None else {"source": source}
        artifacts = resolve_zonos2_speaker_artifacts(
            **keyword,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
            verify_integrity=verify_integrity,
        )
    config_values = read_json_file(artifacts.config)
    config = Qwen3TTSSpeakerEncoderConfig.from_dict(config_values)
    with torch.device("meta"):
        model = Qwen3TTSSpeakerEncoder(
            config,
            initialize=False,
        )
    _load_speaker_checkpoint(
        model,
        artifacts.checkpoint,
        device=device,
        dtype=dtype,
    )
    model.eval()
    model.requires_grad_(False)
    return model, artifacts


@torch.inference_mode()
def extract_zonos2_speaker_embedding(
    model: Qwen3TTSSpeakerEncoder,
    audio: Any,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Load/resample audio and return the unnormalized ECAPA x-vector."""
    model_device = next(model.parameters()).device
    target_device = model_device if device is None else torch.device(device)
    target_dtype = next(model.parameters()).dtype if dtype is None else dtype
    # Decode at the native rate, then resample with upstream's filter: the
    # generic VoiceHub resampler uses a different windowed-sinc kernel.
    loaded = load_native_audio(audio)
    waveform = resample_zonos2_speaker_audio(
        loaded.waveform.to(device=target_device),
        loaded.sampling_rate,
    )
    features = zonos2_speaker_mel(waveform).to(dtype=target_dtype)
    return model(features)


__all__ = [
    "ZONOS2_SPEAKER_SAMPLE_RATE",
    "extract_zonos2_speaker_embedding",
    "load_zonos2_speaker_encoder",
    "resample_zonos2_speaker_audio",
    "zonos2_speaker_mel",
]
