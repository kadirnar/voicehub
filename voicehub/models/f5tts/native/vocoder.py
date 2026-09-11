"""Native implementation of the released 24 kHz Vocos decoder."""

from __future__ import annotations

import torch
from torch import nn

from voicehub.models.f5tts.native.audio import htk_mel_filter_bank


class NativeSpectrogram(nn.Module):
    """State-compatible subset of ``torchaudio.transforms.Spectrogram``."""

    def __init__(
        self,
        *,
        n_fft: int,
        hop_length: int,
        win_length: int,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length))

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        window = self.window.to(device=waveform.device, dtype=waveform.dtype)
        return torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        ).abs()


class NativeMelScale(nn.Module):
    """State-compatible subset of ``torchaudio.transforms.MelScale``."""

    def __init__(
        self,
        *,
        sample_rate: int,
        n_fft: int,
        n_mels: int,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "fb",
            htk_mel_filter_bank(
                sample_rate=sample_rate,
                n_fft=n_fft,
                n_mels=n_mels,
            ),
        )

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        filters = self.fb.to(device=spectrum.device, dtype=spectrum.dtype)
        return torch.matmul(spectrum.transpose(-1, -2), filters).transpose(-1, -2)


class NativeMelTransform(nn.Module):

    def __init__(
        self,
        *,
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        n_mels: int,
    ) -> None:
        super().__init__()
        self.spectrogram = NativeSpectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
        )
        self.mel_scale = NativeMelScale(
            sample_rate=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.mel_scale(self.spectrogram(waveform))


class MelSpectrogramFeatures(nn.Module):
    """Vocos feature extractor with the released module namespace."""

    def __init__(
        self,
        *,
        sample_rate: int = 24_000,
        n_fft: int = 1_024,
        hop_length: int = 256,
        n_mels: int = 100,
    ) -> None:
        super().__init__()
        self.mel_spec = NativeMelTransform(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.mel_spec(waveform).clamp_min(1e-7).log()


class VocosConvNeXtBlock(nn.Module):

    def __init__(
        self,
        *,
        dim: int,
        intermediate_dim: int,
        layer_scale_init_value: float,
    ) -> None:
        super().__init__()
        self.dwconv = nn.Conv1d(
            dim,
            dim,
            kernel_size=7,
            padding=3,
            groups=dim,
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.dwconv(hidden_states).transpose(1, 2)
        hidden_states = self.norm(hidden_states)
        hidden_states = self.pwconv1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.pwconv2(hidden_states)
        hidden_states = self.gamma * hidden_states
        return residual + hidden_states.transpose(1, 2)


class VocosBackbone(nn.Module):

    def __init__(
        self,
        *,
        input_channels: int = 100,
        dim: int = 512,
        intermediate_dim: int = 1_536,
        num_layers: int = 8,
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.embed = nn.Conv1d(input_channels, dim, kernel_size=7, padding=3)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.convnext = nn.ModuleList(
            VocosConvNeXtBlock(
                dim=dim,
                intermediate_dim=intermediate_dim,
                layer_scale_init_value=1 / num_layers,
            ) for _ in range(num_layers))
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed(features)
        hidden_states = self.norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        for block in self.convnext:
            hidden_states = block(hidden_states)
        return self.final_layer_norm(hidden_states.transpose(1, 2))


class NativeISTFT(nn.Module):

    def __init__(
        self,
        *,
        n_fft: int,
        hop_length: int,
        win_length: int,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length))

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        return torch.istft(
            spectrum,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(
                device=spectrum.device,
                dtype=spectrum.real.dtype,
            ),
            center=True,
        )


class ISTFTHead(nn.Module):

    def __init__(
        self,
        *,
        dim: int = 512,
        n_fft: int = 1_024,
        hop_length: int = 256,
    ) -> None:
        super().__init__()
        self.out = nn.Linear(dim, n_fft + 2)
        self.istft = NativeISTFT(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        predicted = self.out(hidden_states).transpose(1, 2)
        # Complex BF16 tensors are unsupported and cuFFT/istft is more
        # numerically reliable in FP32. Keep the learned Vocos backbone in
        # the requested inference dtype, then cross this spectral boundary
        # explicitly.
        magnitude, phase = predicted.float().chunk(2, dim=1)
        magnitude = magnitude.exp().clamp_max(1e2)
        spectrum = magnitude * (
            torch.cos(phase) + torch.complex(
                torch.zeros_like(phase),
                torch.sin(phase),
            ))
        return self.istft(spectrum)


class NativeVocos(nn.Module):
    """Complete state-compatible Vocos mel-24khz graph."""

    def __init__(self) -> None:
        super().__init__()
        self.feature_extractor = MelSpectrogramFeatures()
        self.backbone = VocosBackbone()
        self.head = ISTFTHead()

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[1] != 100:
            raise ValueError("Vocos expects `[batch, 100, frames]` log-mel input.")
        return self.head(self.backbone(features))

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.decode(self.feature_extractor(waveform))


__all__ = [
    "ISTFTHead",
    "MelSpectrogramFeatures",
    "NativeISTFT",
    "NativeMelScale",
    "NativeMelTransform",
    "NativeSpectrogram",
    "NativeVocos",
    "VocosBackbone",
    "VocosConvNeXtBlock",
]
