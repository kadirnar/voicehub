"""PyTorch-native XTTS v2 discrete autoencoder.

The public XTTS v2 runtime checkpoint does not contain these weights.
Coqui publishes them as a separate ``dvae.pth`` artifact used only by
the GPT fine-tuning data path.  This module owns the graph, while
``dvae_checkpoint.py`` owns the explicit legacy-to-Safetensors boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from voicehub.models.xtts.native.audio import MelSpectrogram
from voicehub.processing.waveform import resample_waveform_kaiser


@dataclass(frozen=True, slots=True)
class XTTS2DVAEConfig:
    """Architecture-bearing fields of Coqui's standalone XTTS v2 DVAE."""

    sample_rate: int = 22_050
    mel_channels: int = 80
    num_tokens: int = 1_024
    codebook_dim: int = 512
    hidden_dim: int = 512
    num_layers: int = 2
    num_resnet_blocks: int = 3
    kernel_size: int = 3
    stride: int = 2
    activation: str = "relu"
    use_transposed_convs: bool = False

    def validate(self) -> None:
        for name in (
                "sample_rate",
                "mel_channels",
                "num_tokens",
                "codebook_dim",
                "hidden_dim",
                "num_layers",
                "kernel_size",
                "stride",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"XTTS v2 DVAE `{name}` must be a positive integer.")
        if (isinstance(self.num_resnet_blocks, bool) or not isinstance(self.num_resnet_blocks, Integral) or
                self.num_resnet_blocks < 0):
            raise ValueError("XTTS v2 DVAE `num_resnet_blocks` must be a non-negative integer.")
        if self.kernel_size % 2 == 0:
            raise ValueError("XTTS v2 DVAE `kernel_size` must be odd.")
        if self.activation not in {"relu", "silu"}:
            raise ValueError("XTTS v2 DVAE activation must be `relu` or `silu`.")
        if not isinstance(self.use_transposed_convs, bool):
            raise TypeError("XTTS v2 DVAE `use_transposed_convs` must be a boolean.")

    @property
    def code_stride_samples(self) -> int:
        """Waveform samples represented by one acoustic code."""
        return 256 * self.stride**self.num_layers


@dataclass(frozen=True, slots=True)
class XTTS2DVAEEncoding:
    """One-codebook output from the frozen XTTS acoustic tokenizer."""

    audio_codes: Tensor
    quantized_latents: Tensor
    encoder_latents: Tensor


@dataclass(frozen=True, slots=True)
class XTTS2DVAEAutoencoderOutput:
    """Explicit autoencoder result kept separate from GPT training."""

    reconstruction: Tensor
    encoding: XTTS2DVAEEncoding


class XTTS2VectorQuantizer(nn.Module):
    """Nearest-neighbour codebook with the published buffer namespace.

    XTTS GPT fine-tuning freezes this module.  Consequently this native
    boundary deliberately does not reproduce the legacy EMA mutation
    that was used when the DVAE itself was originally trained.
    """

    def __init__(self, dimension: int, num_tokens: int) -> None:
        super().__init__()
        self.dimension = int(dimension)
        self.num_tokens = int(num_tokens)
        embed = torch.empty(self.dimension, self.num_tokens)
        nn.init.normal_(embed)
        # These names and shapes exactly match the standalone Coqui artifact.
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(self.num_tokens))
        self.register_buffer("embed_avg", embed.clone())

    def nearest_codes(self, latents: Tensor) -> Tensor:
        if not isinstance(latents, Tensor) or latents.ndim != 3:
            raise ValueError("XTTS v2 DVAE latents must have shape [batch, frames, dimension].")
        if latents.shape[-1] != self.dimension:
            raise ValueError(
                "XTTS v2 DVAE latent dimension mismatch: expected "
                f"{self.dimension}, found {latents.shape[-1]}.")
        flattened = latents.reshape(-1, self.dimension)
        distances = (
            flattened.square().sum(dim=1, keepdim=True) - 2 * flattened @ self.embed +
            self.embed.square().sum(dim=0, keepdim=True))
        return distances.argmin(dim=1).reshape(latents.shape[:-1])

    def embed_codes(self, audio_codes: Tensor) -> Tensor:
        if not isinstance(audio_codes, Tensor) or audio_codes.ndim != 2:
            raise ValueError("XTTS v2 DVAE codes must have shape [batch, frames].")
        if audio_codes.dtype == torch.bool or audio_codes.is_floating_point():
            raise TypeError("XTTS v2 DVAE codes must use an integer dtype.")
        if audio_codes.numel() and bool(((audio_codes < 0) | (audio_codes >= self.num_tokens)).any()):
            raise ValueError("XTTS v2 DVAE code is outside the acoustic vocabulary.")
        return F.embedding(audio_codes.long(), self.embed.transpose(0, 1))

    def forward(self, latents: Tensor) -> tuple[Tensor, Tensor]:
        audio_codes = self.nearest_codes(latents)
        quantized = self.embed_codes(audio_codes)
        return quantized, audio_codes


class _ResidualBlock(nn.Module):

    def __init__(self, channels: int, activation: type[nn.Module]) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1),
            activation(),
            nn.Conv1d(channels, channels, 3, padding=1),
            activation(),
            nn.Conv1d(channels, channels, 1),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs + self.net(inputs)


class _UpsampledConv(nn.Module):

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        *,
        stride: int,
        padding: int,
    ) -> None:
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv1d(
            input_channels,
            output_channels,
            kernel_size,
            padding=padding,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(F.interpolate(
            inputs,
            scale_factor=self.stride,
            mode="nearest",
        ))


class XTTS2DVAE(nn.Module):
    """Standalone XTTS v2 acoustic-code autoencoder.

    The top-level ``encoder``, ``codebook``, and ``decoder`` attributes
    retain the 53-key namespace of Coqui's published ``dvae.pth``.  The
    GPT training path calls :meth:`encode_mel`; reconstruction is
    exposed separately for auditing and future codec work.
    """

    is_stochastic_vae = False

    def __init__(self, config: XTTS2DVAEConfig | None = None) -> None:
        super().__init__()
        self.config = XTTS2DVAEConfig() if config is None else config
        if not isinstance(self.config, XTTS2DVAEConfig):
            raise TypeError("XTTS2DVAE requires an XTTS2DVAEConfig.")
        self.config.validate()
        activation = nn.ReLU if self.config.activation == "relu" else nn.SiLU
        padding = (self.config.kernel_size - 1) // 2

        encoder_channels = [self.config.hidden_dim * 2**index for index in range(self.config.num_layers)]
        encoder_pairs = zip(
            (self.config.mel_channels, *encoder_channels[:-1]),
            encoder_channels,
        )
        encoder_layers: list[nn.Module] = [
            nn.Sequential(
                nn.Conv1d(
                    input_channels,
                    output_channels,
                    self.config.kernel_size,
                    stride=self.config.stride,
                    padding=padding,
                ),
                activation(),
            ) for input_channels, output_channels in encoder_pairs
        ]
        inner_channels = encoder_channels[-1]
        encoder_layers.extend(
            _ResidualBlock(inner_channels, activation) for _ in range(self.config.num_resnet_blocks))
        encoder_layers.append(nn.Conv1d(
            inner_channels,
            self.config.codebook_dim,
            1,
        ))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: list[nn.Module] = [nn.Conv1d(
            self.config.codebook_dim,
            inner_channels,
            1,
        )]
        decoder_layers.extend(
            _ResidualBlock(inner_channels, activation) for _ in range(self.config.num_resnet_blocks))
        decoder_channels = (
            inner_channels,
            *reversed(encoder_channels),
        )
        decoder_pairs = zip(
            decoder_channels[:-1],
            decoder_channels[1:],
        )
        for input_channels, output_channels in decoder_pairs:
            if self.config.use_transposed_convs:
                upsample: nn.Module = nn.ConvTranspose1d(
                    input_channels,
                    output_channels,
                    self.config.kernel_size,
                    stride=self.config.stride,
                    padding=padding,
                )
            else:
                upsample = _UpsampledConv(
                    input_channels,
                    output_channels,
                    self.config.kernel_size,
                    stride=self.config.stride,
                    padding=padding,
                )
            decoder_layers.append(nn.Sequential(upsample, activation()))
        decoder_layers.append(nn.Conv1d(
            self.config.hidden_dim,
            self.config.mel_channels,
            1,
        ))
        self.decoder = nn.Sequential(*decoder_layers)
        self.codebook = XTTS2VectorQuantizer(
            self.config.codebook_dim,
            self.config.num_tokens,
        )

    @property
    def quantizer(self) -> XTTS2VectorQuantizer:
        """Non-registering alias used by the shared codec component view."""
        return self.codebook

    def encode_latents(self, mel_spectrogram: Tensor) -> Tensor:
        if not isinstance(mel_spectrogram, Tensor) or mel_spectrogram.ndim != 3:
            raise ValueError("XTTS v2 DVAE mel input must have shape [batch, mel_channels, frames].")
        if mel_spectrogram.shape[1] != self.config.mel_channels:
            raise ValueError(
                "XTTS v2 DVAE mel-channel mismatch: expected "
                f"{self.config.mel_channels}, found {mel_spectrogram.shape[1]}.")
        if not mel_spectrogram.is_floating_point():
            raise TypeError("XTTS v2 DVAE mel input must be floating-point.")
        if mel_spectrogram.shape[-1] == 0:
            raise ValueError("XTTS v2 DVAE mel input cannot be empty.")
        return self.encoder(mel_spectrogram).transpose(1, 2)

    def encode_mel(self, mel_spectrogram: Tensor) -> XTTS2DVAEEncoding:
        encoder_latents = self.encode_latents(mel_spectrogram)
        quantized, audio_codes = self.codebook(encoder_latents)
        return XTTS2DVAEEncoding(
            audio_codes=audio_codes,
            quantized_latents=quantized,
            encoder_latents=encoder_latents,
        )

    @torch.no_grad()
    def get_codebook_indices(self, mel_spectrogram: Tensor) -> Tensor:
        """Compatibility spelling for Coqui's frozen training boundary."""
        return self.encode_mel(mel_spectrogram).audio_codes

    def decode_latents(self, quantized_latents: Tensor) -> Tensor:
        if not isinstance(quantized_latents, Tensor) or quantized_latents.ndim != 3:
            raise ValueError("XTTS v2 DVAE quantized latents must have shape "
                             "[batch, frames, dimension].")
        if quantized_latents.shape[-1] != self.config.codebook_dim:
            raise ValueError(
                "XTTS v2 DVAE quantized latent dimension mismatch: expected "
                f"{self.config.codebook_dim}, found {quantized_latents.shape[-1]}.")
        return self.decoder(quantized_latents.transpose(1, 2))

    def decode_codes(self, audio_codes: Tensor) -> Tensor:
        return self.decode_latents(self.codebook.embed_codes(audio_codes))

    def encode(self, mel_spectrogram: Tensor) -> Tensor:
        """Encode normalized XTTS mel features into one acoustic codebook."""
        return self.encode_mel(mel_spectrogram).audio_codes

    def decode(self, audio_codes: Tensor) -> Tensor:
        """Decode the one-codebook representation back to normalized mel."""
        return self.decode_codes(audio_codes)

    def autoencode(self, mel_spectrogram: Tensor) -> XTTS2DVAEAutoencoderOutput:
        encoding = self.encode_mel(mel_spectrogram)
        return XTTS2DVAEAutoencoderOutput(
            reconstruction=self.decode_latents(encoding.quantized_latents),
            encoding=encoding,
        )

    def forward(self, mel_spectrogram: Tensor) -> Tensor:
        """Return acoustic codes, the only output consumed by XTTS GPT
        training."""
        return self.encode_mel(mel_spectrogram).audio_codes


class XTTS2DVAEMelProcessor(nn.Module):
    """Frozen, native waveform-to-DVAE-mel preprocessing."""

    def __init__(
        self,
        mel_stats: Tensor,
        *,
        sample_rate: int = 22_050,
        mel_channels: int = 80,
    ) -> None:
        super().__init__()
        if not isinstance(mel_stats, Tensor) or mel_stats.ndim != 1:
            raise ValueError("XTTS v2 DVAE mel statistics must be a rank-one tensor.")
        if mel_stats.shape[0] != mel_channels:
            raise ValueError(
                "XTTS v2 DVAE mel-stat count mismatch: expected "
                f"{mel_channels}, found {mel_stats.shape[0]}.")
        if not mel_stats.is_floating_point():
            raise TypeError("XTTS v2 DVAE mel statistics must be floating-point.")
        if not bool(torch.isfinite(mel_stats).all()) or bool((mel_stats == 0).any()):
            raise ValueError("XTTS v2 DVAE mel statistics must be finite and non-zero.")
        self.sample_rate = int(sample_rate)
        self.mel_channels = int(mel_channels)
        self.register_buffer(
            "mel_stats",
            mel_stats.detach().clone().contiguous(),
            persistent=False,
        )
        self.transform = MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=1_024,
            hop_length=256,
            win_length=1_024,
            n_mels=self.mel_channels,
            f_min=0,
            f_max=8_000,
            power=2.0,
            slaney_norm=True,
        )

    @staticmethod
    def _waveform_batch(waveform: Tensor) -> Tensor:
        if not isinstance(waveform, Tensor):
            raise TypeError("XTTS v2 DVAE waveform input must be a PyTorch tensor.")
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.ndim == 3 and waveform.shape[1] == 1:
            waveform = waveform.squeeze(1)
        elif waveform.ndim != 2:
            raise ValueError(
                "XTTS v2 DVAE waveform must have shape [time], [batch, time], "
                "or [batch, 1, time].")
        if not waveform.is_floating_point():
            raise TypeError("XTTS v2 DVAE waveform must be floating-point.")
        if waveform.shape[-1] < 513:
            raise ValueError("XTTS v2 DVAE waveform must contain at least 513 samples.")
        return waveform

    def forward(
        self,
        waveform: Tensor,
        *,
        sample_rate: int | None = None,
    ) -> Tensor:
        waveform = self._waveform_batch(waveform)
        source_rate = self.sample_rate if sample_rate is None else sample_rate
        if isinstance(source_rate, bool) or not isinstance(source_rate, Integral) or source_rate <= 0:
            raise ValueError("XTTS v2 DVAE waveform sample rate must be positive.")
        if source_rate != self.sample_rate:
            waveform = resample_waveform_kaiser(
                waveform,
                int(source_rate),
                self.sample_rate,
                lowpass_filter_width=64,
                rolloff=0.9475937167399596,
                beta=14.769656459379492,
            )
        mel = self.transform(waveform)
        mel = torch.log(mel.clamp_min(1e-5))
        return mel / self.mel_stats.to(
            device=mel.device,
            dtype=mel.dtype,
        )[None, :, None]


class XTTS2TrainingAudioEncoder(nn.Module):
    """Optional separately loaded data-preparation graph for GPT fine-
    tuning."""

    def __init__(
        self,
        dvae: XTTS2DVAE,
        mel_stats: Tensor,
    ) -> None:
        super().__init__()
        if not isinstance(dvae, XTTS2DVAE):
            raise TypeError("XTTS2TrainingAudioEncoder requires an XTTS2DVAE.")
        self.dvae = dvae
        self.mel_processor = XTTS2DVAEMelProcessor(
            mel_stats,
            sample_rate=dvae.config.sample_rate,
            mel_channels=dvae.config.mel_channels,
        )
        self.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(
        self,
        waveform: Tensor,
        *,
        sample_rate: int | None = None,
    ) -> Tensor:
        mel = self.mel_processor(
            waveform,
            sample_rate=sample_rate,
        )
        reference = next(self.dvae.parameters())
        return self.dvae(mel.to(
            device=reference.device,
            dtype=reference.dtype,
        ))


__all__ = [
    "XTTS2DVAE",
    "XTTS2DVAEAutoencoderOutput",
    "XTTS2DVAEConfig",
    "XTTS2DVAEEncoding",
    "XTTS2DVAEMelProcessor",
    "XTTS2TrainingAudioEncoder",
    "XTTS2VectorQuantizer",
]
