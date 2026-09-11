"""Checkpoint-exact native GPT-SoVITS classic S2 graphs."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional

from voicehub.models.gptsovits.native.configuration import GPTSoVITSS2Config
from voicehub.models.gptsovits.native.quantizer import ResidualVectorQuantizer
from voicehub.models.gptsovits.native.style import MRTE, MelStyleEncoder
from voicehub.models.inflecttts.native import attentions, commons
from voicehub.models.inflecttts.native.modeling import DiscriminatorP, DiscriminatorS, Generator
from voicehub.models.inflecttts.native.modeling import PosteriorEncoder as _VITSPosteriorEncoder
from voicehub.models.inflecttts.native.modeling import ResidualCouplingBlock


class TextEncoder(nn.Module):

    def __init__(self, config: GPTSoVITSS2Config) -> None:
        super().__init__()
        self.config = config
        self.ssl_proj = nn.Conv1d(
            config.ssl_channels,
            config.hidden_channels,
            1,
        )
        self.encoder_ssl = attentions.Encoder(
            config.hidden_channels,
            config.filter_channels,
            config.attention_heads,
            config.layers // 2,
            config.kernel_size,
            config.dropout,
        )
        self.encoder_text = attentions.Encoder(
            config.hidden_channels,
            config.filter_channels,
            config.attention_heads,
            config.layers,
            config.kernel_size,
            config.dropout,
        )
        self.text_embedding = nn.Embedding(
            config.phoneme_vocabulary_size,
            config.hidden_channels,
        )
        self.mrte = MRTE()
        self.encoder2 = attentions.Encoder(
            config.hidden_channels,
            config.filter_channels,
            config.attention_heads,
            config.layers // 2,
            config.kernel_size,
            config.dropout,
        )
        self.proj = nn.Conv1d(
            config.hidden_channels,
            config.inter_channels * 2,
            1,
        )

    def forward(
        self,
        semantic_hidden: Tensor,
        semantic_lengths: Tensor,
        phoneme_ids: Tensor,
        phoneme_lengths: Tensor,
        style: Tensor,
        speed: float = 1.0,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        semantic_mask = commons.sequence_mask(
            semantic_lengths,
            semantic_hidden.shape[2],
        ).unsqueeze(1).to(semantic_hidden.dtype)
        semantic_hidden = self.ssl_proj(semantic_hidden * semantic_mask) * semantic_mask
        semantic_hidden = self.encoder_ssl(
            semantic_hidden * semantic_mask,
            semantic_mask,
        )
        phoneme_mask = commons.sequence_mask(
            phoneme_lengths,
            phoneme_ids.shape[1],
        ).unsqueeze(1).to(semantic_hidden.dtype)
        phoneme_hidden = self.text_embedding(phoneme_ids).transpose(1, 2)
        phoneme_hidden = self.encoder_text(
            phoneme_hidden * phoneme_mask,
            phoneme_mask,
        )
        hidden = self.mrte(
            semantic_hidden,
            semantic_mask,
            phoneme_hidden,
            phoneme_mask,
            style,
        )
        hidden = self.encoder2(hidden * semantic_mask, semantic_mask)
        if speed != 1.0:
            hidden = functional.interpolate(
                hidden,
                size=int(hidden.shape[-1] / speed) + 1,
                mode="linear",
            )
            semantic_mask = functional.interpolate(
                semantic_mask,
                size=hidden.shape[-1],
                mode="nearest",
            )
        statistics = self.proj(hidden) * semantic_mask
        mean, log_scale = statistics.split(
            self.config.inter_channels,
            dim=1,
        )
        return hidden, mean, log_scale, semantic_mask


class PosteriorEncoder(_VITSPosteriorEncoder):

    def forward(
        self,
        inputs: Tensor,
        lengths: Tensor,
        g: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return super().forward(
            inputs,
            lengths,
            g=None if g is None else g.detach(),
        )


class MultiPeriodDiscriminator(nn.Module):
    """Checkpoint-compatible classic and Pro discriminator inventories."""

    def __init__(
            self,
            use_spectral_norm: bool = False,
            *,
            periods: tuple[int, ...] = (2, 3, 5, 7, 11),
    ) -> None:
        super().__init__()
        if not periods or any(isinstance(period, bool) or not isinstance(period, int) or period < 2
                              for period in periods):
            raise ValueError("GPT-SoVITS discriminator periods must be integers >= 2.")
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=use_spectral_norm),
            *(DiscriminatorP(
                period,
                use_spectral_norm=use_spectral_norm,
            ) for period in periods),
        ])

    def forward(
        self,
        real: Tensor,
        generated: Tensor,
    ) -> tuple[list[Tensor], list[Tensor], list[list[Tensor]], list[list[Tensor]]]:
        real_scores = []
        generated_scores = []
        real_features = []
        generated_features = []
        for discriminator in self.discriminators:
            real_score, real_feature = discriminator(real)
            generated_score, generated_feature = discriminator(generated)
            real_scores.append(real_score)
            generated_scores.append(generated_score)
            real_features.append(real_feature)
            generated_features.append(generated_feature)
        return (
            real_scores,
            generated_scores,
            real_features,
            generated_features,
        )


class GPTSoVITSSynthesizer(nn.Module):
    """Classic S2 generator retaining all released state-dict names."""

    def __init__(self, config: GPTSoVITSS2Config | None = None) -> None:
        super().__init__()
        self.config = config or GPTSoVITSS2Config()
        config = self.config
        self.spec_channels = config.spectrogram_channels
        self.inter_channels = config.inter_channels
        self.hidden_channels = config.hidden_channels
        self.filter_channels = config.filter_channels
        self.n_heads = config.attention_heads
        self.n_layers = config.layers
        self.kernel_size = config.kernel_size
        self.p_dropout = config.dropout
        self.resblock = config.resblock
        self.resblock_kernel_sizes = config.resblock_kernel_sizes
        self.resblock_dilation_sizes = config.resblock_dilation_sizes
        self.upsample_rates = config.upsample_rates
        self.upsample_initial_channel = config.upsample_initial_channels
        self.upsample_kernel_sizes = config.upsample_kernel_sizes
        self.segment_size = config.segment_frames
        self.n_speakers = 300
        self.gin_channels = config.gin_channels
        self.version = config.version
        self.use_sdp = True
        self.enc_p = TextEncoder(config)
        self.dec = Generator(
            config.inter_channels,
            config.resblock,
            config.resblock_kernel_sizes,
            config.resblock_dilation_sizes,
            config.upsample_rates,
            config.upsample_initial_channels,
            config.upsample_kernel_sizes,
            gin_channels=config.gin_channels,
        )
        self.enc_q = PosteriorEncoder(
            config.spectrogram_channels,
            config.inter_channels,
            config.hidden_channels,
            5,
            1,
            config.posterior_layers,
            gin_channels=config.gin_channels,
        )
        self.flow = ResidualCouplingBlock(
            config.inter_channels,
            config.hidden_channels,
            5,
            1,
            4,
            gin_channels=config.gin_channels,
        )
        self.ref_enc = MelStyleEncoder(
            config.style_channels,
            style_vector_dim=config.gin_channels,
        )
        stride = 2 if config.semantic_frame_rate == "25hz" else 1
        kernel = 2 if stride == 2 else 1
        self.ssl_proj = nn.Conv1d(
            config.ssl_channels,
            config.ssl_channels,
            kernel,
            stride=stride,
        )
        self.quantizer = ResidualVectorQuantizer(
            dimension=config.ssl_channels,
            quantizers=1,
            bins=1_024,
        )
        self.freeze_quantizer = config.freeze_quantizer
        if config.requires_speaker_embedding:
            assert config.speaker_embedding_dim is not None
            self.sv_emb = nn.Linear(
                config.speaker_embedding_dim,
                config.gin_channels,
            )
            self.ge_to512 = nn.Linear(config.gin_channels, 512)
            self.prelu = nn.PReLU(num_parameters=config.gin_channels)

    def _style(
        self,
        spectrogram: Tensor,
        spectrogram_lengths: Tensor,
        speaker_embedding: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        mask = commons.sequence_mask(
            spectrogram_lengths,
            spectrogram.shape[2],
        ).unsqueeze(1).to(spectrogram.dtype)
        style = self.ref_enc(
            spectrogram[:, :self.config.style_channels] * mask,
            mask,
        )
        text_style = style
        if self.config.requires_speaker_embedding:
            assert self.config.speaker_embedding_dim is not None
            if speaker_embedding is None:
                raise ValueError(
                    f"GPT-SoVITS {self.config.version} requires prepared "
                    f"{self.config.speaker_embedding_dim}-dimensional speaker embeddings.")
            if speaker_embedding.ndim == 3 and speaker_embedding.shape[1] == 1:
                speaker_embedding = speaker_embedding[:, 0]
            expected = (
                spectrogram.shape[0],
                self.config.speaker_embedding_dim,
            )
            if tuple(speaker_embedding.shape) != expected:
                raise ValueError("GPT-SoVITS speaker embeddings must have shape "
                                 f"{expected}.")
            style = self.prelu(style + self.sv_emb(speaker_embedding).unsqueeze(-1))
            text_style = self.ge_to512(style.transpose(1, 2)).transpose(1, 2)
        elif speaker_embedding is not None:
            raise ValueError(f"GPT-SoVITS {self.config.version} does not consume speaker embeddings.")
        return style, text_style, mask

    def forward(
        self,
        ssl_features: Tensor,
        spectrogram: Tensor,
        spectrogram_lengths: Tensor,
        phoneme_ids: Tensor,
        phoneme_lengths: Tensor,
        speaker_embedding: Tensor | None = None,
    ) -> tuple[
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
            Tensor,
    ]:
        style, text_style, _ = self._style(
            spectrogram,
            spectrogram_lengths,
            speaker_embedding,
        )
        self.quantizer.eval()
        ssl_hidden = self.ssl_proj(ssl_features)
        quantized, _, commitment_loss, selected = self.quantizer(
            ssl_hidden,
            layers=[0],
        )
        quantized = functional.interpolate(
            quantized,
            size=quantized.shape[-1] * 2,
            mode="nearest",
        )
        _, mean_prior, logs_prior, mask = self.enc_p(
            quantized,
            spectrogram_lengths,
            phoneme_ids,
            phoneme_lengths,
            text_style,
        )
        latent, mean_posterior, logs_posterior, posterior_mask = self.enc_q(
            spectrogram,
            spectrogram_lengths,
            g=style,
        )
        latent_prior = self.flow(latent, posterior_mask, g=style)
        latent_slice, slice_ids = commons.rand_slice_segments(
            latent,
            spectrogram_lengths,
            self.config.segment_frames,
        )
        waveform = self.dec(latent_slice, g=style)
        statistics = selected[0] if selected else quantized
        return (
            waveform,
            commitment_loss,
            slice_ids,
            mask,
            posterior_mask,
            (
                latent,
                latent_prior,
                mean_prior,
                logs_prior,
                mean_posterior,
                logs_posterior,
            ),
            statistics,
        )

    @torch.no_grad()
    def decode(
        self,
        semantic_codes: Tensor,
        phoneme_ids: Tensor,
        reference_spectrogram: Tensor,
        *,
        speaker_embedding: Tensor | None = None,
        noise_scale: float = 0.5,
        speed: float = 1.0,
    ) -> Tensor:
        if semantic_codes.ndim == 2:
            semantic_codes = semantic_codes.unsqueeze(0)
        if semantic_codes.ndim != 3 or semantic_codes.shape[0] != 1:
            raise ValueError("S2 semantic codes must have shape [1, codebooks, time].")
        if phoneme_ids.ndim == 1:
            phoneme_ids = phoneme_ids.unsqueeze(0)
        if reference_spectrogram.ndim != 3:
            raise ValueError("Reference spectrogram must have shape [1, 1025, frames].")
        reference_lengths = torch.tensor(
            [reference_spectrogram.shape[2]],
            device=reference_spectrogram.device,
            dtype=torch.long,
        )
        style, text_style, _ = self._style(
            reference_spectrogram,
            reference_lengths,
            speaker_embedding,
        )
        semantic_lengths = torch.tensor(
            [semantic_codes.shape[2] * 2],
            device=semantic_codes.device,
            dtype=torch.long,
        )
        phoneme_lengths = torch.tensor(
            [phoneme_ids.shape[1]],
            device=phoneme_ids.device,
            dtype=torch.long,
        )
        quantized = self.quantizer.decode(semantic_codes)
        quantized = functional.interpolate(
            quantized,
            size=quantized.shape[-1] * 2,
            mode="nearest",
        )
        _, mean, log_scale, mask = self.enc_p(
            quantized,
            semantic_lengths,
            phoneme_ids,
            phoneme_lengths,
            text_style,
            speed,
        )
        latent_prior = mean + torch.randn_like(mean) * torch.exp(log_scale) * noise_scale
        latent = self.flow(latent_prior, mask, g=style, reverse=True)
        return self.dec(latent * mask, g=style)

    @torch.no_grad()
    def extract_latent(self, ssl_features: Tensor) -> Tensor:
        projected = self.ssl_proj(ssl_features)
        _, codes, _, _ = self.quantizer(projected)
        return codes.transpose(0, 1)


def build_s2_generator(config: GPTSoVITSS2Config | None = None, ) -> GPTSoVITSSynthesizer:
    return GPTSoVITSSynthesizer(config)


def build_s2_discriminator(config: GPTSoVITSS2Config | None = None, ) -> MultiPeriodDiscriminator:
    resolved = config or GPTSoVITSS2Config()
    return MultiPeriodDiscriminator(
        resolved.use_spectral_norm,
        periods=resolved.discriminator_periods,
    )


__all__ = [
    "GPTSoVITSSynthesizer",
    "MultiPeriodDiscriminator",
    "TextEncoder",
    "build_s2_discriminator",
    "build_s2_generator",
]
