"""VoiceHub-native OpenVoice V2 tone-color conversion graph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from voicehub.models.openvoice.native.configuration import OpenVoiceConverterConfig
from voicehub.models.openvoice.source.openvoice.models import SynthesizerTrn


@dataclass
class OpenVoiceConverterOutput:
    """Converted waveform and optional reconstructed fine-tuning loss."""

    loss: Tensor | None
    waveform: Tensor
    waveform_mask: Tensor
    source_latent: Tensor
    converted_latent: Tensor


class OpenVoiceToneColorConverter(SynthesizerTrn):
    """Exact 486-tensor V2 converter with a typed training interface."""

    def __init__(
        self,
        config: OpenVoiceConverterConfig | dict[str, Any],
    ) -> None:
        config = (
            config
            if isinstance(config, OpenVoiceConverterConfig) else OpenVoiceConverterConfig.from_dict(config))
        self.config = config
        super().__init__(
            0,
            config.spectrogram_channels,
            inter_channels=config.inter_channels,
            hidden_channels=config.hidden_channels,
            filter_channels=config.filter_channels,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            kernel_size=config.kernel_size,
            p_dropout=config.dropout,
            resblock=config.resblock,
            resblock_kernel_sizes=config.resblock_kernel_sizes,
            resblock_dilation_sizes=config.resblock_dilation_sizes,
            upsample_rates=config.upsample_rates,
            upsample_initial_channel=config.upsample_initial_channel,
            upsample_kernel_sizes=config.upsample_kernel_sizes,
            n_speakers=0,
            gin_channels=config.speaker_embedding_size,
            zero_g=config.zero_generator_conditioning,
        )

    def extract_speaker_embedding(
        self,
        spectrograms: Tensor,
        *,
        segment_mask: Tensor | None = None,
    ) -> Tensor:
        """Average reference-encoder embeddings across validated segments."""
        embeddings = self.extract_speaker_embeddings(
            spectrograms,
            segment_mask=segment_mask,
        )
        return embeddings.mean(dim=0, keepdim=True)

    def extract_speaker_embeddings(
        self,
        spectrograms: Tensor,
        *,
        lengths: Tensor | None = None,
        segment_mask: Tensor | None = None,
    ) -> Tensor:
        """Encode one right-padded reference spectrogram per batch item.

        The released reference encoder has no padding-mask input.
        Cropping every item before the encoder therefore preserves the
        unpadded computation and keeps gradients connected to the
        reference encoder.
        """
        if (spectrograms.ndim != 3 or spectrograms.shape[1] != self.config.spectrogram_channels):
            raise ValueError(
                "OpenVoice reference spectrograms must have shape "
                f"[segments, {self.config.spectrogram_channels}, frames].")
        if not spectrograms.is_floating_point():
            raise TypeError("OpenVoice spectrograms must be floating point.")
        if lengths is not None and segment_mask is not None:
            raise ValueError("Supply OpenVoice reference `lengths` or `segment_mask`, "
                             "not both.")
        if segment_mask is not None:
            if segment_mask.shape != (
                    spectrograms.shape[0],
                    spectrograms.shape[2],
            ):
                raise ValueError("OpenVoice reference mask must have shape [segments, frames].")
            mask = segment_mask.to(device=spectrograms.device, dtype=torch.bool)
            lengths = mask.long().sum(dim=-1)
            expected = (
                torch.arange(
                    spectrograms.shape[-1],
                    device=spectrograms.device,
                )[None, :] < lengths[:, None])
            if not bool(torch.equal(mask, expected)):
                raise ValueError("OpenVoice reference masks must describe right padding.")
        if lengths is None:
            lengths = torch.full(
                (spectrograms.shape[0], ),
                spectrograms.shape[-1],
                device=spectrograms.device,
                dtype=torch.long,
            )
        if (lengths.ndim != 1 or lengths.shape[0] != spectrograms.shape[0]):
            raise ValueError("OpenVoice reference lengths must have shape [batch].")
        if lengths.dtype == torch.bool or lengths.is_floating_point():
            raise TypeError("OpenVoice reference lengths must use integer dtype.")
        if bool(((lengths < 1) | (lengths > spectrograms.shape[-1])).any()):
            raise ValueError("OpenVoice reference lengths are outside frame bounds.")
        embeddings = []
        for index, length in enumerate(lengths.tolist()):
            reference = spectrograms[
                index:index + 1,
                :,
                :int(length),
            ]
            embeddings.append(self.ref_enc(reference.transpose(1, 2)).unsqueeze(-1))
        return torch.cat(embeddings, dim=0)

    @staticmethod
    def _embedding(
        value: Tensor,
        *,
        batch_size: int,
        channels: int,
        name: str,
    ) -> Tensor:
        if value.ndim == 2:
            value = value.unsqueeze(-1)
        if value.ndim != 3 or value.shape[1:] != (channels, 1):
            raise ValueError(f"OpenVoice `{name}` must have shape [batch, {channels}, 1].")
        if value.shape[0] not in {1, batch_size}:
            raise ValueError(f"OpenVoice `{name}` batch must be one or {batch_size}.")
        if value.shape[0] == 1 and batch_size > 1:
            value = value.expand(batch_size, -1, -1)
        return value

    def forward(
        self,
        source_spectrogram: Tensor,
        source_lengths: Tensor,
        source_embedding: Tensor | None = None,
        target_embedding: Tensor | None = None,
        *,
        source_reference_spectrogram: Tensor | None = None,
        source_reference_lengths: Tensor | None = None,
        target_reference_spectrogram: Tensor | None = None,
        target_reference_lengths: Tensor | None = None,
        target_waveform: Tensor | None = None,
        target_lengths: Tensor | None = None,
        tau: float = 0.3,
        reduction: str = "mean",
    ) -> OpenVoiceConverterOutput:
        """Convert speech and optionally optimize paired reconstruction.

        The public repository does not release its original
        discriminator or loss. ``target_waveform`` therefore activates a
        clearly reconstructed paired waveform objective; it is not
        presented as upstream parity.
        """
        if (source_spectrogram.ndim != 3 or source_spectrogram.shape[1] != self.config.spectrogram_channels):
            raise ValueError(
                "OpenVoice source spectrogram must have shape "
                f"[batch, {self.config.spectrogram_channels}, frames].")
        batch_size = source_spectrogram.shape[0]
        if (source_lengths.ndim != 1 or source_lengths.shape[0] != batch_size):
            raise ValueError("OpenVoice source lengths must have shape [batch].")
        if source_lengths.dtype == torch.bool or source_lengths.is_floating_point():
            raise TypeError("OpenVoice source lengths must use integer dtype.")
        if bool(((source_lengths < 1) | (source_lengths > source_spectrogram.shape[-1])).any()):
            raise ValueError("OpenVoice source lengths are outside frame bounds.")
        if source_embedding is None:
            if source_reference_spectrogram is None:
                raise ValueError(
                    "OpenVoice requires `source_embedding` or "
                    "`source_reference_spectrogram`.")
            source_embedding = self.extract_speaker_embeddings(
                source_reference_spectrogram,
                lengths=source_reference_lengths,
            )
        elif source_reference_spectrogram is not None:
            raise ValueError("Supply OpenVoice `source_embedding` or a source reference, "
                             "not both.")
        if target_embedding is None:
            if target_reference_spectrogram is None:
                raise ValueError(
                    "OpenVoice requires `target_embedding` or "
                    "`target_reference_spectrogram`.")
            target_embedding = self.extract_speaker_embeddings(
                target_reference_spectrogram,
                lengths=target_reference_lengths,
            )
        elif target_reference_spectrogram is not None:
            raise ValueError("Supply OpenVoice `target_embedding` or a target reference, "
                             "not both.")
        source_embedding = self._embedding(
            source_embedding,
            batch_size=batch_size,
            channels=self.config.speaker_embedding_size,
            name="source_embedding",
        )
        target_embedding = self._embedding(
            target_embedding,
            batch_size=batch_size,
            channels=self.config.speaker_embedding_size,
            name="target_embedding",
        )
        if (isinstance(tau, bool) or not isinstance(tau, (int, float)) or not 0.0 <= float(tau) <= 1.0):
            raise ValueError("OpenVoice `tau` must be in [0, 1].")
        waveform, mask, latents = self.voice_conversion(
            source_spectrogram,
            source_lengths.to(
                device=source_spectrogram.device,
                dtype=torch.long,
            ),
            source_embedding.to(
                device=source_spectrogram.device,
                dtype=source_spectrogram.dtype,
            ),
            target_embedding.to(
                device=source_spectrogram.device,
                dtype=source_spectrogram.dtype,
            ),
            tau=float(tau),
        )
        source_latent, _, converted_latent = latents
        loss = None
        if target_waveform is not None:
            if target_waveform.ndim == 2:
                target_waveform = target_waveform.unsqueeze(1)
            if target_waveform.ndim != 3 or target_waveform.shape[:2] != (
                    batch_size,
                    1,
            ):
                raise ValueError(
                    "OpenVoice target waveform must have shape "
                    "[batch, samples] or [batch, 1, samples].")
            if target_lengths is None:
                target_lengths = torch.full(
                    (batch_size, ),
                    target_waveform.shape[-1],
                    device=target_waveform.device,
                    dtype=torch.long,
                )
            if (target_lengths.ndim != 1 or target_lengths.shape[0] != batch_size):
                raise ValueError("OpenVoice target lengths must have shape [batch].")
            if (target_lengths.dtype == torch.bool or target_lengths.is_floating_point()):
                raise TypeError("OpenVoice target lengths must use integer dtype.")
            if bool(((target_lengths < self.config.hop_length) |
                     (target_lengths > target_waveform.shape[-1])).any()):
                raise ValueError("OpenVoice target lengths are outside waveform bounds.")
            generated_lengths = (mask.squeeze(1).sum(dim=-1).long() * self.config.hop_length)
            valid_lengths = torch.minimum(
                generated_lengths,
                target_lengths.to(
                    device=generated_lengths.device,
                    dtype=torch.long,
                ),
            )
            common = min(waveform.shape[-1], target_waveform.shape[-1])
            errors = F.smooth_l1_loss(
                waveform[..., :common].float(),
                target_waveform[..., :common].to(
                    device=waveform.device,
                    dtype=torch.float32,
                ),
                reduction="none",
            )
            sample_mask = (
                torch.arange(common, device=waveform.device)[None, None, :] < valid_lengths[:, None, None])
            errors = errors * sample_mask
            counts = sample_mask.sum(dim=(1, 2)).clamp_min(1)
            per_example = errors.sum(dim=(1, 2)) / counts
            if reduction == "mean":
                loss = per_example.mean()
            elif reduction == "sum":
                loss = per_example.sum()
            elif reduction == "none":
                loss = per_example
            else:
                raise ValueError("OpenVoice reduction must be 'mean', 'sum', or 'none'.")
        return OpenVoiceConverterOutput(
            loss=loss,
            waveform=waveform,
            waveform_mask=mask,
            source_latent=source_latent,
            converted_latent=converted_latent,
        )


__all__ = [
    "OpenVoiceConverterOutput",
    "OpenVoiceToneColorConverter",
]
