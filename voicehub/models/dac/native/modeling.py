"""Native Descript DAC model builder and typed codec outputs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from torch import Tensor

from voicehub.components.audio.codecs.dac.model.dac import DAC
from voicehub.models.dac.native.configuration import DacConfig


@dataclass(frozen=True, slots=True)
class DacEncoderOutput:
    quantized_representation: Tensor
    audio_codes: Tensor
    projected_latents: Tensor
    commitment_loss: Tensor
    codebook_loss: Tensor


@dataclass(frozen=True, slots=True)
class DacDecoderOutput:
    audio_values: Tensor


class DacModel(DAC):
    """Published DAC graph configured without an upstream model runtime."""

    def __init__(self, config: DacConfig | Mapping[str, Any]) -> None:
        self.config = DacConfig.coerce(config)
        super().__init__(
            encoder_dim=self.config.encoder_hidden_size,
            encoder_rates=list(self.config.downsampling_ratios),
            latent_dim=self.config.hidden_size,
            decoder_dim=self.config.decoder_hidden_size,
            decoder_rates=list(self.config.upsampling_ratios),
            n_codebooks=self.config.n_codebooks,
            codebook_size=self.config.codebook_size,
            codebook_dim=self.config.codebook_dim,
            quantizer_dropout=self.config.quantizer_dropout,
            sample_rate=self.config.sampling_rate,
        )

    def encode_output(
        self,
        audio_values: Tensor,
        *,
        n_quantizers: int | None = None,
    ) -> DacEncoderOutput:
        values = self.encode(audio_values, n_quantizers=n_quantizers)
        return DacEncoderOutput(
            quantized_representation=values[0],
            audio_codes=values[1],
            projected_latents=values[2],
            commitment_loss=values[3],
            codebook_loss=values[4],
        )

    def decode_output(self, quantized_representation: Tensor) -> DacDecoderOutput:
        return DacDecoderOutput(audio_values=self.decode(quantized_representation))

    def quantizer_loss(self, output: DacEncoderOutput) -> Tensor:
        if not isinstance(output, DacEncoderOutput):
            raise TypeError("`output` must be a DacEncoderOutput.")
        return (
            self.config.commitment_loss_weight * output.commitment_loss +
            self.config.codebook_loss_weight * output.codebook_loss)


__all__ = [
    "DacDecoderOutput",
    "DacEncoderOutput",
    "DacModel",
]
