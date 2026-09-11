"""VoiceHub-native Descript DAC adapter used by Zonos."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import Tensor

from voicehub.models.dac.native.checkpoint import (
    DESCRIPT_DAC_44KHZ_REVISION,
    HuggingFaceDacCheckpointAdapter,
)
from voicehub.models.dac.native.configuration import DacConfig
from voicehub.models.dac.native.modeling import DacModel
from voicehub.checkpointing import SafeTensorReader
from voicehub.hub import read_json_file, resolve_pretrained_file
from voicehub.processing.waveform import resample_waveform

_DEFAULT_DAC_REPOSITORY = "descript/dac_44khz"
_DEFAULT_DAC_REVISION = DESCRIPT_DAC_44KHZ_REVISION


def _load_native_dac(
    checkpoint_path: str | Path,
    config_path: str | Path,
    *,
    device: torch.device,
) -> DacModel:
    configuration_values = read_json_file(config_path)
    configuration = DacConfig.from_dict(configuration_values)
    if (
        configuration.sampling_rate != 44_100
        or configuration.n_codebooks != 9
        or configuration.codebook_size != 1_024
        or configuration.hop_length != 512
    ):
        raise ValueError(
            "Zonos requires the 44.1 kHz DAC layout with nine 1,024-entry "
            "codebooks and a 512-sample hop."
        )
    with torch.device("meta"):
        model = DacModel(configuration)
    with SafeTensorReader(checkpoint_path) as reader:
        HuggingFaceDacCheckpointAdapter().load_assign(
            model,
            reader,
            configuration_values,
            strict=True,
        )
    return model.to(device=device).eval().requires_grad_(False)


class DACAutoencoder:
    """Lazy native DAC runtime with the historical Zonos codec interface."""

    codebook_size = 1_024
    num_codebooks = 9
    sampling_rate = 44_100
    hop_length = 512

    def __init__(
        self,
        *,
        repository: str | Path = _DEFAULT_DAC_REPOSITORY,
        revision: str = _DEFAULT_DAC_REVISION,
        token: str | bool | None = None,
    ) -> None:
        self.repository = repository
        self.revision = revision
        self.token = token
        self._device = torch.device("cpu")
        self._dac: DacModel | None = None

    @property
    def dac(self) -> DacModel:
        return self._ensure_loaded()

    def _ensure_loaded(self) -> DacModel:
        if self._dac is None:
            checkpoint = resolve_pretrained_file(
                self.repository,
                "model.safetensors",
                revision=self.revision,
                token=self.token,
            )
            configuration = resolve_pretrained_file(
                self.repository,
                "config.json",
                revision=self.revision,
                token=self.token,
            )
            self._dac = _load_native_dac(
                checkpoint,
                configuration,
                device=self._device,
            )
        return self._dac

    def to(self, device: str | torch.device) -> DACAutoencoder:
        self._device = torch.device(device)
        if self._dac is not None:
            self._dac.to(device=self._device)
        return self

    def preprocess(self, waveform: Tensor, sample_rate: int) -> Tensor:
        if not isinstance(waveform, Tensor) or waveform.ndim < 1:
            raise ValueError(
                "DAC audio must be a PyTorch tensor with a time axis."
            )
        if (
            isinstance(sample_rate, bool)
            or not isinstance(sample_rate, int)
            or sample_rate <= 0
        ):
            raise ValueError("DAC sample rate must be a positive integer.")
        flattened = waveform.reshape(-1, waveform.shape[-1])
        if sample_rate != self.sampling_rate:
            resampled = torch.stack(
                tuple(
                    resample_waveform(
                        channel,
                        sample_rate,
                        self.sampling_rate,
                    )
                    for channel in flattened
                ),
                dim=0,
            )
            waveform = resampled.reshape(
                *waveform.shape[:-1],
                resampled.shape[-1],
            )
        right_padding = (
            math.ceil(waveform.shape[-1] / self.hop_length)
            * self.hop_length
            - waveform.shape[-1]
        )
        return torch.nn.functional.pad(waveform, (0, right_padding))

    def encode(self, waveform: Tensor) -> Tensor:
        model = self._ensure_loaded()
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(1)
        if waveform.ndim != 3 or waveform.shape[1] != 1:
            raise ValueError("DAC input must have shape [batch, 1, time].")
        return model.encode_output(waveform).audio_codes

    def decode(self, codes: Tensor) -> Tensor:
        if not isinstance(codes, Tensor) or codes.ndim != 3:
            raise ValueError(
                "DAC codes must have shape [batch, codebook, time]."
            )
        if codes.shape[1] > self.num_codebooks:
            raise ValueError(
                f"DAC supports at most {self.num_codebooks} codebooks."
            )
        model = self._ensure_loaded()
        latents, _, _ = model.quantizer.from_codes(codes.long())
        with torch.autocast(
            model.device.type,
            torch.float16,
            enabled=model.device.type != "cpu",
        ):
            return model.decode_output(latents).audio_values.float()


__all__ = ["DACAutoencoder"]
