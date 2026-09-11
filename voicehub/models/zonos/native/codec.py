"""Frozen VoiceHub-native Descript DAC boundary for Zonos v0.1."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor

from voicehub.checkpointing import SafeTensorReader
from voicehub.hub import read_json_file, resolve_pretrained_file
from voicehub.models.dac.native.checkpoint import HuggingFaceDacCheckpointAdapter
from voicehub.models.dac.native.configuration import DacConfig
from voicehub.models.dac.native.modeling import DacModel
from voicehub.models.zonos.native.metadata import ZONOS_DAC_REPOSITORY, ZONOS_DAC_REVISION
from voicehub.processing.waveform import resample_waveform


@runtime_checkable
class ZonosCodec(Protocol):
    """Minimal frozen codec contract consumed by the native runtime."""

    sample_rate: int
    hop_length: int
    num_codebooks: int
    codebook_size: int

    def encode(self, waveform: Tensor, *, sample_rate: int) -> Tensor:
        """Return codes with shape ``[batch, codebook, time]``."""

    def decode(self, codes: Tensor) -> Tensor:
        """Return audio with shape ``[batch, channel, time]``."""


def load_zonos_dac_model(
    checkpoint_path: str | Path,
    config_path: str | Path,
    *,
    device: torch.device | str,
) -> DacModel:
    values = read_json_file(config_path)
    config = DacConfig.from_dict(values)
    if (config.sampling_rate != 44_100 or config.n_codebooks != 9 or config.codebook_size != 1_024 or
            config.hop_length != 512):
        raise ValueError(
            "Zonos requires the 44.1 kHz DAC graph with nine 1,024-entry "
            "codebooks and a 512-sample hop.")
    with torch.device("meta"):
        model = DacModel(config)
    with SafeTensorReader(checkpoint_path) as reader:
        HuggingFaceDacCheckpointAdapter().load_assign(
            model,
            reader,
            values,
            strict=True,
        )
    remaining = [name for name, value in model.state_dict().items() if value.device.type == "meta"]
    if remaining:
        raise RuntimeError("Native DAC loading left meta tensors: " + ", ".join(remaining[:12]))
    return model.to(device=device).eval().requires_grad_(False)


class ZonosDACCodec:
    """Lazy safe loader for the exact DAC checkpoint used by Zonos."""

    sample_rate = 44_100
    sampling_rate = sample_rate
    hop_length = 512
    num_codebooks = 9
    codebook_size = 1_024

    def __init__(
        self,
        *,
        repository: str | Path = ZONOS_DAC_REPOSITORY,
        revision: str | None = ZONOS_DAC_REVISION,
        cache_dir: str | None = None,
        token: str | bool | None = None,
        local_files_only: bool = False,
        device: torch.device | str = "cpu",
        model: DacModel | None = None,
    ) -> None:
        self.repository = repository
        self.revision = revision
        self.cache_dir = cache_dir
        self.token = token
        self.local_files_only = local_files_only
        self.device = torch.device(device)
        self._model = model
        if model is not None:
            self._validate_model(model)
            model.to(device=self.device).eval().requires_grad_(False)

    @staticmethod
    def _validate_model(model: DacModel) -> None:
        if not isinstance(model, DacModel):
            raise TypeError("Injected Zonos codec model must be DacModel.")
        config = model.config
        if (
                config.sampling_rate,
                config.n_codebooks,
                config.codebook_size,
                config.hop_length,
        ) != (44_100, 9, 1_024, 512):
            raise ValueError("Injected DAC graph is not compatible with Zonos v0.1.")

    @property
    def model(self) -> DacModel:
        if self._model is None:
            common = {
                "revision": self.revision,
                "cache_dir": self.cache_dir,
                "token": self.token,
                "local_files_only": self.local_files_only,
            }
            checkpoint = resolve_pretrained_file(
                self.repository,
                "model.safetensors",
                **common,
            )
            configuration = resolve_pretrained_file(
                self.repository,
                "config.json",
                **common,
            )
            self._model = load_zonos_dac_model(
                checkpoint,
                configuration,
                device=self.device,
            )
        return self._model

    def to(self, device: torch.device | str) -> ZonosDACCodec:
        self.device = torch.device(device)
        if self._model is not None:
            self._model.to(device=self.device)
        return self

    def _preprocess(self, waveform: Tensor, *, sample_rate: int) -> Tensor:
        if not isinstance(waveform, Tensor) or waveform.ndim not in {1, 2, 3}:
            raise ValueError(
                "Zonos DAC waveform must have shape [time], "
                "[batch, time], or [batch, 1, time].")
        if (isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0):
            raise ValueError("Zonos DAC sample rate must be positive.")
        if waveform.ndim == 1:
            waveform = waveform.reshape(1, 1, -1)
        elif waveform.ndim == 2:
            waveform = waveform.unsqueeze(1)
        if waveform.shape[1] != 1:
            raise ValueError("Zonos DAC input must be mono.")
        waveform = waveform.to(dtype=torch.float32)
        if not bool(torch.isfinite(waveform).all()):
            raise ValueError("Zonos DAC waveform must contain finite samples.")
        if sample_rate != self.sample_rate:
            waveform = torch.stack(
                [resample_waveform(
                    row[0],
                    sample_rate,
                    self.sample_rate,
                ) for row in waveform]).unsqueeze(1)
        padding = (math.ceil(waveform.shape[-1] / self.hop_length) * self.hop_length - waveform.shape[-1])
        return torch.nn.functional.pad(
            waveform,
            (0, padding),
        ).to(device=self.device)

    @torch.inference_mode()
    def encode(self, waveform: Tensor, *, sample_rate: int) -> Tensor:
        prepared = self._preprocess(
            waveform,
            sample_rate=sample_rate,
        )
        return self.model.encode_output(prepared).audio_codes

    @torch.inference_mode()
    def decode(self, codes: Tensor) -> Tensor:
        if not isinstance(codes, Tensor) or codes.ndim != 3:
            raise ValueError("Zonos DAC codes must have shape "
                             "[batch, codebook, time].")
        if codes.shape[1] != self.num_codebooks:
            raise ValueError(f"Zonos DAC requires {self.num_codebooks} codebooks.")
        if codes.dtype == torch.bool or codes.is_floating_point():
            raise TypeError("Zonos DAC codes must use an integer dtype.")
        if bool(((codes < 0) | (codes >= self.codebook_size)).any()):
            raise ValueError("Zonos DAC code values must be in [0, 1023].")
        model = self.model
        latents, _, _ = model.quantizer.from_codes(codes.to(device=self.device, dtype=torch.long), )
        with torch.autocast(
                self.device.type,
                torch.float16,
                enabled=self.device.type != "cpu",
        ):
            return model.decode_output(latents).audio_values.float()


__all__ = [
    "ZonosCodec",
    "ZonosDACCodec",
    "load_zonos_dac_model",
]
