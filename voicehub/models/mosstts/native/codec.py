"""Native MOSS Audio Tokenizer v1/v2 runtime contracts and quantizers."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import torch
from torch import Tensor, nn
from torch.nn import functional


def _positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"`{name}` must be a positive integer.")
    return value


@dataclass(frozen=True, slots=True)
class MossAudioCodecConfig:
    """Audited executable boundary shared by MOSS codec v1 and v2."""

    version: int
    sample_rate: int
    downsample_rate: int
    channels: int
    code_dimension: int
    rvq_dimension: int
    output_dimension: int
    num_quantizers: int
    codebook_size: int
    codebook_dimension: int
    quantizer_type: str = "rlfq"
    channel_interleave: bool = False

    def __post_init__(self) -> None:
        if self.version not in (1, 2):
            raise ValueError("MOSS codec version must be 1 or 2.")
        for name in (
                "sample_rate",
                "downsample_rate",
                "channels",
                "code_dimension",
                "rvq_dimension",
                "output_dimension",
                "num_quantizers",
                "codebook_size",
                "codebook_dimension",
        ):
            _positive_integer(name, getattr(self, name))
        if self.quantizer_type not in {"rvq", "spec_rvq", "rlfq", "random_prefix_rlfq"}:
            raise ValueError(f"Unsupported MOSS codec quantizer {self.quantizer_type!r}.")
        if self.version == 1:
            expected = (24_000, 1_920, 1)
        else:
            expected = (48_000, 3_840, 2)
        actual = (self.sample_rate, self.downsample_rate, self.channels)
        if actual != expected:
            raise ValueError(
                f"MOSS codec v{self.version} requires "
                f"sample_rate/downsample_rate/channels={expected!r}; "
                f"found {actual!r}.")
        if self.version == 1 and self.channel_interleave:
            raise ValueError("MOSS codec v1 does not use channel interleaving.")
        if self.version == 2 and not self.channel_interleave:
            raise ValueError("MOSS codec v2 requires channel interleaving.")

    @classmethod
    def from_dict(
        cls,
        values: Mapping[str, object],
        *,
        version: int | None = None,
    ) -> MossAudioCodecConfig:
        if not isinstance(values, Mapping):
            raise TypeError("MOSS codec configuration must be a mapping.")
        sample_rate = int(values.get("sampling_rate", values.get("sample_rate", 0)))
        channels = int(values.get("number_channels", 1))
        if version is None:
            version = 2 if sample_rate == 48_000 or channels == 2 else 1
        quantizer = values.get("quantizer_kwargs")
        if not isinstance(quantizer, Mapping):
            raise KeyError("MOSS codec config is missing `quantizer_kwargs`.")
        quantizer_type = str(quantizer.get(
            "quantizer_type",
            values.get("quantizer_type", ""),
        ))
        return cls(
            version=int(version),
            sample_rate=sample_rate,
            downsample_rate=int(values.get("downsample_rate", 0)),
            channels=channels,
            code_dimension=int(values.get("code_dim", 0)),
            rvq_dimension=int(quantizer.get("rvq_dim", 0)),
            output_dimension=int(quantizer.get("output_dim", 0)),
            num_quantizers=int(quantizer.get("num_quantizers", 0)),
            codebook_size=int(quantizer.get("codebook_size", 0)),
            codebook_dimension=int(quantizer.get("codebook_dim", 0)),
            quantizer_type=quantizer_type,
            channel_interleave=bool(values.get("enable_channel_interleave", False)),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "model_type": "moss-audio-tokenizer",
            "voicehub_codec_version": self.version,
            "sampling_rate": self.sample_rate,
            "sample_rate": self.sample_rate,
            "downsample_rate": self.downsample_rate,
            "number_channels": self.channels,
            "enable_channel_interleave": self.channel_interleave,
            "code_dim": self.code_dimension,
            "quantizer_type": self.quantizer_type,
            "quantizer_kwargs": {
                "input_dim": self.code_dimension,
                "rvq_dim": self.rvq_dimension,
                "output_dim": self.output_dimension,
                "num_quantizers": self.num_quantizers,
                "codebook_size": self.codebook_size,
                "codebook_dim": self.codebook_dimension,
                "quantizer_type": self.quantizer_type,
            },
        }


@dataclass(frozen=True)
class MossCodecEncodeOutput:
    audio_codes: Tensor
    audio_code_lengths: Tensor


@dataclass(frozen=True)
class MossCodecDecodeOutput:
    waveform: Tensor
    waveform_lengths: Tensor
    sample_rate: int


@runtime_checkable
class MossAudioCodec(Protocol):
    """Complete codec interface accepted by the native MOSS runtime."""

    config: MossAudioCodecConfig

    def encode(
        self,
        waveforms: Tensor,
        lengths: Tensor | None = None,
        *,
        num_quantizers: int | None = None,
    ) -> MossCodecEncodeOutput:
        ...

    def decode(
        self,
        audio_codes: Tensor,
        lengths: Tensor | None = None,
    ) -> MossCodecDecodeOutput:
        ...


class MossCodecUnavailable:
    """Explicit boundary used when no complete native codec was supplied."""

    def __init__(self, config: MossAudioCodecConfig) -> None:
        self.config = config

    @staticmethod
    def _error(operation: str) -> RuntimeError:
        return RuntimeError(
            f"MOSS raw-audio {operation} is unavailable because codec loading "
            "was explicitly disabled. Load the runtime with `load_codec=True` "
            "or supply pre-encoded RVQ codes.")

    def encode(
        self,
        waveforms: Tensor,
        lengths: Tensor | None = None,
        *,
        num_quantizers: int | None = None,
    ) -> MossCodecEncodeOutput:
        del waveforms, lengths, num_quantizers
        raise self._error("encoding")

    def decode(
        self,
        audio_codes: Tensor,
        lengths: Tensor | None = None,
    ) -> MossCodecDecodeOutput:
        del audio_codes, lengths
        raise self._error("decoding")


class NativeMossAudioCodec(nn.Module):
    """Checkpoint-exact raw-waveform codec backed only by native PyTorch."""

    def __init__(
        self,
        model: nn.Module,
        config: MossAudioCodecConfig,
        *,
        architecture_config: Any | None = None,
        artifacts: Any | None = None,
        checkpoint_report: Any | None = None,
        frozen: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(model, nn.Module):
            raise TypeError("`model` must be a native MOSS codec graph.")
        if not isinstance(config, MossAudioCodecConfig):
            raise TypeError("`config` must be MossAudioCodecConfig.")
        self.model = model
        self.config = config
        self.architecture_config = architecture_config
        self.artifacts = artifacts
        self.checkpoint_report = checkpoint_report
        self.frozen = bool(frozen)
        if self.frozen:
            self.model.requires_grad_(False)
            self.model.eval()

    @classmethod
    def from_pretrained(
        cls,
        source: str | Path,
        *,
        revision: str | None = None,
        cache_dir: str | None = None,
        token: str | bool | None = None,
        local_files_only: bool = False,
        device: str | torch.device = "cpu",
        encoder_decoder_dtype: torch.dtype | None = None,
        frozen: bool = True,
    ) -> NativeMossAudioCodec:
        """Resolve and load an official or fine-tuned Safetensors codec."""
        from voicehub.models.mosstts.native.codec_checkpoint import load_moss_audio_tokenizer

        loaded = load_moss_audio_tokenizer(
            source,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
            device=device,
            encoder_decoder_dtype=encoder_decoder_dtype,
        )
        return cls(
            loaded.model,
            loaded.codec_config,
            architecture_config=loaded.architecture_config,
            artifacts=loaded.artifacts,
            checkpoint_report=loaded.report,
            frozen=frozen,
        )

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def input_dtype(self) -> torch.dtype:
        encoder = getattr(self.model, "encoder", None)
        if encoder is not None:
            parameter = next(encoder.parameters(), None)
            if parameter is not None:
                return parameter.dtype
        return next(self.model.parameters()).dtype

    def _waveform_batch(
        self,
        waveforms: Tensor,
        lengths: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        values = torch.as_tensor(waveforms)
        if values.ndim == 2 and self.config.channels == 1:
            values = values.unsqueeze(1)
        if values.ndim != 3:
            raise ValueError("MOSS codec waveforms must have shape [batch, channels, time].")
        if values.shape[1] != self.config.channels:
            raise ValueError(
                f"MOSS codec v{self.config.version} requires "
                f"{self.config.channels} waveform channel(s); found "
                f"{values.shape[1]}.")
        if values.shape[-1] < 1:
            raise ValueError("MOSS codec waveforms cannot be empty.")
        if not values.is_floating_point():
            values = values.float()
        if not bool(torch.isfinite(values).all()):
            raise ValueError("MOSS codec waveforms must contain finite values.")
        if lengths is None:
            normalized_lengths = torch.full(
                (values.shape[0], ),
                values.shape[-1],
                dtype=torch.long,
                device=self.device,
            )
        else:
            normalized_lengths = torch.as_tensor(
                lengths,
                dtype=torch.long,
                device=self.device,
            )
        if normalized_lengths.shape != (values.shape[0], ):
            raise ValueError("MOSS codec waveform lengths must have shape [batch].")
        if bool(((normalized_lengths < 1) | (normalized_lengths > values.shape[-1])).any()):
            raise ValueError("MOSS codec waveform lengths are outside the input.")
        values = values.to(device=self.device, dtype=self.input_dtype)
        positions = torch.arange(values.shape[-1], device=self.device)
        padding_mask = positions.unsqueeze(0) < normalized_lengths.unsqueeze(1)
        return values, normalized_lengths, padding_mask

    def _code_batch(
        self,
        audio_codes: Tensor,
        lengths: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        codes = torch.as_tensor(audio_codes)
        if codes.ndim != 3:
            raise ValueError("MOSS codec codes must have shape [batch, time, quantizers].")
        if (codes.dtype == torch.bool or codes.is_floating_point() or codes.is_complex()):
            raise TypeError("MOSS codec codes must use an integer dtype.")
        if not 1 <= codes.shape[2] <= self.config.num_quantizers:
            raise ValueError("MOSS codec codebook count is outside the checkpoint.")
        if codes.shape[1] < 1:
            raise ValueError("MOSS codec code sequences cannot be empty.")
        if bool(((codes < 0) | (codes >= self.config.codebook_size)).any()):
            raise ValueError("MOSS codec codes contain an out-of-range token.")
        codes = codes.to(device=self.device, dtype=torch.long)
        if lengths is None:
            normalized_lengths = torch.full(
                (codes.shape[0], ),
                codes.shape[1],
                dtype=torch.long,
                device=self.device,
            )
        else:
            normalized_lengths = torch.as_tensor(
                lengths,
                dtype=torch.long,
                device=self.device,
            )
        if normalized_lengths.shape != (codes.shape[0], ):
            raise ValueError("MOSS codec code lengths must have shape [batch].")
        if bool(((normalized_lengths < 1) | (normalized_lengths > codes.shape[1])).any()):
            raise ValueError("MOSS codec code lengths are outside the input.")
        positions = torch.arange(codes.shape[1], device=self.device)
        padding_mask = positions.unsqueeze(0) < normalized_lengths.unsqueeze(1)
        return (
            codes.permute(2, 0, 1).contiguous(),
            normalized_lengths,
            padding_mask,
        )

    def encode(
        self,
        waveforms: Tensor,
        lengths: Tensor | None = None,
        *,
        num_quantizers: int | None = None,
    ) -> MossCodecEncodeOutput:
        values, _, padding_mask = self._waveform_batch(waveforms, lengths)
        if num_quantizers is not None:
            num_quantizers = _positive_integer("num_quantizers", num_quantizers)
            if num_quantizers > self.config.num_quantizers:
                raise ValueError("`num_quantizers` exceeds the codec checkpoint.")
        context = torch.no_grad() if self.frozen else torch.enable_grad()
        with context:
            output = self.model.encode(
                input_values=values,
                padding_mask=padding_mask,
                num_quantizers=num_quantizers,
                return_dict=True,
            )
        codes = output.audio_codes
        code_lengths = output.audio_codes_lengths
        if codes is None or code_lengths is None:
            raise RuntimeError("Native MOSS codec encoder returned no codes.")
        if codes.ndim != 3 or codes.shape[1] != values.shape[0]:
            raise RuntimeError("Native MOSS codec encoder returned invalid codes.")
        return MossCodecEncodeOutput(
            audio_codes=codes.permute(1, 2, 0).contiguous(),
            audio_code_lengths=code_lengths.to(dtype=torch.long),
        )

    def decode(
        self,
        audio_codes: Tensor,
        lengths: Tensor | None = None,
    ) -> MossCodecDecodeOutput:
        codes, _, padding_mask = self._code_batch(audio_codes, lengths)
        context = torch.no_grad() if self.frozen else torch.enable_grad()
        with context:
            output = self.model.decode(
                audio_codes=codes,
                padding_mask=padding_mask,
                num_quantizers=codes.shape[0],
                return_dict=True,
            )
        waveform = output.audio
        waveform_lengths = output.audio_lengths
        if waveform is None or waveform_lengths is None:
            raise RuntimeError("Native MOSS codec decoder returned no waveform.")
        if waveform.ndim != 3 or waveform.shape[0] != codes.shape[1]:
            raise RuntimeError("Native MOSS codec decoder returned invalid audio.")
        return MossCodecDecodeOutput(
            waveform=waveform,
            waveform_lengths=waveform_lengths.to(dtype=torch.long),
            sample_rate=self.config.sample_rate,
        )

    def forward(
        self,
        waveforms: Tensor,
        lengths: Tensor | None = None,
        *,
        num_quantizers: int | None = None,
    ) -> tuple[MossCodecEncodeOutput, MossCodecDecodeOutput]:
        encoded = self.encode(
            waveforms,
            lengths,
            num_quantizers=num_quantizers,
        )
        decoded = self.decode(
            encoded.audio_codes,
            encoded.audio_code_lengths,
        )
        return encoded, decoded


def _weight_normalized_pointwise(
    input_channels: int,
    output_channels: int,
    *,
    device=None,
    dtype=None,
) -> nn.Module:
    convolution = nn.Conv1d(
        input_channels,
        output_channels,
        kernel_size=1,
        device=device,
        dtype=dtype,
    )
    return nn.utils.parametrizations.weight_norm(convolution)


class MossVectorQuantizer(nn.Module):
    """One source-compatible nearest-neighbour vector quantizer."""

    def __init__(
        self,
        input_dimension: int,
        codebook_size: int,
        codebook_dimension: int,
        *,
        normalize: bool,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.input_dimension = _positive_integer(
            "input_dimension",
            input_dimension,
        )
        self.codebook_size = _positive_integer("codebook_size", codebook_size)
        self.codebook_dimension = _positive_integer(
            "codebook_dimension",
            codebook_dimension,
        )
        self.normalize = bool(normalize)
        if input_dimension == codebook_dimension:
            self.in_proj = nn.Identity()
            self.out_proj = nn.Identity()
        else:
            self.in_proj = _weight_normalized_pointwise(
                input_dimension,
                codebook_dimension,
                device=device,
                dtype=dtype,
            )
            self.out_proj = _weight_normalized_pointwise(
                codebook_dimension,
                input_dimension,
                device=device,
                dtype=dtype,
            )
        self.codebook = nn.Embedding(
            codebook_size,
            codebook_dimension,
            device=device,
            dtype=dtype,
        )

    def decode_code_without_projection(self, token_ids: Tensor) -> Tensor:
        if token_ids.ndim != 2:
            raise ValueError("Quantizer token IDs must have shape [batch, time].")
        if token_ids.dtype == torch.bool or token_ids.is_floating_point():
            raise TypeError("Quantizer token IDs must use an integer dtype.")
        if bool(((token_ids < 0) | (token_ids >= self.codebook_size)).any()):
            raise ValueError("Quantizer token IDs are outside the codebook.")
        return functional.embedding(
            token_ids,
            self.codebook.weight,
        ).transpose(1, 2)

    def decode_code(self, token_ids: Tensor) -> Tensor:
        return self.out_proj(self.decode_code_without_projection(token_ids).float()).float()

    def _nearest(self, latents: Tensor) -> tuple[Tensor, Tensor]:
        flat = latents.transpose(1, 2).reshape(
            -1,
            latents.shape[1],
        ).float()
        codebook = self.codebook.weight.float()
        if self.normalize:
            flat = functional.normalize(flat, dim=-1)
            codebook = functional.normalize(codebook, dim=-1)
        distances = (
            flat.square().sum(dim=1, keepdim=True) - 2.0 * flat @ codebook.t() +
            codebook.square().sum(dim=1).unsqueeze(0))
        token_ids = distances.argmin(dim=1).reshape(
            latents.shape[0],
            latents.shape[-1],
        )
        quantized = self.decode_code_without_projection(token_ids).float()
        return quantized, token_ids

    def forward(self, values: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if values.ndim != 3:
            raise ValueError("Quantizer values must have shape [batch, channels, time].")
        encoded = self.in_proj(values.float()).float()
        quantized, token_ids = self._nearest(encoded)
        if self.normalize:
            quantized = encoded + (quantized - encoded).detach()
        return self.out_proj(quantized).float(), token_ids, encoded


class MossResidualQuantizer(nn.Module):
    """Residual VQ/LFQ stack matching the official tensor namespace."""

    def __init__(
        self,
        config: MossAudioCodecConfig,
        *,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if not isinstance(config, MossAudioCodecConfig):
            raise TypeError("`config` must be MossAudioCodecConfig.")
        self.config = config
        if config.code_dimension == config.rvq_dimension:
            self.input_proj = nn.Identity()
        else:
            self.input_proj = _weight_normalized_pointwise(
                config.code_dimension,
                config.rvq_dimension,
                device=device,
                dtype=dtype,
            )
        if config.rvq_dimension == config.output_dimension:
            self.output_proj = nn.Identity()
        else:
            self.output_proj = _weight_normalized_pointwise(
                config.rvq_dimension,
                config.output_dimension,
                device=device,
                dtype=dtype,
            )
        normalize = config.quantizer_type in {
            "rlfq",
            "random_prefix_rlfq",
        }
        self.quantizers = nn.ModuleList([
            MossVectorQuantizer(
                config.rvq_dimension,
                config.codebook_size,
                config.codebook_dimension,
                normalize=normalize,
                device=device,
                dtype=dtype,
            ) for _ in range(config.num_quantizers)
        ])

    def _quantizer_count(self, value: int | None) -> int:
        if value is None:
            return self.config.num_quantizers
        count = _positive_integer("num_quantizers", value)
        if count > self.config.num_quantizers:
            raise ValueError("`num_quantizers` exceeds the checkpoint inventory.")
        return count

    def forward(
        self,
        values: Tensor,
        lengths: Tensor,
        *,
        num_quantizers: int | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        count = self._quantizer_count(num_quantizers)
        if values.ndim != 3:
            raise ValueError("Residual quantizer values require [batch, channels, time].")
        if lengths.shape != (values.shape[0], ):
            raise ValueError("Quantizer lengths must have shape [batch].")
        if lengths.dtype == torch.bool or lengths.is_floating_point():
            raise TypeError("Quantizer lengths must use an integer dtype.")
        if bool(((lengths < 0) | (lengths > values.shape[-1])).any()):
            raise ValueError("Quantizer lengths are outside the input extent.")
        encoded = self.input_proj(values).float()
        mask = (torch.arange(encoded.shape[-1], device=encoded.device).unsqueeze(0)
                < lengths.unsqueeze(1)).unsqueeze(1)
        residual = encoded
        quantized = torch.zeros_like(encoded)
        token_ids = []
        for quantizer in self.quantizers[:count]:
            update, indices, _ = quantizer(residual * mask)
            quantized = quantized + update * mask
            residual = residual - update * mask
            token_ids.append(indices)
        return (
            self.output_proj(quantized),
            torch.stack(token_ids),
            lengths,
        )

    def decode_codes(self, audio_codes: Tensor) -> Tensor:
        if audio_codes.ndim != 3:
            raise ValueError("Audio codes must have shape [quantizers, batch, time].")
        count = int(audio_codes.shape[0])
        if not 1 <= count <= self.config.num_quantizers:
            raise ValueError("Audio-code quantizer count is unsupported.")
        if (audio_codes.dtype == torch.bool or audio_codes.is_floating_point() or audio_codes.is_complex()):
            raise TypeError("Audio codes must use an integer dtype.")
        if bool(((audio_codes < 0) | (audio_codes >= self.config.codebook_size)).any()):
            raise ValueError("Audio codes contain an out-of-range token.")
        batch_size, time = audio_codes.shape[1:]
        values = torch.zeros(
            batch_size,
            self.config.rvq_dimension,
            time,
            device=audio_codes.device,
            dtype=torch.float32,
        )
        for index, quantizer in enumerate(self.quantizers[:count]):
            values = values + quantizer.decode_code(audio_codes[index])
        return self.output_proj(values)


def validate_preencoded_codes(
    audio_codes: Tensor,
    config: MossAudioCodecConfig,
    *,
    expected_quantizers: int | None = None,
) -> Tensor:
    """Validate and normalize one ``[time, quantizers]`` code matrix."""
    if not isinstance(audio_codes, Tensor) or audio_codes.ndim != 2:
        raise ValueError("Pre-encoded MOSS audio must have shape [time, quantizers].")
    if (audio_codes.dtype == torch.bool or audio_codes.is_floating_point() or audio_codes.is_complex()):
        raise TypeError("Pre-encoded MOSS audio must use an integer dtype.")
    quantizers = (
        config.num_quantizers if expected_quantizers is None else _positive_integer(
            "expected_quantizers", expected_quantizers))
    if audio_codes.shape[1] != quantizers:
        raise ValueError(
            f"Pre-encoded MOSS audio requires {quantizers} quantizers; "
            f"found {audio_codes.shape[1]}.")
    if audio_codes.shape[0] < 1:
        raise ValueError("Pre-encoded MOSS audio cannot be empty.")
    if bool(((audio_codes < 0) | (audio_codes >= config.codebook_size)).any()):
        raise ValueError("Pre-encoded MOSS audio contains an invalid code.")
    return audio_codes.to(dtype=torch.long)


def codec_duration_seconds(
    frame_count: int,
    config: MossAudioCodecConfig,
) -> float:
    """Convert codec-frame count to waveform duration."""
    if isinstance(frame_count, bool) or not isinstance(frame_count, int):
        raise TypeError("`frame_count` must be an integer.")
    if frame_count < 0:
        raise ValueError("`frame_count` cannot be negative.")
    duration = frame_count * config.downsample_rate / config.sample_rate
    if not math.isfinite(duration):
        raise RuntimeError("Codec duration calculation overflowed.")
    return duration


__all__ = [
    "MossAudioCodec",
    "MossAudioCodecConfig",
    "MossCodecDecodeOutput",
    "MossCodecEncodeOutput",
    "MossCodecUnavailable",
    "MossResidualQuantizer",
    "MossVectorQuantizer",
    "NativeMossAudioCodec",
    "codec_duration_seconds",
    "validate_preencoded_codes",
]
