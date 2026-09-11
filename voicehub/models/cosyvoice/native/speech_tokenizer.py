"""PyTorch-native CosyVoice 3 supervised semantic speech tokenizer.

The published CosyVoice 3 artifact exposes this frozen encoder only as
``speech_tokenizer_v3.onnx``.  The graph below is an audited reconstruction of
that immutable artifact and the Apache-2.0 S3Tokenizer v3 implementation at
``xingchensong/S3Tokenizer@9bf5d845b5e043ffaf4657f4942939091c7697a2``.

Normal inference never imports or executes ONNX.  The one-time converter in
``checkpoint.py`` verifies the complete published file before extracting its
initializers into a strict Safetensors checkpoint.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional

from voicehub.optimization.protocols import OptimizationCompileTarget
from voicehub.processing.audio import mel_filter_bank
from voicehub.processing.waveform import NativeAudio, load_native_audio


def _positive_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"`{name}` must be an integer.")
    if value <= 0:
        raise ValueError(f"`{name}` must be positive.")
    return value


@dataclass(frozen=True, slots=True)
class CosyVoiceSpeechTokenizerConfig:
    """Frozen S3Tokenizer v3 encoder and finite-scalar quantizer geometry."""

    sample_rate: int = 16_000
    n_fft: int = 400
    hop_length: int = 160
    n_mels: int = 128
    hidden_size: int = 1_280
    num_attention_heads: int = 20
    num_hidden_layers: int = 12
    fsmn_kernel_size: int = 31
    convolution_stride: int = 2
    fsq_dimension: int = 8
    fsq_level: int = 3
    max_positions: int = 2_048
    max_segment_frames: int = 3_000
    segment_overlap_seconds: int = 4
    use_sdpa: bool = False

    def __post_init__(self) -> None:
        for name in (
                "sample_rate",
                "n_fft",
                "hop_length",
                "n_mels",
                "hidden_size",
                "num_attention_heads",
                "num_hidden_layers",
                "fsmn_kernel_size",
                "convolution_stride",
                "fsq_dimension",
                "fsq_level",
                "max_positions",
                "max_segment_frames",
                "segment_overlap_seconds",
        ):
            _positive_integer(name, getattr(self, name))
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("`hidden_size` must be divisible by `num_attention_heads`.")
        if self.head_dimension % 2:
            raise ValueError("Speech-tokenizer attention head size must be even.")
        if self.fsmn_kernel_size % 2 == 0:
            raise ValueError("`fsmn_kernel_size` must be odd.")
        if self.convolution_stride != 2:
            raise ValueError("The audited CosyVoice 3 tokenizer requires stride-two convolutions.")
        if self.fsq_level != 3 or self.fsq_dimension != 8:
            raise ValueError(
                "The audited CosyVoice 3 tokenizer requires eight ternary "
                "finite-scalar dimensions.")
        if self.max_segment_frames <= self.segment_overlap_frames:
            raise ValueError("Tokenizer segment overlap must be shorter than a segment.")
        if not isinstance(self.use_sdpa, bool):
            raise TypeError("`use_sdpa` must be a boolean.")

    @property
    def head_dimension(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def codebook_size(self) -> int:
        return self.fsq_level**self.fsq_dimension

    @property
    def segment_overlap_frames(self) -> int:
        return (self.segment_overlap_seconds * self.sample_rate // self.hop_length)

    @property
    def segment_stride_frames(self) -> int:
        return self.max_segment_frames - self.segment_overlap_frames

    @classmethod
    def tiny(cls) -> CosyVoiceSpeechTokenizerConfig:
        """Return a small graph with the exact public checkpoint topology."""
        return cls(
            n_mels=8,
            hidden_size=32,
            num_attention_heads=4,
            num_hidden_layers=2,
            fsmn_kernel_size=7,
            max_positions=128,
            max_segment_frames=400,
            segment_overlap_seconds=1,
        )

    @classmethod
    def from_dict(
        cls,
        values: Mapping[str, Any],
    ) -> CosyVoiceSpeechTokenizerConfig:
        if not isinstance(values, Mapping):
            raise TypeError("Speech-tokenizer configuration must be a mapping.")
        unknown = set(values) - {field_name for field_name in cls.__dataclass_fields__}
        if unknown:
            raise ValueError("Unknown speech-tokenizer configuration fields: " + ", ".join(sorted(unknown)))
        return cls(**dict(values))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _FloatLayerNorm(nn.LayerNorm):

    def forward(self, value: Tensor) -> Tensor:
        return functional.layer_norm(
            value.float(),
            self.normalized_shape,
            None if self.weight is None else self.weight.float(),
            None if self.bias is None else self.bias.float(),
            self.eps,
        ).to(dtype=value.dtype)


class _DTypeLinear(nn.Linear):

    def forward(self, value: Tensor) -> Tensor:
        return functional.linear(
            value,
            self.weight.to(dtype=value.dtype),
            None if self.bias is None else self.bias.to(dtype=value.dtype),
        )


class _DTypeConv1d(nn.Conv1d):

    def _conv_forward(
        self,
        value: Tensor,
        weight: Tensor,
        bias: Tensor | None,
    ) -> Tensor:
        return super()._conv_forward(
            value,
            weight.to(dtype=value.dtype),
            None if bias is None else bias.to(dtype=value.dtype),
        )


def _non_padding_mask(lengths: Tensor, maximum: int) -> Tensor:
    positions = torch.arange(
        maximum,
        device=lengths.device,
        dtype=lengths.dtype,
    )
    return positions.unsqueeze(0) < lengths.unsqueeze(1)


def _rope_tables(
    dimension: int,
    positions: int,
    *,
    device: torch.device | str | None = None,
) -> tuple[Tensor, Tensor]:
    inverse = 1.0 / (
        10_000.0**(torch.arange(
            0,
            dimension,
            2,
            dtype=torch.float32,
            device=device,
        ) / dimension))
    phase = torch.outer(
        torch.arange(
            positions,
            dtype=torch.float32,
            device=device,
        ),
        inverse,
    )
    return (
        torch.cat((phase.cos(), phase.cos()), dim=-1),
        torch.cat((phase.sin(), phase.sin()), dim=-1),
    )


def _apply_rotary(
    query: Tensor,
    key: Tensor,
    cosine: Tensor,
    sine: Tensor,
) -> tuple[Tensor, Tensor]:
    cosine = cosine[None, :, None].to(dtype=query.dtype)
    sine = sine[None, :, None].to(dtype=query.dtype)

    def rotate(value: Tensor) -> Tensor:
        left, right = value.chunk(2, dim=-1)
        return torch.cat((-right, left), dim=-1)

    return (
        query * cosine + rotate(query) * sine,
        key * cosine + rotate(key) * sine,
    )


class _FSMNMultiHeadAttention(nn.Module):

    def __init__(self, config: CosyVoiceSpeechTokenizerConfig) -> None:
        super().__init__()
        self.n_head = config.num_attention_heads
        self.use_sdpa = config.use_sdpa
        dimension = config.hidden_size
        self.query = _DTypeLinear(dimension, dimension)
        self.key = _DTypeLinear(dimension, dimension, bias=False)
        self.value = _DTypeLinear(dimension, dimension)
        self.out = _DTypeLinear(dimension, dimension)
        self.fsmn_block = nn.Conv1d(
            dimension,
            dimension,
            config.fsmn_kernel_size,
            stride=1,
            padding=0,
            groups=dimension,
            bias=False,
        )
        self.left_padding = (config.fsmn_kernel_size - 1) // 2
        self.right_padding = (config.fsmn_kernel_size - 1 - self.left_padding)

    def _fsmn(self, value: Tensor, valid: Tensor) -> Tensor:
        batch, frames, _heads, _dimension = value.shape
        mask = valid.unsqueeze(-1).to(dtype=value.dtype)
        flattened = value.reshape(batch, frames, -1) * mask
        memory = functional.pad(
            flattened.transpose(1, 2),
            (self.left_padding, self.right_padding),
        )
        memory = self.fsmn_block(memory).transpose(1, 2)
        return (memory + flattened) * mask

    def forward(
        self,
        value: Tensor,
        *,
        valid: Tensor,
        cosine: Tensor,
        sine: Tensor,
    ) -> Tensor:
        query = self.query(value)
        key = self.key(value)
        projected_value = self.value(value)
        batch, frames, dimension = query.shape
        head_dimension = dimension // self.n_head
        query = query.reshape(
            batch,
            frames,
            self.n_head,
            head_dimension,
        )
        key = key.reshape(
            batch,
            frames,
            self.n_head,
            head_dimension,
        )
        projected_value = projected_value.reshape(
            batch,
            frames,
            self.n_head,
            head_dimension,
        )
        query, key = _apply_rotary(query, key, cosine, sine)
        memory = self._fsmn(projected_value, valid)
        scale = head_dimension**-0.25
        query = query.permute(0, 2, 1, 3) * scale
        projected_value = projected_value.permute(0, 2, 1, 3)
        attention_bias = (~valid)[:, None, None].to(dtype=query.dtype, ) * -1.0e10
        if self.use_sdpa:
            key = key.permute(0, 2, 1, 3) * scale
            attended = functional.scaled_dot_product_attention(
                query,
                key,
                projected_value,
                attn_mask=attention_bias,
                dropout_p=0.0,
                scale=1.0,
            )
        else:
            key = key.permute(0, 2, 3, 1) * scale
            scores = (query @ key) + attention_bias
            probabilities = scores.float().softmax(dim=-1).to(dtype=query.dtype)
            attended = probabilities @ projected_value
        attended = attended.permute(0, 2, 1, 3).flatten(start_dim=2)
        return self.out(attended) + memory


class _ResidualAttentionBlock(nn.Module):

    def __init__(self, config: CosyVoiceSpeechTokenizerConfig) -> None:
        super().__init__()
        self.attn = _FSMNMultiHeadAttention(config)
        self.attn_ln = _FloatLayerNorm(
            config.hidden_size,
            eps=1e-5,
        )
        self.mlp = nn.Sequential(
            _DTypeLinear(config.hidden_size, config.hidden_size * 4),
            nn.GELU(),
            _DTypeLinear(config.hidden_size * 4, config.hidden_size),
        )
        self.mlp_ln = _FloatLayerNorm(
            config.hidden_size,
            eps=1e-5,
        )

    def forward(
        self,
        value: Tensor,
        *,
        valid: Tensor,
        cosine: Tensor,
        sine: Tensor,
    ) -> Tensor:
        value = value + self.attn(
            self.attn_ln(value),
            valid=valid,
            cosine=cosine,
            sine=sine,
        )
        return value + self.mlp(self.mlp_ln(value))


class CosyVoiceSpeechTokenizerEncoder(nn.Module):
    """Two-stage convolutional subsampler and RoPE/FSMN transformer."""

    def __init__(self, config: CosyVoiceSpeechTokenizerConfig) -> None:
        super().__init__()
        self.config = config
        self.stride = config.convolution_stride
        self.conv1 = _DTypeConv1d(
            config.n_mels,
            config.hidden_size,
            kernel_size=3,
            stride=self.stride,
            padding=1,
        )
        self.conv2 = _DTypeConv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.blocks = nn.ModuleList(
            [_ResidualAttentionBlock(config) for _ in range(config.num_hidden_layers)])
        cosine, sine = _rope_tables(
            config.head_dimension,
            config.max_positions,
        )
        self.register_buffer("_rope_cosine", cosine, persistent=False)
        self.register_buffer("_rope_sine", sine, persistent=False)

    def materialize_runtime_buffers(
        self,
        device: torch.device | str,
    ) -> None:
        cosine, sine = _rope_tables(
            self.config.head_dimension,
            self.config.max_positions,
            device=device,
        )
        self._rope_cosine = cosine
        self._rope_sine = sine

    def forward(
        self,
        features: Tensor,
        feature_lengths: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if not isinstance(features, Tensor) or features.ndim != 3:
            raise ValueError("Speech-tokenizer features must have shape [batch, mel, frame].")
        if features.shape[1] != self.config.n_mels:
            raise ValueError(f"Speech-tokenizer features require {self.config.n_mels} mel bins.")
        if not features.is_floating_point():
            raise TypeError("Speech-tokenizer features must be floating point.")
        if (not isinstance(feature_lengths, Tensor) or feature_lengths.shape != (features.shape[0], )):
            raise ValueError("`feature_lengths` must have one value per feature batch item.")
        lengths = feature_lengths.to(
            device=features.device,
            dtype=torch.long,
        )
        if bool((lengths <= 0).any()) or bool((lengths > features.shape[-1]).any()):
            raise ValueError("Speech-tokenizer feature lengths are out of range.")

        valid = _non_padding_mask(lengths, features.shape[-1])
        value = functional.gelu(self.conv1(features * valid.unsqueeze(1)))
        lengths = (lengths - 1) // self.stride + 1
        valid = _non_padding_mask(lengths, value.shape[-1])
        value = functional.gelu(self.conv2(value * valid.unsqueeze(1)))
        lengths = (lengths - 1) // 2 + 1
        valid = _non_padding_mask(lengths, value.shape[-1])
        value = value.transpose(1, 2)
        if value.shape[1] > self.config.max_positions:
            raise ValueError("Speech-tokenizer sequence exceeds the configured RoPE table.")
        cosine = self._rope_cosine[:value.shape[1]].to(device=value.device)
        sine = self._rope_sine[:value.shape[1]].to(device=value.device)
        for block in self.blocks:
            value = block(
                value,
                valid=valid,
                cosine=cosine,
                sine=sine,
            )
        return value, lengths


class CosyVoiceFiniteScalarQuantizer(nn.Module):
    """Eight-dimensional ternary FSQ used by the 6,561-token vocabulary."""

    def __init__(self, config: CosyVoiceSpeechTokenizerConfig) -> None:
        super().__init__()
        self.config = config
        self._codebook = nn.Module()
        self._codebook.project_down = _DTypeLinear(
            config.hidden_size,
            config.fsq_dimension,
        )

    @property
    def codebook_size(self) -> int:
        return self.config.codebook_size

    def encode(self, value: Tensor) -> Tensor:
        shape = value.shape[:-1]
        projected = self._codebook.project_down(value.reshape(-1, value.shape[-1])).float()
        digits = (projected.tanh() * 0.9990000128746033).round() + 1.0
        powers = torch.pow(
            float(self.config.fsq_level),
            torch.arange(
                self.config.fsq_dimension,
                device=value.device,
                dtype=digits.dtype,
            ),
        )
        return (digits * powers).sum(dim=-1).reshape(shape).to(dtype=torch.long)

    def forward(self, value: Tensor) -> Tensor:
        return self.encode(value)


class CosyVoiceSpeechTokenizer(nn.Module):
    """Frozen waveform-to-semantic-token boundary for CosyVoice 3."""

    is_stochastic_vae = False
    sample_rate = 16_000

    def __init__(
        self,
        config: CosyVoiceSpeechTokenizerConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or CosyVoiceSpeechTokenizerConfig()
        self.sample_rate = self.config.sample_rate
        self.encoder = CosyVoiceSpeechTokenizerEncoder(self.config)
        self.quantizer = CosyVoiceFiniteScalarQuantizer(self.config)
        filters = mel_filter_bank(
            sample_rate=self.config.sample_rate,
            n_fft=self.config.n_fft,
            n_mels=self.config.n_mels,
        )
        self.register_buffer("_mel_filters", filters, persistent=False)
        self.register_buffer(
            "_window",
            torch.hann_window(self.config.n_fft),
            persistent=False,
        )
        self.freeze()

    @property
    def codebook_size(self) -> int:
        return self.config.codebook_size

    def freeze(self) -> CosyVoiceSpeechTokenizer:
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        return self

    def materialize_runtime_buffers(
        self,
        device: torch.device | str,
    ) -> None:
        self.encoder.materialize_runtime_buffers(device)
        self._mel_filters = mel_filter_bank(
            sample_rate=self.config.sample_rate,
            n_fft=self.config.n_fft,
            n_mels=self.config.n_mels,
            device=device,
        )
        self._window = torch.hann_window(
            self.config.n_fft,
            device=device,
        )

    def optimization_module_roots(self, ) -> tuple[tuple[str, nn.Module], ...]:
        return (
            ("cosyvoice.speech_tokenizer.encoder", self.encoder),
            ("cosyvoice.speech_tokenizer.quantizer", self.quantizer),
        )

    def codec_optimization_compile_targets(
        self,
        mode: str,
    ) -> tuple[OptimizationCompileTarget, ...]:
        if mode not in {"inference", "training"}:
            raise ValueError(f"Unsupported optimization mode {mode!r}.")
        return (
            OptimizationCompileTarget(
                "codec.encode.cosyvoice_s3",
                self,
                "forward",
                component="encode",
            ), )

    def forward(
        self,
        features: Tensor,
        feature_lengths: Tensor,
    ) -> tuple[Tensor, Tensor]:
        hidden, token_lengths = self.encoder(
            features,
            feature_lengths,
        )
        tokens = self.quantizer(hidden)
        return tokens, token_lengths

    def log_mel_spectrogram(self, waveform: Tensor) -> Tensor:
        waveform = torch.as_tensor(waveform)
        if waveform.ndim != 1:
            raise ValueError("Tokenizer waveform input must be one-dimensional.")
        if not waveform.is_floating_point():
            waveform = waveform.float()
        if waveform.numel() <= self.config.n_fft // 2:
            raise ValueError("Tokenizer waveform is too short for reflect-padded STFT.")
        spectrum = torch.stft(
            waveform,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            window=self._window.to(
                device=waveform.device,
                dtype=waveform.dtype,
            ),
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )
        power = spectrum[..., :-1].abs().square()
        mel = self._mel_filters.to(
            device=waveform.device,
            dtype=power.dtype,
        ) @ power
        logarithm = mel.clamp_min(1e-10).log10()
        logarithm = torch.maximum(logarithm, logarithm.amax() - 8.0)
        return (logarithm + 4.0) / 4.0

    def _encode_long_features(
        self,
        features: Tensor,
    ) -> Tensor:
        maximum = self.config.max_segment_frames
        stride = self.config.segment_stride_frames
        segments = []
        start = 0
        while start < features.shape[-1]:
            segment = features[:, start:start + maximum]
            valid = segment.shape[-1]
            if valid < maximum:
                segment = functional.pad(segment, (0, maximum - valid))
            tokens, token_lengths = self(
                segment.unsqueeze(0),
                torch.tensor(
                    [valid],
                    dtype=torch.long,
                    device=features.device,
                ),
            )
            segments.append(tokens[0, :token_lengths[0]])
            start += stride
        overlap_tokens = (self.config.segment_overlap_seconds * 25 // 2)
        trimmed = []
        for index, segment in enumerate(segments):
            left = 0 if index == 0 else overlap_tokens
            right = (segment.shape[0] if index == len(segments) - 1 else -overlap_tokens)
            trimmed.append(segment[left:right])
        return torch.cat(trimmed)

    @torch.inference_mode()
    def encode_waveforms(
        self,
        waveforms: Tensor | Sequence[Tensor | NativeAudio],
        *,
        sampling_rate: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Extract padded semantic tokens from raw mono audio.

        Tensor inputs are interpreted as one waveform for rank one or an
        equal-length batch for rank two.  A sequence may mix tensors and
        :class:`NativeAudio` values.  Plain tensors default to the
        tokenizer's 16 kHz rate unless ``sampling_rate`` is explicit.
        """
        if isinstance(waveforms, Tensor):
            if waveforms.ndim == 1:
                entries: Sequence[Tensor | NativeAudio] = (waveforms, )
            elif waveforms.ndim == 2:
                entries = tuple(waveforms)
            else:
                raise ValueError("Raw tokenizer tensor must have shape [sample] or "
                                 "[batch, sample].")
        else:
            if isinstance(waveforms, (str, bytes)) or not isinstance(waveforms, Sequence):
                raise TypeError("`waveforms` must be a tensor or sequence of audio tensors.")
            entries = waveforms
        if not entries:
            raise ValueError("Tokenizer waveform batch cannot be empty.")

        device = next(self.parameters()).device
        encoded: list[Tensor] = []
        for entry in entries:
            if isinstance(entry, NativeAudio):
                audio = load_native_audio(
                    entry,
                    target_sampling_rate=self.config.sample_rate,
                )
            else:
                audio = load_native_audio(
                    entry,
                    sampling_rate=(self.config.sample_rate if sampling_rate is None else sampling_rate),
                    target_sampling_rate=self.config.sample_rate,
                )
            features = self.log_mel_spectrogram(audio.waveform.to(device=device))
            if features.shape[-1] > self.config.max_segment_frames:
                tokens = self._encode_long_features(features)
            else:
                values, lengths = self(
                    features.unsqueeze(0),
                    torch.tensor(
                        [features.shape[-1]],
                        dtype=torch.long,
                        device=device,
                    ),
                )
                tokens = values[0, :lengths[0]]
            encoded.append(tokens)

        lengths = torch.tensor(
            [value.shape[0] for value in encoded],
            dtype=torch.long,
            device=device,
        )
        output = torch.zeros(
            len(encoded),
            int(lengths.max().item()),
            dtype=torch.long,
            device=device,
        )
        for index, value in enumerate(encoded):
            output[index, :value.shape[0]] = value
        return output, lengths


__all__ = [
    "CosyVoiceFiniteScalarQuantizer",
    "CosyVoiceSpeechTokenizer",
    "CosyVoiceSpeechTokenizerConfig",
    "CosyVoiceSpeechTokenizerEncoder",
]
