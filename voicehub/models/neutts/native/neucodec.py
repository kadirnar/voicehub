"""VoiceHub-owned NeuCodec graph and frontend.

NeuCodec shares the audited XCodec2 neural topology but uses a different
frontend and a 24 kHz synthesis hop.  This module owns those semantics
while reusing the already-native PyTorch building blocks.
"""

from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from voicehub.models.llasa.xcodec2 import (
    Wav2Vec2BertSemanticModel,
    XCodec2Decoder,
    XCodec2DownSample1d,
    XCodec2Encoder,
    XCodec2FiniteScalarQuantization,
    XCodec2ISTFTHead,
    XCodec2Quantizer,
    XCodec2SemanticAdapter,
    XCodec2SnakeBeta,
    XCodec2UpSample1d,
    kaiser_sinc_filter1d,
)
from voicehub.models.neutts.native.configuration import NeuCodecConfig
from voicehub.processing.kaldi import KaldiFbankConfig, kaldi_fbank


@dataclass(frozen=True, slots=True)
class NeuCodecFeatures:
    input_values: Tensor
    input_features: Tensor
    padding_mask: Tensor
    input_features_mask: Tensor


@dataclass(frozen=True, slots=True)
class NeuCodecEncoderOutput:
    audio_codes: Tensor
    latents: Tensor | None = None
    audio_codes_mask: Tensor | None = None


@dataclass(frozen=True, slots=True)
class NeuCodecDecoderOutput:
    audio_values: Tensor


@dataclass(frozen=True, slots=True)
class NeuCodecOutput:
    audio_values: Tensor
    audio_codes: Tensor
    latents: Tensor | None = None
    audio_codes_mask: Tensor | None = None


class NeuCodecFeatureExtractor(nn.Module):
    """PyTorch/stdlib frontend matching the official 16 kHz processor."""

    def __init__(self, config: NeuCodecConfig) -> None:
        super().__init__()
        self.config = config
        self.kaldi_config = KaldiFbankConfig(
            sample_frequency=float(config.input_sampling_rate),
            frame_length=25.0,
            frame_shift=10.0,
            num_mel_bins=80,
            dither=0.0,
            low_frequency=20.0,
            high_frequency=float(config.input_sampling_rate // 2),
            preemphasis_coefficient=0.97,
            remove_dc_offset=True,
            use_log_fbank=True,
            use_energy=False,
            snip_edges=True,
            window_type="povey",
        )

    def validate_preprocessor_config(self, values: dict[str, Any]) -> None:
        if not isinstance(values, dict):
            raise TypeError("NeuCodec preprocessor config must be a mapping.")
        expected = {
            "feature_extractor_type": "NeuCodecFeatureExtractor",
            "feature_size": 80,
            "frame_length": 400,
            "frame_shift": 160,
            "hop_length": self.config.encoder_hop_length,
            "num_mel_bins": 80,
            "padding_side": "right",
            "padding_value": 1,
            "return_attention_mask": True,
            "sampling_rate": self.config.input_sampling_rate,
            "stride": 2,
        }
        mismatches = {
            name: (expected_value, values[name])
            for name, expected_value in expected.items() if name in values and values[name] != expected_value
        }
        if mismatches:
            details = ", ".join(
                f"{name}={actual!r} (expected {expected_value!r})"
                for name, (expected_value, actual) in sorted(mismatches.items()))
            raise ValueError("Unsupported NeuCodec frontend metadata: " + details + ".")

    def forward(
        self,
        waveforms: Tensor,
        waveform_lengths: Tensor | None = None,
    ) -> NeuCodecFeatures:
        if not isinstance(waveforms, Tensor):
            raise TypeError("NeuCodec waveforms must be a PyTorch tensor.")
        if waveforms.ndim == 1:
            waveforms = waveforms.unsqueeze(0)
        if waveforms.ndim == 3 and waveforms.shape[1] == 1:
            waveforms = waveforms[:, 0]
        if waveforms.ndim != 2:
            raise ValueError("NeuCodec waveforms must have shape [batch, samples].")
        if not waveforms.is_floating_point() or waveforms.is_complex():
            raise TypeError("NeuCodec waveforms must use a real floating dtype.")
        if not torch.isfinite(waveforms).all():
            raise ValueError("NeuCodec waveforms contain NaN or infinite values.")
        batch_size, maximum = waveforms.shape
        if waveform_lengths is None:
            lengths = torch.full(
                (batch_size, ),
                maximum,
                dtype=torch.long,
                device=waveforms.device,
            )
        else:
            lengths = torch.as_tensor(
                waveform_lengths,
                dtype=torch.long,
                device=waveforms.device,
            )
            if tuple(lengths.shape) != (batch_size, ):
                raise ValueError("`waveform_lengths` must contain one value per example.")
            if bool((lengths <= 0).any()) or bool((lengths > maximum).any()):
                raise ValueError("NeuCodec waveform lengths must lie inside the batch.")

        hop = self.config.encoder_hop_length
        padded_lengths = (lengths + 1 + hop - 1) // hop * hop
        padded_width = int(padded_lengths.max().item())
        input_values = waveforms.new_zeros((batch_size, 1, padded_width))
        padding_mask = torch.zeros(
            (batch_size, padded_width),
            dtype=torch.long,
            device=waveforms.device,
        )
        feature_rows: list[Tensor] = []
        feature_lengths: list[int] = []
        for index, length in enumerate(lengths.tolist()):
            copied = int(length)
            input_values[index, 0, :copied] = waveforms[index, :copied]
            # One valid zero is part of the published processor contract.
            padding_mask[index, :copied + 1] = 1
            padded_length = int(padded_lengths[index].item())
            features = kaldi_fbank(
                input_values[index, :, :padded_length] * (2**15),
                self.kaldi_config,
            )
            if features.shape[0] < 2:
                raise ValueError("NeuCodec audio is too short to normalize semantic frames.")
            features = (features -
                        features.mean(dim=0)) / torch.sqrt(features.var(dim=0, unbiased=True) + 1e-7)
            feature_rows.append(features)
            feature_lengths.append(features.shape[0])

        maximum_frames = max(feature_lengths)
        if maximum_frames % 2:
            maximum_frames += 1
        padded_features = waveforms.new_full(
            (batch_size, maximum_frames, 80),
            1.0,
        )
        feature_mask = torch.zeros(
            (batch_size, maximum_frames),
            dtype=torch.long,
            device=waveforms.device,
        )
        for index, features in enumerate(feature_rows):
            padded_features[index, :features.shape[0]] = features
            feature_mask[index, :features.shape[0]] = 1
        return NeuCodecFeatures(
            input_values=input_values,
            input_features=padded_features.reshape(
                batch_size,
                maximum_frames // 2,
                160,
            ),
            padding_mask=padding_mask,
            input_features_mask=feature_mask.reshape(
                batch_size,
                maximum_frames // 2,
                2,
            ).amin(dim=-1),
        )


class NeuCodecModel(nn.Module):
    """Complete trainable graph for the safe official NeuCodec conversion."""

    def __init__(
        self,
        config: NeuCodecConfig | dict[str, Any],
        *,
        initialize: bool = True,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        self.config = (config if isinstance(config, NeuCodecConfig) else NeuCodecConfig.from_dict(config))
        self.hop_length = self.config.encoder_hop_length
        context = torch.device(device) if device is not None else nullcontext()
        with context:
            self.feature_extractor = NeuCodecFeatureExtractor(self.config)
            self.semantic_encoder = Wav2Vec2BertSemanticModel(self.config.semantic_model_config)
            self.semantic_adapter = XCodec2SemanticAdapter(self.config)
            self.acoustic_encoder = XCodec2Encoder(self.config)
            self.fc_encoder = nn.Linear(
                self.config.quantization_dim,
                self.config.quantization_dim,
            )
            self.quantizer = XCodec2Quantizer(self.config)
            self.acoustic_decoder = XCodec2Decoder(self.config)
        if initialize:
            self.apply(self._initialize_module)
            self._reset_derived_buffers()

    @property
    def input_sampling_rate(self) -> int:
        return self.config.input_sampling_rate

    @property
    def output_sampling_rate(self) -> int:
        return self.config.output_sampling_rate

    @property
    def sampling_rate(self) -> int:
        """The waveform rate produced by :meth:`decode_code`."""
        return self.output_sampling_rate

    def _initialize_module(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.initializer_range,
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.initializer_range,
            )
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif module.__class__.__name__ == "XCodec2RMSNorm":
            nn.init.ones_(module.weight)
        elif isinstance(module, XCodec2SnakeBeta):
            nn.init.zeros_(module.alpha)
            nn.init.zeros_(module.beta)
        elif isinstance(module, Wav2Vec2BertSemanticModel):
            if module.masked_spec_embed is not None:
                nn.init.uniform_(module.masked_spec_embed)

    def _reset_derived_buffers(self) -> None:
        for module in self.modules():
            if isinstance(module, XCodec2ISTFTHead):
                module.window.copy_(
                    torch.hann_window(
                        module.n_fft,
                        device=module.window.device,
                        dtype=module.window.dtype,
                    ))
            elif isinstance(module, XCodec2FiniteScalarQuantization):
                levels, basis, codebook = module._compute_buffers(device=module.levels.device)
                module.levels.copy_(levels)
                module.basis.copy_(basis)
                module.codebook.copy_(codebook)
            elif isinstance(module, XCodec2DownSample1d):
                module.filter.copy_(
                    kaiser_sinc_filter1d(
                        module.cutoff,
                        module.half_width,
                        module.kernel_size,
                    ).to(module.filter.device))
            elif isinstance(module, XCodec2UpSample1d):
                module.filter.copy_(
                    kaiser_sinc_filter1d(
                        0.5 / module.ratio,
                        0.6 / module.ratio,
                        module.kernel_size,
                    ).to(module.filter.device))

    def encode(
        self,
        input_values: Tensor,
        input_features: Tensor,
        *,
        padding_mask: Tensor | None = None,
        input_features_mask: Tensor | None = None,
        output_latents: bool = False,
    ) -> NeuCodecEncoderOutput:
        with torch.no_grad():
            semantic = self.semantic_encoder(
                input_features,
                attention_mask=input_features_mask,
            )
        semantic = self.semantic_adapter(semantic.transpose(1, 2))
        acoustic = self.acoustic_encoder(input_values)
        semantic, acoustic = self._align_encoder_frames(semantic, acoustic)
        hidden_states = self.fc_encoder(torch.cat((semantic, acoustic), dim=1).transpose(1, 2))
        latents, audio_codes = self.quantizer(hidden_states)
        latents = latents.transpose(1, 2)
        audio_codes = audio_codes.transpose(1, 2)
        code_mask = None
        if padding_mask is not None:
            lengths = padding_mask.sum(dim=-1, keepdim=True)
            token_lengths = lengths // self.config.encoder_hop_length
            positions = torch.arange(
                audio_codes.shape[-1],
                device=padding_mask.device,
            ).view(1, -1)
            code_mask = (positions < token_lengths).to(padding_mask.dtype)
        return NeuCodecEncoderOutput(
            audio_codes=audio_codes,
            latents=latents if output_latents else None,
            audio_codes_mask=code_mask,
        )

    @staticmethod
    def _align_encoder_frames(
        semantic: Tensor,
        acoustic: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Match the released NeuCodec's safe shortest-stream alignment."""
        frame_count = min(semantic.shape[-1], acoustic.shape[-1])
        if frame_count <= 0:
            raise RuntimeError("NeuCodec encoders produced no alignable audio frames.")
        if semantic.shape[-1] != frame_count:
            semantic = semantic[..., :frame_count]
        if acoustic.shape[-1] != frame_count:
            acoustic = acoustic[..., :frame_count]
        return semantic, acoustic

    def encode_audio(
        self,
        waveforms: Tensor,
        waveform_lengths: Tensor | None = None,
    ) -> NeuCodecEncoderOutput:
        features = self.feature_extractor(waveforms, waveform_lengths)
        return self.encode(
            features.input_values,
            features.input_features,
            padding_mask=features.padding_mask,
            input_features_mask=features.input_features_mask,
        )

    def encode_code(
        self,
        input_waveform: Tensor,
        sample_rate: int = 16_000,
    ) -> Tensor:
        if sample_rate != self.input_sampling_rate:
            raise ValueError(
                f"NeuCodec requires {self.input_sampling_rate} Hz input; "
                f"received {sample_rate} Hz.")
        return self.encode_audio(input_waveform).audio_codes

    def decode(
        self,
        *,
        audio_codes: Tensor | None = None,
        latents: Tensor | None = None,
    ) -> NeuCodecDecoderOutput:
        if (audio_codes is None) == (latents is None):
            raise ValueError("Specify exactly one of `audio_codes` or `latents`.")
        if audio_codes is not None:
            if audio_codes.ndim != 3 or audio_codes.shape[1] != 1:
                raise ValueError("NeuCodec `audio_codes` must have shape [batch, 1, frames].")
            decoded_latents = self.quantizer.from_codes(audio_codes.transpose(1, 2))
        else:
            if latents is None or latents.ndim != 3:
                raise ValueError("NeuCodec `latents` must have shape [batch, channels, frames].")
            decoded_latents = latents.transpose(1, 2)
        return NeuCodecDecoderOutput(audio_values=self.acoustic_decoder(decoded_latents))

    def decode_code(self, audio_codes: Tensor) -> Tensor:
        return self.decode(audio_codes=audio_codes).audio_values

    def forward(
        self,
        input_values: Tensor,
        input_features: Tensor,
        *,
        padding_mask: Tensor | None = None,
        input_features_mask: Tensor | None = None,
        output_latents: bool = False,
    ) -> NeuCodecOutput:
        output_length = round(input_values.shape[-1] * self.output_sampling_rate / self.input_sampling_rate)
        encoded = self.encode(
            input_values,
            input_features,
            padding_mask=padding_mask,
            input_features_mask=input_features_mask,
            output_latents=True,
        )
        decoded = self.decode(latents=encoded.latents)
        return NeuCodecOutput(
            audio_values=decoded.audio_values[..., :output_length],
            audio_codes=encoded.audio_codes,
            latents=encoded.latents if output_latents else None,
            audio_codes_mask=encoded.audio_codes_mask,
        )

    def save_pretrained(self, directory: str | Path) -> Path:
        from voicehub.checkpointing import save_safetensors

        target = Path(directory).expanduser()
        target.mkdir(parents=True, exist_ok=True)
        (target / "config.json").write_text(
            json.dumps(
                self.config.to_dict(),
                indent=2,
                sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )
        (target / "preprocessor_config.json").write_text(
            json.dumps(
                {
                    "feature_extractor_type": "NeuCodecFeatureExtractor",
                    "feature_size": 80,
                    "frame_length": 400,
                    "frame_shift": 160,
                    "hop_length": self.config.encoder_hop_length,
                    "num_mel_bins": 80,
                    "padding_side": "right",
                    "padding_value": 1,
                    "return_attention_mask": True,
                    "sampling_rate": self.input_sampling_rate,
                    "stride": 2,
                },
                indent=2,
                sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )
        save_safetensors(
            self.state_dict(),
            target / "model.safetensors",
            metadata={
                "format": "pt",
                "architecture": "neucodec",
                "producer": "voicehub",
            },
        )
        return target.resolve()


__all__ = [
    "NeuCodecDecoderOutput",
    "NeuCodecEncoderOutput",
    "NeuCodecFeatureExtractor",
    "NeuCodecFeatures",
    "NeuCodecModel",
    "NeuCodecOutput",
]
