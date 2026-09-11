"""Validated native configuration for the published Qwen3-TTS 12 Hz family.

The schema mirrors the official ``config.json`` files without importing
Transformers. Unknown keys are retained when round-tripping so
checkpoints created by newer compatible releases remain loadable.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


def _positive(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"`{name}` must be a positive integer.")
    return value


def _nonnegative(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"`{name}` must be a non-negative integer.")
    return value


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"`{name}` must be a mapping.")
    return {str(key): item for key, item in value.items()}


@dataclass(slots=True)
class Qwen3TTSSpeakerEncoderConfig:
    mel_dim: int = 128
    enc_dim: int = 1024
    enc_channels: tuple[int, ...] = (512, 512, 512, 512, 1536)
    enc_kernel_sizes: tuple[int, ...] = (5, 3, 3, 3, 1)
    enc_dilations: tuple[int, ...] = (1, 2, 3, 4, 1)
    enc_attention_channels: int = 128
    enc_res2net_scale: int = 8
    enc_se_channels: int = 128
    sample_rate: int = 24_000
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any] | None) -> Qwen3TTSSpeakerEncoderConfig:
        source = _mapping(values, name="speaker_encoder_config")
        known = {
            name: source.pop(name)
            for name in tuple(source) if name in cls.__dataclass_fields__ and name != "extra"
        }
        for name in ("enc_channels", "enc_kernel_sizes", "enc_dilations"):
            if name in known:
                known[name] = tuple(int(item) for item in known[name])
        result = cls(**known, extra=source)
        result.validate()
        return result

    def validate(self) -> None:
        for name in (
                "mel_dim",
                "enc_dim",
                "enc_attention_channels",
                "enc_res2net_scale",
                "enc_se_channels",
                "sample_rate",
        ):
            _positive(getattr(self, name), name=name)
        if not (len(self.enc_channels) == len(self.enc_kernel_sizes) == len(self.enc_dilations) >= 3):
            raise ValueError(
                "Speaker encoder channels, kernels, and dilations must have "
                "the same length of at least three.")
        if any(item <= 0 for values in (
                self.enc_channels,
                self.enc_kernel_sizes,
                self.enc_dilations,
        ) for item in values):
            raise ValueError("Speaker encoder dimensions must be positive.")
        if any(channel % self.enc_res2net_scale for channel in self.enc_channels[1:-1]):
            raise ValueError("Each Res2Net channel count must be divisible by its scale.")

    def to_dict(self) -> dict[str, Any]:
        output = asdict(self)
        extra = output.pop("extra")
        for name in ("enc_channels", "enc_kernel_sizes", "enc_dilations"):
            output[name] = list(output[name])
        output.update(extra)
        return output


@dataclass(slots=True)
class Qwen3TTSCodePredictorConfig:
    vocab_size: int = 2048
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 5
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 65_536
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    attention_bias: bool = False
    attention_dropout: float = 0.0
    num_code_groups: int = 16
    use_cache: bool = True
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any] | None) -> Qwen3TTSCodePredictorConfig:
        source = _mapping(values, name="code_predictor_config")
        known = {
            name: source.pop(name)
            for name in tuple(source) if name in cls.__dataclass_fields__ and name != "extra"
        }
        result = cls(**known, extra=source)
        result.validate()
        return result

    def validate(self) -> None:
        for name in (
                "vocab_size",
                "hidden_size",
                "intermediate_size",
                "num_hidden_layers",
                "num_attention_heads",
                "num_key_value_heads",
                "head_dim",
                "max_position_embeddings",
                "num_code_groups",
        ):
            _positive(getattr(self, name), name=name)
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("Code predictor attention heads must be divisible by key/value heads.")
        if self.num_code_groups < 2:
            raise ValueError("Code predictor requires at least two codebooks.")
        if self.hidden_act != "silu":
            raise ValueError("Published Qwen3-TTS checkpoints require SiLU.")
        if self.initializer_range <= 0 or self.rms_norm_eps <= 0 or self.rope_theta <= 0:
            raise ValueError("Code predictor floating-point constants must be positive.")
        if not 0 <= self.attention_dropout < 1:
            raise ValueError("Code predictor attention dropout must be in [0, 1).")

    def to_dict(self) -> dict[str, Any]:
        output = asdict(self)
        extra = output.pop("extra")
        output.update(extra)
        return output


@dataclass(slots=True)
class Qwen3TTSTalkerConfig:
    code_predictor_config: Qwen3TTSCodePredictorConfig = field(default_factory=Qwen3TTSCodePredictorConfig)
    vocab_size: int = 3072
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 32_768
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    rope_scaling: dict[str, Any] | None = None
    attention_bias: bool = False
    attention_dropout: float = 0.0
    num_code_groups: int = 16
    text_hidden_size: int = 2048
    text_vocab_size: int = 151_936
    codec_eos_token_id: int = 2150
    codec_think_id: int = 2154
    codec_nothink_id: int = 2155
    codec_think_bos_id: int = 2156
    codec_think_eos_id: int = 2157
    codec_pad_id: int = 2148
    codec_bos_id: int = 2149
    position_id_per_seconds: int = 13
    spk_id: dict[str, int] = field(default_factory=dict)
    spk_is_dialect: dict[str, str | bool] = field(default_factory=dict)
    codec_language_id: dict[str, int] = field(default_factory=dict)
    use_cache: bool = True
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any] | None) -> Qwen3TTSTalkerConfig:
        source = _mapping(values, name="talker_config")
        predictor = Qwen3TTSCodePredictorConfig.from_dict(source.pop("code_predictor_config", None))
        known = {
            name: source.pop(name)
            for name in tuple(source)
            if name in cls.__dataclass_fields__ and name not in {"extra", "code_predictor_config"}
        }
        for name in ("spk_id", "spk_is_dialect", "codec_language_id"):
            if name in known:
                known[name] = _mapping(known[name], name=name)
        scaling = known.get("rope_scaling")
        if scaling is not None:
            known["rope_scaling"] = _mapping(scaling, name="rope_scaling")
        result = cls(
            code_predictor_config=predictor,
            **known,
            extra=source,
        )
        result.validate()
        return result

    def validate(self) -> None:
        for name in (
                "vocab_size",
                "hidden_size",
                "intermediate_size",
                "num_hidden_layers",
                "num_attention_heads",
                "num_key_value_heads",
                "head_dim",
                "max_position_embeddings",
                "num_code_groups",
                "text_hidden_size",
                "text_vocab_size",
                "position_id_per_seconds",
        ):
            _positive(getattr(self, name), name=name)
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("Talker attention heads must be divisible by key/value heads.")
        if self.num_code_groups != self.code_predictor_config.num_code_groups:
            raise ValueError("Talker and code predictor must declare the same codebook count.")
        if self.hidden_act != "silu":
            raise ValueError("Published Qwen3-TTS checkpoints require SiLU.")
        token_ids = (
            self.codec_eos_token_id,
            self.codec_think_id,
            self.codec_nothink_id,
            self.codec_think_bos_id,
            self.codec_think_eos_id,
            self.codec_pad_id,
            self.codec_bos_id,
        )
        for value in token_ids:
            _nonnegative(value, name="codec token ID")
            if value >= self.vocab_size:
                raise ValueError("Codec control token IDs must fit the codec vocabulary.")
        if set(self.spk_id) != set(self.spk_is_dialect):
            raise ValueError("Speaker IDs and dialect metadata must name identical speakers.")
        for name, value in self.spk_id.items():
            if not name or isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("Speaker metadata contains an invalid entry.")
            if not 0 <= value < self.vocab_size:
                raise ValueError(f"Speaker {name!r} has an out-of-range codec ID.")
        for name, value in self.codec_language_id.items():
            if (not name or isinstance(value, bool) or not isinstance(value, int) or
                    not 0 <= value < self.vocab_size):
                raise ValueError("Language metadata contains an invalid entry.")

    @property
    def mrope_section(self) -> tuple[int, int, int]:
        values = (self.rope_scaling or {}).get("mrope_section", (24, 20, 20))
        section = tuple(int(item) for item in values)
        if len(section) != 3 or sum(section) * 2 != self.head_dim:
            raise ValueError("Qwen3-TTS mRoPE sections must contain three half-head partitions.")
        return section  # type: ignore[return-value]

    @property
    def mrope_interleaved(self) -> bool:
        return bool((self.rope_scaling or {}).get("interleaved", False))

    def to_dict(self) -> dict[str, Any]:
        output = asdict(self)
        extra = output.pop("extra")
        output["code_predictor_config"] = self.code_predictor_config.to_dict()
        output.update(extra)
        return output


@dataclass(slots=True)
class Qwen3TTSDecoderConfig:
    attention_bias: bool = False
    attention_dropout: float = 0.0
    latent_dim: int = 1024
    codebook_dim: int = 512
    codebook_size: int = 2048
    decoder_dim: int = 1536
    hidden_act: str = "silu"
    hidden_size: int = 512
    intermediate_size: int = 1024
    layer_scale_initial_scale: float = 0.01
    max_position_embeddings: int = 8000
    head_dim: int = 64
    num_attention_heads: int = 16
    num_hidden_layers: int = 8
    num_key_value_heads: int = 16
    num_quantizers: int = 16
    num_semantic_quantizers: int = 1
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10_000.0
    sliding_window: int = 72
    upsample_rates: tuple[int, ...] = (8, 5, 4, 3)
    upsampling_ratios: tuple[int, ...] = (2, 2)
    vector_quantization_hidden_dimension: int = 512
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any] | None) -> Qwen3TTSDecoderConfig:
        source = _mapping(values, name="decoder_config")
        known = {
            name: source.pop(name)
            for name in tuple(source) if name in cls.__dataclass_fields__ and name != "extra"
        }
        for name in ("upsample_rates", "upsampling_ratios"):
            if name in known:
                known[name] = tuple(int(item) for item in known[name])
        result = cls(**known, extra=source)
        result.validate()
        return result

    def validate(self) -> None:
        for name in (
                "latent_dim",
                "codebook_dim",
                "codebook_size",
                "decoder_dim",
                "hidden_size",
                "intermediate_size",
                "max_position_embeddings",
                "head_dim",
                "num_attention_heads",
                "num_hidden_layers",
                "num_key_value_heads",
                "num_quantizers",
                "num_semantic_quantizers",
                "sliding_window",
                "vector_quantization_hidden_dimension",
        ):
            _positive(getattr(self, name), name=name)
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("Decoder attention heads must divide evenly.")
        if self.num_quantizers <= self.num_semantic_quantizers:
            raise ValueError("Decoder requires acoustic residual quantizers.")
        if self.codebook_dim % 2:
            raise ValueError("Qwen3-TTS decoder codebook dimension must be even.")
        if self.decoder_dim % (2**len(self.upsample_rates)):
            raise ValueError("Decoder channels must divide evenly across every upsampling block.")
        if any(value <= 0 for value in self.upsample_rates + self.upsampling_ratios):
            raise ValueError("Decoder upsampling factors must be positive.")
        if self.hidden_act != "silu":
            raise ValueError("Published Qwen3-TTS decoder checkpoints require SiLU.")

    @property
    def total_upsample(self) -> int:
        result = 1
        for value in self.upsample_rates + self.upsampling_ratios:
            result *= value
        return result

    def to_dict(self) -> dict[str, Any]:
        output = asdict(self)
        extra = output.pop("extra")
        output["upsample_rates"] = list(output["upsample_rates"])
        output["upsampling_ratios"] = list(output["upsampling_ratios"])
        output.update(extra)
        return output


@dataclass(slots=True)
class Qwen3TTSEncoderConfig:
    """Checkpoint-exact configuration of the published 12 Hz encoder.

    Qwen subclasses the Apache-2.0 Transformers 4.57.3 Mimi encoder and
    removes its decoder.  These fields are the complete construction
    contract used by the pinned speech-tokenizer checkpoint.
    """

    sampling_rate: int = 24_000
    frame_rate: float = 12.5
    audio_channels: int = 1
    hidden_size: int = 512
    num_filters: int = 64
    num_residual_layers: int = 1
    upsampling_ratios: tuple[int, ...] = (8, 6, 5, 4)
    kernel_size: int = 7
    last_kernel_size: int = 3
    residual_kernel_size: int = 3
    dilation_growth_rate: int = 2
    use_causal_conv: bool = True
    pad_mode: str = "constant"
    compress: int = 2
    trim_right_ratio: float = 1.0
    codebook_size: int = 2048
    codebook_dim: int = 256
    num_quantizers: int = 32
    use_conv_shortcut: bool = False
    vector_quantization_hidden_dimension: int = 256
    num_semantic_quantizers: int = 1
    upsample_groups: int = 512
    num_hidden_layers: int = 8
    intermediate_size: int = 2048
    num_attention_heads: int = 8
    num_key_value_heads: int = 8
    head_dim: int = 64
    hidden_act: str = "gelu"
    max_position_embeddings: int = 8000
    initializer_range: float = 0.02
    norm_eps: float = 1e-5
    use_cache: bool = False
    use_streaming: bool = False
    rope_theta: float = 10_000.0
    sliding_window: int = 250
    attention_dropout: float = 0.0
    layer_scale_initial_scale: float = 0.01
    attention_bias: bool = False
    normalize: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls,
        values: Mapping[str, Any] | None,
    ) -> Qwen3TTSEncoderConfig:
        source = _mapping(values, name="encoder_config")
        frame_rate = source.pop("_frame_rate", source.pop("frame_rate", 12.5))
        known = {
            name: source.pop(name)
            for name in tuple(source) if name in cls.__dataclass_fields__ and name != "extra"
        }
        if "upsampling_ratios" in known:
            known["upsampling_ratios"] = tuple(int(item) for item in known["upsampling_ratios"])
        result = cls(
            frame_rate=float(frame_rate),
            **known,
            extra=source,
        )
        result.validate()
        return result

    @property
    def encodec_frame_rate(self) -> int:
        return math.ceil(self.sampling_rate / math.prod(self.upsampling_ratios))

    @property
    def downsample_factor(self) -> int:
        if math.isclose(
                float(self.encodec_frame_rate),
                float(self.frame_rate),
        ):
            return 1
        return 2

    @property
    def total_downsample(self) -> int:
        return math.prod(self.upsampling_ratios) * self.downsample_factor

    def validate(self) -> None:
        for name in (
                "sampling_rate",
                "audio_channels",
                "hidden_size",
                "num_filters",
                "num_residual_layers",
                "kernel_size",
                "last_kernel_size",
                "residual_kernel_size",
                "dilation_growth_rate",
                "compress",
                "codebook_size",
                "codebook_dim",
                "num_quantizers",
                "vector_quantization_hidden_dimension",
                "num_semantic_quantizers",
                "upsample_groups",
                "num_hidden_layers",
                "intermediate_size",
                "num_attention_heads",
                "num_key_value_heads",
                "head_dim",
                "max_position_embeddings",
                "sliding_window",
        ):
            _positive(getattr(self, name), name=name)
        if any(value <= 0 for value in self.upsampling_ratios):
            raise ValueError("Encoder sampling ratios must be positive.")
        if self.audio_channels != 1:
            raise ValueError("Published Qwen3-TTS encoder checkpoints are mono.")
        if not self.use_causal_conv or self.pad_mode != "constant":
            raise ValueError("Published Qwen3-TTS encoders require constant-padded "
                             "causal convolutions.")
        if self.use_streaming or self.use_cache:
            raise ValueError(
                "The published Qwen3-TTS encoder graph is non-streaming "
                "and does not use a KV cache.")
        if self.hidden_act != "gelu":
            raise ValueError("Published Qwen3-TTS encoder checkpoints require GELU.")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("Encoder hidden size must divide evenly across heads.")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("Encoder attention heads must divide across key/value heads.")
        if self.head_dim * self.num_attention_heads != self.hidden_size:
            raise ValueError("Encoder head dimensions do not reconstruct hidden size.")
        if self.vector_quantization_hidden_dimension != self.codebook_dim:
            raise ValueError("Qwen3-TTS encoder VQ hidden and codebook dimensions must match.")
        if self.num_semantic_quantizers >= self.num_quantizers:
            raise ValueError("Encoder requires acoustic residual quantizers.")
        if self.frame_rate <= 0:
            raise ValueError("Encoder frame rate must be positive.")
        ratio = self.encodec_frame_rate / self.frame_rate
        if not (math.isclose(ratio, 1.0) or math.isclose(ratio, 2.0)):
            raise ValueError(
                "Native Qwen3-TTS supports the published Mimi frame-rate "
                "conversion of one or two only.")
        if self.initializer_range <= 0 or self.norm_eps <= 0:
            raise ValueError("Encoder initialization and normalization constants must be positive.")
        if self.rope_theta <= 1 or self.layer_scale_initial_scale <= 0:
            raise ValueError("Encoder RoPE and layer-scale constants are invalid.")
        if not 0 <= self.attention_dropout < 1:
            raise ValueError("Encoder attention dropout must be in [0, 1).")
        if self.trim_right_ratio != 1.0:
            raise ValueError("Published encoder checkpoints require full right trimming.")
        if self.normalize:
            raise ValueError("Published Qwen3-TTS encoder checkpoints are not normalized.")

    def to_dict(self) -> dict[str, Any]:
        output = asdict(self)
        extra = output.pop("extra")
        output["_frame_rate"] = output.pop("frame_rate")
        output["upsampling_ratios"] = list(output["upsampling_ratios"])
        output.update(extra)
        return output


@dataclass(slots=True)
class Qwen3TTSTokenizerConfig:
    decoder_config: Qwen3TTSDecoderConfig = field(default_factory=Qwen3TTSDecoderConfig)
    encoder_config: Qwen3TTSEncoderConfig | None = None
    encoder_valid_num_quantizers: int = 16
    input_sample_rate: int = 24_000
    output_sample_rate: int = 24_000
    decode_upsample_rate: int = 1920
    encode_downsample_rate: int = 1920
    model_type: str = "qwen3_tts_tokenizer_12hz"
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> Qwen3TTSTokenizerConfig:
        source = _mapping(values, name="speech tokenizer config")
        decoder = Qwen3TTSDecoderConfig.from_dict(source.pop("decoder_config", None))
        encoder_values = source.pop("encoder_config", None)
        encoder = (None if encoder_values is None else Qwen3TTSEncoderConfig.from_dict(encoder_values))
        known = {
            name: source.pop(name)
            for name in tuple(source)
            if name in cls.__dataclass_fields__ and name not in {"extra", "decoder_config", "encoder_config"}
        }
        result = cls(
            decoder_config=decoder,
            encoder_config=encoder,
            **known,
            extra=source,
        )
        result.validate()
        return result

    def validate(self) -> None:
        for name in (
                "encoder_valid_num_quantizers",
                "input_sample_rate",
                "output_sample_rate",
                "decode_upsample_rate",
                "encode_downsample_rate",
        ):
            _positive(getattr(self, name), name=name)
        if self.model_type != "qwen3_tts_tokenizer_12hz":
            raise ValueError("Native Qwen3-TTS currently supports the published 12 Hz tokenizer only.")
        if self.encoder_valid_num_quantizers != self.decoder_config.num_quantizers:
            raise ValueError("Tokenizer and decoder codebook counts disagree.")
        if self.decode_upsample_rate != self.decoder_config.total_upsample:
            raise ValueError("Tokenizer decode rate does not match decoder upsampling.")
        if self.encoder_config is not None:
            self.encoder_config.validate()
            if self.input_sample_rate != self.encoder_config.sampling_rate:
                raise ValueError("Tokenizer input rate does not match its encoder.")
            if self.encoder_valid_num_quantizers > self.encoder_config.num_quantizers:
                raise ValueError("Tokenizer requests unavailable encoder codebooks.")
            if self.encode_downsample_rate != self.encoder_config.total_downsample:
                raise ValueError("Tokenizer encode rate does not match encoder downsampling.")

    def to_dict(self) -> dict[str, Any]:
        output = asdict(self)
        extra = output.pop("extra")
        output["decoder_config"] = self.decoder_config.to_dict()
        output["encoder_config"] = (None if self.encoder_config is None else self.encoder_config.to_dict())
        output.update(extra)
        return output


@dataclass(slots=True)
class Qwen3TTSArchitectureConfig:
    talker_config: Qwen3TTSTalkerConfig
    speaker_encoder_config: Qwen3TTSSpeakerEncoderConfig
    tokenizer_type: str
    tts_model_size: str
    tts_model_type: str
    im_start_token_id: int = 151_644
    im_end_token_id: int = 151_645
    tts_pad_token_id: int = 151_671
    tts_bos_token_id: int = 151_672
    tts_eos_token_id: int = 151_673
    assistant_token_id: int | None = None
    model_type: str = "qwen3_tts"
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> Qwen3TTSArchitectureConfig:
        source = _mapping(values, name="Qwen3-TTS config")
        talker = Qwen3TTSTalkerConfig.from_dict(source.pop("talker_config", None))
        speaker = Qwen3TTSSpeakerEncoderConfig.from_dict(source.pop("speaker_encoder_config", None))
        known = {
            name: source.pop(name)
            for name in tuple(source) if name in cls.__dataclass_fields__ and
            name not in {"extra", "talker_config", "speaker_encoder_config"}
        }
        result = cls(
            talker_config=talker,
            speaker_encoder_config=speaker,
            **known,
            extra=source,
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.model_type != "qwen3_tts":
            raise ValueError("Native Qwen3-TTS requires `model_type='qwen3_tts'`.")
        if self.tokenizer_type != "qwen3_tts_tokenizer_12hz":
            raise ValueError("Only Qwen3-TTS 12 Hz checkpoints are supported.")
        if self.tts_model_type not in {"base", "custom_voice", "voice_design"}:
            raise ValueError(f"Unsupported Qwen3-TTS role {self.tts_model_type!r}.")
        if self.tts_model_size not in {"0b6", "1b7"}:
            raise ValueError(f"Unsupported Qwen3-TTS size {self.tts_model_size!r}.")
        for value in (
                self.im_start_token_id,
                self.im_end_token_id,
                self.tts_pad_token_id,
                self.tts_bos_token_id,
                self.tts_eos_token_id,
        ):
            _nonnegative(value, name="text token ID")
            if value >= self.talker_config.text_vocab_size:
                raise ValueError("TTS text control token IDs exceed the text vocabulary.")
        if self.tts_model_type == "base" and self.speaker_encoder_config.enc_dim != self.talker_config.hidden_size:
            raise ValueError("Base checkpoint speaker embeddings must match talker hidden size.")

    def to_dict(self) -> dict[str, Any]:
        output = asdict(self)
        extra = output.pop("extra")
        output["talker_config"] = self.talker_config.to_dict()
        output["speaker_encoder_config"] = self.speaker_encoder_config.to_dict()
        output.update(extra)
        return output


__all__ = [
    "Qwen3TTSArchitectureConfig",
    "Qwen3TTSCodePredictorConfig",
    "Qwen3TTSDecoderConfig",
    "Qwen3TTSEncoderConfig",
    "Qwen3TTSSpeakerEncoderConfig",
    "Qwen3TTSTalkerConfig",
    "Qwen3TTSTokenizerConfig",
]
