"""Validated native configurations for the official MOSS-TTS releases.

Only architecture fields that affect executable mathematics are
normalized here.  Unknown metadata is retained for round-tripping, while
unsupported published variants are rejected rather than being guessed
from a repository name.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from voicehub.models.causal_lm.native.configuration import Qwen3Config

MossTTSVariant = Literal["delay", "local", "local_v1_5", "realtime"]


def _integer(name: str, value: Any, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"`{name}` must be an integer.")
    if value < minimum:
        raise ValueError(f"`{name}` must be at least {minimum}; found {value}.")
    return value


def _positive_float(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"`{name}` must be a real number.")
    result = float(value)
    if result <= 0.0:
        raise ValueError(f"`{name}` must be greater than zero.")
    return result


def _qwen3(value: Mapping[str, Any] | Qwen3Config) -> Qwen3Config:
    if isinstance(value, Qwen3Config):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("MOSS-TTS language configuration must be a mapping.")
    return Qwen3Config.from_dict(value)


@dataclass(frozen=True, slots=True)
class MossGPT2Config:
    """The one-layer RoPE depth decoder used by Local v1.5."""

    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    max_position_embeddings: int
    layer_norm_epsilon: float = 1e-6
    initializer_range: float = 0.02
    activation_function: str = "silu"
    attention_dropout: float = 0.0
    residual_dropout: float = 0.0
    embedding_dropout: float = 0.0
    rope_base: float = 1_000_000.0

    def __post_init__(self) -> None:
        _integer("hidden_size", self.hidden_size, minimum=1)
        _integer("intermediate_size", self.intermediate_size, minimum=1)
        _integer("num_hidden_layers", self.num_hidden_layers, minimum=1)
        _integer("num_attention_heads", self.num_attention_heads, minimum=1)
        _integer("max_position_embeddings", self.max_position_embeddings, minimum=1)
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("Local hidden size must be divisible by its attention heads.")
        if self.activation_function != "silu":
            raise ValueError("The audited Local v1.5 graph requires SiLU.")
        _positive_float("layer_norm_epsilon", self.layer_norm_epsilon)
        _positive_float("initializer_range", self.initializer_range)
        _positive_float("rope_base", self.rope_base)
        for name in ("attention_dropout", "residual_dropout", "embedding_dropout"):
            value = float(getattr(self, name))
            if not 0.0 <= value < 1.0:
                raise ValueError(f"`{name}` must be in [0, 1).")

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> MossGPT2Config:
        if not isinstance(values, Mapping):
            raise TypeError("MOSS-TTS local GPT configuration must be a mapping.")
        return cls(
            hidden_size=int(values.get("n_embd", values.get("hidden_size", 0))),
            intermediate_size=int(values.get(
                "n_inner",
                values.get("intermediate_size", 0),
            )),
            num_hidden_layers=int(values.get(
                "n_layer",
                values.get("num_hidden_layers", 0),
            )),
            num_attention_heads=int(values.get(
                "n_head",
                values.get("num_attention_heads", 0),
            )),
            max_position_embeddings=int(
                values.get(
                    "n_positions",
                    values.get("max_position_embeddings", 0),
                )),
            layer_norm_epsilon=float(values.get(
                "layer_norm_epsilon",
                1e-6,
            )),
            initializer_range=float(values.get("initializer_range", 0.02)),
            activation_function=str(values.get("activation_function", "silu")),
            attention_dropout=float(values.get("attn_pdrop", 0.0)),
            residual_dropout=float(values.get("resid_pdrop", 0.0)),
            embedding_dropout=float(values.get("embd_pdrop", 0.0)),
            rope_base=float(values.get("rope_base", 1_000_000.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": "gpt2",
            "n_embd": self.hidden_size,
            "n_inner": self.intermediate_size,
            "n_layer": self.num_hidden_layers,
            "n_head": self.num_attention_heads,
            "n_positions": self.max_position_embeddings,
            "n_ctx": self.max_position_embeddings,
            "layer_norm_epsilon": self.layer_norm_epsilon,
            "initializer_range": self.initializer_range,
            "activation_function": self.activation_function,
            "attn_pdrop": self.attention_dropout,
            "resid_pdrop": self.residual_dropout,
            "embd_pdrop": self.embedding_dropout,
            "position_embedding_type": "rope",
            "rope_base": self.rope_base,
        }


@dataclass(frozen=True, slots=True)
class MossTTSConfig:
    """Executable configuration for one audited MOSS-TTS graph."""

    variant: MossTTSVariant
    language_config: Qwen3Config
    n_vq: int
    audio_vocab_size: int
    audio_codebook_sizes: tuple[int, ...]
    audio_pad_token_id: int
    pad_token_id: int
    im_start_token_id: int
    im_end_token_id: int
    audio_start_token_id: int
    audio_end_token_id: int
    sample_rate: int
    audio_user_slot_token_id: int | None = None
    audio_assistant_slot_token_id: int | None = None
    audio_assistant_delay_slot_token_id: int | None = None
    local_config: Qwen3Config | MossGPT2Config | None = None
    additional_mlp_ffn_hidden_size: int | None = None
    local_text_head_mode: str | None = None
    reference_audio_pad_token_id: int | None = None
    text_pad_token_id: int | None = None
    initializer_range: float = 0.02
    codec_repository: str = ""
    extra_config: Mapping[str, Any] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.variant not in {"delay", "local", "local_v1_5", "realtime"}:
            raise ValueError(f"Unsupported MOSS-TTS variant {self.variant!r}.")
        _integer("n_vq", self.n_vq, minimum=1)
        _integer("audio_vocab_size", self.audio_vocab_size, minimum=2)
        if len(self.audio_codebook_sizes) != self.n_vq:
            raise ValueError("`audio_codebook_sizes` must contain one entry per RVQ channel.")
        if any(size != self.audio_vocab_size for size in self.audio_codebook_sizes):
            raise ValueError("Non-uniform MOSS-TTS codebook sizes have not been reference-audited.")
        _integer("audio_pad_token_id", self.audio_pad_token_id, minimum=0)
        _integer("sample_rate", self.sample_rate, minimum=1)
        _positive_float("initializer_range", self.initializer_range)
        if self.variant in {"delay", "local"}:
            if self.n_vq != 32 or self.sample_rate != 24_000:
                raise ValueError("Published Delay/Local graphs require 32 RVQ channels at 24 kHz.")
            if self.audio_pad_token_id != self.audio_vocab_size:
                raise ValueError("Delay/Local audio padding must equal the audio vocabulary size.")
            if self.audio_assistant_delay_slot_token_id is None:
                raise ValueError("Delay/Local graphs require the delay-slot token.")
        elif self.variant == "local_v1_5":
            if self.n_vq != 12 or self.sample_rate != 48_000:
                raise ValueError("Published Local v1.5 requires 12 RVQ channels at 48 kHz.")
            if not isinstance(self.local_config, MossGPT2Config):
                raise TypeError("Local v1.5 requires its audited GPT2-RoPE depth decoder.")
            if self.local_config.num_hidden_layers != 1:
                raise ValueError("Published Local v1.5 has exactly one local depth layer.")
            if self.local_text_head_mode != "binary":
                raise ValueError("Only the published binary Local v1.5 text head is supported.")
        else:
            if self.n_vq != 16 or self.sample_rate != 24_000:
                raise ValueError("Published Realtime requires 16 RVQ channels at 24 kHz.")
            if self.audio_vocab_size != 1027 or self.audio_pad_token_id != 1024:
                raise ValueError("Published Realtime requires audio vocabulary 1027 and pad 1024.")
            if not isinstance(self.local_config, Qwen3Config):
                raise TypeError("Realtime requires its audited Qwen3 local decoder.")
        if not self.codec_repository:
            raise ValueError("A MOSS-TTS config must name its separately versioned codec.")

    @property
    def model_type(self) -> str:
        return {
            "delay": "moss_tts_delay",
            "local": "moss_tts_delay",
            "local_v1_5": "moss_tts_local",
            "realtime": "moss_tts_realtime",
        }[self.variant]

    @property
    def channels(self) -> int:
        return self.n_vq + 1

    @classmethod
    def from_dict(
        cls,
        values: Mapping[str, Any],
        *,
        variant: str | None = None,
    ) -> MossTTSConfig:
        if not isinstance(values, Mapping):
            raise TypeError("MOSS-TTS configuration must be a mapping.")
        source = copy.deepcopy(dict(values))
        model_type = str(source.get("model_type", "")).strip()
        if variant is None:
            native_variant = source.get("voicehub_variant")
            if isinstance(native_variant, str) and native_variant.strip():
                variant = native_variant
            elif model_type == "moss_tts_local":
                variant = "local_v1_5"
            elif model_type == "moss_tts_realtime":
                variant = "realtime"
            elif model_type == "moss_tts_delay":
                variant = (
                    "local" if any(
                        name in source for name in (
                            "local_hidden_size",
                            "local_num_layers",
                            "local_ffn_hidden_size",
                        )) else "delay")
            else:
                raise ValueError(f"Unsupported MOSS-TTS model_type {model_type!r}.")
        normalized = str(variant).strip().lower().replace("-", "_").replace(".", "_")
        normalized = {"local_v15": "local_v1_5"}.get(normalized, normalized)
        if normalized not in {"delay", "local", "local_v1_5", "realtime"}:
            raise ValueError(f"Unsupported MOSS-TTS variant {variant!r}.")

        language_values = source.get(
            "qwen3_config",
            source.get("language_config"),
        )
        if language_values is None:
            raise KeyError("MOSS-TTS config is missing its language graph.")
        language = _qwen3(language_values)
        n_vq = int(source.get("rvq", source.get("n_vq", 0)))
        audio_vocab_size = int(source.get("audio_vocab_size", 0))
        codebook_values = source.get("audio_codebook_sizes")
        if codebook_values is None:
            codebook_sizes = (audio_vocab_size, ) * n_vq
        elif isinstance(codebook_values, Sequence) and not isinstance(codebook_values, (str, bytes)):
            codebook_sizes = tuple(int(item) for item in codebook_values)
        else:
            raise TypeError("`audio_codebook_sizes` must be a sequence.")

        if normalized == "local_v1_5":
            local_values = source.get("gpt2_config")
            if not isinstance(local_values, Mapping):
                raise KeyError("Local v1.5 config is missing `gpt2_config`.")
            local: Qwen3Config | MossGPT2Config = MossGPT2Config.from_dict(local_values)
            codec_repository = str(
                source.get(
                    "audio_tokenizer_name_or_path",
                    "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2",
                ))
            sample_rate = int(source.get("sampling_rate", 48_000))
        elif normalized == "local":
            local_dict = dict(language.to_dict())
            local_dict.update({
                "hidden_size": source.get("local_hidden_size"),
                "intermediate_size": source.get("local_ffn_hidden_size"),
                "num_hidden_layers": source.get("local_num_layers"),
            })
            # The published Local checkpoint derives a four-layer Qwen depth
            # decoder from the 28-layer backbone configuration.  Per-layer
            # metadata inherited from the backbone is therefore not part of
            # the local graph and retaining it would describe an incoherent
            # configuration.
            local_dict.pop("layer_types", None)
            local_dict.pop("max_window_layers", None)
            local = Qwen3Config.from_dict(local_dict)
            codec_repository = "OpenMOSS-Team/MOSS-Audio-Tokenizer"
            sample_rate = int(source.get("sampling_rate", 24_000))
        elif normalized == "realtime":
            local_values = source.get("local_config")
            if not isinstance(local_values, Mapping):
                raise KeyError("Realtime config is missing `local_config`.")
            local_dict = dict(local_values)
            local_dict["model_type"] = "qwen3"
            local_dict["vocab_size"] = audio_vocab_size
            local_dict["pad_token_id"] = int(source.get("audio_pad_token", 1024))
            local = Qwen3Config.from_dict(local_dict)
            codec_repository = "OpenMOSS-Team/MOSS-Audio-Tokenizer"
            sample_rate = int(source.get("sampling_rate", 24_000))
        else:
            local = None
            codec_repository = "OpenMOSS-Team/MOSS-Audio-Tokenizer"
            sample_rate = int(source.get("sampling_rate", 24_000))

        known = {
            "additional_mlp_ffn_hidden_size",
            'architectures',
            "audio_assistant_delay_slot_token_id",
            "audio_assistant_gen_slot_token_id",
            "audio_assistant_slot_token_id",
            "audio_codebook_sizes",
            "audio_end_token_id",
            "audio_pad_code",
            "audio_pad_token",
            "audio_pad_token_id",
            "audio_start_token_id",
            "audio_tokenizer_name_or_path",
            "audio_user_slot_token_id",
            "audio_vocab_size",
            "auto_map",
            "dtype",
            "gpt2_config",
            "im_end_token_id",
            "im_start_token_id",
            "initializer_range",
            "language_config",
            "local_ffn_hidden_size",
            "local_hidden_size",
            "local_num_layers",
            "local_text_head_mode",
            "local_transformer_layers",
            "model_type",
            "n_vq",
            "pad_token_id",
            "processor_class",
            "qwen3_config",
            "reference_audio_pad",
            "rvq",
            "sampling_rate",
            "text_pad",
            "tie_word_embeddings",
            "transformers_version",
            "vocab_size",
            "voicehub_variant",
        }
        return cls(
            variant=normalized,  # type: ignore[arg-type]
            language_config=language,
            n_vq=n_vq,
            audio_vocab_size=audio_vocab_size,
            audio_codebook_sizes=codebook_sizes,
            audio_pad_token_id=int(
                source.get(
                    "audio_pad_token",
                    source.get(
                        "audio_pad_code",
                        source.get("audio_pad_token_id", audio_vocab_size),
                    ),
                )),
            pad_token_id=int(
                source.get("pad_token_id") if source.get("pad_token_id") is not None else (
                    language.pad_token_id if language.pad_token_id is not None else 151_643)),
            im_start_token_id=int(source.get("im_start_token_id", 151_644)),
            im_end_token_id=int(source.get("im_end_token_id", 151_645)),
            audio_start_token_id=int(source.get("audio_start_token_id", 151_652)),
            audio_end_token_id=int(source.get("audio_end_token_id", 151_653)),
            sample_rate=sample_rate,
            audio_user_slot_token_id=(
                None if source.get("audio_user_slot_token_id") is None else int(
                    source["audio_user_slot_token_id"])),
            audio_assistant_slot_token_id=(
                None if source.get(
                    "audio_assistant_slot_token_id",
                    source.get("audio_assistant_gen_slot_token_id"),
                ) is None else int(
                    source.get(
                        "audio_assistant_slot_token_id",
                        source.get("audio_assistant_gen_slot_token_id"),
                    ))),
            audio_assistant_delay_slot_token_id=(
                None if source.get("audio_assistant_delay_slot_token_id") is None else int(
                    source["audio_assistant_delay_slot_token_id"])),
            local_config=local,
            additional_mlp_ffn_hidden_size=(
                None if source.get("additional_mlp_ffn_hidden_size") is None else int(
                    source["additional_mlp_ffn_hidden_size"])),
            local_text_head_mode=(
                None if source.get("local_text_head_mode") is None else str(source["local_text_head_mode"])),
            reference_audio_pad_token_id=(
                None if source.get("reference_audio_pad") is None else int(source["reference_audio_pad"])),
            text_pad_token_id=(None if source.get("text_pad") is None else int(source["text_pad"])),
            initializer_range=float(source.get("initializer_range", 0.02)),
            codec_repository=codec_repository,
            extra_config={
                name: value
                for name, value in source.items() if name not in known
            },
        )

    def to_dict(self) -> dict[str, Any]:
        result = copy.deepcopy(dict(self.extra_config))
        result.update({
            "model_type":
            self.model_type,
            'architectures': [{
                "delay": "MossTTSDelayModel",
                "local": "MossTTSDelayModel",
                "local_v1_5": "MossTTSLocalModel",
                "realtime": "MossTTSRealtime",
            }[self.variant]],
            "language_config":
            self.language_config.to_dict(),
            "n_vq":
            self.n_vq,
            "audio_vocab_size":
            self.audio_vocab_size,
            "audio_codebook_sizes":
            list(self.audio_codebook_sizes),
            "audio_pad_token_id":
            self.audio_pad_token_id,
            "pad_token_id":
            self.pad_token_id,
            "im_start_token_id":
            self.im_start_token_id,
            "im_end_token_id":
            self.im_end_token_id,
            "audio_start_token_id":
            self.audio_start_token_id,
            "audio_end_token_id":
            self.audio_end_token_id,
            "sampling_rate":
            self.sample_rate,
            "initializer_range":
            self.initializer_range,
            "voicehub_variant":
            self.variant,
            "audio_tokenizer_name_or_path":
            self.codec_repository,
        })
        if self.audio_user_slot_token_id is not None:
            result["audio_user_slot_token_id"] = self.audio_user_slot_token_id
        if self.audio_assistant_slot_token_id is not None:
            result["audio_assistant_slot_token_id"] = self.audio_assistant_slot_token_id
            result["audio_assistant_gen_slot_token_id"] = self.audio_assistant_slot_token_id
        if self.audio_assistant_delay_slot_token_id is not None:
            result["audio_assistant_delay_slot_token_id"] = self.audio_assistant_delay_slot_token_id
        if self.local_config is not None:
            if isinstance(self.local_config, MossGPT2Config):
                result["qwen3_config"] = self.language_config.to_dict()
                result["gpt2_config"] = self.local_config.to_dict()
                result["local_text_head_mode"] = self.local_text_head_mode
            else:
                result["local_config"] = self.local_config.to_dict()
                if self.variant == "local":
                    result.update({
                        "local_hidden_size": self.local_config.hidden_size,
                        "local_ffn_hidden_size": self.local_config.intermediate_size,
                        "local_num_layers": self.local_config.num_hidden_layers,
                    })
        if self.additional_mlp_ffn_hidden_size is not None:
            result["additional_mlp_ffn_hidden_size"] = self.additional_mlp_ffn_hidden_size
        if self.reference_audio_pad_token_id is not None:
            result["reference_audio_pad"] = self.reference_audio_pad_token_id
        if self.text_pad_token_id is not None:
            result["text_pad"] = self.text_pad_token_id
        return result


__all__ = [
    "MossGPT2Config",
    "MossTTSConfig",
    "MossTTSVariant",
]
