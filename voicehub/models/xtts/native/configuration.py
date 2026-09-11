"""Immutable, dependency-free XTTS v2 checkpoint configuration."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping

from voicehub.tokenization.assets import read_bounded_asset

XTTS2_LANGUAGES = (
    "en",
    "es",
    "fr",
    "de",
    "it",
    "pt",
    "pl",
    "tr",
    "ru",
    "nl",
    "cs",
    "ar",
    "zh-cn",
    "hu",
    "ko",
    "ja",
    "hi",
)


@dataclass(frozen=True, slots=True)
class XTTS2AudioConfig:
    sample_rate: int = 22_050
    output_sample_rate: int = 24_000


@dataclass(frozen=True, slots=True)
class XTTS2ModelArgs:
    gpt_batch_size: int = 1
    kv_cache: bool = True
    gpt_max_audio_tokens: int = 605
    gpt_max_text_tokens: int = 402
    gpt_max_prompt_tokens: int = 70
    gpt_layers: int = 30
    gpt_n_model_channels: int = 1_024
    gpt_n_heads: int = 16
    gpt_number_text_tokens: int = 6_681
    gpt_start_text_token: int | None = None
    gpt_stop_text_token: int | None = None
    gpt_num_audio_tokens: int = 1_026
    gpt_start_audio_token: int = 1_024
    gpt_stop_audio_token: int = 1_025
    gpt_code_stride_len: int = 1_024
    gpt_use_masking_gt_prompt_approach: bool = True
    gpt_use_perceiver_resampler: bool = True
    input_sample_rate: int = 22_050
    output_sample_rate: int = 24_000
    output_hop_length: int = 256
    decoder_input_dim: int = 1_024
    d_vector_dim: int = 512
    cond_d_vector_in_each_upsampling_layer: bool = True


@dataclass(frozen=True, slots=True)
class XTTS2Config:
    """The architecture-bearing subset of the published ``config.json``.

    Unknown trainer fields remain outside the native runtime.
    Architecture fields are immutable after parsing so a loaded graph
    cannot silently diverge from the checkpoint header that was
    validated for it.
    """

    model_args: XTTS2ModelArgs = XTTS2ModelArgs()
    audio: XTTS2AudioConfig = XTTS2AudioConfig()
    languages: tuple[str, ...] = XTTS2_LANGUAGES
    temperature: float = 0.75
    length_penalty: float = 1.0
    repetition_penalty: float = 5.0
    top_k: int = 50
    top_p: float = 0.85
    gpt_cond_len: int = 30
    gpt_cond_chunk_len: int = 4
    max_ref_len: int = 30
    sound_norm_refs: bool = False

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> XTTS2Config:
        if not isinstance(value, Mapping):
            raise TypeError("XTTS v2 configuration must be a mapping.")
        model_args = _dataclass_from_mapping(
            XTTS2ModelArgs,
            value.get("model_args", {}),
        )
        audio = _dataclass_from_mapping(
            XTTS2AudioConfig,
            value.get("audio", {}),
        )
        languages = value.get("languages", XTTS2_LANGUAGES)
        if (not isinstance(languages, (list, tuple)) or not languages or
                any(not isinstance(item, str) or not item for item in languages)):
            raise ValueError("XTTS v2 `languages` must contain language codes.")
        kwargs = {
            name: value.get(name, field.default)
            for field in fields(cls) if (name := field.name) not in {"model_args", "audio", "languages"}
        }
        result = cls(
            model_args=model_args,
            audio=audio,
            languages=tuple(languages),
            **kwargs,
        )
        result.validate()
        return result

    @classmethod
    def from_json(cls, path: str | Path) -> XTTS2Config:
        import json

        source = Path(path).expanduser().resolve()
        try:
            value = json.loads(read_bounded_asset(
                source,
                max_bytes=1024 * 1024,
            ).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
            raise ValueError(f"Invalid XTTS v2 checkpoint configuration: {error}.") from error
        return cls.from_mapping(value)

    def validate(self) -> None:
        args = self.model_args
        positive = {
            "audio.sample_rate": self.audio.sample_rate,
            "audio.output_sample_rate": self.audio.output_sample_rate,
            "gpt_batch_size": args.gpt_batch_size,
            "gpt_max_audio_tokens": args.gpt_max_audio_tokens,
            "gpt_max_text_tokens": args.gpt_max_text_tokens,
            "gpt_max_prompt_tokens": args.gpt_max_prompt_tokens,
            "gpt_layers": args.gpt_layers,
            "gpt_n_model_channels": args.gpt_n_model_channels,
            "gpt_n_heads": args.gpt_n_heads,
            "gpt_number_text_tokens": args.gpt_number_text_tokens,
            "gpt_num_audio_tokens": args.gpt_num_audio_tokens,
            "gpt_code_stride_len": args.gpt_code_stride_len,
            "input_sample_rate": args.input_sample_rate,
            "output_sample_rate": args.output_sample_rate,
            "output_hop_length": args.output_hop_length,
            "decoder_input_dim": args.decoder_input_dim,
            "d_vector_dim": args.d_vector_dim,
            "gpt_cond_len": self.gpt_cond_len,
            "gpt_cond_chunk_len": self.gpt_cond_chunk_len,
            "max_ref_len": self.max_ref_len,
        }
        for name, item in positive.items():
            if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
                raise ValueError(f"XTTS v2 `{name}` must be a positive integer.")
        for name in (
                "kv_cache",
                "gpt_use_masking_gt_prompt_approach",
                "gpt_use_perceiver_resampler",
                "cond_d_vector_in_each_upsampling_layer",
        ):
            if not isinstance(getattr(args, name), bool):
                raise TypeError(f"XTTS v2 `{name}` must be a boolean.")
        if not isinstance(self.sound_norm_refs, bool):
            raise TypeError("XTTS v2 `sound_norm_refs` must be a boolean.")
        if args.gpt_n_model_channels % args.gpt_n_heads:
            raise ValueError("XTTS v2 model channels must divide evenly into heads.")
        if self.gpt_cond_chunk_len > self.gpt_cond_len:
            raise ValueError("XTTS v2 `gpt_cond_chunk_len` cannot exceed `gpt_cond_len`.")
        if self.audio.sample_rate != args.input_sample_rate:
            raise ValueError("XTTS v2 audio and model input sample rates must agree.")
        if self.audio.output_sample_rate != args.output_sample_rate:
            raise ValueError("XTTS v2 audio and model output sample rates must agree.")
        for name in ("gpt_start_audio_token", "gpt_stop_audio_token"):
            token = getattr(args, name)
            if (isinstance(token, bool) or not isinstance(token, int) or
                    not 0 <= token < args.gpt_num_audio_tokens):
                raise ValueError(f"XTTS v2 `{name}` is outside its acoustic vocabulary.")
        if args.gpt_start_audio_token == args.gpt_stop_audio_token:
            raise ValueError("XTTS v2 start- and stop-audio tokens must be distinct.")
        for name in ("gpt_start_text_token", "gpt_stop_text_token"):
            token = getattr(args, name)
            if token is not None and (isinstance(token, bool) or not isinstance(token, int) or
                                      not 0 <= token < args.gpt_number_text_tokens):
                raise ValueError(f"XTTS v2 `{name}` is outside its text vocabulary.")
        if (isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or self.top_k < 0):
            raise ValueError("XTTS v2 `top_k` must be a non-negative integer.")
        for name in ("temperature", "length_penalty", "repetition_penalty"):
            item = getattr(self, name)
            if (isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item) or
                    item <= 0):
                raise ValueError(f"XTTS v2 `{name}` must be finite and greater than zero.")
        if (isinstance(self.top_p, bool) or not isinstance(self.top_p, (int, float)) or
                not math.isfinite(self.top_p) or not 0 < self.top_p <= 1):
            raise ValueError("XTTS v2 `top_p` must be finite and in the interval (0, 1].")
        if len(set(self.languages)) != len(self.languages):
            raise ValueError("XTTS v2 `languages` must not contain duplicates.")
        if any(item != item.strip().lower() or "[" in item or "]" in item for item in self.languages):
            raise ValueError("XTTS v2 language codes must be normalized lowercase values.")


def _dataclass_from_mapping(kind, value):
    if not isinstance(value, Mapping):
        raise TypeError(f"XTTS v2 `{kind.__name__}` input must be a mapping.")
    allowed = {field.name for field in fields(kind)}
    return kind(**{name: item for name, item in value.items() if name in allowed})


__all__ = [
    "XTTS2AudioConfig",
    "XTTS2Config",
    "XTTS2ModelArgs",
    "XTTS2_LANGUAGES",
]
