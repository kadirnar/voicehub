"""Validated configuration for the audited ESPnet LibriSpeech graph."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from math import isfinite
from types import MappingProxyType
from typing import Any


def _integer(name: str, value: int, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"`{name}` must be an integer.")
    if value < minimum:
        raise ValueError(f"`{name}` must be at least {minimum}.")
    return value


def _real(
    name: str,
    value: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    maximum_exclusive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"`{name}` must be a real number.")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"`{name}` must be finite.")
    if minimum is not None and result < minimum:
        raise ValueError(f"`{name}` must be at least {minimum}.")
    if maximum is not None:
        invalid = result >= maximum if maximum_exclusive else result > maximum
        if invalid:
            qualifier = "less than" if maximum_exclusive else "at most"
            raise ValueError(f"`{name}` must be {qualifier} {maximum}.")
    return result


@dataclass(frozen=True, slots=True)
class ESPnetLibriSpeechTransformerConfig:
    """Complete graph contract for one published ESPnet 0.8.0 release.

    ``librispeech-transformer-e18`` is deliberately immutable: accepting
    an arbitrary ESPnet YAML under the same architecture ID would turn
    strict checkpoint validation into guesswork.  Small tests and
    research graphs can use ``variant="custom"``.
    """

    variant: str = "librispeech-transformer-e18"
    sampling_rate: int = 16_000
    n_fft: int = 512
    win_length: int = 512
    hop_length: int = 128
    n_mels: int = 80
    f_min: float = 0.0
    f_max: float = 8_000.0
    center: bool = True
    pad_mode: str = "reflect"
    normalized_stft: bool = False
    onesided_stft: bool = True
    vocabulary_size: int = 5_000
    blank_token_id: int = 0
    unknown_token_id: int = 1
    sos_eos_token_id: int = 4_999
    ignore_token_id: int = -1
    encoder_dimension: int = 512
    encoder_attention_heads: int = 8
    encoder_linear_units: int = 2_048
    encoder_blocks: int = 18
    decoder_attention_heads: int = 8
    decoder_linear_units: int = 2_048
    decoder_blocks: int = 6
    dropout_rate: float = 0.1
    positional_dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    normalize_before: bool = True
    ctc_weight: float = 0.3
    label_smoothing: float = 0.1
    length_normalized_loss: bool = False
    ctc_dropout_rate: float = 0.0
    global_mvn_epsilon: float = 1.0e-20
    apply_spec_augment: bool = True
    time_warp_window: int = 5
    frequency_mask_width: tuple[int, int] = (0, 30)
    frequency_masks: int = 2
    time_mask_width: tuple[int, int] = (0, 40)
    time_masks: int = 2
    language_model_layers: int = 4
    language_model_units: int = 2_048
    language_model_dropout: float = 0.0
    beam_size: int = 10
    language_model_weight: float = 0.6
    length_bonus: float = 0.0
    minimum_decode_ratio: float = 0.0
    maximum_decode_ratio: float = 1.0
    ctc_candidate_ratio: float | None = None
    extra_config: Mapping[str, Any] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.variant, str) or not self.variant.strip():
            raise ValueError("`variant` must be a non-empty string.")
        variant = self.variant.strip().lower().replace("_", "-")
        if variant not in {"librispeech-transformer-e18", "custom"}:
            raise ValueError(
                "Native ESPnet supports the audited "
                "`librispeech-transformer-e18` graph or `custom`.")
        object.__setattr__(self, "variant", variant)
        for name in (
                "sampling_rate",
                "n_fft",
                "win_length",
                "hop_length",
                "n_mels",
                "vocabulary_size",
                "encoder_dimension",
                "encoder_attention_heads",
                "encoder_linear_units",
                "encoder_blocks",
                "decoder_attention_heads",
                "decoder_linear_units",
                "decoder_blocks",
                "time_warp_window",
                "frequency_masks",
                "time_masks",
                "language_model_layers",
                "language_model_units",
                "beam_size",
        ):
            object.__setattr__(self, name, _integer(name, getattr(self, name)))
        if self.n_fft < self.win_length:
            raise ValueError("`n_fft` cannot be smaller than `win_length`.")
        if self.win_length > self.sampling_rate:
            raise ValueError("`win_length` cannot exceed one second.")
        if self.encoder_dimension % self.encoder_attention_heads:
            raise ValueError("Encoder dimension must be divisible by its head count.")
        if self.encoder_dimension % self.decoder_attention_heads:
            raise ValueError("Decoder dimension must be divisible by its head count.")
        if self.n_mels < 13:
            raise ValueError("Conv2d6 requires at least 13 input features.")
        for name in (
                "center",
                "normalized_stft",
                "onesided_stft",
                "normalize_before",
                "length_normalized_loss",
                "apply_spec_augment",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"`{name}` must be a boolean.")
        if self.pad_mode not in {"constant", "reflect"}:
            raise ValueError("`pad_mode` must be 'constant' or 'reflect'.")
        for name in (
                "dropout_rate",
                "positional_dropout_rate",
                "attention_dropout_rate",
                "label_smoothing",
                "ctc_dropout_rate",
                "language_model_dropout",
        ):
            object.__setattr__(
                self,
                name,
                _real(
                    name,
                    getattr(self, name),
                    minimum=0.0,
                    maximum=1.0,
                    maximum_exclusive=True,
                ),
            )
        for name in ("ctc_weight", "language_model_weight"):
            object.__setattr__(
                self,
                name,
                _real(
                    name,
                    getattr(self, name),
                    minimum=0.0,
                    maximum=1.0,
                ),
            )
        for name in (
                "f_min",
                "f_max",
                "global_mvn_epsilon",
                "length_bonus",
                "minimum_decode_ratio",
                "maximum_decode_ratio",
        ):
            minimum = None if name == "length_bonus" else 0.0
            object.__setattr__(
                self,
                name,
                _real(name, getattr(self, name), minimum=minimum),
            )
        if self.ctc_candidate_ratio is not None:
            object.__setattr__(
                self,
                "ctc_candidate_ratio",
                _real(
                    "ctc_candidate_ratio",
                    self.ctc_candidate_ratio,
                    minimum=1.0,
                ),
            )
        if not 0.0 <= self.f_min < self.f_max <= self.sampling_rate / 2:
            raise ValueError("Mel bounds must satisfy 0 <= f_min < f_max <= Nyquist.")
        if self.maximum_decode_ratio <= self.minimum_decode_ratio:
            raise ValueError("`maximum_decode_ratio` must exceed `minimum_decode_ratio`.")
        for name in (
                "blank_token_id",
                "unknown_token_id",
                "sos_eos_token_id",
        ):
            value = _integer(name, getattr(self, name), minimum=0)
            if value >= self.vocabulary_size:
                raise ValueError(f"`{name}` must be smaller than the vocabulary.")
            object.__setattr__(self, name, value)
        if len({
                self.blank_token_id,
                self.unknown_token_id,
                self.sos_eos_token_id,
        }) != 3:
            raise ValueError("Blank, unknown, and SOS/EOS IDs must be distinct.")
        if (isinstance(self.ignore_token_id, bool) or not isinstance(self.ignore_token_id, int)):
            raise TypeError("`ignore_token_id` must be an integer.")
        for name in ("frequency_mask_width", "time_mask_width"):
            value = getattr(self, name)
            if (not isinstance(value, (tuple, list)) or len(value) != 2 or
                    any(isinstance(item, bool) or not isinstance(item, int) for item in value)):
                raise TypeError(f"`{name}` must contain two integers.")
            resolved = tuple(value)
            if resolved[0] < 0 or resolved[1] <= resolved[0]:
                raise ValueError(f"`{name}` must be an increasing non-negative range.")
            object.__setattr__(self, name, resolved)
        if not isinstance(self.extra_config, Mapping):
            raise TypeError("`extra_config` must be a mapping.")
        object.__setattr__(
            self,
            "extra_config",
            MappingProxyType(copy.deepcopy(dict(self.extra_config))),
        )
        if self.variant == "librispeech-transformer-e18":
            expected = {
                "sampling_rate": 16_000,
                "n_fft": 512,
                "win_length": 512,
                "hop_length": 128,
                "n_mels": 80,
                "f_min": 0.0,
                "f_max": 8_000.0,
                "center": True,
                "pad_mode": "reflect",
                "normalized_stft": False,
                "onesided_stft": True,
                "vocabulary_size": 5_000,
                "blank_token_id": 0,
                "unknown_token_id": 1,
                "sos_eos_token_id": 4_999,
                "ignore_token_id": -1,
                "encoder_dimension": 512,
                "encoder_attention_heads": 8,
                "encoder_linear_units": 2_048,
                "encoder_blocks": 18,
                "decoder_attention_heads": 8,
                "decoder_linear_units": 2_048,
                "decoder_blocks": 6,
                "dropout_rate": 0.1,
                "positional_dropout_rate": 0.1,
                "attention_dropout_rate": 0.1,
                "normalize_before": True,
                "ctc_weight": 0.3,
                "label_smoothing": 0.1,
                "length_normalized_loss": False,
                "ctc_dropout_rate": 0.0,
                "global_mvn_epsilon": 1.0e-20,
                "apply_spec_augment": True,
                "time_warp_window": 5,
                "frequency_mask_width": (0, 30),
                "frequency_masks": 2,
                "time_mask_width": (0, 40),
                "time_masks": 2,
                "language_model_layers": 4,
                "language_model_units": 2_048,
                "language_model_dropout": 0.0,
            }
            changed = [
                name for name, expected_value in expected.items() if getattr(self, name) != expected_value
            ]
            if changed:
                raise ValueError(
                    "Official ESPnet checkpoint compatibility fixes these "
                    f"fields: {', '.join(changed)}. Use `variant='custom'` "
                    "for another graph.")

    @property
    def subsampled_feature_dimension(self) -> int:
        return ((self.n_mels - 1) // 2 - 1) // 3

    @property
    def minimum_feature_frames(self) -> int:
        """Smallest frame count accepted by the 3x3 then 5x5 frontend."""
        return 11

    @property
    def minimum_waveform_samples(self) -> int:
        """Smallest centered-STFT waveform that yields eleven frames."""
        return (self.minimum_feature_frames - 1) * self.hop_length

    @classmethod
    def from_dict(
        cls,
        values: Mapping[str, Any],
    ) -> ESPnetLibriSpeechTransformerConfig:
        if not isinstance(values, Mapping):
            raise TypeError("ESPnet configuration must be a mapping.")
        source = copy.deepcopy(dict(values))
        allowed = {item.name for item in fields(cls)}
        ignored = {
            "architecture",
            'architectures',
            "checkpoint_format",
            "model_type",
            "name_or_path",
            "source_artifact_revision",
            "source_asr_sha256",
            "source_asr_tensor_fingerprint",
            "source_lm_sha256",
            "source_lm_tensor_fingerprint",
            "source_revision",
            "source_tokenizer_sha256",
            "voicehub_provider",
        }
        extra = {key: source.pop(key) for key in tuple(source) if key not in allowed and key not in ignored}
        for key in ignored:
            source.pop(key, None)
        configured_extra = source.pop("extra_config", {})
        if configured_extra and not isinstance(configured_extra, Mapping):
            raise TypeError("`extra_config` must be a mapping.")
        extra.update(dict(configured_extra))
        source["extra_config"] = extra
        for name in ("frequency_mask_width", "time_mask_width"):
            if name in source:
                source[name] = tuple(source[name])
        return cls(**source)

    @classmethod
    def coerce(
        cls,
        value: ESPnetLibriSpeechTransformerConfig | Mapping[str, Any] | None,
    ) -> ESPnetLibriSpeechTransformerConfig:
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        return cls.from_dict(value)

    def to_dict(self) -> dict[str, Any]:
        result = {
            item.name: copy.deepcopy(getattr(self, item.name))
            for item in fields(self) if item.name != "extra_config"
        }
        result.update(copy.deepcopy(dict(self.extra_config)))
        result.update({
            "model_type": "asr_espnet",
            "architecture": "espnet-librispeech-transformer-e18",
        })
        return result


__all__ = ["ESPnetLibriSpeechTransformerConfig"]
