"""Validated configuration for the native SpeechBrain CRDNN ASR graph."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
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
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"`{name}` must be a real number.")
    result = float(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"`{name}` must be at least {minimum}.")
    if maximum is not None and result > maximum:
        raise ValueError(f"`{name}` must be at most {maximum}.")
    return result


def _integer_tuple(
    name: str,
    value: Sequence[int],
    *,
    minimum: int = 1,
) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"`{name}` must be a sequence of integers.")
    result = tuple(_integer(f"{name}[{index}]", item, minimum=minimum) for index, item in enumerate(value))
    if not result:
        raise ValueError(f"`{name}` cannot be empty.")
    return result


@dataclass(frozen=True, slots=True)
class SpeechBrainCRDNNASRConfig:
    """Complete graph and recipe contract for the BPE-1000 release.

    The default values are the exact public
    ``speechbrain/asr-crdnn-rnnlm-librispeech`` architecture.  ``custom``
    configurations are useful for tests and research but are not accepted as
    official-checkpoint compatible.
    """

    variant: str = "librispeech-bpe-1000"
    sampling_rate: int = 16_000
    n_fft: int = 400
    win_length: int = 400
    hop_length: int = 160
    n_mels: int = 40
    f_min: float = 0.0
    f_max: float = 8_000.0
    top_db: float = 80.0
    normalization_epsilon: float = 1e-10
    normalization_update_until_epoch: int = 3
    cnn_channels: tuple[int, ...] = (128, 256)
    cnn_kernel_size: tuple[int, int] = (3, 3)
    inter_layer_pooling_size: tuple[int, ...] = (2, 2)
    time_pooling_size: int = 4
    rnn_layers: int = 4
    rnn_neurons: int = 1_024
    rnn_bidirectional: bool = True
    dnn_blocks: int = 2
    dnn_neurons: int = 512
    embedding_size: int = 128
    decoder_neurons: int = 1_024
    attention_dim: int = 1_024
    attention_channels: int = 10
    attention_kernel_size: int = 100
    output_neurons: int = 1_000
    lm_rnn_layers: int = 2
    lm_rnn_neurons: int = 2_048
    lm_dnn_blocks: int = 1
    lm_dnn_neurons: int = 512
    freeze_language_model: bool = True
    dropout: float = 0.15
    lm_dropout: float = 0.0
    negative_slope: float = 0.01
    blank_token_id: int = 0
    bos_token_id: int = 0
    eos_token_id: int = 0
    beam_size: int = 80
    minimum_decode_ratio: float = 0.0
    maximum_decode_ratio: float = 1.0
    eos_threshold: float = 1.5
    maximum_attention_shift: int = 240
    lm_weight: float = 0.5
    coverage_penalty: float = 1.5
    temperature: float = 1.25
    lm_temperature: float = 1.25
    label_smoothing: float = 0.1
    ctc_weight: float = 0.5
    number_of_ctc_epochs: int = 5
    extra_config: Mapping[str, Any] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.variant, str) or not self.variant.strip():
            raise ValueError("`variant` must be a non-empty string.")
        variant = self.variant.strip().lower().replace("_", "-")
        if variant not in {"librispeech-bpe-1000", "custom"}:
            raise ValueError(
                "SpeechBrain CRDNN ASR supports the audited "
                "`librispeech-bpe-1000` graph or `custom`.")
        object.__setattr__(self, "variant", variant)
        for name in (
                "sampling_rate",
                "n_fft",
                "win_length",
                "hop_length",
                "n_mels",
                "time_pooling_size",
                "rnn_layers",
                "rnn_neurons",
                "dnn_blocks",
                "dnn_neurons",
                "embedding_size",
                "decoder_neurons",
                "attention_dim",
                "attention_channels",
                "attention_kernel_size",
                "output_neurons",
                "lm_rnn_layers",
                "lm_rnn_neurons",
                "lm_dnn_blocks",
                "lm_dnn_neurons",
                "beam_size",
                "maximum_attention_shift",
        ):
            object.__setattr__(
                self,
                name,
                _integer(name, getattr(self, name)),
            )
        object.__setattr__(
            self,
            "normalization_update_until_epoch",
            _integer(
                "normalization_update_until_epoch",
                self.normalization_update_until_epoch,
                minimum=0,
            ),
        )
        object.__setattr__(
            self,
            "number_of_ctc_epochs",
            _integer(
                "number_of_ctc_epochs",
                self.number_of_ctc_epochs,
                minimum=0,
            ),
        )
        if self.n_fft < self.win_length:
            raise ValueError("`n_fft` cannot be smaller than `win_length`.")
        if self.win_length > self.sampling_rate:
            raise ValueError("`win_length` cannot exceed one second.")
        channels = _integer_tuple("cnn_channels", self.cnn_channels)
        pooling = _integer_tuple(
            "inter_layer_pooling_size",
            self.inter_layer_pooling_size,
        )
        kernel = _integer_tuple("cnn_kernel_size", self.cnn_kernel_size)
        if len(kernel) != 2 or any(value % 2 == 0 for value in kernel):
            raise ValueError("`cnn_kernel_size` must contain two odd dimensions.")
        if len(channels) != len(pooling):
            raise ValueError("`inter_layer_pooling_size` must match `cnn_channels`.")
        frequency = self.n_mels
        for amount in pooling:
            if frequency < amount:
                raise ValueError("CNN pooling collapses the mel dimension.")
            frequency //= amount
        object.__setattr__(self, "cnn_channels", channels)
        object.__setattr__(self, "inter_layer_pooling_size", pooling)
        object.__setattr__(self, "cnn_kernel_size", (kernel[0], kernel[1]))
        for name in (
                "freeze_language_model",
                "rnn_bidirectional",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"`{name}` must be a boolean.")
        for name in (
                "f_min",
                "f_max",
                "top_db",
                "normalization_epsilon",
                "negative_slope",
                "eos_threshold",
                "temperature",
                "lm_temperature",
        ):
            object.__setattr__(
                self,
                name,
                _real(name, getattr(self, name), minimum=0.0),
            )
        if not 0.0 <= self.f_min < self.f_max <= self.sampling_rate / 2:
            raise ValueError("Mel bounds must satisfy 0 <= f_min < f_max <= Nyquist.")
        for name in (
                "dropout",
                "lm_dropout",
                "label_smoothing",
                "ctc_weight",
                "lm_weight",
        ):
            value = _real(name, getattr(self, name), minimum=0.0, maximum=1.0)
            if name in {"dropout", "lm_dropout", "label_smoothing"} and value >= 1.0:
                raise ValueError(f"`{name}` must be less than 1.")
            object.__setattr__(self, name, value)
        for name in ("minimum_decode_ratio", "maximum_decode_ratio"):
            object.__setattr__(
                self,
                name,
                _real(name, getattr(self, name), minimum=0.0),
            )
        if self.maximum_decode_ratio <= self.minimum_decode_ratio:
            raise ValueError("`maximum_decode_ratio` must exceed `minimum_decode_ratio`.")
        object.__setattr__(
            self,
            "coverage_penalty",
            _real("coverage_penalty", self.coverage_penalty, minimum=0.0),
        )
        for name in ("blank_token_id", "bos_token_id", "eos_token_id"):
            value = _integer(name, getattr(self, name), minimum=0)
            if value >= self.output_neurons:
                raise ValueError(f"`{name}` must be smaller than vocabulary size.")
            object.__setattr__(self, name, value)
        if not isinstance(self.extra_config, Mapping):
            raise TypeError("`extra_config` must be a mapping.")
        object.__setattr__(
            self,
            "extra_config",
            MappingProxyType(copy.deepcopy(dict(self.extra_config))),
        )
        if self.variant == "librispeech-bpe-1000":
            expected = {
                "sampling_rate": 16_000,
                "n_fft": 400,
                "win_length": 400,
                "hop_length": 160,
                "n_mels": 40,
                "f_min": 0.0,
                "f_max": 8_000.0,
                "top_db": 80.0,
                "cnn_channels": (128, 256),
                "cnn_kernel_size": (3, 3),
                "inter_layer_pooling_size": (2, 2),
                "time_pooling_size": 4,
                "rnn_layers": 4,
                "rnn_neurons": 1_024,
                "rnn_bidirectional": True,
                "dnn_blocks": 2,
                "dnn_neurons": 512,
                "embedding_size": 128,
                "decoder_neurons": 1_024,
                "attention_dim": 1_024,
                "attention_channels": 10,
                "attention_kernel_size": 100,
                "output_neurons": 1_000,
                "lm_rnn_layers": 2,
                "lm_rnn_neurons": 2_048,
                "lm_dnn_blocks": 1,
                "lm_dnn_neurons": 512,
                "freeze_language_model": True,
                "blank_token_id": 0,
                "bos_token_id": 0,
                "eos_token_id": 0,
            }
            changed = [name for name, value in expected.items() if getattr(self, name) != value]
            if changed:
                raise ValueError(
                    "Official SpeechBrain checkpoint compatibility fixes "
                    f"these fields: {', '.join(changed)}. Use "
                    "`variant='custom'` for another graph.")

    @property
    def cnn_output_frequency(self) -> int:
        result = self.n_mels
        for amount in self.inter_layer_pooling_size:
            result //= amount
        return result

    @property
    def encoder_rnn_input_size(self) -> int:
        return self.cnn_output_frequency * self.cnn_channels[-1]

    @property
    def encoder_rnn_output_size(self) -> int:
        return self.rnn_neurons * (2 if self.rnn_bidirectional else 1)

    @classmethod
    def from_dict(
        cls,
        values: Mapping[str, Any],
    ) -> SpeechBrainCRDNNASRConfig:
        if not isinstance(values, Mapping):
            raise TypeError("SpeechBrain ASR configuration must be a mapping.")
        source = copy.deepcopy(dict(values))
        allowed = {item.name for item in fields(cls)}
        ignored = {
            "architecture",
            'architectures',
            "checkpoint_format",
            "model_type",
            "name_or_path",
            "source_artifact_revision",
            "source_checkpoint_sha256",
            "source_hyperparams_sha256",
            "source_lm_sha256",
            "source_normalizer_sha256",
            "source_tensor_fingerprint",
            "source_tokenizer_sha256",
            "source_training_revision",
        }
        extra = {key: source.pop(key) for key in tuple(source) if key not in allowed and key not in ignored}
        for key in ignored:
            source.pop(key, None)
        configured_extra = source.pop("extra_config", {})
        if configured_extra and not isinstance(configured_extra, Mapping):
            raise TypeError("`extra_config` must be a mapping.")
        extra.update(dict(configured_extra))
        source["extra_config"] = extra
        for name in (
                "cnn_channels",
                "cnn_kernel_size",
                "inter_layer_pooling_size",
        ):
            if name in source:
                source[name] = tuple(source[name])
        return cls(**source)

    @classmethod
    def coerce(
        cls,
        value: SpeechBrainCRDNNASRConfig | Mapping[str, Any],
    ) -> SpeechBrainCRDNNASRConfig:
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
            "model_type": "asr_speechbrain",
            "architecture": "speechbrain-crdnn-asr",
        })
        return result


__all__ = ["SpeechBrainCRDNNASRConfig"]
