"""Native Whisper prompting and autoregressive decoding policy.

The prompt layout, control-token suppression, and timestamp pairing
rules are based on OpenAI Whisper ``decoding.py`` at revision
``04f449b8a437f1bbd3dba5c9f826aca972e7709a``.  Language/task validation
and Hugging Face generation-config field names were checked against
Transformers ``generation_whisper.py`` and
``WhisperTimeStampLogitsProcessor`` at revision
``ebea912f0bb6f9e28ad2df04acd9b4df035933a9``.  This implementation
executes only VoiceHub's native model and generation engine.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence, Set
from dataclasses import dataclass, field
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

import torch
from torch import Tensor

from voicehub.generation import (
    AutoregressiveGenerator,
    GenerationConfig,
    GenerationOutput,
    GenerationStepInput,
    GenerationStepOutput,
)

_TASKS = frozenset({"transcribe", "translate"})


def _integer(name: str, value: Any, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"`{name}` must be an integer.")
    result = int(value)
    if result < minimum:
        raise ValueError(f"`{name}` must be at least {minimum}.")
    return result


def _optional_integer(name: str, value: Any) -> int | None:
    return None if value is None else _integer(name, value)


def _token_tuple(name: str, values: Sequence[int] | None) -> tuple[int, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"`{name}` must be a sequence of token IDs.")
    normalized = tuple(_integer(name, value) for value in values)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"`{name}` cannot contain duplicate token IDs.")
    return normalized


def _ordered_token_sequence(
    name: str,
    values: Iterable[int] | Tensor,
    *,
    maximum_length: int,
) -> tuple[int, ...]:
    """Materialize a bounded, ordered stream of token IDs.

    Native tokenizers return :class:`voicehub.tokenization.Encoding`,
    which deliberately exposes the iterable protocol without inheriting
    from ``Sequence``.  A bound is required here so a malformed or
    unbounded tokenizer iterator cannot stall generation.
    """
    if isinstance(values, Tensor):
        if values.ndim != 1:
            raise ValueError(f"`{name}` token tensor must have one dimension.")
        values = values.tolist()
    if isinstance(values, (str, bytes, Mapping, Set)):
        raise TypeError(f"`{name}` must be an ordered iterable of token IDs.")
    try:
        iterator = iter(values)
    except TypeError as error:
        raise TypeError(f"`{name}` must be an ordered iterable of token IDs.") from error

    normalized: list[int] = []
    for index, value in enumerate(iterator):
        if index >= maximum_length:
            raise ValueError(
                f"`{name}` contains more than the model's "
                f"{maximum_length}-token decoder context.")
        normalized.append(_integer(f"{name}[{index}]", value))
    return tuple(normalized)


def _required_config_integer(values: Mapping[str, Any], name: str) -> int:
    try:
        value = values[name]
    except KeyError:
        raise ValueError(f"Hugging Face Whisper generation config is missing `{name}`.") from None
    return _integer(name, value)


@runtime_checkable
class WhisperTokenizerProtocol(Protocol):
    """Minimal tokenizer surface used only for string prompts."""

    def encode(self, text: str) -> Iterable[int]:
        """Encode one prompt without adding Whisper control tokens."""
        ...


@dataclass(frozen=True, slots=True)
class WhisperTokenSet:
    """Model-format token metadata, independent from tokenizer internals."""

    eot: int
    sot: int
    translate: int
    transcribe: int
    sot_lm: int
    sot_prev: int
    no_speech: int | None
    no_timestamps: int | None
    timestamp_begin: int | None
    language_tokens: Mapping[str, int] = field(default_factory=dict)
    non_speech_tokens: tuple[int, ...] = ()
    blank_token_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        for name in (
                "eot",
                "sot",
                "translate",
                "transcribe",
                "sot_lm",
                "sot_prev",
        ):
            object.__setattr__(self, name, _integer(name, getattr(self, name)))
        for name in ("no_speech", "no_timestamps", "timestamp_begin"):
            object.__setattr__(
                self,
                name,
                _optional_integer(name, getattr(self, name)),
            )

        if self.timestamp_begin is not None:
            if self.no_timestamps is None:
                raise ValueError("`no_timestamps` is required when timestamp tokens exist.")
            if self.timestamp_begin != self.no_timestamps + 1:
                raise ValueError(
                    "Standard Whisper format requires `timestamp_begin` to "
                    "immediately follow `no_timestamps`.")

        if not isinstance(self.language_tokens, Mapping):
            raise TypeError("`language_tokens` must be a mapping.")
        languages: dict[str, int] = {}
        for language, token_id in self.language_tokens.items():
            if not isinstance(language, str) or not language.strip():
                raise ValueError("Language identifiers must be non-empty strings.")
            normalized = language.strip().lower()
            if normalized.startswith("<|") and normalized.endswith("|>"):
                normalized = normalized[2:-2]
            if normalized in languages:
                raise ValueError(f"`language_tokens` contains duplicate language {normalized!r}.")
            languages[normalized] = _integer("language token", token_id)
        if len(set(languages.values())) != len(languages):
            raise ValueError("Every language must use a distinct token ID.")
        object.__setattr__(
            self,
            "language_tokens",
            MappingProxyType(languages),
        )
        object.__setattr__(
            self,
            "non_speech_tokens",
            _token_tuple("non_speech_tokens", self.non_speech_tokens),
        )
        object.__setattr__(
            self,
            "blank_token_ids",
            _token_tuple("blank_token_ids", self.blank_token_ids),
        )

    @property
    def is_multilingual(self) -> bool:
        """Whether this tokenizer format exposes language control tokens."""
        return bool(self.language_tokens)

    def language_token(self, language: str) -> int:
        """Resolve and validate a model-supported language code."""
        if not isinstance(language, str) or not language.strip():
            raise ValueError("Whisper language must be a non-empty string.")
        normalized = language.strip().lower()
        if normalized.startswith("<|") and normalized.endswith("|>"):
            normalized = normalized[2:-2]
        try:
            return self.language_tokens[normalized]
        except KeyError:
            choices = ", ".join(self.language_tokens)
            raise ValueError(
                f"Whisper language {language!r} is not supported by this "
                f"checkpoint. Available codes: {choices}.") from None

    def task_token(self, task: str) -> int:
        """Return the control token for a validated Whisper task."""
        if task == "transcribe":
            return self.transcribe
        if task == "translate":
            return self.translate
        choices = ", ".join(sorted(_TASKS))
        raise ValueError(f"Whisper task must be one of {choices}; found {task!r}.")

    def all_token_ids(self) -> tuple[int, ...]:
        """Return all explicit IDs that must fit the model vocabulary."""
        values = [
            self.eot,
            self.sot,
            self.translate,
            self.transcribe,
            self.sot_lm,
            self.sot_prev,
            *self.language_tokens.values(),
            *self.non_speech_tokens,
            *self.blank_token_ids,
        ]
        for optional in (
                self.no_speech,
                self.no_timestamps,
                self.timestamp_begin,
        ):
            if optional is not None:
                values.append(optional)
        return tuple(values)

    @classmethod
    def from_huggingface_config(
        cls,
        values: Mapping[str, Any],
    ) -> WhisperTokenSet:
        """Read the official ``generation_config.json`` token metadata.

        Hugging Face records the task and language tables explicitly.
        Its standard Whisper format places ``sot_lm``, ``sot_prev``, and
        ``no_speech`` immediately before ``no_timestamps``; explicit
        ``*_token_id`` fields override those layout-derived defaults.
        """
        if not isinstance(values, Mapping):
            raise TypeError("Whisper generation configuration must be a mapping.")
        no_timestamps = _required_config_integer(
            values,
            "no_timestamps_token_id",
        )
        task_to_id = values.get("task_to_id")
        if task_to_id is None:
            task_to_id = {
                "translate": no_timestamps - 5,
                "transcribe": no_timestamps - 4,
            }
        if not isinstance(task_to_id, Mapping):
            raise TypeError("`task_to_id` must be a mapping or None.")
        for task in _TASKS:
            if task not in task_to_id:
                raise ValueError(f"`task_to_id` is missing the {task!r} task.")

        language_tokens = values.get("lang_to_id") or {}
        if not isinstance(language_tokens, Mapping):
            raise TypeError("`lang_to_id` must be a mapping.")
        if values.get("is_multilingual") is True and not language_tokens:
            raise ValueError("A multilingual Whisper generation config must define `lang_to_id`.")
        return cls(
            eot=_required_config_integer(values, "eos_token_id"),
            sot=_required_config_integer(values, "decoder_start_token_id"),
            translate=_integer("translate token", task_to_id["translate"]),
            transcribe=_integer("transcribe token", task_to_id["transcribe"]),
            sot_lm=_integer(
                "sot_lm_token_id",
                values.get("sot_lm_token_id", no_timestamps - 3),
            ),
            sot_prev=_integer(
                "prev_sot_token_id",
                values.get("prev_sot_token_id", no_timestamps - 2),
            ),
            no_speech=_integer(
                "no_speech_token_id",
                values.get("no_speech_token_id", no_timestamps - 1),
            ),
            no_timestamps=no_timestamps,
            timestamp_begin=no_timestamps + 1,
            language_tokens=language_tokens,
            non_speech_tokens=_token_tuple(
                "suppress_tokens",
                values.get("suppress_tokens"),
            ),
            blank_token_ids=_token_tuple(
                "begin_suppress_tokens",
                values.get("begin_suppress_tokens"),
            ),
        )


Prompt = str | Iterable[int] | Tensor | None
Language = str | Sequence[str] | None


@dataclass(frozen=True, slots=True)
class WhisperDecodingConfig:
    """Whisper policy layered over the model-neutral generation options."""

    generation: GenerationConfig = field(default_factory=GenerationConfig)
    task: str = "transcribe"
    language: Language = None
    prompt: Prompt = None
    prefix: Prompt = None
    suppress_tokens: tuple[int, ...] = ()
    suppress_non_speech: bool = True
    suppress_blank: bool = True
    return_timestamps: bool = False
    max_initial_timestamp: float | None = 1.0
    time_precision: float = 0.02

    def __post_init__(self) -> None:
        if not isinstance(self.generation, GenerationConfig):
            raise TypeError("`generation` must be a GenerationConfig.")
        if not isinstance(self.task, str):
            raise TypeError("`task` must be a string.")
        task = self.task.strip().lower()
        if task not in _TASKS:
            choices = ", ".join(sorted(_TASKS))
            raise ValueError(f"`task` must be one of {choices}; found {self.task!r}.")
        object.__setattr__(self, "task", task)

        language = self.language
        if isinstance(language, str):
            if not language.strip():
                raise ValueError("`language` cannot be an empty string.")
            object.__setattr__(self, "language", language.strip().lower())
        elif language is not None:
            if (isinstance(language, (str, bytes)) or not isinstance(language, Sequence)):
                raise TypeError("`language` must be a string, a sequence of strings, or None.")
            normalized_languages = tuple(language)
            if not normalized_languages:
                raise ValueError("A language sequence cannot be empty.")
            if any(not isinstance(item, str) or not item.strip() for item in normalized_languages):
                raise TypeError("Every item in a language sequence must be a non-empty string.")
            object.__setattr__(
                self,
                "language",
                tuple(item.strip().lower() for item in normalized_languages),
            )

        for name in ("prompt", "prefix"):
            value = getattr(self, name)
            if isinstance(value, bytes):
                raise TypeError(f"`{name}` cannot be bytes.")
            if value is None or isinstance(value, str):
                continue
            if isinstance(value, (Mapping, Set)):
                raise TypeError(f"`{name}` must be text, an ordered iterable of token IDs, or None.")
            try:
                iter(value)
            except TypeError as error:
                raise TypeError(
                    f"`{name}` must be text, an ordered iterable of token IDs, or None.") from error

        object.__setattr__(
            self,
            "suppress_tokens",
            _token_tuple("suppress_tokens", self.suppress_tokens),
        )
        for name in (
                "suppress_non_speech",
                "suppress_blank",
                "return_timestamps",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"`{name}` must be a boolean.")

        if self.max_initial_timestamp is not None:
            value = self.max_initial_timestamp
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError("`max_initial_timestamp` must be a real number.")
            value = float(value)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("`max_initial_timestamp` must be finite and non-negative.")
            object.__setattr__(self, "max_initial_timestamp", value)
        if isinstance(self.time_precision, bool) or not isinstance(self.time_precision, Real):
            raise TypeError("`time_precision` must be a real number.")
        precision = float(self.time_precision)
        if not math.isfinite(precision) or precision <= 0.0:
            raise ValueError("`time_precision` must be finite and positive.")
        object.__setattr__(self, "time_precision", precision)


def _validate_logits_and_history(logits: Tensor, history: Tensor) -> None:
    if not isinstance(logits, Tensor) or not isinstance(history, Tensor):
        raise TypeError("Whisper logits and token history must be tensors.")
    if logits.ndim != 2 or history.ndim != 2:
        raise ValueError(
            "Whisper logits/history require [batch, vocabulary] and "
            "[batch, sequence] shapes.")
    if logits.shape[0] != history.shape[0] or not logits.shape[0]:
        raise ValueError("Whisper logits and history batch sizes must match.")
    if logits.shape[1] < 1:
        raise ValueError("Whisper logits vocabulary cannot be empty.")
    if not logits.is_floating_point():
        raise TypeError("Whisper logits must use a floating-point dtype.")
    if history.dtype == torch.bool or history.is_floating_point() or history.is_complex():
        raise TypeError("Whisper token history must use an integer dtype.")
    if logits.device != history.device:
        raise ValueError("Whisper logits and token history must use one device.")


def apply_whisper_suppression(
    logits: Tensor,
    history: Tensor,
    *,
    suppress_tokens: Sequence[int],
    begin_suppress_tokens: Sequence[int],
    sample_begin: int,
) -> Tensor:
    """Apply permanent and first-step token suppression without mutation."""
    _validate_logits_and_history(logits, history)
    sample_begin = _integer("sample_begin", sample_begin)
    if sample_begin > history.shape[1]:
        raise ValueError("`sample_begin` cannot exceed the token history length.")
    permanent = _token_tuple("suppress_tokens", suppress_tokens)
    at_begin = _token_tuple("begin_suppress_tokens", begin_suppress_tokens)
    configured = permanent + at_begin
    if configured and max(configured) >= logits.shape[-1]:
        raise ValueError("A suppressed token ID is outside the model vocabulary.")

    processed = logits.clone()
    if permanent:
        processed[:, permanent] = -torch.inf
    if history.shape[1] == sample_begin and at_begin:
        processed[:, at_begin] = -torch.inf
    return processed


def apply_whisper_timestamp_rules(
    logits: Tensor,
    history: Tensor,
    *,
    token_set: WhisperTokenSet,
    sample_begin: int,
    max_initial_timestamp_index: int | None,
) -> Tensor:
    """Enforce Whisper's paired, monotonic timestamp-token grammar."""
    _validate_logits_and_history(logits, history)
    sample_begin = _integer("sample_begin", sample_begin)
    if sample_begin > history.shape[1]:
        raise ValueError("`sample_begin` cannot exceed the token history length.")
    if max_initial_timestamp_index is not None:
        max_initial_timestamp_index = _integer(
            "max_initial_timestamp_index",
            max_initial_timestamp_index,
        )
    timestamp_begin = token_set.timestamp_begin
    no_timestamps = token_set.no_timestamps
    if timestamp_begin is None or no_timestamps is None:
        raise ValueError("This Whisper model format does not define timestamps.")
    if max(timestamp_begin, no_timestamps, token_set.eot) >= logits.shape[-1]:
        raise ValueError("Whisper timestamp metadata contains an ID outside the vocabulary.")

    processed = logits.clone()
    processed[:, no_timestamps] = -torch.inf

    for row in range(history.shape[0]):
        sampled = history[row, sample_begin:]
        last_was_timestamp = (sampled.numel() >= 1 and sampled[-1] >= timestamp_begin)
        penultimate_was_timestamp = (sampled.numel() < 2 or sampled[-2] >= timestamp_begin)

        if last_was_timestamp:
            if penultimate_was_timestamp:
                processed[row, timestamp_begin:] = -torch.inf
            else:
                processed[row, :token_set.eot] = -torch.inf

        timestamps = sampled[sampled >= timestamp_begin]
        if timestamps.numel():
            if last_was_timestamp and not penultimate_was_timestamp:
                timestamp_last = int(timestamps[-1].item())
            else:
                timestamp_last = int(timestamps[-1].item()) + 1
            processed[row, timestamp_begin:timestamp_last] = -torch.inf

    if history.shape[1] == sample_begin:
        processed[:, :timestamp_begin] = -torch.inf
        if max_initial_timestamp_index is not None:
            last_allowed = timestamp_begin + max_initial_timestamp_index
            processed[:, last_allowed + 1:] = -torch.inf

    log_probabilities = torch.log_softmax(processed.float(), dim=-1)
    for row in range(history.shape[0]):
        timestamp_probability = log_probabilities[
            row,
            timestamp_begin:,
        ].logsumexp(dim=-1)
        max_text_probability = log_probabilities[
            row,
            :timestamp_begin,
        ].max()
        if timestamp_probability > max_text_probability:
            processed[row, :timestamp_begin] = -torch.inf
    return processed


@dataclass(frozen=True, slots=True)
class WhisperGenerationOutput:
    """Generated sequences plus the resolved Whisper prompt metadata."""

    generation: GenerationOutput
    prompt_length: int
    language_token_ids: Tensor | None

    @property
    def sequences(self) -> Tensor:
        return self.generation.sequences

    @property
    def generated_sequences(self) -> Tensor:
        """Return tokens emitted after the complete Whisper prompt."""
        return self.sequences[:, self.prompt_length:]

    @property
    def cache(self) -> Any | None:
        return self.generation.cache


class WhisperGenerationAdapter:
    """Bind a native Whisper model to VoiceHub's autoregressive engine."""

    def __init__(
        self,
        model: Any,
        token_set: WhisperTokenSet,
        *,
        tokenizer: WhisperTokenizerProtocol | None = None,
        generator: AutoregressiveGenerator | None = None,
    ) -> None:
        for method in ("encode", "decode"):
            if not callable(getattr(model, method, None)):
                raise TypeError(f"Whisper model must expose a callable `{method}`.")
        model_config = getattr(model, "config", None)
        for attribute in ("vocab_size", "max_target_positions"):
            if not isinstance(getattr(model_config, attribute, None), int):
                raise TypeError(
                    "Whisper model config must define integer `vocab_size` and "
                    "`max_target_positions` fields.")
        if not isinstance(token_set, WhisperTokenSet):
            raise TypeError("`token_set` must be a WhisperTokenSet.")
        if tokenizer is not None and not isinstance(tokenizer, WhisperTokenizerProtocol):
            raise TypeError("`tokenizer` must implement WhisperTokenizerProtocol.")
        if generator is not None and not isinstance(generator, AutoregressiveGenerator):
            raise TypeError("`generator` must be an AutoregressiveGenerator.")

        vocabulary_size = model_config.vocab_size
        if token_set.all_token_ids() and max(token_set.all_token_ids()) >= vocabulary_size:
            raise ValueError("Whisper token metadata contains an ID outside the model vocabulary.")
        self.model = model
        self.token_set = token_set
        self.tokenizer = tokenizer
        self.generator = generator or AutoregressiveGenerator()

    def _encode_prompt(self, value: Prompt, *, name: str) -> tuple[int, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            if not value.strip():
                return ()
            if self.tokenizer is None:
                raise ValueError(f"A tokenizer is required when `{name}` is a string.")
            value = self.tokenizer.encode(" " + value.strip())
        tokens = _ordered_token_sequence(
            name,
            value,
            maximum_length=self.model.config.max_target_positions,
        )
        if tokens and max(tokens) >= self.model.config.vocab_size:
            raise ValueError(f"`{name}` contains a token outside the model vocabulary.")
        return tokens

    def _resolve_requested_languages(
        self,
        language: Language,
        *,
        batch_size: int,
    ) -> tuple[int, ...] | None:
        if not self.token_set.is_multilingual:
            if language is not None:
                raise ValueError("An English-only Whisper format does not accept `language`.")
            return None
        if language is None:
            return None
        languages = ((language, ) * batch_size if isinstance(language, str) else tuple(language))
        if len(languages) != batch_size:
            raise ValueError("A per-row Whisper language list must match the audio batch size.")
        return tuple(self.token_set.language_token(item) for item in languages)

    def _detect_languages(
        self,
        encoder_hidden_states: Tensor,
        *,
        encoder_attention_mask: Tensor | None,
    ) -> tuple[int, ...]:
        language_ids = tuple(self.token_set.language_tokens.values())
        if not language_ids:
            raise ValueError("Language detection requires multilingual token metadata.")
        start_tokens = torch.full(
            (encoder_hidden_states.shape[0], 1),
            self.token_set.sot,
            dtype=torch.long,
            device=encoder_hidden_states.device,
        )
        output = self.model.decode(
            start_tokens,
            encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=False,
        )
        logits = output.logits[:, -1].float()
        candidates = torch.tensor(
            language_ids,
            dtype=torch.long,
            device=logits.device,
        )
        selected = logits.index_select(-1, candidates).argmax(dim=-1)
        return tuple(candidates[selected].tolist())

    def _initial_tokens(
        self,
        config: WhisperDecodingConfig,
        *,
        batch_size: int,
        language_ids: tuple[int, ...] | None,
        device: torch.device,
    ) -> Tensor:
        if not self.token_set.is_multilingual and config.task == "translate":
            raise ValueError("Translation requires a multilingual Whisper checkpoint.")
        if config.return_timestamps and self.token_set.timestamp_begin is None:
            raise ValueError("This Whisper model format does not provide timestamp tokens.")

        prefix = self._encode_prompt(config.prefix, name="prefix")
        prompt = self._encode_prompt(config.prompt, name="prompt")
        context_budget = (self.model.config.max_target_positions - config.generation.max_new_tokens)
        if context_budget < 1:
            raise ValueError("`max_new_tokens` leaves no room for a Whisper start token.")

        rows: list[list[int]] = []
        for row in range(batch_size):
            base = [self.token_set.sot]
            if self.token_set.is_multilingual:
                if language_ids is None:
                    raise RuntimeError("A multilingual prompt requires a language ID.")
                base.extend((
                    language_ids[row],
                    self.token_set.task_token(config.task),
                ))
            if not config.return_timestamps and self.token_set.no_timestamps is not None:
                base.append(self.token_set.no_timestamps)

            if len(base) > context_budget:
                raise ValueError(
                    "Whisper control tokens and requested generation length "
                    "exceed the decoder context.")
            available_prefix = context_budget - len(base)
            row_prefix = list(prefix[-available_prefix:]) if available_prefix else []
            current = base + row_prefix

            if prompt:
                remaining = context_budget - len(current)
                prompt_limit = min(
                    self.model.config.max_target_positions // 2 - 1,
                    max(remaining - 1, 0),
                )
                if prompt_limit < 1:
                    raise ValueError(
                        "The Whisper prompt cannot fit alongside control, "
                        "prefix, and requested output tokens.")
                current = [
                    self.token_set.sot_prev,
                    *prompt[-prompt_limit:],
                    *current,
                ]
            rows.append(current)

        lengths = {len(row) for row in rows}
        if len(lengths) != 1:
            raise RuntimeError("Whisper prompt rows must have equal lengths.")
        return torch.tensor(
            rows,
            dtype=torch.long,
            device=device,
        )

    def _suppression_sets(
        self,
        config: WhisperDecodingConfig,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        permanent = list(config.suppress_tokens)
        if config.suppress_non_speech:
            permanent.extend(self.token_set.non_speech_tokens)
        permanent.extend((
            self.token_set.transcribe,
            self.token_set.translate,
            self.token_set.sot,
            self.token_set.sot_prev,
            self.token_set.sot_lm,
        ))
        if self.token_set.no_speech is not None:
            permanent.append(self.token_set.no_speech)

        at_begin: list[int] = []
        if config.suppress_blank:
            at_begin.extend(self.token_set.blank_token_ids)
            if self.tokenizer is not None:
                at_begin.extend(self.tokenizer.encode(" "))
            at_begin.append(self.token_set.eot)
        return tuple(sorted(set(permanent))), tuple(sorted(set(at_begin)))

    def _encoder_outputs(
        self,
        input_features: Tensor | None,
        encoder_outputs: Tensor | None,
        attention_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor | None]:
        if (input_features is None) == (encoder_outputs is None):
            raise ValueError("Provide exactly one of `input_features` and `encoder_outputs`.")
        if encoder_outputs is not None:
            if not isinstance(encoder_outputs, Tensor) or encoder_outputs.ndim != 3:
                raise ValueError("`encoder_outputs` must have [batch, audio, width] shape.")
            return encoder_outputs, attention_mask

        if not isinstance(input_features, Tensor):
            raise TypeError("`input_features` must be a PyTorch tensor.")
        encoded = self.model.encode(
            input_features,
            attention_mask=attention_mask,
        )
        if attention_mask is None:
            return encoded, None
        encoder_mask = self.model.encoder.downsample_attention_mask(
            attention_mask,
            batch_size=input_features.shape[0],
            input_frames=input_features.shape[2],
            device=encoded.device,
        )
        return encoded, encoder_mask

    @torch.inference_mode()
    def generate(
        self,
        input_features: Tensor | None = None,
        *,
        encoder_outputs: Tensor | None = None,
        attention_mask: Tensor | None = None,
        config: WhisperDecodingConfig | None = None,
    ) -> WhisperGenerationOutput:
        """Generate one token sequence per encoded audio row."""
        config = config or WhisperDecodingConfig()
        if not isinstance(config, WhisperDecodingConfig):
            raise TypeError("`config` must be a WhisperDecodingConfig.")
        encoded, encoder_mask = self._encoder_outputs(
            input_features,
            encoder_outputs,
            attention_mask,
        )
        batch_size = encoded.shape[0]
        if batch_size < 1:
            raise ValueError("Whisper generation requires a non-empty batch.")

        language_ids = self._resolve_requested_languages(
            config.language,
            batch_size=batch_size,
        )
        if self.token_set.is_multilingual and language_ids is None:
            language_ids = self._detect_languages(
                encoded,
                encoder_attention_mask=encoder_mask,
            )
        initial_tokens = self._initial_tokens(
            config,
            batch_size=batch_size,
            language_ids=language_ids,
            device=encoded.device,
        )
        sample_begin = initial_tokens.shape[1]

        generation_config = config.generation
        updates: dict[str, Any] = {}
        if not generation_config.eos_token_ids:
            updates["eos_token_id"] = self.token_set.eot
        if generation_config.pad_token_id is None:
            updates["pad_token_id"] = self.token_set.eot
        if updates:
            generation_config = generation_config.with_updates(**updates)

        suppress_tokens, begin_suppress_tokens = self._suppression_sets(config)
        max_initial_timestamp_index = (
            None if config.max_initial_timestamp is None else round(
                config.max_initial_timestamp / config.time_precision))
        history = initial_tokens

        def decoder_step(step: GenerationStepInput) -> GenerationStepOutput:
            nonlocal history
            if step.step_index == 0:
                history = initial_tokens
            elif step.cache is None:
                history = step.token_ids
            else:
                history = torch.cat((history, step.token_ids), dim=-1)

            decoded = self.model.decode(
                step.token_ids,
                encoded,
                encoder_attention_mask=encoder_mask,
                past_key_values=step.cache,
                use_cache=step.use_cache,
            )
            logits = apply_whisper_suppression(
                decoded.logits[:, -1],
                history,
                suppress_tokens=suppress_tokens,
                begin_suppress_tokens=begin_suppress_tokens,
                sample_begin=sample_begin,
            )
            if config.return_timestamps:
                logits = apply_whisper_timestamp_rules(
                    logits,
                    history,
                    token_set=self.token_set,
                    sample_begin=sample_begin,
                    max_initial_timestamp_index=max_initial_timestamp_index,
                )
            return GenerationStepOutput(
                logits=logits,
                cache=decoded.past_key_values,
            )

        output = self.generator.generate(
            decoder_step,
            initial_tokens,
            generation_config,
        )
        language_tensor = (
            None if language_ids is None else torch.tensor(
                language_ids,
                dtype=torch.long,
                device=initial_tokens.device,
            ))
        return WhisperGenerationOutput(
            generation=output,
            prompt_length=sample_begin,
            language_token_ids=language_tensor,
        )


__all__ = [
    "Language",
    "Prompt",
    "WhisperDecodingConfig",
    "WhisperGenerationAdapter",
    "WhisperGenerationOutput",
    "WhisperTokenSet",
    "WhisperTokenizerProtocol",
    "apply_whisper_suppression",
    "apply_whisper_timestamp_rules",
]
