"""Cache-aware autoregressive generation without third-party model runtimes."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import torch
from torch import Tensor

from voicehub.generation.config import GenerationConfig
from voicehub.generation.logits import _raise_processing_error
from voicehub.generation.sampling import _sample_next_token, create_generator
from voicehub.generation.stopping import EosStoppingCriterion, StoppingCriterion, evaluate_stopping_criteria


@dataclass(frozen=True, slots=True)
class GenerationStepInput:
    """Input passed to one native decoder step."""

    token_ids: Tensor
    cache: Any | None
    use_cache: bool
    step_index: int


@dataclass(frozen=True, slots=True)
class GenerationStepOutput:
    """Last-token logits and an opaque model-specific cache."""

    logits: Tensor
    cache: Any | None = None


DecoderStep = Callable[[GenerationStepInput], GenerationStepOutput]


@runtime_checkable
class LogitsProcessor(Protocol):
    """Transform next-token logits from the complete generated history.

    Processors are request-local policies such as constrained grammars.
    They must return a floating-point tensor with the same shape and
    device as the supplied logits. The generator passes a clone, so a
    processor may use efficient in-place masking without mutating model-
    owned output buffers.
    """

    def __call__(self, input_ids: Tensor, logits: Tensor) -> Tensor:
        """Return processed ``[batch, vocabulary]`` logits."""
        ...


@dataclass(frozen=True, slots=True)
class GenerationOutput:
    """Generated token sequences and row-wise completion metadata."""

    sequences: Tensor
    generated_lengths: Tensor
    finished: Tensor
    cache: Any | None

    @property
    def batch_size(self) -> int:
        return self.sequences.shape[0]


class AutoregressiveGenerator:
    """Run a model-neutral, cache-aware decoding loop.

    The decoder owns its cache representation. When it returns a
    non-``None`` cache, subsequent requests contain only the newly
    emitted token. If cache use is disabled, or the decoder declines to
    return one, it receives the complete sequence on every step.
    """

    def generate(
            self,
            decoder_step: DecoderStep,
            input_ids: Tensor,
            config: GenerationConfig,
            *,
            initial_cache: Any | None = None,
            stopping_criteria: Sequence[StoppingCriterion] = (),
            logits_processors: Sequence[LogitsProcessor] = (),
    ) -> GenerationOutput:
        """Generate up to ``max_new_tokens`` tokens for every input row."""
        self._validate_request(decoder_step, input_ids, config)
        processors = self._validate_logits_processors(logits_processors)
        sequences = input_ids
        batch_size = input_ids.shape[0]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
        generated_lengths = torch.zeros(batch_size, dtype=torch.long, device=input_ids.device)
        cache = initial_cache if config.use_cache else None
        decoder_tokens = input_ids
        generator: torch.Generator | None = None

        criteria = list(stopping_criteria)
        if config.eos_token_ids:
            criteria.insert(0, EosStoppingCriterion(config.eos_token_ids))
        if criteria and config.effective_pad_token_id is None:
            raise ValueError(
                "`pad_token_id` is required for row-wise stopping when no EOS token is configured.")

        for step_index in range(config.max_new_tokens):
            step_output = decoder_step(
                GenerationStepInput(
                    token_ids=decoder_tokens,
                    cache=cache,
                    use_cache=config.use_cache,
                    step_index=step_index,
                ))
            logits = self._last_token_logits(step_output, batch_size)
            self._validate_token_configuration(config, logits.shape[-1])
            logits = self._process_logits(
                processors,
                sequences,
                logits,
            )
            if config.do_sample and generator is None:
                generator = create_generator(logits.device, config.seed)

            next_tokens, processed, violation = _sample_next_token(
                logits,
                sequences,
                config,
                generator=generator,
            )
            if next_tokens.device != sequences.device:
                raise ValueError("Decoder logits and input IDs must be on the same device.")

            if criteria:
                # Finished rows emit padding; masking is a no-op for the other
                # rows, so no host-side `finished.any()` is needed.
                next_tokens = next_tokens.masked_fill(finished, config.effective_pad_token_id)

            history = sequences
            sequences = torch.cat((sequences, next_tokens[:, None]), dim=-1)
            generated_lengths += (~finished).long()
            done = None
            if criteria:
                newly_finished = evaluate_stopping_criteria(
                    criteria,
                    sequences,
                    next_tokens,
                    step_index,
                )
                finished |= newly_finished
                done = finished.all()

            # The only host synchronization of a step: logit validation and
            # the stop decision are read back together.
            if done is None:
                invalid, stop = bool(violation), False
            else:
                invalid, stop = torch.stack((violation, done)).tolist()
            if invalid:
                _raise_processing_error(
                    logits,
                    history,
                    processed,
                    repetition_penalty=config.repetition_penalty,
                )
            cache = step_output.cache if config.use_cache else None
            if stop:
                break
            decoder_tokens = next_tokens[:, None] if cache is not None else sequences

        return GenerationOutput(
            sequences=sequences,
            generated_lengths=generated_lengths,
            finished=finished,
            cache=cache,
        )

    @staticmethod
    def _validate_request(
        decoder_step: DecoderStep,
        input_ids: Tensor,
        config: GenerationConfig,
    ) -> None:
        if not callable(decoder_step):
            raise TypeError("`decoder_step` must be callable.")
        if not isinstance(input_ids, Tensor):
            raise TypeError("`input_ids` must be a PyTorch tensor.")
        if input_ids.ndim != 2:
            raise ValueError(
                f"`input_ids` must have shape [batch, sequence], found {tuple(input_ids.shape)}.")
        if input_ids.shape[0] == 0 or input_ids.shape[1] == 0:
            raise ValueError("`input_ids` must have a non-empty batch and sequence.")
        if input_ids.dtype == torch.bool or input_ids.is_floating_point() or input_ids.is_complex():
            raise TypeError("`input_ids` must use an integer dtype.")
        if not isinstance(config, GenerationConfig):
            raise TypeError("`config` must be a GenerationConfig.")

    @staticmethod
    def _last_token_logits(step_output: GenerationStepOutput, batch_size: int) -> Tensor:
        if not isinstance(step_output, GenerationStepOutput):
            raise TypeError("A decoder step must return GenerationStepOutput.")
        logits = step_output.logits
        if not isinstance(logits, Tensor):
            raise TypeError("Decoder logits must be a PyTorch tensor.")
        if logits.ndim == 3:
            if logits.shape[1] == 0:
                raise ValueError("Decoder logits cannot have an empty time dimension.")
            logits = logits[:, -1, :]
        if logits.ndim != 2 or logits.shape[0] != batch_size:
            raise ValueError(
                "Decoder logits must have shape [batch, vocabulary] or "
                f"[batch, time, vocabulary]; found {tuple(step_output.logits.shape)}.")
        return logits

    @staticmethod
    def _validate_token_configuration(config: GenerationConfig, vocabulary_size: int) -> None:
        configured_ids = config.eos_token_ids
        if config.pad_token_id is not None:
            configured_ids += (config.pad_token_id, )
        if configured_ids and max(configured_ids) >= vocabulary_size:
            raise ValueError("A configured EOS or pad token ID is outside the decoder vocabulary.")

    @staticmethod
    def _validate_logits_processors(processors: Sequence[LogitsProcessor], ) -> tuple[LogitsProcessor, ...]:
        if isinstance(processors, (str, bytes)) or not isinstance(
                processors,
                Sequence,
        ):
            raise TypeError("`logits_processors` must be a sequence.")
        resolved = tuple(processors)
        if not all(callable(processor) for processor in resolved):
            raise TypeError("Every logits processor must be callable.")
        return resolved

    @staticmethod
    def _process_logits(
        processors: tuple[LogitsProcessor, ...],
        input_ids: Tensor,
        logits: Tensor,
    ) -> Tensor:
        if not processors:
            return logits
        processed = logits.clone()
        expected_shape = logits.shape
        expected_device = logits.device
        for processor in processors:
            processed = processor(input_ids, processed)
            if not isinstance(processed, Tensor):
                raise TypeError("A logits processor must return a PyTorch tensor.")
            if processed.shape != expected_shape:
                raise ValueError(
                    "A logits processor changed the logits shape from "
                    f"{tuple(expected_shape)} to {tuple(processed.shape)}.")
            if processed.device != expected_device:
                raise ValueError("A logits processor changed the logits device.")
            if not processed.is_floating_point():
                raise TypeError("Processed logits must use a floating-point dtype.")
        return processed
