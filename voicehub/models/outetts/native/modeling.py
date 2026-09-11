"""OuteTTS causal LM with the source-required local repetition window."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor

from voicehub.generation.config import GenerationConfig
from voicehub.generation.engine import (
    AutoregressiveGenerator,
    GenerationOutput,
    GenerationStepInput,
    GenerationStepOutput,
)
from voicehub.generation.logits import apply_repetition_penalty
from voicehub.generation.stopping import StoppingCriterion
from voicehub.models.causal_lm.native.modeling import CausalLMForCausalLM
from voicehub.neural.cache import DynamicKVCache


class RecentWindowRepetitionProcessor:
    """Apply repetition penalty only to the most recent protocol tokens."""

    def __init__(self, penalty: float, window: int = 64) -> None:
        if isinstance(window, bool) or not isinstance(window, int) or window < 1:
            raise ValueError("OuteTTS repetition window must be positive.")
        self.penalty = float(penalty)
        self.window = window

    def __call__(self, input_ids: Tensor, logits: Tensor) -> Tensor:
        return apply_repetition_penalty(
            logits,
            input_ids[:, -self.window:],
            self.penalty,
        )


class OuteTTSForCausalLM(CausalLMForCausalLM):
    """Dense Llama/Qwen decoder with OuteTTS generation semantics."""

    def generate(
            self,
            input_ids: Tensor,
            *,
            attention_mask: Tensor | None = None,
            generation_config: GenerationConfig | None = None,
            repetition_window: int = 64,
            stopping_criteria: Sequence[StoppingCriterion] = (),
    ) -> GenerationOutput:
        if (attention_mask is not None and (not isinstance(attention_mask, Tensor) or
                                            tuple(attention_mask.shape) != tuple(input_ids.shape))):
            raise ValueError("`attention_mask` must have the same shape as `input_ids`.")
        config = generation_config or GenerationConfig(
            eos_token_id=self.config.eos_token_id,
            pad_token_id=self.config.pad_token_id,
            use_cache=self.config.use_cache,
        )
        if not isinstance(config, GenerationConfig):
            raise TypeError("`generation_config` must be a VoiceHub GenerationConfig.")
        updates: dict[str, Any] = {}
        if config.eos_token_id is None and self.config.eos_token_id is not None:
            updates["eos_token_id"] = self.config.eos_token_id
        if config.pad_token_id is None and self.config.pad_token_id is not None:
            updates["pad_token_id"] = self.config.pad_token_id
        if updates:
            config = config.with_updates(**updates)
        repetition_processor = RecentWindowRepetitionProcessor(
            config.repetition_penalty,
            repetition_window,
        )
        engine_config = config.with_updates(repetition_penalty=1.0)
        prompt_mask = (None if attention_mask is None else attention_mask.to(device=input_ids.device))
        if prompt_mask is not None and bool(prompt_mask.all()):
            prompt_mask = None
        prompt_length = input_ids.shape[1]

        def decoder_step(step: GenerationStepInput) -> GenerationStepOutput:
            past_length = (step.cache.sequence_length() if isinstance(step.cache, DynamicKVCache) else 0)
            key_length = past_length + step.token_ids.shape[1]
            if key_length < prompt_length:
                raise RuntimeError("Decoder cache length is shorter than the prompt mask.")
            generated = key_length - prompt_length
            step_mask = prompt_mask
            if generated and prompt_mask is not None:
                step_mask = torch.cat(
                    (
                        prompt_mask,
                        torch.ones(
                            prompt_mask.shape[0],
                            generated,
                            dtype=prompt_mask.dtype,
                            device=prompt_mask.device,
                        ),
                    ),
                    dim=-1,
                )
            output = self(
                step.token_ids,
                attention_mask=step_mask,
                past_key_values=step.cache,
                use_cache=step.use_cache,
            )
            return GenerationStepOutput(
                logits=output.logits,
                cache=output.past_key_values,
            )

        return AutoregressiveGenerator().generate(
            decoder_step,
            input_ids,
            engine_config,
            stopping_criteria=stopping_criteria,
            logits_processors=(repetition_processor, ),
        )


__all__ = [
    "OuteTTSForCausalLM",
    "RecentWindowRepetitionProcessor",
]
