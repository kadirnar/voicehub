"""Compatibility facade over VoiceHub's native Llama decoder.

The active T3 implementation calls the decoder directly.  This
historical class name remains importable for downstream users without
inheriting from or importing Transformers.
"""

from __future__ import annotations

import torch
from torch import nn

from voicehub.models.causal_lm.native.modeling import CausalLMOutput


class T3HuggingfaceBackend(nn.Module):
    """Project a native Llama decoder's hidden states to speech logits."""

    def __init__(
        self,
        config,
        llama: nn.Module,
        *,
        speech_enc,
        speech_head,
        latents_queue=None,
        logits_queue=None,
        alignment_stream_analyzer=None,
    ) -> None:
        super().__init__()
        del latents_queue, logits_queue
        self.config = config
        self.model = llama
        self.speech_enc = speech_enc
        self.speech_head = speech_head
        self.alignment_stream_analyzer = alignment_stream_analyzer

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values=None,
        use_cache: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = True,
        return_dict: bool = True,
    ) -> CausalLMOutput:
        if not return_dict:
            raise ValueError("The native T3 compatibility backend returns structured outputs.")
        output = self.model(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        logits = self.speech_head(output.last_hidden_state)
        if self.alignment_stream_analyzer is not None:
            logits = self.alignment_stream_analyzer.step(logits)
        return CausalLMOutput(
            logits=logits.float(),
            past_key_values=output.past_key_values,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )


__all__ = ["T3HuggingfaceBackend"]
