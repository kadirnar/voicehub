"""Native batched attention/RNNLM beam search for SpeechBrain CRDNN."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from voicehub.models.asr_speechbrain.native.configuration import SpeechBrainCRDNNASRConfig
from voicehub.models.asr_speechbrain.native.modeling import SpeechBrainCRDNNForASR


@dataclass(frozen=True, slots=True)
class SpeechBrainBeamResult:
    """Top decoded token sequence and normalized score for each item."""

    token_ids: tuple[tuple[int, ...], ...]
    scores: tuple[float, ...]


def _select_hidden(
    hidden: tuple[Tensor, Tensor] | Tensor | None,
    indexes: Tensor,
) -> tuple[Tensor, Tensor] | Tensor | None:
    if hidden is None:
        return None
    if isinstance(hidden, tuple):
        return (
            hidden[0].index_select(1, indexes),
            hidden[1].index_select(1, indexes),
        )
    return hidden.index_select(1, indexes)


class SpeechBrainRNNLMBeamSearch:
    """Pinned SpeechBrain attention beam search with RNNLM shallow fusion."""

    def __init__(
        self,
        model: SpeechBrainCRDNNForASR,
        config: SpeechBrainCRDNNASRConfig | None = None,
    ) -> None:
        if not isinstance(model, SpeechBrainCRDNNForASR):
            raise TypeError("`model` must be SpeechBrainCRDNNForASR.")
        self.model = model
        self.config = model.config if config is None else SpeechBrainCRDNNASRConfig.coerce(config)

    @torch.no_grad()
    def __call__(
        self,
        encoder_states: Tensor,
        relative_lengths: Tensor,
        *,
        beam_size: int | None = None,
        lm_weight: float | None = None,
    ) -> SpeechBrainBeamResult:
        config = self.config
        beam = config.beam_size if beam_size is None else beam_size
        if isinstance(beam, bool) or not isinstance(beam, int) or beam < 1:
            raise ValueError("`beam_size` must be a positive integer.")
        if beam > config.output_neurons:
            raise ValueError("`beam_size` cannot exceed the vocabulary size.")
        resolved_lm_weight = config.lm_weight if lm_weight is None else lm_weight
        if (isinstance(resolved_lm_weight, bool) or not isinstance(resolved_lm_weight, (int, float)) or
                not 0.0 <= float(resolved_lm_weight) <= 1.0):
            raise ValueError("`lm_weight` must be a real number in [0, 1].")
        resolved_lm_weight = float(resolved_lm_weight)
        if encoder_states.ndim != 3:
            raise ValueError("`encoder_states` must have shape [batch, frames, channels].")
        relative_lengths = torch.as_tensor(
            relative_lengths,
            dtype=encoder_states.dtype,
            device=encoder_states.device,
        )
        if (relative_lengths.ndim != 1 or relative_lengths.shape[0] != encoder_states.shape[0]):
            raise ValueError("`relative_lengths` must have shape [batch].")

        batch_size = encoder_states.shape[0]
        encoder_lengths = torch.round(encoder_states.shape[1] * relative_lengths, ).long()
        expanded_states = encoder_states.repeat_interleave(beam, dim=0)
        expanded_lengths = encoder_lengths.repeat_interleave(beam, dim=0)
        attention_state = self.model.decoder.attention.initialize(
            expanded_states,
            expanded_lengths,
        )
        context = expanded_states.new_zeros(
            batch_size * beam,
            config.attention_dim,
        )
        decoder_hidden: Tensor | None = None
        lm_hidden: tuple[Tensor, Tensor] | None = None
        input_tokens = torch.full(
            (batch_size * beam, ),
            config.bos_token_id,
            dtype=torch.long,
            device=encoder_states.device,
        )
        offsets = torch.arange(
            batch_size,
            device=encoder_states.device,
        ) * beam
        sequence_scores = encoder_states.new_full(
            (batch_size * beam, ),
            float("-inf"),
        )
        sequence_scores[offsets] = 0.0
        alive = torch.empty(
            batch_size * beam,
            0,
            dtype=torch.long,
            device=encoder_states.device,
        )
        completed: list[list[tuple[Tensor, float]]] = [[] for _ in range(batch_size)]
        previous_peaks = torch.zeros(
            batch_size * beam,
            dtype=torch.long,
            device=encoder_states.device,
        )
        coverage: Tensor | None = None
        fallback_scores: Tensor | None = None
        minimum_steps = int(encoder_states.shape[1] * config.minimum_decode_ratio, )
        maximum_steps = max(
            1,
            int(encoder_states.shape[1] * config.maximum_decode_ratio),
        )

        for step in range(maximum_steps):
            if all(len(rows) >= beam for rows in completed):
                break
            embedded = self.model.embedding(input_tokens)
            (
                decoder_output,
                decoder_hidden,
                context,
                attention,
                attention_state,
            ) = self.model.decoder.forward_step(
                embedded,
                decoder_hidden,
                context,
                expanded_states,
                attention_state,
            )
            acoustic = (self.model.sequence_linear(decoder_output) / config.temperature).log_softmax(dim=-1)
            acoustic = acoustic.clone()
            if config.maximum_attention_shift > 0:
                peaks = attention.argmax(dim=-1)
                allowed = (peaks <= previous_peaks + config.maximum_attention_shift) & (
                    peaks > previous_peaks - config.maximum_attention_shift)
                acoustic = acoustic.masked_fill(
                    ~allowed.unsqueeze(1),
                    -1e20,
                )
                previous_peaks = peaks
            if step < minimum_steps:
                acoustic[:, config.eos_token_id] = -1e20
            maximum = acoustic.max(dim=-1).values
            eos_allowed = (acoustic[:, config.eos_token_id] > config.eos_threshold * maximum)
            acoustic[:, config.eos_token_id] = torch.where(
                eos_allowed,
                acoustic[:, config.eos_token_id],
                acoustic.new_full((), -1e20),
            )

            if resolved_lm_weight > 0.0:
                lm_logits, lm_hidden = self.model.language_model(
                    input_tokens,
                    lm_hidden,
                )
                lm_scores = (lm_logits / config.lm_temperature).log_softmax(dim=-1)
                combined = acoustic + resolved_lm_weight * lm_scores
            else:
                combined = acoustic
            vocabulary_size = combined.shape[-1]
            candidate_scores = (sequence_scores.unsqueeze(1) + combined)
            candidate_scores = candidate_scores / (step + 1)
            candidate_scores, candidates = candidate_scores.view(
                batch_size,
                beam * vocabulary_size,
            ).topk(
                beam, dim=-1)
            input_tokens = (candidates % vocabulary_size).reshape(batch_size * beam)
            predecessor = (candidates // vocabulary_size + offsets.unsqueeze(1)).reshape(batch_size * beam)
            sequence_scores = (candidate_scores.reshape(batch_size * beam) * (step + 1))

            decoder_hidden = _select_hidden(
                decoder_hidden,
                predecessor,
            )
            context = context.index_select(0, predecessor)
            attention_state = attention_state.index_select(predecessor)
            lm_hidden = _select_hidden(lm_hidden, predecessor)
            previous_peaks = previous_peaks.index_select(0, predecessor)

            if config.coverage_penalty > 0.0:
                current_attention = attention.index_select(0, predecessor)
                if coverage is None:
                    coverage = current_attention
                else:
                    coverage = (coverage.index_select(0, predecessor) + current_attention)
                penalty = (
                    torch.maximum(
                        coverage,
                        coverage.new_full((), 0.5),
                    ).sum(dim=-1) - coverage.shape[-1] * 0.5)
                candidate_scores = (
                    candidate_scores.reshape(batch_size * beam) -
                    (penalty / (step + 1)) * config.coverage_penalty).reshape(batch_size, beam)
            fallback_scores = candidate_scores.reshape(batch_size * beam)

            alive = torch.cat(
                (
                    alive.index_select(0, predecessor),
                    input_tokens.unsqueeze(1),
                ),
                dim=1,
            )
            is_eos = input_tokens.eq(config.eos_token_id)
            for flat_index in torch.nonzero(
                    is_eos,
                    as_tuple=False,
            ).flatten().tolist():
                batch_index = flat_index // beam
                if len(completed[batch_index]) < beam:
                    completed[batch_index].append((
                        alive[flat_index].clone(),
                        float(candidate_scores.reshape(-1)[flat_index].item()),
                    ))
            sequence_scores.masked_fill_(is_eos, float("-inf"))

        final_scores = (
            sequence_scores / max(1, alive.shape[1]) if fallback_scores is None else fallback_scores)
        for flat_index in range(batch_size * beam):
            batch_index = flat_index // beam
            if len(completed[batch_index]) < beam:
                completed[batch_index].append((
                    alive[flat_index].clone(),
                    float(final_scores[flat_index].item()),
                ))
        token_rows: list[tuple[int, ...]] = []
        scores: list[float] = []
        for rows in completed:
            tokens, score = max(rows, key=lambda item: item[1])
            values = tokens.tolist()
            if config.eos_token_id in values:
                values = values[:values.index(config.eos_token_id)]
            token_rows.append(tuple(int(value) for value in values))
            scores.append(float(score))
        return SpeechBrainBeamResult(
            token_ids=tuple(token_rows),
            scores=tuple(scores),
        )


__all__ = [
    "SpeechBrainBeamResult",
    "SpeechBrainRNNLMBeamSearch",
]
