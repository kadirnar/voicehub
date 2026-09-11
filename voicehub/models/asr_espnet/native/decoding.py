"""Joint attention, CTC-prefix, and recurrent-LM beam search."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from voicehub.models.asr_espnet.native.configuration import ESPnetLibriSpeechTransformerConfig
from voicehub.models.asr_espnet.native.modeling import (
    ESPnetLibriSpeechTransformerForASR,
    ESPnetSequentialRNNLanguageModel,
)

_LOG_ZERO = -1.0e10


@dataclass(slots=True)
class ESPnetDecodedBatch:
    """Best token sequence and score for every batch row."""

    token_ids: tuple[tuple[int, ...], ...]
    scores: tuple[float, ...]


@dataclass(slots=True)
class _Hypothesis:
    tokens: tuple[int, ...]
    score: float
    ctc_state: Tensor
    ctc_score: float
    lm_state: tuple[Tensor, Tensor] | None


class ESPnetCTCPrefixScorer:
    """Torch port of ESPnet's Algorithm-2 CTC prefix recurrence."""

    def __init__(
        self,
        log_probabilities: Tensor,
        *,
        blank_token_id: int,
        eos_token_id: int,
    ) -> None:
        if log_probabilities.ndim != 2 or log_probabilities.shape[0] < 1:
            raise ValueError("CTC probabilities must have shape [frames, tokens].")
        self.values = log_probabilities.float()
        self.blank_token_id = blank_token_id
        self.eos_token_id = eos_token_id
        self.initial_state = self.values.new_full(
            (self.values.shape[0], 2),
            _LOG_ZERO,
        )
        self.initial_state[0, 1] = self.values[0, blank_token_id]
        for index in range(1, self.values.shape[0]):
            self.initial_state[index,
                               1] = (self.initial_state[index - 1, 1] + self.values[index, blank_token_id])

    def extend(
        self,
        prefix: tuple[int, ...],
        candidates: Tensor,
        previous_state: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if not prefix:
            raise ValueError("CTC prefixes must include SOS.")
        candidate_ids = torch.as_tensor(
            candidates,
            dtype=torch.long,
            device=self.values.device,
        )
        if candidate_ids.ndim != 1 or candidate_ids.numel() == 0:
            raise ValueError("CTC candidates must be a non-empty token vector.")
        output_length = len(prefix) - 1
        count = candidate_ids.numel()
        frames = self.values.shape[0]
        states = self.values.new_full((frames, 2, count), _LOG_ZERO)
        emissions = self.values.index_select(1, candidate_ids)
        if output_length == 0:
            states[0, 0] = emissions[0]
        elif output_length - 1 < frames:
            states[output_length - 1] = _LOG_ZERO
        previous_sum = torch.logsumexp(previous_state, dim=1)
        transition = previous_sum.unsqueeze(1).expand(-1, count).clone()
        repeated = candidate_ids == prefix[-1]
        if repeated.any():
            transition[:, repeated] = previous_state[:, 1].unsqueeze(1)
        start = max(output_length, 1)
        if start >= frames:
            scores = self.values.new_full((count, ), _LOG_ZERO)
        else:
            scores = states[start - 1, 0].clone()
            blank = self.values[:, self.blank_token_id]
            for frame in range(start, frames):
                states[frame, 0] = (
                    torch.logaddexp(
                        states[frame - 1, 0],
                        transition[frame - 1],
                    ) + emissions[frame])
                states[frame,
                       1] = (torch.logaddexp(
                           states[frame - 1, 0],
                           states[frame - 1, 1],
                       ) + blank[frame])
                scores = torch.logaddexp(
                    scores,
                    transition[frame - 1] + emissions[frame],
                )
        eos = candidate_ids == self.eos_token_id
        if eos.any():
            scores[eos] = previous_sum[-1]
        scores[candidate_ids == self.blank_token_id] = _LOG_ZERO
        return scores, states.permute(2, 0, 1).contiguous()


class ESPnetJointBeamSearch:
    """Single-utterance joint scorer matching the published components."""

    def __init__(
        self,
        model: ESPnetLibriSpeechTransformerForASR,
        config: ESPnetLibriSpeechTransformerConfig,
        *,
        language_model: ESPnetSequentialRNNLanguageModel | None = None,
    ) -> None:
        if not isinstance(model, ESPnetLibriSpeechTransformerForASR):
            raise TypeError("`model` must be the native ESPnet ASR graph.")
        if (language_model is not None and not isinstance(language_model, ESPnetSequentialRNNLanguageModel)):
            raise TypeError("`language_model` must be the native ESPnet RNNLM.")
        self.model = model
        self.config = ESPnetLibriSpeechTransformerConfig.coerce(config)
        self.language_model = language_model

    def _decode_one(
        self,
        memory: Tensor,
        *,
        beam_size: int,
    ) -> tuple[tuple[int, ...], float]:
        ctc_log_probabilities = self.model.ctc.ctc_lo(memory).log_softmax(dim=-1)
        ctc = ESPnetCTCPrefixScorer(
            ctc_log_probabilities,
            blank_token_id=self.config.blank_token_id,
            eos_token_id=self.config.sos_eos_token_id,
        )
        initial = _Hypothesis(
            tokens=(self.config.sos_eos_token_id, ),
            score=0.0,
            ctc_state=ctc.initial_state,
            ctc_score=0.0,
            lm_state=None,
        )
        active = [initial]
        ended: list[_Hypothesis] = []
        minimum_length = max(
            0,
            int(memory.shape[0] * self.config.minimum_decode_ratio),
        )
        maximum_length = max(
            minimum_length + 1,
            int(math.ceil(memory.shape[0] * self.config.maximum_decode_ratio)),
        )
        # The pinned ESPnet 0.8 inference entrypoint scores CTC over the full
        # vocabulary. A candidate ratio is an explicit approximate decoding
        # optimization for custom deployments, never the release default.
        candidate_count = self.config.vocabulary_size
        if self.config.ctc_candidate_ratio is not None:
            candidate_count = min(
                candidate_count,
                max(
                    beam_size,
                    int(math.ceil(beam_size * self.config.ctc_candidate_ratio)),
                ),
            )
        for step in range(maximum_length):
            expanded: list[_Hypothesis] = []
            for hypothesis in active:
                prefix = torch.tensor(
                    hypothesis.tokens,
                    dtype=torch.long,
                    device=memory.device,
                )
                attention_scores = self.model.decoder.score(prefix, memory)
                preselection = ((1.0 - self.config.ctc_weight) * attention_scores)
                next_lm_state = hypothesis.lm_state
                lm_scores = attention_scores.new_zeros(attention_scores.shape)
                if self.language_model is not None:
                    lm_values, next_lm_state = self.language_model.score(
                        prefix[-1],
                        hypothesis.lm_state,
                    )
                    lm_scores = lm_values.squeeze(0)
                    preselection = (preselection + self.config.language_model_weight * lm_scores)
                preselection = preselection.clone()
                preselection[self.config.blank_token_id] = _LOG_ZERO
                if step < minimum_length:
                    preselection[self.config.sos_eos_token_id] = _LOG_ZERO
                candidate_ids = torch.topk(
                    preselection,
                    candidate_count,
                ).indices
                if (step >= minimum_length and not torch.any(candidate_ids == self.config.sos_eos_token_id)):
                    candidate_ids[-1] = self.config.sos_eos_token_id
                ctc_scores, ctc_states = ctc.extend(
                    hypothesis.tokens,
                    candidate_ids,
                    hypothesis.ctc_state,
                )
                local_ctc = ctc_scores - hypothesis.ctc_score
                local_scores = (
                    (1.0 - self.config.ctc_weight) * attention_scores.index_select(0, candidate_ids) +
                    self.config.ctc_weight * local_ctc +
                    self.config.language_model_weight * lm_scores.index_select(0, candidate_ids) +
                    self.config.length_bonus)
                for index, token in enumerate(candidate_ids.tolist()):
                    child = _Hypothesis(
                        tokens=(*hypothesis.tokens, token),
                        score=(hypothesis.score + float(local_scores[index].item())),
                        ctc_state=ctc_states[index],
                        ctc_score=float(ctc_scores[index].item()),
                        lm_state=next_lm_state,
                    )
                    if token == self.config.sos_eos_token_id:
                        ended.append(child)
                    else:
                        expanded.append(child)
            if not expanded:
                break
            active = sorted(
                expanded,
                key=lambda value: value.score,
                reverse=True,
            )[:beam_size]
            ended = sorted(
                ended,
                key=lambda value: value.score,
                reverse=True,
            )[:beam_size]
        candidates = ended or active
        best = max(candidates, key=lambda value: value.score)
        tokens = best.tokens[1:]
        if tokens and tokens[-1] == self.config.sos_eos_token_id:
            tokens = tokens[:-1]
        return tokens, best.score

    def __call__(
        self,
        encoder_states: Tensor,
        encoder_lengths: Tensor,
        *,
        beam_size: int | None = None,
    ) -> ESPnetDecodedBatch:
        if encoder_states.ndim != 3:
            raise ValueError("Encoder states must have shape [batch, frames, hidden].")
        lengths = torch.as_tensor(
            encoder_lengths,
            dtype=torch.long,
            device=encoder_states.device,
        )
        if lengths.ndim != 1 or lengths.shape[0] != encoder_states.shape[0]:
            raise ValueError("Encoder lengths must have shape [batch].")
        resolved_beam = self.config.beam_size if beam_size is None else beam_size
        if (isinstance(resolved_beam, bool) or not isinstance(resolved_beam, int) or resolved_beam < 1):
            raise ValueError("Beam size must be a positive integer.")
        sequences = []
        scores = []
        for index, length in enumerate(lengths):
            tokens, score = self._decode_one(
                encoder_states[index, :int(length.item())],
                beam_size=resolved_beam,
            )
            sequences.append(tokens)
            scores.append(score)
        return ESPnetDecodedBatch(
            token_ids=tuple(sequences),
            scores=tuple(scores),
        )


__all__ = [
    "ESPnetCTCPrefixScorer",
    "ESPnetDecodedBatch",
    "ESPnetJointBeamSearch",
]
