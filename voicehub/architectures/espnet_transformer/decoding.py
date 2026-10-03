"""Joint attention, CTC-prefix, and recurrent-LM beam search."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch
from torch import Tensor

from voicehub.architectures.espnet_transformer.configuration import ESPnetLibriSpeechTransformerConfig
from voicehub.architectures.espnet_transformer.modeling import (
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
        previous_sum = torch.logaddexp(previous_state[:, 0], previous_state[:, 1])
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

    def extend_batch(
        self,
        prefixes: Tensor,
        candidates: Tensor,
        previous_states: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Run :meth:`extend` for equal-length prefixes in one recurrence.

        ``prefixes`` is ``[hypotheses, length]`` (SOS first),
        ``candidates`` is ``[hypotheses, count]`` and
        ``previous_states`` is ``[hypotheses, frames, 2]``.  Returns
        ``[hypotheses, count]`` scores and ``[hypotheses, count, frames,
        2]`` states with the same element-wise arithmetic as
        :meth:`extend`.
        """
        prefixes = torch.as_tensor(prefixes, dtype=torch.long, device=self.values.device)
        candidate_ids = torch.as_tensor(candidates, dtype=torch.long, device=self.values.device)
        if prefixes.ndim != 2 or prefixes.shape[1] == 0:
            raise ValueError("CTC prefixes must have shape [hypotheses, length] and include SOS.")
        if candidate_ids.ndim != 2 or candidate_ids.shape[0] != prefixes.shape[0] or candidate_ids.shape[
                1] == 0:
            raise ValueError("CTC candidates must have shape [hypotheses, count].")
        frames = self.values.shape[0]
        if previous_states.shape != (prefixes.shape[0], frames, 2):
            raise ValueError("CTC states must have shape [hypotheses, frames, 2].")
        hypotheses, count = candidate_ids.shape
        output_length = prefixes.shape[1] - 1
        # [frames, hypotheses, count]
        emissions = self.values.index_select(1, candidate_ids.reshape(-1)).view(frames, hypotheses, count)
        states = self.values.new_full((frames, 2, hypotheses, count), _LOG_ZERO)
        if output_length == 0:
            states[0, 0] = emissions[0]
        previous = previous_states.permute(1, 0, 2)  # [frames, hypotheses, 2]
        previous_sum = torch.logaddexp(previous[:, :, 0], previous[:, :, 1])  # [frames, hypotheses]
        transition = previous_sum.unsqueeze(2).expand(-1, -1, count).clone()
        repeated = candidate_ids == prefixes[:, -1:]
        if output_length > 0 and repeated.any():
            transition = torch.where(
                repeated.unsqueeze(0),
                previous[:, :, 1].unsqueeze(2).expand(-1, -1, count),
                transition,
            )
        start = max(output_length, 1)
        if start >= frames:
            scores = self.values.new_full((hypotheses, count), _LOG_ZERO)
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
            scores = torch.where(eos, previous_sum[-1].unsqueeze(1).expand(-1, count), scores)
        scores = scores.masked_fill(candidate_ids == self.blank_token_id, _LOG_ZERO)
        return scores, states.permute(2, 3, 0, 1)


def _end_detect(
        ended: list[_Hypothesis],
        step: int,
        *,
        window: int = 3,
        threshold: float = math.log(math.exp(-10.0)),
) -> bool:
    """Return whether ESPnet's ``end_detect`` stops the search.

    This is Eq. 50 of Watanabe et al. over the ended hypotheses. Lengths
    count SOS and EOS exactly like ESPnet's ``yseq``.
    """
    if not ended:
        return False
    best = max(value.score for value in ended)
    count = 0
    for offset in range(window):
        same_length = [value.score for value in ended if len(value.tokens) == step - offset]
        if same_length and max(same_length) - best < threshold:
            count += 1
    return count == window


class ESPnetJointBeamSearch:
    """Single-utterance joint scorer matching the published components.

    The search mirrors ``espnet.nets.beam_search.BeamSearch`` at the
    pinned ESPnet 0.8.0 revision as driven by
    ``espnet2/bin/asr_inference.py``: every running hypothesis proposes
    its ``beam_size`` best tokens, the pooled proposals are pruned to
    ``beam_size`` (ended ones included), an EOS-terminated proposal
    leaves the beam, and with ``maximum_decode_ratio == 0`` the search
    runs for at most one step per encoder frame and stops by ESPnet's
    end detection.  A scorer whose weight is zero is not evaluated at
    all, exactly like ESPnet.
    """

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
        config = self.config
        vocabulary = config.vocabulary_size
        eos = config.sos_eos_token_id
        decoder_weight = 1.0 - config.ctc_weight
        use_ctc = config.ctc_weight != 0.0
        use_lm = (self.language_model is not None and config.language_model_weight != 0.0)
        ctc = None
        initial_ctc_state = memory.new_zeros((memory.shape[0], 2))
        if use_ctc:
            ctc_log_probabilities = self.model.ctc.ctc_lo(memory).log_softmax(dim=-1)
            ctc = ESPnetCTCPrefixScorer(
                ctc_log_probabilities,
                blank_token_id=config.blank_token_id,
                eos_token_id=eos,
            )
            initial_ctc_state = ctc.initial_state
        active = [
            _Hypothesis(
                tokens=(config.sos_eos_token_id, ),
                score=0.0,
                ctc_state=initial_ctc_state,
                ctc_score=0.0,
                lm_state=None,
            )
        ]
        ended: list[_Hypothesis] = []
        frames = memory.shape[0]
        end_detection = config.maximum_decode_ratio == 0.0
        maximum_length = (frames if end_detection else max(1, int(config.maximum_decode_ratio * frames)))
        minimum_length = int(config.minimum_decode_ratio * frames)
        # The pinned ESPnet 0.8 inference entrypoint scores CTC over the full
        # vocabulary. A candidate ratio is an explicit approximate decoding
        # optimization for custom deployments, never the release default.
        candidate_count = vocabulary
        if use_ctc and config.ctc_candidate_ratio is not None:
            candidate_count = min(
                candidate_count,
                max(
                    beam_size,
                    int(math.ceil(beam_size * config.ctc_candidate_ratio)),
                ),
            )
        for step in range(maximum_length):
            pool: list[_Hypothesis] = []
            count = len(active)
            prefixes = torch.tensor(
                [hypothesis.tokens for hypothesis in active],
                dtype=torch.long,
                device=memory.device,
            )
            weighted = memory.new_zeros((count, vocabulary))
            if decoder_weight != 0.0:
                weighted = weighted + decoder_weight * self.model.decoder.score_batch(prefixes, memory)
            if config.length_bonus != 0.0:
                weighted = weighted + config.length_bonus
            next_lm_states: list[tuple[Tensor, Tensor] | None] = [None] * count
            if use_lm:
                lm_scores, next_lm_states = self.language_model.score_batch(
                    prefixes[:, -1],
                    [hypothesis.lm_state for hypothesis in active],
                )
                weighted = weighted + config.language_model_weight * lm_scores
            if step < minimum_length:
                weighted[:, eos] = _LOG_ZERO
            ctc_scores = ctc_states = candidate_ids = None
            if ctc is not None:
                if candidate_count < vocabulary:
                    preselection = weighted.clone()
                    preselection[:, config.blank_token_id] = _LOG_ZERO
                    candidate_ids = torch.topk(preselection, candidate_count, dim=-1).indices
                    if step >= minimum_length:
                        missing = ~torch.any(candidate_ids == eos, dim=-1)
                        candidate_ids[missing, -1] = eos
                else:
                    candidate_ids = torch.arange(
                        vocabulary,
                        device=memory.device,
                    ).expand(count, -1)
                ctc_scores, ctc_states = ctc.extend_batch(
                    prefixes,
                    candidate_ids,
                    torch.stack([hypothesis.ctc_state for hypothesis in active]),
                )
                previous_ctc = torch.tensor(
                    [hypothesis.ctc_score for hypothesis in active],
                    dtype=ctc_scores.dtype,
                    device=ctc_scores.device,
                )
                local_ctc = ctc_scores - previous_ctc.unsqueeze(1)
                if candidate_count < vocabulary:
                    # ESPnet's ``beam`` masks tokens pruned before CTC scoring.
                    partial = weighted.gather(1, candidate_ids) + config.ctc_weight * local_ctc
                    weighted = torch.full_like(weighted, float("-inf"))
                    weighted.scatter_(1, candidate_ids, partial)
                else:
                    weighted = weighted + config.ctc_weight * local_ctc
            weighted = weighted + torch.tensor(
                [hypothesis.score for hypothesis in active],
                dtype=weighted.dtype,
                device=weighted.device,
            ).unsqueeze(1)
            top_scores, top_ids = weighted.topk(min(beam_size, vocabulary), dim=-1)
            if candidate_ids is not None and candidate_count < vocabulary:
                positions = (candidate_ids.unsqueeze(1) == top_ids.unsqueeze(2)).int().argmax(dim=2)
            else:
                positions = top_ids
            top_scores_list = top_scores.tolist()
            position_list = positions.tolist()
            top_ids_list = top_ids.tolist()
            selected_ctc = (None if ctc_scores is None else ctc_scores.gather(1, positions).tolist())
            for row, hypothesis in enumerate(active):
                for column, token in enumerate(top_ids_list[row]):
                    pool.append(
                        _Hypothesis(
                            tokens=(*hypothesis.tokens, token),
                            score=top_scores_list[row][column],
                            ctc_state=(
                                hypothesis.ctc_state if ctc_states is None else
                                ctc_states[row, position_list[row][column]].clone()),
                            ctc_score=(
                                hypothesis.ctc_score if selected_ctc is None else selected_ctc[row][column]),
                            lm_state=next_lm_states[row],
                        ))
                # ESPnet prunes the pooled proposals after every hypothesis
                # with a stable sort, so equal scores keep proposal order.
                pool = sorted(
                    pool,
                    key=lambda value: value.score,
                    reverse=True,
                )[:min(len(pool), beam_size)]
            if step == maximum_length - 1:
                # ESPnet appends EOS to every pooled hypothesis at the last
                # step, including one that already proposed EOS.
                pool = [replace(value, tokens=(*value.tokens, eos)) for value in pool]
            active = []
            for hypothesis in pool:
                if hypothesis.tokens[-1] == eos:
                    ended.append(hypothesis)
                else:
                    active.append(hypothesis)
            if end_detection and _end_detect(ended, step):
                break
            if not active:
                break
        candidates = ended or active
        best = max(candidates, key=lambda value: value.score)
        tokens = best.tokens[1:]
        if tokens and tokens[-1] == eos:
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
