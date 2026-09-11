"""Native CTC prefix search and U2++ attention rescoring."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from voicehub.models.asr_wenet.native.modeling import WeNetU2PPForASR


def _log_add(*values: float) -> float:
    if all(value == -float("inf") for value in values):
        return -float("inf")
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


@dataclass(frozen=True)
class WeNetDecodeHypothesis:
    token_ids: tuple[int, ...]
    score: float
    token_frames: tuple[int, ...] = ()
    confidence: float | None = None
    token_confidences: tuple[float, ...] = ()


@dataclass
class _PrefixScore:
    blank: float = -float("inf")
    nonblank: float = -float("inf")
    viterbi_blank: float = -float("inf")
    viterbi_nonblank: float = -float("inf")
    blank_frames: tuple[int, ...] = ()
    nonblank_frames: tuple[int, ...] = ()
    current_probability: float = -float("inf")

    @property
    def total(self) -> float:
        return _log_add(self.blank, self.nonblank)

    @property
    def viterbi(self) -> float:
        return max(self.viterbi_blank, self.viterbi_nonblank)

    @property
    def frames(self) -> tuple[int, ...]:
        return (self.blank_frames if self.viterbi_blank > self.viterbi_nonblank else self.nonblank_frames)


def ctc_greedy_decode(
    log_probabilities: Tensor,
    lengths: Tensor,
    *,
    blank_token_id: int = 0,
) -> tuple[WeNetDecodeHypothesis, ...]:
    results = []
    for batch_index, length in enumerate(lengths.tolist()):
        row = log_probabilities[batch_index, :int(length)]
        scores, token_ids = row.max(dim=-1)
        collapsed: list[int] = []
        frames: list[int] = []
        previous = None
        for frame, token_id in enumerate(token_ids.tolist()):
            if token_id != blank_token_id and token_id != previous:
                collapsed.append(token_id)
                frames.append(frame)
            previous = token_id
        results.append(
            WeNetDecodeHypothesis(
                token_ids=tuple(collapsed),
                score=float(scores.sum().item()),
                token_frames=tuple(frames),
            ))
    return tuple(results)


def ctc_prefix_beam_search(
    log_probabilities: Tensor,
    lengths: Tensor,
    *,
    beam_size: int,
    blank_token_id: int = 0,
) -> tuple[tuple[WeNetDecodeHypothesis, ...], ...]:
    if isinstance(beam_size, bool) or not isinstance(beam_size, int):
        raise TypeError("`beam_size` must be an integer.")
    if beam_size <= 0:
        raise ValueError("`beam_size` must be positive.")
    batch_results = []
    for batch_index, length in enumerate(lengths.tolist()):
        probabilities = log_probabilities[batch_index, :int(length)]
        current: list[tuple[tuple[int, ...], _PrefixScore]] = [(
            (),
            _PrefixScore(
                blank=0.0,
                viterbi_blank=0.0,
                viterbi_nonblank=0.0,
            ),
        )]
        for frame, log_probability in enumerate(probabilities):
            next_scores: defaultdict[tuple[int, ...], _PrefixScore] = (defaultdict(_PrefixScore))
            top_count = min(beam_size, log_probability.numel())
            for token_tensor in log_probability.topk(top_count).indices:
                token = int(token_tensor.item())
                probability = float(log_probability[token].item())
                for prefix, source in current:
                    last = prefix[-1] if prefix else None
                    if token == blank_token_id:
                        target = next_scores[prefix]
                        target.blank = _log_add(
                            target.blank,
                            source.total + probability,
                        )
                        candidate = source.viterbi + probability
                        if candidate > target.viterbi_blank:
                            target.viterbi_blank = candidate
                            target.blank_frames = source.frames
                    elif token == last:
                        unchanged = next_scores[prefix]
                        unchanged.nonblank = _log_add(
                            unchanged.nonblank,
                            source.nonblank + probability,
                        )
                        candidate = source.viterbi_nonblank + probability
                        if candidate > unchanged.viterbi_nonblank:
                            unchanged.viterbi_nonblank = candidate
                            frames = list(source.nonblank_frames)
                            if probability > unchanged.current_probability:
                                unchanged.current_probability = probability
                                if frames:
                                    frames[-1] = frame
                            unchanged.nonblank_frames = tuple(frames)
                        extended = next_scores[prefix + (token, )]
                        extended.nonblank = _log_add(
                            extended.nonblank,
                            source.blank + probability,
                        )
                        candidate = source.viterbi_blank + probability
                        if candidate > extended.viterbi_nonblank:
                            extended.viterbi_nonblank = candidate
                            extended.current_probability = probability
                            extended.nonblank_frames = (source.blank_frames + (frame, ))
                    else:
                        extended = next_scores[prefix + (token, )]
                        extended.nonblank = _log_add(
                            extended.nonblank,
                            source.total + probability,
                        )
                        candidate = source.viterbi + probability
                        if candidate > extended.viterbi_nonblank:
                            extended.viterbi_nonblank = candidate
                            extended.current_probability = probability
                            extended.nonblank_frames = source.frames + (frame, )
            current = sorted(
                next_scores.items(),
                key=lambda item: item[1].total,
                reverse=True,
            )[:beam_size]
        batch_results.append(
            tuple(
                WeNetDecodeHypothesis(
                    token_ids=prefix,
                    score=score.total,
                    token_frames=score.frames,
                ) for prefix, score in current))
    return tuple(batch_results)


def attention_rescore(
    model: WeNetU2PPForASR,
    nbest: tuple[WeNetDecodeHypothesis, ...],
    encoder_output: Tensor,
    *,
    ctc_weight: float = 0.3,
    reverse_weight: float = 0.3,
) -> WeNetDecodeHypothesis:
    if not nbest:
        return WeNetDecodeHypothesis((), 0.0)
    device = encoder_output.device
    hypotheses = [torch.tensor(item.token_ids, dtype=torch.long, device=device) for item in nbest]
    padded = pad_sequence(
        hypotheses,
        batch_first=True,
        padding_value=model.eos,
    )
    sos = padded.new_full((padded.size(0), 1), model.sos)
    decoder_input = torch.cat((sos, padded), dim=1)
    hypothesis_lengths = torch.tensor(
        [len(item.token_ids) + 1 for item in nbest],
        dtype=torch.long,
        device=device,
    )
    forward, reverse = model.forward_attention_decoder(
        decoder_input,
        hypothesis_lengths,
        encoder_output,
        reverse_weight,
    )
    best_index = 0
    best_score = -float("inf")
    confidences: list[float] = []
    token_confidences: list[tuple[float, ...]] = []
    for index, hypothesis in enumerate(nbest):
        forward_score = forward.new_zeros(())
        confidence_values = []
        for offset, token_id in enumerate(hypothesis.token_ids):
            value = forward[index, offset, token_id]
            forward_score = forward_score + value
            confidence_values.append(math.exp(float(value.item())))
        forward_score = (forward_score + forward[index, len(hypothesis.token_ids), model.eos])
        score = forward_score
        if reverse_weight > 0.0:
            reverse_score = reverse.new_zeros(())
            for offset, token_id in enumerate(hypothesis.token_ids):
                reverse_offset = len(hypothesis.token_ids) - offset - 1
                value = reverse[index, reverse_offset, token_id]
                reverse_score = reverse_score + value
                confidence_values[offset] = (confidence_values[offset] + math.exp(float(value.item()))) / 2.0
            reverse_score = (reverse_score + reverse[index, len(hypothesis.token_ids), model.eos])
            score = ((1.0 - reverse_weight) * forward_score + reverse_weight * reverse_score)
        confidence = math.exp(float(score.item()) / (len(hypothesis.token_ids) + 1))
        score = score + ctc_weight * hypothesis.score
        numeric_score = float(score.item())
        confidences.append(confidence)
        token_confidences.append(tuple(confidence_values))
        if numeric_score > best_score:
            best_score = numeric_score
            best_index = index
    selected = nbest[best_index]
    return WeNetDecodeHypothesis(
        token_ids=selected.token_ids,
        score=best_score,
        token_frames=selected.token_frames,
        confidence=confidences[best_index],
        token_confidences=token_confidences[best_index],
    )


__all__ = [
    "WeNetDecodeHypothesis",
    "attention_rescore",
    "ctc_greedy_decode",
    "ctc_prefix_beam_search",
]
