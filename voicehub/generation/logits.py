"""Numerically robust, non-mutating logit transformations."""

from __future__ import annotations

import math

import torch
from torch import Tensor

_NON_FINITE_MESSAGE = "`logits` cannot contain NaN or positive infinity."
_NO_CANDIDATE_MESSAGE = "Every logit row must contain at least one finite candidate."
_TOKEN_RANGE_MESSAGE = "`token_ids` contains an ID outside the logits vocabulary."


def _validate_logits_structure(logits: Tensor) -> None:
    if not isinstance(logits, Tensor):
        raise TypeError("`logits` must be a PyTorch tensor.")
    if logits.ndim != 2:
        raise ValueError(f"`logits` must have shape [batch, vocabulary], found {tuple(logits.shape)}.")
    if logits.shape[0] == 0 or logits.shape[1] == 0:
        raise ValueError("`logits` must have a non-empty batch and vocabulary.")
    if not logits.is_floating_point():
        raise TypeError("`logits` must use a floating-point dtype.")


def _logit_value_violation(logits: Tensor) -> Tensor:
    """Return a device-side flag that is true when logit values are invalid.

    Fusing every value check into one scalar lets callers pay a single
    host synchronization for it, or defer the read entirely.
    """
    return ((torch.isnan(logits) | torch.isposinf(logits)).any() | ~torch.isfinite(logits).any(dim=-1).all())


def _raise_for_logit_values(logits: Tensor) -> None:
    """Raise the precise value error; only reached after a fused check
    failed."""
    if torch.isnan(logits).any() or torch.isposinf(logits).any():
        raise ValueError(_NON_FINITE_MESSAGE)
    if not torch.isfinite(logits).any(dim=-1).all():
        raise ValueError(_NO_CANDIDATE_MESSAGE)


def _validate_logits(logits: Tensor) -> None:
    _validate_logits_structure(logits)
    if _logit_value_violation(logits):
        _raise_for_logit_values(logits)


def apply_repetition_penalty(logits: Tensor, token_ids: Tensor, penalty: float) -> Tensor:
    """Apply the sign-aware repetition penalty used by autoregressive models.

    Positive scores are divided by ``penalty`` and negative scores are
    multiplied by it. The input tensor is never changed.
    """
    _validate_logits(logits)
    result, out_of_range = _apply_repetition_penalty(logits, token_ids, penalty)
    if out_of_range is not None and out_of_range:
        raise ValueError(_TOKEN_RANGE_MESSAGE)
    return result


def _apply_repetition_penalty(
    logits: Tensor,
    token_ids: Tensor,
    penalty: float,
) -> tuple[Tensor, Tensor | None]:
    """Penalize without reading device values.

    Returns the result and a device-side flag for token IDs outside the
    vocabulary (``None`` when no ID was used). Invalid IDs are clamped
    for the lookup, so deferring that check can never trigger a device-
    side indexing assertion.
    """
    _validate_logits_structure(logits)
    if not isinstance(penalty, (int, float)) or isinstance(penalty, bool):
        raise TypeError("`penalty` must be a real number.")
    penalty = float(penalty)
    if not math.isfinite(penalty) or penalty <= 0.0:
        raise ValueError("`penalty` must be finite and greater than zero.")
    if not isinstance(token_ids, Tensor):
        raise TypeError("`token_ids` must be a PyTorch tensor.")
    if token_ids.ndim != 2 or token_ids.shape[0] != logits.shape[0]:
        raise ValueError("`token_ids` must have shape [batch, sequence] and share the logits batch size.")
    if token_ids.numel() == 0 or penalty == 1.0:
        return logits.clone(), None
    if token_ids.dtype == torch.bool or token_ids.is_floating_point() or token_ids.is_complex():
        raise TypeError("`token_ids` must use an integer dtype.")
    if token_ids.device != logits.device:
        raise ValueError("`token_ids` and `logits` must be on the same device.")
    vocabulary_size = logits.shape[-1]
    index = token_ids.long()
    out_of_range = ((index < 0) | (index >= vocabulary_size)).any()
    index = index.clamp(0, vocabulary_size - 1)

    adjusted = logits.gather(dim=-1, index=index)
    adjusted = torch.where(adjusted < 0, adjusted * penalty, adjusted / penalty)
    result = logits.clone()
    result.scatter_(dim=-1, index=index, src=adjusted)
    return result, out_of_range


def filter_top_k(logits: Tensor, top_k: int, *, min_tokens_to_keep: int = 1) -> Tensor:
    """Mask every token outside the ``top_k`` candidates in each row."""
    _validate_logits(logits)
    return _filter_top_k(logits, top_k, min_tokens_to_keep=min_tokens_to_keep)


def _filter_top_k(logits: Tensor, top_k: int, *, min_tokens_to_keep: int = 1) -> Tensor:
    _validate_logits_structure(logits)
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError("`top_k` must be an integer.")
    if isinstance(min_tokens_to_keep, bool) or not isinstance(min_tokens_to_keep, int):
        raise TypeError("`min_tokens_to_keep` must be an integer.")
    if top_k < 0:
        raise ValueError("`top_k` cannot be negative.")
    if min_tokens_to_keep < 1:
        raise ValueError("`min_tokens_to_keep` must be greater than zero.")

    vocabulary_size = logits.shape[-1]
    candidates = min(vocabulary_size, max(top_k, min_tokens_to_keep))
    if candidates >= vocabulary_size:
        return logits.clone()
    threshold = torch.topk(logits, candidates, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))


def filter_top_p(logits: Tensor, top_p: float, *, min_tokens_to_keep: int = 1) -> Tensor:
    """Apply nucleus filtering while always retaining the highest score."""
    _validate_logits(logits)
    return _filter_top_p(logits, top_p, min_tokens_to_keep=min_tokens_to_keep)


def _filter_top_p(logits: Tensor, top_p: float, *, min_tokens_to_keep: int = 1) -> Tensor:
    _validate_logits_structure(logits)
    if not isinstance(top_p, (int, float)) or isinstance(top_p, bool):
        raise TypeError("`top_p` must be a real number.")
    top_p = float(top_p)
    if not math.isfinite(top_p) or not 0.0 <= top_p <= 1.0:
        raise ValueError("`top_p` must be finite and in the interval [0, 1].")
    if isinstance(min_tokens_to_keep, bool) or not isinstance(min_tokens_to_keep, int):
        raise TypeError("`min_tokens_to_keep` must be an integer.")
    if min_tokens_to_keep < 1:
        raise ValueError("`min_tokens_to_keep` must be greater than zero.")
    if top_p == 1.0:
        return logits.clone()

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    probabilities = torch.softmax(sorted_logits.float(), dim=-1)
    remove_sorted = probabilities.cumsum(dim=-1) > top_p
    remove_sorted[..., 1:] = remove_sorted[..., :-1].clone()
    remove_sorted[..., :min_tokens_to_keep] = False

    remove = torch.zeros_like(remove_sorted)
    remove.scatter_(dim=-1, index=sorted_indices, src=remove_sorted)
    return logits.masked_fill(remove, float("-inf"))


def filter_min_p(logits: Tensor, min_p: float, *, min_tokens_to_keep: int = 1) -> Tensor:
    """Mask tokens whose probability is too small relative to the row
    maximum."""
    _validate_logits(logits)
    return _filter_min_p(logits, min_p, min_tokens_to_keep=min_tokens_to_keep)


def _filter_min_p(logits: Tensor, min_p: float, *, min_tokens_to_keep: int = 1) -> Tensor:
    _validate_logits_structure(logits)
    if not isinstance(min_p, (int, float)) or isinstance(min_p, bool):
        raise TypeError("`min_p` must be a real number.")
    min_p = float(min_p)
    if not math.isfinite(min_p) or not 0.0 <= min_p <= 1.0:
        raise ValueError("`min_p` must be finite and in the interval [0, 1].")
    if isinstance(min_tokens_to_keep, bool) or not isinstance(min_tokens_to_keep, int):
        raise TypeError("`min_tokens_to_keep` must be an integer.")
    if min_tokens_to_keep < 1:
        raise ValueError("`min_tokens_to_keep` must be greater than zero.")
    if min_p == 0.0:
        return logits.clone()

    probabilities = torch.softmax(logits.float(), dim=-1)
    threshold = probabilities.amax(dim=-1, keepdim=True) * min_p
    remove = probabilities < threshold
    if min_tokens_to_keep > 0:
        keep_indices = torch.topk(
            logits,
            min(min_tokens_to_keep, logits.shape[-1]),
            dim=-1,
        ).indices
        remove.scatter_(dim=-1, index=keep_indices, value=False)
    return logits.masked_fill(remove, float("-inf"))


def process_logits(
    logits: Tensor,
    token_ids: Tensor,
    *,
    do_sample: bool,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    min_p: float | None = None,
    repetition_penalty: float = 1.0,
) -> Tensor:
    """Apply native generation transforms in a stable, documented order."""
    result, violation = _process_logits(
        logits,
        token_ids,
        do_sample=do_sample,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        min_p=min_p,
        repetition_penalty=repetition_penalty,
    )
    if violation:
        _raise_processing_error(logits, token_ids, result, repetition_penalty=repetition_penalty)
    return result


def _process_logits(
    logits: Tensor,
    token_ids: Tensor,
    *,
    do_sample: bool,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    min_p: float | None = None,
    repetition_penalty: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """Run :func:`process_logits` and return a deferred validity flag.

    No transform can hide an invalid value: NaN and +inf survive the
    repetition penalty, temperature and every filter (none of them masks
    such a score), and a row without a finite candidate keeps none.
    Checking the final logits once is therefore equivalent to checking
    every intermediate, and the caller decides when to read the flag.
    Pass a true flag to :func:`_raise_processing_error`.
    """
    if not isinstance(do_sample, bool):
        raise TypeError("`do_sample` must be a boolean.")
    result, out_of_range = _apply_repetition_penalty(logits, token_ids, repetition_penalty)
    if do_sample:
        if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
            raise TypeError("`temperature` must be a real number.")
        temperature = float(temperature)
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("`temperature` must be finite and greater than zero.")
        result = result / temperature
        if top_k is not None and top_k > 0:
            result = _filter_top_k(result, top_k)
        if top_p is not None and top_p < 1.0:
            result = _filter_top_p(result, top_p)
        if min_p is not None and min_p > 0.0:
            result = _filter_min_p(result, min_p)
    violation = _logit_value_violation(result)
    if out_of_range is not None:
        violation = violation | out_of_range
    return result, violation


def _raise_processing_error(
    logits: Tensor,
    token_ids: Tensor,
    processed: Tensor,
    *,
    repetition_penalty: float,
) -> None:
    """Raise the error the eager checks would have raised, in their order."""
    _raise_for_logit_values(logits)
    if float(repetition_penalty) != 1.0 and token_ids.numel():
        if ((token_ids < 0) | (token_ids >= logits.shape[-1])).any():
            raise ValueError(_TOKEN_RANGE_MESSAGE)
    _raise_for_logit_values(processed)
    raise RuntimeError("Generation logits failed validation.")
