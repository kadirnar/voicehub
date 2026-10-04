"""Request-local token selection for native generation."""

from __future__ import annotations

from numbers import Integral

import torch
from torch import Tensor

from voicehub.generation.config import GenerationConfig
from voicehub.generation.logits import _process_logits, _raise_processing_error

_TORCH_SEED_MIN = -(2**63)
_TORCH_SEED_MAX = 2**64 - 1


def create_generator(device: torch.device | str, seed: int | None = None) -> torch.Generator:
    """Create an isolated random generator without touching global RNG
    state."""
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, Integral):
            raise TypeError("`seed` must be an integer or None.")
        seed = int(seed)
        if not _TORCH_SEED_MIN <= seed <= _TORCH_SEED_MAX:
            raise ValueError(
                "`seed` must be in PyTorch's supported interval "
                f"[{_TORCH_SEED_MIN}, {_TORCH_SEED_MAX}].")
    generator = torch.Generator(device=torch.device(device))
    if seed is None:
        generator.seed()
    else:
        generator.manual_seed(seed)
    return generator


def sample_next_token(
    logits: Tensor,
    token_ids: Tensor,
    config: GenerationConfig,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Select one token per row according to a validated configuration."""
    next_tokens, processed, violation = _sample_next_token(
        logits,
        token_ids,
        config,
        generator=generator,
    )
    if violation:
        _raise_processing_error(
            logits,
            token_ids,
            processed,
            repetition_penalty=config.repetition_penalty,
        )
    return next_tokens


def _sample_next_token(
    logits: Tensor,
    token_ids: Tensor,
    config: GenerationConfig,
    *,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Select tokens without reading device values.

    Returns the tokens, the processed logits and the deferred validity
    flag of :func:`voicehub.generation.logits._process_logits`; the
    caller must raise through ``_raise_processing_error`` when the flag
    is true. Keeping the flag on the device lets a decoding loop combine
    it with its stop decision into one host synchronization per step.
    """
    processed, violation = _process_logits(
        logits,
        token_ids,
        do_sample=config.do_sample,
        temperature=config.temperature,
        top_k=config.top_k,
        top_p=config.top_p,
        min_p=config.min_p,
        repetition_penalty=config.repetition_penalty,
    )
    if not config.do_sample:
        return processed.argmax(dim=-1), processed, violation
    if generator is None:
        generator = create_generator(processed.device, config.seed)
    elif generator.device != processed.device:
        raise ValueError("The request generator and logits must use the same device.")

    # Valid logits (no NaN/+inf, a finite score in every row) always give
    # finite probabilities with a positive row sum. Until the caller reads
    # the flag, invalid logits draw from a uniform distribution instead, so
    # they surface as a ValueError rather than a device-side multinomial
    # assertion.
    probabilities = torch.softmax(processed.float(), dim=-1).masked_fill(violation, 1.0)
    next_tokens = torch.multinomial(
        probabilities,
        num_samples=1,
        replacement=True,
        generator=generator,
    ).squeeze(-1)
    return next_tokens, processed, violation
