"""Exact Fish S2 Dual-AR sampling and repetition-aware fallback."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from voicehub.models.fishtts.native.modeling import FishS2ForConditionalGeneration

REPETITION_WINDOW = 10
REPETITION_FALLBACK_TEMPERATURE = 1.0
REPETITION_FALLBACK_TOP_P = 0.9


def _validate_sampling(
    *,
    temperature: float,
    top_p: float,
    top_k: int,
) -> None:
    if (isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or
            not math.isfinite(float(temperature)) or not 0.0 < float(temperature) < 2.0):
        raise ValueError("Fish temperature must be finite and in (0, 2).")
    if (isinstance(top_p, bool) or not isinstance(top_p, (int, float)) or not math.isfinite(float(top_p)) or
            not 0.0 < float(top_p) <= 1.0):
        raise ValueError("Fish top-p must be finite and in (0, 1].")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("Fish top-k must be a positive integer.")


def logits_to_probabilities(
    logits: Tensor,
    *,
    temperature: float,
    top_p: float,
    top_k: int,
) -> Tensor:
    """Apply Fish's simultaneous top-k and nucleus truncation."""
    _validate_sampling(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    if logits.ndim != 1:
        raise ValueError("Fish sampling logits must be rank one.")
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative = torch.cumsum(
        torch.softmax(sorted_logits, dim=-1),
        dim=-1,
    )
    ranks = torch.arange(logits.shape[-1], device=logits.device)
    remove_sorted = (cumulative > float(top_p)) | (ranks >= top_k)
    remove_sorted[0] = False
    remove = torch.zeros_like(remove_sorted).scatter(
        dim=-1,
        index=sorted_indices,
        src=remove_sorted,
    )
    filtered = logits.masked_fill(remove, float("-inf"))
    return torch.softmax(
        filtered / max(float(temperature), 1e-5),
        dim=-1,
    )


def sample_exponential_race(probabilities: Tensor) -> Tensor:
    """Fish's synchronization-free categorical sample."""
    if probabilities.ndim != 1:
        raise ValueError("Fish probabilities must be rank one.")
    if not torch.isfinite(probabilities).all() or probabilities.sum() <= 0:
        raise ValueError("Fish probabilities must be finite and non-empty.")
    exponential = -torch.log(torch.rand_like(probabilities))
    return torch.argmax(probabilities / exponential).to(dtype=torch.long)


def sample_logits(
    logits: Tensor,
    *,
    temperature: float,
    top_p: float,
    top_k: int,
) -> Tensor:
    return sample_exponential_race(
        logits_to_probabilities(
            logits,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        ))


def _sample_main_token(
    logits: Tensor,
    *,
    model: FishS2ForConditionalGeneration,
    previous_tokens: Tensor,
    temperature: float,
    top_p: float,
    top_k: int,
) -> Tensor:
    constrained = torch.full_like(logits, float("-inf"))
    constrained[model.config.semantic_begin_id:model.config.semantic_end_id +
                1] = logits[model.config.semantic_begin_id:model.config.semantic_end_id + 1]
    constrained[model.config.im_end_id] = logits[model.config.im_end_id]
    normal = sample_logits(
        constrained,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    # The source recipe always consumes the second random draw, even when
    # repetition fallback is not selected.
    fallback = sample_logits(
        constrained,
        temperature=REPETITION_FALLBACK_TEMPERATURE,
        top_p=REPETITION_FALLBACK_TOP_P,
        top_k=top_k,
    )
    repeated = previous_tokens.eq(normal).any()
    semantic = normal.ge(model.config.semantic_begin_id) & normal.le(model.config.semantic_end_id)
    return torch.where(repeated & semantic, fallback, normal)


def _sample_codebooks(
    slow_hidden: Tensor,
    semantic_token: Tensor,
    *,
    model: FishS2ForConditionalGeneration,
    temperature: float,
    top_p: float,
    top_k: int,
) -> Tensor:
    model.reset_fast_caches()
    position = torch.tensor(
        [0],
        device=slow_hidden.device,
        dtype=torch.long,
    )
    model.forward_generate_fast(slow_hidden, position)
    first_code = (semantic_token - model.config.semantic_begin_id).clamp(
        min=0, max=model.config.codebook_size - 1)
    codes = [first_code]
    hidden = model.fast_embeddings(first_code.view(1, 1))
    for codebook_index in range(1, model.config.num_codebooks):
        position = torch.tensor(
            [codebook_index],
            device=slow_hidden.device,
            dtype=torch.long,
        )
        logits = model.forward_generate_fast(hidden, position)[0, -1]
        code = sample_logits(
            logits,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        codes.append(code)
        hidden = model.fast_embeddings(code.view(1, 1))
    return torch.stack(codes)


@torch.inference_mode()
def generate_fish_codes(
    model: FishS2ForConditionalGeneration,
    prompt: Tensor,
    *,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: int = 30,
) -> Tensor:
    """Continue a Fish conversation and return ``[codebook, frame]`` IDs."""
    if not isinstance(model, FishS2ForConditionalGeneration):
        raise TypeError("`model` must be a native Fish semantic model.")
    if (prompt.ndim != 2 or prompt.shape[0] != model.config.num_codebooks + 1):
        raise ValueError("Fish prompt must have shape [num_codebooks + 1, time].")
    if (isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens <= 0):
        raise ValueError("`max_new_tokens` must be a positive integer.")
    _validate_sampling(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    prompt_length = prompt.shape[-1]
    maximum = model.max_seq_len - prompt_length
    if maximum <= 0:
        raise ValueError("Fish prompt reaches the model context limit.")
    steps = min(max_new_tokens, maximum)
    parameter = next(model.parameters())
    prompt = prompt.to(device=parameter.device, dtype=torch.long)
    if (model.max_batch_size < 1 or model.max_sequence_length < model.max_seq_len):
        model.setup_caches(
            max_batch_size=1,
            max_seq_len=model.max_seq_len,
            dtype=parameter.dtype,
        )
    previous = torch.zeros(
        REPETITION_WINDOW,
        dtype=torch.long,
        device=parameter.device,
    )
    generated: list[Tensor] = []
    current = prompt.unsqueeze(0)
    positions = torch.arange(
        prompt_length,
        device=parameter.device,
        dtype=torch.long,
    )
    for _ in range(steps):
        slow = model.forward_generate(current, positions)
        main = _sample_main_token(
            slow.logits[0, -1],
            model=model,
            previous_tokens=previous,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        if int(main.item()) == model.config.im_end_id:
            break
        if not (model.config.semantic_begin_id <= int(main.item()) <= model.config.semantic_end_id):
            raise RuntimeError("Fish constrained sampling produced a non-semantic token.")
        codes = _sample_codebooks(
            slow.hidden_states[:, -1:],
            main,
            model=model,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        generated.append(codes)
        previous = previous.roll(-1)
        previous[-1] = main
        next_token = torch.cat((main.view(1), codes)).view(
            1,
            model.config.num_codebooks + 1,
            1,
        )
        current = next_token
        positions = torch.tensor(
            [prompt_length + len(generated) - 1],
            device=parameter.device,
            dtype=torch.long,
        )
    if not generated:
        raise RuntimeError("Fish generation returned no semantic audio frames.")
    return torch.stack(generated, dim=1)


__all__ = [
    "REPETITION_FALLBACK_TEMPERATURE",
    "REPETITION_FALLBACK_TOP_P",
    "REPETITION_WINDOW",
    "generate_fish_codes",
    "logits_to_probabilities",
    "sample_exponential_race",
    "sample_logits",
]
