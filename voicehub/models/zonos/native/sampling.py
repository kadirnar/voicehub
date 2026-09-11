"""Autoregressive delayed-codebook sampling for native Zonos v0.1."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from voicehub.models.zonos.native.modeling import ZonosForCausalLM
from voicehub.models.zonos.native.pattern import apply_delay_pattern, revert_delay_pattern


@dataclass(frozen=True, slots=True)
class ZonosSamplingOptions:
    """Validated generation controls matching the released sampler."""

    max_new_tokens: int = 86 * 30
    cfg_scale: float = 2.0
    temperature: float = 1.0
    top_p: float = 0.0
    top_k: int = 0
    min_p: float = 0.1
    linear: float = 0.0
    confidence: float = 0.0
    quadratic: float = 0.0
    repetition_penalty: float = 3.0
    repetition_penalty_window: int = 2

    def __post_init__(self) -> None:
        for name in ("max_new_tokens", "repetition_penalty_window"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"`{name}` must be a positive integer.")
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or self.top_k < 0:
            raise ValueError("`top_k` must be a non-negative integer.")
        for name in (
                "cfg_scale",
                "temperature",
                "top_p",
                "min_p",
                "linear",
                "confidence",
                "quadratic",
                "repetition_penalty",
        ):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
                raise ValueError(f"`{name}` must be finite.")
        if self.cfg_scale < 0:
            raise ValueError("`cfg_scale` must be non-negative.")
        if self.temperature < 0:
            raise ValueError("`temperature` must be non-negative.")
        for name in ("top_p", "min_p"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"`{name}` must be in [0, 1].")
        if self.linear < 0:
            raise ValueError("`linear` must be non-negative.")
        if self.repetition_penalty <= 0:
            raise ValueError("`repetition_penalty` must be greater than zero.")


def _multinomial(
    probabilities: Tensor,
    *,
    generator: torch.Generator | None,
) -> Tensor:
    noise = torch.empty_like(probabilities).exponential_(
        1,
        generator=generator,
    )
    return torch.argmax(
        probabilities / noise,
        dim=-1,
        keepdim=True,
    ).to(torch.int64)


def _unified_probabilities(
    probabilities: Tensor,
    *,
    linear: float,
    confidence: float,
    quadratic: float,
) -> Tensor:
    log_probabilities = torch.log(probabilities.clamp_min(1e-20))
    entropy = -torch.sum(
        probabilities * log_probabilities,
        dim=-1,
        keepdim=True,
    )
    scores = (log_probabilities * (linear + entropy * confidence) - log_probabilities.square() * quadratic)
    return scores.softmax(dim=-1)


def _top_k(probabilities: Tensor, value: int) -> Tensor:
    threshold = torch.topk(
        probabilities,
        min(value, probabilities.shape[-1]),
        dim=-1,
    ).values[..., -1:]
    probabilities = torch.where(
        probabilities < threshold,
        0.0,
        probabilities,
    )
    return probabilities / probabilities.sum(dim=-1, keepdim=True)


def _top_p(probabilities: Tensor, value: float) -> Tensor:
    sorted_probabilities, sorted_indices = torch.sort(
        probabilities,
        dim=-1,
        descending=True,
    )
    cumulative = torch.cumsum(sorted_probabilities, dim=-1)
    remove = cumulative - sorted_probabilities > value
    sorted_probabilities = sorted_probabilities.masked_fill(remove, 0.0)
    filtered = torch.zeros_like(probabilities).scatter(
        -1,
        sorted_indices,
        sorted_probabilities,
    )
    return filtered / filtered.sum(dim=-1, keepdim=True)


def _min_p(probabilities: Tensor, value: float) -> Tensor:
    maximum = probabilities.max(dim=-1, keepdim=True).values
    filtered = probabilities.masked_fill(
        probabilities < value * maximum,
        0.0,
    )
    return filtered / filtered.sum(dim=-1, keepdim=True)


def apply_repetition_penalty(
    logits: Tensor,
    generated_tokens: Tensor,
    *,
    penalty: float,
    window: int,
) -> Tensor:
    if penalty == 1.0:
        return logits
    recent = generated_tokens[..., -window:].clamp(
        0,
        logits.shape[-1] - 1,
    ).long()
    factors = torch.ones_like(logits).scatter_reduce(
        2,
        recent,
        torch.full_like(recent, penalty, dtype=logits.dtype),
        reduce="prod",
    )
    return torch.where(logits <= 0, logits * factors, logits / factors)


def sample_zonos_token(
    logits: Tensor,
    *,
    options: ZonosSamplingOptions,
    generated_tokens: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample one token for every codebook."""
    if not isinstance(logits, Tensor) or logits.ndim != 3:
        raise ValueError("Zonos logits must have shape [batch, codebook, vocabulary].")
    if generated_tokens is not None:
        logits = apply_repetition_penalty(
            logits,
            generated_tokens,
            penalty=options.repetition_penalty,
            window=options.repetition_penalty_window,
        )
    if options.temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)
    probabilities = torch.softmax(
        logits / options.temperature,
        dim=-1,
    )
    if options.linear > 0:
        probabilities = _unified_probabilities(
            probabilities,
            linear=options.linear,
            confidence=options.confidence,
            quadratic=options.quadratic,
        )
    if options.top_p > 0:
        probabilities = _top_p(probabilities, options.top_p)
    if options.top_k > 0:
        probabilities = _top_k(probabilities, options.top_k)
    if options.min_p > 0:
        probabilities = _min_p(probabilities, options.min_p)
    return _multinomial(probabilities, generator=generator)


@torch.inference_mode()
def generate_zonos_codes(
    model: ZonosForCausalLM,
    prefix_conditioning: Tensor,
    *,
    options: ZonosSamplingOptions | None = None,
    audio_prefix_codes: Tensor | None = None,
    generator: torch.Generator | None = None,
    callback: Callable[[Tensor, int, int], bool] | None = None,
) -> Tensor:
    """Generate source-layout DAC codes with endpoint-aware stopping."""
    if not isinstance(model, ZonosForCausalLM):
        raise TypeError("`model` must be a native ZonosForCausalLM.")
    options = ZonosSamplingOptions() if options is None else options
    if not isinstance(options, ZonosSamplingOptions):
        raise TypeError("`options` must be ZonosSamplingOptions or None.")
    if (not isinstance(prefix_conditioning, Tensor) or prefix_conditioning.ndim != 3):
        raise ValueError("Zonos prefix conditioning must have shape "
                         "[batch, time, hidden_size].")
    if prefix_conditioning.shape[-1] != model.config.backbone.d_model:
        raise ValueError("Zonos prefix conditioning hidden size does not match the model.")
    if options.cfg_scale == 1.0:
        # ``prepare_conditioning`` returns conditional then unconditional
        # batches. Guidance-free decoding consumes only the first half.
        if prefix_conditioning.shape[0] % 2 == 0:
            prefix_conditioning = prefix_conditioning.chunk(2, dim=0)[0]
        batch_size = prefix_conditioning.shape[0]
        cache_batch_size = batch_size
    else:
        if prefix_conditioning.shape[0] % 2:
            raise ValueError(
                "Guided Zonos conditioning must contain matching conditional "
                "and unconditional batches.")
        batch_size = prefix_conditioning.shape[0] // 2
        cache_batch_size = prefix_conditioning.shape[0]
    prefix_conditioning = prefix_conditioning.to(
        device=model.device,
        dtype=model.dtype,
    )

    prefix_audio_length = 0
    if audio_prefix_codes is not None:
        if (not isinstance(audio_prefix_codes, Tensor) or audio_prefix_codes.ndim != 3):
            raise ValueError("Zonos audio prefix must have shape "
                             "[batch, codebook, time].")
        if audio_prefix_codes.shape[:2] != (
                batch_size,
                model.config.num_codebooks,
        ):
            raise ValueError(
                "Zonos audio prefix batch/codebook dimensions do not match "
                "the conditioning.")
        if (audio_prefix_codes.dtype == torch.bool or audio_prefix_codes.is_floating_point()):
            raise TypeError("Zonos audio prefix must use an integer dtype.")
        if bool(((audio_prefix_codes < 0) | (audio_prefix_codes >= model.config.codebook_size)).any()):
            raise ValueError("Zonos audio prefix tokens must be in [0, 1023].")
        audio_prefix_codes = audio_prefix_codes.to(
            device=model.device,
            dtype=torch.long,
        )
        prefix_audio_length = audio_prefix_codes.shape[-1]

    unknown_token = -1
    audio_sequence_length = (prefix_audio_length + options.max_new_tokens)
    total_sequence_length = (
        prefix_conditioning.shape[1] + audio_sequence_length + model.config.num_codebooks)
    cache = model.setup_cache(
        batch_size=cache_batch_size,
        max_sequence_length=total_sequence_length,
    )
    codes = torch.full(
        (
            batch_size,
            model.config.num_codebooks,
            audio_sequence_length,
        ),
        unknown_token,
        dtype=torch.long,
        device=model.device,
    )
    if audio_prefix_codes is not None:
        codes[..., :prefix_audio_length] = audio_prefix_codes
    delayed = apply_delay_pattern(codes, model.masked_token_id)
    delayed_prefix = delayed[..., :prefix_audio_length + 1]
    logits = model.prefill(
        prefix_conditioning,
        delayed_prefix,
        cache,
        cfg_scale=options.cfg_scale,
    )
    next_token = sample_zonos_token(
        logits,
        options=options,
        generator=generator,
    )
    offset = delayed_prefix.shape[-1]
    frame = delayed[..., offset:offset + 1]
    frame.masked_scatter_(frame == unknown_token, next_token)

    consumed = (prefix_conditioning.shape[1] + prefix_audio_length + 1)
    cache.sequence_offset += consumed
    cache.lengths_per_sample.add_(consumed)

    logit_bias = torch.zeros_like(logits)
    logit_bias[:, 1:, model.eos_token_id] = -torch.inf
    stopping = torch.zeros(
        batch_size,
        dtype=torch.bool,
        device=model.device,
    )
    maximum_steps = delayed.shape[-1] - offset
    remaining = torch.full(
        (batch_size, ),
        maximum_steps,
        dtype=torch.long,
        device=model.device,
    )
    step = 0
    while bool((remaining > 0).any()):
        offset += 1
        logits = model.decode_step(
            delayed[..., offset - 1:offset],
            cache,
            cfg_scale=options.cfg_scale,
        )
        logits = logits + logit_bias
        next_token = sample_zonos_token(
            logits,
            options=options,
            generated_tokens=delayed[..., :offset],
            generator=generator,
        )
        eos_in_first = next_token[:, 0, 0] == model.eos_token_id
        remaining[eos_in_first] = torch.minimum(
            remaining[eos_in_first],
            torch.full_like(
                remaining[eos_in_first],
                model.config.num_codebooks,
            ),
        )
        stopping |= eos_in_first
        endpoint_codebook = (model.config.num_codebooks - remaining).clamp(max=model.config.num_codebooks - 1)
        for batch_index in range(batch_size):
            if bool(stopping[batch_index]):
                codebook_index = int(endpoint_codebook[batch_index].item())
                next_token[batch_index, :codebook_index] = (model.masked_token_id)
                next_token[
                    batch_index,
                    codebook_index,
                ] = model.eos_token_id
        frame = delayed[..., offset:offset + 1]
        frame.masked_scatter_(frame == unknown_token, next_token)
        cache.sequence_offset += 1
        cache.lengths_per_sample.add_(1)
        remaining.sub_(1)
        step += 1
        if callback is not None and not callback(
                frame,
                step,
                maximum_steps,
        ):
            break

    output = revert_delay_pattern(delayed)
    output.masked_fill_(
        output >= model.config.codebook_size,
        0,
    )
    output = output[..., :max(0, offset - model.config.num_codebooks)]
    if output.shape[-1] == 0:
        raise RuntimeError("Zonos generation ended before producing a complete DAC frame.")
    return output


__all__ = [
    "ZonosSamplingOptions",
    "apply_repetition_penalty",
    "generate_zonos_codes",
    "sample_zonos_token",
]
