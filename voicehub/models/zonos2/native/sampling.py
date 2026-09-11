"""Autoregressive sampling for the VoiceHub-native ZONOS2 graph."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import torch
from torch import Tensor

from voicehub.models.zonos2.native.modeling import Zonos2ForCausalLM


@dataclass(slots=True)
class Zonos2SamplingOptions:
    max_new_tokens: int = 1_024
    temperature: float = 1.15
    top_k: int = 106
    top_p: float = 0.0
    min_p: float = 0.18
    repetition_window: int = 50
    repetition_penalty: float = 1.2
    repetition_codebooks: int = 8
    seed: int | None = None

    def validate(self) -> None:
        for name in ("max_new_tokens", "repetition_window"):
            value = getattr(self, name)
            minimum = 1 if name == "max_new_tokens" else 0
            if (isinstance(value, bool) or not isinstance(value, int) or value < minimum):
                raise ValueError(f"`{name}` must be an integer >= {minimum}.")
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or self.top_k < 0:
            raise ValueError("`top_k` must be a non-negative integer.")
        if (isinstance(self.repetition_codebooks, bool) or not isinstance(self.repetition_codebooks, int) or
                self.repetition_codebooks < -1):
            raise ValueError("`repetition_codebooks` must be -1 or non-negative.")
        for name in ("temperature", "repetition_penalty"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or
                    value < 0):
                raise ValueError(f"`{name}` must be finite and non-negative.")
        if self.repetition_penalty == 0:
            raise ValueError("`repetition_penalty` must be greater than zero.")
        for name in ("top_p", "min_p"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or
                    not 0 <= value <= 1):
                raise ValueError(f"`{name}` must be in [0, 1].")
        if self.seed is not None and (isinstance(self.seed, bool) or not isinstance(self.seed, int)):
            raise TypeError("`seed` must be an integer or None.")


def apply_repetition_penalty(
    logits: Tensor,
    generated: list[Tensor],
    options: Zonos2SamplingOptions,
) -> Tensor:
    """Apply repetition penalties independently to each audio codebook."""
    if (not generated or options.repetition_window == 0 or options.repetition_penalty == 1.0):
        return logits
    result = logits.clone()
    codebook_count = (
        result.shape[1] if options.repetition_codebooks < 0 else min(
            result.shape[1], options.repetition_codebooks))
    recent = torch.stack(
        generated[-options.repetition_window:],
        dim=0,
    ).to(device=result.device)
    for codebook in range(codebook_count):
        token_ids = recent[:, codebook].long().unique()
        token_ids = token_ids[(token_ids >= 0) & (token_ids < result.shape[-1])]
        if token_ids.numel() == 0:
            continue
        selected = result[:, codebook, token_ids]
        result[:, codebook, token_ids] = torch.where(
            selected > 0,
            selected / options.repetition_penalty,
            selected * options.repetition_penalty,
        )
    return result


def sample_zonos2_codes(
    logits: Tensor,
    *,
    generated: list[Tensor],
    options: Zonos2SamplingOptions,
    generator: torch.Generator | None,
) -> Tensor:
    """Sample one ``[codebooks]`` frame from batch-size-one logits."""
    if logits.ndim != 3 or logits.shape[0] != 1:
        raise ValueError("ZONOS2 sampling currently expects logits [1, codebooks, vocab].")
    filtered = apply_repetition_penalty(
        logits.float(),
        generated,
        options,
    )
    if options.temperature <= 1e-5:
        return filtered.argmax(dim=-1)[0].long()
    filtered = filtered / max(options.temperature, 1e-8)
    vocabulary_size = filtered.shape[-1]
    if 0 < options.top_k < vocabulary_size:
        threshold = torch.topk(
            filtered,
            options.top_k,
            dim=-1,
        ).values[..., -1:]
        filtered = filtered.masked_fill(
            filtered < threshold,
            float("-inf"),
        )
    probabilities = torch.softmax(filtered, dim=-1)
    if 0.0 < options.top_p < 1.0:
        sorted_probabilities, sorted_indices = probabilities.sort(
            dim=-1,
            descending=True,
        )
        cumulative = sorted_probabilities.cumsum(dim=-1)
        remove = cumulative - sorted_probabilities > options.top_p
        sorted_probabilities = sorted_probabilities.masked_fill(remove, 0.0)
        probabilities = torch.zeros_like(probabilities).scatter(
            -1,
            sorted_indices,
            sorted_probabilities,
        )
    if options.min_p > 0:
        maximum = probabilities.amax(dim=-1, keepdim=True)
        probabilities = probabilities.masked_fill(
            probabilities < maximum * options.min_p,
            0.0,
        )
    totals = probabilities.sum(dim=-1, keepdim=True)
    invalid = totals <= 0
    probabilities = probabilities / totals.clamp_min(1e-8)
    if invalid.any():
        greedy = filtered.argmax(dim=-1, keepdim=True)
        fallback = torch.zeros_like(probabilities).scatter(
            -1,
            greedy,
            1.0,
        )
        probabilities = torch.where(invalid, fallback, probabilities)
    return torch.multinomial(
        probabilities[0],
        num_samples=1,
        generator=generator,
    ).squeeze(-1)


@torch.inference_mode()
def generate_zonos2_codes(
    model: Zonos2ForCausalLM,
    prompt: Tensor,
    *,
    options: Zonos2SamplingOptions | None = None,
    speaker_embedding: Tensor | None = None,
    speaker_position: int | None = None,
) -> tuple[Tensor, int | None]:
    """Generate delayed DAC frames using the model's native KV cache."""
    options = Zonos2SamplingOptions() if options is None else options
    options.validate()
    if prompt.ndim != 3 or prompt.shape[0] != 1:
        raise ValueError("ZONOS2 generation prompt must have shape [1, time, streams].")
    total_length = prompt.shape[1] + options.max_new_tokens
    if total_length > model.config.max_seqlen:
        raise ValueError(
            f"Prompt ({prompt.shape[1]}) plus generation "
            f"({options.max_new_tokens}) exceeds max_seqlen="
            f"{model.config.max_seqlen}.")
    prompt = prompt.to(device=model.device)
    cache = model.create_kv_cache(
        batch_size=1,
        max_length=total_length,
    )
    output = model(
        prompt,
        kv_cache=cache,
        speaker_embedding=speaker_embedding,
        speaker_position=speaker_position,
    )
    logits = output.logits[:, -1]
    generated: list[Tensor] = []
    eos_frame = None
    eos_countdown = 0
    generator = None
    if options.seed is not None:
        generator = torch.Generator(device=model.device)
        generator.manual_seed(options.seed)

    for step in range(options.max_new_tokens):
        codes = sample_zonos2_codes(
            logits,
            generated=generated,
            options=options,
            generator=generator,
        )
        generated.append(codes)
        if eos_frame is None:
            eos_codebooks = torch.nonzero(
                codes == model.config.eoa_id,
                as_tuple=False,
            ).flatten()
            if eos_codebooks.numel():
                eos_frame = max(
                    0,
                    step - int(eos_codebooks.max().item()),
                )
                eos_countdown = model.config.n_codebooks + 1
        if eos_frame is not None:
            eos_countdown -= 1
            if eos_countdown <= 0:
                break
        next_row = torch.full(
            (1, 1, model.config.frame_width),
            model.config.audio_pad_id,
            dtype=torch.long,
            device=model.device,
        )
        next_row[0, 0, :-1] = codes
        next_row[0, 0, -1] = model.config.text_vocab
        logits = model(next_row, kv_cache=cache).logits[:, -1]
    if not generated:
        raise RuntimeError("ZONOS2 generated no audio-code frames.")
    return torch.stack(generated, dim=0), eos_frame


__all__ = [
    "Zonos2SamplingOptions",
    "apply_repetition_penalty",
    "generate_zonos2_codes",
    "sample_zonos2_codes",
]
