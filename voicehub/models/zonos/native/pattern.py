"""Delayed-codebook sequence transforms used by Zonos v0.1."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def apply_delay_pattern(codes: Tensor, mask_token_id: int) -> Tensor:
    """Delay codebook ``q`` by ``q + 1`` autoregressive positions."""
    if not isinstance(codes, Tensor) or codes.ndim != 3:
        raise ValueError("Zonos codes must have shape [batch, codebook, time].")
    if codes.shape[1] == 0:
        raise ValueError("Zonos codes must contain at least one codebook.")
    padded = F.pad(
        codes,
        (0, codes.shape[1]),
        value=mask_token_id,
    )
    return torch.stack(
        [padded[:, index].roll(index + 1, dims=-1) for index in range(padded.shape[1])],
        dim=1,
    )


def revert_delay_pattern(codes: Tensor) -> Tensor:
    """Remove the codebook delays from a generated sequence."""
    if not isinstance(codes, Tensor) or codes.ndim != 3:
        raise ValueError("Delayed Zonos codes must have shape [batch, codebook, time].")
    _, num_codebooks, sequence_length = codes.shape
    if sequence_length < num_codebooks + 1:
        raise ValueError("Delayed Zonos sequence is too short to contain one audio frame.")
    return torch.stack(
        [
            codes[
                :,
                index,
                index + 1:sequence_length - num_codebooks + index + 1,
            ] for index in range(num_codebooks)
        ],
        dim=1,
    )


def add_endpoint_and_delay(
    audio_codes: Tensor,
    *,
    lengths: Tensor | None,
    eos_token_id: int,
    mask_token_id: int,
) -> tuple[Tensor, Tensor]:
    """Build source-faithful teacher inputs and endpoint-aware targets."""
    if not isinstance(audio_codes, Tensor) or audio_codes.ndim != 3:
        raise ValueError("Zonos audio codes must have shape [batch, codebook, time].")
    batch_size, num_codebooks, padded_length = audio_codes.shape
    if batch_size == 0 or num_codebooks == 0 or padded_length == 0:
        raise ValueError("Zonos audio codes cannot have an empty dimension.")
    if audio_codes.dtype == torch.bool or audio_codes.is_floating_point():
        raise TypeError("Zonos audio codes must use an integer dtype.")
    if lengths is None:
        lengths = torch.full(
            (batch_size, ),
            padded_length,
            dtype=torch.long,
            device=audio_codes.device,
        )
    elif not isinstance(lengths, Tensor) or lengths.shape != (batch_size, ):
        raise ValueError("Zonos audio-code lengths must have shape [batch].")
    else:
        lengths = lengths.to(
            device=audio_codes.device,
            dtype=torch.long,
        )
    if bool(((lengths <= 0) | (lengths > padded_length)).any()):
        raise ValueError(
            "Zonos audio-code lengths must be positive and no larger than "
            "the padded time dimension.")
    valid = (torch.arange(padded_length, device=audio_codes.device)[None, None, :]
             < lengths[:, None, None]).expand_as(audio_codes)
    active = audio_codes.masked_select(valid)
    if bool(((active < 0) | (active >= eos_token_id)).any()):
        raise ValueError(f"Valid Zonos codec tokens must be in [0, {eos_token_id - 1}].")
    endpoint = torch.full(
        (batch_size, num_codebooks, padded_length + 1),
        mask_token_id,
        dtype=torch.long,
        device=audio_codes.device,
    )
    endpoint[..., :padded_length] = torch.where(
        valid,
        audio_codes.long(),
        mask_token_id,
    )
    endpoint.scatter_(
        dim=-1,
        index=lengths[:, None, None].expand(-1, num_codebooks, 1),
        value=eos_token_id,
    )
    delayed = apply_delay_pattern(endpoint, mask_token_id)
    return delayed[..., :-1], delayed[..., 1:]


__all__ = [
    "add_endpoint_and_delay",
    "apply_delay_pattern",
    "revert_delay_pattern",
]
