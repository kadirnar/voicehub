"""Source-compatible training sequence construction for ConversationTTS."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class ConversationTTSProtocol:
    """Token layout published by the ConversationTTS training pipeline."""

    audio_num_codebooks: int = 32
    audio_codebook_size: int = 2_048
    audio_vocab_size: int = 2_051
    text_vocab_size: int = 128_256
    text_empty_token_id: int = 0
    text_padding_token_id: int = 128_002
    audio_eos_token_id: int = 0
    audio_padding_token_id: int = 2_050
    maximum_sequence_length: int = 2_048

    def __post_init__(self) -> None:
        for name in (
                "audio_num_codebooks",
                "audio_codebook_size",
                "audio_vocab_size",
                "text_vocab_size",
                "maximum_sequence_length",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"`{name}` must be a positive integer.")
        if self.audio_codebook_size > self.audio_vocab_size:
            raise ValueError("`audio_codebook_size` cannot exceed `audio_vocab_size`.")
        for name, upper_bound in (
            ("text_empty_token_id", self.text_vocab_size),
            ("text_padding_token_id", self.text_vocab_size),
            ("audio_eos_token_id", self.audio_vocab_size),
            ("audio_padding_token_id", self.audio_vocab_size),
        ):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < upper_bound):
                raise ValueError(f"`{name}` must be an integer in [0, {upper_bound}).")


def _integer_tensor(
    value: Tensor | Sequence[int],
    *,
    name: str,
    dimensions: int,
) -> Tensor:
    tensor = value if isinstance(value, Tensor) else torch.as_tensor(value)
    if tensor.ndim != dimensions:
        raise ValueError(
            f"`{name}` must have {dimensions} dimensions, found "
            f"shape {tuple(tensor.shape)!r}.")
    if tensor.dtype == torch.bool or tensor.is_floating_point():
        raise TypeError(f"`{name}` must use an integer dtype.")
    return tensor.long()


def build_conversationtts_sequence(
    text_token_ids: Tensor | Sequence[int],
    audio_codes: Tensor | Sequence[Sequence[int]],
    *,
    protocol: ConversationTTSProtocol | None = None,
) -> tuple[Tensor, Tensor]:
    """Build one exact text-then-audio sequence and its stream mask."""
    layout = protocol or ConversationTTSProtocol()
    text = _integer_tensor(
        text_token_ids,
        name="text_token_ids",
        dimensions=1,
    )
    codes = _integer_tensor(
        audio_codes,
        name="audio_codes",
        dimensions=2,
    )
    if text.numel() == 0:
        raise ValueError("ConversationTTS text tokens cannot be empty.")
    if codes.shape[0] != layout.audio_num_codebooks:
        raise ValueError(
            "ConversationTTS audio codes must have shape "
            f"[{layout.audio_num_codebooks}, time], found "
            f"{tuple(codes.shape)!r}.")
    if codes.shape[-1] == 0:
        raise ValueError("ConversationTTS audio codes cannot be empty.")
    if bool((text < 0).any()) or bool((text >= layout.text_vocab_size).any()):
        raise ValueError("ConversationTTS text token IDs are outside the configured "
                         "vocabulary.")
    if bool((codes < 0).any()) or bool((codes >= layout.audio_codebook_size).any()):
        raise ValueError("ConversationTTS audio codes are outside the Mimi codebook.")

    number_of_streams = layout.audio_num_codebooks + 1
    text_frames = torch.zeros(
        text.shape[0],
        number_of_streams,
        dtype=torch.long,
        device=text.device,
    )
    text_mask = torch.zeros_like(text_frames, dtype=torch.bool)
    text_frames[:, -1] = text
    text_mask[:, -1] = True

    codes = codes.to(device=text.device)
    eos = torch.full(
        (layout.audio_num_codebooks, 1),
        layout.audio_eos_token_id,
        dtype=torch.long,
        device=text.device,
    )
    codes = torch.cat((codes, eos), dim=-1)
    audio_frames = torch.full(
        (codes.shape[-1], number_of_streams),
        layout.text_empty_token_id,
        dtype=torch.long,
        device=text.device,
    )
    audio_mask = torch.zeros_like(audio_frames, dtype=torch.bool)
    audio_frames[:, :-1] = codes.transpose(0, 1)
    audio_mask[:, :-1] = True

    sequence = torch.cat((text_frames, audio_frames), dim=0)
    mask = torch.cat((text_mask, audio_mask), dim=0)
    if sequence.shape[0] > layout.maximum_sequence_length:
        raise ValueError(
            "ConversationTTS sequence exceeds the released "
            f"{layout.maximum_sequence_length}-frame context: "
            f"{sequence.shape[0]} frames.")
    return sequence, mask


def collate_conversationtts_sequences(
    examples: Sequence[tuple[Tensor, Tensor]],
    *,
    protocol: ConversationTTSProtocol | None = None,
) -> dict[str, Tensor | int]:
    """Pad source-form sequences and produce the model's shifted inputs."""
    layout = protocol or ConversationTTSProtocol()
    if not examples:
        raise ValueError("ConversationTTS batches cannot be empty.")
    number_of_streams = layout.audio_num_codebooks + 1
    maximum_length = max(sequence.shape[0] for sequence, _ in examples)
    batch_size = len(examples)
    sequences = torch.empty(
        batch_size,
        maximum_length,
        number_of_streams,
        dtype=torch.long,
    )
    sequences[..., :-1] = layout.audio_padding_token_id
    sequences[..., -1] = layout.text_padding_token_id
    masks = torch.zeros_like(sequences, dtype=torch.bool)
    lengths = torch.empty(batch_size, dtype=torch.long)

    for index, (sequence, mask) in enumerate(examples):
        if sequence.ndim != 2 or sequence.shape[-1] != number_of_streams:
            raise ValueError(
                "ConversationTTS sequence examples must have shape "
                f"[time, {number_of_streams}].")
        if mask.shape != sequence.shape:
            raise ValueError("ConversationTTS sequence masks must match their sequence.")
        length = sequence.shape[0]
        sequences[index, :length] = sequence.to(device="cpu")
        masks[index, :length] = mask.to(device="cpu", dtype=torch.bool)
        lengths[index] = length

    return {
        "tokens": sequences[:, :-1],
        "labels": sequences[:, 1:, :-1],
        "tokens_mask": masks,
        "loss_mask": masks[:, 1:, 0],
        "sequence_lengths": lengths,
        "ignore_id": layout.audio_padding_token_id,
        "residual_ignore_id": layout.audio_padding_token_id,
    }


__all__ = [
    "ConversationTTSProtocol",
    "build_conversationtts_sequence",
    "collate_conversationtts_sequences",
]
