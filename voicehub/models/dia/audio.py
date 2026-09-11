"""Compatibility functions for Dia's native delay protocol."""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor

from voicehub.models.dia.native.processing import DiaProcessor


def build_delay_indices(
    batch_size: int,
    sequence_length: int,
    num_channels: int,
    delay_pattern: Sequence[int],
) -> tuple[Tensor, Tensor]:
    return DiaProcessor.build_indices(
        batch_size,
        sequence_length,
        num_channels,
        delay_pattern,
    )


def apply_audio_delay(
    audio: Tensor,
    pad_value: int,
    bos_value: int,
    precomputed_indices: tuple[Tensor, Tensor],
) -> Tensor:
    return DiaProcessor.apply_audio_delay(
        audio,
        pad_token_id=pad_value,
        bos_token_id=bos_value,
        precomputed_indices=precomputed_indices,
    )


def build_revert_indices(
    batch_size: int,
    sequence_length: int,
    num_channels: int,
    delay_pattern: Sequence[int],
) -> tuple[Tensor, Tensor]:
    return DiaProcessor.build_indices(
        batch_size,
        sequence_length,
        num_channels,
        delay_pattern,
        revert=True,
    )


def revert_audio_delay(
    audio: Tensor,
    pad_value: int,
    precomputed_indices: tuple[Tensor, Tensor],
    sequence_length: int,
) -> Tensor:
    if sequence_length != audio.shape[1]:
        raise ValueError("`sequence_length` must equal the delayed audio sequence length.")
    return DiaProcessor.apply_audio_delay(
        audio,
        pad_token_id=pad_value,
        bos_token_id=pad_value,
        precomputed_indices=precomputed_indices,
    )


__all__ = [
    "apply_audio_delay",
    "build_delay_indices",
    "build_revert_indices",
    "revert_audio_delay",
]
