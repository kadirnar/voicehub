"""Dependency-free ZONOS2 byte prompting and delayed-codebook batching."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor

from voicehub.models.zonos2.native.configuration import Zonos2ArchitectureConfig

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3
LEGACY_SYMBOL_VOCAB_SIZE = 192
BYTE_VOCAB_SIZE = 256
BYTE_TEXT_VOCAB_SIZE = LEGACY_SYMBOL_VOCAB_SIZE + BYTE_VOCAB_SIZE

# Published 0.2-second DAC prefix at 44.1 kHz (17 frames, 9 codebooks).
SILENCE_TOKENS_0_2S = (
    (568, 778, 338, 524, 967, 360, 728, 550, 90),
    (568, 778, 10, 674, 364, 981, 741, 378, 731),
    (568, 804, 10, 674, 364, 981, 568, 378, 731),
    (568, 804, 10, 674, 364, 981, 568, 378, 731),
    (568, 804, 10, 674, 364, 981, 568, 378, 731),
    (568, 804, 10, 674, 364, 981, 568, 378, 731),
    (568, 804, 10, 674, 364, 981, 568, 378, 731),
    (568, 804, 10, 674, 364, 981, 568, 378, 731),
    (568, 804, 10, 674, 364, 981, 568, 378, 731),
    (568, 804, 10, 674, 364, 981, 568, 378, 731),
    (568, 804, 10, 674, 364, 981, 568, 378, 731),
    (568, 804, 10, 674, 364, 981, 568, 378, 731),
    (568, 804, 10, 674, 364, 981, 568, 378, 731),
    (568, 804, 10, 674, 364, 981, 568, 378, 731),
    (568, 804, 10, 674, 364, 981, 568, 378, 731),
    (568, 804, 10, 674, 364, 981, 568, 378, 731),
    (568, 778, 721, 842, 264, 974, 989, 507, 308),
)


def text_to_byte_ids(text: str) -> list[int]:
    """Encode text exactly as the published raw UTF-8 tokenizer."""
    if not isinstance(text, str):
        raise TypeError("ZONOS2 text must be a string.")
    if not text:
        raise ValueError("ZONOS2 text cannot be empty.")
    return [
        BOS_ID,
        *(value + LEGACY_SYMBOL_VOCAB_SIZE for value in text.encode("utf-8")),
        EOS_ID,
    ]


def conditioning_base(config: Zonos2ArchitectureConfig) -> int:
    background_count = 2 if config.speaker_background_token_enabled else 0
    accurate_count = (1 if config.accurate_mode_token_enabled and background_count else 0)
    base = (
        config.text_vocab - config.speaking_rate_num_buckets - sum(config.quality_bucket_counts) -
        background_count - accurate_count)
    if base < BYTE_TEXT_VOCAB_SIZE:
        raise ValueError("ZONOS2 conditioning tokens overlap the UTF-8 byte vocabulary.")
    return base


def speaking_rate_token(
    config: Zonos2ArchitectureConfig,
    bucket: int,
) -> int:
    if not 0 <= bucket < config.speaking_rate_num_buckets:
        raise ValueError(
            "ZONOS2 speaking-rate bucket must be in "
            f"[0, {config.speaking_rate_num_buckets - 1}].")
    return conditioning_base(config) + bucket


def quality_token(
    config: Zonos2ArchitectureConfig,
    feature_index: int,
    bucket: int,
) -> int:
    if not 0 <= feature_index < len(config.quality_features):
        raise ValueError("ZONOS2 quality feature index is out of range.")
    count = config.quality_bucket_counts[feature_index]
    if not 0 <= bucket < count:
        name = config.quality_features[feature_index]
        raise ValueError(f"ZONOS2 quality bucket for {name!r} must be in [0, {count - 1}].")
    return (
        conditioning_base(config) + config.speaking_rate_num_buckets +
        sum(config.quality_bucket_counts[:feature_index]) + bucket)


def speaker_background_token(
    config: Zonos2ArchitectureConfig,
    *,
    clean: bool,
) -> int:
    if not config.speaker_background_token_enabled:
        raise ValueError("This ZONOS2 checkpoint has no background token.")
    return (
        conditioning_base(config) + config.speaking_rate_num_buckets + sum(config.quality_bucket_counts) +
        (0 if clean else 1))


def accurate_mode_token(config: Zonos2ArchitectureConfig) -> int:
    if (not config.speaker_background_token_enabled or not config.accurate_mode_token_enabled):
        raise ValueError("This ZONOS2 checkpoint has no accurate-mode token.")
    return (
        conditioning_base(config) + config.speaking_rate_num_buckets + sum(config.quality_bucket_counts) + 2)


def shear(codes: Tensor, pad_id: int) -> Tensor:
    """Apply the published delay pattern to ``[frames, codebooks]``."""
    if codes.ndim != 2:
        raise ValueError("ZONOS2 codes must have shape [frames, codebooks].")
    frames, codebooks = codes.shape
    if frames == 0 or codebooks == 0:
        raise ValueError("ZONOS2 codes cannot have an empty dimension.")
    padded = codes.new_full((codebooks - 1 + frames, codebooks), pad_id)
    padded[codebooks - 1:] = codes
    rows = (
        codebooks - 1 + torch.arange(frames, device=codes.device).unsqueeze(1) -
        torch.arange(codebooks, device=codes.device))
    return padded.gather(0, rows)


def shear_up(codes: Tensor, pad_id: int) -> Tensor:
    """Remove the ZONOS2 delay pattern."""
    if codes.ndim < 2:
        raise ValueError("ZONOS2 codes must have at least two dimensions.")
    frames, codebooks = codes.shape[-2:]
    output = codes.new_full(codes.shape, pad_id)
    for codebook in range(codebooks):
        if frames > codebook:
            output[..., :frames - codebook, codebook] = (codes[..., codebook:, codebook])
    return output


def delay_audio_completion(
    codes: Tensor,
    *,
    pad_id: int,
    eoa_id: int,
) -> Tensor:
    """Delay a complete target and append one aligned EOA per codebook.

    The published inference prompt's :func:`shear` retains its original
    length. Training needs the delayed tail as targets, so this helper
    emits ``frames + codebooks`` rows and explicitly places end-of-audio
    at ``frames + codebook_index``.
    """
    if codes.ndim != 2:
        raise ValueError("ZONOS2 audio codes must be [frames, codebooks].")
    if codes.shape[0] == 0 or codes.shape[1] == 0:
        raise ValueError("ZONOS2 audio codes cannot be empty.")
    if codes.dtype == torch.bool or codes.is_floating_point():
        raise TypeError("ZONOS2 audio codes must use an integer dtype.")
    frames, codebooks = codes.shape
    delayed = codes.new_full((frames + codebooks, codebooks), pad_id)
    frame_positions = torch.arange(frames, device=codes.device)
    for codebook in range(codebooks):
        delayed[frame_positions + codebook, codebook] = codes[:, codebook]
        delayed[frames + codebook, codebook] = eoa_id
    return delayed


def _marker_row(
    config: Zonos2ArchitectureConfig,
    text_token: int,
    *,
    device: torch.device | str | None,
) -> Tensor:
    row = torch.full(
        (1, config.frame_width),
        config.audio_pad_id,
        dtype=torch.long,
        device=device,
    )
    row[:, -1] = text_token
    return row


def build_zonos2_prompt(
    config: Zonos2ArchitectureConfig,
    text: str,
    *,
    speaking_rate_bucket: int | None = None,
    quality_buckets: Sequence[int | None] | None = None,
    include_speaker_slot: bool = False,
    clean_speaker_background: bool = False,
    accurate_mode: bool = True,
    prepend_silence: bool = True,
    device: torch.device | str | None = None,
) -> tuple[Tensor, int | None]:
    """Build one checkpoint-compatible prompt and speaker-slot position."""
    rows: list[Tensor] = []
    speaker_position = None
    if include_speaker_slot:
        if not config.speaker_enabled:
            raise ValueError("This ZONOS2 checkpoint does not support speakers.")
        speaker_position = 0
        rows.append(_marker_row(config, config.text_vocab, device=device))
        if config.speaker_background_token_enabled:
            rows.append(
                _marker_row(
                    config,
                    speaker_background_token(
                        config,
                        clean=clean_speaker_background,
                    ),
                    device=device,
                ))
        if config.accurate_mode_token_enabled and accurate_mode:
            rows.append(_marker_row(
                config,
                accurate_mode_token(config),
                device=device,
            ))

    if speaking_rate_bucket is not None:
        rows.append(
            _marker_row(
                config,
                speaking_rate_token(config, int(speaking_rate_bucket)),
                device=device,
            ))
    if quality_buckets is not None:
        if len(quality_buckets) != len(config.quality_features):
            raise ValueError(
                f"Expected {len(config.quality_features)} quality buckets, "
                f"received {len(quality_buckets)}.")
        for feature_index, bucket in enumerate(quality_buckets):
            if bucket is None:
                continue
            rows.append(
                _marker_row(
                    config,
                    quality_token(config, feature_index, int(bucket)),
                    device=device,
                ))

    text_tokens = text_to_byte_ids(text)
    text_rows = torch.full(
        (len(text_tokens), config.frame_width),
        config.audio_pad_id,
        dtype=torch.long,
        device=device,
    )
    text_rows[:, -1] = torch.tensor(
        text_tokens,
        dtype=torch.long,
        device=device,
    )
    rows.append(text_rows)

    if prepend_silence:
        if config.n_codebooks > len(SILENCE_TOKENS_0_2S[0]):
            raise ValueError("Published silence prefix supports at most nine codebooks.")
        silence = torch.tensor(
            SILENCE_TOKENS_0_2S,
            dtype=torch.long,
            device=device,
        )[:, :config.n_codebooks]
        silence = shear(silence, config.audio_pad_id)
        silence_rows = torch.full(
            (silence.shape[0], config.frame_width),
            config.audio_pad_id,
            dtype=torch.long,
            device=device,
        )
        silence_rows[:, :-1] = silence
        silence_rows[:, -1] = config.text_vocab
        rows.append(silence_rows)
    return torch.cat(rows, dim=0).unsqueeze(0), speaker_position


def _normalize_audio_batch(
    audio_codes: Tensor | Sequence[Tensor],
    *,
    n_codebooks: int,
) -> list[Tensor]:
    if isinstance(audio_codes, Tensor):
        if audio_codes.ndim == 2:
            values = [audio_codes]
        elif audio_codes.ndim == 3:
            values = list(audio_codes.unbind(dim=0))
        else:
            raise ValueError(
                "ZONOS2 audio codes must be [frames, codebooks] or "
                "[batch, frames, codebooks].")
    elif isinstance(audio_codes, Sequence):
        values = list(audio_codes)
    else:
        raise TypeError("ZONOS2 audio codes must be a tensor or tensor sequence.")
    if not values or any(not isinstance(item, Tensor) for item in values):
        raise TypeError("Every ZONOS2 audio-code example must be a tensor.")
    for item in values:
        if item.ndim != 2 or item.shape[-1] != n_codebooks:
            raise ValueError(f"Each ZONOS2 audio target must be [frames, {n_codebooks}].")
    return values


def prepare_zonos2_training_batch(
    config: Zonos2ArchitectureConfig,
    texts: str | Sequence[str],
    audio_codes: Tensor | Sequence[Tensor],
    *,
    speaker_embeddings: Tensor | None = None,
    prepend_silence: bool = True,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Create a padded, prompt-masked teacher-forcing batch.

    Audio tokenization is intentionally outside this function.  Training
    data must supply DAC codebooks from the pinned 44.1-kHz codec; this
    keeps the frozen codec boundary explicit and makes preprocessing
    cacheable.
    """
    normalized_texts = [texts] if isinstance(texts, str) else list(texts)
    if not normalized_texts or any(not isinstance(text, str) or not text for text in normalized_texts):
        raise ValueError("ZONOS2 training texts must be non-empty strings.")
    normalized_codes = _normalize_audio_batch(
        audio_codes,
        n_codebooks=config.n_codebooks,
    )
    if len(normalized_texts) != len(normalized_codes):
        raise ValueError("ZONOS2 text and audio-code batch sizes must match.")
    if speaker_embeddings is not None:
        if speaker_embeddings.ndim == 1:
            speaker_embeddings = speaker_embeddings.unsqueeze(0)
        if speaker_embeddings.shape != (
                len(normalized_texts),
                config.speaker_embedding_dim,
        ):
            raise ValueError(
                "ZONOS2 speaker embeddings must have shape "
                f"[batch, {config.speaker_embedding_dim}].")

    sequences: list[Tensor] = []
    labels: list[Tensor] = []
    prompt_lengths: list[int] = []
    for text, codes in zip(normalized_texts, normalized_codes):
        prompt, _ = build_zonos2_prompt(
            config,
            text,
            include_speaker_slot=speaker_embeddings is not None,
            prepend_silence=prepend_silence,
            device=device,
        )
        prompt = prompt[0]
        delayed = delay_audio_completion(
            codes.to(device=device, dtype=torch.long),
            pad_id=config.audio_pad_id,
            eoa_id=config.eoa_id,
        )
        completion = torch.full(
            (delayed.shape[0], config.frame_width),
            config.audio_pad_id,
            dtype=torch.long,
            device=device,
        )
        completion[:, :-1] = delayed
        completion[:, -1] = config.text_vocab
        sequence = torch.cat((prompt, completion), dim=0)
        target = torch.full(
            (sequence.shape[0], config.n_codebooks),
            -100,
            dtype=torch.long,
            device=device,
        )
        target[prompt.shape[0]:] = delayed
        sequences.append(sequence)
        labels.append(target)
        prompt_lengths.append(prompt.shape[0])

    maximum_length = max(sequence.shape[0] for sequence in sequences)
    batch_size = len(sequences)
    input_ids = torch.full(
        (batch_size, maximum_length, config.frame_width),
        config.audio_pad_id,
        dtype=torch.long,
        device=device,
    )
    input_ids[..., -1] = config.text_vocab
    target_ids = torch.full(
        (batch_size, maximum_length, config.n_codebooks),
        -100,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros(
        (batch_size, maximum_length),
        dtype=torch.bool,
        device=device,
    )
    loss_mask = torch.zeros_like(attention_mask)
    for index, (sequence, target, prompt_length) in enumerate(zip(sequences, labels, prompt_lengths)):
        length = sequence.shape[0]
        input_ids[index, :length] = sequence
        target_ids[index, :length] = target
        attention_mask[index, :length] = True
        loss_mask[index, prompt_length:length] = True

    batch: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": target_ids,
        "loss_mask": loss_mask,
    }
    if speaker_embeddings is not None:
        batch["speaker_embedding"] = speaker_embeddings.to(
            device=device,
            dtype=torch.float32,
        )
        batch["speaker_position"] = 0
    return batch


__all__ = [
    "BOS_ID",
    "BYTE_TEXT_VOCAB_SIZE",
    "EOS_ID",
    "LEGACY_SYMBOL_VOCAB_SIZE",
    "PAD_ID",
    "SILENCE_TOKENS_0_2S",
    "UNK_ID",
    "accurate_mode_token",
    "build_zonos2_prompt",
    "conditioning_base",
    "delay_audio_completion",
    "prepare_zonos2_training_batch",
    "quality_token",
    "shear",
    "shear_up",
    "speaker_background_token",
    "speaking_rate_token",
    "text_to_byte_ids",
]
