"""Fish S2 conversation protocol and pre-tokenized training helpers."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from voicehub.models.fishtts.native.tokenization import (
    IM_END,
    IM_START,
    MODALITY_VOICE,
    FishTokenizer,
    normalize_fish_text,
)

_SPEAKER_TURN = re.compile(r"(<\|speaker:\d+\|>)")


@dataclass(frozen=True, slots=True)
class FishConversationTurn:
    role: str
    text: str | None = None
    codes: Tensor | None = None
    modality: str | None = None

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant"}:
            raise ValueError("Fish message role is unsupported.")
        if self.text is None and self.codes is None:
            raise ValueError("A Fish turn requires text, codes, or both.")
        if self.modality not in {None, "voice"}:
            raise ValueError("Fish turn modality must be None or 'voice'.")


def split_speaker_turns(text: str) -> tuple[str, ...]:
    normalized = normalize_fish_text(text)
    parts = _SPEAKER_TURN.split(normalized)
    turns: list[str] = []
    current = ""
    for value in parts:
        if _SPEAKER_TURN.fullmatch(value):
            if current.strip():
                turns.append(current.strip())
            current = value
        else:
            current += value
    if current.strip():
        turns.append(current.strip())
    return tuple(turns) if turns else (normalized, )


def group_speaker_turns(
    turns: Sequence[str],
    *,
    maximum_turns: int = 5,
    maximum_utf8_bytes: int = 512,
) -> tuple[str, ...]:
    if (isinstance(maximum_turns, bool) or not isinstance(maximum_turns, int) or maximum_turns <= 0 or
            isinstance(maximum_utf8_bytes, bool) or not isinstance(maximum_utf8_bytes, int) or
            maximum_utf8_bytes <= 0):
        raise ValueError("Fish chunk limits must be positive.")
    batches: list[str] = []
    current: list[str] = []
    byte_count = 0
    for raw_turn in turns:
        turn = normalize_fish_text(raw_turn)
        turn_bytes = len(turn.encode("utf-8"))
        if turn_bytes > maximum_utf8_bytes and not current:
            # Splitting arbitrary UTF-8 text would change token context.
            # Keep an oversized speaker turn whole and let the model-context
            # bound fail explicitly if necessary.
            batches.append(turn)
            continue
        if current and (len(current) >= maximum_turns or byte_count + 1 + turn_bytes > maximum_utf8_bytes):
            batches.append("\n".join(current))
            current = []
            byte_count = 0
        if current:
            byte_count += 1
        current.append(turn)
        byte_count += turn_bytes
    if current:
        batches.append("\n".join(current))
    return tuple(batches)


def _validate_codes(
    codes: Tensor,
    tokenizer: FishTokenizer,
) -> Tensor:
    values = torch.as_tensor(codes)
    if (values.dtype == torch.bool or values.is_floating_point() or values.is_complex()):
        raise TypeError("Fish codes must use an integer dtype.")
    values = values.detach().to(
        device="cpu",
        dtype=torch.long,
    )
    expected = tokenizer.config.num_codebooks
    if values.ndim != 2 or values.shape[0] != expected:
        raise ValueError(f"Fish codes must have shape [{expected}, time].")
    if values.shape[1] == 0:
        raise ValueError("Fish code sequences cannot be empty.")
    if (int(values.min().item()) < 0 or int(values.max().item()) >= tokenizer.config.codebook_size):
        raise ValueError("Fish codes contain an out-of-range ID.")
    return values


def _encode_turn(
    turn: FishConversationTurn,
    tokenizer: FishTokenizer,
) -> Tensor:
    modality = MODALITY_VOICE if turn.modality == "voice" else ""
    prefix = tokenizer.encode(
        f"{IM_START}{turn.role}\n{modality}",
        allow_protocol_tokens=True,
    )
    suffix = tokenizer.encode(f"{IM_END}\n", allow_protocol_tokens=True)
    channel_count = tokenizer.config.num_codebooks + 1
    prefix_values = torch.zeros(
        (channel_count, len(prefix)),
        dtype=torch.long,
    )
    prefix_values[0] = torch.tensor(prefix, dtype=torch.long)
    parts = [prefix_values]
    if turn.text is not None:
        normalized_text = normalize_fish_text(turn.text)
        content = tokenizer.encode(
            normalized_text,
            allow_protocol_tokens=True,
        )
        text_values = torch.zeros(
            (channel_count, len(content)),
            dtype=torch.long,
        )
        text_values[0] = torch.tensor(content, dtype=torch.long)
        parts.append(text_values)
    if turn.codes is not None:
        codes = _validate_codes(turn.codes, tokenizer)
        code_primary = codes[0] + tokenizer.semantic_begin_id
        parts.append(torch.cat((code_primary.unsqueeze(0), codes), dim=0))
    suffix_values = torch.zeros(
        (channel_count, len(suffix)),
        dtype=torch.long,
    )
    suffix_values[0] = torch.tensor(suffix, dtype=torch.long)
    parts.append(suffix_values)
    return torch.cat(parts, dim=1)


def build_fish_prompt(
        text: str,
        tokenizer: FishTokenizer,
        *,
        reference_text: str | None = None,
        reference_codes: Tensor | None = None,
        history: Sequence[FishConversationTurn] = (),
) -> Tensor:
    """Build the exact system/user/assistant-voice prefix."""
    normalized_text = normalize_fish_text(text)
    if (reference_text is None) != (reference_codes is None):
        raise ValueError("Fish reference text and codes must be supplied together.")
    turns: list[FishConversationTurn] = []
    if reference_text is None:
        turns.append(FishConversationTurn(
            role="system",
            text="convert the provided text to speech",
        ))
    else:
        normalized_reference = normalize_fish_text(reference_text)
        if _SPEAKER_TURN.search(normalized_reference) is None:
            normalized_reference = "<|speaker:0|>" + normalized_reference
        turns.append(
            FishConversationTurn(
                role="system",
                text=(
                    "convert the provided text to speech reference to "
                    "the following:\n\nText:\n" + normalized_reference + "\n\nSpeech:\n"),
                codes=_validate_codes(reference_codes, tokenizer),
            ))
    turns.extend(history)
    turns.append(FishConversationTurn(role="user", text=normalized_text))

    # The inference prefix deliberately has no IM_END: generation starts
    # immediately after assistant + voice modality.
    assistant_prefix = tokenizer.encode(
        f"{IM_START}assistant\n{MODALITY_VOICE}",
        allow_protocol_tokens=True,
    )
    encoded = [_encode_turn(turn, tokenizer) for turn in turns]
    assistant = torch.zeros(
        (
            tokenizer.config.num_codebooks + 1,
            len(assistant_prefix),
        ),
        dtype=torch.long,
    )
    assistant[0] = torch.tensor(assistant_prefix, dtype=torch.long)
    return torch.cat((*encoded, assistant), dim=1)


def append_generated_turn(
    history: Sequence[FishConversationTurn],
    *,
    text: str,
    codes: Tensor,
) -> tuple[FishConversationTurn, ...]:
    return (
        *history,
        FishConversationTurn(
            role="user",
            text=normalize_fish_text(text),
        ),
        FishConversationTurn(
            role="assistant",
            codes=codes.detach().cpu(),
            modality="voice",
        ),
    )


__all__ = [
    "FishConversationTurn",
    "append_generated_turn",
    "build_fish_prompt",
    "group_speaker_turns",
    "split_speaker_turns",
]
