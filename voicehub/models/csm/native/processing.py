"""Native text and pre-encoded-audio preparation for Sesame CSM."""

from __future__ import annotations

import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from voicehub.models.csm.native.configuration import CSMArchitectureConfig
from voicehub.models.csm.native.metadata import CSM_BOS_TOKEN_ID, CSM_EOS_TOKEN_ID, CSM_TOKENIZER_FILE
from voicehub.tokenization import ByteBPETokenizer


class CSMTextTokenizer:
    """Dependency-free Llama-3 tokenizer with CSM's BOS/EOS template."""

    def __init__(
        self,
        tokenizer: ByteBPETokenizer,
        *,
        source_path: str | Path | None = None,
    ) -> None:
        if not isinstance(tokenizer, ByteBPETokenizer):
            raise TypeError("CSM requires a `ByteBPETokenizer`.")
        if tokenizer.token_id_space_size < CSM_EOS_TOKEN_ID + 1:
            raise ValueError("CSM tokenizer vocabulary does not contain Llama-3 BOS/EOS.")
        self.tokenizer = tokenizer
        self.source_path = (None if source_path is None else Path(source_path).expanduser().resolve())

    @classmethod
    def from_file(cls, path: str | Path) -> CSMTextTokenizer:
        source = Path(path).expanduser().resolve()
        tokenizer = ByteBPETokenizer.from_tokenizer_json(
            source,
            prefix_token_ids=(CSM_BOS_TOKEN_ID, ),
            suffix_token_ids=(CSM_EOS_TOKEN_ID, ),
            pad_token_id=CSM_EOS_TOKEN_ID,
        )
        return cls(tokenizer, source_path=source)

    def encode(self, text: str, *, speaker: int) -> tuple[int, ...]:
        if isinstance(speaker, bool) or not isinstance(speaker, int) or speaker < 0:
            raise ValueError("CSM speaker IDs must be non-negative integers.")
        return self.tokenizer.encode(f"[{speaker}]{text}").input_ids

    def save_pretrained(self, directory: str | Path) -> Path:
        if self.source_path is None:
            raise RuntimeError("This in-memory CSM tokenizer has no source asset to export.")
        output = Path(directory).expanduser()
        output.mkdir(parents=True, exist_ok=True)
        destination = output / CSM_TOKENIZER_FILE
        if self.source_path != destination.resolve():
            shutil.copyfile(self.source_path, destination)
        return destination.resolve()


@dataclass(frozen=True)
class CSMCodeSegment:
    """One speaker/text/audio-code context segment."""

    speaker: int
    text: str
    audio_codes: Tensor


class CSMProcessor:
    """Build source-layout frame tensors without a processor framework."""

    def __init__(
        self,
        tokenizer: CSMTextTokenizer,
        config: CSMArchitectureConfig,
    ) -> None:
        if not isinstance(tokenizer, CSMTextTokenizer):
            raise TypeError("`tokenizer` must be a `CSMTextTokenizer`.")
        if not isinstance(config, CSMArchitectureConfig):
            raise TypeError("`config` must be a `CSMArchitectureConfig`.")
        self.tokenizer = tokenizer
        self.config = config
        if (tokenizer.tokenizer.token_id_space_size > config.text_vocabulary_size):
            raise ValueError("Tokenizer ID space exceeds CSM text embedding rows.")

    @property
    def sample_rate(self) -> int:
        return self.config.sample_rate

    def text_frames(
        self,
        text: str,
        *,
        speaker: int,
        device=None,
    ) -> tuple[Tensor, Tensor]:
        ids = self.tokenizer.encode(text, speaker=speaker)
        width = self.config.num_audio_codebooks + 1
        tokens = torch.zeros(
            len(ids),
            width,
            dtype=torch.long,
            device=device,
        )
        mask = torch.zeros_like(tokens, dtype=torch.bool)
        tokens[:, -1] = torch.tensor(ids, device=device, dtype=torch.long)
        mask[:, -1] = True
        return tokens, mask

    def audio_frames(
        self,
        audio_codes: Tensor,
        *,
        append_eos: bool = True,
        device=None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if not isinstance(audio_codes, Tensor):
            raise TypeError("`audio_codes` must be a PyTorch tensor.")
        codes = audio_codes
        if codes.ndim == 3:
            if codes.shape[0] != 1:
                raise ValueError("Per-segment CSM codes may only have a singleton batch.")
            codes = codes[0]
        if (codes.ndim != 2 or codes.shape[0] != self.config.num_audio_codebooks):
            raise ValueError("CSM audio codes must have shape [num_codebooks, frames].")
        if codes.dtype == torch.bool or codes.is_floating_point():
            raise TypeError("CSM audio codes must use an integer dtype.")
        if codes.numel() and (int(codes.min()) < 0 or int(codes.max()) >= self.config.audio_vocabulary_size):
            raise ValueError("CSM audio code is outside the vocabulary.")
        codes = codes.to(device=device, dtype=torch.long)
        if append_eos:
            codes = torch.cat(
                (
                    codes,
                    torch.zeros(
                        self.config.num_audio_codebooks,
                        1,
                        dtype=torch.long,
                        device=codes.device,
                    ),
                ),
                dim=1,
            )
        frame_count = codes.shape[1]
        width = self.config.num_audio_codebooks + 1
        tokens = torch.zeros(
            frame_count,
            width,
            dtype=torch.long,
            device=codes.device,
        )
        mask = torch.zeros_like(tokens, dtype=torch.bool)
        tokens[:, :-1] = codes.transpose(0, 1)
        mask[:, :-1] = True
        return tokens, mask, codes.transpose(0, 1)

    def prompt(
            self,
            text: str,
            *,
            speaker: int,
            context: Sequence[CSMCodeSegment] = (),
            device=None,
    ) -> tuple[Tensor, Tensor]:
        token_parts = []
        mask_parts = []
        for segment in context:
            if not isinstance(segment, CSMCodeSegment):
                raise TypeError("CSM context must contain `CSMCodeSegment` values.")
            text_tokens, text_mask = self.text_frames(
                segment.text,
                speaker=segment.speaker,
                device=device,
            )
            audio_tokens, audio_mask, _ = self.audio_frames(
                segment.audio_codes,
                device=device,
            )
            token_parts.extend((text_tokens, audio_tokens))
            mask_parts.extend((text_mask, audio_mask))
        text_tokens, text_mask = self.text_frames(
            text,
            speaker=speaker,
            device=device,
        )
        token_parts.append(text_tokens)
        mask_parts.append(text_mask)
        return (
            torch.cat(token_parts, dim=0).unsqueeze(0),
            torch.cat(mask_parts, dim=0).unsqueeze(0),
        )

    def _record_frames(
        self,
        record: Mapping[str, Any],
        *,
        device=None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        segments = record.get("segments")
        if segments is None:
            if "text" not in record or "audio_codes" not in record:
                raise ValueError(
                    "Native CSM records require `text` and pre-encoded "
                    "`audio_codes`, or a non-empty `segments` sequence.")
            segments = (record, )
        if (isinstance(segments, (str, bytes)) or not isinstance(segments, Sequence) or not segments):
            raise ValueError("CSM `segments` must be a non-empty sequence.")
        token_parts = []
        mask_parts = []
        label_parts = []
        for raw_segment in segments:
            if not isinstance(raw_segment, Mapping):
                raise TypeError("Every CSM training segment must be a mapping.")
            if "text" not in raw_segment or "audio_codes" not in raw_segment:
                raise ValueError(
                    "Every CSM segment requires `text` and pre-encoded "
                    "`audio_codes`; use a runtime with Mimi for raw audio.")
            speaker = raw_segment.get(
                "speaker_id",
                raw_segment.get("speaker", 0),
            )
            text_tokens, text_mask = self.text_frames(
                str(raw_segment["text"]),
                speaker=speaker,
                device=device,
            )
            text_labels = torch.full(
                (
                    text_tokens.shape[0],
                    self.config.num_audio_codebooks,
                ),
                -100,
                dtype=torch.long,
                device=text_tokens.device,
            )
            audio_tokens, audio_mask, audio_labels = self.audio_frames(
                raw_segment["audio_codes"],
                device=device,
            )
            token_parts.extend((text_tokens, audio_tokens))
            mask_parts.extend((text_mask, audio_mask))
            label_parts.extend((text_labels, audio_labels))
        return (
            torch.cat(token_parts, dim=0),
            torch.cat(mask_parts, dim=0),
            torch.cat(label_parts, dim=0),
        )

    def training_batch(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        depth_decoder_labels_ratio: float = 1.0,
        generator: torch.Generator | None = None,
        device=None,
    ) -> dict[str, Tensor]:
        """Collate pre-encoded conversations for the published objective."""
        if (isinstance(records, (str, bytes)) or not isinstance(records, Sequence) or not records):
            raise ValueError("CSM training requires a non-empty record sequence.")
        ratio = float(depth_decoder_labels_ratio)
        if not 0.0 <= ratio <= 1.0:
            raise ValueError("`depth_decoder_labels_ratio` must be between zero and one.")
        prepared = [self._record_frames(record, device=device) for record in records]
        maximum = max(tokens.shape[0] for tokens, _, _ in prepared)
        if maximum > self.config.backbone.max_sequence_length:
            raise ValueError(
                "CSM training sequence exceeds the configured backbone "
                f"context ({maximum} > "
                f"{self.config.backbone.max_sequence_length}).")
        batch_size = len(prepared)
        width = self.config.num_audio_codebooks + 1
        tokens = torch.zeros(
            batch_size,
            maximum,
            width,
            dtype=torch.long,
            device=device,
        )
        token_mask = torch.zeros_like(tokens, dtype=torch.bool)
        labels = torch.full(
            (
                batch_size,
                maximum,
                self.config.num_audio_codebooks,
            ),
            -100,
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros(
            batch_size,
            maximum,
            dtype=torch.bool,
            device=device,
        )
        for index, (item_tokens, item_mask, item_labels) in enumerate(prepared):
            length = item_tokens.shape[0]
            tokens[index, :length] = item_tokens
            token_mask[index, :length] = item_mask
            labels[index, :length] = item_labels
            attention_mask[index, :length] = True
        if ratio < 1.0:
            audio_frames = (labels[..., 0] != -100).nonzero(as_tuple=False)
            skip_count = int(audio_frames.shape[0] * (1.0 - ratio))
            if skip_count:
                permutation = torch.randperm(
                    audio_frames.shape[0],
                    generator=generator,
                    device=audio_frames.device,
                )[:skip_count]
                skipped = audio_frames[permutation]
                labels[skipped[:, 0], skipped[:, 1], 1:] = -100
        return {
            "tokens": tokens,
            "tokens_mask": token_mask,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def save_pretrained(self, directory: str | Path) -> Path:
        return self.tokenizer.save_pretrained(directory)


__all__ = [
    "CSMCodeSegment",
    "CSMProcessor",
    "CSMTextTokenizer",
]
