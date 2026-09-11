"""Native Dia byte tokenization, audio preparation, and delay patterns."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from voicehub.models.dac.native.modeling import DacModel
from voicehub.models.dia.native.configuration import DiaArchitectureConfig
from voicehub.processing.waveform import load_native_audio


class DiaBatch(dict[str, Tensor]):
    """Tensor mapping with the small device-transfer API expected by
    wrappers."""

    def to(
        self,
        device: str | torch.device,
        *,
        non_blocking: bool = False,
    ) -> DiaBatch:
        return DiaBatch({
            name: value.to(
                device=device,
                non_blocking=non_blocking,
            )
            for name, value in self.items()
        })


class DiaByteTokenizer:
    """UTF-8 byte tokenizer with Dia's two speaker control tokens."""

    pad_token_id = 0
    unk_token_id = 0
    speaker_1_token_id = 1
    speaker_2_token_id = 2
    vocab_size = 256
    _CONTROLS = {
        "<pad>": pad_token_id,
        "[S1]": speaker_1_token_id,
        "[S2]": speaker_2_token_id,
    }

    def __init__(self, *, max_length: int = 1_024) -> None:
        if (isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 1):
            raise ValueError("Dia tokenizer `max_length` must be positive.")
        self.max_length = max_length

    def encode(self, text: str) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("Dia text must be a string.")
        if not text:
            raise ValueError("Dia text cannot be empty.")
        token_ids: list[int] = []
        cursor = 0
        while cursor < len(text):
            control = next(
                (value for token, value in self._CONTROLS.items() if text.startswith(token, cursor)),
                None,
            )
            if control is not None:
                token_ids.append(control)
                matched = next(
                    token for token, value in self._CONTROLS.items()
                    if value == control and text.startswith(token, cursor))
                cursor += len(matched)
                continue
            next_control = min(
                (index for token in self._CONTROLS if (index := text.find(token, cursor)) >= 0),
                default=len(text),
            )
            chunk = text[cursor:next_control]
            token_ids.extend(chunk.encode("utf-8"))
            cursor = next_control
        if len(token_ids) > self.max_length:
            raise ValueError(
                f"Dia text encodes to {len(token_ids)} bytes/tokens, exceeding "
                f"the {self.max_length}-token limit.")
        return token_ids

    def encode_batch(
        self,
        texts: str | Sequence[str],
        *,
        padding: bool = True,
    ) -> DiaBatch:
        texts_are_sequence = isinstance(texts, Sequence) and not isinstance(
            texts,
            (bytes, bytearray),
        )
        if isinstance(texts, str):
            texts = (texts, )
        elif texts_are_sequence:
            texts = tuple(texts)
        else:
            raise TypeError("Dia text must be a string or sequence of strings.")
        if not texts:
            raise ValueError("Dia text batch cannot be empty.")
        encoded = tuple(self.encode(text) for text in texts)
        lengths = tuple(len(row) for row in encoded)
        if not padding and len(set(lengths)) != 1:
            raise ValueError("Unpadded Dia batches require equal encoded text lengths.")
        width = max(lengths)
        input_ids = torch.full(
            (len(encoded), width),
            self.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros_like(input_ids)
        for index, row in enumerate(encoded):
            input_ids[index, :len(row)] = torch.tensor(row, dtype=torch.long)
            attention_mask[index, :len(row)] = 1
        return DiaBatch({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        })

    def decode(self, token_ids: Sequence[int]) -> str:
        output = bytearray()
        pieces: list[str] = []
        controls = {
            self.speaker_1_token_id: "[S1]",
            self.speaker_2_token_id: "[S2]",
        }

        def flush() -> None:
            if output:
                pieces.append(bytes(output).decode("utf-8", errors="ignore"))
                output.clear()

        for value in token_ids:
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError("Dia token IDs must be integers.")
            if value in controls:
                flush()
                pieces.append(controls[value])
            elif value == self.pad_token_id:
                continue
            elif 0 <= value < self.vocab_size:
                output.append(value)
            else:
                raise ValueError(f"Dia token ID {value} is outside byte range.")
        flush()
        return "".join(pieces)


@dataclass
class DiaProcessor:
    """Prepare text and native DAC tokens for inference or teacher forcing."""

    config: DiaArchitectureConfig
    audio_tokenizer: DacModel | None = None
    tokenizer: DiaByteTokenizer | None = None
    sampling_rate: int = 44_100
    hop_length: int = 512

    def __post_init__(self) -> None:
        self.config = DiaArchitectureConfig.coerce(self.config)
        if self.tokenizer is None:
            self.tokenizer = DiaByteTokenizer(max_length=self.config.encoder_config.max_position_embeddings, )
        if (isinstance(self.sampling_rate, bool) or not isinstance(self.sampling_rate, int) or
                self.sampling_rate < 1):
            raise ValueError("Dia processor sampling rate must be positive.")
        if (isinstance(self.hop_length, bool) or not isinstance(self.hop_length, int) or self.hop_length < 1):
            raise ValueError("Dia processor hop length must be positive.")
        if self.audio_tokenizer is not None:
            codec_config = self.audio_tokenizer.config
            if codec_config.n_codebooks != self.config.decoder_config.num_channels:
                raise ValueError(
                    "Dia requires a DAC tokenizer with exactly "
                    f"{self.config.decoder_config.num_channels} codebooks.")
            if codec_config.codebook_size != self.config.decoder_config.eos_token_id:
                raise ValueError("Dia DAC codebook size must equal the audio EOS token ID.")
            if codec_config.sampling_rate != self.sampling_rate:
                raise ValueError("Dia and DAC sampling rates must match exactly.")

    @property
    def device(self) -> torch.device:
        if self.audio_tokenizer is None:
            return torch.device("cpu")
        return next(self.audio_tokenizer.parameters()).device

    def freeze_audio_tokenizer(self) -> DacModel:
        codec = self._require_audio_tokenizer()
        codec.requires_grad_(False)
        codec.eval()
        return codec

    def _require_audio_tokenizer(self) -> DacModel:
        if self.audio_tokenizer is None:
            raise RuntimeError("Dia audio processing requires the native 44.1 kHz DAC "
                               "tokenizer.")
        return self.audio_tokenizer

    @staticmethod
    def build_indices(
        batch_size: int,
        sequence_length: int,
        num_channels: int,
        delay_pattern: Sequence[int],
        *,
        revert: bool = False,
    ) -> tuple[Tensor, Tensor]:
        for name, value in (
            ("batch_size", batch_size),
            ("sequence_length", sequence_length),
            ("num_channels", num_channels),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"`{name}` must be a positive integer.")
        delays = tuple(delay_pattern)
        if len(delays) != num_channels:
            raise ValueError("Delay pattern length must equal the number of channels.")
        delay_array = torch.tensor(delays, dtype=torch.int32)
        sequence = torch.arange(sequence_length, dtype=torch.int32)
        sequence = sequence[None, :, None].expand(
            batch_size,
            sequence_length,
            num_channels,
        )
        sequence = (sequence + delay_array[None, None] if revert else sequence - delay_array[None, None])
        valid_sequence = sequence.clamp(0, sequence_length - 1)
        batch = torch.arange(batch_size, dtype=torch.int32)
        batch = batch[:, None, None].expand_as(sequence)
        channels = torch.arange(num_channels, dtype=torch.int32)
        channels = channels[None, None].expand_as(sequence)
        indices = torch.stack(
            (
                batch.reshape(-1),
                valid_sequence.reshape(-1),
                channels.reshape(-1),
            ),
            dim=1,
        ).long()
        return sequence, indices

    @staticmethod
    def apply_audio_delay(
        audio: Tensor,
        *,
        pad_token_id: int,
        bos_token_id: int,
        precomputed_indices: tuple[Tensor, Tensor],
    ) -> Tensor:
        if audio.ndim != 3:
            raise ValueError("Dia audio codes must have shape [batch, sequence, channels].")
        sequence, all_indices = precomputed_indices
        sequence = sequence.to(audio.device)
        all_indices = all_indices.to(audio.device)
        batch, valid_sequence, channels = all_indices.unbind(dim=-1)
        gathered = audio[batch, valid_sequence, channels].view_as(audio)
        return torch.where(
            sequence < 0,
            bos_token_id,
            torch.where(
                sequence >= audio.shape[1],
                pad_token_id,
                gathered,
            ),
        )

    def _normalize_audio_batch(
        self,
        audio: Any,
        *,
        batch_size: int,
    ) -> tuple[Tensor, ...]:
        audio_is_sequence = isinstance(audio, Sequence) and not isinstance(
            audio,
            (bytes, bytearray),
        )
        if isinstance(audio, (str, Path, Mapping, Tensor)):
            values = (audio, )
        elif audio_is_sequence:
            # A flat numeric sequence is one waveform. Nested values are a
            # waveform batch.
            values = ((audio, ) if not audio or isinstance(audio[0], (int, float)) else tuple(audio))
        else:
            raise TypeError("Dia audio must be a waveform, audio mapping, path, or batch.")
        if len(values) != batch_size:
            raise ValueError("Dia requires the same number of text and audio samples.")
        waveforms = []
        for value in values:
            native = load_native_audio(
                value,
                target_sampling_rate=self.sampling_rate,
                sampling_rate=(
                    value.get("sampling_rate") if isinstance(value, Mapping) else
                    self.sampling_rate if not isinstance(value, (str, Path)) else None),
            )
            waveforms.append(native.waveform)
        return tuple(waveforms)

    def _encode_audio(
        self,
        audio: Any,
        *,
        batch_size: int,
        generation: bool,
    ) -> tuple[Tensor, Tensor]:
        codec = self._require_audio_tokenizer()
        waveforms = self._normalize_audio_batch(
            audio,
            batch_size=batch_size,
        )
        compression_rate = codec.config.hop_length
        padded_lengths = tuple(
            math.ceil(waveform.shape[-1] / self.hop_length) * self.hop_length for waveform in waveforms)
        maximum_length = max(padded_lengths)
        maximum_encoded_length = maximum_length // compression_rate
        maximum_delay = max(self.config.delay_pattern)
        input_rows = []
        attention_rows = []
        codec_device = next(codec.parameters()).device
        with torch.no_grad():
            for waveform, padded_length in zip(waveforms, padded_lengths):
                sample = F.pad(
                    waveform,
                    (0, padded_length - waveform.shape[-1]),
                )
                output = codec.encode_output(sample[None, None].to(codec_device), )
                codes = output.audio_codes.transpose(1, 2).cpu()
                encoded_length = codes.shape[1]
                padding_length = maximum_encoded_length - encoded_length
                if not generation:
                    codes = F.pad(
                        codes,
                        (0, 0, 0, 1),
                        value=self.config.decoder_config.eos_token_id,
                    )
                codes = F.pad(
                    codes,
                    (0, 0, padding_length + 1, 0),
                    value=self.config.decoder_config.bos_token_id,
                )
                valid = encoded_length + 1 + maximum_delay
                if not generation:
                    valid += 1
                mask = torch.tensor(
                    [0] * padding_length + [1] * valid,
                    dtype=torch.long,
                )[None]
                input_rows.append(codes)
                attention_rows.append(mask)
        return torch.cat(input_rows), torch.cat(attention_rows)

    def __call__(
        self,
        *,
        text: str | Sequence[str],
        audio: Any | None = None,
        generation: bool = True,
        output_labels: bool = False,
        padding: bool = True,
        return_tensors: str = "pt",
        **kwargs: Any,
    ) -> DiaBatch:
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(f"Unsupported native Dia processor options: {names}.")
        if return_tensors != "pt":
            raise ValueError("Native Dia supports `return_tensors='pt'` only.")
        if not isinstance(generation, bool) or not isinstance(output_labels, bool):
            raise TypeError("Dia generation and output_labels must be booleans.")
        if generation and output_labels:
            raise ValueError("Dia cannot create labels during generation.")
        if not generation and audio is None:
            raise ValueError("Dia teacher forcing requires target audio.")
        text_inputs = self.tokenizer.encode_batch(text, padding=padding)
        batch_size = text_inputs["input_ids"].shape[0]
        maximum_delay = max(self.config.delay_pattern)
        decoder = self.config.decoder_config

        if audio is None:
            decoder_ids = torch.full(
                (batch_size, 1, decoder.num_channels),
                decoder.bos_token_id,
                dtype=torch.long,
            )
            decoder_mask = torch.ones(
                batch_size,
                1 + maximum_delay,
                dtype=torch.long,
            )
        else:
            decoder_ids, decoder_mask = self._encode_audio(
                audio,
                batch_size=batch_size,
                generation=generation,
            )

        maximum_sequence = decoder_mask.shape[1]
        maximum_audio = maximum_sequence - maximum_delay
        prefill = torch.full(
            (batch_size, maximum_sequence, decoder.num_channels),
            decoder.pad_token_id,
            dtype=torch.long,
        )
        prefill[:, :maximum_audio] = decoder_ids
        indices = self.build_indices(
            batch_size,
            maximum_sequence,
            decoder.num_channels,
            self.config.delay_pattern,
        )
        delayed = self.apply_audio_delay(
            prefill,
            pad_token_id=decoder.pad_token_id,
            bos_token_id=decoder.bos_token_id,
            precomputed_indices=indices,
        )
        result = DiaBatch(text_inputs)
        result.update({
            "decoder_input_ids": delayed,
            "decoder_attention_mask": decoder_mask,
        })
        if output_labels:
            labels = delayed[:, 1:].clone()
            labels[labels == decoder.pad_token_id] = -100
            labels[labels == decoder.bos_token_id] = -100
            result["labels"] = labels.transpose(1, 2).reshape(
                batch_size * decoder.num_channels,
                -1,
            ).contiguous().long()
            result["decoder_input_ids"] = delayed[:, :-1]
            result["decoder_attention_mask"] = decoder_mask[:, :-1]
        return result

    def get_audio_prompt_len(self, decoder_attention_mask: Tensor) -> int:
        if decoder_attention_mask.ndim != 2:
            raise ValueError("Dia decoder attention mask must have shape [batch, sequence].")
        return decoder_attention_mask.shape[1] - max(self.config.delay_pattern)

    def batch_decode(
        self,
        decoder_input_ids: Tensor,
        *,
        audio_prompt_len: int | None = None,
    ) -> list[Tensor]:
        codec = self._require_audio_tokenizer()
        if decoder_input_ids.ndim != 3:
            raise ValueError("Dia decoder output must have shape [batch, sequence, channels].")
        decoder = self.config.decoder_config
        if audio_prompt_len is None:
            starts = (decoder_input_ids[:, :, 0] == decoder.bos_token_id).sum(dim=-1)
        else:
            if (isinstance(audio_prompt_len, bool) or not isinstance(audio_prompt_len, int) or
                    audio_prompt_len < 0):
                raise ValueError("`audio_prompt_len` must be non-negative.")
            starts = torch.full(
                (decoder_input_ids.shape[0], ),
                audio_prompt_len,
                dtype=torch.long,
                device=decoder_input_ids.device,
            )
        ends = (
            decoder_input_ids.shape[1] - (decoder_input_ids[:, :, 0] == decoder.pad_token_id).sum(dim=-1) - 1)
        indices = self.build_indices(
            decoder_input_ids.shape[0],
            decoder_input_ids.shape[1],
            decoder_input_ids.shape[2],
            self.config.delay_pattern,
            revert=True,
        )
        undelayed = self.apply_audio_delay(
            decoder_input_ids,
            pad_token_id=-1,
            bos_token_id=-1,
            precomputed_indices=indices,
        ).transpose(1, 2)
        device = next(codec.parameters()).device
        audios = []
        with torch.no_grad():
            for index in range(undelayed.shape[0]):
                start = int(starts[index].item())
                end = int(ends[index].item())
                if end <= start:
                    raise ValueError("Dia generated no complete DAC frames to decode.")
                codes = undelayed[index:index + 1, :, start:end].to(device)
                if ((codes < 0).any() or (codes >= codec.config.codebook_size).any()):
                    raise ValueError("Dia output contains a special token inside decoded "
                                     "DAC frames.")
                audio = codec.decode_codes(codes.long()).cpu().squeeze(0).squeeze(0)
                audios.append(audio)
        return audios

    def decode(
        self,
        decoder_input_ids: Tensor,
        *,
        audio_prompt_len: int | None = None,
    ) -> Tensor:
        if decoder_input_ids.shape[0] != 1:
            raise ValueError("Dia decode() requires exactly one sequence.")
        return self.batch_decode(
            decoder_input_ids,
            audio_prompt_len=audio_prompt_len,
        )[0]


__all__ = [
    "DiaBatch",
    "DiaByteTokenizer",
    "DiaProcessor",
]
