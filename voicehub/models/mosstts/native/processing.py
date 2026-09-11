"""Native multichannel processors for Delay, Local, Local v1.5, and
Realtime."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from voicehub.models.mosstts.native.configuration import MossTTSConfig
from voicehub.models.mosstts.native.tokenization import IM_END, IM_START, REALTIME_AUDIO_PAD, MossTextTokenizer

AUDIO_PLACEHOLDER = "<|audio|>"
_USER_TEMPLATE = """<user_inst>
- Reference(s):
{reference}
- Instruction:
{instruction}
- Tokens:
{tokens}
- Quality:
{quality}
- Sound Event:
{sound_event}
- Ambient Sound:
{ambient_sound}
- Language:
{language}
- Text:
{text}
</user_inst>"""

_REALTIME_SYSTEM_PROMPT = (
    f"{IM_START}system\n"
    "You are a highly expressive text-to-speech (TTS) engine developed by "
    "Mosi Intelligence. \nYou possess natural language understanding, "
    "emotional modeling, and multi-style speech generation capabilities, "
    "allowing you to generate the corresponding speech based on the text "
    f"given in the assistant.{IM_END}\n")


@dataclass(frozen=True)
class MossProcessorBatch:
    input_ids: Tensor
    attention_mask: Tensor
    labels: Tensor | None = None

    def to_dict(self) -> dict[str, Tensor]:
        output = {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
        }
        if self.labels is not None:
            output["labels"] = self.labels
        return output


@dataclass(frozen=True)
class MossGeneratedCodes:
    prompt_audio_frames: int
    audio_codes: Tensor


@dataclass(frozen=True)
class MossRealtimePrompt:
    input_ids: Tensor
    attention_mask: Tensor
    text_ids: Tensor
    text_cursor: int


def _template_value(value: object | None) -> str:
    return "None" if value is None else str(value)


class MossTTSProcessor:
    """Build the exact release-specific text/audio token matrix.

    This processor consumes explicit ``[time, n_vq]`` integer code
    matrices. :class:`MossTTSRuntime` owns raw-waveform loading and
    native codec encoding before records cross this deterministic token
    boundary.
    """

    def __init__(
        self,
        config: MossTTSConfig,
        tokenizer: MossTextTokenizer,
    ) -> None:
        if not isinstance(config, MossTTSConfig):
            raise TypeError("`config` must be MossTTSConfig.")
        if not isinstance(tokenizer, MossTextTokenizer):
            raise TypeError("`tokenizer` must be MossTextTokenizer.")
        self.config = config
        self.tokenizer = tokenizer

    def _codes(self, value: Tensor, *, name: str) -> Tensor:
        if not isinstance(value, Tensor) or value.ndim != 2:
            raise ValueError(f"`{name}` must have shape [time, {self.config.n_vq}].")
        if value.shape[1] != self.config.n_vq:
            raise ValueError(f"`{name}` requires {self.config.n_vq} codebooks; found "
                             f"{value.shape[1]}.")
        if (value.dtype == torch.bool or value.is_floating_point() or value.is_complex()):
            raise TypeError(f"`{name}` must use an integer dtype.")
        if value.shape[0] < 1:
            raise ValueError(f"`{name}` cannot be empty.")
        if bool(((value < 0) | (value >= self.config.audio_vocab_size)).any()):
            raise ValueError(f"`{name}` contains an out-of-range code.")
        return value.to(dtype=torch.long)

    def _text_rows(
        self,
        token_ids: Sequence[int],
        *,
        device: torch.device | str | None = None,
    ) -> Tensor:
        rows = torch.full(
            (len(token_ids), self.config.channels),
            self.config.audio_pad_token_id,
            dtype=torch.long,
            device=device,
        )
        if token_ids:
            rows[:, 0] = torch.tensor(
                tuple(int(value) for value in token_ids),
                dtype=torch.long,
                device=rows.device,
            )
        return rows

    def _audio_rows(self, audio_codes: Tensor, *, text_token_id: int) -> Tensor:
        audio_codes = self._codes(audio_codes, name="audio_codes")
        rows = torch.empty(
            audio_codes.shape[0],
            self.config.channels,
            dtype=torch.long,
            device=audio_codes.device,
        )
        rows[:, 0] = int(text_token_id)
        rows[:, 1:] = audio_codes
        return rows

    def _render_user(
        self,
        text: str,
        *,
        reference_count: int,
        instruction: str | None,
        duration_tokens: int | None,
        quality: str | None,
        sound_event: str | None,
        ambient_sound: str | None,
        language: str | None,
    ) -> str:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("MOSS-TTS text must be a non-empty string.")
        if (duration_tokens is not None and (isinstance(duration_tokens, bool) or
                                             not isinstance(duration_tokens, int) or duration_tokens <= 0)):
            raise ValueError("`duration_tokens` must be a positive integer.")
        reference = (
            "None" if reference_count == 0 else "\n".join(AUDIO_PLACEHOLDER for _ in range(reference_count)))
        return _USER_TEMPLATE.format(
            reference=reference,
            instruction=_template_value(instruction),
            tokens=_template_value(duration_tokens),
            quality=_template_value(quality),
            sound_event=_template_value(sound_event),
            ambient_sound=_template_value(ambient_sound),
            language=_template_value(language),
            text=text,
        )

    @staticmethod
    def apply_delay_pattern(audio_codes: Tensor, pad_code: int) -> Tensor:
        if not isinstance(audio_codes, Tensor) or audio_codes.ndim != 2:
            raise ValueError("Delay input must have shape [time, codebooks].")
        delayed = torch.full(
            (
                audio_codes.shape[0] + audio_codes.shape[1] - 1,
                audio_codes.shape[1],
            ),
            int(pad_code),
            dtype=audio_codes.dtype,
            device=audio_codes.device,
        )
        for index in range(audio_codes.shape[1]):
            delayed[index:index + audio_codes.shape[0], index] = (audio_codes[:, index])
        return delayed

    @staticmethod
    def remove_delay_pattern(delayed_codes: Tensor) -> Tensor:
        if not isinstance(delayed_codes, Tensor) or delayed_codes.ndim != 2:
            raise ValueError("Delayed codes must have shape [time, codebooks].")
        length = delayed_codes.shape[0] - delayed_codes.shape[1] + 1
        if length <= 0:
            return delayed_codes.new_empty((0, delayed_codes.shape[1]))
        output = delayed_codes.new_empty((length, delayed_codes.shape[1]))
        for index in range(delayed_codes.shape[1]):
            output[:, index] = delayed_codes[
                index:index + length,
                index,
            ]
        return output

    def _delay_audio_block(
        self,
        audio_codes: Tensor,
        *,
        role: str,
    ) -> Tensor:
        audio_codes = self._codes(audio_codes, name="audio_codes")
        if role == "user":
            if self.config.audio_user_slot_token_id is None:
                raise ValueError("Delay config has no user audio-slot token.")
            generation_id = self.config.audio_user_slot_token_id
            delay_id = generation_id
        elif role == "assistant":
            if (self.config.audio_assistant_slot_token_id is None or
                    self.config.audio_assistant_delay_slot_token_id is None):
                raise ValueError("Delay config has incomplete assistant audio tokens.")
            generation_id = self.config.audio_assistant_slot_token_id
            delay_id = self.config.audio_assistant_delay_slot_token_id
        else:
            raise ValueError("MOSS audio role must be user or assistant.")
        delayed = self.apply_delay_pattern(
            audio_codes,
            self.config.audio_pad_token_id,
        )
        text_ids = torch.full(
            (delayed.shape[0], ),
            generation_id,
            dtype=torch.long,
            device=delayed.device,
        )
        if self.config.n_vq > 1:
            text_ids[-(self.config.n_vq - 1):] = delay_id
        return torch.cat([text_ids.unsqueeze(1), delayed], dim=1)

    def _delay_generation_prompt(
        self,
        text: str,
        *,
        reference_codes: Sequence[Tensor],
        instruction: str | None,
        duration_tokens: int | None,
        quality: str | None,
        sound_event: str | None,
        ambient_sound: str | None,
        language: str | None,
    ) -> Tensor:
        references = [
            self._codes(item, name=f"reference_codes[{index}]") for index, item in enumerate(reference_codes)
        ]
        content = self._render_user(
            text,
            reference_count=len(references),
            instruction=instruction,
            duration_tokens=duration_tokens,
            quality=quality,
            sound_event=sound_event,
            ambient_sound=ambient_sound,
            language=language,
        )
        if not references:
            rendered = self.tokenizer.apply_chat_template(
                role="user",
                content=content,
                add_generation_prompt=True,
            )
            return self._text_rows(self.tokenizer.encode_ids(rendered))

        # Replace placeholders structurally so audio matrices never pass
        # through text tokenization.
        parts = content.split(AUDIO_PLACEHOLDER)
        rows: list[Tensor] = []
        prefix = f"{IM_START}user\n"
        for index, (before, reference) in enumerate(zip(parts, references)):
            text_part = prefix + before if index == 0 else before
            rows.append(
                self._text_rows(
                    self.tokenizer.encode_ids(text_part) + [self.config.audio_start_token_id],
                    device=reference.device,
                ))
            rows.append(self._delay_audio_block(reference, role="user"))
            rows.append(self._text_rows(
                [self.config.audio_end_token_id],
                device=reference.device,
            ))
            prefix = ""
        suffix = (parts[-1] + f"{IM_END}\n{IM_START}assistant\n")
        rows.append(self._text_rows(
            self.tokenizer.encode_ids(suffix),
            device=references[0].device,
        ))
        return torch.cat(rows)

    def _local_v15_prompt(
        self,
        text: str,
        *,
        reference_codes: Sequence[Tensor],
        instruction: str | None,
        duration_tokens: int | None,
        quality: str | None,
        sound_event: str | None,
        ambient_sound: str | None,
        language: str | None,
    ) -> Tensor:
        references = [
            self._codes(item, name=f"reference_codes[{index}]") for index, item in enumerate(reference_codes)
        ]
        fields = {
            "instruction": instruction,
            "tokens": duration_tokens,
            "quality": quality,
            "sound_event": sound_event,
            "ambient_sound": ambient_sound,
            "language": language,
        }
        prefix = ([self.config.im_start_token_id] +
                  self.tokenizer.encode_ids("user\n<user_inst>\n- Reference(s):\n"))
        after_reference = (
            "\n- Instruction:\n" + _template_value(fields["instruction"]) + "\n- Tokens:\n" +
            _template_value(fields["tokens"]) + "\n- Quality:\n" + _template_value(fields["quality"]) +
            "\n- Sound Event:\n" + _template_value(fields["sound_event"]) + "\n- Ambient Sound:\n" +
            _template_value(fields["ambient_sound"]) + "\n- Language:\n" +
            _template_value(fields["language"]) + "\n- Text:\n")
        suffix = (
            self.tokenizer.encode_ids(after_reference) + self.tokenizer.encode_ids(text) +
            self.tokenizer.encode_ids("\n</user_inst>") + [self.config.im_end_token_id] +
            self.tokenizer.encode_ids("\n") + [self.config.im_start_token_id] +
            self.tokenizer.encode_ids("assistant\n") + [self.config.audio_start_token_id])
        if not references:
            return self._text_rows(prefix + self.tokenizer.encode_ids("None") + suffix)
        if self.config.audio_user_slot_token_id is None:
            raise ValueError("Local v1.5 config has no user-slot token.")
        rows = [self._text_rows(prefix, device=references[0].device)]
        for reference in references:
            rows.append(self._text_rows(
                [self.config.audio_start_token_id],
                device=reference.device,
            ))
            rows.append(self._audio_rows(
                reference,
                text_token_id=self.config.audio_user_slot_token_id,
            ))
            rows.append(self._text_rows(
                [self.config.audio_end_token_id],
                device=reference.device,
            ))
        rows.append(self._text_rows(suffix, device=references[0].device))
        return torch.cat(rows)

    def build_generation_prompt(
        self,
        text: str,
        *,
        reference_codes: Sequence[Tensor] = (),
        instruction: str | None = None,
        duration_tokens: int | None = None,
        quality: str | None = None,
        sound_event: str | None = None,
        ambient_sound: str | None = None,
        language: str | None = None,
        device: str | torch.device | None = None,
    ) -> MossProcessorBatch:
        # Validate the public fields consistently before choosing a
        # release-specific matrix layout.  The Local v1.5 path assembles its
        # prompt structurally and therefore does not otherwise call
        # ``_render_user``.
        self._render_user(
            text,
            reference_count=len(reference_codes),
            instruction=instruction,
            duration_tokens=duration_tokens,
            quality=quality,
            sound_event=sound_event,
            ambient_sound=ambient_sound,
            language=language,
        )
        if self.config.variant == "realtime":
            raise NotImplementedError(
                "MOSS-TTS-Realtime high-level prompt streaming has not been "
                "differentially audited. Use `build_training_record` for "
                "SFT data; buffered generation fails closed.")
        if self.config.variant == "local_v1_5":
            rows = self._local_v15_prompt(
                text,
                reference_codes=reference_codes,
                instruction=instruction,
                duration_tokens=duration_tokens,
                quality=quality,
                sound_event=sound_event,
                ambient_sound=ambient_sound,
                language=language,
            )
        else:
            rows = self._delay_generation_prompt(
                text,
                reference_codes=reference_codes,
                instruction=instruction,
                duration_tokens=duration_tokens,
                quality=quality,
                sound_event=sound_event,
                ambient_sound=ambient_sound,
                language=language,
            )
        rows = rows.to(device=device)
        return MossProcessorBatch(
            input_ids=rows.unsqueeze(0),
            attention_mask=torch.ones(
                1,
                rows.shape[0],
                dtype=torch.bool,
                device=rows.device,
            ),
        )

    def build_realtime_generation_prompt(
        self,
        text: str,
        *,
        reference_codes: Sequence[Tensor] = (),
        device: str | torch.device | None = None,
        prefill_text_tokens: int = 12,
    ) -> MossRealtimePrompt:
        """Build the audited buffered Realtime prefill state."""
        if self.config.variant != "realtime":
            raise ValueError("Realtime prompt construction requires the Realtime graph.")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("MOSS-TTS text must be non-empty.")
        if len(reference_codes) > 1:
            raise ValueError("Realtime MOSS accepts at most one reference.")
        if (isinstance(prefill_text_tokens, bool) or not isinstance(prefill_text_tokens, int) or
                prefill_text_tokens <= 0):
            raise ValueError("`prefill_text_tokens` must be a positive integer.")
        if self.config.reference_audio_pad_token_id is None:
            raise ValueError("Realtime config has no reference-audio pad token.")

        system_text = _REALTIME_SYSTEM_PROMPT
        reference = None
        if reference_codes:
            reference = self._codes(
                reference_codes[0],
                name="reference_codes[0]",
            )
            system_text += (
                f"{IM_START}context\n"
                "The assistant section should be synthesized using the "
                "following voice timbre:" + REALTIME_AUDIO_PAD * reference.shape[0] + f"{IM_END}\n")
        system_ids = self.tokenizer.encode_ids(system_text)
        system = self._text_rows(
            system_ids,
            device=reference.device if reference is not None else None,
        )
        if reference is not None:
            positions = torch.tensor(
                [
                    index for index, token_id in enumerate(system_ids)
                    if token_id == self.config.reference_audio_pad_token_id
                ],
                dtype=torch.long,
                device=system.device,
            )
            if positions.numel() != reference.shape[0]:
                raise ValueError("Realtime reference placeholders do not match codec frames.")
            system[positions, 1:] = reference.to(system.device)

        assistant = self._text_rows(
            self.tokenizer.encode_ids(f"{IM_START}assistant\n"),
            device=system.device,
        )
        text_ids = torch.tensor(
            self.tokenizer.encode_ids(text),
            dtype=torch.long,
            device=system.device,
        )
        if text_ids.numel() == 0:
            raise ValueError("Realtime text produced no tokenizer IDs.")
        cursor = min(text_ids.numel(), prefill_text_tokens)
        text_prefill = torch.full(
            (cursor, self.config.channels),
            self.config.audio_pad_token_id,
            dtype=torch.long,
            device=system.device,
        )
        text_prefill[:, 0] = text_ids[:cursor]
        text_prefill[-1, 1] = 1_025
        rows = torch.cat([system, assistant, text_prefill]).to(device=device)
        text_ids = text_ids.to(device=rows.device)
        return MossRealtimePrompt(
            input_ids=rows.unsqueeze(0),
            attention_mask=torch.ones(
                1,
                rows.shape[0],
                dtype=torch.bool,
                device=rows.device,
            ),
            text_ids=text_ids,
            text_cursor=int(cursor),
        )

    def _assistant_target(self, audio_codes: Tensor) -> Tensor:
        audio_codes = self._codes(audio_codes, name="speech_tokens")
        if self.config.audio_assistant_slot_token_id is None:
            raise ValueError("MOSS config has no assistant audio-slot token.")
        if self.config.variant in {"delay", "local"}:
            return torch.cat([
                self._text_rows(
                    [self.config.audio_start_token_id],
                    device=audio_codes.device,
                ),
                self._delay_audio_block(audio_codes, role="assistant"),
                self._text_rows(
                    [
                        self.config.audio_end_token_id,
                        self.config.im_end_token_id,
                    ],
                    device=audio_codes.device,
                ),
            ])
        if self.config.variant == "local_v1_5":
            return torch.cat([
                self._audio_rows(
                    audio_codes,
                    text_token_id=self.config.audio_assistant_slot_token_id,
                ),
                self._text_rows(
                    [
                        self.config.audio_end_token_id,
                        self.config.im_end_token_id,
                    ],
                    device=audio_codes.device,
                ),
            ])
        raise ValueError("Realtime uses its aligned training processor.")

    def _realtime_training_rows(
        self,
        text: str,
        audio_codes: Tensor,
        *,
        reference_codes: Sequence[Tensor],
    ) -> tuple[Tensor, int]:
        audio_codes = self._codes(audio_codes, name="speech_tokens")
        if len(reference_codes) > 1:
            raise ValueError("Realtime MOSS accepts at most one reference code matrix.")
        system_text = _REALTIME_SYSTEM_PROMPT
        reference = None
        if reference_codes:
            reference = self._codes(
                reference_codes[0],
                name="reference_codes[0]",
            )
            system_text += (
                f"{IM_START}context\n"
                "The assistant section should be synthesized using the "
                "following voice timbre:" + REALTIME_AUDIO_PAD * reference.shape[0] + f"{IM_END}\n")
        system_ids = self.tokenizer.encode_ids(system_text)
        system = self._text_rows(
            system_ids,
            device=audio_codes.device,
        )
        if reference is not None:
            positions = torch.tensor(
                [
                    index for index, token_id in enumerate(system_ids)
                    if token_id == self.config.reference_audio_pad_token_id
                ],
                device=system.device,
            )
            if positions.numel() != reference.shape[0]:
                raise ValueError("Realtime reference placeholders do not match codes.")
            system[positions, 1:] = reference

        prefill = f"{IM_END}\n{IM_START}user\n"
        response = f"{IM_END}\n{IM_START}assistant\n"
        text_ids = self.tokenizer.encode_ids(prefill + text)
        response_ids = self.tokenizer.encode_ids(response)
        delay = 12
        padding_count = max(
            audio_codes.shape[0] + delay - len(self.tokenizer.encode_ids(text)) + 1,
            audio_codes.shape[0] + 1,
        )
        text_ids += [self.config.text_pad_token_id] * padding_count
        user = self._text_rows(text_ids, device=audio_codes.device)
        text_start = len(self.tokenizer.encode_ids(prefill))
        audio_start = (
            text_start + delay if len(self.tokenizer.encode_ids(text)) >= delay else user.shape[0] -
            audio_codes.shape[0] - 1)
        user[
            audio_start:audio_start + audio_codes.shape[0],
            1:,
        ] = audio_codes
        # Official processor marks BOS/EOS only in the first audio channel.
        if audio_start > 0:
            user[audio_start - 1, 1] = 1025
        if audio_start + audio_codes.shape[0] < user.shape[0]:
            user[audio_start + audio_codes.shape[0], 1] = 1026
        response_rows = self._text_rows(
            response_ids,
            device=audio_codes.device,
        )
        prompt_length = system.shape[0] + audio_start
        return torch.cat([system, user, response_rows]), prompt_length

    def build_training_record(
        self,
        *,
        text: str,
        speech_tokens: Tensor,
        reference_codes: Sequence[Tensor] = (),
        instruction: str | None = None,
        duration_tokens: int | None = None,
        quality: str | None = None,
        sound_event: str | None = None,
        ambient_sound: str | None = None,
        language: str | None = None,
        device: str | torch.device | None = None,
    ) -> MossProcessorBatch:
        """Prepare source-style next-token labels from pre-encoded audio."""
        speech_tokens = self._codes(
            speech_tokens,
            name="speech_tokens",
        )
        if self.config.variant == "realtime":
            full, prompt_length = self._realtime_training_rows(
                text,
                speech_tokens,
                reference_codes=reference_codes,
            )
        else:
            prompt = self.build_generation_prompt(
                text,
                reference_codes=reference_codes,
                instruction=instruction,
                duration_tokens=duration_tokens,
                quality=quality,
                sound_event=sound_event,
                ambient_sound=ambient_sound,
                language=language,
            ).input_ids[0].to(speech_tokens.device)
            prompt_length = prompt.shape[0]
            full = torch.cat([prompt, self._assistant_target(speech_tokens)])
        full = full.to(device=device)
        input_ids = full[:-1]
        labels = full[1:].clone()
        labels[:max(prompt_length - 1, 0)] = -100
        labels[:, 1:] = labels[:, 1:].masked_fill(
            labels[:, 1:].eq(self.config.audio_pad_token_id),
            -100,
        )
        attention_mask = torch.ones(
            1,
            input_ids.shape[0],
            dtype=torch.bool,
            device=input_ids.device,
        )
        return MossProcessorBatch(
            input_ids=input_ids.unsqueeze(0),
            attention_mask=attention_mask,
            labels=labels.unsqueeze(0),
        )

    def collate_training(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        device: str | torch.device | None = None,
    ) -> MossProcessorBatch:
        if not records:
            raise ValueError("MOSS-TTS training batch cannot be empty.")
        prepared = [
            self.build_training_record(
                text=str(record["text"]),
                speech_tokens=record["speech_tokens"],
                reference_codes=record.get("reference_codes", ()),
                instruction=record.get("instruction"),
                duration_tokens=record.get("duration_tokens"),
                quality=record.get("quality"),
                sound_event=record.get("sound_event"),
                ambient_sound=record.get("ambient_sound"),
                language=record.get("language"),
                device=device,
            ) for record in records
        ]
        lengths = torch.tensor(
            [item.input_ids.shape[1] for item in prepared],
            device=prepared[0].input_ids.device,
        )
        maximum = int(lengths.max().item())
        batch_size = len(prepared)
        input_ids = torch.full(
            (batch_size, maximum, self.config.channels),
            self.config.audio_pad_token_id,
            dtype=torch.long,
            device=lengths.device,
        )
        input_ids[..., 0] = self.config.pad_token_id
        labels = torch.full_like(input_ids, -100)
        mask = torch.zeros(
            batch_size,
            maximum,
            dtype=torch.bool,
            device=lengths.device,
        )
        for index, item in enumerate(prepared):
            length = item.input_ids.shape[1]
            offset = maximum - length
            input_ids[index, offset:] = item.input_ids[0]
            labels[index, offset:] = item.labels[0]
            mask[index, offset:] = True
        return MossProcessorBatch(
            input_ids=input_ids,
            attention_mask=mask,
            labels=labels,
        )

    def decode_generated(
        self,
        output: Sequence[tuple[int, Tensor]],
    ) -> tuple[MossGeneratedCodes, ...]:
        decoded: list[MossGeneratedCodes] = []
        for prompt_frames, sequence in output:
            if sequence.ndim != 2 or sequence.shape[1] != self.config.channels:
                raise ValueError("Generated MOSS sequence has an invalid channel layout.")
            audio = sequence[:, 1:]
            if self.config.variant in {"delay", "local"}:
                audio = self.remove_delay_pattern(audio)
            non_padding = ~audio.eq(self.config.audio_pad_token_id).all(dim=1)
            indices = torch.where(non_padding)[0]
            if indices.numel() == 0:
                codes = audio.new_empty((0, self.config.n_vq))
            else:
                codes = audio[indices]
            trim = min(max(int(prompt_frames), 0), codes.shape[0])
            decoded.append(MossGeneratedCodes(
                prompt_audio_frames=trim,
                audio_codes=codes[trim:].detach(),
            ))
        return tuple(decoded)


__all__ = [
    "AUDIO_PLACEHOLDER",
    "MossGeneratedCodes",
    "MossProcessorBatch",
    "MossRealtimePrompt",
    "MossTTSProcessor",
]
