"""Dependency-free waveform and prompt processing for VibeVoice."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from voicehub.models.vibevoice.native.metadata import VIBEVOICE_HOP_LENGTH, VIBEVOICE_SAMPLE_RATE
from voicehub.models.vibevoice.native.tokenization import (
    ASR_AUDIO,
    ASR_AUDIO_END,
    ASR_AUDIO_START,
    IM_END,
    IM_START,
    VibeVoiceTokenizer,
)

_ASR_SYSTEM_PROMPT = (
    "You are a helpful assistant that transcribes audio input into text "
    "output in JSON format.")
_TTS_SYSTEM_PROMPT = (
    " Transform the text provided by various speakers into speech output, "
    "utilizing the distinct voice of each respective speaker.\n")
_SPEAKER_LINE = re.compile(
    r"^Speaker\s+(\d+)\s*:\s*(.*)$",
    flags=re.IGNORECASE,
)
_NON_SEQUENCE_AUDIO_TYPES = (str, bytes, bytearray)


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"`{name}` must be a positive integer.")
    return value


def _as_mono_waveform(value: Any) -> Tensor:
    try:
        waveform = torch.as_tensor(value).detach()
    except (TypeError, ValueError, RuntimeError) as error:
        raise TypeError("VibeVoice audio must contain numeric samples.") from error
    if waveform.ndim == 2 and 1 in waveform.shape:
        waveform = waveform.squeeze()
    if waveform.ndim != 1:
        raise ValueError("Each VibeVoice waveform must be mono with shape [samples].")
    if waveform.numel() == 0:
        raise ValueError("VibeVoice audio cannot be empty.")
    if waveform.dtype == torch.bool or waveform.is_complex():
        raise TypeError("VibeVoice audio must contain real-valued samples.")
    waveform = waveform.to(dtype=torch.float32, device="cpu").contiguous()
    if not bool(torch.isfinite(waveform).all()):
        raise ValueError("VibeVoice audio contains NaN or infinite samples.")
    return waveform


def _audio_batch(value: Any) -> tuple[Tensor, ...]:
    if isinstance(value, Tensor):
        if value.ndim == 1:
            return (_as_mono_waveform(value), )
        if value.ndim == 2:
            return tuple(_as_mono_waveform(row) for row in value)
        raise ValueError("VibeVoice audio tensor must have shape [samples] or [batch, samples].")
    if isinstance(value, Sequence) and not isinstance(
            value,
            _NON_SEQUENCE_AUDIO_TYPES,
    ):
        if not value:
            raise ValueError("VibeVoice audio batch cannot be empty.")
        first = value[0]
        if isinstance(first, (int, float)):
            return (_as_mono_waveform(value), )
        return tuple(_as_mono_waveform(item) for item in value)
    return (_as_mono_waveform(value), )


@dataclass(frozen=True, slots=True)
class VibeVoiceAudioBatch:
    input_values: Tensor
    padding_mask: Tensor
    sample_lengths: Tensor


class VibeVoiceAudioProcessor:
    """Published -25 dBFS normalization and 3,200-sample padding."""

    def __init__(
        self,
        *,
        sample_rate: int = VIBEVOICE_SAMPLE_RATE,
        hop_length: int = VIBEVOICE_HOP_LENGTH,
        normalize_audio: bool = True,
        target_dbfs: float = -25.0,
        epsilon: float = 1e-6,
    ) -> None:
        self.sample_rate = _positive_integer(
            sample_rate,
            name="sample_rate",
        )
        self.hop_length = _positive_integer(
            hop_length,
            name="hop_length",
        )
        if not isinstance(normalize_audio, bool):
            raise TypeError("`normalize_audio` must be a boolean.")
        if (isinstance(target_dbfs, bool) or not isinstance(target_dbfs, (int, float)) or
                not math.isfinite(float(target_dbfs))):
            raise ValueError("`target_dbfs` must be finite.")
        if (isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)) or
                not math.isfinite(float(epsilon)) or epsilon <= 0):
            raise ValueError("`epsilon` must be finite and positive.")
        self.normalize_audio = normalize_audio
        self.target_dbfs = float(target_dbfs)
        self.epsilon = float(epsilon)

    def normalize(self, waveform: Tensor) -> Tensor:
        waveform = _as_mono_waveform(waveform).clone()
        if not self.normalize_audio:
            return waveform
        rms = waveform.square().mean().sqrt()
        scale = (10.0**(self.target_dbfs / 20.0)) / (rms + self.epsilon)
        waveform.mul_(scale)
        peak = waveform.abs().max()
        if bool(peak > 1.0):
            waveform.div_(peak + self.epsilon)
        return waveform

    def __call__(
        self,
        audio: Any,
        *,
        sampling_rate: int,
        pad_to_multiple_of: int | None = None,
    ) -> VibeVoiceAudioBatch:
        if sampling_rate != self.sample_rate:
            raise ValueError(
                f"VibeVoice requires {self.sample_rate} Hz audio; received "
                f"{sampling_rate} Hz. Resample explicitly before processing.")
        rows = tuple(self.normalize(row) for row in _audio_batch(audio))
        multiple = (
            self.hop_length if pad_to_multiple_of is None else _positive_integer(
                pad_to_multiple_of,
                name="pad_to_multiple_of",
            ))
        lengths = torch.tensor(
            [row.numel() for row in rows],
            dtype=torch.long,
        )
        width = int(lengths.max().item())
        width = ((width + multiple - 1) // multiple) * multiple
        values = torch.zeros(
            len(rows),
            1,
            width,
            dtype=torch.float32,
        )
        mask = torch.zeros(
            len(rows),
            width,
            dtype=torch.long,
        )
        for index, row in enumerate(rows):
            length = row.numel()
            values[index, 0, :length] = row
            mask[index, :length] = 1
        return VibeVoiceAudioBatch(
            input_values=values,
            padding_mask=mask,
            sample_lengths=lengths,
        )


@dataclass(frozen=True, slots=True)
class VibeVoiceASRBatch:
    input_ids: Tensor
    attention_mask: Tensor
    input_values: Tensor
    padding_mask: Tensor
    labels: Tensor | None = None

    def as_dict(self) -> dict[str, Tensor]:
        result = {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
            "input_values": self.input_values,
            "padding_mask": self.padding_mask,
        }
        if self.labels is not None:
            result["labels"] = self.labels
        return result


def render_vibevoice_asr_prompt(
    *,
    audio_tokens: int,
    duration_seconds: float,
    context: str | None = None,
    system_prompt: str = _ASR_SYSTEM_PROMPT,
) -> str:
    """Render the audited checkpoint chat template without Jinja execution."""
    _positive_integer(audio_tokens, name="audio_tokens")
    if (isinstance(duration_seconds, bool) or not isinstance(duration_seconds, (int, float)) or
            not math.isfinite(float(duration_seconds)) or duration_seconds <= 0):
        raise ValueError("ASR audio duration must be finite and positive.")
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise ValueError("ASR system prompt must be a non-empty string.")
    if context is not None and not isinstance(context, str):
        raise TypeError("ASR context must be a string or None.")
    audio_placeholder = ASR_AUDIO * audio_tokens
    if context:
        request = (
            f"This is a {duration_seconds:.2f} seconds audio, with extra "
            f"info: {context}\n\n"
            "Please transcribe it with these keys: Start time, End time, "
            "Speaker ID, Content")
    else:
        request = (
            f"This is a {duration_seconds:.2f} seconds audio, please "
            "transcribe it with these keys: Start time, End time, Speaker ID, "
            "Content")
    return (
        f"{IM_START}system\n{system_prompt}{IM_END}\n"
        f"{IM_START}user\n"
        f"{ASR_AUDIO_START}{audio_placeholder}{ASR_AUDIO_END}\n"
        f"{request}{IM_END}\n")


class VibeVoiceASRProcessor:
    """Exact prompt expansion and training-label masking for ASR-HF."""

    def __init__(
        self,
        tokenizer: VibeVoiceTokenizer,
        *,
        audio_processor: VibeVoiceAudioProcessor | None = None,
    ) -> None:
        if not isinstance(tokenizer, VibeVoiceTokenizer):
            raise TypeError("ASR processor requires a VibeVoiceTokenizer.")
        self.tokenizer = tokenizer
        self.audio_processor = audio_processor or VibeVoiceAudioProcessor()

    def __call__(
        self,
        audio: Any,
        *,
        sampling_rate: int,
        prompt: str | Sequence[str | None] | None = None,
        output_labels: bool = False,
    ) -> VibeVoiceASRBatch:
        audio_batch = self.audio_processor(
            audio,
            sampling_rate=sampling_rate,
            pad_to_multiple_of=self.audio_processor.hop_length,
        )
        batch_size = audio_batch.input_values.shape[0]
        if prompt is None or isinstance(prompt, str):
            prompts = (prompt, ) * batch_size
        elif isinstance(prompt, Sequence):
            prompts = tuple(prompt)
            if len(prompts) != batch_size:
                raise ValueError("ASR prompt count must match the audio batch size.")
            if any(value is not None and not isinstance(value, str) for value in prompts):
                raise TypeError("ASR prompts must be strings or None.")
        else:
            raise TypeError("ASR prompt must be a string, sequence, or None.")

        rendered: list[str] = []
        for length, context in zip(
                audio_batch.sample_lengths.tolist(),
                prompts,
        ):
            tokens = math.ceil(length / self.audio_processor.hop_length)
            rendered.append(
                render_vibevoice_asr_prompt(
                    audio_tokens=tokens,
                    duration_seconds=length / self.audio_processor.sample_rate,
                    context=context,
                ))
        encoded = self.tokenizer.encode_batch(
            rendered,
            padding=True,
        )
        input_ids = torch.tensor(encoded.input_ids, dtype=torch.long)
        attention_mask = torch.tensor(
            encoded.attention_mask,
            dtype=torch.long,
        )
        labels = None
        if output_labels:
            labels = input_ids.clone()
            for token_id in (
                    self.tokenizer.asr_audio_id,
                    self.tokenizer.asr_audio_start_id,
                    self.tokenizer.asr_audio_end_id,
                    self.tokenizer.pad_token_id,
            ):
                labels.masked_fill_(input_ids.eq(token_id), -100)
        return VibeVoiceASRBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_values=audio_batch.input_values,
            padding_mask=audio_batch.padding_mask,
            labels=labels,
        )

    @staticmethod
    def format_training_target(value: str | Sequence[Mapping[str, Any]]) -> str:
        """Serialize the author-published compact segment JSON target."""
        if isinstance(value, str):
            candidate = value.strip()
            if not candidate:
                raise ValueError("ASR training target cannot be empty.")
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError as error:
                raise ValueError("String ASR targets must be compact JSON segment arrays.") from error
            value = parsed
        if not isinstance(value, Sequence) or isinstance(
                value,
                _NON_SEQUENCE_AUDIO_TYPES,
        ):
            raise TypeError("ASR training target must be JSON or a segment sequence.")
        segments: list[dict[str, Any]] = []
        for item in value:
            if not isinstance(item, Mapping):
                raise TypeError("ASR training segments must be mappings.")
            source = dict(item)
            content = source.get("Content", source.get("text"))
            start = source.get("Start", source.get("start"))
            end = source.get("End", source.get("end"))
            speaker = source.get("Speaker", source.get("speaker"))
            if not isinstance(content, str):
                raise TypeError("ASR training segment text must be a string.")
            if (isinstance(start, bool) or not isinstance(start, (int, float)) or isinstance(end, bool) or
                    not isinstance(end, (int, float))):
                raise TypeError("ASR training segment timestamps must be numbers.")
            if (not math.isfinite(float(start)) or not math.isfinite(float(end)) or start < 0 or end < start):
                raise ValueError("ASR training segment timestamps are invalid.")
            formatted: dict[str, Any] = {
                "Start": round(float(start), 2),
                "End": round(float(end), 2),
            }
            if speaker is not None:
                valid_speaker = (not isinstance(speaker, bool) and isinstance(speaker, (int, str)))
                if not valid_speaker:
                    raise TypeError("ASR training speaker must be an integer or string.")
                formatted["Speaker"] = speaker
            formatted["Content"] = content
            segments.append(formatted)
        if not segments:
            raise ValueError("ASR training target must contain a segment.")
        return json.dumps(
            segments,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def prepare_training_batch(
        self,
        audio: Any,
        targets: (str
                  | Sequence[Mapping[str, Any]]
                  | Sequence[str | Sequence[Mapping[str, Any]]]),
        *,
        sampling_rate: int,
        prompt: str | Sequence[str | None] | None = None,
    ) -> VibeVoiceASRBatch:
        """Build assistant-completion-only labels from the official recipe."""
        audio_batch = self.audio_processor(
            audio,
            sampling_rate=sampling_rate,
            pad_to_multiple_of=self.audio_processor.hop_length,
        )
        batch_size = audio_batch.input_values.shape[0]
        if isinstance(targets, str):
            target_values: tuple[Any, ...] = (targets, )
        elif (isinstance(targets, Sequence) and targets and isinstance(targets[0], Mapping)):
            target_values = (targets, )
        elif isinstance(targets, Sequence):
            target_values = tuple(targets)
        else:
            raise TypeError("ASR training targets must be a target or batch.")
        if len(target_values) != batch_size:
            raise ValueError("ASR training target count must match the audio batch.")
        if prompt is None or isinstance(prompt, str):
            prompts = (prompt, ) * batch_size
        elif isinstance(prompt, Sequence):
            prompts = tuple(prompt)
            if len(prompts) != batch_size:
                raise ValueError("ASR training prompt count must match the audio batch.")
        else:
            raise TypeError("ASR training prompt must be a string or sequence.")

        rows: list[tuple[int, ...]] = []
        label_rows: list[tuple[int, ...]] = []
        for length, context, target in zip(
                audio_batch.sample_lengths.tolist(),
                prompts,
                target_values,
        ):
            prompt_text = render_vibevoice_asr_prompt(
                audio_tokens=math.ceil(length / self.audio_processor.hop_length),
                duration_seconds=length / self.audio_processor.sample_rate,
                context=context,
            )
            prompt_ids = self.tokenizer.encode(prompt_text).input_ids
            serialized = self.format_training_target(target)
            completion_ids = self.tokenizer.encode(f"{IM_START}assistant\n{serialized}{IM_END}\n").input_ids
            rows.append((*prompt_ids, *completion_ids))
            label_rows.append((*((-100, ) * len(prompt_ids)), *completion_ids))
        width = max(len(row) for row in rows)
        input_ids = torch.full(
            (batch_size, width),
            self.tokenizer.model_padding_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros(
            batch_size,
            width,
            dtype=torch.long,
        )
        labels = torch.full(
            (batch_size, width),
            -100,
            dtype=torch.long,
        )
        for index, (row, row_labels) in enumerate(zip(rows, label_rows), ):
            length = len(row)
            input_ids[index, :length] = torch.tensor(
                row,
                dtype=torch.long,
            )
            attention_mask[index, :length] = 1
            labels[index, :length] = torch.tensor(
                row_labels,
                dtype=torch.long,
            )
        return VibeVoiceASRBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_values=audio_batch.input_values,
            padding_mask=audio_batch.padding_mask,
            labels=labels,
        )

    def decode(
        self,
        token_ids: Iterable[int],
        *,
        return_format: str = "raw",
    ) -> str | list[dict[str, Any]]:
        if return_format not in {
                "raw",
                "parsed",
                "transcription_only",
        }:
            raise ValueError("`return_format` must be raw, parsed, or transcription_only.")
        text = self.tokenizer.decode(
            token_ids,
            skip_special_tokens=return_format != "raw",
        )
        if return_format == "raw":
            return text
        segments = self.extract_segments(text)
        if isinstance(segments, str):
            return segments
        if return_format == "parsed":
            return segments
        return " ".join(str(segment.get("Content", "")) for segment in segments).strip()

    @staticmethod
    def extract_segments(text: str) -> list[dict[str, Any]] | str:
        if not isinstance(text, str):
            raise TypeError("Decoded ASR output must be a string.")
        candidate = text.strip()
        if candidate.startswith("assistant"):
            candidate = candidate[len("assistant"):].strip()
        if not candidate.startswith("["):
            return text
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            return text
        if not isinstance(value, list) or not all(isinstance(item, dict) and "Content" in item
                                                  for item in value):
            return text
        output: list[dict[str, Any]] = []
        for item in value:
            normalized = dict(item)
            for key in ("Start", "End"):
                timestamp = normalized.get(key)
                if timestamp is not None:
                    valid_timestamp = (
                        not isinstance(timestamp, bool) and isinstance(timestamp, (int, float)))
                    if not valid_timestamp:
                        return text
                    normalized[key] = float(timestamp)
            output.append(normalized)
        return output


def parse_vibevoice_script(script: str) -> tuple[tuple[int, str], ...]:
    """Parse and deterministically zero-base ``Speaker N: text`` lines."""
    if not isinstance(script, str) or not script.strip():
        raise ValueError("VibeVoice script must be a non-empty string.")
    parsed: list[tuple[int, str]] = []
    for line in script.splitlines():
        if not line.strip():
            continue
        match = _SPEAKER_LINE.fullmatch(line.strip())
        if match is None:
            raise ValueError("Every VibeVoice script line must use `Speaker N: text`.")
        text = match.group(2).strip()
        if not text:
            raise ValueError("VibeVoice speaker text cannot be empty.")
        parsed.append((int(match.group(1)), " " + text))
    if not parsed:
        raise ValueError("VibeVoice script contains no speaker turns.")
    offset = min(speaker for speaker, _ in parsed)
    if offset > 0:
        parsed = [(speaker - offset, text) for speaker, text in parsed]
    return tuple(parsed)


@dataclass(frozen=True, slots=True)
class VibeVoiceTTSPrompt:
    input_ids: Tensor
    attention_mask: Tensor
    acoustic_input_mask: Tensor
    speech_tensors: Tensor | None
    speech_masks: Tensor | None
    parsed_script: tuple[tuple[int, str], ...]

    def as_dict(self) -> dict[str, Tensor | None]:
        return {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
            "acoustic_input_mask": self.acoustic_input_mask,
            "speech_tensors": self.speech_tensors,
            "speech_masks": self.speech_masks,
        }


class VibeVoiceTTSProcessor:
    """Audited non-streaming 1.5B text and reference-voice prompt."""

    def __init__(
        self,
        tokenizer: VibeVoiceTokenizer,
        *,
        audio_processor: VibeVoiceAudioProcessor | None = None,
    ) -> None:
        if not isinstance(tokenizer, VibeVoiceTokenizer):
            raise TypeError("TTS processor requires a VibeVoiceTokenizer.")
        self.tokenizer = tokenizer
        self.audio_processor = audio_processor or VibeVoiceAudioProcessor()

    def __call__(
        self,
        script: str,
        *,
        reference_audio: Any | None = None,
        sampling_rate: int = VIBEVOICE_SAMPLE_RATE,
    ) -> VibeVoiceTTSPrompt:
        turns = parse_vibevoice_script(script)
        token_ids = list(self.tokenizer.encode(_TTS_SYSTEM_PROMPT).input_ids)
        acoustic_mask = [False] * len(token_ids)
        speech_tensors = None
        speech_masks = None

        if reference_audio is not None:
            audio = self.audio_processor(
                reference_audio,
                sampling_rate=sampling_rate,
            )
            if audio.input_values.shape[0] > len({speaker for speaker, _ in turns}):
                raise ValueError("Reference-audio count exceeds the number of speakers.")
            prefix = self.tokenizer.encode(" Voice input:\n").input_ids
            token_ids.extend(prefix)
            acoustic_mask.extend((False, ) * len(prefix))
            latent_lengths = torch.ceil(audio.sample_lengths.float() / self.audio_processor.hop_length).to(
                torch.long)
            latent_width = int(latent_lengths.max().item())
            speech_masks = torch.zeros(
                audio.input_values.shape[0],
                latent_width,
                dtype=torch.bool,
            )
            for speaker_id, length in enumerate(latent_lengths.tolist()):
                speaker_prefix = self.tokenizer.encode(f" Speaker {speaker_id}:").input_ids
                newline = self.tokenizer.encode("\n").input_ids
                controls = (
                    self.tokenizer.speech_start_id,
                    *((self.tokenizer.speech_diffusion_id, ) * length),
                    self.tokenizer.speech_end_id,
                )
                token_ids.extend((*speaker_prefix, *controls, *newline))
                acoustic_mask.extend((
                    *((False, ) * (len(speaker_prefix) + 1)),
                    *((True, ) * length),
                    False,
                    *((False, ) * len(newline)),
                ))
                speech_masks[speaker_id, :length] = True
            speech_tensors = audio.input_values[:, 0]

        text_prefix = self.tokenizer.encode(" Text input:\n").input_ids
        token_ids.extend(text_prefix)
        acoustic_mask.extend((False, ) * len(text_prefix))
        for speaker_id, text in turns:
            row = self.tokenizer.encode(f" Speaker {speaker_id}:{text}\n").input_ids
            token_ids.extend(row)
            acoustic_mask.extend((False, ) * len(row))
        output_prefix = self.tokenizer.encode(" Speech output:\n").input_ids
        token_ids.extend((*output_prefix, self.tokenizer.speech_start_id))
        acoustic_mask.extend((False, ) * (len(output_prefix) + 1))

        return VibeVoiceTTSPrompt(
            input_ids=torch.tensor([token_ids], dtype=torch.long),
            attention_mask=torch.ones(
                1,
                len(token_ids),
                dtype=torch.long,
            ),
            acoustic_input_mask=torch.tensor(
                [acoustic_mask],
                dtype=torch.bool,
            ),
            speech_tensors=speech_tensors,
            speech_masks=speech_masks,
            parsed_script=turns,
        )


def validate_vibevoice_training_record(
    record: Mapping[str, Tensor],
    *,
    semantic_dimension: int,
) -> None:
    """Validate the explicit 1.5B preprocessed fine-tuning contract.

    Microsoft did not publish a raw-waveform TTS training serializer.
    VoiceHub therefore accepts the checkpoint graph's explicit tensors
    and masks rather than inventing speaker/audio alignment from
    ambiguous files.
    """
    required = {
        "input_ids",
        "attention_mask",
        "speech_tensors",
        "speech_masks",
        "speeches_loss_input",
        "speech_semantic_tensors",
        "acoustic_input_mask",
        "acoustic_loss_mask",
    }
    missing = required - set(record)
    if missing:
        raise ValueError("VibeVoice training record is missing: " + ", ".join(sorted(missing)) + ".")
    input_ids = record["input_ids"]
    attention = record["attention_mask"]
    acoustic_input = record["acoustic_input_mask"]
    acoustic_loss = record["acoustic_loss_mask"]
    if (input_ids.ndim != 1 or attention.shape != input_ids.shape or
            acoustic_input.shape != input_ids.shape or acoustic_loss.shape != input_ids.shape):
        raise ValueError("VibeVoice token IDs and token masks must be aligned rank-one tensors.")
    speeches = record["speech_tensors"]
    masks = record["speech_masks"]
    selected = record["speeches_loss_input"]
    semantics = record["speech_semantic_tensors"]
    if (speeches.ndim != 2 or masks.ndim != 2 or selected.shape != masks.shape or semantics.ndim != 3 or
            semantics.shape[:2] != masks.shape or semantics.shape[-1] != semantic_dimension or
            speeches.shape[0] != masks.shape[0]):
        raise ValueError("VibeVoice speech segments, latent masks, and semantic latents "
                         "are not aligned.")
    if int(acoustic_input.bool().sum()) != int(masks.bool().sum()):
        raise ValueError("VibeVoice acoustic placeholders must map one-to-one to latents.")
    target_tokens = int(acoustic_loss.bool().sum())
    target_latents = int((selected.bool() & masks.bool()).sum())
    if target_tokens == 0 or target_tokens != target_latents:
        raise ValueError("VibeVoice diffusion targets must map one-to-one to selected "
                         "speech latents.")


__all__ = [
    "VibeVoiceASRBatch",
    "VibeVoiceASRProcessor",
    "VibeVoiceAudioBatch",
    "VibeVoiceAudioProcessor",
    "VibeVoiceTTSProcessor",
    "VibeVoiceTTSPrompt",
    "parse_vibevoice_script",
    "render_vibevoice_asr_prompt",
    "validate_vibevoice_training_record",
]
