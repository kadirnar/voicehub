"""Native waveform, language-prompt, and transcript processing for Nemotron."""

from __future__ import annotations

import copy
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from voicehub.hub import read_json_file, write_json_file
from voicehub.models.asr_nemotron.native.configuration import NemotronASRArchitectureConfig, NemotronFrontendConfig
from voicehub.models.asr_nemotron.native.frontend import NemotronLogMelFrontend
from voicehub.models.asr_nemotron.native.tokenization import (
    METASPACE,
    PAD_TOKEN,
    PUBLISHED_BLANK_TOKEN,
    UNK_TOKEN,
    NemotronASRTokenizer,
)


def _validate_prompt_dictionary(
    value: Any,
    *,
    num_prompts: int,
) -> dict[str, int]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("Nemotron processor requires a non-empty prompt dictionary.")
    result: dict[str, int] = {}
    for language, prompt_id in value.items():
        if not isinstance(language, str) or not language.strip():
            raise ValueError("Nemotron prompt names must be non-empty strings.")
        if (isinstance(prompt_id, bool) or not isinstance(prompt_id, int) or
                not 0 <= prompt_id < num_prompts):
            raise ValueError(f"Nemotron prompt {language!r} has invalid ID {prompt_id!r}.")
        result[language] = prompt_id
    if "auto" not in result:
        raise ValueError("Nemotron prompt dictionary must contain automatic detection.")
    return result


def _waveform_rows(
    waveforms: Any,
    *,
    waveform_lengths: Tensor | None,
) -> tuple[tuple[Tensor, ...], Tensor]:
    if isinstance(waveforms, Tensor):
        if waveforms.ndim == 1:
            rows = (waveforms, )
        elif waveforms.ndim == 2:
            rows = tuple(waveforms[index] for index in range(waveforms.shape[0]))
        else:
            raise ValueError("Nemotron waveforms must have rank one or two.")
    elif (isinstance(waveforms, Sequence) and not isinstance(waveforms, (str, bytes, bytearray))):
        if not waveforms:
            raise ValueError("Nemotron waveform input cannot be empty.")
        first = waveforms[0]
        if isinstance(first, (int, float)):
            rows = (torch.as_tensor(waveforms), )
        else:
            rows = tuple(torch.as_tensor(row) for row in waveforms)
    else:
        rows = (torch.as_tensor(waveforms), )
    if not rows:
        raise ValueError("Nemotron waveform batch cannot be empty.")

    converted: list[Tensor] = []
    for index, row in enumerate(rows):
        if row.ndim != 1 or row.numel() == 0:
            raise ValueError(f"Nemotron waveform {index} must be one non-empty mono row.")
        if row.dtype == torch.bool or row.is_complex():
            raise TypeError(f"Nemotron waveform {index} must contain real samples.")
        value = row.to(dtype=torch.float32)
        if not torch.isfinite(value).all():
            raise ValueError(f"Nemotron waveform {index} contains non-finite samples.")
        converted.append(value)

    if waveform_lengths is None:
        lengths = torch.tensor(
            [row.numel() for row in converted],
            dtype=torch.long,
            device=converted[0].device,
        )
    else:
        lengths = torch.as_tensor(
            waveform_lengths,
            dtype=torch.long,
            device=converted[0].device,
        )
        if lengths.shape != (len(converted), ):
            raise ValueError("`waveform_lengths` must contain one value per waveform.")
        for index, (length, row) in enumerate(zip(lengths, converted)):
            if int(length) <= 0 or int(length) > row.numel():
                raise ValueError(f"Nemotron waveform length {index} is invalid.")
    return tuple(converted), lengths


def _pad_waveforms(
    rows: tuple[Tensor, ...],
    lengths: Tensor,
) -> Tensor:
    device = rows[0].device
    if any(row.device != device for row in rows):
        raise ValueError("Nemotron waveform rows must be on the same device.")
    width = max(row.numel() for row in rows)
    batch = torch.zeros(
        (len(rows), width),
        dtype=torch.float32,
        device=device,
    )
    for index, row in enumerate(rows):
        batch[index, :int(lengths[index])] = row[:int(lengths[index])]
    return batch


class NemotronASRProcessor:
    """Prepare exact native RNN-T inputs without Transformers or NeMo."""

    def __init__(
        self,
        config: NemotronASRArchitectureConfig,
        tokenizer: NemotronASRTokenizer,
        *,
        processor_config: Mapping[str, Any],
        processor_config_path: Path | None = None,
        tokenizer_config_path: Path | None = None,
    ) -> None:
        if not isinstance(config, NemotronASRArchitectureConfig):
            raise TypeError("`config` must be a NemotronASRArchitectureConfig.")
        if not isinstance(tokenizer, NemotronASRTokenizer):
            raise TypeError("`tokenizer` must be a NemotronASRTokenizer.")
        if not isinstance(processor_config, Mapping):
            raise TypeError("`processor_config` must be a mapping.")
        values = copy.deepcopy(dict(processor_config))
        if values.get("processor_class") != "Nemotron3_5AsrProcessor":
            raise ValueError("Nemotron processor class does not match the checkpoint.")
        if values.get("blank_token") != PUBLISHED_BLANK_TOKEN:
            raise ValueError("Nemotron processor must declare '<blank>'.")
        if values.get("num_prompts") != config.num_prompts:
            raise ValueError("Nemotron processor/model prompt counts do not match.")
        prompt_dictionary = _validate_prompt_dictionary(
            values.get("prompt_dictionary"),
            num_prompts=config.num_prompts,
        )
        supported = tuple(values.get(
            "supported_num_lookahead_tokens",
            (),
        ))
        if supported != config.encoder_config.supported_num_lookahead_tokens:
            raise ValueError("Nemotron processor/model lookahead sets do not match.")
        default_lookahead = values.get("default_num_lookahead_tokens")
        if default_lookahead != config.encoder_config.default_num_lookahead_tokens:
            raise ValueError("Nemotron processor/model default lookahead does not match.")
        if tokenizer.pad_token_id != config.blank_token_id:
            raise ValueError("The released Nemotron model blank must equal the tokenizer "
                             "padding ID.")
        if tokenizer.blank_token_id < config.vocab_size:
            raise ValueError("Expected the published tokenizer blank/model vocabulary "
                             "boundary mismatch.")
        if tokenizer.token_id_space_size != config.vocab_size + 1:
            raise ValueError("Nemotron tokenizer/model ID spaces do not match the "
                             "published checkpoint.")

        frontend_config = NemotronFrontendConfig.from_processor_dict(values)
        if frontend_config.feature_size != config.encoder_config.num_mel_bins:
            raise ValueError("Nemotron frontend/model mel dimensions do not match.")
        self.config = config
        self.tokenizer = tokenizer
        self.frontend_config = frontend_config
        self.frontend = NemotronLogMelFrontend(frontend_config)
        self.feature_extractor = self.frontend
        self.prompt_dictionary = prompt_dictionary
        self.num_prompts = config.num_prompts
        self.supported_num_lookahead_tokens = supported
        self.default_num_lookahead_tokens = default_lookahead
        self.model_blank_token_id = config.blank_token_id
        self.blank_token_id = tokenizer.blank_token_id
        self._processor_config = values
        self._processor_config_path = processor_config_path
        self._tokenizer_config_path = tokenizer_config_path

    @classmethod
    def from_artifacts(
        cls,
        *,
        config: NemotronASRArchitectureConfig,
        tokenizer_json: str | Path,
        tokenizer_config: str | Path,
        processor_config: str | Path,
    ) -> NemotronASRProcessor:
        tokenizer_path = Path(tokenizer_json).expanduser().resolve()
        tokenizer_config_path = (Path(tokenizer_config).expanduser().resolve())
        tokenizer_values = read_json_file(tokenizer_config_path)
        expected = {
            "backend": "tokenizers",
            "clean_up_tokenization_spaces": False,
            "pad_token": PAD_TOKEN,
            "processor_class": "Nemotron3_5AsrProcessor",
            "tokenizer_class": "ParakeetTokenizer",
            "unk_token": UNK_TOKEN,
        }
        for name, expected_value in expected.items():
            if tokenizer_values.get(name) != expected_value:
                raise ValueError(
                    f"Nemotron tokenizer setting {name!r} is "
                    f"{tokenizer_values.get(name)!r}; expected "
                    f"{expected_value!r}.")
        processor_path = Path(processor_config).expanduser().resolve()
        return cls(
            config,
            NemotronASRTokenizer.from_tokenizer_json(tokenizer_path),
            processor_config=read_json_file(processor_path),
            processor_config_path=processor_path,
            tokenizer_config_path=tokenizer_config_path,
        )

    @property
    def sample_rate(self) -> int:
        return self.frontend_config.sampling_rate

    @property
    def frame_seconds(self) -> float:
        return (
            self.frontend_config.hop_length / self.frontend_config.sampling_rate *
            self.config.encoder_config.subsampling_factor)

    @property
    def streaming_latency_ms(self) -> int:
        return round((self.default_num_lookahead_tokens + 1) * self.frame_seconds * 1000)

    @property
    def supported_streaming_latencies_ms(self) -> dict[int, int]:
        return {
            value: round((value + 1) * self.frame_seconds * 1000)
            for value in self.supported_num_lookahead_tokens
        }

    @property
    def num_mel_frames_first_audio_chunk(self) -> int:
        return (1 + self.config.encoder_config.subsampling_factor * self.default_num_lookahead_tokens)

    @property
    def num_mel_frames_per_audio_chunk(self) -> int:
        return (self.config.encoder_config.subsampling_factor * (self.default_num_lookahead_tokens + 1))

    @property
    def num_samples_first_audio_chunk(self) -> int:
        return ((self.num_mel_frames_first_audio_chunk - 1) * self.frontend_config.hop_length +
                self.frontend_config.win_length // 2)

    @property
    def num_samples_per_audio_chunk(self) -> int:
        return (
            self.num_mel_frames_per_audio_chunk * self.frontend_config.hop_length +
            self.frontend_config.win_length)

    def set_num_lookahead_tokens(
        self,
        value: int,
    ) -> NemotronASRProcessor:
        if (isinstance(value, bool) or not isinstance(value, int) or
                value not in self.supported_num_lookahead_tokens):
            raise ValueError(
                f"Unsupported Nemotron lookahead {value!r}; expected one "
                f"of {self.supported_num_lookahead_tokens}.")
        self.default_num_lookahead_tokens = value
        return self

    def resolve_prompt_ids(
        self,
        language: str | Sequence[str],
        *,
        batch_size: int,
        device: torch.device | str | None = None,
    ) -> Tensor:
        if isinstance(language, str):
            languages = (language, ) * batch_size
        elif isinstance(language, Sequence):
            languages = tuple(language)
        else:
            raise TypeError("Nemotron language must be a string or sequence.")
        if len(languages) != batch_size:
            raise ValueError(f"Received {len(languages)} languages for a batch of "
                             f"{batch_size}.")
        prompt_ids: list[int] = []
        for language_name in languages:
            if not isinstance(language_name, str) or not language_name.strip():
                raise ValueError("Nemotron languages must be non-empty strings.")
            normalized = language_name.strip()
            try:
                prompt_ids.append(self.prompt_dictionary[normalized])
            except KeyError as error:
                raise ValueError(
                    f"Unknown Nemotron language {normalized!r}. Supported "
                    f"values: {sorted(self.prompt_dictionary)!r}.") from error
        return torch.tensor(
            prompt_ids,
            dtype=torch.long,
            device=device,
        )

    def prepare_audio_batch(
        self,
        waveforms: Any,
        *,
        sampling_rate: int,
        waveform_lengths: Tensor | None = None,
        is_streaming: bool = False,
        is_first_audio_chunk: bool = True,
    ) -> dict[str, Tensor]:
        if not isinstance(is_streaming, bool):
            raise TypeError("`is_streaming` must be a boolean.")
        if not isinstance(is_first_audio_chunk, bool):
            raise TypeError("`is_first_audio_chunk` must be a boolean.")
        if not is_streaming and not is_first_audio_chunk:
            raise ValueError("Offline Nemotron processing must use the first-chunk "
                             "frontend.")
        rows, lengths = _waveform_rows(
            waveforms,
            waveform_lengths=waveform_lengths,
        )
        padded = _pad_waveforms(rows, lengths)
        input_features, attention_mask = self.frontend(
            padded,
            lengths,
            sampling_rate=sampling_rate,
            center=is_first_audio_chunk,
        )
        return {
            "input_features": input_features,
            "attention_mask": attention_mask,
            "waveform_lengths": lengths,
        }

    def prepare_stream_chunk(
        self,
        waveforms: Any,
        *,
        sampling_rate: int,
        is_first_audio_chunk: bool,
    ) -> Tensor:
        prepared = self.prepare_audio_batch(
            waveforms,
            sampling_rate=sampling_rate,
            is_streaming=True,
            is_first_audio_chunk=is_first_audio_chunk,
        )
        expected = (
            self.num_mel_frames_first_audio_chunk
            if is_first_audio_chunk else self.num_mel_frames_per_audio_chunk)
        valid_lengths = prepared["attention_mask"].sum(dim=-1)
        if torch.any(valid_lengths != expected):
            raise ValueError(
                "Nemotron stream chunk has an invalid sample count; "
                f"expected exactly {expected} valid mel frames.")
        return prepared["input_features"][:, :expected]

    def encode_labels(
        self,
        texts: Sequence[str],
    ) -> dict[str, Tensor]:
        if isinstance(texts, (str, bytes)) or not isinstance(
                texts,
                Sequence,
        ):
            raise TypeError("`texts` must be a sequence of transcripts.")
        rows = tuple(self.tokenizer.encode(text).input_ids for text in texts)
        if not rows:
            raise ValueError("Nemotron label batch cannot be empty.")
        if any(token_id >= self.config.vocab_size for row in rows for token_id in row):
            raise ValueError(
                "Nemotron transcript produced a tokenizer-only token outside "
                "the model vocabulary.")
        maximum = max(len(row) for row in rows)
        labels = torch.full(
            (len(rows), maximum),
            self.model_blank_token_id,
            dtype=torch.long,
        )
        decoder_input_ids = torch.full(
            (len(rows), maximum + 1),
            self.model_blank_token_id,
            dtype=torch.long,
        )
        label_lengths = torch.tensor(
            [len(row) for row in rows],
            dtype=torch.long,
        )
        for index, row in enumerate(rows):
            values = torch.tensor(row, dtype=torch.long)
            labels[index, :len(row)] = values
            decoder_input_ids[index, 1:len(row) + 1] = values
        return {
            "labels": labels,
            "label_lengths": label_lengths,
            "decoder_input_ids": decoder_input_ids,
        }

    def __call__(
        self,
        audio: Any,
        text: str | Sequence[str] | None = None,
        *,
        sampling_rate: int = 16_000,
        waveform_lengths: Tensor | None = None,
        is_streaming: bool = False,
        is_first_audio_chunk: bool = True,
        language: str | Sequence[str] = "auto",
        num_lookahead_tokens: int | None = None,
        return_tensors: str = "pt",
    ) -> dict[str, Any]:
        if return_tensors != "pt":
            raise ValueError("Native Nemotron processing returns PyTorch tensors only.")
        prepared: dict[str, Any] = self.prepare_audio_batch(
            audio,
            sampling_rate=sampling_rate,
            waveform_lengths=waveform_lengths,
            is_streaming=is_streaming,
            is_first_audio_chunk=is_first_audio_chunk,
        )
        batch_size = prepared["input_features"].shape[0]
        prepared["prompt_ids"] = self.resolve_prompt_ids(
            language,
            batch_size=batch_size,
            device=prepared["input_features"].device,
        )
        lookahead = (
            self.default_num_lookahead_tokens if num_lookahead_tokens is None else num_lookahead_tokens)
        if lookahead not in self.supported_num_lookahead_tokens:
            raise ValueError(
                f"Unsupported Nemotron lookahead {lookahead!r}; expected "
                f"one of {self.supported_num_lookahead_tokens}.")
        prepared["num_lookahead_tokens"] = lookahead
        if text is not None:
            texts = (text, ) if isinstance(text, str) else tuple(text)
            if len(texts) != batch_size:
                raise ValueError("Nemotron requires one transcript per waveform.")
            prepared.update(self.encode_labels(texts))
        return prepared

    @staticmethod
    def _sequence_rows(sequences: Any, ) -> tuple[tuple[int, ...], ...]:
        if isinstance(sequences, Tensor):
            values = sequences.detach().cpu()
            if values.ndim == 1:
                values = values.unsqueeze(0)
            if values.ndim != 2:
                raise ValueError("Nemotron sequences must have rank one or two.")
            return tuple(tuple(int(value) for value in row.tolist()) for row in values)
        rows = tuple(sequences)
        if not rows:
            return ()
        if isinstance(rows[0], int):
            return (tuple(int(value) for value in rows), )
        return tuple(tuple(int(value) for value in row) for row in rows)

    def _timestamp_offsets(
        self,
        token_ids: tuple[int, ...],
        durations: tuple[int, ...],
    ) -> list[dict[str, float | str]]:
        if len(token_ids) != len(durations):
            raise ValueError("Nemotron token and duration lengths must match.")
        frame = 0
        offsets: list[dict[str, float | str]] = []
        buffered = bytearray()
        buffered_start: int | None = None
        buffered_end = 0

        def flush_bytes() -> None:
            nonlocal buffered_start
            if buffered and buffered_start is not None:
                offsets.append({
                    "token": buffered.decode("utf-8", errors="replace"),
                    "start": buffered_start * self.frame_seconds,
                    "end": buffered_end * self.frame_seconds,
                })
                buffered.clear()
                buffered_start = None

        for token_id, advance in zip(token_ids, durations):
            start = frame
            frame += advance
            if (token_id == self.model_blank_token_id or token_id in self.tokenizer.special_token_ids):
                flush_bytes()
                continue
            token = self.tokenizer.token_for_id(token_id)
            byte_value = self.tokenizer._byte_value(token)
            if byte_value is not None:
                if buffered_start is None:
                    buffered_start = start
                buffered.append(byte_value)
                buffered_end = start + 1
                continue
            flush_bytes()
            rendered = token.replace(METASPACE, " ")
            if rendered:
                offsets.append({
                    "token": rendered,
                    "start": start * self.frame_seconds,
                    "end": (start + 1) * self.frame_seconds,
                })
        flush_bytes()
        return offsets

    def batch_decode(
        self,
        sequences: Any,
        *,
        durations: Any | None = None,
        skip_special_tokens: bool = True,
    ) -> list[str] | tuple[list[str], list[list[dict[str, float | str]]]]:
        rows = self._sequence_rows(sequences)
        decoded = [
            self.tokenizer.decode(
                (token_id for token_id in row if token_id != self.model_blank_token_id),
                skip_special_tokens=skip_special_tokens,
            ) for row in rows
        ]
        if durations is None:
            return decoded
        duration_rows = self._sequence_rows(durations)
        if len(duration_rows) != len(rows):
            raise ValueError("Nemotron duration batch size must match sequences.")
        offsets = [
            self._timestamp_offsets(token_ids, advances) for token_ids, advances in zip(rows, duration_rows)
        ]
        return decoded, offsets

    def decode(
        self,
        sequences: Any,
        *,
        durations: Any | None = None,
        skip_special_tokens: bool = True,
    ) -> str | tuple[str, list[dict[str, float | str]]]:
        decoded = self.batch_decode(
            sequences,
            durations=durations,
            skip_special_tokens=skip_special_tokens,
        )
        if durations is None:
            if len(decoded) != 1:
                raise ValueError("Use `batch_decode` for multiple Nemotron sequences.")
            return decoded[0]
        texts, offsets = decoded
        if len(texts) != 1:
            raise ValueError("Use `batch_decode` for multiple Nemotron sequences.")
        return texts[0], offsets[0]

    def detected_language(self, sequences: Any) -> str | None:
        rows = self._sequence_rows(sequences)
        if len(rows) != 1:
            raise ValueError("Use one Nemotron sequence for language detection.")
        language_by_token_id = {
            token_id: language
            for language in self.prompt_dictionary if language != "auto"
            for token_id in (self.tokenizer.special_tokens.get(f"<{language}>"), ) if token_id is not None
        }
        return next(
            (language_by_token_id[token_id] for token_id in rows[0] if token_id in language_by_token_id),
            None,
        )

    def save_pretrained(self, directory: str | Path) -> Path:
        destination = Path(directory).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(destination)
        tokenizer_target = destination / "tokenizer_config.json"
        if self._tokenizer_config_path is not None:
            if tokenizer_target.resolve() != self._tokenizer_config_path:
                shutil.copy2(
                    self._tokenizer_config_path,
                    tokenizer_target,
                )
        else:
            write_json_file(
                tokenizer_target,
                {
                    "backend":
                    "tokenizers",
                    "clean_up_tokenization_spaces":
                    False,
                    "extra_special_tokens": [
                        token for token, _ in sorted(
                            self.tokenizer.special_tokens.items(),
                            key=lambda item: item[1],
                        ) if token not in {
                            PAD_TOKEN,
                            PUBLISHED_BLANK_TOKEN,
                            UNK_TOKEN,
                        }
                    ],
                    "pad_token":
                    PAD_TOKEN,
                    "processor_class":
                    "Nemotron3_5AsrProcessor",
                    "tokenizer_class":
                    "ParakeetTokenizer",
                    "unk_token":
                    UNK_TOKEN,
                },
            )
        values = copy.deepcopy(self._processor_config)
        values["default_num_lookahead_tokens"] = (self.default_num_lookahead_tokens)
        write_json_file(
            destination / "processor_config.json",
            values,
        )
        return destination


__all__ = ["NemotronASRProcessor"]
