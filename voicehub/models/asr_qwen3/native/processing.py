"""Native audio/text preparation and output parsing for Qwen3-ASR."""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from voicehub.models.asr_qwen3.native.configuration import Qwen3ASRArchitectureConfig
from voicehub.models.asr_qwen3.native.languages import LANGUAGE_CODES, normalize_qwen3_asr_language
from voicehub.models.asr_qwen3.native.modeling import qwen3_asr_audio_output_lengths
from voicehub.models.asr_qwen3.native.tokenization import (
    ASR_TEXT,
    AUDIO_END,
    AUDIO_PAD,
    AUDIO_START,
    IM_END,
    IM_START,
    Qwen3ASRTokenizer,
)
from voicehub.processing.audio import LogMelSpectrogram
from voicehub.processing.waveform import NativeAudio, load_native_audio

SAMPLE_RATE = 16_000
MAX_ASR_INPUT_SECONDS = 1_200
MIN_ASR_INPUT_SECONDS = 0.5


def detect_and_fix_repetitions(text: str, *, threshold: int = 20) -> str:
    """Collapse pathological character/pattern loops from generation."""
    if not isinstance(text, str):
        raise TypeError("`text` must be a string.")
    if threshold < 2:
        raise ValueError("`threshold` must be at least two.")

    def characters(value: str) -> str:
        output: list[str] = []
        index = 0
        while index < len(value):
            end = index + 1
            while end < len(value) and value[end] == value[index]:
                end += 1
            output.append(value[index] if end - index > threshold else value[index:end])
            index = end
        return "".join(output)

    def patterns(value: str, *, maximum_pattern: int = 20) -> str:
        minimum = threshold * 2
        if len(value) < minimum:
            return value
        output: list[str] = []
        index = 0
        while index <= len(value) - minimum:
            match: tuple[str, int] | None = None
            for size in range(1, maximum_pattern + 1):
                if index + size * threshold > len(value):
                    break
                pattern = value[index:index + size]
                if all(value[index + repeat * size:index + (repeat + 1) * size] == pattern
                       for repeat in range(1, threshold)):
                    end = index + threshold * size
                    while (end + size <= len(value) and value[end:end + size] == pattern):
                        end += size
                    match = pattern, end
                    break
            if match is None:
                output.append(value[index])
                index += 1
                continue
            pattern, end = match
            output.append(pattern)
            output.append(patterns(value[end:], maximum_pattern=maximum_pattern))
            return "".join(output)
        output.append(value[index:])
        return "".join(output)

    return patterns(characters(text))


def parse_qwen3_asr_output(
    raw: str,
    *,
    forced_language: str | None = None,
) -> tuple[str | None, str]:
    """Parse the official ``language X<asr_text>...`` response protocol."""
    if raw is None:
        return None, ""
    value = detect_and_fix_repetitions(str(raw).strip())
    if not value:
        return None, ""
    if forced_language is not None:
        return forced_language, value
    if ASR_TEXT not in value:
        return None, value
    metadata, text = value.split(ASR_TEXT, 1)
    if "language none" in metadata.lower():
        return None, text.strip()
    language = None
    for line in metadata.splitlines():
        line = line.strip()
        if line.lower().startswith("language "):
            candidate = line[len("language "):].strip()
            if candidate:
                candidate = candidate[:1].upper() + candidate[1:].lower()
                language = candidate
            break
    return language, text.strip()


def _broadcast(
    value: Any,
    length: int,
    *,
    name: str,
) -> tuple[Any, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray, Tensor)):
        values = tuple(value)
        if len(values) != length:
            raise ValueError(f"`{name}` must contain one value per audio ({length}).")
        return values
    return (value, ) * length


class Qwen3ASRProcessor:
    """Build log-mel/audio-placeholder batches for inference and training."""

    def __init__(
        self,
        config: Qwen3ASRArchitectureConfig,
        tokenizer: Qwen3ASRTokenizer,
        *,
        preprocessor_config_path: Path | None = None,
        generation_config_path: Path | None = None,
        chat_template_path: Path | None = None,
    ) -> None:
        if not isinstance(config, Qwen3ASRArchitectureConfig):
            raise TypeError("`config` must be Qwen3ASRArchitectureConfig.")
        if not isinstance(tokenizer, Qwen3ASRTokenizer):
            raise TypeError("`tokenizer` must be Qwen3ASRTokenizer.")
        self.config = config
        self.tokenizer = tokenizer
        self.preprocessor_config_path = preprocessor_config_path
        self.generation_config_path = generation_config_path
        self.chat_template_path = chat_template_path
        self.feature_extractor = LogMelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=400,
            hop_length=160,
            n_mels=config.audio_config.num_mel_bins,
            dynamic_range=8.0,
            whisper_scaling=True,
        )

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE

    @property
    def supported_languages(self) -> tuple[str, ...]:
        return self.config.support_languages

    def normalize_language(self, language: str | None) -> str | None:
        return normalize_qwen3_asr_language(
            language,
            supported_languages=self.supported_languages,
        )

    @staticmethod
    def _context_with_hotwords(
        context: str | None,
        hotwords: str | Sequence[str] | None,
    ) -> str:
        if context is not None and not isinstance(context, str):
            raise TypeError("`context` must be a string or None.")
        normalized = context.strip() if context else ""
        if hotwords is None:
            return normalized
        words = (hotwords, ) if isinstance(hotwords, str) else tuple(hotwords)
        if any(not isinstance(word, str) or not word.strip() for word in words):
            raise ValueError("`hotwords` must contain non-empty strings.")
        vocabulary = "Vocabulary: " + ", ".join(word.strip() for word in words)
        return f"{normalized}\n{vocabulary}" if normalized else vocabulary

    def build_prompt(
        self,
        *,
        context: str = "",
        language: str | None = None,
        audio_tokens: int = 1,
    ) -> str:
        if not isinstance(context, str):
            raise TypeError("`context` must be a string.")
        if (isinstance(audio_tokens, bool) or not isinstance(audio_tokens, int) or audio_tokens <= 0):
            raise ValueError("`audio_tokens` must be a positive integer.")
        prompt = (
            f"{IM_START}system\n{context}{IM_END}\n"
            f"{IM_START}user\n{AUDIO_START}"
            f"{AUDIO_PAD * audio_tokens}{AUDIO_END}{IM_END}\n"
            f"{IM_START}assistant\n")
        canonical = self.normalize_language(language)
        if canonical is not None:
            prompt += f"language {canonical}{ASR_TEXT}"
        return prompt

    @staticmethod
    def _split_long_audio(audio: NativeAudio) -> tuple[NativeAudio, ...]:
        maximum = MAX_ASR_INPUT_SECONDS * SAMPLE_RATE
        waveform = audio.waveform
        if waveform.numel() <= maximum:
            chunks = [audio]
        else:
            chunks = []
            start = 0
            search = 5 * SAMPLE_RATE
            window = max(4, SAMPLE_RATE // 10)
            while waveform.numel() - start > maximum:
                target = start + maximum
                left = max(start, target - search)
                right = min(waveform.numel(), target + search)
                region = waveform[left:right].abs()
                if region.numel() > window:
                    energy = torch.nn.functional.avg_pool1d(
                        region.reshape(1, 1, -1),
                        kernel_size=window,
                        stride=1,
                    ).reshape(-1)
                    window_start = int(energy.argmin().item())
                    quiet_window = region[window_start:window_start + window]
                    boundary = (left + window_start + int(quiet_window.argmin().item()))
                else:
                    boundary = target
                boundary = min(
                    max(boundary, start + 1),
                    waveform.numel(),
                )
                chunks.append(NativeAudio(
                    waveform=waveform[start:boundary],
                    sampling_rate=SAMPLE_RATE,
                ))
                start = boundary
            chunks.append(NativeAudio(
                waveform=waveform[start:],
                sampling_rate=SAMPLE_RATE,
            ))
        minimum = int(MIN_ASR_INPUT_SECONDS * SAMPLE_RATE)
        return tuple(
            NativeAudio(
                waveform=(
                    torch.nn.functional.pad(
                        chunk.waveform,
                        (0, minimum - chunk.waveform.numel()),
                    ) if chunk.waveform.numel() < minimum else chunk.waveform),
                sampling_rate=chunk.sampling_rate,
                path=chunk.path,
            ) for chunk in chunks)

    def materialize_audio(
        self,
        audio: Any,
        *,
        sampling_rate: int | None,
    ) -> NativeAudio:
        materialized = load_native_audio(
            audio,
            sampling_rate=sampling_rate,
            target_sampling_rate=SAMPLE_RATE,
        )
        waveform = materialized.waveform
        peak = float(waveform.abs().amax().item())
        if peak > 1.0:
            waveform = waveform / peak
        waveform = waveform.clamp(-1.0, 1.0)
        return NativeAudio(
            waveform=waveform,
            sampling_rate=materialized.sampling_rate,
            path=materialized.path,
        )

    def long_audio_chunks(
        self,
        audio: Any,
        *,
        sampling_rate: int | None,
    ) -> tuple[NativeAudio, ...]:
        return self._split_long_audio(self.materialize_audio(audio, sampling_rate=sampling_rate))

    def _audio_features(
        self,
        audios: Sequence[Any],
        *,
        sampling_rates: Sequence[int | None],
    ) -> tuple[Tensor, Tensor, tuple[float, ...]]:
        if not audios:
            raise ValueError("At least one audio input is required.")
        materialized = tuple(
            self.materialize_audio(audio, sampling_rate=rate) for audio, rate in zip(audios, sampling_rates))
        lengths = tuple(int(item.waveform.numel()) for item in materialized)
        maximum_samples = max(max(lengths), 400)
        padded = torch.zeros(
            len(materialized),
            maximum_samples,
            dtype=torch.float32,
        )
        for index, item in enumerate(materialized):
            padded[index, :item.waveform.numel()] = item.waveform.to(
                device="cpu",
                dtype=torch.float32,
            )
        input_features = self.feature_extractor.process({"waveform": padded})["input_features"]
        frame_count = input_features.shape[-1]
        feature_lengths = torch.tensor(
            [
                min(
                    frame_count,
                    max(
                        1,
                        (length + self.feature_extractor.hop_length - 1) // self.feature_extractor.hop_length,
                    ),
                ) for length in lengths
            ],
            dtype=torch.long,
        )
        feature_mask = (torch.arange(frame_count).unsqueeze(0) < feature_lengths.unsqueeze(1))
        return (
            input_features,
            feature_mask,
            tuple(length / SAMPLE_RATE for length in lengths),
        )

    def _token_batch(
        self,
        texts: Sequence[str],
        *,
        prefix_lengths: Sequence[int] | None = None,
    ) -> dict[str, Tensor]:
        encodings = tuple(self.tokenizer.encode(text) for text in texts)
        maximum = max(len(value.input_ids) for value in encodings)
        input_ids = torch.full(
            (len(encodings), maximum),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros(
            (len(encodings), maximum),
            dtype=torch.bool,
        )
        labels = (
            torch.full(
                (len(encodings), maximum),
                -100,
                dtype=torch.long,
            ) if prefix_lengths is not None else None)
        for index, encoding in enumerate(encodings):
            ids = torch.tensor(encoding.input_ids, dtype=torch.long)
            left = maximum - ids.numel()
            input_ids[index, left:] = ids
            attention_mask[index, left:] = True
            if labels is not None:
                prefix_length = int(prefix_lengths[index])
                if not 0 <= prefix_length <= ids.numel():
                    raise ValueError("Invalid completion prefix length.")
                labels[index, left + prefix_length:] = ids[prefix_length:]
                labels[index, input_ids[index] == self.tokenizer.pad_token_id] = -100
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if labels is not None:
            result["labels"] = labels
        return result

    def prepare_inference_batch(
        self,
        audios: Sequence[Any],
        *,
        sampling_rates: Sequence[int | None],
        contexts: Sequence[str] | None = None,
        languages: Sequence[str | None] | None = None,
        hotwords: Sequence[str | Sequence[str] | None] | None = None,
    ) -> dict[str, Any]:
        count = len(audios)
        if len(sampling_rates) != count:
            raise ValueError("Sampling-rate batch size mismatch.")
        contexts = contexts or ("", ) * count
        languages = languages or (None, ) * count
        hotwords = hotwords or (None, ) * count
        if not (len(contexts) == len(languages) == len(hotwords) == count):
            raise ValueError("Qwen3-ASR conditioning batch size mismatch.")
        features, feature_mask, durations = self._audio_features(
            audios,
            sampling_rates=sampling_rates,
        )
        feature_lengths = feature_mask.long().sum(-1)
        output_lengths = qwen3_asr_audio_output_lengths(feature_lengths)
        prompts = tuple(
            self.build_prompt(
                context=self._context_with_hotwords(context, words),
                language=language,
                audio_tokens=int(output_length.item()),
            ) for context, language, words, output_length in zip(
                contexts,
                languages,
                hotwords,
                output_lengths,
            ))
        result: dict[str, Any] = self._token_batch(prompts)
        result.update({
            "input_features": features,
            "feature_attention_mask": feature_mask,
            "durations": durations,
            "forced_languages": tuple(self.normalize_language(value) for value in languages),
        })
        return result

    def prepare_training_batch(
        self,
        audios: Sequence[Any],
        texts: Sequence[str],
        *,
        sampling_rates: Sequence[int | None],
        contexts: Sequence[str] | None = None,
        languages: Sequence[str | None] | None = None,
    ) -> dict[str, Tensor]:
        count = len(audios)
        if len(texts) != count or len(sampling_rates) != count:
            raise ValueError("Qwen3-ASR training columns have different sizes.")
        contexts = contexts or ("", ) * count
        languages = languages or (None, ) * count
        if len(contexts) != count or len(languages) != count:
            raise ValueError("Qwen3-ASR training conditioning size mismatch.")
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Every Qwen3-ASR transcript must be non-empty.")
        features, feature_mask, _ = self._audio_features(
            audios,
            sampling_rates=sampling_rates,
        )
        output_lengths = qwen3_asr_audio_output_lengths(feature_mask.long().sum(-1))
        prefixes = tuple(
            self.build_prompt(
                context=context,
                audio_tokens=int(output_length.item()),
            ) for context, output_length in zip(contexts, output_lengths))
        targets = []
        for text, language in zip(texts, languages):
            target = text.strip()
            has_protocol = (ASR_TEXT in target and target.lower().startswith("language "))
            if not has_protocol:
                canonical = self.normalize_language(language)
                target = (f"language {canonical or 'None'}{ASR_TEXT}{target}")
            targets.append(target)
        prefix_lengths = tuple(len(self.tokenizer.encode(prefix).input_ids) for prefix in prefixes)
        full_texts = tuple(prefix + target + IM_END for prefix, target in zip(prefixes, targets))
        result = self._token_batch(
            full_texts,
            prefix_lengths=prefix_lengths,
        )
        result.update({
            "input_features": features,
            "feature_attention_mask": feature_mask,
        })
        return result

    def save_pretrained(self, directory: str | Path) -> Path:
        target = Path(directory).expanduser()
        target.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(target)
        for source, filename in (
            (self.preprocessor_config_path, "preprocessor_config.json"),
            (self.generation_config_path, "generation_config.json"),
            (self.chat_template_path, "chat_template.json"),
        ):
            if source is None:
                continue
            destination = target / filename
            if source.resolve() != destination.resolve():
                shutil.copy2(source, destination)
        return target


__all__ = [
    "LANGUAGE_CODES",
    "MAX_ASR_INPUT_SECONDS",
    "MIN_ASR_INPUT_SECONDS",
    "SAMPLE_RATE",
    "Qwen3ASRProcessor",
    "detect_and_fix_repetitions",
    "normalize_qwen3_asr_language",
    "parse_qwen3_asr_output",
]
