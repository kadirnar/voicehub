"""Native 16 kHz frontend and prompt-aware Cohere ASR processor."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from voicehub.models.asr_cohere.native.configuration import SUPPORTED_LANGUAGES, CohereAsrConfig
from voicehub.models.asr_cohere.native.modeling import LOG_ZERO_GUARD, NORMALIZATION_EPSILON, FilterbankFeatures
from voicehub.models.asr_cohere.native.tokenization import CohereAsrTokenizer

_NO_SPACE_LANGUAGES = frozenset({"ja", "zh"})


class CohereAsrFeatureExtractor:
    """Official deterministic log-mel frontend implemented in PyTorch."""

    model_input_names = ("input_features", "attention_mask")

    def __init__(
        self,
        featurizer: FilterbankFeatures,
        config: CohereAsrConfig,
    ) -> None:
        if not isinstance(featurizer, FilterbankFeatures):
            raise TypeError("`featurizer` must be the model's FilterbankFeatures module.")
        self.featurizer = featurizer
        self.config = CohereAsrConfig.coerce(config)
        self.feature_size = self.config.encoder_config.num_mel_bins
        self.sampling_rate = self.config.sample_rate
        self.hop_length = self.config.hop_length
        self.n_fft = self.config.n_fft
        self.win_length = self.config.win_length
        self.preemphasis = self.config.preemphasis
        self.dither = self.config.dither
        self.max_audio_clip_s = self.config.max_audio_clip_s
        self.overlap_chunk_second = self.config.overlap_chunk_second
        self.min_energy_window_samples = (self.config.min_energy_window_samples)

    @staticmethod
    def _waveforms(
        audio: Any,
        *,
        device: torch.device | str | None,
    ) -> tuple[torch.Tensor, ...]:
        if isinstance(audio, torch.Tensor):
            if audio.ndim == 1:
                values = (audio, )
            elif audio.ndim == 2:
                values = tuple(audio[index] for index in range(audio.shape[0]))
            else:
                raise ValueError("Cohere ASR audio tensor must have shape [time] or "
                                 "[batch, time].")
        elif isinstance(audio, Sequence) and not isinstance(audio, (str, bytes)):
            if not audio:
                raise ValueError("Cohere ASR audio cannot be empty.")
            first = audio[0]
            if isinstance(first, (int, float)):
                values = (torch.as_tensor(audio), )
            else:
                values = tuple(torch.as_tensor(value) for value in audio)
        else:
            values = (torch.as_tensor(audio), )
        normalized = []
        for waveform in values:
            if waveform.ndim != 1:
                raise ValueError(
                    "Cohere ASR accepts mono waveforms only; downmix at the "
                    "audio-loading boundary.")
            if waveform.numel() < 2:
                raise ValueError("Cohere ASR audio must contain at least two samples.")
            if not torch.isfinite(waveform).all():
                raise ValueError("Cohere ASR audio contains NaN or infinity.")
            normalized.append(waveform.to(device=device, dtype=torch.float32))
        return tuple(normalized)

    def _find_split_point(
        self,
        waveform: torch.Tensor,
        start: int,
        end: int,
    ) -> int:
        segment = waveform[start:end]
        window_size = self.min_energy_window_samples
        if segment.numel() <= window_size:
            return (start + end) // 2
        quietest = start
        minimum_energy = float("inf")
        upper = segment.numel() - window_size
        for offset in range(0, upper, window_size):
            window = segment[offset:offset + window_size]
            energy = torch.sqrt(torch.mean(window.square())).item()
            if energy < minimum_energy:
                minimum_energy = energy
                quietest = start + offset
        return quietest

    def split_long_waveform(
        self,
        waveform: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Split at source-verified quiet boundaries without overlap."""
        chunk_size = max(
            1,
            int(round(self.max_audio_clip_s * self.sampling_rate)),
        )
        boundary_context = max(
            1,
            int(round(self.overlap_chunk_second * self.sampling_rate)),
        )
        if waveform.numel() <= chunk_size:
            return (waveform, )
        ranges: list[tuple[int, int]] = []
        start = 0
        total = waveform.numel()
        while start < total:
            if start + chunk_size >= total:
                ranges.append((start, total))
                break
            search_start = max(
                start,
                start + chunk_size - boundary_context,
            )
            search_end = min(start + chunk_size, total)
            split = (
                start + chunk_size if search_end <= search_start else self._find_split_point(
                    waveform,
                    search_start,
                    search_end,
                ))
            split = max(start + 1, min(split, total))
            ranges.append((start, split))
            start = split
        return tuple(waveform[left:right] for left, right in ranges if right > left)

    def _chunk(
        self,
        waveforms: tuple[torch.Tensor, ...],
    ) -> tuple[tuple[torch.Tensor, ...], tuple[tuple[int, int | None], ...]]:
        threshold = max(
            0.0,
            self.max_audio_clip_s - self.overlap_chunk_second,
        )
        chunks = []
        index = []
        for sample_index, waveform in enumerate(waveforms):
            duration = waveform.numel() / self.sampling_rate
            if duration <= threshold:
                chunks.append(waveform)
                index.append((sample_index, None))
                continue
            split = self.split_long_waveform(waveform)
            for chunk_index, value in enumerate(split):
                chunks.append(value)
                index.append((sample_index, chunk_index))
        return tuple(chunks), tuple(index)

    def _features(
        self,
        waveforms: tuple[torch.Tensor, ...],
    ) -> dict[str, torch.Tensor]:
        lengths = torch.tensor(
            [waveform.numel() for waveform in waveforms],
            device=waveforms[0].device,
            dtype=torch.long,
        )
        feature_lengths = torch.div(
            lengths,
            self.hop_length,
            rounding_mode="floor",
        )
        if torch.any(feature_lengths < 2):
            raise ValueError("Cohere ASR audio must produce at least two valid frames.")
        maximum = int(lengths.max())
        padded = torch.zeros(
            len(waveforms),
            maximum,
            device=waveforms[0].device,
            dtype=torch.float32,
        )
        for index, waveform in enumerate(waveforms):
            padded[index, :waveform.numel()] = waveform
        sample_mask = (torch.arange(maximum, device=padded.device)[None, :] < lengths[:, None])
        if self.dither > 0.0:
            padded = padded.clone()
            generator = torch.Generator(device=padded.device)
            for index, length in enumerate(lengths.tolist()):
                generator.manual_seed(int(length))
                padded[index, :length] += self.dither * torch.randn(
                    length,
                    device=padded.device,
                    dtype=padded.dtype,
                    generator=generator,
                )
        emphasized = torch.cat(
            (
                padded[:, :1],
                padded[:, 1:] - self.preemphasis * padded[:, :-1],
            ),
            dim=1,
        )
        emphasized = emphasized.masked_fill(~sample_mask, 0.0)
        spectrum = torch.stft(
            emphasized,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.featurizer.window.to(
                device=padded.device,
                dtype=torch.float32,
            ),
            center=True,
            pad_mode="constant",
            return_complex=True,
        )
        power = spectrum.abs().square()
        filters = self.featurizer.fb.to(
            device=padded.device,
            dtype=power.dtype,
        )
        features = torch.matmul(filters, power)
        features = torch.log(features + LOG_ZERO_GUARD).transpose(1, 2)
        attention_mask = (
            torch.arange(
                features.shape[1],
                device=features.device,
            )[None, :] < feature_lengths[:, None])
        expanded_mask = attention_mask.unsqueeze(-1)
        masked = features * expanded_mask
        mean = masked.sum(dim=1) / feature_lengths.unsqueeze(-1)
        variance = ((masked - mean.unsqueeze(1)).square() * expanded_mask).sum(dim=1) / (feature_lengths -
                                                                                         1).unsqueeze(-1)
        standard_deviation = torch.sqrt(variance)
        features = (features - mean.unsqueeze(1)) / (standard_deviation.unsqueeze(1) + NORMALIZATION_EPSILON)
        features = features * expanded_mask
        return {
            "input_features": features,
            "attention_mask": attention_mask,
        }

    def __call__(
        self,
        audio: Any,
        *,
        sampling_rate: int,
        device: torch.device | str | None = None,
        chunk_long_audio: bool = True,
    ) -> dict[str, Any]:
        if sampling_rate != self.sampling_rate:
            raise ValueError(
                f"Cohere ASR expects {self.sampling_rate} Hz audio; received "
                f"{sampling_rate} Hz.")
        waveforms = self._waveforms(audio, device=device)
        if chunk_long_audio:
            waveforms, chunk_index = self._chunk(waveforms)
        else:
            maximum_samples = int(round(self.max_audio_clip_s * self.sampling_rate))
            if any(value.numel() > maximum_samples for value in waveforms):
                raise ValueError("Cohere ASR waveform exceeds the verified single-clip "
                                 "duration.")
            chunk_index = tuple((index, None) for index in range(len(waveforms)))
        result: dict[str, Any] = self._features(waveforms)
        result["audio_chunk_index"] = chunk_index
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_extractor_type": "VoiceHubCohereAsrFeatureExtractor",
            "feature_size": self.feature_size,
            "sampling_rate": self.sampling_rate,
            "hop_length": self.hop_length,
            "n_fft": self.n_fft,
            "win_length": self.win_length,
            "preemphasis": self.preemphasis,
            "dither": self.dither,
            "max_audio_clip_s": self.max_audio_clip_s,
            "overlap_chunk_second": self.overlap_chunk_second,
            "min_energy_window_samples": self.min_energy_window_samples,
            "padding_value": 0.0,
            "return_attention_mask": True,
        }


class CohereAsrProcessor:
    """Compose frontend, tokenizer, prompts, and teacher-forcing targets."""

    def __init__(
        self,
        feature_extractor: CohereAsrFeatureExtractor,
        tokenizer: CohereAsrTokenizer,
        config: CohereAsrConfig,
    ) -> None:
        if not isinstance(feature_extractor, CohereAsrFeatureExtractor):
            raise TypeError("Invalid Cohere ASR feature extractor.")
        if not isinstance(tokenizer, CohereAsrTokenizer):
            raise TypeError("Invalid Cohere ASR tokenizer.")
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.config = CohereAsrConfig.coerce(config)
        if tokenizer.token_id_space_size != self.config.vocab_size:
            raise ValueError("Cohere ASR tokenizer and checkpoint vocabulary sizes "
                             "disagree.")
        expected = {
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "bos_token_id": tokenizer.bos_token_id,
        }
        for name, value in expected.items():
            if getattr(self.config, name) != value:
                raise ValueError(f"Cohere ASR tokenizer and checkpoint {name} disagree.")

    @classmethod
    def from_files(
        cls,
        *,
        featurizer: FilterbankFeatures,
        config: CohereAsrConfig,
        tokenizer_path: str | Path,
        tokenizer_config_path: str | Path,
    ) -> CohereAsrProcessor:
        return cls(
            CohereAsrFeatureExtractor(featurizer, config),
            CohereAsrTokenizer.from_files(
                tokenizer_path,
                tokenizer_config_path,
            ),
            config,
        )

    def get_decoder_prompt_ids(
        self,
        language: str,
        *,
        punctuation: bool = True,
    ) -> tuple[int, ...]:
        if language not in SUPPORTED_LANGUAGES:
            supported = ", ".join(SUPPORTED_LANGUAGES)
            raise ValueError(
                f"Unsupported Cohere ASR language {language!r}; expected one "
                f"of: {supported}.")
        if not isinstance(punctuation, bool):
            raise TypeError("`punctuation` must be a boolean.")
        punctuation_token = "<|pnc|>" if punctuation else "<|nopnc|>"
        tokens = (
            "▁",
            "<|startofcontext|>",
            "<|startoftranscript|>",
            "<|emo:undefined|>",
            f"<|{language}|>",
            f"<|{language}|>",
            punctuation_token,
            "<|noitn|>",
            "<|notimestamp|>",
            "<|nodiarize|>",
        )
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        assert isinstance(ids, list)
        if self.tokenizer.unk_token_id in ids:
            raise ValueError("Cohere ASR tokenizer is missing a required prompt token.")
        return tuple(ids)

    @staticmethod
    def _texts(text: str | Sequence[str]) -> tuple[str, ...]:
        values = (text, ) if isinstance(text, str) else tuple(text)
        if not values or any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError("Cohere ASR transcripts must be non-empty strings.")
        return values

    def _training_targets(
        self,
        texts: tuple[str, ...],
        prompt_ids: tuple[int, ...],
        *,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        sequences = []
        labels = []
        prompt_label_count = len(prompt_ids) - 1
        for text in texts:
            transcript = self.tokenizer.encode(text).input_ids
            if not transcript:
                raise ValueError("Cohere ASR transcript produced no tokenizer IDs.")
            complete = prompt_ids + transcript + (self.tokenizer.eos_token_id, )
            input_row = complete[:-1]
            label_row = list(complete[1:])
            if self.config.mask_prompt_loss:
                label_row[:prompt_label_count] = [-100] * prompt_label_count
            sequences.append(input_row)
            labels.append(tuple(label_row))
        maximum = max(len(row) for row in sequences)
        input_tensor = torch.full(
            (len(sequences), maximum),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        label_tensor = torch.full(
            (len(sequences), maximum),
            -100,
            dtype=torch.long,
            device=device,
        )
        decoder_mask = torch.zeros(
            len(sequences),
            maximum,
            dtype=torch.bool,
            device=device,
        )
        for index, (input_row, label_row) in enumerate(zip(sequences, labels)):
            length = len(input_row)
            input_tensor[index, :length] = torch.tensor(
                input_row,
                dtype=torch.long,
                device=device,
            )
            label_tensor[index, :length] = torch.tensor(
                label_row,
                dtype=torch.long,
                device=device,
            )
            decoder_mask[index, :length] = True
        return {
            "decoder_input_ids": input_tensor,
            "decoder_attention_mask": decoder_mask,
            "labels": label_tensor,
        }

    def __call__(
        self,
        audio: Any,
        *,
        language: str,
        text: str | Sequence[str] | None = None,
        punctuation: bool = True,
        sampling_rate: int = 16_000,
        device: torch.device | str | None = None,
    ) -> dict[str, Any]:
        prepared = self.feature_extractor(
            audio,
            sampling_rate=sampling_rate,
            device=device,
            chunk_long_audio=text is None,
        )
        prompt_ids = self.get_decoder_prompt_ids(
            language,
            punctuation=punctuation,
        )
        batch = prepared["input_features"].shape[0]
        target_device = prepared["input_features"].device
        if text is None:
            prepared["decoder_input_ids"] = torch.tensor(
                [prompt_ids] * batch,
                dtype=torch.long,
                device=target_device,
            )
            prepared["decoder_attention_mask"] = torch.ones(
                batch,
                len(prompt_ids),
                dtype=torch.bool,
                device=target_device,
            )
            return prepared
        texts = self._texts(text)
        if len(texts) != batch:
            raise ValueError("Cohere ASR training requires one transcript per waveform.")
        prepared.update(self._training_targets(
            texts,
            prompt_ids,
            device=target_device,
        ))
        return prepared

    @staticmethod
    def reassemble_chunk_texts(
        texts: Sequence[str],
        audio_chunk_index: Sequence[tuple[int, int | None]],
        *,
        language: str,
    ) -> tuple[str, ...]:
        if len(texts) != len(audio_chunk_index):
            raise ValueError("Cohere ASR chunk texts and index lengths disagree.")
        if not audio_chunk_index:
            return ()
        maximum = max(sample for sample, _ in audio_chunk_index)
        output = [""] * (maximum + 1)
        chunked: dict[int, list[tuple[int, str]]] = {}
        for (sample, chunk), text in zip(audio_chunk_index, texts):
            if chunk is None:
                output[sample] = text
            else:
                chunked.setdefault(sample, []).append((chunk, text))
        separator = "" if language in _NO_SPACE_LANGUAGES else " "
        for sample, values in chunked.items():
            ordered = [text.strip() for _, text in sorted(values) if text and text.strip()]
            output[sample] = separator.join(ordered)
        return tuple(output)

    def save_pretrained(self, directory: str | Path) -> Path:
        destination = Path(directory).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(destination)
        (destination / "preprocessor_config.json").write_text(
            json.dumps(
                self.feature_extractor.to_dict(),
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        (destination / "processor_config.json").write_text(
            json.dumps(
                {
                    "processor_class": "VoiceHubCohereAsrProcessor",
                    "supported_languages": list(SUPPORTED_LANGUAGES),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        return destination


__all__ = [
    "CohereAsrFeatureExtractor",
    "CohereAsrProcessor",
]
