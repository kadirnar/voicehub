"""Native text, raw-audio, and masked-token preparation for OmniVoice."""

from __future__ import annotations

import re
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional

from voicehub.audio import load_audio
from voicehub.models.asr_qwen3.native.tokenization import qwen2_pretokenize
from voicehub.models.omnivoice.native.codec import HiggsAudioV2Tokenizer
from voicehub.models.omnivoice.native.configuration import OmniVoiceArchitectureConfig
from voicehub.tokenization import ByteBPETokenizer, Encoding

END_OF_TEXT = "<|endoftext|>"
IM_END = "<|im_end|>"
DENOISE = "<|denoise|>"
LANG_START = "<|lang_start|>"
LANG_END = "<|lang_end|>"
INSTRUCT_START = "<|instruct_start|>"
INSTRUCT_END = "<|instruct_end|>"
TEXT_START = "<|text_start|>"
TEXT_END = "<|text_end|>"

PUBLISHED_TOKEN_IDS = {
    END_OF_TEXT: 151_643,
    IM_END: 151_645,
    DENOISE: 151_669,
    LANG_START: 151_670,
    LANG_END: 151_671,
    INSTRUCT_START: 151_672,
    INSTRUCT_END: 151_673,
    TEXT_START: 151_674,
    TEXT_END: 151_675,
}

_NONVERBAL_PATTERN = re.compile(
    r"\[(laughter|sigh|confirmation-en|question-en|question-ah|question-oh|"
    r"question-ei|question-yi|surprise-ah|surprise-oh|surprise-wa|"
    r"surprise-yo|dissatisfaction-hnn)\]")


class OmniVoiceTokenizer:
    """Exact Qwen2 byte-BPE boundary from the OmniVoice checkpoint."""

    def __init__(
        self,
        tokenizer: ByteBPETokenizer,
        *,
        tokenizer_json_path: Path | None = None,
        validate_published_ids: bool = True,
    ) -> None:
        if not isinstance(tokenizer, ByteBPETokenizer):
            raise TypeError("`tokenizer` must be a ByteBPETokenizer.")
        if validate_published_ids:
            if tokenizer.token_id_space_size != 151_676:
                raise ValueError("OmniVoice tokenizer must declare exactly 151,676 IDs.")
            available = {
                **dict(tokenizer.special_tokens),
                **dict(tokenizer.added_tokens),
            }
            for spelling, expected in PUBLISHED_TOKEN_IDS.items():
                actual = available.get(spelling)
                if actual != expected:
                    raise ValueError(
                        f"OmniVoice token {spelling!r} must use ID "
                        f"{expected}; found {actual!r}.")
        self._tokenizer = tokenizer
        self.tokenizer_json_path = tokenizer_json_path

    @classmethod
    def from_tokenizer_json(
        cls,
        path: str | Path,
    ) -> OmniVoiceTokenizer:
        source = Path(path).expanduser().resolve()
        tokenizer = ByteBPETokenizer.from_tokenizer_json(
            source,
            pretokenizer=qwen2_pretokenize,
            pad_token_id=PUBLISHED_TOKEN_IDS[END_OF_TEXT],
            padding_side="right",
        )
        return cls(tokenizer, tokenizer_json_path=source)

    @property
    def pad_token_id(self) -> int:
        value = self._tokenizer.pad_token_id
        if value is None:
            raise RuntimeError("OmniVoice tokenizer has no pad token.")
        return value

    @property
    def eos_token_id(self) -> int:
        return PUBLISHED_TOKEN_IDS[IM_END]

    @property
    def vocabulary_size(self) -> int:
        return self._tokenizer.vocabulary_size

    @property
    def token_id_space_size(self) -> int:
        return self._tokenizer.token_id_space_size

    def encode(self, text: str) -> Encoding:
        return self._tokenizer.encode(
            text,
            allowed_special="all",
            disallowed_special=(),
        )

    def encode_nonverbal_text(self, text: str) -> tuple[int, ...]:
        """Tokenize supported non-verbal tags independently of context."""
        if not isinstance(text, str):
            raise TypeError("`text` must be a string.")
        pieces: list[int] = []
        cursor = 0
        for match in _NONVERBAL_PATTERN.finditer(text):
            if match.start() > cursor:
                pieces.extend(self.encode(text[cursor:match.start()]).input_ids)
            pieces.extend(self.encode(match.group()).input_ids)
            cursor = match.end()
        if cursor < len(text):
            pieces.extend(self.encode(text[cursor:]).input_ids)
        if not pieces and text:
            pieces.extend(self.encode(text).input_ids)
        return tuple(pieces)

    def decode(
        self,
        token_ids,
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        return self._tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
        )

    def save_pretrained(self, directory: str | Path) -> Path:
        if self.tokenizer_json_path is None:
            raise RuntimeError("An in-memory tokenizer has no source asset to export.")
        destination = Path(directory).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / "tokenizer.json"
        if target.resolve() != self.tokenizer_json_path:
            shutil.copy2(self.tokenizer_json_path, target)
        return target


def combine_text(text: str, reference_text: str | None = None) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("OmniVoice target text must be non-empty.")
    if reference_text is not None and not isinstance(reference_text, str):
        raise TypeError("`reference_text` must be a string or None.")
    result = (
        f"{reference_text.strip()} {text.strip()}"
        if reference_text and reference_text.strip() else text.strip())
    result = re.sub(r"[\r\n]+", "", result)
    result = result.replace("\uff08", "(").replace("\uff09", ")")
    result = re.sub(r"[ \t]+", " ", result)
    result = re.sub(
        r"(?<=[\u4e00-\u9fff])\s+|\s+(?=[\u4e00-\u9fff])",
        "",
        result,
    )
    return result


def style_prompt(
    *,
    language: str | None,
    instruction: str | None,
    denoise: bool,
) -> str:
    if language is not None and not isinstance(language, str):
        raise TypeError("`language` must be a string or None.")
    if instruction is not None and not isinstance(instruction, str):
        raise TypeError("`instruction` must be a string or None.")
    prefix = DENOISE if denoise else ""
    return (
        f"{prefix}{LANG_START}{language or 'None'}{LANG_END}"
        f"{INSTRUCT_START}{instruction or 'None'}{INSTRUCT_END}")


@dataclass(frozen=True, slots=True)
class OmniVoiceMaskingConfig:
    """Published stochastic fine-tuning policy."""

    prompt_ratio_range: tuple[float, float] = (0.0, 0.3)
    mask_ratio_range: tuple[float, float] = (0.0, 1.0)
    drop_cond_ratio: float = 0.1
    language_ratio: float = 0.8
    use_pinyin_ratio: float = 0.3
    instruct_ratio: float = 1.0
    only_instruct_ratio: float = 0.5
    normalize_raw_audio: bool = True

    def __post_init__(self) -> None:
        for name in ("prompt_ratio_range", "mask_ratio_range"):
            values = tuple(float(value) for value in getattr(self, name))
            if len(values) != 2 or not 0 <= values[0] <= values[1] <= 1:
                raise ValueError(f"`{name}` must be an ordered pair in [0, 1].")
            object.__setattr__(self, name, values)
        for name in (
                "drop_cond_ratio",
                "language_ratio",
                "use_pinyin_ratio",
                "instruct_ratio",
                "only_instruct_ratio",
        ):
            value = float(getattr(self, name))
            if not 0 <= value <= 1:
                raise ValueError(f"`{name}` must be in [0, 1].")
            object.__setattr__(self, name, value)
        if not isinstance(self.normalize_raw_audio, bool):
            raise TypeError("`normalize_raw_audio` must be a boolean.")


def _random_uniform(
    bounds: tuple[float, float],
    *,
    generator: torch.Generator | None,
) -> float:
    low, high = bounds
    if low == high:
        return low
    value = torch.rand((), generator=generator).item()
    return low + (high - low) * value


def _random_event(
    probability: float,
    *,
    generator: torch.Generator | None,
) -> bool:
    if probability <= 0:
        return False
    if probability >= 1:
        return True
    return bool(torch.rand((), generator=generator).item() < probability)


def _metadata(record: Mapping[str, Any]) -> dict[str, Any]:
    label = record.get("label")
    if label is None:
        return dict(record)
    if not isinstance(label, Mapping):
        raise TypeError("OmniVoice `label` must be a mapping.")
    merged = dict(record)
    merged.update(label)
    return merged


def _mono_waveform(value: Any) -> Tensor:
    waveform = torch.as_tensor(value)
    if not waveform.is_floating_point():
        waveform = waveform.float()
    if waveform.ndim == 1:
        return waveform
    if waveform.ndim != 2:
        raise ValueError("Raw audio must have shape [time] or [channel, time].")
    if waveform.shape[0] <= 8:
        return waveform.mean(dim=0)
    if waveform.shape[1] <= 8:
        return waveform.mean(dim=1)
    raise ValueError("Raw audio does not have a plausible channel axis.")


class OmniVoiceSampleProcessor:
    """Convert raw or pre-tokenized records into exact training examples."""

    def __init__(
        self,
        tokenizer: OmniVoiceTokenizer,
        model_config: OmniVoiceArchitectureConfig,
        *,
        masking: OmniVoiceMaskingConfig | None = None,
        audio_tokenizer: HiggsAudioV2Tokenizer | None = None,
    ) -> None:
        if not isinstance(tokenizer, OmniVoiceTokenizer):
            raise TypeError("`tokenizer` must be an OmniVoiceTokenizer.")
        if not isinstance(model_config, OmniVoiceArchitectureConfig):
            raise TypeError("`model_config` must be an OmniVoiceArchitectureConfig.")
        if (audio_tokenizer is not None and not isinstance(audio_tokenizer, HiggsAudioV2Tokenizer)):
            raise TypeError("`audio_tokenizer` must be a HiggsAudioV2Tokenizer or None.")
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.masking = masking or OmniVoiceMaskingConfig()
        self.audio_tokenizer = audio_tokenizer

    def _audio_tokens(self, record: Mapping[str, Any]) -> Tensor:
        supplied = record.get("audio_tokens")
        if supplied is not None:
            tokens = torch.as_tensor(supplied)
        else:
            if self.audio_tokenizer is None:
                raise ValueError(
                    "Raw-waveform OmniVoice preparation requires the native "
                    "Higgs Audio V2 tokenizer.")
            value = record.get("waveform", record.get("audio"))
            if value is None:
                raise ValueError("OmniVoice records require `audio_tokens`, `waveform`, "
                                 "or `audio`.")
            sample_rate = record.get(
                "sampling_rate",
                record.get("sample_rate"),
            )
            if sample_rate is not None and (isinstance(sample_rate, bool) or
                                            not isinstance(sample_rate, int) or sample_rate <= 0):
                raise ValueError(
                    "Raw OmniVoice `sampling_rate`, when provided, must be "
                    "a positive integer.")
            loaded = load_audio(
                value,
                sampling_rate=sample_rate,
                target_sampling_rate=self.audio_tokenizer.sample_rate,
            )
            waveform = _mono_waveform(loaded.waveform)
            if self.masking.normalize_raw_audio:
                waveform = (waveform / (waveform.abs().max() + 1e-7) * 0.9)
            codec_device = self.audio_tokenizer.device
            tokens = self.audio_tokenizer.encode(waveform.to(codec_device)[None,
                                                                           None, :]).audio_codes[0].cpu()
        if (tokens.dtype == torch.bool or tokens.is_floating_point() or tokens.is_complex()):
            raise TypeError("`audio_tokens` must use an integer dtype.")
        tokens = tokens.long()
        if (tokens.ndim != 2 or tokens.shape[0] != self.model_config.num_audio_codebook or
                tokens.shape[1] == 0):
            raise ValueError(
                "`audio_tokens` must have shape "
                f"[{self.model_config.num_audio_codebook}, frames].")
        if ((tokens < 0).any() or (tokens >= self.model_config.audio_mask_id).any()):
            raise ValueError(
                "Preprocessed OmniVoice tokens must be real codec IDs; "
                "the mask ID is reserved for online corruption.")
        return tokens

    def __call__(
        self,
        record: Mapping[str, Any],
        *,
        generator: torch.Generator | None = None,
    ) -> dict[str, Tensor | int]:
        if not isinstance(record, Mapping):
            raise TypeError("OmniVoice training records must be mappings.")
        metadata = _metadata(record)
        clean_start = metadata.get("clean_start_token_idx")
        if clean_start is not None:
            if (isinstance(clean_start, bool) or not isinstance(clean_start, int) or clean_start < 0):
                raise ValueError("`clean_start_token_idx` must be a non-negative integer.")
            drop_conditioning = False
        else:
            drop_conditioning = _random_event(
                self.masking.drop_cond_ratio,
                generator=generator,
            )

        if drop_conditioning:
            prompt_ratio = 0.0
            use_language = False
            use_instruction = False
        else:
            prompt_ratio = _random_uniform(
                self.masking.prompt_ratio_range,
                generator=generator,
            )
            use_language = _random_event(
                self.masking.language_ratio,
                generator=generator,
            )
            use_instruction = _random_event(
                self.masking.instruct_ratio,
                generator=generator,
            )
            if use_instruction and _random_event(
                    self.masking.only_instruct_ratio,
                    generator=generator,
            ):
                prompt_ratio = 0.0
        mask_ratio = _random_uniform(
            self.masking.mask_ratio_range,
            generator=generator,
        )

        language = (str(metadata.get("language_id", "None")) if use_language else "None")
        instruction = (str(metadata.get("instruct", "None")) if use_instruction else "None")
        style_ids = torch.tensor(
            self.tokenizer.encode(
                style_prompt(
                    language=language,
                    instruction=instruction,
                    denoise=clean_start is not None,
                )).input_ids,
            dtype=torch.long,
        ).repeat(self.model_config.num_audio_codebook, 1)
        style_labels = torch.full_like(style_ids, -100)

        text_value = metadata.get("text")
        if not isinstance(text_value, str) or not text_value:
            raise ValueError("OmniVoice training records require non-empty `text`.")
        pinyin = metadata.get("text_pinyin")
        if (isinstance(pinyin, str) and _random_event(
                self.masking.use_pinyin_ratio,
                generator=generator,
        )):
            text_value = pinyin
        text_ids = torch.tensor(
            self.tokenizer.encode_nonverbal_text(f"{TEXT_START}{text_value}{TEXT_END}"),
            dtype=torch.long,
        ).repeat(self.model_config.num_audio_codebook, 1)
        text_labels = torch.full_like(text_ids, -100)

        audio_tokens = self._audio_tokens(record)
        prompt_length = (
            clean_start if clean_start is not None else int(audio_tokens.shape[1] * prompt_ratio))
        if prompt_length > audio_tokens.shape[1]:
            raise ValueError("`clean_start_token_idx` exceeds the audio token length.")
        audio_inputs = audio_tokens.clone()
        audio_labels = audio_tokens.clone()
        region = audio_tokens[:, prompt_length:]
        token_mask = (torch.rand(
            region.shape,
            generator=generator,
            device=region.device,
        ) < mask_ratio)
        audio_inputs[:, prompt_length:][token_mask] = (self.model_config.audio_mask_id)
        audio_labels[:, prompt_length:][~token_mask] = -100
        if not drop_conditioning:
            audio_labels[:, :prompt_length] = -100

        if drop_conditioning:
            input_ids = audio_inputs
            labels = audio_labels
            audio_mask = torch.ones(
                input_ids.shape[1],
                dtype=torch.bool,
            )
        else:
            input_ids = torch.cat(
                [style_ids, text_ids, audio_inputs],
                dim=1,
            )
            labels = torch.cat(
                [style_labels, text_labels, audio_labels],
                dim=1,
            )
            audio_mask = torch.zeros(
                input_ids.shape[1],
                dtype=torch.bool,
            )
            audio_mask[style_ids.shape[1] + text_ids.shape[1]:] = True
        return {
            "input_ids": input_ids,
            "labels": labels,
            "audio_mask": audio_mask,
            "length": input_ids.shape[1],
        }


class OmniVoicePaddingCollator:
    """Pad processed examples for native bidirectional attention."""

    def __init__(self, pad_token_id: int) -> None:
        if isinstance(pad_token_id, bool) or not isinstance(pad_token_id, int):
            raise TypeError("`pad_token_id` must be an integer.")
        self.pad_token_id = pad_token_id

    def __call__(
        self,
        samples: Sequence[Mapping[str, Any]],
    ) -> dict[str, Tensor]:
        if not samples:
            return {}
        maximum = max(int(sample["length"]) for sample in samples)
        batch_size = len(samples)
        input_ids = []
        labels = []
        audio_masks = []
        positions = []
        valid = torch.zeros(batch_size, maximum, dtype=torch.bool)
        for index, sample in enumerate(samples):
            length = int(sample["length"])
            padding = maximum - length
            input_ids.append(functional.pad(
                sample["input_ids"],
                (0, padding),
                value=self.pad_token_id,
            ))
            labels.append(functional.pad(
                sample["labels"],
                (0, padding),
                value=-100,
            ))
            audio_masks.append(functional.pad(
                sample["audio_mask"],
                (0, padding),
                value=False,
            ))
            positions.append(functional.pad(
                torch.arange(length),
                (0, padding),
                value=0,
            ))
            valid[index, :length] = True
        attention_mask = valid[:, None, None, :].expand(
            batch_size,
            1,
            maximum,
            maximum,
        ).contiguous()
        return {
            "input_ids": torch.stack(input_ids),
            "labels": torch.stack(labels),
            "audio_mask": torch.stack(audio_masks),
            "position_ids": torch.stack(positions),
            "attention_mask": attention_mask,
        }


class OmniVoicePackingCollator:
    """Pack processed examples while retaining document boundaries."""

    def __init__(self, pad_token_id: int, batch_tokens: int) -> None:
        if (isinstance(batch_tokens, bool) or not isinstance(batch_tokens, int) or batch_tokens <= 0):
            raise ValueError("`batch_tokens` must be a positive integer.")
        self.pad_token_id = int(pad_token_id)
        self.batch_tokens = batch_tokens

    def __call__(
        self,
        samples: Sequence[Mapping[str, Any]],
    ) -> dict[str, Tensor]:
        if not samples:
            return {}
        used = sum(int(sample["length"]) for sample in samples)
        if used > self.batch_tokens:
            raise ValueError("Packed OmniVoice examples exceed `batch_tokens`.")
        padding = self.batch_tokens - used
        input_ids = functional.pad(
            torch.cat([sample["input_ids"] for sample in samples], dim=1),
            (0, padding),
            value=self.pad_token_id,
        )
        labels = functional.pad(
            torch.cat([sample["labels"] for sample in samples], dim=1),
            (0, padding),
            value=-100,
        )
        audio_mask = functional.pad(
            torch.cat([sample["audio_mask"] for sample in samples]),
            (0, padding),
            value=False,
        )
        position_ids = functional.pad(
            torch.cat([torch.arange(int(sample["length"])) for sample in samples]),
            (0, padding),
            value=0,
        )
        document_ids = functional.pad(
            torch.cat([
                torch.full(
                    (int(sample["length"]), ),
                    index,
                    dtype=torch.int32,
                ) for index, sample in enumerate(samples)
            ]),
            (0, padding),
            value=-1,
        )
        return {
            "input_ids": input_ids.unsqueeze(0),
            "labels": labels.unsqueeze(0),
            "audio_mask": audio_mask.unsqueeze(0),
            "position_ids": position_ids.unsqueeze(0),
            "document_ids": document_ids.unsqueeze(0),
        }


__all__ = [
    "DENOISE",
    "END_OF_TEXT",
    "IM_END",
    "INSTRUCT_END",
    "INSTRUCT_START",
    "LANG_END",
    "LANG_START",
    "PUBLISHED_TOKEN_IDS",
    "TEXT_END",
    "TEXT_START",
    "OmniVoiceMaskingConfig",
    "OmniVoicePackingCollator",
    "OmniVoicePaddingCollator",
    "OmniVoiceSampleProcessor",
    "OmniVoiceTokenizer",
    "combine_text",
    "style_prompt",
]
