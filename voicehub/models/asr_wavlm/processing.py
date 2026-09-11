"""WavLM's Wav2Vec2-compatible audio and character CTC processor."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicehub.hub import read_json_file
from voicehub.models.asr_wav2vec2.native.tokenization import Wav2Vec2CTCTokenizer
from voicehub.models.asr_wav2vec2.processing import Wav2Vec2Processor


def _special_token(value: Any, *, name: str, default: str) -> str:
    if value is None:
        return default
    if isinstance(value, Mapping):
        value = value.get("content")
    if not isinstance(value, str) or not value:
        raise ValueError(f"WavLM {name!r} must contain a token string.")
    return value


class WavLMProcessor(Wav2Vec2Processor):
    """Native processor supporting WavLM's separately stored special tokens."""

    @classmethod
    def from_artifacts(
        cls,
        *,
        vocabulary: str | Path,
        added_tokens: str | Path | None = None,
        tokenizer_config: str | Path | None,
        special_tokens_map: str | Path | None,
        preprocessor_config: str | Path | None,
        target_language: str | None = None,
    ) -> WavLMProcessor:
        """Build a processor from declarative, non-executable artifacts."""
        vocabulary_values = read_json_file(vocabulary)
        if added_tokens is not None:
            added_values = read_json_file(added_tokens)
            overlap = set(vocabulary_values) & set(added_values)
            if overlap:
                names = ", ".join(sorted(repr(name) for name in overlap))
                raise ValueError(f"WavLM added tokens duplicate vocabulary entries: {names}.")
            vocabulary_values.update(added_values)

        tokenizer_values = ({} if tokenizer_config is None else read_json_file(tokenizer_config))
        special_values = ({} if special_tokens_map is None else read_json_file(special_tokens_map))
        defaults = {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
            "word_delimiter_token": "|",
            "replace_word_delimiter_char": " ",
        }
        token_options = {
            name:
            _special_token(
                special_values.get(name, tokenizer_values.get(name)),
                name=name,
                default=default,
            )
            for name, default in defaults.items()
        }
        do_lower_case = tokenizer_values.get("do_lower_case", False)
        if not isinstance(do_lower_case, bool):
            raise TypeError("WavLM tokenizer `do_lower_case` must be a boolean.")
        resolved_language = (
            target_language if target_language is not None else tokenizer_values.get("target_lang"))
        tokenizer = Wav2Vec2CTCTokenizer(
            vocabulary_values,
            do_lower_case=do_lower_case,
            target_language=resolved_language,
            **token_options,
        )

        processor_values = ({} if preprocessor_config is None else read_json_file(preprocessor_config))
        if processor_values.get("feature_size", 1) != 1:
            raise ValueError("Native WavLM consumes raw mono waveforms and requires "
                             "`feature_size=1`.")
        if processor_values.get("padding_side", "right") != "right":
            raise ValueError("Native WavLM requires right-padded waveform batches.")
        return cls(
            tokenizer,
            sampling_rate=processor_values.get("sampling_rate", 16_000),
            do_normalize=processor_values.get("do_normalize", True),
            padding_value=processor_values.get("padding_value", 0.0),
            return_attention_mask=processor_values.get(
                "return_attention_mask",
                tokenizer_values.get("return_attention_mask", False),
            ),
        )


__all__ = ["WavLMProcessor"]
