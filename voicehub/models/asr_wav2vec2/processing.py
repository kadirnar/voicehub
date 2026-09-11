"""Native audio-and-text processor for Wav2Vec2 CTC."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from voicehub.hub import read_json_file, write_json_file
from voicehub.models.asr_wav2vec2.native.processing import Wav2Vec2FeatureExtractor
from voicehub.models.asr_wav2vec2.native.tokenization import Wav2Vec2CTCTokenizer


def _special_token(value: Any, *, name: str, default: str) -> str:
    if value is None:
        return default
    if isinstance(value, Mapping):
        value = value.get("content")
    if not isinstance(value, str) or not value:
        raise ValueError(f"Wav2Vec2 {name!r} must contain a token string.")
    return value


class Wav2Vec2Processor(Wav2Vec2FeatureExtractor):
    """Apply official waveform normalization and CTC text tokenization."""

    def __init__(
        self,
        tokenizer: Wav2Vec2CTCTokenizer,
        *,
        sampling_rate: int = 16_000,
        do_normalize: bool = True,
        padding_value: float = 0.0,
        return_attention_mask: bool = False,
    ) -> None:
        if not isinstance(tokenizer, Wav2Vec2CTCTokenizer):
            raise TypeError("`tokenizer` must be a Wav2Vec2CTCTokenizer.")
        self.tokenizer = tokenizer
        super().__init__(
            sampling_rate=sampling_rate,
            do_normalize=do_normalize,
            padding_value=padding_value,
            return_attention_mask=return_attention_mask,
        )

    @classmethod
    def from_artifacts(
        cls,
        *,
        vocabulary: str | Path,
        tokenizer_config: str | Path | None,
        special_tokens_map: str | Path | None,
        preprocessor_config: str | Path | None,
        target_language: str | None = None,
    ) -> Wav2Vec2Processor:
        """Build a processor from declarative Hugging Face files."""
        tokenizer_values = ({} if tokenizer_config is None else read_json_file(tokenizer_config))
        special_values = ({} if special_tokens_map is None else read_json_file(special_tokens_map))
        token_options = {}
        defaults = {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
            "word_delimiter_token": "|",
            "replace_word_delimiter_char": " ",
        }
        for name, default in defaults.items():
            value = special_values.get(
                name,
                tokenizer_values.get(name),
            )
            token_options[name] = _special_token(
                value,
                name=name,
                default=default,
            )
        do_lower_case = tokenizer_values.get("do_lower_case", False)
        if not isinstance(do_lower_case, bool):
            raise TypeError("Wav2Vec2 tokenizer `do_lower_case` must be a boolean.")
        resolved_language = (
            target_language if target_language is not None else tokenizer_values.get("target_lang"))
        tokenizer = Wav2Vec2CTCTokenizer.from_vocab_file(
            vocabulary,
            do_lower_case=do_lower_case,
            target_language=resolved_language,
            **token_options,
        )

        processor_values = ({} if preprocessor_config is None else read_json_file(preprocessor_config))
        feature_size = processor_values.get("feature_size", 1)
        if feature_size != 1:
            raise ValueError("Native Wav2Vec2 consumes raw mono waveforms and requires "
                             "`feature_size=1`.")
        padding_side = processor_values.get("padding_side", "right")
        if padding_side != "right":
            raise ValueError("Native Wav2Vec2 requires right-padded waveform batches.")
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

    def encode_labels(
        self,
        texts: Sequence[str],
        *,
        pad: bool,
    ) -> Any:
        """Encode one or more transcripts as PyTorch label tensors."""
        import torch

        values = tuple(texts)
        if not values:
            raise ValueError("Wav2Vec2 transcript batch cannot be empty.")
        if any(not isinstance(text, str) or not text.strip() for text in values):
            raise ValueError("Wav2Vec2 transcripts must contain non-empty strings.")
        encodings = tuple(self.tokenizer.encode(text) for text in values)
        if not pad:
            if len(encodings) != 1:
                raise ValueError("Unpadded Wav2Vec2 labels require one transcript.")
            return torch.tensor(
                encodings[0].input_ids,
                dtype=torch.long,
            )
        maximum = max(len(encoding) for encoding in encodings)
        labels = torch.full(
            (len(encodings), maximum),
            -100,
            dtype=torch.long,
        )
        for index, encoding in enumerate(encodings):
            if encoding.input_ids:
                labels[index, :len(encoding)] = torch.tensor(
                    encoding.input_ids,
                    dtype=torch.long,
                )
        return labels

    def to_dict(self) -> dict[str, Any]:
        """Return the official preprocessor fields."""
        values = super().to_dict()
        values["processor_class"] = "Wav2Vec2Processor"
        return values

    def save_pretrained(self, directory: str | Path) -> None:
        """Write the complete vocabulary and preprocessing contract."""
        destination = Path(directory).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(destination)
        write_json_file(
            destination / "preprocessor_config.json",
            self.to_dict(),
        )


__all__ = ["Wav2Vec2Processor"]
