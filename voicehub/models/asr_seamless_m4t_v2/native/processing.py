"""Joint native audio/text processing for SeamlessM4T-v2 S2T."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path

from torch import Tensor

from voicehub.models.asr_seamless_m4t_v2.native.configuration import SeamlessM4Tv2S2TConfig
from voicehub.models.asr_seamless_m4t_v2.native.frontend import SeamlessM4Tv2FeatureBatch, SeamlessM4Tv2FeatureExtractor
from voicehub.models.asr_seamless_m4t_v2.native.tokenization import (
    SEAMLESS_M4T_V2_LANGUAGE_TO_ID,
    SeamlessM4Tv2Tokenizer,
)


class SeamlessM4Tv2Processor:
    """One immutable processor for inference and full-model fine-tuning."""

    def __init__(
        self,
        config: SeamlessM4Tv2S2TConfig,
        tokenizer: SeamlessM4Tv2Tokenizer,
    ) -> None:
        if not isinstance(config, SeamlessM4Tv2S2TConfig):
            raise TypeError("`config` must be SeamlessM4Tv2S2TConfig.")
        if not isinstance(tokenizer, SeamlessM4Tv2Tokenizer):
            raise TypeError("`tokenizer` must be SeamlessM4Tv2Tokenizer.")
        if config.vocab_size < max(tokenizer.language_to_id.values()) + 1:
            raise ValueError("The S2T vocabulary cannot represent every language token.")
        self.config = config
        self.tokenizer = tokenizer
        self.feature_extractor = SeamlessM4Tv2FeatureExtractor(config)

    @classmethod
    def from_files(
        cls,
        config: SeamlessM4Tv2S2TConfig,
        tokenizer_model: str | Path,
        *,
        added_tokens: str | Path,
    ) -> SeamlessM4Tv2Processor:
        return cls(
            config,
            SeamlessM4Tv2Tokenizer.from_files(
                tokenizer_model,
                added_tokens=added_tokens,
                expected_sentencepiece_size=(256_000 if config.variant == "seamless-m4t-v2-large" else None),
            ),
        )

    def process_audio(
        self,
        waveforms: Tensor | Sequence[Tensor],
        *,
        sampling_rate: int,
    ) -> SeamlessM4Tv2FeatureBatch:
        return self.feature_extractor(
            waveforms,
            sampling_rate=sampling_rate,
        )

    def encode_labels(
        self,
        texts: Sequence[str],
        *,
        target_language: str,
        padding_value: int = -100,
    ) -> Tensor:
        return self.tokenizer.batch_encode_targets(
            texts,
            language=target_language,
            padding_value=padding_value,
        )

    def generation_language_id(self, target_language: str) -> int:
        return self.tokenizer.language_token_id(target_language)

    def decode(
        self,
        token_ids: Iterable[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
        )

    def batch_decode(
        self,
        sequences: Iterable[Iterable[int]],
        *,
        skip_special_tokens: bool = True,
    ) -> list[str]:
        return self.tokenizer.batch_decode(
            sequences,
            skip_special_tokens=skip_special_tokens,
        )

    def save_pretrained(self, directory: str | Path) -> Path:
        destination = Path(directory).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        tokenizer_model = self.tokenizer.save_pretrained(destination)
        preprocessor = {
            "feature_extractor_type": "SeamlessM4TFeatureExtractor",
            "feature_size": self.config.num_mel_bins,
            "language_code": [f"__{language}__" for language in SEAMLESS_M4T_V2_LANGUAGE_TO_ID],
            "num_mel_bins": self.config.num_mel_bins,
            "padding_side": "right",
            "padding_value": 0.0,
            "processor_class": "SeamlessM4TProcessor",
            "return_attention_mask": True,
            "sampling_rate": self.config.sampling_rate,
            "stride": self.config.feature_stride,
        }
        tokenizer_config = {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "model_max_length": self.config.max_position_embeddings,
            "pad_token": "<pad>",
            "src_lang": "__eng__",
            "tgt_lang": "__eng__",
            "tokenizer_class": "SeamlessM4TTokenizer",
            "unk_token": "<unk>",
        }
        special_tokens = {
            "additional_special_tokens": [f"__{language}__" for language in SEAMLESS_M4T_V2_LANGUAGE_TO_ID],
            "bos_token": "<s>",
            "cls_token": "<s>",
            "eos_token": "</s>",
            "pad_token": "<pad>",
            "sep_token": "</s>",
            "unk_token": "<unk>",
        }
        for filename, values in (
            ("preprocessor_config.json", preprocessor),
            ("tokenizer_config.json", tokenizer_config),
            ("special_tokens_map.json", special_tokens),
        ):
            (destination / filename).write_text(
                json.dumps(values, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        return tokenizer_model


__all__ = ["SeamlessM4Tv2Processor"]
