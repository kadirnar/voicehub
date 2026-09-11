from __future__ import annotations

import json
from pathlib import Path

import torch

from voicehub.checkpointing import save_safetensors
from voicehub.models.asr_moonshine.native import MoonshineConfig, MoonshineForConditionalGeneration

TINY_MOONSHINE_VOCABULARY = {
    "<unk>": 0,
    "<s>": 1,
    "</s>": 2,
    "▁": 3,
    "h": 4,
    "i": 5,
    "▁h": 6,
    "▁hi": 7,
    "!": 8,
    "<0xF0>": 9,
    "<0x9F>": 10,
    "<0x99>": 11,
    "<0x82>": 12,
}


def tiny_moonshine_config(**overrides) -> MoonshineConfig:
    values = {
        "vocab_size": len(TINY_MOONSHINE_VOCABULARY),
        "hidden_size": 8,
        "intermediate_size": 16,
        "encoder_num_hidden_layers": 1,
        "decoder_num_hidden_layers": 1,
        "encoder_num_attention_heads": 2,
        "decoder_num_attention_heads": 2,
        "partial_rotary_factor": 0.5,
        "max_position_embeddings": 12,
        "pad_head_dim_to_multiple_of": 4,
    }
    values.update(overrides)
    return MoonshineConfig(**values)


def tiny_tokenizer_document() -> dict:
    added_tokens = [{
        "id": token_id,
        "content": token,
        "single_word": False,
        "lstrip": False,
        "rstrip": False,
        "normalized": False,
        "special": True,
    } for token, token_id in tuple(TINY_MOONSHINE_VOCABULARY.items())[:3]]
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added_tokens,
        "normalizer": {
            "type":
            "Sequence",
            "normalizers": [
                {
                    "type": "Prepend",
                    "prepend": "▁"
                },
                {
                    "type": "Replace",
                    "pattern": {
                        "String": " "
                    },
                    "content": "▁",
                },
            ],
        },
        "pre_tokenizer": None,
        "post_processor": {
            "type":
            "TemplateProcessing",
            "single": [
                {
                    "SpecialToken": {
                        "id": "<s>",
                        "type_id": 0
                    }
                },
                {
                    "Sequence": {
                        "id": "A",
                        "type_id": 0
                    }
                },
            ],
            "pair": [
                {
                    "SpecialToken": {
                        "id": "<s>",
                        "type_id": 0
                    }
                },
                {
                    "Sequence": {
                        "id": "A",
                        "type_id": 0
                    }
                },
                {
                    "SpecialToken": {
                        "id": "<s>",
                        "type_id": 1
                    }
                },
                {
                    "Sequence": {
                        "id": "B",
                        "type_id": 1
                    }
                },
            ],
            "special_tokens": {
                "<s>": {
                    "id": "<s>",
                    "ids": [1],
                    "tokens": ["<s>"],
                }
            },
        },
        "decoder": {
            "type":
            "Sequence",
            "decoders": [
                {
                    "type": "Replace",
                    "pattern": {
                        "String": "▁"
                    },
                    "content": " ",
                },
                {
                    "type": "ByteFallback"
                },
                {
                    "type": "Fuse"
                },
                {
                    "type": "Strip",
                    "content": " ",
                    "start": 1,
                    "stop": 0,
                },
            ],
        },
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": "<unk>",
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": True,
            "byte_fallback": True,
            "ignore_merges": False,
            "vocab": TINY_MOONSHINE_VOCABULARY,
            "merges": ["▁ h", "▁h i"],
        },
    }


def write_tiny_moonshine_artifact(root: Path, ) -> tuple[MoonshineConfig, MoonshineForConditionalGeneration]:
    torch.manual_seed(907)
    config = tiny_moonshine_config()
    reference = MoonshineForConditionalGeneration(config)
    (root / "config.json").write_text(
        json.dumps(config.to_dict()),
        encoding="utf-8",
    )
    (root / "generation_config.json").write_text(
        json.dumps({
            "_from_model_config": True,
            "bos_token_id": config.bos_token_id,
            "decoder_start_token_id": config.decoder_start_token_id,
            "eos_token_id": config.eos_token_id,
            "pad_token_id": config.pad_token_id,
            "max_length": config.max_position_embeddings,
        }),
        encoding="utf-8",
    )
    (root / "preprocessor_config.json").write_text(
        json.dumps({
            "do_normalize": False,
            "feature_extractor_type": "Wav2Vec2FeatureExtractor",
            "feature_size": 1,
            "padding_side": "right",
            "padding_value": 0.0,
            "return_attention_mask": True,
            "sampling_rate": 16_000,
        }),
        encoding="utf-8",
    )
    (root / "tokenizer.json").write_text(
        json.dumps(tiny_tokenizer_document(), ensure_ascii=False),
        encoding="utf-8",
    )
    save_safetensors(
        reference.state_dict(),
        root / "model.safetensors",
    )
    return config, reference
