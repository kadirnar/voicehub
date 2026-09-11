"""Pinned provenance and checkpoint contracts for native OuteTTS."""

from __future__ import annotations

from types import MappingProxyType

OUTETTS_SOURCE_REPOSITORY = "edwko/OuteTTS"
OUTETTS_SOURCE_REVISION = "f5eac6e70d792844c6a6959d900a47af2c061a5b"
OUTETTS_SOURCE_LICENSE = "Apache-2.0"
OUTETTS_TRAINING_SOURCE_REVISION = "05f51ffd06769a10155e77523cb9b32e8dc55d9a"

OUTETTS_CHECKPOINTS = MappingProxyType({
    "OuteAI/Llama-OuteTTS-1.0-1B":
    MappingProxyType({
        "revision": "f445d0d301c7fc9e47fd545eb9f9d2912afe7a77",
        "family": "llama",
        "license": "CC-BY-NC-SA-4.0",
        "checkpoint_sha256": "03543805868adcbc81a2259177d626f557cfa8be6fa19480a47ba30a920b8003",
        "checkpoint_size": 2_496_811_440,
        "tensor_count": 146,
        "value_count": 1_248_397_312,
        "inventory_fingerprint": "d0abf3c1f2c3da50018193a0d70c3fedba575f07faa206b957298c25c2a4bf36",
        "config_sha256": "a7635d5a8bc28e6932e0ce559daf4916cc2eabf7862b1e1f22625f5fd8870636",
        "config_size": 838,
        "tokenizer_sha256": "ec54a55e3c6fc7318ea02cbd1a6eb1fb180bff1b58acbf068504a25f1b407b7b",
        "tokenizer_size": 18_366_636,
    }),
    "OuteAI/OuteTTS-1.0-0.6B":
    MappingProxyType({
        "revision": "e7bcd87b0ca47fd8c46317c8f745a5e4e19c7b5c",
        "family": "qwen3",
        "license": "Apache-2.0",
        "checkpoint_sha256": "aa74d15aaa97b766fc40e9e61e8c67d205d84b7a0934354fa0309c0085cb55fe",
        "checkpoint_size": 1_204_062_656,
        "tensor_count": 310,
        "value_count": 602_013_696,
        "inventory_fingerprint": "414b5b7b1b890218ab0cd269bfacff25c9a8dc716994af2013b43dad4c115d6e",
        "config_sha256": "86c56de4ef688e91b92a2f4aec6601da234be981fa9bc93ff0d7215e6afb3448",
        "config_size": 753,
        "tokenizer_sha256": "028c21d3dc9a635a658abc0ad900ae6b7884b55333bc9c18774c1de355f41b1a",
        "tokenizer_size": 12_579_186,
    }),
})

OUTETTS_DAC = MappingProxyType({
    "repository":
    "ibm-research/DAC.speech.v1.0",
    "revision":
    "1ea7f64cd0678415e2d8c32d67b190722cb9b149",
    "filename":
    "weights_24khz_1.5kbps_v1.0.pth",
    "license":
    "CDLA-Permissive-2.0",
    "sha256":
    "d77ca0b04df942ec64e6a7a162bcac093b1127700acdaec0079f40d32c4405fb",
    "size":
    295_731_578,
    "tensor_count":
    252,
    "value_count":
    73_909_506,
    "inventory_fingerprint":
    "c145b57cfa3d9b7976db61060f748ffbc02c9bf3bc2360c15129373e6461bea8",
})

OUTETTS_TRAINING_SOURCE = MappingProxyType({
    "repository": OUTETTS_SOURCE_REPOSITORY,
    "revision": OUTETTS_TRAINING_SOURCE_REVISION,
    "recipe": "docs/finetuning.md",
    "objective": "completion-only-causal-language-modeling",
})

NATIVE_OUTETTS_FORMAT = "voicehub-outetts-v1"

__all__ = [
    "NATIVE_OUTETTS_FORMAT",
    "OUTETTS_CHECKPOINTS",
    "OUTETTS_DAC",
    "OUTETTS_SOURCE_LICENSE",
    "OUTETTS_SOURCE_REPOSITORY",
    "OUTETTS_SOURCE_REVISION",
    "OUTETTS_TRAINING_SOURCE",
    "OUTETTS_TRAINING_SOURCE_REVISION",
]
