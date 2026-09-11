"""Immutable provenance and reference checkpoint metadata for NeuTTS."""

from __future__ import annotations

from types import MappingProxyType

NEUTTS_SOURCE_REPOSITORY = "neuphonic/neutts"
NEUTTS_SOURCE_REVISION = "ac69851f28fc63a487917e7c2e27f0d75c759cba"
NEUCODEC_SOURCE_REPOSITORY = "neuphonic/neucodec"
NEUCODEC_SOURCE_REVISION = "ed3e6cd1bdc374ce14a21355e5eee66a777149ce"
NEUCODEC_TRANSFORMERS_REFERENCE_REVISION = ("91d0737eea53da41c98b45da1cd8116e518aad21")

NEUTTS_VARIANTS = MappingProxyType({
    "neuphonic/neutts-air":
    MappingProxyType({
        "revision":
        "a1f5d69afe0fe73076a42196320d9c340049127d",
        "model_sha256": ("85c7db53fbe8d62be9bc29a0743661adcb0067552488f185"
                         "b5f2eb2f1ee4179f"),
        "model_size":
        1_495_893_752,
        "tensor_count":
        291,
        "value_count":
        747_930_496,
        "namespace_fingerprint": ("f652315023931100e131d815294c04855a840e1f9c0a05fc"
                                  "7ab4cd6b87fd1be0"),
        "config_sha256": ("aacd57062fcda540db985b0f134d049ad3eeef9a79a849b0"
                          "0961e74702004068"),
        "tokenizer_sha256": ("74c466530bd698626a5b6a424d204711c58dfff0a6b3dd8b"
                             "4dbac1e1e8c9aa87"),
        "license":
        "Apache-2.0",
        "input_format":
        "phonemes",
        "language":
        "en-us",
    }),
    "neuphonic/neutts-nano":
    MappingProxyType({
        "revision":
        "94c32e783cb1d00097a85fd3e5b12db90f9f3fb0",
        "model_sha256": ("00efc179c49eb1808b785c0722fbd6da78f46a51f0840c50"
                         "b446bcb3e5e0c7ee"),
        "model_size":
        914_843_656,
        "tensor_count":
        218,
        "value_count":
        228_704_832,
        "namespace_fingerprint": ("0e2cef4d0078a3e40f2a9620c9f5a99df04d14286718358d"
                                  "96fb053bd1977510"),
        "license":
        "NeuTTS-Open-License-1.0",
        "input_format":
        "phonemes",
        "language":
        "en-us",
    }),
    "neuphonic/neutts-nano-german":
    MappingProxyType({
        "revision": "6184a0baa58ce22b1db0c67ecac186acf7c667c9",
        "model_sha256": ("444a125e912af8c895c72bc1c32f36f7590cfa9f0115a6fd"
                         "27157bacc3215c85"),
        "model_size": 914_843_656,
        "tensor_count": 218,
        "license": "NeuTTS-Open-License-1.0",
        "input_format": "phonemes",
        "language": "de",
    }),
    "neuphonic/neutts-nano-french":
    MappingProxyType({
        "revision": "bd430226747f37a4b6a6e3e8c067b7fbf070fc16",
        "model_sha256": ("225828d7283b41f4395a40aa5cc5f62312d6852343a72454"
                         "fbd4886557091de8"),
        "model_size": 914_843_656,
        "tensor_count": 218,
        "license": "NeuTTS-Open-License-1.0",
        "input_format": "phonemes",
        "language": "fr-fr",
    }),
    "neuphonic/neutts-nano-spanish":
    MappingProxyType({
        "revision": "acce2ee6b1223dd1d3e399c8db89d85facd3f330",
        "model_sha256": ("b422b7032453b22e526d2ffea1621a88cd2752729cd91dcd"
                         "09888186099d3376"),
        "model_size": 914_843_656,
        "tensor_count": 218,
        "license": "NeuTTS-Open-License-1.0",
        "input_format": "phonemes",
        "language": "es",
    }),
    "neuphonic/neutts-2e":
    MappingProxyType({
        "revision":
        "412aaab11c6b727c6c0fe2552db305c234e568da",
        "model_sha256": ("c86c35dfbc0a722566201bb2393868a738bb137ba9b431e0"
                         "77e9b5a73b999c88"),
        "model_size":
        472_114_168,
        "tensor_count":
        310,
        "value_count":
        236_039_680,
        "namespace_fingerprint": ("1aaa14b6df5b80ded6ddd8c60a56778be2d5b8b5de1b65c1"
                                  "7bdbdd7c0a3cad0f"),
        "license":
        "NeuTTS-Open-License-1.0",
        "input_format":
        "BPE",
        "language":
        "en-us",
    }),
})

NEUCODEC_REFERENCE = MappingProxyType({
    "model_id":
    "neuphonic/neucodec",
    "revision":
    "30c1fdd19e68aee65d542cf043750d4c0165893e",
    "filename":
    "model.safetensors",
    "sha256": ("c4ccda95a16eb6f3ab02aa574946b01870f16a4af0da6bdb"
               "5afaa962e4d9f0d9"),
    "size":
    2_519_855_456,
    "tensor_count":
    811,
    "value_count":
    629_937_706,
    "namespace_fingerprint": ("9ec8ada5d387b47180200d70bbf0e19d4bfd7ef6d13431b0"
                              "ff34e3a5d11405a7"),
    "license":
    "Apache-2.0",
})

NEUTTS_TRAINING_SOURCE = MappingProxyType({
    "repository": NEUTTS_SOURCE_REPOSITORY,
    "revision": NEUTTS_SOURCE_REVISION,
    "recipe": "examples/finetune.py",
    "model_family": "neutts-air",
})

__all__ = [
    "NEUCODEC_REFERENCE",
    "NEUCODEC_SOURCE_REPOSITORY",
    "NEUCODEC_SOURCE_REVISION",
    "NEUCODEC_TRANSFORMERS_REFERENCE_REVISION",
    "NEUTTS_SOURCE_REPOSITORY",
    "NEUTTS_SOURCE_REVISION",
    "NEUTTS_TRAINING_SOURCE",
    "NEUTTS_VARIANTS",
]
