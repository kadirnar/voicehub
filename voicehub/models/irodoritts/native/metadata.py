"""Immutable provenance and checkpoint inventories for Irodori-TTS."""

from __future__ import annotations

IRODORI_SOURCE_REVISION = "eaf74d6a19138f743acb5b71a445fd25a57db987"
IRODORI_SOURCE_LICENSE = "MIT"
IRODORI_TOKENIZER_ID = "llm-jp/llm-jp-3-150m"
IRODORI_TOKENIZER_REVISION = "b112feef602fff752e4dac4c30af6a2c2fa41c7a"
IRODORI_TOKENIZER_LICENSE = "Apache-2.0"
IRODORI_CODEC_ID = "Aratako/Semantic-DACVAE-Japanese-32dim"
IRODORI_CODEC_REVISION = "47376ee24834d7a05a48ebabfe3cde29b3c5e214"
IRODORI_CODEC_SOURCE_REVISION = "414c20785fc3a28373073ea8ef7a1316eeeaca6e"
IRODORI_CODEC_LICENSE = "MIT"
IRODORI_CODEC_SOURCE_LICENSE = "Apache-2.0"

IRODORI_TOKENIZER_ASSETS = {
    "tokenizer.json": {
        "bytes": 6_416_433,
        "git_blob_oid": "0a87b0e61eb74fc708e32aede9a0c19aa4484a40",
    },
    "tokenizer_config.json": {
        "bytes": 494,
        "git_blob_oid": "09aa8575abe71b38280c766ae331a8b70d68622f",
    },
    "special_tokens_map.json": {
        "bytes": 237,
        "git_blob_oid": "8644c8f41e420e9707cea943b96e19b36748fcff",
    },
}

IRODORI_CODEC_CHECKPOINT = {
    "filename": "weights.pth",
    "file_bytes": 429_620_065,
    "git_blob_oid": "3a92e126da9ed6a54af8b1da5a7daac23646d118",
    "lfs_sha256": "db120339c5ee7eca1912cdf29bc612b947a0808e69c3cebfb4936b45a762c1d5",
    "sample_rate": 48_000,
    "hop_length": 1_920,
    "latent_dim": 32,
    "constructor": {
        "encoder_dim": 64,
        "encoder_rates": [2, 8, 10, 12],
        "latent_dim": 1024,
        "decoder_dim": 1536,
        "decoder_rates": [12, 10, 8, 2],
        "n_codebooks": 16,
        "codebook_size": 1024,
        "codebook_dim": 32,
        "quantizer_dropout": False,
        "sample_rate": 48_000,
    },
}

IRODORI_CHECKPOINTS = {
    "v3": {
        "model_id": "Aratako/Irodori-TTS-500M-v3",
        "revision": "236c1e56591279fc24e3c1bf6609fc06e48dde28",
        "filename": "model.safetensors",
        "file_bytes": 2_048_269_748,
        "git_blob_oid": "08a1af7f1d510a66852ac8eaea12855f5d110902",
        "lfs_sha256": "c4b8e7e982697664f829b7fb6bea307a25bd7ee013ad0d6114efc3e326acbd54",
        "header_bytes": 71_976,
        "header_fingerprint": "d5b6315112155d53e249112f4e38e9fe58a5ff6bd3881c83607b918073c493cc",
        "tensors": 637,
        "parameters": 512_049_441,
        "tensor_data_bytes": 2_048_197_764,
        "dtype_counts": {
            "F32": 637
        },
    },
    "v3-voice-design": {
        "model_id": "Aratako/Irodori-TTS-600M-v3-VoiceDesign",
        "revision": "e863a3a93e652e09afeff3e84823a206a0a60314",
        "filename": "model.safetensors",
        "file_bytes": 2_468_332_708,
        "git_blob_oid": "bc008cb3e80a027a2d8601dcd7d2a749d333af56",
        "lfs_sha256": "93c1f8356857ab4297073f452d01c29015e0db5c83c62109800f8566900f4497",
        "header_bytes": 90_136,
        "header_fingerprint": "4473942682ef97c9621456c681f8df6e3d98b1d24804f4c26dd68fbe446ef207",
        "tensors": 790,
        "parameters": 617_060_641,
        "tensor_data_bytes": 2_468_242_564,
        "dtype_counts": {
            "F32": 790
        },
    },
    "v2": {
        "model_id": "Aratako/Irodori-TTS-500M-v2",
        "revision": "8fd631cafb911dde466bc30dd558a0dc55e8ccae",
        "filename": "model.safetensors",
        "file_bytes": 1_980_044_416,
        "git_blob_oid": "e0cf30e5b17074177536cc32aefa832431750833",
        "lfs_sha256": "e5add885303babe328eae2c426475e41bc2f2aca2bcb010a2ceb6b6ddd1b8d9c",
        "header_bytes": 68_600,
        "header_fingerprint": "c5331ecf8b35c370a4b7556cf785d260fe84b1218d7cdc9adb795f99fa7528a8",
        "tensors": 613,
        "parameters": 494_993_952,
        "tensor_data_bytes": 1_979_975_808,
        "dtype_counts": {
            "F32": 613
        },
    },
    "v2-voice-design": {
        "model_id": "Aratako/Irodori-TTS-500M-v2-VoiceDesign",
        "revision": "456e55708e7183f5c7faa1448209d54aa8991451",
        "filename": "model.safetensors",
        "file_bytes": 2_045_071_384,
        "git_blob_oid": "0041dd5e0b4a75ea5ff660c63caf0702bd57d852",
        "lfs_sha256": "8b703c28e88f160dee0258b1136f8fe1ea68c063b45fc28375b5a134d6ce1131",
        "header_bytes": 71_568,
        "header_fingerprint": "6fe453fdd64ea1f1b3b7191124a7b3855503a7849e69a53dea63e0744f4fd376",
        "tensors": 636,
        "parameters": 511_249_952,
        "tensor_data_bytes": 2_044_999_808,
        "dtype_counts": {
            "F32": 636
        },
    },
}

IRODORI_REJECTED_FORMATS = {
    "gguf": "The published Irodori graph is continuous flow matching, not a GGUF runtime.",
    "mlx": "MLX conversions are provider-specific and not accepted as native checkpoints.",
    "quantized": "Quantized tensors have no audited training or numerical contract.",
    "v1": "Irodori v1 has an incompatible graph and inference protocol.",
}

__all__ = [
    "IRODORI_CHECKPOINTS",
    "IRODORI_CODEC_CHECKPOINT",
    "IRODORI_CODEC_ID",
    "IRODORI_CODEC_REVISION",
    "IRODORI_REJECTED_FORMATS",
    "IRODORI_SOURCE_REVISION",
    "IRODORI_TOKENIZER_ASSETS",
    "IRODORI_TOKENIZER_ID",
    "IRODORI_TOKENIZER_REVISION",
]
