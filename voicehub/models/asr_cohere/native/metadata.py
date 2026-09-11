"""Pinned source and checkpoint metadata for native Cohere Transcribe."""

COHERE_TRANSFORMERS_REVISION = ("af71155683b4d34dd92d8f037392fa6bf334035e")
COHERE_CHECKPOINT_REVISION = ("b1eacc2686a3d08ceaae5f24a88b1d519620bc09")
COHERE_TOKENIZERS_REVISION = ("f383101a26663708484cac0727792aad74f78234")
SPM_PRECOMPILED_REVISION = ("3795b14343d25782df654b49f5c3e9f2a7db7d6f")

COHERE_ASR_CHECKPOINTS = {
    "CohereLabs/cohere-transcribe-03-2026": {
        "revision": COHERE_CHECKPOINT_REVISION,
        "license": "Apache-2.0",
        "checkpoint_bytes": 4_131_862_976,
        "header_bytes": 254_488,
        "tensor_data_bytes": 4_131_608_480,
        "tensors": 2_152,
        "state_values": 2_065_804_096,
        "parameters": 2_047_822_080,
        "header_fingerprint": ("06a76e1e91f509c865013ce962a695a05b6a50ae0290d1258910c660ccb06292"),
        "files": {
            "README.md": "a80f4d9e509aac67dc14aef6d9386501d00737980bbedfe83060a277796449f3",
            "config.json": "5de7e586cec6d8f51225c8d5fe17a56a3043dda9af8c42f9cb01dd545905eb18",
            "configuration_cohere_asr.py": "8cfd12b210d9e13d4e46acf0600d7c5561a66e7c3ca64d06a3fa62c98e98f769",
            "generation_config.json": "a4837377b5696b1d04033536f25a6d0a11d5e613c6d7d75bf20ffc463ae642c6",
            "model.safetensors": "987bd3e141c7bfdb5a78f5db11397ee7737308357e6cc0a3f36a4979b158137a",
            "modeling_cohere_asr.py": "ca5a0b67a1cba76e86e54a3bfe350255a17868f27a4b2faa400c2daa520193e0",
            "preprocessor_config.json": "9f297d330646ecc8ebb9dc5784f48b7c35b118c913e306a1ccd0192f2c976332",
            "processing_cohere_asr.py": "151f965b721c068897cbf199296aa3d4d652dffaa2387d6edef7f10a9b16fb74",
            "processor_config.json": "d5eff85971bab7f42480856f8fb397d8dd966858d67049def99aa8a665aa87d5",
            "special_tokens_map.json": "1814ce01458ff6a72b04a6618e75f18ce627be4dc17619cd3a7cd7f71e137f0f",
            "tokenization_cohere_asr.py": "6b3df3814b6604d0ba9f35e1d058b128937c7d404f6b9dc275d574e013f010b2",
            "tokenizer.json": "780ccca2de2ccd289971b1fb7d4f0b5ec2dc908872f6e350181c7eba9db3fa9f",
            "tokenizer.model": "6d21e6a83b2d0d3e1241a7817e4bef8eb63bcb7cfe4a2675af9a35ff3bbf0e14",
            "tokenizer_config.json": "b462e16e04c9dbae82289b6ef9b080d8dbad331586ece3ebded0bd09996db636",
        },
    },
}

__all__ = [
    "COHERE_ASR_CHECKPOINTS",
    "COHERE_CHECKPOINT_REVISION",
    "COHERE_TOKENIZERS_REVISION",
    "COHERE_TRANSFORMERS_REVISION",
    "SPM_PRECOMPILED_REVISION",
]
