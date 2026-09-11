"""Pinned provenance and immutable artifact facts for VibeVoice."""

from __future__ import annotations

from types import MappingProxyType

MICROSOFT_VIBEVOICE_SOURCE = "https://github.com/microsoft/VibeVoice"
MICROSOFT_VIBEVOICE_SOURCE_REVISION = "94da20d98b2fa7688e9cbfaf7692ddb4954f7600"
MICROSOFT_VIBEVOICE_LICENSE = "MIT"

TRANSFORMERS_SOURCE = "https://github.com/huggingface/transformers"
TRANSFORMERS_VIBEVOICE_ASR_INITIAL_REVISION = "f2f0d17d160bc239de46771d6ad84a4e5c2bc2e2"
TRANSFORMERS_VIBEVOICE_ASR_AUDITED_REVISION = "af71155683b4d34dd92d8f037392fa6bf334035e"
TRANSFORMERS_LICENSE = "Apache-2.0"

VIBEVOICE_ASR_REPOSITORY = "microsoft/VibeVoice-ASR-HF"
VIBEVOICE_ASR_REVISION = "f22241c2062b3b25272bf117397e03d73381037a"
VIBEVOICE_TTS_REPOSITORY = "microsoft/VibeVoice-1.5B"
VIBEVOICE_TTS_REVISION = "c00898d257e6b46004e3e2866a47534085fb685a"
VIBEVOICE_REALTIME_REPOSITORY = "microsoft/VibeVoice-Realtime-0.5B"
VIBEVOICE_REALTIME_REVISION = "6bce5f06044837fe6d2c5d7a71a84f0416bd57e4"

QWEN_1_5B_TOKENIZER_REPOSITORY = "Qwen/Qwen2.5-1.5B"
QWEN_1_5B_TOKENIZER_REVISION = "8faed761d45a263340a0528343f099c05c9a4323"
QWEN_0_5B_TOKENIZER_REPOSITORY = "Qwen/Qwen2.5-0.5B"
QWEN_0_5B_TOKENIZER_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
QWEN_LICENSE = "Apache-2.0"

_ASR_SHARDS = {
    "model-00001-of-00008.safetensors": {
        "size": 2_488_346_304,
        "header_size": 4_792,
        "sha256": "580189757a1c737ecc6fad16e633b922bc73220268356e31362c24365f081095",
    },
    "model-00002-of-00008.safetensors": {
        "size": 2_389_316_008,
        "header_size": 8_608,
        "sha256": "65e419d28fc8c87d6938ec611c7b425b38e391c80a2df59c2b4935f6fd65c7c2",
    },
    "model-00003-of-00008.safetensors": {
        "size": 2_466_376_400,
        "header_size": 7_880,
        "sha256": "30f574c764550d26c654cc3f5dd000b90b0c3305db955267c577001c1013fdf7",
    },
    "model-00004-of-00008.safetensors": {
        "size": 2_466_376_432,
        "header_size": 7_912,
        "sha256": "8fa3d9350f0a8cc97524617ff582e451159bfdba2ef0b2e372a627518ef681b1",
    },
    "model-00005-of-00008.safetensors": {
        "size": 2_499_431_160,
        "header_size": 8_944,
        "sha256": "5015ecacc86d897d20d5e42ba52ef83355d42f587d3b158ff897ec94847bd083",
    },
    "model-00006-of-00008.safetensors": {
        "size": 1_831_435_704,
        "header_size": 5_552,
        "sha256": "7463d75607185c118925cea6a60a5698aaa6f428e626bcee774572bdc28ef55d",
    },
    "model-00007-of-00008.safetensors": {
        "size": 2_482_226_384,
        "header_size": 74_440,
        "sha256": "b52aa2fab9640bcac5201a278f540b018c439ed4e99516d9ff30ea1ae399525e",
    },
    "model-00008-of-00008.safetensors": {
        "size": 37_262_376,
        "header_size": 800,
        "sha256": "a66ced85b619507e1970dc51be45786ff55803ccc63658765e412ea23d9a8ace",
    },
}

_TTS_SHARDS = {
    "model-00001-of-00003.safetensors": {
        "size": 1_975_317_828,
        "header_size": 25_912,
        "sha256": "c5f0a61ddeaeb028e3af540ba4dee7933ad30f9f30b6e1320dd9c875a2daa033",
    },
    "model-00002-of-00003.safetensors": {
        "size": 1_983_051_688,
        "header_size": 59_040,
        "sha256": "81c3891f7b2493eb48a9eb6f5be0df48d4f1a4bfd952d84e21683ca6d0bf7969",
    },
    "model-00003-of-00003.safetensors": {
        "size": 1_449_832_938,
        "header_size": 73_504,
        "sha256": "cb6e7e5e86b4a41fffbe1f3aaf445d0d50b5e21ed47574101b777f77d75fa196",
    },
}

VIBEVOICE_CHECKPOINTS = MappingProxyType({
    VIBEVOICE_ASR_REPOSITORY:
    MappingProxyType({
        "revision":
        VIBEVOICE_ASR_REVISION,
        "model_type":
        "vibevoice_asr",
        "index_filename":
        "model.safetensors.index.json",
        "index_size":
        91_965,
        "index_sha256":
        "c807b82f9bb711f0fd3cc9cc138abccb3bcb1e27a002605cc4fbb3c113d322f3",
        "shards":
        MappingProxyType({
            name: MappingProxyType(value)
            for name, value in _ASR_SHARDS.items()
        }),
        "tensors":
        901,
        "parameters":
        8_330_325_888,
        "tensor_bytes":
        16_660_651_776,
        "dtype":
        "BF16",
        "header_fingerprint":
        "013b5db1ca72e154c055182ace9a75d3a348ca4e922fadca608a7eb4e1ae4015",
        "license":
        MICROSOFT_VIBEVOICE_LICENSE,
    }),
    VIBEVOICE_TTS_REPOSITORY:
    MappingProxyType({
        "revision":
        VIBEVOICE_TTS_REVISION,
        "model_type":
        "vibevoice",
        "index_filename":
        "model.safetensors.index.json",
        "index_size":
        122_616,
        "index_sha256":
        "067db9b10fdecee3a5588aa00206794156c7125f5e85f3f2234e0e6d821ee629",
        "shards":
        MappingProxyType({
            name: MappingProxyType(value)
            for name, value in _TTS_SHARDS.items()
        }),
        "tensors":
        1_204,
        "parameters":
        2_704_021_987,
        "tensor_bytes":
        5_408_043_974,
        "dtype":
        "BF16",
        "header_fingerprint":
        "925fb3708482f1cfe7f4614a53731876ef8923435f36eda579c1d5e082f8001c",
        "license":
        MICROSOFT_VIBEVOICE_LICENSE,
    }),
    VIBEVOICE_REALTIME_REPOSITORY:
    MappingProxyType({
        "revision": VIBEVOICE_REALTIME_REVISION,
        "model_type": "vibevoice_streaming",
        "filename": "model.safetensors",
        "size": 2_035_332_888,
        "header_size": 79_432,
        "sha256": "7758b150b8139deb48ac1ff6f181f745c8fedd5511232fd974b3eb217d83b514",
        "tensors": 605,
        "parameters": 1_017_626_724,
        "tensor_bytes": 2_035_253_448,
        "dtype": "BF16",
        "header_fingerprint": "7fb07bd53c540e4b8de42c13d4a09615725e9b700b50a8f34b70906d7864395b",
        "license": MICROSOFT_VIBEVOICE_LICENSE,
    }),
})

VIBEVOICE_STATIC_ASSETS = MappingProxyType({
    VIBEVOICE_ASR_REPOSITORY:
    MappingProxyType({
        "config.json": (
            2_920,
            "89e5f9e4932e72cd8ace355e80b8c5c0e4ac6f6a92e97bf47e8b8ffdca1035e0",
        ),
        "processor_config.json": (
            537,
            "918e2554dde40a1558a2c9c5f1ae4e077b594c00c14276f776ad67c73a2eb6a6",
        ),
        "tokenizer.json": (
            11_421_892,
            "3fd169731d2cbde95e10bf356d66d5997fd885dd8dbb6fb4684da3f23b2585d8",
        ),
        "tokenizer_config.json": (
            714,
            "64029a57ca4f977c2f50fe95dda323a923d4b5f9e31836257610c0362c8e683c",
        ),
        "generation_config.json": (
            281,
            "0d18916d2ae79a3cebc7b05d99fd66203c6180a64dfeb4586856c104bcd46e32",
        ),
        "chat_template.jinja": (
            1_243,
            "facaa74472ce1cc68fd19be60062e202ec5ce8b7b87c28a28a7d27f6adfef58d",
        ),
    }),
    VIBEVOICE_TTS_REPOSITORY:
    MappingProxyType({
        "config.json": (
            2_762,
            "b0d6db52f45dd5179b42217c55112aa2ee66e7ad7b40b0950833c9f6780a0f21",
        ),
        "preprocessor_config.json": (
            351,
            "56bef7fb56db168e31dff685337c4acb6a35b5ce9c68d255128b01395b442c43",
        ),
    }),
    VIBEVOICE_REALTIME_REPOSITORY:
    MappingProxyType({
        "config.json": (
            2_117,
            "caee2691e790b04054bbe14a753b40149fa7c0c16fadb58d9adf5412343dcf57",
        ),
        "preprocessor_config.json": (
            360,
            "ebf514b5d30a012e5ae00d9a19d01e735e35b27768c3926d980815db8fa742e5",
        ),
    }),
})

QWEN_TOKENIZER_ASSETS = MappingProxyType({
    "tokenizer.json": (
        7_031_645,
        "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    ),
    "tokenizer_config.json": (
        7_228,
        "c91efca15ceff6e9ee9424db58a6f59cd41294e550a86cbd07e3c1fb500b34f9",
    ),
})

VIBEVOICE_SAMPLE_RATE = 24_000
VIBEVOICE_HOP_LENGTH = 3_200

__all__ = [
    "MICROSOFT_VIBEVOICE_LICENSE",
    "MICROSOFT_VIBEVOICE_SOURCE",
    "MICROSOFT_VIBEVOICE_SOURCE_REVISION",
    "QWEN_0_5B_TOKENIZER_REPOSITORY",
    "QWEN_0_5B_TOKENIZER_REVISION",
    "QWEN_1_5B_TOKENIZER_REPOSITORY",
    "QWEN_1_5B_TOKENIZER_REVISION",
    "QWEN_LICENSE",
    "QWEN_TOKENIZER_ASSETS",
    "TRANSFORMERS_LICENSE",
    "TRANSFORMERS_SOURCE",
    "TRANSFORMERS_VIBEVOICE_ASR_AUDITED_REVISION",
    "TRANSFORMERS_VIBEVOICE_ASR_INITIAL_REVISION",
    "VIBEVOICE_ASR_REPOSITORY",
    "VIBEVOICE_ASR_REVISION",
    "VIBEVOICE_CHECKPOINTS",
    "VIBEVOICE_HOP_LENGTH",
    "VIBEVOICE_REALTIME_REPOSITORY",
    "VIBEVOICE_REALTIME_REVISION",
    "VIBEVOICE_SAMPLE_RATE",
    "VIBEVOICE_STATIC_ASSETS",
    "VIBEVOICE_TTS_REPOSITORY",
    "VIBEVOICE_TTS_REVISION",
]
