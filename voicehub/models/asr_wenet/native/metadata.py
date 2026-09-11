"""Pinned provenance for the native GigaSpeech U2++ implementation."""

WENET_SOURCE_REPOSITORY = "https://github.com/wenet-e2e/wenet"
WENET_SOURCE_REVISION = "a50d4208f13bbf3a0746e606ac29176cd2e87e6b"
WENET_SOURCE_DATE = "2021-10-25"
WENET_SOURCE_LICENSE = "Apache-2.0"
WENET_CHECKPOINT_LISTING_URL = (
    "https://github.com/wenet-e2e/wenet/blob/"
    f"{WENET_SOURCE_REVISION}/examples/gigaspeech/s0/README.md#conformer-u2-result")

GIGASPEECH_MODEL_NAME = "gigaspeech-u2pp-conformer"
GIGASPEECH_MODEL_VERSION = "20210728"
GIGASPEECH_ARCHIVE_FILENAME = "20210728_u2pp_conformer_exp.tar.gz"
GIGASPEECH_UPSTREAM_URL = (
    "http://mobvoi-speech-public.ufile.ucloud.cn/public/wenet/gigaspeech/"
    "20210728_u2pp_conformer_exp.tar.gz")
GIGASPEECH_MIRROR_REPOSITORY = "openspeech/wenet-models"
GIGASPEECH_MIRROR_REVISION = "90acd57d17169a15d5ceab462c6e7db3bd003921"
GIGASPEECH_MIRROR_FILENAME = "gigaspeech_u2pp_conformer_exp.tar.gz"
GIGASPEECH_MODEL_URL = (
    f"https://huggingface.co/{GIGASPEECH_MIRROR_REPOSITORY}/resolve/"
    f"{GIGASPEECH_MIRROR_REVISION}/{GIGASPEECH_MIRROR_FILENAME}")
GIGASPEECH_ARCHIVE_SIZE = 503_845_602
GIGASPEECH_ARCHIVE_SHA256 = ("061ccfa51d64ebe7ea091a5a13ae31e37d9c36f4eface5c7bafc80bd4a06b26e")
GIGASPEECH_WEIGHTS_SHA256 = ("d8a5a94f08fd30ba1c10fb031da91c86c014710ad699379f76316c63b057b424")
GIGASPEECH_CMVN_SHA256 = ("db7e66555c03cf84df1a50bc4f0bd7a3fc912c8b7c22fe8379789b29ffa9b4c6")
GIGASPEECH_CONFIG_SHA256 = ("176b14b1db3a14491b3790404e8bb24dbf40ed161f095d43109188b10132a683")
GIGASPEECH_TOKENIZER_SHA256 = ("b6f023cf53956c5d96c3677fe6a898913e2caf6ed30be1f34a2f7c0a2ddd6d34")
GIGASPEECH_UNITS_SHA256 = ("0b5d0ecb55ce532312daeaf2b4f2bde8fd6feaa5734f0b6f4989c9831abc7d0f")
GIGASPEECH_TENSOR_COUNT = 670
GIGASPEECH_STATE_VALUES = 136_225_077
GIGASPEECH_TENSOR_FINGERPRINT = ("c1956c8a895e342aa4f53f824b1729f41fa3a861b6b08a1fe5e3d55b48ff45c3")
GIGASPEECH_TENSOR_FINGERPRINT_FORMAT = ("SHA-256 of sorted name|portable-dtype|dimxdim rows joined by LF")
GIGASPEECH_CHECKPOINT_LICENSE = "NOT DECLARED"
GIGASPEECH_CHECKPOINT_PROVIDER = "external-archive"
GIGASPEECH_DOCUMENTATION_PATH = "path/to/converted-wenet-u2pp"
GIGASPEECH_CHECKPOINT_STATUS = (
    "Original upstream archive unavailable (HTTP 404 and TLS failures "
    "verified 2026-08-04); exact bytes are available from the immutable "
    f"{GIGASPEECH_MIRROR_REPOSITORY} mirror at {GIGASPEECH_MIRROR_REVISION}")
GIGASPEECH_DOCUMENTATION_NOTE = (
    "The registry identifier is not a Hugging Face repository and the "
    "original upstream archive endpoints are unavailable. VoiceHub verifies "
    "an immutable mirror against the published 503,845,602-byte archive's "
    "SHA-256. Convert that trust-gated pickle archive first, then replace the "
    "path below with the resulting VoiceHub-native directory containing "
    "model.safetensors, config.json, tokenizer.model, and units.txt.")

__all__ = [name for name in globals() if name.startswith(("GIGASPEECH_", "WENET_"))]
