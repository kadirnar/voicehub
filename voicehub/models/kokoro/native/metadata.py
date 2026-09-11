"""Immutable Kokoro source and checkpoint provenance.

This module deliberately contains constants only. Architecture discovery
can therefore expose the pinned provenance without importing PyTorch,
checkpoint readers, or the Kokoro model graph.
"""

KOKORO_SOURCE_REVISION = "dfb907a02bba8152ca444717ca5d78747ccb4bec"
KOKORO_CHECKPOINT_REVISION = "cbc78411372edb46f7e42030b241834a55fe0cb6"
KOKORO_PYTORCH_SHA256 = ("496dba118d1a58f5f3db2efc88dbdc216e0483fc89fe6e47ee1f2c53f18ad1e4")
KOKORO_LEGACY_TENSOR_COUNT = 548
KOKORO_LEGACY_PARAMETER_COUNT = 81_763_410
KOKORO_LEGACY_HEADER_FINGERPRINT = ("2726c65f540ea996938b1e39edb20469be5f773a918388ac138a46318d5a5b5c")
KOKORO_NATIVE_FORMAT = "voicehub-kokoro-v1"

__all__ = [
    "KOKORO_CHECKPOINT_REVISION",
    "KOKORO_LEGACY_HEADER_FINGERPRINT",
    "KOKORO_LEGACY_PARAMETER_COUNT",
    "KOKORO_LEGACY_TENSOR_COUNT",
    "KOKORO_NATIVE_FORMAT",
    "KOKORO_PYTORCH_SHA256",
    "KOKORO_SOURCE_REVISION",
]
