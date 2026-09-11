"""Immutable provenance and audited checkpoint facts for MOSS-TTS."""

from __future__ import annotations

from types import MappingProxyType

OPENMOSS_TTS_SOURCE = "https://github.com/OpenMOSS/MOSS-TTS"
OPENMOSS_TTS_SOURCE_REVISION = "58b20a0d5fcc6766658d50967a90a9d890009a46"
OPENMOSS_CODEC_SOURCE = "https://github.com/OpenMOSS/MOSS-Audio-Tokenizer"
OPENMOSS_CODEC_SOURCE_REVISION = "8c50ac4c5d7287d2ed6ea20a08c90ca439887d23"
OPENMOSS_LICENSE = "Apache-2.0"

MOSS_TTS_DELAY_REPOSITORY = "OpenMOSS-Team/MOSS-TTS"
MOSS_TTS_DELAY_REVISION = "b6b0229853ff63c68fa6aeceb380d8c016f55daf"
MOSS_TTS_DELAY_V15_REPOSITORY = "OpenMOSS-Team/MOSS-TTS-v1.5"
MOSS_TTS_DELAY_V15_REVISION = "cdd3b911b1585e3f2dbc7775ef10f9926f58850a"
MOSS_TTS_LOCAL_REPOSITORY = "OpenMOSS-Team/MOSS-TTS-Local-Transformer"
MOSS_TTS_LOCAL_REVISION = "12aa734e4f11a7b3fdf4eb0ad2aa2029675ffc2e"
MOSS_TTS_LOCAL_V15_REPOSITORY = "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5"
MOSS_TTS_LOCAL_V15_REVISION = "be7766a6735b98bd793f7c79fb720b4d0f5d13b8"
MOSS_TTS_REALTIME_REPOSITORY = "OpenMOSS-Team/MOSS-TTS-Realtime"
MOSS_TTS_REALTIME_REVISION = "75682787d8e2fcc73faca37ba2931453ca9c4022"
MOSS_CODEC_V1_REPOSITORY = "OpenMOSS-Team/MOSS-Audio-Tokenizer"
MOSS_CODEC_V1_REVISION = "3cd226ba2947efa357ef453bcad111b6eafba782"
MOSS_CODEC_V2_REPOSITORY = "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2"
MOSS_CODEC_V2_REVISION = "f6e20e543b33d2c252a7ef71bdf8aa71e5ff9169"


def _checkpoint(
    *,
    revision: str,
    variant: str,
    tensors: int,
    parameters: int,
    tensor_bytes: int,
    fingerprint: str,
    sample_rate: int,
    codec: str,
) -> MappingProxyType:
    return MappingProxyType({
        "revision": revision,
        "variant": variant,
        "tensors": tensors,
        "parameters": parameters,
        "tensor_bytes": tensor_bytes,
        "dtype": "BF16",
        "header_fingerprint": fingerprint,
        "sample_rate": sample_rate,
        "codec": codec,
        "license": OPENMOSS_LICENSE,
    })


MOSS_TTS_CHECKPOINTS = MappingProxyType({
    MOSS_TTS_DELAY_REPOSITORY:
    _checkpoint(
        revision=MOSS_TTS_DELAY_REVISION,
        variant="delay",
        tensors=463,
        parameters=8_489_841_664,
        tensor_bytes=16_979_683_328,
        fingerprint="3491bdffba10bba013848d67673e000cd333c6e46a891a96b056d980c43697c7",
        sample_rate=24_000,
        codec=MOSS_CODEC_V1_REPOSITORY,
    ),
    MOSS_TTS_DELAY_V15_REPOSITORY:
    _checkpoint(
        revision=MOSS_TTS_DELAY_V15_REVISION,
        variant="delay",
        tensors=463,
        parameters=8_489_841_664,
        tensor_bytes=16_979_683_328,
        fingerprint="3491bdffba10bba013848d67673e000cd333c6e46a891a96b056d980c43697c7",
        sample_rate=24_000,
        codec=MOSS_CODEC_V1_REPOSITORY,
    ),
    MOSS_TTS_LOCAL_REPOSITORY:
    _checkpoint(
        revision=MOSS_TTS_LOCAL_REVISION,
        variant="local",
        tensors=556,
        parameters=3_060_606_464,
        tensor_bytes=6_121_212_928,
        fingerprint="c6eb55387bde7a3221828e79fc3d1b6d6a101eca6c1b59342b85bde402d068c2",
        sample_rate=24_000,
        codec=MOSS_CODEC_V1_REPOSITORY,
    ),
    MOSS_TTS_LOCAL_V15_REPOSITORY:
    _checkpoint(
        revision=MOSS_TTS_LOCAL_V15_REVISION,
        variant="local_v1_5",
        tensors=438,
        parameters=4_550_403_584,
        tensor_bytes=9_100_807_168,
        fingerprint="62da29b32d667d031f6c59067207cdda88064f2f326235df560fa91e6f76a440",
        sample_rate=48_000,
        codec=MOSS_CODEC_V2_REPOSITORY,
    ),
    MOSS_TTS_REALTIME_REPOSITORY:
    _checkpoint(
        revision=MOSS_TTS_REALTIME_REVISION,
        variant="realtime",
        tensors=403,
        parameters=2_331_940_864,
        tensor_bytes=4_663_881_728,
        fingerprint="20409dc259f40a1c86e3a9b9dc087d66e787da3124f011a1217ed04f0361c495",
        sample_rate=24_000,
        codec=MOSS_CODEC_V1_REPOSITORY,
    ),
})

MOSS_CODEC_CHECKPOINTS = MappingProxyType({
    MOSS_CODEC_V1_REPOSITORY:
    MappingProxyType({
        "revision": MOSS_CODEC_V1_REVISION,
        "version": 1,
        "tensors": 1_600,
        "parameters": 1_774_566_400,
        "tensor_bytes": 7_098_265_600,
        "dtype": "F32",
        "header_fingerprint": "7133d25edd0d529bb3d0c78eedd7c5c2278e9a6b663881e1bfce90df7a8da6db",
        "sample_rate": 24_000,
        "channels": 1,
        "downsample_rate": 1_920,
        "license": OPENMOSS_LICENSE,
    }),
    MOSS_CODEC_V2_REPOSITORY:
    MappingProxyType({
        "revision": MOSS_CODEC_V2_REVISION,
        "version": 2,
        "tensors": 2_094,
        "parameters": 2_123_701_248,
        "tensor_bytes": 8_494_804_992,
        "dtype": "F32",
        "header_fingerprint": "c9548ffba0bcc696522eb2fe1a433010bebac20f832d7ca81926ba2a371c121d",
        "sample_rate": 48_000,
        "channels": 2,
        "downsample_rate": 3_840,
        "license": OPENMOSS_LICENSE,
    }),
})

MOSS_TTS_REVISIONS = MappingProxyType({
    repository: str(facts["revision"])
    for repository, facts in MOSS_TTS_CHECKPOINTS.items()
})
MOSS_CODEC_REVISIONS = MappingProxyType({
    repository: str(facts["revision"])
    for repository, facts in MOSS_CODEC_CHECKPOINTS.items()
})

__all__ = [
    "MOSS_CODEC_CHECKPOINTS",
    "MOSS_CODEC_REVISIONS",
    "MOSS_CODEC_V1_REPOSITORY",
    "MOSS_CODEC_V1_REVISION",
    "MOSS_CODEC_V2_REPOSITORY",
    "MOSS_CODEC_V2_REVISION",
    "MOSS_TTS_CHECKPOINTS",
    "MOSS_TTS_DELAY_REPOSITORY",
    "MOSS_TTS_DELAY_REVISION",
    "MOSS_TTS_DELAY_V15_REPOSITORY",
    "MOSS_TTS_DELAY_V15_REVISION",
    "MOSS_TTS_LOCAL_REPOSITORY",
    "MOSS_TTS_LOCAL_REVISION",
    "MOSS_TTS_LOCAL_V15_REPOSITORY",
    "MOSS_TTS_LOCAL_V15_REVISION",
    "MOSS_TTS_REALTIME_REPOSITORY",
    "MOSS_TTS_REALTIME_REVISION",
    "MOSS_TTS_REVISIONS",
    "OPENMOSS_CODEC_SOURCE",
    "OPENMOSS_CODEC_SOURCE_REVISION",
    "OPENMOSS_LICENSE",
    "OPENMOSS_TTS_SOURCE",
    "OPENMOSS_TTS_SOURCE_REVISION",
]
