"""Immutable GPT-SoVITS source and checkpoint metadata."""

from __future__ import annotations

from dataclasses import dataclass

GPT_SOVITS_REPOSITORY = "lj1995/GPT-SoVITS"
GPT_SOVITS_REVISION = "336b2ec4e8d4ac74740798dd40af44e74659ecaf"
GPT_SOVITS_SOURCE_REVISION = "d523079fc05d9a8028d6085bffe4a2757c32abb6"
GPT_SOVITS_LICENSE = "MIT"


@dataclass(frozen=True, slots=True)
class GPTSoVITSCheckpointMetadata:
    """Immutable identity and state inventory for one published component."""

    filename: str
    subfolder: str
    sha256: str
    inventory_fingerprint: str
    tensor_count: int
    parameter_count: int


@dataclass(frozen=True, slots=True)
class GPTSoVITSVariantMetadata:
    """One coherent public S1/S2 release family."""

    variant: str
    s1: GPTSoVITSCheckpointMetadata
    s2_generator: GPTSoVITSCheckpointMetadata
    s2_discriminator: GPTSoVITSCheckpointMetadata


_S1_V1 = GPTSoVITSCheckpointMetadata(
    filename="s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt",
    subfolder="",
    sha256="b1c1e17e9c99547a89388f72048cd6e1b41b5a18b170e86a46dfde0324d63eb1",
    inventory_fingerprint="3194c642cb45ea9b2bf4f7df6ef1e5ed5434d3b03fe380a7dc72b40c21f3c5df",
    tensor_count=295,
    parameter_count=77_493_762,
)
_S1_V2 = GPTSoVITSCheckpointMetadata(
    filename="s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
    subfolder="gsv-v2final-pretrained",
    sha256="732f94e63b148066e24c7f9d2637f3374083e637635f07fbdb695dee20ddbe1f",
    inventory_fingerprint="f2c51a99cc5008555175fceba4e4b5bb2e76d7ead7364d6d14ac524f0ee4f037",
    tensor_count=295,
    parameter_count=77_606_402,
)
_S2_D_CLASSIC_V1 = GPTSoVITSCheckpointMetadata(
    filename="s2D488k.pth",
    subfolder="",
    sha256="fc579c1db3c1e21b721001cf99d7a584214280df19b002e200b630a34fa06eb8",
    inventory_fingerprint="d093b4d9d5d5d9f9296f2fd004a15c8658e6deabaf011a095d78a96a7147a632",
    tensor_count=111,
    parameter_count=46_747_132,
)
_S2_D_CLASSIC_V2 = GPTSoVITSCheckpointMetadata(
    filename="s2D2333k.pth",
    subfolder="gsv-v2final-pretrained",
    sha256="8ae7fe8dd8c8f2e718de359e00edac88b0c71ab2fd10b07ad4cc45070eb8a836",
    inventory_fingerprint="d093b4d9d5d5d9f9296f2fd004a15c8658e6deabaf011a095d78a96a7147a632",
    tensor_count=111,
    parameter_count=46_747_132,
)
_S2_D_PRO = GPTSoVITSCheckpointMetadata(
    filename="s2Dv2Pro.pth",
    subfolder="v2Pro",
    sha256="257a7f36ae49e9ff08790d2cd3d398592530852caba810197f8606e884a52c54",
    inventory_fingerprint="d55dddbd9b41ec132816dcc395c8c8fbd1447b6a3be135a77bb2e7ce443a3565",
    tensor_count=147,
    parameter_count=63_189_440,
)
_S2_D_PRO_PLUS = GPTSoVITSCheckpointMetadata(
    filename="s2Dv2ProPlus.pth",
    subfolder="v2Pro",
    sha256="635cd84bf6f7f9b8d41c88c7106f81d782c794c61f931845214ea037b0c5bef2",
    inventory_fingerprint="d55dddbd9b41ec132816dcc395c8c8fbd1447b6a3be135a77bb2e7ce443a3565",
    tensor_count=147,
    parameter_count=63_189_440,
)

GPT_SOVITS_VARIANTS = {
    "v1":
    GPTSoVITSVariantMetadata(
        variant="v1",
        s1=_S1_V1,
        s2_generator=GPTSoVITSCheckpointMetadata(
            filename="s2G488k.pth",
            subfolder="",
            sha256="020a014e1e01e550e510f2f61fae5e5f5b6aab40f15c22f1f12f724df507e835",
            inventory_fingerprint="b42605d1b1d10ee313c0612a82f2ddbb720e7c63e816883ce54f6a7fb2d18f68",
            tensor_count=776,
            parameter_count=52_846_337,
        ),
        s2_discriminator=_S2_D_CLASSIC_V1,
    ),
    "v2":
    GPTSoVITSVariantMetadata(
        variant="v2",
        s1=_S1_V2,
        s2_generator=GPTSoVITSCheckpointMetadata(
            filename="s2G2333k.pth",
            subfolder="gsv-v2final-pretrained",
            sha256="924fdccaa3c574bf139c25c9759aa1ed3b3f99e19a7c529ee996c2bc17663695",
            inventory_fingerprint="eb37acdc56b12de10bb84d63e844bc07e26cdf3a004f390a7653a12a864895ec",
            tensor_count=776,
            parameter_count=52_883_969,
        ),
        s2_discriminator=_S2_D_CLASSIC_V2,
    ),
    "v2Pro":
    GPTSoVITSVariantMetadata(
        variant="v2Pro",
        s1=_S1_V2,
        s2_generator=GPTSoVITSCheckpointMetadata(
            filename="s2Gv2Pro.pth",
            subfolder="v2Pro",
            sha256="0f8ead815234365edf045c6d86370ed6e4f440e8195be77ff0ea72684ad406a5",
            inventory_fingerprint="69724c95323889309cd9898c4d39fc98e10793ca6495f018170baa0d1bb57949",
            tensor_count=781,
            parameter_count=81_001_985,
        ),
        s2_discriminator=_S2_D_PRO,
    ),
    "v2ProPlus":
    GPTSoVITSVariantMetadata(
        variant="v2ProPlus",
        s1=_S1_V2,
        s2_generator=GPTSoVITSCheckpointMetadata(
            filename="s2Gv2ProPlus.pth",
            subfolder="v2Pro",
            sha256="d42a22bbbf65fb2bbdd45ad6a66841156977db45c7aabe0a6992ff378d9c7d3b",
            inventory_fingerprint="299534fe99cb07d58faf3f6ab3f8070f4a1fa9cb4994f866c684c52cded56a7a",
            tensor_count=781,
            parameter_count=99_912_321,
        ),
        s2_discriminator=_S2_D_PRO_PLUS,
    ),
}

# Backward-compatible names continue to identify the default V2 release.
S1_FILENAME = _S1_V2.filename
S1_SUBFOLDER = _S1_V2.subfolder
S1_SHA256 = _S1_V2.sha256
S1_INVENTORY = _S1_V2.inventory_fingerprint
S1_TENSORS = _S1_V2.tensor_count
S1_VALUES = _S1_V2.parameter_count

S2_GENERATOR_FILENAME = GPT_SOVITS_VARIANTS["v2"].s2_generator.filename
S2_GENERATOR_SUBFOLDER = GPT_SOVITS_VARIANTS["v2"].s2_generator.subfolder
S2_GENERATOR_SHA256 = GPT_SOVITS_VARIANTS["v2"].s2_generator.sha256
S2_GENERATOR_INVENTORY = GPT_SOVITS_VARIANTS["v2"].s2_generator.inventory_fingerprint
S2_GENERATOR_TENSORS = GPT_SOVITS_VARIANTS["v2"].s2_generator.tensor_count
S2_GENERATOR_VALUES = GPT_SOVITS_VARIANTS["v2"].s2_generator.parameter_count

S2_DISCRIMINATOR_FILENAME = GPT_SOVITS_VARIANTS["v2"].s2_discriminator.filename
S2_DISCRIMINATOR_SUBFOLDER = GPT_SOVITS_VARIANTS["v2"].s2_discriminator.subfolder
S2_DISCRIMINATOR_SHA256 = GPT_SOVITS_VARIANTS["v2"].s2_discriminator.sha256
S2_DISCRIMINATOR_INVENTORY = GPT_SOVITS_VARIANTS["v2"].s2_discriminator.inventory_fingerprint
S2_DISCRIMINATOR_TENSORS = GPT_SOVITS_VARIANTS["v2"].s2_discriminator.tensor_count
S2_DISCRIMINATOR_VALUES = GPT_SOVITS_VARIANTS["v2"].s2_discriminator.parameter_count

LEGACY_NATIVE_FORMAT = "voicehub-native-gpt-sovits-v2"
LEGACY_NATIVE_FORMAT_VERSION = 1
NATIVE_FORMAT = "voicehub-native-gpt-sovits"
NATIVE_FORMAT_VERSION = 2
NATIVE_CONFIG_FILENAME = "gptsovits_config.json"
NATIVE_S1_FILENAME = "s1_model.safetensors"
NATIVE_S2_GENERATOR_FILENAME = "s2_generator.safetensors"
NATIVE_S2_DISCRIMINATOR_FILENAME = "s2_discriminator.safetensors"

__all__ = [
    "GPT_SOVITS_LICENSE",
    "GPT_SOVITS_REPOSITORY",
    "GPT_SOVITS_REVISION",
    "GPT_SOVITS_SOURCE_REVISION",
    "GPT_SOVITS_VARIANTS",
    "GPTSoVITSCheckpointMetadata",
    "GPTSoVITSVariantMetadata",
    "LEGACY_NATIVE_FORMAT",
    "LEGACY_NATIVE_FORMAT_VERSION",
    "NATIVE_CONFIG_FILENAME",
    "NATIVE_FORMAT",
    "NATIVE_FORMAT_VERSION",
    "NATIVE_S1_FILENAME",
    "NATIVE_S2_DISCRIMINATOR_FILENAME",
    "NATIVE_S2_GENERATOR_FILENAME",
    "S1_FILENAME",
    "S1_INVENTORY",
    "S1_SHA256",
    "S1_SUBFOLDER",
    "S1_TENSORS",
    "S1_VALUES",
    "S2_DISCRIMINATOR_FILENAME",
    "S2_DISCRIMINATOR_INVENTORY",
    "S2_DISCRIMINATOR_SHA256",
    "S2_DISCRIMINATOR_SUBFOLDER",
    "S2_DISCRIMINATOR_TENSORS",
    "S2_DISCRIMINATOR_VALUES",
    "S2_GENERATOR_FILENAME",
    "S2_GENERATOR_INVENTORY",
    "S2_GENERATOR_SHA256",
    "S2_GENERATOR_SUBFOLDER",
    "S2_GENERATOR_TENSORS",
    "S2_GENERATOR_VALUES",
]
