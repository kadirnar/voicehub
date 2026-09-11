"""Pinned source and release metadata for VoiceHub's MeloTTS graph."""

from __future__ import annotations

MELOTTS_SOURCE_REPOSITORY = "https://github.com/myshell-ai/MeloTTS"
MELOTTS_SOURCE_REVISION = "209145371cff8fc3bd60d7be902ea69cbdb7965a"
MELOTTS_SOURCE_LICENSE = "MIT"
MELOTTS_NATIVE_FORMAT = "voicehub-melotts-v1"

# Language aliases retain the repository names used by the upstream API, but
# pin immutable snapshots instead of following mutable ``main`` branches.
MELOTTS_RELEASES = {
    "EN": (
        "myshell-ai/MeloTTS-English",
        "bb4fb7346d566d277ba8c8c7dbfdf6786139b8ef",
        "039116c927c70eaa4458d315ea83aaaa99e1fca1c621b50c8ca56b4a5700eb77",
        "acd278040eaf9536908e2b965273df5a731c44d8f0da66cc5fed7972772ed23c",
    ),
    "EN_V2": (
        "myshell-ai/MeloTTS-English-v2",
        "a53e3509c4ee4ff16d79272feb2474ff864e18f3",
        "fbe2f4196068b472982651148b912387e03efd42d07c771ce126a60408c3118a",
        "794226eb7c1745f3ca281b290613d5f39aa5b0d3b16a117009966f4aaf184757",
    ),
    "EN_NEWEST": (
        "myshell-ai/MeloTTS-English-v3",
        "f7c4a35392c0e9be24a755f1edb4c3f63040f759",
        "66db2f9d3fba6e94e7f430668188130129c0428a5a1646bcd8bd453a966e778a",
        "959433dc1c6df618922560b4b7bc8c7af0a4b7ceaea267480c5d9ae8a3cfe536",
    ),
    "FR": (
        "myshell-ai/MeloTTS-French",
        "1e9bf590262392d8bffb679b0a3b0c16b0f9fdaf",
        "361e84109451acb2f0331c2c9f3c5437e9a502380bbf8741b02545e41b062139",
        "fdf967d514f91582e451c482cab655e5d736821c3ba87ede8bb0625709642b29",
    ),
    "JP": (
        "myshell-ai/MeloTTS-Japanese",
        "367f8795464b531b4e97c1515bddfc1243e60891",
        "207def0d31bf7623e20f4a5e690f217747661bf495319c0139303122b6debcc3",
        "96ae783e6ec0177aa810e2a645aec5d136a6f4992fdea26ee92b7b04d9688ad0",
    ),
    "ES": (
        "myshell-ai/MeloTTS-Spanish",
        "dbb5496df39d11a66c1d5f5a9ca357c3c9fb95fb",
        "54488d922a2983f4d6a7f57158cc5f2714ea117994e1b22147a8579791554221",
        "9077a7e7e5fd8e42f3f922641c401f1936971c08465a3e7ccb19d57a659e72ae",
    ),
    "ZH": (
        "myshell-ai/MeloTTS-Chinese",
        "af5d207a364ea4208c6f589c89f57f88414bdd16",
        "d58b5acdab89ad2bbd65325affab309ae3cb964834b02f9a60587474e81c8bb9",
        "a74e9eadffff065c75eb6dfa040efa72cad23e72cfea70d39190bc174fb97093",
    ),
    "KR": (
        "myshell-ai/MeloTTS-Korean",
        "0207e5adfc90129a51b6b03d89be6d84360ed323",
        "74543376976dfadde45ba34336fa79c7e95509f43a7c2e701b22c0f71fd7695c",
        "48e3ff3fd0b5348e095f0468e60ae727507564100f58142ef3a922ead6e0a4d0",
    ),
}

MELOTTS_EN_NEWEST_CHECKPOINT_SHA256 = ("959433dc1c6df618922560b4b7bc8c7af0a4b7ceaea267480c5d9ae8a3cfe536")
MELOTTS_EN_NEWEST_TENSOR_COUNT = 1_051
MELOTTS_EN_NEWEST_PARAMETER_COUNT = 51_808_433
# SHA-256 over sorted ``name|torch.dtype|shape`` rows from the pinned
# English-v3 ``checkpoint.pth`` payload.
MELOTTS_EN_NEWEST_INVENTORY_FINGERPRINT = ("c505248490ac8de6668aa818388cfa5ca4bcf2ce75a7aacfa9a35dfe6b15816d")
MELOTTS_GENERATOR_COMPONENTS = (
    "dec",
    "dp",
    "emb_g",
    "enc_p",
    "enc_q",
    "flow",
    "sdp",
)

__all__ = [
    "MELOTTS_EN_NEWEST_CHECKPOINT_SHA256",
    "MELOTTS_EN_NEWEST_INVENTORY_FINGERPRINT",
    "MELOTTS_EN_NEWEST_PARAMETER_COUNT",
    "MELOTTS_EN_NEWEST_TENSOR_COUNT",
    "MELOTTS_GENERATOR_COMPONENTS",
    "MELOTTS_NATIVE_FORMAT",
    "MELOTTS_RELEASES",
    "MELOTTS_SOURCE_LICENSE",
    "MELOTTS_SOURCE_REPOSITORY",
    "MELOTTS_SOURCE_REVISION",
]
