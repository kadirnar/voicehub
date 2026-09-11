"""Pinned provenance and checkpoint facts for native SeamlessM4T-v2 S2T."""

from __future__ import annotations

from types import MappingProxyType

SEAMLESS_M4T_V2_REPOSITORY = "facebook/seamless-m4t-v2-large"
SEAMLESS_M4T_V2_REVISION = "5f8cc790b19fc3f67a61c105133b20b34e3dcb76"
SEAMLESS_M4T_V2_LICENSE = "CC-BY-NC-4.0"
SEAMLESS_M4T_V2_LICENSE_URL = ("https://creativecommons.org/licenses/by-nc/4.0/legalcode")

TRANSFORMERS_SOURCE_REPOSITORY = "https://github.com/huggingface/transformers"
TRANSFORMERS_SOURCE_VERSION = "5.14.1"
TRANSFORMERS_SOURCE_REVISION = "a08ace4bbd97e721c98751deec37d87b026acadc"

SEAMLESS_M4T_V2_SHARDS = MappingProxyType({
    "model-00001-of-00002.safetensors":
    MappingProxyType({
        "size": 4_999_163_080,
        "sha256": "85cab984fbc111f8713827c440499453b9e66f262862866eeed8725c302ba2ac",
    }),
    "model-00002-of-00002.safetensors":
    MappingProxyType({
        "size": 4_238_114_628,
        "sha256": "9536dc05892a6ca8410bcfea763dde422e2430c3cc1f3acd26c55182c8989017",
    }),
})

SEAMLESS_M4T_V2_CHECKPOINTS = MappingProxyType({
    SEAMLESS_M4T_V2_REPOSITORY:
    MappingProxyType({
        "revision": SEAMLESS_M4T_V2_REVISION,
        "checkpoint": "model.safetensors.index.json",
        "index_etag": "f37bceddd602f9ffd93d6d329e5b2ed67b5251c6",
        "full_tensor_count": 2_232,
        "full_parameter_count": 2_309_249_669,
        "full_tensor_bytes": 9_236_998_676,
        "full_header_fingerprint": "e415ce94fe9bda5d062f9c777b4033eb1a340f75bdb20a3ae455aa7e690fb82e",
        "s2t_tensor_count": 1_429,
        "s2t_parameter_count": 1_501_842_240,
        "s2t_tensor_bytes": 6_007_368_960,
        "s2t_header_fingerprint": "2f12727c9a2e9b844e57efaef0aa42dcaf5394425633561591e285455cb39bef",
        "dtype": "F32",
        "license": SEAMLESS_M4T_V2_LICENSE,
        "shards": SEAMLESS_M4T_V2_SHARDS,
    }),
})

__all__ = [
    "SEAMLESS_M4T_V2_CHECKPOINTS",
    "SEAMLESS_M4T_V2_LICENSE",
    "SEAMLESS_M4T_V2_LICENSE_URL",
    "SEAMLESS_M4T_V2_REPOSITORY",
    "SEAMLESS_M4T_V2_REVISION",
    "SEAMLESS_M4T_V2_SHARDS",
    "TRANSFORMERS_SOURCE_REPOSITORY",
    "TRANSFORMERS_SOURCE_REVISION",
    "TRANSFORMERS_SOURCE_VERSION",
]
