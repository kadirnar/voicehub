"""Immutable public provenance for Google MedASR."""

from __future__ import annotations

MEDASR_MODEL_ID = "google/medasr"
MEDASR_MODEL_REVISION = "ae1e4845b4b07479735d93e1e591e566435b7104"
MEDASR_WEIGHTS_INTRODUCED_REVISION = ("33dc1b0c1b01cb2886b1561d5d0fb4abb9d63b03")
MEDASR_SOURCE_REVISION = "65dc261512cbdb1ee72b88ae5b222f2605aad8e5"
MEDASR_RECIPE_REVISION = "ad843cb81b3e610e1868ed38f7230a70b66ed7e8"

# The official gated tree exposes the Git blob ID and file size but redacts
# the LFS object hash. The preservation copy recorded in SOURCE.json has the
# identical Git blob ID, which includes the LFS pointer contents, and exposes
# the underlying SHA-256. Header bytes were range-read only; no weight payload
# was downloaded during the audit.
MEDASR_CHECKPOINT = {
    "filename": "model.safetensors",
    "file_bytes": 421_172_424,
    "git_blob_oid": "46301a945e56697801394e9949fd90aacda7dfca",
    "lfs_sha256": ("03f61725f0c799624a4408e79ba3b0b3f43e9b54e994b1ba23949ae7b5a4a698"),
    "header_bytes": 41_016,
    "header_fingerprint": ("c302fca93cbde75690b26e9015b44f7e05dd39790a266aaab332b669363e5090"),
    "tensors": 368,
    "parameters": 105_282_833,
    "floating_parameters": 105_282_816,
    "tensor_data_bytes": 421_131_400,
    "dtype_counts": {
        "F32": 351,
        "I64": 17,
    },
}

MEDASR_ASSET_GIT_OIDS = {
    "added_tokens.json": "86274a18d83647b93e1eb775b4b3550a91766234",
    "config.json": "d6911cb9a3fa6833dd912282095e04d21ec5dfdd",
    "preprocessor_config.json": ("f0bf3713b4145f21d174e886e248966aba9a6ac6"),
    "processor_config.json": ("a8295064f6e1e847ad3726a2b71faec0bbfa55c9"),
    "spiece.model": "64bf355757fd19b5b288e2e00c8fa2275032be7e",
    "tokenizer.json": "e52b5511d41c42837a7bbd9ea717a7e973a06b0a",
    "tokenizer_config.json": ("189b1c53c64c059e58ec13c752b8892ec392b723"),
}

__all__ = [
    "MEDASR_ASSET_GIT_OIDS",
    "MEDASR_CHECKPOINT",
    "MEDASR_MODEL_ID",
    "MEDASR_MODEL_REVISION",
    "MEDASR_RECIPE_REVISION",
    "MEDASR_SOURCE_REVISION",
    "MEDASR_WEIGHTS_INTRODUCED_REVISION",
]
