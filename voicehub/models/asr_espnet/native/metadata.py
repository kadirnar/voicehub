"""Immutable provenance for the audited ESPnet LibriSpeech release."""

ESPNET_SOURCE_REPOSITORY = "https://github.com/espnet/espnet"
ESPNET_SOURCE_REVISION = "75db853dd26a40d3d4dd979b2ff2457fbbb0cd69"
ESPNET_SOURCE_LICENSE = "Apache-2.0"

ESPNET_REPOSITORY = (
    "espnet/"
    "shinji-watanabe-librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_"
    "valid.acc.best")
ESPNET_LEGACY_ALIAS = (
    "espnet/"
    "kan-bayashi_librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_"
    "valid.acc.best")
ESPNET_REVISION = "bc6bbd771cec698f070640ee677a66719181f0a2"
ESPNET_CHECKPOINT_LICENSE = "CC-BY-4.0"

ESPNET_ASR_FILENAME = ("exp/asr_train_asr_transformer_e18_raw_bpe_sp/54epoch.pth")
ESPNET_ASR_SIZE = 397_647_953
ESPNET_ASR_SHA256 = ("1ab4b8ea9c656aac1044f628564f8476121f5624b62c26bf81c923cea2a09578")
ESPNET_ASR_TENSOR_COUNT = 462
ESPNET_ASR_STATE_VALUES = 99_385_344
ESPNET_ASR_TENSOR_FINGERPRINT = ("0709694e76ea178d45c7d7765efeb90f58d83e95136beb1a54aed1e34ff6fd7e")

ESPNET_LM_FILENAME = "exp/lm_train_lm_adam_bpe/17epoch.pth"
ESPNET_LM_SIZE = 619_075_639
ESPNET_LM_SHA256 = ("ed0c5dbed55057a513efcba58cabaa0ae3e8ceb754cff1ea19920cc2160ee091")
ESPNET_LM_TENSOR_COUNT = 19
ESPNET_LM_STATE_VALUES = 154_768_264
ESPNET_LM_SOURCE_TENSOR_FINGERPRINT = ("f1d94bea3bc029fc9da0417efdbd98a94d281456f032aacc1452bbbd6dc4a274")
ESPNET_LM_NATIVE_TENSOR_FINGERPRINT = ("7e9df002d479913094d49aa3bac60bd01bbf3d2901662c5e9bea356805fb4f02")

ESPNET_TOKENIZER_FILENAME = "data/token_list/bpe_unigram5000/bpe.model"
ESPNET_TOKENIZER_SIZE = 324_480
ESPNET_TOKENIZER_SHA256 = ("bdf1e0d937462a6e0487016db635feba953802be53b8d9d51cb59bc2cdcc4786")
ESPNET_CONFIG_FILENAME = ("exp/asr_train_asr_transformer_e18_raw_bpe_sp/config.yaml")
ESPNET_CONFIG_SIZE = 82_131
ESPNET_CONFIG_SHA256 = ("16351b9bf79631d1df0a4645a858dc330c40434cf03470408c9c8fd446b6ea19")
ESPNET_LM_CONFIG_FILENAME = "exp/lm_train_lm_adam_bpe/config.yaml"
ESPNET_LM_CONFIG_SIZE = 80_817
ESPNET_LM_CONFIG_SHA256 = ("eaf73708cc99b959374e33355e2e6902a7a1a96c76bd2ee3ff5b4c6d20840610")
ESPNET_TOKEN_LIST_SIZE = 46_827
ESPNET_TOKEN_LIST_SHA256 = ("48ec6eedbee6a22e2a9b51adeb425af3c39db23128086c015240f591601a3ea3")

ESPNET_TENSOR_FINGERPRINT_FORMAT = ("SHA-256 of checkpoint-order name|FloatStorage|dimxdim rows joined by LF")

__all__ = [name for name in globals() if name.startswith("ESPNET_")]
