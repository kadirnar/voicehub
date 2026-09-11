"""Pinned provenance for VoiceHub's native Supertonic 3 runtime."""

from __future__ import annotations

SUPERTONIC_SOURCE_REPOSITORY = "https://github.com/supertone-inc/supertonic"
SUPERTONIC_SOURCE_REVISION = ("7e2804f96016a7028cb1ed627353c61c1e9dd281")
SUPERTONIC_SOURCE_LICENSE = "MIT"

SUPERTONIC_CHECKPOINT_REPOSITORY = "Supertone/supertonic-3"
SUPERTONIC_CHECKPOINT_REVISION = ("3cadd1ee6394adea1bd021217a0e650ede09a323")
SUPERTONIC_CHECKPOINT_LICENSE = "OpenRAIL-M"

SUPERTONIC_LANGUAGES = frozenset({
    "ar",
    "bg",
    "cs",
    "da",
    "de",
    "el",
    "en",
    "es",
    "et",
    "fi",
    "fr",
    "hi",
    "hr",
    "hu",
    "id",
    "it",
    "ja",
    "ko",
    "lt",
    "lv",
    "na",
    "nl",
    "pl",
    "pt",
    "ro",
    "ru",
    "sk",
    "sl",
    "sv",
    "tr",
    "uk",
    "vi",
})

SUPERTONIC_GRAPH_FILES = {
    "duration_predictor": "duration_predictor.onnx",
    "text_encoder": "text_encoder.onnx",
    "vector_estimator": "vector_estimator.onnx",
    "vocoder": "vocoder.onnx",
}
SUPERTONIC_GRAPH_INTEGRITY = {
    "duration_predictor": (
        3_700_147,
        "c3eb91414d5ff8a7a239b7fe9e34e7e2bf8a8140d8375ffb14718b1c639325db",
        "ac0abf65413d0459a2af38c060f25b2936355191d596d2cffb486d388892a94c",
    ),
    "text_encoder": (
        36_416_150,
        "c7befd5ea8c3119769e8a6c1486c4edc6a3bc8365c67621c881bbb774b9902ff",
        "0adb860249d27c81bf6e50e23812902d822422a7036d5a52a664e4c8a826bb58",
    ),
    "vector_estimator": (
        256_534_781,
        "883ac868ea0275ef0e991524dc64f16b3c0376efd7c320af6b53f5b780d7c61c",
        "0d587ff295de03b0f5d6454dfb116eabd1b18aad74489db1bdaa12a15612c991",
    ),
    "vocoder": (
        101_424_195,
        "085de76dd8e8d5836d6ca66826601f615939218f90e519f70ee8a36ed2a4c4ba",
        "bc899685c63c1bc8ec7d8d649bbd6375d1053885410f4f1d98cbfd50aabfbfa5",
    ),
}
SUPERTONIC_PROCESSOR_INTEGRITY = {
    "tts.json": (
        8_253,
        "42078d3aef1cd43ab43021f3c54f47d2d75ceb4e75f627f118890128b06a0d09",
    ),
    "unicode_indexer.json": (
        277_676,
        "9bf7346e43883a81f8645c81224f786d43c5b57f3641f6e7671a7d6c493cb24f",
    ),
}
SUPERTONIC_STYLE_INTEGRITY = {
    "F1": (
        292_046,
        "bbdec6ee00231c2c742ad05483df5334cab3b52fda3ba38e6a07059c4563dbc2",
    ),
    "F2": (
        292_423,
        "7c722c6a72707b1a77f035d67f0d1351ba187738e06f7683e8c72b1df3477fc6",
    ),
    "F3": (
        290_794,
        "12f6ef2573baa2defa1128069cb59f203e3ab67c92af77b42df8a0e3a2f7c6ab",
    ),
    "F4": (
        291_808,
        "c2fa764c1225a76dfc3e2c73e8aa4f70d9ee48793860eb34c295fff01c2e032b",
    ),
    "F5": (
        291_479,
        "45966e73316415626cf41a7d1c6f3b4c70dbc1ba2bee5c1978ef0ce33244fc8d",
    ),
    "M1": (
        291_748,
        "e35604687f5d23694b8e91593a93eec0e4eca6c0b02bb8ed69139ab2ea6b0a5b",
    ),
    "M2": (
        292_055,
        "b76cbf62bac707c710cf0ae5aba5e31eea1a6339a9734bfae33ab98499534a50",
    ),
    "M3": (
        290_198,
        "ea1ac35ccb91b0d7ecad533a2fbd0eec10c91513d8951e3b25fbba99954e159b",
    ),
    "M4": (
        291_522,
        "ca8eefad4fcd989c9379032ff3e50738adc547eeb5e221b82593a6d7b3bac303",
    ),
    "M5": (
        291_469,
        "dd22b92740314321f8ae11c5e87f8dd60d060f15dd3a632b5adf77f471f77af2",
    ),
}

__all__ = [
    "SUPERTONIC_CHECKPOINT_LICENSE",
    "SUPERTONIC_CHECKPOINT_REPOSITORY",
    "SUPERTONIC_CHECKPOINT_REVISION",
    "SUPERTONIC_GRAPH_FILES",
    "SUPERTONIC_GRAPH_INTEGRITY",
    "SUPERTONIC_LANGUAGES",
    "SUPERTONIC_PROCESSOR_INTEGRITY",
    "SUPERTONIC_SOURCE_LICENSE",
    "SUPERTONIC_SOURCE_REPOSITORY",
    "SUPERTONIC_SOURCE_REVISION",
    "SUPERTONIC_STYLE_INTEGRITY",
]
