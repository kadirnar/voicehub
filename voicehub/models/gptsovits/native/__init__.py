"""VoiceHub-native GPT-SoVITS classic-S2 architecture.

The package intentionally keeps heavyweight PyTorch modules lazy.
Importing ``voicehub.models.gptsovits.native`` is therefore safe for
registry discovery.
"""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "GPTSoVITSS1Config": (
        "voicehub.models.gptsovits.native.configuration",
        "GPTSoVITSS1Config",
    ),
    "GPTSoVITSS2Config": (
        "voicehub.models.gptsovits.native.configuration",
        "GPTSoVITSS2Config",
    ),
    "GPTSoVITSSemanticModel": (
        "voicehub.models.gptsovits.native.semantic",
        "GPTSoVITSSemanticModel",
    ),
    "GPTSoVITSSynthesizer": (
        "voicehub.models.gptsovits.native.modeling",
        "GPTSoVITSSynthesizer",
    ),
    "GPTSoVITSRuntime": (
        "voicehub.models.gptsovits.native.runtime",
        "GPTSoVITSRuntime",
    ),
    "GPTSoVITSStagedTrainingModel": (
        "voicehub.models.gptsovits.native.training",
        "GPTSoVITSStagedTrainingModel",
    ),
    "build_s2_discriminator": (
        "voicehub.models.gptsovits.native.modeling",
        "build_s2_discriminator",
    ),
    "build_s2_generator": (
        "voicehub.models.gptsovits.native.modeling",
        "build_s2_generator",
    ),
    "build_staged_training_model": (
        "voicehub.models.gptsovits.native.training",
        "build_staged_training_model",
    ),
    "convert_gptsovits_legacy_checkpoints": (
        "voicehub.models.gptsovits.native.checkpoint",
        "convert_gptsovits_legacy_checkpoints",
    ),
    "export_gptsovits_checkpoint": (
        "voicehub.models.gptsovits.native.checkpoint",
        "export_gptsovits_checkpoint",
    ),
}


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


__all__ = sorted(_EXPORTS)
