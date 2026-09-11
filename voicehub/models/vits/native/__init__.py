"""VoiceHub-owned VITS/MMS-TTS architecture with lazy public exports."""

from __future__ import annotations

import importlib
from typing import Any

_PACKAGE = "voicehub.models.vits.native."
_EXPORTS = {
    "DEFAULT_VITS_ALIASES": _PACKAGE + "registration",
    "FACEBOOK_MMS_TTS_ENG_HEADER_FINGERPRINT": _PACKAGE + "checkpoint",
    "FACEBOOK_MMS_TTS_ENG_REVISION": _PACKAGE + "checkpoint",
    "HFVitsCheckpointAdapter": _PACKAGE + "checkpoint",
    "HifiGanResidualBlock": _PACKAGE + "modeling",
    "HuggingFaceVitsCheckpointAdapter": _PACKAGE + "checkpoint",
    "NativeVitsCheckpointAdapter": _PACKAGE + "checkpoint",
    "ORIGINAL_VITS_REVISION": _PACKAGE + "checkpoint",
    "TRANSFORMERS_VITS_REVISION": _PACKAGE + "checkpoint",
    "TextNormalizer": _PACKAGE + "frontend",
    "TextPhonemizer": _PACKAGE + "frontend",
    "TextRomanizer": _PACKAGE + "frontend",
    "VITS_TRAINING_SUPPORT": _PACKAGE + "losses",
    "VitsAcousticConfig": _PACKAGE + "training",
    "VitsAcousticFrontend": _PACKAGE + "training",
    "VitsAdversarialTrainingModel": _PACKAGE + "training",
    "VitsConfig": _PACKAGE + "configuration",
    "VitsFrontendAssetError": _PACKAGE + "frontend",
    "VitsFrontendCapabilityError": _PACKAGE + "frontend",
    "VitsFrontendConfig": _PACKAGE + "frontend",
    "VitsFrontendError": _PACKAGE + "frontend",
    "VitsGeneratorLoss": _PACKAGE + "losses",
    "VitsGenerationError": _PACKAGE + "modeling",
    "VitsInferenceOutput": _PACKAGE + "modeling",
    "VitsInputError": _PACKAGE + "modeling",
    "VitsModel": _PACKAGE + "modeling",
    "VitsMultiPeriodDiscriminator": _PACKAGE + "losses",
    "VitsSamplingConfig": _PACKAGE + "modeling",
    "VitsTokenizer": _PACKAGE + "frontend",
    "VitsTrainingOutput": _PACKAGE + "modeling",
    "VitsTrainingSupport": _PACKAGE + "losses",
    "create_vits_architecture_spec": _PACKAGE + "registration",
    "build_slaney_mel_filter": _PACKAGE + "training",
    "discriminator_loss": _PACKAGE + "losses",
    "feature_matching_loss": _PACKAGE + "losses",
    "generate_path": _PACKAGE + "alignment",
    "generator_adversarial_loss": _PACKAGE + "losses",
    "huggingface_vits_tensor_mapping": _PACKAGE + "checkpoint",
    "huggingface_vits_tensor_shapes": _PACKAGE + "checkpoint",
    "maximum_path": _PACKAGE + "alignment",
    "native_vits_tensor_names": _PACKAGE + "checkpoint",
    "native_vits_tensor_shapes": _PACKAGE + "checkpoint",
    "register_vits_architecture": _PACKAGE + "registration",
    "safetensors_header_fingerprint": _PACKAGE + "checkpoint",
    "sequence_mask": _PACKAGE + "alignment",
    "vits_kl_loss": _PACKAGE + "losses",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve public components only when a caller requests one."""
    try:
        module_name = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return stable results for interactive discovery."""
    return sorted((*globals(), *_EXPORTS))
