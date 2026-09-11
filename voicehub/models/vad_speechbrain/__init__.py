"""Public native SpeechBrain VAD provider with lazy component exports."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from voicehub.models.vad_speechbrain.configuration import SpeechBrainVADConfig
    from voicehub.models.vad_speechbrain.modeling import SpeechBrainVADForVoiceActivityDetection
    from voicehub.models.vad_speechbrain.training_vad_speechbrain import (
        NativeSpeechBrainVADTrainingAdapter,
        SpeechBrainVADTrainingDataset,
    )

_PUBLIC_COMPONENTS = {
    "NativeSpeechBrainVADTrainingAdapter": ("voicehub.models.vad_speechbrain.training_vad_speechbrain"),
    "SpeechBrainVADConfig": ("voicehub.models.vad_speechbrain.configuration"),
    "SpeechBrainVADForVoiceActivityDetection": ("voicehub.models.vad_speechbrain.modeling"),
    "SpeechBrainVADTrainingDataset": ("voicehub.models.vad_speechbrain.training_vad_speechbrain"),
}

__all__ = sorted(_PUBLIC_COMPONENTS)


def __getattr__(name: str) -> Any:
    try:
        module_name = _PUBLIC_COMPONENTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
