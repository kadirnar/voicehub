"""Public native Silero VAD provider with lazy component exports."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from voicehub.models.vad_silero.configuration import SileroVADConfig
    from voicehub.models.vad_silero.modeling import SileroVADForVoiceActivityDetection
    from voicehub.models.vad_silero.training_vad_silero import NativeSileroVADTrainingAdapter, SileroVADTrainingDataset

_PUBLIC_COMPONENTS = {
    "NativeSileroVADTrainingAdapter": ("voicehub.models.vad_silero.training_vad_silero"),
    "SileroVADConfig": ("voicehub.models.vad_silero.configuration"),
    "SileroVADForVoiceActivityDetection": ("voicehub.models.vad_silero.modeling"),
    "SileroVADTrainingDataset": ("voicehub.models.vad_silero.training_vad_silero"),
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
