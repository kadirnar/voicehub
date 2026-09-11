"""Public native FSMN VAD provider with lazy component exports."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from voicehub.models.vad_funasr.configuration import FunASRVADConfig
    from voicehub.models.vad_funasr.modeling import FunASRVADForVoiceActivityDetection
    from voicehub.models.vad_funasr.streaming import FSMNVADStreamingSession
    from voicehub.models.vad_funasr.training_vad_funasr import FSMNVADTrainingDataset, NativeFSMNVADTrainingAdapter

_PUBLIC_COMPONENTS = {
    "FSMNVADStreamingSession": "voicehub.models.vad_funasr.streaming",
    "FSMNVADTrainingDataset": ("voicehub.models.vad_funasr.training_vad_funasr"),
    "FunASRVADConfig": ("voicehub.models.vad_funasr.configuration"),
    "FunASRVADForVoiceActivityDetection": ("voicehub.models.vad_funasr.modeling"),
    "NativeFSMNVADTrainingAdapter": ("voicehub.models.vad_funasr.training_vad_funasr"),
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
