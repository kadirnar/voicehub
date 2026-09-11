"""VoiceHub-owned NeuTTS and NeuCodec with lazy public exports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_PACKAGE = "voicehub.models.neutts.native."
_EXPORTS = {
    "LinearScalingRotaryEmbedding": _PACKAGE + "modeling",
    "NeuCodecArtifacts": _PACKAGE + "artifacts",
    "NeuCodecCheckpointAdapter": _PACKAGE + "checkpoint",
    "NeuCodecConfig": _PACKAGE + "configuration",
    "NeuCodecDecoderOutput": _PACKAGE + "neucodec",
    "NeuCodecEncoderOutput": _PACKAGE + "neucodec",
    "NeuCodecFeatureExtractor": _PACKAGE + "neucodec",
    "NeuCodecFeatures": _PACKAGE + "neucodec",
    "NeuCodecModel": _PACKAGE + "neucodec",
    "NeuCodecOutput": _PACKAGE + "neucodec",
    "NeuTTSArtifacts": _PACKAGE + "artifacts",
    "NeuTTSBackbone": _PACKAGE + "modeling",
    "NeuTTSBackboneConfig": _PACKAGE + "configuration",
    "NeuTTSCheckpointAdapter": _PACKAGE + "checkpoint",
    "NeuTTSRuntime": _PACKAGE + "modeling",
    "NeuTTSTokenizer": _PACKAGE + "tokenization",
    "DEFAULT_NEUTTS_ALIASES": _PACKAGE + "registration",
    "create_neutts_architecture_spec": _PACKAGE + "registration",
    "register_neutts_architecture": _PACKAGE + "registration",
    "resolve_neucodec_artifacts": _PACKAGE + "artifacts",
    "resolve_neutts_artifacts": _PACKAGE + "artifacts",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_EXPORTS))
