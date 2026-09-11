"""MOSS-TTS public provider with dependency-light lazy exports."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "MossTTS": "voicehub.models.mosstts.modeling",
    "MossTTSConfig": "voicehub.models.mosstts.modeling",
    "MossTTSForTextToSpeech": "voicehub.models.mosstts.modeling",
    "MossPreencodedDataset": "voicehub.models.mosstts.native.training",
    "MossTTSDataset": "voicehub.models.mosstts.native.training",
    "NativeMossAudioCodec": "voicehub.models.mosstts.native.codec",
    "NativeMossTTSTrainingAdapter": "voicehub.models.mosstts.native.training",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name = _EXPORTS[name]
    except KeyError:
        raise AttributeError(name) from None
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_EXPORTS))
