"""Lazy public exports for native, serializable speech processing."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_MODULES = {
    "GlobalFeatureNormalization": "voicehub.processing.kaldi",
    "KaldiFbank": "voicehub.processing.kaldi",
    "KaldiFbankConfig": "voicehub.processing.kaldi",
    "LogMelSpectrogram": "voicehub.processing.audio",
    "ModelBatch": "voicehub.processing.schema",
    "NativeAudio": "voicehub.processing.waveform",
    "PROCESSING_OPERATIONS": "voicehub.processing.graph",
    "PadOrTrimAudio": "voicehub.processing.audio",
    "ProcessingOperation": "voicehub.processing.graph",
    "ProcessingOperationRegistry": "voicehub.processing.graph",
    "ProcessorGraph": "voicehub.processing.graph",
    "SpeechProcessor": "voicehub.processing.base",
    "TrainingExample": "voicehub.processing.schema",
    "decode_pcm_wave": "voicehub.processing.waveform",
    "htk_mel_filter_bank": "voicehub.processing.audio",
    "kaldi_fbank": "voicehub.processing.kaldi",
    "kaldi_mel_filter_bank": "voicehub.processing.kaldi",
    "load_global_cmvn": "voicehub.processing.kaldi",
    "load_native_audio": "voicehub.processing.waveform",
    "load_pcm_wave": "voicehub.processing.waveform",
    "mel_filter_bank": "voicehub.processing.audio",
    "normalize_waveform": "voicehub.processing.waveform",
    "resample_waveform": "voicehub.processing.waveform",
    "resample_waveform_hann": "voicehub.processing.waveform",
    "resample_waveform_kaiser": "voicehub.processing.waveform",
    "save_pcm_wave": "voicehub.processing.waveform",
}

__all__ = sorted(_MODULES)


def __getattr__(name: str) -> Any:
    try:
        module_name = _MODULES[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(
        import_module(module_name),
        name,
    )
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_MODULES))
