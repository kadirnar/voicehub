"""VoiceHub-owned MOSS-TTS architecture with lazy public exports."""

from __future__ import annotations

import importlib
from typing import Any

_PACKAGE = "voicehub.models.mosstts.native."
_EXPORTS = {
    "DEFAULT_MOSSTTS_ALIASES": _PACKAGE + "registration",
    "MOSS_CODEC_CHECKPOINTS": _PACKAGE + "metadata",
    "MOSS_CODEC_REVISIONS": _PACKAGE + "metadata",
    "MOSS_TTS_CHECKPOINTS": _PACKAGE + "metadata",
    "MOSS_TTS_REVISIONS": _PACKAGE + "metadata",
    "MossAudioCodec": _PACKAGE + "codec",
    "MossAudioCodecConfig": _PACKAGE + "codec",
    "MossAudioTokenizerConfig": _PACKAGE + "codec_configuration",
    "MossAudioTokenizerV1Model": _PACKAGE + "codec_modeling_v1",
    "MossAudioTokenizerV2Model": _PACKAGE + "codec_modeling",
    "MossCheckpointReport": _PACKAGE + "checkpoint",
    "MossCodecArtifacts": _PACKAGE + "artifacts",
    "MossCodecDecodeOutput": _PACKAGE + "codec",
    "MossCodecEncodeOutput": _PACKAGE + "codec",
    "MossCodecUnavailable": _PACKAGE + "codec",
    "MossDelayModel": _PACKAGE + "modeling",
    "MossGeneratedCodes": _PACKAGE + "processing",
    "MossGPT2Config": _PACKAGE + "configuration",
    "MossLocalV15Model": _PACKAGE + "modeling",
    "MossOldLocalModel": _PACKAGE + "modeling",
    "MossPreencodedDataset": _PACKAGE + "training",
    "MossProcessorBatch": _PACKAGE + "processing",
    "MossRealtimeModel": _PACKAGE + "modeling",
    "MossResidualQuantizer": _PACKAGE + "codec",
    "MossTTSArtifacts": _PACKAGE + "artifacts",
    "MossTTSConfig": _PACKAGE + "configuration",
    "MossTTSOutput": _PACKAGE + "modeling",
    "MossTTSProcessor": _PACKAGE + "processing",
    "MossTTSRuntime": _PACKAGE + "runtime",
    "MossTTSDataset": _PACKAGE + "training",
    "MossTextTokenizer": _PACKAGE + "tokenization",
    "MossVectorQuantizer": _PACKAGE + "codec",
    "NativeMossTTSTrainingAdapter": _PACKAGE + "training",
    "NativeMossAudioCodec": _PACKAGE + "codec",
    "build_moss_audio_tokenizer": _PACKAGE + "codec_checkpoint",
    "build_mosstts_model": _PACKAGE + "modeling",
    "create_mosstts_architecture_spec": _PACKAGE + "registration",
    "default_mosstts_codec_config": _PACKAGE + "runtime",
    "export_mosstts_checkpoint": _PACKAGE + "checkpoint",
    "export_moss_audio_tokenizer_checkpoint": _PACKAGE + "codec_checkpoint",
    "inspect_mosstts_checkpoint": _PACKAGE + "checkpoint",
    "load_mosstts_checkpoint": _PACKAGE + "checkpoint",
    "load_mosstts_runtime": _PACKAGE + "runtime",
    "load_moss_audio_tokenizer": _PACKAGE + "codec_checkpoint",
    "load_moss_audio_tokenizer_checkpoint": _PACKAGE + "codec_checkpoint",
    "register_mosstts_architecture": _PACKAGE + "registration",
    "resolve_mosstts_artifacts": _PACKAGE + "artifacts",
    "resolve_moss_codec_artifacts": _PACKAGE + "artifacts",
    "resolve_mosstts_dtype": _PACKAGE + "runtime",
    "save_mosstts_pretrained": _PACKAGE + "checkpoint",
    "save_moss_audio_tokenizer_pretrained": _PACKAGE + "codec_checkpoint",
    "validate_mosstts_tied_weights": _PACKAGE + "checkpoint",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_EXPORTS))
