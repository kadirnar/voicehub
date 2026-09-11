"""VoiceHub-native SpeechBrain CRDNN ASR architecture."""

from __future__ import annotations

from importlib import import_module

_PACKAGE = "voicehub.models.asr_speechbrain.native."
_EXPORTS = {
    "NATIVE_SPEECHBRAIN_ASR_FILENAME": _PACKAGE + "checkpoint",
    "NATIVE_SPEECHBRAIN_ASR_FORMAT": _PACKAGE + "checkpoint",
    "NATIVE_SPEECHBRAIN_ASR_TOKENIZER": _PACKAGE + "checkpoint",
    "SpeechBrainASRArtifacts": _PACKAGE + "artifacts",
    "SpeechBrainASRFrontend": _PACKAGE + "frontend",
    "SpeechBrainASROutput": _PACKAGE + "modeling",
    "SpeechBrainASRSafeTensorsCheckpointAdapter": _PACKAGE + "checkpoint",
    "SpeechBrainBeamResult": _PACKAGE + "decoding",
    "SpeechBrainCRDNNASRConfig": _PACKAGE + "configuration",
    "SpeechBrainCRDNNForASR": _PACKAGE + "modeling",
    "SpeechBrainRNNLMBeamSearch": _PACKAGE + "decoding",
    "convert_speechbrain_asr_checkpoints": _PACKAGE + "checkpoint",
    "create_speechbrain_asr_architecture_spec": _PACKAGE + "registration",
    "native_speechbrain_asr_tensor_shapes": _PACKAGE + "checkpoint",
    "register_speechbrain_asr_architecture": _PACKAGE + "registration",
    "resolve_speechbrain_asr_artifacts": _PACKAGE + "artifacts",
    "speechbrain_asr_source_tensor_mapping": _PACKAGE + "checkpoint",
    "speechbrain_lm_source_tensor_mapping": _PACKAGE + "checkpoint",
    "speechbrain_sequence_loss": _PACKAGE + "modeling",
}


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_EXPORTS))


__all__ = (
    "NATIVE_SPEECHBRAIN_ASR_FILENAME",
    "NATIVE_SPEECHBRAIN_ASR_FORMAT",
    "NATIVE_SPEECHBRAIN_ASR_TOKENIZER",
    "SpeechBrainASRArtifacts",
    "SpeechBrainASRFrontend",
    "SpeechBrainASROutput",
    "SpeechBrainASRSafeTensorsCheckpointAdapter",
    "SpeechBrainBeamResult",
    "SpeechBrainCRDNNASRConfig",
    "SpeechBrainCRDNNForASR",
    "SpeechBrainRNNLMBeamSearch",
    "convert_speechbrain_asr_checkpoints",
    "create_speechbrain_asr_architecture_spec",
    "native_speechbrain_asr_tensor_shapes",
    "register_speechbrain_asr_architecture",
    "resolve_speechbrain_asr_artifacts",
    "speechbrain_asr_source_tensor_mapping",
    "speechbrain_lm_source_tensor_mapping",
    "speechbrain_sequence_loss",
)
