"""One lazy public API for TTS, STT, VAD, and audio codecs."""

from importlib import import_module

__version__ = "0.4.0"

_EXPORTS = {
    "ASRInferenceConfig": "voicehub.inference_configuration",
    "ASROutput": "voicehub.outputs",
    "ASRSegment": "voicehub.outputs",
    "ASRWord": "voicehub.outputs",
    "AudioCodecPipeline": "voicehub.pipelines",
    "AudioInput": "voicehub.audio",
    "AudioOutput": "voicehub.outputs",
    "AutoConfig": "voicehub.auto",
    "AutoModel": "voicehub.auto",
    "AutoModelForAudioCodec": "voicehub.auto",
    "AutoModelForSpeechRecognition": "voicehub.auto",
    "AutoModelForTextToSpeech": "voicehub.auto",
    "AutoModelForVoiceActivityDetection": "voicehub.auto",
    "AutoProcessor": "voicehub.auto",
    "AutomaticSpeechRecognitionPipeline": "voicehub.pipelines",
    "CodecOutput": "voicehub.outputs",
    "ModelRegistry": "voicehub.registry",
    "ModelSpec": "voicehub.registry",
    "Pipeline": "voicehub.pipelines",
    "PreTrainedASRModel": "voicehub.models.audio",
    "PreTrainedCodecModel": "voicehub.models.codec",
    "PreTrainedSpeechModel": "voicehub.model",
    "PreTrainedTTSModel": "voicehub.models.tts",
    "PreTrainedVADModel": "voicehub.models.audio",
    "SpeechTask": "voicehub.tasks",
    "TTSGenerationConfig": "voicehub.generation.configuration",
    "TTSOutput": "voicehub.outputs",
    "TextToSpeechPipeline": "voicehub.pipelines",
    "VADInferenceConfig": "voicehub.inference_configuration",
    "VADOutput": "voicehub.outputs",
    "VADSegment": "voicehub.outputs",
    "VoiceActivityDetectionPipeline": "voicehub.pipelines",
    "VoiceHubConfig": "voicehub.configuration",
    "get_model_spec": "voicehub.registry",
    "list_model_specs": "voicehub.registry",
    "load_audio": "voicehub.audio",
    "pipeline": "voicehub.pipelines",
    "register_model_spec": "voicehub.registry",
    "unregister_model_spec": "voicehub.registry",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
