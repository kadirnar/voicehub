"""Speech-task identifiers shared by registries, factories, and metadata."""

from __future__ import annotations

from enum import Enum


class SpeechTask(str, Enum):
    """Canonical public tasks supported by VoiceHub."""

    TEXT_TO_SPEECH = "text-to-speech"
    AUTOMATIC_SPEECH_RECOGNITION = "automatic-speech-recognition"
    VOICE_ACTIVITY_DETECTION = "voice-activity-detection"
    AUDIO_CODEC = "audio-codec"

    @classmethod
    def coerce(cls, value: SpeechTask | str) -> SpeechTask:
        """Normalize a task enum, canonical name, or documented alias."""
        if isinstance(value, cls):
            return value
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Speech tasks must be non-empty strings.")
        normalized = value.strip().lower().replace("_", "-")
        aliases = {
            "tts": cls.TEXT_TO_SPEECH.value,
            "speech-synthesis": cls.TEXT_TO_SPEECH.value,
            "asr": cls.AUTOMATIC_SPEECH_RECOGNITION.value,
            "speech-recognition": cls.AUTOMATIC_SPEECH_RECOGNITION.value,
            "speech-to-text": cls.AUTOMATIC_SPEECH_RECOGNITION.value,
            "stt": cls.AUTOMATIC_SPEECH_RECOGNITION.value,
            "vad": cls.VOICE_ACTIVITY_DETECTION.value,
            "speech-activity-detection": cls.VOICE_ACTIVITY_DETECTION.value,
            "codec": cls.AUDIO_CODEC.value,
            "audio-tokenization": cls.AUDIO_CODEC.value,
        }
        normalized = aliases.get(normalized, normalized)
        try:
            return cls(normalized)
        except ValueError as exc:
            choices = ", ".join(task.value for task in cls)
            raise ValueError(f"Unknown speech task {value!r}. Expected one of: {choices}.") from exc

    def __str__(self) -> str:
        return self.value
