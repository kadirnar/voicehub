"""Task-aware inference pipelines for normalized VoiceHub outputs."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicehub.auto import (
    AutoModelForAudioCodec,
    AutoModelForSpeechRecognition,
    AutoModelForTextToSpeech,
    AutoModelForVoiceActivityDetection,
)
from voicehub.errors import UnknownModelError
from voicehub.registry import get_model_spec
from voicehub.tasks import SpeechTask


class Pipeline:
    """Small task adapter around a loaded or lazily constructed speech
    model."""

    task: SpeechTask
    inference_method: str

    def __init__(self, model):
        self.model = model
        self._validate_model()

    def _validate_model(self) -> None:
        config = getattr(self.model, "config", None)
        model_type = getattr(config, "model_type", None)
        if isinstance(model_type, str) and model_type.strip():
            try:
                spec = get_model_spec(model_type)
            except UnknownModelError:
                pass
            else:
                if spec.task is not self.task:
                    raise ValueError(
                        f"Model {model_type!r} is registered for task "
                        f"{spec.task.value!r}, not {self.task.value!r}.")
        inference = getattr(self.model, self.inference_method, None)
        if not callable(inference):
            raise TypeError(
                f"A {self.task.value!r} pipeline requires a model with a "
                f"callable `{self.inference_method}()` method.")

    @property
    def model_type(self) -> str | None:
        """Return the configured model type when the model declares one."""
        config = getattr(self.model, "config", None)
        return getattr(config, "model_type", None)

    @property
    def device(self):
        """Return the device selected by the wrapped model."""
        return getattr(self.model, "device", None)

    @property
    def processor(self):
        """Return the processor owned by the wrapped model, when available."""
        return getattr(self.model, "processor", None)

    def __call__(self, inputs, **kwargs):
        """Run task inference and return the model's normalized output
        unchanged."""
        return getattr(self.model, self.inference_method)(inputs, **kwargs)

    def load(self) -> Pipeline:
        """Load the wrapped runtime eagerly and return this pipeline."""
        load = getattr(self.model, "load", None)
        if not callable(load):
            raise TypeError("The wrapped model does not expose `load()`.")
        load()
        return self

    def save_pretrained(self, directory: str | Path):
        """Delegate portable artifact serialization to the wrapped model."""
        save_pretrained = getattr(self.model, "save_pretrained", None)
        if not callable(save_pretrained):
            raise TypeError("The wrapped model does not expose `save_pretrained()`.")
        return save_pretrained(directory)


class TextToSpeechPipeline(Pipeline):
    """Pipeline that maps text to :class:`~voicehub.TTSOutput`."""

    task = SpeechTask.TEXT_TO_SPEECH
    inference_method = "generate"


class AutomaticSpeechRecognitionPipeline(Pipeline):
    """Pipeline that maps audio to :class:`~voicehub.ASROutput`."""

    task = SpeechTask.AUTOMATIC_SPEECH_RECOGNITION
    inference_method = "transcribe"


class VoiceActivityDetectionPipeline(Pipeline):
    """Pipeline that maps audio to :class:`~voicehub.VADOutput`."""

    task = SpeechTask.VOICE_ACTIVITY_DETECTION
    inference_method = "detect"


class AudioCodecPipeline(Pipeline):
    """Encode audio on call; decode the returned CodecOutput explicitly."""

    task = SpeechTask.AUDIO_CODEC
    inference_method = "encode"

    def _validate_model(self) -> None:
        super()._validate_model()
        if not callable(getattr(self.model, "decode", None)):
            raise TypeError("An audio-codec pipeline requires a callable decode() method.")

    def decode(self, encoded, **kwargs):
        return self.model.decode(encoded, **kwargs)


_PIPELINE_BY_TASK = {
    SpeechTask.TEXT_TO_SPEECH: TextToSpeechPipeline,
    SpeechTask.AUTOMATIC_SPEECH_RECOGNITION: AutomaticSpeechRecognitionPipeline,
    SpeechTask.VOICE_ACTIVITY_DETECTION: VoiceActivityDetectionPipeline,
    SpeechTask.AUDIO_CODEC: AudioCodecPipeline,
}

_AUTO_FACTORY_BY_TASK = {
    SpeechTask.TEXT_TO_SPEECH: AutoModelForTextToSpeech,
    SpeechTask.AUTOMATIC_SPEECH_RECOGNITION: AutoModelForSpeechRecognition,
    SpeechTask.VOICE_ACTIVITY_DETECTION: AutoModelForVoiceActivityDetection,
    SpeechTask.AUDIO_CODEC: AutoModelForAudioCodec,
}

_RESERVED_MODEL_KWARGS = frozenset({
    "config",
    "config_kwargs",
    "device",
    "inference_strategy",
    "model_type",
    "pretrained_model_name_or_path",
})


def _coerce_options(name: str, options: Mapping[str, Any] | None) -> dict[str, Any]:
    if options is None:
        return {}
    if not isinstance(options, Mapping):
        raise TypeError(f"`{name}` must be a mapping or None.")
    values = dict(options)
    if any(not isinstance(key, str) or not key for key in values):
        raise ValueError(f"`{name}` keys must be non-empty strings.")
    return values


def pipeline(
    task: SpeechTask | str,
    model=None,
    *,
    model_type: str | None = None,
    config=None,
    device: str | None = None,
    inference_strategy=None,
    config_kwargs: Mapping[str, Any] | None = None,
    model_kwargs: Mapping[str, Any] | None = None,
) -> Pipeline:
    """Create a task pipeline from a checkpoint source or existing model.

    Checkpoint sources are resolved through the task-specific auto
    factory and retain its lazy-loading default. Existing model objects
    are wrapped without changing device, runtime, or inference-strategy
    state.
    """
    resolved_task = SpeechTask.coerce(task)
    resolved_config_kwargs = _coerce_options("config_kwargs", config_kwargs)
    resolved_model_kwargs = _coerce_options("model_kwargs", model_kwargs)
    reserved = sorted(_RESERVED_MODEL_KWARGS.intersection(resolved_model_kwargs))
    if reserved:
        names = ", ".join(f"`{name}`" for name in reserved)
        raise ValueError(f"`model_kwargs` contains reserved pipeline options: {names}.")

    is_checkpoint_source = model is None or isinstance(model, (str, Path))
    if is_checkpoint_source:
        loader_kwargs = dict(resolved_model_kwargs)
        if model_type is not None:
            loader_kwargs["model_type"] = model_type
        if config is not None:
            loader_kwargs["config"] = config
        if device is not None:
            loader_kwargs["device"] = device
        if inference_strategy is not None:
            loader_kwargs["inference_strategy"] = inference_strategy
        if resolved_config_kwargs:
            loader_kwargs["config_kwargs"] = resolved_config_kwargs
        source = "" if model is None else model
        model = _AUTO_FACTORY_BY_TASK[resolved_task].from_pretrained(
            source,
            **loader_kwargs,
        )
    elif any((
            model_type is not None,
            config is not None,
            device is not None,
            inference_strategy is not None,
            bool(resolved_config_kwargs),
            bool(resolved_model_kwargs),
    )):
        raise TypeError(
            "Loader options cannot be used when wrapping an existing model; "
            "configure the model before passing it to `pipeline()`.")

    return _PIPELINE_BY_TASK[resolved_task](model)
