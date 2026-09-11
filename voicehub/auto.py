"""Automatic configuration, processor, and task-specific model factories."""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from pathlib import Path

from voicehub.configuration import VoiceHubConfig
from voicehub.errors import UnknownModelError
from voicehub.hub import read_json_file, resolve_pretrained_file
from voicehub.inference_strategy import InferenceStrategy
from voicehub.processing.processor import VoiceHubProcessor
from voicehub.registry import (
    MODEL_REGISTRY,
    ModelSpec,
    get_default_model_spec,
    get_model_spec,
    register_model_spec,
    unregister_model_spec,
)
from voicehub.tasks import SpeechTask


def _load_class(module_name: str, class_name: str):
    module = import_module(module_name)
    return getattr(module, class_name)


_FACTORY_NAME_BY_TASK = {
    SpeechTask.TEXT_TO_SPEECH: "AutoModelForTextToSpeech",
    SpeechTask.AUTOMATIC_SPEECH_RECOGNITION: "AutoModelForSpeechRecognition",
    SpeechTask.VOICE_ACTIVITY_DETECTION: "AutoModelForVoiceActivityDetection",
    SpeechTask.AUDIO_CODEC: "AutoModelForAudioCodec",
}

_PROCESSOR_HUB_KWARGS = (
    "subfolder",
    "cache_dir",
    "revision",
    "token",
    "local_files_only",
)


def _get_task_model_spec(
    model_type: str,
    *,
    expected_task: SpeechTask | None,
) -> ModelSpec:
    """Resolve *model_type* and reject cross-task factory usage early."""
    spec = get_model_spec(model_type)
    if expected_task is None or spec.task is expected_task:
        return spec

    expected_factory = _FACTORY_NAME_BY_TASK[expected_task]
    registered_factory = _FACTORY_NAME_BY_TASK[spec.task]
    raise ValueError(
        f"Model {model_type!r} is registered for task {spec.task.value!r}, "
        f"so it cannot be loaded by {expected_factory}. "
        f"Use {registered_factory} instead.")


class AutoConfig:
    """Instantiate the registered configuration class for a model type."""

    def __init__(self):
        raise OSError("AutoConfig must be created with for_model/from_pretrained.")

    @classmethod
    def for_model(cls, model_type: str, **kwargs) -> VoiceHubConfig:
        spec = get_model_spec(model_type)
        config_class = _load_class(spec.config_module, spec.config_class)
        kwargs.setdefault('architectures', [spec.class_name])
        config = config_class(**kwargs)
        if config.model_type == "voicehub":
            config.model_type = spec.model_type
        return config

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        model_type: str | None = None,
        **kwargs,
    ) -> VoiceHubConfig:
        source = Path(pretrained_model_name_or_path).expanduser()
        is_direct_checkpoint = (source.is_file() and source.suffix.lower() != ".json")
        if model_type is None:
            if is_direct_checkpoint:
                raise ValueError(
                    "A raw checkpoint file does not identify its model type. "
                    "Pass `model_type` explicitly.")
            if source.is_file():
                config_path = source
            else:
                config_path = resolve_pretrained_file(
                    pretrained_model_name_or_path,
                    "config.json",
                    subfolder=kwargs.get("subfolder", ""),
                    cache_dir=kwargs.get("cache_dir"),
                    revision=kwargs.get("revision"),
                    token=kwargs.get("token"),
                    local_files_only=kwargs.get("local_files_only", False),
                )
            model_type = read_json_file(config_path).get("model_type")
            if not model_type:
                raise ValueError("config.json does not contain `model_type`; pass model_type explicitly.")
            normalized_model_type = (
                model_type.strip().lower() if isinstance(model_type, str) else model_type)
            if normalized_model_type not in MODEL_REGISTRY:
                raise UnknownModelError(
                    f"Checkpoint model type {model_type!r} is not a canonical "
                    "VoiceHub provider. Select a task-specific auto factory or "
                    "pass `model_type` explicitly.")

        spec = get_model_spec(model_type)
        config_class = _load_class(spec.config_module, spec.config_class)
        config = config_class.from_pretrained(
            pretrained_model_name_or_path,
            **kwargs,
        )
        if not config.architectures:
            config.architectures = [spec.class_name]
        return config


class _BaseAutoModel:
    """Shared implementation for task-constrained speech-model factories."""

    task: SpeechTask | None = None

    def __init__(self):
        raise OSError(f"{self.__class__.__name__} must be created with "
                      "from_config/from_pretrained.")

    @classmethod
    def _get_spec(cls, model_type: str) -> ModelSpec:
        return _get_task_model_spec(
            model_type,
            expected_task=cls.task,
        )

    @classmethod
    def _get_default_spec(cls) -> ModelSpec:
        spec = get_default_model_spec(cls.task) if cls.task is not None else None
        if spec is None:
            raise ValueError(
                f"{cls.__name__} has no registry-declared default checkpoint. "
                "Pass a model path or `model_type`.")
        return spec

    @classmethod
    def available_models(cls, *, task: SpeechTask | str | None = None) -> tuple[ModelSpec, ...]:
        """List compatible backends without importing their runtimes."""
        from voicehub.registry import list_model_specs

        if task is not None and cls.task is not None and SpeechTask.coerce(task) is not cls.task:
            raise ValueError("The requested task does not match this factory.")
        return list_model_specs(task=cls.task if task is None else task)

    @classmethod
    def register(
        cls,
        config_class: type[VoiceHubConfig],
        model_class: type,
        *,
        model_type: str | None = None,
        task: SpeechTask | str | None = None,
        default_model_path: str = "",
        aliases: tuple[str, ...] = (),
        capabilities: tuple[str, ...] = (),
        architecture: str | None = None,
        components: tuple[str, ...] = (),
        default_for_task: bool = False,
        install_extra: str | None = None,
        exist_ok: bool = False,
    ) -> ModelSpec:
        """Register a model/config pair for this task-specific factory.

        The signature mirrors the extension flow used by Transformers
        auto classes while retaining VoiceHub's task boundary. The
        registry stores import paths, not the class objects, so normal
        discovery stays lazy.
        """
        if not isinstance(config_class, type) or not issubclass(
                config_class,
                VoiceHubConfig,
        ):
            raise TypeError("`config_class` must inherit VoiceHubConfig.")
        resolved_model_type = model_type or getattr(config_class, "model_type", None)
        if not isinstance(resolved_model_type, str) or not resolved_model_type.strip():
            raise ValueError("`model_type` is required when config_class has no model_type.")
        if resolved_model_type.strip().lower() == "voicehub":
            raise ValueError("Extension config classes must declare a unique `model_type`.")
        resolved_task = cls.task if task is None else SpeechTask.coerce(task)
        if resolved_task is None:
            raise ValueError("AutoModel.register requires an explicit task.")
        if cls.task is not None and resolved_task is not cls.task:
            raise ValueError("The registered task does not match this factory.")
        spec = ModelSpec.from_classes(
            model_type=resolved_model_type,
            model_class=model_class,
            config_class=config_class,
            default_model_path=default_model_path,
            install_extra=install_extra,
            capabilities=capabilities,
            task=resolved_task,
            architecture=architecture,
            components=components,
            default_for_task=default_for_task,
        )
        register_model_spec(
            spec,
            aliases=aliases,
            exist_ok=exist_ok,
        )
        return spec

    @classmethod
    def unregister(
        cls,
        model_type: str,
        *,
        missing_ok: bool = False,
    ) -> ModelSpec | None:
        """Unregister a model owned by this task-specific factory."""
        try:
            spec = get_model_spec(model_type)
        except UnknownModelError:
            if missing_ok:
                return None
            raise
        if cls.task is not None and spec.task is not cls.task:
            raise ValueError(
                f"Model {model_type!r} belongs to {spec.task.value!r}, not "
                f"{cls.task.value!r}.")
        return unregister_model_spec(model_type, missing_ok=missing_ok)

    @classmethod
    def from_config(cls, config: VoiceHubConfig, **kwargs):
        """Construct the selected task through its shared pretrained
        lifecycle."""
        spec = cls._get_spec(config.model_type)
        model_class = _load_class(spec.module, spec.class_name)
        return model_class.from_pretrained("", config=config, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path = "",
        *,
        model_type: str | None = None,
        config: VoiceHubConfig | None = None,
        inference_strategy: str | InferenceStrategy | None = None,
        config_kwargs: Mapping[str, object] | None = None,
        **kwargs,
    ):
        if config_kwargs is None:
            config_values = {}
        elif isinstance(config_kwargs, Mapping):
            config_values = dict(config_kwargs)
        else:
            raise TypeError("`config_kwargs` must be a mapping or None.")
        if any(not isinstance(key, str) or not key for key in config_values):
            raise ValueError("`config_kwargs` keys must be non-empty strings.")
        if "model_type" in config_values:
            raise ValueError(
                "Pass `model_type` as the top-level factory argument, not "
                "inside `config_kwargs`.")
        if config is not None and config_values:
            raise TypeError("Pass configuration through either `config` or "
                            "`config_kwargs`, not both.")

        empty_source = (
            isinstance(pretrained_model_name_or_path, str) and not pretrained_model_name_or_path.strip())
        if config is None:
            if model_type is None:
                if empty_source:
                    spec = cls._get_default_spec()
                    config_class = _load_class(
                        spec.config_module,
                        spec.config_class,
                    )
                    config = config_class(
                        name_or_path=spec.default_model_path,
                        **config_values,
                    )
                else:
                    try:
                        config = AutoConfig.from_pretrained(
                            pretrained_model_name_or_path,
                            **config_values,
                        )
                        spec = cls._get_spec(config.model_type)
                    except UnknownModelError:
                        spec = get_default_model_spec(cls.task) if cls.task is not None else None
                        if spec is None:
                            raise
                        config_class = _load_class(
                            spec.config_module,
                            spec.config_class,
                        )
                        config = config_class(
                            name_or_path=pretrained_model_name_or_path,
                            **config_values,
                        )
            else:
                spec = cls._get_spec(model_type)
                config_class = _load_class(spec.config_module, spec.config_class)
                config = config_class(
                    name_or_path=(spec.default_model_path if empty_source else pretrained_model_name_or_path),
                    **config_values,
                )
                config.model_type = spec.model_type
        else:
            if model_type is not None and config.model_type == "voicehub":
                spec = cls._get_spec(model_type)
                config.model_type = spec.model_type
            else:
                configured_spec = cls._get_spec(config.model_type)
                if model_type is None:
                    spec = configured_spec
                else:
                    spec = cls._get_spec(model_type)
                if spec.model_type != configured_spec.model_type:
                    raise ValueError(
                        f"Explicit model_type {model_type!r} resolves to "
                        f"{spec.model_type!r}, but the supplied config targets "
                        f"{configured_spec.model_type!r}.")
        model_class = _load_class(spec.module, spec.class_name)
        return model_class.from_pretrained(
            "" if empty_source else pretrained_model_name_or_path,
            config=config,
            inference_strategy=inference_strategy,
            **kwargs,
        )


class AutoModelForTextToSpeech(_BaseAutoModel):
    """Load a registered text-to-speech model."""

    task = SpeechTask.TEXT_TO_SPEECH


class AutoModelForSpeechRecognition(_BaseAutoModel):
    """Load a registered automatic speech-recognition model."""

    task = SpeechTask.AUTOMATIC_SPEECH_RECOGNITION


class AutoModelForVoiceActivityDetection(_BaseAutoModel):
    """Load a registered voice-activity-detection model."""

    task = SpeechTask.VOICE_ACTIVITY_DETECTION


class AutoModelForAudioCodec(_BaseAutoModel):
    """Load a registered audio codec with encode and decode operations."""

    task = SpeechTask.AUDIO_CODEC


class AutoModel(_BaseAutoModel):
    """One factory for all registered TTS, STT, VAD, and codec models."""


class AutoProcessor:
    """Create the processor paired with a VoiceHub speech configuration."""

    def __init__(self):
        raise OSError("AutoProcessor must be created with from_config/from_pretrained.")

    @classmethod
    def _get_processor_class(
        cls,
        config: VoiceHubConfig,
    ) -> type[VoiceHubProcessor]:
        """Resolve the declared processor class without constructing it."""
        spec = get_model_spec(config.model_type)
        return _load_class(spec.processor_module, spec.processor_class)

    @classmethod
    def from_config(
        cls,
        config: VoiceHubConfig,
        **kwargs,
    ) -> VoiceHubProcessor:
        """Construct the processor class declared by the model."""
        return cls._get_processor_class(config)(**kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path = "",
        *,
        model_type: str | None = None,
        config: VoiceHubConfig | None = None,
        config_kwargs: Mapping[str, object] | None = None,
        **kwargs,
    ) -> VoiceHubProcessor:
        """Construct a processor without importing a model runtime."""
        if config_kwargs is None:
            config_values = {}
        elif isinstance(config_kwargs, Mapping):
            config_values = dict(config_kwargs)
        else:
            raise TypeError("`config_kwargs` must be a mapping or None.")
        if any(not isinstance(key, str) or not key for key in config_values):
            raise ValueError("`config_kwargs` keys must be non-empty strings.")
        if "model_type" in config_values:
            raise ValueError(
                "Pass `model_type` as the top-level factory argument, not "
                "inside `config_kwargs`.")
        if config is not None and config_values:
            raise TypeError("Pass configuration through either `config` or "
                            "`config_kwargs`, not both.")

        if config is None:
            if model_type is None:
                config = AutoConfig.from_pretrained(
                    pretrained_model_name_or_path,
                    **config_values,
                )
            else:
                config = AutoConfig.for_model(
                    model_type,
                    name_or_path=pretrained_model_name_or_path,
                    **config_values,
                )
        processor_class = cls._get_processor_class(config)
        processor_values = dict(kwargs)
        for name in _PROCESSOR_HUB_KWARGS:
            if name in config_values:
                processor_values.setdefault(name, config_values[name])
        return processor_class.from_pretrained(
            pretrained_model_name_or_path,
            **processor_values,
        )
