"""Task-aware model catalogue and extension registry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from types import MappingProxyType
from typing import Iterable, Iterator, Mapping

from voicehub.errors import UnknownModelError
from voicehub.models.manifests import BuiltinModelManifest, discover_builtin_model_manifests
from voicehub.tasks import SpeechTask

_DEFAULT_PROCESSOR_BY_TASK = {
    SpeechTask.TEXT_TO_SPEECH: (
        "voicehub.processing_utils",
        "VoiceHubProcessor",
    ),
    SpeechTask.AUTOMATIC_SPEECH_RECOGNITION: (
        "voicehub.processing_utils",
        "AudioProcessor",
    ),
    SpeechTask.VOICE_ACTIVITY_DETECTION: (
        "voicehub.processing_utils",
        "AudioProcessor",
    ),
}


def _normalize_identifier(value: str, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    normalized = value.strip().lower()
    if not normalized:
        raise ValueError(f"{name} must be a non-empty string.")
    return normalized


@dataclass(frozen=True)
class ModelSpec:
    """Metadata required to discover and lazily import a backend."""

    model_type: str
    module: str
    class_name: str
    default_model_path: str
    install_extra: str | None = None
    capabilities: tuple[str, ...] = ("text-to-speech", )
    config_module: str = "voicehub.configuration_utils"
    config_class: str = "VoiceHubConfig"
    task: SpeechTask | str = SpeechTask.TEXT_TO_SPEECH
    architecture: str | None = None
    components: tuple[str, ...] = ()
    default_for_task: bool = False
    processor_module: str | None = None
    processor_class: str | None = None

    def __post_init__(self) -> None:
        model_type = _normalize_identifier(self.model_type, name="model_type")
        object.__setattr__(self, "model_type", model_type)

        for field_name in (
                "module",
                "class_name",
                "config_module",
                "config_class",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string.")
            object.__setattr__(self, field_name, value.strip())
        install_extra = self.install_extra
        if install_extra is not None:
            if not isinstance(install_extra, str) or not install_extra.strip():
                raise ValueError("install_extra must be a non-empty string or None.")
            object.__setattr__(
                self,
                "install_extra",
                install_extra.strip(),
            )
        if not isinstance(self.default_model_path, str):
            raise TypeError("default_model_path must be a string.")

        task = SpeechTask.coerce(self.task)
        object.__setattr__(self, "task", task)
        if not isinstance(self.default_for_task, bool):
            raise TypeError("default_for_task must be a boolean.")

        processor_module = self.processor_module
        processor_class = self.processor_class
        if processor_module is None and processor_class is None:
            processor_module, processor_class = _DEFAULT_PROCESSOR_BY_TASK[task]
        elif processor_module is None or processor_class is None:
            raise ValueError("processor_module and processor_class must be declared together.")
        for field_name, value in (
            ("processor_module", processor_module),
            ("processor_class", processor_class),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string.")
            object.__setattr__(self, field_name, value.strip())

        capabilities = ((self.capabilities, ) if isinstance(self.capabilities, str) else tuple(
            self.capabilities))
        if any(not isinstance(capability, str) or not capability.strip() for capability in capabilities):
            raise ValueError("capabilities must contain non-empty strings.")
        capabilities = tuple(capability.strip().lower() for capability in capabilities)
        if (task is not SpeechTask.TEXT_TO_SPEECH and capabilities == (SpeechTask.TEXT_TO_SPEECH.value, )):
            capabilities = ()
        object.__setattr__(
            self,
            "capabilities",
            tuple(dict.fromkeys((task.value, *capabilities))),
        )

        architecture = self.architecture
        if architecture is not None:
            architecture = _normalize_identifier(
                architecture,
                name="architecture",
            )
        object.__setattr__(self, "architecture", architecture)

        components = ((self.components, ) if isinstance(self.components, str) else tuple(self.components))
        normalized_components = tuple(
            _normalize_identifier(component, name="component") for component in components)
        if len(set(normalized_components)) != len(normalized_components):
            raise ValueError("components must not contain duplicates.")
        object.__setattr__(self, "components", normalized_components)

    def supports_task(self, task: SpeechTask | str) -> bool:
        """Return whether this implementation belongs to *task*."""
        return self.task is SpeechTask.coerce(task)

    @property
    def display_name(self) -> str:
        """Return a presentation label without changing the registry key."""
        suffixes = (
            "ForTextToSpeech",
            "ForSpeechRecognition",
            "ForVoiceActivityDetection",
        )
        return next(
            (self.class_name[:-len(suffix)] for suffix in suffixes if self.class_name.endswith(suffix)),
            self.class_name,
        )

    @property
    def is_voicehub_native(self) -> bool:
        """Whether the executable architecture is owned by VoiceHub."""
        return "voicehub-native" in self.capabilities

    @property
    def native_architecture(self):
        """Return the native architecture declaration, when applicable."""
        if not self.is_voicehub_native:
            return None
        if self.architecture is None:
            raise RuntimeError(f"Native model {self.model_type!r} does not declare an "
                               "architecture ID.")
        from voicehub.architectures import get_architecture_spec

        return get_architecture_spec(self.architecture)

    @property
    def license(self):
        """Return special model/checkpoint license metadata, if present."""
        from voicehub.policies.licensing import get_model_license

        return get_model_license(self.model_type)

    @property
    def training(self):
        """Return the mandatory training profile for this backend."""
        from voicehub.training.specs import get_training_spec

        return get_training_spec(self.model_type)

    @classmethod
    def from_classes(
            cls,
            *,
            model_type: str,
            model_class: type,
            config_class: type,
            default_model_path: str = "",
            install_extra: str | None = None,
            capabilities: tuple[str, ...] = (),
            task: SpeechTask | str = SpeechTask.TEXT_TO_SPEECH,
            architecture: str | None = None,
            components: tuple[str, ...] = (),
            default_for_task: bool = False,
    ) -> ModelSpec:
        """Build a lazy registry declaration from Python classes.

        This is the extension-friendly counterpart to spelling out
        module and class names manually. Only import paths are retained,
        so later factory discovery remains lazy.
        """
        for name, value in (
            ("model_class", model_class),
            ("config_class", config_class),
        ):
            if not isinstance(value, type):
                raise TypeError(f"`{name}` must be a class.")
            if (not value.__module__ or not value.__name__ or value.__qualname__ != value.__name__):
                raise ValueError(f"`{name}` must have an importable module and name.")
        processor_type = getattr(model_class, "processor_class", None)
        if processor_type is not None:
            if not isinstance(processor_type, type):
                raise TypeError("`model_class.processor_class` must be a class.")
            if (not processor_type.__module__ or not processor_type.__name__ or
                    processor_type.__qualname__ != processor_type.__name__):
                raise ValueError("`model_class.processor_class` must have an importable "
                                 "module and name.")
        return cls(
            model_type=model_type,
            module=model_class.__module__,
            class_name=model_class.__name__,
            default_model_path=default_model_path,
            install_extra=install_extra,
            capabilities=capabilities,
            config_module=config_class.__module__,
            config_class=config_class.__name__,
            task=task,
            architecture=architecture,
            components=components,
            default_for_task=default_for_task,
            processor_module=(None if processor_type is None else processor_type.__module__),
            processor_class=(None if processor_type is None else processor_type.__name__),
        )


def model_spec_from_manifest(manifest: BuiltinModelManifest, ) -> ModelSpec:
    """Project one activated, source-only manifest into lazy registry
    metadata."""
    if not isinstance(manifest, BuiltinModelManifest):
        raise TypeError("`manifest` must be a BuiltinModelManifest.")
    package = f"voicehub.models.{manifest.model_type}"
    return ModelSpec(
        model_type=manifest.model_type,
        module=f"{package}.modeling_{manifest.model_type}",
        class_name=manifest.model_class,
        default_model_path=manifest.default_checkpoint,
        install_extra=manifest.install_extra,
        capabilities=manifest.capabilities,
        config_module=f"{package}.configuration_{manifest.model_type}",
        config_class=manifest.config_class,
        task=manifest.task,
        architecture=manifest.architecture,
        components=manifest.components,
        default_for_task=manifest.default_for_task,
    )


def discover_manifest_model_specs(models_root: str | Path | None = None) -> tuple[ModelSpec, ...]:
    """Discover lazy specs for activated package-local integration
    manifests."""
    return tuple(
        model_spec_from_manifest(manifest) for manifest in discover_builtin_model_manifests(models_root))


class ModelRegistry:
    """Thread-safe catalogue of lazily imported speech models.

    Registry instances are isolated, which makes extensions and tests
    composable. The process-wide :data:`MODEL_CATALOG` below preserves
    the existing module-level API and its read-only live mappings.
    """

    def __init__(
            self,
            specs: Iterable[ModelSpec] = (),
            *,
            aliases: Mapping[str, str] | None = None,
    ) -> None:
        self._lock = RLock()
        self._specs: dict[str, ModelSpec] = {}
        self._aliases: dict[str, str] = {}
        self._order: list[str] = []
        self._spec_view: Mapping[str, ModelSpec] = MappingProxyType(self._specs)
        self._alias_view: Mapping[str, str] = MappingProxyType(self._aliases)

        for spec in specs:
            self.register(spec)
        for alias, model_type in dict(aliases or {}).items():
            self.register_alias(alias, model_type)

    @property
    def specs(self) -> Mapping[str, ModelSpec]:
        """Read-only live view of canonical model declarations."""
        return self._spec_view

    @property
    def aliases(self) -> Mapping[str, str]:
        """Read-only live view of model aliases."""
        return self._alias_view

    def __len__(self) -> int:
        with self._lock:
            return len(self._specs)

    def __iter__(self) -> Iterator[str]:
        with self._lock:
            return iter(tuple(self._order))

    def __contains__(self, model_type: object) -> bool:
        if not isinstance(model_type, str):
            return False
        try:
            canonical = self.normalize(model_type)
        except (TypeError, ValueError):
            return False
        with self._lock:
            return canonical in self._specs

    def normalize(self, model_type: str) -> str:
        """Resolve a model alias to its canonical identifier."""
        normalized = _normalize_identifier(model_type, name="model_type")
        with self._lock:
            return self._aliases.get(normalized, normalized)

    def get(self, model_type: str) -> ModelSpec:
        """Return one declaration or raise with the available choices."""
        canonical = self.normalize(model_type)
        with self._lock:
            spec = self._specs.get(canonical)
            available = ", ".join(self._order)
        if spec is None:
            raise UnknownModelError(f"Unknown model type {model_type!r}. Available models: {available}.")
        return spec

    def get_default(self, task: SpeechTask | str) -> ModelSpec | None:
        """Return the model declared as the task default, when one exists."""
        resolved_task = SpeechTask.coerce(task)
        with self._lock:
            return next(
                (
                    self._specs[name] for name in self._order
                    if self._specs[name].task is resolved_task and self._specs[name].default_for_task),
                None,
            )

    def list(
        self,
        *,
        task: SpeechTask | str | None = None,
        native: bool | None = None,
    ) -> tuple[ModelSpec, ...]:
        """List declarations in stable registration order."""
        resolved_task = None if task is None else SpeechTask.coerce(task)
        if native is not None and not isinstance(native, bool):
            raise TypeError("`native` must be a boolean or None.")
        with self._lock:
            specs = tuple(self._specs[name] for name in self._order)
        if resolved_task is not None:
            specs = tuple(spec for spec in specs if spec.task is resolved_task)
        if native is not None:
            specs = tuple(spec for spec in specs if spec.is_voicehub_native is native)
        return specs

    def _validated_alias(
        self,
        alias: str,
        canonical: str,
        *,
        exist_ok: bool,
    ) -> str:
        normalized = _normalize_identifier(alias, name="alias")
        if normalized == canonical or normalized in self._specs:
            raise ValueError(f"Model alias {alias!r} collides with a registered model type.")
        existing = self._aliases.get(normalized)
        if existing is not None and (existing != canonical or not exist_ok):
            raise ValueError(f"Model alias {alias!r} is already registered for {existing!r}.")
        return normalized

    def register(
            self,
            spec: ModelSpec,
            *,
            aliases: Iterable[str] = (),
            exist_ok: bool = False,
    ) -> None:
        """Register or deliberately replace one model declaration."""
        if not isinstance(spec, ModelSpec):
            raise TypeError("Model registry entries must be ModelSpec instances.")
        if not isinstance(exist_ok, bool):
            raise TypeError("`exist_ok` must be a boolean.")
        aliases = tuple(aliases)
        if any(not isinstance(alias, str) for alias in aliases):
            raise TypeError("Model aliases must be strings.")

        with self._lock:
            if spec.model_type in self._aliases:
                target = self._aliases[spec.model_type]
                raise ValueError(
                    f"Model type {spec.model_type!r} collides with an alias "
                    f"for {target!r}.")
            if spec.model_type in self._specs and not exist_ok:
                raise ValueError(f"A model backend is already registered for {spec.model_type!r}.")
            if spec.default_for_task:
                existing_default = next(
                    (
                        self._specs[name] for name in self._order if name != spec.model_type and
                        self._specs[name].task is spec.task and self._specs[name].default_for_task),
                    None,
                )
                if existing_default is not None:
                    raise ValueError(
                        f"Task {spec.task.value!r} already declares default model "
                        f"{existing_default.model_type!r}.")
            normalized_aliases = tuple(
                self._validated_alias(alias, spec.model_type, exist_ok=exist_ok) for alias in aliases)
            if len(normalized_aliases) != len(set(normalized_aliases)):
                raise ValueError("Model aliases must not contain duplicates.")

            is_new = spec.model_type not in self._specs
            self._specs[spec.model_type] = spec
            if is_new:
                self._order.append(spec.model_type)
            for alias in normalized_aliases:
                self._aliases[alias] = spec.model_type

    def unregister(
        self,
        model_type: str,
        *,
        missing_ok: bool = False,
    ) -> ModelSpec | None:
        """Remove a model and all aliases that resolve to it."""
        if not isinstance(missing_ok, bool):
            raise TypeError("`missing_ok` must be a boolean.")
        canonical = self.normalize(model_type)
        with self._lock:
            try:
                spec = self._specs.pop(canonical)
            except KeyError:
                if missing_ok:
                    return None
                raise UnknownModelError(f"No model backend is registered for {model_type!r}.") from None
            self._order.remove(canonical)
            stale = tuple(alias for alias, target in self._aliases.items() if target == canonical)
            for alias in stale:
                del self._aliases[alias]
            return spec

    def register_alias(
        self,
        alias: str,
        model_type: str,
        *,
        exist_ok: bool = False,
    ) -> None:
        """Register an alias for an existing model."""
        if not isinstance(exist_ok, bool):
            raise TypeError("`exist_ok` must be a boolean.")
        with self._lock:
            canonical = self.normalize(model_type)
            if canonical not in self._specs:
                raise UnknownModelError(f"Cannot register an alias for unknown model type {model_type!r}.")
            normalized = self._validated_alias(
                alias,
                canonical,
                exist_ok=exist_ok,
            )
            self._aliases[normalized] = canonical

    def unregister_alias(
        self,
        alias: str,
        *,
        missing_ok: bool = False,
    ) -> str | None:
        """Remove an alias and return its former canonical target."""
        if not isinstance(missing_ok, bool):
            raise TypeError("`missing_ok` must be a boolean.")
        normalized = _normalize_identifier(alias, name="alias")
        with self._lock:
            try:
                return self._aliases.pop(normalized)
            except KeyError:
                if missing_ok:
                    return None
                raise KeyError(f"No model alias is registered for {alias!r}.") from None

    def clear(self) -> None:
        """Remove every model and alias from this registry instance."""
        with self._lock:
            self._specs.clear()
            self._aliases.clear()
            self._order.clear()


_MODEL_SPECS = (
    ModelSpec(
        "orpheustts",
        "voicehub.models.orpheustts.modeling_orpheustts",
        "OrpheusTTSForTextToSpeech",
        "canopylabs/orpheus-3b-0.1-ft",
        None,
        (
            "text-to-speech",
            "expressive-speech",
            "safetensors",
            "fine-tuning",
            "voicehub-native",
            "native-runtime",
        ),
        "voicehub.models.orpheustts.configuration_orpheustts",
        "OrpheusTTSConfig",
        architecture="causal-lm",
        default_for_task=True,
    ),
    ModelSpec(
        "dia",
        "voicehub.models.dia.modeling_dia",
        "DiaForTextToSpeech",
        "nari-labs/Dia-1.6B-0626",
        None,
        (
            "text-to-speech",
            "dialogue",
            "safetensors",
            "fine-tuning",
            "voicehub-native",
            "native-runtime",
        ),
        "voicehub.models.dia.configuration_dia",
        "DiaConfig",
        architecture="dia",
        components=("dac", ),
    ),
    ModelSpec(
        "vui",
        "voicehub.models.vui.modeling_vui",
        "VuiForTextToSpeech",
        "vui-abraham-100m.pt",
        None,
        (
            "text-to-speech",
            "fine-tuning",
            "safetensors",
            "standalone-safetensors-export",
            "voicehub-native",
            "native-runtime",
            "preprocessed-training",
        ),
        "voicehub.models.vui.configuration_vui",
        "VuiConfig",
        architecture="vui",
    ),
    ModelSpec(
        "chatterbox",
        "voicehub.models.chatterbox.modeling_chatterbox",
        "ChatterboxForTextToSpeech",
        "ResembleAI/chatterbox",
        None,
        (
            "text-to-speech",
            "voice-cloning",
            "fine-tuning",
            "safetensors",
            "voicehub-native",
            "native-runtime",
            "raw-audio-fine-tuning",
        ),
        "voicehub.models.chatterbox.configuration_chatterbox",
        "ChatterboxConfig",
        architecture="chatterbox",
        components=("conformer", ),
    ),
    ModelSpec(
        "kokoro",
        "voicehub.models.kokoro.modeling_kokoro",
        "KokoroForTextToSpeech",
        "hexgrad/Kokoro-82M",
        None,
        (
            "text-to-speech",
            "multilingual",
            "fine-tuning",
            "safetensors",
            "voicehub-native",
            "native-runtime",
        ),
        "voicehub.models.kokoro.configuration_kokoro",
        "KokoroConfig",
        architecture="kokoro",
    ),
    ModelSpec(
        "echo",
        "voicehub.models.echo.modeling_echo",
        "EchoTTSForTextToSpeech",
        "jordand/echo-tts-base",
        None,
        (
            "text-to-speech",
            "voice-cloning",
            "fine-tuning",
            "flow-matching",
            "safetensors",
            "voicehub-native",
            "native-runtime",
        ),
        "voicehub.models.echo.configuration_echo",
        "EchoTTSConfig",
        architecture="echo-dit",
    ),
    ModelSpec(
        "conversationtts",
        "voicehub.models.conversationtts.modeling_conversationtts",
        "ConversationTTSForTextToSpeech",
        "AudioFoundation/SpeechFoundation",
        None,
        (
            "text-to-speech",
            "voice-cloning",
            "conversation",
            "multilingual",
            "fine-tuning",
            "safetensors",
            "voicehub-native",
            "native-runtime",
            "raw-audio-fine-tuning",
            "preencoded-code-fine-tuning",
            "noncommercial",
        ),
        "voicehub.models.conversationtts.configuration_conversationtts",
        "ConversationTTSConfig",
        architecture="conversationtts",
    ),
    ModelSpec(
        "llasa",
        "voicehub.models.llasa.modeling_llasa",
        "LlasaForTextToSpeech",
        "HKUSTAudio/Llasa-1B-Multilingual",
        None,
        (
            "text-to-speech",
            "voice-cloning",
            "multilingual",
            "fine-tuning",
            "safetensors",
            "voicehub-native",
            "native-runtime",
            "raw-audio-fine-tuning",
            "preencoded-code-fine-tuning",
        ),
        "voicehub.models.llasa.configuration_llasa",
        "LlasaConfig",
        architecture="llasa",
    ),
    ModelSpec(
        "cosyvoice",
        "voicehub.models.cosyvoice.modeling_cosyvoice",
        "CosyVoiceForTextToSpeech",
        "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        None,
        (
            "text-to-speech",
            "voice-cloning",
            "multilingual",
            "fine-tuning",
            "flow-matching",
            "adversarial-vocoder-training",
            "safetensors",
            "voicehub-native",
            "native-runtime",
            "precomputed-speaker-embedding",
            "preencoded-speech-token-fine-tuning",
        ),
        "voicehub.models.cosyvoice.configuration_cosyvoice",
        "CosyVoiceConfig",
        architecture="cosyvoice-native",
        components=("conformer", ),
    ),
    ModelSpec(
        "f5tts",
        "voicehub.models.f5tts.modeling_f5tts",
        "F5TTSForTextToSpeech",
        "F5TTS_v1_Base",
        None,
        (
            "text-to-speech",
            "voice-cloning",
            "fine-tuning",
            "flow-matching",
            "safetensors",
            "voicehub-native",
            "native-runtime",
        ),
        "voicehub.models.f5tts.configuration_f5tts",
        "F5TTSConfig",
        architecture="f5tts",
        components=("vocos", ),
    ),
    ModelSpec(
        "gptsovits",
        "voicehub.models.gptsovits.modeling_gptsovits",
        "GPTSoVITSForTextToSpeech",
        "lj1995/GPT-SoVITS",
        None,
        (
            "text-to-speech",
            "voice-cloning",
            "multilingual",
            "fine-tuning",
            "safetensors",
            "voicehub-native",
            "native-runtime",
            "preprocessed-training",
            "gpt-sovits-v1",
            "gpt-sovits-v2",
            "gpt-sovits-v2-pro",
            "gpt-sovits-v2-pro-plus",
            "prepared-pro-speaker-conditioning",
            "variant-aware-safetensors-export",
        ),
        "voicehub.models.gptsovits.configuration_gptsovits",
        "GPTSoVITSConfig",
        architecture="gptsovits",
    ),
    ModelSpec(
        "melotts",
        "voicehub.models.melotts.modeling_melotts",
        "MeloTTSForTextToSpeech",
        "EN",
        None,
        (
            "text-to-speech",
            "multilingual",
            "fine-tuning",
            "safetensors",
            "voicehub-native",
            "native-runtime",
            "preprocessed-training",
            "explicit-linguistic-features",
        ),
        "voicehub.models.melotts.configuration_melotts",
        "MeloTTSConfig",
        architecture="melotts",
    ),
    ModelSpec(
        "openvoice",
        "voicehub.models.openvoice.modeling_openvoice",
        "OpenVoiceForTextToSpeech",
        "myshell-ai/OpenVoiceV2",
        None,
        (
            "text-to-speech",
            "voice-cloning",
            "multilingual",
            "fine-tuning",
            "safetensors",
            "voicehub-native",
            "native-runtime",
            "paired-waveform-training",
            "explicit-base-waveform",
        ),
        "voicehub.models.openvoice.configuration_openvoice",
        "OpenVoiceConfig",
        architecture="openvoice-v2-converter",
        components=("wavmark", ),
    ),
    ModelSpec(
        "outetts",
        "voicehub.models.outetts.modeling_outetts",
        "OuteTTSForTextToSpeech",
        "OuteAI/Llama-OuteTTS-1.0-1B",
        None,
        (
            "text-to-speech",
            "voice-cloning",
            "fine-tuning",
            "safetensors",
            "voicehub-native",
            "native-runtime",
            "preprocessed-training",
            "speaker-profile-training",
        ),
        "voicehub.models.outetts.configuration_outetts",
        "OuteTTSConfig",
        architecture="outetts",
        components=("dac", ),
    ),
    ModelSpec(
        "parlertts",
        "voicehub.models.parlertts.modeling_parlertts",
        "ParlerTTSForTextToSpeech",
        "parler-tts/parler-tts-mini-v1",
        None,
        (
            "text-to-speech",
            "prompted-style",
            "fine-tuning",
            "safetensors",
            "voicehub-native",
            "native-runtime",
            "raw-audio-fine-tuning",
        ),
        "voicehub.models.parlertts.configuration_parlertts",
        "ParlerTTSConfig",
        architecture="parlertts",
        components=("dac", ),
    ),
    ModelSpec(
        "styletts2",
        "voicehub.models.styletts2.modeling_styletts2",
        "StyleTTS2ForTextToSpeech",
        "",
        None,
        (
            "text-to-speech",
            "voice-cloning",
            "fine-tuning",
            "safetensors",
            "voicehub-native",
            "native-runtime",
            "preprocessed-training",
            "explicit-phonemes",
        ),
        "voicehub.models.styletts2.configuration_styletts2",
        "StyleTTS2Config",
        architecture="styletts2",
    ),
    ModelSpec(
        "mosstts",
        "voicehub.models.mosstts.modeling_mosstts",
        "MossTTSForTextToSpeech",
        "OpenMOSS-Team/MOSS-TTS-v1.5",
        None,
        (
            "text-to-speech",
            "voice-cloning",
            "multilingual",
            "fine-tuning",
            "safetensors",
            "voicehub-native",
            "native-runtime",
            "delay-variant",
            "local-variant",
            "local-v1.5-variant",
            "realtime-variant",
            "raw-audio-fine-tuning",
            "preencoded-rvq-fine-tuning",
            "native-codec-v1",
            "native-codec-v2",
            "buffered-generation",
        ),
        "voicehub.models.mosstts.configuration_mosstts",
        "MossTTSConfig",
        architecture="moss-tts",
    ),
    ModelSpec(
        "qwen3tts",
        "voicehub.models.qwen3tts.modeling_qwen3tts",
        "Qwen3TTSForTextToSpeech",
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        None,
        (
            "text-to-speech",
            "voice-cloning",
            "voice-design",
            "multilingual",
            "fine-tuning",
            "lora-fine-tuning",
            "default-checkpoint-inference-only",
            "safetensors",
            "voicehub-native",
            "native-runtime",
        ),
        "voicehub.models.qwen3tts.configuration_qwen3tts",
        "Qwen3TTSConfig",
        architecture="qwen3-tts",
    ),
    ModelSpec(
        "irodoritts",
        "voicehub.models.irodoritts.modeling_irodoritts",
        "IrodoriTTSForTextToSpeech",
        "Aratako/Irodori-TTS-500M-v3",
        None,
        (
            "text-to-speech",
            "voice-cloning",
            "voice-design",
            "multilingual",
            "fine-tuning",
            "flow-matching",
            "safetensors",
            "voicehub-native",
            "native-runtime",
            "raw-audio-fine-tuning",
            "preencoded-latent-fine-tuning",
            "duration-prediction",
        ),
        "voicehub.models.irodoritts.configuration_irodoritts",
        "IrodoriTTSConfig",
        architecture="irodoritts-rf-dit",
    ),
    ModelSpec(
        "zonos",
        "voicehub.models.zonos.modeling_zonos",
        "ZonosForTextToSpeech",
        "Zyphra/Zonos-v0.1-transformer",
        None,
        (
            "text-to-speech",
            "voice-cloning",
            "multilingual",
            "fine-tuning",
            "safetensors",
            "voicehub-native",
            "native-runtime",
        ),
        "voicehub.models.zonos.configuration_zonos",
        "ZonosConfig",
        architecture="zonos",
        components=("dac", ),
    ),
    ModelSpec(
        "zonos2",
        "voicehub.models.zonos2.modeling_zonos2",
        "Zonos2ForTextToSpeech",
        "Zyphra/ZONOS2",
        None,
        (
            "text-to-speech",
            "voice-cloning",
            "multilingual",
            "fine-tuning",
            "safetensors",
            "voicehub-native",
            "native-runtime",
        ),
        "voicehub.models.zonos2.configuration_zonos2",
        "Zonos2Config",
        architecture="zonos2",
        components=("dac", ),
    ),
    ModelSpec(
        "voxcpm",
        "voicehub.models.voxcpm.modeling_voxcpm",
        "VoxCPMForTextToSpeech",
        "openbmb/VoxCPM2",
        None,
        (
            "text-to-speech",
            "voice-cloning",
            "voice-design",
            "audio-continuation",
            "multilingual",
            "fine-tuning",
            "safetensors",
            "voicehub-native",
            "native-runtime",
        ),
        "voicehub.models.voxcpm.configuration_voxcpm",
        "VoxCPMConfig",
        architecture="voxcpm2",
    ),
    ModelSpec(
        "omnivoice",
        "voicehub.models.omnivoice.modeling_omnivoice",
        "OmniVoiceForTextToSpeech",
        "k2-fsa/OmniVoice",
        None,
        (
            "text-to-speech",
            "voice-cloning",
            "voice-design",
            "multilingual",
            "fine-tuning",
            "safetensors",
            "voicehub-native",
            "native-runtime",
            "raw-audio-fine-tuning",
            "preencoded-code-fine-tuning",
        ),
        "voicehub.models.omnivoice.configuration_omnivoice",
        "OmniVoiceConfig",
        architecture="omnivoice",
    ),
    ModelSpec(
        "higgstts",
        "voicehub.models.higgstts.modeling_higgstts",
        "HiggsTTSForTextToSpeech",
        "bosonai/higgs-tts-2-3b-base",
        None,
        (
            "text-to-speech",
            "voice-cloning",
            "expressive-speech",
            "multilingual",
            "fine-tuning",
            "safetensors",
            "voicehub-native",
            "native-runtime",
            "raw-audio-fine-tuning",
            "preencoded-code-fine-tuning",
        ),
        "voicehub.models.higgstts.configuration_higgstts",
        "HiggsTTSConfig",
        architecture="higgs_audio_v2",
    ),
    ModelSpec(
        "xtts",
        "voicehub.models.xtts.modeling_xtts",
        "XTTSForTextToSpeech",
        "coqui/XTTS-v2",
        None,
        (
            "text-to-speech",
            "voice-cloning",
            "multilingual",
            "fine-tuning",
            "safetensors",
            "voicehub-native",
            "native-runtime",
            "preencoded-code-fine-tuning",
            "gpt-fine-tuning",
            "restricted-pickle-conversion",
        ),
        "voicehub.models.xtts.configuration_xtts",
        "XTTSConfig",
        architecture="xtts2",
    ),
    ModelSpec(
        "vibevoice",
        "voicehub.models.vibevoice.modeling_vibevoice",
        "VibeVoiceForTextToSpeech",
        "microsoft/VibeVoice-Realtime-0.5B",
        None,
        (
            "text-to-speech",
            "voice-prompt",
            "fine-tuning",
            "default-checkpoint-inference-only",
            "safetensors",
            "voicehub-native",
            "native-runtime",
            "preprocessed-training",
            "verified-low-level-realtime-stages",
            "high-level-generation-fails-closed",
        ),
        "voicehub.models.vibevoice.configuration_vibevoice",
        "VibeVoiceConfig",
        architecture="vibevoice-tts",
    ),
    ModelSpec(
        "fishtts",
        "voicehub.models.fishtts.modeling_fishtts",
        "FishTTSForTextToSpeech",
        "fishaudio/s2-pro",
        None,
        (
            "text-to-speech",
            "voice-cloning",
            "multilingual",
            "fine-tuning",
            "safetensors",
            "voicehub-native",
            "native-runtime",
            "preprocessed-training",
            "noncommercial",
        ),
        "voicehub.models.fishtts.configuration_fishtts",
        "FishTTSConfig",
        architecture="fish-s2",
        components=("dac", ),
    ),
    ModelSpec(
        "csm",
        "voicehub.models.csm.modeling_csm",
        "CSMForTextToSpeech",
        "sesame/csm-1b",
        None,
        (
            "text-to-speech",
            "voice-cloning",
            "conversation",
            "safetensors",
            "fine-tuning",
            "raw-audio-training",
            "preencoded-code-training",
            "voicehub-native",
            "native-runtime",
        ),
        "voicehub.models.csm.configuration_csm",
        "CSMConfig",
        architecture="csm",
    ),
    ModelSpec(
        "neutts",
        "voicehub.models.neutts.modeling_neutts",
        "NeuTTSForTextToSpeech",
        "neuphonic/neutts-2e",
        None,
        (
            "text-to-speech",
            "voice-cloning",
            "multilingual",
            "emotion",
            "safetensors",
            "fine-tuning",
            "default-checkpoint-inference-only",
            "raw-audio-training",
            "preencoded-code-training",
            "voicehub-native",
            "native-runtime",
        ),
        "voicehub.models.neutts.configuration_neutts",
        "NeuTTSConfig",
        architecture="neutts",
    ),
    ModelSpec(
        "supertonic",
        "voicehub.models.supertonic.modeling_supertonic",
        "SupertonicForTextToSpeech",
        "Supertone/supertonic-3",
        None,
        (
            "text-to-speech",
            "multilingual",
            "fine-tuning",
            "safetensors",
            "voicehub-native",
            "native-runtime",
            "preprocessed-training",
        ),
        "voicehub.models.supertonic.configuration_supertonic",
        "SupertonicConfig",
        architecture="supertonic",
    ),
    ModelSpec(
        "inflecttts",
        "voicehub.models.inflecttts.modeling_inflecttts",
        "InflectTTSForTextToSpeech",
        "owensong/Inflect-Micro-v2",
        None,
        (
            "text-to-speech",
            "fine-tuning",
            "safetensors",
            "voicehub-native",
            "native-runtime",
            "preprocessed-training",
            "vits-warm-start",
            "explicit-phonemes",
        ),
        "voicehub.models.inflecttts.configuration_inflecttts",
        "InflectTTSConfig",
        architecture="inflecttts",
    ),
    ModelSpec(
        "bark",
        "voicehub.models.bark.modeling_bark",
        "BarkForTextToSpeech",
        "suno/bark-small",
        None,
        (
            "text-to-speech",
            "expressive-speech",
            "voice-prompt",
            "safetensors",
            "fine-tuning",
            "voicehub-native",
            "native-runtime",
            "preencoded-stage-training",
            "restricted-pickle-conversion",
        ),
        "voicehub.models.bark.configuration_bark",
        "BarkConfig",
        architecture="bark",
        components=("encodec", ),
    ),
    ModelSpec(
        "speecht5",
        "voicehub.models.speecht5.modeling_speecht5",
        "SpeechT5ForTextToSpeech",
        "microsoft/speecht5_tts",
        None,
        (
            "text-to-speech",
            "speaker-embedding",
            "safetensors",
            "fine-tuning",
            "voicehub-native",
            "native-runtime",
            "raw-audio-fine-tuning",
            "inference-reloadable-training-export",
        ),
        "voicehub.models.speecht5.configuration_speecht5",
        "SpeechT5Config",
        architecture="speecht5",
    ),
    ModelSpec(
        "vits",
        "voicehub.models.vits.modeling_vits",
        "VitsForTextToSpeech",
        "facebook/mms-tts-eng",
        None,
        (
            "text-to-speech",
            "multilingual",
            "mms-tts",
            "safetensors",
            "fine-tuning",
            "voicehub-native",
            "native-runtime",
            "raw-audio-training",
            "preprocessed-training",
            "adversarial-training",
            "generator-warm-start",
            "explicit-acoustic-training-config",
        ),
        "voicehub.models.vits.configuration_vits",
        "VitsConfig",
        architecture="vits",
    ),
)

_AUDIO_INPUT_MODEL_SPECS = (
    ModelSpec(
        "asr_transformers",
        "voicehub.models.asr_transformers.modeling_asr_transformers",
        "TransformersASRForSpeechRecognition",
        "openai/whisper-small",
        None,
        (
            "multilingual",
            "timestamps",
            "safetensors",
            "fine-tuning",
            "ctc",
            "speech-seq2seq",
            "voicehub-native",
            "native-runtime",
        ),
        "voicehub.models.asr_transformers.configuration_asr_transformers",
        "TransformersASRConfig",
        task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
        architecture="native-asr-dispatch",
        default_for_task=True,
    ),
    ModelSpec(
        "asr_whisper",
        "voicehub.models.asr_whisper_native.modeling_asr_whisper_native",
        "WhisperForSpeechRecognition",
        "openai/whisper-large-v3-turbo",
        None,
        (
            "multilingual",
            "translation",
            "timestamps",
            "safetensors",
            "fine-tuning",
            "voicehub-native",
        ),
        "voicehub.models.asr_whisper_native.configuration_asr_whisper_native",
        "WhisperASRConfig",
        task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
        architecture="whisper",
    ),
    ModelSpec(
        "asr_tiron",
        "voicehub.models.asr_tiron.modeling_asr_tiron",
        "TironForSpeechRecognition",
        "Trelis/tiron",
        None,
        (
            "multilingual",
            "speaker-attribution",
            "timestamps",
            "safetensors",
            "fine-tuning",
            "constrained-decoding",
            "voicehub-native",
        ),
        "voicehub.models.asr_tiron.configuration_asr_tiron",
        "TironASRConfig",
        task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
        architecture="whisper",
    ),
    ModelSpec(
        "asr_qwen3",
        "voicehub.models.asr_qwen3.modeling_asr_qwen3",
        "Qwen3ASRForSpeechRecognition",
        "Qwen/Qwen3-ASR-0.6B",
        None,
        (
            "multilingual",
            "language-identification",
            "hotwords",
            "long-form",
            "safetensors",
            "fine-tuning",
            "lora",
            "voicehub-native",
            "native-runtime",
        ),
        "voicehub.models.asr_qwen3.configuration_asr_qwen3",
        "Qwen3ASRConfig",
        task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
        architecture="qwen3-asr",
    ),
    ModelSpec(
        "asr_vibevoice",
        "voicehub.models.asr_vibevoice.modeling_asr_vibevoice",
        "VibeVoiceForSpeechRecognition",
        "microsoft/VibeVoice-ASR-HF",
        None,
        (
            "multilingual",
            "speaker-attribution",
            "timestamps",
            "hotwords",
            "long-form",
            "safetensors",
            "fine-tuning",
            "voicehub-native",
            "native-runtime",
        ),
        "voicehub.models.asr_vibevoice.configuration_asr_vibevoice",
        "VibeVoiceASRConfig",
        task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
        architecture="vibevoice-asr",
    ),
    ModelSpec(
        "asr_granite_speech",
        "voicehub.models.asr_granite_speech.modeling_asr_granite_speech",
        "GraniteSpeechForSpeechRecognition",
        "ibm-granite/granite-speech-4.1-2b",
        None,
        (
            "multilingual",
            "hotwords",
            "translation",
            "safetensors",
            "fine-tuning",
            "lora",
            "voicehub-native",
            "native-runtime",
        ),
        "voicehub.models.asr_granite_speech.configuration_asr_granite_speech",
        "GraniteSpeechASRConfig",
        task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
        architecture="granite-speech",
    ),
    ModelSpec(
        "asr_parakeet_tdt",
        "voicehub.models.asr_parakeet_tdt.modeling_asr_parakeet_tdt",
        "ParakeetTDTForSpeechRecognition",
        "nvidia/parakeet-tdt-0.6b-v3",
        None,
        (
            "multilingual",
            "timestamps",
            "long-form",
            "safetensors",
            "fine-tuning",
            "voicehub-native",
            "native-runtime",
        ),
        "voicehub.models.asr_parakeet_tdt.configuration_asr_parakeet_tdt",
        "ParakeetTDTASRConfig",
        task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
        architecture="parakeet-tdt",
    ),
    ModelSpec(
        "asr_nemotron",
        "voicehub.models.asr_nemotron.modeling_asr_nemotron",
        "NemotronForSpeechRecognition",
        "nvidia/nemotron-3.5-asr-streaming-0.6b",
        None,
        (
            "multilingual",
            "language-identification",
            "timestamps",
            "streaming-architecture",
            "safetensors",
            "fine-tuning",
            "voicehub-native",
            "native-runtime",
        ),
        "voicehub.models.asr_nemotron.configuration_asr_nemotron",
        "NemotronASRConfig",
        task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
        architecture="nemotron-3.5-rnnt",
    ),
    ModelSpec(
        "asr_cohere",
        "voicehub.models.asr_cohere.modeling_asr_cohere",
        "CohereForSpeechRecognition",
        "CohereLabs/cohere-transcribe-03-2026",
        None,
        (
            "multilingual",
            "long-form",
            "punctuation",
            "gated-checkpoint",
            "safetensors",
            "fine-tuning",
            "voicehub-native",
            "native-runtime",
        ),
        "voicehub.models.asr_cohere.configuration_asr_cohere",
        "CohereASRConfig",
        task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
        architecture="cohere-asr",
    ),
    ModelSpec(
        "asr_medasr",
        "voicehub.models.asr_medasr.modeling_asr_medasr",
        "MedASRForSpeechRecognition",
        "google/medasr",
        None,
        (
            "medical",
            "gated-checkpoint",
            "safetensors",
            "fine-tuning",
            "voicehub-native",
            "native-runtime",
        ),
        "voicehub.models.asr_medasr.configuration_asr_medasr",
        "MedASRConfig",
        task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
        architecture="lasr-ctc",
    ),
    ModelSpec(
        "asr_wav2vec2",
        "voicehub.models.asr_wav2vec2.modeling_asr_wav2vec2",
        "Wav2Vec2ForSpeechRecognition",
        "facebook/wav2vec2-base-960h",
        None,
        (
            "timestamps",
            "safetensors",
            "fine-tuning",
            "voicehub-native",
        ),
        "voicehub.models.asr_wav2vec2.configuration_asr_wav2vec2",
        "Wav2Vec2ASRConfig",
        task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
        architecture="wav2vec2",
    ),
    ModelSpec(
        "asr_hubert",
        "voicehub.models.asr_hubert.modeling_asr_hubert",
        "HubertForSpeechRecognition",
        "facebook/hubert-large-ls960-ft",
        None,
        (
            "timestamps",
            "safetensors",
            "fine-tuning",
            "voicehub-native",
        ),
        "voicehub.models.asr_hubert.configuration_asr_hubert",
        "HubertASRConfig",
        task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
        architecture="hubert",
    ),
    ModelSpec(
        "asr_wavlm",
        "voicehub.models.asr_wavlm.modeling_asr_wavlm",
        "WavLMForSpeechRecognition",
        "patrickvonplaten/wavlm-libri-clean-100h-base-plus",
        None,
        (
            "timestamps",
            "safetensors",
            "fine-tuning",
            "voicehub-native",
        ),
        "voicehub.models.asr_wavlm.configuration_asr_wavlm",
        "WavLMASRConfig",
        task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
        architecture="wavlm",
    ),
    ModelSpec(
        "asr_moonshine",
        "voicehub.models.asr_moonshine.modeling_asr_moonshine",
        "MoonshineForSpeechRecognition",
        "UsefulSensors/moonshine-tiny",
        None,
        (
            "safetensors",
            "fine-tuning",
            "compact",
            "voicehub-native",
        ),
        "voicehub.models.asr_moonshine.configuration_asr_moonshine",
        "MoonshineASRConfig",
        task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
        architecture="moonshine",
    ),
    ModelSpec(
        "asr_seamless_m4t_v2",
        "voicehub.models.asr_seamless_m4t_v2.modeling_asr_seamless_m4t_v2",
        "SeamlessM4Tv2ForSpeechRecognition",
        "facebook/seamless-m4t-v2-large",
        None,
        (
            "multilingual",
            "safetensors",
            "fine-tuning",
            "voicehub-native",
            "native-runtime",
            "greedy-decoding",
            "full-model-training",
        ),
        "voicehub.models.asr_seamless_m4t_v2.configuration_asr_seamless_m4t_v2",
        "SeamlessM4Tv2ASRConfig",
        task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
        architecture="seamless-m4t-v2-s2t",
    ),
    ModelSpec(
        "asr_faster_whisper",
        "voicehub.models.asr_native.faster_whisper",
        "FasterWhisperForSpeechRecognition",
        "openai/whisper-small",
        None,
        (
            "multilingual",
            "translation",
            "timestamps",
            "safetensors",
            "fine-tuning",
            "voicehub-native",
        ),
        "voicehub.models.asr_native.configuration",
        "FasterWhisperConfig",
        task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
        architecture="whisper",
    ),
    ModelSpec(
        "asr_whisperx",
        "voicehub.models.asr_native.whisperx",
        "WhisperXForSpeechRecognition",
        "openai/whisper-small",
        None,
        (
            "multilingual",
            "word-timestamps",
            "alignment",
            "safetensors",
            "fine-tuning",
            "voicehub-native",
            "native-runtime",
        ),
        "voicehub.models.asr_native.configuration",
        "WhisperXConfig",
        task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
        architecture="whisper",
    ),
    ModelSpec(
        "asr_openai_whisper",
        "voicehub.models.asr_native.openai_whisper",
        "OpenAIWhisperForSpeechRecognition",
        "openai/whisper-small",
        None,
        (
            "multilingual",
            "translation",
            "timestamps",
            "safetensors",
            "fine-tuning",
            "voicehub-native",
        ),
        "voicehub.models.asr_native.configuration",
        "OpenAIWhisperConfig",
        task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
        architecture="whisper",
    ),
    ModelSpec(
        "asr_nemo",
        "voicehub.models.asr_nemo",
        "NeMoASRForSpeechRecognition",
        "nvidia/nemo/stt_en_quartznet15x5",
        None,
        (
            "english",
            "timestamps",
            "safetensors",
            "fine-tuning",
            "voicehub-native",
            "ctc",
        ),
        "voicehub.models.asr_nemo",
        "NeMoASRConfig",
        task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
        architecture="nemo-asr",
    ),
    ModelSpec(
        "asr_speechbrain",
        "voicehub.models.asr_native.speechbrain",
        "SpeechBrainASRForSpeechRecognition",
        "speechbrain/asr-crdnn-rnnlm-librispeech",
        None,
        (
            "english",
            "beam-search",
            "safetensors",
            "fine-tuning",
            "voicehub-native",
            "crdnn",
            "ctc-seq2seq",
            "rnnlm-shallow-fusion",
        ),
        "voicehub.models.asr_native.configuration",
        "SpeechBrainASRConfig",
        task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
        architecture="speechbrain-crdnn-asr",
    ),
    ModelSpec(
        "asr_funasr",
        "voicehub.models.asr_native.funasr",
        "FunASRForSpeechRecognition",
        "FunAudioLLM/SenseVoiceSmall",
        None,
        (
            "multilingual",
            "timestamps",
            "language-identification",
            "emotion-recognition",
            "audio-events",
            "fine-tuning",
            "safetensors",
            "voicehub-native",
            "native-runtime",
        ),
        "voicehub.models.asr_native.configuration",
        "FunASRConfig",
        task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
        architecture="sensevoice-small",
    ),
    ModelSpec(
        "asr_espnet",
        "voicehub.models.asr_native.espnet",
        "ESPnetASRForSpeechRecognition",
        ("espnet/shinji-watanabe-librispeech_asr_train_asr_transformer_"
         "e18_raw_bpe_sp_valid.acc.best"),
        None,
        (
            "english",
            "safetensors",
            "fine-tuning",
            "voicehub-native",
            "native-runtime",
            "raw-audio-fine-tuning",
            "hybrid-ctc-attention",
        ),
        "voicehub.models.asr_native.configuration",
        "ESPnetASRConfig",
        task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
        architecture="espnet-librispeech-transformer-e18",
    ),
    ModelSpec(
        "asr_wenet",
        "voicehub.models.asr_wenet",
        "WeNetASRForSpeechRecognition",
        "wenet/gigaspeech-u2pp-conformer",
        None,
        (
            "english",
            "timestamps",
            "safetensors",
            "fine-tuning",
            "voicehub-native",
            "ctc",
            "attention-rescoring",
        ),
        "voicehub.models.asr_wenet",
        "WeNetASRConfig",
        task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
        architecture="wenet-asr",
    ),
    ModelSpec(
        "vad_transformers",
        "voicehub.models.vad_transformers.modeling_vad_transformers",
        "TransformersVADForVoiceActivityDetection",
        "",
        None,
        (
            "frame-scores",
            "safetensors",
            "fine-tuning",
            "voicehub-native",
            "native-runtime",
        ),
        "voicehub.models.vad_transformers.configuration_vad_transformers",
        "TransformersVADConfig",
        task=SpeechTask.VOICE_ACTIVITY_DETECTION,
        architecture="wav2vec2",
        default_for_task=True,
    ),
    ModelSpec(
        "vad_silero",
        "voicehub.models.vad_silero.modeling_vad_silero",
        "SileroVADForVoiceActivityDetection",
        "safestack/silero-vad",
        None,
        (
            "voicehub-native",
            "safetensors",
            "jit-weight-import",
            "frame-scores",
            "streaming",
            "fine-tuning",
        ),
        "voicehub.models.vad_silero.configuration_vad_silero",
        "SileroVADConfig",
        task=SpeechTask.VOICE_ACTIVITY_DETECTION,
        architecture="silero-vad",
    ),
    ModelSpec(
        "vad_webrtc",
        "voicehub.models.vad_webrtc.modeling_vad_webrtc",
        "WebRTCVADForVoiceActivityDetection",
        "webrtc-vad",
        None,
        (
            "fixed-point",
            "voicehub-native",
            "native-runtime",
            "streaming",
        ),
        "voicehub.models.vad_webrtc.configuration_vad_webrtc",
        "WebRTCVADConfig",
        task=SpeechTask.VOICE_ACTIVITY_DETECTION,
        architecture="webrtc-vad",
    ),
    ModelSpec(
        "vad_pyannote",
        "voicehub.models.vad_pyannote.modeling_vad_pyannote",
        "PyannoteVADForVoiceActivityDetection",
        "pyannote/voice-activity-detection",
        None,
        (
            "voicehub-native",
            "gated-checkpoint",
            "trusted-checkpoint-conversion",
            "safetensors",
            "frame-scores",
            "fine-tuning",
        ),
        "voicehub.models.vad_pyannote.configuration_vad_pyannote",
        "PyannoteVADConfig",
        task=SpeechTask.VOICE_ACTIVITY_DETECTION,
        architecture="pyannet",
    ),
    ModelSpec(
        "vad_speechbrain",
        "voicehub.models.vad_speechbrain.modeling_vad_speechbrain",
        "SpeechBrainVADForVoiceActivityDetection",
        "speechbrain/vad-crdnn-libriparty",
        None,
        (
            "voicehub-native",
            "safetensors",
            "trusted-checkpoint-conversion",
            "frame-scores",
            "fine-tuning",
            "offline-bidirectional",
        ),
        "voicehub.models.vad_speechbrain.configuration_vad_speechbrain",
        "SpeechBrainVADConfig",
        task=SpeechTask.VOICE_ACTIVITY_DETECTION,
        architecture="speechbrain-crdnn-vad",
    ),
    ModelSpec(
        "vad_nemo",
        "voicehub.models.vad_nemo.modeling_vad_nemo",
        "NeMoVADForVoiceActivityDetection",
        "nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0",
        None,
        (
            "voicehub-native",
            "safetensors",
            "trusted-checkpoint-conversion",
            "frame-scores",
            "fine-tuning",
        ),
        "voicehub.models.vad_nemo.configuration_vad_nemo",
        "NeMoVADConfig",
        task=SpeechTask.VOICE_ACTIVITY_DETECTION,
        architecture="marblenet-vad",
    ),
    ModelSpec(
        "vad_funasr",
        "voicehub.models.vad_funasr.modeling_vad_funasr",
        "FunASRVADForVoiceActivityDetection",
        "funasr/fsmn-vad",
        None,
        (
            "voicehub-native",
            "safetensors",
            "trusted-checkpoint-conversion",
            "frame-scores",
            "streaming",
            "fine-tuning",
            "modelscope-compatible",
        ),
        "voicehub.models.vad_funasr.configuration_vad_funasr",
        "FunASRVADConfig",
        task=SpeechTask.VOICE_ACTIVITY_DETECTION,
        architecture="fsmn-vad",
    ),
    ModelSpec(
        "vad_auditok",
        "voicehub.models.vad_auditok.modeling_vad_auditok",
        "AuditokVADForVoiceActivityDetection",
        "auditok-energy-vad",
        None,
        (
            "energy-based",
            "adaptive-threshold",
            "algorithmic",
            "voicehub-native",
        ),
        "voicehub.models.vad_auditok.configuration_vad_auditok",
        "AuditokVADConfig",
        task=SpeechTask.VOICE_ACTIVITY_DETECTION,
        architecture="energy-vad",
    ),
    ModelSpec(
        "vad_sherpa_onnx",
        "voicehub.models.vad_sherpa_onnx.modeling_vad_sherpa_onnx",
        "SherpaONNXVADForVoiceActivityDetection",
        "safestack/silero-vad",
        None,
        (
            "voicehub-native",
            "safetensors",
            "explicit-onnx-weight-conversion",
            "fine-tuning",
            "streaming",
            "sherpa-compatible-segmentation",
            "silero",
            "ten-vad",
        ),
        "voicehub.models.vad_sherpa_onnx.configuration_vad_sherpa_onnx",
        "SherpaONNXVADConfig",
        task=SpeechTask.VOICE_ACTIVITY_DETECTION,
        architecture="native-vad-dispatch",
    ),
    ModelSpec(
        "vad_pyannote_segmentation",
        "voicehub.models.vad_pyannote_segmentation."
        "modeling_vad_pyannote_segmentation",
        "PyannoteSegmentationVADForVoiceActivityDetection",
        "pyannote/segmentation-3.0",
        None,
        (
            "voicehub-native",
            "gated-checkpoint",
            "trusted-checkpoint-conversion",
            "safetensors",
            "powerset",
            "frame-scores",
            "fine-tuning",
        ),
        "voicehub.models.vad_pyannote_segmentation."
        "configuration_vad_pyannote_segmentation",
        "PyannoteSegmentationVADConfig",
        task=SpeechTask.VOICE_ACTIVITY_DETECTION,
        architecture="pyannet",
    ),
    ModelSpec(
        "vad_pyannote_brouhaha",
        "voicehub.models.vad_pyannote_brouhaha."
        "modeling_vad_pyannote_brouhaha",
        "PyannoteBrouhahaVADForVoiceActivityDetection",
        "pyannote/brouhaha",
        None,
        (
            "gated-checkpoint",
            "voicehub-native",
            "trusted-checkpoint-conversion",
            "safetensors",
            "frame-scores",
            "snr",
            "c50",
            "fine-tuning",
        ),
        "voicehub.models.vad_pyannote_brouhaha."
        "configuration_vad_pyannote_brouhaha",
        "PyannoteBrouhahaVADConfig",
        task=SpeechTask.VOICE_ACTIVITY_DETECTION,
        architecture="pyannet",
    ),
)

_MODEL_SPECS = _MODEL_SPECS + _AUDIO_INPUT_MODEL_SPECS

_BUILTIN_MODEL_ALIASES: dict[str, str] = {
    "conversation-tts": "conversationtts",
    "conversation_tts": "conversationtts",
    "cosy-voice": "cosyvoice",
    "f5": "f5tts",
    "f5-tts": "f5tts",
    "f5_tts": "f5tts",
    "gpt-sovits": "gptsovits",
    "gpt_sovits": "gptsovits",
    "higgs": "higgstts",
    "higgs-tts": "higgstts",
    "inflect": "inflecttts",
    "inflect-tts": "inflecttts",
    "irodori": "irodoritts",
    "irodori-tts": "irodoritts",
    "llasa-tts": "llasa",
    "llasa_tts": "llasa",
    "melo": "melotts",
    "melo-tts": "melotts",
    "melo_tts": "melotts",
    "moss": "mosstts",
    "moss-tts": "mosstts",
    "omni-voice": "omnivoice",
    "omni_voice": "omnivoice",
    "open-voice": "openvoice",
    "oute-tts": "outetts",
    "oute_tts": "outetts",
    "parler": "parlertts",
    "parler-tts": "parlertts",
    "parler_tts": "parlertts",
    "qwen3-tts": "qwen3tts",
    "qwen3_tts": "qwen3tts",
    "style-tts2": "styletts2",
    "style_tts2": "styletts2",
    "supertonic3": "supertonic",
    "bark-tts": "bark",
    "speech-t5": "speecht5",
    "speech_t5": "speecht5",
    "mms-tts": "vits",
    "mms_tts": "vits",
    "vits-tts": "vits",
    "vibe-voice": "vibevoice",
    "vibe_voice": "vibevoice",
    "vox-cpm": "voxcpm",
    "vox_cpm": "voxcpm",
    "zonos-2": "zonos2",
    "zonos_2": "zonos2",
    "transformers-asr": "asr_transformers",
    "hf-asr": "asr_transformers",
    "whisper-transformers": "asr_transformers",
    "hf-whisper": "asr_whisper",
    "whisper-large-v3-turbo": "asr_whisper",
    "tiron": "asr_tiron",
    "tiron-asr": "asr_tiron",
    "qwen-asr": "asr_qwen3",
    "qwen3-asr": "asr_qwen3",
    "vibevoice-asr": "asr_vibevoice",
    "granite-asr": "asr_granite_speech",
    "granite-speech": "asr_granite_speech",
    "parakeet": "asr_parakeet_tdt",
    "parakeet-tdt": "asr_parakeet_tdt",
    "parakeet-tdt-v3": "asr_parakeet_tdt",
    "nemotron-asr": "asr_nemotron",
    "nemotron-3.5-asr": "asr_nemotron",
    "cohere-asr": "asr_cohere",
    "cohere-transcribe": "asr_cohere",
    "medasr": "asr_medasr",
    "wav2vec2": "asr_wav2vec2",
    "wav2vec2-asr": "asr_wav2vec2",
    "hubert": "asr_hubert",
    "hubert-asr": "asr_hubert",
    "wavlm": "asr_wavlm",
    "wavlm-asr": "asr_wavlm",
    "moonshine": "asr_moonshine",
    "moonshine-asr": "asr_moonshine",
    "seamless-m4t": "asr_seamless_m4t_v2",
    "seamless-m4t-v2": "asr_seamless_m4t_v2",
    "mms-asr": "asr_transformers",
    "parakeet-transformers": "asr_parakeet_tdt",
    "faster-whisper": "asr_faster_whisper",
    "whisperx": "asr_whisperx",
    "openai-whisper": "asr_openai_whisper",
    "nemo-asr": "asr_nemo",
    "speechbrain-asr": "asr_speechbrain",
    "funasr": "asr_funasr",
    "espnet-asr": "asr_espnet",
    "wenet-asr": "asr_wenet",
    "transformers-vad": "vad_transformers",
    "silero-vad": "vad_silero",
    "webrtc-vad": "vad_webrtc",
    "pyannote-vad": "vad_pyannote",
    "speechbrain-vad": "vad_speechbrain",
    "nemo-vad": "vad_nemo",
    "funasr-vad": "vad_funasr",
    "fsmn-vad": "vad_funasr",
    "auditok-vad": "vad_auditok",
    "energy-vad": "vad_auditok",
    "sherpa-onnx-vad": "vad_sherpa_onnx",
    "sherpa-vad": "vad_sherpa_onnx",
    "pyannote-segmentation-vad": "vad_pyannote_segmentation",
    "pyannote-segmentation-3.0": "vad_pyannote_segmentation",
    "brouhaha-vad": "vad_pyannote_brouhaha",
    "pyannote-brouhaha": "vad_pyannote_brouhaha",
}

_DISCOVERED_BUILTIN_MANIFESTS = discover_builtin_model_manifests()
_DISCOVERED_MODEL_SPECS = tuple(
    model_spec_from_manifest(manifest) for manifest in _DISCOVERED_BUILTIN_MANIFESTS)
_CENTRAL_MODEL_TYPES = {spec.model_type for spec in _MODEL_SPECS}
_DUPLICATE_MANIFEST_MODELS = sorted(
    manifest.model_type for manifest in _DISCOVERED_BUILTIN_MANIFESTS
    if manifest.model_type in _CENTRAL_MODEL_TYPES)
if _DUPLICATE_MANIFEST_MODELS:
    raise ValueError(
        "Manifest-discovered models duplicate legacy central ModelSpec entries: "
        f"{_DUPLICATE_MANIFEST_MODELS!r}.")
for _manifest in _DISCOVERED_BUILTIN_MANIFESTS:
    for _alias in _manifest.aliases:
        if _alias in _BUILTIN_MODEL_ALIASES:
            raise ValueError(f"Manifest alias {_alias!r} duplicates a legacy central alias declaration.")
        _BUILTIN_MODEL_ALIASES[_alias] = _manifest.model_type
_MODEL_SPECS += _DISCOVERED_MODEL_SPECS

MODEL_CATALOG = ModelRegistry(
    _MODEL_SPECS,
    aliases=_BUILTIN_MODEL_ALIASES,
)
MODEL_REGISTRY: Mapping[str, ModelSpec] = MODEL_CATALOG.specs
MODEL_ALIASES: Mapping[str, str] = MODEL_CATALOG.aliases


def normalize_model_type(model_type: str) -> str:
    """Normalize a public model identifier to its canonical registry key."""
    return MODEL_CATALOG.normalize(model_type)


def get_model_spec(model_type: str) -> ModelSpec:
    """Return registry metadata or raise an error containing valid choices."""
    return MODEL_CATALOG.get(model_type)


def get_default_model_spec(task: SpeechTask | str) -> ModelSpec | None:
    """Return the registry-declared default for a speech task, if present."""
    return MODEL_CATALOG.get_default(task)


def list_model_specs(
    *,
    task: SpeechTask | str | None = None,
    native: bool | None = None,
) -> tuple[ModelSpec, ...]:
    """Return registered models with task and native-runtime filters."""
    return MODEL_CATALOG.list(task=task, native=native)


def register_model_alias(
    alias: str,
    model_type: str,
    *,
    exist_ok: bool = False,
) -> None:
    """Register a public alias, allowing idempotence when requested."""
    MODEL_CATALOG.register_alias(
        alias,
        model_type,
        exist_ok=exist_ok,
    )


def unregister_model_alias(
    alias: str,
    *,
    missing_ok: bool = False,
) -> str | None:
    """Remove a public alias and return its former canonical target."""
    return MODEL_CATALOG.unregister_alias(alias, missing_ok=missing_ok)


def register_model_spec(
        spec: ModelSpec,
        *,
        aliases: Iterable[str] = (),
        exist_ok: bool = False,
) -> None:
    """Register or explicitly replace one lazily imported model backend."""
    MODEL_CATALOG.register(
        spec,
        aliases=aliases,
        exist_ok=exist_ok,
    )


def unregister_model_spec(
    model_type: str,
    *,
    missing_ok: bool = False,
) -> ModelSpec | None:
    """Remove a model backend and every alias that resolves to it."""
    return MODEL_CATALOG.unregister(model_type, missing_ok=missing_ok)
