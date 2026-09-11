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
        "voicehub.processing.processor",
        "VoiceHubProcessor",
    ),
    SpeechTask.AUTOMATIC_SPEECH_RECOGNITION: (
        "voicehub.processing.processor",
        "AudioProcessor",
    ),
    SpeechTask.VOICE_ACTIVITY_DETECTION: (
        "voicehub.processing.processor",
        "AudioProcessor",
    ),
    SpeechTask.AUDIO_CODEC: (
        "voicehub.processing.processor",
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
    config_module: str = "voicehub.configuration"
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
            "ForAudioCodec",
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
        from voicehub.runtime import get_architecture_spec

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
        module=f"{package}.modeling",
        class_name=manifest.model_class,
        default_model_path=manifest.default_checkpoint,
        install_extra=manifest.install_extra,
        capabilities=manifest.capabilities,
        config_module=f"{package}.configuration",
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


_BUILTIN_CATALOG: ModelRegistry | None = None
_BUILTIN_CATALOG_LOCK = RLock()


def _builtin_catalog() -> ModelRegistry:
    """Construct the process catalog on first use, independent of import
    order."""
    global _BUILTIN_CATALOG
    with _BUILTIN_CATALOG_LOCK:
        if _BUILTIN_CATALOG is None:
            from voicehub.models.catalog import _BUILTIN_MODEL_ALIASES, _MODEL_SPECS

            _BUILTIN_CATALOG = ModelRegistry(_MODEL_SPECS, aliases=_BUILTIN_MODEL_ALIASES)
        return _BUILTIN_CATALOG


def __getattr__(name: str):
    if name not in {"MODEL_CATALOG", "MODEL_REGISTRY", "MODEL_ALIASES"}:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    catalog = _builtin_catalog()
    value = {
        "MODEL_CATALOG": catalog,
        "MODEL_REGISTRY": catalog.specs,
        "MODEL_ALIASES": catalog.aliases
    }[name]
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | {"MODEL_CATALOG", "MODEL_REGISTRY", "MODEL_ALIASES"})


def normalize_model_type(model_type: str) -> str:
    """Normalize a public model identifier to its canonical registry key."""
    return _builtin_catalog().normalize(model_type)


def get_model_spec(model_type: str) -> ModelSpec:
    """Return registry metadata or raise an error containing valid choices."""
    return _builtin_catalog().get(model_type)


def get_default_model_spec(task: SpeechTask | str) -> ModelSpec | None:
    """Return the registry-declared default for a speech task, if present."""
    return _builtin_catalog().get_default(task)


def list_model_specs(
    *,
    task: SpeechTask | str | None = None,
    native: bool | None = None,
) -> tuple[ModelSpec, ...]:
    """Return registered models with task and native-runtime filters."""
    return _builtin_catalog().list(task=task, native=native)


def register_model_alias(
    alias: str,
    model_type: str,
    *,
    exist_ok: bool = False,
) -> None:
    """Register a public alias, allowing idempotence when requested."""
    _builtin_catalog().register_alias(
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
    return _builtin_catalog().unregister_alias(alias, missing_ok=missing_ok)


def register_model_spec(
        spec: ModelSpec,
        *,
        aliases: Iterable[str] = (),
        exist_ok: bool = False,
) -> None:
    """Register or explicitly replace one lazily imported model backend."""
    _builtin_catalog().register(
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
    return _builtin_catalog().unregister(model_type, missing_ok=missing_ok)
