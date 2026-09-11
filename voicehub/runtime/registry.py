"""Thread-safe registry for native VoiceHub architecture specifications."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from threading import RLock
from types import MappingProxyType

from voicehub.runtime.specifications import ArchitectureError, ArchitectureSpec, normalize_architecture_id
from voicehub.tasks import SpeechTask


class UnknownArchitectureError(LookupError, ArchitectureError):
    """Raised when an architecture identifier is not registered."""


class ArchitectureRegistrationError(ValueError, ArchitectureError):
    """Raised when a registry mutation would make discovery ambiguous."""


class ArchitectureRegistry:
    """Mutable architecture catalogue with immutable, live public views.

    Registration and alias updates are atomic.  Listing uses a stable
    insertion order, including when an existing specification is
    deliberately replaced.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._specs: dict[str, ArchitectureSpec] = {}
        self._aliases: dict[str, str] = {}
        self._order: list[str] = []
        self._spec_view: Mapping[str, ArchitectureSpec] = MappingProxyType(self._specs)
        self._alias_view: Mapping[str, str] = MappingProxyType(self._aliases)

    @property
    def specs(self) -> Mapping[str, ArchitectureSpec]:
        """Read-only live view of canonical architecture specifications."""
        return self._spec_view

    @property
    def aliases(self) -> Mapping[str, str]:
        """Read-only live view of public aliases."""
        return self._alias_view

    def __len__(self) -> int:
        with self._lock:
            return len(self._specs)

    def __contains__(self, architecture_id: object) -> bool:
        if not isinstance(architecture_id, str):
            return False
        try:
            canonical = self.normalize(architecture_id)
        except (TypeError, ValueError):
            return False
        with self._lock:
            return canonical in self._specs

    def __iter__(self) -> Iterator[str]:
        with self._lock:
            return iter(tuple(self._order))

    def normalize(self, architecture_id: str) -> str:
        """Resolve an alias to a canonical architecture identifier."""
        normalized = normalize_architecture_id(architecture_id)
        with self._lock:
            return self._aliases.get(normalized, normalized)

    def get(self, architecture_id: str) -> ArchitectureSpec:
        """Return one specification or raise an informative lookup error."""
        canonical = self.normalize(architecture_id)
        with self._lock:
            spec = self._specs.get(canonical)
            available = ", ".join(self._order)
        if spec is None:
            suffix = f" Available architectures: {available}." if available else ""
            raise UnknownArchitectureError(f"Unknown architecture {architecture_id!r}.{suffix}")
        return spec

    def list(
        self,
        *,
        task: SpeechTask | str | None = None,
        training: bool | None = None,
        streaming: bool | None = None,
    ) -> tuple[ArchitectureSpec, ...]:
        """Return specifications in stable order with capability filters."""
        resolved_task = None if task is None else SpeechTask.coerce(task)
        for field_name, value in (
            ("training", training),
            ("streaming", streaming),
        ):
            if value is not None and not isinstance(value, bool):
                raise TypeError(f"{field_name} filter must be a boolean or None.")

        with self._lock:
            specs = tuple(self._specs[name] for name in self._order)
        if resolved_task is not None:
            specs = tuple(spec for spec in specs if spec.supports_task(resolved_task))
        if training is not None:
            specs = tuple(spec for spec in specs if spec.capabilities.training is training)
        if streaming is not None:
            specs = tuple(spec for spec in specs if spec.capabilities.streaming is streaming)
        return specs

    def _validate_alias(
        self,
        alias: str,
        canonical: str,
        *,
        exist_ok: bool,
    ) -> str:
        normalized = normalize_architecture_id(alias)
        if normalized == canonical:
            raise ArchitectureRegistrationError(
                f"Architecture alias {alias!r} is identical to its canonical ID.")
        if normalized in self._specs:
            raise ArchitectureRegistrationError(
                f"Architecture alias {alias!r} collides with a registered ID.")
        existing = self._aliases.get(normalized)
        if existing is not None and (existing != canonical or not exist_ok):
            raise ArchitectureRegistrationError(
                f"Architecture alias {alias!r} is already registered for "
                f"{existing!r}.")
        return normalized

    def register(
            self,
            spec: ArchitectureSpec,
            *,
            aliases: Iterable[str] = (),
            exist_ok: bool = False,
    ) -> None:
        """Register or deliberately replace one architecture specification."""
        if not isinstance(spec, ArchitectureSpec):
            raise TypeError("Architecture registry entries must be ArchitectureSpec instances.")
        if not isinstance(exist_ok, bool):
            raise TypeError("exist_ok must be a boolean.")
        aliases = tuple(aliases)
        if any(not isinstance(alias, str) for alias in aliases):
            raise TypeError("Architecture aliases must be strings.")

        with self._lock:
            alias_target = self._aliases.get(spec.architecture_id)
            if alias_target is not None:
                raise ArchitectureRegistrationError(
                    f"Architecture ID {spec.architecture_id!r} collides with "
                    f"an alias for {alias_target!r}.")
            if spec.architecture_id in self._specs and not exist_ok:
                raise ArchitectureRegistrationError(
                    f"Architecture {spec.architecture_id!r} is already registered.")

            normalized_aliases = tuple(
                self._validate_alias(
                    alias,
                    spec.architecture_id,
                    exist_ok=exist_ok,
                ) for alias in aliases)
            if len(set(normalized_aliases)) != len(normalized_aliases):
                raise ArchitectureRegistrationError("Architecture aliases must not contain duplicates.")

            is_new = spec.architecture_id not in self._specs
            self._specs[spec.architecture_id] = spec
            if is_new:
                self._order.append(spec.architecture_id)
            for alias in normalized_aliases:
                self._aliases[alias] = spec.architecture_id

    def unregister(
        self,
        architecture_id: str,
        *,
        missing_ok: bool = False,
    ) -> ArchitectureSpec | None:
        """Remove an architecture and every alias that targets it."""
        if not isinstance(missing_ok, bool):
            raise TypeError("missing_ok must be a boolean.")
        canonical = self.normalize(architecture_id)
        with self._lock:
            try:
                spec = self._specs.pop(canonical)
            except KeyError:
                if missing_ok:
                    return None
                raise UnknownArchitectureError(
                    f"No architecture is registered for {architecture_id!r}.") from None
            self._order.remove(canonical)
            stale_aliases = tuple(alias for alias, target in self._aliases.items() if target == canonical)
            for alias in stale_aliases:
                del self._aliases[alias]
            return spec

    def register_alias(
        self,
        alias: str,
        architecture_id: str,
        *,
        exist_ok: bool = False,
    ) -> None:
        """Register a public alias for an existing architecture."""
        if not isinstance(exist_ok, bool):
            raise TypeError("exist_ok must be a boolean.")
        with self._lock:
            canonical = self.normalize(architecture_id)
            if canonical not in self._specs:
                raise UnknownArchitectureError(
                    f"Cannot register an alias for unknown architecture "
                    f"{architecture_id!r}.")
            normalized = self._validate_alias(
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
            raise TypeError("missing_ok must be a boolean.")
        normalized = normalize_architecture_id(alias)
        with self._lock:
            try:
                return self._aliases.pop(normalized)
            except KeyError:
                if missing_ok:
                    return None
                raise KeyError(f"No architecture alias is registered for {alias!r}.") from None

    def clear(self) -> None:
        """Remove all entries.

        This method primarily supports isolated registries in tests and
        plugin hosts.  Production callers should normally unregister
        targeted entries.
        """
        with self._lock:
            self._specs.clear()
            self._aliases.clear()
            self._order.clear()


ARCHITECTURE_REGISTRY = ArchitectureRegistry()
ARCHITECTURES = ARCHITECTURE_REGISTRY.specs
ARCHITECTURE_ALIASES = ARCHITECTURE_REGISTRY.aliases

_BUILTINS_READY = False
_BUILTINS_LOCK = RLock()


def _ensure_builtins() -> None:
    global _BUILTINS_READY
    with _BUILTINS_LOCK:
        if not _BUILTINS_READY:
            from voicehub.runtime.catalog import register_builtin_architectures

            register_builtin_architectures()
            _BUILTINS_READY = True


def get_architecture_spec(architecture_id: str) -> ArchitectureSpec:
    """Return a specification from the process-wide architecture registry."""
    _ensure_builtins()
    return ARCHITECTURE_REGISTRY.get(architecture_id)


def list_architecture_specs(
    *,
    task: SpeechTask | str | None = None,
    training: bool | None = None,
    streaming: bool | None = None,
) -> tuple[ArchitectureSpec, ...]:
    """List process-wide architecture specifications."""
    _ensure_builtins()
    return ARCHITECTURE_REGISTRY.list(
        task=task,
        training=training,
        streaming=streaming,
    )


def register_architecture_spec(
        spec: ArchitectureSpec,
        *,
        aliases: Iterable[str] = (),
        exist_ok: bool = False,
) -> None:
    """Register a specification in the process-wide architecture registry."""
    _ensure_builtins()
    ARCHITECTURE_REGISTRY.register(
        spec,
        aliases=aliases,
        exist_ok=exist_ok,
    )


def unregister_architecture_spec(
    architecture_id: str,
    *,
    missing_ok: bool = False,
) -> ArchitectureSpec | None:
    """Remove a specification from the process-wide architecture registry."""
    _ensure_builtins()
    return ARCHITECTURE_REGISTRY.unregister(
        architecture_id,
        missing_ok=missing_ok,
    )


def register_architecture_alias(
    alias: str,
    architecture_id: str,
    *,
    exist_ok: bool = False,
) -> None:
    """Register an alias in the process-wide architecture registry."""
    _ensure_builtins()
    ARCHITECTURE_REGISTRY.register_alias(
        alias,
        architecture_id,
        exist_ok=exist_ok,
    )


def unregister_architecture_alias(
    alias: str,
    *,
    missing_ok: bool = False,
) -> str | None:
    """Remove an alias from the process-wide architecture registry."""
    _ensure_builtins()
    return ARCHITECTURE_REGISTRY.unregister_alias(
        alias,
        missing_ok=missing_ok,
    )


__all__ = [
    "ARCHITECTURES",
    "ARCHITECTURE_ALIASES",
    "ARCHITECTURE_REGISTRY",
    "ArchitectureRegistrationError",
    "ArchitectureRegistry",
    "UnknownArchitectureError",
    "get_architecture_spec",
    "list_architecture_specs",
    "register_architecture_alias",
    "register_architecture_spec",
    "unregister_architecture_alias",
    "unregister_architecture_spec",
]
