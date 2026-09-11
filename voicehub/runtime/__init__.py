"""Lazy access to internal native runtime metadata and contracts."""

from importlib import import_module

_EXPORTS = {
    "ARCHITECTURES": "voicehub.runtime.registry",
    "ARCHITECTURE_ALIASES": "voicehub.runtime.registry",
    "ARCHITECTURE_REGISTRY": "voicehub.runtime.registry",
    "ArchitectureCapabilities": "voicehub.runtime.specifications",
    "ArchitectureCompatibilityError": "voicehub.runtime.runtime",
    "ArchitectureError": "voicehub.runtime.specifications",
    "ArchitectureRegistrationError": "voicehub.runtime.registry",
    "ArchitectureRegistry": "voicehub.runtime.registry",
    "ArchitectureSpec": "voicehub.runtime.specifications",
    "BUILTIN_ARCHITECTURE_REGISTRARS": "voicehub.runtime.catalog",
    "CompatibilityIssue": "voicehub.runtime.runtime",
    "ComponentResolutionError": "voicehub.runtime.specifications",
    "LazyComponent": "voicehub.runtime.specifications",
    "LazyComponentRef": "voicehub.runtime.specifications",
    "LazyComponentReference": "voicehub.runtime.specifications",
    "RuntimeBundle": "voicehub.runtime.runtime",
    "RuntimeRequest": "voicehub.runtime.runtime",
    "RuntimeRequirements": "voicehub.runtime.runtime",
    "UnknownArchitectureError": "voicehub.runtime.registry",
    "ensure_compatible": "voicehub.runtime.runtime",
    "get_architecture_spec": "voicehub.runtime.registry",
    "inspect_compatibility": "voicehub.runtime.runtime",
    "list_architecture_specs": "voicehub.runtime.registry",
    "normalize_architecture_id": "voicehub.runtime.specifications",
    "register_architecture_alias": "voicehub.runtime.registry",
    "register_architecture_spec": "voicehub.runtime.registry",
    "register_builtin_architectures": "voicehub.runtime.catalog",
    "unregister_architecture_alias": "voicehub.runtime.registry",
    "unregister_architecture_spec": "voicehub.runtime.registry",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(name)
    if name in {"ARCHITECTURES", "ARCHITECTURE_REGISTRY", "ARCHITECTURE_ALIASES"}:
        from voicehub.runtime.registry import _ensure_builtins

        _ensure_builtins()
    value = getattr(import_module(module), name)
    globals()[name] = value
    return value
