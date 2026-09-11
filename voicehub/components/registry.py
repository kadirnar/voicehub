"""Registry connecting reusable components to model backends."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class ComponentSpec:
    """Provenance and import metadata for one shared component."""

    name: str
    category: str
    module: str
    upstream: str
    license_id: str
    commercial_use: bool | None = True


_COMPONENT_SPECS = (
    ComponentSpec(
        "dac",
        "audio-codec",
        "voicehub.components.audio.codecs.dac",
        "https://github.com/descriptinc/descript-audio-codec",
        "MIT",
    ),
    ComponentSpec(
        "encodec",
        "audio-codec",
        "voicehub.components.audio.codecs.encodec",
        "https://github.com/facebookresearch/encodec",
        "MIT",
    ),
    ComponentSpec(
        "vocos",
        "vocoder",
        "voicehub.components.audio.vocoders.vocos",
        "https://github.com/gemelo-ai/vocos",
        "MIT",
    ),
    ComponentSpec(
        "wavmark",
        "watermarking",
        "voicehub.components.audio.watermarking.wavmark",
        "https://github.com/wavmark/wavmark",
        "MIT",
    ),
    ComponentSpec(
        "conformer",
        "neural-block",
        "voicehub.components.neural.conformer",
        "https://github.com/lucidrains/conformer",
        "MIT",
    ),
)

COMPONENT_REGISTRY: Mapping[str,
                            ComponentSpec] = MappingProxyType({spec.name: spec
                                                               for spec in _COMPONENT_SPECS})


class _ModelComponents(Mapping[str, tuple[str, ...]]):
    """Read-only live view of component links declared by model specs."""

    def __getitem__(self, model_type: str) -> tuple[str, ...]:
        from voicehub.errors import UnknownModelError
        from voicehub.registry import get_model_spec

        try:
            components = get_model_spec(model_type).components
        except (TypeError, ValueError, UnknownModelError):
            raise KeyError(model_type) from None
        if not components:
            raise KeyError(model_type)
        return components

    def __iter__(self) -> Iterator[str]:
        from voicehub.registry import list_model_specs

        return iter(tuple(sorted(spec.model_type for spec in list_model_specs() if spec.components)))

    def __len__(self) -> int:
        from voicehub.registry import list_model_specs

        return sum(bool(spec.components) for spec in list_model_specs())


MODEL_COMPONENTS: Mapping[str, tuple[str, ...]] = _ModelComponents()


def get_component_spec(name: str) -> ComponentSpec:
    """Return metadata for one reusable component."""
    try:
        return COMPONENT_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(COMPONENT_REGISTRY)
        raise KeyError(f"Unknown component {name!r}. Available components: {available}.") from exc


def list_component_specs() -> tuple[ComponentSpec, ...]:
    """Return shared components in stable display order."""
    return _COMPONENT_SPECS


def components_for_model(model_type: str) -> tuple[ComponentSpec, ...]:
    """Return shared components linked to a model backend."""
    return tuple(get_component_spec(name) for name in MODEL_COMPONENTS.get(model_type, ()))
