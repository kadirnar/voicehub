"""Discoverable optimization surface for registered VITS-family models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from voicehub.kernel_operations import VITS_FUSED_ADD_TANH_SIGMOID
from voicehub.tasks import SpeechTask

VITS_FAMILY_FEATURE = "vits-family"
VITS_WAVENET_GATE_FEATURE = "vits-wavenet-gate"


class VITSArchitectureKind(str, Enum):
    """How a registered public model incorporates the VITS architecture."""

    CLASSIC = "classic"
    VITS2 = "vits2"
    HYBRID_ACOUSTIC = "hybrid-acoustic"
    CONVERTER = "converter"

    @classmethod
    def coerce(cls, value: VITSArchitectureKind | str) -> VITSArchitectureKind:
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise TypeError("VITS architecture kinds must be strings.")
        try:
            return cls(value.strip().lower())
        except ValueError as error:
            choices = ", ".join(item.value for item in cls)
            raise ValueError(f"Unknown VITS architecture kind {value!r}; expected: {choices}.") from error


@dataclass(frozen=True, slots=True)
class VITSModelOptimizationSupport:
    """One public model's family role and shared optimization contract."""

    model_type: str
    architecture: str
    kind: VITSArchitectureKind
    training: bool
    distributed_training: bool
    optimization_passes: tuple[str, ...]
    kernel_operations: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "model_type": self.model_type,
            "architecture": self.architecture,
            "kind": self.kind.value,
            "training": self.training,
            "distributed_training": self.distributed_training,
            "optimization_passes": list(self.optimization_passes),
            "kernel_operations": list(self.kernel_operations),
        }


def list_vits_model_optimization_support() -> tuple[VITSModelOptimizationSupport, ...]:
    """List registered models marked by architecture traits, not name
    checks."""
    from voicehub.registry import list_model_specs
    from voicehub.runtime import get_architecture_spec

    output = []
    for model in list_model_specs(task=SpeechTask.TEXT_TO_SPEECH):
        if model.architecture is None:
            continue
        architecture = get_architecture_spec(model.architecture)
        capabilities = architecture.capabilities
        if not capabilities.has_feature(VITS_FAMILY_FEATURE):
            continue
        raw_kind = architecture.metadata.get("vits_architecture_kind")
        if not isinstance(raw_kind, str):
            raise TypeError(
                f"VITS-family architecture {architecture.architecture_id!r} "
                "does not declare metadata.vits_architecture_kind.")
        kernel_operations = ((VITS_FUSED_ADD_TANH_SIGMOID, )
                             if capabilities.has_feature(VITS_WAVENET_GATE_FEATURE) else ())
        output.append(
            VITSModelOptimizationSupport(
                model_type=model.model_type,
                architecture=architecture.architecture_id,
                kind=VITSArchitectureKind.coerce(raw_kind),
                training=capabilities.training,
                distributed_training=capabilities.distributed_training,
                optimization_passes=capabilities.optimization_passes,
                kernel_operations=kernel_operations,
            ))
    return tuple(sorted(output, key=lambda item: item.model_type))


def get_vits_model_optimization_support(model_type: str, ) -> VITSModelOptimizationSupport:
    """Return one registered VITS model's support or fail explicitly."""
    if not isinstance(model_type, str) or not model_type.strip():
        raise ValueError("`model_type` must be a non-empty string.")
    from voicehub.registry import get_model_spec

    canonical = get_model_spec(model_type).model_type
    for support in list_vits_model_optimization_support():
        if support.model_type == canonical:
            return support
    raise ValueError(f"Model {canonical!r} is not a registered VITS-family model.")


__all__ = [
    "VITS_FAMILY_FEATURE",
    "VITS_WAVENET_GATE_FEATURE",
    "VITSArchitectureKind",
    "VITSModelOptimizationSupport",
    "get_vits_model_optimization_support",
    "list_vits_model_optimization_support",
]
