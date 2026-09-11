"""Lazy public imports for VoiceHub's native Sesame CSM architecture."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from voicehub.models.csm.native.artifacts import CSMArtifacts, resolve_csm_artifacts
    from voicehub.models.csm.native.checkpoint import (
        CSMCheckpointReport,
        export_csm_checkpoint,
        inspect_csm_checkpoint,
        load_csm_checkpoint,
        validate_csm_checkpoint,
    )
    from voicehub.models.csm.native.configuration import CSMArchitectureConfig, CSMTransformerConfig
    from voicehub.models.csm.native.modeling import CSMModel, CSMOutput
    from voicehub.models.csm.native.processing import CSMCodeSegment, CSMProcessor, CSMTextTokenizer
    from voicehub.models.csm.native.registration import (
        DEFAULT_CSM_ALIASES,
        create_csm_architecture_spec,
        register_csm_architecture,
    )
    from voicehub.models.csm.native.runtime import CSMCodec, CSMRuntime, load_csm_runtime

_PUBLIC_IMPORTS = {
    "CSMArchitectureConfig": (
        "voicehub.models.csm.native.configuration",
        "CSMArchitectureConfig",
    ),
    "CSMArtifacts": (
        "voicehub.models.csm.native.artifacts",
        "CSMArtifacts",
    ),
    "CSMCheckpointReport": (
        "voicehub.models.csm.native.checkpoint",
        "CSMCheckpointReport",
    ),
    "CSMCodeSegment": (
        "voicehub.models.csm.native.processing",
        "CSMCodeSegment",
    ),
    "CSMCodec": (
        "voicehub.models.csm.native.runtime",
        "CSMCodec",
    ),
    "CSMModel": (
        "voicehub.models.csm.native.modeling",
        "CSMModel",
    ),
    "CSMOutput": (
        "voicehub.models.csm.native.modeling",
        "CSMOutput",
    ),
    "CSMProcessor": (
        "voicehub.models.csm.native.processing",
        "CSMProcessor",
    ),
    "CSMRuntime": (
        "voicehub.models.csm.native.runtime",
        "CSMRuntime",
    ),
    "CSMTextTokenizer": (
        "voicehub.models.csm.native.processing",
        "CSMTextTokenizer",
    ),
    "CSMTransformerConfig": (
        "voicehub.models.csm.native.configuration",
        "CSMTransformerConfig",
    ),
    "DEFAULT_CSM_ALIASES": (
        "voicehub.models.csm.native.registration",
        "DEFAULT_CSM_ALIASES",
    ),
    "create_csm_architecture_spec": (
        "voicehub.models.csm.native.registration",
        "create_csm_architecture_spec",
    ),
    "export_csm_checkpoint": (
        "voicehub.models.csm.native.checkpoint",
        "export_csm_checkpoint",
    ),
    "inspect_csm_checkpoint": (
        "voicehub.models.csm.native.checkpoint",
        "inspect_csm_checkpoint",
    ),
    "load_csm_checkpoint": (
        "voicehub.models.csm.native.checkpoint",
        "load_csm_checkpoint",
    ),
    "load_csm_runtime": (
        "voicehub.models.csm.native.runtime",
        "load_csm_runtime",
    ),
    "register_csm_architecture": (
        "voicehub.models.csm.native.registration",
        "register_csm_architecture",
    ),
    "resolve_csm_artifacts": (
        "voicehub.models.csm.native.artifacts",
        "resolve_csm_artifacts",
    ),
    "validate_csm_checkpoint": (
        "voicehub.models.csm.native.checkpoint",
        "validate_csm_checkpoint",
    ),
}


def __getattr__(name: str):
    try:
        module_name, attribute = _PUBLIC_IMPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    from importlib import import_module

    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


__all__ = list(_PUBLIC_IMPORTS)
