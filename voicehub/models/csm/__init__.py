"""VoiceHub-native Sesame CSM model family."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from voicehub.models.csm.configuration import CSMConfig
    from voicehub.models.csm.modeling import CSMTTS, CSMForTextToSpeech
    from voicehub.models.csm.native import CSMArchitectureConfig, CSMModel

_PUBLIC_IMPORTS = {
    "CSMArchitectureConfig": (
        "voicehub.models.csm.native.configuration",
        "CSMArchitectureConfig",
    ),
    "CSMModel": (
        "voicehub.models.csm.native.modeling",
        "CSMModel",
    ),
    "CSMConfig": (
        "voicehub.models.csm.configuration",
        "CSMConfig",
    ),
    "CSMForTextToSpeech": (
        "voicehub.models.csm.modeling",
        "CSMForTextToSpeech",
    ),
    "CSMTTS": (
        "voicehub.models.csm.modeling",
        "CSMTTS",
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
