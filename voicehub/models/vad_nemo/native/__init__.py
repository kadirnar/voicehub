"""Native MarbleNet VAD architecture with lazy public component imports."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "MarbleNetVADConfig": (
        "voicehub.models.vad_nemo.native.configuration",
        "MarbleNetVADConfig",
    ),
    "MarbleNetVADModel": (
        "voicehub.models.vad_nemo.native.modeling",
        "MarbleNetVADModel",
    ),
    "MarbleNetVADOutput": (
        "voicehub.models.vad_nemo.native.modeling",
        "MarbleNetVADOutput",
    ),
    "MarbleNetVADSafeTensorsCheckpointAdapter": (
        "voicehub.models.vad_nemo.native.checkpoint",
        "MarbleNetVADSafeTensorsCheckpointAdapter",
    ),
    "convert_nemo_marblenet_checkpoint": (
        "voicehub.models.vad_nemo.native.checkpoint",
        "convert_nemo_marblenet_checkpoint",
    ),
    "marblenet_vad_loss": (
        "voicehub.models.vad_nemo.native.objective",
        "marblenet_vad_loss",
    ),
}


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_EXPORTS))


__all__ = sorted(_EXPORTS)
