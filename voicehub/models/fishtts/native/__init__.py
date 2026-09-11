"""VoiceHub-owned Fish Speech S2 architecture with lazy imports."""

from importlib import import_module

_PACKAGE = "voicehub.models.fishtts.native."
_EXPORTS = {
    "FishCodecConfig": _PACKAGE + "configuration",
    "FishModifiedDAC": _PACKAGE + "codec",
    "FishS2Config": _PACKAGE + "configuration",
    "FishS2ForConditionalGeneration": _PACKAGE + "modeling",
    "FishS2Runtime": _PACKAGE + "runtime",
    "FishTokenizer": _PACKAGE + "tokenization",
    "convert_legacy_fish_codec": _PACKAGE + "checkpoint",
    "export_fish_codec_checkpoint": _PACKAGE + "checkpoint",
    "export_fish_semantic_checkpoint": _PACKAGE + "checkpoint",
    "generate_fish_codes": _PACKAGE + "sampling",
    "load_fish_codec_checkpoint": _PACKAGE + "checkpoint",
    "load_fish_semantic_checkpoint": _PACKAGE + "checkpoint",
    "write_fish_license_files": _PACKAGE + "checkpoint",
}


def __getattr__(name: str):
    try:
        module_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_EXPORTS))


__all__ = sorted(_EXPORTS)
