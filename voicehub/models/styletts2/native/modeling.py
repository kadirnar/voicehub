"""Native assembly of the released StyleTTS 2 deployable graph."""

from __future__ import annotations

from typing import Any

from torch import nn

from voicehub.models.kokoro.native.albert import KokoroAlbertModel
from voicehub.models.styletts2.native.configuration import StyleTTS2ArchitectureConfig
from voicehub.models.styletts2.source.styletts2.models import StyleTTS2Modules, build_model

DEPLOYABLE_STYLETTS2_COMPONENTS = (
    "bert",
    "bert_encoder",
    "decoder",
    "diffusion",
    "predictor",
    "predictor_encoder",
    "style_encoder",
    "text_encoder",
)

StyleTTS2ConfigInput = StyleTTS2ArchitectureConfig | dict[str, Any] | None


def build_styletts2_model(config: StyleTTS2ConfigInput = None, ) -> StyleTTS2Modules:
    """Build either released StyleTTS 2 decoder profile."""
    if config is None:
        config = StyleTTS2ArchitectureConfig()
    elif not isinstance(config, StyleTTS2ArchitectureConfig):
        config = StyleTTS2ArchitectureConfig.from_dict(config)
    bert = KokoroAlbertModel(config.plbert)
    model = build_model(
        config,
        text_aligner=None,
        pitch_extractor=None,
        bert=bert,
        include_discriminators=False,
    )
    actual = tuple(name for name, module in model.items() if isinstance(module, nn.Module))
    if set(actual) != set(DEPLOYABLE_STYLETTS2_COMPONENTS):
        raise RuntimeError("Native StyleTTS 2 graph has an unexpected component inventory: "
                           f"{actual!r}.")
    return model


__all__ = [
    "DEPLOYABLE_STYLETTS2_COMPONENTS",
    "StyleTTS2Modules",
    "build_styletts2_model",
]
