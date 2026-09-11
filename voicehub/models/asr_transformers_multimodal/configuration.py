"""Native compatibility dispatch for historical multimodal ASR configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from voicehub.models.asr_qwen3.configuration import Qwen3ASRConfig
from voicehub.models.asr_vibevoice.configuration import VibeVoiceASRConfig

_PROVIDER_ALIASES = {
    "asr-qwen3": "qwen3",
    "asr-vibevoice": "vibevoice",
    "qwen": "qwen3",
    "qwen3": "qwen3",
    "qwen3-asr": "qwen3",
    "vibe": "vibevoice",
    "vibevoice": "vibevoice",
    "vibevoice-asr": "vibevoice",
}
_PROVIDER_CONFIGS = {
    "qwen3": Qwen3ASRConfig,
    "vibevoice": VibeVoiceASRConfig,
}


def _normalize_provider(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("`provider` and `model_type` must be non-empty strings.")
    normalized = value.strip().lower().replace("_", "-")
    try:
        return _PROVIDER_ALIASES[normalized]
    except KeyError as error:
        choices = ", ".join(sorted(_PROVIDER_CONFIGS))
        raise ValueError(
            f"Unsupported native multimodal ASR provider {value!r}; "
            f"choose one of: {choices}.") from error


def _provider_from_source(value: Any) -> str | None:
    if not isinstance(value, (str, Path)):
        return None
    normalized = str(value).strip().lower().replace("_", "-")
    if "vibevoice" in normalized:
        return "vibevoice"
    if "qwen3" in normalized or "qwen-3" in normalized:
        return "qwen3"
    return None


def _resolve_provider(
    *,
    provider: Any = None,
    model_type: Any = None,
    source: Any = None,
) -> str:
    """Resolve one explicit native multimodal family without guessing."""
    values = [
        candidate for candidate in (
            _normalize_provider(provider),
            _normalize_provider(model_type),
            _provider_from_source(source),
        ) if candidate is not None
    ]
    if not values:
        raise ValueError(
            "The generic multimodal ASR compatibility API requires "
            "`provider='qwen3'` or `provider='vibevoice'`, a matching "
            "`model_type`, or an identifiable checkpoint name.")
    if any(candidate != values[0] for candidate in values[1:]):
        raise ValueError("Conflicting multimodal ASR provider hints were supplied.")
    return values[0]


def _load_config_class(provider: str):
    return _PROVIDER_CONFIGS[provider]


class MultimodalTransformersASRConfig:
    """Compatibility factory returning a dedicated native ASR config.

    A single generic configuration cannot describe both Qwen3-ASR's
    encoder-decoder graph and VibeVoice-ASR's causal multimodal graph.
    The historical name therefore remains as an explicit factory instead
    of pretending those architectures share one executable base class.
    """

    def __new__(
        cls,
        *,
        provider: str | None = None,
        model_type: str | None = None,
        **kwargs: Any,
    ):
        resolved = _resolve_provider(
            provider=provider,
            model_type=model_type,
            source=kwargs.get("name_or_path"),
        )
        return _load_config_class(resolved)(**kwargs)

    @classmethod
    def from_dict(cls, values: dict[str, Any]):
        if not isinstance(values, dict):
            raise TypeError("`values` must be a dictionary.")
        return cls(**values)


__all__ = [
    "MultimodalTransformersASRConfig",
    "Qwen3ASRConfig",
    "VibeVoiceASRConfig",
]
