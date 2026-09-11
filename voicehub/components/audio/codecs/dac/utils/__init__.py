"""Native checkpoint resolution helpers for the bundled DAC runtime."""

from __future__ import annotations

from pathlib import Path

import torch

from voicehub.models.dac.native.checkpoint import (
    DESCRIPT_DAC_44KHZ_REVISION,
    HuggingFaceDacCheckpointAdapter,
)
from voicehub.models.dac.native.configuration import DacConfig
from voicehub.models.dac.native.modeling import DacModel
from voicehub.checkpointing import SafeTensorReader
from voicehub.components.audio.codecs.dac.model.dac import DAC
from voicehub.hub import read_json_file, resolve_pretrained_file

_MODEL_REPOSITORIES = {
    "16khz": (
        "descript/dac_16khz",
        "7c2fc5e759f1f501aefc6e7a0265cc57f5d17ba7",
    ),
    "24khz": (
        "descript/dac_24khz",
        "6ba020b5ba7d9d8076fb90db7e67f27e31980f6e",
    ),
    "44khz": (
        "descript/dac_44khz",
        DESCRIPT_DAC_44KHZ_REVISION,
    ),
}


def _model_source(
    model_type: str,
    *,
    revision: str,
) -> tuple[Path, Path]:
    try:
        repository, pinned_revision = _MODEL_REPOSITORIES[model_type]
    except KeyError as error:
        choices = ", ".join(sorted(_MODEL_REPOSITORIES))
        raise ValueError(f"`model_type` must be one of: {choices}.") from error
    resolved_revision = pinned_revision if revision == "latest" else revision
    checkpoint = resolve_pretrained_file(
        repository,
        "model.safetensors",
        revision=resolved_revision,
    )
    configuration = resolve_pretrained_file(
        repository,
        "config.json",
        revision=resolved_revision,
    )
    return checkpoint, configuration


def download(
    model_type: str = "44khz",
    model_bitrate: str = "8kbps",
    tag: str = "latest",
) -> Path:
    """Resolve a pinned, safe DAC checkpoint into VoiceHub's Hub cache."""
    normalized_type = str(model_type).strip().lower()
    if model_bitrate != "8kbps":
        raise ValueError(
            "The native Safetensors repositories expose the 8 kbps DAC "
            "variants. Supply an explicit converted checkpoint for another "
            "bitrate."
        )
    if not isinstance(tag, str) or not tag.strip():
        raise ValueError("`tag` must be a non-empty revision string.")
    checkpoint, _ = _model_source(
        normalized_type,
        revision=tag.strip(),
    )
    return checkpoint


def _safetensors_source(
    path: str | Path,
) -> tuple[Path, Path]:
    source = Path(path).expanduser()
    if source.is_dir():
        checkpoint = source / "model.safetensors"
        configuration = source / "config.json"
    else:
        checkpoint = source
        configuration = source.with_name("config.json")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"DAC checkpoint was not found: {checkpoint}.")
    if not configuration.is_file():
        raise FileNotFoundError(
            f"DAC configuration was not found: {configuration}."
        )
    return checkpoint, configuration


def _load_safetensors(checkpoint: Path, configuration: Path) -> DacModel:
    values = read_json_file(configuration)
    config = DacConfig.from_dict(values)
    with torch.device("meta"):
        model = DacModel(config)
    with SafeTensorReader(checkpoint) as reader:
        HuggingFaceDacCheckpointAdapter().load_assign(
            model,
            reader,
            values,
            strict=True,
        )
    return model


def load_model(
    model_type: str = "44khz",
    model_bitrate: str = "8kbps",
    tag: str = "latest",
    load_path: str | Path | None = None,
) -> DAC:
    """Load a native DAC graph from safe Hub or local artifacts."""
    if load_path:
        checkpoint, configuration = _safetensors_source(load_path)
    else:
        if model_bitrate != "8kbps":
            raise ValueError(
                "The native Hub loader supports 8 kbps DAC checkpoints."
            )
        checkpoint, configuration = _model_source(
            str(model_type).strip().lower(),
            revision=tag,
        )
    return _load_safetensors(checkpoint, configuration)


__all__ = [
    "download",
    "load_model",
]
