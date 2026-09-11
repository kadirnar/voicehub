"""Strict native loading and export for MOSS Audio Tokenizer v1/v2."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch
from torch import Tensor, nn

from voicehub.checkpointing import SafeTensorReader, ShardedSafeTensorReader, save_safetensors
from voicehub.checkpointing.errors import CheckpointCompatibilityError, CheckpointIntegrityError
from voicehub.hub import read_json_file, write_json_file
from voicehub.models.mosstts.native.artifacts import MossCodecArtifacts, resolve_moss_codec_artifacts
from voicehub.models.mosstts.native.checkpoint import (
    MossCheckpointReport,
    inspect_mosstts_checkpoint,
    write_mosstts_license_files,
)
from voicehub.models.mosstts.native.codec import MossAudioCodecConfig
from voicehub.models.mosstts.native.codec_configuration import MossAudioTokenizerConfig
from voicehub.models.mosstts.native.metadata import MOSS_CODEC_CHECKPOINTS, OPENMOSS_LICENSE

_FLOAT_DTYPES = frozenset({"BF16", "F16", "F32", "F64"})
_NATIVE_FORMAT = "voicehub-moss-audio-tokenizer-v1"


class _TensorReader(Protocol):

    def keys(self) -> tuple[str, ...]:
        ...

    def tensor_shape(self, name: str) -> tuple[int, ...]:
        ...

    def get_tensor(self, name: str) -> Tensor:
        ...


@dataclass(frozen=True, slots=True)
class LoadedMossAudioTokenizer:
    """A fully validated native codec graph and its immutable provenance."""

    model: nn.Module
    architecture_config: MossAudioTokenizerConfig
    codec_config: MossAudioCodecConfig
    artifacts: MossCodecArtifacts
    report: MossCheckpointReport


def _open_reader(path: Path):
    if path.name.endswith(".safetensors.index.json"):
        return ShardedSafeTensorReader(path)
    if path.suffix == ".safetensors":
        return SafeTensorReader(path)
    raise ValueError("MOSS codec checkpoints must use Safetensors.")


def _record(reader: _TensorReader, name: str):
    record = getattr(reader, "record", None)
    if callable(record):
        return record(name)
    if isinstance(reader, ShardedSafeTensorReader):
        shard = reader.index.shard_path(name)
        return reader._reader(shard).record(name)
    raise TypeError(f"{type(reader).__name__} does not expose tensor metadata.")


def build_moss_audio_tokenizer(
    config: MossAudioTokenizerConfig,
    codec_config: MossAudioCodecConfig,
    *,
    device: str | torch.device = "meta",
) -> nn.Module:
    """Build the checkpoint-exact v1 or v2 graph on the requested device."""
    if not isinstance(config, MossAudioTokenizerConfig):
        raise TypeError("`config` must be MossAudioTokenizerConfig.")
    if not isinstance(codec_config, MossAudioCodecConfig):
        raise TypeError("`codec_config` must be MossAudioCodecConfig.")
    target = torch.device(device)
    with target:
        if codec_config.version == 1:
            from voicehub.models.mosstts.native.codec_modeling_v1 import MossAudioTokenizerModel
        else:
            from voicehub.models.mosstts.native.codec_modeling import MossAudioTokenizerModel
        model = MossAudioTokenizerModel(config)
    return model


def _official_facts(
    source: str | None,
    revision: str | None,
) -> Mapping[str, object] | None:
    if source is None or source not in MOSS_CODEC_CHECKPOINTS:
        return None
    facts = MOSS_CODEC_CHECKPOINTS[source]
    expected_revision = str(facts["revision"])
    if revision is None or revision.lower() != expected_revision.lower():
        raise CheckpointIntegrityError(
            f"Official MOSS codec repository {source!r} must resolve to "
            f"audited revision {expected_revision}; found {revision!r}.")
    return facts


def _validate_official_report(
    report: MossCheckpointReport,
    facts: Mapping[str, object] | None,
) -> None:
    if facts is None:
        return
    expected = {
        "tensor_count": int(facts["tensors"]),
        "parameter_count": int(facts["parameters"]),
        "tensor_bytes": int(facts["tensor_bytes"]),
        "header_fingerprint": str(facts["header_fingerprint"]),
        "dtype_names": (str(facts["dtype"]), ),
    }
    actual = {
        "tensor_count": report.tensor_count,
        "parameter_count": report.parameter_count,
        "tensor_bytes": report.tensor_bytes,
        "header_fingerprint": report.header_fingerprint,
        "dtype_names": report.dtype_names,
    }
    if actual != expected:
        raise CheckpointIntegrityError(
            "Official MOSS codec header does not match the audited inventory: "
            f"expected={expected!r}, actual={actual!r}.")


def _validate_layout(
    model: nn.Module,
    reader: _TensorReader,
) -> tuple[str, ...]:
    expected = {name: tuple(value.shape) for name, value in model.state_dict(keep_vars=True).items()}
    available = set(reader.keys())
    missing = sorted(set(expected) - available)
    unexpected = sorted(available - set(expected))
    mismatched = sorted((name, reader.tensor_shape(name), expected[name])
                        for name in set(expected) & available if reader.tensor_shape(name) != expected[name])
    unsupported_dtypes = sorted({
        _record(reader, name).dtype
        for name in available if _record(reader, name).dtype not in _FLOAT_DTYPES
    })
    if missing or unexpected or mismatched or unsupported_dtypes:
        raise CheckpointCompatibilityError(
            "MOSS codec checkpoint is incompatible: "
            f"missing={missing[:12]!r}, "
            f"unexpected={unexpected[:12]!r}, "
            f"shape_mismatches={mismatched[:12]!r}, "
            f"unsupported_dtypes={unsupported_dtypes!r}.")
    return tuple(sorted(expected))


def load_moss_audio_tokenizer_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    device: str | torch.device,
    encoder_decoder_dtype: torch.dtype | None = None,
    source: str | None = None,
    revision: str | None = None,
) -> MossCheckpointReport:
    """Validate the full header and assign every tensor without pickle."""
    if not isinstance(model, nn.Module) or not isinstance(
            getattr(model, "config", None),
            MossAudioTokenizerConfig,
    ):
        raise TypeError("`model` must be a native MOSS Audio Tokenizer graph.")
    if encoder_decoder_dtype is not None and (not isinstance(encoder_decoder_dtype, torch.dtype) or
                                              not encoder_decoder_dtype.is_floating_point):
        raise TypeError("Codec encoder/decoder dtype must be floating point or None.")

    report = inspect_mosstts_checkpoint(path)
    _validate_official_report(report, _official_facts(source, revision))
    with _open_reader(report.path) as reader:
        names = _validate_layout(model, reader)
        with torch.no_grad():
            for name in names:
                value = reader.get_tensor(name)
                target_dtype = value.dtype
                is_encoder_or_decoder = name.startswith(("encoder.", "decoder."))
                if (encoder_decoder_dtype is not None and value.is_floating_point() and
                        is_encoder_or_decoder):
                    target_dtype = encoder_decoder_dtype
                model.load_state_dict(
                    {name: value.to(
                        device=device,
                        dtype=target_dtype,
                    )},
                    strict=False,
                    assign=True,
                )
                del value

    remaining = [name for name, value in model.state_dict().items() if value.device.type == "meta"]
    if remaining:
        raise CheckpointCompatibilityError(
            "MOSS codec assignment left meta tensors unresolved: " + ", ".join(remaining[:12]) + ".")
    return report


def load_moss_audio_tokenizer(
    source: str | Path,
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
    device: str | torch.device = "cpu",
    encoder_decoder_dtype: torch.dtype | None = None,
) -> LoadedMossAudioTokenizer:
    """Resolve, build, and strictly load a complete native codec."""
    artifacts = resolve_moss_codec_artifacts(
        source,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    raw_config = read_json_file(artifacts.config)
    architecture_config = MossAudioTokenizerConfig.from_dict(raw_config)
    version = raw_config.get("voicehub_codec_version")
    codec_config = MossAudioCodecConfig.from_dict(
        raw_config,
        version=int(version) if version is not None else None,
    )
    model = build_moss_audio_tokenizer(
        architecture_config,
        codec_config,
        device="meta",
    )
    report = load_moss_audio_tokenizer_checkpoint(
        model,
        artifacts.checkpoint,
        device=device,
        encoder_decoder_dtype=encoder_decoder_dtype,
        source=artifacts.source,
        revision=artifacts.revision,
    )
    model.eval()
    return LoadedMossAudioTokenizer(
        model=model,
        architecture_config=architecture_config,
        codec_config=codec_config,
        artifacts=artifacts,
        report=report,
    )


def export_moss_audio_tokenizer_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    state_override: Mapping[str, Tensor] | None = None,
) -> Path:
    """Export one exact, inference-reloadable codec state dictionary."""
    expected_state = model.state_dict()
    state = dict(expected_state) if state_override is None else dict(state_override)
    expected = set(expected_state)
    actual = set(state)
    mismatched = sorted(
        name for name in expected & actual if tuple(state[name].shape) != tuple(expected_state[name].shape))
    if expected != actual or mismatched:
        raise ValueError(
            "MOSS codec export is incomplete: "
            f"missing={sorted(expected - actual)[:12]!r}, "
            f"unexpected={sorted(actual - expected)[:12]!r}, "
            f"shape_mismatches={mismatched[:12]!r}.")
    if any(value.device.type == "meta" for value in state.values()):
        raise ValueError("MOSS codec cannot export unresolved meta tensors.")
    return save_safetensors(
        {
            name: value.detach()
            for name, value in state.items()
        },
        path,
        metadata={
            "format": _NATIVE_FORMAT,
            "license": OPENMOSS_LICENSE,
            "producer": "voicehub",
        },
    ).resolve()


def save_moss_audio_tokenizer_pretrained(
    model: nn.Module,
    directory: str | Path,
    *,
    state_override: Mapping[str, Tensor] | None = None,
) -> Path:
    """Save a portable native codec checkpoint and full graph config."""
    if not isinstance(getattr(model, "config", None), MossAudioTokenizerConfig):
        raise TypeError("`model` must be a native MOSS Audio Tokenizer graph.")
    destination = Path(directory).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    export_moss_audio_tokenizer_checkpoint(
        model,
        destination / "model.safetensors",
        state_override=state_override,
    )
    write_json_file(destination / "config.json", model.config.to_dict())
    write_mosstts_license_files(destination)
    return destination.resolve()


__all__ = [
    "LoadedMossAudioTokenizer",
    "build_moss_audio_tokenizer",
    "export_moss_audio_tokenizer_checkpoint",
    "load_moss_audio_tokenizer",
    "load_moss_audio_tokenizer_checkpoint",
    "save_moss_audio_tokenizer_pretrained",
]
