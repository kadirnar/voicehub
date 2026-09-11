"""Strict Safetensors loading and portable export for native MOSS-TTS."""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import Tensor, nn

from voicehub.checkpointing import SafeTensorReader, ShardedSafeTensorReader, save_safetensors
from voicehub.checkpointing.errors import CheckpointCompatibilityError, CheckpointIntegrityError
from voicehub.hub import write_json_file
from voicehub.models.mosstts.native.configuration import MossTTSConfig
from voicehub.models.mosstts.native.metadata import MOSS_TTS_CHECKPOINTS, OPENMOSS_LICENSE
from voicehub.models.mosstts.native.modeling import MossLocalV15Model, MossTTSModel

_FLOAT_DTYPES = frozenset({"BF16", "F16", "F32", "F64"})
_NATIVE_FORMAT = "voicehub-mosstts-v1"


class _TensorReader(Protocol):

    def keys(self) -> tuple[str, ...]:
        ...

    def tensor_shape(self, name: str) -> tuple[int, ...]:
        ...

    def get_tensor(self, name: str) -> Tensor:
        ...


@dataclass(frozen=True, slots=True)
class MossCheckpointReport:
    path: Path
    tensor_count: int
    parameter_count: int
    tensor_bytes: int
    header_fingerprint: str
    dtype_names: tuple[str, ...]


def _open_reader(path: Path):
    if path.name.endswith(".safetensors.index.json"):
        return ShardedSafeTensorReader(path)
    if path.suffix == ".safetensors":
        return SafeTensorReader(path)
    raise ValueError("MOSS-TTS steady-state checkpoints must use Safetensors.")


def _record(reader: _TensorReader, name: str) -> Any:
    record = getattr(reader, "record", None)
    if callable(record):
        return record(name)
    if isinstance(reader, ShardedSafeTensorReader):
        shard = reader.index.shard_path(name)
        return reader._reader(shard).record(name)
    raise TypeError(f"{type(reader).__name__} does not expose tensor metadata.")


def mosstts_header_fingerprint(reader: _TensorReader) -> str:
    """Fingerprint tensor names, dtypes, and shapes without reading
    payloads."""
    rows = []
    for name in sorted(reader.keys()):
        item = _record(reader, name)
        shape = "x".join(str(value) for value in item.shape)
        rows.append(f"{name}|{item.dtype}|{shape}")
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def inspect_mosstts_checkpoint(path: str | Path, ) -> MossCheckpointReport:
    source = Path(path).expanduser().resolve()
    with _open_reader(source) as reader:
        records = [_record(reader, name) for name in reader.keys()]
        report = MossCheckpointReport(
            path=source,
            tensor_count=len(records),
            parameter_count=sum(item.number_of_elements for item in records),
            tensor_bytes=sum(item.number_of_bytes for item in records),
            header_fingerprint=mosstts_header_fingerprint(reader),
            dtype_names=tuple(sorted({item.dtype
                                      for item in records})),
        )
    return report


def _expected_shapes(model: nn.Module) -> dict[str, tuple[int, ...]]:
    return {name: tuple(value.shape) for name, value in model.state_dict(keep_vars=True).items()}


def _official_facts(
    source: str | None,
    revision: str | None,
) -> Mapping[str, object] | None:
    if source is None or source not in MOSS_TTS_CHECKPOINTS:
        return None
    facts = MOSS_TTS_CHECKPOINTS[source]
    expected_revision = str(facts["revision"])
    if revision is None or revision.lower() != expected_revision.lower():
        raise CheckpointIntegrityError(
            f"Official MOSS-TTS repository {source!r} must resolve to audited "
            f"revision {expected_revision}; found {revision!r}.")
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
    }
    actual = {
        "tensor_count": report.tensor_count,
        "parameter_count": report.parameter_count,
        "tensor_bytes": report.tensor_bytes,
        "header_fingerprint": report.header_fingerprint,
    }
    if actual != expected:
        raise CheckpointIntegrityError(
            "Official MOSS-TTS checkpoint header does not match the audited "
            f"inventory: expected={expected!r}, actual={actual!r}.")


def _validate_layout(
    model: MossTTSModel,
    reader: _TensorReader,
) -> tuple[str, ...]:
    expected = _expected_shapes(model)
    available = set(reader.keys())
    missing = sorted(set(expected) - available)
    unexpected = sorted(available - set(expected))
    mismatched = sorted((
        name,
        reader.tensor_shape(name),
        expected[name],
    ) for name in set(expected) & available if reader.tensor_shape(name) != expected[name])
    unsupported_dtypes = sorted({
        _record(reader, name).dtype
        for name in available if _record(reader, name).dtype not in _FLOAT_DTYPES
    })
    if missing or unexpected or mismatched or unsupported_dtypes:
        raise CheckpointCompatibilityError(
            "MOSS-TTS checkpoint is incompatible: "
            f"missing={missing[:12]!r}, "
            f"unexpected={unexpected[:12]!r}, "
            f"shape_mismatches={mismatched[:12]!r}, "
            f"unsupported_dtypes={unsupported_dtypes!r}.")
    return tuple(sorted(expected))


def _local_v15_tied_pairs(model: MossLocalV15Model, ) -> tuple[tuple[str, str], ...]:
    return (
        (
            "transformer.embed_tokens.weight",
            "text_lm_head.weight",
        ),
        *tuple((
            f"audio_embeddings.{index}.weight",
            f"audio_lm_heads.{index}.weight",
        ) for index in range(model.config.n_vq)),
    )


def _validate_checkpoint_ties(
    model: MossTTSModel,
    reader: _TensorReader,
) -> None:
    """Verify every source-declared Local v1.5 tied tensor before mutation."""
    if not isinstance(model, MossLocalV15Model):
        return
    for canonical_name, alias_name in _local_v15_tied_pairs(model):
        canonical = reader.get_tensor(canonical_name)
        alias = reader.get_tensor(alias_name)
        try:
            if not torch.equal(canonical, alias):
                raise CheckpointCompatibilityError(
                    "MOSS-TTS Local v1.5 tied tensors disagree: "
                    f"{canonical_name!r} and {alias_name!r}.")
        finally:
            del canonical, alias


def validate_mosstts_tied_weights(model: MossTTSModel) -> None:
    """Check that the in-memory Local v1.5 aliases share storage."""
    if not isinstance(model, MossLocalV15Model):
        return
    parameters = dict(model.named_parameters(remove_duplicate=False))
    for canonical_name, alias_name in _local_v15_tied_pairs(model):
        canonical = parameters[canonical_name]
        alias = parameters[alias_name]
        if canonical is not alias or canonical.data_ptr() != alias.data_ptr():
            raise CheckpointCompatibilityError(
                "MOSS-TTS Local v1.5 failed to restore tied weights for "
                f"{canonical_name!r} and {alias_name!r}.")


def _materialize_rotary_buffers(
    model: MossTTSModel,
    device: str | torch.device,
) -> None:
    # RoPE frequencies are non-persistent deterministic buffers, so they do
    # not appear in the checkpoint inventory.  A meta graph must rebuild
    # them from each attention layer's validated config after parameter
    # assignment.
    from voicehub.models.causal_lm.native.modeling import CausalSelfAttention
    from voicehub.neural.rotary import RotaryEmbedding

    for module in model.modules():
        if not isinstance(module, CausalSelfAttention):
            continue
        rotary = module.rotary
        if rotary.inverse_frequency.device.type != "meta":
            continue
        replacement = RotaryEmbedding(
            rotary.dimension,
            base=module.config.rope_theta,
            scaling=module.config.rope_scaling,
            device=device,
        )
        rotary.inverse_frequency = replacement.inverse_frequency


def load_mosstts_checkpoint(
    model: MossTTSModel,
    path: str | Path,
    *,
    device: str | torch.device,
    dtype: torch.dtype | None = None,
    source: str | None = None,
    revision: str | None = None,
) -> MossCheckpointReport:
    """Validate the complete header and tied values, then assign by tensor."""
    if not isinstance(model, nn.Module) or not isinstance(
            getattr(model, "config", None),
            MossTTSConfig,
    ):
        raise TypeError("`model` must be a native MOSS-TTS graph.")
    if dtype is not None and (not isinstance(dtype, torch.dtype) or not dtype.is_floating_point):
        raise TypeError("MOSS-TTS load dtype must be floating point or None.")

    report = inspect_mosstts_checkpoint(path)
    _validate_official_report(
        report,
        _official_facts(source, revision),
    )
    with _open_reader(report.path) as reader:
        names = _validate_layout(model, reader)
        _validate_checkpoint_ties(model, reader)
        with torch.no_grad():
            for name in names:
                value = reader.get_tensor(name)
                target_dtype = (dtype if dtype is not None and value.is_floating_point() else value.dtype)
                model.load_state_dict(
                    {
                        name: value.to(
                            device=device,
                            dtype=target_dtype,
                        ),
                    },
                    strict=False,
                    assign=True,
                )
                del value

    if isinstance(model, MossLocalV15Model):
        model.tie_weights()
        validate_mosstts_tied_weights(model)
    _materialize_rotary_buffers(model, device)
    remaining = [name for name, value in model.state_dict().items() if value.device.type == "meta"]
    if remaining:
        raise CheckpointCompatibilityError(
            "MOSS-TTS checkpoint assignment left meta tensors unresolved: " + ", ".join(remaining[:12]) + ".")
    return report


def export_mosstts_checkpoint(
    model: MossTTSModel,
    path: str | Path,
    *,
    state_override: Mapping[str, Tensor] | None = None,
) -> Path:
    """Write one exact, inference-reloadable native state dictionary."""
    expected_state = model.state_dict()
    state = (dict(expected_state) if state_override is None else dict(state_override))
    expected = set(expected_state)
    actual = set(state)
    mismatched = sorted(
        name for name in expected & actual if tuple(state[name].shape) != tuple(expected_state[name].shape))
    if expected != actual or mismatched:
        raise ValueError(
            "MOSS-TTS export is incomplete: "
            f"missing={sorted(expected - actual)[:12]!r}, "
            f"unexpected={sorted(actual - expected)[:12]!r}, "
            f"shape_mismatches={mismatched[:12]!r}.")
    if any(value.device.type == "meta" for value in state.values()):
        raise ValueError("MOSS-TTS cannot export unresolved meta tensors.")
    if isinstance(model, MossLocalV15Model):
        # The portable checkpoint preserves both published names.  Validate
        # their values before the Safetensors writer materializes aliases.
        for canonical_name, alias_name in _local_v15_tied_pairs(model):
            if not torch.equal(
                    state[canonical_name].detach().cpu(),
                    state[alias_name].detach().cpu(),
            ):
                raise ValueError(
                    "MOSS-TTS export has inconsistent tied tensors: "
                    f"{canonical_name!r} and {alias_name!r}.")
    return save_safetensors(
        {
            name: value.detach()
            for name, value in state.items()
        },
        path,
        metadata={
            "architecture": model.config.variant,
            "format": _NATIVE_FORMAT,
            "license": OPENMOSS_LICENSE,
            "producer": "voicehub",
        },
    ).resolve()


def write_mosstts_license_files(directory: str | Path) -> Path:
    destination = Path(directory).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    license_source = Path(__file__).with_name("THIRD_PARTY_LICENSE")
    source_manifest = Path(__file__).with_name("SOURCE.json")
    if not license_source.is_file() or not source_manifest.is_file():
        raise FileNotFoundError("Bundled MOSS-TTS provenance/license files are missing.")
    shutil.copy2(license_source, destination / license_source.name)
    shutil.copy2(source_manifest, destination / source_manifest.name)
    return destination.resolve()


def save_mosstts_pretrained(
    model: MossTTSModel,
    directory: str | Path,
    *,
    state_override: Mapping[str, Tensor] | None = None,
) -> Path:
    destination = Path(directory).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    export_mosstts_checkpoint(
        model,
        destination / "model.safetensors",
        state_override=state_override,
    )
    write_json_file(
        destination / "config.json",
        model.config.to_dict(),
    )
    write_mosstts_license_files(destination)
    return destination.resolve()


__all__ = [
    "MossCheckpointReport",
    "export_mosstts_checkpoint",
    "inspect_mosstts_checkpoint",
    "load_mosstts_checkpoint",
    "mosstts_header_fingerprint",
    "save_mosstts_pretrained",
    "validate_mosstts_tied_weights",
    "write_mosstts_license_files",
]
