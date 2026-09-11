"""Strict PyanNet checkpoint import and one-time Lightning conversion.

Published Pyannote and Brouhaha artifacts are pickle-based Lightning
checkpoints.  Native execution never reads them.  Conversion is guarded
by an explicit trust acknowledgement, uses PyTorch's restricted
``weights_only`` unpickler with a small allowlist of inert compatibility
records, validates the complete tensor inventory, and writes Safetensors
for every subsequent load.
"""

from __future__ import annotations

import hashlib
import typing
from collections import defaultdict
from collections.abc import Mapping
from enum import IntEnum
from pathlib import Path
from typing import Any

from voicehub.checkpointing import CheckpointAdapter, CopyTensor, SafeTensorReader, TensorPlan, save_safetensors
from voicehub.hub import write_json_file
from voicehub.models.vad_pyannote.native.configuration import PyanNetConfig
from voicehub.models.vad_pyannote.native.metadata import (
    ASTEROID_FILTERBANKS_SOURCE_REVISION,
    BROUHAHA_REPOSITORY_CHECKPOINT_SHA256,
    BROUHAHA_SOURCE_REVISION,
    PYANNOTE_AUDIO_3_SOURCE_REVISION,
    PYANNOTE_BROUHAHA_REVISION,
    PYANNOTE_SEGMENTATION_3_REVISION,
    PYANNOTE_SEGMENTATION_3_SHA256,
    PYANNOTE_SEGMENTATION_REVISION,
    PYANNOTE_SEGMENTATION_SHA256,
    PYANNOTE_VAD_PIPELINE_REVISION,
)


def config_for_variant(variant: str) -> PyanNetConfig:
    normalized = str(variant).strip().lower().replace("_", "-")
    if normalized == "segmentation":
        return PyanNetConfig.segmentation()
    if normalized in {"segmentation-3", "segmentation-3.0", "powerset-segmentation"}:
        return PyanNetConfig.segmentation_3()
    if normalized == "brouhaha":
        return PyanNetConfig.brouhaha()
    raise ValueError("`variant` must be 'segmentation', 'segmentation-3.0', or "
                     "'brouhaha'.")


def native_pyannet_tensor_shapes(config: PyanNetConfig | Mapping[str, Any], ) -> dict[str, tuple[int, ...]]:
    """Return the exact persistent namespace for one graph."""
    from voicehub.models.vad_pyannote.native.modeling import PyanNet

    model = PyanNet(PyanNetConfig.coerce(config))
    return {name: tuple(tensor.shape) for name, tensor in model.state_dict().items()}


def tensor_inventory_fingerprint(
    tensor_shapes: Mapping[str, tuple[int, ...]],
    *,
    dtype: str = "F32",
) -> str:
    rows = [
        f"{name}|{dtype}|{'x'.join(str(item) for item in shape)}"
        for name, shape in sorted(tensor_shapes.items())
    ]
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


class PyanNetSafeTensorsCheckpointAdapter(CheckpointAdapter):
    """Identity-map a strictly validated native PyanNet Safetensors file."""

    architecture_id = "pyannet"
    adapter_id = "voicehub-pyannet-safetensors"
    adapter_version = "1"

    def probe(
        self,
        files: tuple[Path, ...],
        config: Mapping[str, Any],
    ) -> bool:
        try:
            PyanNetConfig.from_dict(config)
        except (TypeError, ValueError):
            return False
        return any(path.suffix == ".safetensors" for path in files)

    def tensor_plan(self, config: Mapping[str, Any]) -> TensorPlan:
        shapes = native_pyannet_tensor_shapes(config)
        return TensorPlan(rules=tuple(CopyTensor(name, name) for name in sorted(shapes)), )


class _OpaqueRecord:
    """Inert target for metadata objects accepted by ``weights_only``."""

    def __new__(cls, *args: Any, **kwargs: Any) -> _OpaqueRecord:
        del args, kwargs
        return object.__new__(cls)

    def __setstate__(self, state: Any) -> None:
        self.state = state


class _OpaqueList(list):

    def __setstate__(self, state: Any) -> None:
        self.state = state


class _OpaqueDict(dict):

    def __setstate__(self, state: Any) -> None:
        self.state = state


class _Problem(IntEnum):
    VALUE_1 = 1
    VALUE_2 = 2
    VALUE_3 = 3
    VALUE_4 = 4


class _Resolution(IntEnum):
    VALUE_1 = 1
    VALUE_2 = 2


def _restricted_safe_globals() -> list[Any]:
    import torch

    values: list[Any] = [
        typing.Any,
        list,
        dict,
        int,
        defaultdict,
        torch.torch_version.TorchVersion,
    ]
    aliases = (
        (
            _OpaqueRecord,
            "pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint",
        ),
        (
            _OpaqueRecord,
            "pytorch_lightning.callbacks.early_stopping.EarlyStopping",
        ),
        (_OpaqueList, "omegaconf.listconfig.ListConfig"),
        (_OpaqueRecord, "omegaconf.base.ContainerMetadata"),
        (_OpaqueDict, "omegaconf.dictconfig.DictConfig"),
        (_OpaqueRecord, "omegaconf.nodes.AnyNode"),
        (_OpaqueRecord, "omegaconf.base.Metadata"),
        (_OpaqueRecord, "pyannote.audio.core.model.Introspection"),
        (_OpaqueRecord, "pyannote.audio.core.task.Specifications"),
        (_Problem, "pyannote.audio.core.task.Problem"),
        (_Resolution, "pyannote.audio.core.task.Resolution"),
    )
    values.extend(aliases)
    return values


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def convert_pyannote_lightning_checkpoint(
    source: str | Path,
    destination: str | Path,
    *,
    variant: str,
    trust_pickle_checkpoint: bool = False,
    expected_sha256: str | None = None,
) -> Path:
    """Convert a reviewed Lightning artifact to a native safe directory.

    ``trust_pickle_checkpoint=True`` must be supplied on every
    conversion call.  It is deliberately not a configuration default and
    is not persisted. The restricted loader still rejects executable
    globals, but the explicit acknowledgement keeps this one-time
    boundary visible to operators.
    """
    if trust_pickle_checkpoint is not True:
        raise ValueError(
            "Pyannote publishes pickle-based Lightning checkpoints. Review "
            "the artifact origin, then pass `trust_pickle_checkpoint=True` "
            "for this one-time Safetensors conversion.")
    source_path = Path(source).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Pyannote Lightning checkpoint was not found: {source_path}.")
    if expected_sha256 is not None:
        if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64):
            raise ValueError("`expected_sha256` must be a SHA-256 hex digest.")
        actual = _file_sha256(source_path)
        if actual.lower() != expected_sha256.lower():
            raise ValueError(
                "Pyannote checkpoint SHA-256 mismatch: "
                f"expected {expected_sha256}, found {actual}.")

    import torch

    config = config_for_variant(variant)
    with torch.serialization.safe_globals(_restricted_safe_globals()):
        payload = torch.load(
            source_path,
            map_location="cpu",
            weights_only=True,
        )
    if not isinstance(payload, Mapping):
        raise TypeError("Lightning checkpoint root must be a mapping.")
    state = payload.get("state_dict")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Lightning checkpoint must contain a non-empty `state_dict`.")
    if any(not isinstance(name, str) or not isinstance(tensor, torch.Tensor)
           for name, tensor in state.items()):
        raise TypeError("Lightning `state_dict` must map string names to tensors only.")

    expected_shapes = native_pyannet_tensor_shapes(config)
    source_names = set(state)
    expected_names = set(expected_shapes)
    if source_names != expected_names:
        missing = sorted(expected_names - source_names)
        unexpected = sorted(source_names - expected_names)
        raise ValueError(
            "Lightning tensor namespace is incompatible with native PyanNet "
            f"(missing={missing}, unexpected={unexpected}).")
    mismatches = {
        name: (tuple(state[name].shape), expected_shapes[name])
        for name in expected_names if tuple(state[name].shape) != expected_shapes[name]
    }
    if mismatches:
        raise ValueError(f"Lightning checkpoint tensor shape mismatch: {mismatches}.")

    output = Path(destination).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    safe_path = output / "model.safetensors"
    source_sha = _file_sha256(source_path)
    save_safetensors(
        {name: state[name].detach().cpu().contiguous()
         for name in sorted(expected_names)},
        safe_path,
        metadata={
            "format": "voicehub-pyannet-v1",
            "source_sha256": source_sha,
            "variant": config.variant,
        },
    )
    values = config.to_dict()
    values.update({
        "checkpoint_format": "voicehub-pyannet-v1",
        "source_checkpoint_sha256": source_sha,
        "source_checkpoint_name": source_path.name,
    })
    write_json_file(output / "config.json", values)
    with SafeTensorReader(safe_path) as reader:
        adapter = PyanNetSafeTensorsCheckpointAdapter()
        from voicehub.models.vad_pyannote.native.modeling import PyanNet

        adapter.load_streaming(
            PyanNet(config),
            reader,
            values,
            strict=True,
        )
    return output


__all__ = [
    "ASTEROID_FILTERBANKS_SOURCE_REVISION",
    "BROUHAHA_REPOSITORY_CHECKPOINT_SHA256",
    "BROUHAHA_SOURCE_REVISION",
    "PYANNOTE_AUDIO_3_SOURCE_REVISION",
    "PYANNOTE_BROUHAHA_REVISION",
    "PYANNOTE_SEGMENTATION_3_REVISION",
    "PYANNOTE_SEGMENTATION_3_SHA256",
    "PYANNOTE_SEGMENTATION_REVISION",
    "PYANNOTE_SEGMENTATION_SHA256",
    "PYANNOTE_VAD_PIPELINE_REVISION",
    "PyanNetSafeTensorsCheckpointAdapter",
    "config_for_variant",
    "convert_pyannote_lightning_checkpoint",
    "native_pyannet_tensor_shapes",
    "tensor_inventory_fingerprint",
]
