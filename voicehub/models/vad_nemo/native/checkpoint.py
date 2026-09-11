"""Strict one-time conversion for official NeMo MarbleNet VAD artifacts."""

from __future__ import annotations

import hashlib
import io
import tarfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicehub.checkpointing import CheckpointAdapter, CopyTensor, SafeTensorReader, TensorPlan, save_safetensors
from voicehub.hub import write_json_file
from voicehub.models.vad_nemo.native.configuration import MarbleNetVADConfig
from voicehub.models.vad_nemo.native.metadata import (
    MARBLENET_VAD_CONFIG_SHA256,
    MARBLENET_VAD_SHA256,
    MARBLENET_VAD_TENSOR_FINGERPRINT,
    MARBLENET_VAD_WEIGHTS_SHA256,
)

NATIVE_MARBLENET_VAD_FORMAT = "voicehub-marblenet-vad-v1"
NATIVE_MARBLENET_VAD_FILENAME = "model.safetensors"
_MAX_ARCHIVE_MEMBER_BYTES = 16 * 1024 * 1024
_ConfigInput = MarbleNetVADConfig | Mapping[str, Any] | None
_TensorShapes = dict[str, tuple[int, ...]]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def tensor_inventory_fingerprint(tensors: Mapping[str, Any]) -> str:
    """Hash names, portable dtypes, and shapes without tensor values."""
    rows = []
    for name, tensor in sorted(tensors.items()):
        dtype = {
            "torch.float32": "F32",
            "torch.int64": "I64",
        }.get(str(tensor.dtype), str(tensor.dtype))
        shape = "x".join(str(item) for item in tensor.shape)
        rows.append(f"{name}|{dtype}|{shape}")
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def native_marblenet_vad_tensor_shapes(config: _ConfigInput = None) -> _TensorShapes:
    from voicehub.models.vad_nemo.native.modeling import MarbleNetVADModel

    model = MarbleNetVADModel(MarbleNetVADConfig.coerce(config or {}))
    return {name: tuple(tensor.shape) for name, tensor in model.state_dict().items()}


class MarbleNetVADSafeTensorsCheckpointAdapter(CheckpointAdapter):
    """Identity-map a validated native MarbleNet Safetensors checkpoint."""

    architecture_id = "marblenet-vad"
    adapter_id = "voicehub-marblenet-vad-safetensors"
    adapter_version = "1"

    def probe(
        self,
        files: tuple[Path, ...],
        config: Mapping[str, Any],
    ) -> bool:
        try:
            MarbleNetVADConfig.from_dict(config)
        except (TypeError, ValueError):
            return False
        return any(path.suffix == ".safetensors" for path in files)

    def tensor_plan(self, config: Mapping[str, Any]) -> TensorPlan:
        shapes = native_marblenet_vad_tensor_shapes(config)
        return TensorPlan(rules=tuple(CopyTensor(name, name) for name in sorted(shapes)), )


def _read_nemo_members(source: Path) -> tuple[bytes, bytes]:
    try:
        archive = tarfile.open(source, mode="r:*")
    except (tarfile.TarError, OSError) as error:
        raise ValueError(f"Invalid NeMo archive: {source}.") from error
    with archive:
        selected: dict[str, tarfile.TarInfo] = {}
        for member in archive.getmembers():
            if member.issym() or member.islnk() or member.isdev():
                raise ValueError("NeMo conversion refuses links and device entries.")
            basename = Path(member.name).name
            if basename not in {"model_config.yaml", "model_weights.ckpt"}:
                continue
            if not member.isfile():
                raise ValueError(f"NeMo member {member.name!r} is not a regular file.")
            if member.size <= 0 or member.size > _MAX_ARCHIVE_MEMBER_BYTES:
                raise ValueError(f"NeMo member {member.name!r} has an unsafe size.")
            if basename in selected:
                raise ValueError(f"NeMo archive contains duplicate {basename!r}.")
            selected[basename] = member
        missing = {"model_config.yaml", "model_weights.ckpt"} - set(selected)
        if missing:
            raise ValueError(f"NeMo archive is missing {sorted(missing)!r}.")
        values = []
        for name in ("model_config.yaml", "model_weights.ckpt"):
            stream = archive.extractfile(selected[name])
            if stream is None:
                raise ValueError(f"Could not read NeMo member {name!r}.")
            payload = stream.read(_MAX_ARCHIVE_MEMBER_BYTES + 1)
            if len(payload) != selected[name].size:
                raise ValueError(f"NeMo member {name!r} has inconsistent size.")
            values.append(payload)
        return values[0], values[1]


def _load_restricted_state(payload: bytes | Path) -> Mapping[str, Any]:
    import torch

    source: Any = payload if isinstance(payload, Path) else io.BytesIO(payload)
    try:
        value = torch.load(
            source,
            map_location="cpu",
            weights_only=True,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError("Could not read the restricted NeMo tensor state.") from error
    if not isinstance(value, Mapping):
        raise TypeError("NeMo checkpoint root must be a mapping.")
    state = value.get("state_dict", value)
    if not isinstance(state, Mapping) or not state:
        raise ValueError("NeMo checkpoint must contain a non-empty tensor state.")
    if any(not isinstance(name, str) or not isinstance(tensor, torch.Tensor)
           for name, tensor in state.items()):
        raise TypeError("NeMo state must map string names to tensors only.")
    return state


def convert_nemo_marblenet_checkpoint(
    source: str | Path,
    destination: str | Path,
    *,
    trust_pickle_checkpoint: bool = False,
    expected_sha256: str | None = None,
) -> Path:
    """Convert a reviewed `.nemo`/`.ckpt` artifact to native Safetensors."""
    if trust_pickle_checkpoint is not True:
        raise ValueError(
            "NeMo publishes a pickle-based checkpoint. Review the artifact "
            "origin, then pass `trust_pickle_checkpoint=True` for this "
            "one-time restricted conversion.")
    source_path = Path(source).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"NeMo checkpoint was not found: {source_path}.")
    if source_path.suffix.lower() not in {".nemo", ".ckpt"}:
        raise ValueError("NeMo conversion accepts only `.nemo` or `.ckpt` files.")
    actual_sha = _file_sha256(source_path)
    if expected_sha256 is not None and actual_sha.lower() != expected_sha256.lower():
        raise ValueError(
            "NeMo checkpoint SHA-256 mismatch: "
            f"expected {expected_sha256}, found {actual_sha}.")

    config_sha = None
    weights_sha = None
    if source_path.suffix.lower() == ".nemo":
        config_payload, weights_payload = _read_nemo_members(source_path)
        config_sha = _bytes_sha256(config_payload)
        weights_sha = _bytes_sha256(weights_payload)
        if actual_sha == MARBLENET_VAD_SHA256:
            if config_sha != MARBLENET_VAD_CONFIG_SHA256:
                raise ValueError("Official NeMo model_config.yaml digest mismatch.")
            if weights_sha != MARBLENET_VAD_WEIGHTS_SHA256:
                raise ValueError("Official NeMo model_weights.ckpt digest mismatch.")
        state = _load_restricted_state(weights_payload)
    else:
        state = _load_restricted_state(source_path)

    config = MarbleNetVADConfig()
    expected_shapes = native_marblenet_vad_tensor_shapes(config)
    expected_names = set(expected_shapes)
    source_names = set(state)
    if source_names != expected_names:
        raise ValueError(
            "NeMo tensor namespace is incompatible with multilingual "
            "Frame-VAD MarbleNet "
            f"(missing={sorted(expected_names - source_names)}, "
            f"unexpected={sorted(source_names - expected_names)}).")
    mismatches = {
        name: (tuple(state[name].shape), expected_shapes[name])
        for name in expected_names if tuple(state[name].shape) != expected_shapes[name]
    }
    if mismatches:
        raise ValueError(f"NeMo tensor shape mismatch: {mismatches}.")
    inventory = tensor_inventory_fingerprint(state)
    if actual_sha == MARBLENET_VAD_SHA256 and inventory != MARBLENET_VAD_TENSOR_FINGERPRINT:
        raise ValueError("Official NeMo tensor inventory fingerprint mismatch.")

    output = Path(destination).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    safe_path = output / NATIVE_MARBLENET_VAD_FILENAME
    save_safetensors(
        {name: state[name].detach().cpu().contiguous()
         for name in sorted(expected_names)},
        safe_path,
        metadata={
            "architecture": "marblenet-vad",
            "format": NATIVE_MARBLENET_VAD_FORMAT,
            "source_sha256": actual_sha,
            "tensor_fingerprint": inventory,
        },
    )
    values = config.to_dict()
    values.update({
        "checkpoint_format": NATIVE_MARBLENET_VAD_FORMAT,
        "source_checkpoint_name": source_path.name,
        "source_checkpoint_sha256": actual_sha,
        "source_config_sha256": config_sha,
        "source_weights_sha256": weights_sha,
        "source_tensor_fingerprint": inventory,
    })
    write_json_file(output / "config.json", values)

    from voicehub.models.vad_nemo.native.modeling import MarbleNetVADModel

    with SafeTensorReader(safe_path) as reader:
        MarbleNetVADSafeTensorsCheckpointAdapter().load_streaming(
            MarbleNetVADModel(config),
            reader,
            values,
            strict=True,
        )
    return output


__all__ = [
    "MarbleNetVADSafeTensorsCheckpointAdapter",
    "NATIVE_MARBLENET_VAD_FILENAME",
    "NATIVE_MARBLENET_VAD_FORMAT",
    "convert_nemo_marblenet_checkpoint",
    "native_marblenet_vad_tensor_shapes",
    "tensor_inventory_fingerprint",
]
