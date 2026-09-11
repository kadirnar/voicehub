"""Strict conversion for the audited NVIDIA QuartzNet15x5 NeMo artifact."""

from __future__ import annotations

import hashlib
import io
import tarfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicehub.checkpointing import CheckpointAdapter, CopyTensor, SafeTensorReader, TensorPlan, save_safetensors
from voicehub.hub import write_json_file
from voicehub.models.asr_nemo.native.configuration import NeMoQuartzNetCTCConfig
from voicehub.models.asr_nemo.native.metadata import (
    QUARTZNET_CONFIG_SHA256,
    QUARTZNET_SHA256,
    QUARTZNET_STATE_VALUES,
    QUARTZNET_TENSOR_COUNT,
    QUARTZNET_TENSOR_FINGERPRINT,
    QUARTZNET_WEIGHTS_SHA256,
)

NATIVE_NEMO_CTC_FORMAT = "voicehub-nemo-quartznet-ctc-v1"
NATIVE_NEMO_CTC_FILENAME = "model.safetensors"
_MAX_CONFIG_BYTES = 1024 * 1024
_MAX_WEIGHTS_BYTES = 96 * 1024 * 1024
NeMoCTCConfigLike = NeMoQuartzNetCTCConfig | Mapping[str, Any]


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def tensor_inventory_fingerprint(tensors: Mapping[str, Any]) -> str:
    """Hash stable tensor names, portable dtypes, and shapes."""
    rows = []
    for name, tensor in sorted(tensors.items()):
        dtype = {
            "torch.float32": "F32",
            "torch.int64": "I64",
        }.get(str(tensor.dtype), str(tensor.dtype))
        shape = "x".join(str(item) for item in tensor.shape)
        rows.append(f"{name}|{dtype}|{shape}")
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def native_nemo_ctc_tensor_shapes(config: NeMoCTCConfigLike | None = None, ) -> dict[str, tuple[int, ...]]:
    from voicehub.models.asr_nemo.native.modeling import NeMoQuartzNetForCTC

    model = NeMoQuartzNetForCTC(NeMoQuartzNetCTCConfig.coerce(config or {}), )
    return {name: tuple(tensor.shape) for name, tensor in model.state_dict().items()}


class NeMoCTCSafeTensorsCheckpointAdapter(CheckpointAdapter):
    """Identity-map a validated native QuartzNet CTC checkpoint."""

    architecture_id = "nemo-quartznet-ctc"
    adapter_id = "voicehub-nemo-quartznet-ctc-safetensors"
    adapter_version = "1"

    def probe(
        self,
        files: tuple[Path, ...],
        config: Mapping[str, Any],
    ) -> bool:
        try:
            NeMoQuartzNetCTCConfig.from_dict(config)
        except (TypeError, ValueError):
            return False
        return any(path.suffix == ".safetensors" for path in files)

    def tensor_plan(self, config: Mapping[str, Any]) -> TensorPlan:
        shapes = native_nemo_ctc_tensor_shapes(config)
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
            limit = (_MAX_CONFIG_BYTES if basename == "model_config.yaml" else _MAX_WEIGHTS_BYTES)
            if member.size <= 0 or member.size > limit:
                raise ValueError(f"NeMo member {member.name!r} has an unsafe size.")
            if basename in selected:
                raise ValueError(f"NeMo archive contains duplicate {basename!r}.")
            selected[basename] = member
        missing = {"model_config.yaml", "model_weights.ckpt"} - set(selected)
        if missing:
            raise ValueError(f"NeMo archive is missing {sorted(missing)!r}.")
        payloads = []
        for name in ("model_config.yaml", "model_weights.ckpt"):
            member = selected[name]
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"Could not read NeMo member {name!r}.")
            payload = stream.read(member.size + 1)
            if len(payload) != member.size:
                raise ValueError(f"NeMo member {name!r} has inconsistent size.")
            payloads.append(payload)
        return payloads[0], payloads[1]


def _load_restricted_state(payload: bytes) -> Mapping[str, Any]:
    import torch

    try:
        value = torch.load(
            io.BytesIO(payload),
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


def convert_nemo_quartznet_checkpoint(
    source: str | Path,
    destination: str | Path,
    *,
    expected_sha256: str = QUARTZNET_SHA256,
) -> Path:
    """Convert the hash-pinned public `.nemo` file to native Safetensors.

    Only the exact audited archive is accepted. Arbitrary NeMo pickle
    checkpoints are deliberately outside this converter's trust
    boundary.
    """
    source_path = Path(source).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"NeMo checkpoint was not found: {source_path}.")
    if source_path.suffix.lower() != ".nemo":
        raise ValueError("QuartzNet conversion accepts only a `.nemo` archive.")
    actual_sha = file_sha256(source_path)
    if actual_sha.lower() != expected_sha256.lower():
        raise ValueError(
            "Unsupported NeMo archive. Native conversion is limited to the "
            "audited NVIDIA QuartzNet15x5 checkpoint "
            f"{expected_sha256}; found {actual_sha}.")

    config_payload, weights_payload = _read_nemo_members(source_path)
    config_sha = _bytes_sha256(config_payload)
    weights_sha = _bytes_sha256(weights_payload)
    if config_sha != QUARTZNET_CONFIG_SHA256:
        raise ValueError("Official QuartzNet model_config.yaml digest mismatch.")
    if weights_sha != QUARTZNET_WEIGHTS_SHA256:
        raise ValueError("Official QuartzNet model_weights.ckpt digest mismatch.")
    state = _load_restricted_state(weights_payload)

    config = NeMoQuartzNetCTCConfig()
    expected_shapes = native_nemo_ctc_tensor_shapes(config)
    expected_names = set(expected_shapes)
    source_names = set(state)
    if source_names != expected_names:
        raise ValueError(
            "NeMo tensor namespace is incompatible with the audited "
            "QuartzNet15x5 graph "
            f"(missing={sorted(expected_names - source_names)}, "
            f"unexpected={sorted(source_names - expected_names)}).")
    mismatches = {
        name: (tuple(state[name].shape), expected_shapes[name])
        for name in expected_names if tuple(state[name].shape) != expected_shapes[name]
    }
    if mismatches:
        raise ValueError(f"NeMo QuartzNet tensor shape mismatch: {mismatches}.")
    inventory = tensor_inventory_fingerprint(state)
    if inventory != QUARTZNET_TENSOR_FINGERPRINT:
        raise ValueError("Official QuartzNet tensor inventory fingerprint mismatch.")
    if len(state) != QUARTZNET_TENSOR_COUNT:
        raise ValueError("Official QuartzNet tensor count mismatch.")
    state_values = sum(tensor.numel() for tensor in state.values())
    if state_values != QUARTZNET_STATE_VALUES:
        raise ValueError("Official QuartzNet state-value count mismatch.")

    output = Path(destination).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    safe_path = output / NATIVE_NEMO_CTC_FILENAME
    save_safetensors(
        {name: state[name].detach().cpu().contiguous()
         for name in sorted(expected_names)},
        safe_path,
        metadata={
            "architecture": "nemo-quartznet-ctc",
            "format": NATIVE_NEMO_CTC_FORMAT,
            "source_sha256": actual_sha,
            "tensor_fingerprint": inventory,
        },
    )
    values = config.to_dict()
    values.update({
        "checkpoint_format": NATIVE_NEMO_CTC_FORMAT,
        "model_type": "asr_nemo",
        "source_checkpoint_name": source_path.name,
        "source_checkpoint_sha256": actual_sha,
        "source_config_sha256": config_sha,
        "source_weights_sha256": weights_sha,
        "source_tensor_fingerprint": inventory,
        "voicehub_provider": "asr_nemo",
    })
    write_json_file(output / "config.json", values)

    from voicehub.models.asr_nemo.native.modeling import NeMoQuartzNetForCTC

    with SafeTensorReader(safe_path) as reader:
        NeMoCTCSafeTensorsCheckpointAdapter().load_streaming(
            NeMoQuartzNetForCTC(config),
            reader,
            values,
            strict=True,
        )
    return output


__all__ = [
    "NATIVE_NEMO_CTC_FILENAME",
    "NATIVE_NEMO_CTC_FORMAT",
    "NeMoCTCSafeTensorsCheckpointAdapter",
    "convert_nemo_quartznet_checkpoint",
    "file_sha256",
    "native_nemo_ctc_tensor_shapes",
    "tensor_inventory_fingerprint",
]
