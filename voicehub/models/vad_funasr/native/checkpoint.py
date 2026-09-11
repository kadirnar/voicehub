"""Strict one-time conversion for the official pickle FSMN VAD artifact."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicehub.checkpointing import CheckpointAdapter, CopyTensor, SafeTensorReader, TensorPlan, save_safetensors
from voicehub.hub import write_json_file
from voicehub.models.vad_funasr.native.configuration import FSMNVADConfig
from voicehub.models.vad_funasr.native.frontend import parse_kaldi_cmvn
from voicehub.models.vad_funasr.native.metadata import (
    FUNASR_CMVN_SHA256,
    FUNASR_MODEL_SHA256,
    FUNASR_OFFICIAL_TENSOR_FINGERPRINT,
)

NATIVE_FSMN_VAD_FORMAT = "voicehub-fsmn-vad-v1"
NATIVE_FSMN_VAD_FILENAME = "model.safetensors"
_TensorShapes = dict[str, tuple[int, ...]]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_inventory_fingerprint(tensors: Mapping[str, Any], ) -> str:
    rows = []
    for name, tensor in sorted(tensors.items()):
        shape = tuple(tensor.shape)
        dtype = {
            "torch.float32": "F32",
            "torch.float64": "F64",
            "torch.float16": "F16",
            "torch.bfloat16": "BF16",
        }.get(str(tensor.dtype), str(tensor.dtype))
        rows.append(f"{name}|{dtype}|{'x'.join(str(item) for item in shape)}")
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def native_fsmn_vad_tensor_shapes(config: FSMNVADConfig | Mapping[str, Any] | None = None, ) -> _TensorShapes:
    from voicehub.models.vad_funasr.native.modeling import FSMNVADModel

    resolved = FSMNVADConfig.coerce(config or {})
    return {name: tuple(tensor.shape) for name, tensor in FSMNVADModel(resolved).state_dict().items()}


class FSMNVADSafeTensorsCheckpointAdapter(CheckpointAdapter):
    """Identity-map a complete native FSMN VAD Safetensors artifact."""

    architecture_id = "fsmn-vad"
    adapter_id = "voicehub-fsmn-vad-safetensors"
    adapter_version = "1"

    def probe(
        self,
        files: tuple[Path, ...],
        config: Mapping[str, Any],
    ) -> bool:
        try:
            FSMNVADConfig.from_dict(config)
        except (TypeError, ValueError):
            return False
        return any(path.suffix == ".safetensors" for path in files)

    def tensor_plan(self, config: Mapping[str, Any]) -> TensorPlan:
        shapes = native_fsmn_vad_tensor_shapes(config)
        return TensorPlan(rules=tuple(CopyTensor(name, name) for name in sorted(shapes)), )


def _validate_official_state(
    payload: Any,
    *,
    config: FSMNVADConfig,
) -> dict[str, Any]:
    import torch

    if not isinstance(payload, Mapping) or not payload:
        raise TypeError("The official FSMN VAD checkpoint must be a non-empty state dict.")
    if any(not isinstance(name, str) or not isinstance(tensor, torch.Tensor)
           for name, tensor in payload.items()):
        raise TypeError("The official FSMN VAD state dict must map names to tensors only.")
    from voicehub.models.vad_funasr.native.modeling import FSMNVADModel

    expected = {
        name: tuple(tensor.shape)
        for name, tensor in FSMNVADModel(config).state_dict().items() if name.startswith("encoder.")
    }
    names = set(payload)
    if names != set(expected):
        raise ValueError(
            "Official FSMN VAD tensor namespace mismatch "
            f"(missing={sorted(set(expected) - names)}, "
            f"unexpected={sorted(names - set(expected))}).")
    mismatches = {
        name: (tuple(payload[name].shape), shape)
        for name, shape in expected.items() if tuple(payload[name].shape) != shape
    }
    if mismatches:
        raise ValueError(f"Official FSMN VAD tensor shape mismatch: {mismatches}.")
    fingerprint = tensor_inventory_fingerprint(payload)
    if fingerprint != FUNASR_OFFICIAL_TENSOR_FINGERPRINT:
        raise ValueError(
            "Official FSMN VAD tensor inventory fingerprint mismatch: "
            f"expected {FUNASR_OFFICIAL_TENSOR_FINGERPRINT}, "
            f"found {fingerprint}.")
    return {name: tensor.detach().cpu().contiguous() for name, tensor in payload.items()}


def convert_funasr_fsmn_checkpoint(
    checkpoint: str | Path,
    cmvn_file: str | Path,
    destination: str | Path,
    *,
    trust_pickle_checkpoint: bool = False,
    expected_checkpoint_sha256: str | None = None,
    expected_cmvn_sha256: str | None = None,
) -> Path:
    """Convert reviewed official files into a strict native directory.

    The released ``model.pt`` uses Python pickle. Conversion therefore
    requires explicit acknowledgement even though PyTorch's restricted
    ``weights_only`` loader is used. Steady-state runtime reads only the
    resulting Safetensors file.
    """
    if trust_pickle_checkpoint is not True:
        raise ValueError(
            "FunASR publishes FSMN VAD weights as a pickle-based `model.pt`. "
            "Review the artifact origin and pass "
            "`trust_pickle_checkpoint=True` for one-time conversion.")
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    cmvn_path = Path(cmvn_file).expanduser().resolve()
    for name, path in (
        ("checkpoint", checkpoint_path),
        ("CMVN", cmvn_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"FSMN VAD {name} file was not found: {path}.")
    expected_checkpoint_sha256 = (
        FUNASR_MODEL_SHA256 if expected_checkpoint_sha256 is None else expected_checkpoint_sha256)
    expected_cmvn_sha256 = (FUNASR_CMVN_SHA256 if expected_cmvn_sha256 is None else expected_cmvn_sha256)
    for name, path, expected in (
        ("checkpoint", checkpoint_path, expected_checkpoint_sha256),
        ("CMVN", cmvn_path, expected_cmvn_sha256),
    ):
        if (not isinstance(expected, str) or len(expected) != 64 or
                any(character not in "0123456789abcdefABCDEF" for character in expected)):
            raise ValueError(f"Expected {name} SHA-256 must be a hex digest.")
        actual = _file_sha256(path)
        if actual.lower() != expected.lower():
            raise ValueError(f"FSMN VAD {name} SHA-256 mismatch: "
                             f"expected {expected}, found {actual}.")

    import torch

    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    config = FSMNVADConfig()
    state = _validate_official_state(payload, config=config)
    shift, scale = parse_kaldi_cmvn(
        cmvn_path,
        expected_dimension=config.input_dim,
    )
    state.update({
        "frontend.cmvn_shift": shift,
        "frontend.cmvn_scale": scale,
    })
    expected_shapes = native_fsmn_vad_tensor_shapes(config)
    if set(state) != set(expected_shapes):
        raise RuntimeError("Internal FSMN VAD conversion did not produce a complete state dict.")

    output = Path(destination).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    safe_path = output / NATIVE_FSMN_VAD_FILENAME
    save_safetensors(
        state,
        safe_path,
        metadata={
            "format": NATIVE_FSMN_VAD_FORMAT,
            "source_checkpoint_sha256": _file_sha256(checkpoint_path),
            "source_cmvn_sha256": _file_sha256(cmvn_path),
            "source_tensor_fingerprint": FUNASR_OFFICIAL_TENSOR_FINGERPRINT,
        },
    )
    values = config.to_dict()
    values.update({
        "checkpoint_format": NATIVE_FSMN_VAD_FORMAT,
        "source_checkpoint_name": checkpoint_path.name,
        "source_checkpoint_sha256": _file_sha256(checkpoint_path),
        "source_cmvn_name": cmvn_path.name,
        "source_cmvn_sha256": _file_sha256(cmvn_path),
        "source_tensor_fingerprint": FUNASR_OFFICIAL_TENSOR_FINGERPRINT,
    })
    write_json_file(output / "config.json", values)

    from voicehub.models.vad_funasr.native.modeling import FSMNVADModel

    with SafeTensorReader(safe_path) as reader:
        FSMNVADSafeTensorsCheckpointAdapter().load_streaming(
            FSMNVADModel(config),
            reader,
            values,
            strict=True,
        )
    return output


__all__ = [
    "FSMNVADSafeTensorsCheckpointAdapter",
    "NATIVE_FSMN_VAD_FILENAME",
    "NATIVE_FSMN_VAD_FORMAT",
    "convert_funasr_fsmn_checkpoint",
    "native_fsmn_vad_tensor_shapes",
    "tensor_inventory_fingerprint",
]
