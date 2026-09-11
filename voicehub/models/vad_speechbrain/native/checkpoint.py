"""Strict conversion of the published SpeechBrain CRDNN VAD checkpoint."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicehub.checkpointing import CheckpointAdapter, CopyTensor, SafeTensorReader, TensorPlan, save_safetensors
from voicehub.hub import write_json_file
from voicehub.models.vad_speechbrain.native.configuration import SpeechBrainCRDNNVADConfig
from voicehub.models.vad_speechbrain.native.metadata import (
    SPEECHBRAIN_TRAINING_SOURCE_REVISION,
    SPEECHBRAIN_VAD_HPARAMS_SHA256,
    SPEECHBRAIN_VAD_MODEL_SHA256,
    SPEECHBRAIN_VAD_REVISION,
    SPEECHBRAIN_VAD_TENSOR_FINGERPRINT,
)

NATIVE_SPEECHBRAIN_VAD_FORMAT = "voicehub-speechbrain-crdnn-vad-v1"
NATIVE_SPEECHBRAIN_VAD_FILENAME = "model.safetensors"
_ConfigLike = SpeechBrainCRDNNVADConfig | Mapping[str, Any] | None

_SOURCE_TO_NATIVE = {
    "0.norm1.norm.weight": "initial_norm.weight",
    "0.norm1.norm.bias": "initial_norm.bias",
    "0.cnn1.conv_1.conv.weight": "cnn_blocks.0.conv_1.weight",
    "0.cnn1.conv_1.conv.bias": "cnn_blocks.0.conv_1.bias",
    "0.cnn1.norm_1.norm.weight": "cnn_blocks.0.norm_1.weight",
    "0.cnn1.norm_1.norm.bias": "cnn_blocks.0.norm_1.bias",
    "0.cnn1.conv_2.conv.weight": "cnn_blocks.0.conv_2.weight",
    "0.cnn1.conv_2.conv.bias": "cnn_blocks.0.conv_2.bias",
    "0.cnn1.norm_2.norm.weight": "cnn_blocks.0.norm_2.weight",
    "0.cnn1.norm_2.norm.bias": "cnn_blocks.0.norm_2.bias",
    "0.cnn2.conv_1.conv.weight": "cnn_blocks.1.conv_1.weight",
    "0.cnn2.conv_1.conv.bias": "cnn_blocks.1.conv_1.bias",
    "0.cnn2.norm_1.norm.weight": "cnn_blocks.1.norm_1.weight",
    "0.cnn2.norm_1.norm.bias": "cnn_blocks.1.norm_1.bias",
    "0.cnn2.conv_2.conv.weight": "cnn_blocks.1.conv_2.weight",
    "0.cnn2.conv_2.conv.bias": "cnn_blocks.1.conv_2.bias",
    "0.cnn2.norm_2.norm.weight": "cnn_blocks.1.norm_2.weight",
    "0.cnn2.norm_2.norm.bias": "cnn_blocks.1.norm_2.bias",
    "2.dnn1.linear.w.weight": "dnn_blocks.0.linear.weight",
    "2.dnn1.linear.w.bias": "dnn_blocks.0.linear.bias",
    "2.dnn1.norm.norm.weight": "dnn_blocks.0.norm.weight",
    "2.dnn1.norm.norm.bias": "dnn_blocks.0.norm.bias",
    "2.dnn1.norm.norm.running_mean": "dnn_blocks.0.norm.running_mean",
    "2.dnn1.norm.norm.running_var": "dnn_blocks.0.norm.running_var",
    "2.dnn1.norm.norm.num_batches_tracked": ("dnn_blocks.0.norm.num_batches_tracked"),
    "2.dnn2.linear.w.weight": "dnn_blocks.1.linear.weight",
    "2.dnn2.linear.w.bias": "dnn_blocks.1.linear.bias",
    "2.dnn2.norm.norm.weight": "dnn_blocks.1.norm.weight",
    "2.dnn2.norm.norm.bias": "dnn_blocks.1.norm.bias",
    "2.dnn2.norm.norm.running_mean": "dnn_blocks.1.norm.running_mean",
    "2.dnn2.norm.norm.running_var": "dnn_blocks.1.norm.running_var",
    "2.dnn2.norm.norm.num_batches_tracked": ("dnn_blocks.1.norm.num_batches_tracked"),
    "2.lin.w.weight": "output.weight",
}
for _name in (
        "weight_ih_l0",
        "weight_hh_l0",
        "bias_ih_l0",
        "bias_hh_l0",
        "weight_ih_l0_reverse",
        "weight_hh_l0_reverse",
        "bias_ih_l0_reverse",
        "bias_hh_l0_reverse",
        "weight_ih_l1",
        "weight_hh_l1",
        "bias_ih_l1",
        "bias_hh_l1",
        "weight_ih_l1_reverse",
        "weight_hh_l1_reverse",
        "bias_ih_l1_reverse",
        "bias_hh_l1_reverse",
):
    _SOURCE_TO_NATIVE[f"1.rnn.{_name}"] = f"rnn.{_name}"


def speechbrain_source_tensor_mapping() -> dict[str, str]:
    """Return the reviewed upstream-to-native tensor namespace."""
    return dict(_SOURCE_TO_NATIVE)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_inventory_fingerprint(tensors: Mapping[str, Any]) -> str:
    rows = []
    for name, tensor in sorted(tensors.items()):
        dtype = {
            "torch.float32": "F32",
            "torch.float64": "F64",
            "torch.float16": "F16",
            "torch.bfloat16": "BF16",
            "torch.int64": "I64",
        }.get(str(tensor.dtype), str(tensor.dtype))
        shape = "x".join(str(item) for item in tensor.shape)
        rows.append(f"{name}|{dtype}|{shape}")
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def native_speechbrain_vad_tensor_shapes(config: _ConfigLike = None, ) -> dict[str, tuple[int, ...]]:
    from voicehub.models.vad_speechbrain.native.modeling import SpeechBrainCRDNNVADModel

    resolved = SpeechBrainCRDNNVADConfig.coerce(config or {})
    return {
        name: tuple(tensor.shape)
        for name, tensor in SpeechBrainCRDNNVADModel(resolved).state_dict().items()
    }


class SpeechBrainVADSafeTensorsCheckpointAdapter(CheckpointAdapter):
    """Identity-load a complete VoiceHub-native CRDNN artifact."""

    architecture_id = "speechbrain-crdnn-vad"
    adapter_id = "voicehub-speechbrain-crdnn-vad-safetensors"
    adapter_version = "1"

    def probe(
        self,
        files: tuple[Path, ...],
        config: Mapping[str, Any],
    ) -> bool:
        try:
            SpeechBrainCRDNNVADConfig.from_dict(config)
        except (TypeError, ValueError):
            return False
        return any(path.suffix == ".safetensors" for path in files)

    def tensor_plan(self, config: Mapping[str, Any]) -> TensorPlan:
        shapes = native_speechbrain_vad_tensor_shapes(config)
        return TensorPlan(rules=tuple(CopyTensor(name, name) for name in sorted(shapes)), )


def _validate_digest(
    path: Path,
    expected: str | None,
    *,
    label: str,
) -> str:
    actual = _file_sha256(path)
    if expected is None:
        return actual
    if (not isinstance(expected, str) or len(expected) != 64 or
            any(character not in "0123456789abcdefABCDEF" for character in expected)):
        raise ValueError(f"Expected {label} SHA-256 must be a hex digest.")
    if actual.lower() != expected.lower():
        raise ValueError(
            f"SpeechBrain VAD {label} SHA-256 mismatch: "
            f"expected {expected}, found {actual}.")
    return actual


def convert_speechbrain_vad_checkpoint(
    checkpoint: str | Path,
    destination: str | Path,
    *,
    hyperparams_file: str | Path | None = None,
    trust_pickle_checkpoint: bool = False,
    expected_checkpoint_sha256: str | None = None,
    expected_hyperparams_sha256: str | None = None,
) -> Path:
    """Convert a reviewed pickle checkpoint into strict Safetensors.

    ``torch.load(weights_only=True)`` constrains deserialization, but
    the source format is still pickle.  Explicit acknowledgement is
    therefore mandatory; steady-state inference and training never
    reopen it.
    """
    if trust_pickle_checkpoint is not True:
        raise ValueError(
            "SpeechBrain publishes `model.ckpt` in a pickle-based format. "
            "Review its origin and pass `trust_pickle_checkpoint=True` for "
            "one-time restricted conversion.")
    source = Path(checkpoint).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"SpeechBrain VAD checkpoint was not found: {source}.")
    source_sha = _validate_digest(
        source,
        expected_checkpoint_sha256,
        label="checkpoint",
    )
    hparams_sha = None
    if hyperparams_file is not None:
        hparams = Path(hyperparams_file).expanduser().resolve()
        if not hparams.is_file():
            raise FileNotFoundError(f"SpeechBrain VAD hyperparams were not found: {hparams}.")
        hparams_sha = _validate_digest(
            hparams,
            expected_hyperparams_sha256,
            label="hyperparams",
        )
    elif expected_hyperparams_sha256 is not None:
        raise ValueError("`hyperparams_file` is required when verifying its SHA-256.")

    import torch

    payload = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or not payload:
        raise TypeError("SpeechBrain `model.ckpt` must be a non-empty state dict.")
    if any(not isinstance(name, str) or not isinstance(tensor, torch.Tensor)
           for name, tensor in payload.items()):
        raise TypeError("SpeechBrain checkpoint entries must map names to tensors.")
    if set(payload) != set(_SOURCE_TO_NATIVE):
        raise ValueError(
            "SpeechBrain checkpoint tensor namespace mismatch "
            f"(missing={sorted(set(_SOURCE_TO_NATIVE) - set(payload))}, "
            f"unexpected={sorted(set(payload) - set(_SOURCE_TO_NATIVE))}).")
    fingerprint = tensor_inventory_fingerprint(payload)
    if fingerprint != SPEECHBRAIN_VAD_TENSOR_FINGERPRINT:
        raise ValueError(
            "SpeechBrain checkpoint inventory fingerprint mismatch: "
            f"expected {SPEECHBRAIN_VAD_TENSOR_FINGERPRINT}, found {fingerprint}.")
    config = SpeechBrainCRDNNVADConfig()
    shapes = native_speechbrain_vad_tensor_shapes(config)
    state = {
        native: payload[source_name].detach().cpu().contiguous()
        for source_name, native in _SOURCE_TO_NATIVE.items()
    }
    if set(state) != set(shapes):
        raise RuntimeError("Internal SpeechBrain VAD conversion is incomplete.")
    mismatches = {
        name: (tuple(state[name].shape), shapes[name])
        for name in shapes if tuple(state[name].shape) != shapes[name]
    }
    if mismatches:
        raise ValueError(f"SpeechBrain VAD tensor shape mismatch: {mismatches}.")

    output = Path(destination).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    safe_path = output / NATIVE_SPEECHBRAIN_VAD_FILENAME
    metadata = {
        "format": NATIVE_SPEECHBRAIN_VAD_FORMAT,
        "source_checkpoint_sha256": source_sha,
        "source_tensor_fingerprint": fingerprint,
        "source_artifact_revision": SPEECHBRAIN_VAD_REVISION,
    }
    if hparams_sha is not None:
        metadata["source_hyperparams_sha256"] = hparams_sha
    save_safetensors(state, safe_path, metadata=metadata)
    values = config.to_dict()
    values.update({
        "checkpoint_format": NATIVE_SPEECHBRAIN_VAD_FORMAT,
        "source_checkpoint_name": source.name,
        "source_checkpoint_sha256": source_sha,
        "source_hyperparams_sha256": hparams_sha,
        "source_tensor_fingerprint": fingerprint,
        "source_artifact_revision": SPEECHBRAIN_VAD_REVISION,
        "source_training_revision": SPEECHBRAIN_TRAINING_SOURCE_REVISION,
    })
    write_json_file(output / "config.json", values)

    from voicehub.models.vad_speechbrain.native.modeling import SpeechBrainCRDNNVADModel

    with SafeTensorReader(safe_path) as reader:
        SpeechBrainVADSafeTensorsCheckpointAdapter().load_streaming(
            SpeechBrainCRDNNVADModel(config),
            reader,
            values,
            strict=True,
        )
    return output


__all__ = [
    "NATIVE_SPEECHBRAIN_VAD_FILENAME",
    "NATIVE_SPEECHBRAIN_VAD_FORMAT",
    "SpeechBrainVADSafeTensorsCheckpointAdapter",
    "convert_speechbrain_vad_checkpoint",
    "native_speechbrain_vad_tensor_shapes",
    "speechbrain_source_tensor_mapping",
    "tensor_inventory_fingerprint",
]
