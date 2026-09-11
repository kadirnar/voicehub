"""Strict ONNX weight conversion and native TEN VAD checkpoint loading."""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicehub.checkpointing import (
    CheckpointAdapter,
    CopyTensor,
    ONNXTensor,
    SafeTensorReader,
    TensorPlan,
    read_onnx_model,
    save_safetensors,
)
from voicehub.hub import write_json_file
from voicehub.models.vad_ten.native.configuration import TENVADConfig
from voicehub.models.vad_ten.native.metadata import (
    SHERPA_ONNX_REVISION,
    TEN_VAD_GRAPH_FINGERPRINT,
    TEN_VAD_INITIALIZER_INVENTORY_FINGERPRINT,
    TEN_VAD_ONNX_SHA256,
    TEN_VAD_REVISION,
    TEN_VAD_SHERPA_ONNX_SHA256,
    TEN_VAD_SOURCE_LICENSE,
)

NATIVE_TEN_VAD_FORMAT = "voicehub-native-ten-vad-v1"
NATIVE_TEN_VAD_FILENAME = "model.safetensors"

_INPUTS = ("input_1", "input_2", "input_3", "input_6", "input_7")
_OUTPUTS = ("output_1", "output_2", "output_3", "output_6", "output_7")
_OFFICIAL_DIGESTS = frozenset({
    TEN_VAD_ONNX_SHA256,
    TEN_VAD_SHERPA_ONNX_SHA256,
})
_ConfigInput = TENVADConfig | Mapping[str, Any] | None
_TensorShapes = dict[str, tuple[int, ...]]
_SOURCE_TO_NATIVE = {
    "const_fold_opt__178": "spatial_depthwise.weight",
    "StatefulPartitionedCall/vad_model/separable_conv2d/"
    "separable_conv2d/ReadVariableOp_1:0": "spatial_pointwise.weight",
    "StatefulPartitionedCall/vad_model/separable_conv2d/"
    "BiasAdd/ReadVariableOp:0": "spatial_pointwise.bias",
    "const_fold_opt__179": "temporal_depthwise_1.weight",
    "StatefulPartitionedCall/vad_model/separable_conv1d/ExpandDims_2:0": "temporal_pointwise_1.weight",
    "StatefulPartitionedCall/vad_model/separable_conv1d/"
    "BiasAdd/ReadVariableOp:0": "temporal_pointwise_1.bias",
    "const_fold_opt__180": "temporal_depthwise_2.weight",
    "StatefulPartitionedCall/vad_model/separable_conv1d_1/ExpandDims_2:0": "temporal_pointwise_2.weight",
    "StatefulPartitionedCall/vad_model/separable_conv1d_1/"
    "BiasAdd/ReadVariableOp:0": "temporal_pointwise_2.bias",
    "W0__70": "lstm_1.weight_ih",
    "R0__71": "lstm_1.weight_hh",
    "W0__99": "lstm_2.weight_ih",
    "R0__100": "lstm_2.weight_hh",
    "StatefulPartitionedCall/vad_model/dense_3/Tensordot/"
    "ReadVariableOp:0": "dense.weight",
    "StatefulPartitionedCall/vad_model/dense_3/BiasAdd/"
    "ReadVariableOp:0": "dense.bias",
    "StatefulPartitionedCall/vad_model/dense_5/Tensordot/"
    "ReadVariableOp:0": "output.weight",
    "StatefulPartitionedCall/vad_model/dense_5/BiasAdd/"
    "ReadVariableOp:0": "output.bias",
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _framed(payload: bytes) -> bytes:
    """Return an unambiguous length-prefixed byte string."""
    return struct.pack(">Q", len(payload)) + payload


def _canonical_attribute_value(value: Any) -> bytes:
    """Encode the parser's bounded attribute types without lossy formatting."""
    if type(value) is int:
        return b"i" + _framed(str(value).encode("ascii"))
    if type(value) is float:
        return b"f" + struct.pack(">d", value)
    if type(value) is bytes:
        return b"b" + _framed(value)
    if type(value) is tuple:
        return b"l" + _framed(b"".join(_framed(_canonical_attribute_value(item)) for item in value))
    if isinstance(value, ONNXTensor):
        fields = (
            value.name.encode("utf-8"),
            str(value.data_type).encode("ascii"),
            _canonical_attribute_value(value.dimensions),
            value.raw_data,
            _canonical_attribute_value(value.float_data),
            _canonical_attribute_value(value.int32_data),
            _canonical_attribute_value(value.int64_data),
            _canonical_attribute_value(value.double_data),
            _canonical_attribute_value(value.uint64_data),
        )
        return b"t" + _framed(b"".join(_framed(field) for field in fields))
    raise TypeError("Unsupported ONNX attribute value in TEN graph fingerprint: "
                    f"{type(value).__name__}.")


def _graph_fingerprint(model: Any) -> str:
    """Fingerprint graph semantics, including every reviewed node attribute."""
    digest = hashlib.sha256(b"voicehub-ten-vad-graph-v2\0")
    for node in model.graph.nodes:
        digest.update(_framed(node.domain.encode("utf-8")))
        digest.update(_framed(node.op_type.encode("utf-8")))
        for connections in (node.inputs, node.outputs):
            digest.update(_framed(str(len(connections)).encode("ascii")))
            for value in connections:
                digest.update(_framed(value.encode("utf-8")))
        digest.update(_framed(str(len(node.attributes)).encode("ascii")))
        for name, attribute in sorted(node.attributes.items()):
            digest.update(_framed(name.encode("utf-8")))
            digest.update(_framed(str(attribute.attribute_type).encode("ascii")))
            digest.update(_framed(_canonical_attribute_value(attribute.value)))
    return digest.hexdigest()


def _inventory_fingerprint(model: Any) -> str:
    rows = [
        f"{name}|{tensor.data_type}|"
        f"{'x'.join(str(item) for item in tensor.dimensions)}"
        for name, tensor in sorted(model.graph.initializers.items())
    ]
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def native_ten_vad_tensor_shapes(config: _ConfigInput = None) -> _TensorShapes:
    from voicehub.models.vad_ten.native.modeling import TENVADModel

    resolved = TENVADConfig.coerce(config or {})
    return {name: tuple(tensor.shape) for name, tensor in TENVADModel(resolved).state_dict().items()}


class TENVADSafeTensorsCheckpointAdapter(CheckpointAdapter):
    """Strict identity adapter for a complete native TEN artifact."""

    architecture_id = "ten-vad"
    adapter_id = "voicehub-native-ten-vad"
    adapter_version = "1"

    def probe(
        self,
        files: tuple[Path, ...],
        config: Mapping[str, Any],
    ) -> bool:
        try:
            TENVADConfig.from_dict(config)
        except (TypeError, ValueError):
            return False
        return any(path.suffix == ".safetensors" for path in files)

    def tensor_plan(self, config: Mapping[str, Any]) -> TensorPlan:
        shapes = native_ten_vad_tensor_shapes(config)
        return TensorPlan(rules=tuple(CopyTensor(name, name) for name in sorted(shapes)), )


def _metadata_vector(
    metadata: Mapping[str, str],
    key: str,
    *,
    size: int,
):
    import torch

    try:
        raw = metadata[key]
    except KeyError:
        return None
    try:
        values = [float(item) for item in raw.split(",")]
    except ValueError as error:
        raise ValueError(f"TEN ONNX metadata {key!r} is not numeric.") from error
    if len(values) != size or any(not math.isfinite(value) for value in values):
        raise ValueError(f"TEN ONNX metadata {key!r} must contain {size} finite values.")
    return torch.tensor(values, dtype=torch.float32)


def _validate_graph(model: Any) -> None:
    if model.ir_version != 4:
        raise ValueError(f"TEN ONNX requires IR version 4, found {model.ir_version}.")
    if dict(model.opsets).get("") != 9:
        raise ValueError("TEN ONNX requires the default-domain opset 9 graph.")
    if tuple(item.name for item in model.graph.inputs) != _INPUTS:
        raise ValueError("TEN ONNX input namespace does not match the reviewed graph.")
    if tuple(item.name for item in model.graph.outputs) != _OUTPUTS:
        raise ValueError("TEN ONNX output namespace does not match the reviewed graph.")
    graph_fingerprint = _graph_fingerprint(model)
    if graph_fingerprint != TEN_VAD_GRAPH_FINGERPRINT:
        raise ValueError(
            "TEN ONNX operator graph fingerprint mismatch: expected "
            f"{TEN_VAD_GRAPH_FINGERPRINT}, found {graph_fingerprint}.")
    inventory = _inventory_fingerprint(model)
    if inventory != TEN_VAD_INITIALIZER_INVENTORY_FINGERPRINT:
        raise ValueError(
            "TEN ONNX initializer inventory mismatch: expected "
            f"{TEN_VAD_INITIALIZER_INVENTORY_FINGERPRINT}, found {inventory}.")
    model_type = model.metadata.get("model_type")
    if model_type is not None and model_type != "ten-vad":
        raise ValueError(f"TEN ONNX metadata declares unsupported model_type {model_type!r}.")


def _converted_state(model_proto: Any, config: TENVADConfig):
    import torch

    from voicehub.models.vad_ten.native.modeling import TENVADModel

    model = TENVADModel(config)
    state = {name: tensor.detach().cpu().contiguous() for name, tensor in model.state_dict().items()}
    initializers = model_proto.graph.initializers
    for source, target in _SOURCE_TO_NATIVE.items():
        tensor = initializers[source].to_torch().to(dtype=torch.float32)
        if source.startswith(("W0__", "R0__")):
            tensor = tensor.squeeze(0)
        elif target in {"dense.weight", "output.weight"}:
            tensor = tensor.transpose(0, 1)
        state[target] = tensor.contiguous()

    for source, prefix in (("B0__72", "lstm_1"), ("B0__101", "lstm_2")):
        bias = initializers[source].to_torch().to(dtype=torch.float32).reshape(-1)
        if bias.numel() != config.recurrent_size * 8:
            raise ValueError(f"TEN ONNX recurrent bias {source!r} has the wrong size.")
        split = config.recurrent_size * 4
        state[f"{prefix}.bias_ih"] = bias[:split].contiguous()
        state[f"{prefix}.bias_hh"] = bias[split:].contiguous()

    mean = _metadata_vector(model_proto.metadata, "mean", size=41)
    inv_stddev = _metadata_vector(
        model_proto.metadata,
        "inv_stddev",
        size=41,
    )
    window = _metadata_vector(model_proto.metadata, "window", size=768)
    supplied = (mean, inv_stddev, window)
    if any(value is not None for value in supplied):
        if any(value is None for value in supplied):
            raise ValueError("Sherpa TEN ONNX metadata must provide mean, inv_stddev, and window together.")
        state["frontend.mean"] = mean
        state["frontend.inv_stddev"] = inv_stddev
        state["frontend.window"] = window

    expected = native_ten_vad_tensor_shapes(config)
    if set(state) != set(expected):
        raise RuntimeError("Internal TEN ONNX conversion did not produce a complete state.")
    mismatches = {
        name: (tuple(state[name].shape), expected[name])
        for name in expected if tuple(state[name].shape) != expected[name]
    }
    if mismatches:
        raise ValueError(f"TEN ONNX tensor shape mismatch: {mismatches}.")
    return state


def convert_ten_vad_onnx_checkpoint(
    checkpoint: str | Path,
    destination: str | Path,
    *,
    trust_onnx_checkpoint: bool = False,
    expected_source_sha256: str | None = None,
    window_size: int = 256,
) -> Path:
    """Convert one reviewed TEN graph into Safetensors and JSON.

    Parsing does not import or execute ONNX.  Explicit acknowledgement
    is still required because the source license is non-standard and
    callers must review the artifact's origin before creating a
    derivative.
    """
    if trust_onnx_checkpoint is not True:
        raise ValueError(
            "Review the TEN artifact and its non-standard source license, "
            "then pass `trust_onnx_checkpoint=True` for one-time conversion.")
    source = Path(checkpoint).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"TEN ONNX checkpoint was not found: {source}.")
    source_sha = _file_sha256(source)
    if expected_source_sha256 is not None:
        if (not isinstance(expected_source_sha256, str) or len(expected_source_sha256) != 64 or
                any(character not in "0123456789abcdefABCDEF" for character in expected_source_sha256)):
            raise ValueError("Expected TEN source SHA-256 must be a hex digest.")
        if source_sha != expected_source_sha256.lower():
            raise ValueError(
                "TEN ONNX SHA-256 mismatch: expected "
                f"{expected_source_sha256.lower()}, found {source_sha}.")
    model_proto = read_onnx_model(source)
    _validate_graph(model_proto)
    config = TENVADConfig(window_size=window_size)
    state = _converted_state(model_proto, config)

    output = Path(destination).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / NATIVE_TEN_VAD_FILENAME
    graph_fingerprint = _graph_fingerprint(model_proto)
    inventory = _inventory_fingerprint(model_proto)
    save_safetensors(
        state,
        checkpoint_path,
        metadata={
            "format": NATIVE_TEN_VAD_FORMAT,
            "architecture": "ten-vad",
            "source_onnx_sha256": source_sha,
            "source_graph_fingerprint": graph_fingerprint,
            "source_inventory_fingerprint": inventory,
            "source_revision": TEN_VAD_REVISION,
            "source_license": TEN_VAD_SOURCE_LICENSE,
            "official_source": str(source_sha in _OFFICIAL_DIGESTS).lower(),
        },
    )
    checkpoint_sha = _file_sha256(checkpoint_path)
    values = config.to_dict()
    values.update({
        "checkpoint_format": NATIVE_TEN_VAD_FORMAT,
        "checkpoint_sha256": checkpoint_sha,
        "source_onnx_name": source.name,
        "source_onnx_sha256": source_sha,
        "source_graph_fingerprint": graph_fingerprint,
        "source_inventory_fingerprint": inventory,
        "source_revision": TEN_VAD_REVISION,
        "sherpa_compatibility_revision": SHERPA_ONNX_REVISION,
        "source_license": TEN_VAD_SOURCE_LICENSE,
        "official_source": source_sha in _OFFICIAL_DIGESTS,
        "training_recipe": "voicehub-reconstructed-window-bce",
        "upstream_training_recipe_published": False,
    })
    write_json_file(output / "config.json", values)

    from voicehub.models.vad_ten.native.modeling import TENVADModel

    with SafeTensorReader(checkpoint_path) as reader:
        declared = reader.metadata.get("format")
        if declared != NATIVE_TEN_VAD_FORMAT:
            raise RuntimeError("Converted TEN Safetensors format marker was not preserved.")
        TENVADSafeTensorsCheckpointAdapter().load_streaming(
            TENVADModel(config),
            reader,
            values,
            strict=True,
        )
    return output


__all__ = [
    "NATIVE_TEN_VAD_FILENAME",
    "NATIVE_TEN_VAD_FORMAT",
    "TENVADSafeTensorsCheckpointAdapter",
    "convert_ten_vad_onnx_checkpoint",
    "native_ten_vad_tensor_shapes",
]
