"""Safe native weight I/O for imported Supertonic graphs."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch

from voicehub.checkpointing import SafeTensorReader, save_safetensors
from voicehub.models.supertonic.native.runtime import NativeSupertonicRuntime
from voicehub.neural.onnx import NativeONNXGraph

NATIVE_SUPERTONIC_WEIGHT_DIRECTORY = "native_weights"
SUPERTONIC_GRAPH_ROLES = (
    "duration_predictor",
    "text_encoder",
    "vector_estimator",
    "vocoder",
)


def _graph(runtime: NativeSupertonicRuntime, role: str) -> NativeONNXGraph:
    if role not in SUPERTONIC_GRAPH_ROLES:
        raise ValueError(f"Unknown Supertonic graph role {role!r}.")
    graph = getattr(runtime, role)
    if not isinstance(graph, NativeONNXGraph):
        raise TypeError(f"Supertonic runtime {role!r} is not a native graph.")
    return graph


def load_supertonic_graph_weights(
    graph: NativeONNXGraph,
    path: str | Path,
) -> None:
    """Strictly stream one native Safetensors state into a graph."""
    source = Path(path).expanduser()
    with SafeTensorReader(source) as reader:
        expected = set(graph.initializer_names)
        provided = set(reader.keys())
        if expected != provided:
            raise ValueError(
                f"Supertonic state {source.name!r} namespace mismatch: "
                f"{len(expected - provided)} missing, "
                f"{len(provided - expected)} unexpected.")
        with torch.no_grad():
            for name in graph.initializer_names:
                target = graph.initializer_tensor(name)
                record = reader.record(name)
                if record.shape != tuple(target.shape):
                    raise ValueError(
                        f"Supertonic tensor {name!r} has shape "
                        f"{record.shape}; expected {tuple(target.shape)}.")
                value = reader.get_tensor(name)
                if value.dtype != target.dtype:
                    raise ValueError(
                        f"Supertonic tensor {name!r} has dtype "
                        f"{value.dtype}; expected {target.dtype}.")
                if (not target.is_floating_point() and not torch.equal(value, target.detach().cpu())):
                    raise ValueError(
                        f"Supertonic structural initializer {name!r} "
                        "differs from the reviewed ONNX graph.")
                target.copy_(value.to(device=target.device))


def load_supertonic_native_weights(
    runtime: NativeSupertonicRuntime,
    paths: Mapping[str, Path],
) -> None:
    """Load all four graph states or reject a partial artifact."""
    if not isinstance(paths, Mapping):
        raise TypeError("Supertonic native weight paths must be a mapping.")
    if not paths:
        return
    expected = set(SUPERTONIC_GRAPH_ROLES)
    if set(paths) != expected:
        raise ValueError("Supertonic native weight artifact must contain all graph roles.")
    for role in SUPERTONIC_GRAPH_ROLES:
        load_supertonic_graph_weights(
            _graph(runtime, role),
            paths[role],
        )


def save_supertonic_native_weights(
    runtime: NativeSupertonicRuntime,
    directory: str | Path,
) -> dict[str, Path]:
    """Write deterministic original-name Safetensors for every graph."""
    destination = (Path(directory).expanduser() / NATIVE_SUPERTONIC_WEIGHT_DIRECTORY)
    destination.mkdir(parents=True, exist_ok=True)
    result = {}
    for role in SUPERTONIC_GRAPH_ROLES:
        path = destination / f"{role}.safetensors"
        graph = _graph(runtime, role)
        save_safetensors(
            graph.onnx_state_dict(),
            path,
            metadata={
                "architecture": "supertonic-3",
                "graph_role": role,
                "tensor_namespace": "onnx-original",
            },
        )
        result[role] = path
    return result


__all__ = [
    "NATIVE_SUPERTONIC_WEIGHT_DIRECTORY",
    "SUPERTONIC_GRAPH_ROLES",
    "load_supertonic_graph_weights",
    "load_supertonic_native_weights",
    "save_supertonic_native_weights",
]
