"""VoiceHub-managed Mimi codec used by Sesame CSM.

The executable Mimi graph is the pinned Moshi source already retained
inside VoiceHub.  This module owns construction and strict Safetensors
loading, so the public CSM path does not import Moshi's Hub loader,
SentencePiece, Safetensors package, or any model framework.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from voicehub.checkpointing import SafeTensorReader
from voicehub.checkpointing.errors import CheckpointCompatibilityError
from voicehub.models.csm.native.metadata import (
    MIMI_CHECKPOINT_HEADER_FINGERPRINT,
    MIMI_CHECKPOINT_PARAMETER_COUNT,
    MIMI_CHECKPOINT_TENSOR_COUNT,
)
from voicehub.models.csm.source.moshi.models.compression import MimiModel
from voicehub.models.csm.source.moshi.modules import SEANetDecoder, SEANetEncoder, transformer
from voicehub.models.csm.source.moshi.quantization import SplitResidualVectorQuantizer

MIMI_SAMPLE_RATE = 24_000
MIMI_FRAME_RATE = 12.5
MIMI_NUM_CODEBOOKS = 32
MIMI_CARDINALITY = 2_048

_SEANET_CONFIG = {
    "channels": 1,
    "dimension": 512,
    "causal": True,
    "n_filters": 64,
    "n_residual_layers": 1,
    "activation": "ELU",
    "compress": 2,
    "dilation_base": 2,
    "disable_norm_outer_blocks": 0,
    "kernel_size": 7,
    "residual_kernel_size": 3,
    "last_kernel_size": 3,
    "norm": "none",
    "pad_mode": "constant",
    "ratios": [8, 6, 5, 4],
    "true_skip": True,
}
_QUANTIZER_CONFIG = {
    "dimension": 256,
    "n_q": MIMI_NUM_CODEBOOKS,
    "bins": MIMI_CARDINALITY,
    "input_dimension": _SEANET_CONFIG["dimension"],
    "output_dimension": _SEANET_CONFIG["dimension"],
}
_TRANSFORMER_CONFIG = {
    "d_model": _SEANET_CONFIG["dimension"],
    "num_heads": 8,
    "num_layers": 8,
    "causal": True,
    "layer_scale": 0.01,
    "context": 250,
    "conv_layout": True,
    "max_period": 10_000,
    "gating": "none",
    "norm": "layer_norm",
    "positional_embedding": "rope",
    "dim_feedforward": 2_048,
    "input_dimension": _SEANET_CONFIG["dimension"],
    "output_dimensions": [_SEANET_CONFIG["dimension"]],
}


@dataclass(frozen=True, slots=True)
class MimiCheckpointReport:
    path: Path
    tensor_count: int
    parameter_count: int
    header_fingerprint: str


def build_mimi(*, device: str | torch.device = "cpu") -> MimiModel:
    """Construct the exact 24 kHz/32-codebook Mimi graph used by CSM.

    The default-device context matters for this 96M-parameter graph: it avoids
    allocating a complete temporary CPU copy when the requested destination is
    CUDA, and it makes metadata-only construction on ``meta`` genuinely
    allocation free.
    """
    resolved_device = torch.device(device)
    with resolved_device:
        encoder = SEANetEncoder(**_SEANET_CONFIG)
        decoder = SEANetDecoder(**_SEANET_CONFIG)
        encoder_transformer = transformer.ProjectedTransformer(
            device=resolved_device,
            **_TRANSFORMER_CONFIG,
        )
        decoder_transformer = transformer.ProjectedTransformer(
            device=resolved_device,
            **_TRANSFORMER_CONFIG,
        )
        quantizer = SplitResidualVectorQuantizer(**_QUANTIZER_CONFIG)
        codec = MimiModel(
            encoder,
            decoder,
            quantizer,
            channels=1,
            sample_rate=MIMI_SAMPLE_RATE,
            frame_rate=MIMI_FRAME_RATE,
            encoder_frame_rate=(MIMI_SAMPLE_RATE / encoder.hop_length),
            causal=True,
            resample_method="conv",
            encoder_transformer=encoder_transformer,
            decoder_transformer=decoder_transformer,
        )
    codec.set_num_codebooks(MIMI_NUM_CODEBOOKS)
    return codec


def _inventory_fingerprint(reader: SafeTensorReader) -> str:
    rows = [(
        f"{name}|{reader.record(name).dtype}|"
        f"{'x'.join(str(value) for value in reader.tensor_shape(name))}") for name in sorted(reader.keys())]
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def _checkpoint_tensor_name(graph_name: str) -> str:
    """Map the retained Moshi graph namespace to the released checkpoint.

    The pinned Mimi artifact predates Moshi's later ``ModuleList``
    wrapper for per-step attention weights.  The executable graph has
    one element in each list, while the artifact uses PyTorch's original
    ``in_proj_weight``/``out_proj.weight`` spellings.  This translation
    is deterministic and shape-preserving; no tensor is split,
    concatenated, or approximated.
    """
    if ".self_attn.in_projs.0.weight" in graph_name:
        return graph_name.replace(
            ".self_attn.in_projs.0.weight",
            ".self_attn.in_proj_weight",
        )
    if ".self_attn.out_projs.0.weight" in graph_name:
        return graph_name.replace(
            ".self_attn.out_projs.0.weight",
            ".self_attn.out_proj.weight",
        )
    return graph_name


def mimi_checkpoint_inventory(codec: nn.Module, ) -> dict[str, tuple[str, tuple[int, ...]]]:
    """Return the exact released-checkpoint namespace for a Mimi graph."""
    inventory: dict[str, tuple[str, tuple[int, ...]]] = {}
    for graph_name, value in codec.state_dict(keep_vars=True).items():
        checkpoint_name = _checkpoint_tensor_name(graph_name)
        if checkpoint_name in inventory:
            raise RuntimeError(
                "Mimi checkpoint namespace translation produced a duplicate "
                f"tensor name: {checkpoint_name!r}.")
        inventory[checkpoint_name] = (
            "F32",
            tuple(value.shape),
        )
    return inventory


def _validate_mimi_layout(
    codec: nn.Module,
    reader: SafeTensorReader,
) -> tuple[tuple[str, str], ...]:
    targets = codec.state_dict(keep_vars=True)
    checkpoint_to_graph = {_checkpoint_tensor_name(graph_name): graph_name for graph_name in targets}
    expected = mimi_checkpoint_inventory(codec)
    actual = set(reader.keys())
    missing = sorted(set(expected) - actual)
    unexpected = sorted(actual - set(expected))
    mismatched = sorted((
        name,
        reader.tensor_shape(name),
        expected[name][1],
    ) for name in set(expected) & actual if reader.tensor_shape(name) != expected[name][1])
    if missing or unexpected or mismatched:
        raise CheckpointCompatibilityError(
            "Mimi checkpoint does not match CSM's native codec graph: "
            f"missing={missing[:12]!r}, unexpected={unexpected[:12]!r}, "
            f"shape_mismatches={mismatched[:12]!r}.")
    return tuple(
        (checkpoint_name, checkpoint_to_graph[checkpoint_name]) for checkpoint_name in sorted(expected))


def load_mimi_checkpoint(
    codec: MimiModel,
    path: str | Path,
    *,
    require_official_inventory: bool = True,
) -> MimiCheckpointReport:
    """Validate the complete Mimi header, then stream-copy its tensors."""
    source = Path(path).expanduser().resolve()
    if source.suffix.lower() != ".safetensors":
        raise ValueError("Native Mimi checkpoints must use Safetensors.")
    with SafeTensorReader(source) as reader:
        tensor_map = _validate_mimi_layout(codec, reader)
        parameter_count = sum(reader.record(name).number_of_elements for name in reader.keys())
        fingerprint = _inventory_fingerprint(reader)
        report = MimiCheckpointReport(
            path=source,
            tensor_count=len(reader.keys()),
            parameter_count=parameter_count,
            header_fingerprint=fingerprint,
        )
        if require_official_inventory and (report.tensor_count != MIMI_CHECKPOINT_TENSOR_COUNT or
                                           report.parameter_count != MIMI_CHECKPOINT_PARAMETER_COUNT or
                                           report.header_fingerprint != MIMI_CHECKPOINT_HEADER_FINGERPRINT):
            raise CheckpointCompatibilityError("Mimi checkpoint is not the audited CSM codec artifact.")
        targets = codec.state_dict(keep_vars=True)
        with torch.no_grad():
            for checkpoint_name, graph_name in tensor_map:
                targets[graph_name].copy_(
                    reader.get_tensor(
                        checkpoint_name,
                        device=targets[graph_name].device,
                        dtype=targets[graph_name].dtype,
                    ), )
    codec.requires_grad_(False)
    codec.eval()
    return report


def load_mimi(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
    require_official_inventory: bool = True,
) -> MimiModel:
    codec = build_mimi(device=device)
    load_mimi_checkpoint(
        codec,
        path,
        require_official_inventory=require_official_inventory,
    )
    return codec


__all__ = [
    "MIMI_CARDINALITY",
    "MIMI_FRAME_RATE",
    "MIMI_NUM_CODEBOOKS",
    "MIMI_SAMPLE_RATE",
    "MimiCheckpointReport",
    "build_mimi",
    "load_mimi",
    "load_mimi_checkpoint",
    "mimi_checkpoint_inventory",
]
