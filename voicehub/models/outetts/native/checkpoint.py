"""Strict OuteTTS language-model and DAC checkpoint loading."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from voicehub.checkpointing import SafeTensorReader, save_safetensors
from voicehub.checkpointing.errors import CheckpointCompatibilityError
from voicehub.hub import read_json_file, write_json_file
from voicehub.models.causal_lm.native.checkpoint import (
    huggingface_causal_lm_tensor_mapping,
    open_causal_lm_tensor_source,
)
from voicehub.models.causal_lm.native.configuration import CausalLMConfig
from voicehub.models.dac.native.configuration import DacConfig
from voicehub.models.dac.native.modeling import DacModel

from .artifacts import OuteTTSArtifacts, OuteTTSDacArtifacts
from .metadata import NATIVE_OUTETTS_FORMAT, OUTETTS_CHECKPOINTS, OUTETTS_DAC
from .modeling import OuteTTSForCausalLM

OUTETTS_DAC_CONFIG = DacConfig(
    encoder_hidden_size=64,
    downsampling_ratios=(2, 4, 5, 8),
    decoder_hidden_size=1_536,
    n_codebooks=2,
    codebook_size=1_024,
    codebook_dim=8,
    sampling_rate=24_000,
)

_DTYPE_NAMES = {
    torch.bool: "BOOL",
    torch.uint8: "U8",
    torch.int8: "I8",
    torch.int16: "I16",
    torch.int32: "I32",
    torch.int64: "I64",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.float32: "F32",
    torch.float64: "F64",
}


def tensor_inventory_fingerprint(inventory: Mapping[str, tuple[str, tuple[int, ...]]], ) -> str:
    rows = [
        f"{name}|{dtype}|{'x'.join(str(item) for item in shape)}"
        for name, (dtype, shape) in sorted(inventory.items())
    ]
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def _expected_state_shapes(module: nn.Module) -> dict[str, tuple[int, ...]]:
    return {name: tuple(value.shape) for name, value in module.state_dict(keep_vars=True).items()}


def _published_lm_inventory(artifacts: OuteTTSArtifacts, ) -> tuple[int, int, str] | None:
    reference = OUTETTS_CHECKPOINTS.get(artifacts.source)
    if reference is None or artifacts.revision != reference["revision"]:
        return None
    return (
        int(reference["tensor_count"]),
        int(reference["value_count"]),
        str(reference["inventory_fingerprint"]),
    )


def _reader_inventory(reader) -> tuple[int, int, str | None]:
    names = tuple(reader.keys())
    value_count = sum(math.prod(reader.tensor_shape(name)) for name in names)
    if not isinstance(reader, SafeTensorReader):
        return len(names), value_count, None
    inventory = {name: (reader.record(name).dtype, reader.tensor_shape(name)) for name in names}
    return (
        len(names),
        value_count,
        tensor_inventory_fingerprint(inventory),
    )


def load_outetts_language_model(
    artifacts: OuteTTSArtifacts,
    *,
    device: str | torch.device,
    dtype: torch.dtype | None,
) -> tuple[OuteTTSForCausalLM, CausalLMConfig]:
    """Validate a complete safe header before assigning any LM tensor."""
    config_values = read_json_file(artifacts.config)
    config = CausalLMConfig.from_dict(config_values)
    if config.model_type not in {"llama", "qwen2", "qwen3"}:
        raise ValueError("OuteTTS 1.0 requires a dense Llama or Qwen causal LM.")
    reference = OUTETTS_CHECKPOINTS.get(artifacts.source)
    if (reference is not None and artifacts.revision == reference["revision"] and
            config.model_type != reference["family"]):
        raise ValueError(
            "Published OuteTTS checkpoint family does not match config: "
            f"expected {reference['family']!r}, found {config.model_type!r}.")
    with torch.device("meta"):
        model = OuteTTSForCausalLM(
            config,
            initialize=False,
            device="meta",
        )
    expected_shapes = _expected_state_shapes(model)
    mapping = dict(huggingface_causal_lm_tensor_mapping(config))
    # Mapping helper returns (source, target); assignment needs target->source.
    target_to_source = {target: source for source, target in huggingface_causal_lm_tensor_mapping(config)}
    if set(target_to_source) != set(expected_shapes):
        raise RuntimeError("OuteTTS causal-LM adapter does not cover the native graph.")
    with open_causal_lm_tensor_source(artifacts.checkpoint) as reader:
        published = _published_lm_inventory(artifacts)
        observed = _reader_inventory(reader)
        if published is not None:
            comparable = (observed if observed[2] is not None else (observed[0], observed[1], published[2]))
            if comparable != published:
                raise CheckpointCompatibilityError(
                    "Published OuteTTS tensor inventory verification failed: "
                    f"found={observed!r}, expected={published!r}.")
        available = set(reader.keys())
        consumed = set(mapping)
        permitted = {
            "lm_head.weight",
            *(
                name for name in available
                if name.endswith("rotary_emb.inv_freq") or name.endswith("rotary_emb.original_inv_freq")),
        }
        missing_sources = sorted(consumed - available)
        unexpected_sources = sorted(available - consumed - permitted)
        mismatches = []
        for target, source in target_to_source.items():
            if source not in available:
                continue
            checkpoint_shape = tuple(reader.tensor_shape(source))
            if checkpoint_shape != expected_shapes[target]:
                mismatches.append((target, checkpoint_shape, expected_shapes[target]))
        if missing_sources or unexpected_sources or mismatches:
            raise CheckpointCompatibilityError(
                "OuteTTS language-model checkpoint is incompatible: "
                f"missing={missing_sources!r}, "
                f"unexpected={unexpected_sources!r}, "
                f"shape_mismatches={mismatches!r}.")
        for target in sorted(target_to_source):
            source = target_to_source[target]
            value = reader.get_tensor(source)
            target_dtype = (dtype if dtype is not None and value.is_floating_point() else value.dtype)
            model.load_state_dict(
                {target: value.to(
                    device=device,
                    dtype=target_dtype,
                )},
                strict=False,
                assign=True,
            )
    if config.tie_word_embeddings:
        model.tie_weights()
    remaining = [name for name, value in model.state_dict().items() if value.device.type == "meta"]
    if remaining:
        raise CheckpointCompatibilityError(
            "OuteTTS LM loading left meta tensors: " + ", ".join(remaining[:12]))
    return model, config


def _native_dac_from_safetensors(
    artifacts: OuteTTSDacArtifacts,
    *,
    device: str | torch.device,
    dtype: torch.dtype | None,
) -> DacModel:
    if artifacts.config is None:
        raise ValueError("Native OuteTTS DAC is missing its configuration.")
    config_values = read_json_file(artifacts.config)
    checkpoint_format = config_values.get("voicehub_checkpoint_format")
    if checkpoint_format != "native-state-dict-v1":
        raise ValueError(
            "OuteTTS DAC Safetensors must declare "
            "`voicehub_checkpoint_format='native-state-dict-v1'`.")
    config = DacConfig.from_dict(config_values)
    if config != OUTETTS_DAC_CONFIG:
        raise ValueError("OuteTTS V3 requires the audited 24 kHz, two-codebook DAC config.")
    with torch.device("meta"):
        codec = DacModel(config)
    expected = _expected_state_shapes(codec)
    with SafeTensorReader(artifacts.checkpoint) as reader:
        names = set(reader.keys())
        missing = sorted(set(expected) - names)
        unexpected = sorted(names - set(expected))
        mismatches = [(name, reader.tensor_shape(name), expected[name])
                      for name in sorted(set(expected) & names)
                      if reader.tensor_shape(name) != expected[name]]
        if missing or unexpected or mismatches:
            raise CheckpointCompatibilityError(
                "OuteTTS native DAC checkpoint is incompatible: "
                f"missing={missing!r}, unexpected={unexpected!r}, "
                f"shape_mismatches={mismatches!r}.")
        for name in sorted(expected):
            value = reader.get_tensor(name)
            target_dtype = (dtype if dtype is not None and value.is_floating_point() else value.dtype)
            codec.load_state_dict(
                {name: value.to(
                    device=device,
                    dtype=target_dtype,
                )},
                strict=False,
                assign=True,
            )
    return codec


def _legacy_state(payload: Any) -> Mapping[str, Tensor]:
    if not isinstance(payload, Mapping):
        raise CheckpointCompatibilityError("Pinned OuteTTS DAC checkpoint root must be a mapping.")
    state = payload.get("state_dict")
    if not isinstance(state, Mapping) or not state:
        raise CheckpointCompatibilityError("Pinned OuteTTS DAC checkpoint requires a tensor `state_dict`.")
    if any(not isinstance(name, str) or not isinstance(value, Tensor) for name, value in state.items()):
        raise CheckpointCompatibilityError("Pinned OuteTTS DAC state must map names to tensors only.")
    return state


def _legacy_dac(
    artifacts: OuteTTSDacArtifacts,
    *,
    device: str | torch.device,
    dtype: torch.dtype | None,
) -> DacModel:
    try:
        payload = torch.load(
            artifacts.checkpoint,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError as error:  # pragma: no cover - old PyTorch
        raise RuntimeError(
            "OuteTTS DAC conversion requires PyTorch with "
            "`torch.load(weights_only=True)`.") from error
    state = _legacy_state(payload)
    with torch.device("meta"):
        codec = DacModel(OUTETTS_DAC_CONFIG)
    expected = _expected_state_shapes(codec)
    names = set(state)
    missing = sorted(set(expected) - names)
    unexpected = sorted(names - set(expected))
    mismatches = [(name, tuple(state[name].shape), expected[name]) for name in sorted(set(expected) & names)
                  if tuple(state[name].shape) != expected[name]]
    inventory = {
        name: (
            _DTYPE_NAMES.get(value.dtype, str(value.dtype)),
            tuple(value.shape),
        )
        for name, value in state.items()
    }
    observed = (
        len(state),
        sum(value.numel() for value in state.values()),
        tensor_inventory_fingerprint(inventory),
    )
    published = (
        OUTETTS_DAC["tensor_count"],
        OUTETTS_DAC["value_count"],
        OUTETTS_DAC["inventory_fingerprint"],
    )
    if missing or unexpected or mismatches or observed != published:
        raise CheckpointCompatibilityError(
            "Pinned OuteTTS DAC inventory verification failed: "
            f"missing={missing!r}, unexpected={unexpected!r}, "
            f"shape_mismatches={mismatches!r}, found={observed!r}, "
            f"expected={published!r}.")
    for name in sorted(expected):
        value = state[name]
        target_dtype = (dtype if dtype is not None and value.is_floating_point() else value.dtype)
        codec.load_state_dict(
            {name: value.to(
                device=device,
                dtype=target_dtype,
            )},
            strict=False,
            assign=True,
        )
    return codec


def load_outetts_dac(
    artifacts: OuteTTSDacArtifacts,
    *,
    device: str | torch.device,
    dtype: torch.dtype | None,
) -> DacModel:
    codec = (
        _legacy_dac(artifacts, device=device, dtype=dtype)
        if artifacts.legacy else _native_dac_from_safetensors(
            artifacts,
            device=device,
            dtype=dtype,
        ))
    codec.requires_grad_(False)
    codec.eval()
    return codec


def save_outetts_dac(codec: DacModel, directory: str | Path) -> Path:
    destination = Path(directory).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    values = codec.config.to_dict()
    values["voicehub_checkpoint_format"] = "native-state-dict-v1"
    values["voicehub_parent_format"] = NATIVE_OUTETTS_FORMAT
    write_json_file(destination / "config.json", values)
    save_safetensors(
        {
            name: value.detach()
            for name, value in codec.state_dict().items()
        },
        destination / "model.safetensors",
        metadata={
            "architecture": "dac",
            "format": "native-state-dict-v1",
            "producer": "voicehub",
        },
    )
    return destination


__all__ = [
    "OUTETTS_DAC_CONFIG",
    "load_outetts_dac",
    "load_outetts_language_model",
    "save_outetts_dac",
    "tensor_inventory_fingerprint",
]
