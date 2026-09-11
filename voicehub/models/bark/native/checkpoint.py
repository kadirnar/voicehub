"""Strict Bark checkpoint validation, conversion, export, and reload."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from voicehub.checkpointing import SafeTensorReader, save_safetensors

from .configuration import BarkArchitectureConfig, BarkGenerationConfig
from .metadata import (
    BARK_CHECKPOINT_REVISION,
    BARK_CHECKPOINT_SHA256,
    BARK_CHECKPOINT_SIZE,
    BARK_INVENTORY_FINGERPRINT,
    BARK_NATIVE_FORMAT,
    BARK_STATE_VALUES,
    BARK_TENSOR_COUNT,
)
from .modeling import BarkModel

_CONFIG_METADATA = "voicehub_config"
_GENERATION_METADATA = "voicehub_generation_config"
_DTYPES = {
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


def _provider_name(internal_name: str) -> str:
    if not internal_name.startswith("codec_model."):
        return internal_name
    name = internal_name
    name = name.replace("codec_model.encoder.model.", "codec_model.encoder.layers.", 1)
    name = name.replace("codec_model.decoder.model.", "codec_model.decoder.layers.", 1)
    name = name.replace(
        "codec_model.quantizer.vq.layers.",
        "codec_model.quantizer.layers.",
        1,
    )
    name = name.replace("._codebook.", ".codebook.")
    name = name.replace(".convtr.convtr.", ".conv.")
    name = name.replace(".conv.conv.", ".conv.")
    # The 2023 checkpoint predates torch parametrizations and publishes the
    # legacy weight-normalization namespace verbatim.
    return name


def provider_to_internal_names(model: BarkModel) -> dict[str, str]:
    """Return the audited bijection from release names to native state."""
    mapping: dict[str, str] = {}
    for internal_name in model.state_dict():
        provider_name = _provider_name(internal_name)
        if provider_name in mapping:
            raise RuntimeError(f"Bark checkpoint mapping collides at {provider_name!r}.")
        mapping[provider_name] = internal_name
    return mapping


def provider_state_dict(model: BarkModel) -> dict[str, Tensor]:
    """Expose native state under the exact pinned checkpoint namespace."""
    state = model.state_dict()
    return {
        provider_name: state[internal_name]
        for provider_name, internal_name in provider_to_internal_names(model).items()
    }


def tensor_inventory_fingerprint(tensors: Mapping[str, Tensor]) -> str:
    rows: list[str] = []
    for name in sorted(tensors):
        tensor = tensors[name]
        if not isinstance(name, str) or not name:
            raise ValueError("Bark tensor names must be non-empty strings.")
        if not isinstance(tensor, Tensor):
            raise TypeError(f"Bark state entry {name!r} is not a tensor.")
        try:
            dtype = _DTYPES[tensor.dtype]
        except KeyError as error:
            raise TypeError(f"Unsupported Bark tensor dtype {tensor.dtype}.") from error
        shape = "x".join(str(item) for item in tensor.shape)
        rows.append(f"{name}|{dtype}|{shape}")
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def verify_native_graph_contract(model: BarkModel) -> None:
    """Assert the native graph still matches the audited release inventory."""
    state = provider_state_dict(model)
    count = len(state)
    values = sum(tensor.numel() for tensor in state.values())
    fingerprint = tensor_inventory_fingerprint(state)
    if (count != BARK_TENSOR_COUNT or values != BARK_STATE_VALUES or
            fingerprint != BARK_INVENTORY_FINGERPRINT):
        raise RuntimeError(
            "The native Bark graph no longer matches the pinned checkpoint "
            f"contract (count={count}, values={values}, "
            f"fingerprint={fingerprint}).")


def _validate_shapes(
    names: set[str],
    shape_of: Any,
    model: BarkModel,
    *,
    enforce_release_contract: bool = False,
) -> dict[str, str]:
    mapping = provider_to_internal_names(model)
    expected_names = set(mapping)
    if names != expected_names:
        raise ValueError(
            "Bark checkpoint namespace mismatch "
            f"(missing={sorted(expected_names - names)!r}, "
            f"unexpected={sorted(names - expected_names)!r}).")
    internal_state = model.state_dict()
    mismatches = {
        name: (
            tuple(shape_of(name)),
            tuple(internal_state[mapping[name]].shape),
        )
        for name in sorted(expected_names)
        if tuple(shape_of(name)) != tuple(internal_state[mapping[name]].shape)
    }
    if mismatches:
        raise ValueError(f"Bark checkpoint tensor shape mismatch: {mismatches}.")
    if enforce_release_contract:
        if len(names) != BARK_TENSOR_COUNT:
            raise ValueError("Bark checkpoint tensor count mismatch.")
        values = sum(math.prod(shape_of(name)) for name in names)
        if values != BARK_STATE_VALUES:
            raise ValueError("Bark checkpoint state-value count mismatch.")
    return mapping


def _copy_safetensors(reader: SafeTensorReader, model: BarkModel) -> None:
    mapping = _validate_shapes(
        set(reader.keys()),
        reader.tensor_shape,
        model,
    )
    state = model.state_dict()
    with torch.no_grad():
        for provider_name in sorted(mapping):
            target = state[mapping[provider_name]]
            source = reader.get_tensor(
                provider_name,
                device=target.device,
                dtype=target.dtype,
            )
            if provider_name.startswith("fine_acoustics.lm_heads."):
                # These heads are aliases of input codebook embeddings. The
                # release serializes both names; reject a corrupted artifact
                # instead of silently letting the second copy win.
                if not torch.equal(source, target):
                    raise ValueError(
                        f"Tied Bark tensor {provider_name!r} disagrees with "
                        "its input embedding.")
                continue
            target.copy_(source)


def load_bark_safetensors(
    model: BarkModel,
    checkpoint: str | Path,
) -> BarkModel:
    """Stream a complete safe Bark artifact into an existing graph."""
    if not isinstance(model, BarkModel):
        raise TypeError("`model` must be a BarkModel.")
    path = Path(checkpoint).expanduser().resolve()
    if path.suffix.lower() != ".safetensors":
        raise ValueError("Native Bark checkpoints must use Safetensors.")
    with SafeTensorReader(path) as reader:
        _copy_safetensors(reader, model)
    return model


def save_bark_safetensors(
    model: BarkModel,
    checkpoint: str | Path,
    *,
    metadata: Mapping[str, str] | None = None,
) -> Path:
    """Export a complete provider-free Bark artifact."""
    if not isinstance(model, BarkModel):
        raise TypeError("`model` must be a BarkModel.")
    path = Path(checkpoint).expanduser()
    if path.suffix.lower() != ".safetensors":
        raise ValueError("Bark export path must end with `.safetensors`.")
    values = {
        "architecture":
        "bark",
        "format":
        BARK_NATIVE_FORMAT,
        "source_revision":
        BARK_CHECKPOINT_REVISION,
        _CONFIG_METADATA:
        json.dumps(
            model.config.to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
        _GENERATION_METADATA:
        json.dumps(
            model.generation_config.to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
    }
    extra = dict(metadata or {})
    conflicts = set(values) & set(extra)
    if conflicts:
        raise ValueError("Bark export metadata cannot override reserved keys "
                         f"{sorted(conflicts)!r}.")
    values.update(extra)
    output = save_safetensors(
        {
            name: tensor.detach()
            for name, tensor in provider_state_dict(model).items()
        },
        path,
        metadata=values,
    )
    with SafeTensorReader(output) as reader:
        _validate_shapes(
            set(reader.keys()),
            reader.tensor_shape,
            model,
        )
    return output


def load_bark_model_from_safetensors(
    checkpoint: str | Path,
    *,
    config: BarkArchitectureConfig | None = None,
    generation_config: BarkGenerationConfig | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
) -> BarkModel:
    """Reconstruct a Bark graph from safe metadata or explicit config."""
    path = Path(checkpoint).expanduser().resolve()
    with SafeTensorReader(path) as reader:
        if config is None:
            encoded = reader.metadata.get(_CONFIG_METADATA)
            if encoded is None:
                raise ValueError("Bark Safetensors is missing its construction config.")
            try:
                raw_config = json.loads(encoded)
            except json.JSONDecodeError as error:
                raise ValueError("Bark Safetensors contains invalid config JSON.") from error
            config = BarkArchitectureConfig.from_dict(raw_config)
        if generation_config is None:
            encoded_generation = reader.metadata.get(_GENERATION_METADATA)
            if encoded_generation is None:
                raise ValueError("Bark Safetensors is missing its generation config.")
            try:
                raw_generation = json.loads(encoded_generation)
            except json.JSONDecodeError as error:
                raise ValueError("Bark Safetensors contains invalid generation JSON.") from error
            generation_config = BarkGenerationConfig.from_dict(raw_generation)
    model = BarkModel(config, generation_config=generation_config)
    if dtype is not None:
        model = model.to(dtype=dtype)
    model = model.to(device=device)
    return load_bark_safetensors(model, path)


def _verify_official_file(path: Path) -> str:
    size = path.stat().st_size
    if size != BARK_CHECKPOINT_SIZE:
        raise ValueError(f"Pinned Bark checkpoint size is {size}, expected "
                         f"{BARK_CHECKPOINT_SIZE}.")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    actual = digest.hexdigest()
    if actual != BARK_CHECKPOINT_SHA256:
        raise ValueError(
            f"Pinned Bark checkpoint SHA-256 is {actual}, expected "
            f"{BARK_CHECKPOINT_SHA256}.")
    return actual


def convert_official_bark_checkpoint(
    source: str | Path,
    destination: str | Path,
    *,
    config: BarkArchitectureConfig,
    generation_config: BarkGenerationConfig,
    trust_official_pickle: bool = False,
) -> Path:
    """Convert the exact pinned legacy archive to native Safetensors."""
    if not trust_official_pickle:
        raise PermissionError(
            "The pinned Bark release is a legacy pickle archive. Pass "
            "`trust_official_pickle=True` only after accepting that exact "
            "digest-pinned source; VoiceHub still uses `weights_only=True` "
            "and validates size, digest, namespace, shapes, and inventory.")
    path = Path(source).expanduser().resolve()
    _verify_official_file(path)
    try:
        state = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError("Could not read the pinned Bark tensor archive.") from error
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Pinned Bark archive did not contain a state mapping.")
    if any(not isinstance(name, str) or not isinstance(value, Tensor) for name, value in state.items()):
        raise TypeError("Pinned Bark archive must map names to tensors only.")
    with torch.device("meta"):
        graph = BarkModel(config, generation_config=generation_config)
    _validate_shapes(
        set(state),
        lambda name: tuple(state[name].shape),
        graph,
        enforce_release_contract=True,
    )
    fingerprint = tensor_inventory_fingerprint(state)
    if fingerprint != BARK_INVENTORY_FINGERPRINT:
        raise ValueError("Pinned Bark archive tensor inventory fingerprint mismatch.")
    return save_safetensors(
        {
            name: tensor.detach()
            for name, tensor in state.items()
        },
        destination,
        metadata={
            "architecture":
            "bark",
            "format":
            BARK_NATIVE_FORMAT,
            "source_revision":
            BARK_CHECKPOINT_REVISION,
            "source_sha256":
            BARK_CHECKPOINT_SHA256,
            _CONFIG_METADATA:
            json.dumps(
                config.to_dict(),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
            _GENERATION_METADATA:
            json.dumps(
                generation_config.to_dict(),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
        },
    )


def load_official_bark_checkpoint(
    model: BarkModel,
    source: str | Path,
    *,
    trust_official_pickle: bool = False,
) -> BarkModel:
    """Load the exact digest-pinned release without a provider runtime."""
    if not isinstance(model, BarkModel):
        raise TypeError("`model` must be a BarkModel.")
    if not trust_official_pickle:
        raise PermissionError(
            "Loading the Bark legacy archive requires "
            "`trust_official_pickle=True` for the exact pinned artifact.")
    path = Path(source).expanduser().resolve()
    _verify_official_file(path)
    try:
        state = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError("Could not read the pinned Bark tensor archive.") from error
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Pinned Bark archive did not contain a state mapping.")
    if any(not isinstance(name, str) or not isinstance(value, Tensor) for name, value in state.items()):
        raise TypeError("Pinned Bark archive must map names to tensors only.")
    mapping = _validate_shapes(
        set(state),
        lambda name: tuple(state[name].shape),
        model,
        enforce_release_contract=True,
    )
    fingerprint = tensor_inventory_fingerprint(state)
    if fingerprint != BARK_INVENTORY_FINGERPRINT:
        raise ValueError("Pinned Bark archive tensor inventory fingerprint mismatch.")
    internal = {mapping[name]: tensor for name, tensor in state.items()}
    model.load_state_dict(internal, strict=True)
    return model


__all__ = [
    "convert_official_bark_checkpoint",
    "load_official_bark_checkpoint",
    "load_bark_model_from_safetensors",
    "load_bark_safetensors",
    "provider_state_dict",
    "provider_to_internal_names",
    "save_bark_safetensors",
    "tensor_inventory_fingerprint",
    "verify_native_graph_contract",
]
