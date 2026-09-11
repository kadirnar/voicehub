"""Native Parakeet TDT checkpoint lifecycle for inference and training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from voicehub.checkpointing import SafeTensorReader, ShardedSafeTensorReader, save_safetensors
from voicehub.hub import read_json_file, write_json_file
from voicehub.models.asr_parakeet_tdt.native.artifacts import ParakeetTDTArtifacts, resolve_parakeet_tdt_artifacts
from voicehub.models.asr_parakeet_tdt.native.checkpoint import (
    ParakeetTDTCheckpointAdapter,
    validate_published_parakeet_tdt_inventory,
)
from voicehub.models.asr_parakeet_tdt.native.configuration import ParakeetTDTConfig
from voicehub.models.asr_parakeet_tdt.native.modeling import ParakeetForTDT
from voicehub.models.asr_parakeet_tdt.native.processing import ParakeetProcessor

_DEFAULT_MAXIMUM_SHARD_BYTES = 1_000_000_000


def resolve_parakeet_tdt_dtype(
    value: str,
    device: str | torch.device,
) -> torch.dtype:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Parakeet TDT dtype must be a non-empty string.")
    aliases = {
        "auto": "auto",
        "bf16": "bfloat16",
        "float": "float32",
        "fp16": "float16",
        "fp32": "float32",
        "half": "float16",
    }
    normalized = aliases.get(value.strip().lower(), value.strip().lower())
    resolved_device = torch.device(device)
    if normalized == "auto":
        if resolved_device.type == "cpu":
            return torch.float32
        if resolved_device.type == "cuda":
            return (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
        if resolved_device.type == "mps":
            return torch.float16
        return torch.float32
    dtype = getattr(torch, normalized, None)
    if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
        raise ValueError(f"Unsupported Parakeet TDT dtype {value!r}.")
    if resolved_device.type == "cpu" and dtype == torch.float16:
        raise ValueError("Parakeet TDT float16 execution is unsupported on CPU; use "
                         "float32 or bfloat16.")
    return dtype


def _validate_processor(
    processor: ParakeetProcessor,
    config: ParakeetTDTConfig,
) -> None:
    feature_extractor = processor.feature_extractor
    if feature_extractor.feature_size != config.encoder_config.num_mel_bins:
        raise ValueError(
            "Parakeet processor/model mel dimensions disagree: "
            f"{feature_extractor.feature_size} and "
            f"{config.encoder_config.num_mel_bins}.")
    if processor.subsampling_factor != config.encoder_config.subsampling_factor:
        raise ValueError("Parakeet processor/model subsampling factors disagree.")
    tokenizer = processor.tokenizer
    if tokenizer.token_id_space_size != config.vocab_size:
        raise ValueError("Parakeet tokenizer/checkpoint vocabulary sizes disagree.")
    expected_ids = {
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "blank_token_id": tokenizer.blank_token_id,
    }
    for name, value in expected_ids.items():
        if getattr(config, name) != value:
            raise ValueError(f"Parakeet tokenizer/checkpoint {name} values disagree.")


def _generation_defaults(config: ParakeetTDTConfig) -> dict[str, Any]:
    return {
        "decoder_start_token_id": config.blank_token_id,
        "eos_token_id": config.eos_token_id,
        "pad_token_id": config.pad_token_id,
        "suppress_tokens": list(range(
            config.vocab_size,
            config.vocab_size + len(config.durations),
        )),
    }


def _validate_generation(
    values: dict[str, Any],
    config: ParakeetTDTConfig,
) -> None:
    expected = _generation_defaults(config)
    for name in ("decoder_start_token_id", "eos_token_id", "pad_token_id"):
        if values.get(name, expected[name]) != expected[name]:
            raise ValueError(f"Parakeet generation {name} disagrees with the checkpoint.")
    suppressed = values.get("suppress_tokens", expected["suppress_tokens"])
    if tuple(suppressed) != tuple(expected["suppress_tokens"]):
        raise ValueError("Parakeet generation duration-logit suppression is incoherent.")


@dataclass
class ParakeetTDTRuntime:
    """Loaded native graph, processor, source, and decoding metadata."""

    model: ParakeetForTDT
    processor: ParakeetProcessor
    config: ParakeetTDTConfig
    artifacts: ParakeetTDTArtifacts
    generation_config: dict[str, Any]

    def prepare_for_training(self) -> ParakeetTDTRuntime:
        self.model.train()
        return self

    def prepare_for_inference(self) -> ParakeetTDTRuntime:
        self.model.eval()
        return self


def load_parakeet_tdt_runtime(
    source: str | Path,
    *,
    device: str | torch.device,
    compute_dtype: str = "auto",
    revision: str | None = None,
    cache_dir: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
    for_training: bool = False,
) -> ParakeetTDTRuntime:
    """Load an official or VoiceHub-exported Parakeet TDT artifact."""
    artifacts = resolve_parakeet_tdt_artifacts(
        source,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    config = ParakeetTDTConfig.from_dict(read_json_file(artifacts.config))
    processor = ParakeetProcessor.from_files(
        artifacts.processor_config,
        artifacts.tokenizer,
        artifacts.tokenizer_config,
    )
    _validate_processor(processor, config)
    generation = (
        read_json_file(artifacts.generation_config)
        if artifacts.generation_config is not None else _generation_defaults(config))
    _validate_generation(generation, config)
    dtype = resolve_parakeet_tdt_dtype(compute_dtype, device)
    with torch.device("meta"):
        model = ParakeetForTDT(config, initialize=False)
    reader_type = (ShardedSafeTensorReader if artifacts.is_sharded else SafeTensorReader)
    with reader_type(artifacts.checkpoint) as reader:
        validate_published_parakeet_tdt_inventory(
            reader,
            source=artifacts.source,
            revision=artifacts.revision,
        )
        ParakeetTDTCheckpointAdapter().load_assign_streaming(
            model,
            reader,
            config,
            device=device,
            dtype=dtype,
            strict=True,
        )
    runtime = ParakeetTDTRuntime(
        model=model,
        processor=processor,
        config=config,
        artifacts=artifacts,
        generation_config=generation,
    )
    return (runtime.prepare_for_training() if for_training else runtime.prepare_for_inference())


def _save_state_dict(
    state_dict: dict[str, torch.Tensor],
    directory: Path,
    *,
    maximum_shard_bytes: int,
) -> Path:
    if (isinstance(maximum_shard_bytes, bool) or not isinstance(maximum_shard_bytes, int) or
            maximum_shard_bytes < 1):
        raise ValueError("`maximum_shard_bytes` must be a positive integer.")
    groups: list[list[str]] = []
    group: list[str] = []
    group_bytes = 0
    total_bytes = 0
    for name in sorted(state_dict):
        tensor = state_dict[name]
        size = tensor.numel() * tensor.element_size()
        total_bytes += size
        if group and group_bytes + size > maximum_shard_bytes:
            groups.append(group)
            group = []
            group_bytes = 0
        group.append(name)
        group_bytes += size
    if group:
        groups.append(group)
    metadata = {
        "format": "pt",
        "voicehub_architecture": "parakeet-tdt",
    }
    if len(groups) == 1:
        return save_safetensors(
            {name: state_dict[name]
             for name in groups[0]},
            directory / "model.safetensors",
            metadata=metadata,
        )
    weight_map: dict[str, str] = {}
    count = len(groups)
    for index, names in enumerate(groups, start=1):
        filename = f"model-{index:05d}-of-{count:05d}.safetensors"
        save_safetensors(
            {name: state_dict[name]
             for name in names},
            directory / filename,
            metadata=metadata,
        )
        weight_map.update({name: filename for name in names})
    index_path = directory / "model.safetensors.index.json"
    write_json_file(
        index_path,
        {
            "metadata": {
                "total_size": total_bytes
            },
            "weight_map": weight_map,
        },
    )
    return index_path


def save_parakeet_tdt_runtime(
    runtime: ParakeetTDTRuntime,
    directory: str | Path,
    *,
    state_dict: dict[str, torch.Tensor] | None = None,
    maximum_shard_bytes: int = _DEFAULT_MAXIMUM_SHARD_BYTES,
) -> Path:
    """Export a strict, inference-ready native Safetensors artifact."""
    if not isinstance(runtime, ParakeetTDTRuntime):
        raise TypeError("`runtime` must be ParakeetTDTRuntime.")
    destination = Path(directory).expanduser()
    state = (dict(runtime.model.state_dict()) if state_dict is None else dict(state_dict))
    model_state = runtime.model.state_dict()
    expected = set(model_state)
    if set(state) != expected:
        missing = sorted(expected - set(state))
        extra = sorted(set(state) - expected)
        raise ValueError(
            "Parakeet TDT export requires the exact model namespace; "
            f"missing={missing[:5]!r}, extra={extra[:5]!r}.")
    for name, value in state.items():
        reference = model_state[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Parakeet TDT export value {name!r} is not a tensor.")
        if value.layout is not torch.strided:
            raise TypeError(f"Parakeet TDT export tensor {name!r} must use strided layout.")
        if value.is_complex():
            raise TypeError(f"Parakeet TDT export tensor {name!r} cannot be complex.")
        if value.is_quantized:
            raise TypeError(f"Parakeet TDT export tensor {name!r} cannot be quantized.")
        if tuple(value.shape) != tuple(reference.shape):
            raise ValueError(
                f"Parakeet TDT export tensor {name!r} has shape "
                f"{tuple(value.shape)}, expected {tuple(reference.shape)}.")
        if value.dtype != reference.dtype:
            raise TypeError(
                f"Parakeet TDT export tensor {name!r} has dtype {value.dtype}, "
                f"expected {reference.dtype}.")
        if value.device.type == "meta":
            raise ValueError(f"Parakeet TDT export tensor {name!r} is not materialized.")
    # Do not create a destination until the entire caller-supplied state has
    # passed validation. A rejected export therefore cannot leave a partial
    # artifact directory.
    destination.mkdir(parents=True, exist_ok=True)
    _save_state_dict(
        state,
        destination,
        maximum_shard_bytes=maximum_shard_bytes,
    )
    values = runtime.config.to_dict()
    values["voicehub_checkpoint_format"] = "native-parakeet-tdt-v1"
    write_json_file(destination / "config.json", values)
    runtime.processor.save_pretrained(destination)
    write_json_file(
        destination / "generation_config.json",
        runtime.generation_config,
    )
    return destination


__all__ = [
    "ParakeetTDTRuntime",
    "load_parakeet_tdt_runtime",
    "resolve_parakeet_tdt_dtype",
    "save_parakeet_tdt_runtime",
]
