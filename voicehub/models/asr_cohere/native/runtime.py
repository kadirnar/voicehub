"""Native Cohere Transcribe checkpoint lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from voicehub.checkpointing import SafeTensorReader, ShardedSafeTensorReader, save_safetensors
from voicehub.hub import read_json_file, write_json_file
from voicehub.models.asr_cohere.native.artifacts import CohereAsrArtifacts, resolve_cohere_asr_artifacts
from voicehub.models.asr_cohere.native.checkpoint import (
    CohereAsrCheckpointAdapter,
    validate_published_cohere_asr_inventory,
)
from voicehub.models.asr_cohere.native.configuration import CohereAsrConfig
from voicehub.models.asr_cohere.native.modeling import CohereAsrForConditionalGeneration
from voicehub.models.asr_cohere.native.processing import CohereAsrProcessor
from voicehub.models.asr_cohere.native.tokenization import CohereAsrTokenizer

_DEFAULT_MAXIMUM_SHARD_BYTES = 1_000_000_000


def resolve_cohere_asr_dtype(
    value: str,
    device: str | torch.device,
) -> torch.dtype:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Cohere ASR dtype must be a non-empty string.")
    aliases = {
        "auto": "auto",
        "bf16": "bfloat16",
        "float": "float32",
        "fp16": "float16",
        "fp32": "float32",
        "half": "float16",
    }
    normalized = aliases.get(
        value.strip().lower(),
        value.strip().lower(),
    )
    resolved_device = torch.device(device)
    if normalized == "auto":
        if resolved_device.type == "cuda":
            return (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
        if resolved_device.type == "mps":
            return torch.float16
        return torch.float32
    dtype = getattr(torch, normalized, None)
    if (not isinstance(dtype, torch.dtype) or not dtype.is_floating_point):
        raise ValueError(f"Unsupported Cohere ASR dtype {value!r}.")
    if resolved_device.type == "cpu" and dtype == torch.float16:
        raise ValueError("Cohere ASR float16 execution is unsupported on CPU; use "
                         "float32 or bfloat16.")
    return dtype


def _validate_preprocessor(
    values: dict[str, Any],
    config: CohereAsrConfig,
) -> None:
    if not isinstance(values, dict):
        raise TypeError("Cohere ASR preprocessor config must be an object.")
    expected = {
        "feature_size": config.encoder_config.num_mel_bins,
        "sampling_rate": config.sample_rate,
        "n_fft": config.n_fft,
    }
    aliases = {
        "hop_length": ("hop_length", "n_window_stride"),
        "win_length": ("win_length", "n_window_size"),
    }
    for name, expected_value in expected.items():
        value = values.get(name)
        if value is not None and value != expected_value:
            raise ValueError(f"Cohere ASR preprocessor {name}={value!r}; expected "
                             f"{expected_value!r}.")
    for target, names in aliases.items():
        supplied = next(
            (values[name] for name in names if name in values),
            None,
        )
        expected_value = getattr(config, target)
        if supplied is not None and supplied != expected_value:
            raise ValueError(
                f"Cohere ASR preprocessor {target}={supplied!r}; expected "
                f"{expected_value!r}.")


def _generation_defaults(config: CohereAsrConfig) -> dict[str, Any]:
    return {
        "bos_token_id": config.bos_token_id,
        "decoder_start_token_id": config.decoder_start_token_id,
        "eos_token_id": config.eos_token_id,
        "pad_token_id": config.pad_token_id,
    }


def _validate_generation(
    values: dict[str, Any],
    config: CohereAsrConfig,
) -> None:
    if not isinstance(values, dict):
        raise TypeError("Cohere ASR generation config must be an object.")
    expected = _generation_defaults(config)
    for name, default in expected.items():
        if values.get(name, default) != default:
            raise ValueError(f"Cohere ASR generation {name} disagrees with the model.")


@dataclass
class CohereAsrRuntime:
    """Loaded native graph, processor, source, and generation metadata."""

    model: CohereAsrForConditionalGeneration
    processor: CohereAsrProcessor
    config: CohereAsrConfig
    artifacts: CohereAsrArtifacts
    generation_config: dict[str, Any]

    def prepare_for_training(self) -> CohereAsrRuntime:
        self.model.train()
        return self

    def prepare_for_inference(self) -> CohereAsrRuntime:
        self.model.eval()
        return self


def load_cohere_asr_runtime(
    source: str | Path,
    *,
    device: str | torch.device,
    compute_dtype: str = "auto",
    revision: str | None = None,
    cache_dir: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
    for_training: bool = False,
) -> CohereAsrRuntime:
    """Load an official or VoiceHub-exported Cohere ASR artifact."""
    artifacts = resolve_cohere_asr_artifacts(
        source,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    config_values = read_json_file(artifacts.config)
    config = CohereAsrConfig.from_dict(config_values)
    _validate_preprocessor(
        read_json_file(artifacts.preprocessor_config),
        config,
    )
    generation = read_json_file(artifacts.generation_config)
    _validate_generation(generation, config)
    dtype = resolve_cohere_asr_dtype(compute_dtype, device)
    with torch.device("meta"):
        model = CohereAsrForConditionalGeneration(
            config,
            initialize=False,
        )
    reader_type = (ShardedSafeTensorReader if artifacts.is_sharded else SafeTensorReader)
    with reader_type(artifacts.checkpoint) as reader:
        validate_published_cohere_asr_inventory(
            reader,
            source=artifacts.source,
            revision=artifacts.revision,
        )
        CohereAsrCheckpointAdapter().load_assign_streaming(
            model,
            reader,
            config,
            device=device,
            dtype=dtype,
            strict=True,
        )
    tokenizer = CohereAsrTokenizer.from_files(
        artifacts.tokenizer,
        artifacts.tokenizer_config,
    )
    processor = CohereAsrProcessor.from_files(
        featurizer=model.preprocessor.featurizer,
        config=config,
        tokenizer_path=artifacts.tokenizer,
        tokenizer_config_path=artifacts.tokenizer_config,
    )
    if tokenizer.assets.vocabulary != processor.tokenizer.assets.vocabulary:
        raise RuntimeError("Cohere ASR tokenizer construction was nondeterministic.")
    runtime = CohereAsrRuntime(
        model=model,
        processor=processor,
        config=config,
        artifacts=artifacts,
        generation_config=generation,
    )
    return (runtime.prepare_for_training() if for_training else runtime.prepare_for_inference())


def _validate_maximum_shard_bytes(value: int) -> None:
    if (isinstance(value, bool) or not isinstance(value, int) or value < 1):
        raise ValueError("`maximum_shard_bytes` must be a positive integer.")


def _save_state_dict(
    state_dict: dict[str, torch.Tensor],
    directory: Path,
    *,
    maximum_shard_bytes: int,
) -> Path:
    groups: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0
    total_bytes = 0
    for name in sorted(state_dict):
        tensor = state_dict[name]
        size = tensor.numel() * tensor.element_size()
        total_bytes += size
        if current and current_bytes + size > maximum_shard_bytes:
            groups.append(current)
            current = []
            current_bytes = 0
        current.append(name)
        current_bytes += size
    if current:
        groups.append(current)
    metadata = {
        "format": "pt",
        "voicehub_architecture": "cohere-asr",
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
        filename = (f"model-{index:05d}-of-{count:05d}.safetensors")
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


def save_cohere_asr_runtime(
    runtime: CohereAsrRuntime,
    directory: str | Path,
    *,
    state_dict: dict[str, torch.Tensor] | None = None,
    maximum_shard_bytes: int = _DEFAULT_MAXIMUM_SHARD_BYTES,
) -> Path:
    """Export a strict inference-ready native Safetensors directory."""
    if not isinstance(runtime, CohereAsrRuntime):
        raise TypeError("`runtime` must be CohereAsrRuntime.")
    _validate_maximum_shard_bytes(maximum_shard_bytes)
    destination = Path(directory).expanduser()
    model_state = runtime.model.state_dict()
    state = (dict(model_state) if state_dict is None else dict(state_dict))
    expected = set(model_state)
    if set(state) != expected:
        missing = sorted(expected - set(state))
        extra = sorted(set(state) - expected)
        raise ValueError(
            "Cohere ASR export requires the exact model namespace; "
            f"missing={missing[:5]!r}, extra={extra[:5]!r}.")
    for name, value in state.items():
        reference = model_state[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Cohere ASR export value {name!r} is not a tensor.")
        if value.layout is not torch.strided:
            raise TypeError(f"Cohere ASR export tensor {name!r} must use strided layout.")
        if value.is_complex():
            raise TypeError(f"Cohere ASR export tensor {name!r} cannot be complex.")
        if value.is_quantized:
            raise TypeError(f"Cohere ASR export tensor {name!r} cannot be quantized.")
        if tuple(value.shape) != tuple(reference.shape):
            raise ValueError(
                f"Cohere ASR export tensor {name!r} has shape "
                f"{tuple(value.shape)}, expected {tuple(reference.shape)}.")
        if value.dtype != reference.dtype:
            raise TypeError(
                f"Cohere ASR export tensor {name!r} has dtype {value.dtype}, "
                f"expected {reference.dtype}.")
        if value.device.type == "meta":
            raise ValueError(f"Cohere ASR export tensor {name!r} is not materialized.")
    if not torch.equal(state["transf_decoder._embedding.token_embedding.weight"],
                       state["log_softmax.mlp.layer0.weight"]):
        raise ValueError("Cohere ASR export requires tied input/output embeddings.")
    # Caller-provided state is fully validated before the destination exists.
    destination.mkdir(parents=True, exist_ok=True)
    _save_state_dict(
        state,
        destination,
        maximum_shard_bytes=maximum_shard_bytes,
    )
    values = runtime.config.to_dict()
    values["voicehub_checkpoint_format"] = "native-cohere-asr-v1"
    write_json_file(destination / "config.json", values)
    runtime.processor.save_pretrained(destination)
    write_json_file(
        destination / "generation_config.json",
        runtime.generation_config,
    )
    return destination


__all__ = [
    "CohereAsrRuntime",
    "load_cohere_asr_runtime",
    "resolve_cohere_asr_dtype",
    "save_cohere_asr_runtime",
]
