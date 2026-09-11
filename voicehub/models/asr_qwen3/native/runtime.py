"""Native Qwen3-ASR checkpoint lifecycle shared by inference and training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from voicehub.checkpointing import SafeTensorReader, ShardedSafeTensorReader, save_safetensors
from voicehub.hub import read_json_file, write_json_file
from voicehub.models.asr_qwen3.native.artifacts import Qwen3ASRArtifacts, resolve_qwen3_asr_artifacts
from voicehub.models.asr_qwen3.native.checkpoint import (
    Qwen3ASRCheckpointAdapter,
    validate_published_qwen3_asr_inventory,
)
from voicehub.models.asr_qwen3.native.configuration import Qwen3ASRArchitectureConfig
from voicehub.models.asr_qwen3.native.modeling import Qwen3ASRForConditionalGeneration
from voicehub.models.asr_qwen3.native.processing import SAMPLE_RATE, Qwen3ASRProcessor
from voicehub.models.asr_qwen3.native.tokenization import Qwen3ASRTokenizer

_MAXIMUM_EXPORT_SHARD_BYTES = 1_000_000_000


def resolve_qwen3_asr_dtype(
    dtype_name: str,
    device: str | torch.device,
) -> torch.dtype:
    if not isinstance(dtype_name, str) or not dtype_name.strip():
        raise ValueError("Qwen3-ASR dtype must be a non-empty string.")
    aliases = {
        "auto": "auto",
        "bf16": "bfloat16",
        "float": "float32",
        "fp16": "float16",
        "fp32": "float32",
        "half": "float16",
    }
    normalized = aliases.get(
        dtype_name.strip().lower(),
        dtype_name.strip().lower(),
    )
    resolved_device = torch.device(device)
    if normalized == "auto":
        if resolved_device.type == "cpu":
            return torch.float32
        if resolved_device.type == "mps":
            return torch.float16
        if resolved_device.type == "cuda":
            return (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
        return torch.float32
    dtype = getattr(torch, normalized, None)
    if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
        raise ValueError(f"Unsupported Qwen3-ASR dtype {dtype_name!r}.")
    if resolved_device.type == "cpu" and dtype == torch.float16:
        raise ValueError("Qwen3-ASR float16 execution is unsupported on CPU; use "
                         "float32 or bfloat16.")
    return dtype


def _validate_preprocessor(
    values: dict[str, Any],
    config: Qwen3ASRArchitectureConfig,
) -> None:
    expected = {
        "feature_size": config.audio_config.num_mel_bins,
        "sampling_rate": SAMPLE_RATE,
        "hop_length": 160,
        "n_fft": 400,
    }
    for name, expected_value in expected.items():
        value = values.get(name)
        if value is not None and value != expected_value:
            raise ValueError(
                f"Qwen3-ASR preprocessor {name!r} is {value!r}; expected "
                f"{expected_value!r}.")


@dataclass
class Qwen3ASRRuntime:
    """Loaded graph, processor, immutable source, and generation metadata."""

    model: Qwen3ASRForConditionalGeneration
    processor: Qwen3ASRProcessor
    config: Qwen3ASRArchitectureConfig
    artifacts: Qwen3ASRArtifacts
    generation_config: dict[str, Any]

    def prepare_for_training(self) -> Qwen3ASRRuntime:
        self.model.train()
        return self

    def prepare_for_inference(self) -> Qwen3ASRRuntime:
        self.model.eval()
        return self


def load_qwen3_asr_runtime(
    source: str | Path,
    *,
    device: str | torch.device,
    compute_dtype: str = "auto",
    revision: str | None = None,
    cache_dir: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
    for_training: bool = False,
) -> Qwen3ASRRuntime:
    """Load an official or portable Qwen3-ASR Safetensors artifact."""
    artifacts = resolve_qwen3_asr_artifacts(
        source,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    config_values = read_json_file(artifacts.config)
    config = Qwen3ASRArchitectureConfig.from_dict(config_values)
    if artifacts.preprocessor_config is not None:
        _validate_preprocessor(
            read_json_file(artifacts.preprocessor_config),
            config,
        )
    tokenizer = Qwen3ASRTokenizer.from_files(
        artifacts.vocab,
        artifacts.merges,
        artifacts.tokenizer_config,
    )
    if tokenizer.audio_token_id != config.audio_token_id:
        raise ValueError("Qwen3-ASR tokenizer/checkpoint audio token IDs disagree.")
    dtype = resolve_qwen3_asr_dtype(compute_dtype, device)
    with torch.device("meta"):
        model = Qwen3ASRForConditionalGeneration(
            config,
            initialize=False,
            tie_weights=False,
        )
    reader_type = (ShardedSafeTensorReader if artifacts.is_sharded else SafeTensorReader)
    with reader_type(artifacts.checkpoint) as reader:
        validate_published_qwen3_asr_inventory(
            reader,
            source=artifacts.source,
            revision=artifacts.revision,
        )
        Qwen3ASRCheckpointAdapter().load_assign_streaming(
            model,
            reader,
            config,
            device=device,
            dtype=dtype,
            strict=True,
        )
    processor = Qwen3ASRProcessor(
        config,
        tokenizer,
        preprocessor_config_path=artifacts.preprocessor_config,
        generation_config_path=artifacts.generation_config,
        chat_template_path=artifacts.chat_template,
    )
    generation = (
        read_json_file(artifacts.generation_config) if artifacts.generation_config is not None else {
            "do_sample": False,
            "eos_token_id": [151_643, 151_645],
            "pad_token_id": 151_643,
        })
    runtime = Qwen3ASRRuntime(
        model=model,
        processor=processor,
        config=config,
        artifacts=artifacts,
        generation_config=generation,
    )
    return (runtime.prepare_for_training() if for_training else runtime.prepare_for_inference())


def _save_sharded_state_dict(
    state_dict: dict[str, torch.Tensor],
    directory: Path,
    *,
    maximum_shard_bytes: int = _MAXIMUM_EXPORT_SHARD_BYTES,
) -> Path:
    if maximum_shard_bytes < 1:
        raise ValueError("`maximum_shard_bytes` must be positive.")
    groups: list[list[str]] = []
    current: list[str] = []
    current_size = 0
    total_size = 0
    for name in sorted(state_dict):
        tensor = state_dict[name]
        size = tensor.numel() * tensor.element_size()
        total_size += size
        if current and current_size + size > maximum_shard_bytes:
            groups.append(current)
            current = []
            current_size = 0
        current.append(name)
        current_size += size
    if current:
        groups.append(current)
    if len(groups) == 1:
        return save_safetensors(
            {name: state_dict[name]
             for name in groups[0]},
            directory / "model.safetensors",
            metadata={
                "format": "pt",
                "voicehub_architecture": "qwen3-asr",
            },
        )
    weight_map: dict[str, str] = {}
    count = len(groups)
    for index, group in enumerate(groups, start=1):
        filename = f"model-{index:05d}-of-{count:05d}.safetensors"
        save_safetensors(
            {name: state_dict[name]
             for name in group},
            directory / filename,
            metadata={
                "format": "pt",
                "voicehub_architecture": "qwen3-asr",
            },
        )
        weight_map.update({name: filename for name in group})
    index_path = directory / "model.safetensors.index.json"
    write_json_file(
        index_path,
        {
            "metadata": {
                "total_size": total_size
            },
            "weight_map": weight_map,
        },
    )
    return index_path


def save_qwen3_asr_runtime(
    runtime: Qwen3ASRRuntime,
    directory: str | Path,
    *,
    state_dict: dict[str, torch.Tensor] | None = None,
) -> Path:
    """Export an inference-ready official-schema native artifact."""
    if not isinstance(runtime, Qwen3ASRRuntime):
        raise TypeError("`runtime` must be Qwen3ASRRuntime.")
    target = Path(directory).expanduser()
    target.mkdir(parents=True, exist_ok=True)
    state = (dict(runtime.model.state_dict()) if state_dict is None else dict(state_dict))
    _save_sharded_state_dict(state, target)
    config_values = runtime.config.to_dict()
    config_values["voicehub_checkpoint_format"] = "native-qwen3-asr-v1"
    write_json_file(target / "config.json", config_values)
    runtime.processor.save_pretrained(target)
    if runtime.artifacts.generation_config is None:
        write_json_file(
            target / "generation_config.json",
            runtime.generation_config,
        )
    if runtime.artifacts.preprocessor_config is None:
        write_json_file(
            target / "preprocessor_config.json",
            {
                "chunk_length": 30,
                "feature_extractor_type": "VoiceHubWhisperFeatureExtractor",
                "feature_size": runtime.config.audio_config.num_mel_bins,
                "hop_length": 160,
                "n_fft": 400,
                "padding_side": "right",
                "padding_value": 0.0,
                "processor_class": "Qwen3ASRProcessor",
                "return_attention_mask": True,
                "sampling_rate": SAMPLE_RATE,
            },
        )
    return target


__all__ = [
    "Qwen3ASRRuntime",
    "load_qwen3_asr_runtime",
    "resolve_qwen3_asr_dtype",
    "save_qwen3_asr_runtime",
]
