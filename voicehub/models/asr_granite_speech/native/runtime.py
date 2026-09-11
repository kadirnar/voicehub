"""Native Granite Speech checkpoint lifecycle for inference and training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from voicehub.checkpointing import SafeTensorReader, ShardedSafeTensorReader, save_safetensors
from voicehub.hub import read_json_file, write_json_file
from voicehub.models.asr_granite_speech.native.artifacts import GraniteSpeechArtifacts, resolve_granite_speech_artifacts
from voicehub.models.asr_granite_speech.native.checkpoint import (
    GraniteSpeechCheckpointAdapter,
    native_granite_speech_tensor_shapes,
    validate_published_granite_speech_inventory,
)
from voicehub.models.asr_granite_speech.native.configuration import GraniteSpeechArchitectureConfig
from voicehub.models.asr_granite_speech.native.modeling import GraniteSpeechForConditionalGeneration
from voicehub.models.asr_granite_speech.native.processing import GraniteSpeechProcessor
from voicehub.models.asr_granite_speech.native.tokenization import GraniteSpeechTokenizer

_MAXIMUM_EXPORT_SHARD_BYTES = 1_000_000_000


def resolve_granite_speech_dtype(
    dtype_name: str,
    device: str | torch.device,
) -> torch.dtype:
    if not isinstance(dtype_name, str) or not dtype_name.strip():
        raise ValueError("Granite Speech dtype must be a non-empty string.")
    normalized = {
        "auto": "auto",
        "bf16": "bfloat16",
        "float": "float32",
        "fp16": "float16",
        "fp32": "float32",
        "half": "float16",
    }.get(dtype_name.strip().lower(),
          dtype_name.strip().lower())
    target = torch.device(device)
    if normalized == "auto":
        if target.type == "cuda":
            return (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
        if target.type == "mps":
            return torch.float16
        return torch.float32
    dtype = getattr(torch, normalized, None)
    if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
        raise ValueError(f"Unsupported Granite Speech dtype {dtype_name!r}.")
    if target.type == "cpu" and dtype == torch.float16:
        raise ValueError(
            "Granite Speech float16 execution is unsupported on CPU; use "
            "float32 or bfloat16.")
    return dtype


def _validate_preprocessor(
    values: dict[str, Any],
    config: GraniteSpeechArchitectureConfig,
) -> None:
    expected = {
        "sampling_rate": 16_000,
        "projector_window_size": config.window_size,
        "projector_downsample_rate": config.downsample_rate,
    }
    for name, expected_value in expected.items():
        value = values.get(name)
        if value is not None and value != expected_value:
            raise ValueError(
                f"Granite Speech preprocessor {name!r} is {value!r}; "
                f"expected {expected_value!r}.")
    mel = values.get("melspec_kwargs", {})
    if not isinstance(mel, dict):
        raise ValueError("Granite Speech `melspec_kwargs` must be a mapping.")
    mel_expected = {
        "sample_rate": 16_000,
        "n_fft": 512,
        "win_length": 400,
        "hop_length": 160,
        "n_mels": config.encoder_config.input_dim // 2,
    }
    for name, expected_value in mel_expected.items():
        value = mel.get(name)
        if value is not None and value != expected_value:
            raise ValueError(
                f"Granite Speech mel setting {name!r} is {value!r}; "
                f"expected {expected_value!r}.")


@dataclass
class GraniteSpeechRuntime:
    """Loaded graph, native processor, and immutable artifact identity."""

    model: GraniteSpeechForConditionalGeneration
    processor: GraniteSpeechProcessor
    config: GraniteSpeechArchitectureConfig
    artifacts: GraniteSpeechArtifacts
    generation_config: dict[str, Any]

    def prepare_for_training(self) -> GraniteSpeechRuntime:
        self.model.train()
        return self

    def prepare_for_inference(self) -> GraniteSpeechRuntime:
        self.model.eval()
        return self


def load_granite_speech_runtime(
    source: str | Path,
    *,
    device: str | torch.device,
    compute_dtype: str = "auto",
    revision: str | None = None,
    cache_dir: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
    for_training: bool = False,
) -> GraniteSpeechRuntime:
    """Load an official or VoiceHub-exported Safetensors artifact."""
    artifacts = resolve_granite_speech_artifacts(
        source,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    config = GraniteSpeechArchitectureConfig.from_dict(read_json_file(artifacts.config), )
    _validate_preprocessor(
        read_json_file(artifacts.preprocessor_config),
        config,
    )
    tokenizer = GraniteSpeechTokenizer.from_files(
        artifacts.tokenizer,
        tokenizer_config=artifacts.tokenizer_config,
        special_tokens_map=artifacts.special_tokens_map,
        added_tokens=artifacts.added_tokens,
        chat_template=artifacts.chat_template,
    )
    processor = GraniteSpeechProcessor(
        config,
        tokenizer,
        preprocessor_config_path=artifacts.preprocessor_config,
        processor_config_path=artifacts.processor_config,
    )
    dtype = resolve_granite_speech_dtype(
        compute_dtype,
        device,
    )
    with torch.device("meta"):
        model = GraniteSpeechForConditionalGeneration(
            config,
            initialize=False,
        )
    reader_type = (ShardedSafeTensorReader if artifacts.is_sharded else SafeTensorReader)
    with reader_type(artifacts.checkpoint) as reader:
        validate_published_granite_speech_inventory(
            reader,
            source=artifacts.source,
            revision=artifacts.revision,
        )
        GraniteSpeechCheckpointAdapter().load_assign_streaming(
            model,
            reader,
            config,
            device=device,
            dtype=dtype,
            strict=True,
        )
    generation_config = (
        read_json_file(artifacts.generation_config) if artifacts.generation_config is not None else {
            "do_sample": False,
            "eos_token_id": config.text_config.eos_token_id,
            "pad_token_id": config.text_config.pad_token_id,
        })
    runtime = GraniteSpeechRuntime(
        model=model,
        processor=processor,
        config=config,
        artifacts=artifacts,
        generation_config=generation_config,
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
                "voicehub_architecture": "granite-speech",
            },
        )
    weight_map: dict[str, str] = {}
    shard_count = len(groups)
    for index, group in enumerate(groups, start=1):
        filename = (f"model-{index:05d}-of-{shard_count:05d}.safetensors")
        save_safetensors(
            {name: state_dict[name]
             for name in group},
            directory / filename,
            metadata={
                "format": "pt",
                "voicehub_architecture": "granite-speech",
            },
        )
        weight_map.update({name: filename for name in group})
    index_path = directory / "model.safetensors.index.json"
    write_json_file(
        index_path,
        {
            "metadata": {
                "total_size": total_size,
            },
            "weight_map": weight_map,
        },
    )
    return index_path


def save_granite_speech_runtime(
    runtime: GraniteSpeechRuntime,
    directory: str | Path,
    *,
    state_dict: dict[str, torch.Tensor] | None = None,
) -> Path:
    """Export a safe, reloadable artifact in the official tensor schema."""
    if not isinstance(runtime, GraniteSpeechRuntime):
        raise TypeError("`runtime` must be GraniteSpeechRuntime.")
    state = (dict(runtime.model.state_dict()) if state_dict is None else dict(state_dict))
    expected_shapes = native_granite_speech_tensor_shapes(runtime.config, )
    expected = set(expected_shapes)
    received = set(state)
    missing = sorted(expected - received)
    unexpected = sorted(received - expected)
    mismatched = sorted(
        (
            name,
            tuple(state[name].shape),
            expected_shapes[name],
        ) for name in expected & received
        if (isinstance(state[name], torch.Tensor) and tuple(state[name].shape) != expected_shapes[name]))
    invalid = sorted(
        name for name, value in state.items() if (
            not isinstance(value, torch.Tensor) or value.device.type == "meta" or value.is_complex() or
            value.is_quantized or value.layout != torch.strided or
            (name.endswith("num_batches_tracked") and value.dtype != torch.int64) or
            (not name.endswith("num_batches_tracked") and not value.is_floating_point())))
    if missing or unexpected or mismatched or invalid:
        raise ValueError(
            "Granite Speech export state is incompatible: "
            f"missing={missing[:8]!r}, "
            f"unexpected={unexpected[:8]!r}, "
            f"shape_mismatches={mismatched[:8]!r}, "
            f"invalid_tensors={invalid[:8]!r}.")
    target = Path(directory).expanduser()
    target.mkdir(parents=True, exist_ok=True)
    _save_sharded_state_dict(state, target)
    config_values = runtime.config.to_dict()
    config_values["voicehub_checkpoint_format"] = ("native-granite-speech-v1")
    write_json_file(
        target / "config.json",
        config_values,
    )
    runtime.processor.save_pretrained(target)
    write_json_file(
        target / "generation_config.json",
        runtime.generation_config,
    )
    return target


__all__ = [
    "GraniteSpeechRuntime",
    "load_granite_speech_runtime",
    "resolve_granite_speech_dtype",
    "save_granite_speech_runtime",
]
