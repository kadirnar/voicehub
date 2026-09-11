"""Native Nemotron 3.5 ASR checkpoint lifecycle."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from voicehub.checkpointing import SafeTensorReader, save_safetensors
from voicehub.hub import read_json_file, write_json_file
from voicehub.models.asr_nemotron.native.artifacts import NemotronASRArtifacts, resolve_nemotron_asr_artifacts
from voicehub.models.asr_nemotron.native.checkpoint import (
    NemotronASRCheckpointAdapter,
    native_nemotron_asr_tensor_shapes,
    validate_published_nemotron_asr_inventory,
)
from voicehub.models.asr_nemotron.native.configuration import NemotronASRArchitectureConfig
from voicehub.models.asr_nemotron.native.modeling import Nemotron3_5ASRForRNNT
from voicehub.models.asr_nemotron.native.processing import NemotronASRProcessor

NATIVE_NEMOTRON_ASR_FORMAT = "voicehub-nemotron-3.5-rnnt-v1"


def validate_nemotron_asr_generation_config(
    values: Mapping[str, Any],
    config: NemotronASRArchitectureConfig,
) -> dict[str, Any]:
    """Validate generation settings that affect native RNN-T semantics."""
    if not isinstance(values, Mapping):
        raise TypeError("Nemotron generation configuration must be a mapping.")
    if not isinstance(config, NemotronASRArchitectureConfig):
        raise TypeError("`config` must be a NemotronASRArchitectureConfig.")
    validated = dict(values)
    required = {
        "blank_token_id": config.blank_token_id,
        "max_symbols_per_step": config.max_symbols_per_step,
    }
    for name, expected in required.items():
        value = validated.get(name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"Nemotron generation setting `{name}` must be an integer.")
        if value != expected:
            raise ValueError(
                f"Nemotron generation setting `{name}` is {value}; "
                f"the model graph requires {expected}.")

    lookahead = validated.get("num_lookahead_tokens")
    supported = config.encoder_config.supported_num_lookahead_tokens
    if (isinstance(lookahead, bool) or not isinstance(lookahead, int) or lookahead not in supported):
        raise ValueError(
            "Nemotron generation setting `num_lookahead_tokens` must be "
            f"one of {supported}; found {lookahead!r}.")

    declared_supported = validated.get("supported_num_lookahead_tokens")
    if declared_supported is not None:
        if (isinstance(declared_supported, (str, bytes)) or not isinstance(declared_supported,
                                                                           (list, tuple)) or
                any(isinstance(value, bool) or not isinstance(value, int) for value in declared_supported) or
                tuple(declared_supported) != supported):
            raise ValueError(
                "Nemotron generation setting "
                "`supported_num_lookahead_tokens` does not match the "
                f"model graph: expected {supported}.")
    declared_default = validated.get("default_num_lookahead_tokens")
    if declared_default is not None:
        if (isinstance(declared_default, bool) or not isinstance(declared_default, int) or
                declared_default != config.encoder_config.default_num_lookahead_tokens):
            raise ValueError(
                "Nemotron generation setting "
                "`default_num_lookahead_tokens` does not match the "
                "model graph.")
    return validated


def resolve_nemotron_asr_dtype(
    dtype_name: str,
    device: str | torch.device,
) -> torch.dtype:
    if not isinstance(dtype_name, str) or not dtype_name.strip():
        raise ValueError("Nemotron ASR dtype must be a non-empty string.")
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
        raise ValueError(f"Unsupported Nemotron ASR dtype {dtype_name!r}.")
    if target.type == "cpu" and dtype == torch.float16:
        raise ValueError("Nemotron ASR float16 execution is unsupported on CPU; use "
                         "float32 or bfloat16.")
    if target.type == "mps" and dtype == torch.bfloat16:
        raise ValueError("Nemotron ASR bfloat16 execution is unsupported on MPS.")
    return dtype


@dataclass
class NemotronASRRuntime:
    """Loaded model, native processor, and immutable artifact identity."""

    model: Nemotron3_5ASRForRNNT
    processor: NemotronASRProcessor
    config: NemotronASRArchitectureConfig
    artifacts: NemotronASRArtifacts
    generation_config: dict[str, Any]

    def prepare_for_training(self) -> NemotronASRRuntime:
        self.model.train()
        return self

    def prepare_for_inference(self) -> NemotronASRRuntime:
        self.model.eval()
        return self


def load_nemotron_asr_runtime(
    source: str | Path,
    *,
    device: str | torch.device,
    compute_dtype: str = "auto",
    revision: str | None = None,
    cache_dir: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
    for_training: bool = False,
) -> NemotronASRRuntime:
    """Load a coherent, safe Nemotron artifact with VoiceHub-owned code."""
    artifacts = resolve_nemotron_asr_artifacts(
        source,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    config = NemotronASRArchitectureConfig.from_dict(read_json_file(artifacts.config), )
    processor = NemotronASRProcessor.from_artifacts(
        config=config,
        tokenizer_json=artifacts.tokenizer,
        tokenizer_config=artifacts.tokenizer_config,
        processor_config=artifacts.processor_config,
    )
    dtype = resolve_nemotron_asr_dtype(compute_dtype, device)
    with torch.device("meta"):
        model = Nemotron3_5ASRForRNNT(
            config,
            initialize=False,
        )
    with SafeTensorReader(artifacts.checkpoint) as reader:
        validate_published_nemotron_asr_inventory(
            reader,
            source=artifacts.source,
            revision=artifacts.revision,
        )
        NemotronASRCheckpointAdapter().load_assign_streaming(
            model,
            reader,
            config,
            device=device,
            dtype=dtype,
            strict=True,
        )
    generation_config = validate_nemotron_asr_generation_config(
        (
            read_json_file(artifacts.generation_config) if artifacts.generation_config is not None else {
                "blank_token_id": config.blank_token_id,
                "max_symbols_per_step": config.max_symbols_per_step,
                "num_lookahead_tokens": config.encoder_config.default_num_lookahead_tokens,
            }),
        config,
    )
    processor.set_num_lookahead_tokens(generation_config["num_lookahead_tokens"], )
    runtime = NemotronASRRuntime(
        model=model,
        processor=processor,
        config=config,
        artifacts=artifacts,
        generation_config=generation_config,
    )
    return (runtime.prepare_for_training() if for_training else runtime.prepare_for_inference())


def _validate_export_state(
    state: dict[str, torch.Tensor],
    config: NemotronASRArchitectureConfig,
) -> None:
    expected_shapes = native_nemotron_asr_tensor_shapes(config)
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
        name for name, tensor in state.items() if (
            not isinstance(tensor, torch.Tensor) or tensor.device.type == "meta" or
            not tensor.is_floating_point() or tensor.is_complex() or tensor.is_quantized or
            tensor.layout != torch.strided))
    if missing or unexpected or mismatched or invalid:
        raise ValueError(
            "Nemotron ASR export state is incompatible: "
            f"missing={missing[:8]!r}, "
            f"unexpected={unexpected[:8]!r}, "
            f"shape_mismatches={mismatched[:8]!r}, "
            f"invalid_tensors={invalid[:8]!r}.")


def save_nemotron_asr_runtime(
    runtime: NemotronASRRuntime,
    directory: str | Path,
    *,
    state_dict: dict[str, torch.Tensor] | None = None,
) -> Path:
    """Export a safe, reloadable single-file Nemotron artifact."""
    if not isinstance(runtime, NemotronASRRuntime):
        raise TypeError("`runtime` must be a NemotronASRRuntime.")
    state = (dict(runtime.model.state_dict()) if state_dict is None else dict(state_dict))
    _validate_export_state(state, runtime.config)
    generation_config = validate_nemotron_asr_generation_config(
        runtime.generation_config,
        runtime.config,
    )
    target = Path(directory).expanduser()
    target.mkdir(parents=True, exist_ok=True)
    save_safetensors(
        state,
        target / "model.safetensors",
        metadata={
            "format": "pt",
            "voicehub_architecture": "nemotron-3.5-rnnt",
            "voicehub_checkpoint_format": NATIVE_NEMOTRON_ASR_FORMAT,
        },
    )
    config_values = runtime.config.to_dict()
    config_values["voicehub_checkpoint_format"] = (NATIVE_NEMOTRON_ASR_FORMAT)
    write_json_file(target / "config.json", config_values)
    runtime.processor.save_pretrained(target)
    write_json_file(
        target / "generation_config.json",
        generation_config,
    )
    return target


__all__ = [
    "NATIVE_NEMOTRON_ASR_FORMAT",
    "NemotronASRRuntime",
    "load_nemotron_asr_runtime",
    "resolve_nemotron_asr_dtype",
    "save_nemotron_asr_runtime",
    "validate_nemotron_asr_generation_config",
]
