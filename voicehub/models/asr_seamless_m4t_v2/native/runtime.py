"""Native SeamlessM4T-v2 S2T loading, execution, and portable export."""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from voicehub.checkpointing import SafeTensorReader, ShardedSafeTensorReader, save_safetensors
from voicehub.hub import read_json_file, write_json_file
from voicehub.models.asr_seamless_m4t_v2.native.artifacts import (
    SeamlessM4Tv2S2TArtifacts,
    resolve_seamless_m4t_v2_artifacts,
)
from voicehub.models.asr_seamless_m4t_v2.native.checkpoint import (
    SeamlessM4Tv2S2TCheckpointAdapter,
    native_seamless_m4t_v2_tensor_shapes,
    validate_published_seamless_m4t_v2_inventory,
)
from voicehub.models.asr_seamless_m4t_v2.native.configuration import SeamlessM4Tv2S2TConfig
from voicehub.models.asr_seamless_m4t_v2.native.modeling import SeamlessM4Tv2ForSpeechToText
from voicehub.models.asr_seamless_m4t_v2.native.processing import SeamlessM4Tv2Processor
from voicehub.models.asr_seamless_m4t_v2.native.tokenization import SEAMLESS_M4T_V2_LANGUAGE_TO_ID

_DEFAULT_MAXIMUM_SHARD_BYTES = 1_000_000_000
_ALIASES = frozenset({
    "lm_head.weight",
    "text_decoder.embed_tokens.weight",
})


def _validate_architecture(values: dict[str, Any]) -> None:
    if not isinstance(values, dict):
        raise TypeError("SeamlessM4T-v2 model configuration must be an object.")
    model_type = str(values.get("model_type", "")).strip().lower()
    if model_type != "seamless_m4t_v2":
        raise ValueError("Native SeamlessM4T-v2 requires `model_type='seamless_m4t_v2'`.")
    if "auto_map" in values:
        raise ValueError("Native SeamlessM4T-v2 rejects remote-code `auto_map` metadata.")
    architectures = values.get('architectures', ())
    if isinstance(architectures, str):
        architectures = (architectures, )
    if (not isinstance(architectures, (tuple, list)) or isinstance(architectures, (str, bytes))):
        raise TypeError("`architectures` must be a sequence.")
    supported = {
        "SeamlessM4Tv2ForSpeechToText",
        "SeamlessM4Tv2Model",
    }
    if architectures and not any(str(value) in supported for value in architectures):
        raise ValueError("Native SeamlessM4T-v2 accepts only the audited unified or S2T "
                         "architecture.")


def resolve_seamless_m4t_v2_dtype(
    value: str,
    device: str | torch.device,
) -> torch.dtype:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("S2T dtype must be a non-empty string.")
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
        if resolved_device.type == "cuda":
            return (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
        if resolved_device.type == "mps":
            return torch.float16
        return torch.float32
    dtype = getattr(torch, normalized, None)
    if (not isinstance(dtype, torch.dtype) or not dtype.is_floating_point):
        raise ValueError(f"Unsupported S2T dtype {value!r}.")
    if resolved_device.type == "cpu" and dtype == torch.float16:
        raise ValueError("SeamlessM4T-v2 float16 execution is unsupported on CPU.")
    return dtype


def _validate_preprocessor(
    values: dict[str, Any],
    config: SeamlessM4Tv2S2TConfig,
) -> None:
    if not isinstance(values, dict):
        raise TypeError("Preprocessor configuration must be an object.")
    expected = {
        "feature_size": config.num_mel_bins,
        "num_mel_bins": config.num_mel_bins,
        "sampling_rate": config.sampling_rate,
        "stride": config.feature_stride,
    }
    for name, wanted in expected.items():
        if values.get(name, wanted) != wanted:
            raise ValueError(f"SeamlessM4T-v2 preprocessor {name} disagrees with "
                             "the executable graph.")


def _generation_defaults(config: SeamlessM4Tv2S2TConfig, ) -> dict[str, Any]:
    return {
        "bos_token_id": config.bos_token_id,
        "decoder_start_token_id": config.decoder_start_token_id,
        "eos_token_id": config.eos_token_id,
        "max_new_tokens": config.max_new_tokens,
        "pad_token_id": config.pad_token_id,
        "text_decoder_lang_to_code_id": dict(SEAMLESS_M4T_V2_LANGUAGE_TO_ID),
    }


def _validate_generation(
    values: dict[str, Any],
    config: SeamlessM4Tv2S2TConfig,
) -> dict[str, Any]:
    if not isinstance(values, dict):
        raise TypeError("Generation configuration must be an object.")
    expected = _generation_defaults(config)
    for name in (
            "bos_token_id",
            "decoder_start_token_id",
            "eos_token_id",
            "max_new_tokens",
            "pad_token_id",
    ):
        if values.get(name, expected[name]) != expected[name]:
            raise ValueError(f"SeamlessM4T-v2 generation {name} disagrees with the model.")
    languages = values.get(
        "text_decoder_lang_to_code_id",
        expected["text_decoder_lang_to_code_id"],
    )
    if languages != expected["text_decoder_lang_to_code_id"]:
        raise ValueError(
            "SeamlessM4T-v2 generation language IDs disagree with the "
            "audited 98-language table.")
    result = dict(values)
    result.update(expected)
    return result


@dataclass
class SeamlessM4Tv2S2TRuntime:
    model: SeamlessM4Tv2ForSpeechToText
    processor: SeamlessM4Tv2Processor
    config: SeamlessM4Tv2S2TConfig
    artifacts: SeamlessM4Tv2S2TArtifacts
    generation_config: dict[str, Any]

    def prepare_for_training(self) -> SeamlessM4Tv2S2TRuntime:
        self.model.train()
        return self

    def prepare_for_inference(self) -> SeamlessM4Tv2S2TRuntime:
        self.model.eval()
        return self


def load_seamless_m4t_v2_runtime(
    source: str | Path,
    *,
    device: str | torch.device,
    compute_dtype: str = "auto",
    revision: str | None = None,
    cache_dir: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
    for_training: bool = False,
) -> SeamlessM4Tv2S2TRuntime:
    """Load an official unified checkpoint or portable native S2T export."""
    artifacts = resolve_seamless_m4t_v2_artifacts(
        source,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    config_values = read_json_file(artifacts.config)
    _validate_architecture(config_values)
    config = SeamlessM4Tv2S2TConfig.from_dict(config_values)
    _validate_preprocessor(
        read_json_file(artifacts.preprocessor_config),
        config,
    )
    generation = _validate_generation(
        read_json_file(artifacts.generation_config),
        config,
    )
    processor = SeamlessM4Tv2Processor.from_files(
        config,
        artifacts.tokenizer_model,
        added_tokens=artifacts.added_tokens,
    )
    dtype = resolve_seamless_m4t_v2_dtype(compute_dtype, device)
    with torch.device("meta"):
        model = SeamlessM4Tv2ForSpeechToText(
            config,
            initialize=False,
        )
    reader_type = (ShardedSafeTensorReader if artifacts.is_sharded else SafeTensorReader)
    with reader_type(artifacts.checkpoint) as reader:
        published_full = validate_published_seamless_m4t_v2_inventory(
            reader,
            source=artifacts.source,
            revision=artifacts.revision,
        )
        SeamlessM4Tv2S2TCheckpointAdapter().load_assign_streaming(
            model,
            reader,
            config,
            device=device,
            dtype=dtype,
            strict=True,
            allow_verified_full_checkpoint=published_full,
        )
    runtime = SeamlessM4Tv2S2TRuntime(
        model=model,
        processor=processor,
        config=config,
        artifacts=artifacts,
        generation_config=generation,
    )
    return (runtime.prepare_for_training() if for_training else runtime.prepare_for_inference())


def _portable_state(
    model: SeamlessM4Tv2ForSpeechToText,
    state_dict: dict[str, torch.Tensor] | None,
) -> dict[str, torch.Tensor]:
    model_state = model.state_dict()
    expected_shapes = native_seamless_m4t_v2_tensor_shapes(model.config)
    supplied = dict(model_state) if state_dict is None else dict(state_dict)
    supplied_names = set(supplied)
    persistent_names = set(expected_shapes)
    full_names = persistent_names | _ALIASES
    if supplied_names != persistent_names and supplied_names != full_names:
        raise ValueError(
            "S2T export requires the exact persistent or tied model namespace; "
            f"missing={sorted(persistent_names - supplied_names)[:5]!r}, "
            f"extra={sorted(supplied_names - full_names)[:5]!r}.")
    if supplied_names == full_names:
        shared = supplied["shared.weight"]
        for alias in _ALIASES:
            if not torch.equal(shared, supplied[alias]):
                raise ValueError("S2T export requires tied decoder embeddings and LM head.")
    portable = {name: supplied[name] for name in sorted(persistent_names)}
    for name, value in portable.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Export value {name!r} is not a tensor.")
        if (value.layout is not torch.strided or value.is_complex() or value.is_quantized or
                not value.is_floating_point()):
            raise TypeError(f"Export tensor {name!r} must be ordinary floating-point.")
        if value.device.type == "meta":
            raise ValueError(f"Export tensor {name!r} is not materialized.")
        if tuple(value.shape) != expected_shapes[name]:
            raise ValueError(
                f"Export tensor {name!r} has shape {tuple(value.shape)}, "
                f"expected {expected_shapes[name]}.")
    return portable


def _save_state(
    state: dict[str, torch.Tensor],
    directory: Path,
    *,
    maximum_shard_bytes: int,
) -> Path:
    groups = []
    current = []
    current_bytes = 0
    total_bytes = 0
    for name, tensor in state.items():
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
        "voicehub_architecture": "seamless-m4t-v2-s2t",
    }
    if len(groups) == 1:
        return save_safetensors(
            {name: state[name].detach().cpu()
             for name in groups[0]},
            directory / "model.safetensors",
            metadata=metadata,
        )
    weight_map = {}
    for index, names in enumerate(groups, start=1):
        filename = f"model-{index:05d}-of-{len(groups):05d}.safetensors"
        save_safetensors(
            {name: state[name].detach().cpu()
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


def save_seamless_m4t_v2_runtime(
    runtime: SeamlessM4Tv2S2TRuntime,
    directory: str | Path,
    *,
    state_dict: dict[str, torch.Tensor] | None = None,
    maximum_shard_bytes: int = _DEFAULT_MAXIMUM_SHARD_BYTES,
) -> Path:
    """Atomically export an inference-ready, S2T-only native artifact."""
    if not isinstance(runtime, SeamlessM4Tv2S2TRuntime):
        raise TypeError("`runtime` must be SeamlessM4Tv2S2TRuntime.")
    if (isinstance(maximum_shard_bytes, bool) or not isinstance(maximum_shard_bytes, int) or
            maximum_shard_bytes < 1):
        raise ValueError("`maximum_shard_bytes` must be positive.")
    state = _portable_state(runtime.model, state_dict)
    destination = Path(directory).expanduser()
    if destination.exists():
        if not destination.is_dir() or any(destination.iterdir()):
            raise FileExistsError(
                "SeamlessM4T-v2 export destination must be absent or an "
                f"empty directory: {destination}.")
        destination.rmdir()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.",
        dir=destination.parent,
    ))
    try:
        _save_state(
            state,
            temporary,
            maximum_shard_bytes=maximum_shard_bytes,
        )
        write_json_file(
            temporary / "config.json",
            runtime.config.to_dict(),
        )
        runtime.processor.save_pretrained(temporary)
        write_json_file(
            temporary / "generation_config.json",
            _generation_defaults(runtime.config),
        )
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


__all__ = [
    "SeamlessM4Tv2S2TRuntime",
    "load_seamless_m4t_v2_runtime",
    "resolve_seamless_m4t_v2_dtype",
    "save_seamless_m4t_v2_runtime",
]
