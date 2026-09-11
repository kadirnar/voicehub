"""Native VibeVoice artifact loading, mode policy, and portable export."""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from voicehub.checkpointing import SafeTensorReader, ShardedSafeTensorReader, save_safetensors
from voicehub.hub import read_json_file, write_json_file
from voicehub.models.vibevoice.native.artifacts import VibeVoiceArtifacts, resolve_vibevoice_artifacts
from voicehub.models.vibevoice.native.checkpoint import (
    VibeVoiceCheckpointAdapter,
    VibeVoiceModel,
    build_vibevoice_model,
    native_vibevoice_tensor_shapes,
    validate_published_vibevoice_inventory,
)
from voicehub.models.vibevoice.native.configuration import (
    VibeVoiceASRConfig,
    VibeVoiceTTSConfig,
    parse_vibevoice_config,
)
from voicehub.models.vibevoice.native.metadata import QWEN_TOKENIZER_ASSETS, VIBEVOICE_STATIC_ASSETS
from voicehub.models.vibevoice.native.modeling import (
    VibeVoiceASRForConditionalGeneration,
    VibeVoiceRealtimeForConditionalGeneration,
)
from voicehub.models.vibevoice.native.processing import (
    VibeVoiceASRProcessor,
    VibeVoiceAudioProcessor,
    VibeVoiceTTSProcessor,
)
from voicehub.models.vibevoice.native.tokenization import VibeVoiceTokenizer

VibeVoiceConfig = VibeVoiceASRConfig | VibeVoiceTTSConfig
VibeVoiceProcessor = VibeVoiceASRProcessor | VibeVoiceTTSProcessor

_MAXIMUM_EXPORT_SHARD_BYTES = 2_000_000_000


def resolve_vibevoice_dtype(
    value: str,
    device: str | torch.device,
) -> torch.dtype:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("VibeVoice dtype must be a non-empty string.")
    normalized = {
        "auto": "auto",
        "bf16": "bfloat16",
        "float": "float32",
        "fp16": "float16",
        "fp32": "float32",
        "half": "float16",
    }.get(value.strip().lower(),
          value.strip().lower())
    target = torch.device(device)
    if normalized == "auto":
        if target.type == "cuda":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if target.type == "mps":
            return torch.float16
        return torch.float32
    dtype = getattr(torch, normalized, None)
    if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
        raise ValueError(f"Unsupported VibeVoice dtype {value!r}.")
    if target.type == "cpu" and dtype == torch.float16:
        raise ValueError("VibeVoice float16 execution is unsupported on CPU; use "
                         "float32 or bfloat16.")
    return dtype


def _validate_audio_processor(
    values: dict[str, Any],
    config: VibeVoiceConfig,
) -> VibeVoiceAudioProcessor:
    if not isinstance(values, dict):
        raise TypeError("VibeVoice processor configuration must be an object.")
    if isinstance(config, VibeVoiceASRConfig):
        audio = values.get("feature_extractor")
        expected_processor = "VibeVoiceAsrProcessor"
    else:
        audio = values.get("audio_processor")
        expected_processor = ("VibeVoiceStreamingProcessor" if config.is_streaming else "VibeVoiceProcessor")
        if values.get("speech_tok_compress_ratio", 3_200) != 3_200:
            raise ValueError("VibeVoice processor compression ratio must be 3,200.")
        if values.get("db_normalize", True) is not True:
            raise ValueError("Published VibeVoice TTS processors require dB normalization.")
    declared_processor = values.get("processor_class")
    if declared_processor is not None and declared_processor != expected_processor:
        raise ValueError(
            f"VibeVoice processor class {declared_processor!r} does not "
            f"match {expected_processor!r}.")
    if not isinstance(audio, dict):
        raise TypeError("VibeVoice processor has no embedded audio configuration.")
    sample_rate = audio.get("sampling_rate", 24_000)
    normalize = audio.get("normalize_audio", True)
    target_dbfs = audio.get("target_dB_FS", -25)
    epsilon = audio.get("eps", 1e-6)
    if sample_rate != 24_000:
        raise ValueError("Published VibeVoice checkpoints require 24 kHz.")
    if normalize is not True:
        raise ValueError("Published VibeVoice checkpoints require waveform normalization.")
    return VibeVoiceAudioProcessor(
        sample_rate=sample_rate,
        hop_length=3_200,
        normalize_audio=normalize,
        target_dbfs=target_dbfs,
        epsilon=epsilon,
    )


def _validate_generation(
    values: dict[str, Any],
    config: VibeVoiceASRConfig,
) -> dict[str, Any]:
    if not isinstance(values, dict):
        raise TypeError("VibeVoice generation config must be an object.")
    expected = {
        "do_sample": False,
        "eos_token_id": config.text_config.eos_token_id,
        "pad_token_id": 151_655,
        "use_cache": True,
    }
    for name, default in expected.items():
        if values.get(name, default) != default:
            raise ValueError(f"VibeVoice generation {name} disagrees with the checkpoint.")
    maximum = values.get(
        "max_new_tokens",
        values.get("max_length", 32_768),
    )
    if maximum != 32_768:
        raise ValueError("Published VibeVoice ASR generation limit must be 32,768.")
    return dict(values)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_file(
    path: Path,
    *,
    expected_size: int,
    expected_sha256: str,
) -> None:
    if path.stat().st_size != expected_size:
        raise ValueError(
            f"VibeVoice asset {path.name!r} has size "
            f"{path.stat().st_size}; expected {expected_size}.")
    if _hash_file(path) != expected_sha256:
        raise ValueError(f"VibeVoice asset {path.name!r} failed SHA-256 verification.")


def _validate_published_static_assets(artifacts: VibeVoiceArtifacts) -> None:
    expected = VIBEVOICE_STATIC_ASSETS.get(artifacts.source)
    if expected is not None:
        available: dict[str, Path | None] = {
            "config.json": artifacts.config,
            "processor_config.json": artifacts.processor_config,
            "preprocessor_config.json": artifacts.processor_config,
            "tokenizer.json": artifacts.tokenizer,
            "tokenizer_config.json": artifacts.tokenizer_config,
            "generation_config.json": artifacts.generation_config,
            "chat_template.jinja": artifacts.chat_template,
        }
        for filename, (size, digest) in expected.items():
            path = available.get(filename)
            if path is None:
                raise FileNotFoundError(f"Published VibeVoice asset {filename!r} is absent.")
            _validate_file(
                path,
                expected_size=size,
                expected_sha256=digest,
            )
    if (artifacts.model_type != "vibevoice_asr" and artifacts.tokenizer_revision is not None):
        for filename, (size, digest) in QWEN_TOKENIZER_ASSETS.items():
            path = (artifacts.tokenizer if filename == "tokenizer.json" else artifacts.tokenizer_config)
            _validate_file(
                path,
                expected_size=size,
                expected_sha256=digest,
            )


@dataclass
class VibeVoiceRuntime:
    """Loaded graph, processor, config, and immutable artifact identity."""

    model: VibeVoiceModel
    processor: VibeVoiceProcessor
    config: VibeVoiceConfig
    artifacts: VibeVoiceArtifacts
    generation_config: dict[str, Any] | None

    def prepare_for_training(self) -> VibeVoiceRuntime:
        if isinstance(
                self.model,
                VibeVoiceRealtimeForConditionalGeneration,
        ):
            raise TypeError(
                "VibeVoice Realtime 0.5B publishes no unified training "
                "forward and cannot be fine-tuned through this runtime.")
        self.model.train()
        if isinstance(
                self.model,
                VibeVoiceASRForConditionalGeneration,
        ):
            frozen = (
                self.model.model.acoustic_tokenizer_encoder,
                self.model.model.semantic_tokenizer_encoder,
            )
        else:
            frozen = (
                self.model.model.acoustic_tokenizer,
                self.model.model.semantic_tokenizer,
            )
        for module in frozen:
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        return self

    def prepare_for_inference(self) -> VibeVoiceRuntime:
        self.model.eval()
        return self


def load_vibevoice_runtime(
    source: str | Path,
    *,
    device: str | torch.device,
    compute_dtype: str = "auto",
    revision: str | None = None,
    cache_dir: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
    for_training: bool = False,
    verify_payload_hashes: bool = False,
) -> VibeVoiceRuntime:
    """Load a published or VoiceHub-exported VibeVoice Safetensors graph."""
    artifacts = resolve_vibevoice_artifacts(
        source,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    _validate_published_static_assets(artifacts)
    config = parse_vibevoice_config(read_json_file(artifacts.config))
    if config.model_type != artifacts.model_type:
        raise ValueError("VibeVoice artifact model type changed after resolution.")
    audio_processor = _validate_audio_processor(
        read_json_file(artifacts.processor_config),
        config,
    )
    vocabulary_limit = (
        config.text_config.vocab_size
        if isinstance(config, VibeVoiceASRConfig) else config.decoder_config.vocab_size)
    tokenizer = VibeVoiceTokenizer.from_files(
        artifacts.tokenizer,
        artifacts.tokenizer_config,
        vocabulary_limit=vocabulary_limit,
    )
    if isinstance(config, VibeVoiceASRConfig):
        if artifacts.generation_config is None:
            raise FileNotFoundError("VibeVoice ASR requires generation_config.json.")
        generation = _validate_generation(
            read_json_file(artifacts.generation_config),
            config,
        )
        processor: VibeVoiceProcessor = VibeVoiceASRProcessor(
            tokenizer,
            audio_processor=audio_processor,
        )
    else:
        generation = None
        processor = VibeVoiceTTSProcessor(
            tokenizer,
            audio_processor=audio_processor,
        )
    dtype = resolve_vibevoice_dtype(compute_dtype, device)
    with torch.device("meta"):
        model = build_vibevoice_model(
            config,
            initialize=False,
        )
    reader_type = ShardedSafeTensorReader if artifacts.is_sharded else SafeTensorReader
    with reader_type(artifacts.checkpoint) as reader:
        validate_published_vibevoice_inventory(
            reader,
            source=artifacts.source,
            revision=artifacts.revision,
            verify_payload_hashes=verify_payload_hashes,
        )
        VibeVoiceCheckpointAdapter().load_assign_streaming(
            model,
            reader,
            config,
            device=device,
            dtype=dtype,
            strict=True,
        )
    runtime = VibeVoiceRuntime(
        model=model,
        processor=processor,
        config=config,
        artifacts=artifacts,
        generation_config=generation,
    )
    return (runtime.prepare_for_training() if for_training else runtime.prepare_for_inference())


def _save_state_dict(
    state_dict: dict[str, Tensor],
    directory: Path,
    *,
    maximum_shard_bytes: int,
    model_type: str,
) -> Path:
    _positive = (
        isinstance(maximum_shard_bytes, int) and not isinstance(maximum_shard_bytes, bool) and
        maximum_shard_bytes > 0)
    if not _positive:
        raise ValueError("`maximum_shard_bytes` must be positive.")
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
        "voicehub_architecture": model_type,
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
    index = directory / "model.safetensors.index.json"
    write_json_file(
        index,
        {
            "metadata": {
                "total_size": total_bytes
            },
            "weight_map": weight_map,
        },
    )
    return index


def save_vibevoice_runtime(
    runtime: VibeVoiceRuntime,
    directory: str | Path,
    *,
    state_dict: dict[str, Tensor] | None = None,
    maximum_shard_bytes: int = _MAXIMUM_EXPORT_SHARD_BYTES,
) -> Path:
    """Export a complete, safe, reloadable VibeVoice directory."""
    if not isinstance(runtime, VibeVoiceRuntime):
        raise TypeError("`runtime` must be a VibeVoiceRuntime.")
    state = dict(runtime.model.state_dict()) if state_dict is None else dict(state_dict)
    expected_shapes = native_vibevoice_tensor_shapes(runtime.config)
    expected = set(expected_shapes)
    received = set(state)
    mismatched = [
        (
            name,
            tuple(state[name].shape),
            expected_shapes[name],
        ) for name in sorted(expected & received)
        if (not isinstance(state[name], Tensor) or tuple(state[name].shape) != expected_shapes[name])
    ]
    invalid = [
        name for name, value in state.items() if (
            not isinstance(value, Tensor) or value.device.type == "meta" or value.is_complex() or
            value.is_quantized or value.layout != torch.strided)
    ]
    if expected != received or mismatched or invalid:
        raise ValueError(
            "VibeVoice export state is incompatible: "
            f"missing={sorted(expected - received)[:8]!r}, "
            f"unexpected={sorted(received - expected)[:8]!r}, "
            f"shape_mismatches={mismatched[:8]!r}, "
            f"invalid={invalid[:8]!r}.")
    target = Path(directory).expanduser()
    target.mkdir(parents=True, exist_ok=True)
    _save_state_dict(
        state,
        target,
        maximum_shard_bytes=maximum_shard_bytes,
        model_type=runtime.config.model_type,
    )
    config_values = runtime.config.to_dict()
    config_values["voicehub_checkpoint_format"] = "native-vibevoice-v1"
    write_json_file(target / "config.json", config_values)
    processor_name = (
        "processor_config.json"
        if isinstance(runtime.config, VibeVoiceASRConfig) else "preprocessor_config.json")
    shutil.copy2(
        runtime.artifacts.processor_config,
        target / processor_name,
    )
    runtime.processor.tokenizer.save_pretrained(target)
    if runtime.generation_config is not None:
        write_json_file(
            target / "generation_config.json",
            runtime.generation_config,
        )
    if runtime.artifacts.chat_template is not None:
        shutil.copy2(
            runtime.artifacts.chat_template,
            target / "chat_template.jinja",
        )
    source_directory = Path(__file__).with_name("source")
    if source_directory.is_dir():
        for filename in ("SOURCE.json", "THIRD_PARTY_NOTICES.md"):
            source = source_directory / filename
            if source.is_file():
                shutil.copy2(source, target / filename)
    return target.resolve()


__all__ = [
    "VibeVoiceRuntime",
    "load_vibevoice_runtime",
    "resolve_vibevoice_dtype",
    "save_vibevoice_runtime",
]
