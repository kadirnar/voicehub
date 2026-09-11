"""Native Dia checkpoint lifecycle shared by inference and training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from voicehub.checkpointing import SafeTensorReader, ShardedSafeTensorReader, save_safetensors
from voicehub.hub import read_json_file, resolve_pretrained_file, write_json_file
from voicehub.models.dac.native.checkpoint import DESCRIPT_DAC_44KHZ_REVISION, HuggingFaceDacCheckpointAdapter
from voicehub.models.dac.native.configuration import DacConfig
from voicehub.models.dac.native.modeling import DacModel
from voicehub.models.dia.native.artifacts import DiaArtifacts, resolve_dia_artifacts
from voicehub.models.dia.native.checkpoint import HuggingFaceDiaCheckpointAdapter
from voicehub.models.dia.native.configuration import DiaArchitectureConfig
from voicehub.models.dia.native.modeling import DiaConditionalGenerationOutput, DiaForConditionalGeneration
from voicehub.models.dia.native.processing import DiaByteTokenizer, DiaProcessor

_DEFAULT_CODEC = "descript/dac_44khz"
_MAX_EXPORT_SHARD_BYTES = 1_000_000_000


def resolve_dia_dtype(
    dtype_name: str,
    device: str | torch.device,
) -> torch.dtype:
    if not isinstance(dtype_name, str) or not dtype_name.strip():
        raise ValueError("Dia compute dtype must be a non-empty string.")
    aliases = {
        "bf16": "bfloat16",
        "fp16": "float16",
        "fp32": "float32",
    }
    normalized = aliases.get(
        dtype_name.strip().lower(),
        dtype_name.strip().lower(),
    )
    dtype = getattr(torch, normalized, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported Dia compute dtype {dtype_name!r}.")
    resolved_device = torch.device(device)
    if resolved_device.type == "cpu" and dtype in {
            torch.float16,
            torch.bfloat16,
    }:
        return torch.float32
    if not dtype.is_floating_point:
        raise ValueError("Dia compute dtype must be floating point.")
    return dtype


def _load_native_dac(
    source: str | Path,
    *,
    device: str | torch.device,
    dtype: torch.dtype,
    revision: str | None,
    cache_dir: str | None,
    token: str | bool | None,
    local_files_only: bool,
) -> DacModel:
    source_path = Path(source).expanduser()
    if source_path.exists():
        root = source_path.resolve()
        config_path = root / "config.json"
        checkpoint_path = root / "model.safetensors"
        if not config_path.is_file() or not checkpoint_path.is_file():
            raise FileNotFoundError(
                "Portable Dia DAC export requires `config.json` and "
                "`model.safetensors`.")
    else:
        config_path = resolve_pretrained_file(
            str(source),
            "config.json",
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        )
        checkpoint_path = resolve_pretrained_file(
            str(source),
            "model.safetensors",
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        )
    config_values = read_json_file(config_path)
    checkpoint_format = config_values.get("voicehub_checkpoint_format")
    config = DacConfig.from_dict(config_values)
    with torch.device("meta"):
        codec = DacModel(config)
    with SafeTensorReader(checkpoint_path) as reader:
        if checkpoint_format == "native-state-dict-v1":
            expected = codec.state_dict()
            expected_names = set(expected)
            checkpoint_names = set(reader.keys())
            missing = sorted(expected_names - checkpoint_names)
            unexpected = sorted(checkpoint_names - expected_names)
            mismatches = sorted(
                name for name in expected_names & checkpoint_names
                if tuple(expected[name].shape) != reader.tensor_shape(name))
            if missing or unexpected or mismatches:
                details = []
                if missing:
                    details.append(f"missing={missing}")
                if unexpected:
                    details.append(f"unexpected={unexpected}")
                if mismatches:
                    details.append(f"shape_mismatches={mismatches}")
                raise ValueError("Portable Dia DAC checkpoint is incompatible: " + "; ".join(details))
            codec.load_state_dict(
                reader.state_dict(device=device, dtype=dtype),
                strict=True,
                assign=True,
            )
        elif checkpoint_format is None:
            HuggingFaceDacCheckpointAdapter().load_assign(
                codec,
                reader,
                config.to_dict(),
                strict=True,
            )
        else:
            raise ValueError("Unsupported Dia DAC checkpoint format "
                             f"{checkpoint_format!r}.")
    codec.to(device=device, dtype=dtype)
    codec.requires_grad_(False)
    codec.eval()
    return codec


def _audio_tokenizer_source(artifacts: DiaArtifacts, ) -> tuple[str | Path, str | None]:
    local_codec = artifacts.config.parent / "audio_tokenizer"
    if local_codec.is_dir():
        return local_codec, None
    if artifacts.audio_tokenizer_config is None:
        return _DEFAULT_CODEC, DESCRIPT_DAC_44KHZ_REVISION
    values = read_json_file(artifacts.audio_tokenizer_config)
    source = values.get("audio_tokenizer_name_or_path", _DEFAULT_CODEC)
    if not isinstance(source, str) or not source.strip():
        raise ValueError("Dia `audio_tokenizer_name_or_path` must be non-empty.")
    if source.startswith("./"):
        source = artifacts.config.parent / source[2:]
    return source, (None if isinstance(source, Path) else DESCRIPT_DAC_44KHZ_REVISION)


def _processor_settings(
    artifacts: DiaArtifacts,
    config: DiaArchitectureConfig,
) -> tuple[int, int, int]:
    sampling_rate = 44_100
    hop_length = 512
    tokenizer_length = config.encoder_config.max_position_embeddings
    if artifacts.preprocessor_config is not None:
        values = read_json_file(artifacts.preprocessor_config)
        sampling_rate = int(values.get("sampling_rate", sampling_rate))
        hop_length = int(values.get("hop_length", hop_length))
    if artifacts.tokenizer_config is not None:
        values = read_json_file(artifacts.tokenizer_config)
        tokenizer_length = int(values.get("max_length", tokenizer_length))
    return sampling_rate, hop_length, tokenizer_length


def _save_sharded_state_dict(
    state_dict: dict[str, torch.Tensor],
    directory: Path,
    *,
    maximum_shard_bytes: int = _MAX_EXPORT_SHARD_BYTES,
) -> Path:
    """Write bounded-memory Safetensors shards and a deterministic index."""
    if maximum_shard_bytes < 1:
        raise ValueError("`maximum_shard_bytes` must be positive.")
    names = tuple(sorted(state_dict))
    groups: list[list[str]] = []
    current: list[str] = []
    current_size = 0
    total_size = 0
    for name in names:
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
                "voicehub_architecture": "dia"
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
                "voicehub_architecture": "dia"
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


@dataclass
class DiaRuntime:
    """Loaded graph, processor, and immutable source metadata."""

    model: DiaForConditionalGeneration
    processor: DiaProcessor
    artifacts: DiaArtifacts

    @property
    def sample_rate(self) -> int:
        return self.processor.sampling_rate

    def prepare_for_training(self) -> DiaRuntime:
        self.model.train()
        self.processor.freeze_audio_tokenizer()
        return self

    def prepare_for_inference(self) -> DiaRuntime:
        self.model.eval()
        self.processor.freeze_audio_tokenizer()
        return self

    def prepare_inputs(self, inputs: Any) -> dict[str, torch.Tensor]:
        if isinstance(inputs, dict) and {
                "input_ids",
                "decoder_input_ids",
                "labels",
        }.issubset(inputs):
            return dict(inputs)
        records = inputs.get("records") if isinstance(inputs, dict) else inputs
        if isinstance(inputs, dict) and records is None:
            texts = inputs.get("text")
            audios = inputs.get("audio")
            texts_are_sequence = isinstance(texts, (list, tuple))
            audios_are_sequence = isinstance(audios, (list, tuple))
            if texts_are_sequence or audios_are_sequence:
                if not texts_are_sequence or not audios_are_sequence:
                    raise TypeError(
                        "Columnar Dia inputs require both `text` and `audio` "
                        "to be sequences.")
                if len(texts) != len(audios):
                    raise ValueError(
                        "Columnar Dia `text` and `audio` sequences must have "
                        "the same length.")
                records = tuple({"text": text, "audio": audio} for text, audio in zip(texts, audios))
        if records is None:
            records = (inputs, )
        if isinstance(records, dict):
            records = (records, )
        if not isinstance(records, (list, tuple)) or not records:
            raise TypeError("Dia training inputs require one or more records.")
        texts = []
        audios = []
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise TypeError(f"Dia training record {index} must be a mapping.")
            text = record.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Dia training record {index} requires non-empty text.")
            if "audio" not in record:
                raise ValueError(f"Dia training record {index} requires target audio.")
            texts.append(text)
            audios.append(record["audio"])
        return dict(self.processor(
            text=texts,
            audio=audios,
            generation=False,
            output_labels=True,
        ))

    def forward_loss(
        self,
        inputs: dict[str, torch.Tensor] | None = None,
        **model_inputs: torch.Tensor,
    ) -> torch.Tensor:
        if inputs is not None:
            if model_inputs:
                raise ValueError("Pass Dia model inputs as a mapping or keywords, not both.")
            model_inputs = dict(inputs)
        output = self.model(**model_inputs)
        if not isinstance(output, DiaConditionalGenerationOutput):
            raise TypeError("Native Dia returned an invalid training output.")
        if output.loss is None or output.loss.ndim != 0:
            raise RuntimeError("Native Dia did not return one scalar loss.")
        return output.loss

    def save_pretrained(self, save_directory: str | Path) -> Path:
        destination = Path(save_directory).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        write_json_file(destination / "config.json", self.model.config.to_dict())
        _save_sharded_state_dict(
            dict(self.model.state_dict()),
            destination,
        )
        decoder = self.model.config.decoder_config
        write_json_file(
            destination / "generation_config.json",
            {
                "bos_token_id": decoder.bos_token_id,
                "do_sample": True,
                "eos_token_id": decoder.eos_token_id,
                "guidance_scale": 3.0,
                "max_length": decoder.max_position_embeddings,
                "pad_token_id": decoder.pad_token_id,
                "temperature": 1.8,
                "top_k": 50,
                "top_p": 0.9,
            },
        )
        write_json_file(
            destination / "preprocessor_config.json",
            {
                "feature_extractor_type": "VoiceHubDiaFeatureExtractor",
                "hop_length": self.processor.hop_length,
                "padding_side": "right",
                "padding_value": 0.0,
                "processor_class": "DiaProcessor",
                "return_attention_mask": True,
                "sampling_rate": self.processor.sampling_rate,
            },
        )
        write_json_file(
            destination / "tokenizer_config.json",
            {
                "max_length": self.processor.tokenizer.max_length,
                "pad_token": "<pad>",
                "processor_class": "DiaProcessor",
                "tokenizer_class": "DiaByteTokenizer",
                "unk_token": "<pad>",
            },
        )
        write_json_file(
            destination / "audio_tokenizer_config.json",
            {
                "audio_tokenizer_class": "DacModel",
                "audio_tokenizer_name_or_path": "./audio_tokenizer",
            },
        )
        codec_directory = destination / "audio_tokenizer"
        codec_directory.mkdir(parents=True, exist_ok=True)
        write_json_file(
            codec_directory / "config.json",
            {
                **self.processor.audio_tokenizer.config.to_dict(),
                "voicehub_checkpoint_format":
                "native-state-dict-v1",
            },
        )
        save_safetensors(
            dict(self.processor.audio_tokenizer.state_dict()),
            codec_directory / "model.safetensors",
            metadata={
                "format": "pt",
                "voicehub_architecture": "dac"
            },
        )
        return destination


def load_dia_runtime(
    source: str | Path,
    *,
    device: str | torch.device,
    compute_dtype: str = "bfloat16",
    revision: str | None = None,
    cache_dir: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
    for_training: bool = False,
) -> DiaRuntime:
    artifacts = resolve_dia_artifacts(
        source,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    config_values = read_json_file(artifacts.config)
    config = DiaArchitectureConfig.from_dict(config_values)
    dtype = resolve_dia_dtype(compute_dtype, device)
    with torch.device("meta"):
        model = DiaForConditionalGeneration(config)
    reader_type = (ShardedSafeTensorReader if artifacts.is_sharded else SafeTensorReader)
    with reader_type(artifacts.checkpoint) as reader:
        HuggingFaceDiaCheckpointAdapter().load_assign_streaming(
            model,
            reader,
            config_values,
            device=device,
            dtype=dtype,
            strict=True,
        )
    codec_source, codec_revision = _audio_tokenizer_source(artifacts)
    codec = _load_native_dac(
        codec_source,
        device=device,
        dtype=dtype,
        revision=codec_revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    sample_rate, hop_length, tokenizer_length = _processor_settings(
        artifacts,
        config,
    )
    processor = DiaProcessor(
        config,
        audio_tokenizer=codec,
        tokenizer=DiaByteTokenizer(max_length=tokenizer_length),
        sampling_rate=sample_rate,
        hop_length=hop_length,
    )
    runtime = DiaRuntime(model, processor, artifacts)
    return (runtime.prepare_for_training() if for_training else runtime.prepare_for_inference())


__all__ = [
    "DiaRuntime",
    "load_dia_runtime",
    "resolve_dia_dtype",
]
