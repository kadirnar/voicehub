"""Complete native Higgs Audio v2 loading, inference, and preparation."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from voicehub.audio import load_audio
from voicehub.hub import read_json_file, write_json_file
from voicehub.models.higgstts.native.artifacts import HiggsAudioV2Artifacts, resolve_higgs_audio_v2_artifacts
from voicehub.models.higgstts.native.checkpoint import export_higgs_checkpoint, load_higgs_checkpoint
from voicehub.models.higgstts.native.configuration import HiggsAudioV2Config
from voicehub.models.higgstts.native.generation import HiggsAudioV2GenerationOutput, HiggsAudioV2Generator
from voicehub.models.higgstts.native.metadata import (
    HIGGS_AUDIO_V2_CHECKPOINT_FILE,
    HIGGS_AUDIO_V2_CODEC_CHECKPOINT_FILE,
    HIGGS_AUDIO_V2_CODEC_CONFIG_FILE,
    HIGGS_AUDIO_V2_CONFIG_FILE,
)
from voicehub.models.higgstts.native.modeling import HiggsAudioV2ForConditionalGeneration
from voicehub.models.higgstts.native.processing import (
    DEFAULT_SCENE_PROMPT,
    DEFAULT_SYSTEM_PROMPT,
    HiggsAudioV2Batch,
    HiggsAudioV2Processor,
    HiggsAudioV2TextTokenizer,
)
from voicehub.models.higgstts.native.tokenizer import HiggsAudioV2TokenizerModel
from voicehub.models.higgstts.native.tokenizer_configuration import HiggsAudioV2TokenizerConfig

_CODEC_DIRECTORY = "audio_tokenizer"


def _dtype(value: str | torch.dtype) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    if not isinstance(value, str):
        raise TypeError("Higgs dtype must be a string or `torch.dtype`.")
    aliases = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    try:
        return aliases[value.lower()]
    except KeyError as error:
        raise ValueError(f"Unsupported Higgs dtype {value!r}.") from error


class HiggsAudioV2Runtime:
    """Native model, text tokenizer, frozen audio tokenizer, and generator."""

    def __init__(
        self,
        model: HiggsAudioV2ForConditionalGeneration,
        processor: HiggsAudioV2Processor,
        *,
        artifacts: HiggsAudioV2Artifacts | None = None,
    ) -> None:
        if not isinstance(model, HiggsAudioV2ForConditionalGeneration):
            raise TypeError("`model` must be HiggsAudioV2ForConditionalGeneration.")
        if not isinstance(processor, HiggsAudioV2Processor):
            raise TypeError("`processor` must be HiggsAudioV2Processor.")
        if processor.model_config != model.config:
            raise ValueError("Higgs runtime model and processor configurations differ.")
        self.model = model
        self.processor = processor
        self.audio_tokenizer = processor.audio_tokenizer
        self.artifacts = artifacts
        self.freeze_audio_tokenizer()

    @property
    def sample_rate(self) -> int:
        return self.processor.sample_rate

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def freeze_audio_tokenizer(self) -> HiggsAudioV2TokenizerModel:
        self.audio_tokenizer.freeze()
        return self.audio_tokenizer

    def prepare_for_training(self) -> None:
        self.model.train()
        self.freeze_audio_tokenizer()

    def prepare_for_inference(self) -> None:
        self.model.eval()
        self.audio_tokenizer.eval()

    def encode_audio(
        self,
        audio: Any,
        *,
        sampling_rate: int | None = None,
    ) -> Tensor:
        loaded = load_audio(
            audio,
            sampling_rate=sampling_rate,
            target_sampling_rate=self.sample_rate,
        )
        waveform = loaded.waveform.to(
            device=self.device,
            dtype=torch.float32,
        )
        with torch.no_grad():
            return self.processor.encode_audio(waveform[None, None])

    def generate(
        self,
        text: str,
        *,
        reference_audio: Any | None = None,
        reference_sampling_rate: int | None = None,
        reference_text: str | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        scene_prompt: str | None = DEFAULT_SCENE_PROMPT,
        max_new_tokens: int = 1_024,
        temperature: float = 1.0,
        top_k: int | None = 50,
        top_p: float = 0.95,
        ras_window: int | None = 7,
        ras_max_repeats: int = 2,
        seed: int | None = None,
    ) -> HiggsAudioV2GenerationOutput:
        reference_codes = (
            None if reference_audio is None else self.encode_audio(
                reference_audio,
                sampling_rate=reference_sampling_rate,
            ))
        batch = self.processor.generation_batch(
            text,
            reference_codes=reference_codes,
            reference_text=reference_text,
            system_prompt=system_prompt,
            scene_prompt=scene_prompt,
            device=self.device,
        )
        self.prepare_for_inference()
        return HiggsAudioV2Generator(
            self.model,
            self.processor,
        ).generate(
            batch,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            ras_window=ras_window,
            ras_max_repeats=ras_max_repeats,
            seed=seed,
        )

    def prepare_training_inputs(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        pad_to_multiple_of: int | None = 8,
    ) -> HiggsAudioV2Batch:
        """Encode consented records into exact delayed-codebook labels."""
        if isinstance(records, (str, bytes)) or not isinstance(
                records,
                Sequence,
        ):
            raise TypeError("Higgs training records must be a sequence.")
        examples = []
        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                raise TypeError(f"Higgs training record {index} must be a mapping.")
            text = record.get("text")
            target_codes = record.get("audio_codes")
            if target_codes is None:
                target_audio = record.get(
                    "audio",
                    record.get("target_audio"),
                )
                if target_audio is None:
                    raise ValueError(f"Higgs training record {index} requires `audio` "
                                     "or `audio_codes`.")
                target_codes = self.encode_audio(
                    target_audio,
                    sampling_rate=record.get("sampling_rate"),
                )
            elif not isinstance(target_codes, Tensor):
                target_codes = torch.as_tensor(
                    target_codes,
                    dtype=torch.long,
                    device=self.device,
                )
            reference_codes = record.get("reference_codes")
            if reference_codes is None and record.get("reference_audio") is not None:
                reference_codes = self.encode_audio(
                    record["reference_audio"],
                    sampling_rate=record.get("reference_sampling_rate"),
                )
            elif reference_codes is not None and not isinstance(
                    reference_codes,
                    Tensor,
            ):
                reference_codes = torch.as_tensor(
                    reference_codes,
                    dtype=torch.long,
                    device=self.device,
                )
            examples.append(
                self.processor.training_example(
                    text,
                    target_codes.to(self.device),
                    reference_codes=(None if reference_codes is None else reference_codes.to(self.device)),
                    reference_text=record.get("reference_text"),
                    system_prompt=record.get(
                        "system_prompt",
                        DEFAULT_SYSTEM_PROMPT,
                    ),
                    scene_prompt=record.get(
                        "scene_prompt",
                        DEFAULT_SCENE_PROMPT,
                    ),
                    device=self.device,
                ))
        self.prepare_for_training()
        return self.processor.collate(
            examples,
            pad_to_multiple_of=pad_to_multiple_of,
        )

    def save_pretrained(
        self,
        directory: str | Path,
        *,
        model_state_override: Mapping[str, Tensor] | None = None,
    ) -> Path:
        """Atomically export a complete, pickle-free native runtime."""
        destination = Path(directory).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and not destination.is_dir():
            raise FileExistsError(f"Higgs export destination is not a directory: "
                                  f"{destination}.")
        if destination.exists() and any(destination.iterdir()):
            raise FileExistsError(f"Refusing to replace non-empty Higgs directory: "
                                  f"{destination}.")
        if destination.exists():
            destination.rmdir()
        temporary = Path(tempfile.mkdtemp(
            prefix=f".{destination.name}.",
            dir=destination.parent,
        ))
        try:
            export_higgs_checkpoint(
                self.model,
                temporary / HIGGS_AUDIO_V2_CHECKPOINT_FILE,
                state_override=model_state_override,
            )
            write_json_file(
                temporary / HIGGS_AUDIO_V2_CONFIG_FILE,
                self.model.config.to_dict(),
            )
            self.processor.tokenizer.save_pretrained(temporary)
            codec_directory = temporary / _CODEC_DIRECTORY
            codec_directory.mkdir()
            export_higgs_checkpoint(
                self.audio_tokenizer,
                codec_directory / HIGGS_AUDIO_V2_CODEC_CHECKPOINT_FILE,
            )
            write_json_file(
                codec_directory / HIGGS_AUDIO_V2_CODEC_CONFIG_FILE,
                self.audio_tokenizer.config.to_dict(),
            )
            write_json_file(
                codec_directory / "preprocessor_config.json",
                {
                    "feature_extractor_type": "DacFeatureExtractor",
                    "feature_size": 1,
                    "hop_length": 1,
                    "padding_side": "right",
                    "padding_value": 0.0,
                    "return_attention_mask": True,
                    "sampling_rate": self.sample_rate,
                },
            )
            os.replace(temporary, destination)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return destination


def load_higgs_audio_v2_runtime(
    source: str | Path,
    *,
    revision: str | None = None,
    codec_source: str | Path | None = None,
    codec_revision: str | None = None,
    device: str | torch.device = "cpu",
    dtype: str | torch.dtype = torch.bfloat16,
    cache_dir: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
    verify_integrity: bool = False,
    verify_checkpoint_integrity: bool = False,
) -> HiggsAudioV2Runtime:
    """Load both official snapshots without an external model runtime."""
    artifacts = resolve_higgs_audio_v2_artifacts(
        source,
        revision=revision,
        codec_source=codec_source,
        codec_revision=codec_revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
        verify_integrity=verify_integrity,
        verify_checkpoint_integrity=verify_checkpoint_integrity,
    )
    model_config = HiggsAudioV2Config.from_dict(read_json_file(artifacts.config))
    codec_config = HiggsAudioV2TokenizerConfig.from_dict(read_json_file(artifacts.codec_config))
    target_device = torch.device(device)
    target_dtype = _dtype(dtype)
    if target_device.type == "cpu" and target_dtype in {
            torch.float16,
            torch.bfloat16,
    }:
        target_dtype = torch.float32
    with torch.device("meta"):
        model = HiggsAudioV2ForConditionalGeneration(
            model_config,
            initialize=False,
        )
        audio_tokenizer = HiggsAudioV2TokenizerModel(
            codec_config,
            initialize=False,
        )
    load_higgs_checkpoint(
        model,
        artifacts.checkpoint,
        device=target_device,
        dtype=target_dtype,
        require_official_inventory=artifacts.official,
    )
    load_higgs_checkpoint(
        audio_tokenizer,
        artifacts.codec_checkpoint,
        device=target_device,
        dtype=torch.float32,
        require_official_inventory=artifacts.official,
    )
    tokenizer = HiggsAudioV2TextTokenizer.from_files(
        artifacts.tokenizer,
        tokenizer_config=artifacts.tokenizer_config,
        special_tokens_map=artifacts.special_tokens_map,
        chat_template=artifacts.chat_template,
    )
    processor = HiggsAudioV2Processor(
        tokenizer,
        audio_tokenizer,
        model_config,
    )
    return HiggsAudioV2Runtime(
        model,
        processor,
        artifacts=artifacts,
    )


__all__ = [
    "HiggsAudioV2Runtime",
    "load_higgs_audio_v2_runtime",
]
