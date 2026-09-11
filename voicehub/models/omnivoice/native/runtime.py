"""Complete native OmniVoice inference, fine-tuning, and export runtime."""

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
from voicehub.models.omnivoice.native.artifacts import (
    CODEC_DIRECTORY,
    CONFIG_FILE,
    MODEL_FILE,
    OmniVoiceArtifacts,
    resolve_omnivoice_artifacts,
)
from voicehub.models.omnivoice.native.checkpoint import export_omnivoice_checkpoint, load_omnivoice_checkpoint
from voicehub.models.omnivoice.native.codec import HiggsAudioV2Tokenizer
from voicehub.models.omnivoice.native.configuration import HiggsAudioV2Config, OmniVoiceArchitectureConfig
from voicehub.models.omnivoice.native.generation import OmniVoiceGenerationConfig, OmniVoiceGenerator, OmniVoicePrompt
from voicehub.models.omnivoice.native.modeling import OmniVoiceModel
from voicehub.models.omnivoice.native.processing import (
    OmniVoiceMaskingConfig,
    OmniVoicePackingCollator,
    OmniVoicePaddingCollator,
    OmniVoiceSampleProcessor,
    OmniVoiceTokenizer,
)


def _dtype(value: str | torch.dtype) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    if not isinstance(value, str):
        raise TypeError("OmniVoice dtype must be a string or torch.dtype.")
    aliases = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    try:
        return aliases[value.lower().removeprefix("torch.")]
    except KeyError as error:
        raise ValueError(f"Unsupported OmniVoice dtype {value!r}.") from error


class OmniVoiceRuntime:
    """Own every executable component needed by OmniVoice."""

    def __init__(
        self,
        model: OmniVoiceModel,
        text_tokenizer: OmniVoiceTokenizer,
        audio_tokenizer: HiggsAudioV2Tokenizer,
        *,
        artifacts: OmniVoiceArtifacts | None = None,
        masking: OmniVoiceMaskingConfig | None = None,
    ) -> None:
        self.generator = OmniVoiceGenerator(
            model,
            text_tokenizer,
            audio_tokenizer,
        )
        self.model = model
        self.text_tokenizer = text_tokenizer
        self.audio_tokenizer = audio_tokenizer
        self.artifacts = artifacts
        self.sample_processor = OmniVoiceSampleProcessor(
            text_tokenizer,
            model.config,
            masking=masking,
            audio_tokenizer=audio_tokenizer,
        )
        self.freeze_audio_tokenizer()

    @property
    def sample_rate(self) -> int:
        return self.audio_tokenizer.sample_rate

    @property
    def device(self) -> torch.device:
        return self.model.device

    def freeze_audio_tokenizer(self) -> None:
        self.audio_tokenizer.requires_grad_(False)
        self.audio_tokenizer.eval()

    def prepare_for_training(self) -> None:
        self.model.train()
        self.freeze_audio_tokenizer()

    def prepare_for_inference(self) -> None:
        self.model.eval()
        self.audio_tokenizer.eval()

    def create_prompt(
        self,
        audio: Any,
        *,
        reference_text: str,
        sampling_rate: int | None = None,
        preprocess_prompt: bool = True,
    ) -> OmniVoicePrompt:
        loaded = load_audio(
            audio,
            sampling_rate=sampling_rate,
            target_sampling_rate=self.sample_rate,
        )
        return self.generator.create_prompt(
            loaded.waveform,
            sampling_rate=loaded.sampling_rate,
            reference_text=reference_text,
            preprocess_prompt=preprocess_prompt,
        )

    def generate(
        self,
        text: str,
        *,
        prompt: OmniVoicePrompt | None = None,
        language: str | None = None,
        instruction: str | None = None,
        duration: float | None = None,
        speed: float = 1.0,
        generation_config: OmniVoiceGenerationConfig | None = None,
        seed: int | None = None,
    ) -> Tensor:
        generator = None
        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                raise ValueError("OmniVoice seed must be a non-negative integer.")
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)
        self.prepare_for_inference()
        return self.generator.generate(
            text,
            prompt=prompt,
            language=language,
            instruction=instruction,
            duration=duration,
            speed=speed,
            config=generation_config,
            generator=generator,
        )

    def prepare_training_inputs(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        generator: torch.Generator | None = None,
        packing_tokens: int | None = None,
    ) -> dict[str, Tensor]:
        if (isinstance(records, (str, bytes, Mapping)) or not isinstance(records, Sequence) or not records):
            raise ValueError("OmniVoice records must be a non-empty sequence.")
        self.prepare_for_training()
        samples = [self.sample_processor(record, generator=generator) for record in records]
        if packing_tokens is None:
            collator = OmniVoicePaddingCollator(self.text_tokenizer.pad_token_id)
        else:
            collator = OmniVoicePackingCollator(
                self.text_tokenizer.pad_token_id,
                packing_tokens,
            )
        return {name: value.to(self.device) for name, value in collator(samples).items()}

    def save_pretrained(self, directory: str | Path) -> Path:
        """Atomically export a complete inference-ready Safetensors runtime."""
        destination = Path(directory).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
            raise FileExistsError("OmniVoice export destination must be absent or empty.")
        if destination.exists():
            destination.rmdir()
        temporary = Path(tempfile.mkdtemp(
            prefix=f".{destination.name}.",
            dir=destination.parent,
        ))
        try:
            export_omnivoice_checkpoint(
                self.model,
                temporary / MODEL_FILE,
            )
            write_json_file(
                temporary / CONFIG_FILE,
                self.model.config.to_dict(),
            )
            self.text_tokenizer.save_pretrained(temporary)
            codec_directory = temporary / CODEC_DIRECTORY
            codec_directory.mkdir()
            export_omnivoice_checkpoint(
                self.audio_tokenizer,
                codec_directory / MODEL_FILE,
            )
            write_json_file(
                codec_directory / CONFIG_FILE,
                self.audio_tokenizer.config.to_dict(),
            )
            os.replace(temporary, destination)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return destination


def load_omnivoice_runtime(
    source: str | Path,
    *,
    revision: str | None = None,
    codec_source: str | Path | None = None,
    codec_revision: str | None = None,
    device: str | torch.device = "cpu",
    dtype: str | torch.dtype = torch.float32,
    cache_dir: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
    verify_integrity: bool = True,
    verify_checkpoint_integrity: bool = False,
) -> OmniVoiceRuntime:
    artifacts = resolve_omnivoice_artifacts(
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
    model_config = OmniVoiceArchitectureConfig.from_mapping(read_json_file(artifacts.model_config))
    codec_config = HiggsAudioV2Config.from_mapping(read_json_file(artifacts.codec_config))
    target_device = torch.device(device)
    target_dtype = _dtype(dtype)
    with torch.device("meta"):
        model = OmniVoiceModel(model_config, initialize=False)
    load_omnivoice_checkpoint(
        model,
        artifacts.model_checkpoint,
        device=target_device,
        dtype=target_dtype,
        require_official_inventory=artifacts.official_model,
    )
    with torch.device("meta"):
        audio_tokenizer = HiggsAudioV2Tokenizer(
            codec_config,
            initialize=False,
        )
    load_omnivoice_checkpoint(
        audio_tokenizer,
        artifacts.codec_checkpoint,
        device=target_device,
        dtype=torch.float32,
        require_official_inventory=artifacts.official_codec,
    )
    text_tokenizer = OmniVoiceTokenizer.from_tokenizer_json(artifacts.text_tokenizer)
    return OmniVoiceRuntime(
        model,
        text_tokenizer,
        audio_tokenizer,
        artifacts=artifacts,
    )


__all__ = ["OmniVoiceRuntime", "load_omnivoice_runtime"]
