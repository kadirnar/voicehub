"""Native VoxCPM2 inference, preprocessing, and portable export."""

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
from voicehub.models.voxcpm.native.artifacts import VoxCPM2Artifacts, resolve_voxcpm2_artifacts
from voicehub.models.voxcpm.native.checkpoint import (
    convert_legacy_voxcpm_codec,
    export_voxcpm_checkpoint,
    load_voxcpm_checkpoint,
)
from voicehub.models.voxcpm.native.codec import VoxCPMAudioVAE
from voicehub.models.voxcpm.native.configuration import VoxCPM2ArchitectureConfig
from voicehub.models.voxcpm.native.metadata import (
    VOXCPM2_CHECKPOINT_FILE,
    VOXCPM2_CODEC_NATIVE_FILE,
    VOXCPM2_CONFIG_FILE,
)
from voicehub.models.voxcpm.native.modeling import VoxCPM2Model
from voicehub.models.voxcpm.native.processing import VoxCPM2Processor, VoxCPM2Tokenizer


def _dtype(value: str | torch.dtype) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    if not isinstance(value, str):
        raise TypeError("VoxCPM dtype must be a string or `torch.dtype`.")
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
        raise ValueError(f"Unsupported VoxCPM dtype {value!r}.") from error


class VoxCPM2Runtime:
    """Complete native VoxCPM2 model, tokenizer, and frozen AudioVAE."""

    def __init__(
        self,
        model: VoxCPM2Model,
        processor: VoxCPM2Processor,
        codec: VoxCPMAudioVAE,
        *,
        artifacts: VoxCPM2Artifacts | None = None,
    ) -> None:
        if not isinstance(model, VoxCPM2Model):
            raise TypeError("`model` must be a native VoxCPM2Model.")
        if not isinstance(processor, VoxCPM2Processor):
            raise TypeError("`processor` must be a native VoxCPM2Processor.")
        if not isinstance(codec, VoxCPMAudioVAE):
            raise TypeError("`codec` must be a native VoxCPMAudioVAE.")
        if processor.codec is not codec:
            raise ValueError("VoxCPM processor and runtime must share one codec.")
        self.model = model
        self.processor = processor
        self.codec = codec
        self.artifacts = artifacts
        self.freeze_codec()

    @property
    def sample_rate(self) -> int:
        return self.codec.out_sample_rate

    @property
    def input_sample_rate(self) -> int:
        return self.codec.sample_rate

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def freeze_codec(self) -> VoxCPMAudioVAE:
        self.codec.requires_grad_(False)
        self.codec.eval()
        return self.codec

    def prepare_for_training(self) -> None:
        self.model.train()
        self.freeze_codec()

    def prepare_for_inference(self) -> None:
        self.model.eval()
        self.codec.eval()

    def encode_audio(
        self,
        audio: Any,
        *,
        sampling_rate: int | None = None,
    ) -> Tensor:
        loaded = load_audio(
            audio,
            sampling_rate=sampling_rate,
            target_sampling_rate=self.input_sample_rate,
        )
        waveform = loaded.waveform.to(
            device=self.device,
            dtype=torch.float32,
        ).squeeze()
        if waveform.ndim != 1:
            raise ValueError("VoxCPM reference audio must be mono.")
        patch_samples = (self.codec.hop_length * self.model.patch_size)
        remainder = waveform.numel() % patch_samples
        if remainder:
            waveform = torch.nn.functional.pad(
                waveform,
                (0, patch_samples - remainder),
            )
        with torch.no_grad():
            encoded = self.codec.encode(
                waveform[None, None],
                self.input_sample_rate,
            )
        return (encoded.transpose(1, 2)[0].unflatten(0, (-1, self.model.patch_size)).cpu())

    def generate(
        self,
        text: str,
        *,
        prompt_audio: Any | None = None,
        prompt_sampling_rate: int | None = None,
        prompt_text: str = "",
        reference_audio: Any | None = None,
        reference_sampling_rate: int | None = None,
        min_length: int = 2,
        max_length: int = 2_000,
        diffusion_steps: int = 10,
        guidance: float = 2.0,
        seed: int | None = None,
    ) -> Tensor:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("VoxCPM generation text cannot be empty.")
        prompt_features = (
            None if prompt_audio is None else self.encode_audio(
                prompt_audio,
                sampling_rate=prompt_sampling_rate,
            ))
        reference_features = (
            None if reference_audio is None else self.encode_audio(
                reference_audio,
                sampling_rate=reference_sampling_rate,
            ))
        prefix = self.processor.generation_prefix(
            text,
            prompt_features=prompt_features,
            prompt_text=prompt_text,
            reference_features=reference_features,
            device=self.device,
        )
        generator = None
        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                raise ValueError("VoxCPM seed must be a non-negative integer.")
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)
        self.prepare_for_inference()
        features = self.model.generate_features(
            **prefix,
            min_length=min_length,
            max_length=max_length,
            diffusion_steps=diffusion_steps,
            guidance=guidance,
            generator=generator,
        )
        return self.codec.decode(features.float()).squeeze(1)

    def prepare_training_inputs(
        self,
        records: Sequence[Mapping[str, Any]],
    ) -> dict[str, Tensor]:
        self.prepare_for_training()
        return self.processor.training_batch(
            records,
            device=self.device,
        )

    def save_pretrained(self, directory: str | Path) -> Path:
        """Atomically export a complete pickle-free runtime."""
        return self.save_pretrained_with_state(directory)

    def save_pretrained_with_state(
        self,
        directory: str | Path,
        *,
        model_state_override: Mapping[str, Tensor] | None = None,
    ) -> Path:
        """Atomically export, optionally using a merged model state."""
        destination = Path(directory).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and not destination.is_dir():
            raise FileExistsError(f"VoxCPM export destination is not a directory: {destination}.")
        if destination.exists() and any(destination.iterdir()):
            raise FileExistsError(f"Refusing to replace non-empty VoxCPM directory: {destination}.")
        if destination.exists():
            destination.rmdir()
        temporary = Path(tempfile.mkdtemp(
            prefix=f".{destination.name}.",
            dir=destination.parent,
        ))
        try:
            export_voxcpm_checkpoint(
                self.model,
                temporary / VOXCPM2_CHECKPOINT_FILE,
                state_override=model_state_override,
            )
            export_voxcpm_checkpoint(
                self.codec,
                temporary / VOXCPM2_CODEC_NATIVE_FILE,
            )
            write_json_file(
                temporary / VOXCPM2_CONFIG_FILE,
                self.model.config.to_dict(),
            )
            self.processor.tokenizer.save_pretrained(temporary)
            os.replace(temporary, destination)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return destination


def load_voxcpm2_runtime(
    source: str | Path,
    *,
    revision: str | None = None,
    codec_path: str | Path | None = None,
    device: str | torch.device = "cpu",
    dtype: str | torch.dtype = torch.float32,
    trust_legacy_codec: bool = False,
    cache_dir: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
    verify_integrity: bool = False,
    verify_checkpoint_integrity: bool = False,
) -> VoxCPM2Runtime:
    """Load official or portable VoxCPM2 artifacts without external
    runtimes."""
    artifacts = resolve_voxcpm2_artifacts(
        source,
        revision=revision,
        codec_path=codec_path,
        allow_legacy_codec=trust_legacy_codec,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
        verify_integrity=verify_integrity,
        verify_checkpoint_integrity=verify_checkpoint_integrity,
    )
    config = VoxCPM2ArchitectureConfig.from_mapping(read_json_file(artifacts.config))
    target_device = torch.device(device)
    target_dtype = _dtype(dtype)
    with torch.device("meta"):
        model = VoxCPM2Model(config)
    load_voxcpm_checkpoint(
        model,
        artifacts.checkpoint,
        device=target_device,
        dtype=target_dtype,
        require_official_inventory=artifacts.official,
    )
    codec_path_resolved = artifacts.codec_checkpoint
    if artifacts.legacy_codec:
        native_path = codec_path_resolved.with_name(VOXCPM2_CODEC_NATIVE_FILE)
        if not native_path.is_file():
            codec_for_conversion = VoxCPMAudioVAE(
                config.audio_vae_config,
                device="cpu",
                dtype=torch.float32,
            )
            convert_legacy_voxcpm_codec(
                codec_for_conversion,
                codec_path_resolved,
                native_path,
                trust_legacy_pickle=trust_legacy_codec,
                verify_official_integrity=artifacts.official,
            )
        codec_path_resolved = native_path
    with torch.device("meta"):
        codec = VoxCPMAudioVAE(config.audio_vae_config)
    load_voxcpm_checkpoint(
        codec,
        codec_path_resolved,
        device=target_device,
        dtype=torch.float32,
        require_official_inventory=artifacts.official,
    )
    tokenizer_instance = VoxCPM2Tokenizer.from_file(
        artifacts.tokenizer,
        config=config,
    )
    processor = VoxCPM2Processor(
        tokenizer_instance,
        config,
        codec=codec,
    )
    return VoxCPM2Runtime(
        model,
        processor,
        codec,
        artifacts=artifacts,
    )


__all__ = ["VoxCPM2Runtime", "load_voxcpm2_runtime"]
