"""Native ZONOS2 lifecycle, conditioning, DAC decode, and portable export."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from voicehub.components.audio.codecs.dac.utils import load_model as load_dac
from voicehub.hub import read_json_file
from voicehub.models.zonos2.native.artifacts import Zonos2Artifacts, resolve_zonos2_artifacts
from voicehub.models.zonos2.native.checkpoint import load_zonos2_checkpoint, save_zonos2_pretrained
from voicehub.models.zonos2.native.configuration import Zonos2ArchitectureConfig
from voicehub.models.zonos2.native.modeling import Zonos2ForCausalLM
from voicehub.models.zonos2.native.prompting import build_zonos2_prompt, prepare_zonos2_training_batch, shear_up
from voicehub.models.zonos2.native.sampling import Zonos2SamplingOptions, generate_zonos2_codes
from voicehub.models.zonos2.native.speaker import extract_zonos2_speaker_embedding, load_zonos2_speaker_encoder
from voicehub.processing import load_native_audio

_RATE_RANGE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*$")
_RATE_OPEN = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*\+\s*$")


def normalize_zonos2_text(text: str) -> str:
    """Portable Unicode/whitespace normalization before raw UTF-8 encoding.

    This intentionally does not claim parity with upstream's optional
    language-specific NeMo FST normalizer, whose generated language
    grammars are not part of the ZONOS2 model checkpoint.
    """
    if not isinstance(text, str):
        raise TypeError("ZONOS2 text must be a string.")
    normalized = " ".join(unicodedata.normalize("NFKC", text).split())
    if not normalized:
        raise ValueError("ZONOS2 text cannot be empty after normalization.")
    return normalized


def speaking_rate_bucket_from_speed(
    config: Zonos2ArchitectureConfig,
    speed: float | None,
) -> int | None:
    if speed is None:
        return None
    if speed <= 0:
        raise ValueError("ZONOS2 speed must be positive.")
    ranges: list[tuple[float, float | None]] = []
    for specification in config.speaking_rate_buckets:
        closed = _RATE_RANGE.match(specification)
        opened = _RATE_OPEN.match(specification)
        if closed:
            ranges.append((float(closed.group(1)), float(closed.group(2))))
        elif opened:
            ranges.append((float(opened.group(1)), None))
        else:
            raise ValueError(f"Invalid ZONOS2 speaking-rate range {specification!r}.")
    if not ranges:
        return None
    neutral_low, neutral_high = ranges[len(ranges) // 2]
    neutral = (max(neutral_low, 15.0) if neutral_high is None else (neutral_low + neutral_high) / 2.0)
    requested = neutral * speed
    for index, (_, high) in enumerate(ranges):
        if high is None or requested < high:
            return index
    return len(ranges) - 1


def resolve_zonos2_dtype(
    value: str | torch.dtype | None,
    *,
    device: torch.device,
) -> torch.dtype:
    if value is None:
        result = torch.bfloat16 if device.type == "cuda" else torch.float32
    elif isinstance(value, torch.dtype):
        result = value
    elif isinstance(value, str):
        normalized = value.lower().removeprefix("torch.")
        normalized = {
            "bf16": "bfloat16",
            "fp16": "float16",
            "fp32": "float32",
        }.get(normalized, normalized)
        result = getattr(torch, normalized, None)
        if not isinstance(result, torch.dtype):
            raise ValueError(f"Unknown ZONOS2 dtype {value!r}.")
    else:
        raise TypeError("ZONOS2 dtype must be a string, torch.dtype, or None.")
    if not result.is_floating_point:
        raise ValueError("ZONOS2 compute dtype must be floating-point.")
    if device.type == "cpu" and result in {torch.float16, torch.bfloat16}:
        return torch.float32
    if device.type == "mps" and result == torch.bfloat16:
        return torch.float16
    return result


@dataclass(slots=True)
class Zonos2Generation:
    audio: Tensor
    audio_codes: Tensor
    sample_rate: int
    eos_frame: int | None
    speaker_embedding: Tensor | None
    text_frontend: str = "raw-utf8"


@dataclass(slots=True)
class NativeZonos2Runtime:
    artifacts: Zonos2Artifacts
    config: Zonos2ArchitectureConfig
    model: Zonos2ForCausalLM
    device: torch.device
    dtype: torch.dtype
    cache_dir: str | None = None
    token: str | bool | None = None
    local_files_only: bool = False
    verify_artifacts: bool = False
    _speaker_encoder: Any = field(default=None, init=False, repr=False)
    _dac: Any = field(default=None, init=False, repr=False)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        architecture: dict[str, Any] | None = None,
        revision: str | None = None,
        cache_dir: str | None = None,
        token: str | bool | None = None,
        local_files_only: bool = False,
        verify_artifacts: bool = False,
        device: str | torch.device = "cpu",
        dtype: str | torch.dtype | None = None,
    ) -> NativeZonos2Runtime:
        resolved_device = torch.device(device)
        resolved_dtype = resolve_zonos2_dtype(dtype, device=resolved_device)
        artifacts = resolve_zonos2_artifacts(
            pretrained_model_name_or_path,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
            verify_integrity=verify_artifacts,
        )
        values = architecture or read_json_file(artifacts.config)
        config = Zonos2ArchitectureConfig.from_dict(values)
        with torch.device("meta"):
            model = Zonos2ForCausalLM(config)
        load_zonos2_checkpoint(
            model,
            artifacts.checkpoint,
            device=resolved_device,
            dtype=resolved_dtype,
        )
        model.eval()
        return cls(
            artifacts=artifacts,
            config=config,
            model=model,
            device=resolved_device,
            dtype=resolved_dtype,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
            verify_artifacts=verify_artifacts,
        )

    def _load_speaker_encoder(self):
        if self._speaker_encoder is None:
            model, _ = load_zonos2_speaker_encoder(
                cache_dir=self.cache_dir,
                token=self.token,
                local_files_only=self.local_files_only,
                verify_integrity=self.verify_artifacts,
                device=self.device,
                # The speaker checkpoint is FP32 and small enough to retain its
                # published precision independently of the acoustic LM.
                dtype=torch.float32,
            )
            self._speaker_encoder = model
        return self._speaker_encoder

    def _load_dac(self):
        if self._dac is None:
            model = load_dac("44khz")
            model.to(device=self.device, dtype=torch.float32)
            model.eval()
            model.requires_grad_(False)
            self._dac = model
        return self._dac

    def embed_speaker(self, audio: Any) -> Tensor:
        return extract_zonos2_speaker_embedding(
            self._load_speaker_encoder(),
            audio,
            device=self.device,
            dtype=torch.float32,
        )

    @torch.inference_mode()
    def encode_audio(
        self,
        audio: Any,
        *,
        sampling_rate: int | None = None,
    ) -> Tensor:
        """Encode a waveform with the frozen 44.1-kHz training codec."""
        loaded = load_native_audio(
            audio,
            sampling_rate=sampling_rate,
            target_sampling_rate=44_100,
        )
        dac = self._load_dac()
        waveform = loaded.waveform.to(
            device=self.device,
            dtype=torch.float32,
        ).view(1, 1, -1)
        waveform = dac.preprocess(waveform, 44_100)
        codes = dac.encode(
            waveform,
            n_quantizers=self.config.n_codebooks,
        )[1]
        if codes.shape[1] != self.config.n_codebooks:
            raise RuntimeError("Pinned DAC produced an incompatible codebook count.")
        return codes[0].transpose(0, 1).long()

    @torch.inference_mode()
    def decode_audio_codes(
        self,
        delayed_codes: Tensor,
        *,
        eos_frame: int | None,
    ) -> Tensor:
        if delayed_codes.ndim != 2:
            raise ValueError("ZONOS2 delayed codes must be [frames, codebooks].")
        complete_frames = max(
            0,
            delayed_codes.shape[0] - (self.config.n_codebooks - 1),
        )
        if eos_frame is not None:
            complete_frames = min(complete_frames, max(0, eos_frame))
        if complete_frames == 0:
            raise RuntimeError("ZONOS2 generation ended before one complete DAC frame.")
        codes = shear_up(
            delayed_codes,
            self.config.audio_pad_id,
        )[:complete_frames]
        codes = codes.clamp(0, self.config.codebook_size - 1)
        dac = self._load_dac()
        codes = codes.transpose(0, 1).unsqueeze(0).to(
            device=self.device,
            dtype=torch.long,
        )
        return dac.decode_codes(codes).float().squeeze(0).squeeze(0).cpu()

    def prepare_training_batch(
        self,
        texts,
        audio_codes,
        *,
        speaker_embeddings: Tensor | None = None,
        prepend_silence: bool = True,
    ) -> dict[str, Any]:
        return prepare_zonos2_training_batch(
            self.config,
            texts,
            audio_codes,
            speaker_embeddings=speaker_embeddings,
            prepend_silence=prepend_silence,
            device=self.device,
        )

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        *,
        options: Zonos2SamplingOptions | None = None,
        speaker_audio: Any | None = None,
        speaker_embedding: Tensor | None = None,
        speed: float | None = None,
        speaking_rate_bucket: int | None = None,
        quality_buckets=None,
        clean_speaker_background: bool = False,
        accurate_mode: bool = True,
        text_normalization: bool = True,
        decode_audio: bool = True,
    ) -> Zonos2Generation:
        if speaker_audio is not None and speaker_embedding is not None:
            raise ValueError("Provide either speaker audio or a speaker embedding, not both.")
        if speed is not None and speaking_rate_bucket is not None:
            raise ValueError("Provide either `speed` or `speaking_rate_bucket`, not both.")
        normalized_text = (normalize_zonos2_text(text) if text_normalization else text)
        if speaker_audio is not None:
            speaker_embedding = self.embed_speaker(speaker_audio)
        rate_bucket = (
            speaking_rate_bucket if speaking_rate_bucket is not None else speaking_rate_bucket_from_speed(
                self.config, speed))
        prompt, speaker_position = build_zonos2_prompt(
            self.config,
            normalized_text,
            speaking_rate_bucket=rate_bucket,
            quality_buckets=quality_buckets,
            include_speaker_slot=speaker_embedding is not None,
            clean_speaker_background=clean_speaker_background,
            accurate_mode=accurate_mode,
            prepend_silence=True,
            device=self.device,
        )
        delayed_codes, eos_frame = generate_zonos2_codes(
            self.model,
            prompt,
            options=options,
            speaker_embedding=speaker_embedding,
            speaker_position=speaker_position,
        )
        audio = (
            self.decode_audio_codes(delayed_codes, eos_frame=eos_frame) if decode_audio else torch.empty(0))
        return Zonos2Generation(
            audio=audio,
            audio_codes=delayed_codes.detach().cpu(),
            sample_rate=44_100,
            eos_frame=eos_frame,
            speaker_embedding=(None if speaker_embedding is None else speaker_embedding.detach().cpu()),
        )

    def save_pretrained(
        self,
        directory: str | Path,
        *,
        state_override: dict[str, Tensor] | None = None,
    ) -> Path:
        return save_zonos2_pretrained(
            self.model,
            directory,
            state_override=state_override,
        )


__all__ = [
    "NativeZonos2Runtime",
    "Zonos2Generation",
    "normalize_zonos2_text",
    "resolve_zonos2_dtype",
    "speaking_rate_bucket_from_speed",
]
