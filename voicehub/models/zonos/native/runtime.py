"""Native Zonos v0.1 lifecycle, conditioning, generation, and codec I/O."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from voicehub.hub import read_json_file
from voicehub.models.zonos.native.artifacts import ZonosArtifacts, resolve_zonos_artifacts
from voicehub.models.zonos.native.checkpoint import load_zonos_checkpoint, save_zonos_pretrained
from voicehub.models.zonos.native.codec import ZonosCodec, ZonosDACCodec
from voicehub.models.zonos.native.configuration import ZonosArchitectureConfig
from voicehub.models.zonos.native.frontend import (
    ZonosPhonemeFrontend,
    batch_phoneme_ids,
    make_condition_dict,
    normalize_language_code,
    resolve_phonemes,
)
from voicehub.models.zonos.native.modeling import ZonosForCausalLM
from voicehub.models.zonos.native.sampling import ZonosSamplingOptions, generate_zonos_codes
from voicehub.processing.waveform import load_native_audio


def resolve_zonos_dtype(
    value: str | torch.dtype | None,
    *,
    device: torch.device,
) -> torch.dtype:
    if value is None or value == "auto":
        return torch.float32 if device.type == "cpu" else torch.bfloat16
    if isinstance(value, str):
        normalized = value.strip().lower()
        aliases = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        try:
            dtype = aliases[normalized]
        except KeyError as error:
            raise ValueError(f"Unknown Zonos dtype {value!r}.") from error
    elif isinstance(value, torch.dtype):
        dtype = value
    else:
        raise TypeError("Zonos dtype must be a string, torch.dtype, or None.")
    if not dtype.is_floating_point:
        raise ValueError("Zonos compute dtype must be floating-point.")
    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError(
            "Native Zonos does not support float16 execution on CPU; use "
            "float32 or bfloat16.")
    return dtype


@dataclass(frozen=True)
class ZonosGeneration:
    codes: Tensor
    audio: Tensor | None
    sample_rate: int
    text_frontend: str


class NativeZonosRuntime:
    """Complete dense-Transformer runtime with explicit optional boundaries."""

    def __init__(
        self,
        *,
        artifacts: ZonosArtifacts,
        config: ZonosArchitectureConfig,
        model: ZonosForCausalLM,
        codec: ZonosCodec | None = None,
        phoneme_frontend: ZonosPhonemeFrontend | None = None,
    ) -> None:
        if not isinstance(artifacts, ZonosArtifacts):
            raise TypeError("`artifacts` must be ZonosArtifacts.")
        if not isinstance(config, ZonosArchitectureConfig):
            raise TypeError("`config` must be ZonosArchitectureConfig.")
        if not isinstance(model, ZonosForCausalLM):
            raise TypeError("`model` must be ZonosForCausalLM.")
        if codec is not None and not isinstance(codec, ZonosCodec):
            raise TypeError("`codec` must implement the ZonosCodec contract.")
        self.artifacts = artifacts
        self.config = config
        self.model = model
        self._codec = codec
        self.phoneme_frontend = phoneme_frontend

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        revision: str | None = None,
        cache_dir: str | None = None,
        token: str | bool | None = None,
        local_files_only: bool = False,
        verify_artifacts: bool = False,
        device: torch.device | str = "cpu",
        dtype: str | torch.dtype | None = "auto",
        codec: ZonosCodec | None = None,
        phoneme_frontend: ZonosPhonemeFrontend | None = None,
    ) -> NativeZonosRuntime:
        resolved_device = torch.device(device)
        resolved_dtype = resolve_zonos_dtype(dtype, device=resolved_device)
        artifacts = resolve_zonos_artifacts(
            pretrained_model_name_or_path,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
            verify_integrity=verify_artifacts,
        )
        config = ZonosArchitectureConfig.from_dict(read_json_file(artifacts.config), )
        with torch.device("meta"):
            model = ZonosForCausalLM(config)
        load_zonos_checkpoint(
            model,
            artifacts.checkpoint,
            device=resolved_device,
            dtype=resolved_dtype,
        )
        model.eval()
        if codec is None:
            codec = ZonosDACCodec(
                cache_dir=cache_dir,
                token=token,
                local_files_only=local_files_only,
                device=resolved_device,
            )
        if codec is not None and hasattr(codec, "to"):
            codec.to(resolved_device)
        return cls(
            artifacts=artifacts,
            config=config,
            model=model,
            codec=codec,
            phoneme_frontend=phoneme_frontend,
        )

    @property
    def device(self) -> torch.device:
        return self.model.device

    @property
    def codec(self) -> ZonosCodec:
        if self._codec is None:
            self._codec = ZonosDACCodec(
                cache_dir=None,
                local_files_only=False,
                device=self.device,
            )
        return self._codec

    def attach_codec(self, codec: ZonosCodec) -> None:
        if not isinstance(codec, ZonosCodec):
            raise TypeError("`codec` must implement the ZonosCodec contract.")
        if (
                codec.sample_rate,
                codec.hop_length,
                codec.num_codebooks,
                codec.codebook_size,
        ) != (44_100, 512, 9, 1_024):
            raise ValueError("Injected codec is incompatible with Zonos v0.1.")
        if hasattr(codec, "to"):
            codec.to(self.device)
        self._codec = codec

    def encode_audio(
        self,
        audio: Any,
        *,
        sampling_rate: int | None = None,
    ) -> Tensor:
        prepared = load_native_audio(
            audio,
            sampling_rate=sampling_rate,
        )
        return self.codec.encode(
            prepared.waveform,
            sample_rate=prepared.sampling_rate,
        )

    def decode_audio(self, codes: Tensor) -> Tensor:
        return self.codec.decode(codes)

    def _resolve_batch_phonemes(
        self,
        texts: str | Sequence[str],
        *,
        language: str | Sequence[str],
        phonemes: str | Sequence[str] | None,
    ) -> tuple[tuple[str, ...], tuple[str, ...], str]:
        text_values = (texts, ) if isinstance(texts, str) else tuple(texts)
        if not text_values or any(not isinstance(value, str) or not value.strip() for value in text_values):
            raise ValueError("Zonos texts must be non-empty strings.")
        if isinstance(language, str):
            languages = (language, ) * len(text_values)
        else:
            languages = tuple(language)
            if len(languages) != len(text_values):
                raise ValueError("Zonos language batch must match the text batch.")
        if phonemes is None:
            phoneme_values: tuple[str | None, ...] = (None, ) * len(text_values)
        elif isinstance(phonemes, str):
            if len(text_values) != 1:
                raise ValueError("A single phoneme string can condition one text only.")
            phoneme_values = (phonemes, )
        else:
            phoneme_values = tuple(phonemes)
            if len(phoneme_values) != len(text_values):
                raise ValueError("Zonos phoneme batch must match the text batch.")
        resolved: list[str] = []
        normalized_languages: list[str] = []
        frontend_ids: list[str] = []
        for text, item_language, item_phonemes in zip(
                text_values,
                languages,
                phoneme_values,
        ):
            value, frontend_id = resolve_phonemes(
                text,
                language=item_language,
                phonemes=item_phonemes,
                frontend=self.phoneme_frontend,
            )
            resolved.append(value)
            normalized_languages.append(normalize_language_code(item_language), )
            frontend_ids.append(frontend_id)
        frontend_id = (frontend_ids[0] if len(set(frontend_ids)) == 1 else "mixed")
        return (
            tuple(resolved),
            tuple(normalized_languages),
            frontend_id,
        )

    def prepare_training_batch(
        self,
        texts: str | Sequence[str],
        audio_codes: Tensor,
        *,
        phonemes: str | Sequence[str] | None = None,
        language: str | Sequence[str] = "en-us",
        speaker_embedding: Tensor | None = None,
        emotion: Tensor | Sequence[float] = (
            0.3077,
            0.0256,
            0.0256,
            0.0256,
            0.0256,
            0.0256,
            0.2564,
            0.3077,
        ),
        fmax: float | Sequence[float] | Tensor = 22_050.0,
        pitch_std: float | Sequence[float] | Tensor = 20.0,
        speaking_rate: float | Sequence[float] | Tensor = 15.0,
        audio_code_lengths: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if not isinstance(audio_codes, Tensor) or audio_codes.ndim != 3:
            raise ValueError("Zonos audio codes must have shape "
                             "[batch, codebook, time].")
        resolved, languages, _ = self._resolve_batch_phonemes(
            texts,
            language=language,
            phonemes=phonemes,
        )
        phoneme_ids, _ = batch_phoneme_ids(
            resolved,
            device=self.device,
        )
        if audio_codes.shape[:2] != (
                phoneme_ids.shape[0],
                self.config.num_codebooks,
        ):
            raise ValueError("Zonos audio-code batch/codebook dimensions do not match "
                             "the phoneme batch.")
        conditioning = make_condition_dict(
            phoneme_ids,
            language=languages,
            speaker_embedding=speaker_embedding,
            emotion=emotion,
            fmax=fmax,
            pitch_std=pitch_std,
            speaking_rate=speaking_rate,
            device=self.device,
        )
        prefix = self.model.prefix_conditioner(conditioning)
        prepared = {
            "prefix_conditioning": prefix,
            "audio_codes": audio_codes.to(
                device=self.device,
                dtype=torch.long,
            ),
        }
        if audio_code_lengths is not None:
            prepared["audio_code_lengths"] = audio_code_lengths.to(
                device=self.device,
                dtype=torch.long,
            )
        return prepared

    def generate(
        self,
        text: str,
        *,
        phonemes: str | None = None,
        language: str = "en-us",
        speaker_embedding: Tensor | None = None,
        emotion: Tensor | Sequence[float] = (
            0.3077,
            0.0256,
            0.0256,
            0.0256,
            0.0256,
            0.0256,
            0.2564,
            0.3077,
        ),
        fmax: float = 22_050.0,
        pitch_std: float = 20.0,
        speaking_rate: float = 15.0,
        options: ZonosSamplingOptions | None = None,
        audio_prefix_codes: Tensor | None = None,
        decode_audio: bool = True,
        generator: torch.Generator | None = None,
    ) -> ZonosGeneration:
        resolved, languages, frontend_id = self._resolve_batch_phonemes(
            text,
            language=language,
            phonemes=phonemes,
        )
        phoneme_ids, _ = batch_phoneme_ids(
            resolved,
            device=self.device,
        )
        condition = make_condition_dict(
            phoneme_ids,
            language=languages,
            speaker_embedding=speaker_embedding,
            emotion=emotion,
            fmax=fmax,
            pitch_std=pitch_std,
            speaking_rate=speaking_rate,
            device=self.device,
        )
        prefix = self.model.prepare_conditioning(condition)
        codes = generate_zonos_codes(
            self.model,
            prefix,
            options=options,
            audio_prefix_codes=audio_prefix_codes,
            generator=generator,
        )
        audio = self.decode_audio(codes) if decode_audio else None
        return ZonosGeneration(
            codes=codes,
            audio=audio,
            sample_rate=self.config.sample_rate,
            text_frontend=frontend_id,
        )

    def save_pretrained(self, directory: str | Path) -> Path:
        return save_zonos_pretrained(self.model, directory)


__all__ = [
    "NativeZonosRuntime",
    "ZonosGeneration",
    "resolve_zonos_dtype",
]
