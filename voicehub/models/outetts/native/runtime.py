"""Composable VoiceHub-native OuteTTS inference and export runtime."""

from __future__ import annotations

import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from voicehub.checkpointing import VoiceHubManifest, build_manifest_files
from voicehub.generation import GenerationConfig
from voicehub.optimization.protocols import OptimizationCompileTarget

from .artifacts import OuteTTSArtifacts, OuteTTSDacArtifacts, resolve_outetts_artifacts, resolve_outetts_dac_artifacts
from .checkpoint import load_outetts_dac, load_outetts_language_model, save_outetts_dac
from .metadata import (
    NATIVE_OUTETTS_FORMAT,
    OUTETTS_CHECKPOINTS,
    OUTETTS_SOURCE_LICENSE,
    OUTETTS_SOURCE_REPOSITORY,
    OUTETTS_SOURCE_REVISION,
    OUTETTS_TRAINING_SOURCE_REVISION,
)
from .prompting import OuteTTSPromptProcessor, SpeakerProfile, load_default_speaker
from .tokenization import OuteTTSTokenizer


def _sentences(text: str, *, maximum_characters: int = 220) -> tuple[str, ...]:
    """Split text into bounded chunks without a language-specific segmenter."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("OuteTTS text must be non-empty.")
    if (isinstance(maximum_characters, bool) or not isinstance(maximum_characters, int) or
            maximum_characters < 16):
        raise ValueError("Chunk size must be an integer of at least 16.")
    pieces: list[str] = []
    current = ""
    for character in text.strip():
        current += character
        boundary = character in ".!?。！？\n"
        if boundary or len(current) >= maximum_characters:
            value = current.strip()
            if value:
                pieces.append(value)
            current = ""
    if current.strip():
        pieces.append(current.strip())
    return tuple(pieces)


class OuteTTSRuntime(nn.Module):
    """Native causal LM, V3 prompt protocol, and frozen 24 kHz DAC."""

    sample_rate = 24_000

    def __init__(
        self,
        language_model: nn.Module,
        tokenizer: OuteTTSTokenizer,
        codec: nn.Module,
        *,
        artifacts: OuteTTSArtifacts | None = None,
        codec_artifacts: OuteTTSDacArtifacts | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(tokenizer, OuteTTSTokenizer):
            raise TypeError("OuteTTS runtime requires OuteTTSTokenizer.")
        self.language_model = language_model
        self.tokenizer = tokenizer
        self.codec = codec
        self.prompt_processor = OuteTTSPromptProcessor(tokenizer)
        self.artifacts = artifacts
        self.codec_artifacts = codec_artifacts
        self.codec.requires_grad_(False)
        self.codec.eval()

    def optimization_compile_targets(
        self,
        mode: str,
    ) -> tuple[OptimizationCompileTarget, ...]:
        """Expose the causal LM and DAC methods used by OuteTTS."""
        if mode == "training":
            return (OptimizationCompileTarget(
                "language_model.forward",
                self.language_model,
                "forward",
            ), )
        if mode != "inference":
            raise ValueError(f"Unsupported optimization mode {mode!r}.")
        return (
            OptimizationCompileTarget(
                "language_model.forward",
                self.language_model,
                "forward",
            ),
            OptimizationCompileTarget(
                "codec.decode_codes",
                self.codec,
                "decode_codes",
            ),
        )

    @property
    def model(self):
        """Compatibility alias for code expecting a backend `.model`."""
        return self.language_model

    def train(self, mode: bool = True):
        super().train(mode)
        self.codec.eval()
        return self

    def resolve_speaker(
        self,
        *,
        speaker: str = "EN-FEMALE-1-NEUTRAL",
        speaker_profile: SpeakerProfile | Mapping[str, Any] | None = None,
        speaker_profile_path: str | Path | None = None,
    ) -> SpeakerProfile:
        supplied = sum(value is not None for value in (speaker_profile, speaker_profile_path))
        if supplied > 1:
            raise ValueError("Pass only one of `speaker_profile` or "
                             "`speaker_profile_path`.")
        if speaker_profile is not None:
            return (
                speaker_profile if isinstance(speaker_profile, SpeakerProfile) else
                SpeakerProfile.from_mapping(speaker_profile))
        if speaker_profile_path is not None:
            return self.prompt_processor.load_speaker(speaker_profile_path)
        normalized = speaker.strip().upper().replace("_", "-")
        if normalized != "EN-FEMALE-1-NEUTRAL":
            raise ValueError(
                "Native OuteTTS currently bundles only "
                "'EN-FEMALE-1-NEUTRAL'. Pass a validated V3 "
                "`speaker_profile` for another voice.")
        return load_default_speaker()

    def _decode_codes(
        self,
        first: list[int],
        second: list[int],
    ) -> Tensor:
        if not first or len(first) != len(second):
            raise RuntimeError("OuteTTS returned no complete two-codebook DAC frames.")
        if max((*first, *second)) >= 1_024:
            raise RuntimeError(
                "OuteTTS generated reserved codec token 1024, which cannot "
                "be decoded by the 1024-entry DAC codebooks.")
        device = next(self.codec.parameters()).device
        codes = torch.tensor(
            [[first, second]],
            dtype=torch.long,
            device=device,
        )
        audio = self.codec.decode_codes(codes)
        if audio.ndim != 3 or audio.shape[0] != 1 or audio.shape[1] != 1:
            raise RuntimeError(
                "Native OuteTTS DAC returned an invalid waveform shape "
                f"{tuple(audio.shape)}.")
        fade_length = min(int(self.sample_rate * 0.015), audio.shape[-1] // 2)
        if fade_length:
            audio = audio.clone()
            audio[..., :fade_length] *= torch.linspace(
                0,
                1,
                fade_length,
                device=audio.device,
                dtype=audio.dtype,
            )
            audio[..., -fade_length:] *= torch.linspace(
                1,
                0,
                fade_length,
                device=audio.device,
                dtype=audio.dtype,
            )
        return audio

    def _generate_one(
        self,
        text: str,
        *,
        speaker: SpeakerProfile,
        max_length: int,
        sampler: Mapping[str, Any],
        seed: int | None,
    ) -> Tensor:
        prompt = self.prompt_processor.completion_prompt(text, speaker)
        prompt_ids = self.prompt_processor.encode(prompt)
        if len(prompt_ids) >= max_length:
            raise ValueError(
                "OuteTTS prompt consumes the complete sequence budget "
                f"({len(prompt_ids)} >= {max_length}). Shorten the text or "
                "speaker profile.")
        device = next(self.language_model.parameters()).device
        input_ids = torch.tensor(
            [prompt_ids],
            dtype=torch.long,
            device=device,
        )
        temperature = float(sampler.get("temperature", 0.4))
        do_sample = temperature > 0
        config = GenerationConfig(
            max_new_tokens=max_length - len(prompt_ids),
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            top_k=int(sampler.get("top_k", 40)),
            top_p=float(sampler.get("top_p", 0.9)),
            min_p=float(sampler.get("min_p", 0.05)),
            repetition_penalty=float(sampler.get("repetition_penalty", 1.1)),
            eos_token_id=(
                self.tokenizer.convert_tokens_to_ids(OuteTTSPromptProcessor.AUDIO_END),
                self.tokenizer.eos_token_id,
            ),
            pad_token_id=self.tokenizer.eos_token_id,
            seed=seed,
            use_cache=True,
        )
        generated = self.language_model.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids, dtype=torch.bool),
            generation_config=config,
            repetition_window=int(sampler.get("repetition_range", 64)),
        )
        completion = generated.sequences[0, input_ids.shape[1]:]
        first, second = self.prompt_processor.extract_audio_codes(completion)
        return self._decode_codes(first, second)

    def generate(
        self,
        text: str,
        *,
        speaker: SpeakerProfile,
        generation_type: str,
        max_length: int,
        sampler: Mapping[str, Any],
        seed: int | None,
    ) -> Tensor:
        if generation_type == "REGULAR":
            chunks = (text, )
        elif generation_type == "CHUNKED":
            chunks = _sentences(text)
        else:
            raise ValueError("Native OuteTTS supports regular and chunked generation only.")
        audio = [
            self._generate_one(
                chunk,
                speaker=speaker,
                max_length=max_length,
                sampler=sampler,
                seed=(None if seed is None else seed + index),
            ) for index, chunk in enumerate(chunks)
        ]
        return torch.cat(audio, dim=-1)

    def save_pretrained(self, directory: str | Path) -> Path:
        """Write one self-contained, integrity-manifested native artifact."""
        destination = Path(directory).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        self.language_model.save_pretrained(destination)
        self.tokenizer.save_pretrained(destination)
        save_outetts_dac(self.codec, destination / "dac")
        default_source = Path(__file__).with_name("default_speaker.json")
        default_destination = destination / "default_speaker.json"
        if default_source.resolve() != default_destination.resolve():
            shutil.copy2(default_source, default_destination)

        paths = [
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "dac/config.json",
            "dac/model.safetensors",
            "default_speaker.json",
        ]
        if (destination / "tokenizer_config.json").is_file():
            paths.append("tokenizer_config.json")
        prior_manifest = (self.artifacts.manifest if self.artifacts is not None else None)
        source = (
            prior_manifest.source if prior_manifest is not None else
            (self.artifacts.source if self.artifacts is not None else OUTETTS_SOURCE_REPOSITORY))
        revision = (
            prior_manifest.source_revision if prior_manifest is not None else
            (self.artifacts.revision if self.artifacts is not None else OUTETTS_SOURCE_REVISION))
        reference = OUTETTS_CHECKPOINTS.get(source)
        manifest = VoiceHubManifest(
            architecture="outetts",
            architecture_version="1",
            checkpoint_format=NATIVE_OUTETTS_FORMAT,
            adapter_version="1",
            source=source,
            source_revision=revision,
            source_license=(
                prior_manifest.source_license if prior_manifest is not None else OUTETTS_SOURCE_LICENSE),
            weight_license=(
                prior_manifest.weight_license if prior_manifest is not None else
                (reference["license"] if reference is not None else None)),
            processor_assets=(
                "tokenizer.json",
                "default_speaker.json",
                "dac/config.json",
                "dac/model.safetensors",
            ),
            training_recipe=("completion-only-causal-language-modeling@" + OUTETTS_TRAINING_SOURCE_REVISION),
            files=build_manifest_files(destination, paths),
            metadata={
                "codec": "descript-dac-24khz-1.5kbps",
                "codec_frozen": True,
                "interface_version": 3,
                "repetition_window": 64,
                "runtime": "voicehub-native",
            },
        )
        manifest.save(destination)
        return destination.resolve()


def load_outetts_runtime(
    source: str | Path,
    *,
    tokenizer_source: str | Path | None = None,
    revision: str | None = None,
    codec_source: str | Path | None = None,
    codec_revision: str | None = None,
    cache_dir: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
) -> OuteTTSRuntime:
    artifacts = resolve_outetts_artifacts(
        source,
        tokenizer_source=tokenizer_source,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    tokenizer = OuteTTSTokenizer.from_tokenizer_json(
        artifacts.tokenizer,
        tokenizer_config_path=artifacts.tokenizer_config,
    )
    language_model, model_config = load_outetts_language_model(
        artifacts,
        device=device,
        dtype=dtype,
    )
    if tokenizer.family != model_config.model_type:
        compatible = (tokenizer.family == "qwen3" and model_config.model_type in {"qwen2", "qwen3"})
        if not compatible:
            raise ValueError(
                "OuteTTS tokenizer and language-model families disagree: "
                f"{tokenizer.family!r} vs {model_config.model_type!r}.")
    if tokenizer.token_id_space_size > model_config.vocab_size:
        raise ValueError("OuteTTS tokenizer declares token IDs outside the language-model "
                         "vocabulary.")
    local_codec = artifacts.root / "dac"
    resolved_codec_source = (local_codec if codec_source is None and local_codec.is_dir() else codec_source)
    codec_artifacts = resolve_outetts_dac_artifacts(
        resolved_codec_source,
        revision=codec_revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    codec = load_outetts_dac(
        codec_artifacts,
        device=device,
        # The released DAC is float32. Keeping it in its audited precision
        # avoids compounding codec error when the language model uses BF16.
        dtype=torch.float32,
    )
    return OuteTTSRuntime(
        language_model,
        tokenizer,
        codec,
        artifacts=artifacts,
        codec_artifacts=codec_artifacts,
    )


__all__ = ["OuteTTSRuntime", "load_outetts_runtime"]
