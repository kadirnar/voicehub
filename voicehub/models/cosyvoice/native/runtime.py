"""Inference, fine-tuning, and export lifecycle for native CosyVoice."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from voicehub.hub import read_json_file, write_json_file
from voicehub.models.cosyvoice.native.artifacts import resolve_cosyvoice_artifacts
from voicehub.models.cosyvoice.native.checkpoint import export_cosyvoice_checkpoint, load_cosyvoice_checkpoint
from voicehub.models.cosyvoice.native.configuration import CosyVoiceArchitectureConfig
from voicehub.models.cosyvoice.native.modeling import CosyVoiceNativeModel, CosyVoiceSynthesisOutput
from voicehub.models.cosyvoice.native.speech_tokenizer import CosyVoiceSpeechTokenizer, CosyVoiceSpeechTokenizerConfig
from voicehub.models.cosyvoice.native.tokenization import CosyVoiceTextTokenizer
from voicehub.optimization.protocols import OptimizationCompileTarget
from voicehub.processing.waveform import NativeAudio, load_native_audio


class CosyVoiceNativeRuntime:
    """Own one native graph and its immutable text tokenizer assets."""

    def __init__(
        self,
        model: CosyVoiceNativeModel,
        tokenizer: CosyVoiceTextTokenizer,
        speech_tokenizer: CosyVoiceSpeechTokenizer | None = None,
    ) -> None:
        if not isinstance(model, CosyVoiceNativeModel):
            raise TypeError("`model` must be CosyVoiceNativeModel.")
        if not isinstance(tokenizer, CosyVoiceTextTokenizer):
            raise TypeError("`tokenizer` must be CosyVoiceTextTokenizer.")
        if (speech_tokenizer is not None and not isinstance(speech_tokenizer, CosyVoiceSpeechTokenizer)):
            raise TypeError("`speech_tokenizer` must be CosyVoiceSpeechTokenizer or None.")
        self.model = model
        self.tokenizer = tokenizer
        self.speech_tokenizer = speech_tokenizer

    @property
    def sample_rate(self) -> int:
        return self.model.config.sample_rate

    @property
    def supports_raw_speech_tokens(self) -> bool:
        return self.speech_tokenizer is not None

    def optimization_module_roots(self, ) -> tuple[tuple[str, torch.nn.Module], ...]:
        roots: list[tuple[str, torch.nn.Module]] = [
            ("cosyvoice.model", self.model),
        ]
        if self.speech_tokenizer is not None:
            roots.append((
                "cosyvoice.speech_tokenizer",
                self.speech_tokenizer,
            ))
        return tuple(roots)

    def optimization_compile_targets(
        self,
        mode: str,
    ) -> tuple[OptimizationCompileTarget, ...]:
        """Expose model stages plus the optional raw-audio tokenizer."""
        targets = list(self.model.optimization_compile_targets(mode))
        if self.speech_tokenizer is not None:
            targets.extend(self.speech_tokenizer.codec_optimization_compile_targets(mode, ))
        return tuple(targets)

    def attach_speech_tokenizer(
        self,
        speech_tokenizer: CosyVoiceSpeechTokenizer,
    ) -> None:
        if not isinstance(speech_tokenizer, CosyVoiceSpeechTokenizer):
            raise TypeError("`speech_tokenizer` must be CosyVoiceSpeechTokenizer.")
        self.speech_tokenizer = speech_tokenizer.freeze()

    def prepare_for_training(self, component: str) -> torch.nn.Module:
        if self.speech_tokenizer is not None:
            self.speech_tokenizer.freeze()
        normalized = str(component).strip().lower().replace("-", "_")
        if normalized.startswith("hifigan"):
            self.model.attach_discriminator(tiny=(self.model.config.hift.base_channels < 64), )
            if normalized == "hifigan_discriminator":
                selected = self.model.hifigan.discriminator
                for parameter in self.model.parameters():
                    parameter.requires_grad_(False)
                for parameter in selected.parameters():
                    parameter.requires_grad_(True)
                return selected
        return self.model.freeze_except(normalized)

    def prepare_for_inference(self) -> None:
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        if self.speech_tokenizer is not None:
            self.speech_tokenizer.freeze()

    def _load_speech_audio(
        self,
        value: Any,
        *,
        sampling_rate: int | None,
    ) -> NativeAudio:
        native_types = (NativeAudio, Mapping, str, Path)
        if sampling_rate is None and not isinstance(value, native_types):
            sampling_rate = 16_000
        return load_native_audio(
            value,
            sampling_rate=sampling_rate,
            target_sampling_rate=16_000,
        )

    @torch.inference_mode()
    def extract_speech_tokens(
        self,
        audio: Any,
        *,
        sampling_rate: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        if self.speech_tokenizer is None:
            raise RuntimeError(
                "Raw speech-token extraction requires an attached native "
                "`speech_tokenizer.safetensors`; supply precomputed tokens "
                "or attach the converted tokenizer.")
        loaded = self._load_speech_audio(
            audio,
            sampling_rate=sampling_rate,
        )
        return self.speech_tokenizer.encode_waveforms((loaded, ))

    @staticmethod
    def _record_speech_audio(record: Mapping[str, Any]) -> Any | None:
        for name in (
                "speech_audio",
                "audio",
                "waveform",
                "audio_path",
        ):
            if record.get(name) is not None:
                return record[name]
        return None

    @staticmethod
    def _record_sampling_rate(record: Mapping[str, Any]) -> int | None:
        for name in (
                "speech_sampling_rate",
                "sampling_rate",
                "sample_rate",
        ):
            if record.get(name) is not None:
                return record[name]
        return None

    def prepare_language_batch(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        device: str | torch.device | None = None,
    ) -> dict[str, Tensor]:
        if not records:
            raise ValueError("CosyVoice language batch cannot be empty.")
        text_items: list[Tensor] = []
        instruction_items: list[Tensor] = []
        speech_items: list[Tensor | None] = []
        raw_audio: list[NativeAudio] = []
        raw_indices: list[int] = []
        for record in records:
            if not isinstance(record, Mapping):
                raise TypeError("Every CosyVoice training record must be a mapping.")
            supplied_text = record.get("text_tokens")
            if supplied_text is None:
                supplied_text = self.tokenizer.encode_tensor(
                    record["text"],
                    device=device,
                )[0]
            text_items.append(torch.as_tensor(
                supplied_text,
                dtype=torch.long,
                device=device,
            ).flatten())
            supplied_instruction = record.get("instruction_tokens")
            if supplied_instruction is None:
                supplied_instruction = self.tokenizer.instruction_tokens(
                    record.get("instruction"),
                    device=device,
                )[0]
            instruction_items.append(
                torch.as_tensor(
                    supplied_instruction,
                    dtype=torch.long,
                    device=device,
                ).flatten())
            supplied_speech = record.get("speech_tokens")
            if supplied_speech is None:
                audio = self._record_speech_audio(record)
                if audio is None:
                    raise ValueError(
                        "CosyVoice SFT records need pre-encoded "
                        "`speech_tokens` or raw "
                        "`speech_audio`/`audio`/`waveform`/`audio_path`.")
                if self.speech_tokenizer is None:
                    raise ValueError(
                        "Raw CosyVoice SFT audio requires an attached native "
                        "speech tokenizer; this runtime accepts precomputed "
                        "`speech_tokens` without it.")
                raw_indices.append(len(speech_items))
                raw_audio.append(
                    self._load_speech_audio(
                        audio,
                        sampling_rate=self._record_sampling_rate(record),
                    ))
                speech_items.append(None)
            else:
                speech_items.append(
                    torch.as_tensor(
                        supplied_speech,
                        dtype=torch.long,
                        device=device,
                    ).flatten())

        if raw_audio:
            assert self.speech_tokenizer is not None
            encoded, encoded_lengths = self.speech_tokenizer.encode_waveforms(raw_audio)
            for raw_index, record_index in enumerate(raw_indices):
                speech_items[record_index] = encoded[raw_index, :encoded_lengths[raw_index]].to(device=device)
        if any(item is None for item in speech_items):
            raise RuntimeError("CosyVoice speech-token preparation was incomplete.")
        resolved_speech_items = [item for item in speech_items if item is not None]

        def pad(items: list[Tensor], value: int = 0) -> tuple[Tensor, Tensor]:
            lengths = torch.tensor(
                [item.numel() for item in items],
                dtype=torch.long,
                device=device,
            )
            maximum = int(lengths.max().item())
            result = items[0].new_full((len(items), maximum), value)
            for index, item in enumerate(items):
                result[index, :item.numel()] = item
            return result, lengths

        text, text_lengths = pad(text_items, self.tokenizer.pad_token_id)
        instruction, instruction_lengths = pad(
            instruction_items,
            self.tokenizer.pad_token_id,
        )
        speech, speech_lengths = pad(resolved_speech_items)
        return {
            "text_tokens": text,
            "text_lengths": text_lengths,
            "instruction_tokens": instruction,
            "instruction_lengths": instruction_lengths,
            "speech_tokens": speech,
            "speech_lengths": speech_lengths,
        }

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        *,
        speaker_embedding: Tensor,
        instruction: str | None = None,
        prompt_speech_tokens: Tensor | None = None,
        prompt_audio: Any | None = None,
        prompt_audio_sample_rate: int | None = None,
        prompt_features: Tensor | None = None,
        min_new_tokens: int = 0,
        max_new_tokens: int = 1_024,
        top_k: int = 25,
        top_p: float = 0.8,
        temperature: float = 1.0,
        flow_steps: int = 10,
        seed: int | None = None,
    ) -> CosyVoiceSynthesisOutput:
        device = next(self.model.parameters()).device
        text_tokens = self.tokenizer.encode_tensor(text, device=device)
        instruction_tokens = self.tokenizer.instruction_tokens(
            instruction,
            device=device,
        )
        speaker_embedding = torch.as_tensor(
            speaker_embedding,
            device=device,
            dtype=next(self.model.flow.parameters()).dtype,
        )
        if speaker_embedding.ndim == 1:
            speaker_embedding = speaker_embedding[None]
        expected = self.model.config.flow.speaker_embedding_dim
        if tuple(speaker_embedding.shape) != (1, expected):
            raise ValueError(f"`speaker_embedding` must have shape [{expected}] or [1, {expected}].")
        if prompt_speech_tokens is not None and prompt_audio is not None:
            raise ValueError("Supply either `prompt_speech_tokens` or raw `prompt_audio`, "
                             "not both.")
        if prompt_audio is not None:
            prompt_speech_tokens, _ = self.extract_speech_tokens(
                prompt_audio,
                sampling_rate=prompt_audio_sample_rate,
            )
        if prompt_speech_tokens is not None:
            prompt_speech_tokens = prompt_speech_tokens.to(
                device=device,
                dtype=torch.long,
            )
            if prompt_speech_tokens.ndim == 1:
                prompt_speech_tokens = prompt_speech_tokens[None]
        if prompt_features is not None:
            prompt_features = prompt_features.to(
                device=device,
                dtype=speaker_embedding.dtype,
            )
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)
        self.prepare_for_inference()
        return self.model.synthesize(
            text_tokens,
            instruction_tokens,
            speaker_embedding,
            prompt_speech_tokens=prompt_speech_tokens,
            prompt_features=prompt_features,
            min_new_tokens=min_new_tokens,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            flow_steps=flow_steps,
            generator=generator,
        )

    def save_pretrained(self, directory: str | Path) -> Path:
        target = Path(directory).expanduser()
        target.mkdir(parents=True, exist_ok=True)
        occupied = [path for path in target.iterdir()]
        if occupied:
            raise FileExistsError("CosyVoice export destination must be empty: "
                                  f"{target}.")
        write_json_file(
            target / "cosyvoice_config.json",
            self.model.config.to_dict(),
        )
        export_cosyvoice_checkpoint(
            self.model.llm,
            target / "llm.safetensors",
            component="llm",
        )
        export_cosyvoice_checkpoint(
            self.model.flow,
            target / "flow.safetensors",
            component="flow",
        )
        export_cosyvoice_checkpoint(
            self.model.hift,
            target / "hift.safetensors",
            component="hift",
        )
        if self.speech_tokenizer is not None:
            write_json_file(
                target / "speech_tokenizer_config.json",
                self.speech_tokenizer.config.to_dict(),
            )
            export_cosyvoice_checkpoint(
                self.speech_tokenizer,
                target / "speech_tokenizer.safetensors",
                component="speech_tokenizer",
            )
        self.tokenizer.save_pretrained(target)
        return target


def load_cosyvoice_runtime(
    source: str | Path,
    *,
    revision: str | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
    cache_dir: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
    require_official_inventory: bool = False,
) -> CosyVoiceNativeRuntime:
    """Strictly load one complete pickle-free native artifact."""
    artifacts = resolve_cosyvoice_artifacts(
        source,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    config = CosyVoiceArchitectureConfig.from_dict(read_json_file(artifacts.config))
    with torch.device("meta"):
        model = CosyVoiceNativeModel(
            config,
            initialize=False,
            dtype=dtype,
        )
    for component, module, path in (
        ("llm", model.llm, artifacts.llm),
        ("flow", model.flow, artifacts.flow),
        ("hift", model.hift, artifacts.hift),
    ):
        load_cosyvoice_checkpoint(
            module,
            path,
            component=component,
            device=device,
            dtype=dtype,
            require_official_inventory=require_official_inventory,
        )
    speech_tokenizer = None
    if artifacts.speech_tokenizer is not None:
        speech_tokenizer_config = (
            CosyVoiceSpeechTokenizerConfig() if artifacts.speech_tokenizer_config is None else
            CosyVoiceSpeechTokenizerConfig.from_dict(read_json_file(artifacts.speech_tokenizer_config)))
        with torch.device("meta"):
            speech_tokenizer = CosyVoiceSpeechTokenizer(speech_tokenizer_config)
        load_cosyvoice_checkpoint(
            speech_tokenizer,
            artifacts.speech_tokenizer,
            component="speech_tokenizer",
            device=device,
            dtype=dtype,
            require_official_inventory=(speech_tokenizer_config == CosyVoiceSpeechTokenizerConfig()),
        )
        speech_tokenizer.freeze()
    tokenizer = CosyVoiceTextTokenizer.from_files(
        artifacts.vocab,
        artifacts.merges,
        artifacts.tokenizer_config,
        validate_published_ids=(config.generation == 3 and config.language.text_vocab_size == 151_936),
    )
    return CosyVoiceNativeRuntime(
        model,
        tokenizer,
        speech_tokenizer,
    )


__all__ = [
    "CosyVoiceNativeRuntime",
    "load_cosyvoice_runtime",
]
