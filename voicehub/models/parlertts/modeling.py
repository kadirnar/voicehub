"""VoiceHub-native Parler-TTS inference and fine-tuning lifecycle."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from voicehub.hub import read_json_file
from voicehub.models._shared import finish_audio_output, resolve_torch_dtype
from voicehub.models.parlertts.configuration import ParlerTTSConfig
from voicehub.models.parlertts.native.artifacts import resolve_parlertts_artifacts
from voicehub.models.parlertts.native.checkpoint import load_parlertts_checkpoint
from voicehub.models.parlertts.native.configuration import ParlerTTSArchitectureConfig
from voicehub.models.parlertts.native.metadata import PARLER_TTS_CHECKPOINT
from voicehub.models.parlertts.native.modeling import ParlerTTSForConditionalGeneration as NativeParlerTTS
from voicehub.models.parlertts.native.modeling import prepare_audio_code_labels
from voicehub.models.parlertts.native.processing import ParlerTextTokenizer
from voicehub.models.tts import PreTrainedTTSModel
from voicehub.outputs import TTSOutput

DEFAULT_DESCRIPTION = (
    "A clear, expressive speaker delivers high-quality speech at a moderate "
    "speed and pitch in a close, noise-free recording.")


class ParlerTTSForTextToSpeech(PreTrainedTTSModel):
    """Prompt-controlled TTS implemented with PyTorch and VoiceHub only."""

    config_class = ParlerTTSConfig
    default_model_name_or_path = PARLER_TTS_CHECKPOINT

    def __init__(
        self,
        config: ParlerTTSConfig | str | None = None,
        *,
        model_path: str | None = None,
        device: str = "auto",
        lazy_load: bool = True,
        **config_overrides: Any,
    ) -> None:
        config = self._coerce_config(
            config,
            model_path=model_path,
            **config_overrides,
        )
        config.validate()
        self.tokenizer: ParlerTextTokenizer | None = None
        self.artifacts = None
        self.architecture_config: ParlerTTSArchitectureConfig | None = None
        self.generation_defaults: dict[str, Any] = {}
        super().__init__(config, device=device, lazy_load=lazy_load)

    def _load_pretrained_model(self) -> None:
        artifacts = resolve_parlertts_artifacts(
            self.config.name_or_path,
            revision=self.config.revision,
            cache_dir=self.config.cache_dir,
            token=self.config.token,
            local_files_only=self.config.local_files_only,
            verify_integrity=self.config.verify_artifacts,
            verify_checkpoint_integrity=self.config.verify_checkpoint,
        )
        config_values = (
            self.config.architecture if self.config.architecture else read_json_file(artifacts.config))
        architecture = ParlerTTSArchitectureConfig.from_dict(config_values)
        generation_values = read_json_file(artifacts.generation_config)
        self.generation_defaults = {
            "max_length": generation_values.get("max_length", 2_580),
            "min_new_tokens": generation_values.get("min_new_tokens", 10),
            "do_sample": generation_values.get("do_sample", True),
            "temperature": generation_values.get("temperature", 1.0),
            "top_k": generation_values.get("top_k", 50),
            "top_p": generation_values.get("top_p"),
            "repetition_penalty": generation_values.get(
                "repetition_penalty",
                1.0,
            ),
            "guidance_scale": generation_values.get("guidance_scale", 1.0),
        }
        dtype = resolve_torch_dtype(
            torch,
            self.config.torch_dtype,
            self.device,
        )
        # Build on meta first so dtype conversion does not transiently allocate
        # both FP32 and reduced-precision copies of this 878M-parameter graph.
        with torch.device("meta"):
            model = NativeParlerTTS(
                architecture,
                attention_implementation=self.config.attention_implementation,
            )
        if dtype != torch.float32:
            model.to(dtype=dtype)
        model.to_empty(device=self.device)
        load_parlertts_checkpoint(
            model,
            artifacts.checkpoint,
            require_official_inventory=artifacts.official_snapshot,
        )
        if self.config.compile_model and not self.is_training_load:
            model.decoder.model.decoder = torch.compile(model.decoder.model.decoder)
        self.model = model
        self.tokenizer = ParlerTextTokenizer.from_model_file(
            artifacts.tokenizer_model,
            eos_token_id=architecture.text_encoder.eos_token_id,
            pad_token_id=architecture.text_encoder.pad_token_id,
            model_vocabulary_size=architecture.text_encoder.vocab_size,
        )
        self.artifacts = artifacts
        self.architecture_config = architecture
        self.config.sample_rate = architecture.sampling_rate

    def _prepare_for_training(self) -> None:
        if self.model is None:
            return
        self.model.train()
        self.model.freeze_encoders(freeze_text_encoder=self.config.freeze_text_encoder, )
        self.model.audio_encoder.eval()

    def _prepare_for_inference(self) -> None:
        if self.model is not None:
            self.model.eval()

    def _validate_generation_inputs(self, model_inputs: dict[str, Any]) -> None:
        description = model_inputs.get("description", DEFAULT_DESCRIPTION)
        if not isinstance(description, str) or not description.strip():
            raise ValueError("`description` must be a non-empty voice description.")

    def _tokenize(self, text: str):
        if self.tokenizer is None:
            raise RuntimeError("Parler-TTS tokenizer is not loaded.")
        return self.tokenizer(text, device=self.device)

    @staticmethod
    def _extract_waveform(generation):
        audio = getattr(generation, "audio_values", None)
        if audio is None:
            audio = getattr(generation, "sequences", generation)
        if audio is None or not hasattr(audio, "detach"):
            raise RuntimeError("Parler-TTS returned no tensor audio waveform.")
        if hasattr(audio, "numel") and audio.numel() == 0:
            raise RuntimeError("Parler-TTS returned an empty audio waveform.")
        return audio.detach().float().cpu().squeeze().contiguous()

    def prepare_training_inputs(
        self,
        inputs: Mapping[str, Any],
        *,
        phase: Any,
    ) -> dict[str, Any]:
        """Tokenize raw text fields while accepting precomputed DAC labels."""
        del phase
        batch = dict(inputs)
        if "description" in batch and "input_ids" not in batch:
            encoded = self._tokenize(batch.pop("description"))
            batch["input_ids"] = encoded.input_ids
            batch["attention_mask"] = encoded.attention_mask
        if "text" in batch and "prompt_input_ids" not in batch:
            encoded = self._tokenize(batch.pop("text"))
            batch["prompt_input_ids"] = encoded.input_ids
            batch["prompt_attention_mask"] = encoded.attention_mask
        audio_lengths = batch.pop("audio_lengths", None)
        audio_code_lengths = batch.pop("audio_code_lengths", None)
        audio_codes = batch.pop("audio_codes", None)
        audio_values = batch.pop("audio_values", None)
        input_values = batch.pop("input_values", None)
        if audio_values is not None and input_values is not None:
            raise ValueError("Provide only one of `audio_values` and `input_values`.")
        audio_values = (input_values if audio_values is None else audio_values)
        if "labels" not in batch:
            if audio_codes is None and audio_values is not None:
                self.model.audio_encoder.eval()
                with torch.no_grad():
                    audio_codes = self.model.audio_encoder.encode(audio_values)
                if audio_lengths is not None:
                    hop_length = self.architecture_config.audio_encoder.hop_length
                    audio_code_lengths = torch.div(
                        torch.as_tensor(
                            audio_lengths,
                            device=audio_codes.device,
                        ) + hop_length - 1,
                        hop_length,
                        rounding_mode="floor",
                    ).clamp_max(audio_codes.shape[-1])
            if audio_codes is not None:
                batch["labels"] = prepare_audio_code_labels(
                    audio_codes,
                    bos_token_id=self.architecture_config.decoder.bos_token_id,
                    eos_token_id=self.architecture_config.decoder.eos_token_id,
                    audio_code_lengths=audio_code_lengths,
                )
        return batch

    def _generate(
        self,
        text: str,
        *,
        description: str = DEFAULT_DESCRIPTION,
        output_file: str | None = None,
        seed: int | None = None,
        **generation_options: Any,
    ) -> TTSOutput:
        description_batch = self._tokenize(description)
        prompt_batch = self._tokenize(text)
        options = dict(self.generation_defaults)
        options.update(generation_options)
        guidance_scale = options.pop("guidance_scale", 1.0)
        if guidance_scale != 1.0:
            raise ValueError("Native Parler-TTS currently supports guidance_scale=1 only.")
        generation = self.model.generate(
            description_batch.input_ids,
            attention_mask=description_batch.attention_mask,
            prompt_input_ids=prompt_batch.input_ids,
            prompt_attention_mask=prompt_batch.attention_mask,
            seed=seed,
            **options,
        )
        return finish_audio_output(
            self._extract_waveform(generation),
            self.sample_rate,
            output_file=output_file,
            metadata={
                "description": description,
                "seed": seed,
                "requested_seed": seed,
                "runtime": "voicehub-native",
            },
        )


ParlerVoiceHubConfig = ParlerTTSConfig
ParlerTTS = ParlerTTSForTextToSpeech
