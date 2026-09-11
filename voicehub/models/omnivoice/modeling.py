"""Public TTS API backed entirely by VoiceHub-native OmniVoice code."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from voicehub.models._shared import finish_audio_output, resolve_torch_dtype, validate_local_file
from voicehub.models.omnivoice.configuration import OmniVoiceConfig
from voicehub.models.tts import PreTrainedTTSModel
from voicehub.outputs import TTSOutput

_PREPARED_KEYS = frozenset({
    "attention_mask",
    "audio_mask",
    "document_ids",
    "input_ids",
    "labels",
    "position_ids",
})


def _batch_item(value: Any, index: int, count: int) -> Any:
    if isinstance(value, Mapping):
        return {name: _batch_item(item, index, count) for name, item in value.items()}
    if isinstance(value, (str, bytes, bytearray)):
        return value
    shape = getattr(value, "shape", ())
    if shape and int(shape[0]) == count:
        return value[index]
    if isinstance(value, Sequence) and len(value) == count:
        return value[index]
    return value


def _training_records(inputs: Mapping[str, Any]) -> list[dict[str, Any]]:
    supplied = inputs.get("records")
    if supplied is not None:
        if (isinstance(supplied, (str, bytes, Mapping)) or not isinstance(supplied, Sequence) or
                not supplied):
            raise ValueError("OmniVoice `records` must be a non-empty sequence.")
        if any(not isinstance(record, Mapping) for record in supplied):
            raise TypeError("Every OmniVoice training record must be a mapping.")
        return [dict(record) for record in supplied]
    text = inputs.get("text")
    if isinstance(text, str):
        return [dict(inputs)]
    if (isinstance(text, Sequence) and not isinstance(text, (str, bytes, bytearray)) and text):
        count = len(text)
        return [{
            name: _batch_item(value, index, count)
            for name, value in inputs.items() if name != "training_phase"
        } for index in range(count)]
    raise ValueError("Raw OmniVoice training inputs require `text` or `records`.")


class OmniVoiceForTextToSpeech(PreTrainedTTSModel):
    """Native multilingual cloning, voice design, and fine-tuning."""

    config_class = OmniVoiceConfig
    default_model_name_or_path = "k2-fsa/OmniVoice"

    def __init__(
        self,
        config: OmniVoiceConfig | str | None = None,
        *,
        model_path: str | None = None,
        device: str = "auto",
        lazy_load: bool = True,
        token: str | bool | None = None,
        **config_overrides: Any,
    ) -> None:
        config = self._coerce_config(
            config,
            model_path=model_path,
            **config_overrides,
        )
        self._runtime = None
        self._hub_token = token
        super().__init__(config, device=device, lazy_load=lazy_load)

    @property
    def native_runtime(self):
        return self._runtime

    def _load_pretrained_model(self) -> None:
        from voicehub.models.omnivoice.native.runtime import OmniVoiceRuntime, load_omnivoice_runtime

        if self._runtime is not None:
            if not isinstance(self._runtime, OmniVoiceRuntime):
                raise TypeError("Injected OmniVoice runtime must be an OmniVoiceRuntime.")
            runtime = self._runtime
        else:
            import torch

            runtime = load_omnivoice_runtime(
                self.config.name_or_path,
                revision=self.config.revision,
                codec_source=self.config.codec_source,
                codec_revision=self.config.codec_revision,
                device=self.device,
                dtype=resolve_torch_dtype(
                    torch,
                    self.config.torch_dtype,
                    self.device,
                ),
                cache_dir=self.config.cache_dir,
                token=self._hub_token,
                local_files_only=self.config.local_files_only,
                verify_integrity=self.config.verify_integrity,
                verify_checkpoint_integrity=(self.config.verify_checkpoint_integrity),
            )
            self._runtime = runtime
        self.model = runtime.model
        self.config.sample_rate = runtime.sample_rate

    def _prepare_for_training(self) -> None:
        if self._runtime is None:
            raise RuntimeError("Native OmniVoice runtime was not loaded.")
        from voicehub.models.omnivoice.native.processing import OmniVoiceMaskingConfig, OmniVoiceSampleProcessor

        masking = OmniVoiceMaskingConfig(**(self.config.training_masking_config or {}))
        self._runtime.sample_processor = OmniVoiceSampleProcessor(
            self._runtime.text_tokenizer,
            self._runtime.model.config,
            masking=masking,
            audio_tokenizer=self._runtime.audio_tokenizer,
        )
        self._runtime.prepare_for_training()
        self.model = self._runtime.model

    def _prepare_for_inference(self) -> None:
        if self._runtime is None:
            raise RuntimeError("Native OmniVoice runtime was not loaded.")
        if self.model is not self._runtime.model:
            self._runtime.model = self.model
            self._runtime.generator.model = self.model
        self._runtime.prepare_for_inference()

    def _set_training_device(self, device: str) -> None:
        super()._set_training_device(device)
        if self._runtime is not None:
            self._runtime.audio_tokenizer.to(device=device)

    def _validate_training_runtime(self) -> None:
        if self.config.use_safetensors is False:
            raise ValueError("Native OmniVoice training requires Safetensors.")

    def _validate_generation_inputs(self, model_inputs: dict[str, Any]) -> None:
        super()._validate_generation_inputs(model_inputs)
        speaker = validate_local_file(
            model_inputs.get("speaker_audio_path"),
            option_name="speaker_audio_path",
        )
        reference_text = model_inputs.get("reference_text")
        if speaker is not None:
            if not isinstance(reference_text, str) or not reference_text.strip():
                raise ValueError(
                    "Native OmniVoice voice cloning requires a non-empty "
                    "`reference_text` with `speaker_audio_path`.")
            model_inputs["speaker_audio_path"] = str(speaker)
        elif reference_text is not None:
            raise ValueError("`reference_text` requires `speaker_audio_path`.")
        for name in ("text", "language", "instruct"):
            value = model_inputs.get(name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"`{name}` must be a non-empty string or None.")
        for name in ("duration", "speed"):
            value = model_inputs.get(name)
            if value is None:
                continue
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                    not math.isfinite(float(value)) or value <= 0):
                raise ValueError(f"`{name}` must be finite and positive.")
        if model_inputs.get("normalize_text", False):
            raise ValueError(
                "Native OmniVoice does not silently use external text "
                "normalizers; normalize text before this boundary.")
        if not isinstance(model_inputs.get("preprocess_prompt", True), bool):
            raise TypeError("`preprocess_prompt` must be a boolean.")
        for name in ("top_p", "max_new_tokens"):
            if model_inputs.get(name) is not None:
                raise ValueError(f"OmniVoice iterative decoding does not support `{name}`.")

    def _generation_config(self, overrides: Mapping[str, Any]):
        from voicehub.models.omnivoice.native.generation import OmniVoiceGenerationConfig

        values = dict(self.config.generation_config)
        values.update({name: value for name, value in overrides.items() if value is not None})
        return OmniVoiceGenerationConfig(**values)

    def _generate(
        self,
        text: str,
        *,
        output_file: str | None = None,
        language: str | None = None,
        speaker_audio_path: str | None = None,
        reference_text: str | None = None,
        instruct: str | None = None,
        speed: float = 1.0,
        duration: float | None = None,
        normalize_text: bool = False,
        seed: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_new_tokens: int | None = None,
        num_step: int | None = None,
        num_steps: int | None = None,
        guidance_scale: float | None = None,
        t_shift: float | None = None,
        time_shift: float | None = None,
        layer_penalty_factor: float | None = None,
        position_temperature: float | None = None,
        class_temperature: float | None = None,
        denoise: bool | None = None,
        preprocess_prompt: bool = True,
        postprocess_output: bool | None = None,
        audio_chunk_duration: float | None = None,
        audio_chunk_threshold: float | None = None,
        pad_duration: float | None = None,
        fade_duration: float | None = None,
    ) -> TTSOutput:
        del normalize_text, top_p, max_new_tokens
        if self._runtime is None:
            raise RuntimeError("Native OmniVoice runtime was not loaded.")
        if temperature is not None:
            # ``class_temperature`` is present in the model generation
            # defaults, so the common ``temperature`` alias necessarily
            # reaches this hook alongside it. The explicit common option
            # takes precedence.
            class_temperature = temperature
        # The singular spellings are retained for compatibility with the
        # original OmniVoice CLI. They override the native defaults supplied
        # under the canonical plural names.
        if num_step is not None:
            num_steps = num_step
        if t_shift is not None:
            time_shift = t_shift
        prompt = None
        if speaker_audio_path is not None:
            prompt = self._runtime.create_prompt(
                speaker_audio_path,
                reference_text=reference_text,
                preprocess_prompt=preprocess_prompt,
            )
        generation = self._generation_config({
            "audio_chunk_duration": audio_chunk_duration,
            "audio_chunk_threshold": audio_chunk_threshold,
            "class_temperature": class_temperature,
            "denoise": denoise,
            "fade_duration": fade_duration,
            "guidance_scale": guidance_scale,
            "layer_penalty_factor": layer_penalty_factor,
            "num_steps": num_steps,
            "pad_duration": pad_duration,
            "position_temperature": position_temperature,
            "postprocess_output": postprocess_output,
            "time_shift": time_shift,
        })
        waveform = self._runtime.generate(
            text,
            prompt=prompt,
            language=language,
            instruction=instruct,
            duration=duration,
            speed=speed,
            generation_config=generation,
            seed=seed,
        ).detach().float().cpu()
        return finish_audio_output(
            waveform,
            self.sample_rate,
            output_file=output_file,
            metadata={
                "architecture": "omnivoice",
                "backend": "voicehub-native",
                "checkpoint_format": "safetensors",
                "language": language,
                "requested_seed": seed,
                "seed": seed,
                "voice_cloning": prompt is not None,
            },
        )

    def prepare_training_inputs(
        self,
        inputs: dict[str, Any],
        *,
        phase: str,
    ) -> dict[str, Any]:
        del phase
        if not isinstance(inputs, Mapping):
            raise TypeError("OmniVoice training inputs must be a mapping.")
        if {"input_ids", "audio_mask", "labels"} <= set(inputs):
            return {name: value for name, value in inputs.items() if name in _PREPARED_KEYS}
        if self._runtime is None:
            raise RuntimeError("OmniVoice preparation requires a loaded native runtime.")
        return self._runtime.prepare_training_inputs(
            _training_records(inputs),
            packing_tokens=self.config.training_packing_tokens,
        )

    def _save_pretrained(self, save_directory: Path) -> None:
        self.load()
        if self._runtime is None:
            raise RuntimeError("Only native OmniVoice runtimes can be exported.")
        self._runtime.save_pretrained(save_directory)

    def export_native_pretrained(self, save_directory: str | Path) -> Path:
        self.load()
        if self._runtime is None:
            raise RuntimeError("Only native OmniVoice runtimes can be exported.")
        return self._runtime.save_pretrained(save_directory)


OmniVoiceTTS = OmniVoiceForTextToSpeech

__all__ = ["OmniVoiceForTextToSpeech", "OmniVoiceTTS"]
