"""Public TTS API backed entirely by VoiceHub-native CosyVoice code."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicehub.models._shared import finish_audio_output, resolve_torch_dtype
from voicehub.models.cosyvoice.configuration import CosyVoiceConfig
from voicehub.models.tts import PreTrainedTTSModel
from voicehub.outputs import TTSOutput


class CosyVoiceForTextToSpeech(PreTrainedTTSModel):
    """CosyVoice 3 inference and source-component fine-tuning."""

    config_class = CosyVoiceConfig
    default_model_name_or_path = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"

    def __init__(
        self,
        config: CosyVoiceConfig | str | None = None,
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

    def _tts_optimization_runtime(self, mode):
        """Include the optional native speech tokenizer during inference."""
        normalized_mode = self._optimization_mode(mode)
        if normalized_mode.value == "inference":
            if self._runtime is None:
                raise RuntimeError("Native CosyVoice runtime was not loaded.")
            return self._runtime
        return super()._tts_optimization_runtime(normalized_mode)

    def _load_pretrained_model(self) -> None:
        from voicehub.models.cosyvoice.native.runtime import CosyVoiceNativeRuntime, load_cosyvoice_runtime

        if self._runtime is not None:
            if not isinstance(self._runtime, CosyVoiceNativeRuntime):
                raise TypeError("Injected CosyVoice runtime has the wrong type.")
            runtime = self._runtime
        else:
            import torch

            runtime = load_cosyvoice_runtime(
                self.config.name_or_path,
                revision=self.config.revision,
                device=self.device,
                dtype=resolve_torch_dtype(
                    torch,
                    self.config.torch_dtype,
                    self.device,
                ),
                cache_dir=self.config.cache_dir,
                token=self._hub_token,
                local_files_only=self.config.local_files_only,
            )
            self._runtime = runtime
        self.model = runtime.model
        self.config.sample_rate = runtime.sample_rate

    def _validate_training_runtime(self) -> None:
        if self.config.use_safetensors is False:
            raise ValueError("CosyVoice fine-tuning requires Safetensors.")

    def _prepare_for_training(self) -> None:
        if self._runtime is None:
            raise RuntimeError("Native CosyVoice runtime was not loaded.")
        self._runtime.prepare_for_training(self.config.training_component)
        self.model = self._runtime.model

    def _prepare_for_inference(self) -> None:
        if self._runtime is None:
            raise RuntimeError("Native CosyVoice runtime was not loaded.")
        if self.model is not self._runtime.model:
            self._runtime.model = self.model
        self._runtime.prepare_for_inference()

    def _validate_generation_inputs(self, model_inputs: dict[str, Any]) -> None:
        super()._validate_generation_inputs(model_inputs)
        speaker_embedding = model_inputs.get("speaker_embedding")
        if speaker_embedding is None:
            raise ValueError(
                "Native CosyVoice requires a precomputed "
                "`speaker_embedding`. The frozen CAMPPlus frontend is not "
                "silently executed.")
        expected = 192
        if self._runtime is not None:
            expected = self._runtime.model.config.flow.speaker_embedding_dim
        shape = getattr(speaker_embedding, "shape", ())
        if shape and tuple(shape) not in {(expected, ), (1, expected)}:
            raise ValueError(f"`speaker_embedding` must have shape [{expected}] or "
                             f"[1, {expected}].")
        for name in ("temperature", "top_p"):
            value = model_inputs.get(name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or
                                      not math.isfinite(float(value)) or value <= 0):
                raise ValueError(f"`{name}` must be finite and positive.")
        seed = model_inputs.get("seed")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
            raise ValueError("`seed` must be a non-negative integer or None.")
        prompt_audio = model_inputs.get("prompt_audio")
        speaker_audio_path = model_inputs.get("speaker_audio_path")
        if prompt_audio is not None and speaker_audio_path is not None:
            raise ValueError("Use either `prompt_audio` or the `speaker_audio_path` alias, "
                             "not both.")
        raw_prompt = (prompt_audio if prompt_audio is not None else speaker_audio_path)
        raw_tokenizer_missing = (self._runtime is None or not self._runtime.supports_raw_speech_tokens)
        if raw_prompt is not None and raw_tokenizer_missing:
            raise ValueError(
                "Raw prompt audio requires an attached native CosyVoice "
                "speech tokenizer. Supply `prompt_speech_tokens` or load an "
                "artifact containing `speech_tokenizer.safetensors`.")
        if (raw_prompt is not None and model_inputs.get("prompt_speech_tokens") is not None):
            raise ValueError("Supply either raw prompt audio or `prompt_speech_tokens`, "
                             "not both.")

    def _generate(
        self,
        text: str,
        *,
        output_file: str | None = None,
        speaker_embedding,
        instruction: str | None = None,
        prompt_speech_tokens=None,
        prompt_audio=None,
        prompt_audio_sample_rate: int | None = None,
        prompt_features=None,
        seed: int | None = None,
        min_new_tokens: int | None = None,
        max_new_tokens: int | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        temperature: float | None = None,
        flow_steps: int | None = None,
        speaker_audio_path: str | None = None,
    ) -> TTSOutput:
        if self._runtime is None:
            raise RuntimeError("Native CosyVoice runtime was not loaded.")
        if prompt_audio is not None and speaker_audio_path is not None:
            raise ValueError("Use either `prompt_audio` or `speaker_audio_path`, not both.")
        if prompt_audio is None:
            prompt_audio = speaker_audio_path
        values = dict(self.config.generation_config)
        values.update({
            name: value
            for name, value in {
                "flow_steps": flow_steps,
                "max_new_tokens": max_new_tokens,
                "min_new_tokens": min_new_tokens,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
            }.items() if value is not None
        })
        output = self._runtime.generate(
            text,
            speaker_embedding=speaker_embedding,
            instruction=instruction,
            prompt_speech_tokens=prompt_speech_tokens,
            prompt_audio=prompt_audio,
            prompt_audio_sample_rate=prompt_audio_sample_rate,
            prompt_features=prompt_features,
            seed=seed,
            **values,
        )
        audio = output.waveform.detach().float().cpu()
        if audio.ndim == 2 and audio.shape[0] == 1:
            audio = audio[0]
        return finish_audio_output(
            audio,
            output.sample_rate,
            output_file=output_file,
            metadata={
                "architecture": "cosyvoice3",
                "backend": "voicehub-native",
                "checkpoint_format": "safetensors",
                "flow_steps": values["flow_steps"],
                "seed": seed,
                "speech_token_count": output.speech_tokens.shape[-1],
            },
        )

    def prepare_training_inputs(
        self,
        inputs: Mapping[str, Any],
        *,
        phase: str,
    ) -> dict[str, Any]:
        if not isinstance(inputs, Mapping):
            raise TypeError("CosyVoice training inputs must be a mapping.")
        if self._runtime is None:
            raise RuntimeError("CosyVoice training preparation requires a runtime.")
        if phase in {"llm", "language_model"}:
            records = inputs.get("records")
            if records is None:
                records = [dict(inputs)]
            return self._runtime.prepare_language_batch(
                records,
                device=self.device,
            )
        return dict(inputs)

    def export_native_pretrained(self, directory: str | Path) -> Path:
        if self._runtime is None:
            raise RuntimeError("CosyVoice export requires a loaded runtime.")
        if self.model is not self._runtime.model:
            self._runtime.model = self.model
        return self._runtime.save_pretrained(directory)


CosyVoiceTTS = CosyVoiceForTextToSpeech

__all__ = [
    "CosyVoiceForTextToSpeech",
    "CosyVoiceTTS",
]
