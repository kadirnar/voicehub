"""Native Granite Speech inference, fine-tuning, and safe export."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral
from pathlib import Path
from typing import Any

from voicehub.models.asr_granite_speech.configuration import GraniteSpeechASRConfig
from voicehub.models.audio import PreTrainedASRModel
from voicehub.outputs import ASROutput

_RAW_TRAINING_FIELDS = frozenset({
    "audio",
    "audio_lengths",
    "language",
    "prompt",
    "sample_rate",
    "sampling_rate",
    "text",
    "transcript",
    "transcription",
})

_TRANSLATION_LANGUAGES = {
    "chinese": ("zh", "Mandarin"),
    "cmn": ("zh", "Mandarin"),
    "de": ("de", "German"),
    "deu": ("de", "German"),
    "english": ("en", "English"),
    "en": ("en", "English"),
    "eng": ("en", "English"),
    "es": ("es", "Spanish"),
    "spa": ("es", "Spanish"),
    "fr": ("fr", "French"),
    "fra": ("fr", "French"),
    "fre": ("fr", "French"),
    "french": ("fr", "French"),
    "german": ("de", "German"),
    "it": ("it", "Italian"),
    "ita": ("it", "Italian"),
    "italian": ("it", "Italian"),
    "ja": ("ja", "Japanese"),
    "japanese": ("ja", "Japanese"),
    "jpn": ("ja", "Japanese"),
    "mandarin": ("zh", "Mandarin"),
    "por": ("pt", "Portuguese"),
    "portuguese": ("pt", "Portuguese"),
    "pt": ("pt", "Portuguese"),
    "spanish": ("es", "Spanish"),
    "zh": ("zh", "Mandarin"),
    "zho": ("zh", "Mandarin"),
}


def _batch_values(
    value: Any,
    *,
    batch_size: int,
    name: str,
    default: Any = None,
) -> tuple[Any, ...]:
    if value is None:
        return (default, ) * batch_size
    if isinstance(value, (str, bytes, Path)):
        return (value, ) * batch_size
    try:
        import torch
    except ModuleNotFoundError:  # pragma: no cover - package invariant
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return (value.item(), ) * batch_size
        if value.ndim != 1:
            raise ValueError(f"`{name}` must be scalar or one-dimensional.")
        values = tuple(value.detach().cpu().tolist())
    elif isinstance(value, Sequence):
        values = tuple(value)
    else:
        return (value, ) * batch_size
    if len(values) != batch_size:
        raise ValueError(f"`{name}` contains {len(values)} values for a batch of "
                         f"{batch_size}.")
    return values


class GraniteSpeechForSpeechRecognition(PreTrainedASRModel):
    """Run IBM Granite Speech Safetensors with VoiceHub-owned code."""

    config_class = GraniteSpeechASRConfig
    default_model_name_or_path = ("ibm-granite/granite-speech-4.1-2b")
    architecture_family = "speech-seq2seq"
    native_checkpoint_format = "native-granite-speech-v1"
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: GraniteSpeechASRConfig | str | Path | None = None,
        *,
        model_path: str | Path | None = None,
        device: str = "auto",
        lazy_load: bool = True,
        token: str | bool | None = None,
        **kwargs: Any,
    ) -> None:
        if token is not None and not isinstance(token, (str, bool)):
            raise TypeError("`token` must be a string, boolean, or None.")
        if isinstance(token, str) and not token.strip():
            raise ValueError("String `token` values must be non-empty.")
        self._hub_token = token
        self.runtime: Any | None = None
        self.artifacts: Any | None = None
        self.native_config: Any | None = None
        self.granite_processor: Any | None = None
        self.training_processor: Any | None = None
        self.transformers_processor: Any | None = None
        self._lora_injection: Any | None = None
        self._lora_base_trainability: dict[str, bool] | None = None
        config = self._coerce_config(
            config,
            model_path=model_path,
            **kwargs,
        )
        super().__init__(
            config,
            device=device,
            lazy_load=lazy_load,
        )

    def prepare_inputs_for_inference(
        self,
        audio: Any,
        *,
        sampling_rate: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return {
            "audio": audio,
            "sampling_rate": sampling_rate,
            **kwargs,
        }

    def _load_pretrained_model(self) -> None:
        from voicehub.models.asr_granite_speech.native.runtime import load_granite_speech_runtime

        source = (self.config.name_or_path or self.default_model_name_or_path)
        runtime = load_granite_speech_runtime(
            source,
            device=self.device,
            compute_dtype=self.config.torch_dtype,
            revision=self.config.revision,
            cache_dir=self.config.cache_dir,
            token=self._hub_token,
            local_files_only=self.config.local_files_only,
            for_training=self.is_training_load,
        )
        self.runtime = runtime
        self.artifacts = runtime.artifacts
        self.native_config = runtime.config
        self.granite_processor = runtime.processor
        self.training_processor = runtime.processor
        # Transitional attribute name; the value is a VoiceHub processor.
        self.transformers_processor = runtime.processor
        self.model = runtime.model

    @staticmethod
    def _translation_language(language: str | None, ) -> tuple[str, str]:
        if not isinstance(language, str) or not language.strip():
            raise ValueError("Granite Speech translation requires a target `language`.")
        normalized = (language.strip().lower().replace("_", "-"))
        try:
            return _TRANSLATION_LANGUAGES[normalized]
        except KeyError as error:
            supported = ("English, French, German, Spanish, Portuguese, Japanese, "
                         "Italian, and Mandarin")
            raise ValueError(
                "Unsupported Granite Speech translation target "
                f"{language!r}. Supported targets: {supported}.") from error

    @staticmethod
    def _validate_request(
        *,
        language: str | None,
        task: str,
        return_timestamps: bool | str,
        chunk_length_s: float | None,
        stride_length_s: float | tuple[float, float] | None,
        batch_size: int | None,
        num_beams: int | None,
    ) -> None:
        if task not in {"transcribe", "translate"}:
            raise ValueError("Granite Speech `task` must be 'transcribe' or "
                             "'translate'.")
        if task == "transcribe" and language is not None:
            raise ValueError(
                "Granite Speech does not expose language-ID forcing. Put "
                "language guidance in `prompt` instead.")
        if task == "translate":
            GraniteSpeechForSpeechRecognition._translation_language(language, )
        if return_timestamps is not False:
            raise ValueError("Granite Speech does not emit timestamps.")
        if chunk_length_s is not None:
            raise ValueError(
                "Granite Speech has no checkpoint-validated overlapping "
                "chunk protocol; `chunk_length_s` is unsupported.")
        if stride_length_s is not None:
            raise ValueError(
                "Granite Speech has no checkpoint-validated stride "
                "stitching protocol; `stride_length_s` is unsupported.")
        if batch_size not in (None, 1):
            raise ValueError("One public transcription request requires `batch_size=1`.")
        if num_beams not in (None, 1):
            raise ValueError(
                "The native Granite Speech decoder supports greedy or "
                "sampling generation; `num_beams` must be 1 or None.")

    def _transcribe(
        self,
        audio: Any,
        *,
        sampling_rate: int | None = None,
        language: str | None = None,
        task: str = "transcribe",
        return_timestamps: bool | str = False,
        chunk_length_s: float | None = None,
        stride_length_s: float | tuple[float, float] | None = None,
        batch_size: int | None = None,
        num_beams: int | None = None,
        max_new_tokens: int | None = None,
        hotwords: str | tuple[str, ...] | list[str] | None = None,
        prompt: str | None = None,
        do_sample: bool | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        min_p: float | None = None,
        repetition_penalty: float | None = None,
        seed: int | None = None,
    ) -> ASROutput:
        import torch

        from voicehub.generation.config import GenerationConfig

        self._validate_request(
            language=language,
            task=task,
            return_timestamps=return_timestamps,
            chunk_length_s=chunk_length_s,
            stride_length_s=stride_length_s,
            batch_size=batch_size,
            num_beams=num_beams,
        )
        if self.model is None or self.runtime is None:
            raise RuntimeError("Granite Speech runtime is not loaded.")
        processor = self.granite_processor
        if processor is None:
            raise RuntimeError("Granite Speech processor is not loaded.")
        output_language = None
        resolved_prompt = prompt
        if task == "translate":
            output_language, target_name = self._translation_language(language, )
            if resolved_prompt is None:
                resolved_prompt = (
                    "<|audio|>translate the speech to "
                    f"{target_name} with proper punctuation and "
                    "capitalization.")
        prepared = processor.prepare_inference_batch(
            (audio, ),
            sampling_rates=(sampling_rate, ),
            prompts=(self.config.transcription_prompt if resolved_prompt is None else resolved_prompt),
            hotwords=hotwords,
        )
        defaults = dict(self.runtime.generation_config)
        generation = GenerationConfig(
            max_new_tokens=(512 if max_new_tokens is None else max_new_tokens),
            do_sample=(bool(defaults.get("do_sample", False)) if do_sample is None else do_sample),
            temperature=(float(defaults.get("temperature", 1.0)) if temperature is None else temperature),
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            repetition_penalty=(1.0 if repetition_penalty is None else repetition_penalty),
            eos_token_id=defaults.get(
                "eos_token_id",
                self.native_config.text_config.eos_token_id,
            ),
            pad_token_id=int(defaults.get(
                "pad_token_id",
                self.native_config.text_config.pad_token_id,
            )),
            seed=seed,
            use_cache=True,
        )
        input_ids = prepared["input_ids"]
        if (input_ids.shape[1] + generation.max_new_tokens
                > self.native_config.text_config.max_position_embeddings):
            raise ValueError(
                "Granite Speech prompt plus `max_new_tokens` exceeds the "
                "checkpoint context window.")
        parameter = next(self.model.parameters())
        input_ids = input_ids.to(parameter.device)
        attention_mask = prepared["attention_mask"].to(parameter.device, )
        input_features = prepared["input_features"].to(
            device=parameter.device,
            dtype=parameter.dtype,
        )
        feature_mask = prepared["input_features_mask"].to(parameter.device, )
        with torch.inference_mode():
            output = self.model.generate(
                input_ids,
                input_features=input_features,
                input_features_mask=feature_mask,
                attention_mask=attention_mask,
                generation_config=generation,
            )
        completion = output.sequences[
            0,
            input_ids.shape[1]:,
        ].detach().cpu().tolist()
        text = processor.tokenizer.decode(
            completion,
            skip_special_tokens=True,
        ).strip()
        duration = (int(prepared["audio_lengths"][0].item()) / processor.sample_rate)
        return ASROutput(
            text=text,
            segments=(),
            language=output_language,
            duration=duration,
            metadata={
                "architecture": "granite-speech",
                "architecture_family": self.architecture_family,
                "backend": "voicehub-native",
                "checkpoint_revision": self.artifacts.revision,
                "decoding": ("sampling" if generation.do_sample else "greedy"),
                "generated_tokens": int(output.generated_lengths[0].item(), ),
                "task": task,
                "timestamps": False,
            },
        )

    @staticmethod
    def _training_audio_rows(
        audio: Any,
        *,
        batch_size: int,
    ) -> tuple[Any, ...]:
        import torch

        if isinstance(audio, torch.Tensor):
            if audio.ndim == 1:
                if batch_size != 1:
                    raise ValueError("Batched transcripts require rank-two audio.")
                return (audio, )
            if audio.ndim == 2:
                if audio.shape[0] != batch_size:
                    raise ValueError("Audio and transcript batch sizes do not match.")
                return tuple(audio[index] for index in range(batch_size))
            raise ValueError("Granite Speech training audio must be rank one or two.")
        if batch_size == 1:
            return (audio, )
        if (isinstance(audio, Sequence) and not isinstance(audio, (str, bytes, bytearray, Path))):
            rows = tuple(audio)
            if len(rows) == batch_size:
                return rows
        raise ValueError("Batched transcripts require one audio value per row.")

    def prepare_training_inputs(
        self,
        inputs: dict[str, Any],
        *,
        phase: str,
    ) -> dict[str, Any]:
        """Build completion-only causal labels from raw audio and text."""
        from voicehub.processing.waveform import load_native_audio

        del phase
        if "input_ids" in inputs:
            if "labels" not in inputs:
                raise ValueError("Cached Granite Speech batches must include "
                                 "completion-only `labels`.")
            return dict(inputs)
        if self.model is None:
            self.load_for_training()
        processor = self.granite_processor
        if processor is None:
            raise RuntimeError("Granite Speech training processor is not loaded.")
        language = inputs.get("language")
        if language is not None:
            if isinstance(language, str):
                values = (language, )
            elif isinstance(language, Sequence):
                values = tuple(language)
            else:
                raise TypeError("Granite Speech training `language` must be a string, "
                                "sequence, or None.")
            if any(value is not None for value in values):
                raise ValueError(
                    "Granite Speech fine-tuning is prompt-conditioned. "
                    "Express language guidance in `prompt`.")
        text = inputs.get(
            "text",
            inputs.get(
                "transcription",
                inputs.get("transcript"),
            ),
        )
        if text is None:
            raise ValueError("Granite Speech training records require "
                             "`text`/`transcription`.")
        if isinstance(text, str):
            texts = (text, )
        elif (isinstance(text, Sequence) and not isinstance(text, (str, bytes))):
            texts = tuple(text)
        else:
            raise TypeError("Granite Speech transcripts must be a string or sequence.")
        if not texts or any(not isinstance(value, str) or not value.strip() for value in texts):
            raise ValueError("Granite Speech transcripts must be non-empty strings.")
        audio = inputs.get("audio")
        if audio is None:
            raise ValueError("Granite Speech training records require `audio`.")
        audios = list(self._training_audio_rows(
            audio,
            batch_size=len(texts),
        ))
        rates = _batch_values(
            inputs.get(
                "sampling_rate",
                inputs.get("sample_rate"),
            ),
            batch_size=len(audios),
            name="sampling_rate",
        )
        length_values = _batch_values(
            inputs.get("audio_lengths"),
            batch_size=len(audios),
            name="audio_lengths",
        )
        for length in length_values:
            if length is None:
                continue
            if (isinstance(length, bool) or not isinstance(length, Integral) or length <= 0):
                raise ValueError("`audio_lengths` must contain positive integers.")
        target_rate = processor.sample_rate
        audios = [
            load_native_audio(
                value,
                sampling_rate=rate,
                target_sampling_rate=target_rate,
                num_samples=length,
            ).waveform for value, rate, length in zip(
                audios,
                rates,
                length_values,
            )
        ]
        prompts = _batch_values(
            inputs.get("prompt"),
            batch_size=len(audios),
            name="prompt",
            default=self.config.transcription_prompt,
        )
        prepared = processor.prepare_training_batch(
            tuple(audios),
            tuple(value.strip() for value in texts),
            sampling_rates=(target_rate, ) * len(audios),
            prompts=prompts,
        )
        if (prepared["input_ids"].shape[1] > self.native_config.text_config.max_position_embeddings):
            raise ValueError("A Granite Speech training example exceeds the checkpoint "
                             "context window.")
        for name, value in inputs.items():
            if (name not in _RAW_TRAINING_FIELDS and name not in prepared):
                prepared[name] = value
        return prepared

    def enable_lora(
        self,
        *,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        target_modules: tuple[str, ...] = (
            "*.q_proj",
            "*.k_proj",
            "*.v_proj",
            "*.o_proj",
            "*.to_q",
            "*.to_kv",
            "*.to_out",
        ),
        freeze_base: bool = True,
        seed: int = 0,
    ) -> Any:
        """Inject VoiceHub-native trainable adapters into the loaded graph."""
        from voicehub.optimization import LoRAConfig, inject_lora

        self.load_for_training()
        if self._lora_injection is not None:
            raise RuntimeError("Granite Speech LoRA is already enabled.")
        original = {name: parameter.requires_grad for name, parameter in self.model.named_parameters()}
        if freeze_base:
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)
        try:
            self._lora_injection = inject_lora(
                self.model,
                LoRAConfig(
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                    target_modules=target_modules,
                    freeze_base=freeze_base,
                    seed=seed,
                ),
            )
        except BaseException:
            for name, parameter in self.model.named_parameters():
                parameter.requires_grad_(original[name])
            raise
        self._lora_base_trainability = original
        return self._lora_injection

    def disable_lora(self) -> None:
        if self._lora_injection is None:
            return
        self._lora_injection.restore()
        self._lora_injection = None
        if self._lora_base_trainability is not None:
            for name, parameter in self.model.named_parameters():
                parameter.requires_grad_(self._lora_base_trainability[name], )
        self._lora_base_trainability = None

    def _portable_state_dict(self) -> dict[str, Any]:
        if self._lora_injection is None:
            return dict(self.model.state_dict())
        portable = {}
        for name, tensor in self.model.state_dict().items():
            if name.endswith((".lora_a", ".lora_b")):
                continue
            target_name = name.replace(".base.", ".")
            value = tensor.detach()
            if name.endswith(".base.weight"):
                module_name = name[:-len(".base.weight")]
                module = self._lora_injection.modules.get(module_name)
                if module is None:
                    raise RuntimeError(
                        "Granite Speech export found an untracked LoRA "
                        f"base module at {module_name!r}.")
                if not module.merged:
                    value = value + module.adapter_delta().detach().to(
                        device=value.device,
                        dtype=value.dtype,
                    )
            if target_name in portable:
                raise RuntimeError(
                    "Granite Speech LoRA export produced duplicate tensor "
                    f"{target_name!r}.")
            portable[target_name] = value
        return portable

    def _save_pretrained(self, save_directory: Path) -> None:
        from voicehub.models.asr_granite_speech.native.runtime import save_granite_speech_runtime

        if self.runtime is None or self.model is None:
            self.load()
        self.runtime.model = self.model
        save_granite_speech_runtime(
            self.runtime,
            save_directory,
            state_dict=self._portable_state_dict(),
        )

    def export_native_pretrained(
        self,
        save_directory: str | Path,
    ) -> Path:
        destination = Path(save_directory).expanduser()
        self._save_pretrained(destination)
        return destination


__all__ = ["GraniteSpeechForSpeechRecognition"]
