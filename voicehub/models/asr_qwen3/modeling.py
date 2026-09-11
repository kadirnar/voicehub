"""Native Qwen3-ASR inference, fine-tuning, and portable export."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral
from pathlib import Path
from typing import Any

from voicehub.models.asr_qwen3.configuration import Qwen3ASRConfig
from voicehub.models.audio import PreTrainedASRModel
from voicehub.outputs import ASROutput

_RAW_TRAINING_FIELDS = frozenset({
    "audio",
    "audio_lengths",
    "context",
    "language",
    "prompt",
    "sample_rate",
    "sampling_rate",
    "text",
    "transcript",
    "transcription",
})


def _batch_values(
    value: Any,
    *,
    batch_size: int,
    name: str,
    default: Any = None,
) -> tuple[Any, ...]:
    """Broadcast a scalar or validate one value per training row."""
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


class Qwen3ASRForSpeechRecognition(PreTrainedASRModel):
    """Run official Qwen3-ASR Safetensors with VoiceHub-owned code."""

    config_class = Qwen3ASRConfig
    default_model_name_or_path = "Qwen/Qwen3-ASR-0.6B"
    architecture_family = "speech-seq2seq"
    native_checkpoint_format = "native-qwen3-asr-v1"
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: Qwen3ASRConfig | str | Path | None = None,
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
        self.qwen3_processor: Any | None = None
        # Compatibility aliases deliberately point to native VoiceHub objects.
        self.training_processor: Any | None = None
        self.transformers_processor: Any | None = None
        self._lora_injection: Any | None = None
        self._lora_base_trainability: dict[str, bool] | None = None
        config = self._coerce_config(
            config,
            model_path=model_path,
            **kwargs,
        )
        super().__init__(config, device=device, lazy_load=lazy_load)

    def prepare_inputs_for_inference(
        self,
        audio: Any,
        *,
        sampling_rate: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Keep raw audio intact until the native runtime is loaded."""
        return {
            "audio": audio,
            "sampling_rate": sampling_rate,
            **kwargs,
        }

    def _load_pretrained_model(self) -> None:
        from voicehub.models.asr_qwen3.native.runtime import load_qwen3_asr_runtime

        source = self.config.name_or_path or self.default_model_name_or_path
        runtime = load_qwen3_asr_runtime(
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
        self.qwen3_processor = runtime.processor
        self.training_processor = runtime.processor
        self.transformers_processor = runtime.processor
        self.model = runtime.model

    @staticmethod
    def _validate_request(
        *,
        task: str,
        return_timestamps: bool | str,
        chunk_length_s: float | None,
        stride_length_s: float | tuple[float, float] | None,
        batch_size: int | None,
        num_beams: int | None,
    ) -> None:
        if task != "transcribe":
            raise ValueError("Qwen3-ASR has no translation decoding mode; "
                             "`task` must be 'transcribe'.")
        if return_timestamps is not False:
            raise ValueError(
                "Qwen3-ASR does not emit timestamps. Use the separate Qwen "
                "forced-aligner architecture after transcription.")
        if chunk_length_s is not None:
            raise ValueError(
                "Qwen3-ASR manages long audio with its validated native "
                "20-minute chunker; `chunk_length_s` is unsupported.")
        if stride_length_s is not None:
            raise ValueError(
                "Qwen3-ASR long-form chunks do not overlap; "
                "`stride_length_s` is unsupported.")
        if batch_size not in (None, 1):
            raise ValueError("One public transcription request requires `batch_size=1`.")
        if num_beams not in (None, 1):
            raise ValueError(
                "The native Qwen3-ASR decoder currently supports greedy or "
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
        context: str | None = None,
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
        from voicehub.models.asr_qwen3.native.processing import parse_qwen3_asr_output

        self._validate_request(
            task=task,
            return_timestamps=return_timestamps,
            chunk_length_s=chunk_length_s,
            stride_length_s=stride_length_s,
            batch_size=batch_size,
            num_beams=num_beams,
        )
        if prompt is not None and context is not None:
            raise ValueError("Pass only one of `prompt` or `context`.")
        if self.model is None or self.runtime is None:
            raise RuntimeError("Qwen3-ASR runtime is not loaded.")
        processor = self.qwen3_processor
        if processor is None:
            raise RuntimeError("Qwen3-ASR processor is not loaded.")
        resolved_context = prompt if prompt is not None else context
        resolved_context = "" if resolved_context is None else resolved_context
        forced_language = processor.normalize_language(language)
        materialized = processor.materialize_audio(
            audio,
            sampling_rate=sampling_rate,
        )
        chunks = processor.long_audio_chunks(
            materialized,
            sampling_rate=materialized.sampling_rate,
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
                (151_643, 151_645),
            ),
            pad_token_id=int(defaults.get("pad_token_id", 151_643)),
            seed=seed,
            use_cache=True,
        )
        parameter = next(self.model.parameters())
        texts: list[str] = []
        detected_languages: list[str] = []
        generated_tokens = 0
        with torch.inference_mode():
            for chunk in chunks:
                prepared = processor.prepare_inference_batch(
                    (chunk, ),
                    sampling_rates=(chunk.sampling_rate, ),
                    contexts=(resolved_context, ),
                    languages=(forced_language, ),
                    hotwords=(hotwords, ),
                )
                input_ids = prepared["input_ids"].to(parameter.device)
                attention_mask = prepared["attention_mask"].to(parameter.device)
                input_features = prepared["input_features"].to(
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
                feature_attention_mask = prepared["feature_attention_mask"].to(parameter.device)
                output = self.model.generate(
                    input_ids,
                    input_features=input_features,
                    attention_mask=attention_mask,
                    feature_attention_mask=feature_attention_mask,
                    generation_config=generation,
                )
                completion = output.sequences[
                    0,
                    input_ids.shape[1]:,
                ].detach().cpu()
                generated_tokens += int(output.generated_lengths[0].item())
                raw = processor.tokenizer.decode(
                    completion,
                    skip_special_tokens=True,
                )
                detected, text = parse_qwen3_asr_output(
                    raw,
                    forced_language=forced_language,
                )
                if text:
                    texts.append(text)
                if detected is not None:
                    detected_languages.append(detected)
        resolved_language = (
            forced_language or (
                detected_languages[0] if detected_languages and
                all(value == detected_languages[0] for value in detected_languages) else None))
        return ASROutput(
            text="".join(texts).strip(),
            segments=(),
            language=resolved_language,
            duration=materialized.duration,
            metadata={
                "architecture": "qwen3-asr",
                "architecture_family": self.architecture_family,
                "backend": "voicehub-native",
                "checkpoint_revision": self.artifacts.revision,
                "chunks": len(chunks),
                "decoding": "sampling" if generation.do_sample else "greedy",
                "generated_tokens": generated_tokens,
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
            raise ValueError("Qwen3-ASR training audio must be rank one or rank two.")
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
        """Build log-mel inputs and completion-only causal labels."""
        from voicehub.processing.waveform import load_native_audio

        del phase
        if "input_ids" in inputs:
            if "labels" not in inputs:
                raise ValueError(
                    "Cached Qwen3-ASR batches must include completion-only "
                    "`labels`; use the native processor when caching data.")
            return dict(inputs)
        if self.model is None:
            self.load_for_training()
        processor = self.qwen3_processor
        if processor is None:
            raise RuntimeError("Qwen3-ASR training processor is not loaded.")
        text = inputs.get(
            "text",
            inputs.get("transcription", inputs.get("transcript")),
        )
        if text is None:
            raise ValueError("Qwen3-ASR training records require "
                             "`text`/`transcription`.")
        if isinstance(text, str):
            texts = (text, )
        elif isinstance(text, Sequence) and not isinstance(text, (str, bytes)):
            texts = tuple(text)
        else:
            raise TypeError("Qwen3-ASR `text`/`transcription` must be a string or "
                            "sequence of strings.")
        if not texts or any(not isinstance(value, str) or not value.strip() for value in texts):
            raise ValueError("Qwen3-ASR transcripts must contain non-empty strings.")
        audio = inputs.get("audio")
        if audio is None:
            raise ValueError("Qwen3-ASR training records require `audio`.")
        audios = list(self._training_audio_rows(audio, batch_size=len(texts)))
        rates = _batch_values(
            inputs.get("sampling_rate", inputs.get("sample_rate")),
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
        target_rate = processor.feature_extractor.sample_rate
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
        contexts = _batch_values(
            inputs.get("context", inputs.get("prompt")),
            batch_size=len(audios),
            name="context",
            default="",
        )
        languages = _batch_values(
            inputs.get("language"),
            batch_size=len(audios),
            name="language",
            default=self.config.training_language,
        )
        prepared = processor.prepare_training_batch(
            tuple(audios),
            tuple(value.strip() for value in texts),
            sampling_rates=(target_rate, ) * len(audios),
            contexts=contexts,
            languages=languages,
        )
        for name, value in inputs.items():
            if name not in _RAW_TRAINING_FIELDS and name not in prepared:
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
            "*.out_proj",
        ),
        freeze_base: bool = True,
        seed: int = 0,
    ) -> Any:
        """Inject VoiceHub-native trainable adapters into the loaded graph."""
        from voicehub.optimization import LoRAConfig, inject_lora

        self.load_for_training()
        if self._lora_injection is not None:
            raise RuntimeError("Qwen3-ASR LoRA is already enabled.")
        original_trainability = {
            name: parameter.requires_grad
            for name, parameter in self.model.named_parameters()
        }
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
                parameter.requires_grad_(original_trainability[name])
            raise
        self._lora_base_trainability = original_trainability
        return self._lora_injection

    def disable_lora(self) -> None:
        """Restore the exact dense graph, discarding active adapters."""
        if self._lora_injection is None:
            return
        self._lora_injection.restore()
        self._lora_injection = None
        if self._lora_base_trainability is not None:
            for name, parameter in self.model.named_parameters():
                parameter.requires_grad_(self._lora_base_trainability[name])
        self._lora_base_trainability = None

    def _portable_state_dict(self) -> dict[str, Any]:
        """Return official-schema tensors for the current dense graph."""
        if self._lora_injection is None:
            return dict(self.model.state_dict())
        if not all(module.merged for module in self._lora_injection.modules.values()):
            raise RuntimeError(
                "LoRA modules must remain merged while their portable "
                "state is serialized.")
        return {
            name.replace(".base.", "."): tensor
            for name, tensor in self.model.state_dict().items() if not name.endswith((".lora_a", ".lora_b"))
        }

    def _save_pretrained(self, save_directory: Path) -> None:
        from voicehub.models.asr_qwen3.native.runtime import save_qwen3_asr_runtime

        if self.runtime is None or self.model is None:
            self.load()
        self.runtime.model = self.model
        if self._lora_injection is not None:
            self._lora_injection.merge()
        try:
            save_qwen3_asr_runtime(
                self.runtime,
                save_directory,
                state_dict=self._portable_state_dict(),
            )
        finally:
            if self._lora_injection is not None:
                self._lora_injection.unmerge()

    def export_native_pretrained(
        self,
        save_directory: str | Path,
    ) -> Path:
        destination = Path(save_directory).expanduser()
        self._save_pretrained(destination)
        return destination


__all__ = ["Qwen3ASRForSpeechRecognition"]
