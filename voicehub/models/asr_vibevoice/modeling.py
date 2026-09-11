"""Native VibeVoice ASR inference, fine-tuning, and export."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from pathlib import Path
from typing import Any

from voicehub.models.asr_vibevoice.configuration import VibeVoiceASRConfig
from voicehub.models.audio import PreTrainedASRModel
from voicehub.outputs import ASROutput, ASRSegment

_PREPROCESSED_FIELDS = frozenset({
    "attention_mask",
    "input_ids",
    "input_values",
    "labels",
    "padding_mask",
})
_RAW_FIELDS = frozenset({
    "audio",
    "audio_lengths",
    "context",
    "customized_context",
    "hotwords",
    "prompt",
    "sample_rate",
    "sampling_rate",
    "segments",
    "text",
    "transcript",
    "transcription",
})


def _batch_values(
    value: Any,
    *,
    batch_size: int,
    name: str,
) -> tuple[Any, ...]:
    if value is None or isinstance(value, (str, bytes, Path)):
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
        raise ValueError(f"`{name}` contains {len(values)} values for batch {batch_size}.")
    return values


class VibeVoiceForSpeechRecognition(PreTrainedASRModel):
    """Run and fine-tune Microsoft VibeVoice ASR with native VoiceHub code."""

    config_class = VibeVoiceASRConfig
    default_model_name_or_path = "microsoft/VibeVoice-ASR-HF"
    architecture_family = "causal-multimodal-lm"
    native_checkpoint_format = "native-vibevoice-asr-v1"
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: VibeVoiceASRConfig | str | Path | None = None,
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
        self.vibevoice_processor: Any | None = None
        self.training_processor: Any | None = None
        self.transformers_processor: Any | None = None
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
        from voicehub.models.vibevoice.native.runtime import load_vibevoice_runtime

        source = self.config.name_or_path or self.default_model_name_or_path
        runtime = load_vibevoice_runtime(
            source,
            device=self.device,
            compute_dtype=self.config.torch_dtype,
            revision=self.config.revision,
            cache_dir=self.config.cache_dir,
            token=self._hub_token,
            local_files_only=self.config.local_files_only,
            for_training=self.is_training_load,
        )
        from voicehub.models.vibevoice.native.configuration import VibeVoiceASRConfig as NativeASRConfig

        if not isinstance(runtime.config, NativeASRConfig):
            raise TypeError("VibeVoice ASR provider received a TTS checkpoint.")
        self.runtime = runtime
        self.artifacts = runtime.artifacts
        self.native_config = runtime.config
        self.vibevoice_processor = runtime.processor
        self.training_processor = runtime.processor
        # Compatibility alias only; the object is VoiceHub-native.
        self.transformers_processor = runtime.processor
        self.model = runtime.model

    def _prepare_for_training(self) -> None:
        if self.runtime is None:
            raise RuntimeError("VibeVoice ASR native runtime is not loaded.")
        self.runtime.prepare_for_training()

    def _prepare_for_inference(self) -> None:
        if self.runtime is None:
            raise RuntimeError("VibeVoice ASR native runtime is not loaded.")
        self.runtime.prepare_for_inference()

    @staticmethod
    def _context(
        *,
        prompt: str | None,
        hotwords: str | Sequence[str] | None,
        language: str | None,
    ) -> str | None:
        parts: list[str] = []
        if prompt is not None:
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("VibeVoice ASR prompt must be non-empty.")
            parts.append(prompt.strip())
        if hotwords is not None:
            if isinstance(hotwords, str):
                words = (hotwords, )
            elif isinstance(hotwords, Sequence):
                words = tuple(hotwords)
            else:
                raise TypeError("VibeVoice hotwords must be a string or sequence.")
            if any(not isinstance(word, str) or not word.strip() for word in words):
                raise ValueError("VibeVoice hotwords must be non-empty.")
            parts.append("Terminology: " + ", ".join(word.strip() for word in words))
        if language is not None:
            if not isinstance(language, str) or not language.strip():
                raise ValueError("VibeVoice ASR language must be non-empty.")
            parts.append("Expected language: " + language.strip())
        return "\n".join(parts) if parts else None

    @staticmethod
    def _validate_request(
        *,
        task: str,
        return_timestamps: bool | str,
        chunk_length_s: float | None,
        stride_length_s: float | tuple[float, float] | None,
        batch_size: int | None,
        num_beams: int | None,
        max_new_tokens: int | None,
    ) -> tuple[bool, int]:
        if task != "transcribe":
            raise ValueError("VibeVoice ASR does not implement speech translation.")
        if return_timestamps == "word":
            raise ValueError(
                "VibeVoice ASR returns speaker-segment timestamps; word "
                "timestamps require a separate aligner.")
        if return_timestamps not in (False, True, "segment"):
            raise ValueError("VibeVoice ASR timestamps accept False, True, or 'segment'.")
        if chunk_length_s is not None or stride_length_s is not None:
            raise ValueError(
                "VibeVoice owns its causal speech-encoder chunking; manual "
                "chunk and stride controls are unsupported.")
        if batch_size not in (None, 1):
            raise ValueError("One VibeVoice ASR request requires `batch_size=1`.")
        if num_beams not in (None, 1):
            raise ValueError("The published VibeVoice ASR generation config is greedy.")
        if max_new_tokens is None:
            maximum = 32_768
        elif (isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or
              not 1 <= max_new_tokens <= 32_768):
            raise ValueError("VibeVoice ASR `max_new_tokens` must be in 1..32768.")
        else:
            maximum = max_new_tokens
        return return_timestamps in (True, "segment"), maximum

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
        seed: int | None = None,
    ) -> ASROutput:
        import torch

        from voicehub.generation.sampling import create_generator
        from voicehub.processing.waveform import load_native_audio

        timestamps, maximum = self._validate_request(
            task=task,
            return_timestamps=return_timestamps,
            chunk_length_s=chunk_length_s,
            stride_length_s=stride_length_s,
            batch_size=batch_size,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
        )
        if self.model is None or self.vibevoice_processor is None:
            raise RuntimeError("VibeVoice ASR runtime is not loaded.")
        context = self._context(
            prompt=prompt,
            hotwords=hotwords,
            language=language,
        )
        materialized = load_native_audio(
            audio,
            sampling_rate=sampling_rate,
            target_sampling_rate=24_000,
        )
        prepared = self.vibevoice_processor(
            materialized.waveform,
            sampling_rate=24_000,
            prompt=context,
        )
        parameter = next(self.model.parameters())
        generator = create_generator(parameter.device, seed)
        model_inputs = {
            "input_ids": prepared.input_ids.to(parameter.device),
            "attention_mask": prepared.attention_mask.to(parameter.device),
            "input_values": prepared.input_values.to(
                device=parameter.device,
                dtype=parameter.dtype,
            ),
            "padding_mask": prepared.padding_mask.to(parameter.device),
        }
        with torch.inference_mode():
            generated = self.model.generate(
                **model_inputs,
                max_new_tokens=maximum,
                eos_token_id=self.vibevoice_processor.tokenizer.eos_token_id,
                generator=generator,
            )
        continuation = generated[0, prepared.input_ids.shape[1]:]
        parsed = self.vibevoice_processor.decode(
            continuation,
            return_format="parsed",
        )
        if isinstance(parsed, str):
            text = parsed.strip()
            segment_values: list[Mapping[str, Any]] = []
        else:
            segment_values = parsed
            text = " ".join(str(segment.get("Content", "")) for segment in parsed).strip()
        segments: tuple[ASRSegment, ...] = ()
        if timestamps:
            normalized_segments: list[ASRSegment] = []
            for segment in segment_values:
                start = max(
                    0.0,
                    min(
                        materialized.duration,
                        float(segment.get("Start", 0.0)),
                    ),
                )
                end = max(
                    0.0,
                    min(
                        materialized.duration,
                        float(segment.get("End", materialized.duration)),
                    ),
                )
                if end < start:
                    continue
                speaker = segment.get("Speaker")
                normalized_segments.append(
                    ASRSegment(
                        text=str(segment.get("Content", "")),
                        start=start,
                        end=end,
                        language=language,
                        speaker=(None if speaker is None else str(speaker)),
                    ))
            segments = tuple(normalized_segments)
        return ASROutput(
            text=text,
            segments=segments,
            language=language,
            duration=materialized.duration,
            metadata={
                "architecture": "vibevoice-asr",
                "architecture_family": "causal-multimodal-lm",
                "backend": "voicehub-native",
                "checkpoint_revision": (None if self.artifacts is None else self.artifacts.revision),
                "decoder": "greedy-autoregressive",
                "seed": seed,
                "structured_segments": len(segment_values),
            },
        )

    @staticmethod
    def _audio_batch(
        audio: Any,
        *,
        target_is_batch: bool,
    ) -> tuple[Any, ...]:
        import torch

        if isinstance(audio, torch.Tensor):
            if audio.ndim == 1:
                return (audio, )
            if audio.ndim == 2:
                return tuple(audio[index] for index in range(audio.shape[0]))
            raise ValueError("VibeVoice training audio must be rank one or two.")
        if target_is_batch:
            if isinstance(audio, (str, bytes, Path)) or not isinstance(audio, Sequence):
                raise ValueError("Batched VibeVoice targets require one audio value per row.")
            if audio and isinstance(audio[0], Real):
                raise ValueError("Batched VibeVoice targets received one flat waveform.")
            return tuple(audio)
        return (audio, )

    @staticmethod
    def _audio_has_batch_axis(audio: Any) -> bool:
        ndim = getattr(audio, "ndim", None)
        if ndim is not None:
            return ndim == 2
        return (
            isinstance(audio, Sequence) and not isinstance(audio, (str, bytes, Path)) and bool(audio) and
            not isinstance(audio[0], Real))

    def prepare_training_inputs(
        self,
        inputs: dict[str, Any],
        *,
        phase: str,
    ) -> dict[str, Any]:
        """Build author-format assistant completion targets from raw rows."""
        from voicehub.processing.waveform import load_native_audio

        del phase
        if _PREPROCESSED_FIELDS.issubset(inputs):
            return dict(inputs)
        if self.model is None:
            self.load_for_training()
        if self.vibevoice_processor is None:
            raise RuntimeError("VibeVoice ASR training processor is not loaded.")
        audio = inputs.get("audio")
        targets = inputs.get(
            "segments",
            inputs.get(
                "text",
                inputs.get(
                    "transcription",
                    inputs.get("transcript"),
                ),
            ),
        )
        if audio is None or targets is None:
            raise ValueError("VibeVoice ASR training requires audio and segment targets.")
        nested_targets = (
            isinstance(targets, Sequence) and targets and isinstance(targets[0], Sequence) and
            not isinstance(targets[0], (str, bytes, Mapping)))
        string_batch = (
            isinstance(targets, Sequence) and not isinstance(targets, (str, bytes)) and targets and
            isinstance(targets[0], str) and self._audio_has_batch_axis(audio))
        if nested_targets or string_batch:
            target_values = tuple(targets)
            target_is_batch = True
        else:
            target_values = (targets, )
            target_is_batch = False
        audio_values = self._audio_batch(
            audio,
            target_is_batch=target_is_batch,
        )
        if len(audio_values) != len(target_values):
            raise ValueError("VibeVoice training requires one target per waveform.")
        rates = _batch_values(
            inputs.get(
                "sampling_rate",
                inputs.get("sample_rate"),
            ),
            batch_size=len(audio_values),
            name="sampling_rate",
        )
        raw_lengths = _batch_values(
            inputs.get("audio_lengths"),
            batch_size=len(audio_values),
            name="audio_lengths",
        )
        from voicehub.processing.waveform import NativeAudio

        materialized = []
        for waveform, rate, raw_length in zip(
                audio_values,
                rates,
                raw_lengths,
        ):
            source = load_native_audio(
                waveform,
                sampling_rate=rate,
            )
            source_waveform = source.waveform
            if raw_length is not None:
                if (isinstance(raw_length, bool) or not isinstance(raw_length, Integral) or raw_length <= 0):
                    raise ValueError("`audio_lengths` must contain positive integers.")
                if int(raw_length) > source_waveform.numel():
                    raise ValueError("`audio_lengths` exceeds a waveform's sample count.")
                source_waveform = source_waveform[:int(raw_length)]
            materialized.append(
                load_native_audio(
                    NativeAudio(
                        waveform=source_waveform,
                        sampling_rate=source.sampling_rate,
                        path=source.path,
                    ),
                    target_sampling_rate=24_000,
                ))
        materialized = tuple(materialized)
        raw_context = inputs.get(
            "context",
            inputs.get(
                "prompt",
                inputs.get("customized_context"),
            ),
        )
        if (raw_context is not None and not isinstance(raw_context, str) and
                isinstance(raw_context, Sequence) and raw_context and
                all(isinstance(value, str) for value in raw_context) and len(audio_values) == 1):
            contexts = ("\n".join(raw_context), )
        else:
            contexts = _batch_values(
                raw_context,
                batch_size=len(audio_values),
                name="context",
            )
        prepared = self.vibevoice_processor.prepare_training_batch(
            tuple(item.waveform for item in materialized),
            target_values,
            sampling_rate=24_000,
            prompt=contexts,
        ).as_dict()
        for name, value in inputs.items():
            if name not in _RAW_FIELDS and name not in prepared:
                prepared[name] = value
        return prepared

    def _save_pretrained(self, save_directory: Path) -> None:
        if self.runtime is None:
            self.load()
        from voicehub.models.vibevoice.native.runtime import save_vibevoice_runtime

        save_vibevoice_runtime(self.runtime, save_directory)

    def export_native_pretrained(
        self,
        save_directory: str | Path,
    ) -> Path:
        destination = Path(save_directory).expanduser()
        self._save_pretrained(destination)
        return destination


__all__ = ["VibeVoiceForSpeechRecognition"]
