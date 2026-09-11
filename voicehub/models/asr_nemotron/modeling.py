"""Native Nemotron 3.5 ASR inference, fine-tuning, and export."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral
from pathlib import Path
from typing import Any

from voicehub.models.asr_nemotron.configuration import NemotronASRConfig
from voicehub.models.audio import PreTrainedASRModel
from voicehub.outputs import ASROutput, ASRSegment, ASRWord

_RAW_FIELDS = frozenset({
    "audio",
    "audio_lengths",
    "language",
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


class NemotronForSpeechRecognition(PreTrainedASRModel):
    """Run NVIDIA Nemotron 3.5 ASR with VoiceHub and PyTorch only."""

    config_class = NemotronASRConfig
    default_model_name_or_path = ("nvidia/nemotron-3.5-asr-streaming-0.6b")
    architecture_family = "rnnt"
    native_checkpoint_format = "voicehub-nemotron-3.5-rnnt-v1"
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: NemotronASRConfig | str | Path | None = None,
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
        self.nemotron_processor: Any | None = None
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
        from voicehub.models.asr_nemotron.native.runtime import load_nemotron_asr_runtime

        source = (self.config.name_or_path or self.default_model_name_or_path)
        runtime = load_nemotron_asr_runtime(
            source,
            device=self.device,
            compute_dtype=self.config.torch_dtype,
            revision=self.config.revision,
            cache_dir=self.config.cache_dir,
            token=self._hub_token,
            local_files_only=self.config.local_files_only,
            for_training=self.is_training_load,
        )
        if self.config.num_lookahead_tokens is not None:
            runtime.processor.set_num_lookahead_tokens(self.config.num_lookahead_tokens, )
        self.runtime = runtime
        self.artifacts = runtime.artifacts
        self.native_config = runtime.config
        self.nemotron_processor = runtime.processor
        self.training_processor = runtime.processor
        # Kept as a migration alias; it points to VoiceHub-native code.
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
        max_new_tokens: int | None,
        hotwords: str | tuple[str, ...] | list[str] | None,
    ) -> bool:
        if task != "transcribe":
            raise ValueError("Nemotron 3.5 ASR does not implement speech translation.")
        if return_timestamps not in (False, True, "word"):
            raise ValueError("Nemotron timestamps accept False, True, or 'word'.")
        if chunk_length_s is not None or stride_length_s is not None:
            raise ValueError(
                "Nemotron owns its cache-aware chunk geometry; common "
                "chunk/stride controls are unsupported.")
        if batch_size not in (None, 1):
            raise ValueError("One public Nemotron request requires `batch_size=1`.")
        if num_beams not in (None, 1):
            raise ValueError("Native Nemotron currently implements greedy RNN-T "
                             "decoding only.")
        if max_new_tokens is not None:
            raise ValueError(
                "Nemotron bounds emissions per encoder frame; "
                "`max_new_tokens` is unsupported.")
        if hotwords is not None:
            raise ValueError("Native Nemotron has no validated hotword decoder.")
        return return_timestamps in (True, "word")

    @staticmethod
    def _timestamp_words(
        offsets: Sequence[Mapping[str, Any]],
        *,
        duration: float,
    ) -> tuple[ASRWord, ...]:
        words: list[ASRWord] = []
        current_text = ""
        current_start: float | None = None
        current_end: float | None = None

        def flush() -> None:
            nonlocal current_text, current_start, current_end
            text = current_text.strip()
            if text:
                words.append(ASRWord(
                    text=text,
                    start=current_start,
                    end=current_end,
                ))
            current_text = ""
            current_start = None
            current_end = None

        for offset in offsets:
            token = str(offset.get("token", ""))
            if not token:
                continue
            start = min(duration, float(offset["start"]))
            end = min(duration, float(offset["end"]))
            boundary = token[:1].isspace()
            rendered = token.strip()
            if not rendered:
                continue
            punctuation = all(not character.isalnum() for character in rendered)
            if boundary and current_text and not punctuation:
                flush()
            if current_start is None:
                current_start = start
            current_text += rendered
            current_end = end
        flush()
        return tuple(words)

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
    ) -> ASROutput:
        import torch

        from voicehub.processing.waveform import load_native_audio

        timestamps = self._validate_request(
            task=task,
            return_timestamps=return_timestamps,
            chunk_length_s=chunk_length_s,
            stride_length_s=stride_length_s,
            batch_size=batch_size,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            hotwords=hotwords,
        )
        if (self.model is None or self.native_config is None or self.nemotron_processor is None):
            raise RuntimeError("Nemotron ASR runtime is not loaded.")
        if language is None:
            prompt_language = self.config.target_language
        elif not isinstance(language, str):
            raise TypeError("Nemotron language must be a string or None.")
        else:
            prompt_language = language.strip()
            if not prompt_language:
                raise ValueError("Nemotron language must be a non-empty string.")
        materialized = load_native_audio(
            audio,
            sampling_rate=sampling_rate,
            target_sampling_rate=16_000,
        )
        waveform = materialized.waveform
        if waveform.numel() < 160:
            waveform = torch.nn.functional.pad(
                waveform,
                (0, 160 - waveform.numel()),
            )
        parameter = next(self.model.parameters())
        prepared = self.nemotron_processor(
            waveform,
            sampling_rate=16_000,
            language=prompt_language,
        )
        input_features = prepared["input_features"].to(
            device=parameter.device,
            dtype=parameter.dtype,
        )
        attention_mask = prepared["attention_mask"].to(device=parameter.device, )
        prompt_ids = prepared["prompt_ids"].to(device=parameter.device, )
        with torch.inference_mode():
            generated = self.model.generate(
                input_features,
                attention_mask,
                prompt_ids=prompt_ids,
                num_lookahead_tokens=prepared["num_lookahead_tokens"],
            )
        decoded = self.nemotron_processor.decode(
            generated.sequences,
            durations=generated.durations if timestamps else None,
            skip_special_tokens=True,
        )
        offsets: list[dict[str, Any]] = []
        if timestamps:
            text, offsets = decoded
        else:
            text = decoded
        detected_language = (
            self.nemotron_processor.detected_language(generated.sequences, )
            if prompt_language == "auto" else None)
        output_language = (detected_language if prompt_language == "auto" else prompt_language)
        segments: tuple[ASRSegment, ...] = ()
        words = self._timestamp_words(
            offsets,
            duration=materialized.duration,
        )
        if timestamps and (text or words):
            segments = (
                ASRSegment(
                    text=text,
                    start=words[0].start if words else 0.0,
                    end=(words[-1].end if words else materialized.duration),
                    language=output_language,
                    words=words,
                ), )
        metadata: dict[str, Any] = {
            "architecture": "nemotron-3.5-rnnt",
            "architecture_family": "rnnt",
            "backend": "voicehub-native",
            "checkpoint_revision": (None if self.artifacts is None else self.artifacts.revision),
            "decoder": "greedy-rnnt",
            "emissions": int(generated.sequences.shape[1] - 1),
            "frame_seconds": self.nemotron_processor.frame_seconds,
            "num_lookahead_tokens": prepared["num_lookahead_tokens"],
            "streaming_latency_ms": self.nemotron_processor.streaming_latency_ms,
        }
        if detected_language is not None:
            metadata["detected_language"] = detected_language
        if timestamps:
            metadata["native_token_timestamps"] = offsets
        return ASROutput(
            text=text,
            segments=segments,
            language=output_language,
            duration=materialized.duration,
            metadata=metadata,
        )

    @staticmethod
    def _audio_batch(
        audio: Any,
        *,
        text_is_batch: bool,
    ) -> tuple[tuple[Any, ...], bool]:
        import torch

        if isinstance(audio, torch.Tensor):
            if audio.ndim == 1:
                return (audio, ), False
            if audio.ndim == 2:
                return (
                    tuple(audio[index] for index in range(audio.shape[0])),
                    True,
                )
            raise ValueError("Nemotron training audio must be rank one or two.")
        if text_is_batch:
            if (isinstance(audio, (str, bytes, bytearray, Path)) or not isinstance(audio, Sequence)):
                raise ValueError("Batched Nemotron transcripts require one audio value "
                                 "per row.")
            return tuple(audio), True
        return (audio, ), False

    def prepare_training_inputs(
        self,
        inputs: dict[str, Any],
        *,
        phase: str,
    ) -> dict[str, Any]:
        """Build log-mels, prompt IDs, labels, and RNN-T prefixes."""
        import torch

        from voicehub.processing.waveform import load_native_audio

        del phase
        required = {
            "input_features",
            "attention_mask",
            "prompt_ids",
            "labels",
            "label_lengths",
            "decoder_input_ids",
        }
        supplied = required & set(inputs)
        if supplied:
            if supplied != required:
                missing = ", ".join(sorted(required - set(inputs)))
                raise ValueError("A cached Nemotron batch is incomplete; missing "
                                 f"{missing}.")
            return dict(inputs)
        if self.model is None:
            self.load_for_training()
        if self.nemotron_processor is None:
            raise RuntimeError("Nemotron training processor is not loaded.")
        audio = inputs.get("audio")
        text = inputs.get(
            "text",
            inputs.get(
                "transcription",
                inputs.get("transcript"),
            ),
        )
        if isinstance(text, str):
            texts = (text, )
            text_is_batch = False
        elif (isinstance(text, Sequence) and not isinstance(text, (str, bytes))):
            texts = tuple(text)
            text_is_batch = True
        else:
            raise ValueError("Nemotron training records require "
                             "`text`/`transcription`.")
        if not texts or any(not isinstance(value, str) or not value.strip() for value in texts):
            raise ValueError("Nemotron training transcripts must be non-empty strings.")
        if audio is None:
            raise ValueError("Nemotron training records require `audio`.")
        audio_values, was_batched = self._audio_batch(
            audio,
            text_is_batch=text_is_batch,
        )
        if len(audio_values) != len(texts):
            raise ValueError("Nemotron requires one transcript per waveform.")
        rates = _batch_values(
            inputs.get(
                "sampling_rate",
                inputs.get("sample_rate"),
            ),
            batch_size=len(audio_values),
            name="sampling_rate",
        )
        lengths = _batch_values(
            inputs.get("audio_lengths"),
            batch_size=len(audio_values),
            name="audio_lengths",
        )
        for length in lengths:
            if length is None:
                continue
            if (isinstance(length, bool) or not isinstance(length, Integral) or length <= 0):
                raise ValueError("`audio_lengths` must contain positive integers.")
        target_rate = self.nemotron_processor.sampling_rate
        waveforms = tuple(
            load_native_audio(
                waveform,
                sampling_rate=rate,
                target_sampling_rate=target_rate,
                num_samples=length,
            ).waveform for waveform, rate, length in zip(
                audio_values,
                rates,
                lengths,
            ))
        waveforms = tuple(
            torch.nn.functional.pad(
                waveform,
                (0, 160 - waveform.numel()),
            ) if waveform.numel() < 160 else waveform for waveform in waveforms)
        default_language = (self.config.training_language or self.config.target_language)
        languages = _batch_values(
            inputs.get("language"),
            batch_size=len(audio_values),
            name="language",
            default=default_language,
        )
        prepared = self.nemotron_processor(
            waveforms,
            text=texts,
            sampling_rate=target_rate,
            language=languages,
        )
        for name, value in inputs.items():
            if name not in _RAW_FIELDS and name not in prepared:
                prepared[name] = value
        if not was_batched:
            return {
                name: (
                    value[0] if
                    (isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == 1) else value)
                for name, value in prepared.items()
            }
        return prepared

    def _save_pretrained(self, save_directory: Path) -> None:
        if self.runtime is None:
            self.load()
        from voicehub.models.asr_nemotron.native.runtime import save_nemotron_asr_runtime

        save_nemotron_asr_runtime(
            self.runtime,
            save_directory,
        )

    def export_native_pretrained(
        self,
        save_directory: str | Path,
    ) -> Path:
        destination = Path(save_directory).expanduser()
        self._save_pretrained(destination)
        return destination


__all__ = ["NemotronForSpeechRecognition"]
