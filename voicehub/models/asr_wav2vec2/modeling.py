"""Native Wav2Vec2 CTC inference and fine-tuning wrapper."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral
from pathlib import Path
from typing import Any

from voicehub.hub import read_json_file, write_json_file
from voicehub.models.asr_wav2vec2.configuration import Wav2Vec2ASRConfig
from voicehub.models.audio import PreTrainedASRModel
from voicehub.outputs import ASROutput, ASRSegment, ASRWord

_RAW_TRAINING_FIELDS = frozenset({
    "audio",
    "audio_lengths",
    "sample_rate",
    "sampling_rate",
    "text",
    "transcript",
    "transcription",
})


def _architecture_names(values: Mapping[str, Any]) -> tuple[str, ...]:
    architectures = values.get('architectures', ())
    if isinstance(architectures, str):
        architectures = (architectures, )
    if not isinstance(architectures, Sequence):
        raise TypeError("Wav2Vec2 checkpoint `architectures` must be a sequence.")
    return tuple(str(value) for value in architectures)


def _batch_scalar_values(
    value: Any,
    *,
    batch_size: int,
    name: str,
) -> tuple[Any, ...]:
    """Normalize a scalar or one-dimensional batch metadata field."""
    if value is None or isinstance(value, (str, bytes)):
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
        values = tuple(value.tolist())
    elif isinstance(value, Sequence):
        values = tuple(value)
    else:
        return (value, ) * batch_size
    if len(values) != batch_size:
        raise ValueError(f"`{name}` contains {len(values)} values for a batch of "
                         f"{batch_size}.")
    return values


class Wav2Vec2ForSpeechRecognition(PreTrainedASRModel):
    """Run and fine-tune Wav2Vec2 CTC using only VoiceHub and PyTorch."""

    config_class = Wav2Vec2ASRConfig
    default_model_name_or_path = "facebook/wav2vec2-base-960h"
    architecture_family = "ctc"
    runtime_name = "Wav2Vec2"
    metadata_architecture = "wav2vec2-ctc"
    native_checkpoint_format = "native-wav2vec2-ctc-v1"
    native_model_architecture = "Wav2Vec2ForCTC"

    def __init__(
        self,
        config: Wav2Vec2ASRConfig | str | Path | None = None,
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
        self.artifacts: Any | None = None
        self.native_config: Any | None = None
        self.ctc_processor: Any | None = None
        # Compatibility aliases are data attributes, not upstream runtimes.
        self.training_processor: Any | None = None
        self.transformers_processor: Any | None = None
        self._default_runtime_language: str | None = None
        self._default_runtime_language_initialized = False
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

    def _model_dtype(self) -> Any:
        import torch

        dtypes = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        configured = self.config.torch_dtype
        if configured == "auto":
            return (torch.float16 if torch.device(self.device).type in {"cuda", "mps"} else torch.float32)
        dtype = dtypes[configured]
        if torch.device(self.device).type == "cpu" and dtype == torch.float16:
            raise ValueError(
                f"Native {self.runtime_name} does not support float16 "
                "execution on CPU; "
                "use float32 or bfloat16.")
        return dtype

    @staticmethod
    def _validate_architecture(values: Mapping[str, Any]) -> None:
        model_type = str(values.get("model_type", "")).strip().lower()
        if model_type not in {"wav2vec2", "asr_wav2vec2"}:
            raise ValueError(
                "Native Wav2Vec2 requires a Wav2Vec2 checkpoint; received "
                f"model type {model_type or '<missing>'!r}.")
        architectures = _architecture_names(values)
        if architectures and not any(name in {"Wav2Vec2ForCTC", "Wav2Vec2ForSpeechRecognition"}
                                     for name in architectures):
            names = ", ".join(architectures)
            raise ValueError(
                "Native Wav2Vec2 requires a CTC checkpoint architecture; "
                f"received: {names}.")

    @classmethod
    def _validate_processor(cls, processor: Any, config: Any) -> None:
        if processor.sampling_rate != config.sampling_rate:
            raise ValueError(
                f"{cls.runtime_name} processor/model sampling-rate mismatch: "
                "processor "
                f"uses {processor.sampling_rate}, model expects "
                f"{config.sampling_rate}.")
        tokenizer = processor.tokenizer
        if tokenizer.vocabulary_size != config.vocab_size:
            raise ValueError(
                f"{cls.runtime_name} tokenizer/model vocabulary mismatch: "
                "tokenizer has "
                f"{tokenizer.vocabulary_size} IDs, model expects "
                f"{config.vocab_size}.")
        expected_ids = {
            "pad_token_id": tokenizer.pad_token_id,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        for name, tokenizer_id in expected_ids.items():
            model_id = getattr(config, name)
            if tokenizer_id != model_id:
                raise ValueError(
                    f"{cls.runtime_name} tokenizer/model {name} mismatch: "
                    "tokenizer "
                    f"uses {tokenizer_id}, model expects {model_id}.")

    def _load_pretrained_model(self) -> None:
        from voicehub.checkpointing import SafeTensorReader, ShardedSafeTensorReader
        from voicehub.models.asr_wav2vec2.native.artifacts import resolve_wav2vec2_artifacts
        from voicehub.models.asr_wav2vec2.native.checkpoint import HuggingFaceWav2Vec2CheckpointAdapter
        from voicehub.models.asr_wav2vec2.native.configuration import Wav2Vec2Config
        from voicehub.models.asr_wav2vec2.native.modeling import Wav2Vec2ForCTC
        from voicehub.models.asr_wav2vec2.processing import Wav2Vec2Processor

        source = self.config.name_or_path or self.default_model_name_or_path
        artifacts = resolve_wav2vec2_artifacts(
            source,
            checkpoint_filename=self.config.checkpoint_filename,
            vocabulary_filename=self.config.vocabulary_filename,
            cache_dir=self.config.cache_dir,
            revision=self.config.revision,
            token=self._hub_token,
            local_files_only=self.config.local_files_only,
        )
        architecture_values = read_json_file(artifacts.config)
        self._validate_architecture(architecture_values)
        native_config = Wav2Vec2Config.from_dict(architecture_values)
        processor = Wav2Vec2Processor.from_artifacts(
            vocabulary=artifacts.vocabulary,
            tokenizer_config=artifacts.tokenizer_config,
            special_tokens_map=artifacts.special_tokens_map,
            preprocessor_config=artifacts.preprocessor_config,
            target_language=self.config.target_language,
        )
        self._validate_processor(processor, native_config)

        model = Wav2Vec2ForCTC(native_config)
        reader_type = (ShardedSafeTensorReader if artifacts.is_sharded else SafeTensorReader)
        with reader_type(artifacts.checkpoint) as reader:
            HuggingFaceWav2Vec2CheckpointAdapter().load_streaming(
                model,
                reader,
                architecture_values,
                strict=True,
            )
        model.to(
            device=self.device,
            dtype=self._model_dtype(),
        )

        self.artifacts = artifacts
        self.native_config = native_config
        self.ctc_processor = processor
        self.training_processor = processor
        self.transformers_processor = processor
        self.model = model

    def _select_runtime_language(self, language: str | None) -> str | None:
        if self.ctc_processor is None or self.native_config is None:
            raise RuntimeError(f"{self.runtime_name} processor is not loaded.")
        tokenizer = self.ctc_processor.tokenizer
        available = tokenizer.available_languages
        if not self._default_runtime_language_initialized:
            self._default_runtime_language = tokenizer.target_language
            self._default_runtime_language_initialized = True
        if language is None:
            language = self._default_runtime_language
            if language is None:
                return None
        if not available:
            configured = self.config.target_language
            if configured is None or language != configured:
                raise ValueError(
                    f"This {self.runtime_name} checkpoint has a single "
                    "vocabulary and "
                    "does not accept a runtime `language` override.")
            return language
        tokenizer.set_target_language(language)
        self._validate_processor(self.ctc_processor, self.native_config)
        return language

    @classmethod
    def _validate_decoding_request(
        cls,
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
            raise ValueError(f"{cls.runtime_name} CTC does not implement speech "
                             "translation.")
        if chunk_length_s is not None or stride_length_s is not None:
            raise ValueError(
                f"Native {cls.runtime_name} currently transcribes the "
                "complete waveform "
                "in one pass; `chunk_length_s` and `stride_length_s` are not "
                "supported.")
        if batch_size not in (None, 1):
            raise ValueError("One public audio request requires `batch_size=1`.")
        if num_beams not in (None, 1):
            raise ValueError(
                f"{cls.runtime_name} CTC uses greedy frame decoding and "
                "requires `num_beams=1`.")
        if max_new_tokens is not None:
            raise ValueError(
                "`max_new_tokens` is a generative decoding option and is "
                f"not valid for {cls.runtime_name} CTC.")
        if hotwords is not None:
            raise ValueError(
                f"Native {cls.runtime_name} does not expose a language-model "
                "hotword decoder.")
        if return_timestamps == "segment":
            raise ValueError(
                f"{cls.runtime_name} CTC timestamp mode supports word "
                "timestamps, not "
                "segment alignment. Use `True`, 'word', or `False`.")
        if return_timestamps not in (False, True, "word"):
            raise ValueError(f"{cls.runtime_name} CTC timestamp mode accepts `False`, "
                             "`True`, or 'word'.")
        return return_timestamps in (True, "word")

    def _pipeline_call_options(
        self,
        *,
        language: str | None,
        task: str,
        return_timestamps: bool | str,
        chunk_length_s: float | None,
        stride_length_s: float | tuple[float, float] | None,
        batch_size: int | None,
        num_beams: int | None,
        max_new_tokens: int | None,
        hotwords: str | tuple[str, ...] | list[str] | None,
        options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Normalize the former pipeline helper without an upstream runtime."""
        word_timestamps = self._validate_decoding_request(
            task=task,
            return_timestamps=return_timestamps,
            chunk_length_s=chunk_length_s,
            stride_length_s=stride_length_s,
            batch_size=batch_size,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            hotwords=hotwords,
        )
        if language is not None:
            self._select_runtime_language(language)
        result = dict(options or {})
        if word_timestamps:
            result["return_timestamps"] = "word"
        return result

    def _word_segments(
        self,
        *,
        decoded: Any,
        logits: Any,
        duration: float,
        language: str | None,
    ) -> tuple[ASRSegment, ...]:
        import torch

        if self.native_config is None:
            raise RuntimeError(f"{self.runtime_name} configuration is not loaded.")
        frame_seconds = (self.native_config.inputs_to_logits_ratio / self.native_config.sampling_rate)
        probabilities = logits.float().softmax(dim=-1).amax(dim=-1)
        words: list[ASRWord] = []
        for offset in decoded.word_offsets:
            start_frame = min(offset.start_offset, probabilities.shape[0])
            end_frame = min(offset.end_offset, probabilities.shape[0])
            confidence = None
            if end_frame > start_frame:
                confidence = float(probabilities[start_frame:end_frame].mean().item())
            words.append(
                ASRWord(
                    text=offset.word,
                    start=min(duration, offset.start_offset * frame_seconds),
                    end=min(duration, offset.end_offset * frame_seconds),
                    confidence=confidence,
                ))
        if not words:
            return ()
        scores = [word.confidence for word in words if word.confidence is not None]
        confidence = (float(torch.tensor(scores).mean().item()) if scores else None)
        return (
            ASRSegment(
                text=decoded.text,
                start=words[0].start,
                end=words[-1].end,
                confidence=confidence,
                language=language,
                words=tuple(words),
            ), )

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

        if self.model is None or self.native_config is None:
            raise RuntimeError(f"{self.runtime_name} runtime is not loaded.")
        if self.ctc_processor is None:
            raise RuntimeError(f"{self.runtime_name} processor is not loaded.")
        word_timestamps = self._validate_decoding_request(
            task=task,
            return_timestamps=return_timestamps,
            chunk_length_s=chunk_length_s,
            stride_length_s=stride_length_s,
            batch_size=batch_size,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            hotwords=hotwords,
        )
        resolved_language = self._select_runtime_language(language)
        materialized = load_native_audio(
            audio,
            sampling_rate=sampling_rate,
            target_sampling_rate=self.native_config.sampling_rate,
        )
        waveform = materialized.waveform
        minimum = self.native_config.minimum_input_samples
        if waveform.numel() < minimum:
            waveform = torch.nn.functional.pad(
                waveform,
                (0, minimum - waveform.numel()),
            )
        model_inputs = self.ctc_processor.prepare_audio_batch((waveform, ))
        parameter = next(self.model.parameters())
        input_values = model_inputs["input_values"].to(
            device=parameter.device,
            dtype=parameter.dtype,
        )
        attention_mask = model_inputs["attention_mask"].to(device=parameter.device, )
        with torch.inference_mode():
            outputs = self.model(
                input_values,
                attention_mask=attention_mask,
            )
        valid_frames = int(outputs.input_lengths[0].item())
        logits = outputs.logits[0, :valid_frames]
        token_ids = logits.argmax(dim=-1).tolist()
        decoded = self.ctc_processor.tokenizer.decode_ctc(
            token_ids,
            skip_special_tokens=False,
            output_word_offsets=word_timestamps,
        )
        segments = (
            self._word_segments(
                decoded=decoded,
                logits=logits,
                duration=materialized.duration,
                language=resolved_language,
            ) if word_timestamps else ())
        return ASROutput(
            text=decoded.text,
            segments=segments,
            language=resolved_language,
            duration=materialized.duration,
            metadata={
                "architecture": self.metadata_architecture,
                "architecture_family": "ctc",
                "backend": "voicehub-native",
                "checkpoint_revision": (None if self.artifacts is None else self.artifacts.revision),
                "logit_frames": valid_frames,
            },
        )

    @classmethod
    def _raw_audio_batch(
        cls,
        audio: Any,
        texts: tuple[str, ...],
        *,
        text_is_batch: bool,
    ) -> tuple[tuple[Any, ...], bool]:
        import torch

        if isinstance(audio, torch.Tensor):
            if audio.ndim == 1:
                return (audio, ), False
            if audio.ndim == 2:
                return tuple(audio[index] for index in range(audio.shape[0])), True
            raise ValueError(f"{cls.runtime_name} training audio must be rank one or rank "
                             "two.")
        if text_is_batch:
            if isinstance(audio, (str, bytes)) or not isinstance(audio, Sequence):
                raise ValueError("Batched transcripts require a sequence of waveforms.")
            return tuple(audio), True
        return (audio, ), False

    def prepare_training_inputs(
        self,
        inputs: dict[str, Any],
        *,
        phase: str,
    ) -> dict[str, Any]:
        """Create native waveform tensors and CTC labels from raw records."""
        import torch

        from voicehub.processing.waveform import load_native_audio

        del phase
        if "input_values" in inputs and "labels" in inputs:
            return {name: value for name, value in inputs.items() if name != "input_lengths"}
        if self.model is None:
            self.load_for_training()
        if self.native_config is None or self.ctc_processor is None:
            raise RuntimeError(f"{self.runtime_name} training processor is not loaded.")
        self._select_runtime_language(None)
        audio = inputs.get("audio")
        text = inputs.get(
            "text",
            inputs.get("transcription", inputs.get("transcript")),
        )
        if isinstance(text, str):
            texts = (text, )
            text_is_batch = False
        elif isinstance(text, Sequence) and not isinstance(text, (bytes, str)):
            texts = tuple(text)
            text_is_batch = True
        else:
            raise ValueError(
                f"{self.runtime_name} training records require non-empty "
                "`text`/`transcription`.")
        if not texts or any(not isinstance(value, str) or not value.strip() for value in texts):
            raise ValueError(
                f"{self.runtime_name} training transcriptions must contain "
                "non-empty strings.")
        if audio is None:
            raise ValueError(f"{self.runtime_name} training records require `audio`.")
        audio_values, was_batched = self._raw_audio_batch(
            audio,
            texts,
            text_is_batch=text_is_batch,
        )
        if len(audio_values) != len(texts):
            raise ValueError(f"{self.runtime_name} training requires one transcript per "
                             "waveform.")

        lengths = _batch_scalar_values(
            inputs.get("audio_lengths"),
            batch_size=len(audio_values),
            name="audio_lengths",
        )
        if any(length is not None and
               (isinstance(length, bool) or not isinstance(length, Integral) or length <= 0)
               for length in lengths):
            raise ValueError("`audio_lengths` must contain positive integers.")
        rates = _batch_scalar_values(
            inputs.get("sampling_rate", inputs.get("sample_rate")),
            batch_size=len(audio_values),
            name="sampling_rate",
        )
        waveforms = tuple(
            load_native_audio(
                value,
                sampling_rate=rate,
                target_sampling_rate=self.native_config.sampling_rate,
                num_samples=(None if length is None else int(length)),
            ).waveform for value, rate, length in zip(
                audio_values,
                rates,
                lengths,
            ))
        minimum = self.native_config.minimum_input_samples
        padded_waveforms = tuple((
            torch.nn.functional.pad(
                waveform,
                (0, minimum - waveform.numel()),
            ) if waveform.numel() < minimum else waveform) for waveform in waveforms)
        prepared = self.ctc_processor.prepare_audio_batch(padded_waveforms)
        prepared["labels"] = self.ctc_processor.encode_labels(
            texts,
            pad=True,
        )
        for name, value in inputs.items():
            if name not in _RAW_TRAINING_FIELDS and name not in prepared:
                prepared[name] = value
        if not was_batched:
            return {
                name: (
                    value[0]
                    if isinstance(value, torch.Tensor) and value.ndim > 1 and value.shape[0] == 1 else value)
                for name, value in prepared.items()
            }
        return prepared

    def _save_pretrained(self, save_directory: Path) -> None:
        from voicehub.checkpointing import save_safetensors

        if (self.model is None or self.native_config is None or self.ctc_processor is None):
            self.load()
        save_directory.mkdir(parents=True, exist_ok=True)
        save_safetensors(
            self.model.state_dict(),
            save_directory / "model.safetensors",
            metadata={"format": self.native_checkpoint_format},
        )
        config_values = self.native_config.to_dict()
        config_values.update({
            'architectures': [self.native_model_architecture],
            "model_type": self.config_class.model_type,
            "voicehub_checkpoint_format": self.native_checkpoint_format,
            "voicehub_provider": self.config_class.model_type,
        })
        write_json_file(save_directory / "config.json", config_values)
        self._select_runtime_language(None)
        self.ctc_processor.save_pretrained(save_directory)

    def export_native_pretrained(
        self,
        save_directory: str | Path,
    ) -> Path:
        """Write a flat, self-contained native CTC artifact."""
        destination = Path(save_directory).expanduser()
        self._save_pretrained(destination)
        return destination


__all__ = ["Wav2Vec2ForSpeechRecognition"]
