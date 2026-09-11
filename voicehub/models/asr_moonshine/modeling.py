"""Native Moonshine inference and fine-tuning wrapper."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral
from pathlib import Path
from typing import Any

from voicehub.hub import read_json_file, write_json_file
from voicehub.models.asr_moonshine.configuration import MoonshineASRConfig
from voicehub.models.audio import PreTrainedASRModel
from voicehub.outputs import ASROutput

_RAW_TRAINING_FIELDS = frozenset({
    "audio",
    "audio_lengths",
    "sample_rate",
    "sampling_rate",
    "text",
    "transcript",
    "transcription",
})
_ENGLISH_ALIASES = frozenset({"en", "eng", "english"})


def _architecture_names(values: Mapping[str, Any]) -> tuple[str, ...]:
    architectures = values.get('architectures', ())
    if isinstance(architectures, str):
        architectures = (architectures, )
    if not isinstance(architectures, Sequence):
        raise TypeError("Moonshine checkpoint `architectures` must be a sequence.")
    return tuple(str(value) for value in architectures)


def _batch_scalar_values(
    value: Any,
    *,
    batch_size: int,
    name: str,
) -> tuple[Any, ...]:
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


class MoonshineForSpeechRecognition(PreTrainedASRModel):
    """Run and fine-tune official Moonshine checkpoints with VoiceHub code."""

    config_class = MoonshineASRConfig
    default_model_name_or_path = "UsefulSensors/moonshine-tiny"
    architecture_family = "speech-seq2seq"
    native_checkpoint_format = "native-moonshine-seq2seq-v1"

    def __init__(
        self,
        config: MoonshineASRConfig | str | Path | None = None,
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
        self.moonshine_processor: Any | None = None
        # Compatibility aliases expose VoiceHub objects, never upstream ones.
        self.training_processor: Any | None = None
        self.transformers_processor: Any | None = None
        self._generation_values: dict[str, Any] = {}
        config = self._coerce_config(
            config,
            model_path=model_path,
            **kwargs,
        )
        super().__init__(config, device=device, lazy_load=lazy_load)

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
                "Native Moonshine does not support float16 execution on CPU; "
                "use float32 or bfloat16.")
        return dtype

    @staticmethod
    def _validate_architecture(values: Mapping[str, Any]) -> None:
        model_type = str(values.get("model_type", "")).strip().lower()
        if model_type not in {"asr_moonshine", "moonshine"}:
            raise ValueError(
                "Native Moonshine requires a Moonshine checkpoint; received "
                f"model type {model_type or '<missing>'!r}.")
        architectures = _architecture_names(values)
        supported = {
            "MoonshineForConditionalGeneration",
            "MoonshineForSpeechRecognition",
        }
        if architectures and not any(architecture in supported for architecture in architectures):
            raise ValueError(
                "Native Moonshine requires a conditional-generation "
                "checkpoint architecture; received: "
                f"{', '.join(architectures)}.")
        if "auto_map" in values:
            # Declarative metadata may be present in legacy exports, but the
            # native runtime must never need or execute it.
            raise ValueError(
                "Moonshine checkpoints containing `auto_map` remote-code "
                "metadata are unsupported. Convert weights into the standard "
                "published Safetensors layout first.")

    @staticmethod
    def _validate_generation(
        values: Mapping[str, Any],
        config: Any,
    ) -> dict[str, Any]:
        expected_ids = {
            "bos_token_id": config.bos_token_id,
            "decoder_start_token_id": config.decoder_start_token_id,
            "eos_token_id": config.eos_token_id,
            "pad_token_id": config.pad_token_id,
        }
        for name, expected in expected_ids.items():
            actual = values.get(name, expected)
            if actual != expected:
                raise ValueError(
                    f"Moonshine generation `{name}` mismatch: checkpoint "
                    f"declares {actual}, model declares {expected}.")
        maximum = values.get(
            "max_length",
            config.max_position_embeddings,
        )
        if (isinstance(maximum, bool) or not isinstance(maximum, int) or
                not 2 <= maximum <= config.max_position_embeddings):
            raise ValueError(
                "Moonshine generation `max_length` must be between 2 and "
                f"{config.max_position_embeddings}.")
        return {
            "bos_token_id": config.bos_token_id,
            "decoder_start_token_id": config.decoder_start_token_id,
            "eos_token_id": config.eos_token_id,
            "pad_token_id": config.pad_token_id,
            "max_length": maximum,
        }

    def _load_pretrained_model(self) -> None:
        from voicehub.checkpointing import SafeTensorReader, ShardedSafeTensorReader
        from voicehub.models.asr_moonshine.native.artifacts import resolve_moonshine_artifacts
        from voicehub.models.asr_moonshine.native.checkpoint import HuggingFaceMoonshineCheckpointAdapter
        from voicehub.models.asr_moonshine.native.configuration import MoonshineConfig
        from voicehub.models.asr_moonshine.native.modeling import MoonshineForConditionalGeneration
        from voicehub.models.asr_moonshine.native.processing import MoonshineProcessor

        source = self.config.name_or_path or self.default_model_name_or_path
        artifacts = resolve_moonshine_artifacts(
            source,
            checkpoint_filename=self.config.checkpoint_filename,
            tokenizer_filename=self.config.tokenizer_filename,
            cache_dir=self.config.cache_dir,
            revision=self.config.revision,
            token=self._hub_token,
            local_files_only=self.config.local_files_only,
        )
        configuration_values = read_json_file(artifacts.config)
        self._validate_architecture(configuration_values)
        native_config = MoonshineConfig.from_dict(configuration_values)
        generation_values = self._validate_generation(
            read_json_file(artifacts.generation_config),
            native_config,
        )
        processor = MoonshineProcessor.from_artifacts(
            tokenizer_path=artifacts.tokenizer,
            preprocessor_config_path=artifacts.preprocessor_config,
            config=native_config,
        )
        if processor.tokenizer.token_id_space_size != native_config.vocab_size:
            raise ValueError(
                "Moonshine tokenizer/model vocabulary mismatch: tokenizer "
                f"uses {processor.tokenizer.token_id_space_size} IDs, model "
                f"expects {native_config.vocab_size}.")

        model = MoonshineForConditionalGeneration(native_config)
        reader_type = (ShardedSafeTensorReader if artifacts.is_sharded else SafeTensorReader)
        with reader_type(artifacts.checkpoint) as reader:
            HuggingFaceMoonshineCheckpointAdapter().load_streaming(
                model,
                reader,
                configuration_values,
                strict=True,
            )
        model.to(device=self.device, dtype=self._model_dtype())
        self.artifacts = artifacts
        self.native_config = native_config
        self._generation_values = generation_values
        self.moonshine_processor = processor
        self.training_processor = processor
        self.transformers_processor = processor
        self.model = model

    @staticmethod
    def _validate_decoding_request(
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
    ) -> None:
        if language is not None:
            if (not isinstance(language, str) or language.strip().lower() not in _ENGLISH_ALIASES):
                raise ValueError(
                    "Published Moonshine tiny/base checkpoints support English "
                    "only; `language` must be 'en', 'eng', 'english', or None.")
        if task != "transcribe":
            raise ValueError(
                "Published Moonshine ASR checkpoints do not expose a speech "
                "translation prompt; `task` must be 'transcribe'.")
        if return_timestamps is not False:
            raise ValueError(
                "Native Moonshine does not expose timestamp token alignment; "
                "`return_timestamps` must be False.")
        if chunk_length_s is not None or stride_length_s is not None:
            raise ValueError(
                "Native Moonshine currently decodes one complete waveform; "
                "chunk and stride options are unsupported.")
        if batch_size not in (None, 1):
            raise ValueError("One public Moonshine request requires `batch_size=1`.")
        if num_beams not in (None, 1):
            raise ValueError(
                "Native Moonshine currently supports deterministic greedy "
                "decoding and requires `num_beams=1`.")
        if max_new_tokens is not None and max_new_tokens <= 0:
            raise ValueError("`max_new_tokens` must be greater than zero.")
        if hotwords is not None:
            raise ValueError(
                "Native Moonshine does not implement an external hotword "
                "language-model decoder.")

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
        """Retain the former preset helper without a delegated pipeline."""
        self._validate_decoding_request(
            language=language,
            task=task,
            return_timestamps=return_timestamps,
            chunk_length_s=chunk_length_s,
            stride_length_s=stride_length_s,
            batch_size=batch_size,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            hotwords=hotwords,
        )
        if options:
            names = ", ".join(sorted(str(name) for name in options))
            raise ValueError(f"Native Moonshine received unsupported option(s): {names}.")
        generate_kwargs = {"num_beams": 1}
        if max_new_tokens is not None:
            generate_kwargs["max_new_tokens"] = max_new_tokens
        return {"generate_kwargs": generate_kwargs}

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

        self._validate_decoding_request(
            language=language,
            task=task,
            return_timestamps=return_timestamps,
            chunk_length_s=chunk_length_s,
            stride_length_s=stride_length_s,
            batch_size=batch_size,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            hotwords=hotwords,
        )
        if (self.model is None or self.native_config is None or self.moonshine_processor is None):
            raise RuntimeError("Moonshine runtime is not loaded.")
        materialized = load_native_audio(
            audio,
            sampling_rate=sampling_rate,
            target_sampling_rate=self.native_config.sampling_rate,
        )
        waveform = materialized.waveform
        original_samples = waveform.numel()
        if waveform.numel() < self.native_config.minimum_input_samples:
            waveform = torch.nn.functional.pad(
                waveform,
                (0, self.native_config.minimum_input_samples - waveform.numel()),
            )
        batch = self.moonshine_processor.prepare_audio_batch((waveform, ))
        parameter = next(self.model.parameters())
        input_values = batch["input_values"].to(
            device=parameter.device,
            dtype=parameter.dtype,
        )
        attention_mask = batch["attention_mask"].to(device=parameter.device)
        generation_options: dict[str, Any] = {"num_beams": 1}
        if max_new_tokens is not None:
            generation_options["max_new_tokens"] = max_new_tokens
        else:
            model_card_limit = max(
                2,
                int(original_samples * 6.5 / self.native_config.sampling_rate),
            )
            generation_options["max_length"] = min(
                self._generation_values["max_length"],
                model_card_limit,
            )
        with torch.inference_mode():
            sequences = self.model.generate(
                input_values,
                attention_mask,
                **generation_options,
            )
        text = self.moonshine_processor.decode(
            sequences[0],
            skip_special_tokens=True,
        )
        token_ids = sequences[0].detach().cpu().tolist()
        decoded_tokens = sum(
            token_id not in {
                self.native_config.bos_token_id,
                self.native_config.eos_token_id,
                self.native_config.pad_token_id,
            } for token_id in token_ids)
        return ASROutput(
            text=text,
            segments=(),
            language="en",
            duration=materialized.duration,
            metadata={
                "architecture": "moonshine",
                "architecture_family": self.architecture_family,
                "backend": "voicehub-native",
                "checkpoint_revision": (None if self.artifacts is None else self.artifacts.revision),
                "decoding": "greedy",
                "generated_tokens": decoded_tokens,
            },
        )

    @staticmethod
    def _raw_audio_batch(
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
            raise ValueError("Moonshine training audio must be rank one or rank two.")
        if text_is_batch:
            if isinstance(audio, (str, bytes)) or not isinstance(audio, Sequence):
                raise ValueError("Batched transcripts require a sequence of waveforms.")
            return tuple(audio), True
        return (audio, ), False

    def _process_training_audio(
        self,
        audio: Any,
        *,
        sampling_rate: int,
    ) -> Mapping[str, Any]:
        """Backward-compatible named-audio processor boundary."""
        processor = self.transformers_processor
        if processor is None:
            raise RuntimeError("Training input preparation requires load_for_training().")
        encoded = processor(
            audio=audio,
            sampling_rate=sampling_rate,
            padding=True,
            return_tensors="pt",
        )
        if not isinstance(encoded, Mapping):
            raise TypeError("The Moonshine processor did not return a mapping.")
        return encoded

    def prepare_training_inputs(
        self,
        inputs: dict[str, Any],
        *,
        phase: str,
    ) -> dict[str, Any]:
        """Create native waveform tensors and teacher-forced text labels."""
        import torch

        from voicehub.processing.waveform import load_native_audio

        del phase
        if "input_values" in inputs and "labels" in inputs:
            return dict(inputs)
        if self.model is None:
            self.load_for_training()
        if self.native_config is None or self.moonshine_processor is None:
            raise RuntimeError("Moonshine training processor is not loaded.")
        audio = inputs.get("audio")
        text = inputs.get(
            "text",
            inputs.get("transcription", inputs.get("transcript")),
        )
        if isinstance(text, str):
            texts = (text, )
            text_is_batch = False
        elif isinstance(text, Sequence) and not isinstance(text, (str, bytes)):
            texts = tuple(text)
            text_is_batch = True
        else:
            raise ValueError("Moonshine training records require non-empty "
                             "`text`/`transcription`.")
        if not texts or any(not isinstance(value, str) or not value.strip() for value in texts):
            raise ValueError("Moonshine training transcriptions must contain non-empty "
                             "strings.")
        if audio is None:
            raise ValueError("Moonshine training records require `audio`.")
        audio_values, was_batched = self._raw_audio_batch(
            audio,
            texts,
            text_is_batch=text_is_batch,
        )
        if len(audio_values) != len(texts):
            raise ValueError("Moonshine training requires one transcript per waveform.")

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
        waveforms = tuple((
            torch.nn.functional.pad(
                waveform,
                (0, minimum - waveform.numel()),
            ) if waveform.numel() < minimum else waveform) for waveform in waveforms)
        prepared = self.moonshine_processor.prepare_audio_batch(waveforms)
        prepared.update(self.moonshine_processor.encode_labels(texts))
        for name, value in inputs.items():
            if name not in _RAW_TRAINING_FIELDS and name not in prepared:
                prepared[name] = value
        if not was_batched:
            return {
                name: (
                    value[0] if
                    (isinstance(value, torch.Tensor) and value.ndim > 1 and value.shape[0] == 1) else value)
                for name, value in prepared.items()
            }
        return prepared

    def _save_pretrained(self, save_directory: Path) -> None:
        from voicehub.checkpointing import save_safetensors

        if (self.model is None or self.native_config is None or self.moonshine_processor is None):
            self.load()
        save_directory.mkdir(parents=True, exist_ok=True)
        save_safetensors(
            self.model.state_dict(),
            save_directory / "model.safetensors",
            metadata={"format": self.native_checkpoint_format},
        )
        values = self.native_config.to_dict()
        values.update({
            'architectures': ["MoonshineForConditionalGeneration"],
            "model_type": self.config_class.model_type,
            "voicehub_checkpoint_format": self.native_checkpoint_format,
            "voicehub_provider": self.config_class.model_type,
        })
        write_json_file(save_directory / "config.json", values)
        write_json_file(
            save_directory / "generation_config.json",
            {
                "_from_model_config": True,
                **self._generation_values,
            },
        )
        self.moonshine_processor.save_pretrained(save_directory)

    def export_native_pretrained(
        self,
        save_directory: str | Path,
    ) -> Path:
        destination = Path(save_directory).expanduser()
        self._save_pretrained(destination)
        return destination


__all__ = ["MoonshineForSpeechRecognition"]
