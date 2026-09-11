"""ConversationTTS inference backed by vendored CC BY-NC source."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from numbers import Integral, Real
from pathlib import Path
from typing import TYPE_CHECKING, Any

from voicehub.dependencies import import_optional
from voicehub.hub import resolve_pretrained_file
from voicehub.models._shared import finish_audio_output, resolve_torch_dtype, seeded_inference, validate_local_file
from voicehub.models.conversationtts.configuration import ConversationTTSConfig
from voicehub.models.tts import PreTrainedTTSModel
from voicehub.outputs import TTSOutput

if TYPE_CHECKING:
    from voicehub.models.conversationtts.native.processing import ConversationTTSProtocol

_AUDIO_FRAME_MILLISECONDS = 40
_MODEL_CONTEXT_TOKENS = 2_048
_MINIMUM_TEXT_PROMPT_TOKENS = 5
_MAX_AUDIO_LENGTH_MILLISECONDS = (
    _MODEL_CONTEXT_TOKENS - _MINIMUM_TEXT_PROMPT_TOKENS) * _AUDIO_FRAME_MILLISECONDS


def resume_for_inference(
    checkpoint: str | Path,
    experiment_directory: str | None,
    model,
    device: str,
):
    """Load checkpoints without importing the PyTorch runtime at module
    import."""
    from voicehub.models.conversationtts.runtime import resume_for_inference as resume

    return resume(
        checkpoint,
        experiment_directory,
        model,
        device,
    )


class ConversationTTSForTextToSpeech(PreTrainedTTSModel):
    """Multilingual conversational synthesis with optional speaker context."""

    config_class = ConversationTTSConfig
    default_model_name_or_path = "AudioFoundation/SpeechFoundation"
    passthrough_generation_options = frozenset()

    def __init__(
        self,
        config: ConversationTTSConfig | str | None = None,
        *,
        model_path: str | None = None,
        device: str = "auto",
        lazy_load: bool = True,
        **config_overrides,
    ):
        config = self._coerce_config(
            config,
            model_path=model_path,
            **config_overrides,
        )
        self._generator = None
        self._generator_module = None
        self._training_text_tokenizer = None
        self._training_audio_tokenizer = None
        self._loaded_for_training = False
        super().__init__(config, device=device, lazy_load=lazy_load)

    def _hub_file(
        self,
        repository_id: str,
        filename: str,
        *,
        revision: str | None = None,
    ) -> Path:
        return resolve_pretrained_file(
            repository_id,
            filename,
            cache_dir=self.config.cache_dir,
            revision=revision,
            local_files_only=self.config.local_files_only,
        )

    def _checkpoint_path(self) -> Path:
        source = Path(self.config.name_or_path).expanduser()
        if source.is_file():
            return source.resolve()
        if source.is_dir():
            candidates = (
                source / "model.safetensors",
                source / "native_export" / "model.safetensors",
                source / self.config.checkpoint_filename,
            )
            for checkpoint in candidates:
                if checkpoint.is_file():
                    return checkpoint.resolve()
            searched = ", ".join(str(path) for path in candidates)
            raise FileNotFoundError("ConversationTTS checkpoint was not found. Searched: "
                                    f"{searched}.")
        return self._hub_file(
            self.config.name_or_path,
            self.config.checkpoint_filename,
            revision=self.config.checkpoint_revision,
        )

    def _text_tokenizer_path(self) -> Path:
        if self.config.text_tokenizer_path:
            path = Path(self.config.text_tokenizer_path).expanduser()
        else:
            path = (Path(__file__).parent / "source" / "conversationtts" / "llama3_2")
        if not path.is_dir():
            raise FileNotFoundError(f"ConversationTTS text tokenizer not found: {path}")
        return path.resolve()

    def _audio_tokenizer_path(self) -> Path:
        if self.config.audio_tokenizer_path:
            path = Path(self.config.audio_tokenizer_path).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"ConversationTTS audio tokenizer not found: {path}")
            return path.resolve()
        return self._hub_file(
            self.config.audio_tokenizer_repo_id,
            self.config.audio_tokenizer_filename,
            revision=self.config.audio_tokenizer_revision,
        )

    def _build_raw_model(self):
        """Construct the differentiable source model without serving state."""
        torch = import_optional(
            "torch",
            model_type="conversationtts",
            install_extra=None,
        )
        model_module = import_optional(
            "voicehub.models.conversationtts.native.modeling",
            model_type="conversationtts",
            install_extra=None,
        )
        model = model_module.ConversationTTSModel(
            model_module.ConversationTTSArchitectureConfig(**self.config.model_args))
        dtype = resolve_torch_dtype(
            torch,
            self.config.torch_dtype,
            self.device,
        )
        model.to(device=self.device, dtype=dtype)
        return model

    def _attach_inference_runtime(self) -> None:
        """Attach tokenizers and KV caches to the current trained weights."""
        if self.model is None:
            raise RuntimeError(
                "ConversationTTS cannot build its inference runtime before "
                "the source model is loaded.")
        if self._generator is not None:
            self.model.eval()
            self._loaded_for_training = False
            return

        generator_module = import_optional(
            "voicehub.models.conversationtts.source.conversationtts."
            "inference.generator",
            model_type="conversationtts",
            install_extra=None,
        )
        was_training = bool(getattr(self.model, "training", False))
        self.model.eval()
        try:
            generator = generator_module.Generator(
                self.model,
                text_tokenizer_path=str(self._text_tokenizer_path()),
                audio_tokenizer_path=str(self._audio_tokenizer_path()),
            )
            sample_rate = int(getattr(generator, "sample_rate", 0))
            if sample_rate <= 0:
                raise ValueError("The ConversationTTS generator reported an invalid "
                                 "sample rate.")
        except BaseException:
            # Generator.setup_caches() runs before its tokenizer loads. A
            # tokenizer failure must not strand a cache-mutated training graph.
            self._clear_inference_caches()
            if was_training:
                self.model.train()
            raise
        self._generator = generator
        self._generator_module = generator_module
        self.config.sample_rate = sample_rate
        self._loaded_for_training = False

    @staticmethod
    def _clear_transformer_caches(transformer) -> None:
        """Release TorchTune KV caches while preserving parameter identity."""
        modules = getattr(transformer, "modules", None)
        if not callable(modules):
            return
        for module in tuple(modules()):
            if hasattr(module, "kv_cache"):
                module.kv_cache = None
            if hasattr(module, "cache_enabled"):
                module.cache_enabled = False

        caches_are_setup = getattr(transformer, "caches_are_setup", None)
        if callable(caches_are_setup) and caches_are_setup():
            raise RuntimeError(
                "ConversationTTS could not remove the inference KV cache from "
                "its training graph.")
        caches_are_enabled = getattr(transformer, "caches_are_enabled", None)
        if callable(caches_are_enabled) and caches_are_enabled():
            raise RuntimeError("ConversationTTS could not disable the inference KV cache for "
                               "training.")

    def _clear_inference_caches(self) -> None:
        """Remove serving-only cache modules and masks from the source
        model."""
        if self.model is None:
            return
        for transformer_name in ("backbone", "decoder"):
            transformer = getattr(self.model, transformer_name, None)
            if transformer is not None:
                self._clear_transformer_caches(transformer)
        for buffer_name in (
                "backbone_causal_mask",
                "decoder_causal_mask",
        ):
            buffers = getattr(self.model, "_buffers", {})
            if buffer_name in buffers:
                delattr(self.model, buffer_name)

    def _prepare_for_training(self) -> None:
        """Return to the raw differentiable graph without changing weights."""
        if self._generator is not None:
            self._training_text_tokenizer = getattr(
                self._generator,
                "_text_tokenizer",
                self._training_text_tokenizer,
            )
            self._training_audio_tokenizer = getattr(
                self._generator,
                "_audio_tokenizer",
                self._training_audio_tokenizer,
            )
        self._generator = None
        self._generator_module = None
        self._clear_inference_caches()
        if self.model is not None and hasattr(self.model, "train"):
            self.model.train()
        self._loaded_for_training = True

    def _prepare_for_inference(self) -> None:
        """Build serving objects lazily around the current trained weights."""
        self._attach_inference_runtime()

    def _load_pretrained_model(self) -> None:
        model = self._build_raw_model()
        resume_for_inference(
            self._checkpoint_path(),
            None,
            model,
            self.device,
        )
        self.model = model
        self._generator = None
        self._generator_module = None
        self._loaded_for_training = self.is_training_load
        if self.is_training_load:
            model.train()
            return
        try:
            self._attach_inference_runtime()
        except BaseException:
            self.model = None
            self._loaded_for_training = False
            raise

    def _training_protocol(self) -> ConversationTTSProtocol:
        from voicehub.models.conversationtts.native.processing import ConversationTTSProtocol

        audio_vocab_size = int(self.config.model_args["audio_vocab_size"])
        text_vocab_size = int(self.config.model_args["text_vocab_size"])
        return ConversationTTSProtocol(
            audio_num_codebooks=int(self.config.model_args["audio_num_codebooks"]),
            audio_codebook_size=min(2_048, audio_vocab_size - 1),
            audio_vocab_size=audio_vocab_size,
            text_vocab_size=text_vocab_size,
            text_padding_token_id=min(128_002, text_vocab_size - 1),
            audio_padding_token_id=audio_vocab_size - 1,
        )

    def _get_training_text_tokenizer(self):
        if self._training_text_tokenizer is None:
            module = import_optional(
                "voicehub.models.conversationtts.source.conversationtts."
                "tools.tokenizer.Text2ID.text_tokenizer",
                model_type="conversationtts",
                install_extra=None,
            )
            self._training_text_tokenizer = module.TextTokenizer(self._text_tokenizer_path())
        return self._training_text_tokenizer

    def _get_training_audio_tokenizer(self):
        if self._training_audio_tokenizer is None:
            module = import_optional(
                "voicehub.models.conversationtts.source.conversationtts."
                "tools.tokenizer.MimiCodec.mimi_tokenizer",
                model_type="conversationtts",
                install_extra=None,
            )
            self._training_audio_tokenizer = module.MimiTokenizer(
                self._audio_tokenizer_path(),
                device=self.device,
            )
            self._training_audio_tokenizer.eval()
            for parameter in self._training_audio_tokenizer.parameters():
                parameter.requires_grad_(False)
        return self._training_audio_tokenizer

    @staticmethod
    def _pop_alias(
        values: dict[str, Any],
        names: tuple[str, ...],
        *,
        description: str,
    ) -> Any:
        present = [name for name in names if name in values]
        if len(present) > 1:
            raise ValueError(f"Provide only one {description}; received {present!r}.")
        return values.pop(present[0]) if present else None

    @staticmethod
    def _text_batch(value: Any) -> tuple[str, ...]:
        if isinstance(value, str):
            texts = (value, )
        elif isinstance(value, Sequence) and not isinstance(value, bytes):
            texts = tuple(value)
        else:
            raise TypeError("ConversationTTS `text` must be a string or sequence of "
                            "strings.")
        if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("ConversationTTS training texts must be non-empty strings.")
        return texts

    @staticmethod
    def _split_tensor_batch(
        value: Any,
        *,
        batch_size: int,
        unbatched_dimensions: int,
        name: str,
    ) -> list[Any]:
        torch = import_optional(
            "torch",
            model_type="conversationtts",
            install_extra=None,
        )
        is_sequence = isinstance(value, Sequence)
        is_path_value = isinstance(value, (str, bytes, Path))
        if isinstance(value, torch.Tensor):
            if value.ndim == unbatched_dimensions and batch_size == 1:
                return [value]
            if (value.ndim == unbatched_dimensions + 1 and value.shape[0] == batch_size):
                return list(value.unbind(0))
            raise ValueError(
                f"ConversationTTS `{name}` does not match batch size "
                f"{batch_size}: shape={tuple(value.shape)!r}.")
        if is_sequence and not is_path_value:
            examples = list(value)
            if batch_size == 1:
                try:
                    tensor = torch.as_tensor(value)
                except (TypeError, ValueError):
                    tensor = None
                if (tensor is not None and tensor.ndim == unbatched_dimensions):
                    return [tensor]
            if len(examples) == batch_size:
                return examples
        raise TypeError(
            f"ConversationTTS `{name}` must contain one tensor-like value "
            "per training example.")

    @staticmethod
    def _split_audio_batch(
        value: Any,
        *,
        batch_size: int,
    ) -> list[Any]:
        torch = import_optional(
            "torch",
            model_type="conversationtts",
            install_extra=None,
        )
        if isinstance(value, Mapping):
            waveform_key = next(
                (name for name in (
                    "array",
                    "waveform",
                    "audio",
                    "input_values",
                ) if name in value),
                None,
            )
            if waveform_key is None:
                raise ValueError("Batched audio mappings require a waveform-like field.")
            waveforms = value[waveform_key]
            if batch_size == 1 and not (isinstance(waveforms, torch.Tensor) and waveforms.ndim >= 2 and
                                        waveforms.shape[0] == 1):
                return [value]
            if (not isinstance(waveforms, torch.Tensor) or waveforms.ndim < 2 or
                    waveforms.shape[0] != batch_size):
                raise ValueError(
                    "Batched audio mapping waveforms must have one leading "
                    "row per training example.")
            examples = []
            for index in range(batch_size):
                example = {}
                for name, item in value.items():
                    if (isinstance(item, torch.Tensor) and item.ndim > 0 and item.shape[0] == batch_size):
                        selected = item[index]
                        example[name] = (selected.item() if selected.ndim == 0 else selected)
                    elif (isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and
                          len(item) == batch_size):
                        example[name] = item[index]
                    else:
                        example[name] = item
                examples.append(example)
            return examples
        if isinstance(value, torch.Tensor):
            if batch_size == 1:
                return [value[0] if value.ndim == 2 and value.shape[0] == 1 else value]
            if value.ndim < 2 or value.shape[0] != batch_size:
                raise ValueError(
                    "Batched ConversationTTS audio tensors require one "
                    "leading row per text.")
            return list(value.unbind(0))
        if isinstance(value, (str, Path)):
            if batch_size != 1:
                raise ValueError("A single audio path cannot serve multiple texts.")
            return [value]
        if isinstance(value, Sequence) and not isinstance(value, bytes):
            examples = list(value)
            if batch_size == 1 and (not examples or isinstance(examples[0], (int, float, bool))):
                return [value]
            if len(examples) == batch_size:
                return examples
        raise TypeError("ConversationTTS `audio` must provide one waveform or path per "
                        "training text.")

    @staticmethod
    def _batch_integers(
        value: Any,
        *,
        batch_size: int,
        name: str,
        allow_none: bool = True,
    ) -> tuple[int | None, ...]:
        if value is None and allow_none:
            return (None, ) * batch_size
        is_sequence = isinstance(value, Sequence)
        is_text_value = isinstance(value, (str, bytes))
        if isinstance(value, Integral) and not isinstance(value, bool):
            items = (int(value), ) * batch_size
        elif is_sequence and not is_text_value:
            items = tuple(value)
        else:
            torch = import_optional(
                "torch",
                model_type="conversationtts",
                install_extra=None,
            )
            if isinstance(value, torch.Tensor):
                if value.ndim == 0:
                    items = (value.item(), ) * batch_size
                elif value.ndim == 1:
                    items = tuple(value.detach().cpu().tolist())
                else:
                    items = ()
            else:
                items = ()
        if len(items) != batch_size or any(
                isinstance(item, bool) or not isinstance(item, Integral) or item <= 0 for item in items):
            raise ValueError(f"`{name}` must contain {batch_size} positive integers.")
        return tuple(int(item) for item in items)

    @staticmethod
    def _mapping_audio_length(
        audio: Any,
        *,
        default: int | None,
    ) -> int | None:
        if default is not None or not isinstance(audio, Mapping):
            return default
        value = audio.get("audio_lengths")
        if value is None:
            return None
        item = getattr(value, "item", None)
        if callable(item):
            try:
                value = item()
            except (RuntimeError, ValueError):
                pass
        if (isinstance(value, bool) or not isinstance(value, Integral) or value <= 0):
            raise ValueError("ConversationTTS mapped `audio_lengths` must be a positive "
                             "integer.")
        return int(value)

    def prepare_training_inputs(
        self,
        inputs: Mapping[str, Any],
        *,
        phase: Any,
    ) -> dict[str, Any]:
        """Convert raw text/audio into the published 33-stream layout."""
        del phase
        batch = dict(inputs)
        required = {"tokens", "labels", "tokens_mask"}
        if required <= set(batch):
            return batch

        text_ids = self._pop_alias(
            batch,
            ("text_token_ids", "text_ids"),
            description="text-token field",
        )
        texts = self._pop_alias(
            batch,
            ("text", "texts"),
            description="text field",
        )
        if (text_ids is None) == (texts is None):
            raise ValueError(
                "ConversationTTS training requires exactly one of raw `text` "
                "or precomputed `text_token_ids`.")
        torch = import_optional(
            "torch",
            model_type="conversationtts",
            install_extra=None,
        )
        if texts is not None:
            text_values = self._text_batch(texts)
            speakers = batch.pop("speaker", None)
            if speakers is not None:
                if isinstance(speakers, int) and not isinstance(speakers, bool):
                    speakers = (speakers, ) * len(text_values)
                elif isinstance(speakers, Sequence):
                    speakers = tuple(speakers)
                else:
                    speakers = ()
                if len(speakers) != len(text_values) or any(
                        isinstance(speaker, bool) or not isinstance(speaker, int) or speaker < 0
                        for speaker in speakers):
                    raise ValueError(
                        "`speaker` must contain one non-negative integer per "
                        "ConversationTTS text.")
                text_values = tuple(f"[{speaker}]{text}" for speaker, text in zip(speakers, text_values))
            tokenizer = self._get_training_text_tokenizer()
            token_examples = [
                torch.tensor(tokenizer.tokenize(text), dtype=torch.long) for text in text_values
            ]
        else:
            if isinstance(text_ids, torch.Tensor):
                if text_ids.ndim == 1:
                    token_examples = [text_ids]
                elif text_ids.ndim == 2:
                    token_examples = list(text_ids.unbind(0))
                else:
                    raise ValueError("`text_token_ids` must have shape [time] or "
                                     "[batch, time].")
            elif isinstance(text_ids, Sequence):
                if not text_ids:
                    raise ValueError("`text_token_ids` cannot be empty.")
                if (isinstance(text_ids[0], Integral) and not isinstance(text_ids[0], bool)):
                    token_examples = [torch.as_tensor(text_ids)]
                else:
                    token_examples = [torch.as_tensor(item) for item in text_ids]
            else:
                raise TypeError("`text_token_ids` must be a tensor or integer sequence.")
            text_lengths = self._batch_integers(
                batch.pop("text_token_lengths", None),
                batch_size=len(token_examples),
                name="text_token_lengths",
            )
            token_examples = [
                tokens[:length] if length is not None else tokens
                for tokens, length in zip(token_examples, text_lengths)
            ]

        batch_size = len(token_examples)
        audio_codes = self._pop_alias(
            batch,
            ("audio_codes", "codes"),
            description="audio-code field",
        )
        raw_audio = self._pop_alias(
            batch,
            ("audio", "audio_values"),
            description="raw-audio field",
        )
        if (audio_codes is None) == (raw_audio is None):
            raise ValueError(
                "ConversationTTS training requires exactly one of "
                "`audio_codes` or raw `audio`.")
        if audio_codes is not None:
            code_examples = self._split_tensor_batch(
                audio_codes,
                batch_size=batch_size,
                unbatched_dimensions=2,
                name="audio_codes",
            )
            code_lengths = self._batch_integers(
                batch.pop("audio_code_lengths", None),
                batch_size=batch_size,
                name="audio_code_lengths",
            )
            code_examples = [
                torch.as_tensor(codes)[..., :length] if length is not None else torch.as_tensor(codes)
                for codes, length in zip(code_examples, code_lengths)
            ]
        else:
            audio_examples = self._split_audio_batch(
                raw_audio,
                batch_size=batch_size,
            )
            audio_lengths = self._batch_integers(
                batch.pop("audio_lengths", None),
                batch_size=batch_size,
                name="audio_lengths",
            )
            rates = self._batch_integers(
                self._pop_alias(
                    batch,
                    (
                        "sampling_rate",
                        "sampling_rates",
                        "audio_sampling_rate",
                        "audio_sampling_rates",
                    ),
                    description="sampling-rate field",
                ),
                batch_size=batch_size,
                name="sampling_rate",
            )
            from voicehub.audio import AudioInput, load_audio

            codec = self._get_training_audio_tokenizer()
            code_examples = []
            for audio, length, rate in zip(
                    audio_examples,
                    audio_lengths,
                    rates,
            ):
                length = self._mapping_audio_length(
                    audio,
                    default=length,
                )
                decoded = load_audio(
                    audio,
                    sampling_rate=rate,
                )
                waveform = decoded.waveform
                if length is not None:
                    if length > waveform.shape[-1]:
                        raise ValueError(
                            "ConversationTTS `audio_lengths` cannot exceed "
                            "the decoded waveform length.")
                    waveform = waveform[..., :length]
                materialized = load_audio(
                    AudioInput(
                        waveform=waveform,
                        sampling_rate=decoded.sampling_rate,
                        path=decoded.path,
                    ),
                    target_sampling_rate=self.config.sample_rate,
                )
                waveform = materialized.waveform
                with torch.no_grad():
                    codes = codec.model.encode(
                        waveform.to(
                            device=codec.device,
                            dtype=next(codec.model.parameters()).dtype,
                        ).reshape(1, 1, -1))
                code_examples.append(codes.squeeze(0).detach().cpu())

        protocol = self._training_protocol()
        from voicehub.models.conversationtts.native.processing import (
            build_conversationtts_sequence,
            collate_conversationtts_sequences,
        )

        sequences = [
            build_conversationtts_sequence(
                tokens,
                codes,
                protocol=protocol,
            ) for tokens, codes in zip(token_examples, code_examples)
        ]
        prepared = collate_conversationtts_sequences(
            sequences,
            protocol=protocol,
        )
        prepared.update(batch)
        return prepared

    def _validate_generation_inputs(self, model_inputs: dict[str, Any]) -> None:
        super()._validate_generation_inputs(model_inputs)
        speaker = model_inputs.get("speaker", 0)
        if isinstance(speaker, bool) or not isinstance(speaker, int) or speaker < 0:
            raise ValueError("`speaker` must be a non-negative integer.")

        speaker_audio = model_inputs.get("speaker_audio_path")
        reference_text = model_inputs.get("reference_text")
        if speaker_audio is not None and (not isinstance(speaker_audio,
                                                         (str, Path)) or not str(speaker_audio).strip()):
            raise ValueError("`speaker_audio_path` must be a non-empty path or None.")
        if reference_text is not None and (not isinstance(reference_text, str) or not reference_text.strip()):
            raise ValueError("`reference_text` must be a non-empty string or None.")
        if (speaker_audio is None) != (reference_text is None):
            raise ValueError("`speaker_audio_path` and `reference_text` must be provided together.")
        speaker_path = validate_local_file(
            speaker_audio,
            option_name="speaker_audio_path",
        )
        if speaker_path is not None:
            model_inputs["speaker_audio_path"] = str(speaker_path)

        max_audio_length_ms = model_inputs.get("max_audio_length_ms", 30_000)
        if (isinstance(max_audio_length_ms, bool) or not isinstance(max_audio_length_ms, Real) or
                not isfinite(max_audio_length_ms) or max_audio_length_ms < _AUDIO_FRAME_MILLISECONDS or
                max_audio_length_ms >= _MAX_AUDIO_LENGTH_MILLISECONDS):
            raise ValueError(
                "`max_audio_length_ms` must be finite and in the interval "
                f"[{_AUDIO_FRAME_MILLISECONDS}, "
                f"{_MAX_AUDIO_LENGTH_MILLISECONDS}).")

        temperature = model_inputs.get("temperature", 0.9)
        if (isinstance(temperature, bool) or not isinstance(temperature, Real) or not isfinite(temperature) or
                temperature <= 0):
            raise ValueError("`temperature` must be finite and greater than zero.")

        top_k = model_inputs.get("top_k", 30)
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("`top_k` must be a positive integer.")
        audio_vocab_size = int(self.config.model_args["audio_vocab_size"])
        if "top_k" in model_inputs and top_k > audio_vocab_size:
            raise ValueError("`top_k` cannot exceed the audio vocabulary size "
                             f"({audio_vocab_size}).")

    def _speaker_context(
        self,
        *,
        speaker: int,
        speaker_audio_path: str | None,
        reference_text: str | None,
    ) -> list:
        if speaker_audio_path is None:
            return []
        return [
            self._generator_module.prepare_prompt(
                reference_text,
                speaker_audio_path,
                segment_id=speaker,
            )
        ]

    def _inference_generator(self):
        if self._generator is None or self._generator_module is None:
            raise RuntimeError(
                "ConversationTTS inference runtime is not initialized. "
                "Call load() before requesting generation.")
        return self._generator

    def _generate(
        self,
        text: str,
        *,
        output_file: str | None = None,
        speaker: int = 0,
        speaker_audio_path: str | None = None,
        reference_text: str | None = None,
        max_audio_length_ms: float = 30_000,
        temperature: float = 0.9,
        top_k: int = 30,
        seed: int | None = None,
        **generation_options,
    ) -> TTSOutput:
        generator = self._inference_generator()
        top_k = min(
            top_k,
            int(self.config.model_args["audio_vocab_size"]),
        )
        context = self._speaker_context(
            speaker=speaker,
            speaker_audio_path=speaker_audio_path,
            reference_text=reference_text,
        )
        with seeded_inference(
                seed,
                device=self.device,
                model_type="conversationtts",
        ) as effective_seed:
            audio = generator.generate_v1(
                text=text,
                speaker=speaker,
                max_audio_length_ms=max_audio_length_ms,
                context=context,
                temperature=temperature,
                topk=top_k,
                **generation_options,
            )
        return finish_audio_output(
            audio,
            self.sample_rate,
            output_file=output_file,
            metadata={
                "speaker": speaker,
                "voice_cloned": bool(context),
                "seed": effective_seed,
                "requested_seed": seed,
                "license": "CC BY-NC 4.0",
                "commercial_use": False,
            },
        )

    def _save_pretrained(self, save_directory: Path) -> None:
        """Export trained weights without serializing serving KV caches."""
        from voicehub.models.conversationtts.native.checkpoint import export_conversationtts_checkpoint

        if self.model is None:
            raise RuntimeError("Load ConversationTTS before exporting its native weights.")
        destination = Path(save_directory).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        export_conversationtts_checkpoint(
            self.model,
            destination / "model.safetensors",
        )
        self.config.save_pretrained(destination)


ConversationTTS = ConversationTTSForTextToSpeech
