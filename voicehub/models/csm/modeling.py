"""Native Sesame CSM inference and lifecycle integration."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicehub.models._shared import finish_audio_output, seeded_inference
from voicehub.models.csm.configuration import CSMConfig
from voicehub.models.csm.native.runtime import CSMCodec, CSMRuntime, load_csm_runtime
from voicehub.models.tts import PreTrainedTTSModel
from voicehub.outputs import TTSOutput


class CSMForTextToSpeech(PreTrainedTTSModel):
    """Conversational CSM with native PyTorch inference and fine-tuning."""

    config_class = CSMConfig
    default_model_name_or_path = "sesame/csm-1b"
    _AUDIO_FRAME_MILLISECONDS = 80
    _AUDIO_VOCAB_SIZE = 2_051

    def __init__(
        self,
        config: CSMConfig | str | None = None,
        *,
        model_path: str | None = None,
        device: str = "auto",
        lazy_load: bool = True,
        token: str | bool | None = None,
        **config_overrides,
    ) -> None:
        # Runtime objects are deliberately removed before config coercion:
        # codecs, postprocessors, and credentials must never be serialized.
        codec: CSMCodec | None = config_overrides.pop("codec", None)
        audio_postprocessor: Any | None = config_overrides.pop(
            "audio_postprocessor",
            None,
        )
        config = self._coerce_config(
            config,
            model_path=model_path,
            **config_overrides,
        )
        self._runtime: CSMRuntime | None = None
        self._training_backend = None
        self._hub_token = token
        self._codec = codec
        self._audio_postprocessor = audio_postprocessor
        super().__init__(config, device=device, lazy_load=lazy_load)

    def _load_pretrained_model(self) -> None:
        if self.is_training_load:
            from voicehub.models.csm.training import load_csm_training_backend

            loader_options = {}
            if self._codec is not None:
                loader_options["codec"] = self._codec
            if self.config.codec_path is not None:
                loader_options["codec_path"] = self.config.codec_path
            if not self.config.load_codec:
                loader_options["include_codec"] = False
            if self.config.revision is not None:
                loader_options["revision"] = self.config.revision
            if self.config.cache_dir is not None:
                loader_options["cache_dir"] = self.config.cache_dir
            if self._hub_token is not None:
                loader_options["token"] = self._hub_token
            if self.config.local_files_only:
                loader_options["local_files_only"] = True
            if self.config.verify_integrity:
                loader_options["verify_integrity"] = True
            if self.config.verify_checkpoint_integrity:
                loader_options["verify_checkpoint_integrity"] = True
            backend = load_csm_training_backend(
                self.config.name_or_path,
                device=self.device,
                torch_dtype=self.config.torch_dtype,
                **loader_options,
            )
            self._training_backend = backend
            self._runtime = getattr(backend, "runtime", None)
            self.model = backend.model
            self.config.sample_rate = int(backend.sample_rate)
            return
        runtime = load_csm_runtime(
            self.config.name_or_path,
            device=self.device,
            dtype=self.config.torch_dtype,
            codec=self._codec,
            codec_path=self.config.codec_path,
            include_codec=self.config.load_codec,
            revision=self.config.revision,
            cache_dir=self.config.cache_dir,
            token=self._hub_token,
            local_files_only=self.config.local_files_only,
            verify_integrity=self.config.verify_integrity,
            verify_checkpoint_integrity=(self.config.verify_checkpoint_integrity),
            audio_postprocessor=self._audio_postprocessor,
        )
        self._runtime = runtime
        self.model = runtime.model
        self.config.sample_rate = runtime.sample_rate

    def _prepare_for_inference(self) -> None:
        if self._runtime is None:
            candidates = (
                self.model,
                getattr(self.model, "_model", None),
                getattr(self.model, "_audio_tokenizer", None),
                getattr(self.model, "_watermarker", None),
            )
            for candidate in candidates:
                evaluate = getattr(candidate, "eval", None)
                if callable(evaluate):
                    evaluate()
            model_config = getattr(self.model, "config", None)
            if model_config is not None and hasattr(model_config, "use_cache"):
                model_config.use_cache = True
            return
        self.model.eval()
        if self._runtime.codec is not None:
            evaluate = getattr(self._runtime.codec, "eval", None)
            if callable(evaluate):
                evaluate()

    def _prepare_for_training(self) -> None:
        if (self._training_backend is not None and self.model is self._training_backend.model):
            freeze = getattr(self._training_backend, "freeze_codec", None)
            if callable(freeze):
                freeze()
            return
        if self._runtime is None:
            self.model = None
            self._loading_for_training = True
            try:
                self.load()
            finally:
                self._loading_for_training = False
        if (self._training_backend is not None and self.model is self._training_backend.model):
            freeze = getattr(self._training_backend, "freeze_codec", None)
            if callable(freeze):
                freeze()
            return
        if self._runtime is None:
            raise RuntimeError("CSM native training runtime was not loaded.")
        if self._training_backend is None:
            from voicehub.models.csm.training import CSMTrainingBackend

            self._training_backend = CSMTrainingBackend.from_runtime(self._runtime, )
        self.model = self._runtime.model
        self.model.train()
        self._training_backend.freeze_codec()

    @property
    def training_backend(self):
        return self._training_backend

    @property
    def _uses_transformers_backend(self) -> bool:
        """Whether an older injected artifact backend is currently attached."""
        return (
            self._training_backend is not None and
            getattr(self._training_backend, "runtime", None) is None and
            self.model is getattr(self._training_backend, "model", self.model))

    @staticmethod
    def _move_processor_output_to_device(
        inputs: Any,
        device: str,
    ) -> Mapping[str, Any]:
        move = getattr(inputs, "to", None)
        if callable(move):
            moved = move(device)
            if not isinstance(moved, Mapping):
                raise TypeError("CSM processor batch `.to()` must return a mapping.")
            return moved
        if not isinstance(inputs, Mapping):
            raise TypeError("CSM processor output must be a mapping.")
        return {
            name: (value.to(device) if callable(getattr(value, "to", None)) else value)
            for name, value in inputs.items()
        }

    def _generate_transformers(
        self,
        text: str,
        *,
        speaker: int,
        speaker_audio_path: str | None,
        reference_text: str | None,
        max_audio_length_ms: float,
        temperature: float,
        top_k: int,
        generation_options: Mapping[str, Any],
    ) -> tuple[Any, int]:
        """Read an older injected training artifact without importing its
        framework."""
        processor = self._training_backend.processor
        conversation = []
        if speaker_audio_path is not None:
            from voicehub.audio import load_audio

            reference = load_audio(
                speaker_audio_path,
                target_sampling_rate=self.sample_rate,
            ).waveform
            conversation.append({
                "role":
                str(speaker),
                "content": [
                    {
                        "type": "text",
                        "text": reference_text,
                    },
                    {
                        "type": "audio",
                        "path": reference,
                    },
                ],
            })
        conversation.append({
            "role": str(speaker),
            "content": [{
                "type": "text",
                "text": text,
            }],
        })
        inputs = processor.apply_chat_template(
            conversation,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = self._move_processor_output_to_device(inputs, self.device)
        options = dict(generation_options)
        options.setdefault(
            "max_new_tokens",
            max(1, int(max_audio_length_ms / 80)),
        )
        if temperature == 0:
            options["do_sample"] = False
            options["depth_decoder_do_sample"] = False
            for name in (
                    "temperature",
                    "top_k",
                    "depth_decoder_temperature",
                    "depth_decoder_top_k",
            ):
                options.pop(name, None)
        else:
            options.setdefault("do_sample", True)
            options.setdefault("temperature", temperature)
            options.setdefault("top_k", top_k)
            options.setdefault("depth_decoder_do_sample", True)
            options.setdefault("depth_decoder_temperature", temperature)
            options.setdefault("depth_decoder_top_k", top_k)
        options.setdefault("use_cache", True)
        generated = self.model.generate(
            **inputs,
            output_audio=True,
            **options,
        )
        audio = getattr(generated, "audio", generated)
        if isinstance(audio, (list, tuple)):
            if not audio:
                raise RuntimeError("Legacy CSM artifact returned no audio.")
            audio = audio[0]
        return audio, len(conversation) - 1

    def prepare_training_inputs(
        self,
        inputs: dict,
        *,
        phase: str,
    ) -> dict:
        del phase
        if self._training_backend is None:
            raise RuntimeError("CSM training inputs require `load_for_training()`.")
        return self._training_backend.prepare_inputs(inputs)

    def _save_pretrained(self, save_directory: Path) -> None:
        """Write a flat, dependency-free CSM runtime artifact."""
        self.load()
        if self._runtime is None:
            raise RuntimeError("Only VoiceHub-native CSM runtimes can be exported.")
        self._runtime.save_pretrained(
            save_directory,
            include_codec=self._runtime.codec is not None,
        )

    def _validate_generation_inputs(self, model_inputs: dict) -> None:
        speaker_audio_path = model_inputs.get("speaker_audio_path")
        reference_text = model_inputs.get("reference_text")
        if speaker_audio_path is not None:
            if (not isinstance(speaker_audio_path, (str, Path)) or not str(speaker_audio_path).strip()):
                raise ValueError("`speaker_audio_path` must be a non-empty local path or "
                                 "None.")
        if reference_text is not None and not isinstance(reference_text, str):
            raise TypeError("`reference_text` must be a string or None.")
        if ((speaker_audio_path is not None) != bool(isinstance(reference_text, str) and
                                                     reference_text.strip())):
            raise ValueError(
                "CSM speaker context requires `speaker_audio_path` and a "
                "non-empty `reference_text` together.")
        if speaker_audio_path is not None:
            reference_path = Path(speaker_audio_path).expanduser()
            if not reference_path.is_file():
                raise FileNotFoundError(f"CSM reference audio was not found: {reference_path}.")
        speaker = model_inputs.get("speaker", 0)
        if (isinstance(speaker, bool) or not isinstance(speaker, int) or speaker < 0):
            raise ValueError("`speaker` must be a non-negative integer.")
        max_audio_length_ms = model_inputs.get(
            "max_audio_length_ms",
            90_000,
        )
        if (isinstance(max_audio_length_ms, bool) or not isinstance(max_audio_length_ms, (int, float)) or
                not math.isfinite(max_audio_length_ms) or max_audio_length_ms <= 0):
            raise ValueError("`max_audio_length_ms` must be finite and greater than zero.")
        if max_audio_length_ms < self._AUDIO_FRAME_MILLISECONDS:
            raise ValueError(
                "`max_audio_length_ms` must be finite and at least "
                f"{self._AUDIO_FRAME_MILLISECONDS}.")
        temperature = model_inputs.get("temperature", 0.9)
        if (isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or
                not math.isfinite(temperature) or temperature < 0):
            raise ValueError("`temperature` must be a finite non-negative number.")
        top_k = model_inputs.get("top_k", 50)
        minimum_top_k = 0 if temperature == 0 else 1
        if (isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < minimum_top_k):
            qualifier = "non-negative" if temperature == 0 else "positive"
            raise ValueError(f"`top_k` must be a {qualifier} integer for this sampling "
                             "mode.")
        if top_k > self._AUDIO_VOCAB_SIZE:
            raise ValueError(
                "`top_k` cannot exceed the CSM audio vocabulary size "
                f"({self._AUDIO_VOCAB_SIZE}).")
        if model_inputs.get("output_audio", True) is not True:
            raise ValueError("CSM text-to-speech generation requires "
                             "`output_audio=True`.")

    def _generate(
        self,
        text: str,
        *,
        output_file: str | None = None,
        speaker: int = 0,
        speaker_audio_path: str | None = None,
        reference_text: str | None = None,
        max_audio_length_ms: float = 90_000,
        temperature: float = 0.9,
        top_k: int = 50,
        seed: int | None = None,
        **generation_options,
    ) -> TTSOutput:
        generation_options.pop("output_audio", None)
        context = ()
        if self._runtime is not None and speaker_audio_path is not None:
            context = (
                self._runtime.context_segment(
                    speaker=speaker,
                    text=reference_text,
                    audio=speaker_audio_path,
                ), )
        with seeded_inference(
                seed,
                device=self.device,
                model_type="csm",
        ) as effective_seed:
            if self._uses_transformers_backend:
                audio, context_count = self._generate_transformers(
                    text,
                    speaker=speaker,
                    speaker_audio_path=speaker_audio_path,
                    reference_text=reference_text,
                    max_audio_length_ms=max_audio_length_ms,
                    temperature=temperature,
                    top_k=top_k,
                    generation_options=generation_options,
                )
                metadata = {
                    "context_segments": context_count,
                    "native_runtime": False,
                    "watermarked": False,
                }
            else:
                if self._runtime is None:
                    raise RuntimeError("CSM native runtime was not loaded.")
                if generation_options:
                    raise ValueError(
                        "Unsupported native CSM generation options: " + ", ".join(sorted(generation_options)))
                audio, metadata = self._runtime.generate(
                    text,
                    speaker=speaker,
                    context=context,
                    max_audio_length_ms=max_audio_length_ms,
                    temperature=temperature,
                    top_k=top_k,
                )
        metadata.update({
            "backend": ("transformers" if self._uses_transformers_backend else "voicehub-native"),
            "speaker": speaker,
            "seed": effective_seed,
            "requested_seed": seed,
        })
        return finish_audio_output(
            audio.detach().float().cpu(),
            self.sample_rate,
            output_file=output_file,
            metadata=metadata,
        )


CSMTTS = CSMForTextToSpeech

__all__ = ["CSMConfig", "CSMForTextToSpeech", "CSMTTS"]
