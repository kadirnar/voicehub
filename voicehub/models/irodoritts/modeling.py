"""VoiceHub-owned Irodori-TTS inference and fine-tuning wrapper."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from voicehub.hub import resolve_pretrained_file
from voicehub.models._shared import finish_audio_output, resolve_model_directory, validate_local_file
from voicehub.models.tts import PreTrainedTTSModel
from voicehub.outputs import TTSOutput

from .configuration import IrodoriTTSConfig


class IrodoriTTSForTextToSpeech(PreTrainedTTSModel):
    """Flow-matching speech synthesis with optional reference or caption."""

    config_class = IrodoriTTSConfig
    default_model_name_or_path = "Aratako/Irodori-TTS-500M-v3"
    passthrough_generation_options = frozenset({
        "cfg_guidance_mode",
        "cfg_max_t",
        "cfg_min_t",
        "cfg_scale",
        "context_kv_cache",
        "decode_mode",
        "lora_adapter",
        "max_caption_len",
        "max_ref_seconds",
        "max_seconds",
        "max_text_len",
        "min_seconds",
        "num_candidates",
        "ref_embed",
        "ref_ensure_max",
        "ref_latent",
        "ref_normalize_db",
        "rescale_k",
        "rescale_sigma",
        "speaker_kv_max_layers",
        "speaker_kv_min_t",
        "speaker_kv_scale",
        "speaker_uncond_mode",
        "sway_coeff",
        "t_schedule_mode",
        "tail_mean_threshold",
        "tail_std_threshold",
        "tail_window_size",
        "trim_tail",
        "truncation_factor",
        "watermark",
    })

    def __init__(
        self,
        config: IrodoriTTSConfig | str | None = None,
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
        config.validate()
        self._runtime_module = None
        self._loaded_for_training = False
        super().__init__(config, device=device, lazy_load=lazy_load)

    def _resolve_checkpoint(self) -> Path:
        source = Path(self.config.name_or_path).expanduser()
        if source.is_file():
            return source.absolute()
        revision = self.config.model_revision
        if revision is None:
            from voicehub.models.irodoritts.native.metadata import IRODORI_CHECKPOINTS

            revision = next(
                (
                    facts["revision"] for facts in IRODORI_CHECKPOINTS.values()
                    if facts["model_id"] == self.config.name_or_path),
                None,
            )
        model_directory = resolve_model_directory(
            self.config.name_or_path,
            model_type="irodoritts",
            revision=revision,
        )
        if self.config.checkpoint_filename:
            checkpoint = model_directory / self.config.checkpoint_filename
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Irodori checkpoint not found: {checkpoint}.")
            return checkpoint.absolute()

        candidates = [
            path for pattern in ("*.safetensors", ) for path in sorted(model_directory.glob(pattern))
            if not path.name.endswith(".speaker.safetensors") and
            path.name not in {"adapter_model.safetensors", "adapter_model.bin"}
        ]
        if not candidates:
            raise FileNotFoundError(f"No Irodori checkpoint found in {model_directory}.")
        if len(candidates) == 1:
            return candidates[0].absolute()

        preferred_names = (
            "model.safetensors",
            "pytorch_model.safetensors",
            "irodori.safetensors",
            "checkpoint.safetensors",
        )
        preferred = [model_directory / name for name in preferred_names if (model_directory / name).is_file()]
        if len(preferred) == 1:
            return preferred[0].absolute()
        names = ", ".join(path.name for path in candidates)
        raise ValueError(
            "Multiple Irodori checkpoints were found "
            f"({names}). Set `checkpoint_filename` explicitly.")

    def _build_runtime(self, runtime):
        checkpoint = self._resolve_checkpoint()
        checkpoint_model_id = checkpoint_revision = None
        configured_source = Path(self.config.name_or_path).expanduser()
        if not configured_source.exists():
            from voicehub.models.irodoritts.native.metadata import IRODORI_CHECKPOINTS

            published = next(
                (
                    facts for facts in IRODORI_CHECKPOINTS.values()
                    if facts["model_id"] == self.config.name_or_path and
                    (self.config.model_revision is None or facts["revision"] == self.config.model_revision)),
                None,
            )
            if published is not None:
                checkpoint_model_id = published["model_id"]
                checkpoint_revision = published["revision"]
        local_tokenizer_json = checkpoint.parent / "tokenizer.json"
        local_tokenizer_config = checkpoint.parent / "tokenizer_config.json"
        if local_tokenizer_json.is_file() and local_tokenizer_config.is_file():
            tokenizer_json = local_tokenizer_json
        else:
            tokenizer_json = resolve_pretrained_file(
                self.config.tokenizer_name_or_path,
                "tokenizer.json",
                revision=self.config.tokenizer_revision,
            )
            resolve_pretrained_file(
                self.config.tokenizer_name_or_path,
                "tokenizer_config.json",
                revision=self.config.tokenizer_revision,
            )
        codec_checkpoint = resolve_pretrained_file(
            self.config.codec_name_or_path,
            "weights.pth",
            revision=self.config.codec_revision,
        )
        key = runtime.RuntimeKey(
            checkpoint=str(checkpoint),
            checkpoint_model_id=checkpoint_model_id,
            checkpoint_revision=checkpoint_revision,
            model_device=self.device,
            codec_repo=str(codec_checkpoint),
            tokenizer_directory=str(tokenizer_json.parent),
            model_precision=self.config.model_precision,
            codec_device=self.device,
            codec_precision=self.config.codec_precision,
            compile_model=(self.config.compile_model and not self.is_training_load),
        )
        model = runtime.InferenceRuntime.from_key(key)
        if not callable(getattr(model, "synthesize", None)):
            raise TypeError("The loaded Irodori runtime does not implement synthesize().")
        sample_rate = int(getattr(getattr(model, "codec", None), "sample_rate", 0))
        if sample_rate <= 0:
            raise ValueError("The loaded Irodori codec reported an invalid sample rate.")
        return model, sample_rate

    def _load_pretrained_model(self) -> None:
        from voicehub.models.irodoritts.native import runtime

        model, sample_rate = self._build_runtime(runtime)
        self.model = model
        self._runtime_module = runtime
        self.config.sample_rate = sample_rate
        self._loaded_for_training = self.is_training_load

    def _prepare_for_training(self) -> None:
        if self._loaded_for_training:
            source_model = getattr(self.model, "model", None)
            if source_model is not None and hasattr(source_model, "train"):
                source_model.train()
            return
        previous_state = (
            self.model,
            self._runtime_module,
            self._loaded_for_training,
        )
        self.model = None
        self._runtime_module = None
        previous_loading_mode = self._loading_for_training
        self._loading_for_training = True
        try:
            self.load()
        except BaseException:
            (
                self.model,
                self._runtime_module,
                self._loaded_for_training,
            ) = previous_state
            raise
        finally:
            self._loading_for_training = previous_loading_mode

    def _prepare_for_inference(self) -> None:
        """Put the runtime's trainable model and codec in serving mode."""
        source_model = getattr(self.model, "model", None)
        if source_model is not None and hasattr(source_model, "eval"):
            source_model.eval()
        codec = getattr(self.model, "codec", None)
        codec_model = getattr(codec, "model", codec)
        if codec_model is not None and hasattr(codec_model, "eval"):
            codec_model.eval()

    def _set_training_device(self, device: str) -> None:
        """Synchronize nested runtime routing after Trainer placement."""
        super()._set_training_device(device)
        runtime = self.model
        if runtime is None:
            return
        current_model_device = getattr(runtime, "model_device", device)
        try:
            runtime.model_device = type(current_model_device)(device)
        except (TypeError, ValueError):
            runtime.model_device = device
        source_model = getattr(runtime, "model", None)
        parameters = getattr(source_model, "parameters", None)
        if callable(parameters):
            first_parameter = next(iter(parameters()), None)
            if first_parameter is not None:
                runtime._model_dtype = first_parameter.dtype

        codec = getattr(runtime, "codec", None)
        codec_model = getattr(codec, "model", None)
        move_codec = getattr(codec_model, "to", None)
        if callable(move_codec):
            moved_codec = move_codec(device)
            if moved_codec is not None:
                codec.model = moved_codec
        if codec is not None:
            runtime.codec_device = getattr(codec, "device", device)

    def _validate_generation_inputs(self, model_inputs: dict[str, Any]) -> None:
        super()._validate_generation_inputs(model_inputs)
        no_reference = model_inputs.get("no_reference", False)
        if not isinstance(no_reference, bool):
            raise TypeError("`no_reference` must be a boolean.")
        reference_options = {
            "speaker_audio_path": model_inputs.get("speaker_audio_path"),
            "ref_latent": model_inputs.get("ref_latent"),
            "ref_embed": model_inputs.get("ref_embed"),
        }
        supplied_references = [name for name, value in reference_options.items() if value is not None]
        if len(supplied_references) > 1:
            raise ValueError("Pass only one of `speaker_audio_path`, `ref_latent`, or "
                             "`ref_embed`.")
        if no_reference and supplied_references:
            raise ValueError("`no_reference=True` cannot be combined with a reference.")
        for name, value in reference_options.items():
            path = validate_local_file(
                value,
                option_name=name,
            )
            if path is not None:
                model_inputs[name] = str(path)

        caption = model_inputs.get("caption")
        if caption is not None and (not isinstance(caption, str) or not caption.strip()):
            raise ValueError("`caption` must be a non-empty string or None.")

        seconds = model_inputs.get("seconds")
        if seconds is not None:
            if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
                raise TypeError("`seconds` must be a finite number or None.")
            if not math.isfinite(float(seconds)) or seconds <= 0:
                raise ValueError("`seconds` must be finite and greater than zero.")

        duration_scale = model_inputs.get("duration_scale", 1.0)
        if (isinstance(duration_scale, bool) or not isinstance(duration_scale, (int, float))):
            raise TypeError("`duration_scale` must be a finite number.")
        if not math.isfinite(float(duration_scale)) or duration_scale <= 0:
            raise ValueError("`duration_scale` must be finite and greater than zero.")

        num_steps = model_inputs.get("num_steps", 40)
        if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps <= 0:
            raise ValueError("`num_steps` must be a positive integer.")
        for name in (
                "cfg_scale_text",
                "cfg_scale_caption",
                "cfg_scale_speaker",
                "cfg_scale",
        ):
            value = model_inputs.get(name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"`{name}` must be a finite number.")
            if not math.isfinite(float(value)) or value < 0:
                raise ValueError(f"`{name}` must be finite and non-negative.")

        min_seconds = model_inputs.get("min_seconds", 0.5)
        max_seconds = model_inputs.get("max_seconds", 30.0)
        for name, value in (
            ("min_seconds", min_seconds),
            ("max_seconds", max_seconds),
        ):
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                    not math.isfinite(float(value)) or value <= 0):
                raise ValueError(f"`{name}` must be finite and positive.")
        if min_seconds > max_seconds:
            raise ValueError("`min_seconds` cannot exceed `max_seconds`.")

        for name in ("max_ref_seconds", "truncation_factor"):
            value = model_inputs.get(name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or
                                      not math.isfinite(float(value)) or value <= 0):
                raise ValueError(f"`{name}` must be finite and positive or None.")

        for name in (
                "max_text_len",
                "max_caption_len",
                "num_candidates",
                "tail_window_size",
        ):
            defaults = {
                "num_candidates": 1,
                "tail_window_size": 20,
            }
            value = model_inputs.get(name, defaults.get(name))
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
                raise ValueError(f"`{name}` must be a positive integer or None.")

        speaker_layers = model_inputs.get("speaker_kv_max_layers")
        if speaker_layers is not None and (isinstance(speaker_layers, bool) or
                                           not isinstance(speaker_layers, int) or speaker_layers < 0):
            raise ValueError("`speaker_kv_max_layers` must be a non-negative integer or "
                             "None.")

        for name, default in (
            ("ref_ensure_max", True),
            ("context_kv_cache", True),
            ("trim_tail", True),
        ):
            if not isinstance(model_inputs.get(name, default), bool):
                raise TypeError(f"`{name}` must be a boolean.")

        for name, supported in (
            ("decode_mode", {"batch", "sequential"}),
            (
                "cfg_guidance_mode",
                {"alternating", "independent", "joint"},
            ),
            ("t_schedule_mode", {"linear", "sway"}),
        ):
            value = model_inputs.get(name)
            if value is not None and (not isinstance(value, str) or value.strip().lower() not in supported):
                choices = ", ".join(sorted(supported))
                raise ValueError(f"`{name}` must be one of: {choices}.")

        cfg_min_t = model_inputs.get("cfg_min_t", 0.5)
        cfg_max_t = model_inputs.get("cfg_max_t", 1.0)
        for name, value in (
            ("cfg_min_t", cfg_min_t),
            ("cfg_max_t", cfg_max_t),
        ):
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                    not math.isfinite(float(value)) or not 0 <= value <= 1):
                raise ValueError(f"`{name}` must be finite and in [0, 1].")
        if cfg_min_t > cfg_max_t:
            raise ValueError("`cfg_min_t` cannot exceed `cfg_max_t`.")
        speaker_kv_min_t = model_inputs.get("speaker_kv_min_t")
        valid_speaker_minimum = (
            isinstance(speaker_kv_min_t, (int, float)) and not isinstance(speaker_kv_min_t, bool) and
            math.isfinite(float(speaker_kv_min_t)) and 0 <= speaker_kv_min_t <= 1)
        if (speaker_kv_min_t is not None and not valid_speaker_minimum):
            raise ValueError("`speaker_kv_min_t` must be finite and in [0, 1] or None.")

        rescale_k = model_inputs.get("rescale_k")
        rescale_sigma = model_inputs.get("rescale_sigma")
        if (rescale_k is None) != (rescale_sigma is None):
            raise ValueError("`rescale_k` and `rescale_sigma` must be provided together.")
        for name, value in (
            ("rescale_k", rescale_k),
            ("rescale_sigma", rescale_sigma),
            ("speaker_kv_scale", model_inputs.get("speaker_kv_scale")),
        ):
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or
                                      not math.isfinite(float(value)) or value <= 0):
                raise ValueError(f"`{name}` must be finite and positive or None.")

        for name, default in (
            ("ref_normalize_db", -16.0),
            ("sway_coeff", -1.0),
            ("tail_std_threshold", 0.05),
            ("tail_mean_threshold", 0.1),
        ):
            value = model_inputs.get(name, default)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or
                                      not math.isfinite(float(value))):
                raise ValueError(f"`{name}` must be finite or None.")

        lora_adapter = model_inputs.get("lora_adapter")
        if lora_adapter is not None and (not isinstance(lora_adapter,
                                                        (str, Path)) or not str(lora_adapter).strip()):
            raise ValueError("`lora_adapter` must be a non-empty path or Hub ID.")
        watermark = model_inputs.get("watermark", False)
        if not isinstance(watermark, bool):
            raise TypeError("`watermark` must be a boolean.")

    def _generate(
        self,
        text: str,
        *,
        output_file: str | None = None,
        speaker_audio_path: str | None = None,
        caption: str | None = None,
        no_reference: bool = False,
        seconds: float | None = None,
        duration_scale: float = 1.0,
        num_steps: int = 40,
        cfg_scale_text: float = 3.0,
        cfg_scale_caption: float = 3.0,
        cfg_scale_speaker: float = 5.0,
        seed: int | None = None,
        **sampling_options,
    ) -> TTSOutput:
        if self.model is None or self._runtime_module is None:
            raise RuntimeError("Irodori-TTS must be loaded before generation.")
        request = self._runtime_module.SamplingRequest(
            text=text,
            caption=caption,
            ref_wav=speaker_audio_path,
            no_ref=(
                no_reference or not any((
                    speaker_audio_path,
                    sampling_options.get("ref_latent"),
                    sampling_options.get("ref_embed"),
                ))),
            seconds=seconds,
            duration_scale=duration_scale,
            num_steps=num_steps,
            cfg_scale_text=cfg_scale_text,
            cfg_scale_caption=cfg_scale_caption,
            cfg_scale_speaker=cfg_scale_speaker,
            seed=seed,
            **sampling_options,
        )
        result = self.model.synthesize(request)
        sample_rate = int(getattr(result, "sample_rate", 0))
        if sample_rate <= 0:
            raise ValueError("Irodori-TTS inference returned an invalid sample rate.")
        if not hasattr(result, "audio"):
            raise TypeError("Irodori-TTS inference returned no audio waveform.")
        self.config.sample_rate = sample_rate
        return finish_audio_output(
            result.audio,
            sample_rate,
            output_file=output_file,
            metadata={
                "caption": caption,
                "seed": result.used_seed,
                "stage_timings": result.stage_timings,
            },
        )

    def prepare_training_inputs(
        self,
        inputs: dict[str, Any],
        *,
        phase: str,
    ) -> dict[str, Any]:
        del phase
        if self.model is None:
            raise RuntimeError("Irodori-TTS must be loaded before preparing training data.")
        from voicehub.models.irodoritts.native.training import IrodoriBatchProcessor

        processor = IrodoriBatchProcessor(
            config=self.model.model_cfg,
            tokenizer=self.model.tokenizer,
            codec=self.model.codec,
            device=self.model.model_device,
        )
        return dict(processor(inputs))

    def get_training_adapter(self):
        from voicehub.models.irodoritts.training import NativeIrodoriTrainingAdapter
        from voicehub.training.specs import get_training_spec

        return NativeIrodoriTrainingAdapter(self, get_training_spec("irodoritts"))

    def _save_pretrained(self, save_directory: Path) -> None:
        if self.model is None:
            raise RuntimeError("Irodori-TTS must be loaded before native export.")
        from voicehub.models.irodoritts.native.checkpoint import save_irodori_safetensors

        save_directory.mkdir(parents=True, exist_ok=True)
        save_irodori_safetensors(
            self.model.model,
            self.model.model_cfg,
            save_directory / "model.safetensors",
        )
        self.model.tokenizer.save_pretrained(save_directory)


IrodoriTTS = IrodoriTTSForTextToSpeech
