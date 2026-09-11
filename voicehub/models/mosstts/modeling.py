"""Public VoiceHub integration for the native MOSS-TTS family.

No Transformers, Accelerate, PEFT, or vendored provider runtime is
imported here.  The executable graphs, Qwen byte-BPE tokenizer,
processors, strict Safetensors loader, and fine-tuning adapter are owned
by VoiceHub.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from os import PathLike
from pathlib import Path
from typing import Any

from voicehub.models._shared import finish_audio_output, seeded_inference, validate_local_file
from voicehub.models.tts import PreTrainedTTSModel
from voicehub.outputs import TTSOutput
from voicehub.training.utils import NATIVE_EXPORT_DIR

from .configuration import MossTTSConfig

_DEFAULT_MODEL = "OpenMOSS-Team/MOSS-TTS-v1.5"
_SUPPORTED_VARIANTS = ("delay", "local", "local_v1_5", "realtime")
_DEFAULT_CODEC_BY_VARIANT = {
    "delay": "OpenMOSS-Team/MOSS-Audio-Tokenizer",
    "local": "OpenMOSS-Team/MOSS-Audio-Tokenizer",
    "local_v1_5": "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2",
    "realtime": "OpenMOSS-Team/MOSS-Audio-Tokenizer",
}


class MossTTSForTextToSpeech(PreTrainedTTSModel):
    """Inference and full semantic-model SFT for all official MOSS variants."""

    config_class = MossTTSConfig
    default_model_name_or_path = _DEFAULT_MODEL
    training_default_model_name_or_path = _DEFAULT_MODEL
    passthrough_generation_options = frozenset({
        "ambient_sound",
        "audio_repetition_penalty",
        "audio_temperature",
        "audio_top_k",
        "audio_top_p",
        "duration_tokens",
        "instruction",
        "language",
        "max_new_tokens",
        "n_vq_for_inference",
        "output_file",
        "quality",
        "seed",
        "sound_event",
        "speaker_audio",
        "speaker_audio_codes",
        "speaker_audio_path",
        "speed",
        "temperature",
        "text_temperature",
        "text_top_k",
        "text_top_p",
        "top_p",
        "use_kv_cache",
    })

    def __init__(
        self,
        config: MossTTSConfig | str | Path | None = None,
        *,
        model_path: str | Path | None = None,
        device: str = "auto",
        lazy_load: bool = True,
        token: str | bool | None = None,
        **config_overrides: Any,
    ) -> None:
        if token is not None and not isinstance(token, (str, bool)):
            raise TypeError("`token` must be a string, boolean, or None.")
        if isinstance(token, str) and not token.strip():
            raise ValueError("String `token` values must be non-empty.")
        normalized = self._coerce_config(
            config,
            model_path=model_path,
            **config_overrides,
        )
        if not isinstance(normalized, MossTTSConfig):
            raise TypeError("MOSS-TTS requires MossTTSConfig.")
        normalized.validate()
        self._hub_token = token
        self._mosstts_runtime = None
        self._variant = ""
        self._codec_name_or_path: str | None = None
        super().__init__(
            normalized,
            device=device,
            lazy_load=lazy_load,
        )

    def _resolve_variant(self) -> str:
        runtime = self._mosstts_runtime
        if runtime is not None:
            return runtime.config.variant
        variant = self.config.normalize_variant(self.config.variant)
        if variant != "auto":
            if variant not in _SUPPORTED_VARIANTS:
                raise ValueError(
                    f"Unsupported MOSS-TTS variant "
                    f"{self.config.variant!r}. Choose one of: auto, "
                    f"{', '.join(_SUPPORTED_VARIANTS)}.")
            return variant
        identifier = str(self.config.name_or_path).lower()
        if "realtime" in identifier:
            return "realtime"
        if "local" in identifier and ("v1.5" in identifier or "v1_5" in identifier):
            return "local_v1_5"
        if "local" in identifier:
            return "local"
        return "delay"

    def _resolve_codec_name_or_path(self, variant: str) -> str:
        configured = self.config.codec_name_or_path
        if configured is None:
            return _DEFAULT_CODEC_BY_VARIANT[variant]
        return str(configured).strip()

    def _runtime_source(self) -> str | Path:
        source = Path(self.config.name_or_path).expanduser()
        native_export = source / NATIVE_EXPORT_DIR
        if source.is_dir() and (native_export / "config.json").is_file():
            return native_export
        return self.config.name_or_path

    def _validate_training_runtime(self) -> None:
        self._resolve_variant()
        identifier = str(self.config.name_or_path).lower()
        if any(marker in identifier for marker in (
                ".gguf",
                "-gguf",
                "/gguf",
                "llama.cpp",
                "llama_cpp",
        )):
            raise ValueError(
                "MOSS-TTS fine-tuning requires an unquantized Safetensors "
                "checkpoint, not a GGUF/llama.cpp serving artifact.")

    def _load_pretrained_model(self) -> None:
        from voicehub.models.mosstts.native.runtime import load_mosstts_runtime

        variant = self.config.normalize_variant(self.config.variant)
        runtime = load_mosstts_runtime(
            self._runtime_source(),
            device=self.device,
            compute_dtype=self.config.compute_dtype,
            variant=None if variant == "auto" else variant,
            revision=self.config.revision,
            cache_dir=(None if self.config.cache_dir is None else str(self.config.cache_dir)),
            token=self._hub_token,
            local_files_only=self.config.local_files_only,
            codec_source=self.config.codec_name_or_path,
            for_training=self.is_training_load,
        )
        self.model = runtime.model
        self._mosstts_runtime = runtime
        self._variant = runtime.config.variant
        self._codec_name_or_path = self._resolve_codec_name_or_path(self._variant)
        self.config.variant = self._variant
        self.config.sample_rate = runtime.sample_rate

    @property
    def training_backend(self):
        runtime = self._mosstts_runtime
        if runtime is not None and runtime.model is self.model:
            return runtime
        return None

    def _prepare_for_training(self) -> None:
        runtime = self.training_backend
        if runtime is None:
            raise RuntimeError("MOSS-TTS native training runtime is not loaded.")
        runtime.prepare_for_training()

    def _prepare_for_inference(self) -> None:
        runtime = self.training_backend
        if runtime is None:
            raise RuntimeError("MOSS-TTS native inference runtime is not loaded.")
        runtime.prepare_for_inference()

    def get_training_adapter(self):
        from voicehub.models.mosstts.native.training import NativeMossTTSTrainingAdapter
        from voicehub.training.specs import get_training_spec

        adapter = NativeMossTTSTrainingAdapter(
            self,
            get_training_spec(self.config.model_type),
        )
        adapter._registered_specialization = True
        return adapter

    def prepare_training_inputs(
        self,
        inputs: dict[str, Any],
        *,
        phase: str,
    ) -> dict[str, Any]:
        del phase
        runtime = self.training_backend
        if runtime is None:
            raise RuntimeError("MOSS-TTS training inputs require "
                               "`load_for_training()` first.")
        if {"input_ids", "attention_mask", "labels"} <= set(inputs):
            return dict(inputs)
        records = inputs.get("records")
        if records is None:
            records = (inputs, )
        return runtime.prepare_training_batch(records).to_dict()

    @staticmethod
    def _reference_codes(value: Any) -> tuple[Any, ...]:
        if value is None:
            return ()
        try:
            import torch
        except ModuleNotFoundError:
            torch = None
        if torch is not None and isinstance(value, torch.Tensor):
            return (value, )
        if (not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray))):
            raise TypeError("`speaker_audio_codes` must be a code matrix or sequence of "
                            "code matrices.")
        return tuple(value)

    def _validate_generation_inputs(
        self,
        model_inputs: dict[str, Any],
    ) -> None:
        text = model_inputs.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("MOSS-TTS text must be a non-empty string.")
        supplied = [
            name for name in (
                "speaker_audio",
                "speaker_audio_codes",
                "speaker_audio_path",
            ) if model_inputs.get(name) is not None
        ]
        if len(supplied) > 1:
            raise ValueError(
                "Pass only one of `speaker_audio`, `speaker_audio_codes`, "
                "or `speaker_audio_path`.")
        if model_inputs.get("speaker_audio_path") is not None:
            validate_local_file(
                model_inputs["speaker_audio_path"],
                option_name="speaker_audio_path",
            )
        max_new_tokens = model_inputs.get("max_new_tokens", 4_096)
        if (isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens <= 0):
            raise ValueError("`max_new_tokens` must be a positive integer.")
        duration = model_inputs.get("duration_tokens")
        if duration is not None and (isinstance(duration, bool) or not isinstance(duration, int) or
                                     duration <= 0):
            raise ValueError("`duration_tokens` must be a positive integer or None.")
        for name in (
                "instruction",
                "language",
                "quality",
                "sound_event",
                "ambient_sound",
        ):
            value = model_inputs.get(name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"`{name}` must be a non-empty string or None.")
        if model_inputs.get("speed") is not None:
            raise ValueError(
                "Native MOSS-TTS has no source-defined speed control. Use "
                "`duration_tokens` on non-Realtime releases.")
        if self._resolve_variant() == "realtime":
            unsupported = [
                name for name in (
                    "instruction",
                    "duration_tokens",
                    "quality",
                    "sound_event",
                    "ambient_sound",
                    "language",
                ) if model_inputs.get(name) is not None
            ]
            if unsupported:
                raise ValueError("MOSS-TTS-Realtime does not support: " + ", ".join(unsupported) + ".")

    @staticmethod
    def _generation_options(values: Mapping[str, Any]) -> dict[str, Any]:
        options = {name: value for name, value in values.items() if value is not None}
        temperature = options.pop("temperature", None)
        if temperature is not None:
            options.setdefault("audio_temperature", temperature)
            options.setdefault("text_temperature", temperature)
        top_p = options.pop("top_p", None)
        if top_p is not None:
            options.setdefault("audio_top_p", top_p)
        return options

    def _generate_code_segments(
        self,
        text: str,
        *,
        reference_codes: Sequence[Any],
        instruction: str | None,
        duration_tokens: int | None,
        quality: str | None,
        sound_event: str | None,
        ambient_sound: str | None,
        language: str | None,
        max_new_tokens: int,
        generation_options: Mapping[str, Any],
    ):
        runtime = self._mosstts_runtime
        if runtime is None:
            raise RuntimeError("MOSS-TTS native runtime is not loaded.")
        options = self._generation_options(generation_options)
        return runtime.generate_codes(
            text,
            reference_codes=reference_codes,
            instruction=instruction,
            duration_tokens=duration_tokens,
            quality=quality,
            sound_event=sound_event,
            ambient_sound=ambient_sound,
            language=language,
            max_new_tokens=max_new_tokens,
            **options,
        )

    def generate_codes(
        self,
        text: str,
        *,
        speaker_audio_codes: Any | None = None,
        speaker_audio: Any | None = None,
        instruction: str | None = None,
        duration_tokens: int | None = None,
        quality: str | None = None,
        sound_event: str | None = None,
        ambient_sound: str | None = None,
        language: str | None = None,
        max_new_tokens: int = 4_096,
        seed: int | None = None,
        **generation_options: Any,
    ):
        inputs = {
            "text": text,
            "speaker_audio_codes": speaker_audio_codes,
            "speaker_audio": speaker_audio,
            "instruction": instruction,
            "duration_tokens": duration_tokens,
            "quality": quality,
            "sound_event": sound_event,
            "ambient_sound": ambient_sound,
            "language": language,
            "max_new_tokens": max_new_tokens,
            "seed": seed,
            **generation_options,
        }
        self._validate_generation_inputs(inputs)
        with self._lifecycle_lock:
            self.load()
            runtime = self._mosstts_runtime
            if runtime is None:
                raise RuntimeError("MOSS-TTS native runtime is not loaded.")
            references = self._reference_codes(speaker_audio_codes)
            if speaker_audio is not None:
                references = (runtime.encode_reference(runtime.load_reference_audio(speaker_audio), ), )
            with seeded_inference(
                    seed,
                    device=self.device,
                    model_type="mosstts",
            ):
                return self._generate_code_segments(
                    text,
                    reference_codes=references,
                    instruction=instruction,
                    duration_tokens=duration_tokens,
                    quality=quality,
                    sound_event=sound_event,
                    ambient_sound=ambient_sound,
                    language=language,
                    max_new_tokens=max_new_tokens,
                    generation_options=generation_options,
                )

    @staticmethod
    def _normalize_waveform(output: Any) -> tuple[Any, int]:
        import torch

        waveform = output.waveform
        length = int(output.waveform_lengths[0].item())
        if waveform.ndim == 3:
            channels = int(waveform.shape[1])
            waveform = waveform[0, :, :length]
            waveform = (waveform[0] if channels == 1 else waveform.mean(dim=0))
        else:
            channels = 1
            waveform = waveform[0, :length]
        if waveform.numel() == 0:
            raise RuntimeError("MOSS-TTS codec returned an empty waveform.")
        return waveform.detach().float().cpu(), channels

    def _generate(
        self,
        text: str,
        *,
        output_file: str | Path | None = None,
        speaker_audio_path: str | PathLike[str] | None = None,
        speaker_audio_codes: Any | None = None,
        speaker_audio: Any | None = None,
        instruction: str | None = None,
        duration_tokens: int | None = None,
        quality: str | None = None,
        sound_event: str | None = None,
        ambient_sound: str | None = None,
        language: str | None = None,
        max_new_tokens: int = 4_096,
        seed: int | None = None,
        speed: float | None = None,
        **generation_options: Any,
    ) -> TTSOutput:
        del speed
        runtime = self._mosstts_runtime
        if runtime is None:
            raise RuntimeError("MOSS-TTS native runtime is not loaded.")
        references = self._reference_codes(speaker_audio_codes)
        if speaker_audio_path is not None:
            speaker_audio = runtime.load_reference_audio(speaker_audio_path)
        if speaker_audio is not None:
            references = (runtime.encode_reference(runtime.load_reference_audio(speaker_audio), ), )
        with seeded_inference(
                seed,
                device=self.device,
                model_type="mosstts",
        ) as effective_seed:
            generated = self._generate_code_segments(
                text,
                reference_codes=references,
                instruction=instruction,
                duration_tokens=duration_tokens,
                quality=quality,
                sound_event=sound_event,
                ambient_sound=ambient_sound,
                language=language,
                max_new_tokens=max_new_tokens,
                generation_options=generation_options,
            )
            decoded = [
                runtime.decode_codes(item.audio_codes) for item in generated if item.audio_codes.numel()
            ]
        if not decoded:
            raise RuntimeError("MOSS-TTS produced no decodable audio codes.")
        normalized = [self._normalize_waveform(item) for item in decoded]
        import torch

        audio = torch.cat([item[0] for item in normalized])
        source_channels = max(item[1] for item in normalized)
        revision = (None if runtime.artifacts is None else runtime.artifacts.revision)
        return finish_audio_output(
            audio,
            self.sample_rate,
            output_file=output_file,
            metadata={
                "backend": "voicehub-native",
                "checkpoint_revision": revision,
                "codec_name_or_path": self._codec_name_or_path,
                "downmixed_to_mono": source_channels > 1,
                "language": language,
                "requested_seed": seed,
                "seed": effective_seed,
                "source_channels": source_channels,
                "variant": self._variant,
            },
        )

    def _save_pretrained(self, save_directory: Path) -> None:
        if self._mosstts_runtime is None:
            self.load()
        runtime = self.training_backend
        if runtime is None:
            raise RuntimeError(
                "Restore MOSS-TTS to its native trainable graph before "
                "exporting an inference-reloadable checkpoint.")
        runtime.save_pretrained(save_directory)


MossTTS = MossTTSForTextToSpeech

__all__ = [
    "MossTTS",
    "MossTTSConfig",
    "MossTTSForTextToSpeech",
]
