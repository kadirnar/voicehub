"""VoiceHub-native VibeVoice TTS loading and fine-tuning lifecycle.

The published realtime checkpoint exposes a staged decoder, diffusion
head, and causal acoustic decoder.  VoiceHub loads those stages
natively, but does not present the upstream cached-prompt loop as
equivalent until its mutable cache format and chunk boundaries have an
independent parity suite.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicehub.models.tts import PreTrainedTTSModel
from voicehub.outputs import TTSOutput

from .configuration import VibeVoiceConfig

_HIGH_LEVEL_GENERATION_ERROR = (
    "VoiceHub has native VibeVoice TTS graphs, but high-level cached-prompt "
    "synthesis is not enabled: cache serialization, chunk boundaries, and "
    "waveform parity have not yet been independently verified. Load the "
    "realtime checkpoint and use `forward_lm`, `forward_tts_lm`, "
    "`sample_speech_latents`, and `decode_speech_latents` directly; the "
    "non-streaming checkpoint is supported for fine-tuning.")


class VibeVoiceForTextToSpeech(PreTrainedTTSModel):
    """Single-speaker realtime VibeVoice generation with cached voice
    prompts."""

    config_class = VibeVoiceConfig
    default_model_name_or_path = "microsoft/VibeVoice-Realtime-0.5B"
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: VibeVoiceConfig | str | None = None,
        *,
        model_path: str | None = None,
        device: str = "auto",
        lazy_load: bool = True,
        token: str | bool | None = None,
        **config_overrides,
    ):
        if token is not None and not isinstance(token, (str, bool)):
            raise TypeError("`token` must be a string, boolean, or None.")
        if isinstance(token, str) and not token.strip():
            raise ValueError("String `token` values must be non-empty.")
        config = self._coerce_config(
            config,
            model_path=model_path,
            **config_overrides,
        )
        config.validate()
        self._hub_token = token
        self._torch = None
        self._processor = None
        self._safe_globals = ()
        self._runtime_kind = None
        self.runtime = None
        super().__init__(config, device=device, lazy_load=lazy_load)

    def _load_pretrained_model(self) -> None:
        import torch

        from voicehub.models.vibevoice.native.configuration import VibeVoiceTTSConfig as NativeVibeVoiceTTSConfig
        from voicehub.models.vibevoice.native.runtime import load_vibevoice_runtime

        runtime = load_vibevoice_runtime(
            self.config.name_or_path,
            device=self.device,
            compute_dtype=self.config.torch_dtype,
            revision=self.config.revision,
            cache_dir=self.config.cache_dir,
            token=self._hub_token,
            local_files_only=self.config.local_files_only,
            for_training=self.is_training_load,
        )
        if not isinstance(runtime.config, NativeVibeVoiceTTSConfig):
            raise TypeError("VibeVoice TTS received an ASR checkpoint.")
        self.runtime = runtime
        self.model = runtime.model
        self._processor = runtime.processor
        self._torch = torch
        self.config.sample_rate = self._checkpoint_sample_rate(runtime.processor)
        self._runtime_kind = (
            "native-realtime-stages" if runtime.config.is_streaming else "native-non-streaming-training")

    def _validate_training_runtime(self) -> None:
        """Reject published realtime checkpoints before graph allocation."""
        identifier = str(self.config.name_or_path)
        source = Path(identifier).expanduser()
        if source.exists():
            root = source if source.is_dir() else source.parent
            configuration_path = root / "config.json"
            if not configuration_path.is_file():
                raise FileNotFoundError(
                    "A local VibeVoice training directory must contain "
                    f"`config.json`: {configuration_path}.")
            from voicehub.hub import read_json_file

            configuration = read_json_file(configuration_path)
            if str(configuration.get("model_type", "")).lower() != "vibevoice":
                raise ValueError(
                    "VibeVoice fine-tuning requires the non-streaming 1.5B "
                    'architecture (`model_type="vibevoice"`).')
            return
        normalized = identifier.strip().lower()
        if normalized in {
                "microsoft/vibevoice-realtime-0.5b",
                "microsoft/vibevoice-asr-hf",
        }:
            raise ValueError(
                "VibeVoice TTS fine-tuning requires a non-streaming "
                '`model_type="vibevoice"` checkpoint. The realtime release '
                "has no unified training forward, and the ASR release belongs "
                "to the speech-recognition provider.")

    def _prepare_for_training(self) -> None:
        if self._runtime_kind != "native-non-streaming-training":
            raise ValueError("VibeVoice fine-tuning requires the non-streaming 1.5B runtime.")
        if self.runtime is None:
            raise RuntimeError("VibeVoice native runtime is not loaded.")
        self.runtime.prepare_for_training()

    def _prepare_for_inference(self) -> None:
        if self.runtime is None:
            raise RuntimeError("VibeVoice native runtime is not loaded.")
        self.runtime.prepare_for_inference()

    @staticmethod
    def _checkpoint_sample_rate(processor: Any) -> int:
        audio_processor = getattr(processor, "audio_processor", None)
        sample_rate = getattr(
            audio_processor,
            "sample_rate",
            getattr(audio_processor, "sampling_rate", None),
        )
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int):
            raise TypeError(
                "The VibeVoice checkpoint processor does not define an "
                "integer audio sampling rate.")
        if sample_rate <= 0:
            raise RuntimeError(
                "The VibeVoice checkpoint processor defines an invalid "
                f"audio sampling rate: {sample_rate}.")
        return sample_rate

    def _validate_generation_inputs(self, model_inputs: dict[str, Any]) -> None:
        voice_prompt_path = model_inputs.get("voice_prompt_path")
        if voice_prompt_path is not None:
            if (not isinstance(voice_prompt_path, (str, Path)) or not str(voice_prompt_path).strip()):
                raise ValueError("`voice_prompt_path` must be a non-empty path or None.")
            prompt_path = Path(voice_prompt_path).expanduser()
            if not prompt_path.is_file():
                raise FileNotFoundError(f"VibeVoice cached voice prompt was not found: {prompt_path}.")
        raise RuntimeError(_HIGH_LEVEL_GENERATION_ERROR)

    def _load_cached_prompt(self, voice_prompt_path: str) -> Mapping[str, Any]:
        with self._torch.serialization.safe_globals(list(self._safe_globals)):
            cached_prompt = self._torch.load(
                str(Path(voice_prompt_path).expanduser()),
                map_location=self.device,
                weights_only=True,
            )
        if not isinstance(cached_prompt, Mapping):
            raise TypeError("VibeVoice cached prompt must contain a mapping.")
        required_sections = ("lm", "tts_lm", "neg_lm", "neg_tts_lm")
        missing = [name for name in required_sections if name not in cached_prompt]
        if missing:
            raise ValueError(
                "VibeVoice cached prompt is missing required section(s): " + ", ".join(missing) + ".")
        return cached_prompt

    def _move_inputs_to_device(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: value.to(self.device) if self._torch.is_tensor(value) else value
            for key, value in inputs.items()
        }

    @staticmethod
    def _generation_kwargs(
        generation_options: Mapping[str, Any],
        *,
        max_new_tokens: int | None,
        cfg_scale: float,
        tokenizer: Any,
        cached_prompt: Mapping[str, Any],
    ) -> dict[str, Any]:
        reserved = {
            "all_prefilled_outputs",
            "cfg_scale",
            "max_new_tokens",
            "tokenizer",
        }
        conflicts = sorted(reserved & set(generation_options))
        if conflicts:
            raise ValueError(
                "VibeVoice generation option(s) are managed by the wrapper: " + ", ".join(conflicts) + ".")
        options = dict(generation_options)
        options.setdefault("generation_config", {"do_sample": False})
        options.update({
            "cfg_scale": cfg_scale,
            "tokenizer": tokenizer,
            "all_prefilled_outputs": copy.deepcopy(cached_prompt),
        })
        if max_new_tokens is not None:
            options["max_new_tokens"] = max_new_tokens
        return options

    def _generate(
        self,
        text: str,
        *,
        output_file: str | None = None,
        voice_prompt_path: str | None = None,
        cfg_scale: float = 1.5,
        max_new_tokens: int | None = None,
        seed: int | None = None,
        **generation_options,
    ) -> TTSOutput:
        del (
            text,
            output_file,
            voice_prompt_path,
            cfg_scale,
            max_new_tokens,
            seed,
            generation_options,
        )
        raise RuntimeError(_HIGH_LEVEL_GENERATION_ERROR)

    def _save_pretrained(self, save_directory: Path) -> None:
        if self.runtime is None:
            self.load_for_training()
        from voicehub.models.vibevoice.native.runtime import save_vibevoice_runtime

        save_vibevoice_runtime(self.runtime, save_directory)


VibeVoiceTTS = VibeVoiceForTextToSpeech
