"""VoiceHub-native F5-TTS inference and fine-tuning lifecycle."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch

from voicehub.models._shared import finish_audio_output, resolve_torch_dtype
from voicehub.models.f5tts.configuration import F5TTSConfig
from voicehub.models.f5tts.native.artifacts import resolve_f5tts_artifacts
from voicehub.models.f5tts.native.checkpoint import load_f5tts_checkpoint, load_vocos_checkpoint
from voicehub.models.f5tts.native.frontend import F5Vocabulary, NativeF5TextFrontend
from voicehub.models.f5tts.native.modeling import F5ConditionalFlowMatcher
from voicehub.models.f5tts.native.runtime import NativeF5TTSRuntime
from voicehub.models.f5tts.native.vocoder import NativeVocos
from voicehub.models.tts import PreTrainedTTSModel
from voicehub.outputs import TTSOutput


class F5TTSForTextToSpeech(PreTrainedTTSModel):
    """F5-TTS voice cloning implemented with PyTorch and VoiceHub only."""

    config_class = F5TTSConfig
    default_model_name_or_path = "F5TTS_v1_Base"

    def __init__(
        self,
        config: F5TTSConfig | str | None = None,
        *,
        model_path: str | None = None,
        device: str = "auto",
        lazy_load: bool = True,
        **config_overrides: Any,
    ) -> None:
        explicit_model_source = (
            model_path is not None or isinstance(config, (str, Path)) or
            (isinstance(config, F5TTSConfig) and bool(config.name_or_path)))
        config = self._coerce_config(
            config,
            model_path=model_path,
            **config_overrides,
        )
        source = Path(config.name_or_path).expanduser()
        if source.is_file():
            configured_checkpoint = str(config.checkpoint_path).strip()
            if configured_checkpoint:
                existing = Path(configured_checkpoint).expanduser()
                if existing.resolve() != source.resolve():
                    raise ValueError(
                        "The direct F5-TTS checkpoint and `checkpoint_path` "
                        "refer to different files.")
            config.checkpoint_path = str(source.resolve())
        elif not explicit_model_source:
            config.name_or_path = config.model_name
        config.validate()
        self.artifacts = None
        super().__init__(config, device=device, lazy_load=lazy_load)

    def _load_pretrained_model(self) -> None:
        architecture = self.config.architecture_config()
        dtype = resolve_torch_dtype(
            torch,
            self.config.torch_dtype,
            self.device,
        )
        if (not self._loading_for_training and dtype != torch.float32 and
                not self.config.allow_unvalidated_reduced_precision_inference):
            raise RuntimeError(
                "F5-TTS reduced-precision inference is disabled by default "
                f"for {str(dtype).removeprefix('torch.')}. Use "
                "`torch_dtype=\"float32\"`, or set "
                "`allow_unvalidated_reduced_precision_inference=True` only "
                "after checkpoint-level quality validation.")
        checkpoint_path = str(self.config.checkpoint_path).strip() or None
        vocabulary_path = str(self.config.vocabulary_path).strip() or None
        artifacts = resolve_f5tts_artifacts(
            self.config.name_or_path,
            model_name=self.config.model_name,
            checkpoint_path=checkpoint_path,
            vocabulary_path=vocabulary_path,
            vocoder_path=self.config.vocoder_path,
            include_vocoder=True,
            revision=self.config.revision,
            cache_dir=self.config.cache_dir,
            token=self.config.token,
            local_files_only=self.config.local_files_only,
            verify_integrity=self.config.verify_artifacts,
        )
        vocabulary = F5Vocabulary.from_file(artifacts.vocabulary)
        if len(vocabulary) != architecture.text_num_embeds:
            raise ValueError(
                f"F5-TTS vocabulary contains {len(vocabulary)} tokens, but "
                f"the configured graph expects {architecture.text_num_embeds}.")
        with torch.device(self.device):
            flow_model = F5ConditionalFlowMatcher(
                architecture,
                ode_method=self.config.ode_method,
            )
        if dtype != torch.float32:
            flow_model.to(dtype=dtype)
        load_f5tts_checkpoint(
            flow_model,
            artifacts.checkpoint,
            use_ema=self.config.use_ema,
            strict=True,
            device=self.device,
        )
        if artifacts.vocoder is None:
            raise RuntimeError("Native F5-TTS did not resolve a Vocos checkpoint.")
        with torch.device(self.device):
            vocoder = NativeVocos()
        if dtype != torch.float32:
            vocoder.to(dtype=dtype)
        load_vocos_checkpoint(
            vocoder,
            artifacts.vocoder,
            device=self.device,
        )
        runtime = NativeF5TTSRuntime(
            flow_model=flow_model,
            vocoder=vocoder,
            frontend=NativeF5TextFrontend(vocabulary),
            allow_unvalidated_reduced_precision_inference=(
                self.config.allow_unvalidated_reduced_precision_inference),
        )
        runtime.to(device=self.device)
        self.artifacts = artifacts
        self.config.sample_rate = architecture.sample_rate
        self.model = runtime

    def _set_training_device(self, device: str) -> None:
        super()._set_training_device(device)
        if self.model is not None:
            self.model.to(device=device)

    def _prepare_for_training(self) -> None:
        if self.model is None:
            return
        prepare = getattr(self.model, "prepare_for_training", None)
        if callable(prepare):
            prepare()

    def _prepare_for_inference(self) -> None:
        if self.model is None:
            return
        prepare = getattr(self.model, "prepare_for_inference", None)
        if callable(prepare):
            prepare()
            return
        # Retain compatibility with lightweight lifecycle test doubles.
        for component_name in ("ema_model", "vocoder"):
            component = getattr(self.model, component_name, None)
            evaluate = getattr(component, "eval", None)
            if callable(evaluate):
                evaluate()

    def _validate_generation_inputs(self, model_inputs: dict[str, Any]) -> None:
        super()._validate_generation_inputs(model_inputs)
        speaker_audio = model_inputs.get("speaker_audio_path")
        if not isinstance(speaker_audio, (str, Path)) or not str(speaker_audio).strip():
            raise ValueError("`speaker_audio_path` must be a non-empty local path.")
        reference_path = Path(speaker_audio).expanduser()
        if not reference_path.is_file():
            raise FileNotFoundError(f"F5-TTS reference audio was not found: {reference_path}.")
        reference_text = model_inputs.get("reference_text", "")
        if not isinstance(reference_text, str):
            raise TypeError("`reference_text` must be a string.")
        numeric_values = {
            "speed": model_inputs.get("speed", 1.0),
            "cfg_strength": model_inputs.get("cfg_strength", 2.0),
            "sway_sampling_coef": model_inputs.get("sway_sampling_coef", -1.0),
            "cross_fade_duration": model_inputs.get("cross_fade_duration", 0.15),
        }
        for name, value in numeric_values.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"`{name}` must be a finite number.")
            if not math.isfinite(float(value)):
                raise ValueError(f"`{name}` must be finite.")
        if numeric_values["speed"] <= 0:
            raise ValueError("`speed` must be greater than zero.")
        if numeric_values["cfg_strength"] < 0:
            raise ValueError("`cfg_strength` must be non-negative.")
        if numeric_values["cross_fade_duration"] < 0:
            raise ValueError("`cross_fade_duration` must be non-negative.")
        nfe_steps = model_inputs.get("nfe_steps", 32)
        if (isinstance(nfe_steps, bool) or not isinstance(nfe_steps, int) or nfe_steps <= 0):
            raise ValueError("`nfe_steps` must be a positive integer.")
        seed = model_inputs.get("seed")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise TypeError("`seed` must be an integer or None.")
        if not isinstance(model_inputs.get("remove_silence", False), bool):
            raise TypeError("`remove_silence` must be a boolean.")
        if not reference_text.strip():
            raise ValueError(
                "`reference_text` is required for native F5-TTS. Run ASR "
                "explicitly when a transcript is unavailable.")

    def _generate(
        self,
        text: str,
        *,
        speaker_audio_path: str,
        reference_text: str = "",
        output_file: str | None = None,
        speed: float = 1.0,
        seed: int | None = None,
        nfe_steps: int = 32,
        cfg_strength: float = 2.0,
        sway_sampling_coef: float = -1.0,
        cross_fade_duration: float = 0.15,
        remove_silence: bool = False,
    ) -> TTSOutput:
        if self.model is None:
            raise RuntimeError("F5-TTS must be loaded before generation.")
        waveform, sample_rate, spectrogram = self.model.infer(
            ref_file=str(Path(speaker_audio_path).expanduser()),
            ref_text=reference_text,
            gen_text=text,
            speed=speed,
            seed=seed,
            nfe_step=nfe_steps,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            cross_fade_duration=cross_fade_duration,
            remove_silence=remove_silence,
        )
        sample_rate = int(sample_rate)
        if sample_rate <= 0:
            raise ValueError("F5-TTS inference returned an invalid sample rate.")
        self.config.sample_rate = sample_rate
        return finish_audio_output(
            waveform,
            sample_rate,
            output_file=output_file,
            metadata={
                "seed": getattr(self.model, "seed", seed),
                "spectrogram": spectrogram,
                "runtime": "voicehub-native",
            },
        )

    def export_native_pretrained(
        self,
        save_directory: str | Path,
    ) -> Path:
        """Export flow weights and frontend assets for fresh inference."""
        if self.model is None:
            self.load_for_training()
        from voicehub.models.f5tts.native.checkpoint import export_f5tts_checkpoint

        destination = Path(save_directory).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        checkpoint = export_f5tts_checkpoint(
            self.model.ema_model,
            destination / "model.safetensors",
        )
        self.model.frontend.vocabulary.save(destination / "vocab.txt")
        self.config.save_pretrained(destination)
        return checkpoint


F5TTS = F5TTSForTextToSpeech

__all__ = ["F5TTS", "F5TTSConfig", "F5TTSForTextToSpeech"]
