"""Public StyleTTS 2 API backed by VoiceHub's native architecture."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from voicehub.models._shared import finish_audio_output, resolve_torch_dtype, seeded_inference
from voicehub.models.styletts2.configuration import StyleTTS2Config
from voicehub.models.tts import PreTrainedTTSModel
from voicehub.outputs import TTSOutput


class StyleTTS2ForTextToSpeech(PreTrainedTTSModel):
    """Style diffusion and voice cloning without provider runtimes."""

    config_class = StyleTTS2Config

    def __init__(
        self,
        config: StyleTTS2Config | str | None = None,
        *,
        model_path: str | None = None,
        device: str = "auto",
        lazy_load: bool = True,
        **config_overrides: Any,
    ) -> None:
        config = self._coerce_config(
            config,
            model_path=model_path,
            **config_overrides,
        )
        super().__init__(config, device=device, lazy_load=lazy_load)
        self.architecture_config = None
        self.training_model = None

    def _load_pretrained_model(self) -> None:
        if not self.config.name_or_path:
            raise ValueError("StyleTTS 2 requires a local native artifact or reviewed "
                             "legacy checkpoint.")
        source = Path(self.config.name_or_path).expanduser()
        if source.is_dir():
            checkpoint_path = source / "model.safetensors"
            config_path = (
                Path(self.config.config_path).expanduser() if self.config.config_path else source /
                "config.json")
        else:
            checkpoint_path = source
            config_path = (
                Path(self.config.config_path).expanduser() if self.config.config_path else
                Path(__file__).parent / "source" / "styletts2" / "Configs" / "config_libritts.yml")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"StyleTTS 2 checkpoint was not found: {checkpoint_path}.")
        if not config_path.is_file():
            raise FileNotFoundError(f"StyleTTS 2 runtime configuration was not found: {config_path}.")
        import torch

        from voicehub.models.styletts2.native.runtime import StyleTTS2Runtime

        dtype = resolve_torch_dtype(torch, self.config.dtype, self.device)
        self.model = StyleTTS2Runtime(
            checkpoint_path=str(checkpoint_path.resolve()),
            config_path=str(config_path.resolve()),
            assets_directory=self.config.assets_directory,
            device=self.device,
            language=self.config.language,
            trust_pickle_checkpoint=self.config.trust_pickle_checkpoint,
            dtype=dtype,
        )
        self.architecture_config = self.model.config
        self.config.sample_rate = self.model.sample_rate

    def _validate_training_runtime(self) -> None:
        if not self.config.enable_native_finetuning:
            raise ValueError(
                "Set `enable_native_finetuning=True` to fine-tune StyleTTS 2 "
                "with explicit phoneme IDs, monotonic alignments, normalized "
                "mels, reference mels, F0/noise targets, and waveforms.")

    def _prepare_for_training(self) -> None:
        from voicehub.models.styletts2.native.runtime import StyleTTS2Runtime
        from voicehub.models.styletts2.native.training import StyleTTS2TrainingModel

        if not isinstance(self.model, StyleTTS2Runtime):
            raise TypeError("StyleTTS 2 training requires the native runtime.")
        self.model.train()
        self.training_model = StyleTTS2TrainingModel(
            self.model.model,
            self.model.config,
            enable_discriminators=(self.config.training_enable_discriminators),
        )

    def prepare_training_inputs(
        self,
        inputs: dict[str, Any],
        *,
        phase: str,
    ) -> dict[str, Any]:
        prepared = dict(inputs)
        prepared["phase"] = phase
        return prepared

    def _validate_generation_inputs(
        self,
        model_inputs: dict[str, Any],
    ) -> None:
        speaker_audio = model_inputs.get("speaker_audio_path")
        if isinstance(speaker_audio, (str, Path)):
            if not str(speaker_audio).strip():
                raise ValueError("`speaker_audio_path` must be non-empty or None.")
            reference_path = Path(speaker_audio).expanduser()
            if not reference_path.is_file():
                raise FileNotFoundError(f"StyleTTS 2 reference audio was not found: "
                                        f"{reference_path}.")

        for name, default in (("alpha", 0.3), ("beta", 0.7)):
            value = model_inputs.get(name, default)
            if (not isinstance(value, (int, float)) or isinstance(value, bool)):
                raise TypeError(f"`{name}` must be numeric.")
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"`{name}` must be in the interval [0, 1].")

        diffusion_steps = model_inputs.get("diffusion_steps", 5)
        if (not isinstance(diffusion_steps, int) or isinstance(diffusion_steps, bool) or diffusion_steps < 2):
            raise ValueError("`diffusion_steps` must be an integer >= 2.")
        embedding_scale = model_inputs.get("embedding_scale", 1.0)
        if (not isinstance(embedding_scale, (int, float)) or isinstance(embedding_scale, bool) or
                not math.isfinite(embedding_scale) or embedding_scale <= 0):
            raise ValueError("`embedding_scale` must be a finite positive number.")
        seed = model_inputs.get("seed")
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
            raise TypeError("`seed` must be an integer or None.")
        text_is_phonemes = model_inputs.get("text_is_phonemes", False)
        if not isinstance(text_is_phonemes, bool):
            raise TypeError("`text_is_phonemes` must be a boolean.")
        if (model_inputs.get("input_ids") is None and text_is_phonemes is not True):
            raise ValueError(
                "Native StyleTTS 2 requires explicit phonemes. Set "
                "`text_is_phonemes=True` or pass `input_ids`.")
        # Whether reference audio is required depends on the checkpoint
        # profile and is checked by the typed runtime after loading.

    def _generate(
        self,
        text: str,
        *,
        output_file: str | None = None,
        speaker_audio_path: Any | None = None,
        alpha: float = 0.3,
        beta: float = 0.7,
        diffusion_steps: int = 5,
        embedding_scale: float = 1.0,
        seed: int | None = None,
        input_ids: Any = None,
        text_is_phonemes: bool = False,
    ) -> TTSOutput:
        with seeded_inference(
                seed,
                device=self.device,
                model_type="styletts2",
        ) as effective_seed:
            audio = self.model.generate(
                text,
                speaker_audio_path=speaker_audio_path,
                alpha=alpha,
                beta=beta,
                diffusion_steps=diffusion_steps,
                embedding_scale=embedding_scale,
                seed=effective_seed,
                input_ids=input_ids,
                text_is_phonemes=text_is_phonemes,
            )
        return finish_audio_output(
            audio,
            self.sample_rate,
            output_file=output_file,
            metadata={
                "seed": effective_seed,
                "voice_cloned": speaker_audio_path is not None,
                "diffusion_steps": diffusion_steps,
                "frontend": ("pretokenized-ids" if input_ids is not None else "explicit-phonemes"),
            },
        )

    def create_native_training_model(
        self,
        *,
        enable_discriminators: bool = True,
        loss_weights: Any = None,
    ):
        self.load()
        from voicehub.models.styletts2.native.training import StyleTTS2TrainingModel

        self.training_model = StyleTTS2TrainingModel(
            self.model.model,
            self.model.config,
            enable_discriminators=enable_discriminators,
            loss_weights=loss_weights,
        )
        return self.training_model

    def _save_pretrained(self, save_directory: Path) -> None:
        self.load()
        from voicehub.models.styletts2.native.checkpoint import save_styletts2_pretrained

        save_styletts2_pretrained(
            self.model.model,
            self.model.config,
            save_directory,
        )


StyleTTS2 = StyleTTS2ForTextToSpeech

__all__ = [
    "StyleTTS2",
    "StyleTTS2Config",
    "StyleTTS2ForTextToSpeech",
]
