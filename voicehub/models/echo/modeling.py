"""Echo-TTS integration using source included in VoiceHub."""

from __future__ import annotations

import math
from functools import partial
from pathlib import Path
from typing import Any

from voicehub.models._shared import finish_audio_output, seeded_inference, validate_local_file, validate_seed
from voicehub.models.tts import PreTrainedTTSModel
from voicehub.outputs import TTSOutput

from .configuration import EchoTTSConfig


class EchoTTSForTextToSpeech(PreTrainedTTSModel):
    """Speaker-conditioned Echo-TTS with local architecture source."""

    config_class = EchoTTSConfig
    default_model_name_or_path = "jordand/echo-tts-base"

    def __init__(
        self,
        config: EchoTTSConfig | str | Path | None = None,
        *,
        model_path: str | Path | None = None,
        device: str = "auto",
        lazy_load: bool = True,
        **config_overrides,
    ):
        config = self._coerce_config(
            config,
            model_path=model_path,
            **config_overrides,
        )
        self.fish_ae = None
        self.pca_state = None
        self._loaded_for_training = False
        super().__init__(config, device=device, lazy_load=lazy_load)

    def _build_runtime_components(self):
        from voicehub.models.echo.sampling import load_fish_ae_from_hf, load_model_from_hf, load_pca_state_from_hf

        model = load_model_from_hf(
            repo_id=self.config.name_or_path,
            device=self.device,
            compile=(self.config.compile_model and not self.is_training_load),
            delete_blockwise_modules=not self.is_training_load,
        )
        fish_ae = load_fish_ae_from_hf(
            repo_id=self.config.codec_name_or_path,
            device=self.device,
        )
        pca_name_or_path = self.config.pca_name_or_path
        if pca_name_or_path is None:
            model_source = Path(self.config.name_or_path).expanduser()
            pca_name_or_path = (model_source.parent if model_source.is_file() else self.config.name_or_path)
        pca_state = load_pca_state_from_hf(
            repo_id=pca_name_or_path,
            device=self.device,
        )
        if model is None or fish_ae is None or pca_state is None:
            raise RuntimeError("Echo-TTS loader returned an incomplete inference runtime.")
        return model, fish_ae, pca_state

    def _load_pretrained_model(self) -> None:
        model, fish_ae, pca_state = self._build_runtime_components()
        sample_rate = int(getattr(fish_ae, "sample_rate", 0))
        if sample_rate <= 0:
            raise ValueError("The loaded Echo-TTS codec reported an invalid sample rate.")
        self.model = model
        self.fish_ae = fish_ae
        self.pca_state = pca_state
        self.config.sample_rate = sample_rate
        self._loaded_for_training = self.is_training_load

    def _prepare_for_training(self) -> None:
        if self._loaded_for_training:
            if self.model is not None and hasattr(self.model, "train"):
                self.model.train()
            return
        previous_state = (
            self.model,
            self.fish_ae,
            self.pca_state,
            self._loaded_for_training,
        )
        self.model = None
        self.fish_ae = None
        self.pca_state = None
        previous_loading_mode = self._loading_for_training
        self._loading_for_training = True
        try:
            self.load()
        except BaseException:
            (
                self.model,
                self.fish_ae,
                self.pca_state,
                self._loaded_for_training,
            ) = previous_state
            raise
        finally:
            self._loading_for_training = previous_loading_mode

    def _prepare_for_inference(self) -> None:
        """Put both the flow model and codec in serving mode."""
        if self.model is not None and hasattr(self.model, "eval"):
            self.model.eval()
        codec = getattr(self.fish_ae, "model", self.fish_ae)
        if codec is not None and hasattr(codec, "eval"):
            codec.eval()

    def _validate_generation_inputs(self, model_inputs: dict[str, Any]) -> None:
        super()._validate_generation_inputs(model_inputs)
        speaker_path = validate_local_file(
            model_inputs.get("speaker_audio_path"),
            option_name="speaker_audio_path",
        )
        if speaker_path is not None:
            model_inputs["speaker_audio_path"] = str(speaker_path)
        num_steps = model_inputs.get("num_steps", 40)
        sequence_length = model_inputs.get("sequence_length", 640)
        for name, value in (
            ("num_steps", num_steps),
            ("sequence_length", sequence_length),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"`{name}` must be a positive integer.")

        numeric_values = {
            "cfg_scale_text": model_inputs.get("cfg_scale_text", 3.0),
            "cfg_scale_speaker": model_inputs.get("cfg_scale_speaker", 8.0),
            "cfg_min_t": model_inputs.get("cfg_min_t", 0.5),
            "cfg_max_t": model_inputs.get("cfg_max_t", 1.0),
        }
        truncation_factor = model_inputs.get("truncation_factor")
        if truncation_factor is not None:
            numeric_values["truncation_factor"] = truncation_factor
        for name, value in numeric_values.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"`{name}` must be a finite number.")
            if not math.isfinite(float(value)):
                raise ValueError(f"`{name}` must be finite.")

        if numeric_values["cfg_scale_text"] < 0:
            raise ValueError("`cfg_scale_text` must be non-negative.")
        if numeric_values["cfg_scale_speaker"] < 0:
            raise ValueError("`cfg_scale_speaker` must be non-negative.")
        cfg_min_t = numeric_values["cfg_min_t"]
        cfg_max_t = numeric_values["cfg_max_t"]
        if not 0 <= cfg_min_t <= cfg_max_t <= 1:
            raise ValueError("`cfg_min_t` and `cfg_max_t` must satisfy "
                             "0 <= cfg_min_t <= cfg_max_t <= 1.")
        if (truncation_factor is not None and numeric_values["truncation_factor"] < 0):
            raise ValueError("`truncation_factor` must be non-negative.")

        for name in ("seed", "rng_seed"):
            validate_seed(
                model_inputs.get(name),
                option_name=name,
            )

    @staticmethod
    def _build_sample_function(
        sample_euler_cfg_independent_guidances,
        *,
        num_steps: int,
        cfg_scale_text: float,
        cfg_scale_speaker: float,
        cfg_min_t: float,
        cfg_max_t: float,
        truncation_factor: float | None,
        sequence_length: int,
    ):
        return partial(
            sample_euler_cfg_independent_guidances,
            num_steps=num_steps,
            cfg_scale_text=cfg_scale_text,
            cfg_scale_speaker=cfg_scale_speaker,
            cfg_min_t=cfg_min_t,
            cfg_max_t=cfg_max_t,
            truncation_factor=truncation_factor,
            rescale_k=None,
            rescale_sigma=None,
            speaker_kv_scale=None,
            speaker_kv_max_layers=None,
            speaker_kv_min_t=None,
            sequence_length=sequence_length,
        )

    def _generate(
        self,
        text: str,
        *,
        output_file: str | None = None,
        speaker_audio_path: str | None = None,
        num_steps: int = 40,
        cfg_scale_text: float = 3.0,
        cfg_scale_speaker: float = 8.0,
        cfg_min_t: float = 0.5,
        cfg_max_t: float = 1.0,
        sequence_length: int = 640,
        seed: int = 0,
        rng_seed: int | None = None,
        truncation_factor: float | None = None,
    ) -> TTSOutput:
        if self.model is None or self.fish_ae is None or self.pca_state is None:
            raise RuntimeError("Echo-TTS must be loaded before generation.")
        from voicehub.models.echo.sampling import load_audio, sample_euler_cfg_independent_guidances, sample_pipeline

        requested_seed = seed if rng_seed is None else rng_seed
        with seeded_inference(
                requested_seed,
                device=self.device,
                model_type="echo",
        ) as effective_seed:
            speaker_audio = None
            if speaker_audio_path:
                speaker_audio = load_audio(speaker_audio_path).to(self.device)
            sample_function = self._build_sample_function(
                sample_euler_cfg_independent_guidances,
                num_steps=num_steps,
                cfg_scale_text=cfg_scale_text,
                cfg_scale_speaker=cfg_scale_speaker,
                cfg_min_t=cfg_min_t,
                cfg_max_t=cfg_max_t,
                truncation_factor=truncation_factor,
                sequence_length=sequence_length,
            )
            audio, normalized_text = sample_pipeline(
                model=self.model,
                fish_ae=self.fish_ae,
                pca_state=self.pca_state,
                sample_fn=sample_function,
                text_prompt=text,
                speaker_audio=speaker_audio,
                rng_seed=effective_seed,
            )
        if not hasattr(audio, "__len__") or len(audio) == 0:
            raise RuntimeError("Echo-TTS generation returned no audio waveform.")
        return finish_audio_output(
            audio[0],
            self.sample_rate,
            output_file=output_file,
            metadata={
                "seed": effective_seed,
                "requested_seed": requested_seed,
                "normalized_text": normalized_text,
            },
        )


EchoTTS = EchoTTSForTextToSpeech
