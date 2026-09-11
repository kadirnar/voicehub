"""VoiceHub-native Inflect Micro/Nano v2 inference and training lifecycle."""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from voicehub.models._shared import finish_audio_output, seeded_inference
from voicehub.models.inflecttts.configuration import InflectTTSConfig
from voicehub.models.inflecttts.native.checkpoint import (
    export_inflect_checkpoint,
    load_inflect_checkpoint,
    load_inflect_discriminator,
    resolve_inflect_artifacts,
)
from voicehub.models.inflecttts.native.modeling import build_inflect_model
from voicehub.models.inflecttts.native.runtime import InflectV2Runtime
from voicehub.models.inflecttts.native.training import InflectV2TrainingModel
from voicehub.models.tts import PreTrainedTTSModel
from voicehub.outputs import TTSOutput


class InflectTTSForTextToSpeech(PreTrainedTTSModel):
    """Native fixed-voice Inflect v2 synthesis and VITS warm-start FT."""

    config_class = InflectTTSConfig
    default_model_name_or_path = "owensong/Inflect-Micro-v2"

    def __init__(
        self,
        config: InflectTTSConfig | str | None = None,
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
        self._native_architecture_config = None
        self._checkpoint_report = None
        self._resolved_artifacts = None
        self.training_model: InflectV2TrainingModel | None = None

    def _load_pretrained_model(self) -> None:
        artifacts = resolve_inflect_artifacts(
            self.config.name_or_path,
            checkpoint_filename=self.config.checkpoint_filename,
            revision=self.config.revision,
            cache_dir=self.config.cache_dir,
            local_files_only=self.config.local_files_only,
        )
        architecture_config = (artifacts.config.for_training() if self.is_training_load else artifacts.config)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="`torch.nn.utils.weight_norm` is deprecated",
                category=FutureWarning,
            )
            generator = build_inflect_model(architecture_config).to(self.device)
        self._checkpoint_report = load_inflect_checkpoint(
            generator,
            artifacts,
            trust_pickle_checkpoint=self.config.trust_pickle_checkpoint,
            allow_fresh_training_components=self.is_training_load,
        )
        self._resolved_artifacts = artifacts
        generator.train(self.is_training_load)
        self._native_architecture_config = architecture_config
        self.model = InflectV2Runtime(generator, architecture_config)
        self.config.sample_rate = architecture_config.sample_rate

    def _validate_training_runtime(self) -> None:
        if not self.config.enable_native_finetuning:
            raise ValueError(
                "Inflect's published checkpoint is deployable generator state, "
                "not a resumable trainer. Set `enable_native_finetuning=True` "
                "to warm-start the exact released generator, freshly "
                "initialize its VITS posterior/discriminators, and provide "
                "checkpoint-compatible phoneme IDs, linear spectrograms, "
                "lengths, and 24 kHz waveforms.")

    def _prepare_for_training(self) -> None:
        if not isinstance(self.model, InflectV2Runtime):
            raise TypeError("Inflect training requires the VoiceHub-native runtime.")
        if self._native_architecture_config is None:
            raise RuntimeError("Inflect architecture config was not loaded.")
        if self.model.generator.inference_only:
            architecture_config = self._native_architecture_config.for_training()
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="`torch.nn.utils.weight_norm` is deprecated",
                    category=FutureWarning,
                )
                training_generator = build_inflect_model(architecture_config).to(self.device)
            public_state = self.model.generator.state_dict()
            incompatible = training_generator.load_state_dict(
                public_state,
                strict=False,
            )
            expected_missing = tuple(
                sorted(name for name in training_generator.state_dict() if name.startswith("enc_q.")))
            if (tuple(sorted(incompatible.missing_keys)) != expected_missing or incompatible.unexpected_keys):
                raise RuntimeError(
                    "Inflect inference-to-training graph expansion produced "
                    "an unexpected checkpoint inventory.")
            self._native_architecture_config = architecture_config
            self.model = InflectV2Runtime(
                training_generator,
                architecture_config,
            )
        self.model.train()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="`torch.nn.utils.weight_norm` is deprecated",
                category=FutureWarning,
            )
            self.training_model = InflectV2TrainingModel(
                self.model.generator,
                self._native_architecture_config,
                enable_discriminator=self.config.training_enable_discriminator,
                loss_weights=self.config.training_loss_weights,
            )
        discriminator_path = getattr(
            self._resolved_artifacts,
            "discriminator_path",
            None,
        )
        if (discriminator_path is not None and self.training_model.discriminator is not None):
            load_inflect_discriminator(
                self.training_model.discriminator,
                discriminator_path,
            )

    def prepare_training_inputs(
        self,
        inputs: dict[str, Any],
        *,
        phase: str,
    ) -> dict[str, Any]:
        prepared = dict(inputs)
        prepared["phase"] = phase
        if "input_lengths" not in prepared:
            attention_mask = prepared.get("attention_mask")
            if attention_mask is not None:
                prepared["input_lengths"] = attention_mask.long().sum(dim=-1)
        if "spectrogram_lengths" not in prepared:
            mask = prepared.get("spectrogram_attention_mask")
            if mask is not None:
                prepared["spectrogram_lengths"] = mask.long().sum(dim=-1)
        return prepared

    def _validate_generation_inputs(
        self,
        model_inputs: dict[str, Any],
    ) -> None:
        speed = model_inputs.get("speed", 1.0)
        if (not isinstance(speed, (int, float)) or isinstance(speed, bool) or not math.isfinite(speed) or
                not 0.5 <= speed <= 2.0):
            raise ValueError("`speed` must be a finite number between 0.5 and 2.0.")
        variation = model_inputs.get("variation", 0.667)
        if (not isinstance(variation, (int, float)) or isinstance(variation, bool) or
                not math.isfinite(variation) or not 0 <= variation <= 1):
            raise ValueError("`variation` must be a finite number between 0 and 1.")
        seed = model_inputs.get("seed", 0)
        if (not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**63):
            raise ValueError("`seed` must be an integer in [0, 2**63).")
        input_ids = model_inputs.get("input_ids")
        phoneme_text = model_inputs.get("phoneme_text")
        input_is_phonemes = model_inputs.get("input_is_phonemes", False)
        if input_ids is not None and (phoneme_text is not None or input_is_phonemes):
            raise ValueError("`input_ids` cannot be combined with phoneme inputs.")

    def _generate(
        self,
        text: str,
        *,
        output_file: str | None = None,
        phoneme_text: str | None = None,
        input_ids: Sequence[int] | Any | None = None,
        input_is_phonemes: bool = False,
        speed: float = 1.0,
        variation: float = 0.667,
        seed: int = 0,
    ) -> TTSOutput:
        with seeded_inference(
                seed,
                device=self.device,
                model_type="inflecttts",
        ):
            sample_rate, audio = self.model.synthesize(
                text,
                phoneme_text=phoneme_text,
                input_ids=input_ids,
                input_is_phonemes=input_is_phonemes,
                speed=speed,
                variation=variation,
                seed=seed,
            )
        sample_rate = int(sample_rate)
        if sample_rate <= 0:
            raise RuntimeError(f"Inflect returned an invalid sample rate: {sample_rate}.")
        if audio is None:
            sample_count = 0
        elif hasattr(audio, "numel"):
            sample_count = audio.numel()
        elif hasattr(audio, "size"):
            sample_count = audio.size
        else:
            sample_count = len(audio)
        if sample_count == 0:
            raise RuntimeError("Inflect returned an empty audio waveform.")
        self.config.sample_rate = sample_rate
        return finish_audio_output(
            audio,
            sample_rate,
            output_file=output_file,
            metadata={
                "speed": speed,
                "variation": variation,
                "seed": seed,
                "frontend": ("exact-token-ids" if input_ids is not None else "explicit-phonemes"),
            },
        )

    def _save_pretrained(self, save_directory: Path) -> None:
        self.load()
        if not isinstance(self.model, InflectV2Runtime):
            raise TypeError("Inflect export requires the native runtime.")
        architecture_config = self._native_architecture_config
        if architecture_config is None:
            raise RuntimeError("Inflect architecture config is unavailable.")
        discriminator = (None if self.training_model is None else self.training_model.discriminator)
        export_inflect_checkpoint(
            self.model.generator,
            architecture_config,
            save_directory,
            discriminator=discriminator,
        )


InflectTTSModel = InflectTTSForTextToSpeech

__all__ = [
    "InflectTTSConfig",
    "InflectTTSForTextToSpeech",
    "InflectTTSModel",
]
