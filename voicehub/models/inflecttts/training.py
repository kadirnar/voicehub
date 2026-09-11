"""Trainer integration for native Inflect v2 warm-start fine-tuning."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicehub.models.inflecttts.native.training import InflectV2TrainingModel
from voicehub.training.adapters import VITSTrainingAdapter
from voicehub.training.collators import DataCollatorForAudioTraining
from voicehub.training.datasets import SpeechDataset


class InflectTTSTrainingAdapter(VITSTrainingAdapter):
    """Run separate generator and discriminator phases on preprocessed data."""

    supports_custom_recipe = True
    native_export_semantics = "inference-reloadable-inflect-v2-safetensors"

    def __init__(self, model: Any, spec: Any) -> None:
        super().__init__(model, spec)
        self.data_collator = DataCollatorForAudioTraining(
            field_schemas={
                "input_ids": {
                    "sequence_dim": 0,
                    "padding_value": 0,
                    "length_field": "input_lengths",
                },
                "spectrogram": {
                    "sequence_dim": -1,
                    "padding_value": 0.0,
                    "length_field": "spectrogram_lengths",
                },
                "audio_values": {
                    "sequence_dim": -1,
                    "padding_value": 0.0,
                },
            })

    def validate_support(self) -> None:
        if getattr(
                self.model.config,
                "enable_native_finetuning",
                False,
        ) is not True:
            raise ValueError(
                "Set `enable_native_finetuning=True` to acknowledge that "
                "Inflect fine-tuning reconstructs fresh training-only VITS "
                "components around the released deployable generator.")
        super().validate_support()

    def setup(self) -> InflectTTSTrainingAdapter:
        super().setup()
        native = getattr(self.model, "training_model", None)
        if not isinstance(native, InflectV2TrainingModel):
            raise TypeError("Inflect fine-tuning requires InflectV2TrainingModel.")
        if self.primary_model is not native:
            raise ValueError("The Inflect recipe must target wrapper.training_model.")
        native.train()
        return self

    def select_training_phase(self, training_phase=None):
        phase = super().select_training_phase(training_phase)
        if (phase.name == "discriminator" and not self.model.config.training_enable_discriminator):
            raise ValueError(
                "Inflect discriminator training is disabled by "
                "`training_enable_discriminator=False`.")
        return phase

    def plan_training_phases(self, step: int):
        phases = super().plan_training_phases(step)
        if self.model.config.training_enable_discriminator:
            return phases
        return tuple(phase for phase in phases if phase.name == "generator")

    def create_dataset(
        self,
        records: Any,
        **kwargs: Any,
    ) -> SpeechDataset:
        self.validate_support()
        return SpeechDataset(records, **kwargs)

    def prepare_training_inputs(
        self,
        inputs: Mapping[str, Any],
        context: Any,
    ) -> Mapping[str, Any]:
        prepared = self.model.prepare_training_inputs(
            dict(inputs),
            phase=context.phase.name,
        )
        accepted = {
            "audio_values",
            "input_ids",
            "input_lengths",
            "phase",
            "spectrogram",
            "spectrogram_lengths",
        }
        return {name: value for name, value in prepared.items() if name in accepted}

    def recipe_resume_configuration(self) -> Mapping[str, Any]:
        configuration = dict(super().recipe_resume_configuration())
        configuration.update({
            "checkpoint_format":
            "voicehub-native-inflect-v2",
            "objective_scope":
            "preprocessed-full-vits-warm-start",
            "public_generator_warm_start":
            True,
            "fresh_posterior_encoder":
            True,
            "fresh_multi_period_discriminator":
            bool(self.model.config.training_enable_discriminator),
            "author_optimizer_state_available":
            False,
            "author_data_pipeline_available":
            False,
        })
        return configuration

    def artifact_manifest(self) -> dict[str, Any]:
        manifest = super().artifact_manifest()
        manifest.update({
            "checkpoint_format":
            "voicehub-native-inflect-v2",
            "native_architecture_family":
            "inflect-v2",
            "training_scope":
            "preprocessed-full-vits-warm-start",
            "generator_checkpoint_compatible":
            True,
            "inference_reloadable":
            True,
            "raw_text_frontend":
            False,
            "required_preprocessing": [
                "checkpoint-compatible en-us phoneme IDs",
                "513-bin linear magnitude spectrogram",
                "24 kHz mono waveform",
            ],
            "fresh_components": [
                "posterior-encoder",
                "multi-period-discriminator",
            ],
        })
        return manifest

    def save_pretrained(self, save_directory: str | Path) -> None:
        self.setup()
        self.model.export_native_pretrained(save_directory)


__all__ = ["InflectTTSTrainingAdapter"]
