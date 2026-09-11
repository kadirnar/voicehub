"""Trainer integration for native MeloTTS feature-level fine-tuning."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from voicehub.models.melotts.native.training import MeloTTSTrainingCollator, MeloTTSTrainingModel
from voicehub.training.adapters import VITSTrainingAdapter
from voicehub.training.datasets import SpeechDataset


class MeloTTSTrainingAdapter(VITSTrainingAdapter):
    """Route VoiceHub phases to the native generator and discriminators."""

    supports_custom_recipe = True
    native_export_semantics = "inference-reloadable-melotts-safetensors"

    def __init__(self, model: Any, spec: Any) -> None:
        super().__init__(model, spec)
        self.data_collator = MeloTTSTrainingCollator()

    def validate_support(self) -> None:
        if (getattr(
                self.model.config,
                "enable_native_finetuning",
                False,
        ) is not True):
            raise ValueError(
                "Set `enable_native_finetuning=True` to acknowledge the "
                "explicit MeloTTS phone/tone/language/BERT feature boundary.")
        super().validate_support()

    def setup(self) -> MeloTTSTrainingAdapter:
        super().setup()
        native = getattr(self.model, "training_model", None)
        if not isinstance(native, MeloTTSTrainingModel):
            raise TypeError("MeloTTS fine-tuning requires MeloTTSTrainingModel.")
        if self.primary_model is not native:
            raise ValueError("MeloTTS recipe must target wrapper.training_model.")
        native.train()
        return self

    def select_training_phase(self, training_phase=None):
        phase = super().select_training_phase(training_phase)
        if (phase.name in {"discriminator", "duration_discriminator"} and
                not self.model.config.training_enable_discriminators):
            raise ValueError(
                "MeloTTS discriminator training is disabled by "
                "`training_enable_discriminators=False`.")
        native = getattr(self.model, "training_model", None)
        if (phase.name == "duration_discriminator" and isinstance(native, MeloTTSTrainingModel) and
                native.duration_discriminator is None):
            raise ValueError("This MeloTTS architecture config disables its duration "
                             "discriminator.")
        return phase

    def plan_training_phases(self, step: int):
        phases = super().plan_training_phases(step)
        if not self.model.config.training_enable_discriminators:
            return tuple(phase for phase in phases if phase.name == "generator")
        native = getattr(self.model, "training_model", None)
        if (isinstance(native, MeloTTSTrainingModel) and native.duration_discriminator is None):
            return tuple(phase for phase in phases if phase.name != "duration_discriminator")
        return phases

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
        native = getattr(self.model, "training_model", None)
        if not isinstance(native, MeloTTSTrainingModel):
            raise TypeError("MeloTTS fine-tuning requires MeloTTSTrainingModel.")
        native.set_step(context.step)
        accepted = {
            *MeloTTSTrainingCollator.REQUIRED_FIELDS,
            "input_lengths",
            "spectrogram_lengths",
            "audio_lengths",
            "speaker_ids",
        }
        prepared = {name: value for name, value in inputs.items() if name in accepted}
        prepared.pop("speaker_id", None)
        prepared["phase"] = context.phase.name
        return prepared

    def recipe_resume_configuration(self) -> Mapping[str, Any]:
        configuration = dict(super().recipe_resume_configuration())
        configuration.update({
            "native_export_semantics": self.native_export_semantics,
            "preprocessing_boundary": ("phones-tones-language-bert-spectrogram-waveform"),
            "discriminators_exported": False,
        })
        return configuration


__all__ = [
    "MeloTTSTrainingAdapter",
    "MeloTTSTrainingCollator",
]
