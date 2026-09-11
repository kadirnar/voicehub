"""Trainer adapter for native staged GPT-SoVITS classic-S2 fine-tuning."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicehub.models.gptsovits.native.training import GPTSoVITSStagedTrainingModel
from voicehub.training.adapters import CompositeTrainingAdapter
from voicehub.training.collators import DataCollatorForAudioTraining
from voicehub.training.datasets import SpeechDataset


class GPTSoVITSTrainingAdapter(CompositeTrainingAdapter):
    """Route independently optimized S1 and S2 phases."""

    supports_custom_recipe = True
    native_export_semantics = "inference-reloadable-staged-safetensors"

    def __init__(self, model: Any, spec: Any) -> None:
        super().__init__(model, spec)
        self.data_collator = DataCollatorForAudioTraining(
            field_schemas={
                "phoneme_ids": {
                    "sequence_dim": -1,
                    "padding_value": 0,
                    "length_field": "phoneme_lengths",
                },
                "semantic_ids": {
                    "sequence_dim": -1,
                    "padding_value": 0,
                    "length_field": "semantic_lengths",
                },
                "bert_features": {
                    "sequence_dim": -1,
                    "padding_value": 0.0,
                },
                "ssl_features": {
                    "sequence_dim": -1,
                    "padding_value": 0.0,
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
                "speaker_embedding": {
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
                "Set `enable_native_finetuning=True` for staged GPT-SoVITS "
                "fine-tuning from checkpoint-compatible prepared tensors.")
        super().validate_support()

    def setup(self) -> GPTSoVITSTrainingAdapter:
        super().setup()
        native = getattr(self.model, "training_model", None)
        if not isinstance(native, GPTSoVITSStagedTrainingModel):
            raise TypeError("GPT-SoVITS training requires GPTSoVITSStagedTrainingModel.")
        if self.primary_model is not native:
            raise ValueError("The GPT-SoVITS recipe must target wrapper.training_model.")
        native.train()
        return self

    def create_dataset(
        self,
        records: Any,
        **kwargs: Any,
    ) -> SpeechDataset:
        self.validate_support()
        return SpeechDataset(records, **kwargs)

    def select_training_phase(self, training_phase=None):
        phase = super().select_training_phase(training_phase)
        if (phase.name == "s2_discriminator" and not self.model.config.training_enable_s2_discriminator):
            raise ValueError(
                "GPT-SoVITS S2 discriminator training is disabled by "
                "`training_enable_s2_discriminator=False`.")
        return phase

    def plan_training_phases(self, step: int):
        phases = super().plan_training_phases(step)
        if self.model.config.training_enable_s2_discriminator:
            return phases
        return tuple(phase for phase in phases if phase.name != "s2_discriminator")

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
            "s1": {
                "phoneme_ids",
                "phoneme_lengths",
                "semantic_ids",
                "semantic_lengths",
                "bert_features",
            },
            "s2_generator": {
                "ssl_features",
                "spectrogram",
                "spectrogram_lengths",
                "audio_values",
                "phoneme_ids",
                "phoneme_lengths",
                "speaker_embedding",
            },
            "s2_discriminator": {
                "ssl_features",
                "spectrogram",
                "spectrogram_lengths",
                "audio_values",
                "phoneme_ids",
                "phoneme_lengths",
                "speaker_embedding",
            },
        }
        try:
            names = accepted[context.phase.name]
        except KeyError as error:
            raise ValueError(f"Unsupported GPT-SoVITS phase {context.phase.name!r}.") from error
        return {name: value for name, value in prepared.items() if name in names}

    def recipe_resume_configuration(self) -> Mapping[str, Any]:
        configuration = dict(super().recipe_resume_configuration())
        discriminator_enabled = bool(self.model.config.training_enable_s2_discriminator)
        variant = getattr(self.model.config, "variant", "v2")
        configuration.update({
            "checkpoint_format":
            "voicehub-native-gpt-sovits",
            "variant":
            variant,
            "stages": [
                "s1",
                "s2_generator",
                *(["s2_discriminator"] if discriminator_enabled else []),
            ],
            "s1_objective":
            "sum-cross-entropy",
            "s2_objective": (
                "vits-lsgan-mel-kl-commitment"
                if discriminator_enabled else "vits-mel-kl-commitment-without-adversarial"),
            "s2_discriminator_enabled":
            discriminator_enabled,
            "author_optimizer_state_available":
            False,
            "raw_frontend_available":
            False,
        })
        return configuration

    def artifact_manifest(self) -> dict[str, Any]:
        manifest = super().artifact_manifest()
        discriminator_enabled = bool(self.model.config.training_enable_s2_discriminator)
        variant = getattr(self.model.config, "variant", "v2")
        required_preprocessing = [
            ("512-entry V1 S1 phoneme IDs" if variant == "v1" else "732-entry V2-family S1 phoneme IDs"),
            ("322-entry V1 S2 phoneme IDs" if variant == "v1" else "732-entry V2-family S2 phoneme IDs"),
            "1,024-dimensional Chinese-RoBERTa features for S1",
            "768-dimensional CNHubert SSL features for S2",
            "1,025-bin linear magnitude spectrogram",
            "32 kHz mono waveform",
        ]
        if variant in {"v2Pro", "v2ProPlus"}:
            required_preprocessing.append("20,480-dimensional ERes2NetV2 speaker-verification embedding")
        manifest.update({
            "checkpoint_format":
            "voicehub-native-gpt-sovits",
            "native_architecture_family":
            "gpt-sovits-classic-s2",
            "variant":
            variant,
            "training_scope": (
                "source-faithful-preprocessed-s1-and-s2"
                if discriminator_enabled else "preprocessed-s1-and-non-adversarial-s2"),
            "inference_reloadable":
            True,
            "supported_variants": [
                "v1",
                "v2",
                "v2Pro",
                "v2ProPlus",
            ],
            "unsupported_variants": [
                "v3",
                "v4",
                "LoRA",
            ],
            "required_preprocessing":
            required_preprocessing,
        })
        return manifest

    def save_pretrained(self, save_directory: str | Path) -> None:
        self.setup()
        self.model.export_native_pretrained(save_directory)


__all__ = ["GPTSoVITSTrainingAdapter"]
