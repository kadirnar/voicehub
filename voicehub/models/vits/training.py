"""Fine-tuning adapter for VoiceHub's native VITS/MMS-TTS runtime."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from voicehub.training.adapters import VITSTrainingAdapter
from voicehub.training.contracts import TrainingPhaseKind, TrainingPhaseSpec, TrainingRecipeKind, TrainingSupport
from voicehub.training.datasets import SpeechDataset


def _generator_training_spec(spec: Any) -> Any:
    """Project the full shared profile into the legacy warm-start phase."""
    generator = TrainingPhaseSpec(
        name="generator",
        component_paths=("training_model", ),
        optimizer_names=("model", ),
        forward_component="training_model",
        label_names=("audio_values", ),
        prediction_keys=("waveform", "audio_values"),
        loss_keys=(
            "loss",
            "waveform_loss",
            "spectral_loss",
            "duration_loss",
            "kl_loss",
        ),
        required_inputs=("input_ids", "spectrogram", "audio_values"),
        kind=TrainingPhaseKind.GENERATOR,
    )
    return replace(
        spec,
        module_paths=("training_model", ),
        component_paths=("training_model", ),
        label_names=("audio_values", ),
        prediction_keys=("waveform", "audio_values"),
        loss_keys=generator.loss_keys,
        native_training=True,
        separate_optimizers=False,
        support=TrainingSupport.PREPROCESSED,
        phases=(generator, ),
        default_phase="generator",
        recipe_kind=TrainingRecipeKind.SINGLE_PHASE,
    )


def _adversarial_training_spec(spec: Any) -> Any:
    """Project the legacy shared profile into the exact native GAN phases."""
    discriminator = TrainingPhaseSpec(
        name="discriminator",
        component_paths=("training_model.discriminator", ),
        optimizer_names=("discriminator", ),
        forward_component="training_model",
        forward_method="discriminator_step",
        label_names=("audio_values", ),
        prediction_keys=("audio_values", ),
        loss_keys=("loss", ),
        required_inputs=("input_ids", "audio_values"),
        kind=TrainingPhaseKind.DISCRIMINATOR,
        frozen_component_paths=("training_model.native_model", ),
        optimizer_step_after_phase=True,
    )
    generator = TrainingPhaseSpec(
        name="generator",
        component_paths=("training_model.native_model", ),
        optimizer_names=("generator", ),
        forward_component="training_model",
        forward_method="generator_step",
        label_names=("audio_values", ),
        prediction_keys=("audio_values", ),
        loss_keys=("loss", ),
        required_inputs=("input_ids", "audio_values"),
        kind=TrainingPhaseKind.GENERATOR,
        frozen_component_paths=("training_model.discriminator", ),
        optimizer_step_after_phase=True,
    )
    return replace(
        spec,
        module_paths=("training_model", ),
        component_paths=(
            "training_model.native_model",
            "training_model.discriminator",
        ),
        label_names=("audio_values", ),
        prediction_keys=("audio_values", ),
        loss_keys=("loss", ),
        native_training=True,
        separate_optimizers=True,
        support=TrainingSupport.PREPROCESSED,
        phases=(discriminator, generator),
        default_phase="generator",
        recipe_kind=TrainingRecipeKind.ADVERSARIAL,
    )


class NativeVitsGeneratorTrainingAdapter(VITSTrainingAdapter):
    """Run either the compatibility warm start or complete VITS GAN recipe.

    ``enable_native_adversarial_training=True`` selects two independently
    optimized phases:

    * the discriminator receives detached generated waveforms;
    * the generator receives mel, duration, KL, feature-matching, and
      adversarial losses while discriminator parameters are frozen.

    The checkpoint's acoustic settings remain explicit because MMS-TTS model
    metadata does not contain the original FFT/mel training configuration.
    The older generator-only reconstruction path remains available for
    portable checkpoints and callers that already provide spectrograms.
    """

    supports_custom_recipe = True
    native_export_semantics = "voicehub-native-vits-generator-safetensors"

    def __init__(self, model: Any, spec: Any) -> None:
        adversarial = getattr(
            getattr(model, "config", None),
            "enable_native_adversarial_training",
            False,
        )
        spec = (_adversarial_training_spec(spec) if adversarial else _generator_training_spec(spec))
        super().__init__(model, spec)

    @property
    def adversarial_training_enabled(self) -> bool:
        return (
            getattr(
                getattr(self.model, "config", None),
                "enable_native_adversarial_training",
                False,
            ) is True)

    @property
    def generator_training_enabled(self) -> bool:
        config = getattr(self.model, "config", None)
        return (
            getattr(config, "enable_native_generator_training", False) is True or
            self.adversarial_training_enabled)

    @property
    def experimental_reconstruction_enabled(self) -> bool:
        """Compatibility spelling for the former opt-in property."""
        return self.generator_training_enabled

    def validate_support(self) -> None:
        if not self.generator_training_enabled:
            raise ValueError(
                "Native VITS training is opt-in. Set "
                "`enable_native_adversarial_training=True` with an explicit "
                "`training_acoustic_config` for the full GAN recipe, or set "
                "`enable_native_generator_training=True` for the legacy "
                "preprocessed generator-only warm-start.")
        if self.adversarial_training_enabled:
            acoustic = getattr(
                self.model.config,
                "training_acoustic_config",
                None,
            )
            if acoustic is None:
                raise ValueError(
                    "Full VITS adversarial fine-tuning requires an explicit "
                    "`training_acoustic_config`. MMS-TTS checkpoint metadata "
                    "does not publish its FFT, hop, window, mel, or segment "
                    "training settings.")
            from voicehub.models.vits.native.training import VitsAcousticConfig

            VitsAcousticConfig.from_mapping(acoustic)
        super().validate_support()

    def setup(self) -> NativeVitsGeneratorTrainingAdapter:
        super().setup()
        if self.primary_model is not getattr(self.model, "training_model", None):
            raise ValueError(
                "Native VITS training must target the wrapper's exact "
                "`training_model` objective facade.")
        native_model = getattr(self.model, "model", None)
        if getattr(self.primary_model, "native_model", None) is not native_model:
            raise ValueError(
                "Native VITS training facade is not attached to the loaded "
                "generator checkpoint.")
        if self.adversarial_training_enabled:
            discriminator = getattr(self.primary_model, "discriminator", None)
            if discriminator is None:
                raise ValueError("Full VITS training requires the native multi-period "
                                 "discriminator.")
        return self

    def create_dataset(self, records: Any, **kwargs: Any) -> SpeechDataset:
        """Create a dataset while retaining raw or preprocessed speech."""
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
            "attention_mask",
            "audio_lengths",
            "audio_values",
            "durations",
            "generator",
            "input_ids",
            "speaker_id",
            "spectrogram",
            "spectrogram_attention_mask",
        }
        return {name: value for name, value in prepared.items() if name in accepted}

    def recipe_resume_configuration(self) -> Mapping[str, Any]:
        configuration = dict(super().recipe_resume_configuration())
        if self.adversarial_training_enabled:
            configuration.update({
                "checkpoint_format":
                "native-vits-v1",
                "objective_scope":
                "source-adversarial-fine-tuning",
                "acoustic_frontend":
                dict(self.model.config.training_acoustic_config),
                "discriminator":
                "scale-plus-five-period",
                "optimizer_phases": [
                    "discriminator",
                    "generator",
                ],
                "generator_objectives": [
                    "mel-reconstruction",
                    "duration",
                    "kl",
                    "feature-matching",
                    "least-squares-adversarial",
                ],
                "full_vits_fine_tuning":
                True,
                "mms_checkpoint_acoustic_metadata_inferred":
                False,
            })
        else:
            configuration.update({
                "checkpoint_format":
                "native-vits-v1",
                "objective_scope":
                "preprocessed-generator-warm-start",
                "posterior_encoder":
                True,
                "monotonic_alignment_search":
                True,
                "duration_objective":
                True,
                "kl_objective":
                True,
                "waveform_reconstruction":
                True,
                "full_vits_fine_tuning":
                False,
                "blocking_requirements": [
                    "checkpoint-specific spectrogram and mel preprocessing",
                    "enable_native_adversarial_training",
                ],
            })
        return configuration

    def artifact_manifest(self) -> dict[str, Any]:
        manifest = super().artifact_manifest()
        manifest.update({
            "checkpoint_format":
            "native-vits-v1",
            "native_architecture_family":
            "vits",
            "training_scope": (
                "source-adversarial-fine-tuning"
                if self.adversarial_training_enabled else "preprocessed-generator-warm-start"),
            "full_vits_fine_tuning":
            self.adversarial_training_enabled,
        })
        return manifest

    def on_training_phase_end(self, context: Any, output: Any) -> Any:
        output = super().on_training_phase_end(context, output)
        output.metadata.update({
            "native_architecture_family":
            "vits",
            "objective": (
                "source-adversarial-fine-tuning"
                if self.adversarial_training_enabled else "preprocessed-generator-warm-start"),
            "full_vits_fine_tuning":
            self.adversarial_training_enabled,
            "checkpoint_acoustic_settings_explicit":
            self.adversarial_training_enabled,
        })
        return output

    def save_pretrained(self, save_directory: str | Path) -> None:
        """Export an inference-ready generator; exact resumes keep the MPD."""
        self.setup()
        destination = Path(save_directory).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        self.model.export_native_pretrained(destination)


NativeVitsAdversarialTrainingAdapter = NativeVitsGeneratorTrainingAdapter
VitsReconstructionTrainingAdapter = NativeVitsGeneratorTrainingAdapter

__all__ = [
    "NativeVitsAdversarialTrainingAdapter",
    "NativeVitsGeneratorTrainingAdapter",
    "VitsReconstructionTrainingAdapter",
]
