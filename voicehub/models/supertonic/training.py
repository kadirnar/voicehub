"""Trainer integration for Supertonic's published differentiable graphs."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicehub.models.supertonic.native.runtime import NativeSupertonicRuntime
from voicehub.training.adapters import FlowMatchingTrainingAdapter
from voicehub.training.datasets import SpeechDataset


class SupertonicTrainingAdapter(FlowMatchingTrainingAdapter):
    """Fine-tune released graph components from prepared supervision.

    This adapter does not claim to reproduce Supertone's unpublished
    raw-audio recipe. It trains the exact published duration, text-to-
    latent, flow-step, and vocoder graph with explicit
    duration/latent/style targets.
    """

    supports_custom_recipe = True
    native_export_semantics = "voicehub-native-supertonic-safetensors-v1"

    def validate_support(self) -> None:
        if not bool(getattr(
                self.model.config,
                "enable_preprocessed_training",
                False,
        )):
            raise ValueError(
                "Supertonic fine-tuning is an explicitly reconstructed "
                "published-graph recipe. Set "
                "`enable_preprocessed_training=True` and provide style "
                "tensors plus duration and/or latent targets.")
        super().validate_support()

    def setup(self) -> SupertonicTrainingAdapter:
        super().setup()
        if not isinstance(self.primary_model, NativeSupertonicRuntime):
            raise TypeError("Supertonic training must target NativeSupertonicRuntime.")
        if self.primary_model is not self.model.model:
            raise ValueError("Supertonic training runtime is detached from its wrapper.")
        return self

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
        return self.model.prepare_training_inputs(
            dict(inputs),
            phase=context.phase.name,
        )

    def recipe_resume_configuration(self) -> Mapping[str, Any]:
        configuration = dict(super().recipe_resume_configuration())
        configuration.update({
            "checkpoint_format":
            "voicehub-native-supertonic-v1",
            "objective_scope":
            "published-inference-graph",
            "supervision":
            "precomputed-style-duration-latent",
            "recipe_status":
            "reconstructed-not-author-verified",
            "raw_audio_recipe":
            False,
            "blocking_requirements": [
                "unpublished audio encoder",
                "unpublished style encoder training path",
                "unpublished optimizer and data recipe",
            ],
        })
        return configuration

    def artifact_manifest(self) -> dict[str, Any]:
        manifest = super().artifact_manifest()
        manifest.update({
            "checkpoint_format": "voicehub-native-supertonic-v1",
            "native_architecture_family": "supertonic-3",
            "training_scope": "published-inference-graph",
            "author_verified_full_recipe": False,
            "requires_precomputed_latents": True,
        })
        return manifest

    def save_pretrained(self, save_directory: str | Path) -> None:
        self.setup()
        destination = Path(save_directory).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        self.model._save_pretrained(destination)


__all__ = ["SupertonicTrainingAdapter"]
