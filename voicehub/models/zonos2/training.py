"""Differentiable fine-tuning adapter for the native ZONOS2 graph."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from voicehub.models.zonos2.native.checkpoint import save_zonos2_pretrained
from voicehub.models.zonos2.native.modeling import Zonos2ForCausalLM
from voicehub.outputs import SpeechTrainingOutput
from voicehub.training.adapters import CausalLMTrainingAdapter
from voicehub.training.contracts import TrainingContext


class Zonos2TrainingAdapter(CausalLMTrainingAdapter):
    """Fine-tune the released model graph with a reconstructed causal loss.

    Zyphra has not published the original data pipeline, optimizer
    recipe, or training loop. This adapter therefore owns only the
    verifiable boundary: source-shaped delayed DAC codebooks, full-model
    gradients, strict Safetensors save/reload, and frozen DAC/speaker
    preprocessing.
    """

    supports_custom_recipe = True
    native_export_semantics = "inference-reloadable-safetensors"
    RECIPE_VERSION = 1

    def _native_model(self) -> Zonos2ForCausalLM | None:
        candidate = self.primary_model
        return (candidate if isinstance(candidate, Zonos2ForCausalLM) else None)

    def setup(self):
        super().setup()
        native = self._native_model()
        if native is None:
            raise TypeError("ZONOS2 fine-tuning requires the VoiceHub-native "
                            "Zonos2ForCausalLM graph.")
        native.train()
        return self

    def prepare_training_inputs(
        self,
        inputs: Mapping[str, Any],
        context: TrainingContext,
    ) -> Mapping[str, Any]:
        prepared = dict(super().prepare_training_inputs(inputs, context))
        input_ids = prepared.get("input_ids")
        labels = prepared.get("labels")
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 3:
            raise ValueError("ZONOS2 input_ids must have shape [batch, time, streams].")
        if not isinstance(labels, torch.Tensor) or labels.ndim != 3:
            raise ValueError("ZONOS2 labels must have shape [batch, time, codebooks].")
        native = self._native_model()
        config = native.config
        if input_ids.shape[-1] != config.frame_width:
            raise ValueError(
                f"ZONOS2 input has {input_ids.shape[-1]} streams; expected "
                f"{config.frame_width}.")
        if labels.shape != (
                input_ids.shape[0],
                input_ids.shape[1],
                config.n_codebooks,
        ):
            raise ValueError(
                "ZONOS2 labels must align with input batch/time and contain "
                f"{config.n_codebooks} codebooks.")
        if input_ids.dtype == torch.bool or input_ids.is_floating_point():
            raise TypeError("ZONOS2 input_ids must use an integer dtype.")
        if labels.dtype == torch.bool or labels.is_floating_point():
            raise TypeError("ZONOS2 labels must use an integer dtype.")
        return prepared

    def on_training_phase_end(
        self,
        context: TrainingContext,
        output: SpeechTrainingOutput,
    ) -> SpeechTrainingOutput:
        output.metadata.update({
            "objective": "reconstructed-delayed-codebook-causal-ce",
            "objective_author_verified": False,
            "codec": "frozen-descript-dac-44khz",
            "speaker_encoder": "frozen-ecapa-tdnn",
            "full_model_gradient_ready": True,
        })
        return output

    def save_pretrained(self, save_directory: str | Path) -> None:
        """Write a strict artifact loadable by fresh native inference."""
        self.setup()
        native = self._native_model()
        save_zonos2_pretrained(native, save_directory)

    def artifact_manifest(self) -> dict[str, Any]:
        manifest = super().artifact_manifest()
        manifest.update({
            "checkpoint_format": "voicehub-zonos2-v1",
            "native_architecture_family": "zonos2-sonic-moe",
            "objective": "reconstructed-delayed-codebook-causal-ce",
            "objective_author_verified": False,
            "training_scope": "full-acoustic-language-model",
            "frozen_components": [
                "descript-dac-44khz",
                "ecapa-speaker-encoder",
            ],
            "raw_audio_preprocessing": "optional-frozen-codec",
            "inference_reloadable": True,
        })
        return manifest


__all__ = ["Zonos2TrainingAdapter"]
