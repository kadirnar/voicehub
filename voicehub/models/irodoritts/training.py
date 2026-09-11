"""Trainer integration for the VoiceHub-owned Irodori-TTS runtime."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicehub.outputs import TTSTrainingOutput
from voicehub.training.adapters import FlowMatchingTrainingAdapter
from voicehub.training.contracts import TrainingContext
from voicehub.training.datasets import SpeechDataset


class NativeIrodoriTrainingAdapter(FlowMatchingTrainingAdapter):
    """Full-model RF-DiT fine-tuning with raw or pre-encoded audio data."""

    supports_custom_recipe = True
    native_export_semantics = "inference-reloadable-native-irodoritts-safetensors"

    def setup(self) -> NativeIrodoriTrainingAdapter:
        super().setup()
        from voicehub.models.irodoritts.native.modeling import TextToLatentRFDiT

        runtime = getattr(self.model, "model", None)
        native = getattr(runtime, "model", None)
        if not isinstance(native, TextToLatentRFDiT):
            raise TypeError("Irodori fine-tuning requires the native RF-DiT graph.")
        if self.primary_model is not native:
            raise ValueError("Irodori fine-tuning must target wrapper.model.model exactly.")
        for parameter in native.parameters():
            parameter.requires_grad_(True)
        native.set_gradient_checkpointing(bool(self.model.config.training_gradient_checkpointing))
        native.train()
        codec_model = getattr(getattr(runtime, "codec", None), "model", None)
        if codec_model is not None:
            codec_model.eval()
            for parameter in codec_model.parameters():
                parameter.requires_grad_(False)
        return self

    def create_dataset(
        self,
        records: Any,
        **kwargs: Any,
    ) -> SpeechDataset:
        return SpeechDataset(records, **kwargs)

    def prepare_training_inputs(
        self,
        inputs: Mapping[str, Any],
        context: TrainingContext,
    ) -> Mapping[str, Any]:
        return self.model.prepare_training_inputs(
            dict(inputs),
            phase=context.phase.name,
        )

    def execute_training_phase(
        self,
        context: TrainingContext,
    ) -> TTSTrainingOutput:
        self.setup()
        if not isinstance(context, TrainingContext):
            raise TypeError("Irodori training requires a TrainingContext.")
        prepared = self.prepare_training_inputs(context.inputs, context)
        prepared = self.prepare_runtime_inputs(prepared)
        from voicehub.models.irodoritts.native.training import irodori_training_step

        objective = self.model.config.training_objective
        native = self.primary_model
        if objective in {"duration", "joint"} and not native.cfg.use_duration_predictor:
            if objective == "duration":
                raise ValueError("This Irodori checkpoint has no duration predictor.")
            objective = "flow"
        outputs = irodori_training_step(
            native,
            prepared,
            objective=objective,
            rf_loss_mode=self.model.config.training_rf_loss_mode,
            duration_loss_weight=self.model.config.training_duration_loss_weight,
            duration_huber_delta=self.model.config.training_duration_huber_delta,
        )
        losses = {name: value for name, value in outputs.items() if name.endswith("_loss") or name == "loss"}
        return TTSTrainingOutput(
            loss=outputs["loss"],
            predictions=outputs.get(
                "velocity",
                outputs.get("duration_prediction"),
            ),
            losses=losses,
            training_phase=context.phase.name,
            optimizer_names=context.phase.optimizer_names,
            metadata={
                "checkpoint_format": "voicehub-native-irodoritts-v1",
                "fine_tuning_scope": "all-irodori-model-parameters",
                "native_architecture_family": "irodoritts-rf-dit",
                "native_objective": objective,
                "raw_audio_supported": True,
                "preencoded_latents_supported": True,
                "codec_trainable": False,
            },
        )

    def recipe_resume_configuration(self) -> Mapping[str, Any]:
        values = dict(super().recipe_resume_configuration())
        values.update({
            "checkpoint_format": "voicehub-native-irodoritts-v1",
            "duration_huber_delta": self.model.config.training_duration_huber_delta,
            "duration_loss_weight": self.model.config.training_duration_loss_weight,
            "full_model": True,
            "gradient_checkpointing": self.model.config.training_gradient_checkpointing,
            "objective": self.model.config.training_objective,
            "optimizer": "adamw",
            "rf_loss_mode": self.model.config.training_rf_loss_mode,
            "sample_rate": 48_000,
            "source_recipe_revision": "eaf74d6a19138f743acb5b71a445fd25a57db987",
        })
        return values

    def artifact_manifest(self) -> dict[str, Any]:
        manifest = super().artifact_manifest()
        manifest.update({
            "checkpoint_format": "voicehub-native-irodoritts-v1",
            "fine_tuning_scope": "all-irodori-model-parameters",
            "inference_reloadable": True,
            "native_architecture_family": "irodoritts-rf-dit",
            "raw_audio_supported": True,
            "preencoded_latents_supported": True,
            "semantic_codec": {
                "fine_tuned": False,
                "reason": "fixed training target and inference decoder",
            },
        })
        return manifest

    def save_pretrained(self, save_directory: str | Path) -> None:
        self.setup()
        self.model.export_native_pretrained(save_directory)


__all__ = ["NativeIrodoriTrainingAdapter"]
