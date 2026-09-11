"""Fine-tuning adapter for VoiceHub-native SenseVoiceSmall."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicehub.training.adapters import CTCTrainingAdapter


class NativeSenseVoiceTrainingAdapter(CTCTrainingAdapter):
    """Train the published CTC plus four-query rich-control objective."""

    supports_custom_recipe = True
    native_export_semantics = "voicehub-native-sensevoice-safetensors"

    def setup(self) -> NativeSenseVoiceTrainingAdapter:
        super().setup()
        if getattr(self.model, "architecture_family", None) != "ctc":
            raise ValueError("Native SenseVoice fine-tuning requires the CTC runtime.")
        if self.primary_model is not getattr(self.model, "model", None):
            raise ValueError(
                "Native SenseVoice fine-tuning must target the wrapper's "
                "exact `model` graph.")
        return self

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
            "feature_lengths",
            "features",
            "label_lengths",
            "labels",
        }
        return {name: value for name, value in prepared.items() if name in accepted}

    def recipe_resume_configuration(self) -> Mapping[str, Any]:
        values = dict(super().recipe_resume_configuration())
        config = getattr(self.model, "native_config", None)
        values.update({
            "checkpoint_format": "voicehub-sensevoice-small-v1",
            "gradient_clip_norm": getattr(
                config,
                "gradient_clip_norm",
                5.0,
            ),
            "learning_rate": getattr(config, "learning_rate", 0.00002),
            "objective": "ctc-plus-rich-control-ce",
            "optimizer": getattr(config, "optimizer", "adamw"),
            "sample_rate": getattr(config, "sampling_rate", 16_000),
            "scheduler": "warmuplr",
            "warmup_steps": getattr(config, "warmup_steps", 25_000),
        })
        return values

    def create_optimizer(
        self,
        name: str,
        parameters: list[tuple[str, Any]],
        training_args: Any,
    ):
        import torch

        if name not in {"default", "model"}:
            raise ValueError("Native SenseVoice declares only the `model` optimizer, "
                             f"found {name!r}.")
        trainable = [parameter for _, parameter in parameters if parameter.requires_grad]
        if not trainable:
            raise ValueError("Native SenseVoice has no trainable parameters.")
        return torch.optim.AdamW(
            trainable,
            lr=training_args.learning_rate,
            betas=(
                training_args.adam_beta1,
                training_args.adam_beta2,
            ),
            eps=training_args.adam_epsilon,
            weight_decay=training_args.weight_decay,
        )

    def create_scheduler(
        self,
        name: str,
        optimizer: Any,
        num_training_steps: int,
        training_args: Any,
    ):
        import torch

        del num_training_steps
        if name not in {"default", "model"}:
            raise ValueError("Native SenseVoice declares only the `model` scheduler, "
                             f"found {name!r}.")
        config = getattr(self.model, "native_config", None)
        warmup_steps = (
            training_args.warmup_steps if training_args.warmup_steps > 0 else int(
                getattr(config, "warmup_steps", 25_000)))
        if warmup_steps <= 0:
            raise ValueError("SenseVoice WarmupLR requires warmup steps.")
        scale = warmup_steps**0.5

        def schedule(current_step: int) -> float:
            step = current_step + 1
            return scale * min(
                step**-0.5,
                step * warmup_steps**-1.5,
            )

        return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)

    def artifact_manifest(self) -> dict[str, Any]:
        manifest = super().artifact_manifest()
        manifest.update({
            "checkpoint_format": "voicehub-sensevoice-small-v1",
            "native_architecture_family": "sensevoice-small",
            "native_objective": "ctc-plus-rich-control-ce",
            "processor_runtime": "voicehub-native",
        })
        return manifest

    def execute_training_phase(self, context: Any):
        output = super().execute_training_phase(context)
        output.metadata.update({
            "native_architecture_family": "sensevoice-small",
            "native_objective": "ctc-plus-rich-control-ce",
        })
        return output

    def save_pretrained(self, save_directory: str | Path) -> None:
        self.setup()
        self.model.export_native_pretrained(save_directory)


__all__ = ["NativeSenseVoiceTrainingAdapter"]
