"""Source-faithful full and LoRA fine-tuning for native VoxCPM2."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from voicehub.training.adapters import FlowMatchingTrainingAdapter
from voicehub.training.datasets import SpeechDataset

_FORWARD_INPUTS = frozenset({
    "audio_feats",
    "audio_mask",
    "generator",
    "labels",
    "loss_mask",
    "position_ids",
    "progress",
    "sample_generate",
    "text_mask",
    "text_tokens",
})


@dataclass(frozen=True)
class VoxCPMTrainingCollator:
    """Preserve raw records until the frozen codec can process the batch."""

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not features:
            return {}
        if any(not isinstance(record, Mapping) for record in features):
            raise TypeError("Every VoxCPM training sample must be a mapping.")
        return {"records": [dict(record) for record in features]}

    def resume_fingerprint(self) -> dict[str, Any]:
        return {
            "type": "voxcpm2-source-records-v1",
            "packing": "[text,101,audio,102]-or-reference-layout",
            "audio_codec": "frozen-audiovae-v2",
        }


class VoxCPMTrainingAdapter(FlowMatchingTrainingAdapter):
    """Optimize the published CFM and stop-token objectives.

    AudioVAE V2 is always frozen. ``training_lora_config=None`` performs
    full-model fine-tuning; a mapping injects the exact official LoRA
    target topology and leaves only adapter parameters trainable.
    """

    native_export_semantics = "merged-voicehub-native-voxcpm2-safetensors"

    def __init__(self, model, spec) -> None:
        super().__init__(model, spec)
        self.data_collator = VoxCPMTrainingCollator()

    def setup(self) -> VoxCPMTrainingAdapter:
        super().setup()
        runtime = getattr(self.model, "native_runtime", None)
        if runtime is None:
            raise ValueError("VoxCPM training requires the native runtime.")
        if self.primary_model is not getattr(runtime, "model", None):
            raise ValueError("VoxCPM training must target the runtime's exact 577-tensor graph.")
        codec = getattr(runtime, "codec", None)
        if codec is None:
            raise ValueError("VoxCPM raw-audio fine-tuning requires AudioVAE V2.")
        trainable_codec = [name for name, parameter in codec.named_parameters() if parameter.requires_grad]
        if trainable_codec:
            raise ValueError(
                "VoxCPM AudioVAE must remain frozen; trainable codec tensors: " +
                ", ".join(trainable_codec[:8]) + ".")
        return self

    def create_dataset(self, records: Any, **kwargs: Any) -> SpeechDataset:
        dataset = SpeechDataset(records, required_fields=("text", ), **kwargs)
        for index in range(len(dataset)):
            record = dataset[index]
            if not any(name in record for name in ("audio", "audio_features", "waveform")):
                raise ValueError(
                    f"VoxCPM training record {index} requires target audio "
                    "or pre-encoded `audio_features`.")
        return dataset

    def prepare_training_inputs(
        self,
        inputs: Mapping[str, Any],
        context: Any,
    ) -> Mapping[str, Any]:
        prepared = self.model.prepare_training_inputs(
            dict(inputs),
            phase=context.phase.name,
        )
        result = {name: value for name, value in prepared.items() if name in _FORWARD_INPUTS}
        progress = context.metadata.get("training_progress")
        if progress is not None and "progress" not in result:
            progress = float(progress)
            if not 0.0 <= progress <= 1.0:
                raise ValueError("VoxCPM `training_progress` must be in [0, 1].")
            result["progress"] = progress
        return result

    def _aggregate_losses(self, losses, phase=None):
        if "diffusion_loss" not in losses or "stop_loss" not in losses:
            return super()._aggregate_losses(losses, phase)
        config = self.model.config
        return (
            losses["diffusion_loss"] * config.training_diffusion_loss_weight +
            losses["stop_loss"] * config.training_stop_loss_weight)

    def recipe_resume_configuration(self) -> Mapping[str, Any]:
        configuration = dict(super().recipe_resume_configuration())
        configuration.update({
            "architecture":
            "voxcpm2",
            "audio_codec":
            "frozen-audiovae-v2",
            "input_sample_rate":
            16_000,
            "output_sample_rate":
            48_000,
            "objective": {
                "diffusion": "conditional-flow-matching",
                "stop": "token-cross-entropy",
            },
            "parameter_efficient": (getattr(self.model, "_active_lora_config", None) is not None),
        })
        return configuration

    def artifact_manifest(self) -> dict[str, Any]:
        manifest = super().artifact_manifest()
        manifest.update({
            "checkpoint_format": "voicehub-native-voxcpm2-v1",
            "native_architecture_family": "voxcpm2",
            "codec_trainable": False,
            "full_finetuning": True,
            "lora_finetuning": True,
            "lora_native_export": "merged-into-standard-model-namespace",
        })
        return manifest

    def on_training_phase_end(self, context: Any, output: Any) -> Any:
        output = super().on_training_phase_end(context, output)
        output.metadata.update({
            "native_architecture_family":
            "voxcpm2",
            "objective":
            "source-cfm-plus-stop-ce",
            "codec_frozen":
            True,
            "parameter_efficient": (getattr(self.model, "_active_lora_config", None) is not None),
        })
        return output

    def save_pretrained(self, save_directory: str | Path) -> None:
        """Export an inference-ready native runtime.

        For LoRA runs the ordinary Trainer checkpoint retains unmerged
        adapter tensors for exact resume, while this warm-start artifact
        contains the merged standard VoxCPM2 namespace plus a separate
        adapter copy.
        """
        self.setup()
        destination = Path(save_directory).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        self.model.export_native_pretrained(destination)


__all__ = ["VoxCPMTrainingAdapter", "VoxCPMTrainingCollator"]
