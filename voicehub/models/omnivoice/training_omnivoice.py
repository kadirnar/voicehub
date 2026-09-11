"""Source-faithful full fine-tuning for native OmniVoice."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from voicehub.training.adapters import CompositeTrainingAdapter
from voicehub.training.datasets import SpeechDataset

_FORWARD_INPUTS = frozenset({
    "attention_mask",
    "audio_mask",
    "document_ids",
    "input_ids",
    "labels",
    "position_ids",
})


@dataclass(frozen=True)
class OmniVoiceTrainingCollator:
    """Keep records raw until the frozen Higgs tokenizer is available."""

    def __call__(
        self,
        features: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        if not features:
            return {}
        if any(not isinstance(record, Mapping) for record in features):
            raise TypeError("Every OmniVoice training sample must be a mapping.")
        return {"records": [dict(record) for record in features]}

    def resume_fingerprint(self) -> dict[str, Any]:
        return {
            "audio_codec": "frozen-higgs-audio-v2",
            "objective": "weighted-codebook-masked-cross-entropy",
            "record_boundary": "raw-waveform-or-eight-codebook-tokens",
            "type": "omnivoice-source-records-v1",
        }


class OmniVoiceTrainingAdapter(CompositeTrainingAdapter):
    """Optimize the published independently averaged codebook CE objective."""

    native_export_semantics = ("inference-ready-voicehub-native-omnivoice-safetensors")

    def __init__(self, model, spec) -> None:
        super().__init__(model, spec)
        self.data_collator = OmniVoiceTrainingCollator()

    def setup(self) -> OmniVoiceTrainingAdapter:
        super().setup()
        runtime = getattr(self.model, "native_runtime", None)
        if runtime is None:
            raise ValueError("OmniVoice training requires the native runtime.")
        if self.primary_model is not getattr(runtime, "model", None):
            raise ValueError("OmniVoice training must target the exact native model graph.")
        codec = getattr(runtime, "audio_tokenizer", None)
        if codec is None:
            raise ValueError("Raw-audio OmniVoice fine-tuning requires Higgs Audio V2.")
        trainable = [name for name, parameter in codec.named_parameters() if parameter.requires_grad]
        if trainable:
            raise ValueError(
                "Higgs Audio V2 must remain frozen; trainable tensors: " + ", ".join(trainable[:8]))
        return self

    def create_dataset(self, records: Any, **kwargs: Any) -> SpeechDataset:
        dataset = SpeechDataset(records, required_fields=("text", ), **kwargs)
        for index in range(len(dataset)):
            record = dataset[index]
            if not any(name in record for name in ("audio", "audio_tokens", "waveform")):
                raise ValueError(
                    f"OmniVoice record {index} requires raw audio or "
                    "preprocessed `audio_tokens`.")
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
        return {name: value for name, value in prepared.items() if name in _FORWARD_INPUTS}

    def recipe_resume_configuration(self) -> Mapping[str, Any]:
        configuration = dict(super().recipe_resume_configuration())
        configuration.update({
            "architecture": "omnivoice",
            "audio_codec": "frozen-higgs-audio-v2",
            "audio_codebooks": 8,
            "audio_frame_rate": 25,
            "attention": "fully-bidirectional",
            "checkpoint_format": "safetensors",
            "objective": {
                "aggregation": "independent-codebook-means",
                "ignore_index": -100,
                "loss": "masked-cross-entropy",
                "published_weights": [8, 8, 6, 6, 4, 4, 2, 2],
            },
        })
        return configuration

    def artifact_manifest(self) -> dict[str, Any]:
        manifest = super().artifact_manifest()
        manifest.update({
            "checkpoint_format": "voicehub-native-omnivoice-v1",
            "codec_trainable": False,
            "full_finetuning": True,
            "native_architecture_family": "omnivoice",
            "preprocessed_audio_contract": "[8, frames] int codec IDs",
        })
        return manifest

    def on_training_phase_end(self, context: Any, output: Any) -> Any:
        output = super().on_training_phase_end(context, output)
        output.metadata.update({
            "codec_frozen": True,
            "native_architecture_family": "omnivoice",
            "objective": "source-weighted-codebook-masked-ce",
        })
        return output

    def save_pretrained(self, save_directory: str | Path) -> None:
        self.setup()
        destination = Path(save_directory).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        self.model.export_native_pretrained(destination)


__all__ = ["OmniVoiceTrainingAdapter", "OmniVoiceTrainingCollator"]
