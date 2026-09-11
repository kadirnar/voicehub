"""Source-faithful full fine-tuning for native Higgs Audio v2."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from voicehub.training.adapters import CausalLMTrainingAdapter
from voicehub.training.datasets import SpeechDataset

_FORWARD_INPUTS = frozenset({
    "attention_mask",
    "audio_input_ids",
    "audio_input_ids_mask",
    "audio_labels",
    "input_ids",
    "labels",
    "output_attentions",
    "output_hidden_states",
    "position_ids",
})


@dataclass(frozen=True)
class HiggsTrainingCollator:
    """Keep raw records intact until the frozen codec can encode the batch."""

    pad_to_multiple_of: int | None = 8

    def __post_init__(self) -> None:
        value = self.pad_to_multiple_of
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
            raise ValueError("`pad_to_multiple_of` must be a positive integer or None.")

    def __call__(
        self,
        features: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        if not features:
            return {}
        if any(not isinstance(record, Mapping) for record in features):
            raise TypeError("Every Higgs training sample must be a mapping.")
        return {"records": [dict(record) for record in features]}

    def resume_fingerprint(self) -> dict[str, Any]:
        return {
            "audio_codec": "frozen-higgs-audio-v2-tokenizer",
            "packing": "chatml-delayed-eight-codebook",
            "pad_to_multiple_of": self.pad_to_multiple_of,
            "type": "higgs-audio-v2-source-records-v1",
        }


class HiggsSFTDataset(SpeechDataset):
    """Validated raw or pre-encoded records for Higgs supervised tuning."""

    def __init__(self, records: Any, **kwargs: Any) -> None:
        super().__init__(records, required_fields=("text", ), **kwargs)
        for index in range(len(self)):
            record = self[index]
            if not any(name in record for name in ("audio", "audio_codes", "target_audio")):
                raise ValueError(
                    f"Higgs training record {index} requires `audio`, "
                    "`target_audio`, or pre-encoded `audio_codes`.")
            has_reference = any(
                record.get(name) is not None for name in ("reference_audio", "reference_codes"))
            reference_text = record.get("reference_text")
            if has_reference and (not isinstance(reference_text, str) or not reference_text.strip()):
                raise ValueError(
                    f"Higgs training record {index} uses reference audio "
                    "and therefore requires non-empty `reference_text`.")
            if not has_reference and reference_text is not None:
                raise ValueError(
                    f"Higgs training record {index} supplies "
                    "`reference_text` without reference audio.")


class HiggsTrainingAdapter(CausalLMTrainingAdapter):
    """Optimize the released joint text/eight-codebook causal objective.

    Boson publishes the model forward and delay/collation semantics, but
    not a complete optimizer or schedule recipe. VoiceHub therefore
    preserves the source objective exactly while owning the surrounding
    training loop.
    """

    native_export_semantics = ("voicehub-native-higgs-audio-v2-safetensors")

    def __init__(self, model: Any, spec: Any) -> None:
        super().__init__(model, spec)
        self.data_collator = HiggsTrainingCollator()

    def setup(self) -> HiggsTrainingAdapter:
        super().setup()
        runtime = getattr(self.model, "native_runtime", None)
        if runtime is None:
            raise ValueError("Higgs training requires the native runtime.")
        if self.primary_model is not getattr(runtime, "model", None):
            raise ValueError("Higgs training must target the runtime's exact native "
                             "dual-FFN decoder.")
        codec = getattr(runtime, "audio_tokenizer", None)
        if codec is None:
            raise ValueError("Higgs raw-audio fine-tuning requires its native tokenizer.")
        trainable_codec = [name for name, parameter in codec.named_parameters() if parameter.requires_grad]
        if trainable_codec:
            raise ValueError(
                "The Higgs audio tokenizer must remain frozen; trainable "
                "codec tensors: " + ", ".join(trainable_codec[:8]) + ".")
        return self

    def create_dataset(
        self,
        records: Any,
        **kwargs: Any,
    ) -> HiggsSFTDataset:
        return HiggsSFTDataset(records, **kwargs)

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

    def _aggregate_losses(self, losses: Any, phase: Any = None):
        if "text_loss" not in losses and "audio_loss" not in losses:
            return super()._aggregate_losses(losses, phase)
        text_weight = self.model.config.training_text_loss_weight
        audio_weight = self.model.config.training_audio_loss_weight
        weighted = []
        if "text_loss" in losses and text_weight:
            weighted.append(losses["text_loss"] * text_weight)
        if "audio_loss" in losses and audio_weight:
            weighted.append(losses["audio_loss"] * audio_weight)
        if not weighted:
            raise ValueError(
                "Higgs received no supervised text or audio tokens for an "
                "enabled objective.")
        return sum(weighted)

    def recipe_resume_configuration(self) -> Mapping[str, Any]:
        configuration = dict(super().recipe_resume_configuration())
        configuration.update({
            "architecture": "higgs_audio_v2",
            "audio_codec": "frozen-higgs-audio-v2-tokenizer",
            "audio_codebooks": 8,
            "checkpoint_format": "safetensors",
            "objective": {
                "audio": "sum-of-eight-delayed-codebook-causal-ce",
                "text": "causal-ce",
            },
            "sample_rate": 24_000,
        })
        return configuration

    def artifact_manifest(self) -> dict[str, Any]:
        manifest = super().artifact_manifest()
        manifest.update({
            "checkpoint_format": "voicehub-native-higgs-audio-v2-v1",
            "codec_trainable": False,
            "full_finetuning": True,
            "native_architecture_family": "higgs_audio_v2",
            "recipe_provenance": "source-forward-and-collation; voicehub-optimizer-orchestration",
        })
        return manifest

    def on_training_phase_end(self, context: Any, output: Any) -> Any:
        output = super().on_training_phase_end(context, output)
        output.metadata.update({
            "codec_frozen": True,
            "native_architecture_family": "higgs_audio_v2",
            "objective": "source-joint-text-plus-delayed-codebook-ce",
            "recipe_provenance": "source-objective-voicehub-orchestration",
        })
        return output

    def save_pretrained(self, save_directory: str | Path) -> None:
        self.setup()
        destination = Path(save_directory).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        self.model.export_native_pretrained(destination)


def load_higgs_training_backend(
    model_name_or_path: str,
    audio_tokenizer_name_or_path: str,
    *,
    device: str,
    torch_dtype: str = "bfloat16",
    **kwargs: Any,
):
    """Compatibility loader returning the native cache-free runtime."""
    from voicehub.models.higgstts.native.runtime import load_higgs_audio_v2_runtime

    return load_higgs_audio_v2_runtime(
        model_name_or_path,
        codec_source=audio_tokenizer_name_or_path,
        device=device,
        dtype=torch_dtype,
        **kwargs,
    )


def __getattr__(name: str) -> Any:
    if name == "HiggsTrainingBackend":
        from voicehub.models.higgstts.native.runtime import HiggsAudioV2Runtime

        return HiggsAudioV2Runtime
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "HiggsSFTDataset",
    "HiggsTrainingAdapter",
    "HiggsTrainingCollator",
    "load_higgs_training_backend",
]
