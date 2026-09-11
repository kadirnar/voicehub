"""Trainer integration for native StyleTTS 2 fine-tuning."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional

from voicehub.models.styletts2.native.training import StyleTTS2TrainingModel
from voicehub.training.adapters import VITSTrainingAdapter
from voicehub.training.datasets import SpeechDataset


class StyleTTS2TrainingCollator:
    """Pad both axes of monotonic alignments without guessing semantics."""

    REQUIRED_FIELDS = (
        "input_ids",
        "alignments",
        "normalized_mel",
        "reference_mel",
        "f0_targets",
        "noise_targets",
        "audio_values",
    )

    @staticmethod
    def _tensor(value: Any, *, name: str) -> Tensor:
        tensor = value if isinstance(value, Tensor) else torch.as_tensor(value)
        if tensor.is_complex():
            raise TypeError(f"StyleTTS 2 `{name}` cannot be complex.")
        return tensor

    @staticmethod
    def _right_pad(value: Tensor, shape: tuple[int, ...]) -> Tensor:
        if value.ndim != len(shape):
            raise ValueError("StyleTTS 2 collator received an invalid rank.")
        padding = []
        for actual, target in reversed(tuple(zip(value.shape, shape))):
            if actual > target:
                raise ValueError("StyleTTS 2 target padding shape is too small.")
            padding.extend((0, target - actual))
        return functional.pad(value, tuple(padding))

    def __call__(self, features: list[Mapping[str, Any]]) -> dict[str, Any]:
        if not features:
            return {}
        rows = [dict(feature) for feature in features]
        for index, row in enumerate(rows):
            missing = [name for name in self.REQUIRED_FIELDS if name not in row]
            if missing:
                raise ValueError(f"StyleTTS 2 sample {index} is missing: " + ", ".join(missing) + ".")
        tensors = {
            name: [self._tensor(row[name], name=name) for row in rows]
            for name in self.REQUIRED_FIELDS
        }
        input_ids = tensors["input_ids"]
        alignments = tensors["alignments"]
        if any(value.ndim != 1 for value in input_ids):
            raise ValueError("Each StyleTTS 2 `input_ids` sample must be 1-D.")
        if any(value.dtype == torch.bool or value.is_floating_point() for value in input_ids):
            raise TypeError("Each StyleTTS 2 `input_ids` sample must use integer dtype.")
        if any(value.shape[0] < 3 or int(value[0]) != 0 for value in input_ids):
            raise ValueError("Each StyleTTS 2 sample requires BOS ID 0 and at least two "
                             "phoneme tokens.")
        if any(value.ndim != 2 for value in alignments):
            raise ValueError("Each StyleTTS 2 alignment must be rank two.")
        for ids, alignment in zip(input_ids, alignments):
            if alignment.shape[0] != ids.shape[0]:
                raise ValueError("StyleTTS 2 alignment text rows must match input IDs.")
            if alignment.shape[1] < ids.shape[0]:
                raise ValueError(
                    "StyleTTS 2 monotonic alignments require at least one "
                    "frame per text token.")
        for index, alignment in enumerate(alignments):
            if bool((alignment < 0).any()):
                raise ValueError(f"StyleTTS 2 sample {index} alignment is negative.")
            if not bool(torch.allclose(
                    alignment.sum(dim=0).float(),
                    torch.ones(
                        alignment.shape[1],
                        dtype=torch.float32,
                        device=alignment.device,
                    ),
                    atol=1e-4,
                    rtol=1e-4,
            )):
                raise ValueError(
                    f"StyleTTS 2 sample {index} does not assign each "
                    "acoustic frame exactly once.")
            if not bool(((alignment.abs() <= 1e-4) | ((alignment - 1.0).abs() <= 1e-4)).all()):
                raise ValueError(f"StyleTTS 2 sample {index} alignment is not binary.")
            token_path = alignment.argmax(dim=0)
            advances = token_path[1:] - token_path[:-1]
            starts_at_first_token = int(token_path[0]) == 0
            ends_at_last_token = int(token_path[-1]) == alignment.shape[0] - 1
            advances_monotonically = not bool(((advances < 0) | (advances > 1)).any())
            if not (starts_at_first_token and ends_at_last_token and advances_monotonically):
                raise ValueError(f"StyleTTS 2 sample {index} has a non-monotonic "
                                 "alignment.")
        max_text = max(value.shape[0] for value in input_ids)
        max_acoustic = max(value.shape[1] for value in alignments)
        result: dict[str, Any] = {
            "input_ids":
            torch.stack([self._right_pad(value, (max_text, )) for value in input_ids]).long(),
            "input_lengths":
            torch.tensor(
                [value.shape[0] for value in input_ids],
                dtype=torch.long,
            ),
            "alignments":
            torch.stack([self._right_pad(
                value,
                (max_text, max_acoustic),
            ) for value in alignments]).float(),
            "alignment_lengths":
            torch.tensor(
                [value.shape[1] for value in alignments],
                dtype=torch.long,
            ),
        }
        for name in ("normalized_mel", "reference_mel"):
            values = [value.unsqueeze(0) if value.ndim == 2 else value for value in tensors[name]]
            if any(value.ndim != 3 or value.shape[0] != 1 for value in values):
                raise ValueError(
                    f"Each StyleTTS 2 `{name}` must have shape "
                    "[1, n_mels, frames] or [n_mels, frames].")
            mel_bins = values[0].shape[1]
            if any(value.shape[1] != mel_bins for value in values):
                raise ValueError("StyleTTS 2 mel bins must be consistent.")
            max_frames = max(value.shape[-1] for value in values)
            result[name] = torch.stack(
                [self._right_pad(value, (1, mel_bins, max_frames)) for value in values]).float()
            result[f"{name}_lengths"] = torch.tensor(
                [value.shape[-1] for value in values],
                dtype=torch.long,
            )
        normalized_lengths = result["normalized_mel_lengths"].tolist()
        for index, (alignment, mel_length) in enumerate(zip(alignments, normalized_lengths)):
            if mel_length != alignment.shape[1] * 2:
                raise ValueError(
                    f"StyleTTS 2 sample {index} normalized mel must have "
                    "twice as many frames as its alignment.")
        for name in ("f0_targets", "noise_targets", "audio_values"):
            values = [
                value.squeeze(0) if value.ndim == 2 and value.shape[0] == 1 else value
                for value in tensors[name]
            ]
            if any(value.ndim != 1 for value in values):
                raise ValueError(f"Each StyleTTS 2 `{name}` sample must be one-dimensional.")
            if name in {"f0_targets", "noise_targets"}:
                for index, (value, alignment) in enumerate(zip(values, alignments)):
                    if value.shape[0] != alignment.shape[1] * 2:
                        raise ValueError(
                            f"StyleTTS 2 sample {index} `{name}` must have "
                            "twice as many frames as its alignment.")
            maximum = max(value.shape[0] for value in values)
            stacked = torch.stack([self._right_pad(value, (maximum, )) for value in values]).float()
            result[name] = (stacked.unsqueeze(1) if name == "audio_values" else stacked)
            if name == "audio_values":
                result["audio_lengths"] = torch.tensor(
                    [value.shape[0] for value in values],
                    dtype=torch.long,
                )
        phases = {row.get("training_phase") for row in rows if row.get("training_phase") is not None}
        if len(phases) > 1:
            raise ValueError("Every StyleTTS 2 sample must select the same training phase.")
        if phases:
            result["training_phase"] = phases.pop()
        return result

    def resume_fingerprint(self) -> dict[str, Any]:
        return {
            "type": "styletts2-preprocessed-v1",
            "alignment_padding": "right-2d",
            "phoneme_padding_id": 0,
        }


class StyleTTS2TrainingAdapter(VITSTrainingAdapter):
    """Run native generator and fresh MPD/MSD phases."""

    supports_custom_recipe = True
    native_export_semantics = "inference-reloadable-styletts2-safetensors"

    def __init__(self, model: Any, spec: Any) -> None:
        super().__init__(model, spec)
        self.data_collator = StyleTTS2TrainingCollator()

    def validate_support(self) -> None:
        if getattr(
                self.model.config,
                "enable_native_finetuning",
                False,
        ) is not True:
            raise ValueError(
                "Set `enable_native_finetuning=True` to acknowledge the "
                "explicit preprocessed StyleTTS 2 training boundary.")
        super().validate_support()

    def setup(self) -> StyleTTS2TrainingAdapter:
        super().setup()
        native = getattr(self.model, "training_model", None)
        if not isinstance(native, StyleTTS2TrainingModel):
            raise TypeError("StyleTTS 2 fine-tuning requires StyleTTS2TrainingModel.")
        if self.primary_model is not native:
            raise ValueError("StyleTTS 2 recipe must target wrapper.training_model.")
        native.train()
        return self

    def select_training_phase(self, training_phase=None):
        phase = super().select_training_phase(training_phase)
        if (phase.name == "discriminator" and not self.model.config.training_enable_discriminators):
            raise ValueError(
                "StyleTTS 2 discriminator training is disabled by "
                "`training_enable_discriminators=False`.")
        return phase

    def plan_training_phases(self, step: int):
        phases = super().plan_training_phases(step)
        if self.model.config.training_enable_discriminators:
            return phases
        return tuple(phase for phase in phases if phase.name == "generator")

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
        accepted = {
            *StyleTTS2TrainingCollator.REQUIRED_FIELDS,
            "input_lengths",
            "alignment_lengths",
            "normalized_mel_lengths",
            "reference_mel_lengths",
            "audio_lengths",
        }
        prepared = {name: value for name, value in inputs.items() if name in accepted}
        prepared["phase"] = context.phase.name
        return prepared

    def recipe_resume_configuration(self) -> Mapping[str, Any]:
        configuration = dict(super().recipe_resume_configuration())
        configuration.update({
            "checkpoint_format": "voicehub-native-styletts2-v1",
            "objective_scope": "preprocessed-teacher-forced-full-generator",
            "fresh_mpd_msd": bool(self.model.config.training_enable_discriminators),
            "raw_g2p_available": False,
            "raw_alignment_available": False,
            "author_optimizer_state_available": False,
        })
        return configuration

    def artifact_manifest(self) -> dict[str, Any]:
        manifest = super().artifact_manifest()
        manifest.update({
            "checkpoint_format":
            "voicehub-native-styletts2-v1",
            "native_architecture_family":
            "styletts2",
            "training_scope":
            "preprocessed-teacher-forced",
            "generator_checkpoint_compatible":
            True,
            "inference_reloadable":
            True,
            "raw_text_frontend":
            False,
            "required_preprocessing": [
                "released 178-symbol phoneme IDs",
                "monotonic text-to-acoustic alignment",
                "normalized 80-bin mel and reference mel with frame lengths",
                "F0 and log-normalization targets",
                "24 kHz mono waveform with sample length",
            ],
            "fresh_components": [
                "multi-period-discriminator",
                "multi-resolution-spectral-discriminator",
            ],
        })
        return manifest

    def save_pretrained(self, save_directory: str | Path) -> None:
        self.setup()
        self.model.export_native_pretrained(save_directory)


__all__ = [
    "StyleTTS2TrainingAdapter",
    "StyleTTS2TrainingCollator",
]
