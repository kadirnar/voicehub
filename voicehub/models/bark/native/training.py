"""Stage-specific, pretokenized fine-tuning for native Bark."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from voicehub.training.adapters import CompositeTrainingAdapter
from voicehub.training.contracts import TrainingContext

from .modeling import BarkCoarseModel, BarkFineModel, BarkSemanticModel


class BarkTokenObjective(nn.Module):
    """One explicitly aligned Bark stage cross-entropy objective."""

    def __init__(
        self,
        component: BarkSemanticModel | BarkCoarseModel | BarkFineModel,
        *,
        shift_labels: bool,
        fine_stage: bool = False,
    ) -> None:
        super().__init__()
        self.component = component
        self.shift_labels = shift_labels
        self.fine_stage = fine_stage

    @staticmethod
    def _scalar_codebook_index(value: Any) -> int:
        if value is None:
            raise ValueError("Bark fine training requires `codebook_idx`.")
        if isinstance(value, Tensor):
            flattened = value.detach().reshape(-1)
            if flattened.numel() == 0:
                raise ValueError("Bark `codebook_idx` cannot be empty.")
            first = int(flattened[0].item())
            if not bool((flattened == first).all().item()):
                raise ValueError("One Bark fine batch may target only one codebook.")
            return first
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("Bark `codebook_idx` must be an integer.")
        return value

    def forward(
        self,
        input_ids: Tensor,
        *,
        labels: Tensor,
        attention_mask: Tensor | None = None,
        codebook_idx: int | Tensor | None = None,
    ) -> dict[str, Tensor]:
        if not isinstance(input_ids, Tensor) or not isinstance(labels, Tensor):
            raise TypeError("Bark training IDs and labels must be tensors.")
        if self.fine_stage:
            if input_ids.ndim != 3 or labels.ndim != 2:
                raise ValueError(
                    "Bark fine training expects input IDs [batch, tokens, 8] "
                    "and labels [batch, tokens].")
            outputs = self.component(
                input_ids,
                codebook_idx=self._scalar_codebook_index(codebook_idx),
                attention_mask=attention_mask,
            )
        else:
            if input_ids.ndim != 2 or labels.ndim != 2:
                raise ValueError("Bark causal training expects IDs and labels shaped "
                                 "[batch, tokens].")
            outputs = self.component(
                input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
        logits = outputs.logits
        common = min(logits.shape[-2], labels.shape[-1])
        if self.shift_labels:
            if common < 2:
                raise ValueError("Bark causal training requires at least two aligned tokens.")
            predictions = logits[:, :common - 1].contiguous()
            targets = labels[:, 1:common].contiguous()
        else:
            predictions = logits[:, :common].contiguous()
            targets = labels[:, :common].contiguous()
        valid = targets.ne(-100)
        if not bool(valid.any().item()):
            raise ValueError("Bark training batch contains no supervised tokens.")
        if bool(((targets[valid] < 0) | (targets[valid] >= predictions.shape[-1])).any().item()):
            raise ValueError("Bark labels contain token IDs outside the stage vocabulary.")
        loss = F.cross_entropy(
            predictions.reshape(-1, predictions.shape[-1]),
            targets.reshape(-1).long(),
            ignore_index=-100,
        )
        return {"loss": loss, "logits": logits}


class BarkTrainingModel(nn.Module):
    """Three independent Bark fine-tuning phase roots."""

    def __init__(
        self,
        semantic: BarkSemanticModel,
        coarse: BarkCoarseModel,
        fine: BarkFineModel,
    ) -> None:
        super().__init__()
        self.semantic = BarkTokenObjective(
            semantic,
            shift_labels=True,
        )
        self.coarse = BarkTokenObjective(
            coarse,
            shift_labels=True,
        )
        self.fine = BarkTokenObjective(
            fine,
            shift_labels=False,
            fine_stage=True,
        )

    @classmethod
    def from_model(cls, model: Any) -> BarkTrainingModel:
        return cls(
            model.semantic,
            model.coarse_acoustics,
            model.fine_acoustics,
        )


class BarkTrainingAdapter(CompositeTrainingAdapter):
    """Train one Bark token stage per phase; keep Encodec frozen."""

    supports_custom_recipe = True
    native_export_semantics = "inference-reloadable-safetensors"

    def setup(self) -> BarkTrainingAdapter:
        super().setup()
        codec = getattr(getattr(self.model, "model", None), "codec_model", None)
        if codec is None:
            raise TypeError("Bark training requires its native codec graph.")
        codec.eval()
        codec.requires_grad_(False)
        training_model = getattr(self.model, "training_model", None)
        if not isinstance(training_model, BarkTrainingModel):
            raise TypeError("Bark wrapper did not expose the native stage training graph.")
        return self

    def prepare_training_inputs(
        self,
        inputs: Mapping[str, Any],
        context: TrainingContext,
    ) -> Mapping[str, Any]:
        return self.model.prepare_training_inputs(
            dict(inputs),
            phase=context.phase.name,
        )

    def save_pretrained(self, save_directory: str | Path) -> None:
        self.setup()
        self.model.export_native_pretrained(save_directory)

    def artifact_manifest(self) -> dict[str, Any]:
        manifest = super().artifact_manifest()
        manifest.update({
            "checkpoint_format": "voicehub-bark-v1",
            "training_scope": "pretokenized-stage-specific",
            "raw_audio_fine_tuning": False,
            "objectives": {
                "semantic": "shifted-token-cross-entropy",
                "coarse": "shifted-interleaved-codebook-cross-entropy",
                "fine": "per-codebook-masked-token-cross-entropy",
            },
            "frozen_components": ["codec_model"],
            "inference_reloadable": True,
        })
        return manifest


__all__ = [
    "BarkTokenObjective",
    "BarkTrainingAdapter",
    "BarkTrainingModel",
]
