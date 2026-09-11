"""Model-local ConversationTTS source-native training adapter."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from voicehub.dependencies import import_optional
from voicehub.outputs import TTSTrainingOutput
from voicehub.training.adapters import CausalLMTrainingAdapter
from voicehub.training.contracts import TrainingContext
from voicehub.training.recipes import SourceRecipeTrainingAdapter


class ConversationTTSTrainingAdapter(
        CausalLMTrainingAdapter,
        SourceRecipeTrainingAdapter,
):
    """Published two-level codec language-model objective and export."""

    native_export_semantics = "inference-reloadable-safetensors"

    def execute_training_phase(
        self,
        context: TrainingContext,
    ) -> TTSTrainingOutput:
        self.setup()
        prepared = self.prepare_batch(context.inputs, context)
        if not isinstance(prepared, Mapping):
            raise TypeError("ConversationTTS input preparation must return a mapping.")
        prepared = dict(prepared)
        required = ("tokens", "labels", "tokens_mask")
        missing = [name for name in required if name not in prepared]
        if missing:
            raise ValueError("ConversationTTS fine-tuning requires: " + ", ".join(missing))
        model = self.primary_model
        c0_logits, residual_logits, residual_labels = model(
            tokens=prepared["tokens"],
            labels=prepared["labels"],
            tokens_mask=prepared["tokens_mask"],
            input_pos=prepared.get("input_pos"),
        )
        source = import_optional(
            "voicehub.models.conversationtts.source.conversationtts."
            "models.model_new",
            model_type="conversationtts",
            install_extra="training",
        )
        labels = prepared["labels"]
        zero_labels = labels[..., 0]
        zero_mask = prepared.get("loss_mask")
        if zero_mask is None:
            zero_mask = prepared["tokens_mask"][:, 1:, 0].bool()
        zero_length = min(c0_logits.shape[1], zero_labels.shape[1], zero_mask.shape[1])
        default_padding_id = int(getattr(
            getattr(model, "config", None),
            "audio_vocab_size",
            2_051,
        )) - 1
        loss_zero, zero_metrics = source.CrossEntropyAndAccuracy_zero(
            c0_logits[:, :zero_length],
            zero_labels[:, :zero_length],
            zero_mask[:, :zero_length],
            ignore_id=int(prepared.get("ignore_id", default_padding_id)),
        )
        residual_weights = prepared.get("residual_loss_weights")
        if residual_weights is None:
            residual_weights = [1.0] * int(residual_labels.shape[-1])
        loss_residual, residual_metrics = source.CrossEntropyAndAccuracy_residual(
            residual_logits,
            residual_labels,
            loss_weights=residual_weights,
            ignore_id=int(prepared.get(
                "residual_ignore_id",
                default_padding_id,
            )),
        )
        loss = loss_zero + loss_residual
        metrics = {
            **zero_metrics,
            **residual_metrics,
        }
        return self._training_output(
            context,
            loss=loss,
            losses={
                "loss": loss,
                "codebook0_loss": loss_zero,
                "residual_loss": loss_residual,
            },
            logits=(c0_logits, residual_logits),
            metadata={"metrics": metrics},
        )

    def save_pretrained(self, save_directory) -> None:
        self.setup()
        export = getattr(self.model, "export_native_pretrained", None)
        if not callable(export):
            raise TypeError(
                "Native ConversationTTS training requires a wrapper with "
                "`export_native_pretrained()`.")
        export(save_directory)

    def artifact_manifest(self) -> dict[str, Any]:
        manifest = super().artifact_manifest()
        manifest.update({
            "checkpoint_format": "voicehub-conversationtts-v1",
            "native_architecture_family": "conversationtts",
            "objective": "published-two-level-masked-codebook-cross-entropy",
            "objective_author_verified": True,
            "training_scope": "full-acoustic-language-model",
            "frozen_components": ["mimi-codec"],
            "raw_audio_preprocessing": "optional-frozen-native-mimi",
            "inference_reloadable": True,
            "commercial_use": False,
        })
        return manifest


__all__ = ["ConversationTTSTrainingAdapter"]
