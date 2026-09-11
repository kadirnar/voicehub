"""Source-faithful Parler-TTS fine-tuning adapter."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from voicehub.hub import read_json_file, write_json_file
from voicehub.models.parlertts.native.checkpoint import export_parlertts_checkpoint
from voicehub.models.parlertts.native.modeling import ParlerTTSForConditionalGeneration
from voicehub.training.adapters import Seq2SeqTrainingAdapter
from voicehub.training.contracts import TrainingContext


class ParlerTTSTrainingAdapter(Seq2SeqTrainingAdapter):
    """Train the exact delayed-codebook cross-entropy objective.

    Upstream freezes DAC for every run. FLAN-T5, the acoustic decoder,
    and text-prompt embeddings remain trainable by default. Set
    ``freeze_text_encoder=True`` on the wrapper configuration for the
    lower-memory upstream option.
    """

    supports_custom_recipe = True
    native_export_semantics = "inference-export"
    RECIPE_VERSION = 1

    def _native_runtime(self) -> ParlerTTSForConditionalGeneration | None:
        runtime = getattr(self.model, "model", None)
        return (runtime if isinstance(runtime, ParlerTTSForConditionalGeneration) else None)

    def setup(self):
        super().setup()
        runtime = self._native_runtime()
        if runtime is None:
            return self
        freeze_text_encoder = bool(
            getattr(
                getattr(self.model, "config", None),
                "freeze_text_encoder",
                False,
            ))
        runtime.freeze_encoders(freeze_text_encoder=freeze_text_encoder)
        runtime.audio_encoder.eval()
        runtime.decoder.train()
        runtime.embed_prompts.train()
        return self

    def prepare_training_inputs(
        self,
        inputs: Mapping[str, Any],
        context: TrainingContext,
    ) -> Mapping[str, Any]:
        prepare = getattr(self.model, "prepare_training_inputs", None)
        prepared = (prepare(dict(inputs), phase=context.phase.name) if callable(prepare) else dict(inputs))
        runtime = self._native_runtime()
        if runtime is None:
            return prepared
        labels = prepared.get("labels")
        if labels is not None:
            if not isinstance(labels, torch.Tensor) or labels.ndim != 3:
                raise ValueError("Parler-TTS labels must have shape "
                                 "[batch, frames, codebooks].")
            expected = runtime.config.decoder.num_codebooks
            if labels.shape[-1] != expected:
                raise ValueError(
                    f"Parler-TTS labels contain {labels.shape[-1]} "
                    f"codebooks; expected {expected}.")
            if labels.dtype == torch.bool or labels.is_floating_point():
                raise TypeError("Parler-TTS labels must use an integer dtype.")
        return prepared

    def save_pretrained(self, save_directory) -> None:
        """Export a fresh-inference graph and all required processor assets."""
        self.setup()
        destination = Path(save_directory)
        destination.mkdir(parents=True, exist_ok=True)
        runtime = self._native_runtime()
        if runtime is None:
            raise TypeError("Parler-TTS portable export requires the native Parler graph.")
        export_parlertts_checkpoint(
            runtime,
            destination / "model.safetensors",
        )
        architecture = getattr(self.model, "architecture_config", None)
        if architecture is None:
            raise TypeError("Parler-TTS export requires architecture metadata.")
        write_json_file(
            destination / "config.json",
            architecture.to_dict(),
        )
        tokenizer = getattr(self.model, "tokenizer", None)
        if tokenizer is None:
            raise TypeError("Parler-TTS export requires its text tokenizer.")
        tokenizer.sentencepiece.save_pretrained(
            destination,
            filename="spiece.model",
        )
        artifacts = getattr(self.model, "artifacts", None)
        generation_path = getattr(artifacts, "generation_config", None)
        if generation_path is None:
            generation_config = {
                "bos_token_id": architecture.decoder.bos_token_id,
                "decoder_start_token_id": architecture.decoder_start_token_id,
                "do_sample": True,
                "eos_token_id": architecture.decoder.eos_token_id,
                "max_length": 2_580,
                "min_new_tokens": 10,
                "pad_token_id": architecture.decoder.pad_token_id,
            }
        else:
            generation_config = read_json_file(generation_path)
        write_json_file(
            destination / "generation_config.json",
            generation_config,
        )

    def artifact_manifest(self) -> dict[str, Any]:
        manifest = super().artifact_manifest()
        freeze_text_encoder = bool(
            getattr(
                getattr(self.model, "config", None),
                "freeze_text_encoder",
                False,
            ))
        default_frozen = ["audio_encoder"]
        if freeze_text_encoder:
            default_frozen.append("text_encoder")
        manifest.update({
            "checkpoint_format":
            "voicehub-parlertts-v1",
            "native_architecture_family":
            "parler-tts-mini-v1",
            "default_training_scope": (
                "text-encoder-decoder-and-prompt-embeddings"
                if not freeze_text_encoder else "decoder-and-prompt-embeddings"),
            "default_frozen_components":
            default_frozen,
            "objective":
            "delayed-codebook-cross-entropy",
            "inference_reloadable":
            True,
        })
        return manifest


__all__ = ["ParlerTTSTrainingAdapter"]
