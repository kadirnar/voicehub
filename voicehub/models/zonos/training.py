"""Differentiable fine-tuning for the native Zonos v0.1 Transformer."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from voicehub.checkpointing import save_safetensors
from voicehub.models.zonos.native.checkpoint import save_zonos_pretrained
from voicehub.models.zonos.native.metadata import NATIVE_ZONOS_FORMAT
from voicehub.models.zonos.native.modeling import ZonosForCausalLM
from voicehub.outputs import TTSTrainingOutput
from voicehub.training.adapters import CausalLMTrainingAdapter
from voicehub.training.contracts import TrainingContext


class ZonosTrainingAdapter(CausalLMTrainingAdapter):
    """Full-model delayed-codebook causal fine-tuning.

    Zyphra publishes the inference graph and checkpoints, but not the
    original optimizer, data pipeline, or training loop.  This adapter
    reconstructs only the objective implied by generation: each codebook
    predicts its next delayed token, including the diagonal EOS cascade.
    The distinction is recorded in every artifact manifest.
    """

    supports_custom_recipe = True
    native_export_semantics = "inference-reloadable-safetensors"
    RECIPE_VERSION = 1

    def _native_model(self) -> ZonosForCausalLM | None:
        candidate = self.primary_model
        return (candidate if isinstance(candidate, ZonosForCausalLM) else None)

    def setup(self):
        super().setup()
        # Compatibility runtimes may expose the frozen codec on the model.
        # The native runtime keeps it outside the trainable graph entirely.
        for name in ("autoencoder", "codec"):
            component = getattr(self.primary_model, name, None)
            if component is None:
                continue
            component = getattr(component, "dac", component)
            if hasattr(component, "eval"):
                component.eval()
            if hasattr(component, "parameters"):
                for parameter in component.parameters():
                    parameter.requires_grad_(False)
        self.primary_model.train()
        return self

    @staticmethod
    def _validate_batch(
        prefix: Any,
        audio_codes: Any,
        lengths: Any,
        *,
        model: Any,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        if not isinstance(prefix, Tensor) or prefix.ndim != 3:
            raise ValueError(
                "Zonos `prefix_conditioning` must have shape "
                "[batch, prefix_time, hidden_size].")
        if not isinstance(audio_codes, Tensor) or audio_codes.ndim != 3:
            raise ValueError("Zonos `audio_codes` must have shape "
                             "[batch, codebook, audio_time].")
        if prefix.shape[0] != audio_codes.shape[0]:
            raise ValueError("Zonos prefix and audio-code batch sizes must match.")
        expected_codebooks = len(getattr(model, "embeddings", ()))
        if audio_codes.shape[1] != expected_codebooks:
            raise ValueError(
                f"Zonos expects {expected_codebooks} codebooks, received "
                f"{audio_codes.shape[1]}.")
        if audio_codes.dtype == torch.bool or audio_codes.is_floating_point():
            raise TypeError("Zonos audio codes must use an integer dtype.")
        if lengths is not None:
            if not isinstance(lengths, Tensor) or lengths.shape != (audio_codes.shape[0], ):
                raise ValueError("Zonos `audio_code_lengths` must have shape [batch].")
            lengths = lengths.to(dtype=torch.long)
        return prefix, audio_codes.long(), lengths

    def execute_training_phase(
        self,
        context: TrainingContext,
    ) -> TTSTrainingOutput:
        self.setup()
        prepared = self.prepare_batch(context.inputs, context)
        if not isinstance(prepared, Mapping):
            raise TypeError("Zonos training input preparation must return a mapping.")
        batch = self.prepare_runtime_inputs(prepared)
        prefix, audio_codes, code_lengths = self._validate_batch(
            batch.get("prefix_conditioning"),
            batch.get("audio_codes"),
            batch.get("audio_code_lengths"),
            model=self.primary_model,
        )
        device = getattr(self.primary_model, "device", None)
        if device is None:
            device = next(self.primary_model.parameters()).device
        dtype = next(self.primary_model.parameters()).dtype
        prefix = prefix.to(device=device, dtype=dtype)
        audio_codes = audio_codes.to(device=device)
        if code_lengths is not None:
            code_lengths = code_lengths.to(device=device)

        logits, labels = self.primary_model.teacher_forced_logits(
            prefix,
            audio_codes,
            audio_code_lengths=code_lengths,
        )
        supervised = labels.ne(int(self.primary_model.masked_token_id))
        if not bool(supervised.any()):
            raise ValueError("Zonos training batch contains no supervised codec token.")
        targets = labels.masked_fill(~supervised, -100)
        codec_loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            ignore_index=-100,
        )
        return TTSTrainingOutput(
            loss=codec_loss,
            logits=logits,
            losses={
                "loss": codec_loss,
                "codec_ce_loss": codec_loss,
            },
            metadata={
                "model_type": "zonos",
                "objective": ("reconstructed-delayed-codebook-causal-cross-entropy"),
                "objective_author_verified": False,
                "supervised_tokens": int(supervised.sum().item()),
                "full_model_gradient_ready": True,
                "codec": "frozen-native-descript-dac-44khz",
            },
            training_phase=context.phase.name,
            optimizer_names=context.phase.optimizer_names,
        )

    def save_pretrained(self, save_directory: str | Path) -> None:
        """Write a strict artifact loadable by fresh native inference."""
        self.setup()
        destination = Path(save_directory).expanduser()
        native = self._native_model()
        if native is not None:
            save_zonos_pretrained(native, destination)
            return

        # Keep the compatibility path used by legacy in-memory test graphs.
        # Production native models always take the strict branch above.
        destination.mkdir(parents=True, exist_ok=True)
        save_safetensors(
            {
                name: value.detach().cpu().contiguous()
                for name, value in self.primary_model.state_dict().items()
            },
            destination / "model.safetensors",
        )
        config = getattr(self.primary_model, "config", None)
        if config is None:
            raise TypeError("Zonos compatibility graph has no configuration.")
        if hasattr(config, "to_dict"):
            values = config.to_dict()
        elif is_dataclass(config):
            values = asdict(config)
        elif isinstance(config, Mapping):
            values = dict(config)
        else:
            raise TypeError("Zonos compatibility configuration is not serializable.")
        (destination / "config.json").write_text(
            json.dumps(
                values,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )

    def artifact_manifest(self) -> dict[str, Any]:
        manifest = super().artifact_manifest()
        manifest.update({
            "checkpoint_format": NATIVE_ZONOS_FORMAT,
            "native_architecture_family": "zonos-v0.1-transformer",
            "objective": ("reconstructed-delayed-codebook-causal-cross-entropy"),
            "objective_author_verified": False,
            "training_scope": "full-acoustic-language-model",
            "frozen_components": ["descript-dac-44khz"],
            "raw_audio_preprocessing": "optional-frozen-native-codec",
            "phoneme_frontend": ("precomputed-or-injected-checkpoint-compatible-frontend"),
            "inference_reloadable": True,
            "hybrid_mamba_support": False,
        })
        return manifest


__all__ = ["ZonosTrainingAdapter"]
