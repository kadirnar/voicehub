"""Architecture-consistent rectified-flow fine-tuning for Echo-TTS.

The public Echo repository is inference-only. VoiceHub reconstructs the
standard velocity objective implied by its rectified-flow sampler, but
does not present this integration as an author-verified reproduction of
the original data or optimization recipe.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicehub.checkpointing import save_safetensors
from voicehub.dependencies import import_optional
from voicehub.outputs import TTSTrainingOutput
from voicehub.training.adapters import FlowMatchingTrainingAdapter
from voicehub.training.contracts import TrainingContext


class EchoTrainingAdapter(FlowMatchingTrainingAdapter):
    """Train Echo's flow network from precomputed codec latents.

    Echo's public checkpoint does not ship a raw-audio data recipe. This
    adapter therefore keeps the Fish codec frozen and requires the exact
    latent, text, and speaker-conditioning tensors consumed by
    ``EchoDiT``.
    """

    supports_custom_recipe = True
    native_export_semantics = "source-compatible-flow-weight-warm-start"

    def setup(self):
        super().setup()
        codec = getattr(self.model, "fish_ae", None)
        if codec is not None:
            codec.eval()
            for parameter in codec.parameters():
                parameter.requires_grad_(False)
        self.primary_model.train()
        return self

    @staticmethod
    def _require_tensor(torch, batch: Mapping[str, Any], name: str, *, ndim: int):
        value = batch.get(name)
        if value is None:
            raise ValueError(f"Echo fine-tuning requires `{name}`.")
        if not torch.is_tensor(value):
            raise TypeError(f"Echo `{name}` must be a tensor.")
        if value.ndim != ndim:
            raise ValueError(
                f"Echo `{name}` must have {ndim} dimensions; received "
                f"shape {tuple(value.shape)}.")
        return value

    def execute_training_phase(
        self,
        context: TrainingContext,
    ) -> TTSTrainingOutput:
        self.setup()
        torch = import_optional(
            "torch",
            model_type="echo",
            install_extra="training",
        )
        batch = dict(context.inputs)
        target = self._require_tensor(
            torch,
            batch,
            "target_latents",
            ndim=3,
        )
        text_ids = self._require_tensor(
            torch,
            batch,
            "text_input_ids",
            ndim=2,
        )
        text_mask = self._require_tensor(
            torch,
            batch,
            "text_mask",
            ndim=2,
        )
        speaker_latents = self._require_tensor(
            torch,
            batch,
            "speaker_latents",
            ndim=3,
        )
        speaker_mask = self._require_tensor(
            torch,
            batch,
            "speaker_mask",
            ndim=2,
        )

        batch_size = target.shape[0]
        if target.shape[-1] != self.primary_model.in_proj.in_features:
            raise ValueError(
                "Echo `target_latents` has the wrong feature dimension: "
                f"expected {self.primary_model.in_proj.in_features}, "
                f"received {target.shape[-1]}.")
        if any(value.shape[0] != batch_size for value in (
                text_ids,
                text_mask,
                speaker_latents,
                speaker_mask,
        )):
            raise ValueError("All Echo training tensors must share one batch size.")
        if text_ids.shape != text_mask.shape:
            raise ValueError("Echo `text_input_ids` and `text_mask` shapes must match.")
        if speaker_latents.shape[:2] != speaker_mask.shape:
            raise ValueError(
                "Echo `speaker_mask` must match the first two dimensions of "
                "`speaker_latents`.")
        patch_size = int(self.primary_model.speaker_patch_size)
        if (speaker_latents.shape[1] < patch_size or speaker_latents.shape[1] % patch_size):
            raise ValueError(
                "Echo speaker conditioning length must be a positive multiple "
                f"of its {patch_size}-frame patch size.")

        device = self.primary_model.device
        dtype = self.primary_model.dtype
        target = target.to(device=device, dtype=dtype)
        text_ids = text_ids.to(device=device, dtype=torch.long)
        text_mask = text_mask.to(device=device, dtype=torch.bool)
        speaker_latents = speaker_latents.to(device=device, dtype=dtype)
        speaker_mask = speaker_mask.to(device=device, dtype=torch.bool)

        noise = batch.get("noise")
        if noise is None:
            noise = torch.randn_like(target)
        elif not torch.is_tensor(noise) or noise.shape != target.shape:
            raise ValueError("Echo `noise` must match `target_latents` exactly.")
        else:
            noise = noise.to(device=device, dtype=dtype)

        timesteps = batch.get("timesteps")
        if timesteps is None:
            timesteps = torch.rand(batch_size, device=device, dtype=dtype)
        elif not torch.is_tensor(timesteps):
            raise TypeError("Echo `timesteps` must be a tensor.")
        else:
            timesteps = timesteps.to(device=device, dtype=dtype)
        if timesteps.ndim == 2 and timesteps.shape[1] == 1:
            timesteps = timesteps[:, 0]
        if timesteps.shape != (batch_size, ):
            raise ValueError("Echo `timesteps` must have shape [batch].")
        if not bool(torch.isfinite(timesteps).all()) or bool(((timesteps < 0) | (timesteps > 1)).any()):
            raise ValueError("Echo `timesteps` must be finite and in [0, 1].")

        interpolation = timesteps[:, None, None]
        noisy_latents = (1 - interpolation) * target + interpolation * noise
        velocity_target = noise - target
        text_cache = self.primary_model.get_kv_cache_text(text_ids, text_mask)
        speaker_cache = self.primary_model.get_kv_cache_speaker(speaker_latents)
        predicted_velocity = self.primary_model(
            noisy_latents,
            timesteps,
            text_mask,
            speaker_mask,
            text_cache,
            speaker_cache,
        )
        if predicted_velocity.shape != target.shape:
            raise RuntimeError(
                "Echo flow network returned shape "
                f"{tuple(predicted_velocity.shape)}, expected {tuple(target.shape)}.")

        squared_error = (predicted_velocity.float() - velocity_target.float()).square()
        latent_mask = batch.get("latent_mask")
        if latent_mask is not None:
            if not torch.is_tensor(latent_mask) or latent_mask.shape != target.shape[:2]:
                raise ValueError("Echo `latent_mask` must have shape [batch, latent_time].")
            weights = latent_mask.to(device=device, dtype=torch.bool).unsqueeze(-1)
            denominator = weights.sum() * target.shape[-1]
            if int(denominator.item()) == 0:
                raise ValueError("Echo `latent_mask` must select at least one frame.")
            flow_loss = (squared_error * weights).sum() / denominator
        else:
            flow_loss = squared_error.mean()

        return TTSTrainingOutput(
            loss=flow_loss,
            logits=predicted_velocity,
            losses={
                "loss": flow_loss,
                "flow_loss": flow_loss,
            },
            metadata={
                "model_type": "echo",
                "objective": "rectified-flow-velocity",
                "preprocessing": "source-shaped-latents",
                "recipe_status": "reconstructed-not-author-verified",
            },
            training_phase=context.phase.name,
            optimizer_names=context.phase.optimizer_names,
        )

    def save_pretrained(self, save_directory) -> None:
        """Write the files consumed by Echo's source loader."""
        self.setup()
        destination = Path(save_directory)
        destination.mkdir(parents=True, exist_ok=True)
        state = {
            name: value.detach().cpu().contiguous()
            for name, value in self.primary_model.state_dict().items()
        }
        save_safetensors(
            state,
            destination / "pytorch_model.safetensors",
        )

        pca = getattr(self.model, "pca_state", None)
        if pca is None:
            raise RuntimeError("Echo export requires the loaded PCA state.")
        pca_state = {
            "pca_components": pca.pca_components.detach().cpu().contiguous(),
            "pca_mean": pca.pca_mean.detach().cpu().contiguous(),
            "latent_scale": self._scalar_tensor(
                pca.latent_scale,
                like=pca.pca_mean,
            ),
        }
        save_safetensors(
            pca_state,
            destination / "pca_state.safetensors",
        )
        config = getattr(self.model, "config", None)
        if config is not None and hasattr(config, "save_pretrained"):
            config.save_pretrained(destination)

    @staticmethod
    def _scalar_tensor(value, *, like):
        torch = import_optional(
            "torch",
            model_type="echo",
            install_extra="training",
        )
        return torch.tensor(
            float(value),
            dtype=like.dtype,
            device="cpu",
        ).contiguous()


__all__ = ["EchoTrainingAdapter"]
