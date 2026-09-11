"""Teacher-forced delayed-codebook fine-tuning for Vui."""

from __future__ import annotations

from voicehub.dependencies import import_optional
from voicehub.outputs import TTSTrainingOutput
from voicehub.training.adapters import CausalLMTrainingAdapter
from voicehub.training.contracts import TrainingContext


class VuiTrainingAdapter(CausalLMTrainingAdapter):
    """Optimize Vui's causal multi-codebook objective.

    Audio is expected as native Fluac code IDs. The frozen codec remains
    available for preprocessing and inference but is never optimized.
    """

    supports_custom_recipe = True
    native_export_semantics = "inference-reloadable-vui-fluac-safetensors"

    def setup(self):
        super().setup()
        codec = getattr(self.primary_model, "codec", None)
        if codec is not None:
            codec.eval()
            for parameter in codec.parameters():
                parameter.requires_grad_(False)
        decoder = getattr(self.primary_model, "decoder", None)
        if decoder is not None and hasattr(decoder, "deallocate_kv_cache"):
            decoder.deallocate_kv_cache()
        self.primary_model.train()
        return self

    def execute_training_phase(
        self,
        context: TrainingContext,
    ) -> TTSTrainingOutput:
        self.setup()
        torch = import_optional(
            "torch",
            model_type="vui",
            install_extra="training",
        )
        functional = import_optional(
            "torch.nn.functional",
            model_type="vui",
            install_extra="training",
        )
        batch = dict(context.inputs)
        input_ids = batch.get("input_ids")
        audio_codes = batch.get("audio_codes")
        if not torch.is_tensor(input_ids) or input_ids.ndim != 2:
            raise ValueError("Vui `input_ids` must have shape [batch, text_time].")
        if not torch.is_tensor(audio_codes) or audio_codes.ndim != 3:
            raise ValueError("Vui `audio_codes` must have shape [batch, codebook, audio_time].")
        if input_ids.shape[0] != audio_codes.shape[0]:
            raise ValueError("Vui text and audio code batch sizes must match.")
        if audio_codes.shape[-1] == 0:
            raise ValueError("Vui `audio_codes` must contain at least one frame.")

        architecture = self.primary_model.config.model
        codebook_count = int(architecture.n_quantizers)
        if audio_codes.shape[1] != codebook_count:
            raise ValueError(f"Vui expects {codebook_count} codebooks, received "
                             f"{audio_codes.shape[1]}.")
        if bool(((audio_codes < 0) | (audio_codes >= architecture.codebook_size)).any()):
            raise ValueError(
                "Vui `audio_codes` must contain source codec IDs in "
                f"[0, {architecture.codebook_size - 1}].")

        device = self.primary_model.device
        input_ids = input_ids.to(device=device, dtype=torch.long)
        audio_codes = audio_codes.to(device=device, dtype=torch.long)
        text_mask = batch.get("text_attention_mask")
        if text_mask is None:
            text_mask = torch.ones_like(input_ids, dtype=torch.bool)
        elif not torch.is_tensor(text_mask) or text_mask.shape != input_ids.shape:
            raise ValueError("Vui `text_attention_mask` must match `input_ids`.")
        else:
            text_mask = text_mask.to(device=device, dtype=torch.bool)
        if input_ids.shape[1] == 0 or not bool(text_mask[:, 0].all()):
            raise ValueError("Every Vui sequence must start with a valid text token.")
        # The source model has no position compaction for ragged prefixes.
        # Right padding is safe because the explicit attention mask below
        # prevents audio positions from attending to padded text keys.
        if bool((text_mask[:, 1:] & ~text_mask[:, :-1]).any()):
            raise ValueError("Vui text batches must use contiguous right padding.")

        pattern = self.primary_model.pattern_provider.get_pattern(int(audio_codes.shape[-1]), )
        sequence, _, pattern_mask = pattern.build_pattern_sequence(
            audio_codes,
            int(architecture.special_token_id),
        )
        model_codes = sequence[..., :-1]
        targets = sequence[..., 1:]
        valid_targets = pattern_mask[:, 1:].to(device=device)
        valid_targets = valid_targets.unsqueeze(0).expand_as(targets)

        code_lengths = batch.get("audio_code_lengths")
        if code_lengths is not None:
            if (not torch.is_tensor(code_lengths) or code_lengths.shape != (audio_codes.shape[0], )):
                raise ValueError("Vui `audio_code_lengths` must have shape [batch].")
            code_lengths = code_lengths.to(device=device, dtype=torch.long)
            if bool(((code_lengths <= 0) | (code_lengths > audio_codes.shape[-1])).any()):
                raise ValueError(
                    "Vui `audio_code_lengths` must be positive and no larger "
                    "than the padded audio length.")
            dense_valid = (
                torch.arange(audio_codes.shape[-1], device=device)[None, None, :]
                < code_lengths[:, None, None]).expand(-1, codebook_count, -1)
            valid_sequence, _, _ = pattern.build_pattern_sequence(
                dense_valid.long(),
                0,
            )
            valid_targets = valid_targets & valid_sequence[..., 1:].bool()

        audio_embeddings = sum(
            embedding(model_codes[:, index])
            for index, embedding in enumerate(self.primary_model.audio_embeddings)) / codebook_count
        text_embeddings = self.primary_model.token_emb(input_ids)
        embeddings = torch.cat((text_embeddings, audio_embeddings), dim=1)
        total_length = embeddings.shape[1]
        if total_length > self.primary_model.decoder.max_seqlen:
            raise ValueError(
                "Vui training sequence exceeds the model context window: "
                f"{total_length} > {self.primary_model.decoder.max_seqlen}.")

        key_mask = torch.cat(
            (
                text_mask,
                torch.ones(
                    audio_embeddings.shape[:2],
                    device=device,
                    dtype=torch.bool,
                ),
            ),
            dim=1,
        )
        causal = torch.ones(
            total_length,
            total_length,
            device=device,
            dtype=torch.bool,
        ).tril()
        attention_mask = causal[None, None] & key_mask[:, None, None, :]
        text_positions = text_mask.long().cumsum(dim=1).sub(1).clamp_min(0)
        audio_positions = (
            text_mask.sum(dim=1, keepdim=True) +
            torch.arange(audio_embeddings.shape[1], device=device)[None, :])
        positions = torch.cat((text_positions, audio_positions), dim=1)
        hidden_states = self.primary_model.decoder(
            embeddings,
            positions,
            attn_mask=attention_mask,
        )
        audio_hidden_states = hidden_states[:, input_ids.shape[1]:]
        logits = torch.stack(
            [head(audio_hidden_states) for head in self.primary_model.audio_heads],
            dim=1,
        ).float()
        labels = targets.masked_fill(~valid_targets, -100)
        codec_loss = functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
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
                "model_type": "vui",
                "objective": "delayed-codebook-causal-ce",
                "supervised_tokens": int(valid_targets.sum().item()),
            },
            training_phase=context.phase.name,
            optimizer_names=context.phase.optimizer_names,
        )

    def save_pretrained(self, save_directory) -> None:
        """Export a standalone native Vui + Fluac Safetensors directory."""
        self.setup()
        from voicehub.models.vui.checkpoint import export_vui_pretrained

        export_vui_pretrained(
            self.primary_model,
            save_directory,
            wrapper_config=getattr(self.model, "config", None),
        )


__all__ = ["VuiTrainingAdapter"]
