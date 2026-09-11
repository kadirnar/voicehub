"""Model-local Qwen3-TTS source-native training adapter."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicehub.dependencies import import_optional
from voicehub.outputs import TTSTrainingOutput
from voicehub.training.adapters import CausalLMTrainingAdapter
from voicehub.training.contracts import TrainingContext
from voicehub.training.recipes import SourceRecipeTrainingAdapter


class Qwen3TTSTrainingAdapter(
        CausalLMTrainingAdapter,
        SourceRecipeTrainingAdapter,
):
    """Official Qwen3-TTS 12 Hz single-speaker SFT objective."""

    native_export_semantics = "inference-export"
    _DEFAULT_LORA_TARGETS = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )

    def __init__(self, model, spec):
        super().__init__(model, spec)
        self._target_speaker_embedding = None
        self._lora_injection = None

    def _configured_lora_targets(self) -> tuple[str, ...]:
        return tuple(
            getattr(
                self.model.config,
                "training_lora_target_modules",
                self._DEFAULT_LORA_TARGETS,
            ))

    def setup(self):
        if self.is_ready:
            return super().setup()
        super().setup()
        model_type = str(getattr(self.primary_model.config, "tts_model_type", "")).lower()
        speaker_encoder = getattr(self.primary_model, "speaker_encoder", None)
        if model_type != "base" or speaker_encoder is None:
            raise ValueError(
                "Qwen3-TTS fine-tuning requires a 12 Hz Base checkpoint with "
                "its speaker encoder. CustomVoice and VoiceDesign artifacts "
                "are inference/export targets, not valid SFT starting points.")
        for parameter in speaker_encoder.parameters():
            parameter.requires_grad_(False)
        lora_rank = getattr(self.model.config, "training_lora_rank", None)
        if lora_rank is not None:
            from voicehub.models.qwen3tts.lora import inject_qwen3_tts_lora

            self._lora_injection = inject_qwen3_tts_lora(
                self.primary_model,
                rank=lora_rank,
                alpha=getattr(
                    self.model.config,
                    "training_lora_alpha",
                    16.0,
                ),
                dropout=getattr(
                    self.model.config,
                    "training_lora_dropout",
                    0.0,
                ),
                target_modules=self._configured_lora_targets(),
                seed=getattr(
                    self.model.config,
                    "training_lora_seed",
                    0,
                ),
            )
        return self

    def recipe_resume_configuration(self):
        configuration = dict(super().recipe_resume_configuration())
        configuration["resolved_sub_talker_loss_weight"] = float(
            getattr(self.model.config, "sub_talker_loss_weight", 0.3), )
        lora_rank = getattr(self.model.config, "training_lora_rank", None)
        configuration.update({
            "parameter_efficient": lora_rank is not None,
            "lora_topology": None if lora_rank is None else {
                "adapter_library":
                "voicehub-native",
                "base_model_frozen":
                True,
                "speaker_encoder_frozen":
                True,
                "decoder_stacks": (
                    "talker",
                    "residual-code-predictor",
                ),
                "injected_module_names":
                ([] if self._lora_injection is None else list(self._lora_injection.module_names)),
            },
        })
        return configuration

    def artifact_manifest(self) -> dict[str, Any]:
        manifest = super().artifact_manifest()
        if getattr(self.model.config, "training_lora_rank", None) is not None:
            manifest["checkpoint_semantics"]["lora_adapter"] = ("strict-adapter-only-safetensors")
            manifest["checkpoint_semantics"]["save_pretrained"] = (
                "clone-merged-inference-export-plus-adapter")
        return manifest

    def create_dataset(self, records, **kwargs):
        del kwargs
        self.setup()
        training = import_optional(
            "voicehub.models.qwen3tts.training",
            model_type="qwen3tts",
            install_extra="training",
        )
        return training.Qwen3TTSSFTDataset(
            records,
            self.model.model.processor,
            self.primary_model.config,
        )

    def execute_training_phase(
        self,
        context: TrainingContext,
    ) -> TTSTrainingOutput:
        self.setup()
        batch = dict(context.inputs)
        required = (
            "input_ids",
            "codec_ids",
            "ref_mels",
            "text_embedding_mask",
            "codec_embedding_mask",
            "attention_mask",
            "codec_0_labels",
            "codec_mask",
        )
        missing = [name for name in required if name not in batch]
        if missing:
            raise ValueError("Qwen3-TTS fine-tuning requires: " + ", ".join(missing))
        model = self.primary_model
        speaker_embedding = model.speaker_encoder(
            batch["ref_mels"].to(
                device=model.device,
                dtype=model.dtype,
            )).detach()
        if self._target_speaker_embedding is None:
            self._target_speaker_embedding = speaker_embedding[0].detach().clone()

        input_ids = batch["input_ids"]
        codec_ids = batch["codec_ids"]
        input_text_ids = input_ids[:, :, 0]
        input_codec_ids = input_ids[:, :, 1]
        text_embeddings = model.talker.get_text_embeddings()(input_text_ids)
        text_embeddings = (model.talker.text_projection(text_embeddings) * batch["text_embedding_mask"])
        codec_embeddings = (
            model.talker.model.codec_embedding(input_codec_ids) * batch["codec_embedding_mask"])
        codec_embeddings = codec_embeddings.clone()
        codec_embeddings[:, 6, :] = speaker_embedding
        input_embeddings = text_embeddings + codec_embeddings

        codec_mask = batch["codec_mask"]
        for channel_index in range(1, codec_ids.shape[-1]):
            channel_embedding = (
                model.talker.code_predictor.get_input_embeddings()[channel_index - 1](
                    codec_ids[:, :, channel_index]))
            input_embeddings = (input_embeddings + channel_embedding * codec_mask.unsqueeze(-1))

        outputs = model.talker(
            inputs_embeds=input_embeddings,
            attention_mask=batch["attention_mask"],
            labels=batch["codec_0_labels"],
            output_hidden_states=False,
        )
        hidden_states = outputs.last_hidden_state
        # The native causal loss uses hidden state t - 1 to predict target t.
        # Preserve that same pairing for the per-frame sub-talker objective.
        next_codec_mask = codec_mask[:, 1:]
        talker_hidden = hidden_states[:, :-1][next_codec_mask]
        talker_codec_ids = codec_ids[:, 1:][next_codec_mask]
        sub_logits, sub_loss = model.talker.forward_sub_talker_finetune(
            talker_codec_ids,
            talker_hidden,
        )
        sub_weight = float(getattr(self.model.config, "sub_talker_loss_weight", 0.3))
        if not math.isfinite(sub_weight) or sub_weight < 0:
            raise ValueError("Qwen3-TTS sub-talker loss weight must be finite and "
                             "non-negative.")
        loss = outputs.loss + sub_weight * sub_loss
        return self._training_output(
            context,
            loss=loss,
            losses={
                "loss": loss,
                "talker_loss": outputs.loss,
                "sub_talker_loss": sub_loss,
            },
            logits=(outputs.logits, sub_logits),
            metadata={
                "parameter_efficient": self._lora_injection is not None,
                "lora_adapter_library": (None if self._lora_injection is None else "voicehub-native"),
            },
        )

    def recipe_state_dict(self) -> Mapping[str, Any]:
        return {
            "target_speaker_embedding": (
                None if self._target_speaker_embedding is None else
                self._target_speaker_embedding.detach().clone())
        }

    def load_recipe_state_dict(
        self,
        state_dict: Mapping[str, Any],
        *,
        strict: bool = True,
    ) -> None:
        if not state_dict:
            return
        if strict and set(state_dict) != {"target_speaker_embedding"}:
            raise ValueError("Qwen3-TTS recipe state must contain only "
                             "'target_speaker_embedding'.")
        embedding = state_dict.get("target_speaker_embedding")
        self._target_speaker_embedding = (None if embedding is None else embedding.detach().clone())

    def save_pretrained(self, save_directory) -> None:
        """Write a directly loadable Hugging Face safetensors directory."""
        self.setup()
        if self._target_speaker_embedding is None:
            return
        model = self.primary_model
        speaker_id = int(getattr(self.model.config, "training_speaker_id", 3000))
        speaker_name = str(getattr(self.model.config, "training_speaker_name", "voicehub"))
        embedding = model.talker.model.codec_embedding.weight
        if not 0 <= speaker_id < embedding.shape[0]:
            raise ValueError(
                f"Qwen3-TTS speaker id {speaker_id} is outside the codec "
                f"embedding table of size {embedding.shape[0]}.")
        talker_config = model.config.talker_config
        destination = Path(save_directory)
        if self._lora_injection is None:
            state_dict = {
                name: value
                for name, value in model.state_dict().items() if not name.startswith("speaker_encoder.")
            }
        else:
            from voicehub.models.qwen3tts.lora import merged_qwen3_tts_state_dict

            state_dict = merged_qwen3_tts_state_dict(
                model,
                self._lora_injection,
                drop_prefixes=("speaker_encoder.", ),
            )
        embedding_module = model.talker.model.codec_embedding
        embedding_key = next(
            (f"{name}.weight" for name, module in model.named_modules() if module is embedding_module),
            None,
        )
        if embedding_key is None or embedding_key not in state_dict:
            raise RuntimeError(
                "Qwen3-TTS export could not locate the talker codec "
                "embedding in the model state.")
        exported_embedding = state_dict[embedding_key].detach().clone()
        exported_embedding[speaker_id].copy_(
            self._target_speaker_embedding.to(
                device=exported_embedding.device,
                dtype=exported_embedding.dtype,
            ))
        state_dict[embedding_key] = exported_embedding

        original_speaker_ids = talker_config.spk_id
        original_dialects = talker_config.spk_is_dialect
        original_model_type = model.config.tts_model_type
        try:
            talker_config.spk_id = {speaker_name: speaker_id}
            talker_config.spk_is_dialect = {speaker_name: False}
            model.config.tts_model_type = "custom_voice"
            model.save_pretrained(
                destination,
                state_dict=state_dict,
                safe_serialization=True,
            )
        finally:
            talker_config.spk_id = original_speaker_ids
            talker_config.spk_is_dialect = original_dialects
            model.config.tts_model_type = original_model_type
        processor = getattr(self.model.model, "processor", None)
        if processor is not None and hasattr(processor, "save_pretrained"):
            processor.save_pretrained(destination)
        speech_tokenizer = getattr(model, "speech_tokenizer", None)
        speech_model = getattr(speech_tokenizer, "model", None)
        feature_extractor = getattr(speech_tokenizer, "feature_extractor", None)
        speech_directory = destination / "speech_tokenizer"
        if speech_model is None or feature_extractor is None:
            raise ValueError(
                "Qwen3-TTS export requires the loaded speech tokenizer model "
                "and feature extractor.")
        speech_model.save_pretrained(
            speech_directory,
            safe_serialization=True,
        )
        feature_extractor.save_pretrained(speech_directory)
        if (self._lora_injection is not None and bool(getattr(
                self.model.config,
                "training_lora_export_adapter",
                True,
        ))):
            from voicehub.models.qwen3tts.lora import save_qwen3_tts_lora_adapter

            save_qwen3_tts_lora_adapter(
                self._lora_injection,
                destination / "lora_adapter",
                target_modules=self._configured_lora_targets(),
                base_model=(getattr(self.model.config, "name_or_path", None) or None),
                target_speaker_embedding=self._target_speaker_embedding,
                speaker_name=speaker_name,
                speaker_id=speaker_id,
            )

    def load_lora_adapter(self, directory):
        """Restore a strict adapter-only export into this training graph."""
        self.setup()
        if self._lora_injection is None:
            raise RuntimeError(
                "Enable Qwen3-TTS LoRA in the training configuration before "
                "loading an adapter-only checkpoint.")
        from voicehub.models.qwen3tts.lora import load_qwen3_tts_lora_adapter

        result = load_qwen3_tts_lora_adapter(
            self._lora_injection,
            directory,
            target_modules=self._configured_lora_targets(),
            expected_base_model=(getattr(self.model.config, "name_or_path", None) or None),
            expected_speaker_name=str(getattr(
                self.model.config,
                "training_speaker_name",
                "voicehub",
            )),
            expected_speaker_id=int(getattr(
                self.model.config,
                "training_speaker_id",
                3000,
            )),
            expected_speaker_embedding_size=int(
                self.primary_model.talker.model.codec_embedding.weight.shape[1]),
        )
        self._target_speaker_embedding = result.target_speaker_embedding
        return result


__all__ = ["Qwen3TTSTrainingAdapter"]
