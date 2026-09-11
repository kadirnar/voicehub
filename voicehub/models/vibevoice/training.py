"""Official causal-language and diffusion objectives for VibeVoice 1.5B."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

from voicehub.dependencies import import_optional
from voicehub.outputs import TTSTrainingOutput
from voicehub.training.adapters import CompositeTrainingAdapter
from voicehub.training.contracts import TrainingContext
from voicehub.training.datasets import SpeechDataset


class VibeVoicePreprocessedCollator:
    """Pad utterances while flattening their source-native speech segments.

    VibeVoice represents text per utterance but acoustic/semantic
    features per speech segment. A generic tensor stack introduces an
    invalid extra batch dimension for the segment tensors, so this
    collator concatenates segments in the same order as their acoustic
    placeholders.
    """

    def __init__(self, *, pad_token_id: int = 0):
        if (isinstance(pad_token_id, bool) or not isinstance(pad_token_id, int) or pad_token_id < 0):
            raise ValueError("VibeVoice `pad_token_id` must be a non-negative integer.")
        self.pad_token_id = pad_token_id

    def resume_fingerprint(self) -> dict[str, object]:
        return {
            "kind": "vibevoice-preprocessed",
            "pad_token_id": self.pad_token_id,
            "segment_batching": "flatten",
        }

    @staticmethod
    def _tensor(torch, record, name: str, *, ndim: int):
        value = record.get(name)
        if not torch.is_tensor(value) or value.ndim != ndim:
            raise ValueError(f"Each VibeVoice record must provide `{name}` with {ndim} dimensions.")
        return value

    def __call__(self, features) -> dict[str, object]:
        torch = import_optional(
            "torch",
            model_type="vibevoice",
            install_extra="training",
        )
        functional = import_optional(
            "torch.nn.functional",
            model_type="vibevoice",
            install_extra="training",
        )
        if not features:
            return {}
        if any(not isinstance(feature, Mapping) for feature in features):
            raise TypeError("VibeVoice training records must be mappings.")

        normalized = []
        semantic_size = None
        max_sequence = 0
        max_samples = 0
        max_latents = 0
        for index, feature in enumerate(features):
            record = dict(feature)
            input_ids = self._tensor(torch, record, "input_ids", ndim=1)
            attention_mask = self._tensor(
                torch,
                record,
                "attention_mask",
                ndim=1,
            )
            acoustic_input_mask = self._tensor(
                torch,
                record,
                "acoustic_input_mask",
                ndim=1,
            )
            acoustic_loss_mask = self._tensor(
                torch,
                record,
                "acoustic_loss_mask",
                ndim=1,
            )
            if not (input_ids.shape == attention_mask.shape == acoustic_input_mask.shape ==
                    acoustic_loss_mask.shape):
                raise ValueError(f"VibeVoice record {index} token fields must share one shape.")

            speech_tensors = self._tensor(
                torch,
                record,
                "speech_tensors",
                ndim=2,
            )
            speech_masks = self._tensor(
                torch,
                record,
                "speech_masks",
                ndim=2,
            )
            loss_selection = self._tensor(
                torch,
                record,
                "speeches_loss_input",
                ndim=2,
            )
            semantics = self._tensor(
                torch,
                record,
                "speech_semantic_tensors",
                ndim=3,
            )
            segment_count = speech_tensors.shape[0]
            if (segment_count == 0 or speech_masks.shape[0] != segment_count or
                    loss_selection.shape != speech_masks.shape or semantics.shape[:2] != speech_masks.shape):
                raise ValueError(f"VibeVoice record {index} speech segment fields are not aligned.")
            if semantic_size is None:
                semantic_size = semantics.shape[-1]
            elif semantics.shape[-1] != semantic_size:
                raise ValueError("All VibeVoice semantic tensors must use one feature size.")

            acoustic_count = int(acoustic_input_mask.bool().sum().item())
            latent_count = int(speech_masks.bool().sum().item())
            target_count = int(acoustic_loss_mask.bool().sum().item())
            selected_count = int((loss_selection.bool() & speech_masks.bool()).sum().item())
            if acoustic_count != latent_count:
                raise ValueError(
                    f"VibeVoice record {index} has {acoustic_count} acoustic "
                    f"placeholders but {latent_count} valid speech latents.")
            if target_count == 0 or target_count != selected_count:
                raise ValueError(
                    f"VibeVoice record {index} has {target_count} target "
                    f"placeholders but {selected_count} selected latents.")

            normalized.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "acoustic_input_mask": acoustic_input_mask,
                "acoustic_loss_mask": acoustic_loss_mask,
                "speech_tensors": speech_tensors,
                "speech_masks": speech_masks,
                "speeches_loss_input": loss_selection,
                "speech_semantic_tensors": semantics,
            })
            max_sequence = max(max_sequence, input_ids.shape[0])
            max_samples = max(max_samples, speech_tensors.shape[1])
            max_latents = max(max_latents, speech_masks.shape[1])

        def pad_sequence(value, padding_value):
            return functional.pad(
                value,
                (0, max_sequence - value.shape[0]),
                value=padding_value,
            )

        input_ids = torch.stack(
            [pad_sequence(record["input_ids"].long(), self.pad_token_id) for record in normalized])
        attention_mask = torch.stack(
            [pad_sequence(record["attention_mask"].long(), 0) for record in normalized])
        acoustic_input_mask = torch.stack(
            [pad_sequence(record["acoustic_input_mask"].bool(), False) for record in normalized])
        acoustic_loss_mask = torch.stack(
            [pad_sequence(record["acoustic_loss_mask"].bool(), False) for record in normalized])

        padded_speech_tensors = [
            functional.pad(
                record["speech_tensors"],
                (0, max_samples - record["speech_tensors"].shape[1]),
                value=0,
            ) for record in normalized
        ]
        speech_tensors = torch.cat(padded_speech_tensors, dim=0)

        padded_speech_masks = [
            functional.pad(
                record["speech_masks"].bool(),
                (0, max_latents - record["speech_masks"].shape[1]),
                value=False,
            ) for record in normalized
        ]
        speech_masks = torch.cat(padded_speech_masks, dim=0)

        padded_speeches_loss_input = [
            functional.pad(
                record["speeches_loss_input"].bool(),
                (0, max_latents - record["speeches_loss_input"].shape[1]),
                value=False,
            ) for record in normalized
        ]
        speeches_loss_input = torch.cat(padded_speeches_loss_input, dim=0)

        padded_speech_semantic_tensors = [
            functional.pad(
                record["speech_semantic_tensors"],
                (
                    0,
                    0,
                    0,
                    max_latents - record["speech_semantic_tensors"].shape[1],
                ),
                value=0,
            ) for record in normalized
        ]
        speech_semantic_tensors = torch.cat(padded_speech_semantic_tensors, dim=0)

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "speech_tensors": speech_tensors,
            "speech_masks": speech_masks,
            "speech_semantic_tensors": speech_semantic_tensors,
            "acoustic_input_mask": acoustic_input_mask,
            "acoustic_loss_mask": acoustic_loss_mask,
            "speeches_loss_input": speeches_loss_input,
        }
        phases = [feature.get("training_phase") for feature in features]
        if any(phase is not None for phase in phases):
            if any(phase != phases[0] for phase in phases):
                raise ValueError("Every VibeVoice record in a batch must use one training phase.")
            batch["training_phase"] = phases[0]
        return batch


def mask_text_labels(
    labels,
    attention_mask,
    acoustic_input_mask,
    *,
    ignore_index: int = -100,
):
    """Build the source recipe's next-token, non-acoustic CE labels."""
    shifted = labels[:, 1:].contiguous()
    valid = attention_mask[:, 1:].eq(1)
    valid = valid & ~acoustic_input_mask[:, 1:].bool()
    output = shifted.clone()
    output[~valid] = ignore_index
    return output


class VibeVoiceTrainingAdapter(CompositeTrainingAdapter):
    """Train the verified non-streaming VibeVoice 1.5B graph.

    The realtime 0.5B runtime intentionally remains rejected: it exposes
    a streaming generation graph, not the combined LM/diffusion training
    forward used here.
    """

    supports_custom_recipe = True
    native_export_semantics = ("voicehub-native-vibevoice-full-model-safetensors-and-processor")

    _REQUIRED_INPUTS = (
        "input_ids",
        "attention_mask",
        "speech_tensors",
        "speech_masks",
        "speeches_loss_input",
        "speech_semantic_tensors",
        "acoustic_input_mask",
        "acoustic_loss_mask",
    )

    def __init__(self, model, spec):
        super().__init__(model, spec)
        pad_token_id = int(getattr(
            getattr(model, "config", None),
            "training_pad_token_id",
            0,
        ))
        self.data_collator = VibeVoicePreprocessedCollator(pad_token_id=pad_token_id, )

    def setup(self):
        super().setup()
        runtime = getattr(self.primary_model, "model", None)
        if runtime is None:
            raise RuntimeError("VibeVoice training runtime has no base model.")
        for name in ("acoustic_tokenizer", "semantic_tokenizer"):
            tokenizer = getattr(runtime, name, None)
            if tokenizer is None:
                raise RuntimeError(f"VibeVoice training runtime is missing `{name}`.")
            tokenizer.eval()
            for parameter in tokenizer.parameters():
                parameter.requires_grad_(False)
        self.primary_model.train()
        # ``train()`` recursively toggles frozen tokenizers, so restore their
        # evaluation behavior after changing the root mode.
        runtime.acoustic_tokenizer.eval()
        runtime.semantic_tokenizer.eval()
        processor = getattr(self.model, "_processor", None)
        tokenizer = getattr(processor, "tokenizer", None)
        checkpoint_pad_id = getattr(tokenizer, "pad_token_id", None)
        if (isinstance(checkpoint_pad_id, int) and not isinstance(checkpoint_pad_id, bool) and
                checkpoint_pad_id >= 0):
            self.data_collator.pad_token_id = checkpoint_pad_id
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            runtime = self.primary_model.model
            runtime.acoustic_tokenizer.eval()
            runtime.semantic_tokenizer.eval()
        return self

    def create_dataset(self, records, **kwargs):
        """Validate records carrying source-native, preprocessed tensors."""
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"VibeVoice's preprocessed dataset does not accept options: {unknown}.")
        dataset = SpeechDataset(
            records,
            required_fields=self._REQUIRED_INPUTS,
        )
        dataset.collate_fn = self.data_collator
        return dataset

    @staticmethod
    def _validate_batch(torch, batch: Mapping[str, object]) -> None:
        missing = [name for name in VibeVoiceTrainingAdapter._REQUIRED_INPUTS if batch.get(name) is None]
        if missing:
            raise ValueError("VibeVoice 1.5B fine-tuning requires: " + ", ".join(missing))
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        acoustic_input_mask = batch["acoustic_input_mask"]
        acoustic_loss_mask = batch["acoustic_loss_mask"]
        for name, value in (
            ("input_ids", input_ids),
            ("attention_mask", attention_mask),
            ("acoustic_input_mask", acoustic_input_mask),
            ("acoustic_loss_mask", acoustic_loss_mask),
        ):
            if not torch.is_tensor(value) or value.ndim != 2:
                raise ValueError(f"VibeVoice `{name}` must have shape [batch, sequence].")
        if not (input_ids.shape == attention_mask.shape == acoustic_input_mask.shape ==
                acoustic_loss_mask.shape):
            raise ValueError(
                "VibeVoice token IDs, attention mask, and acoustic masks must "
                "share one shape.")
        if input_ids.shape[1] < 2:
            raise ValueError("VibeVoice sequences require at least two tokens.")
        if bool(acoustic_loss_mask[:, :2].any()):
            raise ValueError(
                "VibeVoice acoustic targets cannot occupy either of the first "
                "two positions in the source recipe.")

        speech_tensors = batch["speech_tensors"]
        speech_masks = batch["speech_masks"]
        loss_selection = batch["speeches_loss_input"]
        semantics = batch["speech_semantic_tensors"]
        if not torch.is_tensor(speech_tensors) or speech_tensors.ndim != 2:
            raise ValueError("VibeVoice `speech_tensors` must have shape [segments, samples].")
        if not torch.is_tensor(speech_masks) or speech_masks.ndim != 2:
            raise ValueError("VibeVoice `speech_masks` must have shape [segments, latent_time].")
        if (not torch.is_tensor(loss_selection) or loss_selection.shape != speech_masks.shape):
            raise ValueError("VibeVoice `speeches_loss_input` must match `speech_masks`.")
        if (not torch.is_tensor(semantics) or semantics.ndim != 3 or
                semantics.shape[:2] != speech_masks.shape):
            raise ValueError(
                "VibeVoice `speech_semantic_tensors` must have shape "
                "[segments, latent_time, semantic_size].")
        if speech_tensors.shape[0] != speech_masks.shape[0]:
            raise ValueError("VibeVoice waveform and latent-mask segment counts must match.")
        acoustic_count = int(acoustic_input_mask.bool().sum().item())
        latent_count = int(speech_masks.bool().sum().item())
        if acoustic_count != latent_count:
            raise ValueError(
                "VibeVoice acoustic placeholder count must equal the number "
                f"of valid speech latents ({acoustic_count} != {latent_count}).")
        target_count = int(acoustic_loss_mask.bool().sum().item())
        selected_count = int((loss_selection.bool() & speech_masks.bool()).sum().item())
        if target_count == 0 or target_count != selected_count:
            raise ValueError(
                "VibeVoice target placeholders must map one-to-one to selected "
                f"diffusion latents ({target_count} != {selected_count}).")

    def execute_training_phase(
        self,
        context: TrainingContext,
    ) -> TTSTrainingOutput:
        self.setup()
        torch = import_optional(
            "torch",
            model_type="vibevoice",
            install_extra="training",
        )
        functional = import_optional(
            "torch.nn.functional",
            model_type="vibevoice",
            install_extra="training",
        )
        batch = dict(context.inputs)
        self._validate_batch(torch, batch)
        device = next(self.primary_model.parameters()).device
        model_dtype = next(self.primary_model.parameters()).dtype
        model_inputs = {
            name:
            value.to(
                device=device,
                dtype=(model_dtype if name in (
                    "speech_tensors",
                    "speech_semantic_tensors",
                ) else None),
            )
            for name, value in batch.items() if name in self._REQUIRED_INPUTS
        }

        config = self.model.config
        ddpm_batch_mul = getattr(config, "training_ddpm_batch_mul", 1)
        if (isinstance(ddpm_batch_mul, bool) or not isinstance(ddpm_batch_mul, int) or ddpm_batch_mul <= 0):
            raise ValueError("VibeVoice `training_ddpm_batch_mul` must be a positive integer.")
        outputs = self.primary_model(
            **model_inputs,
            ddpm_batch_mul=ddpm_batch_mul,
            use_cache=False,
        )
        logits = outputs.logits
        labels = mask_text_labels(
            model_inputs["input_ids"],
            model_inputs["attention_mask"],
            model_inputs["acoustic_input_mask"],
        )
        if not bool(labels.ne(-100).any()):
            raise ValueError("VibeVoice batch has no non-acoustic next-token targets for CE.")
        shifted_logits = logits[:, :-1].contiguous()
        ce_loss = functional.cross_entropy(
            shifted_logits.view(-1, shifted_logits.shape[-1]),
            labels.view(-1),
            ignore_index=-100,
        )
        diffusion_loss = outputs.diffusion_loss
        if diffusion_loss is None:
            raise RuntimeError("VibeVoice 1.5B forward did not return its diffusion loss.")

        ce_weight = float(getattr(config, "training_ce_loss_weight", 1.0))
        diffusion_weight = float(getattr(config, "training_diffusion_loss_weight", 1.0))
        if (not math.isfinite(ce_weight) or ce_weight < 0 or not math.isfinite(diffusion_weight) or
                diffusion_weight < 0):
            raise ValueError("VibeVoice training loss weights must be finite and non-negative.")
        if ce_weight == 0 and diffusion_weight == 0:
            raise ValueError("At least one VibeVoice training loss weight must be positive.")
        loss = ce_weight * ce_loss + diffusion_weight * diffusion_loss
        return TTSTrainingOutput(
            loss=loss,
            logits=logits,
            losses={
                "loss": loss,
                "ce_loss": ce_loss,
                "diffusion_loss": diffusion_loss,
            },
            metadata={
                "model_type": "vibevoice",
                "checkpoint_family": "non-streaming-1.5b",
                "objective": "masked-causal-ce-plus-diffusion",
                "ddpm_batch_mul": ddpm_batch_mul,
            },
            training_phase=context.phase.name,
            optimizer_names=context.phase.optimizer_names,
        )

    def save_pretrained(self, save_directory) -> None:
        """Export a complete native Safetensors checkpoint and processor."""
        self.setup()
        destination = Path(save_directory)
        native_export = getattr(
            self.model,
            "export_native_pretrained",
            None,
        )
        if callable(native_export):
            native_export(destination)
            return

        # Compatibility for controlled adapter tests and legacy wrappers.
        self.primary_model.save_pretrained(
            destination,
            safe_serialization=True,
        )
        processor = getattr(self.model, "_processor", None)
        if processor is None or not hasattr(processor, "save_pretrained"):
            raise RuntimeError("VibeVoice export requires its loaded processor.")
        processor.save_pretrained(destination)


__all__ = [
    "VibeVoicePreprocessedCollator",
    "VibeVoiceTrainingAdapter",
    "mask_text_labels",
]
