"""Source-native Fish Speech text-to-semantic fine-tuning.

Fish Speech S2 trains the text-to-semantic transformer while treating
the audio codec as an offline tokenizer.  The upstream dataset has
already aligned each label with the prediction at the same sequence
position; this adapter therefore deliberately does not apply another
causal shift.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from voicehub.checkpointing import save_safetensors
from voicehub.dependencies import import_optional
from voicehub.outputs import TTSTrainingOutput
from voicehub.training.adapters import CausalLMTrainingAdapter
from voicehub.training.contracts import TrainingContext
from voicehub.training.recipes import SourceRecipeTrainingAdapter

CODEBOOK_PAD_TOKEN_ID = 0
LABEL_IGNORE_ID = -100


def _get_value(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


class FishSemanticDataset:
    """Validated in-memory dataset for pre-tokenized Fish Speech examples.

    Each record must contain ``tokens`` (or ``inputs``) and ``labels``
    with shape ``[num_codebooks + 1, sequence_length]``.  Audio should
    be encoded with the checkpoint's Fish codec before constructing
    these records.
    """

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        tokenizer,
        num_codebooks: int,
        max_length: int = 4096,
    ) -> None:
        if not records:
            raise ValueError("FishSemanticDataset requires at least one record.")
        if num_codebooks <= 0:
            raise ValueError("num_codebooks must be greater than zero.")
        if max_length <= 0:
            raise ValueError("max_length must be greater than zero.")
        self.records = tuple(dict(record) for record in records)
        self.num_codebooks = int(num_codebooks)
        self.collate_fn = FishTextDataCollator(
            tokenizer=tokenizer,
            max_length=max_length,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        torch = import_optional(
            "torch",
            model_type="fishtts",
            install_extra="training",
        )
        record = self.records[index]
        tokens = record.get("tokens", record.get("inputs"))
        labels = record.get("labels")
        missing = []
        if tokens is None:
            missing.append("tokens")
        if labels is None:
            missing.append("labels")
        if missing:
            raise ValueError(f"Fish Speech record {index} is missing: {', '.join(missing)}.")

        tokens = torch.as_tensor(tokens, dtype=torch.long)
        labels = torch.as_tensor(labels, dtype=torch.long)
        expected_channels = self.num_codebooks + 1
        if tokens.ndim != 2 or tokens.shape[0] != expected_channels:
            raise ValueError(
                "Fish Speech tokens must have shape "
                f"[{expected_channels}, sequence_length], received "
                f"{tuple(tokens.shape)} for record {index}.")
        if labels.shape != tokens.shape:
            raise ValueError(
                "Fish Speech labels must have the same shape as tokens; "
                f"received {tuple(labels.shape)} and {tuple(tokens.shape)} "
                f"for record {index}.")
        return {
            "tokens": tokens.contiguous(),
            "labels": labels.contiguous(),
        }


class FishTextDataCollator:
    """Pad Fish's channel-first token layout exactly like the source recipe.

    ``attention_masks`` uses the source model's key-padding convention:
    ``False`` is a real token and ``True`` is padding.
    """

    def __init__(self, tokenizer, max_length: int = 4096) -> None:
        if max_length <= 0:
            raise ValueError("max_length must be greater than zero.")
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __call__(self, examples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not examples:
            raise ValueError("Cannot collate an empty Fish Speech batch.")
        torch = import_optional(
            "torch",
            model_type="fishtts",
            install_extra="training",
        )
        functional = torch.nn.functional
        normalized = []
        channel_count = None
        maximum = 0
        for index, example in enumerate(examples):
            tokens = example.get("tokens", example.get("inputs"))
            labels = example.get("labels")
            if tokens is None or labels is None:
                raise ValueError(f"Fish Speech example {index} requires tokens and labels.")
            tokens = torch.as_tensor(tokens, dtype=torch.long)
            labels = torch.as_tensor(labels, dtype=torch.long)
            if tokens.ndim != 2 or labels.shape != tokens.shape:
                raise ValueError(
                    "Fish Speech examples require equal rank-2 tokens and "
                    f"labels; received {tuple(tokens.shape)} and "
                    f"{tuple(labels.shape)} at index {index}.")
            if channel_count is None:
                channel_count = tokens.shape[0]
            elif tokens.shape[0] != channel_count:
                raise ValueError(
                    "Every Fish Speech example must use the same number of "
                    "codebook channels.")
            maximum = max(maximum, int(tokens.shape[1]))
            normalized.append((tokens, labels))

        padded_length = min(maximum, self.max_length)
        if padded_length == 0:
            raise ValueError("Fish Speech examples cannot have zero length.")
        end_of_text_id = int(self.tokenizer.get_token_id("<|end_of_text|>"))
        inputs = []
        labels_batch = []
        attention_masks = []
        for tokens, labels in normalized:
            tokens = tokens[:, :padded_length]
            labels = labels[:, :padded_length]
            length = int(tokens.shape[1])
            padding = padded_length - length
            if padding:
                tokens = functional.pad(
                    tokens,
                    (0, padding),
                    value=end_of_text_id,
                )
                tokens[1:, length:] = CODEBOOK_PAD_TOKEN_ID
                labels = functional.pad(
                    labels,
                    (0, padding),
                    value=LABEL_IGNORE_ID,
                )
            key_padding_mask = torch.ones(
                padded_length,
                dtype=torch.bool,
            )
            key_padding_mask[:length] = False
            inputs.append(tokens)
            labels_batch.append(labels)
            attention_masks.append(key_padding_mask)

        return {
            "inputs": torch.stack(inputs),
            "attention_masks": torch.stack(attention_masks),
            "labels": torch.stack(labels_batch),
        }

    def resume_fingerprint(self) -> dict[str, Any]:
        return {
            "max_length": self.max_length,
            "end_of_text_id": int(self.tokenizer.get_token_id("<|end_of_text|>")),
        }


class FishSpeechTrainingAdapter(
        CausalLMTrainingAdapter,
        SourceRecipeTrainingAdapter,
):
    """Fine-tune Fish Speech's semantic transformer from source checkpoints.

    The primary language-model head predicts text and the first semantic
    codebook.  The residual head predicts all codec channels only at
    semantic positions.  Both Naive/full-sequence logits ``[B, T, C,
    V]`` and DualAR's filtered logits ``[N_semantic, C, V]`` are
    accepted.
    """

    supports_custom_recipe = True
    native_export_semantics = "inference-export"

    def validate_support(self) -> None:
        # Keep artifact checks unconditional: a preloaded inference runtime
        # must not bypass the GGUF/quantization guards in the cold loader.
        super().validate_support()

    def setup(self):
        """Resolve the wrapper's already-native differentiable graph."""
        self._freeze_codec()
        super().setup()
        self._freeze_codec()
        return self

    def _freeze_codec(self) -> None:
        codec = getattr(self.model, "_codec", None)
        if codec is None:
            codec = getattr(self.model, "codec", None)
        if codec is None:
            return
        if hasattr(codec, "eval"):
            codec.eval()
        if hasattr(codec, "parameters"):
            for parameter in codec.parameters():
                parameter.requires_grad_(False)

    def execute_training_phase(
        self,
        context: TrainingContext,
    ) -> TTSTrainingOutput:
        self.setup()
        if context.phase.name != "semantic":
            raise ValueError(
                "Fish Speech S2 fine-tuning trains only the text-to-semantic "
                "transformer. The released codec is an offline tokenizer and "
                "does not have an integrated training recipe.")

        batch = dict(context.inputs)
        tokens = batch.get(
            "inputs",
            batch.get("tokens", batch.get("input_ids")),
        )
        labels = batch.get("labels")
        if tokens is None or labels is None:
            raise ValueError("Fish Speech fine-tuning requires 'inputs' (or 'tokens') "
                             "and 'labels'.")

        key_padding_mask = batch.get(
            "attention_masks",
            batch.get("key_padding_mask"),
        )
        if key_padding_mask is None and "attention_mask" in batch:
            # VoiceHub/Hugging Face masks use True for real tokens, whereas
            # Fish's key-padding mask uses True for padding.
            key_padding_mask = ~batch["attention_mask"].bool()

        self._validate_batch(tokens, labels, key_padding_mask)
        forward_kwargs = {
            "inp": tokens,
            "key_padding_mask": key_padding_mask,
        }
        signature = inspect.signature(self.primary_model.forward)
        if ("labels" in signature.parameters or any(parameter.kind is inspect.Parameter.VAR_KEYWORD
                                                    for parameter in signature.parameters.values())):
            forward_kwargs["labels"] = labels
        if key_padding_mask is None:
            forward_kwargs.pop("key_padding_mask")
        outputs = self.primary_model(**forward_kwargs)

        token_logits = _get_value(outputs, "token_logits")
        if token_logits is None:
            token_logits = _get_value(outputs, "logits")
        codebook_logits = _get_value(outputs, "codebook_logits")
        if token_logits is None or codebook_logits is None:
            raise TypeError(
                "Fish Speech's semantic model must return token_logits/logits "
                "and codebook_logits. A bare BaseTransformer has no residual "
                "codebook head; use a NaiveTransformer or DualARTransformer "
                "checkpoint.")

        config = self.primary_model.config
        losses, normalized_codebook_logits, codebook_targets = (
            self.compute_source_losses(
                token_logits=token_logits,
                codebook_logits=codebook_logits,
                labels=labels,
                semantic_begin_id=int(config.semantic_begin_id),
                semantic_end_id=int(config.semantic_end_id),
                num_codebooks=int(config.num_codebooks),
            ))
        base_weight = float(
            batch.get(
                "base_loss_weight",
                getattr(self.model.config, "training_base_loss_weight", 1.0),
            ))
        semantic_weight = float(
            batch.get(
                "semantic_loss_weight",
                getattr(
                    self.model.config,
                    "training_semantic_loss_weight",
                    1.0,
                ),
            ))
        if (not math.isfinite(base_weight) or not math.isfinite(semantic_weight) or base_weight < 0 or
                semantic_weight < 0):
            raise ValueError("Fish Speech loss weights must be finite and non-negative.")
        if base_weight + semantic_weight == 0:
            raise ValueError("At least one Fish Speech loss weight must be positive.")
        total_loss = (base_weight * losses["base_loss"] + semantic_weight * losses["semantic_loss"])
        accuracy = self._top_k_accuracy(
            normalized_codebook_logits,
            codebook_targets,
            k=5,
        )
        return self._training_output(
            context,
            loss=total_loss,
            losses={
                "loss": total_loss,
                **losses,
            },
            logits=(token_logits, codebook_logits),
            metadata={
                "top_5_accuracy": accuracy,
                "semantic_positions": int(normalized_codebook_logits.shape[0]),
                "codec_trainable": False,
            },
        )

    @staticmethod
    def _validate_batch(tokens, labels, key_padding_mask) -> None:
        if tokens.ndim != 3 or labels.ndim != 3:
            raise ValueError(
                "Fish Speech inputs and labels must have shape "
                "[batch, num_codebooks + 1, sequence_length].")
        if tokens.shape != labels.shape:
            raise ValueError(
                "Fish Speech inputs and labels must have the same shape; "
                f"received {tuple(tokens.shape)} and {tuple(labels.shape)}.")
        if key_padding_mask is not None and (key_padding_mask.ndim != 2 or tuple(key_padding_mask.shape)
                                             != (tokens.shape[0], tokens.shape[2])):
            raise ValueError(
                "Fish Speech key-padding masks must have shape "
                f"[{tokens.shape[0]}, {tokens.shape[2]}].")

    @classmethod
    def compute_source_losses(
        cls,
        *,
        token_logits,
        codebook_logits,
        labels,
        semantic_begin_id: int,
        semantic_end_id: int,
        num_codebooks: int,
    ):
        """Compute the two upstream cross-entropies without an extra shift."""
        torch = import_optional(
            "torch",
            model_type="fishtts",
            install_extra="training",
        )
        functional = torch.nn.functional
        if semantic_begin_id > semantic_end_id:
            raise ValueError("semantic_begin_id cannot exceed semantic_end_id.")
        if labels.ndim != 3:
            raise ValueError("Fish Speech labels must be a rank-3 tensor.")
        batch_size, channels, sequence_length = labels.shape
        expected_channels = num_codebooks + 1
        if channels != expected_channels:
            raise ValueError(
                f"Fish Speech expects {expected_channels} label channels, "
                f"received {channels}.")
        if token_logits.ndim != 3 or tuple(token_logits.shape[:2]) != (
                batch_size,
                sequence_length,
        ):
            raise ValueError(
                "Fish Speech token logits must have shape [batch, time, "
                f"vocab], received {tuple(token_logits.shape)}.")

        primary_labels = labels[:, 0].long()
        if not primary_labels.ne(LABEL_IGNORE_ID).any():
            raise ValueError("Fish Speech batch contains no supervised base-token labels.")
        base_loss = functional.cross_entropy(
            token_logits.reshape(-1, token_logits.shape[-1]),
            primary_labels.reshape(-1),
            ignore_index=LABEL_IGNORE_ID,
        )

        semantic_mask = (primary_labels.ge(semantic_begin_id) & primary_labels.le(semantic_end_id))
        codebook_targets = (labels[:, 1:expected_channels].permute(0, 2, 1)[semantic_mask].long())
        if not codebook_targets.ne(LABEL_IGNORE_ID).any():
            raise ValueError("Fish Speech batch contains no supervised semantic codebook "
                             "labels.")
        normalized_logits = cls._select_semantic_logits(
            codebook_logits,
            semantic_mask=semantic_mask,
            num_codebooks=num_codebooks,
        )
        if tuple(normalized_logits.shape[:-1]) != tuple(codebook_targets.shape):
            raise ValueError(
                "Fish Speech codebook logits do not align with semantic "
                f"targets: {tuple(normalized_logits.shape)} versus "
                f"{tuple(codebook_targets.shape)}.")
        semantic_loss = functional.cross_entropy(
            normalized_logits.reshape(-1, normalized_logits.shape[-1]),
            codebook_targets.reshape(-1),
            ignore_index=LABEL_IGNORE_ID,
        )
        return (
            {
                "base_loss": base_loss,
                "semantic_loss": semantic_loss,
            },
            normalized_logits,
            codebook_targets,
        )

    @staticmethod
    def _select_semantic_logits(
        codebook_logits,
        *,
        semantic_mask,
        num_codebooks: int,
    ):
        batch_size, sequence_length = semantic_mask.shape
        semantic_count = int(semantic_mask.sum().item())
        shape = tuple(codebook_logits.shape)
        if codebook_logits.ndim == 4:
            if shape[:3] == (
                    batch_size,
                    sequence_length,
                    num_codebooks,
            ):
                return codebook_logits[semantic_mask]
            if shape[:3] == (
                    batch_size,
                    num_codebooks,
                    sequence_length,
            ):
                return codebook_logits.permute(0, 2, 1, 3)[semantic_mask]
        elif codebook_logits.ndim == 3:
            if shape[:2] == (semantic_count, num_codebooks):
                return codebook_logits
            if shape[:2] == (
                    batch_size * sequence_length,
                    num_codebooks,
            ):
                return codebook_logits.reshape(
                    batch_size,
                    sequence_length,
                    num_codebooks,
                    shape[-1],
                )[semantic_mask]
        elif (codebook_logits.ndim == 2 and shape[0] == semantic_count * num_codebooks):
            return codebook_logits.reshape(
                semantic_count,
                num_codebooks,
                shape[-1],
            )
        raise ValueError(
            "Unsupported Fish Speech codebook-logit shape "
            f"{shape}. Expected [B,T,C,V], [B,C,T,V], "
            "[N_semantic,C,V], or [N_semantic*C,V].")

    @staticmethod
    def _top_k_accuracy(logits, labels, *, k: int):
        torch = import_optional(
            "torch",
            model_type="fishtts",
            install_extra="training",
        )
        mask = (labels.ne(LABEL_IGNORE_ID) & labels.ne(CODEBOOK_PAD_TOKEN_ID))
        if not mask.any():
            return torch.zeros((), device=logits.device)
        indices = logits.topk(min(k, logits.shape[-1]), dim=-1).indices
        correct = indices.eq(labels.unsqueeze(-1)).any(dim=-1) & mask
        return correct.sum() / mask.sum()

    def create_optimizer(self, name, parameters, training_args):
        del name
        torch = import_optional(
            "torch",
            model_type="fishtts",
            install_extra="training",
        )
        decay_parameters = []
        other_parameters = []
        for parameter_name, parameter in parameters:
            normalized = parameter_name.lower()
            no_decay = (
                parameter_name.endswith(".bias") or "norm.weight" in normalized or
                ".embeddings." in normalized)
            (other_parameters if no_decay else decay_parameters).append(parameter)
        groups = []
        if decay_parameters:
            groups.append({
                "params": decay_parameters,
                "weight_decay": training_args.weight_decay,
            })
        if other_parameters:
            groups.append({
                "params": other_parameters,
                "weight_decay": 0.0,
            })
        return torch.optim.AdamW(
            groups,
            lr=training_args.learning_rate,
            betas=(
                float(getattr(
                    self.model.config,
                    "training_adam_beta1",
                    0.9,
                )),
                float(getattr(
                    self.model.config,
                    "training_adam_beta2",
                    0.95,
                )),
            ),
            eps=float(getattr(
                self.model.config,
                "training_adam_epsilon",
                1e-5,
            )),
        )

    def create_scheduler(
        self,
        name,
        optimizer,
        num_training_steps,
        training_args,
    ):
        del name
        torch = import_optional(
            "torch",
            model_type="fishtts",
            install_extra="training",
        )
        configured_warmup = getattr(
            self.model.config,
            "training_warmup_steps",
            None,
        )
        warmup_steps = (
            int(configured_warmup) if configured_warmup is not None else (
                training_args.get_warmup_steps(num_training_steps)
                if training_args.warmup_steps or training_args.warmup_ratio else 10))

        def schedule(step: int) -> float:
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return 1.0

        return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)

    def create_dataset(self, records, **kwargs):
        self.setup()
        tokenizer = getattr(
            self.model,
            "tokenizer",
            getattr(self.primary_model, "tokenizer", None),
        )
        if tokenizer is None:
            raise ValueError("Fish Speech fine-tuning requires the checkpoint tokenizer.")
        num_codebooks = int(self.primary_model.config.num_codebooks)
        configured_max_length = getattr(
            self.model.config,
            "training_max_length",
            None,
        )
        max_length = int(
            kwargs.pop(
                "max_length",
                4096 if configured_max_length is None else configured_max_length,
            ))

        if isinstance(records, (str, Path)):
            raise ValueError(
                "Native Fish training accepts validated pre-tokenized "
                "records. Legacy protobuf datasets require provider-owned "
                "parsers and must be converted before entering the training "
                "runtime.")
        try:
            materialized = tuple(records)
        except TypeError as exc:
            raise TypeError("Fish records must be a sequence of pre-tokenized mappings.") from exc
        if materialized and all(isinstance(item, (str, Path)) for item in materialized):
            raise ValueError(
                "Native Fish training does not load legacy protobuf paths; "
                "convert them to pre-tokenized mappings first.")
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected FishSemanticDataset arguments: {unexpected}.")
        return FishSemanticDataset(
            materialized,
            tokenizer=tokenizer,
            num_codebooks=num_codebooks,
            max_length=max_length,
        )

    def save_pretrained(self, save_directory) -> None:
        """Export a strict semantic + codec Safetensors runtime."""
        self.setup()
        destination = Path(save_directory)
        destination.mkdir(parents=True, exist_ok=True)
        from voicehub.hub import write_json_file
        from voicehub.models.fishtts.native.checkpoint import (
            save_fish_codec_pretrained,
            save_fish_semantic_pretrained,
            write_fish_license_files,
        )
        from voicehub.models.fishtts.native.modeling import FishS2ForConditionalGeneration

        if isinstance(
                self.primary_model,
                FishS2ForConditionalGeneration,
        ):
            save_fish_semantic_pretrained(
                self.primary_model,
                destination,
            )
        else:
            # Small custom graphs used by downstream recipe extensions retain
            # a safe generic warm-start representation.
            state = {
                name: value.detach().cpu().contiguous()
                for name, value in self.primary_model.state_dict().items()
            }
            save_safetensors(
                state,
                destination / "model.safetensors",
                metadata={"format": "voicehub-fish-custom-v1"},
            )
            config = getattr(self.primary_model, "config", None)
            if config is not None:
                to_dict = getattr(config, "to_dict", None)
                if callable(to_dict):
                    write_json_file(
                        destination / "config.json",
                        to_dict(),
                    )
                elif callable(getattr(config, "save", None)):
                    config.save(destination / "config.json")
            write_fish_license_files(destination)
        tokenizer = getattr(self.primary_model, "tokenizer", None)
        if tokenizer is None:
            tokenizer = getattr(self.model, "tokenizer", None)
        if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(destination)

        codec = getattr(self.model, "_codec", None)
        if codec is None:
            codec = getattr(self.model, "codec", None)
        if codec is None:
            ensure_codec_loaded = getattr(
                self.model,
                "ensure_codec_loaded",
                None,
            )
            if callable(ensure_codec_loaded):
                codec = ensure_codec_loaded()
        if codec is None or not callable(getattr(codec, "state_dict", None)):
            raise FileNotFoundError("Fish export requires a loaded native codec module.")
        from voicehub.models.fishtts.native.codec import FishModifiedDAC

        if isinstance(codec, FishModifiedDAC):
            save_fish_codec_pretrained(codec, destination / "codec")
        else:
            codec_directory = destination / "codec"
            codec_directory.mkdir(parents=True, exist_ok=True)
            save_safetensors(
                {
                    name: value.detach().cpu().contiguous()
                    for name, value in codec.state_dict().items()
                },
                codec_directory / "model.safetensors",
                metadata={"format": "voicehub-fish-custom-codec-v1"},
            )


__all__ = [
    "FishSemanticDataset",
    "FishSpeechTrainingAdapter",
    "FishTextDataCollator",
]
