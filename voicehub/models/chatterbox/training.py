"""Source-faithful fine-tuning and preprocessing for native Chatterbox.

The adapter accepts portable ``audio``/``text`` records or the
precomputed tensor schema used by the reviewed community trainer. T3
language-model and S3Gen flow objectives remain separate jobs because
they use different frozen frontends, parameter graphs, and optimizer
state.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

from voicehub.models.chatterbox.checkpoint import export_chatterbox_runtime
from voicehub.models.chatterbox.models.s3gen.const import S3GEN_SR
from voicehub.models.chatterbox.models.s3tokenizer import S3_SR
from voicehub.models.chatterbox.models.t3.modules.cond_enc import T3Cond
from voicehub.models.chatterbox.tts import punc_norm
from voicehub.outputs import TTSTrainingOutput
from voicehub.processing.waveform import load_native_audio, resample_waveform
from voicehub.training.adapters import CompositeTrainingAdapter
from voicehub.training.collators import DataCollatorForAudioTraining
from voicehub.training.contracts import TrainingContext
from voicehub.training.datasets import SpeechDataset


def resize_t3_text_vocabulary(t3: nn.Module, vocabulary_size: int) -> None:
    """Resize T3's text embedding/head with mean-initialized new tokens.

    Chatterbox community fine-tuning uses this operation when a
    replacement tokenizer adds a language.  Mean initialization
    preserves the original embedding scale and avoids introducing
    unusually large random logits.
    """
    if (isinstance(vocabulary_size, bool) or not isinstance(vocabulary_size, int) or vocabulary_size <= 0):
        raise ValueError("Chatterbox text vocabulary size must be a positive integer.")
    current_size = int(t3.text_emb.num_embeddings)
    if vocabulary_size == current_size:
        return
    minimum_size = max(
        int(t3.hp.start_text_token),
        int(t3.hp.stop_text_token),
    ) + 1
    if vocabulary_size < minimum_size:
        raise ValueError(
            "Chatterbox text vocabulary cannot be smaller than its reserved "
            f"token range ({minimum_size}).")

    embedding = t3.text_emb
    head = t3.text_head
    if (embedding.weight.ndim != 2 or head.weight.ndim != 2 or embedding.weight.shape != head.weight.shape):
        raise ValueError(
            "Chatterbox T3 vocabulary resize requires matching text embedding "
            "and projection matrices.")
    embedding_trainable = embedding.weight.requires_grad
    head_trainable = head.weight.requires_grad
    replacement_embedding = nn.Embedding(
        vocabulary_size,
        embedding.embedding_dim,
        device=embedding.weight.device,
        dtype=embedding.weight.dtype,
    )
    replacement_head = nn.Linear(
        head.in_features,
        vocabulary_size,
        bias=False,
        device=head.weight.device,
        dtype=head.weight.dtype,
    )
    copied = min(current_size, vocabulary_size)
    with torch.no_grad():
        replacement_embedding.weight[:copied].copy_(embedding.weight[:copied])
        replacement_head.weight[:copied].copy_(head.weight[:copied])
        if vocabulary_size > current_size:
            replacement_embedding.weight[current_size:].copy_(embedding.weight.mean(dim=0, keepdim=True))
            replacement_head.weight[current_size:].copy_(head.weight.mean(dim=0, keepdim=True))
    replacement_embedding.weight.requires_grad_(embedding_trainable)
    replacement_head.weight.requires_grad_(head_trainable)
    t3.text_emb = replacement_embedding
    t3.text_head = replacement_head
    t3.hp.text_tokens_dict_size = vocabulary_size


class ChatterboxTrainingAdapter(CompositeTrainingAdapter):
    """Fine-tune exactly one released Chatterbox objective per job."""

    supports_custom_recipe = True
    native_export_semantics = "complete-inference-reloadable-runtime"

    _PHASE_ALIASES = {
        "language_model": "language_model",
        "lm": "language_model",
        "t3": "language_model",
        "flow": "flow",
        "s3gen": "flow",
    }

    def __init__(self, model, spec):
        super().__init__(model, spec)
        self._lora_injection = None
        self.data_collator = DataCollatorForAudioTraining(
            field_schemas={
                "audio": {
                    "sequence_dim": -1,
                    "padding_value": 0.0,
                    "length_field": "audio_lengths",
                },
                "text_tokens": {
                    "sequence_dim": 0,
                    "padding_value": 0,
                    "length_field": "text_token_lens",
                },
                "speech_tokens": {
                    "sequence_dim": 0,
                    "padding_value": 0,
                    "length_field": "speech_token_lens",
                },
                "prompt_tokens": {
                    "sequence_dim": 0,
                    "padding_value": 0,
                    "length_field": "prompt_lens",
                    "allow_missing": True,
                },
                "speech_token": {
                    "sequence_dim": 0,
                    "padding_value": 0,
                    "length_field": "speech_token_len",
                },
                "speech_feat": {
                    "sequence_dim": -1,
                    "padding_value": 0.0,
                    "length_field": "speech_feat_len",
                },
            })

    @property
    def selected_phase_name(self) -> str:
        configured = getattr(
            self.model.config,
            "training_component",
            "language_model",
        )
        normalized = str(configured).strip().lower().replace("-", "_")
        try:
            return self._PHASE_ALIASES[normalized]
        except KeyError as error:
            choices = ", ".join(sorted(self._PHASE_ALIASES))
            raise ValueError(
                f"Unknown Chatterbox training_component {configured!r}; "
                f"choose one of: {choices}.") from error

    def setup(self):
        if self.is_ready:
            return super().setup()
        super().setup()
        runtime = self.model.model
        configured_vocabulary_size = getattr(
            self.model.config,
            "training_text_vocab_size",
            None,
        )
        if configured_vocabulary_size is not None:
            if self.selected_phase_name != "language_model":
                raise ValueError(
                    "Chatterbox vocabulary expansion is available only for "
                    "language_model training.")
            resize_t3_text_vocabulary(
                runtime.t3,
                configured_vocabulary_size,
            )
        for component_name in ("t3", "s3gen", "ve"):
            component = getattr(runtime, component_name)
            for parameter in component.parameters():
                parameter.requires_grad_(False)
            component.eval()
        selected = (runtime.t3 if self.selected_phase_name == "language_model" else runtime.s3gen.flow)
        lora_rank = getattr(self.model.config, "training_lora_rank", None)
        if lora_rank is not None:
            if self.selected_phase_name != "language_model":
                raise ValueError(
                    "Chatterbox LoRA is available only for language_model "
                    "training; S3Gen flow uses its published dense objective.")
            from voicehub.optimization import LoRAConfig, inject_lora

            self._lora_injection = inject_lora(
                runtime.t3,
                LoRAConfig(
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
                    target_modules=tuple(
                        getattr(
                            self.model.config,
                            "training_lora_target_modules",
                            (
                                "q_proj",
                                "k_proj",
                                "v_proj",
                                "o_proj",
                                "gate_proj",
                                "up_proj",
                                "down_proj",
                                "spkr_enc",
                            ),
                        )),
                    freeze_base=True,
                    seed=getattr(
                        self.model.config,
                        "training_lora_seed",
                        0,
                    ),
                ),
            )
            lora_modules_to_train = getattr(
                self.model.config,
                "training_lora_modules_to_train",
                ("text_emb", "text_head"),
            )
            for module_name in lora_modules_to_train:
                module = runtime.t3.get_submodule(str(module_name))
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
        else:
            for parameter in selected.parameters():
                parameter.requires_grad_(True)
        selected.train()
        return self

    def train(self, mode: bool = True):
        """Train only the selected phase while frozen frontends stay in
        eval."""
        self.setup()
        runtime = self.model.model
        runtime.t3.eval()
        runtime.s3gen.eval()
        runtime.ve.eval()
        selected = (runtime.t3 if self.selected_phase_name == "language_model" else runtime.s3gen.flow)
        selected.train(mode)
        return self

    def plan_training_phases(self, step: int):
        del step
        return (self.spec.get_phase(self.selected_phase_name), )

    def select_training_phase(self, training_phase=None):
        phase = super().select_training_phase(training_phase)
        if phase.name != self.selected_phase_name:
            raise ValueError(
                f"This Chatterbox runtime was configured for "
                f"{self.selected_phase_name!r}, not {phase.name!r}. Start a "
                "separate job with the other training_component.")
        return phase

    def create_dataset(self, records, **kwargs):
        if isinstance(records, (str, bytes, Mapping)):
            raise TypeError("Chatterbox records must be an iterable of mappings.")
        materialized = tuple(records)
        precomputed_required = ((
            "text_tokens",
            "speech_tokens",
            "speaker_emb",
        ) if self.selected_phase_name == "language_model" else (
            "speech_token",
            "speech_feat",
            "embedding",
        ))
        raw_required = (("audio", "text") if self.selected_phase_name == "language_model" else ("audio", ))
        normalized = []
        modes = set()
        for index, record in enumerate(materialized):
            if not isinstance(record, Mapping):
                raise TypeError(f"Chatterbox record {index} must be a mapping.")
            value = dict(record)
            if "audio" not in value and "audio_path" in value:
                value["audio"] = value.pop("audio_path")
            if all(name in value for name in precomputed_required):
                modes.add("precomputed")
                if self.selected_phase_name == "language_model":
                    value.setdefault(
                        "text_token_lens",
                        int(torch.as_tensor(value["text_tokens"]).numel()),
                    )
                    value.setdefault(
                        "speech_token_lens",
                        int(torch.as_tensor(value["speech_tokens"]).numel()),
                    )
                    if ("prompt_tokens" in value and "prompt_lens" not in value):
                        value["prompt_lens"] = int(torch.as_tensor(value["prompt_tokens"]).numel())
                else:
                    value.setdefault(
                        "speech_token_len",
                        int(torch.as_tensor(value["speech_token"]).numel()),
                    )
                    value.setdefault(
                        "speech_feat_len",
                        int(torch.as_tensor(value["speech_feat"]).shape[-1]),
                    )
            elif all(name in value for name in raw_required):
                modes.add("raw")
            else:
                options = (f"{', '.join(precomputed_required)} or "
                           f"{', '.join(raw_required)}")
                raise ValueError(f"Chatterbox record {index} requires {options}.")
            normalized.append(value)
        if len(modes) > 1:
            raise ValueError("Do not mix raw-audio and precomputed Chatterbox records in "
                             "one dataset.")
        required = (raw_required if modes == {"raw"} else precomputed_required)
        return SpeechDataset(
            normalized,
            required_fields=required,
            transform=kwargs.get("transform"),
        )

    @staticmethod
    def _require(batch: Mapping, fields: tuple[str, ...], *, phase: str) -> None:
        missing = [name for name in fields if name not in batch]
        if missing:
            raise ValueError(
                f"Chatterbox {phase} fine-tuning requires precomputed fields: " + ", ".join(missing))

    @staticmethod
    def _batch_items(
        value: Any,
        *,
        batch_size: int,
        name: str,
    ) -> list[Any]:
        if isinstance(value, torch.Tensor):
            if batch_size == 1 and (value.ndim == 1 or value.ndim == 2 and value.shape[0] != 1):
                return [value]
            if value.ndim >= 1 and value.shape[0] == batch_size:
                return list(value.unbind(0))
        elif isinstance(value, (str, Path, Mapping)):
            if batch_size == 1:
                return [value]
        elif isinstance(value, Sequence):
            values = list(value)
            if len(values) == batch_size:
                return values
            if batch_size == 1 and values and all(isinstance(item, (int, float)) for item in values):
                return [values]
        raise ValueError(f"Chatterbox raw field {name!r} does not contain "
                         f"{batch_size} sample(s).")

    @staticmethod
    def _batch_rates(value: Any, *, batch_size: int) -> list[int | None]:
        if value is None:
            return [None] * batch_size
        if isinstance(value, torch.Tensor):
            values = value.detach().cpu().reshape(-1).tolist()
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            values = list(value)
        else:
            values = [value]
        if len(values) == 1:
            values *= batch_size
        if len(values) != batch_size:
            raise ValueError(
                "Chatterbox sampling_rate must be scalar or contain one "
                "value per audio sample.")
        output = []
        for rate in values:
            if rate is None:
                output.append(None)
            elif (isinstance(rate, bool) or int(rate) != rate or int(rate) <= 0):
                raise ValueError("Chatterbox sampling rates must be positive integers.")
            else:
                output.append(int(rate))
        return output

    @staticmethod
    def _pad_1d(
        values: Sequence[torch.Tensor],
        *,
        padding_value: int = 0,
    ) -> torch.Tensor:
        if not values:
            raise ValueError("Cannot pad an empty Chatterbox batch.")
        length = max(int(value.numel()) for value in values)
        output = values[0].new_full(
            (len(values), length),
            padding_value,
        )
        for index, value in enumerate(values):
            output[index, :value.numel()] = value.reshape(-1)
        return output

    @staticmethod
    def _pad_features(values: Sequence[torch.Tensor]) -> torch.Tensor:
        if not values:
            raise ValueError("Cannot pad an empty Chatterbox feature batch.")
        channels = int(values[0].shape[0])
        if any(value.ndim != 2 or value.shape[0] != channels for value in values):
            raise ValueError("Chatterbox speech features must have matching channels.")
        frames = max(int(value.shape[1]) for value in values)
        output = values[0].new_zeros(len(values), channels, frames)
        for index, value in enumerate(values):
            output[index, :, :value.shape[1]] = value
        return output

    def _raw_audio_batch(
        self,
        batch: Mapping[str, Any],
        *,
        batch_size: int,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        audio = self._batch_items(
            batch["audio"],
            batch_size=batch_size,
            name="audio",
        )
        rates = self._batch_rates(
            batch.get("sampling_rate", batch.get("sample_rate")),
            batch_size=batch_size,
        )
        lengths_value = batch.get("audio_lengths")
        lengths = (
            self._batch_items(
                lengths_value,
                batch_size=batch_size,
                name="audio_lengths",
            ) if lengths_value is not None else [None] * batch_size)
        waveforms_16k = []
        waveforms_24k = []
        for value, rate, length in zip(audio, rates, lengths):
            if length is not None and not isinstance(value, (str, Path, Mapping)):
                length = int(torch.as_tensor(length).item())
                if length <= 0:
                    raise ValueError("Chatterbox audio_lengths must be positive.")
                value = torch.as_tensor(value)[..., :length]
            decoded = load_native_audio(value, sampling_rate=rate)
            waveform = decoded.waveform
            waveforms_16k.append(resample_waveform(
                waveform,
                decoded.sampling_rate,
                S3_SR,
            ))
            waveforms_24k.append(resample_waveform(
                waveform,
                decoded.sampling_rate,
                S3GEN_SR,
            ))
        return waveforms_16k, waveforms_24k

    def _prepare_raw_language_model_batch(
        self,
        batch: Mapping[str, Any],
        context: TrainingContext,
    ) -> dict[str, torch.Tensor]:
        texts_value = batch.get("text")
        if isinstance(texts_value, str):
            texts = [texts_value]
        elif isinstance(texts_value, Sequence):
            texts = list(texts_value)
        else:
            raise TypeError("Chatterbox raw T3 batches require text strings.")
        if not texts or any(not isinstance(text, str) for text in texts):
            raise TypeError("Every Chatterbox raw T3 example must contain a text string.")
        waveforms_16k, _ = self._raw_audio_batch(
            batch,
            batch_size=len(texts),
        )
        runtime = self.model.model
        device = runtime.t3.device
        max_text = int(getattr(self.model.config, "training_max_text_tokens", 256))
        max_speech = int(getattr(self.model.config, "training_max_speech_tokens", 850))
        prompt_duration = float(getattr(self.model.config, "training_prompt_duration", 3.0))
        dropout = float(getattr(
            self.model.config,
            "training_conditioning_dropout",
            0.2,
        ))
        if max_text < 3 or max_text > runtime.t3.hp.max_text_tokens:
            raise ValueError(
                "training_max_text_tokens must be between 3 and the T3 "
                f"limit ({runtime.t3.hp.max_text_tokens}).")
        if max_speech < 3 or max_speech > runtime.t3.hp.max_speech_tokens:
            raise ValueError(
                "training_max_speech_tokens must be between 3 and the T3 "
                f"limit ({runtime.t3.hp.max_speech_tokens}).")
        if not math.isfinite(prompt_duration) or prompt_duration <= 0:
            raise ValueError("training_prompt_duration must be finite and positive.")
        if not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("training_conditioning_dropout must be in [0, 1).")

        text_tokens = []
        speech_tokens = []
        prompt_tokens = []
        prompt_loss_lengths = []
        speaker_embeddings = []
        prompt_samples = round(prompt_duration * S3_SR)
        prompt_limit = int(runtime.t3.hp.speech_cond_prompt_len)
        with torch.inference_mode():
            for text, waveform in zip(texts, waveforms_16k):
                waveform = waveform.to(device)
                speaker = runtime.ve.embeds_from_wavs(
                    [waveform],
                    sample_rate=S3_SR,
                )[0]
                codes, code_lengths = runtime.s3gen.tokenizer([waveform])
                code_length = min(
                    int(code_lengths[0].item()),
                    max_speech - 2,
                )
                if code_length < 1:
                    raise ValueError(
                        "Chatterbox T3 preprocessing produced no speech "
                        "tokens; provide a longer recording.")
                codes = codes[0, :code_length].long()
                speech = torch.cat((
                    codes.new_tensor([runtime.t3.hp.start_speech_token]),
                    codes,
                    codes.new_tensor([runtime.t3.hp.stop_speech_token]),
                ))

                prompt_waveform = waveform[:prompt_samples]
                if prompt_waveform.numel() < prompt_samples:
                    prompt_waveform = torch.nn.functional.pad(
                        prompt_waveform,
                        (0, prompt_samples - prompt_waveform.numel()),
                    )
                prompt, prompt_length = runtime.s3gen.tokenizer(
                    [prompt_waveform],
                    max_len=prompt_limit,
                )
                prompt = prompt[
                    0,
                    :int(prompt_length[0].item()),
                ].long()
                body = runtime.tokenizer.text_to_tokens(punc_norm(text)).reshape(-1).to(
                    device=device, dtype=torch.long)
                body = body[:max_text - 2]
                text_sequence = torch.cat((
                    body.new_tensor([runtime.t3.hp.start_text_token]),
                    body,
                    body.new_tensor([runtime.t3.hp.stop_text_token]),
                ))
                if (context.is_training and dropout > 0 and bool(torch.rand((), device=device) < dropout)):
                    speaker = torch.zeros_like(speaker)
                    prompt = prompt.new_zeros(1)
                    prompt_loss_length = 0
                else:
                    prompt_loss_length = prompt.numel()
                text_tokens.append(text_sequence)
                speech_tokens.append(speech)
                prompt_tokens.append(prompt)
                prompt_loss_lengths.append(prompt_loss_length)
                speaker_embeddings.append(speaker)

        return {
            "text_tokens":
            self._pad_1d(text_tokens),
            "text_token_lens":
            torch.tensor(
                [value.numel() for value in text_tokens],
                device=device,
                dtype=torch.long,
            ),
            "speech_tokens":
            self._pad_1d(speech_tokens),
            "speech_token_lens":
            torch.tensor(
                [value.numel() for value in speech_tokens],
                device=device,
                dtype=torch.long,
            ),
            "speaker_emb":
            torch.stack(speaker_embeddings),
            "prompt_tokens":
            self._pad_1d(prompt_tokens),
            "prompt_lens":
            torch.tensor(
                prompt_loss_lengths,
                device=device,
                dtype=torch.long,
            ),
        }

    def _prepare_raw_flow_batch(
        self,
        batch: Mapping[str, Any],
    ) -> dict[str, torch.Tensor]:
        audio_value = batch["audio"]
        if isinstance(audio_value, torch.Tensor):
            batch_size = 1 if audio_value.ndim == 1 else audio_value.shape[0]
        elif isinstance(audio_value, (str, Path, Mapping)):
            batch_size = 1
        elif isinstance(audio_value, Sequence):
            batch_size = len(audio_value)
        else:
            raise TypeError("Chatterbox raw flow audio has an invalid type.")
        waveforms_16k, waveforms_24k = self._raw_audio_batch(
            batch,
            batch_size=batch_size,
        )
        runtime = self.model.model
        device = runtime.s3gen.device
        max_speech = int(getattr(self.model.config, "training_max_speech_tokens", 850))
        tokens = []
        features = []
        embeddings = []
        with torch.inference_mode():
            for waveform_16k, waveform_24k in zip(
                    waveforms_16k,
                    waveforms_24k,
            ):
                waveform_16k = waveform_16k.to(device)
                waveform_24k = waveform_24k.to(device)
                token, token_length = runtime.s3gen.tokenizer([waveform_16k])
                feature = runtime.s3gen.mel_extractor(waveform_24k.unsqueeze(0))[0]
                usable_tokens = min(
                    int(token_length[0].item()),
                    max_speech,
                    int(feature.shape[1]) // 2,
                )
                if usable_tokens < 1:
                    raise ValueError("Chatterbox flow preprocessing produced no aligned "
                                     "speech frames.")
                tokens.append(token[0, :usable_tokens].long())
                features.append(feature[:, :usable_tokens * 2])
                embeddings.append(runtime.s3gen.speaker_encoder.inference([waveform_16k])[0])
        return {
            "speech_token":
            self._pad_1d(tokens),
            "speech_token_len":
            torch.tensor(
                [value.numel() for value in tokens],
                device=device,
                dtype=torch.long,
            ),
            "speech_feat":
            self._pad_features(features),
            "speech_feat_len":
            torch.tensor(
                [value.shape[1] for value in features],
                device=device,
                dtype=torch.long,
            ),
            "embedding":
            torch.stack(embeddings),
        }

    def prepare_training_inputs(
        self,
        inputs: Mapping,
        context: TrainingContext,
    ) -> dict:
        batch = dict(inputs)
        if context.phase.name == "language_model":
            if "text_tokens" not in batch and "audio" in batch:
                batch = self._prepare_raw_language_model_batch(
                    batch,
                    context,
                )
            required = (
                "text_tokens",
                "text_token_lens",
                "speech_tokens",
                "speech_token_lens",
            )
            self._require(batch, required, phase="T3")
            if "speaker_emb" not in batch and "speaker_embedding" in batch:
                batch["speaker_emb"] = batch.pop("speaker_embedding")
            community_prompt = batch.pop("prompt_tokens", None)
            if (community_prompt is not None and "cond_prompt_speech_tokens" in batch):
                raise ValueError("Pass either prompt_tokens or "
                                 "cond_prompt_speech_tokens, not both.")
            if community_prompt is not None:
                batch["cond_prompt_speech_tokens"] = community_prompt
                if ("prompt_lens" not in batch and bool(getattr(
                        self.model.config,
                        "training_mask_prompt_loss",
                        True,
                ))):
                    batch["prompt_lens"] = torch.full(
                        (community_prompt.shape[0], ),
                        community_prompt.shape[1],
                        device=community_prompt.device,
                        dtype=torch.long,
                    )
            if "t3_cond" not in batch:
                self._require(batch, ("speaker_emb", ), phase="T3 conditioning")
                speaker_embedding = batch.pop("speaker_emb")
                emotion = batch.pop("emotion_adv", None)
                if emotion is None:
                    emotion = torch.full(
                        (speaker_embedding.shape[0], 1, 1),
                        0.5,
                        device=speaker_embedding.device,
                        dtype=speaker_embedding.dtype,
                    )
                else:
                    emotion = torch.as_tensor(
                        emotion,
                        device=speaker_embedding.device,
                        dtype=speaker_embedding.dtype,
                    )
                    if emotion.numel() == 1:
                        emotion = emotion.expand(speaker_embedding.shape[0]).reshape(-1, 1, 1)
                    elif emotion.numel() == speaker_embedding.shape[0]:
                        emotion = emotion.reshape(-1, 1, 1)
                    else:
                        raise ValueError(
                            "Chatterbox emotion_adv must contain one value "
                            "or one value per sample.")
                batch["t3_cond"] = T3Cond(
                    speaker_emb=speaker_embedding,
                    clap_emb=batch.pop("clap_emb", None),
                    cond_prompt_speech_tokens=batch.pop(
                        "cond_prompt_speech_tokens",
                        None,
                    ),
                    cond_prompt_speech_emb=batch.pop(
                        "cond_prompt_speech_emb",
                        None,
                    ),
                    emotion_adv=emotion,
                )
            elif not isinstance(batch["t3_cond"], T3Cond):
                raise TypeError("t3_cond must be a Chatterbox T3Cond value.")
            optional = (
                "labels_text",
                "labels_speech",
                "prompt_lens",
            )
            names = (*required, "t3_cond", *optional)
            return {name: batch[name] for name in names if name in batch}

        if "speech_token" not in batch and "audio" in batch:
            batch = self._prepare_raw_flow_batch(batch)
        required = (
            "speech_token",
            "speech_token_len",
            "speech_feat",
            "speech_feat_len",
            "embedding",
        )
        self._require(batch, required, phase="S3Gen flow")
        return {name: batch[name] for name in required}

    def execute_training_phase(
        self,
        context: TrainingContext,
    ) -> TTSTrainingOutput:
        self.setup()
        phase = self.select_training_phase(context.phase)
        prepared = self.prepare_runtime_inputs(self.prepare_batch(context.inputs, context))
        runtime = self.model.model
        if phase.name == "language_model":
            prepared["t3_cond"].to(device=runtime.t3.device)
            text_loss, speech_loss = runtime.t3.loss(**prepared)
            text_weight = float(getattr(self.model.config, "training_text_loss_weight", 1.0))
            speech_weight = float(getattr(self.model.config, "training_speech_loss_weight", 1.0))
            if (not math.isfinite(text_weight) or not math.isfinite(speech_weight) or text_weight < 0 or
                    speech_weight < 0 or text_weight + speech_weight <= 0):
                raise ValueError(
                    "Chatterbox T3 loss weights must be finite, non-negative, "
                    "and include at least one positive value.")
            weighted_text = text_loss * text_weight
            weighted_speech = speech_loss * speech_weight
            loss = weighted_text + weighted_speech
            losses = {
                "loss": loss,
                "text_loss": weighted_text,
                "speech_token_loss": weighted_speech,
                "raw_text_loss": text_loss,
                "raw_speech_token_loss": speech_loss,
            }
        else:
            losses = runtime.s3gen.compute_loss(prepared)
            loss = losses["loss"]

        return TTSTrainingOutput(
            loss=loss,
            losses=losses,
            metadata={
                "model_type": self.model_type,
                "training_family": self.spec.family_name,
                "training_support": self.spec.support.value,
                "training_phase": phase.name,
                "optimizer_names": phase.optimizer_names,
                "published_objective": True,
                "author_end_to_end_recipe_published": False,
                "accepts_raw_audio": True,
                "precomputed_supervision_supported": True,
                "parameter_efficient": self._lora_injection is not None,
            },
            training_phase=phase.name,
            optimizer_names=phase.optimizer_names,
        )

    def execute_prediction_phase(self, context: TrainingContext):
        return self.execute_training_phase(context)

    def recipe_resume_configuration(self):
        configuration = dict(super().recipe_resume_configuration())
        configuration.update({
            "selected_phase": self.selected_phase_name,
            "published_objective": True,
            "author_end_to_end_recipe_published": False,
            "accepts_raw_audio": True,
            "precomputed_supervision_supported": True,
            "parameter_efficient": self._lora_injection is not None,
        })
        return configuration

    def save_pretrained(self, save_directory) -> None:
        self.setup()
        runtime = self.model.model
        if self._lora_injection is None:
            runtime.save_pretrained(Path(save_directory))
            return
        self._lora_injection.merge()
        try:
            portable_t3_state = {
                name.replace(".base.", "."): tensor
                for name, tensor in runtime.t3.state_dict().items()
                if not name.endswith((".lora_a", ".lora_b"))
            }
            export_chatterbox_runtime(
                runtime,
                Path(save_directory),
                t3_state_dict=portable_t3_state,
            )
        finally:
            self._lora_injection.unmerge()


__all__ = [
    "ChatterboxTrainingAdapter",
    "resize_t3_text_vocabulary",
]
