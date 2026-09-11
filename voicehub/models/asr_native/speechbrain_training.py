"""Native fine-tuning recipe for SpeechBrain's LibriSpeech CRDNN ASR."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from numbers import Integral
from pathlib import Path
from typing import Any

from voicehub.training.adapters import SpeechSeq2SeqTrainingAdapter
from voicehub.training.contracts import TrainingPhaseSpec
from voicehub.training.datasets import ASRDataset, SpeechDataset

_RAW_FIELDS = frozenset({
    "audio",
    "audio_lengths",
    "audio_path",
    "epoch",
    "sample_rate",
    "sampling_rate",
    "text",
    "transcript",
    "transcription",
})
_PREPARED_FIELDS = frozenset({
    "waveforms",
    "waveform_lengths",
    "tokens_bos",
    "tokens_eos",
    "token_lengths",
    "ctc_tokens",
    "ctc_token_lengths",
})


class SpeechBrainASRTrainingDataset(SpeechDataset):
    """Validated raw-audio/transcript records for native CRDNN training."""

    def __init__(
        self,
        records: Iterable[Mapping[str, Any]],
        *,
        transform=None,
    ) -> None:
        if isinstance(records, (str, bytes, Mapping)):
            raise TypeError("`records` must be an iterable of mappings.")
        normalized = []
        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                raise TypeError(f"SpeechBrain ASR record {index} must be a mapping.")
            value = dict(record)
            if _PREPARED_FIELDS <= set(value):
                normalized.append(value)
                continue
            if "audio" not in value and "audio_path" in value:
                value["audio"] = value["audio_path"]
            if "text" not in value:
                for alias in ("transcription", "transcript"):
                    if alias in value:
                        value["text"] = value[alias]
                        break
            if "audio" not in value:
                raise ValueError(f"SpeechBrain ASR record {index} requires `audio` or "
                                 "`audio_path`.")
            text = value.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"SpeechBrain ASR record {index} requires a non-empty "
                                 "transcript.")
            normalized.append(value)
        super().__init__(normalized, transform=transform)


def _numeric_waveform(value: Any) -> bool:
    return (
        isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and bool(value) and
        all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value))


def _audio_batch(value: Any) -> tuple[Any, ...]:
    import torch

    if isinstance(value, torch.Tensor):
        if value.ndim == 1:
            return (value, )
        if value.ndim == 2:
            return tuple(value[index] for index in range(value.shape[0]))
        raise ValueError("SpeechBrain ASR audio must have shape [samples] or "
                         "[batch, samples].")
    if _numeric_waveform(value):
        return (value, )
    if (isinstance(value, Sequence) and not isinstance(value, (str, bytes))):
        if not value:
            raise ValueError("SpeechBrain ASR audio batches cannot be empty.")
        return tuple(value)
    return (value, )


def _batch_values(
    value: Any,
    *,
    batch_size: int,
    name: str,
) -> tuple[Any, ...]:
    import torch

    if value is None or isinstance(value, (str, bytes)):
        return (value, ) * batch_size
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return (value.item(), ) * batch_size
        if value.ndim != 1:
            raise ValueError(f"`{name}` must be scalar or one-dimensional.")
        values = tuple(value.tolist())
    elif isinstance(value, Sequence):
        values = tuple(value)
    else:
        return (value, ) * batch_size
    if len(values) != batch_size:
        raise ValueError(f"`{name}` contains {len(values)} values for a batch of "
                         f"{batch_size}.")
    return values


def _transcripts(inputs: Mapping[str, Any]) -> tuple[str, ...]:
    value = inputs.get(
        "text",
        inputs.get("transcription", inputs.get("transcript")),
    )
    if isinstance(value, str):
        rows = (value, )
    elif (isinstance(value, Sequence) and not isinstance(value, (str, bytes))):
        rows = tuple(value)
    else:
        raise TypeError("SpeechBrain ASR training requires `text`, `transcription`, or "
                        "`transcript`.")
    if not rows or any(not isinstance(text, str) or not text.strip() for text in rows):
        raise ValueError("SpeechBrain ASR transcripts must be non-empty strings.")
    # LibriSpeech's published tokenizer and recipe use uppercase normalized
    # transcripts. Whitespace normalization prevents accidental unknown pieces
    # without changing apostrophes or other authored text.
    return tuple(" ".join(text.split()).upper() for text in rows)


def _word_edit_distance(
    reference: Sequence[str],
    hypothesis: Sequence[str],
) -> int:
    """Return Levenshtein distance using memory linear in the hypothesis."""
    if len(reference) < len(hypothesis):
        return _word_edit_distance(hypothesis, reference)
    previous = list(range(len(hypothesis) + 1))
    for reference_index, reference_word in enumerate(reference, start=1):
        current = [reference_index]
        for hypothesis_index, hypothesis_word in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[hypothesis_index] + 1,
                    previous[hypothesis_index - 1] + (reference_word != hypothesis_word),
                ))
        previous = current
    return previous[-1]


def _corpus_word_error_rate(
    hypotheses: Sequence[str],
    references: Sequence[str],
) -> float:
    if len(hypotheses) != len(references):
        raise ValueError("WER requires one hypothesis per reference transcript.")
    if not references:
        raise ValueError("WER requires at least one reference transcript.")
    errors = 0
    reference_words = 0
    for index, (hypothesis, reference) in enumerate(zip(hypotheses, references)):
        if not isinstance(hypothesis, str):
            raise TypeError(f"WER hypothesis {index} must be a string.")
        if not isinstance(reference, str):
            raise TypeError(f"WER reference {index} must be a string.")
        reference_tokens = reference.split()
        if not reference_tokens:
            raise ValueError(f"WER reference {index} must contain at least one word.")
        hypothesis_tokens = hypothesis.split()
        errors += _word_edit_distance(
            reference_tokens,
            hypothesis_tokens,
        )
        reference_words += len(reference_tokens)
    return errors / reference_words


def prepare_speechbrain_asr_training_batch(
    wrapper: Any,
    inputs: Mapping[str, Any],
    *,
    phase: str,
) -> dict[str, Any]:
    """Build padded waveforms plus BOS/EOS and CTC token targets."""
    import torch
    from torch.nn import functional

    from voicehub.processing.waveform import load_native_audio

    if not isinstance(inputs, Mapping):
        raise TypeError("SpeechBrain ASR training inputs must be a mapping.")
    if not isinstance(phase, str) or not phase:
        raise ValueError("SpeechBrain ASR training `phase` must be non-empty.")
    if _PREPARED_FIELDS <= set(inputs):
        return dict(inputs)
    if wrapper.native_config is None or wrapper.tokenizer is None:
        raise RuntimeError("SpeechBrain ASR must be loaded before preprocessing.")

    texts = _transcripts(inputs)
    if "audio" in inputs:
        raw_audio = inputs["audio"]
    elif "audio_path" in inputs:
        raw_audio = inputs["audio_path"]
    else:
        raise ValueError("SpeechBrain ASR training requires `audio` or `audio_path`.")
    sources = _audio_batch(raw_audio)
    if len(sources) != len(texts):
        raise ValueError("SpeechBrain ASR training requires one transcript per waveform.")

    raw_lengths = _batch_values(
        inputs.get("audio_lengths"),
        batch_size=len(sources),
        name="audio_lengths",
    )
    rates = _batch_values(
        inputs.get("sampling_rate", inputs.get("sample_rate")),
        batch_size=len(sources),
        name="sampling_rate",
    )
    maximum_samples = round(wrapper.config.training_max_duration_s * wrapper.sample_rate, )
    minimum_samples = max(
        wrapper.native_config.win_length,
        wrapper.native_config.hop_length * (wrapper.native_config.time_pooling_size - 1),
    )
    waveforms = []
    for index, (source, raw_length, rate) in enumerate(zip(sources, raw_lengths, rates), ):
        if raw_length is not None:
            if (isinstance(raw_length, bool) or not isinstance(raw_length, Integral) or raw_length <= 0):
                raise ValueError("`audio_lengths` must contain positive integers.")
        materialized = load_native_audio(
            source,
            sampling_rate=rate,
            target_sampling_rate=wrapper.sample_rate,
            num_samples=(None if raw_length is None else int(raw_length)),
        )
        waveform = materialized.waveform
        if waveform.numel() > maximum_samples:
            raise ValueError(
                f"SpeechBrain ASR example {index} is longer than "
                f"{wrapper.config.training_max_duration_s:g} seconds. "
                "Segment long recordings with aligned transcripts instead of "
                "silently truncating them.")
        if waveform.numel() < minimum_samples:
            waveform = functional.pad(
                waveform,
                (0, minimum_samples - waveform.numel()),
            )
        waveforms.append(waveform)

    waveform_lengths = torch.tensor(
        [waveform.numel() for waveform in waveforms],
        dtype=torch.long,
    )
    maximum_waveform = int(waveform_lengths.max().item())
    waveform_batch = torch.stack(
        [functional.pad(
            waveform,
            (0, maximum_waveform - waveform.numel()),
        ) for waveform in waveforms])

    encoded = tuple(tuple(wrapper.tokenizer.encode_as_ids(text)) for text in texts)
    if any(not row for row in encoded):
        raise ValueError("SpeechBrain ASR transcripts must produce at least one token.")
    token_lengths = torch.tensor(
        [len(row) + 1 for row in encoded],
        dtype=torch.long,
    )
    ctc_lengths = torch.tensor(
        [len(row) for row in encoded],
        dtype=torch.long,
    )
    maximum_tokens = int(ctc_lengths.max().item())
    maximum_sequence = maximum_tokens + 1
    bos = wrapper.native_config.bos_token_id
    eos = wrapper.native_config.eos_token_id
    blank = wrapper.native_config.blank_token_id
    tokens_bos = torch.full(
        (len(encoded), maximum_sequence),
        bos,
        dtype=torch.long,
    )
    tokens_eos = torch.full(
        (len(encoded), maximum_sequence),
        eos,
        dtype=torch.long,
    )
    ctc_tokens = torch.full(
        (len(encoded), maximum_tokens),
        blank,
        dtype=torch.long,
    )
    for index, row in enumerate(encoded):
        token_tensor = torch.tensor(row, dtype=torch.long)
        tokens_bos[index, 1:len(row) + 1] = token_tensor
        tokens_eos[index, :len(row)] = token_tensor
        ctc_tokens[index, :len(row)] = token_tensor

    prepared = {
        "waveforms": waveform_batch,
        "waveform_lengths": waveform_lengths,
        "tokens_bos": tokens_bos,
        "tokens_eos": tokens_eos,
        "token_lengths": token_lengths,
        "ctc_tokens": ctc_tokens,
        "ctc_token_lengths": ctc_lengths,
    }
    for name, value in inputs.items():
        if name not in _RAW_FIELDS and name not in prepared:
            prepared[name] = value
    return prepared


class SpeechBrainNewBobScheduler:
    """Validation-WER NewBob scheduler matching the pinned author recipe.

    Per-step ``step()`` calls intentionally do nothing because NewBob is
    an epoch/validation scheduler. Call :meth:`step_validation_wer`
    after a validation pass; this prevents a generic trainer from
    accidentally annealing the learning rate on every optimizer update.
    """

    def __init__(
        self,
        optimizer: Any,
        *,
        improvement_threshold: float = 0.0025,
        annealing_factor: float = 0.8,
        patience: int = 0,
    ) -> None:
        if not 0.0 < annealing_factor <= 1.0:
            raise ValueError("`annealing_factor` must be in (0, 1].")
        if improvement_threshold < 0.0:
            raise ValueError("`improvement_threshold` must be non-negative.")
        if isinstance(patience, bool) or not isinstance(patience, int) or patience < 0:
            raise ValueError("`patience` must be a non-negative integer.")
        self.optimizer = optimizer
        self.improvement_threshold = float(improvement_threshold)
        self.annealing_factor = float(annealing_factor)
        self.patience = patience
        self.current_patience = patience
        self.metric_values: list[float] = []

    def step(self, metric: float | None = None) -> tuple[float, float] | None:
        if metric is None:
            return None
        return self.step_validation_wer(metric)

    def step_validation_wer(self, metric: float) -> tuple[float, float]:
        if (isinstance(metric, bool) or not isinstance(metric, (int, float)) or
                not math.isfinite(float(metric)) or metric < 0.0):
            raise ValueError("Validation WER must be finite and non-negative.")
        old_value = float(self.optimizer.param_groups[0]["lr"])
        new_value = old_value
        if self.metric_values:
            previous = self.metric_values[-1]
            improvement = (0.0 if previous == 0.0 else (previous - float(metric)) / previous)
            if improvement < self.improvement_threshold:
                if self.current_patience == 0:
                    new_value *= self.annealing_factor
                    self.current_patience = self.patience
                else:
                    self.current_patience -= 1
        self.metric_values.append(float(metric))
        for group in self.optimizer.param_groups:
            group["lr"] = new_value
        return old_value, new_value

    def get_last_lr(self) -> list[float]:
        return [float(group["lr"]) for group in self.optimizer.param_groups]

    def state_dict(self) -> dict[str, Any]:
        return {
            "annealing_factor": self.annealing_factor,
            "current_patience": self.current_patience,
            "improvement_threshold": self.improvement_threshold,
            "metric_values": list(self.metric_values),
            "patience": self.patience,
        }

    def load_state_dict(self, values: Mapping[str, Any]) -> None:
        if not isinstance(values, Mapping):
            raise TypeError("NewBob scheduler state must be a mapping.")
        expected = {
            "annealing_factor",
            "current_patience",
            "improvement_threshold",
            "metric_values",
            "patience",
        }
        if set(values) != expected:
            raise ValueError("NewBob scheduler state keys do not match the recipe.")
        self.annealing_factor = float(values["annealing_factor"])
        self.current_patience = int(values["current_patience"])
        self.improvement_threshold = float(values["improvement_threshold"], )
        self.metric_values = [float(value) for value in values["metric_values"]]
        self.patience = int(values["patience"])


class NativeSpeechBrainASRTrainingAdapter(SpeechSeq2SeqTrainingAdapter):
    """Train the exact native graph with the published combined objective."""

    supports_custom_recipe = True
    native_export_semantics = ("voicehub-native-speechbrain-crdnn-asr-safetensors-and-tokenizer")

    def setup(self) -> NativeSpeechBrainASRTrainingAdapter:
        super().setup()
        if self.primary_model is not getattr(self.model, "model", None):
            raise ValueError(
                "Native SpeechBrain ASR tuning must target the wrapper's "
                "exact CRDNN/decoder/RNNLM graph.")
        return self

    def create_dataset(self, records, **kwargs):
        if isinstance(records, ASRDataset) and not kwargs:
            return records
        return SpeechBrainASRTrainingDataset(records, **kwargs)

    def prepare_training_inputs(
        self,
        inputs: Mapping[str, Any],
        context: Any,
    ) -> Mapping[str, Any]:
        prepared = self.model.prepare_training_inputs(
            dict(inputs),
            phase=context.phase.name,
        )
        if context.is_training:
            epoch = (1 if context.epoch is None else max(1, math.floor(float(context.epoch)) + 1))
        else:
            epoch = self.model.native_config.number_of_ctc_epochs + 1
        prepared["epoch"] = epoch
        prepared["update_normalization"] = bool(context.is_training)
        accepted = {
            "ctc_token_lengths",
            "ctc_tokens",
            "epoch",
            "token_lengths",
            "tokens_bos",
            "tokens_eos",
            "update_normalization",
            "waveform_lengths",
            "waveforms",
        }
        return {name: value for name, value in prepared.items() if name in accepted}

    def create_optimizer(self, name, parameters, training_args):
        del name, training_args
        import torch

        return torch.optim.Adadelta(
            [parameter for _, parameter in parameters],
            lr=1.0,
            rho=0.95,
            eps=1e-8,
            weight_decay=0.0,
        )

    def create_scheduler(
        self,
        name,
        optimizer,
        num_training_steps,
        training_args,
    ):
        del name, num_training_steps, training_args
        return SpeechBrainNewBobScheduler(
            optimizer,
            improvement_threshold=0.0025,
            annealing_factor=0.8,
            patience=0,
        )

    def evaluation_label_values(
        self,
        inputs: Mapping[str, Any],
        phase: TrainingPhaseSpec,
    ) -> tuple[Any, ...]:
        if any(name in inputs for name in ("text", "transcript", "transcription")):
            return (list(_transcripts(inputs)), )
        target_name = (
            "tokens_eos" if "tokens_eos" in inputs else "ctc_tokens" if "ctc_tokens" in inputs else None)
        length_name = ("token_lengths" if target_name == "tokens_eos" else "ctc_token_lengths")
        if (target_name is not None and length_name in inputs and self.model.tokenizer is not None):
            targets = inputs[target_name]
            lengths = inputs[length_name]
            target_rows = targets.tolist() if hasattr(targets, "tolist") else targets
            target_lengths = lengths.tolist() if hasattr(lengths, "tolist") else lengths
            if not isinstance(target_rows, (list, tuple)):
                raise TypeError(f"`{target_name}` must contain token rows.")
            if not isinstance(target_lengths, (list, tuple)):
                raise TypeError(f"`{length_name}` must contain row lengths.")
            if len(target_rows) != len(target_lengths):
                raise ValueError(f"`{target_name}` and `{length_name}` batch sizes differ.")
            references = []
            for row, length in zip(target_rows, target_lengths):
                token_count = int(length) - (1 if target_name == "tokens_eos" else 0)
                references.append(
                    self.model.tokenizer.decode_ids(
                        tuple(int(token) for token in row[:max(0, token_count)]), ))
            return (references, )
        return super().evaluation_label_values(
            inputs,
            phase,
        )

    def prepare_evaluation_predictions(
        self,
        outputs: Any,
        context: Any,
        predictions: Any,
    ) -> Any:
        del context
        encoder_states = self._get_value(outputs, "encoder_states")
        relative_lengths = self._get_value(outputs, "relative_lengths")
        if encoder_states is None or relative_lengths is None:
            return predictions
        if self.model.decoder is None or self.model.tokenizer is None:
            raise RuntimeError("SpeechBrain ASR decoder and tokenizer must be loaded "
                               "before evaluation.")
        decoded = self.model.decoder(
            encoder_states,
            relative_lengths,
            lm_weight=0.0,
        )
        return [self.model.tokenizer.decode_ids(token_ids) for token_ids in decoded.token_ids]

    def compute_evaluation_metrics(
        self,
        predictions: Any,
        label_ids: Any,
    ) -> Mapping[str, Any]:
        if not isinstance(predictions, (list, tuple)):
            raise TypeError("SpeechBrain ASR evaluation predictions must be text rows.")
        if not isinstance(label_ids, (list, tuple)):
            raise TypeError("SpeechBrain ASR evaluation labels must be text rows.")
        return {
            "wer": _corpus_word_error_rate(
                predictions,
                label_ids,
            )
        }

    def evaluation_scheduler_metric(
        self,
        metrics: Mapping[str, Any],
    ) -> float | None:
        if not isinstance(metrics, Mapping):
            raise TypeError("Evaluation metrics must be a mapping.")
        value = metrics.get("eval_wer")
        if value is None:
            return None
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                not math.isfinite(float(value)) or float(value) < 0.0):
            raise ValueError("`eval_wer` must be a finite non-negative number.")
        return float(value)

    def recipe_resume_configuration(self) -> Mapping[str, Any]:
        from voicehub.models.asr_speechbrain.native.metadata import (
            SPEECHBRAIN_ASR_REVISION,
            SPEECHBRAIN_ASR_SOURCE_REVISION,
        )

        configuration = dict(super().recipe_resume_configuration())
        native_config = self.model.native_config
        configuration.update({
            "architecture": "speechbrain-crdnn-asr",
            "checkpoint_format": ("voicehub-speechbrain-crdnn-asr-v1"),
            "ctc_epochs": native_config.number_of_ctc_epochs,
            "ctc_weight": native_config.ctc_weight,
            "label_smoothing": native_config.label_smoothing,
            "newbob": {
                "annealing_factor": 0.8,
                "improvement_threshold": 0.0025,
                "patience": 0,
            },
            "objective": ("ctc-plus-label-smoothed-seq2seq-then-seq2seq"),
            "optimizer": {
                "eps": 1e-8,
                "lr": 1.0,
                "name": "Adadelta",
                "rho": 0.95,
            },
            "published_artifact_revision": SPEECHBRAIN_ASR_REVISION,
            "sample_rate": self.model.sample_rate,
            "upstream_training_source_revision": (SPEECHBRAIN_ASR_SOURCE_REVISION),
        })
        return configuration

    def artifact_manifest(self) -> dict[str, Any]:
        manifest = super().artifact_manifest()
        manifest.update({
            "checkpoint_format": ("voicehub-speechbrain-crdnn-asr-v1"),
            "native_architecture_family": "speechbrain-crdnn-asr",
            "optimizer": "Adadelta(lr=1,rho=0.95,eps=1e-8)",
            "processor_runtime": "voicehub-native",
            "scheduler": ("NewBob(validation WER, threshold=0.0025, factor=0.8)"),
            "tokenizer_runtime": "voicehub-sentencepiece-unigram",
        })
        return manifest

    def on_training_phase_end(self, context, output):
        epoch = (1 if context.epoch is None else max(1, math.floor(float(context.epoch)) + 1))
        ctc_active = (context.is_training and epoch <= self.model.native_config.number_of_ctc_epochs)
        output.metadata.update({
            "ctc_active":
            ctc_active,
            "native_architecture_family":
            "speechbrain-crdnn-asr",
            "native_objective": ("combined-ctc-seq2seq" if ctc_active else "label-smoothed-seq2seq"),
            "recipe_epoch":
            epoch,
        })
        return output

    def save_pretrained(self, save_directory: str | Path) -> None:
        self.setup()
        self.model.export_native_pretrained(Path(save_directory).expanduser(), )


__all__ = [
    "NativeSpeechBrainASRTrainingAdapter",
    "SpeechBrainASRTrainingDataset",
    "SpeechBrainNewBobScheduler",
    "prepare_speechbrain_asr_training_batch",
]
