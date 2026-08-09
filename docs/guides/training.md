---
description: Fine-tune supported speech models with one-step validation, exact resume, and portable exports.
---

# Training

VoiceHub provides shared orchestration only when an integration exposes a
differentiable graph, objective, and dataset contract. It does not assign a
generic loss to an inference model.

Read the [training matrix](../models/training-support.md) before choosing a
checkpoint. It records the exact support level, objective, required fields,
frozen components, and artifact boundary for every model.

## Support levels

| Level | Meaning |
| --- | --- |
| `native` | Integrated differentiable objective and trainer route |
| `preprocessed` | Executable objective; caller supplies model-shaped tensors |
| `custom` | Model-specific phases or orchestration are required |
| `inference-only` | No verified gradient path; training fails closed |

Support is checkpoint-aware. A trainable architecture does not make an ONNX,
GGUF, quantized, fused, or inference-pruned artifact trainable.

## Install

```bash
python -m pip install "voicehub[training] @ git+https://github.com/kadirnar/voicehub.git@main"
```

The extra adds dataset, evaluation, and reporting tools. Built-in inference
models remain part of the default package.

## Inspect the selected profile

Do this before loading weights:

```python
from voicehub import get_training_spec

spec = get_training_spec("dia")
print(spec.support.value)
print(spec.family_name)
print(spec.dataset_spec.architecture)
print([phase.name for phase in spec.phases])
```

Query the registry rather than relying on copied model counts:

```python
from collections import Counter

from voicehub import list_training_specs

print(Counter(item.support.value for item in list_training_specs()))
```

## Prepare data

Keep stable IDs, exact transcripts, speaker/session groups, consent, license,
and provenance in the source manifest. Split before model preprocessing:

```python
from voicehub import TTSDataset

source = TTSDataset.from_manifest(
    "data/dia/manifest.jsonl",
    model_type="dia",
    validate_files=True,
)
train_source, validation_source = source.train_test_split(
    validation_fraction=0.1,
    seed=42,
    group_by="session_id",
)
```

Use `ASRDataset` for speech recognition. The
[data preparation guide](data-preparation.md) and
[speech data guide](speech-data.md) describe aliases, audio validation, and
leakage-safe groups.

The selected model owns final preparation:

```python
from voicehub import AutoModelForTextToSpeech

model = AutoModelForTextToSpeech.from_pretrained(
    "nari-labs/Dia-1.6B-0626",
    model_type="dia",
    device="cuda",
    lazy_load=True,
)
model.validate_training_support()
train_dataset = model.create_training_dataset(train_source)
validation_dataset = model.create_training_dataset(validation_source)
```

Inspect one collated batch before training:

```python
features = [train_dataset[0]]
batch = train_dataset.collate_fn(features)
for name, value in batch.items():
    print(name, getattr(value, "shape", type(value).__name__))
```

The generic trainer does not invent codec delays, flow targets, alignments,
adversarial pairs, transducer durations, or detach boundaries. A
`preprocessed` profile must receive exactly the fields listed in its contract.

## Run one optimizer step

Keep `max_steps=1` until the loss is finite, intended parameters receive
gradients, frozen components remain frozen, and save/reload works:

```python
from voicehub import Trainer, TrainingArguments

arguments = TrainingArguments(
    output_dir="runs/dia-smoke",
    max_steps=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=5e-5,
    logging_steps=1,
    eval_strategy="steps",
    eval_steps=1,
    save_strategy="steps",
    save_steps=1,
    report_to="none",
    seed=42,
    data_seed=42,
)
trainer = Trainer(
    model=model,
    args=arguments,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
)
result = trainer.train(resume_from_checkpoint=False)
print(result.training_loss, result.metrics)
```

Increase steps, epochs, batch size, workers, mixed precision, or distributed
scope only after this smoke run passes.

## Evaluate, save, and resume

```python
metrics = trainer.evaluate()
artifact = trainer.save_model("runs/dia-smoke/final")
print(metrics, artifact)
```

`checkpoint-N/` is an exact Trainer resume artifact containing optimizer,
scheduler, random state, callbacks, sampler, strategy, and recipe topology.
Credential fields are rejected, but binary pickle state still requires a trusted, integrity-checked checkpoint:

```python
trainer.train(resume_from_checkpoint=True)
```

`trainer.save_model()` writes a portable final artifact. A standalone
Safetensors file is a weight container, not proof of exact resume state.

Reload the portable artifact in a fresh model:

```python
reloaded = AutoModelForTextToSpeech.from_pretrained(
    "runs/dia-smoke/final",
    device="cuda",
    lazy_load=True,
)
```

Compare pre- and post-training inference with the same prompt, seed, decoding
settings, and conditioning. Listen for intelligibility, artifacts, speaker
similarity, prosody, and memorization.

## Specialized objectives

VoiceHub has distinct adapters for:

- codec/LLM and completion-only TTS;
- diffusion and flow matching;
- VITS adversarial generator/discriminator phases;
- CTC and speech sequence-to-sequence ASR;
- RNN-T and TDT;
- frame and audio classification; and
- model-specific multi-phase recipes.

Do not substitute one family objective for another. Use the
[training matrix](../models/training-support.md) and the selected
`training_spec.phases`.

The repository VITS smoke utility is a **preprocessed generator warm-start**,
not a full adversarial MMS reproduction:

```bash
python scripts/smoke_finetune_vits.py --output-dir runs/vits-smoke
```

## Training optimization

Execution optimization and recipe optimization are separate:

- `TTSOptimizationConfig` selects attention, kernels, and compilation;
- source-specific training profiles select optimizer, precision, batching,
  checkpointing, and EMA only where verified.

Start from the correct eager objective, then follow
[TTS optimization](tts-optimization.md). A serving-optimized model object
should not be reused as a fresh training graph.

## Optional W&B reporting

```python
arguments = TrainingArguments(
    output_dir="runs/dia",
    max_steps=100,
    report_to="wandb",
    wandb_project="voicehub-finetuning",
    wandb_mode="offline",
)
```

Authenticate outside the notebook or script. Credentials are never serialized
into training arguments.

## Failure checklist

- Training validation fails: wrong support level, backend, or artifact type.
- Missing/detached/non-finite loss: inspect the model-owned batch and objective.
- Out of memory: reduce batch size or use a verified memory technique; do not
  silently quantize or lower precision.
- Resume signature mismatch: use the original complete checkpoint and
  unchanged recipe topology.
- Poor validation: verify group-disjoint splits, transcripts, conditioning,
  preprocessing revision, and evaluation policy.

Always record code, checkpoint, and dataset revisions; seeds; device;
precision; arguments; split fingerprints; and licenses. Use only recordings
whose rights and consent permit training.

See the [Trainer concepts](../concepts/trainer.md),
[API reference](../reference/api.md), and [training notebook](notebook.md).
