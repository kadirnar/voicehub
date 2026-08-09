# Trainer

VoiceHub separates speech-model training into four layers:

```text
model integration    loads a differentiable training runtime
training recipe      declares phases, inputs, losses, and parameter ownership
Trainer              runs dataloaders, accumulation, evaluation, and checkpoints
training strategy    performs device, precision, backward, and optimizer operations
```

This separation is deliberate. A language model, CTC or transducer recognizer,
frame classifier, conditional flow model, and VITS generator/discriminator
pair do not share one meaningful fallback objective. `Trainer` provides common
orchestration only after a model integration has exposed a valid training
graph and objective.

Install the training dependencies only when they are needed:

```bash
python -m pip install "voicehub[training] @ git+https://github.com/kadirnar/voicehub.git@main"
```

The main package already contains every built-in inference runtime. The
`training` extra adds only the shared fine-tuning, evaluation, artifact, and
reporting layer.

## Training support is an explicit contract

Every registered model has a `ModelTrainingSpec`, but the presence of a profile
does **not** by itself mean that the current runtime can be fine-tuned.
`TrainingSupport` records the strongest supported boundary:

| Level | Guarantee |
| --- | --- |
| `native` | The integrated runtime exposes a differentiable, backend-native loss that the VoiceHub adapter can execute. |
| `preprocessed` | The differentiable forward path is integrated, but the dataset must already contain backend-shaped tensors. Raw text/audio preparation is outside the generic path. |
| `custom` | The architecture requires model-specific loss or orchestration. A specialized adapter/recipe must be present; a family label alone is not sufficient. |
| `inference-only` | The currently integrated runtime has no verified gradient path. This can describe an ONNX/GGUF backend, a fused engine, an inference-pruned checkpoint, or a wrapper with no training forward. |

`inference-only` describes the VoiceHub integration and checkpoint, not a
theoretical limitation of the architecture. It intentionally fails during
adapter setup instead of producing a plausible-looking but invalid loss.

Inspect the profile before constructing a training job:

```python
from voicehub import get_training_spec

spec = get_training_spec("parlertts")
print(spec.support)
print(spec.is_turnkey)
print(spec.supports_training)
print(spec.has_training_recipe)
print(spec.family_name)
print([phase.name for phase in spec.phases])
```

`is_turnkey` and `supports_training` are true only for `native` and
`preprocessed` profiles. `has_training_recipe` also includes `custom` profiles
whose source recipe is represented but not generically executable. A `custom`
profile is deliberately gated before model loading unless VoiceHub provides a
declarative `adapter_factory` or `AutoTrainingAdapter.register()` has installed
a process-local override.

The profile is the model-type-level boundary. Validate the exact checkpoint,
variant, quantization settings, and runtime before allocating its weights:

```python
model = AutoModelForTextToSpeech.from_pretrained(
    "parler-tts/parler-tts-mini-v1",
    model_type="parlertts",
    lazy_load=True,
)
spec = model.validate_training_support()
```

This catches exclusions such as an inference-only GGUF backbone, a fused
runtime, or training-incompatible quantization options. The common adapter
performs a second check against the loaded module graph, while specialized
loaders can impose tighter checkpoint-family rules. Safetensors are accepted
as weight containers only when they reconstruct that differentiable graph.

The audited TTS model-by-model boundary is listed in the
[TTS training support matrix](../models/training-support.md). ASR and VAD
provider boundaries are listed separately in the
[speech-input support matrix](../models/asr-vad-support.md). Variant and
backend qualifications in those tables are part of the support statement.

## Loading for training

Inference loaders often apply transformations that are unsafe for fine-tuning:
weight-only quantization, ONNX export, graph compilation, module fusion,
weight-normalization removal, EMA-only selection, or deletion of
training-only components.

`PreTrainedTTSModel.load_for_training()` provides a separate lifecycle:

1. `_validate_training_runtime()` rejects an unsupported variant or backend.
2. The checkpoint is loaded while `is_training_load` is true.
3. `_prepare_for_training()` validates or restores training-only state.
4. The adapter resolves the declared source components.

Model implementations should branch on this lifecycle in their loader rather
than undoing destructive inference transformations later. Start training from
a lazy, unloaded wrapper whenever a backend has different inference and
training construction paths:

```python
from voicehub import AutoModelForTextToSpeech

model = AutoModelForTextToSpeech.from_pretrained(
    "your/model",
    model_type="parlertts",
    device="cuda",
    lazy_load=True,
)
```

Do not reuse an object that was already loaded through an inference-only
backend. Construct a new lazy wrapper and let the training adapter request the
training runtime.

## Basic single-phase training

For a runnable raw-data walkthrough from baseline inference through export,
see the [training workflow](../guides/training.md) and its
[companion notebooks](../guides/notebook.md).

For a `native` or `preprocessed` profile, the public loop follows the familiar
Transformers vocabulary:

```python
from voicehub import Trainer, TrainingArguments

arguments = TrainingArguments(
    output_dir="runs/parler-finetune",
    num_train_epochs=10,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    bf16=True,
    eval_strategy="steps",
    eval_steps=250,
    save_strategy="steps",
    save_steps=250,
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=arguments,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    processing_class=model.processor,
)
trainer.train()
```

Use `trainer.train(resume_from_checkpoint=True)` only when `output_dir`
already contains a complete checkpoint.

This example assumes each dataset item already matches the selected model's
training schema. `processing_class` is retained for saving and interoperability;
it is not an implicit raw-text/audio preprocessing pipeline. Tokenization,
audio resampling, codec encoding, speaker conditioning, and family-specific
target construction belong in dataset preparation, a custom collator, the
model's `prepare_training_inputs()` hook, or a specialized training adapter.
When `model.create_training_dataset(...)` returns a dataset with a callable
`collate_fn`, `Trainer` uses that collator automatically unless an explicit
`data_collator` is supplied.

## Batch and collation boundary

Dataset items must be mappings. Common portable fields are:

```text
input_ids              text, semantic, or interleaved token input
attention_mask         mask aligned with input_ids
input_values           waveform, mel, codec, or latent input
labels                 portable primary target
model_inputs           nested mapping of backend-specific tensors
training_phase         optional explicit phase selector
```

`model_inputs` is flattened by the adapter before the backend forward. A key
may appear either at the top level or inside `model_inputs`, but not both:

```python
sample = {
    "model_inputs": {
        "text_tokens": text_tokens,
        "speaker_embedding": speaker_embedding,
        "noisy_latents": noisy_latents,
        "timesteps": timesteps,
    },
    "labels": velocity_target,
}
```

The built-in `DataCollatorForTTSTraining` is intentionally structural:

- it recursively collates nested dictionaries;
- it stacks equal-shaped tensors;
- it pads a variable first dimension, or a variable last dimension when the
  remaining leading dimensions match;
- it pads integer `labels` and `label_ids` with `-100`;
- it pads floating-point features with `0.0`;
- it derives an `attention_mask` only for padded `input_ids`;
- it keeps `training_phase` as one batch-level control and rejects mixed phases;
- it leaves strings and unsupported or ambiguous values as Python lists.

Use `TTSFieldSchema` when a field's sequence dimension is ambiguous or when it
needs lengths or a mask:

```python
from voicehub.training import DataCollatorForTTSTraining, TTSFieldSchema

collator = DataCollatorForTTSTraining(
    field_schemas={
        "model_inputs.mel": TTSFieldSchema(
            sequence_dim=-1,
            padding_side="right",
            length_field="mel_lengths",
            mask_field="mel_mask",
            pad_to_multiple_of=8,
        ),
    },
)
```

Schema paths may be nested. Derived field names without a dot are written next
to the source field; dotted names are written from the batch root. Set
`allow_missing=True` only when an absent sample should become a zero-length
sequence. Configured fields are otherwise strict. `return_input_lengths=True`
adds `input_lengths`; it is disabled by default for compatibility.

The collator does not infer the *meaning* of masks for codec codebooks, mel
frames, or waveforms. It also does not create delay patterns, alignment paths,
noisy flow states, stop labels, pitch targets, or adversarial real/fake pairs.
Declare mechanical padding metadata with a schema and prepare semantic targets
explicitly.

Typical schemas differ by family:

| Family | Inputs that normally must be prepared | Objective boundary |
| --- | --- | --- |
| Causal or codec LM | token framing, codebook layout/delays, causal mask, labels with ignored positions | shifted or model-native masked cross-entropy |
| Encoder-decoder LM | encoder inputs, decoder inputs, decoder labels and masks | teacher-forced model-native cross-entropy |
| Flow matching or diffusion | conditioning, clean state, noise/noisy state, time sample, velocity/noise target, masks | backend-native flow/diffusion loss |
| VITS/acoustic | text/phoneme IDs, lengths, spectrogram, waveform, speaker/language conditioning, alignment auxiliaries | reconstruction, duration, KL, feature-matching, and adversarial losses |
| Hybrid | the union of each component schema plus phase-specific intermediate values | separate component objectives with explicit gradient boundaries |

## Recipe, phase, and optimizer routing

`ModelTrainingSpec` describes an architecture without importing a tensor
framework. Its `family` is extensible and is not a closed enumeration.
`phases` contains one or more `TrainingPhaseSpec` objects.

A phase declares:

- `component_paths`: parameters owned by the phase;
- `optimizer_names`: the optimizer route for those components;
- `forward_component` and `forward_method`: the callable to execute;
- input aliases and required inputs;
- accepted label, prediction, and loss names;
- named loss weights and an optional, explicit fallback objective;
- `frequency` and `offset` for zero-based step scheduling;
- detach and temporary-freeze boundaries;
- optional `optimizer_step_after_phase` for an exact sequential optimizer
  boundary; and
- a semantic kind such as objective, generator, discriminator, or auxiliary.

At a training step, the adapter executes every scheduled phase in declaration
order. `Trainer` backpropagates each returned loss, collects the active
optimizer names, and advances only those optimizers and schedulers at the
gradient-accumulation boundary. Scheduling is based on `global_step`, not on
individual micro-batches. Each optimizer's gradients are normalized by the
number of micro-batches in which that route was active.

VITS and other GAN recipes often require the discriminator to update before a
fresh generator forward. Setting `optimizer_step_after_phase=True` on every
scheduled phase performs that sequence with named optimizers. Because
the next phase must see the just-updated state, this mode currently rejects
gradient accumulation greater than one instead of approximating the recipe.

One optimizer name can own all components in a phase, or names can map
one-to-one to component paths. A parameter may not be assigned to two optimizer
names: adapter setup raises instead of stepping shared storage twice.

For manual phase execution, include `training_phase` in the batch:

```python
batch = {
    "training_phase": "discriminator",
    "model_inputs": discriminator_inputs,
    "labels": real_audio,
}
```

An explicit phase selects one phase. Omitting it lets `Trainer` execute the
scheduled recipe. A generic `compute_loss_func` also represents a single
forward/loss boundary; use a specialized adapter for a true multi-phase recipe.

Architecture-specific optimizer policy can be injected without changing
`Trainer`:

```python
import torch

def optimizer_factory(name, named_parameters, args):
    parameters = [parameter for _, parameter in named_parameters]
    learning_rate = {
        "generator": 2e-4,
        "discriminator": 2e-4,
    }.get(name, args.learning_rate)
    return torch.optim.AdamW(parameters, lr=learning_rate)

trainer = Trainer(
    model=model,
    args=arguments,
    train_dataset=train_dataset,
    optimizer_factory=optimizer_factory,
)
```

`scheduler_factory(name, optimizer, num_training_steps, args)` provides the
corresponding named scheduler boundary. `num_training_steps` is the scheduled
update horizon for that optimizer route, not necessarily the run's global
step count.

## Objective families

### Causal and codec language models

The built-in causal fallback is appropriate only when predictions and labels
are aligned causal token sequences. Speech LMs commonly need more:
audio-control tokens, several delayed codebooks, masked text/audio regions,
codebook-specific heads, or separate text and audio weights. Those semantics
must be implemented by the backend forward or a specialized adapter.

Codec encoders and decoders are normally frozen when they only construct
targets. Do not add a codec to an optimizer merely because it is reachable from
the inference wrapper.

### Flow matching and diffusion

There is no safe universal way to infer a flow target from `labels`. A valid
recipe defines the clean sample, noise distribution, sampled time, conditional
path, target parameterization, masks, and any classifier-free conditioning
dropout. Flow profiles therefore use a native loss or opt into a specific
fallback only when the adapter has established those tensors.

EMA maintenance is also recipe behavior. Selecting an EMA module for inference
is not a substitute for optimizing online parameters and updating EMA after an
optimizer step.

### VITS and adversarial models

A VITS-style fine-tune is not a single waveform regression call. A complete
recipe usually has at least generator and discriminator phases and may add a
duration discriminator. It must define:

- the generator's reconstruction, duration, KL, adversarial, and
  feature-matching terms;
- the real/fake discriminator inputs;
- detachment of generated audio during discriminator training;
- temporary freezing or exclusion of the opposite component;
- update frequency and optimizer ownership;
- checkpoint state for every component.

VoiceHub's native VITS integration implements these boundaries with a
scale-plus-five-period discriminator and independently routed generator and
discriminator phases. An inference-only MMS-TTS snapshot supplies the
generator topology but not its original FFT, hop, window, mel, or segment
settings, so the full route requires an explicit, validated
`training_acoustic_config`. The compatibility generator warm start remains
available, but its artifact manifest does not claim full VITS fine-tuning.

### Hybrid architectures

Models such as semantic-LM plus acoustic-flow plus vocoder systems should expose
one phase per independently callable objective. Intermediate values that cross
phase boundaries must specify whether gradients continue or are detached.
Component discovery is a validation aid, not a recipe: production integrations
should declare exact source paths and loss keys.

## Loss contract

A training forward may return:

```python
SpeechTrainingOutput(loss=loss, logits=logits)
{"loss": loss, "logits": logits}
(loss, logits)
```

Shared adapters normalize TTS phases to the backward-compatible
`TTSTrainingOutput` subclass and ASR/VAD phases to `SpeechTrainingOutput`.
Both carry `training_phase`, `optimizer_names`, individual `losses`, and
metadata. Adapters first look for the phase's declared native loss keys and
apply declared weights. A family fallback runs only when the profile explicitly
permits it.

For a genuinely single-phase external `torch.nn.Module`, a custom loss can be
connected through `compute_loss_func`:

```python
def compute_loss(outputs, labels, num_items_in_batch):
    return acoustic_objective(outputs.audio_values, labels)

trainer = Trainer(
    model=model,
    args=arguments,
    train_dataset=train_dataset,
    compute_loss_func=compute_loss,
)
```

The function receives raw model outputs, labels, and the number of items
represented by the accumulated batch. It cannot supply missing trainable
modules or recover gradients from an inference engine.

## Experiment reporting

Reporting is implemented as a callback boundary rather than embedded in model
adapters or training strategies. Enable the built-in W&B callback with
`report_to="wandb"` and configure it through the `wandb_*` fields on
`TrainingArguments`.

The callback initializes lazily on the world-primary process, records
serializable training and model metadata, namespaces metrics by phase, and
persists its run ID inside VoiceHub's callback checkpoint state. An existing
user-managed `wandb.run` is reused but never finished by VoiceHub. Artifact
upload is deliberately opt-in: `"checkpoint"` uploads only after the atomic
checkpoint completion marker exists, while `"end"` saves and uploads one
portable final artifact.

This separation keeps inference imports lightweight and gives future reporting
backends the same callback lifecycle without coupling them to TTS, ASR, VAD,
or a specific execution strategy.

## Training strategy and optimization boundary

Recipes describe **what** to optimize. `TrainingStrategy` describes **how** to
execute it. The strategy owns model and dataloader preparation, nested input
device movement, autocast and gradient scaling, backward, gradient clipping,
selected optimizer/scheduler steps, `no_sync`, metric gathering, unwrapping,
and runtime checkpoint state. In particular, `prepare_device()` places the
unwrapped graph before explicit optimization, `prepare_training_adapter()`
wraps the already transformed graph into a strategy-owned execution handle,
`execute_training_phase()` routes every recipe phase through that handle,
`prepare_optimization()` may wrap the model and named optimization state
together, and `unwrap_model()` returns the serializable adapter.

The ordering is deliberate:

```text
build graph -> prepare device -> apply persistent passes
            -> create strategy proxy -> create optimizers
            -> prepare optimizer/scheduler runtime
```

Training pass contexts must request persistent results. A nonpersistent pass
is rejected before its `apply()` method runs. If a pass changes parameter
names or topology in a separate-optimizer recipe, it must return a complete
post-transform route for every named optimizer; VoiceHub validates that every
trainable parameter is owned exactly once.

The built-in `"torch"` strategy is single-process. VoiceHub does not currently
pretend to provide built-in DDP, FSDP, DeepSpeed, Accelerate, TPU, or
quantization-aware training. Those runtimes can integrate at the strategy
boundary:

```python
from voicehub.training.strategy import (
    TrainingStrategy,
    register_training_strategy,
)

class MyDistributedStrategy(TrainingStrategy):
    name = "my-distributed"
    # Implement preparation, backward, optimizer, gathering, and state hooks.

register_training_strategy(
    MyDistributedStrategy.name,
    MyDistributedStrategy,
)

trainer = Trainer(
    model=model,
    args=arguments,
    train_dataset=train_dataset,
    training_strategy="my-distributed",
)
```

An inference optimizer belongs on the other side of the model lifecycle. It
may quantize, compile, export, or fuse a trained model for serving, but its
runtime is not assumed to remain differentiable. A future optimization
integration can therefore target inference loading, the training strategy, or
both without changing the phase recipe schema.

## Versioned, resumable checkpoints

Checkpoints are written to a temporary directory and atomically renamed only
after all state is saved. A completed format-version-3 checkpoint contains:

```text
checkpoint-<global_step>/
  config.json                present for a VoiceHub pretrained wrapper
  generation_config.json     present for a VoiceHub pretrained wrapper
  processor_config.json      present for a VoiceHub pretrained wrapper
  model_state.pt
  optimizer.pt
  scheduler.pt
  scaler.pt                 present when gradient scaling is active
  rng_state.pth             PyTorch CPU/CUDA/MPS state when available
  training_runtime.pt       Python/NumPy RNG, callbacks, sampler, strategy
  trainer_state.json
  training_args.json
  training_recipe.json       present when a training adapter is active
  checkpoint_manifest.json
  .complete
```

The root is always a VoiceHub artifact. A deliberate `trainer.save_model()`
also asks the adapter for a native Hugging Face or source-repository export.
Those files are isolated under `native_export/`, so an upstream `config.json`
cannot overwrite VoiceHub's reload metadata. Periodic exact-resume checkpoints
skip this duplicate export. The recipe manifest labels a generated native
export either as an inference export or as a component weight warm start;
XTTS GPT, CosyVoice component, and F5 EMA files are not advertised as complete
source-loadable model directories.

The adapter-supplied recipe manifest is portable metadata, not a credential
store. `Trainer.save_model()` rejects nested credential-shaped fields before
model or native-export state is written or the destination is created, then
rechecks the final manifest before output. Safe descriptive fields such as
`token_count` remain serializable.

An exact checkpoint may retain optimized state only because its pass manifest
is part of the resume identity and every Trainer pass is explicitly
persistent. A public/final portable artifact uses canonical state instead. A
topology/name-changing pass must declare and implement portable export;
otherwise `save_model()` fails before writing an artifact. This prevents an
optimized proxy state dict from being advertised as loadable by a fresh
unoptimized runtime.

The model state is the adapter's versioned component topology for integrated
TTS wrappers. Optimizer and scheduler bundles retain their names, so a
generator/discriminator or multi-component job resumes with the same routing.
Recipe-owned state, such as F5-TTS EMA shadows or Qwen3-TTS's target speaker
embedding, is handed from a portable model load to the concrete Trainer
adapter only after its auxiliary training graph has been constructed.
The manifest records the format version, global step, model type, adapter
identity, optimizer names, training strategy, required files, byte sizes, and
SHA-256 digests.

Format 3 also records a resolved exact-resume signature: effective batch/data
seed and dataloader topology, schedule horizon and per-optimizer warmup,
optimizer/scheduler classes, precision and scaler mode, strategy/world size,
recipe configuration, and ordered stateful callbacks. A dataset or collator
may implement `resume_fingerprint()` to identify its content/revision; this is
the only generic way to detect a same-length dataset whose contents or ordering
changed. Resume rejects an incomplete checkpoint, a newer unsupported format,
a different model type or strategy, a mismatched named-optimizer topology, or
a changed optimization configuration or exact-resume signature. Optimization
records snapshot the pass ID, kind, version, capabilities, configuration, and
result metadata as strict JSON. Legacy checkpoints remain readable, but
formats without this signature cannot make the same topology guarantee.

```python
trainer.train(resume_from_checkpoint=True)  # newest complete numeric checkpoint
```

Exact generic mid-epoch resume requires a stable, sized dataloader and
`dataloader_num_workers=0`; worker prefetch state and arbitrary iterable
cursors cannot be reconstructed portably. Python, NumPy, and framework RNG
state are restored after replaying the deterministic dataloader cursor.
Changing `max_steps`, accumulation, batching, schedule, precision, or recipe
configuration starts a different training plan and is rejected by exact
resume. Load a saved VoiceHub model artifact as a weight-only warm start for
that case.

Evaluation routes through the adapter's selected evaluation phase and marks its
`TrainingContext` as non-training. A batch may still explicitly supply
`training_phase`; genuinely different multi-phase validation objectives can
use phase-tagged samples or separate named evaluation splits.

For reproducibility, keep dataset manifests, preprocessing/codec versions,
model source revision, and custom adapter code beside the run. They are inputs
to the recipe and are not embedded automatically in tensor state.

## Adding a trainable model family

A production integration should:

1. implement a non-destructive `load_for_training()` lifecycle;
2. define the exact backend batch schema and preprocessing ownership;
3. expose a native scalar loss or implement a specialized adapter objective;
4. register explicit component and forward paths;
5. declare phase scheduling, detach/freeze boundaries, and optimizer routes;
6. choose the most conservative truthful `TrainingSupport` level;
7. test gradients, parameter ownership, one optimizer update, evaluation, and
   checkpoint round-trip;
8. document variant/backend exclusions in the model matrix.

Declare a model-specific implementation as the profile's lazy
`adapter_factory="module:callable"`; this keeps the recipe beside its model and
requires no edit to a central provider map. `AutoTrainingAdapter.register()`
installs a process-local model override, and
`AutoTrainingAdapter.register_family()` installs a reusable adapter for a new
family string. Dynamic specs can be registered through
`voicehub.training.specs.register_training_spec`. Keep family adapters
narrow: when two architectures disagree about target construction or loss
semantics, that behavior belongs in their model-specific recipes.
Reusable recipe bases live in `voicehub.training.recipes`. Historical imports
of the built-in model adapters from that module remain lazy compatibility
aliases to their model-local implementations.
