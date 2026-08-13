---
description: Public API, checkpoint, training, and optimization guide for the kokoro integration.
hide:
  - toc
---

<div class="vh-model-detail" data-vh-model-detail data-model-type="kokoro" data-task="text-to-speech" data-training="preprocessed" data-parameter-count="81810022" markdown>

<header class="vh-model-detail__hero" data-vh-model-hero markdown>

<p class="vh-model-detail__namespace" aria-label="Model repository"><span class="vh-model-detail__owner-avatar" aria-hidden="true">HE</span><a href="https://huggingface.co/hexgrad">hexgrad</a><span aria-hidden="true">/</span><strong>Kokoro-82M</strong></p>

# Kokoro {.vh-model-title}

<p class="vh-model-detail__summary">Selects a Kokoro voice ID and explicit speaking speed.</p>
<div class="vh-model-detail__tags" aria-label="Model metadata"><span class="vh-model-detail__chip" data-chip-kind="task">Text to speech</span><span class="vh-model-detail__chip" data-chip-kind="runtime">VoiceHub-native</span><span class="vh-model-detail__chip" data-chip-kind="architecture">kokoro</span><span class="vh-model-detail__chip" data-chip-kind="parameters" aria-describedby="vh-model-parameters-note-kokoro">Parameters: 81.8M</span><span class="vh-model-detail__chip" data-chip-kind="language">Languages: en-US, en-GB +7</span><span class="vh-model-detail__chip" data-chip-kind="training">Training: preprocessed</span><span class="vh-model-detail__chip" data-chip-kind="license">License: Checkpoint-specific</span></div>
<p class="vh-model-detail__parameter-note" id="vh-model-parameters-note-kokoro"><strong>Parameter metadata:</strong> Exact learned-parameter total for VoiceHub&#x27;s audited native primary graph at the registered default selection; separately loaded auxiliary models are excluded.</p>
<div class="vh-model-detail__actions" aria-label="Model actions">
<a class="vh-model-detail__action vh-model-detail__action--primary" href="#usage" data-vh-model-action="use">Use this model</a>
<button class="vh-model-detail__action vh-model-detail__copy" type="button" data-vh-copy-model-id data-model-id="hexgrad/Kokoro-82M" aria-describedby="vh-model-checkpoint-kokoro"><span data-vh-copy-model-id-label>Copy model ID</span></button>
<a class="vh-model-detail__action" href="https://huggingface.co/hexgrad/Kokoro-82M" data-vh-model-action="checkpoint">Checkpoint</a>
<details class="vh-model-detail__resources">
<summary class="vh-model-detail__action">Resources</summary>
<div class="vh-model-detail__resource-menu">
<a href="https://github.com/hexgrad/kokoro" data-vh-model-action="github">Upstream GitHub</a>
<a href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/kokoro/modeling_kokoro.py" data-vh-model-action="source">VoiceHub source</a>
<a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/kokoro.ipynb" data-vh-model-action="colab">Open in Colab</a>
</div>
</details>
</div>
</header>

<nav class="vh-model-detail__tabs" aria-label="Model sections"><a href="#usage" data-vh-model-tab="usage">Usage</a><a href="#overview" data-vh-model-tab="model-card" aria-current="location">Model card</a><a href="#paper-and-github" data-vh-model-tab="sources">Sources</a><a href="#training-and-optimization" data-vh-model-tab="training">Training</a><a href="#checkpoints-provenance-license-and-limitations" data-vh-model-tab="checkpoint">Checkpoint</a><a href="#public-api" data-vh-model-tab="api">Public API</a></nav>

<div class="vh-model-detail__layout" markdown>

<aside class="vh-model-detail__sidebar" data-vh-model-facts aria-labelledby="vh-model-facts-title-kokoro"><h2 id="vh-model-facts-title-kokoro">Model facts</h2><details class="vh-model-detail__facts-disclosure" data-vh-model-facts-disclosure aria-labelledby="vh-model-facts-title-kokoro" open><summary><span>Toggle model facts</span></summary><dl class="vh-model-detail__facts"><div><dt>Task</dt><dd>Text to speech</dd></div><div><dt>Parameters</dt><dd aria-describedby="vh-model-parameters-note-kokoro">81.8M</dd></div><div><dt>Architecture</dt><dd><code>kokoro</code></dd></div><div><dt>Runtime</dt><dd>VoiceHub-native</dd></div><div><dt>Languages</dt><dd><details class="vh-model-detail__languages"><summary>9 documented codes</summary><span><code>en-US</code> <code>en-GB</code> <code>es</code> <code>fr</code> <code>hi</code> <code>it</code> <code>pt-BR</code> <code>ja</code> <code>zh</code></span></details></dd></div><div><dt>Capabilities</dt><dd><details class="vh-model-detail__capabilities"><summary>6 capabilities</summary><span><code>text-to-speech</code> <code>multilingual</code> <code>fine-tuning</code> <code>safetensors</code> <code>voicehub-native</code> <code>native-runtime</code></span></details></dd></div><div><dt>Training</dt><dd><code>preprocessed</code></dd></div><div><dt>License</dt><dd>Checkpoint-specific</dd></div><div><dt>Default checkpoint</dt><dd id="vh-model-checkpoint-kokoro"><a href="https://huggingface.co/hexgrad/Kokoro-82M"><code>hexgrad/Kokoro-82M</code></a></dd></div></dl></details></aside>

<div class="vh-model-detail__main vh-model-detail__content" markdown>

## Usage

Complete the [VoiceHub installation](../../getting-started/installation.md) once,
then run this repository-authored example. Model pages intentionally contain no
package-install command.

This example is maintained against VoiceHub's public API; it is not copied from an upstream demo or package README.

**Model-specific path:** Selects a Kokoro voice ID and explicit speaking speed.

**Inputs and controls:** Voice IDs are checkpoint-specific; `af_heart` belongs to the registered Kokoro release.

```python
from pathlib import Path

from voicehub import AutoModelForTextToSpeech, TTSGenerationConfig

model = AutoModelForTextToSpeech.from_pretrained(
    'hexgrad/Kokoro-82M',
    model_type='kokoro',
    device="cuda",
    lazy_load=True,
)
output = model.generate(
    'VoiceHub keeps model integrations explicit and reproducible.',
    generation_config=TTSGenerationConfig(
        seed=42,
        output_file=Path("output.wav"),
    ),
    voice="af_heart",
    speed=1.0,
)
print(output.file_path, output.sample_rate, output.metadata)
```

Use authorized recordings. Verify hardware needs and pin a revision in production.

## Overview

`kokoro` is a VoiceHub **text to speech**
integration. This page is generated from its registry contract. [Open the `kokoro` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/kokoro.ipynb).

| Property | Value |
| --- | --- |
| Task | Text to speech |
| Architecture | `kokoro` |
| Runtime | `VoiceHub-native` |
| Languages | `en-US`, `en-GB`, `es`, `fr`, … complete audited list below |
| Capabilities | `text-to-speech`, `multilingual`, `fine-tuning`, `safetensors`, `voicehub-native`, `native-runtime` |
| Reusable components | — |
| Normalized output | `TTSOutput` |

### Language support

<details class="vh-language-support" markdown>
<summary>Supported language abbreviations</summary>

`en-US`, `en-GB`, `es`, `fr`, `hi`, `it`, `pt-BR`, `ja`, `zh`

</details>

## Paper and GitHub

- **Paper:** No dedicated upstream research paper is published for this integration.
- **Upstream GitHub:** [Kokoro](https://github.com/hexgrad/kokoro)
- **VoiceHub source:** [VoiceHub model implementation](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/kokoro/modeling_kokoro.py)

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('kokoro')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `kokoro` |
| Configuration class | `KokoroConfig` |
| Architecture class | `KokoroForTextToSpeech` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'hexgrad/Kokoro-82M',
    model_type='kokoro',
)
print(type(processor).__name__)
```

## Inference

The Usage example returns `TTSOutput` through `AutoModelForTextToSpeech`.

### Input and output contract

| Property | Value |
| --- | --- |
| Readiness | `preprocessed` |
| Data architecture | `acoustic` |
| Sample rate | 24,000 Hz |
| Contract getter | `get_tts_dataset_spec('kokoro')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `full-preprocessed` | `durations` | input_ids / phonemes; ref_s / voice; audio_values / audio / labels | Prepared | — |
| `duration-only` | `durations`, `training_phase` | input_ids / phonemes; ref_s / voice | Prepared | — |

Direct acoustic, mel, codec, or waveform regression data. See the [data workflow](../../guides/data-preparation.md).

## Training and optimization

Use `available_optimization_passes()` to discover reversible public passes.
Unsupported runtime or hardware fails closed before mutation.

### Training contract

| Property | Value |
| --- | --- |
| Support | `preprocessed` |
| Family | `acoustic-regression` |
| Recipe | `multi-phase` |
| Default phase | `acoustic` |
| Training checkpoint | `hexgrad/Kokoro-82M` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `duration` | objective | `model.bert`, `model.bert_encoder`, `model.predictor` | `input_ids`, `ref_s`, `durations` | `loss` |
| `acoustic` | objective | `model.bert`, `model.bert_encoder`, `model.predictor`, `model.text_encoder`, `model.decoder` | `input_ids`, `ref_s`, `durations`, `audio_values` | `loss` |

Prepare the exact tensors listed in the data contract before this step. Call `model.validate_training_support()` first, then follow the
[training workflow](../../guides/training.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | [`hexgrad/Kokoro-82M`](https://huggingface.co/hexgrad/Kokoro-82M) |
| Hugging Face ID | [`hexgrad/Kokoro-82M`](https://huggingface.co/hexgrad/Kokoro-82M)<br>Repository availability verified through the Hugging Face model API on 2026-08-11; pin a revision before production use. |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.kokoro.modeling_kokoro.KokoroForTextToSpeech` |
| Configuration | `voicehub.models.kokoro.configuration_kokoro.KokoroConfig` |
| Source provenance | `voicehub/models/kokoro/source/SOURCE.json` |
| License | Checkpoint-specific |

No VoiceHub-specific license override is registered. Verify the checkpoint and upstream source terms before use.

Confirm the checkpoint revision, access terms, provenance, and license.

### Limitations

- No integration-specific checkpoint limitation is registered. Verify the selected checkpoint revision and its documented runtime requirements.
- Validate memory, precision, and optional dependencies on the target system.
- Public optimizations fail closed when the runtime or hardware cannot satisfy
  their validation contract; an unavailable pass is not reported as applied.
- Contract tests do not replace the linked released-checkpoint evidence.

## Public API

Use the stable configuration, processor, and task-model facades below.

<section class="vh-model-api-card" data-vh-model-api-card="configuration" markdown>
<p class="vh-model-api-card__badge-wrap"><span class="vh-model-api-card__badge">Configuration</span></p>

### `KokoroConfig`

<p class="vh-model-api-card__source-wrap"><a class="vh-model-api-card__source" href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/kokoro/configuration_kokoro.py">View source</a></p>
<div class="vh-model-api-card__signature" markdown>

```text
KokoroConfig(**config_kwargs)
```

</div>
<h4>Parameters</h4>
<div class="vh-model-api-card__parameters" markdown>
- `**config_kwargs` — Configuration fields validated by KokoroConfig.
</div>
</section>

<section class="vh-model-api-card" data-vh-model-api-card="model" markdown>
<p class="vh-model-api-card__badge-wrap"><span class="vh-model-api-card__badge">Model</span></p>

### `KokoroForTextToSpeech`

<p class="vh-model-api-card__source-wrap"><a class="vh-model-api-card__source" href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/kokoro/modeling_kokoro.py">View source</a></p>
<div class="vh-model-api-card__signature" markdown>

```text
AutoModelForTextToSpeech.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='kokoro',
    config=None,
    **model_kwargs,
)
```

</div>
<h4>Parameters</h4>
<div class="vh-model-api-card__parameters" markdown>
- `pretrained_model_name_or_path` — Hub ID or compatible local directory.
- `model_type` — Canonical model type; use 'kokoro'.
- `config` — Optional preloaded KokoroConfig instance.
- `**model_kwargs` — Model-specific loading arguments.
</div>
</section>

```python
from voicehub import get_model_spec

spec = get_model_spec('kokoro')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('kokoro')` |
| Load and run | `AutoModelForTextToSpeech` |
| Configure | `KokoroConfig` |
| Process | `AutoProcessor` |
| Model implementation | `KokoroForTextToSpeech` |
| Normalized output | `TTSOutput` |
| Training contract | `get_training_spec('kokoro')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).

</div>

</div>

</div>
