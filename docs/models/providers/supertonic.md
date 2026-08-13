---
description: Public API, checkpoint, training, and optimization guide for the supertonic integration.
hide:
  - toc
---

<div class="vh-model-detail" data-vh-model-detail data-model-type="supertonic" data-task="text-to-speech" data-training="preprocessed" data-parameter-count="" markdown>

<header class="vh-model-detail__hero" data-vh-model-hero markdown>

<p class="vh-model-detail__namespace" aria-label="Model repository"><span class="vh-model-detail__owner-avatar" aria-hidden="true">SU</span><a href="https://huggingface.co/Supertone">Supertone</a><span aria-hidden="true">/</span><strong>supertonic-3</strong></p>

# Supertonic {.vh-model-title}

<p class="vh-model-detail__summary">Selects a Supertonic style ID, language, diffusion-step count, and speaking speed.</p>
<div class="vh-model-detail__tags" aria-label="Model metadata"><span class="vh-model-detail__chip" data-chip-kind="task">Text to speech</span><span class="vh-model-detail__chip" data-chip-kind="runtime">VoiceHub-native</span><span class="vh-model-detail__chip" data-chip-kind="architecture">supertonic</span><span class="vh-model-detail__chip" data-chip-kind="parameters" aria-describedby="vh-model-parameters-note-supertonic">Parameters: Not reported</span><span class="vh-model-detail__chip" data-chip-kind="language">Languages: en, ko +29</span><span class="vh-model-detail__chip" data-chip-kind="training">Training: preprocessed</span><span class="vh-model-detail__chip" data-chip-kind="license">License: Checkpoint-specific</span></div>
<p class="vh-model-detail__parameter-note" id="vh-model-parameters-note-supertonic"><strong>Parameter metadata:</strong> Not reported: the audited metadata available for the registered default does not provide an exact parameter total.</p>
<div class="vh-model-detail__actions" aria-label="Model actions">
<a class="vh-model-detail__action vh-model-detail__action--primary" href="#usage" data-vh-model-action="use">Use this model</a>
<button class="vh-model-detail__action vh-model-detail__copy" type="button" data-vh-copy-model-id data-model-id="Supertone/supertonic-3" aria-describedby="vh-model-checkpoint-supertonic"><span data-vh-copy-model-id-label>Copy model ID</span></button>
<a class="vh-model-detail__action" href="https://huggingface.co/Supertone/supertonic-3" data-vh-model-action="checkpoint">Checkpoint</a>
<details class="vh-model-detail__resources">
<summary class="vh-model-detail__action">Resources</summary>
<div class="vh-model-detail__resource-menu">
<a href="https://github.com/supertone-inc/supertonic" data-vh-model-action="github">Upstream GitHub</a>
<a href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/supertonic/modeling_supertonic.py" data-vh-model-action="source">VoiceHub source</a>
<a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/supertonic.ipynb" data-vh-model-action="colab">Open in Colab</a>
</div>
</details>
</div>
</header>

<nav class="vh-model-detail__tabs" aria-label="Model sections"><a href="#usage" data-vh-model-tab="usage">Usage</a><a href="#overview" data-vh-model-tab="model-card" aria-current="location">Model card</a><a href="#paper-and-github" data-vh-model-tab="sources">Sources</a><a href="#training-and-optimization" data-vh-model-tab="training">Training</a><a href="#checkpoints-provenance-license-and-limitations" data-vh-model-tab="checkpoint">Checkpoint</a><a href="#public-api" data-vh-model-tab="api">Public API</a></nav>

<div class="vh-model-detail__layout" markdown>

<aside class="vh-model-detail__sidebar" data-vh-model-facts aria-labelledby="vh-model-facts-title-supertonic"><h2 id="vh-model-facts-title-supertonic">Model facts</h2><details class="vh-model-detail__facts-disclosure" data-vh-model-facts-disclosure aria-labelledby="vh-model-facts-title-supertonic" open><summary><span>Toggle model facts</span></summary><dl class="vh-model-detail__facts"><div><dt>Task</dt><dd>Text to speech</dd></div><div><dt>Parameters</dt><dd aria-describedby="vh-model-parameters-note-supertonic">Not reported</dd></div><div><dt>Architecture</dt><dd><code>supertonic</code></dd></div><div><dt>Runtime</dt><dd>VoiceHub-native</dd></div><div><dt>Languages</dt><dd><details class="vh-model-detail__languages"><summary>31 documented codes</summary><span><code>en</code> <code>ko</code> <code>ja</code> <code>ar</code> <code>bg</code> <code>cs</code> <code>da</code> <code>de</code> <code>el</code> <code>es</code> <code>et</code> <code>fi</code> <code>fr</code> <code>hi</code> <code>hr</code> <code>hu</code> <code>id</code> <code>it</code> <code>lt</code> <code>lv</code> <code>nl</code> <code>pl</code> <code>pt</code> <code>ro</code> <code>ru</code> <code>sk</code> <code>sl</code> <code>sv</code> <code>tr</code> <code>uk</code> <code>vi</code></span></details></dd></div><div><dt>Capabilities</dt><dd><details class="vh-model-detail__capabilities"><summary>7 capabilities</summary><span><code>text-to-speech</code> <code>multilingual</code> <code>fine-tuning</code> <code>safetensors</code> <code>voicehub-native</code> <code>native-runtime</code> <code>preprocessed-training</code></span></details></dd></div><div><dt>Training</dt><dd><code>preprocessed</code></dd></div><div><dt>License</dt><dd>Checkpoint-specific</dd></div><div><dt>Default checkpoint</dt><dd id="vh-model-checkpoint-supertonic"><a href="https://huggingface.co/Supertone/supertonic-3"><code>Supertone/supertonic-3</code></a></dd></div></dl></details></aside>

<div class="vh-model-detail__main vh-model-detail__content" markdown>

## Usage

Complete the [VoiceHub installation](../../getting-started/installation.md) once,
then run this repository-authored example. Model pages intentionally contain no
package-install command.

This example is maintained against VoiceHub's public API; it is not copied from an upstream demo or package README.

**Model-specific path:** Selects a Supertonic style ID, language, diffusion-step count, and speaking speed.

**Inputs and controls:** Voice/style IDs and languages are validated against files published by the checkpoint.

```python
from pathlib import Path

from voicehub import AutoModelForTextToSpeech, TTSGenerationConfig

model = AutoModelForTextToSpeech.from_pretrained(
    'Supertone/supertonic-3',
    model_type='supertonic',
    device="cuda",
    lazy_load=True,
)
output = model.generate(
    'VoiceHub keeps model integrations explicit and reproducible.',
    generation_config=TTSGenerationConfig(
        seed=42,
        output_file=Path("output.wav"),
    ),
    voice="F1",
    language="en",
    total_steps=5,
    speed=1.05,
)
print(output.file_path, output.sample_rate, output.metadata)
```

Use authorized recordings. Verify hardware needs and pin a revision in production.

## Overview

`supertonic` is a VoiceHub **text to speech**
integration. This page is generated from its registry contract. [Open the `supertonic` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/supertonic.ipynb).

| Property | Value |
| --- | --- |
| Task | Text to speech |
| Architecture | `supertonic` |
| Runtime | `VoiceHub-native` |
| Languages | `en`, `ko`, `ja`, `ar`, … complete audited list below |
| Capabilities | `text-to-speech`, `multilingual`, `fine-tuning`, `safetensors`, `voicehub-native`, `native-runtime`, `preprocessed-training` |
| Reusable components | — |
| Normalized output | `TTSOutput` |

### Language support

<details class="vh-language-support" markdown>
<summary>Supported language abbreviations</summary>

`en`, `ko`, `ja`, `ar`, `bg`, `cs`, `da`, `de`, `el`, `es`, `et`, `fi`, `fr`, `hi`, `hr`, `hu`, `id`, `it`, `lt`, `lv`, `nl`, `pl`, `pt`, `ro`, `ru`, `sk`, `sl`, `sv`, `tr`, `uk`, `vi`

</details>

## Paper and GitHub

- **Paper:** No dedicated upstream research paper is published for this integration.
- **Upstream GitHub:** [Supertonic](https://github.com/supertone-inc/supertonic)
- **VoiceHub source:** [VoiceHub model implementation](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/supertonic/modeling_supertonic.py)

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('supertonic')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `supertonic` |
| Configuration class | `SupertonicConfig` |
| Architecture class | `SupertonicForTextToSpeech` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'Supertone/supertonic-3',
    model_type='supertonic',
)
print(type(processor).__name__)
```

## Inference

The Usage example returns `TTSOutput` through `AutoModelForTextToSpeech`.

### Input and output contract

| Property | Value |
| --- | --- |
| Readiness | `preprocessed` |
| Data architecture | `diffusion` |
| Sample rate | 44,100 Hz |
| Contract getter | `get_tts_dataset_spec('supertonic')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `text-style-object` | `text`, `style` | target_duration / duration / duration_seconds / target_latent / latent / latents | Prepared | — |
| `text-style-tensors` | `text`, `style_ttl`, `style_dp` | target_duration / duration / duration_seconds / target_latent / latent / latents | Prepared | — |
| `tokenized-style-object` | `text_ids`, `style` | text_mask / text_lengths; target_duration / duration / duration_seconds / target_latent / latent / latents | Prepared | — |
| `tokenized-style-tensors` | `text_ids`, `style_ttl`, `style_dp` | text_mask / text_lengths; target_duration / duration / duration_seconds / target_latent / latent / latents | Prepared | — |

Conditional flow-matching, rectified-flow, or diffusion data. See the [data workflow](../../guides/data-preparation.md).

## Training and optimization

Use `available_optimization_passes()` to discover reversible public passes.
Unsupported runtime or hardware fails closed before mutation.

### Training contract

| Property | Value |
| --- | --- |
| Support | `preprocessed` |
| Family | `flow-matching` |
| Recipe | `single-phase` |
| Default phase | `published_graph` |
| Training checkpoint | `Supertone/supertonic-3` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `published_graph` | objective | `model` | `text_ids`, `text_mask`, `style_ttl`, `style_dp` | `loss`, `duration_loss`, `flow_step_loss`, `vocoder_l1_loss` |

Prepare the exact tensors listed in the data contract before this step. Call `model.validate_training_support()` first, then follow the
[training workflow](../../guides/training.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | [`Supertone/supertonic-3`](https://huggingface.co/Supertone/supertonic-3) |
| Hugging Face ID | [`Supertone/supertonic-3`](https://huggingface.co/Supertone/supertonic-3)<br>Repository availability verified through the Hugging Face model API on 2026-08-11; pin a revision before production use. |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.supertonic.modeling_supertonic.SupertonicForTextToSpeech` |
| Configuration | `voicehub.models.supertonic.configuration_supertonic.SupertonicConfig` |
| Source provenance | `voicehub/models/supertonic/source/SOURCE.json` |
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

### `SupertonicConfig`

<p class="vh-model-api-card__source-wrap"><a class="vh-model-api-card__source" href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/supertonic/configuration_supertonic.py">View source</a></p>
<div class="vh-model-api-card__signature" markdown>

```text
SupertonicConfig(**config_kwargs)
```

</div>
<h4>Parameters</h4>
<div class="vh-model-api-card__parameters" markdown>
- `**config_kwargs` — Configuration fields validated by SupertonicConfig.
</div>
</section>

<section class="vh-model-api-card" data-vh-model-api-card="model" markdown>
<p class="vh-model-api-card__badge-wrap"><span class="vh-model-api-card__badge">Model</span></p>

### `SupertonicForTextToSpeech`

<p class="vh-model-api-card__source-wrap"><a class="vh-model-api-card__source" href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/supertonic/modeling_supertonic.py">View source</a></p>
<div class="vh-model-api-card__signature" markdown>

```text
AutoModelForTextToSpeech.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='supertonic',
    config=None,
    **model_kwargs,
)
```

</div>
<h4>Parameters</h4>
<div class="vh-model-api-card__parameters" markdown>
- `pretrained_model_name_or_path` — Hub ID or compatible local directory.
- `model_type` — Canonical model type; use 'supertonic'.
- `config` — Optional preloaded SupertonicConfig instance.
- `**model_kwargs` — Model-specific loading arguments.
</div>
</section>

```python
from voicehub import get_model_spec

spec = get_model_spec('supertonic')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('supertonic')` |
| Load and run | `AutoModelForTextToSpeech` |
| Configure | `SupertonicConfig` |
| Process | `AutoProcessor` |
| Model implementation | `SupertonicForTextToSpeech` |
| Normalized output | `TTSOutput` |
| Training contract | `get_training_spec('supertonic')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).

</div>

</div>

</div>
