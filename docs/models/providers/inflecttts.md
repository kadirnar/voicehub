---
description: Public API, checkpoint, training, and optimization guide for the inflecttts integration.
hide:
  - toc
---

<div class="vh-model-detail" data-vh-model-detail data-model-type="inflecttts" data-task="text-to-speech" data-training="preprocessed" data-parameter-count="" markdown>

<header class="vh-model-detail__hero" data-vh-model-hero markdown>

<p class="vh-model-detail__namespace" aria-label="Model repository"><span class="vh-model-detail__owner-avatar" aria-hidden="true">OW</span><a href="https://huggingface.co/owensong">owensong</a><span aria-hidden="true">/</span><strong>Inflect-Micro-v2</strong></p>

# InflectTTS {.vh-model-title}

<p class="vh-model-detail__summary">Uses Inflect&#x27;s normalized-text frontend with explicit speed and variation controls.</p>
<div class="vh-model-detail__tags" aria-label="Model metadata"><span class="vh-model-detail__chip" data-chip-kind="task">Text to speech</span><span class="vh-model-detail__chip" data-chip-kind="runtime">VoiceHub-native</span><span class="vh-model-detail__chip" data-chip-kind="architecture">inflecttts</span><span class="vh-model-detail__chip" data-chip-kind="parameters" aria-describedby="vh-model-parameters-note-inflecttts">Parameters: Not reported</span><span class="vh-model-detail__chip" data-chip-kind="language">Language: en-US</span><span class="vh-model-detail__chip" data-chip-kind="training">Training: preprocessed</span><span class="vh-model-detail__chip" data-chip-kind="license">License: Checkpoint-specific</span></div>
<p class="vh-model-detail__parameter-note" id="vh-model-parameters-note-inflecttts"><strong>Parameter metadata:</strong> Not reported: the audited metadata available for the registered default does not provide an exact parameter total.</p>
<div class="vh-model-detail__actions" aria-label="Model actions">
<a class="vh-model-detail__action vh-model-detail__action--primary" href="#usage" data-vh-model-action="use">Use this model</a>
<button class="vh-model-detail__action vh-model-detail__copy" type="button" data-vh-copy-model-id data-model-id="owensong/Inflect-Micro-v2" aria-describedby="vh-model-checkpoint-inflecttts"><span data-vh-copy-model-id-label>Copy model ID</span></button>
<a class="vh-model-detail__action" href="https://huggingface.co/owensong/Inflect-Micro-v2" data-vh-model-action="checkpoint">Checkpoint</a>
<details class="vh-model-detail__resources">
<summary class="vh-model-detail__action">Resources</summary>
<div class="vh-model-detail__resource-menu">
<a href="https://github.com/owenawsong/Inflect" data-vh-model-action="github">Upstream GitHub</a>
<a href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/inflecttts/modeling_inflecttts.py" data-vh-model-action="source">VoiceHub source</a>
<a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/inflecttts.ipynb" data-vh-model-action="colab">Open in Colab</a>
</div>
</details>
</div>
</header>

<nav class="vh-model-detail__tabs" aria-label="Model sections"><a href="#usage" data-vh-model-tab="usage">Usage</a><a href="#overview" data-vh-model-tab="model-card" aria-current="location">Model card</a><a href="#paper-and-github" data-vh-model-tab="sources">Sources</a><a href="#training-and-optimization" data-vh-model-tab="training">Training</a><a href="#checkpoints-provenance-license-and-limitations" data-vh-model-tab="checkpoint">Checkpoint</a><a href="#public-api" data-vh-model-tab="api">Public API</a></nav>

<div class="vh-model-detail__layout" markdown>

<aside class="vh-model-detail__sidebar" data-vh-model-facts aria-labelledby="vh-model-facts-title-inflecttts"><h2 id="vh-model-facts-title-inflecttts">Model facts</h2><details class="vh-model-detail__facts-disclosure" data-vh-model-facts-disclosure aria-labelledby="vh-model-facts-title-inflecttts" open><summary><span>Toggle model facts</span></summary><dl class="vh-model-detail__facts"><div><dt>Task</dt><dd>Text to speech</dd></div><div><dt>Parameters</dt><dd aria-describedby="vh-model-parameters-note-inflecttts">Not reported</dd></div><div><dt>Architecture</dt><dd><code>inflecttts</code></dd></div><div><dt>Runtime</dt><dd>VoiceHub-native</dd></div><div><dt>Languages</dt><dd><code>en-US</code></dd></div><div><dt>Capabilities</dt><dd><details class="vh-model-detail__capabilities"><summary>8 capabilities</summary><span><code>text-to-speech</code> <code>fine-tuning</code> <code>safetensors</code> <code>voicehub-native</code> <code>native-runtime</code> <code>preprocessed-training</code> <code>vits-warm-start</code> <code>explicit-phonemes</code></span></details></dd></div><div><dt>Training</dt><dd><code>preprocessed</code></dd></div><div><dt>License</dt><dd>Checkpoint-specific</dd></div><div><dt>Default checkpoint</dt><dd id="vh-model-checkpoint-inflecttts"><a href="https://huggingface.co/owensong/Inflect-Micro-v2"><code>owensong/Inflect-Micro-v2</code></a></dd></div></dl></details></aside>

<div class="vh-model-detail__main vh-model-detail__content" markdown>

## Usage

Complete the [VoiceHub installation](../../getting-started/installation.md) once,
then run this repository-authored example. Model pages intentionally contain no
package-install command.

This example is maintained against VoiceHub's public API; it is not copied from an upstream demo or package README.

**Model-specific path:** Uses Inflect's normalized-text frontend with explicit speed and variation controls.

**Inputs and controls:** Set `input_is_phonemes=True` only when supplying checkpoint-compatible phoneme text.

```python
from pathlib import Path

from voicehub import AutoModelForTextToSpeech, TTSGenerationConfig

model = AutoModelForTextToSpeech.from_pretrained(
    'owensong/Inflect-Micro-v2',
    model_type='inflecttts',
    device="cuda",
    lazy_load=True,
)
output = model.generate(
    'VoiceHub keeps model integrations explicit and reproducible.',
    generation_config=TTSGenerationConfig(
        seed=42,
        output_file=Path("output.wav"),
    ),
    speed=1.0,
    variation=0.3,
)
print(output.file_path, output.sample_rate, output.metadata)
```

Use authorized recordings. Verify hardware needs and pin a revision in production.

## Overview

`inflecttts` is a VoiceHub **text to speech**
integration. This page is generated from its registry contract. [Open the `inflecttts` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/inflecttts.ipynb).

| Property | Value |
| --- | --- |
| Task | Text to speech |
| Architecture | `inflecttts` |
| Runtime | `VoiceHub-native` |
| Languages | `en-US` |
| Capabilities | `text-to-speech`, `fine-tuning`, `safetensors`, `voicehub-native`, `native-runtime`, `preprocessed-training`, `vits-warm-start`, `explicit-phonemes` |
| Reusable components | — |
| Normalized output | `TTSOutput` |

### Language support

<details class="vh-language-support" markdown>
<summary>Supported language abbreviations</summary>

`en-US`

</details>

## Paper and GitHub

- **Paper:** No dedicated upstream research paper is published for this integration.
- **Upstream GitHub:** [Inflect](https://github.com/owenawsong/Inflect)
- **VoiceHub source:** [VoiceHub model implementation](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/inflecttts/modeling_inflecttts.py)

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('inflecttts')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `inflecttts` |
| Configuration class | `InflectTTSConfig` |
| Architecture class | `InflectTTSForTextToSpeech` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'owensong/Inflect-Micro-v2',
    model_type='inflecttts',
)
print(type(processor).__name__)
```

## Inference

The Usage example returns `TTSOutput` through `AutoModelForTextToSpeech`.

### Input and output contract

| Property | Value |
| --- | --- |
| Readiness | `preprocessed` |
| Data architecture | `vits` |
| Sample rate | 24,000 Hz |
| Contract getter | `get_tts_dataset_spec('inflecttts')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `explicit-features` | `input_ids`, `spectrogram`, `audio_values` | — | Prepared | — |

VITS/GAN text, waveform, spectrogram, and adversarial data. See the [data workflow](../../guides/data-preparation.md).

## Training and optimization

Use `available_optimization_passes()` to discover reversible public passes.
Unsupported runtime or hardware fails closed before mutation.

### Training contract

| Property | Value |
| --- | --- |
| Support | `preprocessed` |
| Family | `vits` |
| Recipe | `adversarial` |
| Default phase | `generator` |
| Training checkpoint | `owensong/Inflect-Micro-v2` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `generator` | generator | `training_model.generator` | `input_ids`, `input_lengths`, `spectrogram`, `spectrogram_lengths`, `audio_values` | `loss`, `mel_loss`, `kl_loss`, `duration_loss`, `adversarial_loss`, `feature_matching_loss`, `waveform_loss` |
| `discriminator` | discriminator | `training_model.discriminator` | `input_ids`, `input_lengths`, `spectrogram`, `spectrogram_lengths`, `audio_values` | `loss` |

Prepare the exact tensors listed in the data contract before this step. Call `model.validate_training_support()` first, then follow the
[training workflow](../../guides/training.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | [`owensong/Inflect-Micro-v2`](https://huggingface.co/owensong/Inflect-Micro-v2) |
| Hugging Face ID | [`owensong/Inflect-Micro-v2`](https://huggingface.co/owensong/Inflect-Micro-v2)<br>Repository availability verified through the Hugging Face model API on 2026-08-11; pin a revision before production use. |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.inflecttts.modeling_inflecttts.InflectTTSForTextToSpeech` |
| Configuration | `voicehub.models.inflecttts.configuration_inflecttts.InflectTTSConfig` |
| Source provenance | `voicehub/models/inflecttts/source/SOURCE.json` |
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

### `InflectTTSConfig`

<p class="vh-model-api-card__source-wrap"><a class="vh-model-api-card__source" href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/inflecttts/configuration_inflecttts.py">View source</a></p>
<div class="vh-model-api-card__signature" markdown>

```text
InflectTTSConfig(**config_kwargs)
```

</div>
<h4>Parameters</h4>
<div class="vh-model-api-card__parameters" markdown>
- `**config_kwargs` — Configuration fields validated by InflectTTSConfig.
</div>
</section>

<section class="vh-model-api-card" data-vh-model-api-card="model" markdown>
<p class="vh-model-api-card__badge-wrap"><span class="vh-model-api-card__badge">Model</span></p>

### `InflectTTSForTextToSpeech`

<p class="vh-model-api-card__source-wrap"><a class="vh-model-api-card__source" href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/inflecttts/modeling_inflecttts.py">View source</a></p>
<div class="vh-model-api-card__signature" markdown>

```text
AutoModelForTextToSpeech.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='inflecttts',
    config=None,
    **model_kwargs,
)
```

</div>
<h4>Parameters</h4>
<div class="vh-model-api-card__parameters" markdown>
- `pretrained_model_name_or_path` — Hub ID or compatible local directory.
- `model_type` — Canonical model type; use 'inflecttts'.
- `config` — Optional preloaded InflectTTSConfig instance.
- `**model_kwargs` — Model-specific loading arguments.
</div>
</section>

```python
from voicehub import get_model_spec

spec = get_model_spec('inflecttts')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('inflecttts')` |
| Load and run | `AutoModelForTextToSpeech` |
| Configure | `InflectTTSConfig` |
| Process | `AutoProcessor` |
| Model implementation | `InflectTTSForTextToSpeech` |
| Normalized output | `TTSOutput` |
| Training contract | `get_training_spec('inflecttts')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).

</div>

</div>

</div>
