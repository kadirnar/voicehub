---
description: Public API, checkpoint, training, and optimization guide for the melotts integration.
hide:
  - toc
---

<div class="vh-model-detail" data-vh-model-detail data-model-type="melotts" data-task="text-to-speech" data-training="preprocessed" data-parameter-count="" markdown>

<header class="vh-model-detail__hero" data-vh-model-hero markdown>

<p class="vh-model-detail__namespace" aria-label="Model repository"><span class="vh-model-detail__owner-avatar" aria-hidden="true">VH</span><a href="https://github.com/kadirnar/voicehub">VoiceHub</a><span aria-hidden="true">/</span><strong>melotts</strong></p>

# MeloTTS {.vh-model-title}

<p class="vh-model-detail__summary">Opts into the pinned legacy MeloTTS release explicitly and selects its English speaker table.</p>
<div class="vh-model-detail__tags" aria-label="Model metadata"><span class="vh-model-detail__chip" data-chip-kind="task">Text to speech</span><span class="vh-model-detail__chip" data-chip-kind="runtime">VoiceHub-native</span><span class="vh-model-detail__chip" data-chip-kind="architecture">melotts</span><span class="vh-model-detail__chip" data-chip-kind="parameters" aria-describedby="vh-model-parameters-note-melotts">Parameters: Not reported</span><span class="vh-model-detail__chip" data-chip-kind="language">Languages: en, fr +4</span><span class="vh-model-detail__chip" data-chip-kind="training">Training: preprocessed</span><span class="vh-model-detail__chip" data-chip-kind="license">License: Checkpoint-specific</span></div>
<p class="vh-model-detail__parameter-note" id="vh-model-parameters-note-melotts"><strong>Parameter metadata:</strong> Not reported: the audited metadata available for the registered default does not provide an exact parameter total.</p>
<div class="vh-model-detail__actions" aria-label="Model actions">
<a class="vh-model-detail__action vh-model-detail__action--primary" href="#usage" data-vh-model-action="use">Use this model</a>
<button class="vh-model-detail__action vh-model-detail__copy" type="button" data-vh-copy-model-id data-model-id="EN" aria-describedby="vh-model-checkpoint-melotts"><span data-vh-copy-model-id-label>Copy model ID</span></button>
<details class="vh-model-detail__resources">
<summary class="vh-model-detail__action">Resources</summary>
<div class="vh-model-detail__resource-menu">
<a href="https://github.com/myshell-ai/MeloTTS" data-vh-model-action="github">Upstream GitHub</a>
<a href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/melotts/modeling_melotts.py" data-vh-model-action="source">VoiceHub source</a>
</div>
</details>
</div>
</header>

<nav class="vh-model-detail__tabs" aria-label="Model sections"><a href="#usage" data-vh-model-tab="usage">Usage</a><a href="#overview" data-vh-model-tab="model-card" aria-current="location">Model card</a><a href="#paper-and-github" data-vh-model-tab="sources">Sources</a><a href="#training-and-optimization" data-vh-model-tab="training">Training</a><a href="#checkpoints-provenance-license-and-limitations" data-vh-model-tab="checkpoint">Checkpoint</a><a href="#public-api" data-vh-model-tab="api">Public API</a></nav>

<div class="vh-model-detail__layout" markdown>

<aside class="vh-model-detail__sidebar" data-vh-model-facts aria-labelledby="vh-model-facts-title-melotts"><h2 id="vh-model-facts-title-melotts">Model facts</h2><details class="vh-model-detail__facts-disclosure" data-vh-model-facts-disclosure aria-labelledby="vh-model-facts-title-melotts" open><summary><span>Toggle model facts</span></summary><dl class="vh-model-detail__facts"><div><dt>Task</dt><dd>Text to speech</dd></div><div><dt>Parameters</dt><dd aria-describedby="vh-model-parameters-note-melotts">Not reported</dd></div><div><dt>Architecture</dt><dd><code>melotts</code></dd></div><div><dt>Runtime</dt><dd>VoiceHub-native</dd></div><div><dt>Languages</dt><dd><details class="vh-model-detail__languages"><summary>6 documented codes</summary><span><code>en</code> <code>fr</code> <code>ja</code> <code>es</code> <code>zh</code> <code>ko</code></span></details></dd></div><div><dt>Capabilities</dt><dd><details class="vh-model-detail__capabilities"><summary>8 capabilities</summary><span><code>text-to-speech</code> <code>multilingual</code> <code>fine-tuning</code> <code>safetensors</code> <code>voicehub-native</code> <code>native-runtime</code> <code>preprocessed-training</code> <code>explicit-linguistic-features</code></span></details></dd></div><div><dt>Training</dt><dd><code>preprocessed</code></dd></div><div><dt>License</dt><dd>Checkpoint-specific</dd></div><div><dt>Default checkpoint</dt><dd id="vh-model-checkpoint-melotts"><code>EN</code></dd></div></dl></details></aside>

<div class="vh-model-detail__main vh-model-detail__content" markdown>

## Usage

Complete the [VoiceHub installation](../../getting-started/installation.md) once,
then run this repository-authored example. Model pages intentionally contain no
package-install command.

This example is maintained against VoiceHub's public API; it is not copied from an upstream demo or package README.

**Model-specific path:** Opts into the pinned legacy MeloTTS release explicitly and selects its English speaker table.

**Inputs and controls:** The official release is a reviewed pickle checkpoint; keep `trust_pickle_checkpoint` false for arbitrary files.

```python
from pathlib import Path

from voicehub import AutoModelForTextToSpeech, TTSGenerationConfig, AutoConfig

model = AutoModelForTextToSpeech.from_pretrained(
    'EN',
    model_type='melotts',
    device="cuda",
    lazy_load=True,
    config=AutoConfig.for_model("melotts", trust_pickle_checkpoint=True),
)
output = model.generate(
    'VoiceHub keeps model integrations explicit and reproducible.',
    generation_config=TTSGenerationConfig(
        seed=42,
        output_file=Path("output.wav"),
    ),
    speaker="EN-US",
    speed=1.0,
)
print(output.file_path, output.sample_rate, output.metadata)
```

Use authorized recordings. Verify hardware needs and pin a revision in production.

## Overview

`melotts` is a VoiceHub **text to speech**
integration. This page is generated from its registry contract.

| Property | Value |
| --- | --- |
| Task | Text to speech |
| Architecture | `melotts` |
| Runtime | `VoiceHub-native` |
| Languages | `en`, `fr`, `ja`, `es`, `zh`, `ko` |
| Capabilities | `text-to-speech`, `multilingual`, `fine-tuning`, `safetensors`, `voicehub-native`, `native-runtime`, `preprocessed-training`, `explicit-linguistic-features` |
| Reusable components | — |
| Normalized output | `TTSOutput` |

### Language support

<details class="vh-language-support" markdown>
<summary>Supported language abbreviations</summary>

`en`, `fr`, `ja`, `es`, `zh`, `ko`

</details>

## Paper and GitHub

- **Paper:** No dedicated upstream research paper is published for this integration.
- **Upstream GitHub:** [MeloTTS](https://github.com/myshell-ai/MeloTTS)
- **VoiceHub source:** [VoiceHub model implementation](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/melotts/modeling_melotts.py)

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('melotts')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `melotts` |
| Configuration class | `MeloTTSConfig` |
| Architecture class | `MeloTTSForTextToSpeech` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'EN',
    model_type='melotts',
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
| Sample rate | 44,100 Hz |
| Contract getter | `get_tts_dataset_spec('melotts')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `explicit-features` | `input_ids`, `tone_ids`, `language_ids`, `bert_features`, `ja_bert_features`, `spectrogram`, `audio_values`, `speaker_id` | — | Prepared | — |

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
| Training checkpoint | `EN` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `generator` | generator | `training_model.model` | `input_ids`, `input_lengths`, `tone_ids`, `language_ids`, `bert_features`, `ja_bert_features`, `spectrogram`, `spectrogram_lengths`, `audio_values`, `audio_lengths`, `speaker_ids` | `loss` |
| `discriminator` | discriminator | `training_model.mpd` | `input_ids`, `input_lengths`, `tone_ids`, `language_ids`, `bert_features`, `ja_bert_features`, `spectrogram`, `spectrogram_lengths`, `audio_values`, `audio_lengths`, `speaker_ids` | `loss` |
| `duration_discriminator` | duration-discriminator | `training_model.duration_discriminator` | `input_ids`, `input_lengths`, `tone_ids`, `language_ids`, `bert_features`, `ja_bert_features`, `spectrogram`, `spectrogram_lengths`, `audio_values`, `audio_lengths`, `speaker_ids` | `loss` |

Prepare the exact tensors listed in the data contract before this step. Call `model.validate_training_support()` first, then follow the
[training workflow](../../guides/training.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | `EN` |
| Hugging Face ID | [`myshell-ai/MeloTTS-English`](https://huggingface.co/myshell-ai/MeloTTS-English)<br>Official English MeloTTS repository, verified available on 2026-08-11 and used by the registered EN release alias. |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.melotts.modeling_melotts.MeloTTSForTextToSpeech` |
| Configuration | `voicehub.models.melotts.configuration_melotts.MeloTTSConfig` |
| Source provenance | `voicehub/models/melotts/source/SOURCE.json` |
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

### `MeloTTSConfig`

<p class="vh-model-api-card__source-wrap"><a class="vh-model-api-card__source" href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/melotts/configuration_melotts.py">View source</a></p>
<div class="vh-model-api-card__signature" markdown>

```text
MeloTTSConfig(**config_kwargs)
```

</div>
<h4>Parameters</h4>
<div class="vh-model-api-card__parameters" markdown>
- `**config_kwargs` — Configuration fields validated by MeloTTSConfig.
</div>
</section>

<section class="vh-model-api-card" data-vh-model-api-card="model" markdown>
<p class="vh-model-api-card__badge-wrap"><span class="vh-model-api-card__badge">Model</span></p>

### `MeloTTSForTextToSpeech`

<p class="vh-model-api-card__source-wrap"><a class="vh-model-api-card__source" href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/melotts/modeling_melotts.py">View source</a></p>
<div class="vh-model-api-card__signature" markdown>

```text
AutoModelForTextToSpeech.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='melotts',
    config=None,
    **model_kwargs,
)
```

</div>
<h4>Parameters</h4>
<div class="vh-model-api-card__parameters" markdown>
- `pretrained_model_name_or_path` — Hub ID or compatible local directory.
- `model_type` — Canonical model type; use 'melotts'.
- `config` — Optional preloaded MeloTTSConfig instance.
- `**model_kwargs` — Model-specific loading arguments.
</div>
</section>

```python
from voicehub import get_model_spec

spec = get_model_spec('melotts')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('melotts')` |
| Load and run | `AutoModelForTextToSpeech` |
| Configure | `MeloTTSConfig` |
| Process | `AutoProcessor` |
| Model implementation | `MeloTTSForTextToSpeech` |
| Normalized output | `TTSOutput` |
| Training contract | `get_training_spec('melotts')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).

</div>

</div>

</div>
