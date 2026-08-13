---
description: Public API, checkpoint, training, and optimization guide for the conversationtts integration.
hide:
  - toc
---

<div class="vh-model-detail" data-vh-model-detail data-model-type="conversationtts" data-task="text-to-speech" data-training="native" data-parameter-count="" markdown>

<header class="vh-model-detail__hero" data-vh-model-hero markdown>

<p class="vh-model-detail__namespace" aria-label="Model repository"><span class="vh-model-detail__owner-avatar" aria-hidden="true">AF</span><a href="https://huggingface.co/AudioFoundation">AudioFoundation</a><span aria-hidden="true">/</span><strong>SpeechFoundation</strong></p>

# ConversationTTS {.vh-model-title}

<p class="vh-model-detail__summary">Assigns an explicit conversation speaker and caps the generated audio duration.</p>
<div class="vh-model-detail__tags" aria-label="Model metadata"><span class="vh-model-detail__chip" data-chip-kind="task">Text to speech</span><span class="vh-model-detail__chip" data-chip-kind="runtime">VoiceHub-native</span><span class="vh-model-detail__chip" data-chip-kind="architecture">conversationtts</span><span class="vh-model-detail__chip" data-chip-kind="parameters" aria-describedby="vh-model-parameters-note-conversationtts">Parameters: Not reported</span><span class="vh-model-detail__chip" data-chip-kind="language">Languages: en, zh +1</span><span class="vh-model-detail__chip" data-chip-kind="training">Training: native</span><span class="vh-model-detail__chip" data-chip-kind="license">License: CC-BY-NC-4.0</span></div>
<p class="vh-model-detail__parameter-note" id="vh-model-parameters-note-conversationtts"><strong>Parameter metadata:</strong> Not reported: the audited metadata available for the registered default does not provide an exact parameter total.</p>
<div class="vh-model-detail__actions" aria-label="Model actions">
<a class="vh-model-detail__action vh-model-detail__action--primary" href="#usage" data-vh-model-action="use">Use this model</a>
<button class="vh-model-detail__action vh-model-detail__copy" type="button" data-vh-copy-model-id data-model-id="AudioFoundation/SpeechFoundation" aria-describedby="vh-model-checkpoint-conversationtts"><span data-vh-copy-model-id-label>Copy model ID</span></button>
<a class="vh-model-detail__action" href="https://huggingface.co/AudioFoundation/SpeechFoundation" data-vh-model-action="checkpoint">Checkpoint</a>
<details class="vh-model-detail__resources">
<summary class="vh-model-detail__action">Resources</summary>
<div class="vh-model-detail__resource-menu">
<a href="https://github.com/Audio-Foundation-Models/ConversationTTS" data-vh-model-action="github">Upstream GitHub</a>
<a href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/conversationtts/modeling_conversationtts.py" data-vh-model-action="source">VoiceHub source</a>
<a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/conversationtts.ipynb" data-vh-model-action="colab">Open in Colab</a>
</div>
</details>
</div>
</header>

<nav class="vh-model-detail__tabs" aria-label="Model sections"><a href="#usage" data-vh-model-tab="usage">Usage</a><a href="#overview" data-vh-model-tab="model-card" aria-current="location">Model card</a><a href="#paper-and-github" data-vh-model-tab="sources">Sources</a><a href="#training-and-optimization" data-vh-model-tab="training">Training</a><a href="#checkpoints-provenance-license-and-limitations" data-vh-model-tab="checkpoint">Checkpoint</a><a href="#public-api" data-vh-model-tab="api">Public API</a></nav>

<div class="vh-model-detail__layout" markdown>

<aside class="vh-model-detail__sidebar" data-vh-model-facts aria-labelledby="vh-model-facts-title-conversationtts"><h2 id="vh-model-facts-title-conversationtts">Model facts</h2><details class="vh-model-detail__facts-disclosure" data-vh-model-facts-disclosure aria-labelledby="vh-model-facts-title-conversationtts" open><summary><span>Toggle model facts</span></summary><dl class="vh-model-detail__facts"><div><dt>Task</dt><dd>Text to speech</dd></div><div><dt>Parameters</dt><dd aria-describedby="vh-model-parameters-note-conversationtts">Not reported</dd></div><div><dt>Architecture</dt><dd><code>conversationtts</code></dd></div><div><dt>Runtime</dt><dd>VoiceHub-native</dd></div><div><dt>Languages</dt><dd><code>en</code> <code>zh</code> <code>yue</code></dd></div><div><dt>Capabilities</dt><dd><details class="vh-model-detail__capabilities"><summary>11 capabilities</summary><span><code>text-to-speech</code> <code>voice-cloning</code> <code>conversation</code> <code>multilingual</code> <code>fine-tuning</code> <code>safetensors</code> <code>voicehub-native</code> <code>native-runtime</code> <code>raw-audio-fine-tuning</code> <code>preencoded-code-fine-tuning</code> <code>noncommercial</code></span></details></dd></div><div><dt>Training</dt><dd><code>native</code></dd></div><div><dt>License</dt><dd><a href="https://github.com/Audio-Foundation-Models/ConversationTTS">CC-BY-NC-4.0</a></dd></div><div><dt>Default checkpoint</dt><dd id="vh-model-checkpoint-conversationtts"><a href="https://huggingface.co/AudioFoundation/SpeechFoundation"><code>AudioFoundation/SpeechFoundation</code></a></dd></div></dl></details></aside>

<div class="vh-model-detail__main vh-model-detail__content" markdown>

## Usage

Complete the [VoiceHub installation](../../getting-started/installation.md) once,
then run this repository-authored example. Model pages intentionally contain no
package-install command.

This example is maintained against VoiceHub's public API; it is not copied from an upstream demo or package README.

**Model-specific path:** Assigns an explicit conversation speaker and caps the generated audio duration.

**Inputs and controls:** Use stable integer speaker IDs when building multi-turn context.

```python
from pathlib import Path

from voicehub import AutoModelForTextToSpeech, TTSGenerationConfig

model = AutoModelForTextToSpeech.from_pretrained(
    'AudioFoundation/SpeechFoundation',
    model_type='conversationtts',
    device="cuda",
    lazy_load=True,
)
output = model.generate(
    'VoiceHub keeps model integrations explicit and reproducible.',
    generation_config=TTSGenerationConfig(
        seed=42,
        output_file=Path("output.wav"),
    ),
    speaker=0,
    max_audio_length_ms=30_000,
    temperature=0.9,
    top_k=50,
)
print(output.file_path, output.sample_rate, output.metadata)
```

Use authorized recordings. Verify hardware needs and pin a revision in production.

## Overview

`conversationtts` is a VoiceHub **text to speech**
integration. This page is generated from its registry contract. [Open the `conversationtts` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/conversationtts.ipynb).

| Property | Value |
| --- | --- |
| Task | Text to speech |
| Architecture | `conversationtts` |
| Runtime | `VoiceHub-native` |
| Languages | `en`, `zh`, `yue` |
| Capabilities | `text-to-speech`, `voice-cloning`, `conversation`, `multilingual`, `fine-tuning`, `safetensors`, `voicehub-native`, `native-runtime`, `raw-audio-fine-tuning`, `preencoded-code-fine-tuning`, `noncommercial` |
| Reusable components | — |
| Normalized output | `TTSOutput` |

### Language support

<details class="vh-language-support" markdown>
<summary>Supported language abbreviations</summary>

`en`, `zh`, `yue`

These are the languages explicitly named in the upstream release README's podcast data.

</details>

## Paper and GitHub

- **Paper:** No dedicated upstream research paper is published for this integration.
- **Upstream GitHub:** [ConversationTTS](https://github.com/Audio-Foundation-Models/ConversationTTS)
- **VoiceHub source:** [VoiceHub model implementation](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/conversationtts/modeling_conversationtts.py)

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('conversationtts')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `conversationtts` |
| Configuration class | `ConversationTTSConfig` |
| Architecture class | `ConversationTTSForTextToSpeech` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'AudioFoundation/SpeechFoundation',
    model_type='conversationtts',
)
print(type(processor).__name__)
```

## Inference

The Usage example returns `TTSOutput` through `AutoModelForTextToSpeech`.

### Input and output contract

| Property | Value |
| --- | --- |
| Readiness | `integrated-raw` |
| Data architecture | `codec-lm` |
| Sample rate | 24,000 Hz |
| Contract getter | `get_tts_dataset_spec('conversationtts')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `raw-text-audio` | — | text / texts; audio / audio_values | Source | at most one: text / texts; audio / audio_values; forbidden: text_token_ids, text_ids, audio_codes, codes |
| `raw-text-code` | — | text / texts; audio_codes / codes | Prepared | at most one: text / texts; audio_codes / codes; forbidden: text_token_ids, text_ids, audio, audio_values |
| `tokenized-text-audio` | — | text_token_ids / text_ids; audio / audio_values | Prepared | at most one: text_token_ids / text_ids; audio / audio_values; forbidden: text, texts, audio_codes, codes |
| `tokenized-text-code` | — | text_token_ids / text_ids; audio_codes / codes | Prepared | at most one: text_token_ids / text_ids; audio_codes / codes; forbidden: text, texts, audio, audio_values |
| `multi-codebook-batch` | `tokens`, `labels`, `tokens_mask` | — | Prepared | — |

Autoregressive text/audio-token or codec-language-model data. See the [data workflow](../../guides/data-preparation.md).

## Training and optimization

Use `available_optimization_passes()` to discover reversible public passes.
Unsupported runtime or hardware fails closed before mutation.

### Training contract

| Property | Value |
| --- | --- |
| Support | `native` |
| Family | `causal-lm` |
| Recipe | `single-phase` |
| Default phase | `codec_language_model` |
| Training checkpoint | `AudioFoundation/SpeechFoundation` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `codec_language_model` | objective | `model` | `tokens`, `labels`, `tokens_mask` | `loss`, `codebook0_loss`, `residual_loss` |

The integration accepts its declared source or prepared contract directly. Call `model.validate_training_support()` first, then follow the
[training workflow](../../guides/training.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | [`AudioFoundation/SpeechFoundation`](https://huggingface.co/AudioFoundation/SpeechFoundation) |
| Hugging Face ID | [`AudioFoundation/SpeechFoundation`](https://huggingface.co/AudioFoundation/SpeechFoundation)<br>Repository availability verified through the Hugging Face model API on 2026-08-11; pin a revision before production use. |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.conversationtts.modeling_conversationtts.ConversationTTSForTextToSpeech` |
| Configuration | `voicehub.models.conversationtts.configuration_conversationtts.ConversationTTSConfig` |
| Source provenance | `voicehub/models/conversationtts/source/SOURCE.json` |
| License | [CC-BY-NC-4.0](https://github.com/Audio-Foundation-Models/ConversationTTS) |

Source, checkpoints, datasets, and evaluation tools are non-commercial. Commercial use: **not allowed**.

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

### `ConversationTTSConfig`

<p class="vh-model-api-card__source-wrap"><a class="vh-model-api-card__source" href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/conversationtts/configuration_conversationtts.py">View source</a></p>
<div class="vh-model-api-card__signature" markdown>

```text
ConversationTTSConfig(**config_kwargs)
```

</div>
<h4>Parameters</h4>
<div class="vh-model-api-card__parameters" markdown>
- `**config_kwargs` — Configuration fields validated by ConversationTTSConfig.
</div>
</section>

<section class="vh-model-api-card" data-vh-model-api-card="model" markdown>
<p class="vh-model-api-card__badge-wrap"><span class="vh-model-api-card__badge">Model</span></p>

### `ConversationTTSForTextToSpeech`

<p class="vh-model-api-card__source-wrap"><a class="vh-model-api-card__source" href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/conversationtts/modeling_conversationtts.py">View source</a></p>
<div class="vh-model-api-card__signature" markdown>

```text
AutoModelForTextToSpeech.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='conversationtts',
    config=None,
    **model_kwargs,
)
```

</div>
<h4>Parameters</h4>
<div class="vh-model-api-card__parameters" markdown>
- `pretrained_model_name_or_path` — Hub ID or compatible local directory.
- `model_type` — Canonical model type; use 'conversationtts'.
- `config` — Optional preloaded ConversationTTSConfig instance.
- `**model_kwargs` — Model-specific loading arguments.
</div>
</section>

```python
from voicehub import get_model_spec

spec = get_model_spec('conversationtts')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('conversationtts')` |
| Load and run | `AutoModelForTextToSpeech` |
| Configure | `ConversationTTSConfig` |
| Process | `AutoProcessor` |
| Model implementation | `ConversationTTSForTextToSpeech` |
| Normalized output | `TTSOutput` |
| Training contract | `get_training_spec('conversationtts')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).

</div>

</div>

</div>
