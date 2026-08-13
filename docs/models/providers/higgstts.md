---
description: Public API, checkpoint, training, and optimization guide for the higgstts integration.
hide:
  - toc
---

<div class="vh-model-detail" data-vh-model-detail data-model-type="higgstts" data-task="text-to-speech" data-training="native" data-parameter-count="5771283456" markdown>

<header class="vh-model-detail__hero" data-vh-model-hero markdown>

<p class="vh-model-detail__namespace" aria-label="Model repository"><span class="vh-model-detail__owner-avatar" aria-hidden="true">BO</span><a href="https://huggingface.co/bosonai">bosonai</a><span aria-hidden="true">/</span><strong>higgs-tts-2-3b-base</strong></p>

# HiggsTTS {.vh-model-title}

<p class="vh-model-detail__summary">Provides Higgs Audio with paired reference context and a bounded semantic-token budget.</p>
<div class="vh-model-detail__tags" aria-label="Model metadata"><span class="vh-model-detail__chip" data-chip-kind="task">Text to speech</span><span class="vh-model-detail__chip" data-chip-kind="runtime">VoiceHub-native</span><span class="vh-model-detail__chip" data-chip-kind="architecture">higgs_audio_v2</span><span class="vh-model-detail__chip" data-chip-kind="parameters" aria-describedby="vh-model-parameters-note-higgstts">Parameters: 5.8B</span><span class="vh-model-detail__chip" data-chip-kind="language">Languages: en, zh +2</span><span class="vh-model-detail__chip" data-chip-kind="training">Training: native</span><span class="vh-model-detail__chip" data-chip-kind="license">License: Checkpoint-specific</span></div>
<p class="vh-model-detail__parameter-note" id="vh-model-parameters-note-higgstts"><strong>Parameter metadata:</strong> Exact learned-parameter total for VoiceHub&#x27;s audited native primary graph at the registered default selection; separately loaded auxiliary models are excluded.</p>
<div class="vh-model-detail__actions" aria-label="Model actions">
<a class="vh-model-detail__action vh-model-detail__action--primary" href="#usage" data-vh-model-action="use">Use this model</a>
<button class="vh-model-detail__action vh-model-detail__copy" type="button" data-vh-copy-model-id data-model-id="bosonai/higgs-tts-2-3b-base" aria-describedby="vh-model-checkpoint-higgstts"><span data-vh-copy-model-id-label>Copy model ID</span></button>
<a class="vh-model-detail__action" href="https://huggingface.co/bosonai/higgs-tts-2-3b-base" data-vh-model-action="checkpoint">Checkpoint</a>
<details class="vh-model-detail__resources">
<summary class="vh-model-detail__action">Resources</summary>
<div class="vh-model-detail__resource-menu">
<a href="https://github.com/boson-ai/higgs-audio" data-vh-model-action="github">Upstream GitHub</a>
<a href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/higgstts/modeling_higgstts.py" data-vh-model-action="source">VoiceHub source</a>
<a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/higgstts.ipynb" data-vh-model-action="colab">Open in Colab</a>
</div>
</details>
</div>
</header>

<nav class="vh-model-detail__tabs" aria-label="Model sections"><a href="#usage" data-vh-model-tab="usage">Usage</a><a href="#overview" data-vh-model-tab="model-card" aria-current="location">Model card</a><a href="#paper-and-github" data-vh-model-tab="sources">Sources</a><a href="#training-and-optimization" data-vh-model-tab="training">Training</a><a href="#checkpoints-provenance-license-and-limitations" data-vh-model-tab="checkpoint">Checkpoint</a><a href="#public-api" data-vh-model-tab="api">Public API</a></nav>

<div class="vh-model-detail__layout" markdown>

<aside class="vh-model-detail__sidebar" data-vh-model-facts aria-labelledby="vh-model-facts-title-higgstts"><h2 id="vh-model-facts-title-higgstts">Model facts</h2><details class="vh-model-detail__facts-disclosure" data-vh-model-facts-disclosure aria-labelledby="vh-model-facts-title-higgstts" open><summary><span>Toggle model facts</span></summary><dl class="vh-model-detail__facts"><div><dt>Task</dt><dd>Text to speech</dd></div><div><dt>Parameters</dt><dd aria-describedby="vh-model-parameters-note-higgstts">5.8B</dd></div><div><dt>Architecture</dt><dd><code>higgs_audio_v2</code></dd></div><div><dt>Runtime</dt><dd>VoiceHub-native</dd></div><div><dt>Languages</dt><dd><code>en</code> <code>zh</code> <code>de</code> <code>ko</code></dd></div><div><dt>Capabilities</dt><dd><details class="vh-model-detail__capabilities"><summary>10 capabilities</summary><span><code>text-to-speech</code> <code>voice-cloning</code> <code>expressive-speech</code> <code>multilingual</code> <code>fine-tuning</code> <code>safetensors</code> <code>voicehub-native</code> <code>native-runtime</code> <code>raw-audio-fine-tuning</code> <code>preencoded-code-fine-tuning</code></span></details></dd></div><div><dt>Training</dt><dd><code>native</code></dd></div><div><dt>License</dt><dd>Checkpoint-specific</dd></div><div><dt>Default checkpoint</dt><dd id="vh-model-checkpoint-higgstts"><a href="https://huggingface.co/bosonai/higgs-tts-2-3b-base"><code>bosonai/higgs-tts-2-3b-base</code></a></dd></div></dl></details></aside>

<div class="vh-model-detail__main vh-model-detail__content" markdown>

## Usage

Complete the [VoiceHub installation](../../getting-started/installation.md) once,
then run this repository-authored example. Model pages intentionally contain no
package-install command.

This example is maintained against VoiceHub's public API; it is not copied from an upstream demo or package README.

**Model-specific path:** Provides Higgs Audio with paired reference context and a bounded semantic-token budget.

**Inputs and controls:** The reference transcript must correspond to the authorized speaker audio.

```python
from pathlib import Path

from voicehub import AutoModelForTextToSpeech, TTSGenerationConfig

REFERENCE_AUDIO = Path("reference.wav")
REFERENCE_TEXT = "The reference transcript must exactly match the authorized audio."
if not REFERENCE_AUDIO.is_file():
    raise FileNotFoundError(REFERENCE_AUDIO)

model = AutoModelForTextToSpeech.from_pretrained(
    'bosonai/higgs-tts-2-3b-base',
    model_type='higgstts',
    device="cuda",
    lazy_load=True,
)
output = model.generate(
    'VoiceHub keeps model integrations explicit and reproducible.',
    generation_config=TTSGenerationConfig(
        seed=42,
        output_file=Path("output.wav"),
    ),
    speaker_audio_path=str(REFERENCE_AUDIO),
    reference_text=REFERENCE_TEXT,
    system_prompt="Generate natural, clean speech.",
    max_new_tokens=1_024,
)
print(output.file_path, output.sample_rate, output.metadata)
```

Use authorized recordings. Verify hardware needs and pin a revision in production.

## Overview

`higgstts` is a VoiceHub **text to speech**
integration. This page is generated from its registry contract. [Open the `higgstts` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/higgstts.ipynb).

| Property | Value |
| --- | --- |
| Task | Text to speech |
| Architecture | `higgs_audio_v2` |
| Runtime | `VoiceHub-native` |
| Languages | `en`, `zh`, `de`, `ko` |
| Capabilities | `text-to-speech`, `voice-cloning`, `expressive-speech`, `multilingual`, `fine-tuning`, `safetensors`, `voicehub-native`, `native-runtime`, `raw-audio-fine-tuning`, `preencoded-code-fine-tuning` |
| Reusable components | — |
| Normalized output | `TTSOutput` |

### Language support

<details class="vh-language-support" markdown>
<summary>Supported language abbreviations</summary>

`en`, `zh`, `de`, `ko`

</details>

## Paper and GitHub

- **Paper:** No dedicated upstream research paper is published for this integration.
- **Upstream GitHub:** [Higgs Audio](https://github.com/boson-ai/higgs-audio)
- **VoiceHub source:** [VoiceHub model implementation](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/higgstts/modeling_higgstts.py)

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('higgstts')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `higgstts` |
| Configuration class | `HiggsTTSConfig` |
| Architecture class | `HiggsTTSForTextToSpeech` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'bosonai/higgs-tts-2-3b-base',
    model_type='higgstts',
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
| Contract getter | `get_tts_dataset_spec('higgstts')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `raw-audio` | `text` | audio / target_audio | Source | at most one: audio / target_audio; reference_audio / reference_codes; forbidden: audio_codes; reference_audio requires reference_text; reference_codes requires reference_text; reference_text requires one of reference_audio, reference_codes |
| `audio-codes` | `text`, `audio_codes` | — | Prepared | at most one: reference_audio / reference_codes; forbidden: audio, target_audio; reference_audio requires reference_text; reference_codes requires reference_text; reference_text requires one of reference_audio, reference_codes |

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
| Training checkpoint | `bosonai/higgs-tts-2-3b-base` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `codec_language_model` | objective | `model` | `input_ids`, `attention_mask`, `audio_input_ids`, `audio_input_ids_mask`, `labels`, `audio_labels` | `loss`, `text_loss`, `audio_loss` |

The integration accepts its declared source or prepared contract directly. Call `model.validate_training_support()` first, then follow the
[training workflow](../../guides/training.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | [`bosonai/higgs-tts-2-3b-base`](https://huggingface.co/bosonai/higgs-tts-2-3b-base) |
| Hugging Face ID | [`bosonai/higgs-tts-2-3b-base`](https://huggingface.co/bosonai/higgs-tts-2-3b-base)<br>Repository availability verified through the Hugging Face model API on 2026-08-11; pin a revision before production use. |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.higgstts.modeling_higgstts.HiggsTTSForTextToSpeech` |
| Configuration | `voicehub.models.higgstts.configuration_higgstts.HiggsTTSConfig` |
| Source provenance | `voicehub/models/higgstts/source/SOURCE.json` |
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

### `HiggsTTSConfig`

<p class="vh-model-api-card__source-wrap"><a class="vh-model-api-card__source" href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/higgstts/configuration_higgstts.py">View source</a></p>
<div class="vh-model-api-card__signature" markdown>

```text
HiggsTTSConfig(**config_kwargs)
```

</div>
<h4>Parameters</h4>
<div class="vh-model-api-card__parameters" markdown>
- `**config_kwargs` — Configuration fields validated by HiggsTTSConfig.
</div>
</section>

<section class="vh-model-api-card" data-vh-model-api-card="model" markdown>
<p class="vh-model-api-card__badge-wrap"><span class="vh-model-api-card__badge">Model</span></p>

### `HiggsTTSForTextToSpeech`

<p class="vh-model-api-card__source-wrap"><a class="vh-model-api-card__source" href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/higgstts/modeling_higgstts.py">View source</a></p>
<div class="vh-model-api-card__signature" markdown>

```text
AutoModelForTextToSpeech.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='higgstts',
    config=None,
    **model_kwargs,
)
```

</div>
<h4>Parameters</h4>
<div class="vh-model-api-card__parameters" markdown>
- `pretrained_model_name_or_path` — Hub ID or compatible local directory.
- `model_type` — Canonical model type; use 'higgstts'.
- `config` — Optional preloaded HiggsTTSConfig instance.
- `**model_kwargs` — Model-specific loading arguments.
</div>
</section>

```python
from voicehub import get_model_spec

spec = get_model_spec('higgstts')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('higgstts')` |
| Load and run | `AutoModelForTextToSpeech` |
| Configure | `HiggsTTSConfig` |
| Process | `AutoProcessor` |
| Model implementation | `HiggsTTSForTextToSpeech` |
| Normalized output | `TTSOutput` |
| Training contract | `get_training_spec('higgstts')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).

</div>

</div>

</div>
