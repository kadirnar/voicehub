---
description: Public API, checkpoint, training, and optimization guide for the chatterbox integration.
hide:
  - toc
---

<div class="vh-model-detail" data-vh-model-detail data-model-type="chatterbox" data-task="text-to-speech" data-training="custom" data-parameter-count="797762633" markdown>

<header class="vh-model-detail__hero" data-vh-model-hero markdown>

<p class="vh-model-detail__namespace" aria-label="Model repository"><span class="vh-model-detail__owner-avatar" aria-hidden="true">RA</span><a href="https://huggingface.co/ResembleAI">ResembleAI</a><span aria-hidden="true">/</span><strong>chatterbox</strong></p>

# Chatterbox {.vh-model-title}

<p class="vh-model-detail__summary">Demonstrates Chatterbox voice prompting through VoiceHub&#x27;s normalized reference-audio field.</p>
<div class="vh-model-detail__tags" aria-label="Model metadata"><span class="vh-model-detail__chip" data-chip-kind="task">Text to speech</span><span class="vh-model-detail__chip" data-chip-kind="runtime">VoiceHub-native</span><span class="vh-model-detail__chip" data-chip-kind="architecture">chatterbox</span><span class="vh-model-detail__chip" data-chip-kind="parameters" aria-describedby="vh-model-parameters-note-chatterbox">Parameters: 797.8M</span><span class="vh-model-detail__chip" data-chip-kind="language">Languages: ar, da +21</span><span class="vh-model-detail__chip" data-chip-kind="training">Training: custom</span><span class="vh-model-detail__chip" data-chip-kind="license">License: Checkpoint-specific</span></div>
<p class="vh-model-detail__parameter-note" id="vh-model-parameters-note-chatterbox"><strong>Parameter metadata:</strong> Exact learned-parameter total for VoiceHub&#x27;s audited native primary graph at the registered default selection; separately loaded auxiliary models are excluded.</p>
<div class="vh-model-detail__actions" aria-label="Model actions">
<a class="vh-model-detail__action vh-model-detail__action--primary" href="#usage" data-vh-model-action="use">Use this model</a>
<button class="vh-model-detail__action vh-model-detail__copy" type="button" data-vh-copy-model-id data-model-id="ResembleAI/chatterbox" aria-describedby="vh-model-checkpoint-chatterbox"><span data-vh-copy-model-id-label>Copy model ID</span></button>
<a class="vh-model-detail__action" href="https://huggingface.co/ResembleAI/chatterbox" data-vh-model-action="checkpoint">Checkpoint</a>
<details class="vh-model-detail__resources">
<summary class="vh-model-detail__action">Resources</summary>
<div class="vh-model-detail__resource-menu">
<a href="https://github.com/resemble-ai/chatterbox" data-vh-model-action="github">Upstream GitHub</a>
<a href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/chatterbox/modeling_chatterbox.py" data-vh-model-action="source">VoiceHub source</a>
<a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/chatterbox.ipynb" data-vh-model-action="colab">Open in Colab</a>
</div>
</details>
</div>
</header>

<nav class="vh-model-detail__tabs" aria-label="Model sections"><a href="#usage" data-vh-model-tab="usage">Usage</a><a href="#overview" data-vh-model-tab="model-card" aria-current="location">Model card</a><a href="#paper-and-github" data-vh-model-tab="sources">Sources</a><a href="#training-and-optimization" data-vh-model-tab="training">Training</a><a href="#checkpoints-provenance-license-and-limitations" data-vh-model-tab="checkpoint">Checkpoint</a><a href="#public-api" data-vh-model-tab="api">Public API</a></nav>

<div class="vh-model-detail__layout" markdown>

<aside class="vh-model-detail__sidebar" data-vh-model-facts aria-labelledby="vh-model-facts-title-chatterbox"><h2 id="vh-model-facts-title-chatterbox">Model facts</h2><details class="vh-model-detail__facts-disclosure" data-vh-model-facts-disclosure aria-labelledby="vh-model-facts-title-chatterbox" open><summary><span>Toggle model facts</span></summary><dl class="vh-model-detail__facts"><div><dt>Task</dt><dd>Text to speech</dd></div><div><dt>Parameters</dt><dd aria-describedby="vh-model-parameters-note-chatterbox">797.8M</dd></div><div><dt>Architecture</dt><dd><code>chatterbox</code></dd></div><div><dt>Runtime</dt><dd>VoiceHub-native</dd></div><div><dt>Languages</dt><dd><details class="vh-model-detail__languages"><summary>23 documented codes</summary><span><code>ar</code> <code>da</code> <code>de</code> <code>el</code> <code>en</code> <code>es</code> <code>fi</code> <code>fr</code> <code>he</code> <code>hi</code> <code>it</code> <code>ja</code> <code>ko</code> <code>ms</code> <code>nl</code> <code>no</code> <code>pl</code> <code>pt</code> <code>ru</code> <code>sv</code> <code>sw</code> <code>tr</code> <code>zh</code></span></details></dd></div><div><dt>Capabilities</dt><dd><details class="vh-model-detail__capabilities"><summary>7 capabilities</summary><span><code>text-to-speech</code> <code>voice-cloning</code> <code>fine-tuning</code> <code>safetensors</code> <code>voicehub-native</code> <code>native-runtime</code> <code>raw-audio-fine-tuning</code></span></details></dd></div><div><dt>Training</dt><dd><code>custom</code></dd></div><div><dt>License</dt><dd>Checkpoint-specific</dd></div><div><dt>Default checkpoint</dt><dd id="vh-model-checkpoint-chatterbox"><a href="https://huggingface.co/ResembleAI/chatterbox"><code>ResembleAI/chatterbox</code></a></dd></div></dl></details></aside>

<div class="vh-model-detail__main vh-model-detail__content" markdown>

## Usage

Complete the [VoiceHub installation](../../getting-started/installation.md) once,
then run this repository-authored example. Model pages intentionally contain no
package-install command.

This example is maintained against VoiceHub's public API; it is not copied from an upstream demo or package README.

**Model-specific path:** Demonstrates Chatterbox voice prompting through VoiceHub's normalized reference-audio field.

**Inputs and controls:** Use only a reference recording you are authorized to process; omit the argument for the checkpoint's default voice.

```python
from pathlib import Path

from voicehub import AutoModelForTextToSpeech, TTSGenerationConfig

REFERENCE_AUDIO = Path("reference.wav")
REFERENCE_TEXT = "The reference transcript must exactly match the authorized audio."
if not REFERENCE_AUDIO.is_file():
    raise FileNotFoundError(REFERENCE_AUDIO)

model = AutoModelForTextToSpeech.from_pretrained(
    'ResembleAI/chatterbox',
    model_type='chatterbox',
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
    max_new_tokens=1_024,
)
print(output.file_path, output.sample_rate, output.metadata)
```

Use authorized recordings. Verify hardware needs and pin a revision in production.

## Overview

`chatterbox` is a VoiceHub **text to speech**
integration. This page is generated from its registry contract. [Open the `chatterbox` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/chatterbox.ipynb).

| Property | Value |
| --- | --- |
| Task | Text to speech |
| Architecture | `chatterbox` |
| Runtime | `VoiceHub-native` |
| Languages | `ar`, `da`, `de`, `el`, … complete audited list below |
| Capabilities | `text-to-speech`, `voice-cloning`, `fine-tuning`, `safetensors`, `voicehub-native`, `native-runtime`, `raw-audio-fine-tuning` |
| Reusable components | `conformer` |
| Normalized output | `TTSOutput` |

### Language support

<details class="vh-language-support" markdown>
<summary>Supported language abbreviations</summary>

`ar`, `da`, `de`, `el`, `en`, `es`, `fi`, `fr`, `he`, `hi`, `it`, `ja`, `ko`, `ms`, `nl`, `no`, `pl`, `pt`, `ru`, `sv`, `sw`, `tr`, `zh`

</details>

## Paper and GitHub

- **Paper:** No dedicated upstream research paper is published for this integration.
- **Upstream GitHub:** [Chatterbox](https://github.com/resemble-ai/chatterbox)
- **VoiceHub source:** [VoiceHub model implementation](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/chatterbox/modeling_chatterbox.py)

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('chatterbox')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `chatterbox` |
| Configuration class | `ChatterboxConfig` |
| Architecture class | `ChatterboxForTextToSpeech` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'ResembleAI/chatterbox',
    model_type='chatterbox',
)
print(type(processor).__name__)
```

## Inference

The Usage example returns `TTSOutput` through `AutoModelForTextToSpeech`.

### Input and output contract

| Property | Value |
| --- | --- |
| Readiness | `integrated-raw` |
| Data architecture | `hybrid` |
| Sample rate | 24,000 Hz |
| Contract getter | `get_tts_dataset_spec('chatterbox')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `t3-raw` | `text` | audio / audio_path | Source | — |
| `flow-raw` | — | audio / audio_path | Source | — |
| `t3-precomputed` | `text_tokens`, `speech_tokens`, `speaker_emb` | — | Prepared | — |
| `flow-precomputed` | `speech_token`, `speech_feat`, `embedding` | — | Prepared | — |

Multi-component language-model, diffusion, acoustic, or GAN data. See the [data workflow](../../guides/data-preparation.md).

## Training and optimization

Use `available_optimization_passes()` to discover reversible public passes.
Unsupported runtime or hardware fails closed before mutation.

### Training contract

| Property | Value |
| --- | --- |
| Support | `custom` |
| Family | `composite` |
| Recipe | `multi-phase` |
| Default phase | `language_model` |
| Training checkpoint | `ResembleAI/chatterbox` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `language_model` | objective | `model.t3` | — | `loss`, `text_loss`, `speech_token_loss` |
| `flow` | objective | `model.s3gen.flow` | — | `loss`, `flow_loss`, `diffusion_loss` |

This profile uses model-specific phases; inspect and honor each phase boundary. Call `model.validate_training_support()` first, then follow the
[training workflow](../../guides/training.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | [`ResembleAI/chatterbox`](https://huggingface.co/ResembleAI/chatterbox) |
| Hugging Face ID | [`ResembleAI/chatterbox`](https://huggingface.co/ResembleAI/chatterbox)<br>Repository availability verified through the Hugging Face model API on 2026-08-11; pin a revision before production use. |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.chatterbox.modeling_chatterbox.ChatterboxForTextToSpeech` |
| Configuration | `voicehub.models.chatterbox.configuration_chatterbox.ChatterboxConfig` |
| Source provenance | `voicehub/models/chatterbox/source/SOURCE.json` |
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

### `ChatterboxConfig`

<p class="vh-model-api-card__source-wrap"><a class="vh-model-api-card__source" href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/chatterbox/configuration_chatterbox.py">View source</a></p>
<div class="vh-model-api-card__signature" markdown>

```text
ChatterboxConfig(**config_kwargs)
```

</div>
<h4>Parameters</h4>
<div class="vh-model-api-card__parameters" markdown>
- `**config_kwargs` — Configuration fields validated by ChatterboxConfig.
</div>
</section>

<section class="vh-model-api-card" data-vh-model-api-card="model" markdown>
<p class="vh-model-api-card__badge-wrap"><span class="vh-model-api-card__badge">Model</span></p>

### `ChatterboxForTextToSpeech`

<p class="vh-model-api-card__source-wrap"><a class="vh-model-api-card__source" href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/chatterbox/modeling_chatterbox.py">View source</a></p>
<div class="vh-model-api-card__signature" markdown>

```text
AutoModelForTextToSpeech.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='chatterbox',
    config=None,
    **model_kwargs,
)
```

</div>
<h4>Parameters</h4>
<div class="vh-model-api-card__parameters" markdown>
- `pretrained_model_name_or_path` — Hub ID or compatible local directory.
- `model_type` — Canonical model type; use 'chatterbox'.
- `config` — Optional preloaded ChatterboxConfig instance.
- `**model_kwargs` — Model-specific loading arguments.
</div>
</section>

```python
from voicehub import get_model_spec

spec = get_model_spec('chatterbox')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('chatterbox')` |
| Load and run | `AutoModelForTextToSpeech` |
| Configure | `ChatterboxConfig` |
| Process | `AutoProcessor` |
| Model implementation | `ChatterboxForTextToSpeech` |
| Normalized output | `TTSOutput` |
| Training contract | `get_training_spec('chatterbox')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).

</div>

</div>

</div>
