---
description: Public API, checkpoint, training, and optimization guide for the asr_moonshine integration.
hide:
  - toc
---

<div class="vh-model-detail" data-vh-model-detail data-model-type="asr_moonshine" data-task="automatic-speech-recognition" data-training="native" data-parameter-count="27092736" markdown>

<header class="vh-model-detail__hero" data-vh-model-hero markdown>

<p class="vh-model-detail__namespace" aria-label="Model repository"><span class="vh-model-detail__owner-avatar" aria-hidden="true">US</span><a href="https://huggingface.co/UsefulSensors">UsefulSensors</a><span aria-hidden="true">/</span><strong>moonshine-tiny</strong></p>

# Moonshine {.vh-model-title}

<p class="vh-model-detail__summary">Uses Moonshine&#x27;s short-form speech path with deterministic decoding.</p>
<div class="vh-model-detail__tags" aria-label="Model metadata"><span class="vh-model-detail__chip" data-chip-kind="task">Automatic speech recognition</span><span class="vh-model-detail__chip" data-chip-kind="runtime">VoiceHub-native</span><span class="vh-model-detail__chip" data-chip-kind="architecture">moonshine</span><span class="vh-model-detail__chip" data-chip-kind="parameters" aria-describedby="vh-model-parameters-note-asr_moonshine">Parameters: 27.1M</span><span class="vh-model-detail__chip" data-chip-kind="language">Language: en</span><span class="vh-model-detail__chip" data-chip-kind="training">Training: native</span><span class="vh-model-detail__chip" data-chip-kind="license">License: Checkpoint-specific</span></div>
<p class="vh-model-detail__parameter-note" id="vh-model-parameters-note-asr_moonshine"><strong>Parameter metadata:</strong> Exact Safetensors total reported by the Hugging Face model API for the registered default checkpoint, retrieved 2026-08-13.</p>
<div class="vh-model-detail__actions" aria-label="Model actions">
<a class="vh-model-detail__action vh-model-detail__action--primary" href="#usage" data-vh-model-action="use">Use this model</a>
<button class="vh-model-detail__action vh-model-detail__copy" type="button" data-vh-copy-model-id data-model-id="UsefulSensors/moonshine-tiny" aria-describedby="vh-model-checkpoint-asr_moonshine"><span data-vh-copy-model-id-label>Copy model ID</span></button>
<a class="vh-model-detail__action" href="https://huggingface.co/UsefulSensors/moonshine-tiny" data-vh-model-action="checkpoint">Checkpoint</a>
<details class="vh-model-detail__resources">
<summary class="vh-model-detail__action">Resources</summary>
<div class="vh-model-detail__resource-menu">
<a href="https://arxiv.org/abs/2410.15608" data-vh-model-action="paper">Paper</a>
<a href="https://github.com/moonshine-ai/moonshine" data-vh-model-action="github">Upstream GitHub</a>
<a href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_moonshine/modeling_asr_moonshine.py" data-vh-model-action="source">VoiceHub source</a>
<a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/asr_moonshine.ipynb" data-vh-model-action="colab">Open in Colab</a>
</div>
</details>
</div>
</header>

<nav class="vh-model-detail__tabs" aria-label="Model sections"><a href="#usage" data-vh-model-tab="usage">Usage</a><a href="#overview" data-vh-model-tab="model-card" aria-current="location">Model card</a><a href="#paper-and-github" data-vh-model-tab="sources">Sources</a><a href="#training-and-optimization" data-vh-model-tab="training">Training</a><a href="#checkpoints-provenance-license-and-limitations" data-vh-model-tab="checkpoint">Checkpoint</a><a href="#public-api" data-vh-model-tab="api">Public API</a></nav>

<div class="vh-model-detail__layout" markdown>

<aside class="vh-model-detail__sidebar" data-vh-model-facts aria-labelledby="vh-model-facts-title-asr_moonshine"><h2 id="vh-model-facts-title-asr_moonshine">Model facts</h2><details class="vh-model-detail__facts-disclosure" data-vh-model-facts-disclosure aria-labelledby="vh-model-facts-title-asr_moonshine" open><summary><span>Toggle model facts</span></summary><dl class="vh-model-detail__facts"><div><dt>Task</dt><dd>Automatic speech recognition</dd></div><div><dt>Parameters</dt><dd aria-describedby="vh-model-parameters-note-asr_moonshine">27.1M</dd></div><div><dt>Architecture</dt><dd><code>moonshine</code></dd></div><div><dt>Runtime</dt><dd>VoiceHub-native</dd></div><div><dt>Languages</dt><dd><code>en</code></dd></div><div><dt>Capabilities</dt><dd><details class="vh-model-detail__capabilities"><summary>5 capabilities</summary><span><code>automatic-speech-recognition</code> <code>safetensors</code> <code>fine-tuning</code> <code>compact</code> <code>voicehub-native</code></span></details></dd></div><div><dt>Training</dt><dd><code>native</code></dd></div><div><dt>License</dt><dd>Checkpoint-specific</dd></div><div><dt>Default checkpoint</dt><dd id="vh-model-checkpoint-asr_moonshine"><a href="https://huggingface.co/UsefulSensors/moonshine-tiny"><code>UsefulSensors/moonshine-tiny</code></a></dd></div></dl></details></aside>

<div class="vh-model-detail__main vh-model-detail__content" markdown>

## Usage

Complete the [VoiceHub installation](../../getting-started/installation.md) once,
then run this repository-authored example. Model pages intentionally contain no
package-install command.

This example is maintained against VoiceHub's public API; it is not copied from an upstream demo or package README.

**Model-specific path:** Uses Moonshine's short-form speech path with deterministic decoding.

**Inputs and controls:** Split very long recordings deliberately instead of assuming short-form checkpoint behavior will scale unchanged.

```python
from pathlib import Path

from voicehub import AutoModelForSpeechRecognition

AUDIO_FILE = Path("speech.wav")
if not AUDIO_FILE.is_file():
    raise FileNotFoundError(AUDIO_FILE)

model = AutoModelForSpeechRecognition.from_pretrained(
    'UsefulSensors/moonshine-tiny',
    model_type='asr_moonshine',
    device="cuda",
    lazy_load=True,
)
output = model.transcribe(
    AUDIO_FILE,
    language="en",
    num_beams=1,
)
print(output.text)
for segment in output.segments:
    print(segment.start, segment.end, segment.text, segment.confidence)
```

Use authorized recordings. Verify hardware needs and pin a revision in production.

## Overview

`asr_moonshine` is a VoiceHub **automatic speech recognition**
integration. This page is generated from its registry contract. [Open the `asr_moonshine` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/asr_moonshine.ipynb).

| Property | Value |
| --- | --- |
| Task | Automatic speech recognition |
| Architecture | `moonshine` |
| Runtime | `VoiceHub-native` |
| Languages | `en` |
| Capabilities | `automatic-speech-recognition`, `safetensors`, `fine-tuning`, `compact`, `voicehub-native` |
| Reusable components | — |
| Normalized output | `ASROutput` |

### Language support

<details class="vh-language-support" markdown>
<summary>Supported language abbreviations</summary>

`en`

</details>

## Paper and GitHub

- **Paper:** [Moonshine: Speech Recognition for Live Transcription and Voice Commands](https://arxiv.org/abs/2410.15608)
- **Upstream GitHub:** [Moonshine](https://github.com/moonshine-ai/moonshine)
- **VoiceHub source:** [VoiceHub model implementation](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_moonshine/modeling_asr_moonshine.py)

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('asr_moonshine')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `asr_moonshine` |
| Configuration class | `MoonshineASRConfig` |
| Architecture class | `MoonshineForSpeechRecognition` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'UsefulSensors/moonshine-tiny',
    model_type='asr_moonshine',
)
print(type(processor).__name__)
```

## Inference

The Usage example returns `ASROutput` through `AutoModelForSpeechRecognition`.

### Input and output contract

| Property | Value |
| --- | --- |
| Readiness | `integrated-raw` |
| Data architecture | `speech-sequence-to-sequence` |
| Sample rate | 16,000 Hz |
| Contract getter | `get_asr_dataset_spec('asr_moonshine')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `raw-audio` | `audio` | text / transcription / transcript | Source | at most one: text / transcription / transcript |
| `moonshine-model-ready` | `input_values`, `labels` | — | Prepared | — |

Moonshine waveform-to-sequence fine-tuning records. See the [data workflow](../../guides/speech-data.md).

## Training and optimization

Use `available_optimization_passes()` to discover reversible public passes.
Unsupported runtime or hardware fails closed before mutation.

### Training contract

| Property | Value |
| --- | --- |
| Support | `native` |
| Family | `speech-sequence-to-sequence` |
| Recipe | `single-phase` |
| Default phase | `speech_recognition` |
| Training checkpoint | `UsefulSensors/moonshine-tiny` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `speech_recognition` | objective | `model` | — | `loss` |

The integration accepts its declared source or prepared contract directly. Call `model.validate_training_support()` first, then follow the
[training workflow](../../guides/training.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | [`UsefulSensors/moonshine-tiny`](https://huggingface.co/UsefulSensors/moonshine-tiny) |
| Hugging Face ID | [`UsefulSensors/moonshine-tiny`](https://huggingface.co/UsefulSensors/moonshine-tiny)<br>Repository availability verified through the Hugging Face model API on 2026-08-11; pin a revision before production use. |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.asr_moonshine.modeling_asr_moonshine.MoonshineForSpeechRecognition` |
| Configuration | `voicehub.models.asr_moonshine.configuration_asr_moonshine.MoonshineASRConfig` |
| Source provenance | `voicehub/architectures/moonshine/SOURCE.json` |
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

### `MoonshineASRConfig`

<p class="vh-model-api-card__source-wrap"><a class="vh-model-api-card__source" href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_moonshine/configuration_asr_moonshine.py">View source</a></p>
<div class="vh-model-api-card__signature" markdown>

```text
MoonshineASRConfig(**config_kwargs)
```

</div>
<h4>Parameters</h4>
<div class="vh-model-api-card__parameters" markdown>
- `**config_kwargs` — Configuration fields validated by MoonshineASRConfig.
</div>
</section>

<section class="vh-model-api-card" data-vh-model-api-card="model" markdown>
<p class="vh-model-api-card__badge-wrap"><span class="vh-model-api-card__badge">Model</span></p>

### `MoonshineForSpeechRecognition`

<p class="vh-model-api-card__source-wrap"><a class="vh-model-api-card__source" href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_moonshine/modeling_asr_moonshine.py">View source</a></p>
<div class="vh-model-api-card__signature" markdown>

```text
AutoModelForSpeechRecognition.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='asr_moonshine',
    config=None,
    **model_kwargs,
)
```

</div>
<h4>Parameters</h4>
<div class="vh-model-api-card__parameters" markdown>
- `pretrained_model_name_or_path` — Hub ID or compatible local directory.
- `model_type` — Canonical model type; use 'asr_moonshine'.
- `config` — Optional preloaded MoonshineASRConfig instance.
- `**model_kwargs` — Model-specific loading arguments.
</div>
</section>

```python
from voicehub import get_model_spec

spec = get_model_spec('asr_moonshine')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('asr_moonshine')` |
| Load and run | `AutoModelForSpeechRecognition` |
| Configure | `MoonshineASRConfig` |
| Process | `AutoProcessor` |
| Model implementation | `MoonshineForSpeechRecognition` |
| Normalized output | `ASROutput` |
| Training contract | `get_training_spec('asr_moonshine')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).

</div>

</div>

</div>
