---
description: Public API, checkpoint, training, and optimization guide for the openvoice integration.
hide:
  - toc
---

<div class="vh-model-detail" data-vh-model-detail data-model-type="openvoice" data-task="text-to-speech" data-training="custom" data-parameter-count="32792226" markdown>

<header class="vh-model-detail__hero" data-vh-model-hero markdown>

<p class="vh-model-detail__namespace" aria-label="Model repository"><span class="vh-model-detail__owner-avatar" aria-hidden="true">MA</span><a href="https://huggingface.co/myshell-ai">myshell-ai</a><span aria-hidden="true">/</span><strong>OpenVoiceV2</strong></p>

# OpenVoice {.vh-model-title}

<p class="vh-model-detail__summary">Runs OpenVoice tone-color transfer from a source utterance to an authorized target-speaker recording.</p>
<div class="vh-model-detail__tags" aria-label="Model metadata"><span class="vh-model-detail__chip" data-chip-kind="task">Text to speech</span><span class="vh-model-detail__chip" data-chip-kind="runtime">VoiceHub-native</span><span class="vh-model-detail__chip" data-chip-kind="architecture">openvoice-v2-converter</span><span class="vh-model-detail__chip" data-chip-kind="parameters" aria-describedby="vh-model-parameters-note-openvoice">Parameters: 32.8M</span><span class="vh-model-detail__chip" data-chip-kind="language">Languages: en, es +4</span><span class="vh-model-detail__chip" data-chip-kind="training">Training: custom</span><span class="vh-model-detail__chip" data-chip-kind="license">License: Checkpoint-specific</span></div>
<p class="vh-model-detail__parameter-note" id="vh-model-parameters-note-openvoice"><strong>Parameter metadata:</strong> Exact learned-parameter total for VoiceHub&#x27;s audited native primary graph at the registered default selection; separately loaded auxiliary models are excluded.</p>
<div class="vh-model-detail__actions" aria-label="Model actions">
<a class="vh-model-detail__action vh-model-detail__action--primary" href="#usage" data-vh-model-action="use">Use this model</a>
<button class="vh-model-detail__action vh-model-detail__copy" type="button" data-vh-copy-model-id data-model-id="myshell-ai/OpenVoiceV2" aria-describedby="vh-model-checkpoint-openvoice"><span data-vh-copy-model-id-label>Copy model ID</span></button>
<a class="vh-model-detail__action" href="https://huggingface.co/myshell-ai/OpenVoiceV2" data-vh-model-action="checkpoint">Checkpoint</a>
<details class="vh-model-detail__resources">
<summary class="vh-model-detail__action">Resources</summary>
<div class="vh-model-detail__resource-menu">
<a href="https://arxiv.org/abs/2312.01479" data-vh-model-action="paper">Paper</a>
<a href="https://github.com/myshell-ai/OpenVoice" data-vh-model-action="github">Upstream GitHub</a>
<a href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/openvoice/modeling_openvoice.py" data-vh-model-action="source">VoiceHub source</a>
<a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/openvoice.ipynb" data-vh-model-action="colab">Open in Colab</a>
</div>
</details>
</div>
</header>

<nav class="vh-model-detail__tabs" aria-label="Model sections"><a href="#usage" data-vh-model-tab="usage">Usage</a><a href="#overview" data-vh-model-tab="model-card" aria-current="location">Model card</a><a href="#paper-and-github" data-vh-model-tab="sources">Sources</a><a href="#training-and-optimization" data-vh-model-tab="training">Training</a><a href="#checkpoints-provenance-license-and-limitations" data-vh-model-tab="checkpoint">Checkpoint</a><a href="#public-api" data-vh-model-tab="api">Public API</a></nav>

<div class="vh-model-detail__layout" markdown>

<aside class="vh-model-detail__sidebar" data-vh-model-facts aria-labelledby="vh-model-facts-title-openvoice"><h2 id="vh-model-facts-title-openvoice">Model facts</h2><details class="vh-model-detail__facts-disclosure" data-vh-model-facts-disclosure aria-labelledby="vh-model-facts-title-openvoice" open><summary><span>Toggle model facts</span></summary><dl class="vh-model-detail__facts"><div><dt>Task</dt><dd>Text to speech</dd></div><div><dt>Parameters</dt><dd aria-describedby="vh-model-parameters-note-openvoice">32.8M</dd></div><div><dt>Architecture</dt><dd><code>openvoice-v2-converter</code></dd></div><div><dt>Runtime</dt><dd>VoiceHub-native</dd></div><div><dt>Languages</dt><dd><details class="vh-model-detail__languages"><summary>6 documented codes</summary><span><code>en</code> <code>es</code> <code>fr</code> <code>zh</code> <code>ja</code> <code>ko</code></span></details></dd></div><div><dt>Capabilities</dt><dd><details class="vh-model-detail__capabilities"><summary>9 capabilities</summary><span><code>text-to-speech</code> <code>voice-cloning</code> <code>multilingual</code> <code>fine-tuning</code> <code>safetensors</code> <code>voicehub-native</code> <code>native-runtime</code> <code>paired-waveform-training</code> <code>explicit-base-waveform</code></span></details></dd></div><div><dt>Training</dt><dd><code>custom</code></dd></div><div><dt>License</dt><dd>Checkpoint-specific</dd></div><div><dt>Default checkpoint</dt><dd id="vh-model-checkpoint-openvoice"><a href="https://huggingface.co/myshell-ai/OpenVoiceV2"><code>myshell-ai/OpenVoiceV2</code></a></dd></div></dl></details></aside>

<div class="vh-model-detail__main vh-model-detail__content" markdown>

## Usage

Complete the [VoiceHub installation](../../getting-started/installation.md) once,
then run this repository-authored example. Model pages intentionally contain no
package-install command.

This example is maintained against VoiceHub's public API; it is not copied from an upstream demo or package README.

**Model-specific path:** Runs OpenVoice tone-color transfer from a source utterance to an authorized target-speaker recording.

**Inputs and controls:** `base.wav` must contain the utterance to convert; `reference.wav` supplies only the target tone color.

```python
from pathlib import Path

from voicehub import AutoModelForTextToSpeech, TTSGenerationConfig

BASE_AUDIO = Path("base.wav")
REFERENCE_AUDIO = Path("reference.wav")
for audio_file in (BASE_AUDIO, REFERENCE_AUDIO):
    if not audio_file.is_file():
        raise FileNotFoundError(audio_file)

model = AutoModelForTextToSpeech.from_pretrained(
    'myshell-ai/OpenVoiceV2',
    model_type='openvoice',
    device="cuda",
    lazy_load=True,
)
output = model.generate(
    'VoiceHub keeps model integrations explicit and reproducible.',
    generation_config=TTSGenerationConfig(
        seed=42,
        output_file=Path("output.wav"),
    ),
    base_audio=str(BASE_AUDIO),
    speaker_audio_path=str(REFERENCE_AUDIO),
    tau=0.3,
)
print(output.file_path, output.sample_rate, output.metadata)
```

Use authorized recordings. Verify hardware needs and pin a revision in production.

## Overview

`openvoice` is a VoiceHub **text to speech**
integration. This page is generated from its registry contract. [Open the `openvoice` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/openvoice.ipynb).

| Property | Value |
| --- | --- |
| Task | Text to speech |
| Architecture | `openvoice-v2-converter` |
| Runtime | `VoiceHub-native` |
| Languages | `en`, `es`, `fr`, `zh`, `ja`, `ko` |
| Capabilities | `text-to-speech`, `voice-cloning`, `multilingual`, `fine-tuning`, `safetensors`, `voicehub-native`, `native-runtime`, `paired-waveform-training`, `explicit-base-waveform` |
| Reusable components | `wavmark` |
| Normalized output | `TTSOutput` |

### Language support

<details class="vh-language-support" markdown>
<summary>Supported language abbreviations</summary>

`en`, `es`, `fr`, `zh`, `ja`, `ko`

</details>

## Paper and GitHub

- **Paper:** [OpenVoice: Versatile Instant Voice Cloning](https://arxiv.org/abs/2312.01479)
- **Upstream GitHub:** [OpenVoice](https://github.com/myshell-ai/OpenVoice)
- **VoiceHub source:** [VoiceHub model implementation](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/openvoice/modeling_openvoice.py)

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('openvoice')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `openvoice` |
| Configuration class | `OpenVoiceConfig` |
| Architecture class | `OpenVoiceForTextToSpeech` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'myshell-ai/OpenVoiceV2',
    model_type='openvoice',
)
print(type(processor).__name__)
```

## Inference

The Usage example returns `TTSOutput` through `AutoModelForTextToSpeech`.

### Input and output contract

| Property | Value |
| --- | --- |
| Readiness | `integrated-raw` |
| Data architecture | `vits` |
| Sample rate | 22,050 Hz |
| Contract getter | `get_tts_dataset_spec('openvoice')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `paired-waveforms` | `source_audio`, `target_audio` | — | Source | — |
| `paired-waveform-aliases` | `audio`, `target_waveform` | — | Source | — |

VITS/GAN text, waveform, spectrogram, and adversarial data. See the [data workflow](../../guides/data-preparation.md).

## Training and optimization

Use `available_optimization_passes()` to discover reversible public passes.
Unsupported runtime or hardware fails closed before mutation.

### Training contract

| Property | Value |
| --- | --- |
| Support | `custom` |
| Family | `vits` |
| Recipe | `single-phase` |
| Default phase | `generator` |
| Training checkpoint | `myshell-ai/OpenVoiceV2` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `generator` | generator | `model.enc_q`, `model.flow`, `model.dec`, `model.ref_enc` | `source_spectrogram`, `source_lengths`, `target_waveform`, `target_lengths` | `loss` |

This profile uses model-specific phases; inspect and honor each phase boundary. Call `model.validate_training_support()` first, then follow the
[training workflow](../../guides/training.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | [`myshell-ai/OpenVoiceV2`](https://huggingface.co/myshell-ai/OpenVoiceV2) |
| Hugging Face ID | [`myshell-ai/OpenVoiceV2`](https://huggingface.co/myshell-ai/OpenVoiceV2)<br>Repository availability verified through the Hugging Face model API on 2026-08-11; pin a revision before production use. |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.openvoice.modeling_openvoice.OpenVoiceForTextToSpeech` |
| Configuration | `voicehub.models.openvoice.configuration_openvoice.OpenVoiceConfig` |
| Source provenance | `voicehub/models/openvoice/source/SOURCE.json` |
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

### `OpenVoiceConfig`

<p class="vh-model-api-card__source-wrap"><a class="vh-model-api-card__source" href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/openvoice/configuration_openvoice.py">View source</a></p>
<div class="vh-model-api-card__signature" markdown>

```text
OpenVoiceConfig(**config_kwargs)
```

</div>
<h4>Parameters</h4>
<div class="vh-model-api-card__parameters" markdown>
- `**config_kwargs` — Configuration fields validated by OpenVoiceConfig.
</div>
</section>

<section class="vh-model-api-card" data-vh-model-api-card="model" markdown>
<p class="vh-model-api-card__badge-wrap"><span class="vh-model-api-card__badge">Model</span></p>

### `OpenVoiceForTextToSpeech`

<p class="vh-model-api-card__source-wrap"><a class="vh-model-api-card__source" href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/openvoice/modeling_openvoice.py">View source</a></p>
<div class="vh-model-api-card__signature" markdown>

```text
AutoModelForTextToSpeech.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='openvoice',
    config=None,
    **model_kwargs,
)
```

</div>
<h4>Parameters</h4>
<div class="vh-model-api-card__parameters" markdown>
- `pretrained_model_name_or_path` — Hub ID or compatible local directory.
- `model_type` — Canonical model type; use 'openvoice'.
- `config` — Optional preloaded OpenVoiceConfig instance.
- `**model_kwargs` — Model-specific loading arguments.
</div>
</section>

```python
from voicehub import get_model_spec

spec = get_model_spec('openvoice')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('openvoice')` |
| Load and run | `AutoModelForTextToSpeech` |
| Configure | `OpenVoiceConfig` |
| Process | `AutoProcessor` |
| Model implementation | `OpenVoiceForTextToSpeech` |
| Normalized output | `TTSOutput` |
| Training contract | `get_training_spec('openvoice')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).

</div>

</div>

</div>
