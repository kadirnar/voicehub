---
description: Public API, checkpoint, training, and optimization guide for the voxcpm integration.
hide:
  - toc
---

<div class="vh-model-detail" data-vh-model-detail data-model-type="voxcpm" data-task="text-to-speech" data-training="native" data-parameter-count="2290004544" markdown>

<header class="vh-model-detail__hero" data-vh-model-hero markdown>

<p class="vh-model-detail__namespace" aria-label="Model repository"><span class="vh-model-detail__owner-avatar" aria-hidden="true">OP</span><a href="https://huggingface.co/openbmb">openbmb</a><span aria-hidden="true">/</span><strong>VoxCPM2</strong></p>

# VoxCPM {.vh-model-title}

<p class="vh-model-detail__summary">Conditions VoxCPM2 on a reference timbre and exposes its diffusion guidance and step count.</p>
<div class="vh-model-detail__tags" aria-label="Model metadata"><span class="vh-model-detail__chip" data-chip-kind="task">Text to speech</span><span class="vh-model-detail__chip" data-chip-kind="runtime">VoiceHub-native</span><span class="vh-model-detail__chip" data-chip-kind="architecture">voxcpm2</span><span class="vh-model-detail__chip" data-chip-kind="parameters" aria-describedby="vh-model-parameters-note-voxcpm">Parameters: 2.3B</span><span class="vh-model-detail__chip" data-chip-kind="language">Languages: zh, en +28</span><span class="vh-model-detail__chip" data-chip-kind="training">Training: native</span><span class="vh-model-detail__chip" data-chip-kind="license">License: Checkpoint-specific</span></div>
<p class="vh-model-detail__parameter-note" id="vh-model-parameters-note-voxcpm"><strong>Parameter metadata:</strong> Exact learned-parameter total for VoiceHub&#x27;s audited native primary graph at the registered default selection; separately loaded auxiliary models are excluded.</p>
<div class="vh-model-detail__actions" aria-label="Model actions">
<a class="vh-model-detail__action vh-model-detail__action--primary" href="#usage" data-vh-model-action="use">Use this model</a>
<button class="vh-model-detail__action vh-model-detail__copy" type="button" data-vh-copy-model-id data-model-id="openbmb/VoxCPM2" aria-describedby="vh-model-checkpoint-voxcpm"><span data-vh-copy-model-id-label>Copy model ID</span></button>
<a class="vh-model-detail__action" href="https://huggingface.co/openbmb/VoxCPM2" data-vh-model-action="checkpoint">Checkpoint</a>
<details class="vh-model-detail__resources">
<summary class="vh-model-detail__action">Resources</summary>
<div class="vh-model-detail__resource-menu">
<a href="https://github.com/OpenBMB/VoxCPM" data-vh-model-action="github">Upstream GitHub</a>
<a href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/voxcpm/modeling_voxcpm.py" data-vh-model-action="source">VoiceHub source</a>
<a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/voxcpm.ipynb" data-vh-model-action="colab">Open in Colab</a>
</div>
</details>
</div>
</header>

<nav class="vh-model-detail__tabs" aria-label="Model sections"><a href="#usage" data-vh-model-tab="usage">Usage</a><a href="#overview" data-vh-model-tab="model-card" aria-current="location">Model card</a><a href="#paper-and-github" data-vh-model-tab="sources">Sources</a><a href="#training-and-optimization" data-vh-model-tab="training">Training</a><a href="#checkpoints-provenance-license-and-limitations" data-vh-model-tab="checkpoint">Checkpoint</a><a href="#public-api" data-vh-model-tab="api">Public API</a></nav>

<div class="vh-model-detail__layout" markdown>

<aside class="vh-model-detail__sidebar" data-vh-model-facts aria-labelledby="vh-model-facts-title-voxcpm"><h2 id="vh-model-facts-title-voxcpm">Model facts</h2><details class="vh-model-detail__facts-disclosure" data-vh-model-facts-disclosure aria-labelledby="vh-model-facts-title-voxcpm" open><summary><span>Toggle model facts</span></summary><dl class="vh-model-detail__facts"><div><dt>Task</dt><dd>Text to speech</dd></div><div><dt>Parameters</dt><dd aria-describedby="vh-model-parameters-note-voxcpm">2.3B</dd></div><div><dt>Architecture</dt><dd><code>voxcpm2</code></dd></div><div><dt>Runtime</dt><dd>VoiceHub-native</dd></div><div><dt>Languages</dt><dd><details class="vh-model-detail__languages"><summary>30 documented codes</summary><span><code>zh</code> <code>en</code> <code>ar</code> <code>my</code> <code>da</code> <code>nl</code> <code>fi</code> <code>fr</code> <code>de</code> <code>el</code> <code>he</code> <code>hi</code> <code>id</code> <code>it</code> <code>ja</code> <code>km</code> <code>ko</code> <code>lo</code> <code>ms</code> <code>no</code> <code>pl</code> <code>pt</code> <code>ru</code> <code>es</code> <code>sw</code> <code>sv</code> <code>tl</code> <code>th</code> <code>tr</code> <code>vi</code></span></details></dd></div><div><dt>Capabilities</dt><dd><details class="vh-model-detail__capabilities"><summary>9 capabilities</summary><span><code>text-to-speech</code> <code>voice-cloning</code> <code>voice-design</code> <code>audio-continuation</code> <code>multilingual</code> <code>fine-tuning</code> <code>safetensors</code> <code>voicehub-native</code> <code>native-runtime</code></span></details></dd></div><div><dt>Training</dt><dd><code>native</code></dd></div><div><dt>License</dt><dd>Checkpoint-specific</dd></div><div><dt>Default checkpoint</dt><dd id="vh-model-checkpoint-voxcpm"><a href="https://huggingface.co/openbmb/VoxCPM2"><code>openbmb/VoxCPM2</code></a></dd></div></dl></details></aside>

<div class="vh-model-detail__main vh-model-detail__content" markdown>

## Usage

Complete the [VoiceHub installation](../../getting-started/installation.md) once,
then run this repository-authored example. Model pages intentionally contain no
package-install command.

This example is maintained against VoiceHub's public API; it is not copied from an upstream demo or package README.

**Model-specific path:** Conditions VoxCPM2 on a reference timbre and exposes its diffusion guidance and step count.

**Inputs and controls:** A prompt transcript is required only with `prompt_audio_path`; the timbre-only field used here is separate.

```python
from pathlib import Path

from voicehub import AutoModelForTextToSpeech, TTSGenerationConfig

REFERENCE_AUDIO = Path("reference.wav")
REFERENCE_TEXT = "The reference transcript must exactly match the authorized audio."
if not REFERENCE_AUDIO.is_file():
    raise FileNotFoundError(REFERENCE_AUDIO)

model = AutoModelForTextToSpeech.from_pretrained(
    'openbmb/VoxCPM2',
    model_type='voxcpm',
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
    cfg_value=2.0,
    inference_timesteps=10,
)
print(output.file_path, output.sample_rate, output.metadata)
```

Use authorized recordings. Verify hardware needs and pin a revision in production.

## Overview

`voxcpm` is a VoiceHub **text to speech**
integration. This page is generated from its registry contract. [Open the `voxcpm` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/voxcpm.ipynb).

| Property | Value |
| --- | --- |
| Task | Text to speech |
| Architecture | `voxcpm2` |
| Runtime | `VoiceHub-native` |
| Languages | `zh`, `en`, `ar`, `my`, … complete audited list below |
| Capabilities | `text-to-speech`, `voice-cloning`, `voice-design`, `audio-continuation`, `multilingual`, `fine-tuning`, `safetensors`, `voicehub-native`, `native-runtime` |
| Reusable components | — |
| Normalized output | `TTSOutput` |

### Language support

<details class="vh-language-support" markdown>
<summary>Supported language abbreviations</summary>

`zh`, `en`, `ar`, `my`, `da`, `nl`, `fi`, `fr`, `de`, `el`, `he`, `hi`, `id`, `it`, `ja`, `km`, `ko`, `lo`, `ms`, `no`, `pl`, `pt`, `ru`, `es`, `sw`, `sv`, `tl`, `th`, `tr`, `vi`

</details>

## Paper and GitHub

- **Paper:** No dedicated upstream research paper is published for this integration.
- **Upstream GitHub:** [VoxCPM](https://github.com/OpenBMB/VoxCPM)
- **VoiceHub source:** [VoiceHub model implementation](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/voxcpm/modeling_voxcpm.py)

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('voxcpm')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `voxcpm` |
| Configuration class | `VoxCPMConfig` |
| Architecture class | `VoxCPMForTextToSpeech` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'openbmb/VoxCPM2',
    model_type='voxcpm',
)
print(type(processor).__name__)
```

## Inference

The Usage example returns `TTSOutput` through `AutoModelForTextToSpeech`.

### Input and output contract

| Property | Value |
| --- | --- |
| Readiness | `integrated-raw` |
| Data architecture | `diffusion` |
| Sample rate | 16,000 Hz |
| Contract getter | `get_tts_dataset_spec('voxcpm')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `raw-waveform` | `text` | audio / waveform | Source | at most one: audio / waveform; forbidden: audio_features |
| `audio-features` | `text`, `audio_features` | — | Prepared | forbidden: audio, waveform |

Conditional flow-matching, rectified-flow, or diffusion data. See the [data workflow](../../guides/data-preparation.md).

## Training and optimization

Use `available_optimization_passes()` to discover reversible public passes.
Unsupported runtime or hardware fails closed before mutation.

### Training contract

| Property | Value |
| --- | --- |
| Support | `native` |
| Family | `flow-matching` |
| Recipe | `single-phase` |
| Default phase | `source_flow_and_stop` |
| Training checkpoint | `openbmb/VoxCPM2` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `source_flow_and_stop` | objective | `model` | `text_tokens`, `text_mask`, `audio_feats`, `audio_mask`, `loss_mask`, `position_ids`, `labels` | `diffusion_loss`, `stop_loss` |

The integration accepts its declared source or prepared contract directly. Call `model.validate_training_support()` first, then follow the
[training workflow](../../guides/training.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | [`openbmb/VoxCPM2`](https://huggingface.co/openbmb/VoxCPM2) |
| Hugging Face ID | [`openbmb/VoxCPM2`](https://huggingface.co/openbmb/VoxCPM2)<br>Repository availability verified through the Hugging Face model API on 2026-08-11; pin a revision before production use. |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.voxcpm.modeling_voxcpm.VoxCPMForTextToSpeech` |
| Configuration | `voicehub.models.voxcpm.configuration_voxcpm.VoxCPMConfig` |
| Source provenance | `voicehub/models/voxcpm/source/SOURCE.json` |
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

### `VoxCPMConfig`

<p class="vh-model-api-card__source-wrap"><a class="vh-model-api-card__source" href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/voxcpm/configuration_voxcpm.py">View source</a></p>
<div class="vh-model-api-card__signature" markdown>

```text
VoxCPMConfig(**config_kwargs)
```

</div>
<h4>Parameters</h4>
<div class="vh-model-api-card__parameters" markdown>
- `**config_kwargs` — Configuration fields validated by VoxCPMConfig.
</div>
</section>

<section class="vh-model-api-card" data-vh-model-api-card="model" markdown>
<p class="vh-model-api-card__badge-wrap"><span class="vh-model-api-card__badge">Model</span></p>

### `VoxCPMForTextToSpeech`

<p class="vh-model-api-card__source-wrap"><a class="vh-model-api-card__source" href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/voxcpm/modeling_voxcpm.py">View source</a></p>
<div class="vh-model-api-card__signature" markdown>

```text
AutoModelForTextToSpeech.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='voxcpm',
    config=None,
    **model_kwargs,
)
```

</div>
<h4>Parameters</h4>
<div class="vh-model-api-card__parameters" markdown>
- `pretrained_model_name_or_path` — Hub ID or compatible local directory.
- `model_type` — Canonical model type; use 'voxcpm'.
- `config` — Optional preloaded VoxCPMConfig instance.
- `**model_kwargs` — Model-specific loading arguments.
</div>
</section>

```python
from voicehub import get_model_spec

spec = get_model_spec('voxcpm')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('voxcpm')` |
| Load and run | `AutoModelForTextToSpeech` |
| Configure | `VoxCPMConfig` |
| Process | `AutoProcessor` |
| Model implementation | `VoxCPMForTextToSpeech` |
| Normalized output | `TTSOutput` |
| Training contract | `get_training_spec('voxcpm')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).

</div>

</div>

</div>
