---
description: Public API, checkpoint, training, and optimization guide for the styletts2 integration.
hide:
  - toc
---

<div class="vh-model-detail" data-vh-model-detail data-model-type="styletts2" data-task="text-to-speech" data-training="preprocessed" data-parameter-count="" markdown>

<header class="vh-model-detail__hero" data-vh-model-hero markdown>

<p class="vh-model-detail__namespace" aria-label="Model repository"><span class="vh-model-detail__owner-avatar" aria-hidden="true">VH</span><a href="https://github.com/kadirnar/voicehub">VoiceHub</a><span aria-hidden="true">/</span><strong>styletts2</strong></p>

# StyleTTS2 {.vh-model-title}

<p class="vh-model-detail__summary">Uses an explicit local VoiceHub artifact and the native phoneme boundary required by StyleTTS 2.</p>
<div class="vh-model-detail__tags" aria-label="Model metadata"><span class="vh-model-detail__chip" data-chip-kind="task">Text to speech</span><span class="vh-model-detail__chip" data-chip-kind="runtime">VoiceHub-native</span><span class="vh-model-detail__chip" data-chip-kind="architecture">styletts2</span><span class="vh-model-detail__chip" data-chip-kind="parameters" aria-describedby="vh-model-parameters-note-styletts2">Parameters: Not reported</span><span class="vh-model-detail__chip" data-chip-kind="language">Language: en-US</span><span class="vh-model-detail__chip" data-chip-kind="training">Training: preprocessed</span><span class="vh-model-detail__chip" data-chip-kind="license">License: Checkpoint-specific</span></div>
<p class="vh-model-detail__parameter-note" id="vh-model-parameters-note-styletts2"><strong>Parameter metadata:</strong> Not reported: the audited metadata available for the registered default does not provide an exact parameter total.</p>
<div class="vh-model-detail__actions" aria-label="Model actions">
<a class="vh-model-detail__action vh-model-detail__action--primary" href="#usage" data-vh-model-action="use">Use this model</a>
<details class="vh-model-detail__resources">
<summary class="vh-model-detail__action">Resources</summary>
<div class="vh-model-detail__resource-menu">
<a href="https://arxiv.org/abs/2306.07691" data-vh-model-action="paper">Paper</a>
<a href="https://github.com/yl4579/StyleTTS2" data-vh-model-action="github">Upstream GitHub</a>
<a href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/styletts2/modeling_styletts2.py" data-vh-model-action="source">VoiceHub source</a>
</div>
</details>
</div>
</header>

<nav class="vh-model-detail__tabs" aria-label="Model sections"><a href="#usage" data-vh-model-tab="usage">Usage</a><a href="#overview" data-vh-model-tab="model-card" aria-current="location">Model card</a><a href="#paper-and-github" data-vh-model-tab="sources">Sources</a><a href="#training-and-optimization" data-vh-model-tab="training">Training</a><a href="#checkpoints-provenance-license-and-limitations" data-vh-model-tab="checkpoint">Checkpoint</a><a href="#public-api" data-vh-model-tab="api">Public API</a></nav>

<div class="vh-model-detail__layout" markdown>

<aside class="vh-model-detail__sidebar" data-vh-model-facts aria-labelledby="vh-model-facts-title-styletts2"><h2 id="vh-model-facts-title-styletts2">Model facts</h2><details class="vh-model-detail__facts-disclosure" data-vh-model-facts-disclosure aria-labelledby="vh-model-facts-title-styletts2" open><summary><span>Toggle model facts</span></summary><dl class="vh-model-detail__facts"><div><dt>Task</dt><dd>Text to speech</dd></div><div><dt>Parameters</dt><dd aria-describedby="vh-model-parameters-note-styletts2">Not reported</dd></div><div><dt>Architecture</dt><dd><code>styletts2</code></dd></div><div><dt>Runtime</dt><dd>VoiceHub-native</dd></div><div><dt>Languages</dt><dd><code>en-US</code></dd></div><div><dt>Capabilities</dt><dd><details class="vh-model-detail__capabilities"><summary>8 capabilities</summary><span><code>text-to-speech</code> <code>voice-cloning</code> <code>fine-tuning</code> <code>safetensors</code> <code>voicehub-native</code> <code>native-runtime</code> <code>preprocessed-training</code> <code>explicit-phonemes</code></span></details></dd></div><div><dt>Training</dt><dd><code>preprocessed</code></dd></div><div><dt>License</dt><dd>Checkpoint-specific</dd></div><div><dt>Default checkpoint</dt><dd id="vh-model-checkpoint-styletts2">Caller-provided compatible artifact</dd></div></dl></details></aside>

<div class="vh-model-detail__main vh-model-detail__content" markdown>

## Usage

Complete the [VoiceHub installation](../../getting-started/installation.md) once,
then run this repository-authored example. Model pages intentionally contain no
package-install command.

This example is maintained against VoiceHub's public API; it is not copied from an upstream demo or package README.

**Model-specific path:** Uses an explicit local VoiceHub artifact and the native phoneme boundary required by StyleTTS 2.

**Inputs and controls:** Convert or review the upstream LibriTTS files first; the HF repository is provenance, not a drop-in VoiceHub directory.

```python
from pathlib import Path

from voicehub import AutoModelForTextToSpeech, TTSGenerationConfig

REFERENCE_AUDIO = Path("reference.wav")
REFERENCE_TEXT = "The reference transcript must exactly match the authorized audio."
if not REFERENCE_AUDIO.is_file():
    raise FileNotFoundError(REFERENCE_AUDIO)

model = AutoModelForTextToSpeech.from_pretrained(
    'checkpoints/styletts2/model.safetensors',
    model_type='styletts2',
    device="cuda",
    lazy_load=True,
)
output = model.generate(
    'həˈloʊ fɹʌm vɔɪs hʌb',
    generation_config=TTSGenerationConfig(
        seed=42,
        output_file=Path("output.wav"),
    ),
    speaker_audio_path=str(REFERENCE_AUDIO),
    text_is_phonemes=True,
    diffusion_steps=5,
    embedding_scale=1.0,
)
print(output.file_path, output.sample_rate, output.metadata)
```

Use authorized recordings. Verify hardware needs and pin a revision in production.

## Overview

`styletts2` is a VoiceHub **text to speech**
integration. This page is generated from its registry contract.

| Property | Value |
| --- | --- |
| Task | Text to speech |
| Architecture | `styletts2` |
| Runtime | `VoiceHub-native` |
| Languages | `en-US` |
| Capabilities | `text-to-speech`, `voice-cloning`, `fine-tuning`, `safetensors`, `voicehub-native`, `native-runtime`, `preprocessed-training`, `explicit-phonemes` |
| Reusable components | — |
| Normalized output | `TTSOutput` |

### Language support

<details class="vh-language-support" markdown>
<summary>Supported language abbreviations</summary>

`en-US`

</details>

## Paper and GitHub

- **Paper:** [StyleTTS 2: Towards Human-Level Text-to-Speech through Style Diffusion](https://arxiv.org/abs/2306.07691)
- **Upstream GitHub:** [StyleTTS 2](https://github.com/yl4579/StyleTTS2)
- **VoiceHub source:** [VoiceHub model implementation](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/styletts2/modeling_styletts2.py)

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('styletts2')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `styletts2` |
| Configuration class | `StyleTTS2Config` |
| Architecture class | `StyleTTS2ForTextToSpeech` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'checkpoints/styletts2/model.safetensors',
    model_type='styletts2',
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
| Contract getter | `get_tts_dataset_spec('styletts2')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `explicit-features` | `input_ids`, `alignments`, `normalized_mel`, `reference_mel`, `f0_targets`, `noise_targets`, `audio_values` | — | Prepared | — |

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
| Training checkpoint | `owner/model-or-local-directory` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `generator` | generator | `training_model.model` | `input_ids`, `input_lengths`, `alignments`, `alignment_lengths`, `normalized_mel`, `normalized_mel_lengths`, `reference_mel`, `reference_mel_lengths`, `f0_targets`, `noise_targets`, `audio_values`, `audio_lengths` | `loss` |
| `discriminator` | discriminator | `training_model.mpd`, `training_model.msd` | `input_ids`, `input_lengths`, `alignments`, `alignment_lengths`, `normalized_mel`, `normalized_mel_lengths`, `reference_mel`, `reference_mel_lengths`, `f0_targets`, `noise_targets`, `audio_values`, `audio_lengths` | `loss` |

Prepare the exact tensors listed in the data contract before this step. Call `model.validate_training_support()` first, then follow the
[training workflow](../../guides/training.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | No default; pass a compatible Hub ID or local directory. |
| Hugging Face ID | [`yl4579/StyleTTS2-LibriTTS`](https://huggingface.co/yl4579/StyleTTS2-LibriTTS)<br>Upstream LibriTTS repository, verified available on 2026-08-11. VoiceHub requires a reviewed local artifact because the published layout is not a native VoiceHub directory. |
| Checkpoint status | No registry default; provide the compatible local artifact described on this page |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.styletts2.modeling_styletts2.StyleTTS2ForTextToSpeech` |
| Configuration | `voicehub.models.styletts2.configuration_styletts2.StyleTTS2Config` |
| Source provenance | `voicehub/models/styletts2/source/SOURCE.json` |
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

### `StyleTTS2Config`

<p class="vh-model-api-card__source-wrap"><a class="vh-model-api-card__source" href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/styletts2/configuration_styletts2.py">View source</a></p>
<div class="vh-model-api-card__signature" markdown>

```text
StyleTTS2Config(**config_kwargs)
```

</div>
<h4>Parameters</h4>
<div class="vh-model-api-card__parameters" markdown>
- `**config_kwargs` — Configuration fields validated by StyleTTS2Config.
</div>
</section>

<section class="vh-model-api-card" data-vh-model-api-card="model" markdown>
<p class="vh-model-api-card__badge-wrap"><span class="vh-model-api-card__badge">Model</span></p>

### `StyleTTS2ForTextToSpeech`

<p class="vh-model-api-card__source-wrap"><a class="vh-model-api-card__source" href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/styletts2/modeling_styletts2.py">View source</a></p>
<div class="vh-model-api-card__signature" markdown>

```text
AutoModelForTextToSpeech.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='styletts2',
    config=None,
    **model_kwargs,
)
```

</div>
<h4>Parameters</h4>
<div class="vh-model-api-card__parameters" markdown>
- `pretrained_model_name_or_path` — Hub ID or compatible local directory.
- `model_type` — Canonical model type; use 'styletts2'.
- `config` — Optional preloaded StyleTTS2Config instance.
- `**model_kwargs` — Model-specific loading arguments.
</div>
</section>

```python
from voicehub import get_model_spec

spec = get_model_spec('styletts2')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('styletts2')` |
| Load and run | `AutoModelForTextToSpeech` |
| Configure | `StyleTTS2Config` |
| Process | `AutoProcessor` |
| Model implementation | `StyleTTS2ForTextToSpeech` |
| Normalized output | `TTSOutput` |
| Training contract | `get_training_spec('styletts2')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).

</div>

</div>

</div>
