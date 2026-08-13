---
description: Public API, checkpoint, training, and optimization guide for the vibevoice integration.
hide:
  - toc
---

<div class="vh-model-detail" data-vh-model-detail data-model-type="vibevoice" data-task="text-to-speech" data-training="preprocessed" data-parameter-count="1017626724" markdown>

<header class="vh-model-detail__hero" data-vh-model-hero markdown>

<p class="vh-model-detail__namespace" aria-label="Model repository"><span class="vh-model-detail__owner-avatar" aria-hidden="true">MI</span><a href="https://huggingface.co/microsoft">microsoft</a><span aria-hidden="true">/</span><strong>VibeVoice-Realtime-0.5B</strong></p>

# VibeVoice {.vh-model-title}

<p class="vh-model-detail__summary">Loads the audited VibeVoice realtime stages without claiming an unverified text-to-waveform loop.</p>
<div class="vh-model-detail__tags" aria-label="Model metadata"><span class="vh-model-detail__chip" data-chip-kind="task">Text to speech</span><span class="vh-model-detail__chip" data-chip-kind="runtime">VoiceHub-native</span><span class="vh-model-detail__chip" data-chip-kind="architecture">vibevoice-tts</span><span class="vh-model-detail__chip" data-chip-kind="parameters" aria-describedby="vh-model-parameters-note-vibevoice">Parameters: 1B</span><span class="vh-model-detail__chip" data-chip-kind="language">Language: en</span><span class="vh-model-detail__chip" data-chip-kind="training">Training: preprocessed</span><span class="vh-model-detail__chip" data-chip-kind="license">License: Checkpoint-specific</span></div>
<p class="vh-model-detail__parameter-note" id="vh-model-parameters-note-vibevoice"><strong>Parameter metadata:</strong> Exact serialized tensor-element total from VoiceHub&#x27;s audited native primary checkpoint; a distinct learned-parameter total is not available.</p>
<div class="vh-model-detail__actions" aria-label="Model actions">
<a class="vh-model-detail__action vh-model-detail__action--primary" href="#usage" data-vh-model-action="use">Use this model</a>
<button class="vh-model-detail__action vh-model-detail__copy" type="button" data-vh-copy-model-id data-model-id="microsoft/VibeVoice-Realtime-0.5B" aria-describedby="vh-model-checkpoint-vibevoice"><span data-vh-copy-model-id-label>Copy model ID</span></button>
<a class="vh-model-detail__action" href="https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B" data-vh-model-action="checkpoint">Checkpoint</a>
<details class="vh-model-detail__resources">
<summary class="vh-model-detail__action">Resources</summary>
<div class="vh-model-detail__resource-menu">
<a href="https://arxiv.org/abs/2508.19205" data-vh-model-action="paper">Paper</a>
<a href="https://github.com/microsoft/VibeVoice" data-vh-model-action="github">Upstream GitHub</a>
<a href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/vibevoice/modeling_vibevoice.py" data-vh-model-action="source">VoiceHub source</a>
<a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/vibevoice.ipynb" data-vh-model-action="colab">Open in Colab</a>
</div>
</details>
</div>
</header>

<nav class="vh-model-detail__tabs" aria-label="Model sections"><a href="#usage" data-vh-model-tab="usage">Usage</a><a href="#overview" data-vh-model-tab="model-card" aria-current="location">Model card</a><a href="#paper-and-github" data-vh-model-tab="sources">Sources</a><a href="#training-and-optimization" data-vh-model-tab="training">Training</a><a href="#checkpoints-provenance-license-and-limitations" data-vh-model-tab="checkpoint">Checkpoint</a><a href="#public-api" data-vh-model-tab="api">Public API</a></nav>

<div class="vh-model-detail__layout" markdown>

<aside class="vh-model-detail__sidebar" data-vh-model-facts aria-labelledby="vh-model-facts-title-vibevoice"><h2 id="vh-model-facts-title-vibevoice">Model facts</h2><details class="vh-model-detail__facts-disclosure" data-vh-model-facts-disclosure aria-labelledby="vh-model-facts-title-vibevoice" open><summary><span>Toggle model facts</span></summary><dl class="vh-model-detail__facts"><div><dt>Task</dt><dd>Text to speech</dd></div><div><dt>Parameters</dt><dd aria-describedby="vh-model-parameters-note-vibevoice">1B</dd></div><div><dt>Architecture</dt><dd><code>vibevoice-tts</code></dd></div><div><dt>Runtime</dt><dd>VoiceHub-native</dd></div><div><dt>Languages</dt><dd><code>en</code></dd></div><div><dt>Capabilities</dt><dd><details class="vh-model-detail__capabilities"><summary>10 capabilities</summary><span><code>text-to-speech</code> <code>voice-prompt</code> <code>fine-tuning</code> <code>default-checkpoint-inference-only</code> <code>safetensors</code> <code>voicehub-native</code> <code>native-runtime</code> <code>preprocessed-training</code> <code>verified-low-level-realtime-stages</code> <code>high-level-generation-fails-closed</code></span></details></dd></div><div><dt>Training</dt><dd><code>preprocessed</code></dd></div><div><dt>License</dt><dd>Checkpoint-specific</dd></div><div><dt>Default checkpoint</dt><dd id="vh-model-checkpoint-vibevoice"><a href="https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B"><code>microsoft/VibeVoice-Realtime-0.5B</code></a></dd></div></dl></details></aside>

<div class="vh-model-detail__main vh-model-detail__content" markdown>

## Usage

Complete the [VoiceHub installation](../../getting-started/installation.md) once,
then run this repository-authored example. Model pages intentionally contain no
package-install command.

This example is maintained against VoiceHub's public API; it is not copied from an upstream demo or package README.

**Model-specific path:** Loads the audited VibeVoice realtime stages without claiming an unverified text-to-waveform loop.

**Inputs and controls:** High-level cached-prompt synthesis intentionally fails closed until cache serialization, chunk boundaries, and waveform parity are verified.

```python
from voicehub import AutoModelForTextToSpeech

model = AutoModelForTextToSpeech.from_pretrained(
    'microsoft/VibeVoice-Realtime-0.5B',
    model_type='vibevoice',
    device="cuda",
    lazy_load=True,
)
model.load()
required_stages = (
    "forward_lm",
    "forward_tts_lm",
    "sample_speech_latents",
    "decode_speech_latents",
)
missing = [name for name in required_stages if not hasattr(model.model, name)]
if missing:
    raise RuntimeError(f"Missing audited VibeVoice stage(s): {', '.join(missing)}")
print("High-level synthesis is not verified; available native stages:", required_stages)
```

Use authorized recordings. Verify hardware needs and pin a revision in production.

## Overview

`vibevoice` is a VoiceHub **text to speech**
integration. This page is generated from its registry contract. [Open the `vibevoice` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/vibevoice.ipynb).

| Property | Value |
| --- | --- |
| Task | Text to speech |
| Architecture | `vibevoice-tts` |
| Runtime | `VoiceHub-native` |
| Languages | `en` |
| Capabilities | `text-to-speech`, `voice-prompt`, `fine-tuning`, `default-checkpoint-inference-only`, `safetensors`, `voicehub-native`, `native-runtime`, `preprocessed-training`, `verified-low-level-realtime-stages`, `high-level-generation-fails-closed` |
| Reusable components | — |
| Normalized output | `TTSOutput` |

### Language support

<details class="vh-language-support" markdown>
<summary>Supported language abbreviations</summary>

`en`

</details>

## Paper and GitHub

- **Paper:** [VibeVoice Technical Report](https://arxiv.org/abs/2508.19205)
- **Upstream GitHub:** [VibeVoice](https://github.com/microsoft/VibeVoice)
- **VoiceHub source:** [VoiceHub model implementation](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/vibevoice/modeling_vibevoice.py)

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('vibevoice')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `vibevoice` |
| Configuration class | `VibeVoiceConfig` |
| Architecture class | `VibeVoiceForTextToSpeech` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'microsoft/VibeVoice-Realtime-0.5B',
    model_type='vibevoice',
)
print(type(processor).__name__)
```

## Inference

The Usage example returns `TTSOutput` through `AutoModelForTextToSpeech`.

### Input and output contract

| Property | Value |
| --- | --- |
| Readiness | `preprocessed` |
| Data architecture | `hybrid` |
| Sample rate | 24,000 Hz |
| Contract getter | `get_tts_dataset_spec('vibevoice')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `lm-diffusion-batch` | `input_ids`, `attention_mask`, `speech_tensors`, `speech_masks`, `speeches_loss_input`, `speech_semantic_tensors`, `acoustic_input_mask`, `acoustic_loss_mask` | — | Prepared | — |

Multi-component language-model, diffusion, acoustic, or GAN data. See the [data workflow](../../guides/data-preparation.md).

## Training and optimization

Use `available_optimization_passes()` to discover reversible public passes.
Unsupported runtime or hardware fails closed before mutation.

### Training contract

| Property | Value |
| --- | --- |
| Support | `preprocessed` |
| Family | `composite` |
| Recipe | `single-phase` |
| Default phase | `lm_diffusion` |
| Training checkpoint | `microsoft/VibeVoice-1.5B` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `lm_diffusion` | objective | `model` | `input_ids`, `attention_mask`, `speech_tensors`, `speech_masks`, `speeches_loss_input`, `speech_semantic_tensors`, `acoustic_input_mask`, `acoustic_loss_mask` | `loss`, `ce_loss`, `diffusion_loss` |

Prepare the exact tensors listed in the data contract before this step. Call `model.validate_training_support()` first, then follow the
[training workflow](../../guides/training.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | [`microsoft/VibeVoice-Realtime-0.5B`](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B) |
| Hugging Face ID | [`microsoft/VibeVoice-Realtime-0.5B`](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B)<br>Repository availability verified through the Hugging Face model API on 2026-08-11; pin a revision before production use. |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.vibevoice.modeling_vibevoice.VibeVoiceForTextToSpeech` |
| Configuration | `voicehub.models.vibevoice.configuration_vibevoice.VibeVoiceConfig` |
| Source provenance | `voicehub/models/vibevoice/source/SOURCE.json` |
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

### `VibeVoiceConfig`

<p class="vh-model-api-card__source-wrap"><a class="vh-model-api-card__source" href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/vibevoice/configuration_vibevoice.py">View source</a></p>
<div class="vh-model-api-card__signature" markdown>

```text
VibeVoiceConfig(**config_kwargs)
```

</div>
<h4>Parameters</h4>
<div class="vh-model-api-card__parameters" markdown>
- `**config_kwargs` — Configuration fields validated by VibeVoiceConfig.
</div>
</section>

<section class="vh-model-api-card" data-vh-model-api-card="model" markdown>
<p class="vh-model-api-card__badge-wrap"><span class="vh-model-api-card__badge">Model</span></p>

### `VibeVoiceForTextToSpeech`

<p class="vh-model-api-card__source-wrap"><a class="vh-model-api-card__source" href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/vibevoice/modeling_vibevoice.py">View source</a></p>
<div class="vh-model-api-card__signature" markdown>

```text
AutoModelForTextToSpeech.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='vibevoice',
    config=None,
    **model_kwargs,
)
```

</div>
<h4>Parameters</h4>
<div class="vh-model-api-card__parameters" markdown>
- `pretrained_model_name_or_path` — Hub ID or compatible local directory.
- `model_type` — Canonical model type; use 'vibevoice'.
- `config` — Optional preloaded VibeVoiceConfig instance.
- `**model_kwargs` — Model-specific loading arguments.
</div>
</section>

```python
from voicehub import get_model_spec

spec = get_model_spec('vibevoice')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('vibevoice')` |
| Load and run | `AutoModelForTextToSpeech` |
| Configure | `VibeVoiceConfig` |
| Process | `AutoProcessor` |
| Model implementation | `VibeVoiceForTextToSpeech` |
| Normalized output | `TTSOutput` |
| Training contract | `get_training_spec('vibevoice')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).

</div>

</div>

</div>
