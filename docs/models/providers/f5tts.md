---
description: Public API, checkpoint, training, and optimization guide for the f5tts integration.
hide:
  - toc
---

<div class="vh-model-detail" data-vh-model-detail data-model-type="f5tts" data-task="text-to-speech" data-training="preprocessed" data-parameter-count="337096804" markdown>

<header class="vh-model-detail__hero" data-vh-model-hero markdown>

<p class="vh-model-detail__namespace" aria-label="Model repository"><span class="vh-model-detail__owner-avatar" aria-hidden="true">VH</span><a href="https://github.com/kadirnar/voicehub">VoiceHub</a><span aria-hidden="true">/</span><strong>f5tts</strong></p>

# F5TTS {.vh-model-title}

<p class="vh-model-detail__summary">Supplies F5-TTS with the mandatory reference waveform and matching transcript.</p>
<div class="vh-model-detail__tags" aria-label="Model metadata"><span class="vh-model-detail__chip" data-chip-kind="task">Text to speech</span><span class="vh-model-detail__chip" data-chip-kind="runtime">VoiceHub-native</span><span class="vh-model-detail__chip" data-chip-kind="architecture">f5tts</span><span class="vh-model-detail__chip" data-chip-kind="parameters" aria-describedby="vh-model-parameters-note-f5tts">Parameters: 337.1M</span><span class="vh-model-detail__chip" data-chip-kind="language">Languages: en, zh</span><span class="vh-model-detail__chip" data-chip-kind="training">Training: preprocessed</span><span class="vh-model-detail__chip" data-chip-kind="license">License: Checkpoint-specific</span></div>
<p class="vh-model-detail__parameter-note" id="vh-model-parameters-note-f5tts"><strong>Parameter metadata:</strong> Exact learned-parameter total for VoiceHub&#x27;s audited native primary graph at the registered default selection; separately loaded auxiliary models are excluded.</p>
<div class="vh-model-detail__actions" aria-label="Model actions">
<a class="vh-model-detail__action vh-model-detail__action--primary" href="#usage" data-vh-model-action="use">Use this model</a>
<button class="vh-model-detail__action vh-model-detail__copy" type="button" data-vh-copy-model-id data-model-id="F5TTS_v1_Base" aria-describedby="vh-model-checkpoint-f5tts"><span data-vh-copy-model-id-label>Copy model ID</span></button>
<details class="vh-model-detail__resources">
<summary class="vh-model-detail__action">Resources</summary>
<div class="vh-model-detail__resource-menu">
<a href="https://arxiv.org/abs/2410.06885" data-vh-model-action="paper">Paper</a>
<a href="https://github.com/SWivid/F5-TTS" data-vh-model-action="github">Upstream GitHub</a>
<a href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/f5tts/modeling_f5tts.py" data-vh-model-action="source">VoiceHub source</a>
</div>
</details>
</div>
</header>

<nav class="vh-model-detail__tabs" aria-label="Model sections"><a href="#usage" data-vh-model-tab="usage">Usage</a><a href="#overview" data-vh-model-tab="model-card" aria-current="location">Model card</a><a href="#paper-and-github" data-vh-model-tab="sources">Sources</a><a href="#training-and-optimization" data-vh-model-tab="training">Training</a><a href="#checkpoints-provenance-license-and-limitations" data-vh-model-tab="checkpoint">Checkpoint</a><a href="#public-api" data-vh-model-tab="api">Public API</a></nav>

<div class="vh-model-detail__layout" markdown>

<aside class="vh-model-detail__sidebar" data-vh-model-facts aria-labelledby="vh-model-facts-title-f5tts"><h2 id="vh-model-facts-title-f5tts">Model facts</h2><details class="vh-model-detail__facts-disclosure" data-vh-model-facts-disclosure aria-labelledby="vh-model-facts-title-f5tts" open><summary><span>Toggle model facts</span></summary><dl class="vh-model-detail__facts"><div><dt>Task</dt><dd>Text to speech</dd></div><div><dt>Parameters</dt><dd aria-describedby="vh-model-parameters-note-f5tts">337.1M</dd></div><div><dt>Architecture</dt><dd><code>f5tts</code></dd></div><div><dt>Runtime</dt><dd>VoiceHub-native</dd></div><div><dt>Languages</dt><dd><code>en</code> <code>zh</code></dd></div><div><dt>Capabilities</dt><dd><details class="vh-model-detail__capabilities"><summary>7 capabilities</summary><span><code>text-to-speech</code> <code>voice-cloning</code> <code>fine-tuning</code> <code>flow-matching</code> <code>safetensors</code> <code>voicehub-native</code> <code>native-runtime</code></span></details></dd></div><div><dt>Training</dt><dd><code>preprocessed</code></dd></div><div><dt>License</dt><dd>Checkpoint-specific</dd></div><div><dt>Default checkpoint</dt><dd id="vh-model-checkpoint-f5tts"><code>F5TTS_v1_Base</code></dd></div></dl></details></aside>

<div class="vh-model-detail__main vh-model-detail__content" markdown>

## Usage

Complete the [VoiceHub installation](../../getting-started/installation.md) once,
then run this repository-authored example. Model pages intentionally contain no
package-install command.

This example is maintained against VoiceHub's public API; it is not copied from an upstream demo or package README.

**Model-specific path:** Supplies F5-TTS with the mandatory reference waveform and matching transcript.

**Inputs and controls:** The transcript must match the reference audio exactly or alignment quality will degrade.

```python
from pathlib import Path

from voicehub import AutoModelForTextToSpeech, TTSGenerationConfig

REFERENCE_AUDIO = Path("reference.wav")
REFERENCE_TEXT = "The reference transcript must exactly match the authorized audio."
if not REFERENCE_AUDIO.is_file():
    raise FileNotFoundError(REFERENCE_AUDIO)

model = AutoModelForTextToSpeech.from_pretrained(
    'F5TTS_v1_Base',
    model_type='f5tts',
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
    nfe_steps=32,
    cfg_strength=2.0,
)
print(output.file_path, output.sample_rate, output.metadata)
```

Use authorized recordings. Verify hardware needs and pin a revision in production.

## Overview

`f5tts` is a VoiceHub **text to speech**
integration. This page is generated from its registry contract.

| Property | Value |
| --- | --- |
| Task | Text to speech |
| Architecture | `f5tts` |
| Runtime | `VoiceHub-native` |
| Languages | `en`, `zh` |
| Capabilities | `text-to-speech`, `voice-cloning`, `fine-tuning`, `flow-matching`, `safetensors`, `voicehub-native`, `native-runtime` |
| Reusable components | `vocos` |
| Normalized output | `TTSOutput` |

### Language support

<details class="vh-language-support" markdown>
<summary>Supported language abbreviations</summary>

`en`, `zh`

</details>

## Paper and GitHub

- **Paper:** [F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech](https://arxiv.org/abs/2410.06885)
- **Upstream GitHub:** [F5-TTS](https://github.com/SWivid/F5-TTS)
- **VoiceHub source:** [VoiceHub model implementation](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/f5tts/modeling_f5tts.py)

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('f5tts')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `f5tts` |
| Configuration class | `F5TTSConfig` |
| Architecture class | `F5TTSForTextToSpeech` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'F5TTS_v1_Base',
    model_type='f5tts',
)
print(type(processor).__name__)
```

## Inference

The Usage example returns `TTSOutput` through `AutoModelForTextToSpeech`.

### Input and output contract

| Property | Value |
| --- | --- |
| Readiness | `preprocessed` |
| Data architecture | `diffusion` |
| Sample rate | 24,000 Hz |
| Contract getter | `get_tts_dataset_spec('f5tts')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `waveform-vocab` | `input_values`, `input_ids` | — | Prepared | — |
| `mel-features` | `input_ids` | mel / mel_spec | Prepared | — |
| `native-ready` | `inp`, `text` | — | Prepared | — |

Conditional flow-matching, rectified-flow, or diffusion data. See the [data workflow](../../guides/data-preparation.md).

## Training and optimization

Use `available_optimization_passes()` to discover reversible public passes.
Unsupported runtime or hardware fails closed before mutation.

### Training contract

| Property | Value |
| --- | --- |
| Support | `preprocessed` |
| Family | `flow-matching` |
| Recipe | `single-phase` |
| Default phase | `flow` |
| Training checkpoint | `F5TTS_v1_Base` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `flow` | objective | `model.ema_model` | `inp`, `text` | `loss` |

Prepare the exact tensors listed in the data contract before this step. Call `model.validate_training_support()` first, then follow the
[training workflow](../../guides/training.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | `F5TTS_v1_Base` |
| Hugging Face ID | [`SWivid/F5-TTS`](https://huggingface.co/SWivid/F5-TTS)<br>Official F5-TTS repository, verified available on 2026-08-11; the registry alias selects the F5TTS_v1_Base files inside it. |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.f5tts.modeling_f5tts.F5TTSForTextToSpeech` |
| Configuration | `voicehub.models.f5tts.configuration_f5tts.F5TTSConfig` |
| Source provenance | `voicehub/models/f5tts/source/SOURCE.json` |
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

### `F5TTSConfig`

<p class="vh-model-api-card__source-wrap"><a class="vh-model-api-card__source" href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/f5tts/configuration_f5tts.py">View source</a></p>
<div class="vh-model-api-card__signature" markdown>

```text
F5TTSConfig(**config_kwargs)
```

</div>
<h4>Parameters</h4>
<div class="vh-model-api-card__parameters" markdown>
- `**config_kwargs` — Configuration fields validated by F5TTSConfig.
</div>
</section>

<section class="vh-model-api-card" data-vh-model-api-card="model" markdown>
<p class="vh-model-api-card__badge-wrap"><span class="vh-model-api-card__badge">Model</span></p>

### `F5TTSForTextToSpeech`

<p class="vh-model-api-card__source-wrap"><a class="vh-model-api-card__source" href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/f5tts/modeling_f5tts.py">View source</a></p>
<div class="vh-model-api-card__signature" markdown>

```text
AutoModelForTextToSpeech.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='f5tts',
    config=None,
    **model_kwargs,
)
```

</div>
<h4>Parameters</h4>
<div class="vh-model-api-card__parameters" markdown>
- `pretrained_model_name_or_path` — Hub ID or compatible local directory.
- `model_type` — Canonical model type; use 'f5tts'.
- `config` — Optional preloaded F5TTSConfig instance.
- `**model_kwargs` — Model-specific loading arguments.
</div>
</section>

```python
from voicehub import get_model_spec

spec = get_model_spec('f5tts')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('f5tts')` |
| Load and run | `AutoModelForTextToSpeech` |
| Configure | `F5TTSConfig` |
| Process | `AutoProcessor` |
| Model implementation | `F5TTSForTextToSpeech` |
| Normalized output | `TTSOutput` |
| Training contract | `get_training_spec('f5tts')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).

</div>

</div>

</div>
