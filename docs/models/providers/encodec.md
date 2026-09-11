---
description: Public API, checkpoint, training, and optimization guide for the encodec integration.
hide:
  - toc
---

<div class="vh-model-detail" data-vh-model-detail data-model-type="encodec" data-task="audio-codec" data-training="inference-only" data-parameter-count="" markdown>

<header class="vh-model-detail__hero" data-vh-model-hero markdown>

<p class="vh-model-detail__namespace" aria-label="Model repository"><span class="vh-model-detail__owner-avatar" aria-hidden="true">VH</span><a href="https://github.com/kadirnar/voicehub">VoiceHub</a><span aria-hidden="true">/</span><strong>encodec</strong></p>

# Encodec {.vh-model-title}

<p class="vh-model-detail__summary">Encode audio while preserving segmented scales and channel information.</p>
<div class="vh-model-detail__tags" aria-label="Model metadata"><span class="vh-model-detail__chip" data-chip-kind="task">Audio codecs</span><span class="vh-model-detail__chip" data-chip-kind="runtime">VoiceHub-native</span><span class="vh-model-detail__chip" data-chip-kind="architecture">encodec</span><span class="vh-model-detail__chip" data-chip-kind="parameters" aria-describedby="vh-model-parameters-note-encodec">Parameters: Not reported</span><span class="vh-model-detail__chip" data-chip-kind="language">Not text-language conditioned</span><span class="vh-model-detail__chip" data-chip-kind="training">Training: inference-only</span><span class="vh-model-detail__chip" data-chip-kind="license">License: Checkpoint-specific</span></div>
<p class="vh-model-detail__parameter-note" id="vh-model-parameters-note-encodec"><strong>Parameter metadata:</strong> Not reported: the audited metadata available for the registered default does not provide an exact parameter total.</p>
<div class="vh-model-detail__actions" aria-label="Model actions">
<a class="vh-model-detail__action vh-model-detail__action--primary" href="#usage" data-vh-model-action="use">Use this model</a>
<button class="vh-model-detail__action vh-model-detail__copy" type="button" data-vh-copy-model-id data-model-id="encodec_24khz" aria-describedby="vh-model-checkpoint-encodec"><span data-vh-copy-model-id-label>Copy model ID</span></button>
<details class="vh-model-detail__resources">
<summary class="vh-model-detail__action">Resources</summary>
<div class="vh-model-detail__resource-menu">
<a href="https://arxiv.org/abs/2210.13438" data-vh-model-action="paper">Paper</a>
<a href="https://github.com/facebookresearch/encodec" data-vh-model-action="github">Upstream GitHub</a>
<a href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/encodec/modeling.py" data-vh-model-action="source">VoiceHub source</a>
</div>
</details>
</div>
</header>

<nav class="vh-model-detail__tabs" aria-label="Model sections"><a href="#usage" data-vh-model-tab="usage">Usage</a><a href="#overview" data-vh-model-tab="model-card" aria-current="location">Model card</a><a href="#paper-and-github" data-vh-model-tab="sources">Sources</a><a href="#training-and-optimization" data-vh-model-tab="training">Training</a><a href="#checkpoints-provenance-license-and-limitations" data-vh-model-tab="checkpoint">Checkpoint</a><a href="#public-api" data-vh-model-tab="api">Public API</a></nav>

<div class="vh-model-detail__layout" markdown>

<aside class="vh-model-detail__sidebar" data-vh-model-facts aria-labelledby="vh-model-facts-title-encodec"><h2 id="vh-model-facts-title-encodec">Model facts</h2><details class="vh-model-detail__facts-disclosure" data-vh-model-facts-disclosure aria-labelledby="vh-model-facts-title-encodec" open><summary><span>Toggle model facts</span></summary><dl class="vh-model-detail__facts"><div><dt>Task</dt><dd>Audio codecs</dd></div><div><dt>Parameters</dt><dd aria-describedby="vh-model-parameters-note-encodec">Not reported</dd></div><div><dt>Architecture</dt><dd><code>encodec</code></dd></div><div><dt>Runtime</dt><dd>VoiceHub-native</dd></div><div><dt>Languages</dt><dd>Audio codecs encode waveforms without selecting a text language.</dd></div><div><dt>Capabilities</dt><dd><details class="vh-model-detail__capabilities"><summary>7 capabilities</summary><span><code>audio-codec</code> <code>voicehub-native</code> <code>encode</code> <code>decode</code> <code>stereo</code> <code>segmented-audio</code> <code>safetensors</code></span></details></dd></div><div><dt>Training</dt><dd><code>inference-only</code></dd></div><div><dt>License</dt><dd>Checkpoint-specific</dd></div><div><dt>Default checkpoint</dt><dd id="vh-model-checkpoint-encodec"><code>encodec_24khz</code></dd></div></dl></details></aside>

<div class="vh-model-detail__main vh-model-detail__content" markdown>

## Usage

Complete the [VoiceHub installation](../../getting-started/installation.md) once,
then run this repository-authored example. Model pages intentionally contain no
package-install command.

This example is maintained against VoiceHub's public API; it is not copied from an upstream demo or package README.

**Model-specific path:** Encode audio while preserving segmented scales and channel information.

**Inputs and controls:** Load a native Safetensors export. The pinned official .th import requires explicit trust_official_pickle configuration.

```python
from voicehub import AutoModelForAudioCodec

model = AutoModelForAudioCodec.from_pretrained(
    'checkpoints/encodec/model.safetensors', model_type='encodec', device="cpu",
)
encoded = model.encode("speech.wav")
reconstructed = model.decode(encoded)
print(encoded.length, reconstructed.sample_rate, reconstructed.audio.shape)
```

Use authorized recordings. Verify hardware needs and pin a revision in production.

## Overview

`encodec` is a VoiceHub **audio codecs**
integration. This page is generated from its registry contract.

| Property | Value |
| --- | --- |
| Task | Audio codecs |
| Architecture | `encodec` |
| Runtime | `VoiceHub-native` |
| Languages | Not text-language conditioned |
| Capabilities | `audio-codec`, `voicehub-native`, `encode`, `decode`, `stereo`, `segmented-audio`, `safetensors` |
| Reusable components | — |
| Normalized output | `CodecOutput` |

### Language support

Audio codecs encode waveforms without selecting a text language.

## Paper and GitHub

- **Paper:** [High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438)
- **Upstream GitHub:** [Encodec](https://github.com/facebookresearch/encodec)
- **VoiceHub source:** [VoiceHub model implementation](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/encodec/modeling.py)

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('encodec')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `encodec` |
| Configuration class | `EncodecCodecConfig` |
| Architecture class | `EncodecForAudioCodec` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'encodec_24khz',
    model_type='encodec',
)
print(type(processor).__name__)
```

## Inference

The Usage example returns `CodecOutput` through `AutoModelForAudioCodec`.

### Input and output contract

| Property | Value |
| --- | --- |
| Label boundary | No verified training dataset contract |
| Required training inputs | — |

Use authorized audio and preserve annotation provenance. See the
[ASR and VAD data workflow](../../guides/speech-data.md).

## Training and optimization

Use `available_optimization_passes()` to discover reversible public passes.
Unsupported runtime or hardware fails closed before mutation.

### Training contract

| Property | Value |
| --- | --- |
| Support | `inference-only` |
| Family | `audio-codec` |
| Recipe | `single-phase` |
| Default phase | `default` |
| Training checkpoint | `encodec_24khz` |
| Native training graph | `no` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `default` | objective | — | — | `loss`, `total_loss` |

This integration is **inference-only**. Choose a verified model from the
[training matrix](../training-support.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | `encodec_24khz` |
| Hugging Face ID | Not published / not applicable<br>Use a VoiceHub native Safetensors export or an explicitly trusted pinned Meta release; the codec loader does not interpret Hugging Face Transformers checkpoints. |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.encodec.modeling.EncodecForAudioCodec` |
| Configuration | `voicehub.models.encodec.configuration.EncodecCodecConfig` |
| Source provenance | `voicehub/components/audio/codecs/encodec/SOURCE.json` |
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

### `EncodecCodecConfig`

<p class="vh-model-api-card__source-wrap"><a class="vh-model-api-card__source" href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/encodec/configuration.py">View source</a></p>
<div class="vh-model-api-card__signature" markdown>

```text
EncodecCodecConfig(**config_kwargs)
```

</div>
<h4>Parameters</h4>
<div class="vh-model-api-card__parameters" markdown>
- `**config_kwargs` — Configuration fields validated by EncodecCodecConfig.
</div>
</section>

<section class="vh-model-api-card" data-vh-model-api-card="model" markdown>
<p class="vh-model-api-card__badge-wrap"><span class="vh-model-api-card__badge">Model</span></p>

### `EncodecForAudioCodec`

<p class="vh-model-api-card__source-wrap"><a class="vh-model-api-card__source" href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/encodec/modeling.py">View source</a></p>
<div class="vh-model-api-card__signature" markdown>

```text
AutoModelForAudioCodec.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='encodec',
    config=None,
    **model_kwargs,
)
```

</div>
<h4>Parameters</h4>
<div class="vh-model-api-card__parameters" markdown>
- `pretrained_model_name_or_path` — Hub ID or compatible local directory.
- `model_type` — Canonical model type; use 'encodec'.
- `config` — Optional preloaded EncodecCodecConfig instance.
- `**model_kwargs` — Model-specific loading arguments.
</div>
</section>

```python
from voicehub import get_model_spec

spec = get_model_spec('encodec')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('encodec')` |
| Load and run | `AutoModelForAudioCodec` |
| Configure | `EncodecCodecConfig` |
| Process | `AutoProcessor` |
| Model implementation | `EncodecForAudioCodec` |
| Normalized output | `CodecOutput` |
| Training contract | `get_training_spec('encodec')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).

</div>

</div>

</div>
