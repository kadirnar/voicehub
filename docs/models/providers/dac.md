---
description: Public API, checkpoint, training, and optimization guide for the dac integration.
hide:
  - toc
---

<div class="vh-model-detail" data-vh-model-detail data-model-type="dac" data-task="audio-codec" data-training="inference-only" data-parameter-count="" markdown>

<header class="vh-model-detail__hero" data-vh-model-hero markdown>

<p class="vh-model-detail__namespace" aria-label="Model repository"><span class="vh-model-detail__owner-avatar" aria-hidden="true">DE</span><a href="https://huggingface.co/descript">descript</a><span aria-hidden="true">/</span><strong>dac_44khz</strong></p>

# Dac {.vh-model-title}

<p class="vh-model-detail__summary">Encode mono audio into discrete codebooks and reconstruct its original sample length.</p>
<div class="vh-model-detail__tags" aria-label="Model metadata"><span class="vh-model-detail__chip" data-chip-kind="task">Audio codecs</span><span class="vh-model-detail__chip" data-chip-kind="runtime">VoiceHub-native</span><span class="vh-model-detail__chip" data-chip-kind="architecture">dac</span><span class="vh-model-detail__chip" data-chip-kind="parameters" aria-describedby="vh-model-parameters-note-dac">Parameters: Not reported</span><span class="vh-model-detail__chip" data-chip-kind="language">Not text-language conditioned</span><span class="vh-model-detail__chip" data-chip-kind="training">Training: inference-only</span><span class="vh-model-detail__chip" data-chip-kind="license">License: Checkpoint-specific</span></div>
<p class="vh-model-detail__parameter-note" id="vh-model-parameters-note-dac"><strong>Parameter metadata:</strong> Not reported: the audited metadata available for the registered default does not provide an exact parameter total.</p>
<div class="vh-model-detail__actions" aria-label="Model actions">
<a class="vh-model-detail__action vh-model-detail__action--primary" href="#usage" data-vh-model-action="use">Use this model</a>
<button class="vh-model-detail__action vh-model-detail__copy" type="button" data-vh-copy-model-id data-model-id="descript/dac_44khz" aria-describedby="vh-model-checkpoint-dac"><span data-vh-copy-model-id-label>Copy model ID</span></button>
<a class="vh-model-detail__action" href="https://huggingface.co/descript/dac_44khz" data-vh-model-action="checkpoint">Checkpoint</a>
<details class="vh-model-detail__resources">
<summary class="vh-model-detail__action">Resources</summary>
<div class="vh-model-detail__resource-menu">
<a href="https://arxiv.org/abs/2306.06546" data-vh-model-action="paper">Paper</a>
<a href="https://github.com/descriptinc/descript-audio-codec" data-vh-model-action="github">Upstream GitHub</a>
<a href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/dac/modeling.py" data-vh-model-action="source">VoiceHub source</a>
<a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/dac.ipynb" data-vh-model-action="colab">Open in Colab</a>
</div>
</details>
</div>
</header>

<nav class="vh-model-detail__tabs" aria-label="Model sections"><a href="#usage" data-vh-model-tab="usage">Usage</a><a href="#overview" data-vh-model-tab="model-card" aria-current="location">Model card</a><a href="#paper-and-github" data-vh-model-tab="sources">Sources</a><a href="#training-and-optimization" data-vh-model-tab="training">Training</a><a href="#checkpoints-provenance-license-and-limitations" data-vh-model-tab="checkpoint">Checkpoint</a><a href="#public-api" data-vh-model-tab="api">Public API</a></nav>

<div class="vh-model-detail__layout" markdown>

<aside class="vh-model-detail__sidebar" data-vh-model-facts aria-labelledby="vh-model-facts-title-dac"><h2 id="vh-model-facts-title-dac">Model facts</h2><details class="vh-model-detail__facts-disclosure" data-vh-model-facts-disclosure aria-labelledby="vh-model-facts-title-dac" open><summary><span>Toggle model facts</span></summary><dl class="vh-model-detail__facts"><div><dt>Task</dt><dd>Audio codecs</dd></div><div><dt>Parameters</dt><dd aria-describedby="vh-model-parameters-note-dac">Not reported</dd></div><div><dt>Architecture</dt><dd><code>dac</code></dd></div><div><dt>Runtime</dt><dd>VoiceHub-native</dd></div><div><dt>Languages</dt><dd>Audio codecs encode waveforms without selecting a text language.</dd></div><div><dt>Capabilities</dt><dd><details class="vh-model-detail__capabilities"><summary>5 capabilities</summary><span><code>audio-codec</code> <code>voicehub-native</code> <code>encode</code> <code>decode</code> <code>safetensors</code></span></details></dd></div><div><dt>Training</dt><dd><code>inference-only</code></dd></div><div><dt>License</dt><dd>Checkpoint-specific</dd></div><div><dt>Default checkpoint</dt><dd id="vh-model-checkpoint-dac"><a href="https://huggingface.co/descript/dac_44khz"><code>descript/dac_44khz</code></a></dd></div></dl></details></aside>

<div class="vh-model-detail__main vh-model-detail__content" markdown>

## Usage

Complete the [VoiceHub installation](../../getting-started/installation.md) once,
then run this repository-authored example. Model pages intentionally contain no
package-install command.

This example is maintained against VoiceHub's public API; it is not copied from an upstream demo or package README.

**Model-specific path:** Encode mono audio into discrete codebooks and reconstruct its original sample length.

**Inputs and controls:** Use a PCM WAVE path or a tensor with an explicit sampling_rate. Output codes retain the original length.

```python
from voicehub import AutoModelForAudioCodec

model = AutoModelForAudioCodec.from_pretrained(
    'descript/dac_44khz', model_type='dac', device="cpu",
)
encoded = model.encode("speech.wav")
reconstructed = model.decode(encoded)
print(encoded.length, reconstructed.sample_rate, reconstructed.audio.shape)
```

Use authorized recordings. Verify hardware needs and pin a revision in production.

## Overview

`dac` is a VoiceHub **audio codecs**
integration. This page is generated from its registry contract. [Open the `dac` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/dac.ipynb).

| Property | Value |
| --- | --- |
| Task | Audio codecs |
| Architecture | `dac` |
| Runtime | `VoiceHub-native` |
| Languages | Not text-language conditioned |
| Capabilities | `audio-codec`, `voicehub-native`, `encode`, `decode`, `safetensors` |
| Reusable components | — |
| Normalized output | `CodecOutput` |

### Language support

Audio codecs encode waveforms without selecting a text language.

## Paper and GitHub

- **Paper:** [High-Fidelity Audio Compression with Improved RVQGAN](https://arxiv.org/abs/2306.06546)
- **Upstream GitHub:** [Descript Audio Codec](https://github.com/descriptinc/descript-audio-codec)
- **VoiceHub source:** [VoiceHub model implementation](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/dac/modeling.py)

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('dac')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `dac` |
| Configuration class | `DacCodecConfig` |
| Architecture class | `DacForAudioCodec` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'descript/dac_44khz',
    model_type='dac',
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
| Training checkpoint | `descript/dac_44khz` |
| Native training graph | `no` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `default` | objective | — | — | `loss`, `total_loss` |

This integration is **inference-only**. Choose a verified model from the
[training matrix](../training-support.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | [`descript/dac_44khz`](https://huggingface.co/descript/dac_44khz) |
| Hugging Face ID | [`descript/dac_44khz`](https://huggingface.co/descript/dac_44khz)<br>Repository availability verified through the Hugging Face model API on 2026-08-11; pin a revision before production use. |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.dac.modeling.DacForAudioCodec` |
| Configuration | `voicehub.models.dac.configuration.DacCodecConfig` |
| Source provenance | No integration-specific bundled `SOURCE.json` is declared for this registry entry. |
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

### `DacCodecConfig`

<p class="vh-model-api-card__source-wrap"><a class="vh-model-api-card__source" href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/dac/configuration.py">View source</a></p>
<div class="vh-model-api-card__signature" markdown>

```text
DacCodecConfig(**config_kwargs)
```

</div>
<h4>Parameters</h4>
<div class="vh-model-api-card__parameters" markdown>
- `**config_kwargs` — Configuration fields validated by DacCodecConfig.
</div>
</section>

<section class="vh-model-api-card" data-vh-model-api-card="model" markdown>
<p class="vh-model-api-card__badge-wrap"><span class="vh-model-api-card__badge">Model</span></p>

### `DacForAudioCodec`

<p class="vh-model-api-card__source-wrap"><a class="vh-model-api-card__source" href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/dac/modeling.py">View source</a></p>
<div class="vh-model-api-card__signature" markdown>

```text
AutoModelForAudioCodec.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='dac',
    config=None,
    **model_kwargs,
)
```

</div>
<h4>Parameters</h4>
<div class="vh-model-api-card__parameters" markdown>
- `pretrained_model_name_or_path` — Hub ID or compatible local directory.
- `model_type` — Canonical model type; use 'dac'.
- `config` — Optional preloaded DacCodecConfig instance.
- `**model_kwargs` — Model-specific loading arguments.
</div>
</section>

```python
from voicehub import get_model_spec

spec = get_model_spec('dac')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('dac')` |
| Load and run | `AutoModelForAudioCodec` |
| Configure | `DacCodecConfig` |
| Process | `AutoProcessor` |
| Model implementation | `DacForAudioCodec` |
| Normalized output | `CodecOutput` |
| Training contract | `get_training_spec('dac')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).

</div>

</div>

</div>
