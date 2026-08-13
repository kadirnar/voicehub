---
description: Public API, checkpoint, training, and optimization guide for the vad_pyannote_segmentation integration.
hide:
  - toc
---

<div class="vh-model-detail" data-vh-model-detail data-model-type="vad_pyannote_segmentation" data-task="voice-activity-detection" data-training="native" data-parameter-count="" markdown>

<header class="vh-model-detail__hero" data-vh-model-hero markdown>

<p class="vh-model-detail__namespace" aria-label="Model repository"><span class="vh-model-detail__owner-avatar" aria-hidden="true">PY</span><a href="https://huggingface.co/pyannote">pyannote</a><span aria-hidden="true">/</span><strong>segmentation-3.0</strong></p>

# PyannoteSegmentationVAD {.vh-model-title}

<p class="vh-model-detail__summary">Runs pyannote segmentation 3.0 as VAD with hysteresis and frame evidence enabled.</p>
<div class="vh-model-detail__tags" aria-label="Model metadata"><span class="vh-model-detail__chip" data-chip-kind="task">Voice activity detection</span><span class="vh-model-detail__chip" data-chip-kind="runtime">VoiceHub-native</span><span class="vh-model-detail__chip" data-chip-kind="architecture">pyannet</span><span class="vh-model-detail__chip" data-chip-kind="parameters" aria-describedby="vh-model-parameters-note-vad_pyannote_segmentation">Parameters: Not reported</span><span class="vh-model-detail__chip" data-chip-kind="language">Not text-language conditioned</span><span class="vh-model-detail__chip" data-chip-kind="training">Training: native</span><span class="vh-model-detail__chip" data-chip-kind="license">License: Checkpoint-specific</span></div>
<p class="vh-model-detail__parameter-note" id="vh-model-parameters-note-vad_pyannote_segmentation"><strong>Parameter metadata:</strong> Not reported: the audited metadata available for the registered default does not provide an exact parameter total.</p>
<div class="vh-model-detail__actions" aria-label="Model actions">
<a class="vh-model-detail__action vh-model-detail__action--primary" href="#usage" data-vh-model-action="use">Use this model</a>
<button class="vh-model-detail__action vh-model-detail__copy" type="button" data-vh-copy-model-id data-model-id="pyannote/segmentation-3.0" aria-describedby="vh-model-checkpoint-vad_pyannote_segmentation"><span data-vh-copy-model-id-label>Copy model ID</span></button>
<a class="vh-model-detail__action" href="https://huggingface.co/pyannote/segmentation-3.0" data-vh-model-action="checkpoint">Checkpoint</a>
<details class="vh-model-detail__resources">
<summary class="vh-model-detail__action">Resources</summary>
<div class="vh-model-detail__resource-menu">
<a href="https://arxiv.org/abs/1911.01255" data-vh-model-action="paper">Paper</a>
<a href="https://github.com/pyannote/pyannote-audio" data-vh-model-action="github">Upstream GitHub</a>
<a href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/vad_pyannote_segmentation/modeling_vad_pyannote_segmentation.py" data-vh-model-action="source">VoiceHub source</a>
<a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/vad_pyannote_segmentation.ipynb" data-vh-model-action="colab">Open in Colab</a>
</div>
</details>
</div>
</header>

<nav class="vh-model-detail__tabs" aria-label="Model sections"><a href="#usage" data-vh-model-tab="usage">Usage</a><a href="#overview" data-vh-model-tab="model-card" aria-current="location">Model card</a><a href="#paper-and-github" data-vh-model-tab="sources">Sources</a><a href="#training-and-optimization" data-vh-model-tab="training">Training</a><a href="#checkpoints-provenance-license-and-limitations" data-vh-model-tab="checkpoint">Checkpoint</a><a href="#public-api" data-vh-model-tab="api">Public API</a></nav>

<div class="vh-model-detail__layout" markdown>

<aside class="vh-model-detail__sidebar" data-vh-model-facts aria-labelledby="vh-model-facts-title-vad_pyannote_segmentation"><h2 id="vh-model-facts-title-vad_pyannote_segmentation">Model facts</h2><details class="vh-model-detail__facts-disclosure" data-vh-model-facts-disclosure aria-labelledby="vh-model-facts-title-vad_pyannote_segmentation" open><summary><span>Toggle model facts</span></summary><dl class="vh-model-detail__facts"><div><dt>Task</dt><dd>Voice activity detection</dd></div><div><dt>Parameters</dt><dd aria-describedby="vh-model-parameters-note-vad_pyannote_segmentation">Not reported</dd></div><div><dt>Architecture</dt><dd><code>pyannet</code></dd></div><div><dt>Runtime</dt><dd>VoiceHub-native</dd></div><div><dt>Languages</dt><dd>The public VAD contract does not select a spoken language; validate checkpoint acoustic coverage on the target languages and recording conditions.</dd></div><div><dt>Capabilities</dt><dd><details class="vh-model-detail__capabilities"><summary>8 capabilities</summary><span><code>voice-activity-detection</code> <code>voicehub-native</code> <code>gated-checkpoint</code> <code>trusted-checkpoint-conversion</code> <code>safetensors</code> <code>powerset</code> <code>frame-scores</code> <code>fine-tuning</code></span></details></dd></div><div><dt>Training</dt><dd><code>native</code></dd></div><div><dt>License</dt><dd>Checkpoint-specific</dd></div><div><dt>Default checkpoint</dt><dd id="vh-model-checkpoint-vad_pyannote_segmentation"><a href="https://huggingface.co/pyannote/segmentation-3.0"><code>pyannote/segmentation-3.0</code></a></dd></div></dl></details></aside>

<div class="vh-model-detail__main vh-model-detail__content" markdown>

## Usage

Complete the [VoiceHub installation](../../getting-started/installation.md) once,
then run this repository-authored example. Model pages intentionally contain no
package-install command.

This example is maintained against VoiceHub's public API; it is not copied from an upstream demo or package README.

**Model-specific path:** Runs pyannote segmentation 3.0 as VAD with hysteresis and frame evidence enabled.

**Inputs and controls:** Segmentation scores are converted to speech regions; they are not diarization labels by themselves.

```python
from pathlib import Path

from voicehub import AutoModelForVoiceActivityDetection

AUDIO_FILE = Path("speech.wav")
if not AUDIO_FILE.is_file():
    raise FileNotFoundError(AUDIO_FILE)

model = AutoModelForVoiceActivityDetection.from_pretrained(
    'pyannote/segmentation-3.0',
    model_type='vad_pyannote_segmentation',
    device="cpu",
    lazy_load=True,
)
output = model.detect(
    AUDIO_FILE,
    onset=0.6,
    offset=0.4,
    return_frames=True,
)
for segment in output.segments:
    print(segment.start, segment.end, segment.score)
```

Use authorized recordings. Verify hardware needs and pin a revision in production.

## Overview

`vad_pyannote_segmentation` is a VoiceHub **voice activity detection**
integration. This page is generated from its registry contract. [Open the `vad_pyannote_segmentation` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/vad_pyannote_segmentation.ipynb).

| Property | Value |
| --- | --- |
| Task | Voice activity detection |
| Architecture | `pyannet` |
| Runtime | `VoiceHub-native` |
| Languages | Not text-language conditioned |
| Capabilities | `voice-activity-detection`, `voicehub-native`, `gated-checkpoint`, `trusted-checkpoint-conversion`, `safetensors`, `powerset`, `frame-scores`, `fine-tuning` |
| Reusable components | — |
| Normalized output | `VADOutput` |

### Language support

The public VAD contract does not select a spoken language; validate checkpoint acoustic coverage on the target languages and recording conditions.

## Paper and GitHub

- **Paper:** [pyannote.audio: neural building blocks for speaker diarization](https://arxiv.org/abs/1911.01255)
- **Upstream GitHub:** [pyannote.audio](https://github.com/pyannote/pyannote-audio)
- **VoiceHub source:** [VoiceHub model implementation](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/vad_pyannote_segmentation/modeling_vad_pyannote_segmentation.py)

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('vad_pyannote_segmentation')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `vad_pyannote_segmentation` |
| Configuration class | `PyannoteSegmentationVADConfig` |
| Architecture class | `PyannoteSegmentationVADForVoiceActivityDetection` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'pyannote/segmentation-3.0',
    model_type='vad_pyannote_segmentation',
)
print(type(processor).__name__)
```

## Inference

The Usage example returns `VADOutput` through `AutoModelForVoiceActivityDetection`.

### Input and output contract

| Property | Value |
| --- | --- |
| Label boundary | Clip-, frame-, or segment-level labels |
| Required training inputs | `waveforms`, `labels` |

Use authorized audio and preserve annotation provenance. See the
[ASR and VAD data workflow](../../guides/speech-data.md).

## Training and optimization

Use `available_optimization_passes()` to discover reversible public passes.
Unsupported runtime or hardware fails closed before mutation.

### Training contract

| Property | Value |
| --- | --- |
| Support | `native` |
| Family | `frame-classification` |
| Recipe | `single-phase` |
| Default phase | `powerset_segmentation` |
| Training checkpoint | `pyannote/segmentation-3.0` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `powerset_segmentation` | objective | `model` | `waveforms`, `labels` | `loss` |

The integration accepts its declared source or prepared contract directly. Call `model.validate_training_support()` first, then follow the
[training workflow](../../guides/training.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | [`pyannote/segmentation-3.0`](https://huggingface.co/pyannote/segmentation-3.0) |
| Hugging Face ID | [`pyannote/segmentation-3.0`](https://huggingface.co/pyannote/segmentation-3.0)<br>Repository availability verified through the Hugging Face model API on 2026-08-11; pin a revision before production use. |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cpu`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.vad_pyannote_segmentation.modeling_vad_pyannote_segmentation.PyannoteSegmentationVADForVoiceActivityDetection` |
| Configuration | `voicehub.models.vad_pyannote_segmentation.configuration_vad_pyannote_segmentation.PyannoteSegmentationVADConfig` |
| Source provenance | `voicehub/architectures/pyannet/SOURCE.json` |
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

### `PyannoteSegmentationVADConfig`

<p class="vh-model-api-card__source-wrap"><a class="vh-model-api-card__source" href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/vad_pyannote_segmentation/configuration_vad_pyannote_segmentation.py">View source</a></p>
<div class="vh-model-api-card__signature" markdown>

```text
PyannoteSegmentationVADConfig(**config_kwargs)
```

</div>
<h4>Parameters</h4>
<div class="vh-model-api-card__parameters" markdown>
- `**config_kwargs` — Configuration fields validated by PyannoteSegmentationVADConfig.
</div>
</section>

<section class="vh-model-api-card" data-vh-model-api-card="model" markdown>
<p class="vh-model-api-card__badge-wrap"><span class="vh-model-api-card__badge">Model</span></p>

### `PyannoteSegmentationVADForVoiceActivityDetection`

<p class="vh-model-api-card__source-wrap"><a class="vh-model-api-card__source" href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/vad_pyannote_segmentation/modeling_vad_pyannote_segmentation.py">View source</a></p>
<div class="vh-model-api-card__signature" markdown>

```text
AutoModelForVoiceActivityDetection.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='vad_pyannote_segmentation',
    config=None,
    **model_kwargs,
)
```

</div>
<h4>Parameters</h4>
<div class="vh-model-api-card__parameters" markdown>
- `pretrained_model_name_or_path` — Hub ID or compatible local directory.
- `model_type` — Canonical model type; use 'vad_pyannote_segmentation'.
- `config` — Optional preloaded PyannoteSegmentationVADConfig instance.
- `**model_kwargs` — Model-specific loading arguments.
</div>
</section>

```python
from voicehub import get_model_spec

spec = get_model_spec('vad_pyannote_segmentation')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('vad_pyannote_segmentation')` |
| Load and run | `AutoModelForVoiceActivityDetection` |
| Configure | `PyannoteSegmentationVADConfig` |
| Process | `AutoProcessor` |
| Model implementation | `PyannoteSegmentationVADForVoiceActivityDetection` |
| Normalized output | `VADOutput` |
| Training contract | `get_training_spec('vad_pyannote_segmentation')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).

</div>

</div>

</div>
