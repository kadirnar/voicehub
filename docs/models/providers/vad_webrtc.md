---
description: Public API, checkpoint, training, and optimization guide for the vad_webrtc integration.
hide:
  - toc
---

<div class="vh-model-detail" data-vh-model-detail data-model-type="vad_webrtc" data-task="voice-activity-detection" data-training="inference-only" data-parameter-count="0" markdown>

<header class="vh-model-detail__hero" data-vh-model-hero markdown>

<p class="vh-model-detail__namespace" aria-label="Model repository"><span class="vh-model-detail__owner-avatar" aria-hidden="true">VH</span><a href="https://github.com/kadirnar/voicehub">VoiceHub</a><span aria-hidden="true">/</span><strong>vad_webrtc</strong></p>

# WebRTCVAD {.vh-model-title}

<p class="vh-model-detail__summary">Runs weightless WebRTC VAD with frame-compatible duration controls.</p>
<div class="vh-model-detail__tags" aria-label="Model metadata"><span class="vh-model-detail__chip" data-chip-kind="task">Voice activity detection</span><span class="vh-model-detail__chip" data-chip-kind="runtime">VoiceHub-native</span><span class="vh-model-detail__chip" data-chip-kind="architecture">webrtc-vad</span><span class="vh-model-detail__chip" data-chip-kind="parameters" aria-describedby="vh-model-parameters-note-vad_webrtc">Parameters: Weightless</span><span class="vh-model-detail__chip" data-chip-kind="language">Not text-language conditioned</span><span class="vh-model-detail__chip" data-chip-kind="training">Training: inference-only</span><span class="vh-model-detail__chip" data-chip-kind="license">License: MIT and BSD-3-Clause</span></div>
<p class="vh-model-detail__parameter-note" id="vh-model-parameters-note-vad_webrtc"><strong>Parameter metadata:</strong> Weightless algorithm; the registered default has no model parameters.</p>
<div class="vh-model-detail__actions" aria-label="Model actions">
<a class="vh-model-detail__action vh-model-detail__action--primary" href="#usage" data-vh-model-action="use">Use this model</a>
<details class="vh-model-detail__resources">
<summary class="vh-model-detail__action">Resources</summary>
<div class="vh-model-detail__resource-menu">
<a href="https://github.com/wiseman/py-webrtcvad" data-vh-model-action="github">Upstream GitHub</a>
<a href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/vad_webrtc/modeling_vad_webrtc.py" data-vh-model-action="source">VoiceHub source</a>
</div>
</details>
</div>
</header>

<nav class="vh-model-detail__tabs" aria-label="Model sections"><a href="#usage" data-vh-model-tab="usage">Usage</a><a href="#overview" data-vh-model-tab="model-card" aria-current="location">Model card</a><a href="#paper-and-github" data-vh-model-tab="sources">Sources</a><a href="#training-and-optimization" data-vh-model-tab="training">Training</a><a href="#checkpoints-provenance-license-and-limitations" data-vh-model-tab="checkpoint">Runtime</a><a href="#public-api" data-vh-model-tab="api">Public API</a></nav>

<div class="vh-model-detail__layout" markdown>

<aside class="vh-model-detail__sidebar" data-vh-model-facts aria-labelledby="vh-model-facts-title-vad_webrtc"><h2 id="vh-model-facts-title-vad_webrtc">Model facts</h2><details class="vh-model-detail__facts-disclosure" data-vh-model-facts-disclosure aria-labelledby="vh-model-facts-title-vad_webrtc" open><summary><span>Toggle model facts</span></summary><dl class="vh-model-detail__facts"><div><dt>Task</dt><dd>Voice activity detection</dd></div><div><dt>Parameters</dt><dd aria-describedby="vh-model-parameters-note-vad_webrtc">Weightless</dd></div><div><dt>Architecture</dt><dd><code>webrtc-vad</code></dd></div><div><dt>Runtime</dt><dd>VoiceHub-native</dd></div><div><dt>Languages</dt><dd>This weightless runtime does not select a spoken language and is not text-language conditioned; validate its implementation, configuration, and recording conditions for the target speech.</dd></div><div><dt>Capabilities</dt><dd><details class="vh-model-detail__capabilities"><summary>5 capabilities</summary><span><code>voice-activity-detection</code> <code>fixed-point</code> <code>voicehub-native</code> <code>native-runtime</code> <code>streaming</code></span></details></dd></div><div><dt>Training</dt><dd><code>inference-only</code></dd></div><div><dt>License</dt><dd><a href="https://github.com/kadirnar/voicehub/blob/main/voicehub/architectures/webrtc_vad/SOURCE.json">MIT and BSD-3-Clause</a></dd></div><div><dt>Runtime identifier</dt><dd id="vh-model-checkpoint-vad_webrtc"><code>webrtc-vad</code></dd></div></dl></details></aside>

<div class="vh-model-detail__main vh-model-detail__content" markdown>

## Usage

Complete the [VoiceHub installation](../../getting-started/installation.md) once,
then run this repository-authored example. Model pages intentionally contain no
package-install command.

This example is maintained against VoiceHub's public API; it is not copied from an upstream demo or package README.

**Model-specific path:** Runs weightless WebRTC VAD with frame-compatible duration controls.

**Inputs and controls:** Input is resampled and framed by VoiceHub; algorithm aggressiveness belongs to the model configuration.

```python
from pathlib import Path

from voicehub import AutoModelForVoiceActivityDetection

AUDIO_FILE = Path("speech.wav")
if not AUDIO_FILE.is_file():
    raise FileNotFoundError(AUDIO_FILE)

model = AutoModelForVoiceActivityDetection.from_pretrained(
    'webrtc-vad',
    model_type='vad_webrtc',
    device="cpu",
    lazy_load=True,
)
output = model.detect(
    AUDIO_FILE,
    min_speech_duration_ms=120,
    min_silence_duration_ms=240,
    speech_pad_ms=30,
)
for segment in output.segments:
    print(segment.start, segment.end, segment.score)
```

Use authorized recordings. Version the implementation and configuration in production.

## Overview

`vad_webrtc` is a VoiceHub **voice activity detection**
integration. This page is generated from its registry contract.

| Property | Value |
| --- | --- |
| Task | Voice activity detection |
| Architecture | `webrtc-vad` |
| Runtime | `VoiceHub-native` |
| Languages | Not text-language conditioned |
| Capabilities | `voice-activity-detection`, `fixed-point`, `voicehub-native`, `native-runtime`, `streaming` |
| Reusable components | — |
| Normalized output | `VADOutput` |

### Language support

This weightless runtime does not select a spoken language and is not text-language conditioned; validate its implementation, configuration, and recording conditions for the target speech.

## Paper and GitHub

- **Paper:** No dedicated upstream research paper is published for this integration.
- **Upstream GitHub:** [py-webrtcvad](https://github.com/wiseman/py-webrtcvad)
- **VoiceHub source:** [VoiceHub model implementation](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/vad_webrtc/modeling_vad_webrtc.py)

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('vad_webrtc')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `vad_webrtc` |
| Configuration class | `WebRTCVADConfig` |
| Architecture class | `WebRTCVADForVoiceActivityDetection` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'webrtc-vad',
    model_type='vad_webrtc',
)
print(type(processor).__name__)
```

## Inference

The Usage example returns `VADOutput` through `AutoModelForVoiceActivityDetection`.

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
| Family | `upstream-native` |
| Recipe | `single-phase` |
| Default phase | `default` |
| Runtime identifier | `webrtc-vad` |
| Native training graph | `no` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `default` | objective | — | — | `loss`, `total_loss` |

This integration is **inference-only**. Choose a verified model from the
[training matrix](../training-support.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Runtime identifier | `webrtc-vad` |
| Hugging Face ID | Not published / not applicable<br>Not applicable: WebRTC VAD is a weightless signal-processing algorithm. |
| Checkpoint status | Not applicable; this is a weightless algorithm with no checkpoint |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cpu`; verify implementation-specific requirements |
| Real-checkpoint evidence | Not applicable; version the implementation, configuration, and source provenance |
| Implementation | `voicehub.models.vad_webrtc.modeling_vad_webrtc.WebRTCVADForVoiceActivityDetection` |
| Configuration | `voicehub.models.vad_webrtc.configuration_vad_webrtc.WebRTCVADConfig` |
| Source provenance | `voicehub/architectures/webrtc_vad/SOURCE.json` |
| License | [MIT and BSD-3-Clause](https://github.com/kadirnar/voicehub/blob/main/voicehub/architectures/webrtc_vad/SOURCE.json) |

This weightless runtime has no checkpoint license. Its audited source record declares **MIT and BSD-3-Clause**; verify those implementation terms.

Confirm the implementation revision, source provenance, access terms, and license.

### Limitations

- Weightless algorithm; the registered default has no model parameters.
- Validate memory, precision, and optional dependencies on the target system.
- Public optimizations fail closed when the runtime or hardware cannot satisfy
  their validation contract; an unavailable pass is not reported as applied.
- Contract tests do not replace implementation and recording-condition validation.

## Public API

Use the stable configuration, processor, and task-model facades below.

<section class="vh-model-api-card" data-vh-model-api-card="configuration" markdown>
<p class="vh-model-api-card__badge-wrap"><span class="vh-model-api-card__badge">Configuration</span></p>

### `WebRTCVADConfig`

<p class="vh-model-api-card__source-wrap"><a class="vh-model-api-card__source" href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/vad_webrtc/configuration_vad_webrtc.py">View source</a></p>
<div class="vh-model-api-card__signature" markdown>

```text
WebRTCVADConfig(**config_kwargs)
```

</div>
<h4>Parameters</h4>
<div class="vh-model-api-card__parameters" markdown>
- `**config_kwargs` — Configuration fields validated by WebRTCVADConfig.
</div>
</section>

<section class="vh-model-api-card" data-vh-model-api-card="model" markdown>
<p class="vh-model-api-card__badge-wrap"><span class="vh-model-api-card__badge">Model</span></p>

### `WebRTCVADForVoiceActivityDetection`

<p class="vh-model-api-card__source-wrap"><a class="vh-model-api-card__source" href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/vad_webrtc/modeling_vad_webrtc.py">View source</a></p>
<div class="vh-model-api-card__signature" markdown>

```text
AutoModelForVoiceActivityDetection.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='vad_webrtc',
    config=None,
    **model_kwargs,
)
```

</div>
<h4>Parameters</h4>
<div class="vh-model-api-card__parameters" markdown>
- `pretrained_model_name_or_path` — Runtime identifier for this weightless implementation.
- `model_type` — Canonical model type; use 'vad_webrtc'.
- `config` — Optional preloaded WebRTCVADConfig instance.
- `**model_kwargs` — Model-specific loading arguments.
</div>
</section>

```python
from voicehub import get_model_spec

spec = get_model_spec('vad_webrtc')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('vad_webrtc')` |
| Load and run | `AutoModelForVoiceActivityDetection` |
| Configure | `WebRTCVADConfig` |
| Process | `AutoProcessor` |
| Model implementation | `WebRTCVADForVoiceActivityDetection` |
| Normalized output | `VADOutput` |
| Training contract | `get_training_spec('vad_webrtc')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).

</div>

</div>

</div>
