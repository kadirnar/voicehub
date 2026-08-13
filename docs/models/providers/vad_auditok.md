---
description: Public API, checkpoint, training, and optimization guide for the vad_auditok integration.
hide:
  - toc
---

<div class="vh-model-detail" data-vh-model-detail data-model-type="vad_auditok" data-task="voice-activity-detection" data-training="inference-only" data-parameter-count="0" markdown>

<header class="vh-model-detail__hero" data-vh-model-hero markdown>

<p class="vh-model-detail__namespace" aria-label="Model repository"><span class="vh-model-detail__owner-avatar" aria-hidden="true">VH</span><a href="https://github.com/kadirnar/voicehub">VoiceHub</a><span aria-hidden="true">/</span><strong>vad_auditok</strong></p>

# AuditokVAD {.vh-model-title}

<p class="vh-model-detail__summary">Runs Auditok&#x27;s weightless energy detector with conservative speech/silence durations.</p>
<div class="vh-model-detail__tags" aria-label="Model metadata"><span class="vh-model-detail__chip" data-chip-kind="task">Voice activity detection</span><span class="vh-model-detail__chip" data-chip-kind="runtime">VoiceHub-native</span><span class="vh-model-detail__chip" data-chip-kind="architecture">energy-vad</span><span class="vh-model-detail__chip" data-chip-kind="parameters" aria-describedby="vh-model-parameters-note-vad_auditok">Parameters: Weightless</span><span class="vh-model-detail__chip" data-chip-kind="language">Not text-language conditioned</span><span class="vh-model-detail__chip" data-chip-kind="training">Training: inference-only</span><span class="vh-model-detail__chip" data-chip-kind="license">License: Source terms require review</span></div>
<p class="vh-model-detail__parameter-note" id="vh-model-parameters-note-vad_auditok"><strong>Parameter metadata:</strong> Weightless algorithm; the registered default has no model parameters.</p>
<div class="vh-model-detail__actions" aria-label="Model actions">
<a class="vh-model-detail__action vh-model-detail__action--primary" href="#usage" data-vh-model-action="use">Use this model</a>
<details class="vh-model-detail__resources">
<summary class="vh-model-detail__action">Resources</summary>
<div class="vh-model-detail__resource-menu">
<a href="https://github.com/amsehili/auditok" data-vh-model-action="github">Upstream GitHub</a>
<a href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/vad_auditok/modeling_vad_auditok.py" data-vh-model-action="source">VoiceHub source</a>
</div>
</details>
</div>
</header>

<nav class="vh-model-detail__tabs" aria-label="Model sections"><a href="#usage" data-vh-model-tab="usage">Usage</a><a href="#overview" data-vh-model-tab="model-card" aria-current="location">Model card</a><a href="#paper-and-github" data-vh-model-tab="sources">Sources</a><a href="#training-and-optimization" data-vh-model-tab="training">Training</a><a href="#checkpoints-provenance-license-and-limitations" data-vh-model-tab="checkpoint">Runtime</a><a href="#public-api" data-vh-model-tab="api">Public API</a></nav>

<div class="vh-model-detail__layout" markdown>

<aside class="vh-model-detail__sidebar" data-vh-model-facts aria-labelledby="vh-model-facts-title-vad_auditok"><h2 id="vh-model-facts-title-vad_auditok">Model facts</h2><details class="vh-model-detail__facts-disclosure" data-vh-model-facts-disclosure aria-labelledby="vh-model-facts-title-vad_auditok" open><summary><span>Toggle model facts</span></summary><dl class="vh-model-detail__facts"><div><dt>Task</dt><dd>Voice activity detection</dd></div><div><dt>Parameters</dt><dd aria-describedby="vh-model-parameters-note-vad_auditok">Weightless</dd></div><div><dt>Architecture</dt><dd><code>energy-vad</code></dd></div><div><dt>Runtime</dt><dd>VoiceHub-native</dd></div><div><dt>Languages</dt><dd>This weightless runtime does not select a spoken language and is not text-language conditioned; validate its implementation, configuration, and recording conditions for the target speech.</dd></div><div><dt>Capabilities</dt><dd><details class="vh-model-detail__capabilities"><summary>5 capabilities</summary><span><code>voice-activity-detection</code> <code>energy-based</code> <code>adaptive-threshold</code> <code>algorithmic</code> <code>voicehub-native</code></span></details></dd></div><div><dt>Training</dt><dd><code>inference-only</code></dd></div><div><dt>License</dt><dd>Source terms require review</dd></div><div><dt>Runtime identifier</dt><dd id="vh-model-checkpoint-vad_auditok"><code>auditok-energy-vad</code></dd></div></dl></details></aside>

<div class="vh-model-detail__main vh-model-detail__content" markdown>

## Usage

Complete the [VoiceHub installation](../../getting-started/installation.md) once,
then run this repository-authored example. Model pages intentionally contain no
package-install command.

This example is maintained against VoiceHub's public API; it is not copied from an upstream demo or package README.

**Model-specific path:** Runs Auditok's weightless energy detector with conservative speech/silence durations.

**Inputs and controls:** Energy thresholds are recording-level heuristics and must be recalibrated after gain changes.

```python
from pathlib import Path

from voicehub import AutoModelForVoiceActivityDetection

AUDIO_FILE = Path("speech.wav")
if not AUDIO_FILE.is_file():
    raise FileNotFoundError(AUDIO_FILE)

model = AutoModelForVoiceActivityDetection.from_pretrained(
    'auditok-energy-vad',
    model_type='vad_auditok',
    device="cpu",
    lazy_load=True,
)
output = model.detect(
    AUDIO_FILE,
    min_speech_duration_ms=200,
    min_silence_duration_ms=300,
    speech_pad_ms=20,
)
for segment in output.segments:
    print(segment.start, segment.end, segment.score)
```

Use authorized recordings. Version the implementation and configuration in production.

## Overview

`vad_auditok` is a VoiceHub **voice activity detection**
integration. This page is generated from its registry contract.

| Property | Value |
| --- | --- |
| Task | Voice activity detection |
| Architecture | `energy-vad` |
| Runtime | `VoiceHub-native` |
| Languages | Not text-language conditioned |
| Capabilities | `voice-activity-detection`, `energy-based`, `adaptive-threshold`, `algorithmic`, `voicehub-native` |
| Reusable components | — |
| Normalized output | `VADOutput` |

### Language support

This weightless runtime does not select a spoken language and is not text-language conditioned; validate its implementation, configuration, and recording conditions for the target speech.

## Paper and GitHub

- **Paper:** No dedicated upstream research paper is published for this integration.
- **Upstream GitHub:** [auditok](https://github.com/amsehili/auditok)
- **VoiceHub source:** [VoiceHub model implementation](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/vad_auditok/modeling_vad_auditok.py)

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('vad_auditok')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `vad_auditok` |
| Configuration class | `AuditokVADConfig` |
| Architecture class | `AuditokVADForVoiceActivityDetection` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'auditok-energy-vad',
    model_type='vad_auditok',
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
| Runtime identifier | `auditok-energy-vad` |
| Native training graph | `no` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `default` | objective | — | — | `loss`, `total_loss` |

This integration is **inference-only**. Choose a verified model from the
[training matrix](../training-support.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Runtime identifier | `auditok-energy-vad` |
| Hugging Face ID | Not published / not applicable<br>Not applicable: Auditok VAD is an energy-based detector with no model weights. |
| Checkpoint status | Not applicable; this is a weightless algorithm with no checkpoint |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cpu`; verify implementation-specific requirements |
| Real-checkpoint evidence | Not applicable; version the implementation, configuration, and source provenance |
| Implementation | `voicehub.models.vad_auditok.modeling_vad_auditok.AuditokVADForVoiceActivityDetection` |
| Configuration | `voicehub.models.vad_auditok.configuration_vad_auditok.AuditokVADConfig` |
| Source provenance | No integration-specific bundled `SOURCE.json` is declared for this registry entry. |
| License | Source terms require review |

This weightless runtime has no checkpoint license. Review the VoiceHub and upstream implementation terms before use.

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

### `AuditokVADConfig`

<p class="vh-model-api-card__source-wrap"><a class="vh-model-api-card__source" href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/vad_auditok/configuration_vad_auditok.py">View source</a></p>
<div class="vh-model-api-card__signature" markdown>

```text
AuditokVADConfig(**config_kwargs)
```

</div>
<h4>Parameters</h4>
<div class="vh-model-api-card__parameters" markdown>
- `**config_kwargs` — Configuration fields validated by AuditokVADConfig.
</div>
</section>

<section class="vh-model-api-card" data-vh-model-api-card="model" markdown>
<p class="vh-model-api-card__badge-wrap"><span class="vh-model-api-card__badge">Model</span></p>

### `AuditokVADForVoiceActivityDetection`

<p class="vh-model-api-card__source-wrap"><a class="vh-model-api-card__source" href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/vad_auditok/modeling_vad_auditok.py">View source</a></p>
<div class="vh-model-api-card__signature" markdown>

```text
AutoModelForVoiceActivityDetection.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='vad_auditok',
    config=None,
    **model_kwargs,
)
```

</div>
<h4>Parameters</h4>
<div class="vh-model-api-card__parameters" markdown>
- `pretrained_model_name_or_path` — Runtime identifier for this weightless implementation.
- `model_type` — Canonical model type; use 'vad_auditok'.
- `config` — Optional preloaded AuditokVADConfig instance.
- `**model_kwargs` — Model-specific loading arguments.
</div>
</section>

```python
from voicehub import get_model_spec

spec = get_model_spec('vad_auditok')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('vad_auditok')` |
| Load and run | `AutoModelForVoiceActivityDetection` |
| Configure | `AuditokVADConfig` |
| Process | `AutoProcessor` |
| Model implementation | `AuditokVADForVoiceActivityDetection` |
| Normalized output | `VADOutput` |
| Training contract | `get_training_spec('vad_auditok')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).

</div>

</div>

</div>
