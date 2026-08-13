---
description: Public API, checkpoint, training, and optimization guide for the asr_wenet integration.
hide:
  - toc
---

<div class="vh-model-detail" data-vh-model-detail data-model-type="asr_wenet" data-task="automatic-speech-recognition" data-training="native" data-parameter-count="" markdown>

<header class="vh-model-detail__hero" data-vh-model-hero markdown>

<p class="vh-model-detail__namespace" aria-label="Model repository"><span class="vh-model-detail__owner-avatar" aria-hidden="true">VH</span><a href="https://github.com/kadirnar/voicehub">VoiceHub</a><span aria-hidden="true">/</span><strong>asr_wenet</strong></p>

# WeNetASR {.vh-model-title}

<p class="vh-model-detail__summary">Loads a reviewed VoiceHub conversion of WeNet GigaSpeech U2++ and requests word timestamps.</p>
<div class="vh-model-detail__tags" aria-label="Model metadata"><span class="vh-model-detail__chip" data-chip-kind="task">Automatic speech recognition</span><span class="vh-model-detail__chip" data-chip-kind="runtime">VoiceHub-native</span><span class="vh-model-detail__chip" data-chip-kind="architecture">wenet-asr</span><span class="vh-model-detail__chip" data-chip-kind="parameters" aria-describedby="vh-model-parameters-note-asr_wenet">Parameters: Not reported</span><span class="vh-model-detail__chip" data-chip-kind="language">Language: en</span><span class="vh-model-detail__chip" data-chip-kind="training">Training: native</span><span class="vh-model-detail__chip" data-chip-kind="license">License: NOT DECLARED</span></div>
<p class="vh-model-detail__parameter-note" id="vh-model-parameters-note-asr_wenet"><strong>Parameter metadata:</strong> Not reported: the audited metadata available for the registered default does not provide an exact parameter total.</p>
<div class="vh-model-detail__actions" aria-label="Model actions">
<a class="vh-model-detail__action vh-model-detail__action--primary" href="#usage" data-vh-model-action="use">Use this model</a>
<button class="vh-model-detail__action vh-model-detail__copy" type="button" data-vh-copy-model-id data-model-id="wenet/gigaspeech-u2pp-conformer" aria-describedby="vh-model-checkpoint-asr_wenet"><span data-vh-copy-model-id-label>Copy model ID</span></button>
<a class="vh-model-detail__action" href="https://github.com/wenet-e2e/wenet/blob/a50d4208f13bbf3a0746e606ac29176cd2e87e6b/examples/gigaspeech/s0/README.md#conformer-u2-result" data-vh-model-action="checkpoint">Checkpoint</a>
<details class="vh-model-detail__resources">
<summary class="vh-model-detail__action">Resources</summary>
<div class="vh-model-detail__resource-menu">
<a href="https://arxiv.org/abs/2102.01547" data-vh-model-action="paper">Paper</a>
<a href="https://github.com/wenet-e2e/wenet" data-vh-model-action="github">Upstream GitHub</a>
<a href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_wenet/__init__.py" data-vh-model-action="source">VoiceHub source</a>
</div>
</details>
</div>
</header>

<nav class="vh-model-detail__tabs" aria-label="Model sections"><a href="#usage" data-vh-model-tab="usage">Usage</a><a href="#overview" data-vh-model-tab="model-card" aria-current="location">Model card</a><a href="#paper-and-github" data-vh-model-tab="sources">Sources</a><a href="#training-and-optimization" data-vh-model-tab="training">Training</a><a href="#checkpoints-provenance-license-and-limitations" data-vh-model-tab="checkpoint">Checkpoint</a><a href="#public-api" data-vh-model-tab="api">Public API</a></nav>

<div class="vh-model-detail__layout" markdown>

<aside class="vh-model-detail__sidebar" data-vh-model-facts aria-labelledby="vh-model-facts-title-asr_wenet"><h2 id="vh-model-facts-title-asr_wenet">Model facts</h2><details class="vh-model-detail__facts-disclosure" data-vh-model-facts-disclosure aria-labelledby="vh-model-facts-title-asr_wenet" open><summary><span>Toggle model facts</span></summary><dl class="vh-model-detail__facts"><div><dt>Task</dt><dd>Automatic speech recognition</dd></div><div><dt>Parameters</dt><dd aria-describedby="vh-model-parameters-note-asr_wenet">Not reported</dd></div><div><dt>Architecture</dt><dd><code>wenet-asr</code></dd></div><div><dt>Runtime</dt><dd>VoiceHub-native</dd></div><div><dt>Languages</dt><dd><code>en</code></dd></div><div><dt>Capabilities</dt><dd><details class="vh-model-detail__capabilities"><summary>8 capabilities</summary><span><code>automatic-speech-recognition</code> <code>english</code> <code>timestamps</code> <code>safetensors</code> <code>fine-tuning</code> <code>voicehub-native</code> <code>ctc</code> <code>attention-rescoring</code></span></details></dd></div><div><dt>Training</dt><dd><code>native</code></dd></div><div><dt>License</dt><dd><a href="https://github.com/wenet-e2e/wenet/blob/a50d4208f13bbf3a0746e606ac29176cd2e87e6b/examples/gigaspeech/s0/README.md#conformer-u2-result">NOT DECLARED</a></dd></div><div><dt>Default checkpoint</dt><dd id="vh-model-checkpoint-asr_wenet"><a href="https://github.com/wenet-e2e/wenet/blob/a50d4208f13bbf3a0746e606ac29176cd2e87e6b/examples/gigaspeech/s0/README.md#conformer-u2-result"><code>wenet/gigaspeech-u2pp-conformer</code></a></dd></div></dl></details></aside>

<div class="vh-model-detail__main vh-model-detail__content" markdown>

## Usage

Complete the [VoiceHub installation](../../getting-started/installation.md) once,
then run this repository-authored example. Model pages intentionally contain no
package-install command.

This example is maintained against VoiceHub's public API; it is not copied from an upstream demo or package README.

**Model-specific path:** Loads a reviewed VoiceHub conversion of WeNet GigaSpeech U2++ and requests word timestamps.

**Inputs and controls:** The external release is not a drop-in HF model; convert it through the audited artifact boundary first.

**Checkpoint note:** The registry identifier is not a Hugging Face repository and the original upstream archive endpoints are unavailable. VoiceHub verifies an immutable mirror against the published 503,845,602-byte archive's SHA-256. Convert that trust-gated pickle archive first, then replace the path below with the resulting VoiceHub-native directory containing model.safetensors, config.json, tokenizer.model, and units.txt.

```python
from pathlib import Path

from voicehub import AutoModelForSpeechRecognition

AUDIO_FILE = Path("speech.wav")
if not AUDIO_FILE.is_file():
    raise FileNotFoundError(AUDIO_FILE)

model = AutoModelForSpeechRecognition.from_pretrained(
    'path/to/converted-wenet-u2pp',
    model_type='asr_wenet',
    device="cuda",
    lazy_load=True,
)
output = model.transcribe(
    AUDIO_FILE,
    language="en",
    return_timestamps="word",
    num_beams=10,
)
print(output.text)
for segment in output.segments:
    print(segment.start, segment.end, segment.text, segment.confidence)
```

Use authorized recordings. Verify hardware needs and pin a revision in production.

## Overview

`asr_wenet` is a VoiceHub **automatic speech recognition**
integration. This page is generated from its registry contract.

| Property | Value |
| --- | --- |
| Task | Automatic speech recognition |
| Architecture | `wenet-asr` |
| Runtime | `VoiceHub-native` |
| Languages | `en` |
| Capabilities | `automatic-speech-recognition`, `english`, `timestamps`, `safetensors`, `fine-tuning`, `voicehub-native`, `ctc`, `attention-rescoring` |
| Reusable components | — |
| Normalized output | `ASROutput` |

### Language support

<details class="vh-language-support" markdown>
<summary>Supported language abbreviations</summary>

`en`

</details>

## Paper and GitHub

- **Paper:** [WeNet: Production Oriented Streaming and Non-Streaming End-to-End Speech Recognition Toolkit](https://arxiv.org/abs/2102.01547)
- **Upstream GitHub:** [WeNet](https://github.com/wenet-e2e/wenet)
- **VoiceHub source:** [VoiceHub model implementation](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_wenet/__init__.py)

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('asr_wenet')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `asr_wenet` |
| Configuration class | `WeNetASRConfig` |
| Architecture class | `WeNetASRForSpeechRecognition` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'path/to/converted-wenet-u2pp',
    model_type='asr_wenet',
)
print(type(processor).__name__)
```

## Inference

The Usage example returns `ASROutput` through `AutoModelForSpeechRecognition`.

### Input and output contract

| Property | Value |
| --- | --- |
| Readiness | `integrated-raw` |
| Data architecture | `hybrid-ctc-attention` |
| Sample rate | 16,000 Hz |
| Contract getter | `get_asr_dataset_spec('asr_wenet')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `raw-audio` | `audio` | text / transcription / transcript | Source | at most one: text / transcription / transcript |
| `wenet-waveform-model-ready` | `input_signal`, `input_signal_length`, `labels`, `label_lengths` | — | Prepared | — |
| `wenet-feature-model-ready` | `features`, `feature_lengths`, `labels`, `label_lengths` | — | Prepared | — |

WeNet U2++ joint CTC/attention fine-tuning records. See the [data workflow](../../guides/speech-data.md).

## Training and optimization

Use `available_optimization_passes()` to discover reversible public passes.
Unsupported runtime or hardware fails closed before mutation.

### Training contract

| Property | Value |
| --- | --- |
| Support | `native` |
| Family | `speech-sequence-to-sequence` |
| Recipe | `single-phase` |
| Default phase | `speech_recognition` |
| Training checkpoint | `wenet/gigaspeech-u2pp-conformer` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `speech_recognition` | objective | `model` | `labels`, `label_lengths` | `loss`, `attention_loss`, `ctc_loss` |

The integration accepts its declared source or prepared contract directly. Call `model.validate_training_support()` first, then follow the
[training workflow](../../guides/training.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | [`wenet/gigaspeech-u2pp-conformer`](https://github.com/wenet-e2e/wenet/blob/a50d4208f13bbf3a0746e606ac29176cd2e87e6b/examples/gigaspeech/s0/README.md#conformer-u2-result) |
| Hugging Face ID | Not published / not applicable<br>No canonical Hugging Face repository for the exact audited GigaSpeech U2++ release; the page links the verified external archive and conversion boundary. |
| Checkpoint status | Original upstream archive unavailable (HTTP 404 and TLS failures verified 2026-08-04); exact bytes are available from the immutable openspeech/wenet-models mirror at 90acd57d17169a15d5ceab462c6e7db3bd003921 |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.asr_wenet.WeNetASRForSpeechRecognition` |
| Configuration | `voicehub.models.asr_wenet.WeNetASRConfig` |
| Source provenance | `voicehub/architectures/wenet_u2pp/SOURCE.json` |
| License | [NOT DECLARED](https://github.com/wenet-e2e/wenet/blob/a50d4208f13bbf3a0746e606ac29176cd2e87e6b/examples/gigaspeech/s0/README.md#conformer-u2-result) |

The published GigaSpeech checkpoint archive does not declare a checkpoint license. The VoiceHub-owned architecture port is Apache-2.0, but that source license is not assumed for the weights. Commercial use: **review required**.

Confirm the checkpoint revision, access terms, provenance, and license.

### Limitations

- The registry identifier is not a Hugging Face repository and the original upstream archive endpoints are unavailable. VoiceHub verifies an immutable mirror against the published 503,845,602-byte archive's SHA-256. Convert that trust-gated pickle archive first, then replace the path below with the resulting VoiceHub-native directory containing model.safetensors, config.json, tokenizer.model, and units.txt.
- Validate memory, precision, and optional dependencies on the target system.
- Public optimizations fail closed when the runtime or hardware cannot satisfy
  their validation contract; an unavailable pass is not reported as applied.
- Contract tests do not replace the linked released-checkpoint evidence.

## Public API

Use the stable configuration, processor, and task-model facades below.

<section class="vh-model-api-card" data-vh-model-api-card="configuration" markdown>
<p class="vh-model-api-card__badge-wrap"><span class="vh-model-api-card__badge">Configuration</span></p>

### `WeNetASRConfig`

<p class="vh-model-api-card__source-wrap"><a class="vh-model-api-card__source" href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_wenet/__init__.py">View source</a></p>
<div class="vh-model-api-card__signature" markdown>

```text
WeNetASRConfig(**config_kwargs)
```

</div>
<h4>Parameters</h4>
<div class="vh-model-api-card__parameters" markdown>
- `**config_kwargs` — Configuration fields validated by WeNetASRConfig.
</div>
</section>

<section class="vh-model-api-card" data-vh-model-api-card="model" markdown>
<p class="vh-model-api-card__badge-wrap"><span class="vh-model-api-card__badge">Model</span></p>

### `WeNetASRForSpeechRecognition`

<p class="vh-model-api-card__source-wrap"><a class="vh-model-api-card__source" href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_wenet/__init__.py">View source</a></p>
<div class="vh-model-api-card__signature" markdown>

```text
AutoModelForSpeechRecognition.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='asr_wenet',
    config=None,
    **model_kwargs,
)
```

</div>
<h4>Parameters</h4>
<div class="vh-model-api-card__parameters" markdown>
- `pretrained_model_name_or_path` — Hub ID or compatible local directory.
- `model_type` — Canonical model type; use 'asr_wenet'.
- `config` — Optional preloaded WeNetASRConfig instance.
- `**model_kwargs` — Model-specific loading arguments.
</div>
</section>

```python
from voicehub import get_model_spec

spec = get_model_spec('asr_wenet')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('asr_wenet')` |
| Load and run | `AutoModelForSpeechRecognition` |
| Configure | `WeNetASRConfig` |
| Process | `AutoProcessor` |
| Model implementation | `WeNetASRForSpeechRecognition` |
| Normalized output | `ASROutput` |
| Training contract | `get_training_spec('asr_wenet')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).

</div>

</div>

</div>
