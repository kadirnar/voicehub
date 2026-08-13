---
description: Public API, checkpoint, training, and optimization guide for the asr_funasr integration.
hide:
  - toc
---

<div class="vh-model-detail" data-vh-model-detail data-model-type="asr_funasr" data-task="automatic-speech-recognition" data-training="native" data-parameter-count="" markdown>

<header class="vh-model-detail__hero" data-vh-model-hero markdown>

<p class="vh-model-detail__namespace" aria-label="Model repository"><span class="vh-model-detail__owner-avatar" aria-hidden="true">FA</span><a href="https://huggingface.co/FunAudioLLM">FunAudioLLM</a><span aria-hidden="true">/</span><strong>SenseVoiceSmall</strong></p>

# FunASR {.vh-model-title}

<p class="vh-model-detail__summary">Runs SenseVoiceSmall&#x27;s native SANM-CTC graph with language detection and word timestamps.</p>
<div class="vh-model-detail__tags" aria-label="Model metadata"><span class="vh-model-detail__chip" data-chip-kind="task">Automatic speech recognition</span><span class="vh-model-detail__chip" data-chip-kind="runtime">VoiceHub-native</span><span class="vh-model-detail__chip" data-chip-kind="architecture">sensevoice-small</span><span class="vh-model-detail__chip" data-chip-kind="parameters" aria-describedby="vh-model-parameters-note-asr_funasr">Parameters: Not reported</span><span class="vh-model-detail__chip" data-chip-kind="language">Languages: zh, en +3</span><span class="vh-model-detail__chip" data-chip-kind="training">Training: native</span><span class="vh-model-detail__chip" data-chip-kind="license">License: Checkpoint-specific</span></div>
<p class="vh-model-detail__parameter-note" id="vh-model-parameters-note-asr_funasr"><strong>Parameter metadata:</strong> Not reported: the audited metadata available for the registered default does not provide an exact parameter total.</p>
<div class="vh-model-detail__actions" aria-label="Model actions">
<a class="vh-model-detail__action vh-model-detail__action--primary" href="#usage" data-vh-model-action="use">Use this model</a>
<button class="vh-model-detail__action vh-model-detail__copy" type="button" data-vh-copy-model-id data-model-id="FunAudioLLM/SenseVoiceSmall" aria-describedby="vh-model-checkpoint-asr_funasr"><span data-vh-copy-model-id-label>Copy model ID</span></button>
<a class="vh-model-detail__action" href="https://huggingface.co/FunAudioLLM/SenseVoiceSmall" data-vh-model-action="checkpoint">Checkpoint</a>
<details class="vh-model-detail__resources">
<summary class="vh-model-detail__action">Resources</summary>
<div class="vh-model-detail__resource-menu">
<a href="https://arxiv.org/abs/2305.11013" data-vh-model-action="paper">Paper</a>
<a href="https://github.com/modelscope/FunASR" data-vh-model-action="github">Upstream GitHub</a>
<a href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_native/funasr.py" data-vh-model-action="source">VoiceHub source</a>
<a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/asr_funasr.ipynb" data-vh-model-action="colab">Open in Colab</a>
</div>
</details>
</div>
</header>

<nav class="vh-model-detail__tabs" aria-label="Model sections"><a href="#usage" data-vh-model-tab="usage">Usage</a><a href="#overview" data-vh-model-tab="model-card" aria-current="location">Model card</a><a href="#paper-and-github" data-vh-model-tab="sources">Sources</a><a href="#training-and-optimization" data-vh-model-tab="training">Training</a><a href="#checkpoints-provenance-license-and-limitations" data-vh-model-tab="checkpoint">Checkpoint</a><a href="#public-api" data-vh-model-tab="api">Public API</a></nav>

<div class="vh-model-detail__layout" markdown>

<aside class="vh-model-detail__sidebar" data-vh-model-facts aria-labelledby="vh-model-facts-title-asr_funasr"><h2 id="vh-model-facts-title-asr_funasr">Model facts</h2><details class="vh-model-detail__facts-disclosure" data-vh-model-facts-disclosure aria-labelledby="vh-model-facts-title-asr_funasr" open><summary><span>Toggle model facts</span></summary><dl class="vh-model-detail__facts"><div><dt>Task</dt><dd>Automatic speech recognition</dd></div><div><dt>Parameters</dt><dd aria-describedby="vh-model-parameters-note-asr_funasr">Not reported</dd></div><div><dt>Architecture</dt><dd><code>sensevoice-small</code></dd></div><div><dt>Runtime</dt><dd>VoiceHub-native</dd></div><div><dt>Languages</dt><dd><details class="vh-model-detail__languages"><summary>5 documented codes</summary><span><code>zh</code> <code>en</code> <code>ja</code> <code>ko</code> <code>yue</code></span></details></dd></div><div><dt>Capabilities</dt><dd><details class="vh-model-detail__capabilities"><summary>10 capabilities</summary><span><code>automatic-speech-recognition</code> <code>multilingual</code> <code>timestamps</code> <code>language-identification</code> <code>emotion-recognition</code> <code>audio-events</code> <code>fine-tuning</code> <code>safetensors</code> <code>voicehub-native</code> <code>native-runtime</code></span></details></dd></div><div><dt>Training</dt><dd><code>native</code></dd></div><div><dt>License</dt><dd>Checkpoint-specific</dd></div><div><dt>Default checkpoint</dt><dd id="vh-model-checkpoint-asr_funasr"><a href="https://huggingface.co/FunAudioLLM/SenseVoiceSmall"><code>FunAudioLLM/SenseVoiceSmall</code></a></dd></div></dl></details></aside>

<div class="vh-model-detail__main vh-model-detail__content" markdown>

## Usage

Complete the [VoiceHub installation](../../getting-started/installation.md) once,
then run this repository-authored example. Model pages intentionally contain no
package-install command.

This example is maintained against VoiceHub's public API; it is not copied from an upstream demo or package README.

**Model-specific path:** Runs SenseVoiceSmall's native SANM-CTC graph with language detection and word timestamps.

**Inputs and controls:** This provider recognizes SenseVoiceSmall only; VAD, punctuation, and speaker models must be composed separately.

```python
from pathlib import Path

from voicehub import AutoModelForSpeechRecognition

AUDIO_FILE = Path("speech.wav")
if not AUDIO_FILE.is_file():
    raise FileNotFoundError(AUDIO_FILE)

model = AutoModelForSpeechRecognition.from_pretrained(
    'FunAudioLLM/SenseVoiceSmall',
    model_type='asr_funasr',
    device="cuda",
    lazy_load=True,
)
output = model.transcribe(
    AUDIO_FILE,
    language="auto",
    return_timestamps="word",
)
print(output.text)
for segment in output.segments:
    print(segment.start, segment.end, segment.text, segment.confidence)
```

Use authorized recordings. Verify hardware needs and pin a revision in production.

## Overview

`asr_funasr` is a VoiceHub **automatic speech recognition**
integration. This page is generated from its registry contract. [Open the `asr_funasr` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/asr_funasr.ipynb).

| Property | Value |
| --- | --- |
| Task | Automatic speech recognition |
| Architecture | `sensevoice-small` |
| Runtime | `VoiceHub-native` |
| Languages | `zh`, `en`, `ja`, `ko`, `yue` |
| Capabilities | `automatic-speech-recognition`, `multilingual`, `timestamps`, `language-identification`, `emotion-recognition`, `audio-events`, `fine-tuning`, `safetensors`, `voicehub-native`, `native-runtime` |
| Reusable components | — |
| Normalized output | `ASROutput` |

### Language support

<details class="vh-language-support" markdown>
<summary>Supported language abbreviations</summary>

`zh`, `en`, `ja`, `ko`, `yue`

</details>

## Paper and GitHub

- **Paper:** [FunASR: A Fundamental End-to-End Speech Recognition Toolkit](https://arxiv.org/abs/2305.11013)
- **Upstream GitHub:** [FunASR](https://github.com/modelscope/FunASR)
- **VoiceHub source:** [VoiceHub model implementation](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_native/funasr.py)

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('asr_funasr')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `asr_funasr` |
| Configuration class | `FunASRConfig` |
| Architecture class | `FunASRForSpeechRecognition` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'FunAudioLLM/SenseVoiceSmall',
    model_type='asr_funasr',
)
print(type(processor).__name__)
```

## Inference

The Usage example returns `ASROutput` through `AutoModelForSpeechRecognition`.

### Input and output contract

| Property | Value |
| --- | --- |
| Readiness | `integrated-raw` |
| Data architecture | `ctc` |
| Sample rate | 16,000 Hz |
| Contract getter | `get_asr_dataset_spec('asr_funasr')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `raw-audio` | `language` | audio / audio_values / input_signal; text / transcription / transcript | Source | at most one: audio / audio_values / input_signal; text / transcription / transcript |
| `sensevoice-feature-transcript` | `features`, `language` | text / transcription / transcript | Prepared | at most one: text / transcription / transcript |
| `sensevoice-model-ready` | `features`, `feature_lengths`, `labels`, `label_lengths` | — | Prepared | — |

SenseVoice CTC records with language, emotion, event, and ITN control. See the [data workflow](../../guides/speech-data.md).

## Training and optimization

Use `available_optimization_passes()` to discover reversible public passes.
Unsupported runtime or hardware fails closed before mutation.

### Training contract

| Property | Value |
| --- | --- |
| Support | `native` |
| Family | `ctc` |
| Recipe | `single-phase` |
| Default phase | `speech_recognition` |
| Training checkpoint | `FunAudioLLM/SenseVoiceSmall` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `speech_recognition` | objective | `model` | `features`, `feature_lengths`, `labels`, `label_lengths` | `loss`, `ctc`, `rich` |

The integration accepts its declared source or prepared contract directly. Call `model.validate_training_support()` first, then follow the
[training workflow](../../guides/training.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | [`FunAudioLLM/SenseVoiceSmall`](https://huggingface.co/FunAudioLLM/SenseVoiceSmall) |
| Hugging Face ID | [`FunAudioLLM/SenseVoiceSmall`](https://huggingface.co/FunAudioLLM/SenseVoiceSmall)<br>Repository availability verified through the Hugging Face model API on 2026-08-11; pin a revision before production use. |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.asr_native.funasr.FunASRForSpeechRecognition` |
| Configuration | `voicehub.models.asr_native.configuration.FunASRConfig` |
| Source provenance | `voicehub/architectures/sensevoice/SOURCE.json` |
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

### `FunASRConfig`

<p class="vh-model-api-card__source-wrap"><a class="vh-model-api-card__source" href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_native/configuration.py">View source</a></p>
<div class="vh-model-api-card__signature" markdown>

```text
FunASRConfig(**config_kwargs)
```

</div>
<h4>Parameters</h4>
<div class="vh-model-api-card__parameters" markdown>
- `**config_kwargs` — Configuration fields validated by FunASRConfig.
</div>
</section>

<section class="vh-model-api-card" data-vh-model-api-card="model" markdown>
<p class="vh-model-api-card__badge-wrap"><span class="vh-model-api-card__badge">Model</span></p>

### `FunASRForSpeechRecognition`

<p class="vh-model-api-card__source-wrap"><a class="vh-model-api-card__source" href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_native/funasr.py">View source</a></p>
<div class="vh-model-api-card__signature" markdown>

```text
AutoModelForSpeechRecognition.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='asr_funasr',
    config=None,
    **model_kwargs,
)
```

</div>
<h4>Parameters</h4>
<div class="vh-model-api-card__parameters" markdown>
- `pretrained_model_name_or_path` — Hub ID or compatible local directory.
- `model_type` — Canonical model type; use 'asr_funasr'.
- `config` — Optional preloaded FunASRConfig instance.
- `**model_kwargs` — Model-specific loading arguments.
</div>
</section>

```python
from voicehub import get_model_spec

spec = get_model_spec('asr_funasr')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('asr_funasr')` |
| Load and run | `AutoModelForSpeechRecognition` |
| Configure | `FunASRConfig` |
| Process | `AutoProcessor` |
| Model implementation | `FunASRForSpeechRecognition` |
| Normalized output | `ASROutput` |
| Training contract | `get_training_spec('asr_funasr')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).

</div>

</div>

</div>
