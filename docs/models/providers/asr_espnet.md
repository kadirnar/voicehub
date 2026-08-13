---
description: Public API, checkpoint, training, and optimization guide for the asr_espnet integration.
hide:
  - toc
---

<div class="vh-model-detail" data-vh-model-detail data-model-type="asr_espnet" data-task="automatic-speech-recognition" data-training="native" data-parameter-count="" markdown>

<header class="vh-model-detail__hero" data-vh-model-hero markdown>

<p class="vh-model-detail__namespace" aria-label="Model repository"><span class="vh-model-detail__owner-avatar" aria-hidden="true">ES</span><a href="https://huggingface.co/espnet">espnet</a><span aria-hidden="true">/</span><strong>shinji-watanabe-librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best</strong></p>

# ESPnetASR {.vh-model-title}

<p class="vh-model-detail__summary">Uses the audited ESPnet LibriSpeech transformer with an explicit beam size.</p>
<div class="vh-model-detail__tags" aria-label="Model metadata"><span class="vh-model-detail__chip" data-chip-kind="task">Automatic speech recognition</span><span class="vh-model-detail__chip" data-chip-kind="runtime">VoiceHub-native</span><span class="vh-model-detail__chip" data-chip-kind="architecture">espnet-librispeech-transformer-e18</span><span class="vh-model-detail__chip" data-chip-kind="parameters" aria-describedby="vh-model-parameters-note-asr_espnet">Parameters: Not reported</span><span class="vh-model-detail__chip" data-chip-kind="language">Language: en</span><span class="vh-model-detail__chip" data-chip-kind="training">Training: native</span><span class="vh-model-detail__chip" data-chip-kind="license">License: Checkpoint-specific</span></div>
<p class="vh-model-detail__parameter-note" id="vh-model-parameters-note-asr_espnet"><strong>Parameter metadata:</strong> Not reported: the audited metadata available for the registered default does not provide an exact parameter total.</p>
<div class="vh-model-detail__actions" aria-label="Model actions">
<a class="vh-model-detail__action vh-model-detail__action--primary" href="#usage" data-vh-model-action="use">Use this model</a>
<button class="vh-model-detail__action vh-model-detail__copy" type="button" data-vh-copy-model-id data-model-id="espnet/shinji-watanabe-librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best" aria-describedby="vh-model-checkpoint-asr_espnet"><span data-vh-copy-model-id-label>Copy model ID</span></button>
<a class="vh-model-detail__action" href="https://huggingface.co/espnet/shinji-watanabe-librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best" data-vh-model-action="checkpoint">Checkpoint</a>
<details class="vh-model-detail__resources">
<summary class="vh-model-detail__action">Resources</summary>
<div class="vh-model-detail__resource-menu">
<a href="https://arxiv.org/abs/1804.00015" data-vh-model-action="paper">Paper</a>
<a href="https://github.com/espnet/espnet" data-vh-model-action="github">Upstream GitHub</a>
<a href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_native/espnet.py" data-vh-model-action="source">VoiceHub source</a>
<a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/asr_espnet.ipynb" data-vh-model-action="colab">Open in Colab</a>
</div>
</details>
</div>
</header>

<nav class="vh-model-detail__tabs" aria-label="Model sections"><a href="#usage" data-vh-model-tab="usage">Usage</a><a href="#overview" data-vh-model-tab="model-card" aria-current="location">Model card</a><a href="#paper-and-github" data-vh-model-tab="sources">Sources</a><a href="#training-and-optimization" data-vh-model-tab="training">Training</a><a href="#checkpoints-provenance-license-and-limitations" data-vh-model-tab="checkpoint">Checkpoint</a><a href="#public-api" data-vh-model-tab="api">Public API</a></nav>

<div class="vh-model-detail__layout" markdown>

<aside class="vh-model-detail__sidebar" data-vh-model-facts aria-labelledby="vh-model-facts-title-asr_espnet"><h2 id="vh-model-facts-title-asr_espnet">Model facts</h2><details class="vh-model-detail__facts-disclosure" data-vh-model-facts-disclosure aria-labelledby="vh-model-facts-title-asr_espnet" open><summary><span>Toggle model facts</span></summary><dl class="vh-model-detail__facts"><div><dt>Task</dt><dd>Automatic speech recognition</dd></div><div><dt>Parameters</dt><dd aria-describedby="vh-model-parameters-note-asr_espnet">Not reported</dd></div><div><dt>Architecture</dt><dd><code>espnet-librispeech-transformer-e18</code></dd></div><div><dt>Runtime</dt><dd>VoiceHub-native</dd></div><div><dt>Languages</dt><dd><code>en</code></dd></div><div><dt>Capabilities</dt><dd><details class="vh-model-detail__capabilities"><summary>8 capabilities</summary><span><code>automatic-speech-recognition</code> <code>english</code> <code>safetensors</code> <code>fine-tuning</code> <code>voicehub-native</code> <code>native-runtime</code> <code>raw-audio-fine-tuning</code> <code>hybrid-ctc-attention</code></span></details></dd></div><div><dt>Training</dt><dd><code>native</code></dd></div><div><dt>License</dt><dd>Checkpoint-specific</dd></div><div><dt>Default checkpoint</dt><dd id="vh-model-checkpoint-asr_espnet"><a href="https://huggingface.co/espnet/shinji-watanabe-librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best"><code>espnet/shinji-watanabe-librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best</code></a></dd></div></dl></details></aside>

<div class="vh-model-detail__main vh-model-detail__content" markdown>

## Usage

Complete the [VoiceHub installation](../../getting-started/installation.md) once,
then run this repository-authored example. Model pages intentionally contain no
package-install command.

This example is maintained against VoiceHub's public API; it is not copied from an upstream demo or package README.

**Model-specific path:** Uses the audited ESPnet LibriSpeech transformer with an explicit beam size.

**Inputs and controls:** This release is English-only and has no calibrated timestamp head.

```python
from pathlib import Path

from voicehub import AutoModelForSpeechRecognition

AUDIO_FILE = Path("speech.wav")
if not AUDIO_FILE.is_file():
    raise FileNotFoundError(AUDIO_FILE)

model = AutoModelForSpeechRecognition.from_pretrained(
    'espnet/shinji-watanabe-librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best',
    model_type='asr_espnet',
    device="cuda",
    lazy_load=True,
)
output = model.transcribe(
    AUDIO_FILE,
    language="en",
    num_beams=10,
)
print(output.text)
for segment in output.segments:
    print(segment.start, segment.end, segment.text, segment.confidence)
```

Use authorized recordings. Verify hardware needs and pin a revision in production.

## Overview

`asr_espnet` is a VoiceHub **automatic speech recognition**
integration. This page is generated from its registry contract. [Open the `asr_espnet` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/asr_espnet.ipynb).

| Property | Value |
| --- | --- |
| Task | Automatic speech recognition |
| Architecture | `espnet-librispeech-transformer-e18` |
| Runtime | `VoiceHub-native` |
| Languages | `en` |
| Capabilities | `automatic-speech-recognition`, `english`, `safetensors`, `fine-tuning`, `voicehub-native`, `native-runtime`, `raw-audio-fine-tuning`, `hybrid-ctc-attention` |
| Reusable components | — |
| Normalized output | `ASROutput` |

### Language support

<details class="vh-language-support" markdown>
<summary>Supported language abbreviations</summary>

`en`

</details>

## Paper and GitHub

- **Paper:** [ESPnet: End-to-End Speech Processing Toolkit](https://arxiv.org/abs/1804.00015)
- **Upstream GitHub:** [ESPnet](https://github.com/espnet/espnet)
- **VoiceHub source:** [VoiceHub model implementation](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_native/espnet.py)

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('asr_espnet')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `asr_espnet` |
| Configuration class | `ESPnetASRConfig` |
| Architecture class | `ESPnetASRForSpeechRecognition` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'espnet/shinji-watanabe-librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best',
    model_type='asr_espnet',
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
| Contract getter | `get_asr_dataset_spec('asr_espnet')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `raw-audio` | — | audio / audio_path; text / transcription / transcript | Source | at most one: audio / audio_path; text / transcription / transcript |
| `espnet-feature-transcript` | `features` | text / transcription / transcript | Prepared | at most one: text / transcription / transcript |
| `espnet-waveform-model-ready` | `waveforms`, `waveform_lengths`, `labels`, `label_lengths` | — | Prepared | — |
| `espnet-feature-model-ready` | `features`, `feature_lengths`, `labels`, `label_lengths` | — | Prepared | — |

ESPnet Transformer joint CTC/attention raw and cached records. See the [data workflow](../../guides/speech-data.md).

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
| Training checkpoint | `espnet/shinji-watanabe-librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `speech_recognition` | objective | `model` | `labels`, `label_lengths` | `loss`, `ctc_loss`, `attention_loss` |

The integration accepts its declared source or prepared contract directly. Call `model.validate_training_support()` first, then follow the
[training workflow](../../guides/training.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | [`espnet/shinji-watanabe-librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best`](https://huggingface.co/espnet/shinji-watanabe-librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best) |
| Hugging Face ID | [`espnet/shinji-watanabe-librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best`](https://huggingface.co/espnet/shinji-watanabe-librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best)<br>Repository availability verified through the Hugging Face model API on 2026-08-11; pin a revision before production use. |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.asr_native.espnet.ESPnetASRForSpeechRecognition` |
| Configuration | `voicehub.models.asr_native.configuration.ESPnetASRConfig` |
| Source provenance | `voicehub/architectures/espnet_transformer/SOURCE.json` |
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

### `ESPnetASRConfig`

<p class="vh-model-api-card__source-wrap"><a class="vh-model-api-card__source" href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_native/configuration.py">View source</a></p>
<div class="vh-model-api-card__signature" markdown>

```text
ESPnetASRConfig(**config_kwargs)
```

</div>
<h4>Parameters</h4>
<div class="vh-model-api-card__parameters" markdown>
- `**config_kwargs` — Configuration fields validated by ESPnetASRConfig.
</div>
</section>

<section class="vh-model-api-card" data-vh-model-api-card="model" markdown>
<p class="vh-model-api-card__badge-wrap"><span class="vh-model-api-card__badge">Model</span></p>

### `ESPnetASRForSpeechRecognition`

<p class="vh-model-api-card__source-wrap"><a class="vh-model-api-card__source" href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_native/espnet.py">View source</a></p>
<div class="vh-model-api-card__signature" markdown>

```text
AutoModelForSpeechRecognition.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='asr_espnet',
    config=None,
    **model_kwargs,
)
```

</div>
<h4>Parameters</h4>
<div class="vh-model-api-card__parameters" markdown>
- `pretrained_model_name_or_path` — Hub ID or compatible local directory.
- `model_type` — Canonical model type; use 'asr_espnet'.
- `config` — Optional preloaded ESPnetASRConfig instance.
- `**model_kwargs` — Model-specific loading arguments.
</div>
</section>

```python
from voicehub import get_model_spec

spec = get_model_spec('asr_espnet')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('asr_espnet')` |
| Load and run | `AutoModelForSpeechRecognition` |
| Configure | `ESPnetASRConfig` |
| Process | `AutoProcessor` |
| Model implementation | `ESPnetASRForSpeechRecognition` |
| Normalized output | `ASROutput` |
| Training contract | `get_training_spec('asr_espnet')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).

</div>

</div>

</div>
