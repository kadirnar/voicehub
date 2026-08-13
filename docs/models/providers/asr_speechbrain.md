---
description: Public API, checkpoint, training, and optimization guide for the asr_speechbrain integration.
hide:
  - toc
---

<div class="vh-model-detail" data-vh-model-detail data-model-type="asr_speechbrain" data-task="automatic-speech-recognition" data-training="native" data-parameter-count="" markdown>

<header class="vh-model-detail__hero" data-vh-model-hero markdown>

<p class="vh-model-detail__namespace" aria-label="Model repository"><span class="vh-model-detail__owner-avatar" aria-hidden="true">SP</span><a href="https://huggingface.co/speechbrain">speechbrain</a><span aria-hidden="true">/</span><strong>asr-crdnn-rnnlm-librispeech</strong></p>

# SpeechBrainASR {.vh-model-title}

<p class="vh-model-detail__summary">Uses the audited SpeechBrain CRDNN/RNNLM decoder with an explicit beam size.</p>
<div class="vh-model-detail__tags" aria-label="Model metadata"><span class="vh-model-detail__chip" data-chip-kind="task">Automatic speech recognition</span><span class="vh-model-detail__chip" data-chip-kind="runtime">VoiceHub-native</span><span class="vh-model-detail__chip" data-chip-kind="architecture">speechbrain-crdnn-asr</span><span class="vh-model-detail__chip" data-chip-kind="parameters" aria-describedby="vh-model-parameters-note-asr_speechbrain">Parameters: Not reported</span><span class="vh-model-detail__chip" data-chip-kind="language">Language: en</span><span class="vh-model-detail__chip" data-chip-kind="training">Training: native</span><span class="vh-model-detail__chip" data-chip-kind="license">License: Apache-2.0</span></div>
<p class="vh-model-detail__parameter-note" id="vh-model-parameters-note-asr_speechbrain"><strong>Parameter metadata:</strong> Not reported: the audited metadata available for the registered default does not provide an exact parameter total.</p>
<div class="vh-model-detail__actions" aria-label="Model actions">
<a class="vh-model-detail__action vh-model-detail__action--primary" href="#usage" data-vh-model-action="use">Use this model</a>
<button class="vh-model-detail__action vh-model-detail__copy" type="button" data-vh-copy-model-id data-model-id="speechbrain/asr-crdnn-rnnlm-librispeech" aria-describedby="vh-model-checkpoint-asr_speechbrain"><span data-vh-copy-model-id-label>Copy model ID</span></button>
<a class="vh-model-detail__action" href="https://huggingface.co/speechbrain/asr-crdnn-rnnlm-librispeech" data-vh-model-action="checkpoint">Checkpoint</a>
<details class="vh-model-detail__resources">
<summary class="vh-model-detail__action">Resources</summary>
<div class="vh-model-detail__resource-menu">
<a href="https://arxiv.org/abs/2106.04624" data-vh-model-action="paper">Paper</a>
<a href="https://github.com/speechbrain/speechbrain" data-vh-model-action="github">Upstream GitHub</a>
<a href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_native/speechbrain.py" data-vh-model-action="source">VoiceHub source</a>
<a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/asr_speechbrain.ipynb" data-vh-model-action="colab">Open in Colab</a>
</div>
</details>
</div>
</header>

<nav class="vh-model-detail__tabs" aria-label="Model sections"><a href="#usage" data-vh-model-tab="usage">Usage</a><a href="#overview" data-vh-model-tab="model-card" aria-current="location">Model card</a><a href="#paper-and-github" data-vh-model-tab="sources">Sources</a><a href="#training-and-optimization" data-vh-model-tab="training">Training</a><a href="#checkpoints-provenance-license-and-limitations" data-vh-model-tab="checkpoint">Checkpoint</a><a href="#public-api" data-vh-model-tab="api">Public API</a></nav>

<div class="vh-model-detail__layout" markdown>

<aside class="vh-model-detail__sidebar" data-vh-model-facts aria-labelledby="vh-model-facts-title-asr_speechbrain"><h2 id="vh-model-facts-title-asr_speechbrain">Model facts</h2><details class="vh-model-detail__facts-disclosure" data-vh-model-facts-disclosure aria-labelledby="vh-model-facts-title-asr_speechbrain" open><summary><span>Toggle model facts</span></summary><dl class="vh-model-detail__facts"><div><dt>Task</dt><dd>Automatic speech recognition</dd></div><div><dt>Parameters</dt><dd aria-describedby="vh-model-parameters-note-asr_speechbrain">Not reported</dd></div><div><dt>Architecture</dt><dd><code>speechbrain-crdnn-asr</code></dd></div><div><dt>Runtime</dt><dd>VoiceHub-native</dd></div><div><dt>Languages</dt><dd><code>en</code></dd></div><div><dt>Capabilities</dt><dd><details class="vh-model-detail__capabilities"><summary>9 capabilities</summary><span><code>automatic-speech-recognition</code> <code>english</code> <code>beam-search</code> <code>safetensors</code> <code>fine-tuning</code> <code>voicehub-native</code> <code>crdnn</code> <code>ctc-seq2seq</code> <code>rnnlm-shallow-fusion</code></span></details></dd></div><div><dt>Training</dt><dd><code>native</code></dd></div><div><dt>License</dt><dd><a href="https://huggingface.co/speechbrain/asr-crdnn-rnnlm-librispeech">Apache-2.0</a></dd></div><div><dt>Default checkpoint</dt><dd id="vh-model-checkpoint-asr_speechbrain"><a href="https://huggingface.co/speechbrain/asr-crdnn-rnnlm-librispeech"><code>speechbrain/asr-crdnn-rnnlm-librispeech</code></a></dd></div></dl></details></aside>

<div class="vh-model-detail__main vh-model-detail__content" markdown>

## Usage

Complete the [VoiceHub installation](../../getting-started/installation.md) once,
then run this repository-authored example. Model pages intentionally contain no
package-install command.

This example is maintained against VoiceHub's public API; it is not copied from an upstream demo or package README.

**Model-specific path:** Uses the audited SpeechBrain CRDNN/RNNLM decoder with an explicit beam size.

**Inputs and controls:** The released LibriSpeech graph is English-only and does not expose calibrated timestamps.

```python
from pathlib import Path

from voicehub import AutoModelForSpeechRecognition

AUDIO_FILE = Path("speech.wav")
if not AUDIO_FILE.is_file():
    raise FileNotFoundError(AUDIO_FILE)

model = AutoModelForSpeechRecognition.from_pretrained(
    'speechbrain/asr-crdnn-rnnlm-librispeech',
    model_type='asr_speechbrain',
    device="cuda",
    lazy_load=True,
)
output = model.transcribe(
    AUDIO_FILE,
    language="en",
    num_beams=8,
)
print(output.text)
for segment in output.segments:
    print(segment.start, segment.end, segment.text, segment.confidence)
```

Use authorized recordings. Verify hardware needs and pin a revision in production.

## Overview

`asr_speechbrain` is a VoiceHub **automatic speech recognition**
integration. This page is generated from its registry contract. [Open the `asr_speechbrain` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/asr_speechbrain.ipynb).

| Property | Value |
| --- | --- |
| Task | Automatic speech recognition |
| Architecture | `speechbrain-crdnn-asr` |
| Runtime | `VoiceHub-native` |
| Languages | `en` |
| Capabilities | `automatic-speech-recognition`, `english`, `beam-search`, `safetensors`, `fine-tuning`, `voicehub-native`, `crdnn`, `ctc-seq2seq`, `rnnlm-shallow-fusion` |
| Reusable components | — |
| Normalized output | `ASROutput` |

### Language support

<details class="vh-language-support" markdown>
<summary>Supported language abbreviations</summary>

`en`

</details>

## Paper and GitHub

- **Paper:** [SpeechBrain: A General-Purpose Speech Toolkit](https://arxiv.org/abs/2106.04624)
- **Upstream GitHub:** [SpeechBrain](https://github.com/speechbrain/speechbrain)
- **VoiceHub source:** [VoiceHub model implementation](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_native/speechbrain.py)

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('asr_speechbrain')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `asr_speechbrain` |
| Configuration class | `SpeechBrainASRConfig` |
| Architecture class | `SpeechBrainASRForSpeechRecognition` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'speechbrain/asr-crdnn-rnnlm-librispeech',
    model_type='asr_speechbrain',
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
| Contract getter | `get_asr_dataset_spec('asr_speechbrain')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `raw-audio` | — | audio / audio_path; text / transcription / transcript | Source | at most one: audio / audio_path; text / transcription / transcript |
| `speechbrain-model-ready` | `waveforms`, `waveform_lengths`, `tokens_bos`, `tokens_eos`, `token_lengths`, `ctc_tokens`, `ctc_token_lengths` | — | Prepared | — |

SpeechBrain CRDNN joint CTC/attention fine-tuning records. See the [data workflow](../../guides/speech-data.md).

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
| Training checkpoint | `speechbrain/asr-crdnn-rnnlm-librispeech` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `speech_recognition` | objective | `model` | `waveforms`, `waveform_lengths`, `tokens_bos`, `tokens_eos`, `token_lengths`, `ctc_tokens`, `ctc_token_lengths` | `loss`, `seq2seq_loss`, `ctc_loss` |

The integration accepts its declared source or prepared contract directly. Call `model.validate_training_support()` first, then follow the
[training workflow](../../guides/training.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | [`speechbrain/asr-crdnn-rnnlm-librispeech`](https://huggingface.co/speechbrain/asr-crdnn-rnnlm-librispeech) |
| Hugging Face ID | [`speechbrain/asr-crdnn-rnnlm-librispeech`](https://huggingface.co/speechbrain/asr-crdnn-rnnlm-librispeech)<br>Repository availability verified through the Hugging Face model API on 2026-08-11; pin a revision before production use. |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.asr_native.speechbrain.SpeechBrainASRForSpeechRecognition` |
| Configuration | `voicehub.models.asr_native.configuration.SpeechBrainASRConfig` |
| Source provenance | `voicehub/architectures/speechbrain_asr/SOURCE.json` |
| License | [Apache-2.0](https://huggingface.co/speechbrain/asr-crdnn-rnnlm-librispeech) |

The pinned CRDNN, RNNLM, tokenizer, and source implementation are Apache-2.0. The original pickle files cross a strict one-time conversion boundary; steady-state artifacts are Safetensors. Commercial use: **allowed by the registered terms**.

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

### `SpeechBrainASRConfig`

<p class="vh-model-api-card__source-wrap"><a class="vh-model-api-card__source" href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_native/configuration.py">View source</a></p>
<div class="vh-model-api-card__signature" markdown>

```text
SpeechBrainASRConfig(**config_kwargs)
```

</div>
<h4>Parameters</h4>
<div class="vh-model-api-card__parameters" markdown>
- `**config_kwargs` — Configuration fields validated by SpeechBrainASRConfig.
</div>
</section>

<section class="vh-model-api-card" data-vh-model-api-card="model" markdown>
<p class="vh-model-api-card__badge-wrap"><span class="vh-model-api-card__badge">Model</span></p>

### `SpeechBrainASRForSpeechRecognition`

<p class="vh-model-api-card__source-wrap"><a class="vh-model-api-card__source" href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_native/speechbrain.py">View source</a></p>
<div class="vh-model-api-card__signature" markdown>

```text
AutoModelForSpeechRecognition.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='asr_speechbrain',
    config=None,
    **model_kwargs,
)
```

</div>
<h4>Parameters</h4>
<div class="vh-model-api-card__parameters" markdown>
- `pretrained_model_name_or_path` — Hub ID or compatible local directory.
- `model_type` — Canonical model type; use 'asr_speechbrain'.
- `config` — Optional preloaded SpeechBrainASRConfig instance.
- `**model_kwargs` — Model-specific loading arguments.
</div>
</section>

```python
from voicehub import get_model_spec

spec = get_model_spec('asr_speechbrain')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('asr_speechbrain')` |
| Load and run | `AutoModelForSpeechRecognition` |
| Configure | `SpeechBrainASRConfig` |
| Process | `AutoProcessor` |
| Model implementation | `SpeechBrainASRForSpeechRecognition` |
| Normalized output | `ASROutput` |
| Training contract | `get_training_spec('asr_speechbrain')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).

</div>

</div>

</div>
