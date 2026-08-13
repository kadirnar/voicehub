---
description: Public API, checkpoint, training, and optimization guide for the asr_parakeet_tdt integration.
hide:
  - toc
---

<div class="vh-model-detail" data-vh-model-detail data-model-type="asr_parakeet_tdt" data-task="automatic-speech-recognition" data-training="native" data-parameter-count="627008134" markdown>

<header class="vh-model-detail__hero" data-vh-model-hero markdown>

<p class="vh-model-detail__namespace" aria-label="Model repository"><span class="vh-model-detail__owner-avatar" aria-hidden="true">NV</span><a href="https://huggingface.co/nvidia">nvidia</a><span aria-hidden="true">/</span><strong>parakeet-tdt-0.6b-v3</strong></p>

# ParakeetTDT {.vh-model-title}

<p class="vh-model-detail__summary">Runs the native Parakeet TDT decoder and returns its calibrated timestamp segments.</p>
<div class="vh-model-detail__tags" aria-label="Model metadata"><span class="vh-model-detail__chip" data-chip-kind="task">Automatic speech recognition</span><span class="vh-model-detail__chip" data-chip-kind="runtime">VoiceHub-native</span><span class="vh-model-detail__chip" data-chip-kind="architecture">parakeet-tdt</span><span class="vh-model-detail__chip" data-chip-kind="parameters" aria-describedby="vh-model-parameters-note-asr_parakeet_tdt">Parameters: 627M</span><span class="vh-model-detail__chip" data-chip-kind="language">Languages: en, es +23</span><span class="vh-model-detail__chip" data-chip-kind="training">Training: native</span><span class="vh-model-detail__chip" data-chip-kind="license">License: CC-BY-4.0</span></div>
<p class="vh-model-detail__parameter-note" id="vh-model-parameters-note-asr_parakeet_tdt"><strong>Parameter metadata:</strong> Exact learned-parameter total for VoiceHub&#x27;s audited native primary graph at the registered default selection; separately loaded auxiliary models are excluded.</p>
<div class="vh-model-detail__actions" aria-label="Model actions">
<a class="vh-model-detail__action vh-model-detail__action--primary" href="#usage" data-vh-model-action="use">Use this model</a>
<button class="vh-model-detail__action vh-model-detail__copy" type="button" data-vh-copy-model-id data-model-id="nvidia/parakeet-tdt-0.6b-v3" aria-describedby="vh-model-checkpoint-asr_parakeet_tdt"><span data-vh-copy-model-id-label>Copy model ID</span></button>
<a class="vh-model-detail__action" href="https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3" data-vh-model-action="checkpoint">Checkpoint</a>
<details class="vh-model-detail__resources">
<summary class="vh-model-detail__action">Resources</summary>
<div class="vh-model-detail__resource-menu">
<a href="https://github.com/NVIDIA/NeMo" data-vh-model-action="github">Upstream GitHub</a>
<a href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_parakeet_tdt/modeling_asr_parakeet_tdt.py" data-vh-model-action="source">VoiceHub source</a>
<a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/asr_parakeet_tdt.ipynb" data-vh-model-action="colab">Open in Colab</a>
</div>
</details>
</div>
</header>

<nav class="vh-model-detail__tabs" aria-label="Model sections"><a href="#usage" data-vh-model-tab="usage">Usage</a><a href="#overview" data-vh-model-tab="model-card" aria-current="location">Model card</a><a href="#paper-and-github" data-vh-model-tab="sources">Sources</a><a href="#training-and-optimization" data-vh-model-tab="training">Training</a><a href="#checkpoints-provenance-license-and-limitations" data-vh-model-tab="checkpoint">Checkpoint</a><a href="#public-api" data-vh-model-tab="api">Public API</a></nav>

<div class="vh-model-detail__layout" markdown>

<aside class="vh-model-detail__sidebar" data-vh-model-facts aria-labelledby="vh-model-facts-title-asr_parakeet_tdt"><h2 id="vh-model-facts-title-asr_parakeet_tdt">Model facts</h2><details class="vh-model-detail__facts-disclosure" data-vh-model-facts-disclosure aria-labelledby="vh-model-facts-title-asr_parakeet_tdt" open><summary><span>Toggle model facts</span></summary><dl class="vh-model-detail__facts"><div><dt>Task</dt><dd>Automatic speech recognition</dd></div><div><dt>Parameters</dt><dd aria-describedby="vh-model-parameters-note-asr_parakeet_tdt">627M</dd></div><div><dt>Architecture</dt><dd><code>parakeet-tdt</code></dd></div><div><dt>Runtime</dt><dd>VoiceHub-native</dd></div><div><dt>Languages</dt><dd><details class="vh-model-detail__languages"><summary>25 documented codes</summary><span><code>en</code> <code>es</code> <code>fr</code> <code>de</code> <code>bg</code> <code>hr</code> <code>cs</code> <code>da</code> <code>nl</code> <code>et</code> <code>fi</code> <code>el</code> <code>hu</code> <code>it</code> <code>lv</code> <code>lt</code> <code>mt</code> <code>pl</code> <code>pt</code> <code>ro</code> <code>sk</code> <code>sl</code> <code>sv</code> <code>ru</code> <code>uk</code></span></details></dd></div><div><dt>Capabilities</dt><dd><details class="vh-model-detail__capabilities"><summary>8 capabilities</summary><span><code>automatic-speech-recognition</code> <code>multilingual</code> <code>timestamps</code> <code>long-form</code> <code>safetensors</code> <code>fine-tuning</code> <code>voicehub-native</code> <code>native-runtime</code></span></details></dd></div><div><dt>Training</dt><dd><code>native</code></dd></div><div><dt>License</dt><dd><a href="https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3">CC-BY-4.0</a></dd></div><div><dt>Default checkpoint</dt><dd id="vh-model-checkpoint-asr_parakeet_tdt"><a href="https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3"><code>nvidia/parakeet-tdt-0.6b-v3</code></a></dd></div></dl></details></aside>

<div class="vh-model-detail__main vh-model-detail__content" markdown>

## Usage

Complete the [VoiceHub installation](../../getting-started/installation.md) once,
then run this repository-authored example. Model pages intentionally contain no
package-install command.

This example is maintained against VoiceHub's public API; it is not copied from an upstream demo or package README.

**Model-specific path:** Runs the native Parakeet TDT decoder and returns its calibrated timestamp segments.

**Inputs and controls:** The registered multilingual release accepts automatic language handling; inspect the returned language metadata.

```python
from pathlib import Path

from voicehub import AutoModelForSpeechRecognition

AUDIO_FILE = Path("speech.wav")
if not AUDIO_FILE.is_file():
    raise FileNotFoundError(AUDIO_FILE)

model = AutoModelForSpeechRecognition.from_pretrained(
    'nvidia/parakeet-tdt-0.6b-v3',
    model_type='asr_parakeet_tdt',
    device="cuda",
    lazy_load=True,
)
output = model.transcribe(
    AUDIO_FILE,
    return_timestamps=True,
)
print(output.text)
for segment in output.segments:
    print(segment.start, segment.end, segment.text, segment.confidence)
```

Use authorized recordings. Verify hardware needs and pin a revision in production.

## Overview

`asr_parakeet_tdt` is a VoiceHub **automatic speech recognition**
integration. This page is generated from its registry contract. [Open the `asr_parakeet_tdt` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/asr_parakeet_tdt.ipynb).

| Property | Value |
| --- | --- |
| Task | Automatic speech recognition |
| Architecture | `parakeet-tdt` |
| Runtime | `VoiceHub-native` |
| Languages | `en`, `es`, `fr`, `de`, … complete audited list below |
| Capabilities | `automatic-speech-recognition`, `multilingual`, `timestamps`, `long-form`, `safetensors`, `fine-tuning`, `voicehub-native`, `native-runtime` |
| Reusable components | — |
| Normalized output | `ASROutput` |

### Language support

<details class="vh-language-support" markdown>
<summary>Supported language abbreviations</summary>

`en`, `es`, `fr`, `de`, `bg`, `hr`, `cs`, `da`, `nl`, `et`, `fi`, `el`, `hu`, `it`, `lv`, `lt`, `mt`, `pl`, `pt`, `ro`, `sk`, `sl`, `sv`, `ru`, `uk`

</details>

## Paper and GitHub

- **Paper:** No dedicated upstream research paper is published for this integration.
- **Upstream GitHub:** [NVIDIA NeMo](https://github.com/NVIDIA/NeMo)
- **VoiceHub source:** [VoiceHub model implementation](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_parakeet_tdt/modeling_asr_parakeet_tdt.py)

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('asr_parakeet_tdt')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `asr_parakeet_tdt` |
| Configuration class | `ParakeetTDTASRConfig` |
| Architecture class | `ParakeetTDTForSpeechRecognition` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'nvidia/parakeet-tdt-0.6b-v3',
    model_type='asr_parakeet_tdt',
)
print(type(processor).__name__)
```

## Inference

The Usage example returns `ASROutput` through `AutoModelForSpeechRecognition`.

### Input and output contract

| Property | Value |
| --- | --- |
| Readiness | `integrated-raw` |
| Data architecture | `tdt` |
| Sample rate | 16,000 Hz |
| Contract getter | `get_asr_dataset_spec('asr_parakeet_tdt')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `raw-audio` | `audio` | text / transcription / transcript | Source | at most one: text / transcription / transcript |
| `parakeet-tdt-model-ready` | `input_features`, `attention_mask`, `labels`, `decoder_input_ids` | — | Prepared | — |

Parakeet token-duration transducer audio and transcript records. See the [data workflow](../../guides/speech-data.md).

## Training and optimization

Use `available_optimization_passes()` to discover reversible public passes.
Unsupported runtime or hardware fails closed before mutation.

### Training contract

| Property | Value |
| --- | --- |
| Support | `native` |
| Family | `tdt` |
| Recipe | `single-phase` |
| Default phase | `speech_recognition` |
| Training checkpoint | `nvidia/parakeet-tdt-0.6b-v3` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `speech_recognition` | objective | `model.encoder`, `model.encoder_projector`, `model.decoder`, `model.joint` | `input_features`, `attention_mask`, `labels`, `decoder_input_ids` | `loss` |

The integration accepts its declared source or prepared contract directly. Call `model.validate_training_support()` first, then follow the
[training workflow](../../guides/training.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) |
| Hugging Face ID | [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)<br>Repository availability verified through the Hugging Face model API on 2026-08-11; pin a revision before production use. |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.asr_parakeet_tdt.modeling_asr_parakeet_tdt.ParakeetTDTForSpeechRecognition` |
| Configuration | `voicehub.models.asr_parakeet_tdt.configuration_asr_parakeet_tdt.ParakeetTDTASRConfig` |
| Source provenance | `voicehub/architectures/parakeet_tdt/SOURCE.json` |
| License | [CC-BY-4.0](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) |

The pinned Parakeet TDT checkpoint and derivatives require CC-BY-4.0 attribution. The VoiceHub-owned architecture port is audited against Apache-2.0 Transformers and NeMo source. Commercial use: **allowed by the registered terms**.

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

### `ParakeetTDTASRConfig`

<p class="vh-model-api-card__source-wrap"><a class="vh-model-api-card__source" href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_parakeet_tdt/configuration_asr_parakeet_tdt.py">View source</a></p>
<div class="vh-model-api-card__signature" markdown>

```text
ParakeetTDTASRConfig(**config_kwargs)
```

</div>
<h4>Parameters</h4>
<div class="vh-model-api-card__parameters" markdown>
- `**config_kwargs` — Configuration fields validated by ParakeetTDTASRConfig.
</div>
</section>

<section class="vh-model-api-card" data-vh-model-api-card="model" markdown>
<p class="vh-model-api-card__badge-wrap"><span class="vh-model-api-card__badge">Model</span></p>

### `ParakeetTDTForSpeechRecognition`

<p class="vh-model-api-card__source-wrap"><a class="vh-model-api-card__source" href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_parakeet_tdt/modeling_asr_parakeet_tdt.py">View source</a></p>
<div class="vh-model-api-card__signature" markdown>

```text
AutoModelForSpeechRecognition.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='asr_parakeet_tdt',
    config=None,
    **model_kwargs,
)
```

</div>
<h4>Parameters</h4>
<div class="vh-model-api-card__parameters" markdown>
- `pretrained_model_name_or_path` — Hub ID or compatible local directory.
- `model_type` — Canonical model type; use 'asr_parakeet_tdt'.
- `config` — Optional preloaded ParakeetTDTASRConfig instance.
- `**model_kwargs` — Model-specific loading arguments.
</div>
</section>

```python
from voicehub import get_model_spec

spec = get_model_spec('asr_parakeet_tdt')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('asr_parakeet_tdt')` |
| Load and run | `AutoModelForSpeechRecognition` |
| Configure | `ParakeetTDTASRConfig` |
| Process | `AutoProcessor` |
| Model implementation | `ParakeetTDTForSpeechRecognition` |
| Normalized output | `ASROutput` |
| Training contract | `get_training_spec('asr_parakeet_tdt')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).

</div>

</div>

</div>
