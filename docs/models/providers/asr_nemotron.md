---
description: Public API, checkpoint, training, and optimization guide for the asr_nemotron integration.
hide:
  - toc
---

<div class="vh-model-detail" data-vh-model-detail data-model-type="asr_nemotron" data-task="automatic-speech-recognition" data-training="native" data-parameter-count="637997088" markdown>

<header class="vh-model-detail__hero" data-vh-model-hero markdown>

<p class="vh-model-detail__namespace" aria-label="Model repository"><span class="vh-model-detail__owner-avatar" aria-hidden="true">NV</span><a href="https://huggingface.co/nvidia">nvidia</a><span aria-hidden="true">/</span><strong>nemotron-3.5-asr-streaming-0.6b</strong></p>

# Nemotron {.vh-model-title}

<p class="vh-model-detail__summary">Uses Nemotron&#x27;s cache-aware native decoder and requests word timestamps.</p>
<div class="vh-model-detail__tags" aria-label="Model metadata"><span class="vh-model-detail__chip" data-chip-kind="task">Automatic speech recognition</span><span class="vh-model-detail__chip" data-chip-kind="runtime">VoiceHub-native</span><span class="vh-model-detail__chip" data-chip-kind="architecture">nemotron-3.5-rnnt</span><span class="vh-model-detail__chip" data-chip-kind="parameters" aria-describedby="vh-model-parameters-note-asr_nemotron">Parameters: 638M</span><span class="vh-model-detail__chip" data-chip-kind="language">Languages: en-US, en-GB +38</span><span class="vh-model-detail__chip" data-chip-kind="training">Training: native</span><span class="vh-model-detail__chip" data-chip-kind="license">License: OpenMDW-1.1</span></div>
<p class="vh-model-detail__parameter-note" id="vh-model-parameters-note-asr_nemotron"><strong>Parameter metadata:</strong> Exact learned-parameter total for VoiceHub&#x27;s audited native primary graph at the registered default selection; separately loaded auxiliary models are excluded.</p>
<div class="vh-model-detail__actions" aria-label="Model actions">
<a class="vh-model-detail__action vh-model-detail__action--primary" href="#usage" data-vh-model-action="use">Use this model</a>
<button class="vh-model-detail__action vh-model-detail__copy" type="button" data-vh-copy-model-id data-model-id="nvidia/nemotron-3.5-asr-streaming-0.6b" aria-describedby="vh-model-checkpoint-asr_nemotron"><span data-vh-copy-model-id-label>Copy model ID</span></button>
<a class="vh-model-detail__action" href="https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b" data-vh-model-action="checkpoint">Checkpoint</a>
<details class="vh-model-detail__resources">
<summary class="vh-model-detail__action">Resources</summary>
<div class="vh-model-detail__resource-menu">
<a href="https://github.com/NVIDIA/NeMo" data-vh-model-action="github">Upstream GitHub</a>
<a href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_nemotron/modeling_asr_nemotron.py" data-vh-model-action="source">VoiceHub source</a>
<a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/asr_nemotron.ipynb" data-vh-model-action="colab">Open in Colab</a>
</div>
</details>
</div>
</header>

<nav class="vh-model-detail__tabs" aria-label="Model sections"><a href="#usage" data-vh-model-tab="usage">Usage</a><a href="#overview" data-vh-model-tab="model-card" aria-current="location">Model card</a><a href="#paper-and-github" data-vh-model-tab="sources">Sources</a><a href="#training-and-optimization" data-vh-model-tab="training">Training</a><a href="#checkpoints-provenance-license-and-limitations" data-vh-model-tab="checkpoint">Checkpoint</a><a href="#public-api" data-vh-model-tab="api">Public API</a></nav>

<div class="vh-model-detail__layout" markdown>

<aside class="vh-model-detail__sidebar" data-vh-model-facts aria-labelledby="vh-model-facts-title-asr_nemotron"><h2 id="vh-model-facts-title-asr_nemotron">Model facts</h2><details class="vh-model-detail__facts-disclosure" data-vh-model-facts-disclosure aria-labelledby="vh-model-facts-title-asr_nemotron" open><summary><span>Toggle model facts</span></summary><dl class="vh-model-detail__facts"><div><dt>Task</dt><dd>Automatic speech recognition</dd></div><div><dt>Parameters</dt><dd aria-describedby="vh-model-parameters-note-asr_nemotron">638M</dd></div><div><dt>Architecture</dt><dd><code>nemotron-3.5-rnnt</code></dd></div><div><dt>Runtime</dt><dd>VoiceHub-native</dd></div><div><dt>Languages</dt><dd><details class="vh-model-detail__languages"><summary>40 documented codes</summary><span><code>en-US</code> <code>en-GB</code> <code>es-US</code> <code>es-ES</code> <code>fr-FR</code> <code>fr-CA</code> <code>it-IT</code> <code>pt-BR</code> <code>pt-PT</code> <code>nl-NL</code> <code>de-DE</code> <code>tr-TR</code> <code>ru-RU</code> <code>ar-AR</code> <code>hi-IN</code> <code>ja-JP</code> <code>ko-KR</code> <code>vi-VN</code> <code>uk-UA</code> <code>pl-PL</code> <code>sv-SE</code> <code>cs-CZ</code> <code>nb-NO</code> <code>da-DK</code> <code>bg-BG</code> <code>fi-FI</code> <code>hr-HR</code> <code>sk-SK</code> <code>zh-CN</code> <code>hu-HU</code> <code>ro-RO</code> <code>et-EE</code> <code>el-GR</code> <code>lt-LT</code> <code>lv-LV</code> <code>mt-MT</code> <code>sl-SI</code> <code>he-IL</code> <code>th-TH</code> <code>nn-NO</code></span></details></dd></div><div><dt>Capabilities</dt><dd><details class="vh-model-detail__capabilities"><summary>9 capabilities</summary><span><code>automatic-speech-recognition</code> <code>multilingual</code> <code>language-identification</code> <code>timestamps</code> <code>streaming-architecture</code> <code>safetensors</code> <code>fine-tuning</code> <code>voicehub-native</code> <code>native-runtime</code></span></details></dd></div><div><dt>Training</dt><dd><code>native</code></dd></div><div><dt>License</dt><dd><a href="https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b">OpenMDW-1.1</a></dd></div><div><dt>Default checkpoint</dt><dd id="vh-model-checkpoint-asr_nemotron"><a href="https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b"><code>nvidia/nemotron-3.5-asr-streaming-0.6b</code></a></dd></div></dl></details></aside>

<div class="vh-model-detail__main vh-model-detail__content" markdown>

## Usage

Complete the [VoiceHub installation](../../getting-started/installation.md) once,
then run this repository-authored example. Model pages intentionally contain no
package-install command.

This example is maintained against VoiceHub's public API; it is not copied from an upstream demo or package README.

**Model-specific path:** Uses Nemotron's cache-aware native decoder and requests word timestamps.

**Inputs and controls:** Chunk geometry is owned by the checkpoint runtime; common chunk and stride overrides intentionally fail closed.

```python
from pathlib import Path

from voicehub import AutoModelForSpeechRecognition

AUDIO_FILE = Path("speech.wav")
if not AUDIO_FILE.is_file():
    raise FileNotFoundError(AUDIO_FILE)

model = AutoModelForSpeechRecognition.from_pretrained(
    'nvidia/nemotron-3.5-asr-streaming-0.6b',
    model_type='asr_nemotron',
    device="cuda",
    lazy_load=True,
)
output = model.transcribe(
    AUDIO_FILE,
    return_timestamps="word",
)
print(output.text)
for segment in output.segments:
    print(segment.start, segment.end, segment.text, segment.confidence)
```

Use authorized recordings. Verify hardware needs and pin a revision in production.

## Overview

`asr_nemotron` is a VoiceHub **automatic speech recognition**
integration. This page is generated from its registry contract. [Open the `asr_nemotron` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/asr_nemotron.ipynb).

| Property | Value |
| --- | --- |
| Task | Automatic speech recognition |
| Architecture | `nemotron-3.5-rnnt` |
| Runtime | `VoiceHub-native` |
| Languages | `en-US`, `en-GB`, `es-US`, `es-ES`, … complete audited list below |
| Capabilities | `automatic-speech-recognition`, `multilingual`, `language-identification`, `timestamps`, `streaming-architecture`, `safetensors`, `fine-tuning`, `voicehub-native`, `native-runtime` |
| Reusable components | — |
| Normalized output | `ASROutput` |

### Language support

<details class="vh-language-support" markdown>
<summary>Supported language abbreviations</summary>

`en-US`, `en-GB`, `es-US`, `es-ES`, `fr-FR`, `fr-CA`, `it-IT`, `pt-BR`, `pt-PT`, `nl-NL`, `de-DE`, `tr-TR`, `ru-RU`, `ar-AR`, `hi-IN`, `ja-JP`, `ko-KR`, `vi-VN`, `uk-UA`, `pl-PL`, `sv-SE`, `cs-CZ`, `nb-NO`, `da-DK`, `bg-BG`, `fi-FI`, `hr-HR`, `sk-SK`, `zh-CN`, `hu-HU`, `ro-RO`, `et-EE`, `el-GR`, `lt-LT`, `lv-LV`, `mt-MT`, `sl-SI`, `he-IL`, `th-TH`, `nn-NO`

The `el-GR`, `lt-LT`, `lv-LV`, `mt-MT`, `sl-SI`, `he-IL`, `th-TH`, and `nn-NO` locales are adaptation-ready and require in-domain fine-tuning; the other listed locales are transcription-ready or broad-coverage.

</details>

## Paper and GitHub

- **Paper:** No dedicated upstream research paper is published for this integration.
- **Upstream GitHub:** [NVIDIA NeMo](https://github.com/NVIDIA/NeMo)
- **VoiceHub source:** [VoiceHub model implementation](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_nemotron/modeling_asr_nemotron.py)

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('asr_nemotron')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `asr_nemotron` |
| Configuration class | `NemotronASRConfig` |
| Architecture class | `NemotronForSpeechRecognition` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'nvidia/nemotron-3.5-asr-streaming-0.6b',
    model_type='asr_nemotron',
)
print(type(processor).__name__)
```

## Inference

The Usage example returns `ASROutput` through `AutoModelForSpeechRecognition`.

### Input and output contract

| Property | Value |
| --- | --- |
| Readiness | `integrated-raw` |
| Data architecture | `rnnt` |
| Sample rate | 16,000 Hz |
| Contract getter | `get_asr_dataset_spec('asr_nemotron')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `raw-audio` | `audio` | text / transcription / transcript | Source | at most one: text / transcription / transcript |
| `nemotron-rnnt-model-ready` | `input_features`, `attention_mask`, `prompt_ids`, `labels`, `label_lengths`, `decoder_input_ids` | — | Prepared | — |

Language-prompted Nemotron RNN-T fine-tuning records. See the [data workflow](../../guides/speech-data.md).

## Training and optimization

Use `available_optimization_passes()` to discover reversible public passes.
Unsupported runtime or hardware fails closed before mutation.

### Training contract

| Property | Value |
| --- | --- |
| Support | `native` |
| Family | `rnnt` |
| Recipe | `single-phase` |
| Default phase | `speech_recognition` |
| Training checkpoint | `nvidia/nemotron-3.5-asr-streaming-0.6b` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `speech_recognition` | objective | `model.encoder`, `model.encoder_projector`, `model.prompt_projector`, `model.decoder`, `model.joint` | `input_features`, `attention_mask`, `prompt_ids`, `labels`, `label_lengths`, `decoder_input_ids` | `loss` |

The integration accepts its declared source or prepared contract directly. Call `model.validate_training_support()` first, then follow the
[training workflow](../../guides/training.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | [`nvidia/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) |
| Hugging Face ID | [`nvidia/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)<br>Repository availability verified through the Hugging Face model API on 2026-08-11; pin a revision before production use. |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.asr_nemotron.modeling_asr_nemotron.NemotronForSpeechRecognition` |
| Configuration | `voicehub.models.asr_nemotron.configuration_asr_nemotron.NemotronASRConfig` |
| Source provenance | `voicehub/architectures/nemotron_asr/SOURCE.json` |
| License | [OpenMDW-1.1](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) |

Use of the checkpoint and derivatives is governed by the OpenMDW-1.1 license. Commercial use: **allowed by the registered terms**.

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

### `NemotronASRConfig`

<p class="vh-model-api-card__source-wrap"><a class="vh-model-api-card__source" href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_nemotron/configuration_asr_nemotron.py">View source</a></p>
<div class="vh-model-api-card__signature" markdown>

```text
NemotronASRConfig(**config_kwargs)
```

</div>
<h4>Parameters</h4>
<div class="vh-model-api-card__parameters" markdown>
- `**config_kwargs` — Configuration fields validated by NemotronASRConfig.
</div>
</section>

<section class="vh-model-api-card" data-vh-model-api-card="model" markdown>
<p class="vh-model-api-card__badge-wrap"><span class="vh-model-api-card__badge">Model</span></p>

### `NemotronForSpeechRecognition`

<p class="vh-model-api-card__source-wrap"><a class="vh-model-api-card__source" href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_nemotron/modeling_asr_nemotron.py">View source</a></p>
<div class="vh-model-api-card__signature" markdown>

```text
AutoModelForSpeechRecognition.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='asr_nemotron',
    config=None,
    **model_kwargs,
)
```

</div>
<h4>Parameters</h4>
<div class="vh-model-api-card__parameters" markdown>
- `pretrained_model_name_or_path` — Hub ID or compatible local directory.
- `model_type` — Canonical model type; use 'asr_nemotron'.
- `config` — Optional preloaded NemotronASRConfig instance.
- `**model_kwargs` — Model-specific loading arguments.
</div>
</section>

```python
from voicehub import get_model_spec

spec = get_model_spec('asr_nemotron')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('asr_nemotron')` |
| Load and run | `AutoModelForSpeechRecognition` |
| Configure | `NemotronASRConfig` |
| Process | `AutoProcessor` |
| Model implementation | `NemotronForSpeechRecognition` |
| Normalized output | `ASROutput` |
| Training contract | `get_training_spec('asr_nemotron')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).

</div>

</div>

</div>
