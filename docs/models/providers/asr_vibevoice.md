---
description: Public API, checkpoint, training, and optimization guide for the asr_vibevoice integration.
hide:
  - toc
---

<div class="vh-model-detail" data-vh-model-detail data-model-type="asr_vibevoice" data-task="automatic-speech-recognition" data-training="native" data-parameter-count="8330325888" markdown>

<header class="vh-model-detail__hero" data-vh-model-hero markdown>

<p class="vh-model-detail__namespace" aria-label="Model repository"><span class="vh-model-detail__owner-avatar" aria-hidden="true">MI</span><a href="https://huggingface.co/microsoft">microsoft</a><span aria-hidden="true">/</span><strong>VibeVoice-ASR-HF</strong></p>

# VibeVoice {.vh-model-title}

<p class="vh-model-detail__summary">Requests VibeVoice-ASR timestamps with a concise transcription prompt.</p>
<div class="vh-model-detail__tags" aria-label="Model metadata"><span class="vh-model-detail__chip" data-chip-kind="task">Automatic speech recognition</span><span class="vh-model-detail__chip" data-chip-kind="runtime">VoiceHub-native</span><span class="vh-model-detail__chip" data-chip-kind="architecture">vibevoice-asr</span><span class="vh-model-detail__chip" data-chip-kind="parameters" aria-describedby="vh-model-parameters-note-asr_vibevoice">Parameters: 8.3B</span><span class="vh-model-detail__chip" data-chip-kind="language">Languages: en, zh +49</span><span class="vh-model-detail__chip" data-chip-kind="training">Training: native</span><span class="vh-model-detail__chip" data-chip-kind="license">License: Checkpoint-specific</span></div>
<p class="vh-model-detail__parameter-note" id="vh-model-parameters-note-asr_vibevoice"><strong>Parameter metadata:</strong> Exact serialized tensor-element total from VoiceHub&#x27;s audited native primary checkpoint; a distinct learned-parameter total is not available.</p>
<div class="vh-model-detail__actions" aria-label="Model actions">
<a class="vh-model-detail__action vh-model-detail__action--primary" href="#usage" data-vh-model-action="use">Use this model</a>
<button class="vh-model-detail__action vh-model-detail__copy" type="button" data-vh-copy-model-id data-model-id="microsoft/VibeVoice-ASR-HF" aria-describedby="vh-model-checkpoint-asr_vibevoice"><span data-vh-copy-model-id-label>Copy model ID</span></button>
<a class="vh-model-detail__action" href="https://huggingface.co/microsoft/VibeVoice-ASR-HF" data-vh-model-action="checkpoint">Checkpoint</a>
<details class="vh-model-detail__resources">
<summary class="vh-model-detail__action">Resources</summary>
<div class="vh-model-detail__resource-menu">
<a href="https://github.com/microsoft/VibeVoice" data-vh-model-action="github">Upstream GitHub</a>
<a href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_vibevoice/modeling_asr_vibevoice.py" data-vh-model-action="source">VoiceHub source</a>
<a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/asr_vibevoice.ipynb" data-vh-model-action="colab">Open in Colab</a>
</div>
</details>
</div>
</header>

<nav class="vh-model-detail__tabs" aria-label="Model sections"><a href="#usage" data-vh-model-tab="usage">Usage</a><a href="#overview" data-vh-model-tab="model-card" aria-current="location">Model card</a><a href="#paper-and-github" data-vh-model-tab="sources">Sources</a><a href="#training-and-optimization" data-vh-model-tab="training">Training</a><a href="#checkpoints-provenance-license-and-limitations" data-vh-model-tab="checkpoint">Checkpoint</a><a href="#public-api" data-vh-model-tab="api">Public API</a></nav>

<div class="vh-model-detail__layout" markdown>

<aside class="vh-model-detail__sidebar" data-vh-model-facts aria-labelledby="vh-model-facts-title-asr_vibevoice"><h2 id="vh-model-facts-title-asr_vibevoice">Model facts</h2><details class="vh-model-detail__facts-disclosure" data-vh-model-facts-disclosure aria-labelledby="vh-model-facts-title-asr_vibevoice" open><summary><span>Toggle model facts</span></summary><dl class="vh-model-detail__facts"><div><dt>Task</dt><dd>Automatic speech recognition</dd></div><div><dt>Parameters</dt><dd aria-describedby="vh-model-parameters-note-asr_vibevoice">8.3B</dd></div><div><dt>Architecture</dt><dd><code>vibevoice-asr</code></dd></div><div><dt>Runtime</dt><dd>VoiceHub-native</dd></div><div><dt>Languages</dt><dd><details class="vh-model-detail__languages"><summary>51 documented codes</summary><span><code>en</code> <code>zh</code> <code>es</code> <code>pt</code> <code>de</code> <code>ja</code> <code>ko</code> <code>fr</code> <code>ru</code> <code>id</code> <code>sv</code> <code>it</code> <code>he</code> <code>nl</code> <code>pl</code> <code>no</code> <code>tr</code> <code>th</code> <code>ar</code> <code>hu</code> <code>ca</code> <code>cs</code> <code>da</code> <code>fa</code> <code>af</code> <code>hi</code> <code>fi</code> <code>et</code> <code>aa</code> <code>el</code> <code>ro</code> <code>vi</code> <code>bg</code> <code>is</code> <code>sl</code> <code>sk</code> <code>lt</code> <code>sw</code> <code>uk</code> <code>kl</code> <code>lv</code> <code>hr</code> <code>ne</code> <code>sr</code> <code>tl</code> <code>yi</code> <code>ms</code> <code>ur</code> <code>mn</code> <code>hy</code> <code>jv</code></span></details></dd></div><div><dt>Capabilities</dt><dd><details class="vh-model-detail__capabilities"><summary>10 capabilities</summary><span><code>automatic-speech-recognition</code> <code>multilingual</code> <code>speaker-attribution</code> <code>timestamps</code> <code>hotwords</code> <code>long-form</code> <code>safetensors</code> <code>fine-tuning</code> <code>voicehub-native</code> <code>native-runtime</code></span></details></dd></div><div><dt>Training</dt><dd><code>native</code></dd></div><div><dt>License</dt><dd>Checkpoint-specific</dd></div><div><dt>Default checkpoint</dt><dd id="vh-model-checkpoint-asr_vibevoice"><a href="https://huggingface.co/microsoft/VibeVoice-ASR-HF"><code>microsoft/VibeVoice-ASR-HF</code></a></dd></div></dl></details></aside>

<div class="vh-model-detail__main vh-model-detail__content" markdown>

## Usage

Complete the [VoiceHub installation](../../getting-started/installation.md) once,
then run this repository-authored example. Model pages intentionally contain no
package-install command.

This example is maintained against VoiceHub's public API; it is not copied from an upstream demo or package README.

**Model-specific path:** Requests VibeVoice-ASR timestamps with a concise transcription prompt.

**Inputs and controls:** Keep the prompt task-focused and verify timestamp granularity for the selected checkpoint revision.

```python
from pathlib import Path

from voicehub import AutoModelForSpeechRecognition

AUDIO_FILE = Path("speech.wav")
if not AUDIO_FILE.is_file():
    raise FileNotFoundError(AUDIO_FILE)

model = AutoModelForSpeechRecognition.from_pretrained(
    'microsoft/VibeVoice-ASR-HF',
    model_type='asr_vibevoice',
    device="cuda",
    lazy_load=True,
)
output = model.transcribe(
    AUDIO_FILE,
    return_timestamps=True,
    prompt="Transcribe every spoken turn.",
)
print(output.text)
for segment in output.segments:
    print(segment.start, segment.end, segment.text, segment.confidence)
```

Use authorized recordings. Verify hardware needs and pin a revision in production.

## Overview

`asr_vibevoice` is a VoiceHub **automatic speech recognition**
integration. This page is generated from its registry contract. [Open the `asr_vibevoice` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/asr_vibevoice.ipynb).

| Property | Value |
| --- | --- |
| Task | Automatic speech recognition |
| Architecture | `vibevoice-asr` |
| Runtime | `VoiceHub-native` |
| Languages | `en`, `zh`, `es`, `pt`, … complete audited list below |
| Capabilities | `automatic-speech-recognition`, `multilingual`, `speaker-attribution`, `timestamps`, `hotwords`, `long-form`, `safetensors`, `fine-tuning`, `voicehub-native`, `native-runtime` |
| Reusable components | — |
| Normalized output | `ASROutput` |

### Language support

<details class="vh-language-support" markdown>
<summary>Supported language abbreviations</summary>

`en`, `zh`, `es`, `pt`, `de`, `ja`, `ko`, `fr`, `ru`, `id`, `sv`, `it`, `he`, `nl`, `pl`, `no`, `tr`, `th`, `ar`, `hu`, `ca`, `cs`, `da`, `fa`, `af`, `hi`, `fi`, `et`, `aa`, `el`, `ro`, `vi`, `bg`, `is`, `sl`, `sk`, `lt`, `sw`, `uk`, `kl`, `lv`, `hr`, `ne`, `sr`, `tl`, `yi`, `ms`, `ur`, `mn`, `hy`, `jv`

</details>

## Paper and GitHub

- **Paper:** No dedicated upstream research paper is published for this integration.
- **Upstream GitHub:** [VibeVoice](https://github.com/microsoft/VibeVoice)
- **VoiceHub source:** [VoiceHub model implementation](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_vibevoice/modeling_asr_vibevoice.py)

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('asr_vibevoice')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `asr_vibevoice` |
| Configuration class | `VibeVoiceASRConfig` |
| Architecture class | `VibeVoiceForSpeechRecognition` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'microsoft/VibeVoice-ASR-HF',
    model_type='asr_vibevoice',
)
print(type(processor).__name__)
```

## Inference

The Usage example returns `ASROutput` through `AutoModelForSpeechRecognition`.

### Input and output contract

| Property | Value |
| --- | --- |
| Readiness | `integrated-raw` |
| Data architecture | `prompted-multimodal` |
| Sample rate | 24,000 Hz |
| Contract getter | `get_asr_dataset_spec('asr_vibevoice')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `segmented-audio` | `audio`, `segments` | — | Source | forbidden: text, transcription, transcript |
| `serialized-audio` | `audio` | text / transcription / transcript | Source | at most one: text / transcription / transcript; forbidden: segments |
| `vibevoice-model-ready` | `input_ids`, `attention_mask`, `input_values`, `padding_mask`, `labels` | — | Prepared | — |

VibeVoice structured long-form ASR targets and multimodal prompt inputs. See the [data workflow](../../guides/speech-data.md).

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
| Training checkpoint | `microsoft/VibeVoice-ASR-HF` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `speech_recognition` | objective | `model.model.multi_modal_projector`, `model.model.language_model`, `model.lm_head` | `input_ids`, `attention_mask`, `input_values`, `padding_mask`, `labels` | `loss` |

The integration accepts its declared source or prepared contract directly. Call `model.validate_training_support()` first, then follow the
[training workflow](../../guides/training.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | [`microsoft/VibeVoice-ASR-HF`](https://huggingface.co/microsoft/VibeVoice-ASR-HF) |
| Hugging Face ID | [`microsoft/VibeVoice-ASR-HF`](https://huggingface.co/microsoft/VibeVoice-ASR-HF)<br>Repository availability verified through the Hugging Face model API on 2026-08-11; pin a revision before production use. |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.asr_vibevoice.modeling_asr_vibevoice.VibeVoiceForSpeechRecognition` |
| Configuration | `voicehub.models.asr_vibevoice.configuration_asr_vibevoice.VibeVoiceASRConfig` |
| Source provenance | `voicehub/architectures/vibevoice/source/SOURCE.json` |
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

### `VibeVoiceASRConfig`

<p class="vh-model-api-card__source-wrap"><a class="vh-model-api-card__source" href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_vibevoice/configuration_asr_vibevoice.py">View source</a></p>
<div class="vh-model-api-card__signature" markdown>

```text
VibeVoiceASRConfig(**config_kwargs)
```

</div>
<h4>Parameters</h4>
<div class="vh-model-api-card__parameters" markdown>
- `**config_kwargs` — Configuration fields validated by VibeVoiceASRConfig.
</div>
</section>

<section class="vh-model-api-card" data-vh-model-api-card="model" markdown>
<p class="vh-model-api-card__badge-wrap"><span class="vh-model-api-card__badge">Model</span></p>

### `VibeVoiceForSpeechRecognition`

<p class="vh-model-api-card__source-wrap"><a class="vh-model-api-card__source" href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_vibevoice/modeling_asr_vibevoice.py">View source</a></p>
<div class="vh-model-api-card__signature" markdown>

```text
AutoModelForSpeechRecognition.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='asr_vibevoice',
    config=None,
    **model_kwargs,
)
```

</div>
<h4>Parameters</h4>
<div class="vh-model-api-card__parameters" markdown>
- `pretrained_model_name_or_path` — Hub ID or compatible local directory.
- `model_type` — Canonical model type; use 'asr_vibevoice'.
- `config` — Optional preloaded VibeVoiceASRConfig instance.
- `**model_kwargs` — Model-specific loading arguments.
</div>
</section>

```python
from voicehub import get_model_spec

spec = get_model_spec('asr_vibevoice')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('asr_vibevoice')` |
| Load and run | `AutoModelForSpeechRecognition` |
| Configure | `VibeVoiceASRConfig` |
| Process | `AutoProcessor` |
| Model implementation | `VibeVoiceForSpeechRecognition` |
| Normalized output | `ASROutput` |
| Training contract | `get_training_spec('asr_vibevoice')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).

</div>

</div>

</div>
