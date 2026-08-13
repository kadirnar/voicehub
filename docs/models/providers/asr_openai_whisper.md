---
description: Public API, checkpoint, training, and optimization guide for the asr_openai_whisper integration.
hide:
  - toc
---

<div class="vh-model-detail" data-vh-model-detail data-model-type="asr_openai_whisper" data-task="automatic-speech-recognition" data-training="native" data-parameter-count="241734912" markdown>

<header class="vh-model-detail__hero" data-vh-model-hero markdown>

<p class="vh-model-detail__namespace" aria-label="Model repository"><span class="vh-model-detail__owner-avatar" aria-hidden="true">OP</span><a href="https://huggingface.co/openai">openai</a><span aria-hidden="true">/</span><strong>whisper-small</strong></p>

# OpenAIWhisper {.vh-model-title}

<p class="vh-model-detail__summary">Runs the original OpenAI Whisper backend with deterministic beam decoding.</p>
<div class="vh-model-detail__tags" aria-label="Model metadata"><span class="vh-model-detail__chip" data-chip-kind="task">Automatic speech recognition</span><span class="vh-model-detail__chip" data-chip-kind="runtime">VoiceHub-native</span><span class="vh-model-detail__chip" data-chip-kind="architecture">whisper</span><span class="vh-model-detail__chip" data-chip-kind="parameters" aria-describedby="vh-model-parameters-note-asr_openai_whisper">Parameters: 241.7M</span><span class="vh-model-detail__chip" data-chip-kind="language">Languages: en, zh +97</span><span class="vh-model-detail__chip" data-chip-kind="training">Training: native</span><span class="vh-model-detail__chip" data-chip-kind="license">License: Checkpoint-specific</span></div>
<p class="vh-model-detail__parameter-note" id="vh-model-parameters-note-asr_openai_whisper"><strong>Parameter metadata:</strong> Exact Safetensors total reported by the Hugging Face model API for the registered default checkpoint, retrieved 2026-08-13.</p>
<div class="vh-model-detail__actions" aria-label="Model actions">
<a class="vh-model-detail__action vh-model-detail__action--primary" href="#usage" data-vh-model-action="use">Use this model</a>
<button class="vh-model-detail__action vh-model-detail__copy" type="button" data-vh-copy-model-id data-model-id="openai/whisper-small" aria-describedby="vh-model-checkpoint-asr_openai_whisper"><span data-vh-copy-model-id-label>Copy model ID</span></button>
<a class="vh-model-detail__action" href="https://huggingface.co/openai/whisper-small" data-vh-model-action="checkpoint">Checkpoint</a>
<details class="vh-model-detail__resources">
<summary class="vh-model-detail__action">Resources</summary>
<div class="vh-model-detail__resource-menu">
<a href="https://arxiv.org/abs/2212.04356" data-vh-model-action="paper">Paper</a>
<a href="https://github.com/openai/whisper" data-vh-model-action="github">Upstream GitHub</a>
<a href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_native/openai_whisper.py" data-vh-model-action="source">VoiceHub source</a>
<a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/asr_openai_whisper.ipynb" data-vh-model-action="colab">Open in Colab</a>
</div>
</details>
</div>
</header>

<nav class="vh-model-detail__tabs" aria-label="Model sections"><a href="#usage" data-vh-model-tab="usage">Usage</a><a href="#overview" data-vh-model-tab="model-card" aria-current="location">Model card</a><a href="#paper-and-github" data-vh-model-tab="sources">Sources</a><a href="#training-and-optimization" data-vh-model-tab="training">Training</a><a href="#checkpoints-provenance-license-and-limitations" data-vh-model-tab="checkpoint">Checkpoint</a><a href="#public-api" data-vh-model-tab="api">Public API</a></nav>

<div class="vh-model-detail__layout" markdown>

<aside class="vh-model-detail__sidebar" data-vh-model-facts aria-labelledby="vh-model-facts-title-asr_openai_whisper"><h2 id="vh-model-facts-title-asr_openai_whisper">Model facts</h2><details class="vh-model-detail__facts-disclosure" data-vh-model-facts-disclosure aria-labelledby="vh-model-facts-title-asr_openai_whisper" open><summary><span>Toggle model facts</span></summary><dl class="vh-model-detail__facts"><div><dt>Task</dt><dd>Automatic speech recognition</dd></div><div><dt>Parameters</dt><dd aria-describedby="vh-model-parameters-note-asr_openai_whisper">241.7M</dd></div><div><dt>Architecture</dt><dd><code>whisper</code></dd></div><div><dt>Runtime</dt><dd>VoiceHub-native</dd></div><div><dt>Languages</dt><dd><details class="vh-model-detail__languages"><summary>99 documented codes</summary><span><code>en</code> <code>zh</code> <code>de</code> <code>es</code> <code>ru</code> <code>ko</code> <code>fr</code> <code>ja</code> <code>pt</code> <code>tr</code> <code>pl</code> <code>ca</code> <code>nl</code> <code>ar</code> <code>sv</code> <code>it</code> <code>id</code> <code>hi</code> <code>fi</code> <code>vi</code> <code>he</code> <code>uk</code> <code>el</code> <code>ms</code> <code>cs</code> <code>ro</code> <code>da</code> <code>hu</code> <code>ta</code> <code>no</code> <code>th</code> <code>ur</code> <code>hr</code> <code>bg</code> <code>lt</code> <code>la</code> <code>mi</code> <code>ml</code> <code>cy</code> <code>sk</code> <code>te</code> <code>fa</code> <code>lv</code> <code>bn</code> <code>sr</code> <code>az</code> <code>sl</code> <code>kn</code> <code>et</code> <code>mk</code> <code>br</code> <code>eu</code> <code>is</code> <code>hy</code> <code>ne</code> <code>mn</code> <code>bs</code> <code>kk</code> <code>sq</code> <code>sw</code> <code>gl</code> <code>mr</code> <code>pa</code> <code>si</code> <code>km</code> <code>sn</code> <code>yo</code> <code>so</code> <code>af</code> <code>oc</code> <code>ka</code> <code>be</code> <code>tg</code> <code>sd</code> <code>gu</code> <code>am</code> <code>yi</code> <code>lo</code> <code>uz</code> <code>fo</code> <code>ht</code> <code>ps</code> <code>tk</code> <code>nn</code> <code>mt</code> <code>sa</code> <code>lb</code> <code>my</code> <code>bo</code> <code>tl</code> <code>mg</code> <code>as</code> <code>tt</code> <code>haw</code> <code>ln</code> <code>ha</code> <code>ba</code> <code>jw</code> <code>su</code></span></details></dd></div><div><dt>Capabilities</dt><dd><details class="vh-model-detail__capabilities"><summary>7 capabilities</summary><span><code>automatic-speech-recognition</code> <code>multilingual</code> <code>translation</code> <code>timestamps</code> <code>safetensors</code> <code>fine-tuning</code> <code>voicehub-native</code></span></details></dd></div><div><dt>Training</dt><dd><code>native</code></dd></div><div><dt>License</dt><dd>Checkpoint-specific</dd></div><div><dt>Default checkpoint</dt><dd id="vh-model-checkpoint-asr_openai_whisper"><a href="https://huggingface.co/openai/whisper-small"><code>openai/whisper-small</code></a></dd></div></dl></details></aside>

<div class="vh-model-detail__main vh-model-detail__content" markdown>

## Usage

Complete the [VoiceHub installation](../../getting-started/installation.md) once,
then run this repository-authored example. Model pages intentionally contain no
package-install command.

This example is maintained against VoiceHub's public API; it is not copied from an upstream demo or package README.

**Model-specific path:** Runs the original OpenAI Whisper backend with deterministic beam decoding.

**Inputs and controls:** This integration is distinct from native Whisper and faster-whisper even when they share an HF checkpoint ID.

```python
from pathlib import Path

from voicehub import AutoModelForSpeechRecognition

AUDIO_FILE = Path("speech.wav")
if not AUDIO_FILE.is_file():
    raise FileNotFoundError(AUDIO_FILE)

model = AutoModelForSpeechRecognition.from_pretrained(
    'openai/whisper-small',
    model_type='asr_openai_whisper',
    device="cuda",
    lazy_load=True,
)
output = model.transcribe(
    AUDIO_FILE,
    language="en",
    task="transcribe",
    num_beams=5,
)
print(output.text)
for segment in output.segments:
    print(segment.start, segment.end, segment.text, segment.confidence)
```

Use authorized recordings. Verify hardware needs and pin a revision in production.

## Overview

`asr_openai_whisper` is a VoiceHub **automatic speech recognition**
integration. This page is generated from its registry contract. [Open the `asr_openai_whisper` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/asr_openai_whisper.ipynb).

| Property | Value |
| --- | --- |
| Task | Automatic speech recognition |
| Architecture | `whisper` |
| Runtime | `VoiceHub-native` |
| Languages | `en`, `zh`, `de`, `es`, … complete audited list below |
| Capabilities | `automatic-speech-recognition`, `multilingual`, `translation`, `timestamps`, `safetensors`, `fine-tuning`, `voicehub-native` |
| Reusable components | — |
| Normalized output | `ASROutput` |

### Language support

<details class="vh-language-support" markdown>
<summary>Supported language abbreviations</summary>

`en`, `zh`, `de`, `es`, `ru`, `ko`, `fr`, `ja`, `pt`, `tr`, `pl`, `ca`, `nl`, `ar`, `sv`, `it`, `id`, `hi`, `fi`, `vi`, `he`, `uk`, `el`, `ms`, `cs`, `ro`, `da`, `hu`, `ta`, `no`, `th`, `ur`, `hr`, `bg`, `lt`, `la`, `mi`, `ml`, `cy`, `sk`, `te`, `fa`, `lv`, `bn`, `sr`, `az`, `sl`, `kn`, `et`, `mk`, `br`, `eu`, `is`, `hy`, `ne`, `mn`, `bs`, `kk`, `sq`, `sw`, `gl`, `mr`, `pa`, `si`, `km`, `sn`, `yo`, `so`, `af`, `oc`, `ka`, `be`, `tg`, `sd`, `gu`, `am`, `yi`, `lo`, `uz`, `fo`, `ht`, `ps`, `tk`, `nn`, `mt`, `sa`, `lb`, `my`, `bo`, `tl`, `mg`, `as`, `tt`, `haw`, `ln`, `ha`, `ba`, `jw`, `su`

</details>

## Paper and GitHub

- **Paper:** [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356)
- **Upstream GitHub:** [Whisper](https://github.com/openai/whisper)
- **VoiceHub source:** [VoiceHub model implementation](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_native/openai_whisper.py)

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('asr_openai_whisper')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `asr_openai_whisper` |
| Configuration class | `OpenAIWhisperConfig` |
| Architecture class | `OpenAIWhisperForSpeechRecognition` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'openai/whisper-small',
    model_type='asr_openai_whisper',
)
print(type(processor).__name__)
```

## Inference

The Usage example returns `ASROutput` through `AutoModelForSpeechRecognition`.

### Input and output contract

| Property | Value |
| --- | --- |
| Readiness | `integrated-raw` |
| Data architecture | `speech-sequence-to-sequence` |
| Sample rate | 16,000 Hz |
| Contract getter | `get_asr_dataset_spec('asr_openai_whisper')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `raw-audio` | `audio` | text / transcription / transcript | Source | at most one: text / transcription / transcript |
| `whisper-model-ready` | `input_features`, `labels` | — | Prepared | — |

OpenAI Whisper-compatible records trained through the native Whisper graph. See the [data workflow](../../guides/speech-data.md).

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
| Training checkpoint | `openai/whisper-small` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `speech_recognition` | objective | `model` | — | `loss` |

The integration accepts its declared source or prepared contract directly. Call `model.validate_training_support()` first, then follow the
[training workflow](../../guides/training.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | [`openai/whisper-small`](https://huggingface.co/openai/whisper-small) |
| Hugging Face ID | [`openai/whisper-small`](https://huggingface.co/openai/whisper-small)<br>Repository availability verified through the Hugging Face model API on 2026-08-11; pin a revision before production use. |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.asr_native.openai_whisper.OpenAIWhisperForSpeechRecognition` |
| Configuration | `voicehub.models.asr_native.configuration.OpenAIWhisperConfig` |
| Source provenance | No integration-specific bundled `SOURCE.json` is declared for this registry entry. |
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

### `OpenAIWhisperConfig`

<p class="vh-model-api-card__source-wrap"><a class="vh-model-api-card__source" href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_native/configuration.py">View source</a></p>
<div class="vh-model-api-card__signature" markdown>

```text
OpenAIWhisperConfig(**config_kwargs)
```

</div>
<h4>Parameters</h4>
<div class="vh-model-api-card__parameters" markdown>
- `**config_kwargs` — Configuration fields validated by OpenAIWhisperConfig.
</div>
</section>

<section class="vh-model-api-card" data-vh-model-api-card="model" markdown>
<p class="vh-model-api-card__badge-wrap"><span class="vh-model-api-card__badge">Model</span></p>

### `OpenAIWhisperForSpeechRecognition`

<p class="vh-model-api-card__source-wrap"><a class="vh-model-api-card__source" href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_native/openai_whisper.py">View source</a></p>
<div class="vh-model-api-card__signature" markdown>

```text
AutoModelForSpeechRecognition.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='asr_openai_whisper',
    config=None,
    **model_kwargs,
)
```

</div>
<h4>Parameters</h4>
<div class="vh-model-api-card__parameters" markdown>
- `pretrained_model_name_or_path` — Hub ID or compatible local directory.
- `model_type` — Canonical model type; use 'asr_openai_whisper'.
- `config` — Optional preloaded OpenAIWhisperConfig instance.
- `**model_kwargs` — Model-specific loading arguments.
</div>
</section>

```python
from voicehub import get_model_spec

spec = get_model_spec('asr_openai_whisper')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('asr_openai_whisper')` |
| Load and run | `AutoModelForSpeechRecognition` |
| Configure | `OpenAIWhisperConfig` |
| Process | `AutoProcessor` |
| Model implementation | `OpenAIWhisperForSpeechRecognition` |
| Normalized output | `ASROutput` |
| Training contract | `get_training_spec('asr_openai_whisper')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).

</div>

</div>

</div>
