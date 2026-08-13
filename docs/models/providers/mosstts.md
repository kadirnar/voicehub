---
description: Public API, checkpoint, training, and optimization guide for the mosstts integration.
hide:
  - toc
---

<div class="vh-model-detail" data-vh-model-detail data-model-type="mosstts" data-task="text-to-speech" data-training="native" data-parameter-count="8489841664" markdown>

<header class="vh-model-detail__hero" data-vh-model-hero markdown>

<p class="vh-model-detail__namespace" aria-label="Model repository"><span class="vh-model-detail__owner-avatar" aria-hidden="true">OM</span><a href="https://huggingface.co/OpenMOSS-Team">OpenMOSS-Team</a><span aria-hidden="true">/</span><strong>MOSS-TTS-v1.5</strong></p>

# MossTTS {.vh-model-title}

<p class="vh-model-detail__summary">Combines MOSS-TTS language, instruction, and quality controls without importing upstream demo code.</p>
<div class="vh-model-detail__tags" aria-label="Model metadata"><span class="vh-model-detail__chip" data-chip-kind="task">Text to speech</span><span class="vh-model-detail__chip" data-chip-kind="runtime">VoiceHub-native</span><span class="vh-model-detail__chip" data-chip-kind="architecture">moss-tts</span><span class="vh-model-detail__chip" data-chip-kind="parameters" aria-describedby="vh-model-parameters-note-mosstts">Parameters: 8.5B</span><span class="vh-model-detail__chip" data-chip-kind="language">Languages: zh, yue +29</span><span class="vh-model-detail__chip" data-chip-kind="training">Training: native</span><span class="vh-model-detail__chip" data-chip-kind="license">License: Checkpoint-specific</span></div>
<p class="vh-model-detail__parameter-note" id="vh-model-parameters-note-mosstts"><strong>Parameter metadata:</strong> Exact learned-parameter total for VoiceHub&#x27;s audited native primary graph at the registered default selection; separately loaded auxiliary models are excluded.</p>
<div class="vh-model-detail__actions" aria-label="Model actions">
<a class="vh-model-detail__action vh-model-detail__action--primary" href="#usage" data-vh-model-action="use">Use this model</a>
<button class="vh-model-detail__action vh-model-detail__copy" type="button" data-vh-copy-model-id data-model-id="OpenMOSS-Team/MOSS-TTS-v1.5" aria-describedby="vh-model-checkpoint-mosstts"><span data-vh-copy-model-id-label>Copy model ID</span></button>
<a class="vh-model-detail__action" href="https://huggingface.co/OpenMOSS-Team/MOSS-TTS-v1.5" data-vh-model-action="checkpoint">Checkpoint</a>
<details class="vh-model-detail__resources">
<summary class="vh-model-detail__action">Resources</summary>
<div class="vh-model-detail__resource-menu">
<a href="https://github.com/OpenMOSS/MOSS-TTS" data-vh-model-action="github">Upstream GitHub</a>
<a href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/mosstts/modeling_mosstts.py" data-vh-model-action="source">VoiceHub source</a>
<a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/mosstts.ipynb" data-vh-model-action="colab">Open in Colab</a>
</div>
</details>
</div>
</header>

<nav class="vh-model-detail__tabs" aria-label="Model sections"><a href="#usage" data-vh-model-tab="usage">Usage</a><a href="#overview" data-vh-model-tab="model-card" aria-current="location">Model card</a><a href="#paper-and-github" data-vh-model-tab="sources">Sources</a><a href="#training-and-optimization" data-vh-model-tab="training">Training</a><a href="#checkpoints-provenance-license-and-limitations" data-vh-model-tab="checkpoint">Checkpoint</a><a href="#public-api" data-vh-model-tab="api">Public API</a></nav>

<div class="vh-model-detail__layout" markdown>

<aside class="vh-model-detail__sidebar" data-vh-model-facts aria-labelledby="vh-model-facts-title-mosstts"><h2 id="vh-model-facts-title-mosstts">Model facts</h2><details class="vh-model-detail__facts-disclosure" data-vh-model-facts-disclosure aria-labelledby="vh-model-facts-title-mosstts" open><summary><span>Toggle model facts</span></summary><dl class="vh-model-detail__facts"><div><dt>Task</dt><dd>Text to speech</dd></div><div><dt>Parameters</dt><dd aria-describedby="vh-model-parameters-note-mosstts">8.5B</dd></div><div><dt>Architecture</dt><dd><code>moss-tts</code></dd></div><div><dt>Runtime</dt><dd>VoiceHub-native</dd></div><div><dt>Languages</dt><dd><details class="vh-model-detail__languages"><summary>31 documented codes</summary><span><code>zh</code> <code>yue</code> <code>en</code> <code>ar</code> <code>cs</code> <code>da</code> <code>de</code> <code>nl</code> <code>es</code> <code>fr</code> <code>fi</code> <code>el</code> <code>he</code> <code>hi</code> <code>hu</code> <code>ja</code> <code>it</code> <code>ko</code> <code>mk</code> <code>ms</code> <code>ru</code> <code>fa</code> <code>pl</code> <code>pt</code> <code>sv</code> <code>ro</code> <code>sw</code> <code>tl</code> <code>th</code> <code>tr</code> <code>vi</code></span></details></dd></div><div><dt>Capabilities</dt><dd><details class="vh-model-detail__capabilities"><summary>16 capabilities</summary><span><code>text-to-speech</code> <code>voice-cloning</code> <code>multilingual</code> <code>fine-tuning</code> <code>safetensors</code> <code>voicehub-native</code> <code>native-runtime</code> <code>delay-variant</code> <code>local-variant</code> <code>local-v1.5-variant</code> <code>realtime-variant</code> <code>raw-audio-fine-tuning</code> <code>preencoded-rvq-fine-tuning</code> <code>native-codec-v1</code> <code>native-codec-v2</code> <code>buffered-generation</code></span></details></dd></div><div><dt>Training</dt><dd><code>native</code></dd></div><div><dt>License</dt><dd>Checkpoint-specific</dd></div><div><dt>Default checkpoint</dt><dd id="vh-model-checkpoint-mosstts"><a href="https://huggingface.co/OpenMOSS-Team/MOSS-TTS-v1.5"><code>OpenMOSS-Team/MOSS-TTS-v1.5</code></a></dd></div></dl></details></aside>

<div class="vh-model-detail__main vh-model-detail__content" markdown>

## Usage

Complete the [VoiceHub installation](../../getting-started/installation.md) once,
then run this repository-authored example. Model pages intentionally contain no
package-install command.

This example is maintained against VoiceHub's public API; it is not copied from an upstream demo or package README.

**Model-specific path:** Combines MOSS-TTS language, instruction, and quality controls without importing upstream demo code.

**Inputs and controls:** Keep instructions descriptive and validate the requested language against the selected checkpoint.

```python
from pathlib import Path

from voicehub import AutoModelForTextToSpeech, TTSGenerationConfig

model = AutoModelForTextToSpeech.from_pretrained(
    'OpenMOSS-Team/MOSS-TTS-v1.5',
    model_type='mosstts',
    device="cuda",
    lazy_load=True,
)
output = model.generate(
    'VoiceHub keeps model integrations explicit and reproducible.',
    generation_config=TTSGenerationConfig(
        seed=42,
        output_file=Path("output.wav"),
    ),
    language="en",
    instruction="Calm, clear studio speech",
    quality="high",
)
print(output.file_path, output.sample_rate, output.metadata)
```

Use authorized recordings. Verify hardware needs and pin a revision in production.

## Overview

`mosstts` is a VoiceHub **text to speech**
integration. This page is generated from its registry contract. [Open the `mosstts` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/mosstts.ipynb).

| Property | Value |
| --- | --- |
| Task | Text to speech |
| Architecture | `moss-tts` |
| Runtime | `VoiceHub-native` |
| Languages | `zh`, `yue`, `en`, `ar`, … complete audited list below |
| Capabilities | `text-to-speech`, `voice-cloning`, `multilingual`, `fine-tuning`, `safetensors`, `voicehub-native`, `native-runtime`, `delay-variant`, `local-variant`, `local-v1.5-variant`, `realtime-variant`, `raw-audio-fine-tuning`, `preencoded-rvq-fine-tuning`, `native-codec-v1`, `native-codec-v2`, `buffered-generation` |
| Reusable components | — |
| Normalized output | `TTSOutput` |

### Language support

<details class="vh-language-support" markdown>
<summary>Supported language abbreviations</summary>

`zh`, `yue`, `en`, `ar`, `cs`, `da`, `de`, `nl`, `es`, `fr`, `fi`, `el`, `he`, `hi`, `hu`, `ja`, `it`, `ko`, `mk`, `ms`, `ru`, `fa`, `pl`, `pt`, `sv`, `ro`, `sw`, `tl`, `th`, `tr`, `vi`

</details>

## Paper and GitHub

- **Paper:** No dedicated upstream research paper is published for this integration.
- **Upstream GitHub:** [MOSS-TTS](https://github.com/OpenMOSS/MOSS-TTS)
- **VoiceHub source:** [VoiceHub model implementation](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/mosstts/modeling_mosstts.py)

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('mosstts')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `mosstts` |
| Configuration class | `MossTTSConfig` |
| Architecture class | `MossTTSForTextToSpeech` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'OpenMOSS-Team/MOSS-TTS-v1.5',
    model_type='mosstts',
)
print(type(processor).__name__)
```

## Inference

The Usage example returns `TTSOutput` through `AutoModelForTextToSpeech`.

### Input and output contract

| Property | Value |
| --- | --- |
| Readiness | `integrated-raw` |
| Data architecture | `codec-lm` |
| Sample rate | Model/checkpoint specific |
| Contract getter | `get_tts_dataset_spec('mosstts')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `raw-audio` | `text` | audio / waveform / audio_path | Source | at most one: audio / waveform / audio_path; forbidden: speech_tokens |
| `preencoded-rvq` | `text`, `speech_tokens` | — | Prepared | forbidden: audio, waveform, audio_path |

Autoregressive text/audio-token or codec-language-model data. See the [data workflow](../../guides/data-preparation.md).

## Training and optimization

Use `available_optimization_passes()` to discover reversible public passes.
Unsupported runtime or hardware fails closed before mutation.

### Training contract

| Property | Value |
| --- | --- |
| Support | `native` |
| Family | `causal-lm` |
| Recipe | `single-phase` |
| Default phase | `semantic_language_model` |
| Training checkpoint | `OpenMOSS-Team/MOSS-TTS-v1.5` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `semantic_language_model` | objective | `model` | `input_ids`, `attention_mask`, `labels` | `loss` |

The integration accepts its declared source or prepared contract directly. Call `model.validate_training_support()` first, then follow the
[training workflow](../../guides/training.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | [`OpenMOSS-Team/MOSS-TTS-v1.5`](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-v1.5) |
| Hugging Face ID | [`OpenMOSS-Team/MOSS-TTS-v1.5`](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-v1.5)<br>Repository availability verified through the Hugging Face model API on 2026-08-11; pin a revision before production use. |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.mosstts.modeling_mosstts.MossTTSForTextToSpeech` |
| Configuration | `voicehub.models.mosstts.configuration_mosstts.MossTTSConfig` |
| Source provenance | `voicehub/models/mosstts/source/SOURCE.json` |
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

### `MossTTSConfig`

<p class="vh-model-api-card__source-wrap"><a class="vh-model-api-card__source" href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/mosstts/configuration_mosstts.py">View source</a></p>
<div class="vh-model-api-card__signature" markdown>

```text
MossTTSConfig(**config_kwargs)
```

</div>
<h4>Parameters</h4>
<div class="vh-model-api-card__parameters" markdown>
- `**config_kwargs` — Configuration fields validated by MossTTSConfig.
</div>
</section>

<section class="vh-model-api-card" data-vh-model-api-card="model" markdown>
<p class="vh-model-api-card__badge-wrap"><span class="vh-model-api-card__badge">Model</span></p>

### `MossTTSForTextToSpeech`

<p class="vh-model-api-card__source-wrap"><a class="vh-model-api-card__source" href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/mosstts/modeling_mosstts.py">View source</a></p>
<div class="vh-model-api-card__signature" markdown>

```text
AutoModelForTextToSpeech.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='mosstts',
    config=None,
    **model_kwargs,
)
```

</div>
<h4>Parameters</h4>
<div class="vh-model-api-card__parameters" markdown>
- `pretrained_model_name_or_path` — Hub ID or compatible local directory.
- `model_type` — Canonical model type; use 'mosstts'.
- `config` — Optional preloaded MossTTSConfig instance.
- `**model_kwargs` — Model-specific loading arguments.
</div>
</section>

```python
from voicehub import get_model_spec

spec = get_model_spec('mosstts')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('mosstts')` |
| Load and run | `AutoModelForTextToSpeech` |
| Configure | `MossTTSConfig` |
| Process | `AutoProcessor` |
| Model implementation | `MossTTSForTextToSpeech` |
| Normalized output | `TTSOutput` |
| Training contract | `get_training_spec('mosstts')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).

</div>

</div>

</div>
