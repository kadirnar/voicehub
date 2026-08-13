---
description: Public API, checkpoint, training, and optimization guide for the fishtts integration.
hide:
  - toc
---

<div class="vh-model-detail" data-vh-model-detail data-model-type="fishtts" data-task="text-to-speech" data-training="preprocessed" data-parameter-count="4561852416" markdown>

<header class="vh-model-detail__hero" data-vh-model-hero markdown>

<p class="vh-model-detail__namespace" aria-label="Model repository"><span class="vh-model-detail__owner-avatar" aria-hidden="true">FI</span><a href="https://huggingface.co/fishaudio">fishaudio</a><span aria-hidden="true">/</span><strong>s2-pro</strong></p>

# FishTTS {.vh-model-title}

<p class="vh-model-detail__summary">Pairs Fish S2 reference audio and text while keeping semantic sampling bounded.</p>
<div class="vh-model-detail__tags" aria-label="Model metadata"><span class="vh-model-detail__chip" data-chip-kind="task">Text to speech</span><span class="vh-model-detail__chip" data-chip-kind="runtime">VoiceHub-native</span><span class="vh-model-detail__chip" data-chip-kind="architecture">fish-s2</span><span class="vh-model-detail__chip" data-chip-kind="parameters" aria-describedby="vh-model-parameters-note-fishtts">Parameters: 4.6B</span><span class="vh-model-detail__chip" data-chip-kind="language">Languages: zh, en +81</span><span class="vh-model-detail__chip" data-chip-kind="training">Training: preprocessed</span><span class="vh-model-detail__chip" data-chip-kind="license">License: Fish-Audio-Research-License</span></div>
<p class="vh-model-detail__parameter-note" id="vh-model-parameters-note-fishtts"><strong>Parameter metadata:</strong> Exact learned-parameter total for VoiceHub&#x27;s audited native primary graph at the registered default selection; separately loaded auxiliary models are excluded.</p>
<div class="vh-model-detail__actions" aria-label="Model actions">
<a class="vh-model-detail__action vh-model-detail__action--primary" href="#usage" data-vh-model-action="use">Use this model</a>
<button class="vh-model-detail__action vh-model-detail__copy" type="button" data-vh-copy-model-id data-model-id="fishaudio/s2-pro" aria-describedby="vh-model-checkpoint-fishtts"><span data-vh-copy-model-id-label>Copy model ID</span></button>
<a class="vh-model-detail__action" href="https://huggingface.co/fishaudio/s2-pro" data-vh-model-action="checkpoint">Checkpoint</a>
<details class="vh-model-detail__resources">
<summary class="vh-model-detail__action">Resources</summary>
<div class="vh-model-detail__resource-menu">
<a href="https://arxiv.org/abs/2411.01156" data-vh-model-action="paper">Paper</a>
<a href="https://github.com/fishaudio/fish-speech" data-vh-model-action="github">Upstream GitHub</a>
<a href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/fishtts/modeling_fishtts.py" data-vh-model-action="source">VoiceHub source</a>
<a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/fishtts.ipynb" data-vh-model-action="colab">Open in Colab</a>
</div>
</details>
</div>
</header>

<nav class="vh-model-detail__tabs" aria-label="Model sections"><a href="#usage" data-vh-model-tab="usage">Usage</a><a href="#overview" data-vh-model-tab="model-card" aria-current="location">Model card</a><a href="#paper-and-github" data-vh-model-tab="sources">Sources</a><a href="#training-and-optimization" data-vh-model-tab="training">Training</a><a href="#checkpoints-provenance-license-and-limitations" data-vh-model-tab="checkpoint">Checkpoint</a><a href="#public-api" data-vh-model-tab="api">Public API</a></nav>

<div class="vh-model-detail__layout" markdown>

<aside class="vh-model-detail__sidebar" data-vh-model-facts aria-labelledby="vh-model-facts-title-fishtts"><h2 id="vh-model-facts-title-fishtts">Model facts</h2><details class="vh-model-detail__facts-disclosure" data-vh-model-facts-disclosure aria-labelledby="vh-model-facts-title-fishtts" open><summary><span>Toggle model facts</span></summary><dl class="vh-model-detail__facts"><div><dt>Task</dt><dd>Text to speech</dd></div><div><dt>Parameters</dt><dd aria-describedby="vh-model-parameters-note-fishtts">4.6B</dd></div><div><dt>Architecture</dt><dd><code>fish-s2</code></dd></div><div><dt>Runtime</dt><dd>VoiceHub-native</dd></div><div><dt>Languages</dt><dd><details class="vh-model-detail__languages"><summary>83 documented codes</summary><span><code>zh</code> <code>en</code> <code>ja</code> <code>ko</code> <code>es</code> <code>pt</code> <code>ar</code> <code>ru</code> <code>fr</code> <code>de</code> <code>sv</code> <code>it</code> <code>tr</code> <code>no</code> <code>nl</code> <code>cy</code> <code>eu</code> <code>ca</code> <code>da</code> <code>gl</code> <code>ta</code> <code>hu</code> <code>fi</code> <code>pl</code> <code>et</code> <code>hi</code> <code>la</code> <code>ur</code> <code>th</code> <code>vi</code> <code>jw</code> <code>bn</code> <code>yo</code> <code>sl</code> <code>cs</code> <code>sw</code> <code>nn</code> <code>he</code> <code>ms</code> <code>uk</code> <code>id</code> <code>kk</code> <code>bg</code> <code>lv</code> <code>my</code> <code>tl</code> <code>sk</code> <code>ne</code> <code>fa</code> <code>af</code> <code>el</code> <code>bo</code> <code>hr</code> <code>ro</code> <code>sn</code> <code>mi</code> <code>yi</code> <code>am</code> <code>be</code> <code>km</code> <code>is</code> <code>az</code> <code>sd</code> <code>br</code> <code>sq</code> <code>ps</code> <code>mn</code> <code>ht</code> <code>ml</code> <code>sr</code> <code>sa</code> <code>te</code> <code>ka</code> <code>bs</code> <code>pa</code> <code>lt</code> <code>kn</code> <code>si</code> <code>hy</code> <code>mr</code> <code>as</code> <code>gu</code> <code>fo</code></span></details></dd></div><div><dt>Capabilities</dt><dd><details class="vh-model-detail__capabilities"><summary>9 capabilities</summary><span><code>text-to-speech</code> <code>voice-cloning</code> <code>multilingual</code> <code>fine-tuning</code> <code>safetensors</code> <code>voicehub-native</code> <code>native-runtime</code> <code>preprocessed-training</code> <code>noncommercial</code></span></details></dd></div><div><dt>Training</dt><dd><code>preprocessed</code></dd></div><div><dt>License</dt><dd><a href="https://github.com/fishaudio/fish-speech">Fish-Audio-Research-License</a></dd></div><div><dt>Default checkpoint</dt><dd id="vh-model-checkpoint-fishtts"><a href="https://huggingface.co/fishaudio/s2-pro"><code>fishaudio/s2-pro</code></a></dd></div></dl></details></aside>

<div class="vh-model-detail__main vh-model-detail__content" markdown>

## Usage

Complete the [VoiceHub installation](../../getting-started/installation.md) once,
then run this repository-authored example. Model pages intentionally contain no
package-install command.

This example is maintained against VoiceHub's public API; it is not copied from an upstream demo or package README.

**Model-specific path:** Pairs Fish S2 reference audio and text while keeping semantic sampling bounded.

**Inputs and controls:** Use either reference audio or precomputed codes, never both; each requires a matching transcript.

```python
from pathlib import Path

from voicehub import AutoModelForTextToSpeech, TTSGenerationConfig

REFERENCE_AUDIO = Path("reference.wav")
REFERENCE_TEXT = "The reference transcript must exactly match the authorized audio."
if not REFERENCE_AUDIO.is_file():
    raise FileNotFoundError(REFERENCE_AUDIO)

model = AutoModelForTextToSpeech.from_pretrained(
    'fishaudio/s2-pro',
    model_type='fishtts',
    device="cuda",
    lazy_load=True,
)
output = model.generate(
    'VoiceHub keeps model integrations explicit and reproducible.',
    generation_config=TTSGenerationConfig(
        seed=42,
        output_file=Path("output.wav"),
    ),
    speaker_audio_path=str(REFERENCE_AUDIO),
    reference_text=REFERENCE_TEXT,
    top_p=0.8,
    temperature=0.8,
    iterative_prompt=True,
)
print(output.file_path, output.sample_rate, output.metadata)
```

Use authorized recordings. Verify hardware needs and pin a revision in production.

## Overview

`fishtts` is a VoiceHub **text to speech**
integration. This page is generated from its registry contract. [Open the `fishtts` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/fishtts.ipynb).

| Property | Value |
| --- | --- |
| Task | Text to speech |
| Architecture | `fish-s2` |
| Runtime | `VoiceHub-native` |
| Languages | `zh`, `en`, `ja`, `ko`, … complete audited list below |
| Capabilities | `text-to-speech`, `voice-cloning`, `multilingual`, `fine-tuning`, `safetensors`, `voicehub-native`, `native-runtime`, `preprocessed-training`, `noncommercial` |
| Reusable components | `dac` |
| Normalized output | `TTSOutput` |

### Language support

<details class="vh-language-support" markdown>
<summary>Supported language abbreviations</summary>

`zh`, `en`, `ja`, `ko`, `es`, `pt`, `ar`, `ru`, `fr`, `de`, `sv`, `it`, `tr`, `no`, `nl`, `cy`, `eu`, `ca`, `da`, `gl`, `ta`, `hu`, `fi`, `pl`, `et`, `hi`, `la`, `ur`, `th`, `vi`, `jw`, `bn`, `yo`, `sl`, `cs`, `sw`, `nn`, `he`, `ms`, `uk`, `id`, `kk`, `bg`, `lv`, `my`, `tl`, `sk`, `ne`, `fa`, `af`, `el`, `bo`, `hr`, `ro`, `sn`, `mi`, `yi`, `am`, `be`, `km`, `is`, `az`, `sd`, `br`, `sq`, `ps`, `mn`, `ht`, `ml`, `sr`, `sa`, `te`, `ka`, `bs`, `pa`, `lt`, `kn`, `si`, `hy`, `mr`, `as`, `gu`, `fo`

</details>

## Paper and GitHub

- **Paper:** [Fish-Speech: Leveraging Large Language Models for Advanced Multilingual TTS](https://arxiv.org/abs/2411.01156)
- **Upstream GitHub:** [Fish Speech](https://github.com/fishaudio/fish-speech)
- **VoiceHub source:** [VoiceHub model implementation](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/fishtts/modeling_fishtts.py)

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('fishtts')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `fishtts` |
| Configuration class | `FishTTSConfig` |
| Architecture class | `FishTTSForTextToSpeech` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'fishaudio/s2-pro',
    model_type='fishtts',
)
print(type(processor).__name__)
```

## Inference

The Usage example returns `TTSOutput` through `AutoModelForTextToSpeech`.

### Input and output contract

| Property | Value |
| --- | --- |
| Readiness | `preprocessed` |
| Data architecture | `codec-lm` |
| Sample rate | 44,100 Hz |
| Contract getter | `get_tts_dataset_spec('fishtts')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `semantic-tokens` | `labels` | tokens / inputs | Prepared | — |

Autoregressive text/audio-token or codec-language-model data. See the [data workflow](../../guides/data-preparation.md).

## Training and optimization

Use `available_optimization_passes()` to discover reversible public passes.
Unsupported runtime or hardware fails closed before mutation.

### Training contract

| Property | Value |
| --- | --- |
| Support | `preprocessed` |
| Family | `causal-lm` |
| Recipe | `single-phase` |
| Default phase | `semantic` |
| Training checkpoint | `fishaudio/s2-pro` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `semantic` | objective | `model` | `inputs`, `labels` | `loss`, `base_loss`, `semantic_loss` |

Prepare the exact tensors listed in the data contract before this step. Call `model.validate_training_support()` first, then follow the
[training workflow](../../guides/training.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | [`fishaudio/s2-pro`](https://huggingface.co/fishaudio/s2-pro) |
| Hugging Face ID | [`fishaudio/s2-pro`](https://huggingface.co/fishaudio/s2-pro)<br>Repository availability verified through the Hugging Face model API on 2026-08-11; pin a revision before production use. |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.fishtts.modeling_fishtts.FishTTSForTextToSpeech` |
| Configuration | `voicehub.models.fishtts.configuration_fishtts.FishTTSConfig` |
| Source provenance | `voicehub/models/fishtts/source/SOURCE.json` |
| License | [Fish-Audio-Research-License](https://github.com/fishaudio/fish-speech) |

Fine-tuned checkpoints are derivative works. Commercial use requires a separate written Fish Audio license. Distribution must include the Fish Audio Research License, retain its exact copyright notice, and prominently display “Built with Fish Audio”. The license also restricts using materials, derivatives, or outputs to create or improve non-Fish foundational generative-AI models. Commercial use: **not allowed**.

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

### `FishTTSConfig`

<p class="vh-model-api-card__source-wrap"><a class="vh-model-api-card__source" href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/fishtts/configuration_fishtts.py">View source</a></p>
<div class="vh-model-api-card__signature" markdown>

```text
FishTTSConfig(**config_kwargs)
```

</div>
<h4>Parameters</h4>
<div class="vh-model-api-card__parameters" markdown>
- `**config_kwargs` — Configuration fields validated by FishTTSConfig.
</div>
</section>

<section class="vh-model-api-card" data-vh-model-api-card="model" markdown>
<p class="vh-model-api-card__badge-wrap"><span class="vh-model-api-card__badge">Model</span></p>

### `FishTTSForTextToSpeech`

<p class="vh-model-api-card__source-wrap"><a class="vh-model-api-card__source" href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/fishtts/modeling_fishtts.py">View source</a></p>
<div class="vh-model-api-card__signature" markdown>

```text
AutoModelForTextToSpeech.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='fishtts',
    config=None,
    **model_kwargs,
)
```

</div>
<h4>Parameters</h4>
<div class="vh-model-api-card__parameters" markdown>
- `pretrained_model_name_or_path` — Hub ID or compatible local directory.
- `model_type` — Canonical model type; use 'fishtts'.
- `config` — Optional preloaded FishTTSConfig instance.
- `**model_kwargs` — Model-specific loading arguments.
</div>
</section>

```python
from voicehub import get_model_spec

spec = get_model_spec('fishtts')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('fishtts')` |
| Load and run | `AutoModelForTextToSpeech` |
| Configure | `FishTTSConfig` |
| Process | `AutoProcessor` |
| Model implementation | `FishTTSForTextToSpeech` |
| Normalized output | `TTSOutput` |
| Training contract | `get_training_spec('fishtts')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).

</div>

</div>

</div>
