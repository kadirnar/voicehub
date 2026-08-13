---
description: Public API, checkpoint, training, and optimization guide for the gptsovits integration.
hide:
  - toc
---

<div class="vh-model-detail" data-vh-model-detail data-model-type="gptsovits" data-task="text-to-speech" data-training="preprocessed" data-parameter-count="128916482" markdown>

<header class="vh-model-detail__hero" data-vh-model-hero markdown>

<p class="vh-model-detail__namespace" aria-label="Model repository"><span class="vh-model-detail__owner-avatar" aria-hidden="true">LJ</span><a href="https://huggingface.co/lj1995">lj1995</a><span aria-hidden="true">/</span><strong>GPT-SoVITS</strong></p>

# GPTSoVITS {.vh-model-title}

<p class="vh-model-detail__summary">Defines both target and prompt languages for GPT-SoVITS zero-shot voice prompting.</p>
<div class="vh-model-detail__tags" aria-label="Model metadata"><span class="vh-model-detail__chip" data-chip-kind="task">Text to speech</span><span class="vh-model-detail__chip" data-chip-kind="runtime">VoiceHub-native</span><span class="vh-model-detail__chip" data-chip-kind="architecture">gptsovits</span><span class="vh-model-detail__chip" data-chip-kind="parameters" aria-describedby="vh-model-parameters-note-gptsovits">Parameters: 128.9M</span><span class="vh-model-detail__chip" data-chip-kind="language">Languages: zh, en +3</span><span class="vh-model-detail__chip" data-chip-kind="training">Training: preprocessed</span><span class="vh-model-detail__chip" data-chip-kind="license">License: Checkpoint-specific</span></div>
<p class="vh-model-detail__parameter-note" id="vh-model-parameters-note-gptsovits"><strong>Parameter metadata:</strong> Exact learned-parameter total for VoiceHub&#x27;s audited native primary graph at the registered default selection; separately loaded auxiliary models are excluded.</p>
<div class="vh-model-detail__actions" aria-label="Model actions">
<a class="vh-model-detail__action vh-model-detail__action--primary" href="#usage" data-vh-model-action="use">Use this model</a>
<button class="vh-model-detail__action vh-model-detail__copy" type="button" data-vh-copy-model-id data-model-id="lj1995/GPT-SoVITS" aria-describedby="vh-model-checkpoint-gptsovits"><span data-vh-copy-model-id-label>Copy model ID</span></button>
<a class="vh-model-detail__action" href="https://huggingface.co/lj1995/GPT-SoVITS" data-vh-model-action="checkpoint">Checkpoint</a>
<details class="vh-model-detail__resources">
<summary class="vh-model-detail__action">Resources</summary>
<div class="vh-model-detail__resource-menu">
<a href="https://github.com/RVC-Boss/GPT-SoVITS" data-vh-model-action="github">Upstream GitHub</a>
<a href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/gptsovits/modeling_gptsovits.py" data-vh-model-action="source">VoiceHub source</a>
<a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/gptsovits.ipynb" data-vh-model-action="colab">Open in Colab</a>
</div>
</details>
</div>
</header>

<nav class="vh-model-detail__tabs" aria-label="Model sections"><a href="#usage" data-vh-model-tab="usage">Usage</a><a href="#overview" data-vh-model-tab="model-card" aria-current="location">Model card</a><a href="#paper-and-github" data-vh-model-tab="sources">Sources</a><a href="#training-and-optimization" data-vh-model-tab="training">Training</a><a href="#checkpoints-provenance-license-and-limitations" data-vh-model-tab="checkpoint">Checkpoint</a><a href="#public-api" data-vh-model-tab="api">Public API</a></nav>

<div class="vh-model-detail__layout" markdown>

<aside class="vh-model-detail__sidebar" data-vh-model-facts aria-labelledby="vh-model-facts-title-gptsovits"><h2 id="vh-model-facts-title-gptsovits">Model facts</h2><details class="vh-model-detail__facts-disclosure" data-vh-model-facts-disclosure aria-labelledby="vh-model-facts-title-gptsovits" open><summary><span>Toggle model facts</span></summary><dl class="vh-model-detail__facts"><div><dt>Task</dt><dd>Text to speech</dd></div><div><dt>Parameters</dt><dd aria-describedby="vh-model-parameters-note-gptsovits">128.9M</dd></div><div><dt>Architecture</dt><dd><code>gptsovits</code></dd></div><div><dt>Runtime</dt><dd>VoiceHub-native</dd></div><div><dt>Languages</dt><dd><details class="vh-model-detail__languages"><summary>5 documented codes</summary><span><code>zh</code> <code>en</code> <code>ja</code> <code>ko</code> <code>yue</code></span></details></dd></div><div><dt>Capabilities</dt><dd><details class="vh-model-detail__capabilities"><summary>14 capabilities</summary><span><code>text-to-speech</code> <code>voice-cloning</code> <code>multilingual</code> <code>fine-tuning</code> <code>safetensors</code> <code>voicehub-native</code> <code>native-runtime</code> <code>preprocessed-training</code> <code>gpt-sovits-v1</code> <code>gpt-sovits-v2</code> <code>gpt-sovits-v2-pro</code> <code>gpt-sovits-v2-pro-plus</code> <code>prepared-pro-speaker-conditioning</code> <code>variant-aware-safetensors-export</code></span></details></dd></div><div><dt>Training</dt><dd><code>preprocessed</code></dd></div><div><dt>License</dt><dd>Checkpoint-specific</dd></div><div><dt>Default checkpoint</dt><dd id="vh-model-checkpoint-gptsovits"><a href="https://huggingface.co/lj1995/GPT-SoVITS"><code>lj1995/GPT-SoVITS</code></a></dd></div></dl></details></aside>

<div class="vh-model-detail__main vh-model-detail__content" markdown>

## Usage

Complete the [VoiceHub installation](../../getting-started/installation.md) once,
then run this repository-authored example. Model pages intentionally contain no
package-install command.

This example is maintained against VoiceHub's public API; it is not copied from an upstream demo or package README.

**Model-specific path:** Defines both target and prompt languages for GPT-SoVITS zero-shot voice prompting.

**Inputs and controls:** Use the language codes accepted by the selected GPT-SoVITS checkpoint and an exact prompt transcript.

```python
from pathlib import Path

from voicehub import AutoModelForTextToSpeech, TTSGenerationConfig

REFERENCE_AUDIO = Path("reference.wav")
REFERENCE_TEXT = "The reference transcript must exactly match the authorized audio."
if not REFERENCE_AUDIO.is_file():
    raise FileNotFoundError(REFERENCE_AUDIO)

model = AutoModelForTextToSpeech.from_pretrained(
    'lj1995/GPT-SoVITS',
    model_type='gptsovits',
    device="cuda",
    lazy_load=True,
)
output = model.generate(
    'VoiceHub keeps model integrations explicit and reproducible.',
    generation_config=TTSGenerationConfig(
        seed=42,
        output_file=Path("output.wav"),
    ),
    text_language="en",
    speaker_audio_path=str(REFERENCE_AUDIO),
    prompt_language="en",
    prompt_text=REFERENCE_TEXT,
    text_split_method="cut5",
)
print(output.file_path, output.sample_rate, output.metadata)
```

Use authorized recordings. Verify hardware needs and pin a revision in production.

## Overview

`gptsovits` is a VoiceHub **text to speech**
integration. This page is generated from its registry contract. [Open the `gptsovits` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/gptsovits.ipynb).

| Property | Value |
| --- | --- |
| Task | Text to speech |
| Architecture | `gptsovits` |
| Runtime | `VoiceHub-native` |
| Languages | `zh`, `en`, `ja`, `ko`, `yue` |
| Capabilities | `text-to-speech`, `voice-cloning`, `multilingual`, `fine-tuning`, `safetensors`, `voicehub-native`, `native-runtime`, `preprocessed-training`, `gpt-sovits-v1`, `gpt-sovits-v2`, `gpt-sovits-v2-pro`, `gpt-sovits-v2-pro-plus`, `prepared-pro-speaker-conditioning`, `variant-aware-safetensors-export` |
| Reusable components | — |
| Normalized output | `TTSOutput` |

### Language support

<details class="vh-language-support" markdown>
<summary>Supported language abbreviations</summary>

`zh`, `en`, `ja`, `ko`, `yue`

Korean and Cantonese support applies to V2 and later variants.

</details>

## Paper and GitHub

- **Paper:** No dedicated upstream research paper is published for this integration.
- **Upstream GitHub:** [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
- **VoiceHub source:** [VoiceHub model implementation](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/gptsovits/modeling_gptsovits.py)

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('gptsovits')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `gptsovits` |
| Configuration class | `GPTSoVITSConfig` |
| Architecture class | `GPTSoVITSForTextToSpeech` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'lj1995/GPT-SoVITS',
    model_type='gptsovits',
)
print(type(processor).__name__)
```

## Inference

The Usage example returns `TTSOutput` through `AutoModelForTextToSpeech`.

### Input and output contract

| Property | Value |
| --- | --- |
| Readiness | `preprocessed` |
| Data architecture | `hybrid` |
| Sample rate | 32,000 Hz |
| Contract getter | `get_tts_dataset_spec('gptsovits')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `s1-preprocessed` | `phoneme_ids`, `semantic_ids`, `bert_features` | — | Prepared | — |
| `s2-preprocessed` | `ssl_features`, `spectrogram`, `audio_values`, `phoneme_ids` | — | Prepared | — |
| `s2-pro-preprocessed` | `ssl_features`, `spectrogram`, `audio_values`, `phoneme_ids`, `speaker_embedding` | — | Prepared | — |

Multi-component language-model, diffusion, acoustic, or GAN data. See the [data workflow](../../guides/data-preparation.md).

## Training and optimization

Use `available_optimization_passes()` to discover reversible public passes.
Unsupported runtime or hardware fails closed before mutation.

### Training contract

| Property | Value |
| --- | --- |
| Support | `preprocessed` |
| Family | `composite` |
| Recipe | `adversarial` |
| Default phase | `s1` |
| Training checkpoint | `lj1995/GPT-SoVITS` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `s1` | objective | `training_model.s1` | `phoneme_ids`, `phoneme_lengths`, `semantic_ids`, `semantic_lengths`, `bert_features` | `loss` |
| `s2_generator` | generator | `training_model.s2.generator` | `ssl_features`, `spectrogram`, `spectrogram_lengths`, `audio_values`, `phoneme_ids`, `phoneme_lengths` | `loss` |
| `s2_discriminator` | discriminator | `training_model.s2.discriminator` | `ssl_features`, `spectrogram`, `spectrogram_lengths`, `audio_values`, `phoneme_ids`, `phoneme_lengths` | `loss` |

Prepare the exact tensors listed in the data contract before this step. Call `model.validate_training_support()` first, then follow the
[training workflow](../../guides/training.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | [`lj1995/GPT-SoVITS`](https://huggingface.co/lj1995/GPT-SoVITS) |
| Hugging Face ID | [`lj1995/GPT-SoVITS`](https://huggingface.co/lj1995/GPT-SoVITS)<br>Repository availability verified through the Hugging Face model API on 2026-08-11; pin a revision before production use. |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.gptsovits.modeling_gptsovits.GPTSoVITSForTextToSpeech` |
| Configuration | `voicehub.models.gptsovits.configuration_gptsovits.GPTSoVITSConfig` |
| Source provenance | `voicehub/models/gptsovits/source/SOURCE.json` |
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

### `GPTSoVITSConfig`

<p class="vh-model-api-card__source-wrap"><a class="vh-model-api-card__source" href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/gptsovits/configuration_gptsovits.py">View source</a></p>
<div class="vh-model-api-card__signature" markdown>

```text
GPTSoVITSConfig(**config_kwargs)
```

</div>
<h4>Parameters</h4>
<div class="vh-model-api-card__parameters" markdown>
- `**config_kwargs` — Configuration fields validated by GPTSoVITSConfig.
</div>
</section>

<section class="vh-model-api-card" data-vh-model-api-card="model" markdown>
<p class="vh-model-api-card__badge-wrap"><span class="vh-model-api-card__badge">Model</span></p>

### `GPTSoVITSForTextToSpeech`

<p class="vh-model-api-card__source-wrap"><a class="vh-model-api-card__source" href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/gptsovits/modeling_gptsovits.py">View source</a></p>
<div class="vh-model-api-card__signature" markdown>

```text
AutoModelForTextToSpeech.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='gptsovits',
    config=None,
    **model_kwargs,
)
```

</div>
<h4>Parameters</h4>
<div class="vh-model-api-card__parameters" markdown>
- `pretrained_model_name_or_path` — Hub ID or compatible local directory.
- `model_type` — Canonical model type; use 'gptsovits'.
- `config` — Optional preloaded GPTSoVITSConfig instance.
- `**model_kwargs` — Model-specific loading arguments.
</div>
</section>

```python
from voicehub import get_model_spec

spec = get_model_spec('gptsovits')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('gptsovits')` |
| Load and run | `AutoModelForTextToSpeech` |
| Configure | `GPTSoVITSConfig` |
| Process | `AutoProcessor` |
| Model implementation | `GPTSoVITSForTextToSpeech` |
| Normalized output | `TTSOutput` |
| Training contract | `get_training_spec('gptsovits')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).

</div>

</div>

</div>
