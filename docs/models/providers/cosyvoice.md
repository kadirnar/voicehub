---
description: Public API, checkpoint, training, and optimization guide for the cosyvoice integration.
hide:
  - toc
---

<div class="vh-model-detail" data-vh-model-detail data-model-type="cosyvoice" data-task="text-to-speech" data-training="custom" data-parameter-count="859185455" markdown>

<header class="vh-model-detail__hero" data-vh-model-hero markdown>

<p class="vh-model-detail__namespace" aria-label="Model repository"><span class="vh-model-detail__owner-avatar" aria-hidden="true">FA</span><a href="https://huggingface.co/FunAudioLLM">FunAudioLLM</a><span aria-hidden="true">/</span><strong>Fun-CosyVoice3-0.5B-2512</strong></p>

# CosyVoice {.vh-model-title}

<p class="vh-model-detail__summary">Loads the required 192-value speaker embedding from a reviewable JSON file.</p>
<div class="vh-model-detail__tags" aria-label="Model metadata"><span class="vh-model-detail__chip" data-chip-kind="task">Text to speech</span><span class="vh-model-detail__chip" data-chip-kind="runtime">VoiceHub-native</span><span class="vh-model-detail__chip" data-chip-kind="architecture">cosyvoice-native</span><span class="vh-model-detail__chip" data-chip-kind="parameters" aria-describedby="vh-model-parameters-note-cosyvoice">Parameters: 859.2M</span><span class="vh-model-detail__chip" data-chip-kind="language">Languages: zh, en +7</span><span class="vh-model-detail__chip" data-chip-kind="training">Training: custom</span><span class="vh-model-detail__chip" data-chip-kind="license">License: Checkpoint-specific</span></div>
<p class="vh-model-detail__parameter-note" id="vh-model-parameters-note-cosyvoice"><strong>Parameter metadata:</strong> Exact learned-parameter total for VoiceHub&#x27;s audited native primary graph at the registered default selection; separately loaded auxiliary models are excluded.</p>
<div class="vh-model-detail__actions" aria-label="Model actions">
<a class="vh-model-detail__action vh-model-detail__action--primary" href="#usage" data-vh-model-action="use">Use this model</a>
<button class="vh-model-detail__action vh-model-detail__copy" type="button" data-vh-copy-model-id data-model-id="FunAudioLLM/Fun-CosyVoice3-0.5B-2512" aria-describedby="vh-model-checkpoint-cosyvoice"><span data-vh-copy-model-id-label>Copy model ID</span></button>
<a class="vh-model-detail__action" href="https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512" data-vh-model-action="checkpoint">Checkpoint</a>
<details class="vh-model-detail__resources">
<summary class="vh-model-detail__action">Resources</summary>
<div class="vh-model-detail__resource-menu">
<a href="https://arxiv.org/abs/2407.05407" data-vh-model-action="paper">Paper</a>
<a href="https://github.com/FunAudioLLM/CosyVoice" data-vh-model-action="github">Upstream GitHub</a>
<a href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/cosyvoice/modeling_cosyvoice.py" data-vh-model-action="source">VoiceHub source</a>
<a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/cosyvoice.ipynb" data-vh-model-action="colab">Open in Colab</a>
</div>
</details>
</div>
</header>

<nav class="vh-model-detail__tabs" aria-label="Model sections"><a href="#usage" data-vh-model-tab="usage">Usage</a><a href="#overview" data-vh-model-tab="model-card" aria-current="location">Model card</a><a href="#paper-and-github" data-vh-model-tab="sources">Sources</a><a href="#training-and-optimization" data-vh-model-tab="training">Training</a><a href="#checkpoints-provenance-license-and-limitations" data-vh-model-tab="checkpoint">Checkpoint</a><a href="#public-api" data-vh-model-tab="api">Public API</a></nav>

<div class="vh-model-detail__layout" markdown>

<aside class="vh-model-detail__sidebar" data-vh-model-facts aria-labelledby="vh-model-facts-title-cosyvoice"><h2 id="vh-model-facts-title-cosyvoice">Model facts</h2><details class="vh-model-detail__facts-disclosure" data-vh-model-facts-disclosure aria-labelledby="vh-model-facts-title-cosyvoice" open><summary><span>Toggle model facts</span></summary><dl class="vh-model-detail__facts"><div><dt>Task</dt><dd>Text to speech</dd></div><div><dt>Parameters</dt><dd aria-describedby="vh-model-parameters-note-cosyvoice">859.2M</dd></div><div><dt>Architecture</dt><dd><code>cosyvoice-native</code></dd></div><div><dt>Runtime</dt><dd>VoiceHub-native</dd></div><div><dt>Languages</dt><dd><details class="vh-model-detail__languages"><summary>9 documented codes</summary><span><code>zh</code> <code>en</code> <code>ja</code> <code>ko</code> <code>de</code> <code>es</code> <code>fr</code> <code>it</code> <code>ru</code></span></details></dd></div><div><dt>Capabilities</dt><dd><details class="vh-model-detail__capabilities"><summary>11 capabilities</summary><span><code>text-to-speech</code> <code>voice-cloning</code> <code>multilingual</code> <code>fine-tuning</code> <code>flow-matching</code> <code>adversarial-vocoder-training</code> <code>safetensors</code> <code>voicehub-native</code> <code>native-runtime</code> <code>precomputed-speaker-embedding</code> <code>preencoded-speech-token-fine-tuning</code></span></details></dd></div><div><dt>Training</dt><dd><code>custom</code></dd></div><div><dt>License</dt><dd>Checkpoint-specific</dd></div><div><dt>Default checkpoint</dt><dd id="vh-model-checkpoint-cosyvoice"><a href="https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512"><code>FunAudioLLM/Fun-CosyVoice3-0.5B-2512</code></a></dd></div></dl></details></aside>

<div class="vh-model-detail__main vh-model-detail__content" markdown>

## Usage

Complete the [VoiceHub installation](../../getting-started/installation.md) once,
then run this repository-authored example. Model pages intentionally contain no
package-install command.

This example is maintained against VoiceHub's public API; it is not copied from an upstream demo or package README.

**Model-specific path:** Loads the required 192-value speaker embedding from a reviewable JSON file.

**Inputs and controls:** The native boundary intentionally does not run an unverified speaker encoder behind the caller's back.

```python
from pathlib import Path
import json

from voicehub import AutoModelForTextToSpeech, TTSGenerationConfig

SPEAKER_EMBEDDING_FILE = Path("speaker_embedding.json")
SPEAKER_EMBEDDING = json.loads(SPEAKER_EMBEDDING_FILE.read_text(encoding="utf-8"))
if len(SPEAKER_EMBEDDING) != 192:
    raise ValueError("CosyVoice expects exactly 192 speaker-embedding values.")

model = AutoModelForTextToSpeech.from_pretrained(
    'FunAudioLLM/Fun-CosyVoice3-0.5B-2512',
    model_type='cosyvoice',
    device="cuda",
    lazy_load=True,
)
output = model.generate(
    'VoiceHub keeps model integrations explicit and reproducible.',
    generation_config=TTSGenerationConfig(
        seed=42,
        output_file=Path("output.wav"),
    ),
    speaker_embedding=SPEAKER_EMBEDDING,
    instruction="Speak clearly.",
    flow_steps=10,
)
print(output.file_path, output.sample_rate, output.metadata)
```

Use authorized recordings. Verify hardware needs and pin a revision in production.

## Overview

`cosyvoice` is a VoiceHub **text to speech**
integration. This page is generated from its registry contract. [Open the `cosyvoice` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/cosyvoice.ipynb).

| Property | Value |
| --- | --- |
| Task | Text to speech |
| Architecture | `cosyvoice-native` |
| Runtime | `VoiceHub-native` |
| Languages | `zh`, `en`, `ja`, `ko`, … complete audited list below |
| Capabilities | `text-to-speech`, `voice-cloning`, `multilingual`, `fine-tuning`, `flow-matching`, `adversarial-vocoder-training`, `safetensors`, `voicehub-native`, `native-runtime`, `precomputed-speaker-embedding`, `preencoded-speech-token-fine-tuning` |
| Reusable components | `conformer` |
| Normalized output | `TTSOutput` |

### Language support

<details class="vh-language-support" markdown>
<summary>Supported language abbreviations</summary>

`zh`, `en`, `ja`, `ko`, `de`, `es`, `fr`, `it`, `ru`

The card additionally names Guangdong, Minnan, Sichuan, Dongbei, Shan3xi, Shan1xi, Shanghai, Tianjin, Shandong, Ningxia, and Gansu Chinese dialects or accents.

</details>

## Paper and GitHub

- **Paper:** [CosyVoice: Multi-Lingual Large Voice Generation Model](https://arxiv.org/abs/2407.05407)
- **Upstream GitHub:** [CosyVoice](https://github.com/FunAudioLLM/CosyVoice)
- **VoiceHub source:** [VoiceHub model implementation](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/cosyvoice/modeling_cosyvoice.py)

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('cosyvoice')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `cosyvoice` |
| Configuration class | `CosyVoiceConfig` |
| Architecture class | `CosyVoiceForTextToSpeech` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'FunAudioLLM/Fun-CosyVoice3-0.5B-2512',
    model_type='cosyvoice',
)
print(type(processor).__name__)
```

## Inference

The Usage example returns `TTSOutput` through `AutoModelForTextToSpeech`.

### Input and output contract

| Property | Value |
| --- | --- |
| Readiness | `integrated-raw` |
| Data architecture | `hybrid` |
| Sample rate | 24,000 Hz |
| Contract getter | `get_tts_dataset_spec('cosyvoice')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `llm-raw-audio` | `text` | speech_audio / audio / waveform / audio_path | Source | at most one: speech_audio / audio / waveform / audio_path; forbidden: speech_tokens; speech_audio requires one of speech_sampling_rate, sampling_rate, sample_rate; audio requires one of speech_sampling_rate, sampling_rate, sample_rate; waveform requires one of speech_sampling_rate, sampling_rate, sample_rate |
| `llm-record` | `text`, `speech_tokens` | — | Prepared | forbidden: speech_audio, audio, waveform, audio_path |

Multi-component language-model, diffusion, acoustic, or GAN data. See the [data workflow](../../guides/data-preparation.md).

## Training and optimization

Use `available_optimization_passes()` to discover reversible public passes.
Unsupported runtime or hardware fails closed before mutation.

### Training contract

| Property | Value |
| --- | --- |
| Support | `custom` |
| Family | `composite` |
| Recipe | `adversarial` |
| Default phase | `llm` |
| Training checkpoint | `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `llm` | objective | `model.llm` | — | `language_model_loss` |
| `flow` | objective | `model.flow` | — | `flow_matching_loss` |
| `hifigan_generator` | generator | `model.hift` | — | `adversarial_loss`, `feature_matching_loss`, `pitch_loss`, `spectral_reconstruction_loss` |
| `hifigan_discriminator` | discriminator | `model.hifigan.discriminator` | — | `discriminator_loss` |

This profile uses model-specific phases; inspect and honor each phase boundary. Call `model.validate_training_support()` first, then follow the
[training workflow](../../guides/training.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | [`FunAudioLLM/Fun-CosyVoice3-0.5B-2512`](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) |
| Hugging Face ID | [`FunAudioLLM/Fun-CosyVoice3-0.5B-2512`](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512)<br>Repository availability verified through the Hugging Face model API on 2026-08-11; pin a revision before production use. |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.cosyvoice.modeling_cosyvoice.CosyVoiceForTextToSpeech` |
| Configuration | `voicehub.models.cosyvoice.configuration_cosyvoice.CosyVoiceConfig` |
| Source provenance | `voicehub/models/cosyvoice/source/SOURCE.json` |
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

### `CosyVoiceConfig`

<p class="vh-model-api-card__source-wrap"><a class="vh-model-api-card__source" href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/cosyvoice/configuration_cosyvoice.py">View source</a></p>
<div class="vh-model-api-card__signature" markdown>

```text
CosyVoiceConfig(**config_kwargs)
```

</div>
<h4>Parameters</h4>
<div class="vh-model-api-card__parameters" markdown>
- `**config_kwargs` — Configuration fields validated by CosyVoiceConfig.
</div>
</section>

<section class="vh-model-api-card" data-vh-model-api-card="model" markdown>
<p class="vh-model-api-card__badge-wrap"><span class="vh-model-api-card__badge">Model</span></p>

### `CosyVoiceForTextToSpeech`

<p class="vh-model-api-card__source-wrap"><a class="vh-model-api-card__source" href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/cosyvoice/modeling_cosyvoice.py">View source</a></p>
<div class="vh-model-api-card__signature" markdown>

```text
AutoModelForTextToSpeech.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='cosyvoice',
    config=None,
    **model_kwargs,
)
```

</div>
<h4>Parameters</h4>
<div class="vh-model-api-card__parameters" markdown>
- `pretrained_model_name_or_path` — Hub ID or compatible local directory.
- `model_type` — Canonical model type; use 'cosyvoice'.
- `config` — Optional preloaded CosyVoiceConfig instance.
- `**model_kwargs` — Model-specific loading arguments.
</div>
</section>

```python
from voicehub import get_model_spec

spec = get_model_spec('cosyvoice')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('cosyvoice')` |
| Load and run | `AutoModelForTextToSpeech` |
| Configure | `CosyVoiceConfig` |
| Process | `AutoProcessor` |
| Model implementation | `CosyVoiceForTextToSpeech` |
| Normalized output | `TTSOutput` |
| Training contract | `get_training_spec('cosyvoice')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).

</div>

</div>

</div>
