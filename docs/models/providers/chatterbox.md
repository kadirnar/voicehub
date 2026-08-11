---
description: Public API, checkpoint, training, and optimization guide for the chatterbox integration.
---

# Chatterbox {.vh-model-title}

## Usage

Complete the [VoiceHub installation](../../getting-started/installation.md) once,
then run this repository-authored example. Model pages intentionally contain no
package-install command.

This example is maintained against VoiceHub's public API; it is not copied from an upstream demo or package README.

**Model-specific path:** Demonstrates Chatterbox voice prompting through VoiceHub's normalized reference-audio field.

**Inputs and controls:** Use only a reference recording you are authorized to process; omit the argument for the checkpoint's default voice.

```python
from pathlib import Path

from voicehub import AutoModelForTextToSpeech, TTSGenerationConfig

REFERENCE_AUDIO = Path("reference.wav")
REFERENCE_TEXT = "The reference transcript must exactly match the authorized audio."
if not REFERENCE_AUDIO.is_file():
    raise FileNotFoundError(REFERENCE_AUDIO)

model = AutoModelForTextToSpeech.from_pretrained(
    'ResembleAI/chatterbox',
    model_type='chatterbox',
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
    max_new_tokens=1_024,
)
print(output.file_path, output.sample_rate, output.metadata)
```

Use authorized recordings. Verify hardware needs and pin a revision in production.

## Overview

`chatterbox` is a VoiceHub **text to speech**
integration. This page is generated from its registry contract. [Open the `chatterbox` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/chatterbox.ipynb).

| Property | Value |
| --- | --- |
| Task | Text to speech |
| Architecture | `chatterbox` |
| Runtime | `VoiceHub-native` |
| Languages | `ar`, `da`, `de`, `el`, `en`, `es`, `fi`, `fr`, `he`, `hi`, `it`, `ja`, `ko`, `ms`, `nl`, `no`, `pl`, `pt`, `ru`, `sv`, `sw`, `tr`, `zh` |
| Capabilities | `text-to-speech`, `voice-cloning`, `fine-tuning`, `safetensors`, `voicehub-native`, `native-runtime`, `raw-audio-fine-tuning` |
| Reusable components | `conformer` |
| Normalized output | `TTSOutput` |

### Language support

<details class="vh-language-support" markdown>
<summary>Supported language abbreviations</summary>

`ar`, `da`, `de`, `el`, `en`, `es`, `fi`, `fr`, `he`, `hi`, `it`, `ja`, `ko`, `ms`, `nl`, `no`, `pl`, `pt`, `ru`, `sv`, `sw`, `tr`, `zh`

</details>

## Paper and GitHub

- **Paper:** No dedicated upstream research paper is published for this integration.
- **Upstream GitHub:** [Chatterbox](https://github.com/resemble-ai/chatterbox)
- **VoiceHub source:** [VoiceHub model implementation](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/chatterbox/modeling_chatterbox.py)

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('chatterbox')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `chatterbox` |
| Configuration class | `ChatterboxConfig` |
| Architecture class | `ChatterboxForTextToSpeech` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'ResembleAI/chatterbox',
    model_type='chatterbox',
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
| Contract getter | `get_tts_dataset_spec('chatterbox')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `t3-raw` | `text` | audio / audio_path | Source | — |
| `flow-raw` | — | audio / audio_path | Source | — |
| `t3-precomputed` | `text_tokens`, `speech_tokens`, `speaker_emb` | — | Prepared | — |
| `flow-precomputed` | `speech_token`, `speech_feat`, `embedding` | — | Prepared | — |

Multi-component language-model, diffusion, acoustic, or GAN data. See the [data workflow](../../guides/data-preparation.md).

## Training and optimization

Use `available_optimization_passes()` to discover reversible public passes.
Unsupported runtime or hardware fails closed before mutation.

### Training contract

| Property | Value |
| --- | --- |
| Support | `custom` |
| Family | `composite` |
| Recipe | `multi-phase` |
| Default phase | `language_model` |
| Training checkpoint | `ResembleAI/chatterbox` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `language_model` | objective | `model.t3` | — | `loss`, `text_loss`, `speech_token_loss` |
| `flow` | objective | `model.s3gen.flow` | — | `loss`, `flow_loss`, `diffusion_loss` |

This profile uses model-specific phases; inspect and honor each phase boundary. Call `model.validate_training_support()` first, then follow the
[training workflow](../../guides/training.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | [`ResembleAI/chatterbox`](https://huggingface.co/ResembleAI/chatterbox) |
| Hugging Face ID | [`ResembleAI/chatterbox`](https://huggingface.co/ResembleAI/chatterbox)<br>Repository availability verified through the Hugging Face model API on 2026-08-11; pin a revision before production use. |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.chatterbox.modeling_chatterbox.ChatterboxForTextToSpeech` |
| Configuration | `voicehub.models.chatterbox.configuration_chatterbox.ChatterboxConfig` |
| Source provenance | `voicehub/models/chatterbox/source/SOURCE.json` |
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

### `ChatterboxConfig`

[View `ChatterboxConfig` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/chatterbox/configuration_chatterbox.py)

```text
ChatterboxConfig(**config_kwargs)
```

### `ChatterboxForTextToSpeech`

[View `ChatterboxForTextToSpeech` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/chatterbox/modeling_chatterbox.py)

```text
AutoModelForTextToSpeech.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='chatterbox',
    config=None,
    **model_kwargs,
)
```

```python
from voicehub import get_model_spec

spec = get_model_spec('chatterbox')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('chatterbox')` |
| Load and run | `AutoModelForTextToSpeech` |
| Configure | `ChatterboxConfig` |
| Process | `AutoProcessor` |
| Model implementation | `ChatterboxForTextToSpeech` |
| Normalized output | `TTSOutput` |
| Training contract | `get_training_spec('chatterbox')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).
