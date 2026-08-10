---
description: Public API, checkpoint, training, and optimization guide for the vits integration.
---

# Vits {.vh-model-title}

## Usage

```bash
python -m pip install "voicehub @ git+https://github.com/kadirnar/voicehub.git@main"
```

Install from source, then choose a compatible checkpoint. Set the text and generation options, then inspect the returned audio.

```python
from pathlib import Path

from voicehub import AutoModelForTextToSpeech, TTSGenerationConfig

model = AutoModelForTextToSpeech.from_pretrained(
    'facebook/mms-tts-eng',
    model_type='vits',
    device="cuda",
    lazy_load=True,
)
generation_kwargs = {}
output = model.generate(
    "VoiceHub keeps model integrations consistent and easy to extend.",
    generation_config=TTSGenerationConfig(
        seed=42,
        output_file=Path("output.wav"),
    ),
    **generation_kwargs,
)
print(output.file_path, output.sample_rate)
```

Use authorized recordings. Verify hardware needs and pin a revision in production.

## Overview

`vits` is a VoiceHub **text to speech**
integration. This page is generated from its registry contract. [Open the `vits` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/vits.ipynb).

| Property | Value |
| --- | --- |
| Task | Text to speech |
| Architecture | `vits` |
| Runtime | `VoiceHub-native` |
| Languages | `en` |
| Capabilities | `text-to-speech`, `multilingual`, `mms-tts`, `safetensors`, `fine-tuning`, `voicehub-native`, `native-runtime`, `raw-audio-training`, `preprocessed-training`, `adversarial-training`, `generator-warm-start`, `explicit-acoustic-training-config` |
| Reusable components | — |
| Normalized output | `TTSOutput` |

### Language support

<details class="vh-language-support" markdown>
<summary>Supported language abbreviations</summary>

`en`

</details>

## Paper and GitHub

- **Paper:** [Conditional Variational Autoencoder with Adversarial Learning for End-to-End TTS](https://arxiv.org/abs/2106.06103)
- **Upstream GitHub:** [VITS](https://github.com/jaywalnut310/vits)
- **VoiceHub source:** [VoiceHub model implementation](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/vits/modeling_vits.py)

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('vits')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `vits` |
| Configuration class | `VitsConfig` |
| Architecture class | `VitsForTextToSpeech` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'facebook/mms-tts-eng',
    model_type='vits',
)
print(type(processor).__name__)
```

## Inference

The Usage example returns `TTSOutput` through `AutoModelForTextToSpeech`.

### Input and output contract

| Property | Value |
| --- | --- |
| Readiness | `integrated-raw` |
| Data architecture | `vits` |
| Sample rate | Model/checkpoint specific |
| Contract getter | `get_tts_dataset_spec('vits')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `raw-adversarial` | `text` | audio / audio_values | Source | — |
| `tokenized-raw-adversarial` | `input_ids` | audio / audio_values | Source | — |
| `precomputed-spectrogram` | `spectrogram` | text / input_ids; audio / audio_values | Prepared | — |

VITS/GAN text, waveform, spectrogram, and adversarial data. See the [data workflow](../../guides/data-preparation.md).

## Training and optimization

Use `available_optimization_passes()` to discover reversible public passes.
Unsupported runtime or hardware fails closed before mutation.

### Training contract

| Property | Value |
| --- | --- |
| Support | `preprocessed` |
| Family | `vits` |
| Recipe | `adversarial` |
| Default phase | `generator` |
| Training checkpoint | `facebook/mms-tts-eng` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `discriminator` | discriminator | `training_model.discriminator` | `input_ids`, `audio_values` | `loss` |
| `generator` | generator | `training_model.native_model` | `input_ids`, `audio_values` | `loss` |

Prepare the exact tensors listed in the data contract before this step. Call `model.validate_training_support()` first, then follow the
[training workflow](../../guides/training.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | [`facebook/mms-tts-eng`](https://huggingface.co/facebook/mms-tts-eng) |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.vits.modeling_vits.VitsForTextToSpeech` |
| Configuration | `voicehub.models.vits.configuration_vits.VitsConfig` |
| Source provenance | `voicehub/architectures/vits/SOURCE.json` |
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

### `VitsConfig`

[View `VitsConfig` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/vits/configuration_vits.py)

```text
VitsConfig(**config_kwargs)
```

### `VitsForTextToSpeech`

[View `VitsForTextToSpeech` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/vits/modeling_vits.py)

```text
AutoModelForTextToSpeech.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='vits',
    config=None,
    **model_kwargs,
)
```

```python
from voicehub import get_model_spec

spec = get_model_spec('vits')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('vits')` |
| Load and run | `AutoModelForTextToSpeech` |
| Configure | `VitsConfig` |
| Process | `AutoProcessor` |
| Model implementation | `VitsForTextToSpeech` |
| Normalized output | `TTSOutput` |
| Training contract | `get_training_spec('vits')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).
