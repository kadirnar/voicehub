---
description: Public API, checkpoint, training, and optimization guide for the parlertts integration.
---

# ParlerTTS {.vh-model-title}

## Usage

Complete the [VoiceHub installation](../../getting-started/installation.md) once,
then run this repository-authored example. Model pages intentionally contain no
package-install command.

This example is maintained against VoiceHub's public API; it is not copied from an upstream demo or package README.

**Model-specific path:** Separates the spoken text from Parler-TTS's acoustic style description.

**Inputs and controls:** Describe voice, pace, and recording conditions in `description`, not in the text to be spoken.

```python
from pathlib import Path

from voicehub import AutoModelForTextToSpeech, TTSGenerationConfig

model = AutoModelForTextToSpeech.from_pretrained(
    'parler-tts/parler-tts-mini-v1',
    model_type='parlertts',
    device="cuda",
    lazy_load=True,
)
output = model.generate(
    'VoiceHub keeps model integrations explicit and reproducible.',
    generation_config=TTSGenerationConfig(
        seed=42,
        output_file=Path("output.wav"),
    ),
    description="A clear, close-mic voice at a steady pace with very little background noise",
)
print(output.file_path, output.sample_rate, output.metadata)
```

Use authorized recordings. Verify hardware needs and pin a revision in production.

## Overview

`parlertts` is a VoiceHub **text to speech**
integration. This page is generated from its registry contract. [Open the `parlertts` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/parlertts.ipynb).

| Property | Value |
| --- | --- |
| Task | Text to speech |
| Architecture | `parlertts` |
| Runtime | `VoiceHub-native` |
| Languages | `en` |
| Capabilities | `text-to-speech`, `prompted-style`, `fine-tuning`, `safetensors`, `voicehub-native`, `native-runtime`, `raw-audio-fine-tuning` |
| Reusable components | `dac` |
| Normalized output | `TTSOutput` |

### Language support

<details class="vh-language-support" markdown>
<summary>Supported language abbreviations</summary>

`en`

</details>

## Paper and GitHub

- **Paper:** [Parler-TTS: A Text-to-Speech Dataset and Model Controlled by Natural Language](https://arxiv.org/abs/2402.01912)
- **Upstream GitHub:** [Parler-TTS](https://github.com/huggingface/parler-tts)
- **VoiceHub source:** [VoiceHub model implementation](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/parlertts/modeling_parlertts.py)

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('parlertts')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `parlertts` |
| Configuration class | `ParlerTTSConfig` |
| Architecture class | `ParlerTTSForTextToSpeech` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'parler-tts/parler-tts-mini-v1',
    model_type='parlertts',
)
print(type(processor).__name__)
```

## Inference

The Usage example returns `TTSOutput` through `AutoModelForTextToSpeech`.

### Input and output contract

| Property | Value |
| --- | --- |
| Readiness | `integrated-raw` |
| Data architecture | `sequence-to-sequence` |
| Sample rate | 44,100 Hz |
| Contract getter | `get_tts_dataset_spec('parlertts')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `waveform-teacher-forcing` | — | description / input_ids; audio_values / input_values | Source | at most one: description / input_ids; audio_values / input_values; forbidden: audio_codes, labels |
| `dac-codes` | `audio_codes` | description / input_ids | Prepared | at most one: description / input_ids; forbidden: audio_values, input_values, labels |
| `delayed-labels` | `labels` | description / input_ids | Prepared | at most one: description / input_ids; forbidden: audio_values, input_values, audio_codes |

Encoder text plus teacher-forced acoustic or codec targets. See the [data workflow](../../guides/data-preparation.md).

## Training and optimization

Use `available_optimization_passes()` to discover reversible public passes.
Unsupported runtime or hardware fails closed before mutation.

### Training contract

| Property | Value |
| --- | --- |
| Support | `native` |
| Family | `sequence-to-sequence` |
| Recipe | `single-phase` |
| Default phase | `default` |
| Training checkpoint | `parler-tts/parler-tts-mini-v1` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `default` | objective | — | — | `loss`, `total_loss` |

The integration accepts its declared source or prepared contract directly. Call `model.validate_training_support()` first, then follow the
[training workflow](../../guides/training.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | [`parler-tts/parler-tts-mini-v1`](https://huggingface.co/parler-tts/parler-tts-mini-v1) |
| Hugging Face ID | [`parler-tts/parler-tts-mini-v1`](https://huggingface.co/parler-tts/parler-tts-mini-v1)<br>Repository availability verified through the Hugging Face model API on 2026-08-11; pin a revision before production use. |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.parlertts.modeling_parlertts.ParlerTTSForTextToSpeech` |
| Configuration | `voicehub.models.parlertts.configuration_parlertts.ParlerTTSConfig` |
| Source provenance | `voicehub/models/parlertts/source/SOURCE.json` |
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

### `ParlerTTSConfig`

[View `ParlerTTSConfig` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/parlertts/configuration_parlertts.py)

```text
ParlerTTSConfig(**config_kwargs)
```

### `ParlerTTSForTextToSpeech`

[View `ParlerTTSForTextToSpeech` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/parlertts/modeling_parlertts.py)

```text
AutoModelForTextToSpeech.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='parlertts',
    config=None,
    **model_kwargs,
)
```

```python
from voicehub import get_model_spec

spec = get_model_spec('parlertts')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('parlertts')` |
| Load and run | `AutoModelForTextToSpeech` |
| Configure | `ParlerTTSConfig` |
| Process | `AutoProcessor` |
| Model implementation | `ParlerTTSForTextToSpeech` |
| Normalized output | `TTSOutput` |
| Training contract | `get_training_spec('parlertts')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).
