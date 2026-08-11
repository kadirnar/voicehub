---
description: Public API, checkpoint, training, and optimization guide for the kokoro integration.
---

# Kokoro {.vh-model-title}

## Usage

Complete the [VoiceHub installation](../../getting-started/installation.md) once,
then run this repository-authored example. Model pages intentionally contain no
package-install command.

This example is maintained against VoiceHub's public API; it is not copied from an upstream demo or package README.

**Model-specific path:** Selects a Kokoro voice ID and explicit speaking speed.

**Inputs and controls:** Voice IDs are checkpoint-specific; `af_heart` belongs to the registered Kokoro release.

```python
from pathlib import Path

from voicehub import AutoModelForTextToSpeech, TTSGenerationConfig

model = AutoModelForTextToSpeech.from_pretrained(
    'hexgrad/Kokoro-82M',
    model_type='kokoro',
    device="cuda",
    lazy_load=True,
)
output = model.generate(
    'VoiceHub keeps model integrations explicit and reproducible.',
    generation_config=TTSGenerationConfig(
        seed=42,
        output_file=Path("output.wav"),
    ),
    voice="af_heart",
    speed=1.0,
)
print(output.file_path, output.sample_rate, output.metadata)
```

Use authorized recordings. Verify hardware needs and pin a revision in production.

## Overview

`kokoro` is a VoiceHub **text to speech**
integration. This page is generated from its registry contract. [Open the `kokoro` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/kokoro.ipynb).

| Property | Value |
| --- | --- |
| Task | Text to speech |
| Architecture | `kokoro` |
| Runtime | `VoiceHub-native` |
| Languages | `en-US`, `en-GB`, `es`, `fr`, `hi`, `it`, `pt-BR`, `ja`, `zh` |
| Capabilities | `text-to-speech`, `multilingual`, `fine-tuning`, `safetensors`, `voicehub-native`, `native-runtime` |
| Reusable components | — |
| Normalized output | `TTSOutput` |

### Language support

<details class="vh-language-support" markdown>
<summary>Supported language abbreviations</summary>

`en-US`, `en-GB`, `es`, `fr`, `hi`, `it`, `pt-BR`, `ja`, `zh`

</details>

## Paper and GitHub

- **Paper:** No dedicated upstream research paper is published for this integration.
- **Upstream GitHub:** [Kokoro](https://github.com/hexgrad/kokoro)
- **VoiceHub source:** [VoiceHub model implementation](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/kokoro/modeling_kokoro.py)

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('kokoro')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `kokoro` |
| Configuration class | `KokoroConfig` |
| Architecture class | `KokoroForTextToSpeech` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'hexgrad/Kokoro-82M',
    model_type='kokoro',
)
print(type(processor).__name__)
```

## Inference

The Usage example returns `TTSOutput` through `AutoModelForTextToSpeech`.

### Input and output contract

| Property | Value |
| --- | --- |
| Readiness | `preprocessed` |
| Data architecture | `acoustic` |
| Sample rate | 24,000 Hz |
| Contract getter | `get_tts_dataset_spec('kokoro')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `full-preprocessed` | `durations` | input_ids / phonemes; ref_s / voice; audio_values / audio / labels | Prepared | — |
| `duration-only` | `durations`, `training_phase` | input_ids / phonemes; ref_s / voice | Prepared | — |

Direct acoustic, mel, codec, or waveform regression data. See the [data workflow](../../guides/data-preparation.md).

## Training and optimization

Use `available_optimization_passes()` to discover reversible public passes.
Unsupported runtime or hardware fails closed before mutation.

### Training contract

| Property | Value |
| --- | --- |
| Support | `preprocessed` |
| Family | `acoustic-regression` |
| Recipe | `multi-phase` |
| Default phase | `acoustic` |
| Training checkpoint | `hexgrad/Kokoro-82M` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `duration` | objective | `model.bert`, `model.bert_encoder`, `model.predictor` | `input_ids`, `ref_s`, `durations` | `loss` |
| `acoustic` | objective | `model.bert`, `model.bert_encoder`, `model.predictor`, `model.text_encoder`, `model.decoder` | `input_ids`, `ref_s`, `durations`, `audio_values` | `loss` |

Prepare the exact tensors listed in the data contract before this step. Call `model.validate_training_support()` first, then follow the
[training workflow](../../guides/training.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | [`hexgrad/Kokoro-82M`](https://huggingface.co/hexgrad/Kokoro-82M) |
| Hugging Face ID | [`hexgrad/Kokoro-82M`](https://huggingface.co/hexgrad/Kokoro-82M)<br>Repository availability verified through the Hugging Face model API on 2026-08-11; pin a revision before production use. |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.kokoro.modeling_kokoro.KokoroForTextToSpeech` |
| Configuration | `voicehub.models.kokoro.configuration_kokoro.KokoroConfig` |
| Source provenance | `voicehub/models/kokoro/source/SOURCE.json` |
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

### `KokoroConfig`

[View `KokoroConfig` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/kokoro/configuration_kokoro.py)

```text
KokoroConfig(**config_kwargs)
```

### `KokoroForTextToSpeech`

[View `KokoroForTextToSpeech` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/kokoro/modeling_kokoro.py)

```text
AutoModelForTextToSpeech.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='kokoro',
    config=None,
    **model_kwargs,
)
```

```python
from voicehub import get_model_spec

spec = get_model_spec('kokoro')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('kokoro')` |
| Load and run | `AutoModelForTextToSpeech` |
| Configure | `KokoroConfig` |
| Process | `AutoProcessor` |
| Model implementation | `KokoroForTextToSpeech` |
| Normalized output | `TTSOutput` |
| Training contract | `get_training_spec('kokoro')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).
