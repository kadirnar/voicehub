---
description: Public API, checkpoint, training, and optimization guide for the speecht5 integration.
---

# SpeechT5 {.vh-model-title}

## Usage

Complete the [VoiceHub installation](../../getting-started/installation.md) once,
then run this repository-authored example. Model pages intentionally contain no
package-install command.

This example is maintained against VoiceHub's public API; it is not copied from an upstream demo or package README.

**Model-specific path:** Passes a reviewed speaker-embedding file through SpeechT5's safe public loader.

**Inputs and controls:** Use Safetensors or NPY for embeddings; omit the field to use the wrapper's neutral zero embedding.

```python
from pathlib import Path

from voicehub import AutoModelForTextToSpeech, TTSGenerationConfig

SPEAKER_EMBEDDING = Path("speaker_embedding.npy")
if not SPEAKER_EMBEDDING.is_file():
    raise FileNotFoundError(SPEAKER_EMBEDDING)

model = AutoModelForTextToSpeech.from_pretrained(
    'microsoft/speecht5_tts',
    model_type='speecht5',
    device="cuda",
    lazy_load=True,
)
output = model.generate(
    'VoiceHub keeps model integrations explicit and reproducible.',
    generation_config=TTSGenerationConfig(
        seed=42,
        output_file=Path("output.wav"),
    ),
    speaker_embedding_path=SPEAKER_EMBEDDING,
    threshold=0.5,
)
print(output.file_path, output.sample_rate, output.metadata)
```

Use authorized recordings. Verify hardware needs and pin a revision in production.

## Overview

`speecht5` is a VoiceHub **text to speech**
integration. This page is generated from its registry contract. [Open the `speecht5` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/speecht5.ipynb).

| Property | Value |
| --- | --- |
| Task | Text to speech |
| Architecture | `speecht5` |
| Runtime | `VoiceHub-native` |
| Languages | `en` |
| Capabilities | `text-to-speech`, `speaker-embedding`, `safetensors`, `fine-tuning`, `voicehub-native`, `native-runtime`, `raw-audio-fine-tuning`, `inference-reloadable-training-export` |
| Reusable components | — |
| Normalized output | `TTSOutput` |

### Language support

<details class="vh-language-support" markdown>
<summary>Supported language abbreviations</summary>

`en`

</details>

## Paper and GitHub

- **Paper:** [SpeechT5: Unified-Modal Encoder-Decoder Pre-Training for Spoken Language Processing](https://arxiv.org/abs/2110.07205)
- **Upstream GitHub:** [SpeechT5](https://github.com/microsoft/SpeechT5)
- **VoiceHub source:** [VoiceHub model implementation](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/speecht5/modeling_speecht5.py)

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('speecht5')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `speecht5` |
| Configuration class | `SpeechT5Config` |
| Architecture class | `SpeechT5ForTextToSpeech` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'microsoft/speecht5_tts',
    model_type='speecht5',
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
| Sample rate | 16,000 Hz |
| Contract getter | `get_tts_dataset_spec('speecht5')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `raw-audio` | `text`, `audio` | — | Source | — |
| `processor-ready` | `input_ids`, `labels` | — | Prepared | — |

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
| Default phase | `spectrogram` |
| Training checkpoint | `microsoft/speecht5_tts` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `spectrogram` | objective | `model` | `input_ids`, `attention_mask`, `labels` | `loss` |

The integration accepts its declared source or prepared contract directly. Call `model.validate_training_support()` first, then follow the
[training workflow](../../guides/training.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | [`microsoft/speecht5_tts`](https://huggingface.co/microsoft/speecht5_tts) |
| Hugging Face ID | [`microsoft/speecht5_tts`](https://huggingface.co/microsoft/speecht5_tts)<br>Repository availability verified through the Hugging Face model API on 2026-08-11; pin a revision before production use. |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.speecht5.modeling_speecht5.SpeechT5ForTextToSpeech` |
| Configuration | `voicehub.models.speecht5.configuration_speecht5.SpeechT5Config` |
| Source provenance | `voicehub/models/speecht5/SOURCE.json` |
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

### `SpeechT5Config`

[View `SpeechT5Config` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/speecht5/configuration_speecht5.py)

```text
SpeechT5Config(**config_kwargs)
```

### `SpeechT5ForTextToSpeech`

[View `SpeechT5ForTextToSpeech` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/speecht5/modeling_speecht5.py)

```text
AutoModelForTextToSpeech.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='speecht5',
    config=None,
    **model_kwargs,
)
```

```python
from voicehub import get_model_spec

spec = get_model_spec('speecht5')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('speecht5')` |
| Load and run | `AutoModelForTextToSpeech` |
| Configure | `SpeechT5Config` |
| Process | `AutoProcessor` |
| Model implementation | `SpeechT5ForTextToSpeech` |
| Normalized output | `TTSOutput` |
| Training contract | `get_training_spec('speecht5')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).
