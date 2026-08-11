---
description: Public API, checkpoint, training, and optimization guide for the asr_wav2vec2 integration.
---

# Wav2Vec2 {.vh-model-title}

## Usage

Complete the [VoiceHub installation](../../getting-started/installation.md) once,
then run this repository-authored example. Model pages intentionally contain no
package-install command.

This example is maintained against VoiceHub's public API; it is not copied from an upstream demo or package README.

**Model-specific path:** Runs the native Wav2Vec2 CTC path and requests word-level alignment where supported.

**Inputs and controls:** CTC decoding is checkpoint-vocabulary specific; do not reuse this ID for arbitrary languages.

```python
from pathlib import Path

from voicehub import AutoModelForSpeechRecognition

AUDIO_FILE = Path("speech.wav")
if not AUDIO_FILE.is_file():
    raise FileNotFoundError(AUDIO_FILE)

model = AutoModelForSpeechRecognition.from_pretrained(
    'facebook/wav2vec2-base-960h',
    model_type='asr_wav2vec2',
    device="cuda",
    lazy_load=True,
)
output = model.transcribe(
    AUDIO_FILE,
    language="en",
    return_timestamps="word",
)
print(output.text)
for segment in output.segments:
    print(segment.start, segment.end, segment.text, segment.confidence)
```

Use authorized recordings. Verify hardware needs and pin a revision in production.

## Overview

`asr_wav2vec2` is a VoiceHub **automatic speech recognition**
integration. This page is generated from its registry contract. [Open the `asr_wav2vec2` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/asr_wav2vec2.ipynb).

| Property | Value |
| --- | --- |
| Task | Automatic speech recognition |
| Architecture | `wav2vec2` |
| Runtime | `VoiceHub-native` |
| Languages | `en` |
| Capabilities | `automatic-speech-recognition`, `timestamps`, `safetensors`, `fine-tuning`, `voicehub-native` |
| Reusable components | — |
| Normalized output | `ASROutput` |

### Language support

<details class="vh-language-support" markdown>
<summary>Supported language abbreviations</summary>

`en`

</details>

## Paper and GitHub

- **Paper:** [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477)
- **Upstream GitHub:** [fairseq](https://github.com/facebookresearch/fairseq)
- **VoiceHub source:** [VoiceHub model implementation](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_wav2vec2/modeling_asr_wav2vec2.py)

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('asr_wav2vec2')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `asr_wav2vec2` |
| Configuration class | `Wav2Vec2ASRConfig` |
| Architecture class | `Wav2Vec2ForSpeechRecognition` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'facebook/wav2vec2-base-960h',
    model_type='asr_wav2vec2',
)
print(type(processor).__name__)
```

## Inference

The Usage example returns `ASROutput` through `AutoModelForSpeechRecognition`.

### Input and output contract

| Property | Value |
| --- | --- |
| Readiness | `integrated-raw` |
| Data architecture | `ctc` |
| Sample rate | 16,000 Hz |
| Contract getter | `get_asr_dataset_spec('asr_wav2vec2')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `raw-audio` | `audio` | text / transcription / transcript | Source | at most one: text / transcription / transcript |
| `waveform-ctc` | `input_values`, `labels` | — | Prepared | — |

Wav2Vec2 waveform and CTC transcript records. See the [data workflow](../../guides/speech-data.md).

## Training and optimization

Use `available_optimization_passes()` to discover reversible public passes.
Unsupported runtime or hardware fails closed before mutation.

### Training contract

| Property | Value |
| --- | --- |
| Support | `native` |
| Family | `ctc` |
| Recipe | `single-phase` |
| Default phase | `speech_recognition` |
| Training checkpoint | `facebook/wav2vec2-base-960h` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `speech_recognition` | objective | `model` | — | `loss` |

The integration accepts its declared source or prepared contract directly. Call `model.validate_training_support()` first, then follow the
[training workflow](../../guides/training.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | [`facebook/wav2vec2-base-960h`](https://huggingface.co/facebook/wav2vec2-base-960h) |
| Hugging Face ID | [`facebook/wav2vec2-base-960h`](https://huggingface.co/facebook/wav2vec2-base-960h)<br>Repository availability verified through the Hugging Face model API on 2026-08-11; pin a revision before production use. |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.asr_wav2vec2.modeling_asr_wav2vec2.Wav2Vec2ForSpeechRecognition` |
| Configuration | `voicehub.models.asr_wav2vec2.configuration_asr_wav2vec2.Wav2Vec2ASRConfig` |
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

### `Wav2Vec2ASRConfig`

[View `Wav2Vec2ASRConfig` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_wav2vec2/configuration_asr_wav2vec2.py)

```text
Wav2Vec2ASRConfig(**config_kwargs)
```

### `Wav2Vec2ForSpeechRecognition`

[View `Wav2Vec2ForSpeechRecognition` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_wav2vec2/modeling_asr_wav2vec2.py)

```text
AutoModelForSpeechRecognition.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='asr_wav2vec2',
    config=None,
    **model_kwargs,
)
```

```python
from voicehub import get_model_spec

spec = get_model_spec('asr_wav2vec2')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('asr_wav2vec2')` |
| Load and run | `AutoModelForSpeechRecognition` |
| Configure | `Wav2Vec2ASRConfig` |
| Process | `AutoProcessor` |
| Model implementation | `Wav2Vec2ForSpeechRecognition` |
| Normalized output | `ASROutput` |
| Training contract | `get_training_spec('asr_wav2vec2')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).
