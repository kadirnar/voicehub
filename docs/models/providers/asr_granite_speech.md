---
description: Public API, checkpoint, training, and optimization guide for the asr_granite_speech integration.
---

# GraniteSpeech {.vh-model-title}

## Usage

```bash
python -m pip install "voicehub @ git+https://github.com/kadirnar/voicehub.git@main"
```

Install from source, then choose a compatible checkpoint. Place a supported recording at `speech.wav` and inspect the transcript.

```python
from voicehub import AutoModelForSpeechRecognition

model = AutoModelForSpeechRecognition.from_pretrained(
    'ibm-granite/granite-speech-4.1-2b',
    model_type='asr_granite_speech',
    device="cuda",
    lazy_load=True,
)
output = model.transcribe("speech.wav")
print(output.text)
for segment in output.segments:
    print(segment.start, segment.end, segment.text)
```

Use authorized recordings. Verify hardware needs and pin a revision in production.

## Overview

`asr_granite_speech` is a VoiceHub **automatic speech recognition**
integration. This page is generated from its registry contract. [Open the `asr_granite_speech` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/asr_granite_speech.ipynb).

| Property | Value |
| --- | --- |
| Task | Automatic speech recognition |
| Architecture | `granite-speech` |
| Runtime | `VoiceHub-native` |
| Languages | Checkpoint-defined; not exhaustively enumerated |
| Capabilities | `automatic-speech-recognition`, `multilingual`, `hotwords`, `translation`, `safetensors`, `fine-tuning`, `lora`, `voicehub-native`, `native-runtime` |
| Reusable components | — |
| Normalized output | `ASROutput` |

### Language support

VoiceHub does not claim one exhaustive language list across compatible checkpoints; verify the selected checkpoint card and processor metadata.

## Paper and GitHub

- **Paper:** No dedicated upstream research paper is published for this integration.
- **Upstream GitHub:** [Granite Speech models](https://github.com/ibm-granite/granite-speech-models)
- **VoiceHub source:** [VoiceHub model implementation](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_granite_speech/modeling_asr_granite_speech.py)

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('asr_granite_speech')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `asr_granite_speech` |
| Configuration class | `GraniteSpeechASRConfig` |
| Architecture class | `GraniteSpeechForSpeechRecognition` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'ibm-granite/granite-speech-4.1-2b',
    model_type='asr_granite_speech',
)
print(type(processor).__name__)
```

## Inference

The Usage example returns `ASROutput` through `AutoModelForSpeechRecognition`.

### Input and output contract

| Property | Value |
| --- | --- |
| Readiness | `integrated-raw` |
| Data architecture | `prompted-multimodal` |
| Sample rate | 16,000 Hz |
| Contract getter | `get_asr_dataset_spec('asr_granite_speech')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `raw-audio` | `audio` | text / transcription / transcript | Source | at most one: text / transcription / transcript; forbidden: language |
| `granite-model-ready` | `input_ids`, `attention_mask`, `input_features`, `input_features_mask`, `labels` | — | Prepared | — |

Prompt-conditioned Granite Speech multimodal fine-tuning records. See the [data workflow](../../guides/speech-data.md).

## Training and optimization

Use `available_optimization_passes()` to discover reversible public passes.
Unsupported runtime or hardware fails closed before mutation.

### Training contract

| Property | Value |
| --- | --- |
| Support | `native` |
| Family | `speech-sequence-to-sequence` |
| Recipe | `single-phase` |
| Default phase | `speech_recognition` |
| Training checkpoint | `ibm-granite/granite-speech-4.1-2b` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `speech_recognition` | objective | `model.encoder`, `model.projector`, `model.language_model` | `input_ids`, `attention_mask`, `input_features`, `input_features_mask`, `labels` | `loss` |

The integration accepts its declared source or prepared contract directly. Call `model.validate_training_support()` first, then follow the
[training workflow](../../guides/training.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | [`ibm-granite/granite-speech-4.1-2b`](https://huggingface.co/ibm-granite/granite-speech-4.1-2b) |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.asr_granite_speech.modeling_asr_granite_speech.GraniteSpeechForSpeechRecognition` |
| Configuration | `voicehub.models.asr_granite_speech.configuration_asr_granite_speech.GraniteSpeechASRConfig` |
| Source provenance | `voicehub/architectures/granite_speech/SOURCE.json` |
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

### `GraniteSpeechASRConfig`

[View `GraniteSpeechASRConfig` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_granite_speech/configuration_asr_granite_speech.py)

```text
GraniteSpeechASRConfig(**config_kwargs)
```

### `GraniteSpeechForSpeechRecognition`

[View `GraniteSpeechForSpeechRecognition` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_granite_speech/modeling_asr_granite_speech.py)

```text
AutoModelForSpeechRecognition.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='asr_granite_speech',
    config=None,
    **model_kwargs,
)
```

```python
from voicehub import get_model_spec

spec = get_model_spec('asr_granite_speech')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('asr_granite_speech')` |
| Load and run | `AutoModelForSpeechRecognition` |
| Configure | `GraniteSpeechASRConfig` |
| Process | `AutoProcessor` |
| Model implementation | `GraniteSpeechForSpeechRecognition` |
| Normalized output | `ASROutput` |
| Training contract | `get_training_spec('asr_granite_speech')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).
