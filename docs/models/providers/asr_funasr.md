---
description: Public API, checkpoint, training, and optimization guide for the asr_funasr integration.
---

# FunASR {.vh-model-title}

## Usage

```bash
python -m pip install "voicehub @ git+https://github.com/kadirnar/voicehub.git@main"
```

Install from source, then choose a compatible checkpoint. Place a supported recording at `speech.wav` and inspect the transcript.

```python
from voicehub import AutoModelForSpeechRecognition

model = AutoModelForSpeechRecognition.from_pretrained(
    'iic/SenseVoiceSmall',
    model_type='asr_funasr',
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

`asr_funasr` is a VoiceHub **automatic speech recognition**
integration. This page is generated from its registry contract. [Open the `asr_funasr` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/asr_funasr.ipynb).

| Property | Value |
| --- | --- |
| Task | Automatic speech recognition |
| Architecture | `sensevoice-small` |
| Runtime | `VoiceHub-native` |
| Languages | Checkpoint-defined; not exhaustively enumerated |
| Capabilities | `automatic-speech-recognition`, `multilingual`, `timestamps`, `language-identification`, `emotion-recognition`, `audio-events`, `fine-tuning`, `safetensors`, `voicehub-native`, `native-runtime` |
| Reusable components | — |
| Normalized output | `ASROutput` |

### Language support

VoiceHub does not claim one exhaustive language list across compatible checkpoints; verify the selected checkpoint card and processor metadata.

## Paper and GitHub

- **Paper:** [FunASR: A Fundamental End-to-End Speech Recognition Toolkit](https://arxiv.org/abs/2305.11013)
- **Upstream GitHub:** [FunASR](https://github.com/modelscope/FunASR)
- **VoiceHub source:** [VoiceHub model implementation](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_native/funasr.py)

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('asr_funasr')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `asr_funasr` |
| Configuration class | `FunASRConfig` |
| Architecture class | `FunASRForSpeechRecognition` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'iic/SenseVoiceSmall',
    model_type='asr_funasr',
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
| Contract getter | `get_asr_dataset_spec('asr_funasr')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `raw-audio` | `language` | audio / audio_values / input_signal; text / transcription / transcript | Source | at most one: audio / audio_values / input_signal; text / transcription / transcript |
| `sensevoice-feature-transcript` | `features`, `language` | text / transcription / transcript | Prepared | at most one: text / transcription / transcript |
| `sensevoice-model-ready` | `features`, `feature_lengths`, `labels`, `label_lengths` | — | Prepared | — |

SenseVoice CTC records with language, emotion, event, and ITN control. See the [data workflow](../../guides/speech-data.md).

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
| Training checkpoint | `iic/SenseVoiceSmall` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `speech_recognition` | objective | `model` | `features`, `feature_lengths`, `labels`, `label_lengths` | `loss`, `ctc`, `rich` |

The integration accepts its declared source or prepared contract directly. Call `model.validate_training_support()` first, then follow the
[training workflow](../../guides/training.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | [`iic/SenseVoiceSmall`](https://huggingface.co/iic/SenseVoiceSmall) |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.asr_native.funasr.FunASRForSpeechRecognition` |
| Configuration | `voicehub.models.asr_native.configuration.FunASRConfig` |
| Source provenance | `voicehub/architectures/sensevoice/SOURCE.json` |
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

### `FunASRConfig`

[View `FunASRConfig` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_native/configuration.py)

```text
FunASRConfig(**config_kwargs)
```

### `FunASRForSpeechRecognition`

[View `FunASRForSpeechRecognition` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_native/funasr.py)

```text
AutoModelForSpeechRecognition.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='asr_funasr',
    config=None,
    **model_kwargs,
)
```

```python
from voicehub import get_model_spec

spec = get_model_spec('asr_funasr')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('asr_funasr')` |
| Load and run | `AutoModelForSpeechRecognition` |
| Configure | `FunASRConfig` |
| Process | `AutoProcessor` |
| Model implementation | `FunASRForSpeechRecognition` |
| Normalized output | `ASROutput` |
| Training contract | `get_training_spec('asr_funasr')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).
