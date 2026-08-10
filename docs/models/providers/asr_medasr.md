---
description: Public API, checkpoint, training, and optimization guide for the asr_medasr integration.
---

# MedASR {.vh-model-title}

## Usage

```bash
python -m pip install "voicehub @ git+https://github.com/kadirnar/voicehub.git@main"
```

Install from source, then choose a compatible checkpoint. Place a supported recording at `speech.wav` and inspect the transcript.

```python
from voicehub import AutoModelForSpeechRecognition

model = AutoModelForSpeechRecognition.from_pretrained(
    'google/medasr',
    model_type='asr_medasr',
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

`asr_medasr` is a VoiceHub **automatic speech recognition**
integration. This page is generated from its registry contract. [Open the `asr_medasr` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/asr_medasr.ipynb).

| Property | Value |
| --- | --- |
| Task | Automatic speech recognition |
| Architecture | `lasr-ctc` |
| Runtime | `VoiceHub-native` |
| Languages | `en` |
| Capabilities | `automatic-speech-recognition`, `medical`, `gated-checkpoint`, `safetensors`, `fine-tuning`, `voicehub-native`, `native-runtime` |
| Reusable components | — |
| Normalized output | `ASROutput` |

### Language support

<details class="vh-language-support" markdown>
<summary>Supported language abbreviations</summary>

`en`

</details>

## Paper and GitHub

- **Paper:** No dedicated upstream research paper is published for this integration.
- **Upstream GitHub:** [MedASR](https://github.com/google-health/medasr)
- **VoiceHub source:** [VoiceHub model implementation](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_medasr/modeling_asr_medasr.py)

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('asr_medasr')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `asr_medasr` |
| Configuration class | `MedASRConfig` |
| Architecture class | `MedASRForSpeechRecognition` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'google/medasr',
    model_type='asr_medasr',
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
| Contract getter | `get_asr_dataset_spec('asr_medasr')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `raw-audio` | `audio` | text / transcription / transcript | Source | at most one: text / transcription / transcript |
| `medasr-model-ready` | `input_features`, `attention_mask`, `labels` | — | Prepared | — |

MedASR native LASR feature and CTC transcript records. See the [data workflow](../../guides/speech-data.md).

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
| Training checkpoint | `google/medasr` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `speech_recognition` | objective | `model.encoder`, `model.ctc_head` | `input_features`, `attention_mask`, `labels` | `loss` |

The integration accepts its declared source or prepared contract directly. Call `model.validate_training_support()` first, then follow the
[training workflow](../../guides/training.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | [`google/medasr`](https://huggingface.co/google/medasr) |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.asr_medasr.modeling_asr_medasr.MedASRForSpeechRecognition` |
| Configuration | `voicehub.models.asr_medasr.configuration_asr_medasr.MedASRConfig` |
| Source provenance | `voicehub/architectures/medasr/SOURCE.json` |
| License | [health-ai-developer-foundations](https://huggingface.co/google/medasr) |

Access requires accepting Google's Health AI Developer Foundations terms. Review the healthcare-specific use restrictions before fine-tuning or deployment. Commercial use: **review required**.

Confirm the checkpoint revision, access terms, provenance, and license.

### Limitations

- No integration-specific checkpoint limitation is registered. Verify the selected checkpoint revision and its documented runtime requirements.
- Validate memory, precision, and optional dependencies on the target system.
- Public optimizations fail closed when the runtime or hardware cannot satisfy
  their validation contract; an unavailable pass is not reported as applied.
- Contract tests do not replace the linked released-checkpoint evidence.

## Public API

Use the stable configuration, processor, and task-model facades below.

### `MedASRConfig`

[View `MedASRConfig` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_medasr/configuration_asr_medasr.py)

```text
MedASRConfig(**config_kwargs)
```

### `MedASRForSpeechRecognition`

[View `MedASRForSpeechRecognition` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_medasr/modeling_asr_medasr.py)

```text
AutoModelForSpeechRecognition.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='asr_medasr',
    config=None,
    **model_kwargs,
)
```

```python
from voicehub import get_model_spec

spec = get_model_spec('asr_medasr')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('asr_medasr')` |
| Load and run | `AutoModelForSpeechRecognition` |
| Configure | `MedASRConfig` |
| Process | `AutoProcessor` |
| Model implementation | `MedASRForSpeechRecognition` |
| Normalized output | `ASROutput` |
| Training contract | `get_training_spec('asr_medasr')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).
