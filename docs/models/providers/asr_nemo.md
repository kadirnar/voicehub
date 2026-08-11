---
description: Public API, checkpoint, training, and optimization guide for the asr_nemo integration.
---

# NeMoASR {.vh-model-title}

## Usage

Complete the [VoiceHub installation](../../getting-started/installation.md) once,
then run this repository-authored example. Model pages intentionally contain no
package-install command.

This example is maintained against VoiceHub's public API; it is not copied from an upstream demo or package README.

**Model-specific path:** Runs VoiceHub's native QuartzNet15x5 graph from the pinned NeMo/NGC source.

**Inputs and controls:** The audited release is English-only and supports CTC word timestamps, not arbitrary NeMo architectures.

```python
from pathlib import Path

from voicehub import AutoModelForSpeechRecognition

AUDIO_FILE = Path("speech.wav")
if not AUDIO_FILE.is_file():
    raise FileNotFoundError(AUDIO_FILE)

model = AutoModelForSpeechRecognition.from_pretrained(
    'nvidia/nemo/stt_en_quartznet15x5',
    model_type='asr_nemo',
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

`asr_nemo` is a VoiceHub **automatic speech recognition**
integration. This page is generated from its registry contract.

| Property | Value |
| --- | --- |
| Task | Automatic speech recognition |
| Architecture | `nemo-asr` |
| Runtime | `VoiceHub-native` |
| Languages | `en` |
| Capabilities | `automatic-speech-recognition`, `english`, `timestamps`, `safetensors`, `fine-tuning`, `voicehub-native`, `ctc` |
| Reusable components | — |
| Normalized output | `ASROutput` |

### Language support

<details class="vh-language-support" markdown>
<summary>Supported language abbreviations</summary>

`en`

</details>

## Paper and GitHub

- **Paper:** [NeMo: a toolkit for building AI applications using Neural Modules](https://arxiv.org/abs/1909.09577)
- **Upstream GitHub:** [NVIDIA NeMo](https://github.com/NVIDIA/NeMo)
- **VoiceHub source:** [VoiceHub model implementation](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_nemo/__init__.py)

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('asr_nemo')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `asr_nemo` |
| Configuration class | `NeMoASRConfig` |
| Architecture class | `NeMoASRForSpeechRecognition` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'nvidia/nemo/stt_en_quartznet15x5',
    model_type='asr_nemo',
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
| Contract getter | `get_asr_dataset_spec('asr_nemo')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `raw-audio` | `audio` | text / transcription / transcript | Source | at most one: text / transcription / transcript |
| `nemo-ctc-waveform-model-ready` | `input_signal`, `input_signal_length`, `labels`, `label_lengths` | — | Prepared | — |
| `nemo-ctc-feature-model-ready` | `processed_signal`, `processed_signal_length`, `labels`, `label_lengths` | — | Prepared | — |

NeMo QuartzNet waveform and CTC transcript records. See the [data workflow](../../guides/speech-data.md).

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
| Training checkpoint | `nvidia/nemo/stt_en_quartznet15x5` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `speech_recognition` | objective | `model` | — | `loss` |

The integration accepts its declared source or prepared contract directly. Call `model.validate_training_support()` first, then follow the
[training workflow](../../guides/training.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | `nvidia/nemo/stt_en_quartznet15x5` |
| Hugging Face ID | Not published / not applicable<br>No canonical Hugging Face repository for the exact audited QuartzNet15x5 release; VoiceHub resolves the pinned NeMo/NGC artifact instead. |
| Checkpoint status | Pinned NeMo/NGC QuartzNet15x5 release; VoiceHub converts the exact audited graph into its safe native artifact boundary |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.asr_nemo.NeMoASRForSpeechRecognition` |
| Configuration | `voicehub.models.asr_nemo.NeMoASRConfig` |
| Source provenance | `voicehub/architectures/nemo_ctc/SOURCE.json` |
| License | [NVIDIA-NGC-Terms](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_en_quartznet15x5) |

The QuartzNet checkpoint is governed by the NVIDIA NGC Terms of Use; the VoiceHub-owned architecture code is Apache-2.0. Commercial use: **review required**.

Confirm the checkpoint revision, access terms, provenance, and license.

### Limitations

- No integration-specific checkpoint limitation is registered. Verify the selected checkpoint revision and its documented runtime requirements.
- Validate memory, precision, and optional dependencies on the target system.
- Public optimizations fail closed when the runtime or hardware cannot satisfy
  their validation contract; an unavailable pass is not reported as applied.
- Contract tests do not replace the linked released-checkpoint evidence.

## Public API

Use the stable configuration, processor, and task-model facades below.

### `NeMoASRConfig`

[View `NeMoASRConfig` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_nemo/__init__.py)

```text
NeMoASRConfig(**config_kwargs)
```

### `NeMoASRForSpeechRecognition`

[View `NeMoASRForSpeechRecognition` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_nemo/__init__.py)

```text
AutoModelForSpeechRecognition.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='asr_nemo',
    config=None,
    **model_kwargs,
)
```

```python
from voicehub import get_model_spec

spec = get_model_spec('asr_nemo')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('asr_nemo')` |
| Load and run | `AutoModelForSpeechRecognition` |
| Configure | `NeMoASRConfig` |
| Process | `AutoProcessor` |
| Model implementation | `NeMoASRForSpeechRecognition` |
| Normalized output | `ASROutput` |
| Training contract | `get_training_spec('asr_nemo')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).
