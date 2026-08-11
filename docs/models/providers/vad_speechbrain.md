---
description: Public API, checkpoint, training, and optimization guide for the vad_speechbrain integration.
---

# SpeechBrainVAD {.vh-model-title}

## Usage

Complete the [VoiceHub installation](../../getting-started/installation.md) once,
then run this repository-authored example. Model pages intentionally contain no
package-install command.

This example is maintained against VoiceHub's public API; it is not copied from an upstream demo or package README.

**Model-specific path:** Uses SpeechBrain CRDNN probabilities with explicit hysteresis and silence merging.

**Inputs and controls:** Tune onset and offset jointly; independent threshold changes can fragment segments.

```python
from pathlib import Path

from voicehub import AutoModelForVoiceActivityDetection

AUDIO_FILE = Path("speech.wav")
if not AUDIO_FILE.is_file():
    raise FileNotFoundError(AUDIO_FILE)

model = AutoModelForVoiceActivityDetection.from_pretrained(
    'speechbrain/vad-crdnn-libriparty',
    model_type='vad_speechbrain',
    device="cpu",
    lazy_load=True,
)
output = model.detect(
    AUDIO_FILE,
    onset=0.6,
    offset=0.4,
    min_silence_duration_ms=250,
)
for segment in output.segments:
    print(segment.start, segment.end, segment.score)
```

Use authorized recordings. Verify hardware needs and pin a revision in production.

## Overview

`vad_speechbrain` is a VoiceHub **voice activity detection**
integration. This page is generated from its registry contract. [Open the `vad_speechbrain` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/vad_speechbrain.ipynb).

| Property | Value |
| --- | --- |
| Task | Voice activity detection |
| Architecture | `speechbrain-crdnn-vad` |
| Runtime | `VoiceHub-native` |
| Languages | Not text-language conditioned |
| Capabilities | `voice-activity-detection`, `voicehub-native`, `safetensors`, `trusted-checkpoint-conversion`, `frame-scores`, `fine-tuning`, `offline-bidirectional` |
| Reusable components | — |
| Normalized output | `VADOutput` |

### Language support

The public VAD contract does not select a spoken language; validate checkpoint acoustic coverage on the target languages and recording conditions.

## Paper and GitHub

- **Paper:** [SpeechBrain: A General-Purpose Speech Toolkit](https://arxiv.org/abs/2106.04624)
- **Upstream GitHub:** [SpeechBrain](https://github.com/speechbrain/speechbrain)
- **VoiceHub source:** [VoiceHub model implementation](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/vad_speechbrain/modeling_vad_speechbrain.py)

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('vad_speechbrain')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `vad_speechbrain` |
| Configuration class | `SpeechBrainVADConfig` |
| Architecture class | `SpeechBrainVADForVoiceActivityDetection` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'speechbrain/vad-crdnn-libriparty',
    model_type='vad_speechbrain',
)
print(type(processor).__name__)
```

## Inference

The Usage example returns `VADOutput` through `AutoModelForVoiceActivityDetection`.

### Input and output contract

| Property | Value |
| --- | --- |
| Label boundary | Clip-, frame-, or segment-level labels |
| Required training inputs | `waveforms`, `labels` |

Use authorized audio and preserve annotation provenance. See the
[ASR and VAD data workflow](../../guides/speech-data.md).

## Training and optimization

Use `available_optimization_passes()` to discover reversible public passes.
Unsupported runtime or hardware fails closed before mutation.

### Training contract

| Property | Value |
| --- | --- |
| Support | `native` |
| Family | `frame-classification` |
| Recipe | `single-phase` |
| Default phase | `voice_activity_detection` |
| Training checkpoint | `speechbrain/vad-crdnn-libriparty` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `voice_activity_detection` | objective | `model` | `waveforms`, `labels` | `loss` |

The integration accepts its declared source or prepared contract directly. Call `model.validate_training_support()` first, then follow the
[training workflow](../../guides/training.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | [`speechbrain/vad-crdnn-libriparty`](https://huggingface.co/speechbrain/vad-crdnn-libriparty) |
| Hugging Face ID | [`speechbrain/vad-crdnn-libriparty`](https://huggingface.co/speechbrain/vad-crdnn-libriparty)<br>Repository availability verified through the Hugging Face model API on 2026-08-11; pin a revision before production use. |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cpu`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.vad_speechbrain.modeling_vad_speechbrain.SpeechBrainVADForVoiceActivityDetection` |
| Configuration | `voicehub.models.vad_speechbrain.configuration_vad_speechbrain.SpeechBrainVADConfig` |
| Source provenance | `voicehub/architectures/speechbrain_vad/SOURCE.json` |
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

### `SpeechBrainVADConfig`

[View `SpeechBrainVADConfig` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/vad_speechbrain/configuration_vad_speechbrain.py)

```text
SpeechBrainVADConfig(**config_kwargs)
```

### `SpeechBrainVADForVoiceActivityDetection`

[View `SpeechBrainVADForVoiceActivityDetection` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/vad_speechbrain/modeling_vad_speechbrain.py)

```text
AutoModelForVoiceActivityDetection.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='vad_speechbrain',
    config=None,
    **model_kwargs,
)
```

```python
from voicehub import get_model_spec

spec = get_model_spec('vad_speechbrain')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('vad_speechbrain')` |
| Load and run | `AutoModelForVoiceActivityDetection` |
| Configure | `SpeechBrainVADConfig` |
| Process | `AutoProcessor` |
| Model implementation | `SpeechBrainVADForVoiceActivityDetection` |
| Normalized output | `VADOutput` |
| Training contract | `get_training_spec('vad_speechbrain')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).
