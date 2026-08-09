---
description: Public API, checkpoint, training, and optimization guide for the vad_nemo integration.
---

# NeMoVAD {.vh-model-title}

## Usage

```bash
python -m pip install "voicehub @ git+https://github.com/kadirnar/voicehub.git@main"
```

Install from source, then choose a compatible checkpoint. Place a recording at `speech.wav`; tune the threshold on labeled audio.

```python
from voicehub import AutoModelForVoiceActivityDetection

model = AutoModelForVoiceActivityDetection.from_pretrained(
    'nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0',
    model_type='vad_nemo',
    device="cpu",
    lazy_load=True,
)
output = model.detect("speech.wav", threshold=0.5)
for segment in output.segments:
    print(segment.start, segment.end, segment.score)
```

Use authorized recordings. Verify hardware needs and pin a revision in production.

## Overview

`vad_nemo` is a VoiceHub **voice activity detection**
integration. This page is generated from its registry contract. [Open the `vad_nemo` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/vad_nemo.ipynb).

| Property | Value |
| --- | --- |
| Task | Voice activity detection |
| Architecture | `marblenet-vad` |
| Runtime | `VoiceHub-native` |
| Languages | Not text-language conditioned |
| Capabilities | `voice-activity-detection`, `voicehub-native`, `safetensors`, `trusted-checkpoint-conversion`, `frame-scores`, `fine-tuning` |
| Reusable components | — |
| Normalized output | `VADOutput` |

### Language support

The public VAD contract does not select a spoken language; validate checkpoint acoustic coverage on the target languages and recording conditions.

## Paper and GitHub

- **Paper:** [MarbleNet: Deep 1D Time-Channel Separable Convolutional Neural Network for VAD](https://arxiv.org/abs/2010.13886)
- **Upstream GitHub:** [NVIDIA NeMo](https://github.com/NVIDIA/NeMo)
- **VoiceHub source:** [VoiceHub model implementation](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/vad_nemo/modeling_vad_nemo.py)

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('vad_nemo')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `vad_nemo` |
| Configuration class | `NeMoVADConfig` |
| Architecture class | `NeMoVADForVoiceActivityDetection` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0',
    model_type='vad_nemo',
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
| Training checkpoint | `nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `voice_activity_detection` | objective | `model` | `waveforms`, `labels` | `loss` |

The integration accepts its declared source or prepared contract directly. Call `model.validate_training_support()` first, then follow the
[training workflow](../../guides/training.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | [`nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0`](https://huggingface.co/nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0) |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cpu`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.vad_nemo.modeling_vad_nemo.NeMoVADForVoiceActivityDetection` |
| Configuration | `voicehub.models.vad_nemo.configuration_vad_nemo.NeMoVADConfig` |
| Source provenance | `voicehub/architectures/marblenet_vad/SOURCE.json` |
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

### `NeMoVADConfig`

[View `NeMoVADConfig` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/vad_nemo/configuration_vad_nemo.py)

```text
NeMoVADConfig(**config_kwargs)
```

### `NeMoVADForVoiceActivityDetection`

[View `NeMoVADForVoiceActivityDetection` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/vad_nemo/modeling_vad_nemo.py)

```text
AutoModelForVoiceActivityDetection.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='vad_nemo',
    config=None,
    **model_kwargs,
)
```

```python
from voicehub import get_model_spec

spec = get_model_spec('vad_nemo')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('vad_nemo')` |
| Load and run | `AutoModelForVoiceActivityDetection` |
| Configure | `NeMoVADConfig` |
| Process | `AutoProcessor` |
| Model implementation | `NeMoVADForVoiceActivityDetection` |
| Normalized output | `VADOutput` |
| Training contract | `get_training_spec('vad_nemo')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).
