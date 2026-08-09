---
description: Public API, checkpoint, training, and optimization guide for the vad_sherpa_onnx integration.
---

# SherpaONNXVAD {.vh-model-title}

## Usage

```bash
python -m pip install "voicehub @ git+https://github.com/kadirnar/voicehub.git@main"
```

Install from source, then choose a compatible checkpoint. Place a recording at `speech.wav`; tune the threshold on labeled audio.

```python
from voicehub import AutoModelForVoiceActivityDetection

model = AutoModelForVoiceActivityDetection.from_pretrained(
    'safestack/silero-vad',
    model_type='vad_sherpa_onnx',
    device="cpu",
    lazy_load=True,
)
output = model.detect("speech.wav", threshold=0.5)
for segment in output.segments:
    print(segment.start, segment.end, segment.score)
```

Use authorized recordings. Verify hardware needs and pin a revision in production.

## Overview

`vad_sherpa_onnx` is a VoiceHub **voice activity detection**
integration. This page is generated from its registry contract. [Open the `vad_sherpa_onnx` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/vad_sherpa_onnx.ipynb).

| Property | Value |
| --- | --- |
| Task | Voice activity detection |
| Architecture | `native-vad-dispatch` |
| Runtime | `VoiceHub-native` |
| Languages | Not text-language conditioned |
| Capabilities | `voice-activity-detection`, `voicehub-native`, `safetensors`, `explicit-onnx-weight-conversion`, `fine-tuning`, `streaming`, `sherpa-compatible-segmentation`, `silero`, `ten-vad` |
| Reusable components | — |
| Normalized output | `VADOutput` |

### Language support

The public VAD contract does not select a spoken language; validate checkpoint acoustic coverage on the target languages and recording conditions.

## Paper and GitHub

- **Paper:** No dedicated upstream research paper is published for this integration.
- **Upstream GitHub:** [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)
- **VoiceHub source:** [VoiceHub model implementation](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/vad_sherpa_onnx/modeling_vad_sherpa_onnx.py)

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('vad_sherpa_onnx')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `vad_sherpa_onnx` |
| Configuration class | `SherpaONNXVADConfig` |
| Architecture class | `SherpaONNXVADForVoiceActivityDetection` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'safestack/silero-vad',
    model_type='vad_sherpa_onnx',
)
print(type(processor).__name__)
```

## Inference

The Usage example returns `VADOutput` through `AutoModelForVoiceActivityDetection`.

### Input and output contract

| Property | Value |
| --- | --- |
| Label boundary | Clip-, frame-, or segment-level labels |
| Required training inputs | `labels` |

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
| Training checkpoint | `safestack/silero-vad` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `voice_activity_detection` | objective | `model` | `labels` | `loss` |

The integration accepts its declared source or prepared contract directly. Call `model.validate_training_support()` first, then follow the
[training workflow](../../guides/training.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | [`safestack/silero-vad`](https://huggingface.co/safestack/silero-vad) |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cpu`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.vad_sherpa_onnx.modeling_vad_sherpa_onnx.SherpaONNXVADForVoiceActivityDetection` |
| Configuration | `voicehub.models.vad_sherpa_onnx.configuration_vad_sherpa_onnx.SherpaONNXVADConfig` |
| Source provenance | `voicehub/architectures/ten_vad/SOURCE.json` |
| License | [LicenseRef-TEN-VAD-Open-Source-License](https://github.com/TEN-framework/ten-vad) |

The provider's optional TEN family is governed by a non-standard license with additional deployment restrictions, including limits on competing with Agora. Review the bundled THIRD_PARTY_LICENSE before conversion, fine-tuning, distribution, or deployment. The default Silero family retains its own checkpoint terms. Commercial use: **review required**.

Confirm the checkpoint revision, access terms, provenance, and license.

### Limitations

- No integration-specific checkpoint limitation is registered. Verify the selected checkpoint revision and its documented runtime requirements.
- Validate memory, precision, and optional dependencies on the target system.
- Public optimizations fail closed when the runtime or hardware cannot satisfy
  their validation contract; an unavailable pass is not reported as applied.
- Contract tests do not replace the linked released-checkpoint evidence.

## Public API

Use the stable configuration, processor, and task-model facades below.

### `SherpaONNXVADConfig`

[View `SherpaONNXVADConfig` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/vad_sherpa_onnx/configuration_vad_sherpa_onnx.py)

```text
SherpaONNXVADConfig(**config_kwargs)
```

### `SherpaONNXVADForVoiceActivityDetection`

[View `SherpaONNXVADForVoiceActivityDetection` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/vad_sherpa_onnx/modeling_vad_sherpa_onnx.py)

```text
AutoModelForVoiceActivityDetection.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='vad_sherpa_onnx',
    config=None,
    **model_kwargs,
)
```

```python
from voicehub import get_model_spec

spec = get_model_spec('vad_sherpa_onnx')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('vad_sherpa_onnx')` |
| Load and run | `AutoModelForVoiceActivityDetection` |
| Configure | `SherpaONNXVADConfig` |
| Process | `AutoProcessor` |
| Model implementation | `SherpaONNXVADForVoiceActivityDetection` |
| Normalized output | `VADOutput` |
| Training contract | `get_training_spec('vad_sherpa_onnx')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).
