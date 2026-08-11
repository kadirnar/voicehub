---
description: Public API, checkpoint, training, and optimization guide for the vad_webrtc integration.
---

# WebRTCVAD {.vh-model-title}

## Usage

Complete the [VoiceHub installation](../../getting-started/installation.md) once,
then run this repository-authored example. Model pages intentionally contain no
package-install command.

This example is maintained against VoiceHub's public API; it is not copied from an upstream demo or package README.

**Model-specific path:** Runs weightless WebRTC VAD with frame-compatible duration controls.

**Inputs and controls:** Input is resampled and framed by VoiceHub; algorithm aggressiveness belongs to the model configuration.

```python
from pathlib import Path

from voicehub import AutoModelForVoiceActivityDetection

AUDIO_FILE = Path("speech.wav")
if not AUDIO_FILE.is_file():
    raise FileNotFoundError(AUDIO_FILE)

model = AutoModelForVoiceActivityDetection.from_pretrained(
    'webrtc-vad',
    model_type='vad_webrtc',
    device="cpu",
    lazy_load=True,
)
output = model.detect(
    AUDIO_FILE,
    min_speech_duration_ms=120,
    min_silence_duration_ms=240,
    speech_pad_ms=30,
)
for segment in output.segments:
    print(segment.start, segment.end, segment.score)
```

Use authorized recordings. Verify hardware needs and pin a revision in production.

## Overview

`vad_webrtc` is a VoiceHub **voice activity detection**
integration. This page is generated from its registry contract.

| Property | Value |
| --- | --- |
| Task | Voice activity detection |
| Architecture | `webrtc-vad` |
| Runtime | `VoiceHub-native` |
| Languages | Not text-language conditioned |
| Capabilities | `voice-activity-detection`, `fixed-point`, `voicehub-native`, `native-runtime`, `streaming` |
| Reusable components | — |
| Normalized output | `VADOutput` |

### Language support

The public VAD contract does not select a spoken language; validate checkpoint acoustic coverage on the target languages and recording conditions.

## Paper and GitHub

- **Paper:** No dedicated upstream research paper is published for this integration.
- **Upstream GitHub:** [py-webrtcvad](https://github.com/wiseman/py-webrtcvad)
- **VoiceHub source:** [VoiceHub model implementation](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/vad_webrtc/modeling_vad_webrtc.py)

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('vad_webrtc')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `vad_webrtc` |
| Configuration class | `WebRTCVADConfig` |
| Architecture class | `WebRTCVADForVoiceActivityDetection` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'webrtc-vad',
    model_type='vad_webrtc',
)
print(type(processor).__name__)
```

## Inference

The Usage example returns `VADOutput` through `AutoModelForVoiceActivityDetection`.

### Input and output contract

| Property | Value |
| --- | --- |
| Label boundary | No verified training dataset contract |
| Required training inputs | — |

Use authorized audio and preserve annotation provenance. See the
[ASR and VAD data workflow](../../guides/speech-data.md).

## Training and optimization

Use `available_optimization_passes()` to discover reversible public passes.
Unsupported runtime or hardware fails closed before mutation.

### Training contract

| Property | Value |
| --- | --- |
| Support | `inference-only` |
| Family | `upstream-native` |
| Recipe | `single-phase` |
| Default phase | `default` |
| Training checkpoint | `webrtc-vad` |
| Native training graph | `no` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `default` | objective | — | — | `loss`, `total_loss` |

This integration is **inference-only**. Choose a verified model from the
[training matrix](../training-support.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | `webrtc-vad` |
| Hugging Face ID | Not published / not applicable<br>Not applicable: WebRTC VAD is a weightless signal-processing algorithm. |
| Checkpoint status | Weightless algorithm; version the implementation and configuration, not model weights |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cpu`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.vad_webrtc.modeling_vad_webrtc.WebRTCVADForVoiceActivityDetection` |
| Configuration | `voicehub.models.vad_webrtc.configuration_vad_webrtc.WebRTCVADConfig` |
| Source provenance | `voicehub/architectures/webrtc_vad/SOURCE.json` |
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

### `WebRTCVADConfig`

[View `WebRTCVADConfig` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/vad_webrtc/configuration_vad_webrtc.py)

```text
WebRTCVADConfig(**config_kwargs)
```

### `WebRTCVADForVoiceActivityDetection`

[View `WebRTCVADForVoiceActivityDetection` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/vad_webrtc/modeling_vad_webrtc.py)

```text
AutoModelForVoiceActivityDetection.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='vad_webrtc',
    config=None,
    **model_kwargs,
)
```

```python
from voicehub import get_model_spec

spec = get_model_spec('vad_webrtc')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('vad_webrtc')` |
| Load and run | `AutoModelForVoiceActivityDetection` |
| Configure | `WebRTCVADConfig` |
| Process | `AutoProcessor` |
| Model implementation | `WebRTCVADForVoiceActivityDetection` |
| Normalized output | `VADOutput` |
| Training contract | `get_training_spec('vad_webrtc')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).
