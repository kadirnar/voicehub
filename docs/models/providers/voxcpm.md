---
description: Public API, checkpoint, training, and optimization guide for the voxcpm integration.
---

# VoxCPM {.vh-model-title}

## Usage

```bash
python -m pip install "voicehub @ git+https://github.com/kadirnar/voicehub.git@main"
```

Install from source, then choose a compatible checkpoint. Set the text and generation options, then inspect the returned audio.

```python
from pathlib import Path

from voicehub import AutoModelForTextToSpeech, TTSGenerationConfig

model = AutoModelForTextToSpeech.from_pretrained(
    'openbmb/VoxCPM2',
    model_type='voxcpm',
    device="cuda",
    lazy_load=True,
)
generation_kwargs = {}
output = model.generate(
    "VoiceHub keeps model integrations consistent and easy to extend.",
    generation_config=TTSGenerationConfig(
        seed=42,
        output_file=Path("output.wav"),
    ),
    **generation_kwargs,
)
print(output.file_path, output.sample_rate)
```

Use authorized recordings. Verify hardware needs and pin a revision in production.

## Overview

`voxcpm` is a VoiceHub **text to speech**
integration. This page is generated from its registry contract. [Open the `voxcpm` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/voxcpm.ipynb).

| Property | Value |
| --- | --- |
| Task | Text to speech |
| Architecture | `voxcpm2` |
| Runtime | `VoiceHub-native` |
| Languages | Checkpoint-defined; not exhaustively enumerated |
| Capabilities | `text-to-speech`, `voice-cloning`, `voice-design`, `audio-continuation`, `multilingual`, `fine-tuning`, `safetensors`, `voicehub-native`, `native-runtime` |
| Reusable components | — |
| Normalized output | `TTSOutput` |

### Language support

VoiceHub does not claim one exhaustive language list across compatible checkpoints; verify the selected checkpoint card and processor metadata.

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('voxcpm')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `voxcpm` |
| Configuration class | `VoxCPMConfig` |
| Architecture class | `VoxCPMForTextToSpeech` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'openbmb/VoxCPM2',
    model_type='voxcpm',
)
print(type(processor).__name__)
```

## Inference

The Usage example returns `TTSOutput` through `AutoModelForTextToSpeech`.

### Input and output contract

| Property | Value |
| --- | --- |
| Readiness | `integrated-raw` |
| Data architecture | `diffusion` |
| Sample rate | 16,000 Hz |
| Contract getter | `get_tts_dataset_spec('voxcpm')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `raw-waveform` | `text` | audio / waveform | Source | at most one: audio / waveform; forbidden: audio_features |
| `audio-features` | `text`, `audio_features` | — | Prepared | forbidden: audio, waveform |

Conditional flow-matching, rectified-flow, or diffusion data. See the [data workflow](../../guides/data-preparation.md).

## Training and optimization

Use `available_optimization_passes()` to discover reversible public passes.
Unsupported runtime or hardware fails closed before mutation.

### Training contract

| Property | Value |
| --- | --- |
| Support | `native` |
| Family | `flow-matching` |
| Recipe | `single-phase` |
| Default phase | `source_flow_and_stop` |
| Training checkpoint | `openbmb/VoxCPM2` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `source_flow_and_stop` | objective | `model` | `text_tokens`, `text_mask`, `audio_feats`, `audio_mask`, `loss_mask`, `position_ids`, `labels` | `diffusion_loss`, `stop_loss` |

The integration accepts its declared source or prepared contract directly. Call `model.validate_training_support()` first, then follow the
[training workflow](../../guides/training.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | [`openbmb/VoxCPM2`](https://huggingface.co/openbmb/VoxCPM2) |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.voxcpm.modeling_voxcpm.VoxCPMForTextToSpeech` |
| Configuration | `voicehub.models.voxcpm.configuration_voxcpm.VoxCPMConfig` |
| Source provenance | `voicehub/models/voxcpm/source/SOURCE.json` |
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

### `VoxCPMConfig`

[View `VoxCPMConfig` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/voxcpm/configuration_voxcpm.py)

```text
VoxCPMConfig(**config_kwargs)
```

### `VoxCPMForTextToSpeech`

[View `VoxCPMForTextToSpeech` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/voxcpm/modeling_voxcpm.py)

```text
AutoModelForTextToSpeech.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='voxcpm',
    config=None,
    **model_kwargs,
)
```

```python
from voicehub import get_model_spec

spec = get_model_spec('voxcpm')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('voxcpm')` |
| Load and run | `AutoModelForTextToSpeech` |
| Configure | `VoxCPMConfig` |
| Process | `AutoProcessor` |
| Model implementation | `VoxCPMForTextToSpeech` |
| Normalized output | `TTSOutput` |
| Training contract | `get_training_spec('voxcpm')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).
