---
description: Public API, checkpoint, training, and optimization guide for the vibevoice integration.
---

# VibeVoice {.vh-model-title}

## Usage

```bash
python -m pip install "voicehub @ git+https://github.com/kadirnar/voicehub.git@main"
```

Install from source, then choose a compatible checkpoint. Set the text and generation options, then inspect the returned audio.

```python
from pathlib import Path

from voicehub import AutoModelForTextToSpeech, TTSGenerationConfig

model = AutoModelForTextToSpeech.from_pretrained(
    'microsoft/VibeVoice-Realtime-0.5B',
    model_type='vibevoice',
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

`vibevoice` is a VoiceHub **text to speech**
integration. This page is generated from its registry contract. [Open the `vibevoice` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/vibevoice.ipynb).

| Property | Value |
| --- | --- |
| Task | Text to speech |
| Architecture | `vibevoice-tts` |
| Runtime | `VoiceHub-native` |
| Languages | Checkpoint-defined; not exhaustively enumerated |
| Capabilities | `text-to-speech`, `voice-prompt`, `fine-tuning`, `default-checkpoint-inference-only`, `safetensors`, `voicehub-native`, `native-runtime`, `preprocessed-training`, `verified-low-level-realtime-stages`, `high-level-generation-fails-closed` |
| Reusable components | — |
| Normalized output | `TTSOutput` |

### Language support

VoiceHub does not claim one exhaustive language list across compatible checkpoints; verify the selected checkpoint card and processor metadata.

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('vibevoice')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `vibevoice` |
| Configuration class | `VibeVoiceConfig` |
| Architecture class | `VibeVoiceForTextToSpeech` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'microsoft/VibeVoice-Realtime-0.5B',
    model_type='vibevoice',
)
print(type(processor).__name__)
```

## Inference

The Usage example returns `TTSOutput` through `AutoModelForTextToSpeech`.

### Input and output contract

| Property | Value |
| --- | --- |
| Readiness | `preprocessed` |
| Data architecture | `hybrid` |
| Sample rate | 24,000 Hz |
| Contract getter | `get_tts_dataset_spec('vibevoice')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `lm-diffusion-batch` | `input_ids`, `attention_mask`, `speech_tensors`, `speech_masks`, `speeches_loss_input`, `speech_semantic_tensors`, `acoustic_input_mask`, `acoustic_loss_mask` | — | Prepared | — |

Multi-component language-model, diffusion, acoustic, or GAN data. See the [data workflow](../../guides/data-preparation.md).

## Training and optimization

Use `available_optimization_passes()` to discover reversible public passes.
Unsupported runtime or hardware fails closed before mutation.

### Training contract

| Property | Value |
| --- | --- |
| Support | `preprocessed` |
| Family | `composite` |
| Recipe | `single-phase` |
| Default phase | `lm_diffusion` |
| Training checkpoint | `microsoft/VibeVoice-1.5B` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `lm_diffusion` | objective | `model` | `input_ids`, `attention_mask`, `speech_tensors`, `speech_masks`, `speeches_loss_input`, `speech_semantic_tensors`, `acoustic_input_mask`, `acoustic_loss_mask` | `loss`, `ce_loss`, `diffusion_loss` |

Prepare the exact tensors listed in the data contract before this step. Call `model.validate_training_support()` first, then follow the
[training workflow](../../guides/training.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | [`microsoft/VibeVoice-Realtime-0.5B`](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B) |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.vibevoice.modeling_vibevoice.VibeVoiceForTextToSpeech` |
| Configuration | `voicehub.models.vibevoice.configuration_vibevoice.VibeVoiceConfig` |
| Source provenance | `voicehub/models/vibevoice/source/SOURCE.json` |
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

### `VibeVoiceConfig`

[View `VibeVoiceConfig` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/vibevoice/configuration_vibevoice.py)

```text
VibeVoiceConfig(**config_kwargs)
```

### `VibeVoiceForTextToSpeech`

[View `VibeVoiceForTextToSpeech` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/vibevoice/modeling_vibevoice.py)

```text
AutoModelForTextToSpeech.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='vibevoice',
    config=None,
    **model_kwargs,
)
```

```python
from voicehub import get_model_spec

spec = get_model_spec('vibevoice')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('vibevoice')` |
| Load and run | `AutoModelForTextToSpeech` |
| Configure | `VibeVoiceConfig` |
| Process | `AutoProcessor` |
| Model implementation | `VibeVoiceForTextToSpeech` |
| Normalized output | `TTSOutput` |
| Training contract | `get_training_spec('vibevoice')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).
