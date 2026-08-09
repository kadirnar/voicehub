---
description: Public API, checkpoint, training, and optimization guide for the echo integration.
---

# EchoTTS {.vh-model-title}

## Usage

```bash
python -m pip install "voicehub @ git+https://github.com/kadirnar/voicehub.git@main"
```

Install from source, then choose a compatible checkpoint. Set the text and generation options, then inspect the returned audio.

```python
from pathlib import Path

from voicehub import AutoModelForTextToSpeech, TTSGenerationConfig

model = AutoModelForTextToSpeech.from_pretrained(
    'jordand/echo-tts-base',
    model_type='echo',
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

`echo` is a VoiceHub **text to speech**
integration. This page is generated from its registry contract. [Open the `echo` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/echo.ipynb).

| Property | Value |
| --- | --- |
| Task | Text to speech |
| Architecture | `echo-dit` |
| Runtime | `VoiceHub-native` |
| Languages | Checkpoint-defined; not exhaustively enumerated |
| Capabilities | `text-to-speech`, `voice-cloning`, `fine-tuning`, `flow-matching`, `safetensors`, `voicehub-native`, `native-runtime` |
| Reusable components | — |
| Normalized output | `TTSOutput` |

### Language support

VoiceHub does not claim one exhaustive language list across compatible checkpoints; verify the selected checkpoint card and processor metadata.

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('echo')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `echo` |
| Configuration class | `EchoTTSConfig` |
| Architecture class | `EchoTTSForTextToSpeech` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'jordand/echo-tts-base',
    model_type='echo',
)
print(type(processor).__name__)
```

## Inference

The Usage example returns `TTSOutput` through `AutoModelForTextToSpeech`.

### Input and output contract

| Property | Value |
| --- | --- |
| Readiness | `preprocessed` |
| Data architecture | `diffusion` |
| Sample rate | Model/checkpoint specific |
| Contract getter | `get_tts_dataset_spec('echo')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `flow-batch` | `target_latents`, `text_input_ids`, `text_mask`, `speaker_latents`, `speaker_mask` | — | Prepared | — |

Conditional flow-matching, rectified-flow, or diffusion data. See the [data workflow](../../guides/data-preparation.md).

## Training and optimization

Use `available_optimization_passes()` to discover reversible public passes.
Unsupported runtime or hardware fails closed before mutation.

### Training contract

| Property | Value |
| --- | --- |
| Support | `preprocessed` |
| Family | `flow-matching` |
| Recipe | `single-phase` |
| Default phase | `flow` |
| Training checkpoint | `jordand/echo-tts-base` |
| Native training graph | `no` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `flow` | objective | `model` | `target_latents`, `text_input_ids`, `text_mask`, `speaker_latents`, `speaker_mask` | `loss`, `flow_loss` |

Prepare the exact tensors listed in the data contract before this step. Call `model.validate_training_support()` first, then follow the
[training workflow](../../guides/training.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | [`jordand/echo-tts-base`](https://huggingface.co/jordand/echo-tts-base) |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.echo.modeling_echo.EchoTTSForTextToSpeech` |
| Configuration | `voicehub.models.echo.configuration_echo.EchoTTSConfig` |
| Source provenance | `voicehub/models/echo/source/SOURCE.json` |
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

### `EchoTTSConfig`

[View `EchoTTSConfig` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/echo/configuration_echo.py)

```text
EchoTTSConfig(**config_kwargs)
```

### `EchoTTSForTextToSpeech`

[View `EchoTTSForTextToSpeech` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/echo/modeling_echo.py)

```text
AutoModelForTextToSpeech.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='echo',
    config=None,
    **model_kwargs,
)
```

```python
from voicehub import get_model_spec

spec = get_model_spec('echo')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('echo')` |
| Load and run | `AutoModelForTextToSpeech` |
| Configure | `EchoTTSConfig` |
| Process | `AutoProcessor` |
| Model implementation | `EchoTTSForTextToSpeech` |
| Normalized output | `TTSOutput` |
| Training contract | `get_training_spec('echo')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).
