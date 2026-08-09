---
description: Public API, checkpoint, training, and optimization guide for the irodoritts integration.
---

# IrodoriTTS {.vh-model-title}

## Usage

```bash
python -m pip install "voicehub @ git+https://github.com/kadirnar/voicehub.git@main"
```

Install from source, then choose a compatible checkpoint. Set the text and generation options, then inspect the returned audio.

```python
from pathlib import Path

from voicehub import AutoModelForTextToSpeech, TTSGenerationConfig

model = AutoModelForTextToSpeech.from_pretrained(
    'Aratako/Irodori-TTS-500M-v3',
    model_type='irodoritts',
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

`irodoritts` is a VoiceHub **text to speech**
integration. This page is generated from its registry contract. [Open the `irodoritts` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/irodoritts.ipynb).

| Property | Value |
| --- | --- |
| Task | Text to speech |
| Architecture | `irodoritts-rf-dit` |
| Runtime | `VoiceHub-native` |
| Languages | Checkpoint-defined; not exhaustively enumerated |
| Capabilities | `text-to-speech`, `voice-cloning`, `voice-design`, `multilingual`, `fine-tuning`, `flow-matching`, `safetensors`, `voicehub-native`, `native-runtime`, `raw-audio-fine-tuning`, `preencoded-latent-fine-tuning`, `duration-prediction` |
| Reusable components | — |
| Normalized output | `TTSOutput` |

### Language support

VoiceHub does not claim one exhaustive language list across compatible checkpoints; verify the selected checkpoint card and processor metadata.

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('irodoritts')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `irodoritts` |
| Configuration class | `IrodoriTTSConfig` |
| Architecture class | `IrodoriTTSForTextToSpeech` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'Aratako/Irodori-TTS-500M-v3',
    model_type='irodoritts',
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
| Sample rate | 48,000 Hz |
| Contract getter | `get_tts_dataset_spec('irodoritts')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `raw-waveform` | `text` | waveform / audio | Source | at most one: waveform / audio; forbidden: target_latent, latent |
| `preencoded-latent` | `text` | target_latent / latent | Prepared | at most one: target_latent / latent; forbidden: waveform, audio |

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
| Default phase | `flow` |
| Training checkpoint | `Aratako/Irodori-TTS-500M-v3` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `flow` | objective | `model.model` | — | `loss`, `flow_loss`, `duration_loss` |

The integration accepts its declared source or prepared contract directly. Call `model.validate_training_support()` first, then follow the
[training workflow](../../guides/training.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | [`Aratako/Irodori-TTS-500M-v3`](https://huggingface.co/Aratako/Irodori-TTS-500M-v3) |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.irodoritts.modeling_irodoritts.IrodoriTTSForTextToSpeech` |
| Configuration | `voicehub.models.irodoritts.configuration_irodoritts.IrodoriTTSConfig` |
| Source provenance | `voicehub/models/irodoritts/source/SOURCE.json` |
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

### `IrodoriTTSConfig`

[View `IrodoriTTSConfig` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/irodoritts/configuration_irodoritts.py)

```text
IrodoriTTSConfig(**config_kwargs)
```

### `IrodoriTTSForTextToSpeech`

[View `IrodoriTTSForTextToSpeech` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/irodoritts/modeling_irodoritts.py)

```text
AutoModelForTextToSpeech.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='irodoritts',
    config=None,
    **model_kwargs,
)
```

```python
from voicehub import get_model_spec

spec = get_model_spec('irodoritts')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('irodoritts')` |
| Load and run | `AutoModelForTextToSpeech` |
| Configure | `IrodoriTTSConfig` |
| Process | `AutoProcessor` |
| Model implementation | `IrodoriTTSForTextToSpeech` |
| Normalized output | `TTSOutput` |
| Training contract | `get_training_spec('irodoritts')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).
