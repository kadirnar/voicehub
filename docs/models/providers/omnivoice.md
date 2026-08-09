---
description: Public API, checkpoint, training, and optimization guide for the omnivoice integration.
---

# OmniVoice {.vh-model-title}

## Usage

```bash
python -m pip install "voicehub @ git+https://github.com/kadirnar/voicehub.git@main"
```

Install from source, then choose a compatible checkpoint. Set the text and generation options, then inspect the returned audio.

```python
from pathlib import Path

from voicehub import AutoModelForTextToSpeech, TTSGenerationConfig

model = AutoModelForTextToSpeech.from_pretrained(
    'k2-fsa/OmniVoice',
    model_type='omnivoice',
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

`omnivoice` is a VoiceHub **text to speech**
integration. This page is generated from its registry contract. [Open the `omnivoice` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/omnivoice.ipynb).

| Property | Value |
| --- | --- |
| Task | Text to speech |
| Architecture | `omnivoice` |
| Runtime | `VoiceHub-native` |
| Languages | Checkpoint-defined; not exhaustively enumerated |
| Capabilities | `text-to-speech`, `voice-cloning`, `voice-design`, `multilingual`, `fine-tuning`, `safetensors`, `voicehub-native`, `native-runtime`, `raw-audio-fine-tuning`, `preencoded-code-fine-tuning` |
| Reusable components | — |
| Normalized output | `TTSOutput` |

### Language support

VoiceHub does not claim one exhaustive language list across compatible checkpoints; verify the selected checkpoint card and processor metadata.

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('omnivoice')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `omnivoice` |
| Configuration class | `OmniVoiceConfig` |
| Architecture class | `OmniVoiceForTextToSpeech` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'k2-fsa/OmniVoice',
    model_type='omnivoice',
)
print(type(processor).__name__)
```

## Inference

The Usage example returns `TTSOutput` through `AutoModelForTextToSpeech`.

### Input and output contract

| Property | Value |
| --- | --- |
| Readiness | `integrated-raw` |
| Data architecture | `hybrid` |
| Sample rate | 24,000 Hz |
| Contract getter | `get_tts_dataset_spec('omnivoice')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `raw-audio` | `text` | audio / waveform | Source | at most one: audio / waveform; forbidden: audio_tokens |
| `audio-tokens` | `text`, `audio_tokens` | — | Prepared | forbidden: audio, waveform |

Multi-component language-model, diffusion, acoustic, or GAN data. See the [data workflow](../../guides/data-preparation.md).

## Training and optimization

Use `available_optimization_passes()` to discover reversible public passes.
Unsupported runtime or hardware fails closed before mutation.

### Training contract

| Property | Value |
| --- | --- |
| Support | `native` |
| Family | `composite` |
| Recipe | `single-phase` |
| Default phase | `masked_audio` |
| Training checkpoint | `k2-fsa/OmniVoice` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `masked_audio` | objective | `model` | `input_ids`, `audio_mask`, `labels` | `loss` |

The integration accepts its declared source or prepared contract directly. Call `model.validate_training_support()` first, then follow the
[training workflow](../../guides/training.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | [`k2-fsa/OmniVoice`](https://huggingface.co/k2-fsa/OmniVoice) |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.omnivoice.modeling_omnivoice.OmniVoiceForTextToSpeech` |
| Configuration | `voicehub.models.omnivoice.configuration_omnivoice.OmniVoiceConfig` |
| Source provenance | `voicehub/models/omnivoice/source/SOURCE.json` |
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

### `OmniVoiceConfig`

[View `OmniVoiceConfig` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/omnivoice/configuration_omnivoice.py)

```text
OmniVoiceConfig(**config_kwargs)
```

### `OmniVoiceForTextToSpeech`

[View `OmniVoiceForTextToSpeech` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/omnivoice/modeling_omnivoice.py)

```text
AutoModelForTextToSpeech.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='omnivoice',
    config=None,
    **model_kwargs,
)
```

```python
from voicehub import get_model_spec

spec = get_model_spec('omnivoice')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('omnivoice')` |
| Load and run | `AutoModelForTextToSpeech` |
| Configure | `OmniVoiceConfig` |
| Process | `AutoProcessor` |
| Model implementation | `OmniVoiceForTextToSpeech` |
| Normalized output | `TTSOutput` |
| Training contract | `get_training_spec('omnivoice')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).
