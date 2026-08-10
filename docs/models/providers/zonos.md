---
description: Public API, checkpoint, training, and optimization guide for the zonos integration.
---

# Zonos {.vh-model-title}

## Usage

```bash
python -m pip install "voicehub @ git+https://github.com/kadirnar/voicehub.git@main"
```

Install from source, then choose a compatible checkpoint. Set the text and generation options, then inspect the returned audio.

```python
from pathlib import Path

from voicehub import AutoModelForTextToSpeech, TTSGenerationConfig

model = AutoModelForTextToSpeech.from_pretrained(
    'Zyphra/Zonos-v0.1-transformer',
    model_type='zonos',
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

`zonos` is a VoiceHub **text to speech**
integration. This page is generated from its registry contract. [Open the `zonos` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/zonos.ipynb).

| Property | Value |
| --- | --- |
| Task | Text to speech |
| Architecture | `zonos` |
| Runtime | `VoiceHub-native` |
| Languages | `en`, `ja`, `zh`, `fr`, `de` |
| Capabilities | `text-to-speech`, `voice-cloning`, `multilingual`, `fine-tuning`, `safetensors`, `voicehub-native`, `native-runtime` |
| Reusable components | `dac` |
| Normalized output | `TTSOutput` |

### Language support

<details class="vh-language-support" markdown>
<summary>Supported language abbreviations</summary>

`en`, `ja`, `zh`, `fr`, `de`

</details>

## Paper and GitHub

- **Paper:** No dedicated upstream research paper is published for this integration.
- **Upstream GitHub:** [Zonos](https://github.com/Zyphra/Zonos)
- **VoiceHub source:** [VoiceHub model implementation](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/zonos/modeling_zonos.py)

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('zonos')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `zonos` |
| Configuration class | `ZonosConfig` |
| Architecture class | `ZonosForTextToSpeech` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'Zyphra/Zonos-v0.1-transformer',
    model_type='zonos',
)
print(type(processor).__name__)
```

## Inference

The Usage example returns `TTSOutput` through `AutoModelForTextToSpeech`.

### Input and output contract

| Property | Value |
| --- | --- |
| Readiness | `preprocessed` |
| Data architecture | `codec-lm` |
| Sample rate | Model/checkpoint specific |
| Contract getter | `get_tts_dataset_spec('zonos')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `codec-batch` | `prefix_conditioning`, `audio_codes` | — | Prepared | — |

Autoregressive text/audio-token or codec-language-model data. See the [data workflow](../../guides/data-preparation.md).

## Training and optimization

Use `available_optimization_passes()` to discover reversible public passes.
Unsupported runtime or hardware fails closed before mutation.

### Training contract

| Property | Value |
| --- | --- |
| Support | `preprocessed` |
| Family | `causal-lm` |
| Recipe | `single-phase` |
| Default phase | `reconstructed_codec_language_model` |
| Training checkpoint | `Zyphra/Zonos-v0.1-transformer` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `reconstructed_codec_language_model` | objective | `model` | `prefix_conditioning`, `audio_codes` | `loss`, `codec_ce_loss` |

Prepare the exact tensors listed in the data contract before this step. Call `model.validate_training_support()` first, then follow the
[training workflow](../../guides/training.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | [`Zyphra/Zonos-v0.1-transformer`](https://huggingface.co/Zyphra/Zonos-v0.1-transformer) |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.zonos.modeling_zonos.ZonosForTextToSpeech` |
| Configuration | `voicehub.models.zonos.configuration_zonos.ZonosConfig` |
| Source provenance | `voicehub/models/zonos/source/SOURCE.json` |
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

### `ZonosConfig`

[View `ZonosConfig` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/zonos/configuration_zonos.py)

```text
ZonosConfig(**config_kwargs)
```

### `ZonosForTextToSpeech`

[View `ZonosForTextToSpeech` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/zonos/modeling_zonos.py)

```text
AutoModelForTextToSpeech.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='zonos',
    config=None,
    **model_kwargs,
)
```

```python
from voicehub import get_model_spec

spec = get_model_spec('zonos')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('zonos')` |
| Load and run | `AutoModelForTextToSpeech` |
| Configure | `ZonosConfig` |
| Process | `AutoProcessor` |
| Model implementation | `ZonosForTextToSpeech` |
| Normalized output | `TTSOutput` |
| Training contract | `get_training_spec('zonos')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).
