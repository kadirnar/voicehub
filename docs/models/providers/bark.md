---
description: Public API, checkpoint, training, and optimization guide for the bark integration.
---

# Bark {.vh-model-title}

## Usage

```bash
python -m pip install "voicehub @ git+https://github.com/kadirnar/voicehub.git@main"
```

Install from source, then choose a compatible checkpoint. Set the text and generation options, then inspect the returned audio.

```python
from pathlib import Path

from voicehub import AutoModelForTextToSpeech, TTSGenerationConfig

model = AutoModelForTextToSpeech.from_pretrained(
    'suno/bark-small',
    model_type='bark',
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

`bark` is a VoiceHub **text to speech**
integration. This page is generated from its registry contract. [Open the `bark` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/bark.ipynb).

| Property | Value |
| --- | --- |
| Task | Text to speech |
| Architecture | `bark` |
| Runtime | `VoiceHub-native` |
| Languages | `de`, `en`, `es`, `fr`, `hi`, `it`, `ja`, `ko`, `pl`, `pt`, `ru`, `tr`, `zh` |
| Capabilities | `text-to-speech`, `expressive-speech`, `voice-prompt`, `safetensors`, `fine-tuning`, `voicehub-native`, `native-runtime`, `preencoded-stage-training`, `restricted-pickle-conversion` |
| Reusable components | `encodec` |
| Normalized output | `TTSOutput` |

### Language support

<details class="vh-language-support" markdown>
<summary>Supported language abbreviations</summary>

`de`, `en`, `es`, `fr`, `hi`, `it`, `ja`, `ko`, `pl`, `pt`, `ru`, `tr`, `zh`

</details>

## Paper and GitHub

- **Paper:** No dedicated upstream research paper is published for this integration.
- **Upstream GitHub:** [Bark](https://github.com/suno-ai/bark)
- **VoiceHub source:** [VoiceHub model implementation](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/bark/modeling_bark.py)

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('bark')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `bark` |
| Configuration class | `BarkConfig` |
| Architecture class | `BarkForTextToSpeech` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'suno/bark-small',
    model_type='bark',
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
| Contract getter | `get_tts_dataset_spec('bark')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `causal-stage` | `input_ids`, `labels`, `training_phase` | — | Prepared | — |
| `fine-stage` | `input_ids`, `labels`, `codebook_idx`, `training_phase` | — | Prepared | — |
| `all-stages` | `semantic_input_ids`, `semantic_labels`, `coarse_input_ids`, `coarse_labels`, `fine_input_ids`, `fine_labels`, `codebook_idx` | — | Prepared | — |

Multi-component language-model, diffusion, acoustic, or GAN data. See the [data workflow](../../guides/data-preparation.md).

## Training and optimization

Use `available_optimization_passes()` to discover reversible public passes.
Unsupported runtime or hardware fails closed before mutation.

### Training contract

| Property | Value |
| --- | --- |
| Support | `preprocessed` |
| Family | `composite` |
| Recipe | `multi-phase` |
| Default phase | `semantic` |
| Training checkpoint | `suno/bark-small` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `semantic` | objective | `training_model.semantic` | `input_ids`, `labels` | `loss` |
| `coarse` | objective | `training_model.coarse` | `input_ids`, `labels` | `loss` |
| `fine` | objective | `training_model.fine` | `input_ids`, `labels`, `codebook_idx` | `loss` |

Prepare the exact tensors listed in the data contract before this step. Call `model.validate_training_support()` first, then follow the
[training workflow](../../guides/training.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | [`suno/bark-small`](https://huggingface.co/suno/bark-small) |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.bark.modeling_bark.BarkForTextToSpeech` |
| Configuration | `voicehub.models.bark.configuration_bark.BarkConfig` |
| Source provenance | `voicehub/architectures/bark/SOURCE.json` |
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

### `BarkConfig`

[View `BarkConfig` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/bark/configuration_bark.py)

```text
BarkConfig(**config_kwargs)
```

### `BarkForTextToSpeech`

[View `BarkForTextToSpeech` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/bark/modeling_bark.py)

```text
AutoModelForTextToSpeech.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='bark',
    config=None,
    **model_kwargs,
)
```

```python
from voicehub import get_model_spec

spec = get_model_spec('bark')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('bark')` |
| Load and run | `AutoModelForTextToSpeech` |
| Configure | `BarkConfig` |
| Process | `AutoProcessor` |
| Model implementation | `BarkForTextToSpeech` |
| Normalized output | `TTSOutput` |
| Training contract | `get_training_spec('bark')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).
