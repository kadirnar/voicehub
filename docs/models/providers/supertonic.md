---
description: Public API, checkpoint, training, and optimization guide for the supertonic integration.
---

# Supertonic {.vh-model-title}

## Usage

```bash
python -m pip install "voicehub @ git+https://github.com/kadirnar/voicehub.git@main"
```

Install from source, then choose a compatible checkpoint. Set the text and generation options, then inspect the returned audio.

```python
from pathlib import Path

from voicehub import AutoModelForTextToSpeech, TTSGenerationConfig

model = AutoModelForTextToSpeech.from_pretrained(
    'Supertone/supertonic-3',
    model_type='supertonic',
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

`supertonic` is a VoiceHub **text to speech**
integration. This page is generated from its registry contract. [Open the `supertonic` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/supertonic.ipynb).

| Property | Value |
| --- | --- |
| Task | Text to speech |
| Architecture | `supertonic` |
| Runtime | `VoiceHub-native` |
| Languages | 32 enumerated languages |
| Capabilities | `text-to-speech`, `multilingual`, `fine-tuning`, `safetensors`, `voicehub-native`, `native-runtime`, `preprocessed-training` |
| Reusable components | — |
| Normalized output | `TTSOutput` |

### Language support

<details class="vh-language-support" markdown>
<summary>32 documented languages</summary>

`ar`, `bg`, `cs`, `da`, `de`, `el`, `en`, `es`, `et`, `fi`, `fr`, `hi`, `hr`, `hu`, `id`, `it`, `ja`, `ko`, `lt`, `lv`, `na`, `nl`, `pl`, `pt`, `ro`, `ru`, `sk`, `sl`, `sv`, `tr`, `uk`, `vi`

</details>

## Paper and GitHub

- **Paper:** No dedicated upstream research paper is published for this integration.
- **Upstream GitHub:** [Supertonic](https://github.com/supertone-inc/supertonic)
- **VoiceHub source:** [VoiceHub model implementation](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/supertonic/modeling_supertonic.py)

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('supertonic')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `supertonic` |
| Configuration class | `SupertonicConfig` |
| Architecture class | `SupertonicForTextToSpeech` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'Supertone/supertonic-3',
    model_type='supertonic',
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
| Sample rate | 44,100 Hz |
| Contract getter | `get_tts_dataset_spec('supertonic')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `text-style-object` | `text`, `style` | target_duration / duration / duration_seconds / target_latent / latent / latents | Prepared | — |
| `text-style-tensors` | `text`, `style_ttl`, `style_dp` | target_duration / duration / duration_seconds / target_latent / latent / latents | Prepared | — |
| `tokenized-style-object` | `text_ids`, `style` | text_mask / text_lengths; target_duration / duration / duration_seconds / target_latent / latent / latents | Prepared | — |
| `tokenized-style-tensors` | `text_ids`, `style_ttl`, `style_dp` | text_mask / text_lengths; target_duration / duration / duration_seconds / target_latent / latent / latents | Prepared | — |

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
| Default phase | `published_graph` |
| Training checkpoint | `Supertone/supertonic-3` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `published_graph` | objective | `model` | `text_ids`, `text_mask`, `style_ttl`, `style_dp` | `loss`, `duration_loss`, `flow_step_loss`, `vocoder_l1_loss` |

Prepare the exact tensors listed in the data contract before this step. Call `model.validate_training_support()` first, then follow the
[training workflow](../../guides/training.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | [`Supertone/supertonic-3`](https://huggingface.co/Supertone/supertonic-3) |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.supertonic.modeling_supertonic.SupertonicForTextToSpeech` |
| Configuration | `voicehub.models.supertonic.configuration_supertonic.SupertonicConfig` |
| Source provenance | `voicehub/models/supertonic/source/SOURCE.json` |
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

### `SupertonicConfig`

[View `SupertonicConfig` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/supertonic/configuration_supertonic.py)

```text
SupertonicConfig(**config_kwargs)
```

### `SupertonicForTextToSpeech`

[View `SupertonicForTextToSpeech` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/supertonic/modeling_supertonic.py)

```text
AutoModelForTextToSpeech.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='supertonic',
    config=None,
    **model_kwargs,
)
```

```python
from voicehub import get_model_spec

spec = get_model_spec('supertonic')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('supertonic')` |
| Load and run | `AutoModelForTextToSpeech` |
| Configure | `SupertonicConfig` |
| Process | `AutoProcessor` |
| Model implementation | `SupertonicForTextToSpeech` |
| Normalized output | `TTSOutput` |
| Training contract | `get_training_spec('supertonic')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).
