---
description: Public API, checkpoint, training, and optimization guide for the outetts integration.
---

# OuteTTS {.vh-model-title}

## Usage

```bash
python -m pip install "voicehub @ git+https://github.com/kadirnar/voicehub.git@main"
```

Install from source, then choose a compatible checkpoint. Set the text and generation options, then inspect the returned audio.

```python
from pathlib import Path

from voicehub import AutoModelForTextToSpeech, TTSGenerationConfig

model = AutoModelForTextToSpeech.from_pretrained(
    'OuteAI/Llama-OuteTTS-1.0-1B',
    model_type='outetts',
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

`outetts` is a VoiceHub **text to speech**
integration. This page is generated from its registry contract. [Open the `outetts` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/outetts.ipynb).

| Property | Value |
| --- | --- |
| Task | Text to speech |
| Architecture | `outetts` |
| Runtime | `VoiceHub-native` |
| Languages | Checkpoint-defined; not exhaustively enumerated |
| Capabilities | `text-to-speech`, `voice-cloning`, `fine-tuning`, `safetensors`, `voicehub-native`, `native-runtime`, `preprocessed-training`, `speaker-profile-training` |
| Reusable components | `dac` |
| Normalized output | `TTSOutput` |

### Language support

VoiceHub does not claim one exhaustive language list across compatible checkpoints; verify the selected checkpoint card and processor metadata.

## Paper and GitHub

- **Paper:** No dedicated upstream research paper is published for this integration.
- **Upstream GitHub:** [OuteTTS](https://github.com/edwko/OuteTTS)
- **VoiceHub source:** [VoiceHub model implementation](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/outetts/modeling_outetts.py)

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('outetts')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `outetts` |
| Configuration class | `OuteTTSConfig` |
| Architecture class | `OuteTTSForTextToSpeech` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'OuteAI/Llama-OuteTTS-1.0-1B',
    model_type='outetts',
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
| Sample rate | 24,000 Hz |
| Contract getter | `get_tts_dataset_spec('outetts')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `v3-profile` | — | speaker_profile / speaker / profile | Prepared | — |
| `inline-v3-profile` | `text`, `words`, `global_features` | — | Prepared | — |
| `tokenized` | `input_ids`, `labels` | — | Prepared | — |

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
| Default phase | `codec_language_model` |
| Training checkpoint | `OuteAI/Llama-OuteTTS-1.0-1B` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `codec_language_model` | objective | `model.language_model` | `input_ids`, `labels` | `loss` |

Prepare the exact tensors listed in the data contract before this step. Call `model.validate_training_support()` first, then follow the
[training workflow](../../guides/training.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | [`OuteAI/Llama-OuteTTS-1.0-1B`](https://huggingface.co/OuteAI/Llama-OuteTTS-1.0-1B) |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.outetts.modeling_outetts.OuteTTSForTextToSpeech` |
| Configuration | `voicehub.models.outetts.configuration_outetts.OuteTTSConfig` |
| Source provenance | `voicehub/models/outetts/source/SOURCE.json` |
| License | [CC-BY-NC-SA-4.0](https://huggingface.co/OuteAI/Llama-OuteTTS-1.0-1B) |

The default Llama-OuteTTS-1.0-1B checkpoint is non-commercial. The supported OuteTTS 0.6B checkpoint is Apache-2.0; review the selected artifact before training or deployment. Commercial use: **not allowed**.

Confirm the checkpoint revision, access terms, provenance, and license.

### Limitations

- No integration-specific checkpoint limitation is registered. Verify the selected checkpoint revision and its documented runtime requirements.
- Validate memory, precision, and optional dependencies on the target system.
- Public optimizations fail closed when the runtime or hardware cannot satisfy
  their validation contract; an unavailable pass is not reported as applied.
- Contract tests do not replace the linked released-checkpoint evidence.

## Public API

Use the stable configuration, processor, and task-model facades below.

### `OuteTTSConfig`

[View `OuteTTSConfig` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/outetts/configuration_outetts.py)

```text
OuteTTSConfig(**config_kwargs)
```

### `OuteTTSForTextToSpeech`

[View `OuteTTSForTextToSpeech` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/outetts/modeling_outetts.py)

```text
AutoModelForTextToSpeech.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='outetts',
    config=None,
    **model_kwargs,
)
```

```python
from voicehub import get_model_spec

spec = get_model_spec('outetts')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('outetts')` |
| Load and run | `AutoModelForTextToSpeech` |
| Configure | `OuteTTSConfig` |
| Process | `AutoProcessor` |
| Model implementation | `OuteTTSForTextToSpeech` |
| Normalized output | `TTSOutput` |
| Training contract | `get_training_spec('outetts')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).
