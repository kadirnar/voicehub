---
description: Public API, checkpoint, training, and optimization guide for the dia integration.
---

# Dia {.vh-model-title}

## Usage

```bash
python -m pip install "voicehub @ git+https://github.com/kadirnar/voicehub.git@main"
```

Install from source, then choose a compatible checkpoint. Set the text and generation options, then inspect the returned audio.

```python
from pathlib import Path

from voicehub import AutoModelForTextToSpeech, TTSGenerationConfig

model = AutoModelForTextToSpeech.from_pretrained(
    'nari-labs/Dia-1.6B-0626',
    model_type='dia',
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

`dia` is a VoiceHub **text to speech**
integration. This page is generated from its registry contract. [Open the `dia` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/dia.ipynb).

| Property | Value |
| --- | --- |
| Task | Text to speech |
| Architecture | `dia` |
| Runtime | `VoiceHub-native` |
| Languages | Checkpoint-defined; not exhaustively enumerated |
| Capabilities | `text-to-speech`, `dialogue`, `safetensors`, `fine-tuning`, `voicehub-native`, `native-runtime` |
| Reusable components | `dac` |
| Normalized output | `TTSOutput` |

### Language support

VoiceHub does not claim one exhaustive language list across compatible checkpoints; verify the selected checkpoint card and processor metadata.

## Paper and GitHub

- **Paper:** No dedicated upstream research paper is published for this integration.
- **Upstream GitHub:** [Dia](https://github.com/nari-labs/dia)
- **VoiceHub source:** [VoiceHub model implementation](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/dia/modeling_dia.py)

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('dia')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `dia` |
| Configuration class | `DiaConfig` |
| Architecture class | `DiaForTextToSpeech` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'nari-labs/Dia-1.6B-0626',
    model_type='dia',
)
print(type(processor).__name__)
```

## Inference

The Usage example returns `TTSOutput` through `AutoModelForTextToSpeech`.

### Input and output contract

| Property | Value |
| --- | --- |
| Readiness | `integrated-raw` |
| Data architecture | `sequence-to-sequence` |
| Sample rate | 44,100 Hz |
| Contract getter | `get_tts_dataset_spec('dia')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `raw-audio` | `text`, `audio` | — | Source | — |
| `processor-ready` | `input_ids`, `attention_mask`, `decoder_input_ids`, `decoder_attention_mask`, `labels` | — | Prepared | — |

Encoder text plus teacher-forced acoustic or codec targets. See the [data workflow](../../guides/data-preparation.md).

## Training and optimization

Use `available_optimization_passes()` to discover reversible public passes.
Unsupported runtime or hardware fails closed before mutation.

### Training contract

| Property | Value |
| --- | --- |
| Support | `native` |
| Family | `sequence-to-sequence` |
| Recipe | `single-phase` |
| Default phase | `codec_language_model` |
| Training checkpoint | `nari-labs/Dia-1.6B-0626` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `codec_language_model` | objective | `model` | `input_ids`, `attention_mask`, `decoder_input_ids`, `decoder_attention_mask`, `labels` | `loss` |

The integration accepts its declared source or prepared contract directly. Call `model.validate_training_support()` first, then follow the
[training workflow](../../guides/training.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | [`nari-labs/Dia-1.6B-0626`](https://huggingface.co/nari-labs/Dia-1.6B-0626) |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.dia.modeling_dia.DiaForTextToSpeech` |
| Configuration | `voicehub.models.dia.configuration_dia.DiaConfig` |
| Source provenance | `voicehub/architectures/dia/SOURCE.json` |
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

### `DiaConfig`

[View `DiaConfig` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/dia/configuration_dia.py)

```text
DiaConfig(**config_kwargs)
```

### `DiaForTextToSpeech`

[View `DiaForTextToSpeech` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/dia/modeling_dia.py)

```text
AutoModelForTextToSpeech.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='dia',
    config=None,
    **model_kwargs,
)
```

```python
from voicehub import get_model_spec

spec = get_model_spec('dia')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('dia')` |
| Load and run | `AutoModelForTextToSpeech` |
| Configure | `DiaConfig` |
| Process | `AutoProcessor` |
| Model implementation | `DiaForTextToSpeech` |
| Normalized output | `TTSOutput` |
| Training contract | `get_training_spec('dia')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).
