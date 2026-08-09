---
description: Public API, checkpoint, training, and optimization guide for the llasa integration.
---

# Llasa {.vh-model-title}

## Usage

```bash
python -m pip install "voicehub @ git+https://github.com/kadirnar/voicehub.git@main"
```

Install from source, then choose a compatible checkpoint. Set the text and generation options, then inspect the returned audio.

```python
from pathlib import Path

from voicehub import AutoModelForTextToSpeech, TTSGenerationConfig

model = AutoModelForTextToSpeech.from_pretrained(
    'HKUSTAudio/Llasa-1B-Multilingual',
    model_type='llasa',
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

`llasa` is a VoiceHub **text to speech**
integration. This page is generated from its registry contract. [Open the `llasa` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/llasa.ipynb).

| Property | Value |
| --- | --- |
| Task | Text to speech |
| Architecture | `llasa` |
| Runtime | `VoiceHub-native` |
| Languages | Checkpoint-defined; not exhaustively enumerated |
| Capabilities | `text-to-speech`, `voice-cloning`, `multilingual`, `fine-tuning`, `safetensors`, `voicehub-native`, `native-runtime`, `raw-audio-fine-tuning`, `preencoded-code-fine-tuning` |
| Reusable components | — |
| Normalized output | `TTSOutput` |

### Language support

VoiceHub does not claim one exhaustive language list across compatible checkpoints; verify the selected checkpoint card and processor metadata.

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('llasa')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `llasa` |
| Configuration class | `LlasaConfig` |
| Architecture class | `LlasaForTextToSpeech` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'HKUSTAudio/Llasa-1B-Multilingual',
    model_type='llasa',
)
print(type(processor).__name__)
```

## Inference

The Usage example returns `TTSOutput` through `AutoModelForTextToSpeech`.

### Input and output contract

| Property | Value |
| --- | --- |
| Readiness | `integrated-raw` |
| Data architecture | `codec-lm` |
| Sample rate | 16,000 Hz |
| Contract getter | `get_tts_dataset_spec('llasa')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `raw-audio` | `text` | audio / audio_codes | Source | — |
| `tokenized` | `input_ids`, `labels` | — | Prepared | — |

Autoregressive text/audio-token or codec-language-model data. See the [data workflow](../../guides/data-preparation.md).

## Training and optimization

Use `available_optimization_passes()` to discover reversible public passes.
Unsupported runtime or hardware fails closed before mutation.

### Training contract

| Property | Value |
| --- | --- |
| Support | `native` |
| Family | `causal-lm` |
| Recipe | `single-phase` |
| Default phase | `codec_language_model` |
| Training checkpoint | `HKUSTAudio/Llasa-1B-Multilingual` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `codec_language_model` | objective | `model` | `input_ids`, `attention_mask`, `labels` | `loss` |

The integration accepts its declared source or prepared contract directly. Call `model.validate_training_support()` first, then follow the
[training workflow](../../guides/training.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | [`HKUSTAudio/Llasa-1B-Multilingual`](https://huggingface.co/HKUSTAudio/Llasa-1B-Multilingual) |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.llasa.modeling_llasa.LlasaForTextToSpeech` |
| Configuration | `voicehub.models.llasa.configuration_llasa.LlasaConfig` |
| Source provenance | `voicehub/models/llasa/source/SOURCE.json` |
| License | [CC-BY-NC-4.0](https://huggingface.co/HKUSTAudio/xcodec2) |

The vendored XCodec2 component is restricted to non-commercial use. Commercial use: **not allowed**.

Confirm the checkpoint revision, access terms, provenance, and license.

### Limitations

- No integration-specific checkpoint limitation is registered. Verify the selected checkpoint revision and its documented runtime requirements.
- Validate memory, precision, and optional dependencies on the target system.
- Public optimizations fail closed when the runtime or hardware cannot satisfy
  their validation contract; an unavailable pass is not reported as applied.
- Contract tests do not replace the linked released-checkpoint evidence.

## Public API

Use the stable configuration, processor, and task-model facades below.

### `LlasaConfig`

[View `LlasaConfig` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/llasa/configuration_llasa.py)

```text
LlasaConfig(**config_kwargs)
```

### `LlasaForTextToSpeech`

[View `LlasaForTextToSpeech` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/llasa/modeling_llasa.py)

```text
AutoModelForTextToSpeech.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='llasa',
    config=None,
    **model_kwargs,
)
```

```python
from voicehub import get_model_spec

spec = get_model_spec('llasa')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('llasa')` |
| Load and run | `AutoModelForTextToSpeech` |
| Configure | `LlasaConfig` |
| Process | `AutoProcessor` |
| Model implementation | `LlasaForTextToSpeech` |
| Normalized output | `TTSOutput` |
| Training contract | `get_training_spec('llasa')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).
