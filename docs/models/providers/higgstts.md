---
description: Public API, checkpoint, training, and optimization guide for the higgstts integration.
---

# HiggsTTS {.vh-model-title}

## Usage

```bash
python -m pip install "voicehub @ git+https://github.com/kadirnar/voicehub.git@main"
```

Install from source, then choose a compatible checkpoint. Set the text and generation options, then inspect the returned audio.

```python
from pathlib import Path

from voicehub import AutoModelForTextToSpeech, TTSGenerationConfig

model = AutoModelForTextToSpeech.from_pretrained(
    'bosonai/higgs-tts-2-3b-base',
    model_type='higgstts',
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

`higgstts` is a VoiceHub **text to speech**
integration. This page is generated from its registry contract. [Open the `higgstts` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/higgstts.ipynb).

| Property | Value |
| --- | --- |
| Task | Text to speech |
| Architecture | `higgs_audio_v2` |
| Runtime | `VoiceHub-native` |
| Languages | Checkpoint-defined; not exhaustively enumerated |
| Capabilities | `text-to-speech`, `voice-cloning`, `expressive-speech`, `multilingual`, `fine-tuning`, `safetensors`, `voicehub-native`, `native-runtime`, `raw-audio-fine-tuning`, `preencoded-code-fine-tuning` |
| Reusable components | — |
| Normalized output | `TTSOutput` |

### Language support

VoiceHub does not claim one exhaustive language list across compatible checkpoints; verify the selected checkpoint card and processor metadata.

## Paper and GitHub

- **Paper:** No dedicated upstream research paper is published for this integration.
- **Upstream GitHub:** [Higgs Audio](https://github.com/boson-ai/higgs-audio)
- **VoiceHub source:** [VoiceHub model implementation](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/higgstts/modeling_higgstts.py)

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('higgstts')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `higgstts` |
| Configuration class | `HiggsTTSConfig` |
| Architecture class | `HiggsTTSForTextToSpeech` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'bosonai/higgs-tts-2-3b-base',
    model_type='higgstts',
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
| Sample rate | 24,000 Hz |
| Contract getter | `get_tts_dataset_spec('higgstts')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `raw-audio` | `text` | audio / target_audio | Source | at most one: audio / target_audio; reference_audio / reference_codes; forbidden: audio_codes; reference_audio requires reference_text; reference_codes requires reference_text; reference_text requires one of reference_audio, reference_codes |
| `audio-codes` | `text`, `audio_codes` | — | Prepared | at most one: reference_audio / reference_codes; forbidden: audio, target_audio; reference_audio requires reference_text; reference_codes requires reference_text; reference_text requires one of reference_audio, reference_codes |

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
| Training checkpoint | `bosonai/higgs-tts-2-3b-base` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `codec_language_model` | objective | `model` | `input_ids`, `attention_mask`, `audio_input_ids`, `audio_input_ids_mask`, `labels`, `audio_labels` | `loss`, `text_loss`, `audio_loss` |

The integration accepts its declared source or prepared contract directly. Call `model.validate_training_support()` first, then follow the
[training workflow](../../guides/training.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | [`bosonai/higgs-tts-2-3b-base`](https://huggingface.co/bosonai/higgs-tts-2-3b-base) |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.higgstts.modeling_higgstts.HiggsTTSForTextToSpeech` |
| Configuration | `voicehub.models.higgstts.configuration_higgstts.HiggsTTSConfig` |
| Source provenance | `voicehub/models/higgstts/source/SOURCE.json` |
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

### `HiggsTTSConfig`

[View `HiggsTTSConfig` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/higgstts/configuration_higgstts.py)

```text
HiggsTTSConfig(**config_kwargs)
```

### `HiggsTTSForTextToSpeech`

[View `HiggsTTSForTextToSpeech` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/higgstts/modeling_higgstts.py)

```text
AutoModelForTextToSpeech.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='higgstts',
    config=None,
    **model_kwargs,
)
```

```python
from voicehub import get_model_spec

spec = get_model_spec('higgstts')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('higgstts')` |
| Load and run | `AutoModelForTextToSpeech` |
| Configure | `HiggsTTSConfig` |
| Process | `AutoProcessor` |
| Model implementation | `HiggsTTSForTextToSpeech` |
| Normalized output | `TTSOutput` |
| Training contract | `get_training_spec('higgstts')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).
