---
description: Public API, checkpoint, training, and optimization guide for the orpheustts integration.
---

# OrpheusTTS {.vh-model-title}

## Usage

```bash
python -m pip install "voicehub @ git+https://github.com/kadirnar/voicehub.git@main"
```

Install from source, then choose a compatible checkpoint. Provide an authorized `reference.wav` and its exact transcript when requested.

```python
from pathlib import Path

from voicehub import AutoModelForTextToSpeech, TTSGenerationConfig

model = AutoModelForTextToSpeech.from_pretrained(
    'canopylabs/orpheus-3b-0.1-ft',
    model_type='orpheustts',
    device="cuda",
    lazy_load=True,
)
generation_kwargs = {
    "voice": "tara",
}
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

`orpheustts` is a VoiceHub **text to speech**
integration. This page is generated from its registry contract. [Open the `orpheustts` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/orpheustts.ipynb).

| Property | Value |
| --- | --- |
| Task | Text to speech |
| Architecture | `causal-lm` |
| Runtime | `VoiceHub-native` |
| Languages | `en` |
| Capabilities | `text-to-speech`, `expressive-speech`, `safetensors`, `fine-tuning`, `voicehub-native`, `native-runtime` |
| Reusable components | — |
| Normalized output | `TTSOutput` |

### Language support

<details class="vh-language-support" markdown>
<summary>Supported language abbreviations</summary>

`en`

</details>

## Paper and GitHub

- **Paper:** No dedicated upstream research paper is published for this integration.
- **Upstream GitHub:** [Orpheus-TTS](https://github.com/canopyai/Orpheus-TTS)
- **VoiceHub source:** [VoiceHub model implementation](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/orpheustts/modeling_orpheustts.py)

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('orpheustts')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `orpheustts` |
| Configuration class | `OrpheusTTSConfig` |
| Architecture class | `OrpheusTTSForTextToSpeech` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'canopylabs/orpheus-3b-0.1-ft',
    model_type='orpheustts',
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
| Contract getter | `get_tts_dataset_spec('orpheustts')` |

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
| Training checkpoint | `canopylabs/orpheus-3b-0.1-ft` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `codec_language_model` | objective | `model` | — | `loss` |

The integration accepts its declared source or prepared contract directly. Call `model.validate_training_support()` first, then follow the
[training workflow](../../guides/training.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | [`canopylabs/orpheus-3b-0.1-ft`](https://huggingface.co/canopylabs/orpheus-3b-0.1-ft) |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.orpheustts.modeling_orpheustts.OrpheusTTSForTextToSpeech` |
| Configuration | `voicehub.models.orpheustts.configuration_orpheustts.OrpheusTTSConfig` |
| Source provenance | `voicehub/models/orpheustts/source/SOURCE.json` |
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

### `OrpheusTTSConfig`

[View `OrpheusTTSConfig` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/orpheustts/configuration_orpheustts.py)

```text
OrpheusTTSConfig(**config_kwargs)
```

### `OrpheusTTSForTextToSpeech`

[View `OrpheusTTSForTextToSpeech` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/orpheustts/modeling_orpheustts.py)

```text
AutoModelForTextToSpeech.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='orpheustts',
    config=None,
    **model_kwargs,
)
```

```python
from voicehub import get_model_spec

spec = get_model_spec('orpheustts')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('orpheustts')` |
| Load and run | `AutoModelForTextToSpeech` |
| Configure | `OrpheusTTSConfig` |
| Process | `AutoProcessor` |
| Model implementation | `OrpheusTTSForTextToSpeech` |
| Normalized output | `TTSOutput` |
| Training contract | `get_training_spec('orpheustts')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).
