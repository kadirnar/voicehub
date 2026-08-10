---
description: Public API, checkpoint, training, and optimization guide for the neutts integration.
---

# NeuTTS {.vh-model-title}

## Usage

```bash
python -m pip install "voicehub @ git+https://github.com/kadirnar/voicehub.git@main"
```

Install from source, then choose a compatible checkpoint. Provide an authorized `reference.wav` and its exact transcript when requested.

```python
from pathlib import Path

REFERENCE_AUDIO = Path("reference.wav")
REFERENCE_TEXT = "This transcript must exactly match the authorized reference audio."

from voicehub import AutoModelForTextToSpeech, TTSGenerationConfig

model = AutoModelForTextToSpeech.from_pretrained(
    'neuphonic/neutts-2e',
    model_type='neutts',
    device="cuda",
    lazy_load=True,
)
generation_kwargs = {
    "speaker_audio_path": str(REFERENCE_AUDIO),
    "reference_text": REFERENCE_TEXT,
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

`neutts` is a VoiceHub **text to speech**
integration. This page is generated from its registry contract. [Open the `neutts` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/neutts.ipynb).

| Property | Value |
| --- | --- |
| Task | Text to speech |
| Architecture | `neutts` |
| Runtime | `VoiceHub-native` |
| Languages | `en` |
| Capabilities | `text-to-speech`, `voice-cloning`, `multilingual`, `emotion`, `safetensors`, `fine-tuning`, `default-checkpoint-inference-only`, `raw-audio-training`, `preencoded-code-training`, `voicehub-native`, `native-runtime` |
| Reusable components | — |
| Normalized output | `TTSOutput` |

### Language support

<details class="vh-language-support" markdown>
<summary>Supported language abbreviations</summary>

`en`

</details>

## Paper and GitHub

- **Paper:** No dedicated upstream research paper is published for this integration.
- **Upstream GitHub:** [NeuTTS](https://github.com/neuphonic/neutts)
- **VoiceHub source:** [VoiceHub model implementation](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/neutts/modeling_neutts.py)

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('neutts')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `neutts` |
| Configuration class | `NeuTTSConfig` |
| Architecture class | `NeuTTSForTextToSpeech` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'neuphonic/neutts-2e',
    model_type='neutts',
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
| Sample rate | Model/checkpoint specific |
| Contract getter | `get_tts_dataset_spec('neutts')` |

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
| Training checkpoint | `neuphonic/neutts-air` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `codec_language_model` | objective | `model.backbone` | `input_ids`, `labels` | `loss` |

The integration accepts its declared source or prepared contract directly. Call `model.validate_training_support()` first, then follow the
[training workflow](../../guides/training.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | [`neuphonic/neutts-2e`](https://huggingface.co/neuphonic/neutts-2e) |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.neutts.modeling_neutts.NeuTTSForTextToSpeech` |
| Configuration | `voicehub.models.neutts.configuration_neutts.NeuTTSConfig` |
| Source provenance | `voicehub/models/neutts/source/SOURCE.json` |
| License | [NeuTTS-Open-License-1.0](https://github.com/neuphonic/neutts) |

NeuTTS Air is Apache-2.0. Other registered variants use the NeuTTS Open License v1.0, which allows limited commercial use below its USD 5,000,000 annual-revenue threshold and requires a separate license at or above that threshold. Commercial use: **review required**.

Confirm the checkpoint revision, access terms, provenance, and license.

### Limitations

- No integration-specific checkpoint limitation is registered. Verify the selected checkpoint revision and its documented runtime requirements.
- Validate memory, precision, and optional dependencies on the target system.
- Public optimizations fail closed when the runtime or hardware cannot satisfy
  their validation contract; an unavailable pass is not reported as applied.
- Contract tests do not replace the linked released-checkpoint evidence.

## Public API

Use the stable configuration, processor, and task-model facades below.

### `NeuTTSConfig`

[View `NeuTTSConfig` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/neutts/configuration_neutts.py)

```text
NeuTTSConfig(**config_kwargs)
```

### `NeuTTSForTextToSpeech`

[View `NeuTTSForTextToSpeech` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/neutts/modeling_neutts.py)

```text
AutoModelForTextToSpeech.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='neutts',
    config=None,
    **model_kwargs,
)
```

```python
from voicehub import get_model_spec

spec = get_model_spec('neutts')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('neutts')` |
| Load and run | `AutoModelForTextToSpeech` |
| Configure | `NeuTTSConfig` |
| Process | `AutoProcessor` |
| Model implementation | `NeuTTSForTextToSpeech` |
| Normalized output | `TTSOutput` |
| Training contract | `get_training_spec('neutts')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).
