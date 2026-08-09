---
description: Public API, checkpoint, training, and optimization guide for the melotts integration.
---

# MeloTTS {.vh-model-title}

## Usage

```bash
python -m pip install "voicehub @ git+https://github.com/kadirnar/voicehub.git@main"
```

Install from source, then choose a compatible checkpoint. Set the text and generation options, then inspect the returned audio.

```python
from pathlib import Path

from voicehub import AutoModelForTextToSpeech, TTSGenerationConfig

model = AutoModelForTextToSpeech.from_pretrained(
    'EN',
    model_type='melotts',
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

`melotts` is a VoiceHub **text to speech**
integration. This page is generated from its registry contract.

| Property | Value |
| --- | --- |
| Task | Text to speech |
| Architecture | `melotts` |
| Runtime | `VoiceHub-native` |
| Languages | 6 enumerated languages |
| Capabilities | `text-to-speech`, `multilingual`, `fine-tuning`, `safetensors`, `voicehub-native`, `native-runtime`, `preprocessed-training`, `explicit-linguistic-features` |
| Reusable components | — |
| Normalized output | `TTSOutput` |

### Language support

<details class="vh-language-support" markdown>
<summary>6 documented languages</summary>

`en`, `fr`, `ja`, `es`, `zh`, `ko`

</details>

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('melotts')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `melotts` |
| Configuration class | `MeloTTSConfig` |
| Architecture class | `MeloTTSForTextToSpeech` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'EN',
    model_type='melotts',
)
print(type(processor).__name__)
```

## Inference

The Usage example returns `TTSOutput` through `AutoModelForTextToSpeech`.

### Input and output contract

| Property | Value |
| --- | --- |
| Readiness | `preprocessed` |
| Data architecture | `vits` |
| Sample rate | 44,100 Hz |
| Contract getter | `get_tts_dataset_spec('melotts')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `explicit-features` | `input_ids`, `tone_ids`, `language_ids`, `bert_features`, `ja_bert_features`, `spectrogram`, `audio_values`, `speaker_id` | — | Prepared | — |

VITS/GAN text, waveform, spectrogram, and adversarial data. See the [data workflow](../../guides/data-preparation.md).

## Training and optimization

Use `available_optimization_passes()` to discover reversible public passes.
Unsupported runtime or hardware fails closed before mutation.

### Training contract

| Property | Value |
| --- | --- |
| Support | `preprocessed` |
| Family | `vits` |
| Recipe | `adversarial` |
| Default phase | `generator` |
| Training checkpoint | `EN` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `generator` | generator | `training_model.model` | `input_ids`, `input_lengths`, `tone_ids`, `language_ids`, `bert_features`, `ja_bert_features`, `spectrogram`, `spectrogram_lengths`, `audio_values`, `audio_lengths`, `speaker_ids` | `loss` |
| `discriminator` | discriminator | `training_model.mpd` | `input_ids`, `input_lengths`, `tone_ids`, `language_ids`, `bert_features`, `ja_bert_features`, `spectrogram`, `spectrogram_lengths`, `audio_values`, `audio_lengths`, `speaker_ids` | `loss` |
| `duration_discriminator` | duration-discriminator | `training_model.duration_discriminator` | `input_ids`, `input_lengths`, `tone_ids`, `language_ids`, `bert_features`, `ja_bert_features`, `spectrogram`, `spectrogram_lengths`, `audio_values`, `audio_lengths`, `speaker_ids` | `loss` |

Prepare the exact tensors listed in the data contract before this step. Call `model.validate_training_support()` first, then follow the
[training workflow](../../guides/training.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | `EN` |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.melotts.modeling_melotts.MeloTTSForTextToSpeech` |
| Configuration | `voicehub.models.melotts.configuration_melotts.MeloTTSConfig` |
| Source provenance | `voicehub/models/melotts/source/SOURCE.json` |
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

### `MeloTTSConfig`

[View `MeloTTSConfig` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/melotts/configuration_melotts.py)

```text
MeloTTSConfig(**config_kwargs)
```

### `MeloTTSForTextToSpeech`

[View `MeloTTSForTextToSpeech` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/melotts/modeling_melotts.py)

```text
AutoModelForTextToSpeech.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='melotts',
    config=None,
    **model_kwargs,
)
```

```python
from voicehub import get_model_spec

spec = get_model_spec('melotts')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('melotts')` |
| Load and run | `AutoModelForTextToSpeech` |
| Configure | `MeloTTSConfig` |
| Process | `AutoProcessor` |
| Model implementation | `MeloTTSForTextToSpeech` |
| Normalized output | `TTSOutput` |
| Training contract | `get_training_spec('melotts')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).
