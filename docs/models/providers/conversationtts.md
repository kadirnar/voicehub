---
description: Public API, checkpoint, training, and optimization guide for the conversationtts integration.
---

# ConversationTTS {.vh-model-title}

## Usage

```bash
python -m pip install "voicehub @ git+https://github.com/kadirnar/voicehub.git@main"
```

Install from source, then choose a compatible checkpoint. Set the text and generation options, then inspect the returned audio.

```python
from pathlib import Path

from voicehub import AutoModelForTextToSpeech, TTSGenerationConfig

model = AutoModelForTextToSpeech.from_pretrained(
    'AudioFoundation/SpeechFoundation',
    model_type='conversationtts',
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

`conversationtts` is a VoiceHub **text to speech**
integration. This page is generated from its registry contract. [Open the `conversationtts` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/conversationtts.ipynb).

| Property | Value |
| --- | --- |
| Task | Text to speech |
| Architecture | `conversationtts` |
| Runtime | `VoiceHub-native` |
| Languages | `en`, `zh`, `yue` |
| Capabilities | `text-to-speech`, `voice-cloning`, `conversation`, `multilingual`, `fine-tuning`, `safetensors`, `voicehub-native`, `native-runtime`, `raw-audio-fine-tuning`, `preencoded-code-fine-tuning`, `noncommercial` |
| Reusable components | — |
| Normalized output | `TTSOutput` |

### Language support

<details class="vh-language-support" markdown>
<summary>Supported language abbreviations</summary>

`en`, `zh`, `yue`

These are the languages explicitly named in the upstream release README's podcast data.

</details>

## Paper and GitHub

- **Paper:** No dedicated upstream research paper is published for this integration.
- **Upstream GitHub:** [ConversationTTS](https://github.com/Audio-Foundation-Models/ConversationTTS)
- **VoiceHub source:** [VoiceHub model implementation](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/conversationtts/modeling_conversationtts.py)

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('conversationtts')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `conversationtts` |
| Configuration class | `ConversationTTSConfig` |
| Architecture class | `ConversationTTSForTextToSpeech` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'AudioFoundation/SpeechFoundation',
    model_type='conversationtts',
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
| Contract getter | `get_tts_dataset_spec('conversationtts')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `raw-text-audio` | — | text / texts; audio / audio_values | Source | at most one: text / texts; audio / audio_values; forbidden: text_token_ids, text_ids, audio_codes, codes |
| `raw-text-code` | — | text / texts; audio_codes / codes | Prepared | at most one: text / texts; audio_codes / codes; forbidden: text_token_ids, text_ids, audio, audio_values |
| `tokenized-text-audio` | — | text_token_ids / text_ids; audio / audio_values | Prepared | at most one: text_token_ids / text_ids; audio / audio_values; forbidden: text, texts, audio_codes, codes |
| `tokenized-text-code` | — | text_token_ids / text_ids; audio_codes / codes | Prepared | at most one: text_token_ids / text_ids; audio_codes / codes; forbidden: text, texts, audio, audio_values |
| `multi-codebook-batch` | `tokens`, `labels`, `tokens_mask` | — | Prepared | — |

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
| Training checkpoint | `AudioFoundation/SpeechFoundation` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `codec_language_model` | objective | `model` | `tokens`, `labels`, `tokens_mask` | `loss`, `codebook0_loss`, `residual_loss` |

The integration accepts its declared source or prepared contract directly. Call `model.validate_training_support()` first, then follow the
[training workflow](../../guides/training.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | [`AudioFoundation/SpeechFoundation`](https://huggingface.co/AudioFoundation/SpeechFoundation) |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.conversationtts.modeling_conversationtts.ConversationTTSForTextToSpeech` |
| Configuration | `voicehub.models.conversationtts.configuration_conversationtts.ConversationTTSConfig` |
| Source provenance | `voicehub/models/conversationtts/source/SOURCE.json` |
| License | [CC-BY-NC-4.0](https://github.com/Audio-Foundation-Models/ConversationTTS) |

Source, checkpoints, datasets, and evaluation tools are non-commercial. Commercial use: **not allowed**.

Confirm the checkpoint revision, access terms, provenance, and license.

### Limitations

- No integration-specific checkpoint limitation is registered. Verify the selected checkpoint revision and its documented runtime requirements.
- Validate memory, precision, and optional dependencies on the target system.
- Public optimizations fail closed when the runtime or hardware cannot satisfy
  their validation contract; an unavailable pass is not reported as applied.
- Contract tests do not replace the linked released-checkpoint evidence.

## Public API

Use the stable configuration, processor, and task-model facades below.

### `ConversationTTSConfig`

[View `ConversationTTSConfig` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/conversationtts/configuration_conversationtts.py)

```text
ConversationTTSConfig(**config_kwargs)
```

### `ConversationTTSForTextToSpeech`

[View `ConversationTTSForTextToSpeech` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/conversationtts/modeling_conversationtts.py)

```text
AutoModelForTextToSpeech.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='conversationtts',
    config=None,
    **model_kwargs,
)
```

```python
from voicehub import get_model_spec

spec = get_model_spec('conversationtts')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('conversationtts')` |
| Load and run | `AutoModelForTextToSpeech` |
| Configure | `ConversationTTSConfig` |
| Process | `AutoProcessor` |
| Model implementation | `ConversationTTSForTextToSpeech` |
| Normalized output | `TTSOutput` |
| Training contract | `get_training_spec('conversationtts')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).
