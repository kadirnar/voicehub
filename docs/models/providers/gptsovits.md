---
description: Public API, checkpoint, training, and optimization guide for the gptsovits integration.
---

# GPTSoVITS {.vh-model-title}

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
    'lj1995/GPT-SoVITS',
    model_type='gptsovits',
    device="cuda",
    lazy_load=True,
)
generation_kwargs = {
    "speaker_audio_path": str(REFERENCE_AUDIO),
    "prompt_text": REFERENCE_TEXT,
    "text_language": "en",
    "prompt_language": "en",
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

`gptsovits` is a VoiceHub **text to speech**
integration. This page is generated from its registry contract. [Open the `gptsovits` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/gptsovits.ipynb).

| Property | Value |
| --- | --- |
| Task | Text to speech |
| Architecture | `gptsovits` |
| Runtime | `VoiceHub-native` |
| Languages | 5 enumerated languages |
| Capabilities | `text-to-speech`, `voice-cloning`, `multilingual`, `fine-tuning`, `safetensors`, `voicehub-native`, `native-runtime`, `preprocessed-training`, `gpt-sovits-v1`, `gpt-sovits-v2`, `gpt-sovits-v2-pro`, `gpt-sovits-v2-pro-plus`, `prepared-pro-speaker-conditioning`, `variant-aware-safetensors-export` |
| Reusable components | — |
| Normalized output | `TTSOutput` |

### Language support

<details class="vh-language-support" markdown>
<summary>5 documented languages</summary>

`zh`, `en`, `ja`, `ko`, `yue`

Korean and Cantonese support applies to V2 and later variants.

</details>

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('gptsovits')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `gptsovits` |
| Configuration class | `GPTSoVITSConfig` |
| Architecture class | `GPTSoVITSForTextToSpeech` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'lj1995/GPT-SoVITS',
    model_type='gptsovits',
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
| Sample rate | 32,000 Hz |
| Contract getter | `get_tts_dataset_spec('gptsovits')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `s1-preprocessed` | `phoneme_ids`, `semantic_ids`, `bert_features` | — | Prepared | — |
| `s2-preprocessed` | `ssl_features`, `spectrogram`, `audio_values`, `phoneme_ids` | — | Prepared | — |
| `s2-pro-preprocessed` | `ssl_features`, `spectrogram`, `audio_values`, `phoneme_ids`, `speaker_embedding` | — | Prepared | — |

Multi-component language-model, diffusion, acoustic, or GAN data. See the [data workflow](../../guides/data-preparation.md).

## Training and optimization

Use `available_optimization_passes()` to discover reversible public passes.
Unsupported runtime or hardware fails closed before mutation.

### Training contract

| Property | Value |
| --- | --- |
| Support | `preprocessed` |
| Family | `composite` |
| Recipe | `adversarial` |
| Default phase | `s1` |
| Training checkpoint | `lj1995/GPT-SoVITS` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `s1` | objective | `training_model.s1` | `phoneme_ids`, `phoneme_lengths`, `semantic_ids`, `semantic_lengths`, `bert_features` | `loss` |
| `s2_generator` | generator | `training_model.s2.generator` | `ssl_features`, `spectrogram`, `spectrogram_lengths`, `audio_values`, `phoneme_ids`, `phoneme_lengths` | `loss` |
| `s2_discriminator` | discriminator | `training_model.s2.discriminator` | `ssl_features`, `spectrogram`, `spectrogram_lengths`, `audio_values`, `phoneme_ids`, `phoneme_lengths` | `loss` |

Prepare the exact tensors listed in the data contract before this step. Call `model.validate_training_support()` first, then follow the
[training workflow](../../guides/training.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | [`lj1995/GPT-SoVITS`](https://huggingface.co/lj1995/GPT-SoVITS) |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.gptsovits.modeling_gptsovits.GPTSoVITSForTextToSpeech` |
| Configuration | `voicehub.models.gptsovits.configuration_gptsovits.GPTSoVITSConfig` |
| Source provenance | `voicehub/models/gptsovits/source/SOURCE.json` |
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

### `GPTSoVITSConfig`

[View `GPTSoVITSConfig` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/gptsovits/configuration_gptsovits.py)

```text
GPTSoVITSConfig(**config_kwargs)
```

### `GPTSoVITSForTextToSpeech`

[View `GPTSoVITSForTextToSpeech` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/gptsovits/modeling_gptsovits.py)

```text
AutoModelForTextToSpeech.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='gptsovits',
    config=None,
    **model_kwargs,
)
```

```python
from voicehub import get_model_spec

spec = get_model_spec('gptsovits')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('gptsovits')` |
| Load and run | `AutoModelForTextToSpeech` |
| Configure | `GPTSoVITSConfig` |
| Process | `AutoProcessor` |
| Model implementation | `GPTSoVITSForTextToSpeech` |
| Normalized output | `TTSOutput` |
| Training contract | `get_training_spec('gptsovits')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).
