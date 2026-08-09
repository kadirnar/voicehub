---
description: Public API, checkpoint, training, and optimization guide for the f5tts integration.
---

# F5TTS {.vh-model-title}

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
    'F5TTS_v1_Base',
    model_type='f5tts',
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

`f5tts` is a VoiceHub **text to speech**
integration. This page is generated from its registry contract.

| Property | Value |
| --- | --- |
| Task | Text to speech |
| Architecture | `f5tts` |
| Runtime | `VoiceHub-native` |
| Languages | 2 enumerated languages |
| Capabilities | `text-to-speech`, `voice-cloning`, `fine-tuning`, `flow-matching`, `safetensors`, `voicehub-native`, `native-runtime` |
| Reusable components | `vocos` |
| Normalized output | `TTSOutput` |

### Language support

<details class="vh-language-support" markdown>
<summary>2 documented languages</summary>

`en`, `zh`

</details>

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('f5tts')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `f5tts` |
| Configuration class | `F5TTSConfig` |
| Architecture class | `F5TTSForTextToSpeech` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'F5TTS_v1_Base',
    model_type='f5tts',
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
| Sample rate | 24,000 Hz |
| Contract getter | `get_tts_dataset_spec('f5tts')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `waveform-vocab` | `input_values`, `input_ids` | — | Prepared | — |
| `mel-features` | `input_ids` | mel / mel_spec | Prepared | — |
| `native-ready` | `inp`, `text` | — | Prepared | — |

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
| Default phase | `flow` |
| Training checkpoint | `F5TTS_v1_Base` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `flow` | objective | `model.ema_model` | `inp`, `text` | `loss` |

Prepare the exact tensors listed in the data contract before this step. Call `model.validate_training_support()` first, then follow the
[training workflow](../../guides/training.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | `F5TTS_v1_Base` |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.f5tts.modeling_f5tts.F5TTSForTextToSpeech` |
| Configuration | `voicehub.models.f5tts.configuration_f5tts.F5TTSConfig` |
| Source provenance | `voicehub/models/f5tts/source/SOURCE.json` |
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

### `F5TTSConfig`

[View `F5TTSConfig` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/f5tts/configuration_f5tts.py)

```text
F5TTSConfig(**config_kwargs)
```

### `F5TTSForTextToSpeech`

[View `F5TTSForTextToSpeech` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/f5tts/modeling_f5tts.py)

```text
AutoModelForTextToSpeech.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='f5tts',
    config=None,
    **model_kwargs,
)
```

```python
from voicehub import get_model_spec

spec = get_model_spec('f5tts')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('f5tts')` |
| Load and run | `AutoModelForTextToSpeech` |
| Configure | `F5TTSConfig` |
| Process | `AutoProcessor` |
| Model implementation | `F5TTSForTextToSpeech` |
| Normalized output | `TTSOutput` |
| Training contract | `get_training_spec('f5tts')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).
