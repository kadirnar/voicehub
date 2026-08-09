---
description: Public API, checkpoint, training, and optimization guide for the openvoice integration.
---

# OpenVoice {.vh-model-title}

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
    'myshell-ai/OpenVoiceV2',
    model_type='openvoice',
    device="cuda",
    lazy_load=True,
)
generation_kwargs = {
    "speaker_audio_path": str(REFERENCE_AUDIO),
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

`openvoice` is a VoiceHub **text to speech**
integration. This page is generated from its registry contract. [Open the `openvoice` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/openvoice.ipynb).

| Property | Value |
| --- | --- |
| Task | Text to speech |
| Architecture | `openvoice-v2-converter` |
| Runtime | `VoiceHub-native` |
| Languages | 6 enumerated languages |
| Capabilities | `text-to-speech`, `voice-cloning`, `multilingual`, `fine-tuning`, `safetensors`, `voicehub-native`, `native-runtime`, `paired-waveform-training`, `explicit-base-waveform` |
| Reusable components | `wavmark` |
| Normalized output | `TTSOutput` |

### Language support

<details class="vh-language-support" markdown>
<summary>6 documented languages</summary>

`en`, `es`, `fr`, `zh`, `ja`, `ko`

</details>

## Paper and GitHub

- **Paper:** [OpenVoice: Versatile Instant Voice Cloning](https://arxiv.org/abs/2312.01479)
- **Upstream GitHub:** [OpenVoice](https://github.com/myshell-ai/OpenVoice)
- **VoiceHub source:** [VoiceHub model implementation](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/openvoice/modeling_openvoice.py)

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('openvoice')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `openvoice` |
| Configuration class | `OpenVoiceConfig` |
| Architecture class | `OpenVoiceForTextToSpeech` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'myshell-ai/OpenVoiceV2',
    model_type='openvoice',
)
print(type(processor).__name__)
```

## Inference

The Usage example returns `TTSOutput` through `AutoModelForTextToSpeech`.

### Input and output contract

| Property | Value |
| --- | --- |
| Readiness | `integrated-raw` |
| Data architecture | `vits` |
| Sample rate | 22,050 Hz |
| Contract getter | `get_tts_dataset_spec('openvoice')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `paired-waveforms` | `source_audio`, `target_audio` | — | Source | — |
| `paired-waveform-aliases` | `audio`, `target_waveform` | — | Source | — |

VITS/GAN text, waveform, spectrogram, and adversarial data. See the [data workflow](../../guides/data-preparation.md).

## Training and optimization

Use `available_optimization_passes()` to discover reversible public passes.
Unsupported runtime or hardware fails closed before mutation.

### Training contract

| Property | Value |
| --- | --- |
| Support | `custom` |
| Family | `vits` |
| Recipe | `single-phase` |
| Default phase | `generator` |
| Training checkpoint | `myshell-ai/OpenVoiceV2` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `generator` | generator | `model.enc_q`, `model.flow`, `model.dec`, `model.ref_enc` | `source_spectrogram`, `source_lengths`, `target_waveform`, `target_lengths` | `loss` |

This profile uses model-specific phases; inspect and honor each phase boundary. Call `model.validate_training_support()` first, then follow the
[training workflow](../../guides/training.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | [`myshell-ai/OpenVoiceV2`](https://huggingface.co/myshell-ai/OpenVoiceV2) |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.openvoice.modeling_openvoice.OpenVoiceForTextToSpeech` |
| Configuration | `voicehub.models.openvoice.configuration_openvoice.OpenVoiceConfig` |
| Source provenance | `voicehub/models/openvoice/source/SOURCE.json` |
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

### `OpenVoiceConfig`

[View `OpenVoiceConfig` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/openvoice/configuration_openvoice.py)

```text
OpenVoiceConfig(**config_kwargs)
```

### `OpenVoiceForTextToSpeech`

[View `OpenVoiceForTextToSpeech` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/openvoice/modeling_openvoice.py)

```text
AutoModelForTextToSpeech.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='openvoice',
    config=None,
    **model_kwargs,
)
```

```python
from voicehub import get_model_spec

spec = get_model_spec('openvoice')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('openvoice')` |
| Load and run | `AutoModelForTextToSpeech` |
| Configure | `OpenVoiceConfig` |
| Process | `AutoProcessor` |
| Model implementation | `OpenVoiceForTextToSpeech` |
| Normalized output | `TTSOutput` |
| Training contract | `get_training_spec('openvoice')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).
