---
description: Public API, checkpoint, training, and optimization guide for the styletts2 integration.
---

# StyleTTS2 {.vh-model-title}

## Usage

Complete the [VoiceHub installation](../../getting-started/installation.md) once,
then run this repository-authored example. Model pages intentionally contain no
package-install command.

This example is maintained against VoiceHub's public API; it is not copied from an upstream demo or package README.

**Model-specific path:** Uses an explicit local VoiceHub artifact and the native phoneme boundary required by StyleTTS 2.

**Inputs and controls:** Convert or review the upstream LibriTTS files first; the HF repository is provenance, not a drop-in VoiceHub directory.

```python
from pathlib import Path

from voicehub import AutoModelForTextToSpeech, TTSGenerationConfig

REFERENCE_AUDIO = Path("reference.wav")
REFERENCE_TEXT = "The reference transcript must exactly match the authorized audio."
if not REFERENCE_AUDIO.is_file():
    raise FileNotFoundError(REFERENCE_AUDIO)

model = AutoModelForTextToSpeech.from_pretrained(
    'checkpoints/styletts2/model.safetensors',
    model_type='styletts2',
    device="cuda",
    lazy_load=True,
)
output = model.generate(
    'həˈloʊ fɹʌm vɔɪs hʌb',
    generation_config=TTSGenerationConfig(
        seed=42,
        output_file=Path("output.wav"),
    ),
    speaker_audio_path=str(REFERENCE_AUDIO),
    text_is_phonemes=True,
    diffusion_steps=5,
    embedding_scale=1.0,
)
print(output.file_path, output.sample_rate, output.metadata)
```

Use authorized recordings. Verify hardware needs and pin a revision in production.

## Overview

`styletts2` is a VoiceHub **text to speech**
integration. This page is generated from its registry contract.

| Property | Value |
| --- | --- |
| Task | Text to speech |
| Architecture | `styletts2` |
| Runtime | `VoiceHub-native` |
| Languages | `en-US` |
| Capabilities | `text-to-speech`, `voice-cloning`, `fine-tuning`, `safetensors`, `voicehub-native`, `native-runtime`, `preprocessed-training`, `explicit-phonemes` |
| Reusable components | — |
| Normalized output | `TTSOutput` |

### Language support

<details class="vh-language-support" markdown>
<summary>Supported language abbreviations</summary>

`en-US`

</details>

## Paper and GitHub

- **Paper:** [StyleTTS 2: Towards Human-Level Text-to-Speech through Style Diffusion](https://arxiv.org/abs/2306.07691)
- **Upstream GitHub:** [StyleTTS 2](https://github.com/yl4579/StyleTTS2)
- **VoiceHub source:** [VoiceHub model implementation](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/styletts2/modeling_styletts2.py)

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('styletts2')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `styletts2` |
| Configuration class | `StyleTTS2Config` |
| Architecture class | `StyleTTS2ForTextToSpeech` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'checkpoints/styletts2/model.safetensors',
    model_type='styletts2',
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
| Sample rate | 24,000 Hz |
| Contract getter | `get_tts_dataset_spec('styletts2')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `explicit-features` | `input_ids`, `alignments`, `normalized_mel`, `reference_mel`, `f0_targets`, `noise_targets`, `audio_values` | — | Prepared | — |

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
| Training checkpoint | `owner/model-or-local-directory` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `generator` | generator | `training_model.model` | `input_ids`, `input_lengths`, `alignments`, `alignment_lengths`, `normalized_mel`, `normalized_mel_lengths`, `reference_mel`, `reference_mel_lengths`, `f0_targets`, `noise_targets`, `audio_values`, `audio_lengths` | `loss` |
| `discriminator` | discriminator | `training_model.mpd`, `training_model.msd` | `input_ids`, `input_lengths`, `alignments`, `alignment_lengths`, `normalized_mel`, `normalized_mel_lengths`, `reference_mel`, `reference_mel_lengths`, `f0_targets`, `noise_targets`, `audio_values`, `audio_lengths` | `loss` |

Prepare the exact tensors listed in the data contract before this step. Call `model.validate_training_support()` first, then follow the
[training workflow](../../guides/training.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | No default; pass a compatible Hub ID or local directory. |
| Hugging Face ID | [`yl4579/StyleTTS2-LibriTTS`](https://huggingface.co/yl4579/StyleTTS2-LibriTTS)<br>Upstream LibriTTS repository, verified available on 2026-08-11. VoiceHub requires a reviewed local artifact because the published layout is not a native VoiceHub directory. |
| Checkpoint status | No registry default; provide the compatible local artifact described on this page |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.styletts2.modeling_styletts2.StyleTTS2ForTextToSpeech` |
| Configuration | `voicehub.models.styletts2.configuration_styletts2.StyleTTS2Config` |
| Source provenance | `voicehub/models/styletts2/source/SOURCE.json` |
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

### `StyleTTS2Config`

[View `StyleTTS2Config` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/styletts2/configuration_styletts2.py)

```text
StyleTTS2Config(**config_kwargs)
```

### `StyleTTS2ForTextToSpeech`

[View `StyleTTS2ForTextToSpeech` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/styletts2/modeling_styletts2.py)

```text
AutoModelForTextToSpeech.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='styletts2',
    config=None,
    **model_kwargs,
)
```

```python
from voicehub import get_model_spec

spec = get_model_spec('styletts2')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('styletts2')` |
| Load and run | `AutoModelForTextToSpeech` |
| Configure | `StyleTTS2Config` |
| Process | `AutoProcessor` |
| Model implementation | `StyleTTS2ForTextToSpeech` |
| Normalized output | `TTSOutput` |
| Training contract | `get_training_spec('styletts2')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).
