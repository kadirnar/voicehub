---
description: Public API, checkpoint, training, and optimization guide for the xtts integration.
---

# XTTS {.vh-model-title}

## Usage

Complete the [VoiceHub installation](../../getting-started/installation.md) once,
then run this repository-authored example. Model pages intentionally contain no
package-install command.

This example is maintained against VoiceHub's public API; it is not copied from an upstream demo or package README.

**Model-specific path:** Supplies the mandatory XTTS v2 speaker reference and a supported language code.

**Inputs and controls:** XTTS rejects missing reference files and unsupported checkpoint language codes before synthesis.

```python
from pathlib import Path

from voicehub import AutoModelForTextToSpeech, TTSGenerationConfig

REFERENCE_AUDIO = Path("reference.wav")
REFERENCE_TEXT = "The reference transcript must exactly match the authorized audio."
if not REFERENCE_AUDIO.is_file():
    raise FileNotFoundError(REFERENCE_AUDIO)

model = AutoModelForTextToSpeech.from_pretrained(
    'coqui/XTTS-v2',
    model_type='xtts',
    device="cuda",
    lazy_load=True,
)
output = model.generate(
    'VoiceHub keeps model integrations explicit and reproducible.',
    generation_config=TTSGenerationConfig(
        seed=42,
        output_file=Path("output.wav"),
    ),
    speaker_audio_path=str(REFERENCE_AUDIO),
    language="en",
    speed=1.0,
)
print(output.file_path, output.sample_rate, output.metadata)
```

Use authorized recordings. Verify hardware needs and pin a revision in production.

## Overview

`xtts` is a VoiceHub **text to speech**
integration. This page is generated from its registry contract. [Open the `xtts` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/xtts.ipynb).

| Property | Value |
| --- | --- |
| Task | Text to speech |
| Architecture | `xtts2` |
| Runtime | `VoiceHub-native` |
| Languages | `en`, `es`, `fr`, `de`, `it`, `pt`, `pl`, `tr`, `ru`, `nl`, `cs`, `ar`, `zh-CN`, `hu`, `ko`, `ja`, `hi` |
| Capabilities | `text-to-speech`, `voice-cloning`, `multilingual`, `fine-tuning`, `safetensors`, `voicehub-native`, `native-runtime`, `preencoded-code-fine-tuning`, `gpt-fine-tuning`, `restricted-pickle-conversion` |
| Reusable components | — |
| Normalized output | `TTSOutput` |

### Language support

<details class="vh-language-support" markdown>
<summary>Supported language abbreviations</summary>

`en`, `es`, `fr`, `de`, `it`, `pt`, `pl`, `tr`, `ru`, `nl`, `cs`, `ar`, `zh-CN`, `hu`, `ko`, `ja`, `hi`

</details>

## Paper and GitHub

- **Paper:** No dedicated upstream research paper is published for this integration.
- **Upstream GitHub:** [Coqui TTS](https://github.com/coqui-ai/TTS)
- **VoiceHub source:** [VoiceHub model implementation](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/xtts/modeling_xtts.py)

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('xtts')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `xtts` |
| Configuration class | `XTTSConfig` |
| Architecture class | `XTTSForTextToSpeech` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'coqui/XTTS-v2',
    model_type='xtts',
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
| Sample rate | 22,050 Hz |
| Contract getter | `get_tts_dataset_spec('xtts')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `native-gpt-tokens` | `text_inputs`, `text_lengths`, `audio_codes`, `wav_lengths` | cond_mels / cond_latents | Prepared | — |
| `native-gpt-waveform` | `text_inputs`, `text_lengths` | wav / audio_values; cond_mels / cond_latents | Prepared | at most one: wav / audio_values; forbidden: audio_codes |

Multi-component language-model, diffusion, acoustic, or GAN data. See the [data workflow](../../guides/data-preparation.md).

## Training and optimization

Use `available_optimization_passes()` to discover reversible public passes.
Unsupported runtime or hardware fails closed before mutation.

### Training contract

| Property | Value |
| --- | --- |
| Support | `preprocessed` |
| Family | `composite` |
| Recipe | `single-phase` |
| Default phase | `language_model` |
| Training checkpoint | `coqui/XTTS-v2` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `language_model` | objective | `model.gpt` | `text_inputs`, `text_lengths`, `audio_codes`, `wav_lengths` | `loss`, `loss_text_ce`, `loss_mel_ce` |

Prepare the exact tensors listed in the data contract before this step. Call `model.validate_training_support()` first, then follow the
[training workflow](../../guides/training.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | [`coqui/XTTS-v2`](https://huggingface.co/coqui/XTTS-v2) |
| Hugging Face ID | [`coqui/XTTS-v2`](https://huggingface.co/coqui/XTTS-v2)<br>Repository availability verified through the Hugging Face model API on 2026-08-11; pin a revision before production use. |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.xtts.modeling_xtts.XTTSForTextToSpeech` |
| Configuration | `voicehub.models.xtts.configuration_xtts.XTTSConfig` |
| Source provenance | `voicehub/models/xtts/source/SOURCE.json` |
| License | [CPML](https://huggingface.co/coqui/XTTS-v2) |

XTTS checkpoint terms are separate from the MPL-2.0 runtime source. Commercial use: **review required**.

Confirm the checkpoint revision, access terms, provenance, and license.

### Limitations

- No integration-specific checkpoint limitation is registered. Verify the selected checkpoint revision and its documented runtime requirements.
- Validate memory, precision, and optional dependencies on the target system.
- Public optimizations fail closed when the runtime or hardware cannot satisfy
  their validation contract; an unavailable pass is not reported as applied.
- Contract tests do not replace the linked released-checkpoint evidence.

## Public API

Use the stable configuration, processor, and task-model facades below.

### `XTTSConfig`

[View `XTTSConfig` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/xtts/configuration_xtts.py)

```text
XTTSConfig(**config_kwargs)
```

### `XTTSForTextToSpeech`

[View `XTTSForTextToSpeech` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/xtts/modeling_xtts.py)

```text
AutoModelForTextToSpeech.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='xtts',
    config=None,
    **model_kwargs,
)
```

```python
from voicehub import get_model_spec

spec = get_model_spec('xtts')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('xtts')` |
| Load and run | `AutoModelForTextToSpeech` |
| Configure | `XTTSConfig` |
| Process | `AutoProcessor` |
| Model implementation | `XTTSForTextToSpeech` |
| Normalized output | `TTSOutput` |
| Training contract | `get_training_spec('xtts')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).
