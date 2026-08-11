---
description: Public API, checkpoint, training, and optimization guide for the csm integration.
---

# CSM {.vh-model-title}

## Usage

Complete the [VoiceHub installation](../../getting-started/installation.md) once,
then run this repository-authored example. Model pages intentionally contain no
package-install command.

This example is maintained against VoiceHub's public API; it is not copied from an upstream demo or package README.

**Model-specific path:** Builds CSM speaker context from a stable speaker index and paired reference recording.

**Inputs and controls:** Reference audio and text must be supplied together; speaker IDs must be non-negative.

```python
from pathlib import Path

from voicehub import AutoModelForTextToSpeech, TTSGenerationConfig

REFERENCE_AUDIO = Path("reference.wav")
REFERENCE_TEXT = "The reference transcript must exactly match the authorized audio."
if not REFERENCE_AUDIO.is_file():
    raise FileNotFoundError(REFERENCE_AUDIO)

model = AutoModelForTextToSpeech.from_pretrained(
    'sesame/csm-1b',
    model_type='csm',
    device="cuda",
    lazy_load=True,
)
output = model.generate(
    'VoiceHub keeps model integrations explicit and reproducible.',
    generation_config=TTSGenerationConfig(
        seed=42,
        output_file=Path("output.wav"),
    ),
    speaker=0,
    speaker_audio_path=str(REFERENCE_AUDIO),
    reference_text=REFERENCE_TEXT,
    max_audio_length_ms=30_000,
)
print(output.file_path, output.sample_rate, output.metadata)
```

Use authorized recordings. Verify hardware needs and pin a revision in production.

## Overview

`csm` is a VoiceHub **text to speech**
integration. This page is generated from its registry contract. [Open the `csm` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/csm.ipynb).

| Property | Value |
| --- | --- |
| Task | Text to speech |
| Architecture | `csm` |
| Runtime | `VoiceHub-native` |
| Languages | `en` |
| Capabilities | `text-to-speech`, `voice-cloning`, `conversation`, `safetensors`, `fine-tuning`, `raw-audio-training`, `preencoded-code-training`, `voicehub-native`, `native-runtime` |
| Reusable components | — |
| Normalized output | `TTSOutput` |

### Language support

<details class="vh-language-support" markdown>
<summary>Supported language abbreviations</summary>

`en`

</details>

## Paper and GitHub

- **Paper:** No dedicated upstream research paper is published for this integration.
- **Upstream GitHub:** [CSM](https://github.com/SesameAILabs/csm)
- **VoiceHub source:** [VoiceHub model implementation](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/csm/modeling_csm.py)

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('csm')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `csm` |
| Configuration class | `CSMConfig` |
| Architecture class | `CSMForTextToSpeech` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'sesame/csm-1b',
    model_type='csm',
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
| Contract getter | `get_tts_dataset_spec('csm')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `conversation` | — | conversation / messages | Source | — |
| `grouped-audios` | `texts`, `speaker_ids`, `audios` | — | Source | — |
| `grouped-concatenated` | `texts`, `speaker_ids`, `audio`, `audio_cut_idxs` | — | Source | — |
| `utterance` | `text`, `audio` | — | Source | — |
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
| Training checkpoint | `sesame/csm-1b` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `codec_language_model` | objective | `model` | — | `loss`, `backbone_loss`, `depth_decoder_loss` |

The integration accepts its declared source or prepared contract directly. Call `model.validate_training_support()` first, then follow the
[training workflow](../../guides/training.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | [`sesame/csm-1b`](https://huggingface.co/sesame/csm-1b) |
| Hugging Face ID | [`sesame/csm-1b`](https://huggingface.co/sesame/csm-1b)<br>Repository availability verified through the Hugging Face model API on 2026-08-11; pin a revision before production use. |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.csm.modeling_csm.CSMForTextToSpeech` |
| Configuration | `voicehub.models.csm.configuration_csm.CSMConfig` |
| Source provenance | `voicehub/models/csm/source/SOURCE.json` |
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

### `CSMConfig`

[View `CSMConfig` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/csm/configuration_csm.py)

```text
CSMConfig(**config_kwargs)
```

### `CSMForTextToSpeech`

[View `CSMForTextToSpeech` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/csm/modeling_csm.py)

```text
AutoModelForTextToSpeech.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='csm',
    config=None,
    **model_kwargs,
)
```

```python
from voicehub import get_model_spec

spec = get_model_spec('csm')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('csm')` |
| Load and run | `AutoModelForTextToSpeech` |
| Configure | `CSMConfig` |
| Process | `AutoProcessor` |
| Model implementation | `CSMForTextToSpeech` |
| Normalized output | `TTSOutput` |
| Training contract | `get_training_spec('csm')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).
