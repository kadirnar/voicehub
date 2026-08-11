---
description: Public API, checkpoint, training, and optimization guide for the asr_hubert integration.
---

# Hubert {.vh-model-title}

## Usage

Complete the [VoiceHub installation](../../getting-started/installation.md) once,
then run this repository-authored example. Model pages intentionally contain no
package-install command.

This example is maintained against VoiceHub's public API; it is not copied from an upstream demo or package README.

**Model-specific path:** Uses the HuBERT CTC fine-tuned head with an explicit English transcription task.

**Inputs and controls:** The base HuBERT family is self-supervised; this exact HF ID includes the ASR head required here.

```python
from pathlib import Path

from voicehub import AutoModelForSpeechRecognition

AUDIO_FILE = Path("speech.wav")
if not AUDIO_FILE.is_file():
    raise FileNotFoundError(AUDIO_FILE)

model = AutoModelForSpeechRecognition.from_pretrained(
    'facebook/hubert-large-ls960-ft',
    model_type='asr_hubert',
    device="cuda",
    lazy_load=True,
)
output = model.transcribe(
    AUDIO_FILE,
    language="en",
    task="transcribe",
)
print(output.text)
for segment in output.segments:
    print(segment.start, segment.end, segment.text, segment.confidence)
```

Use authorized recordings. Verify hardware needs and pin a revision in production.

## Overview

`asr_hubert` is a VoiceHub **automatic speech recognition**
integration. This page is generated from its registry contract. [Open the `asr_hubert` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/asr_hubert.ipynb).

| Property | Value |
| --- | --- |
| Task | Automatic speech recognition |
| Architecture | `hubert` |
| Runtime | `VoiceHub-native` |
| Languages | `en` |
| Capabilities | `automatic-speech-recognition`, `timestamps`, `safetensors`, `fine-tuning`, `voicehub-native` |
| Reusable components | — |
| Normalized output | `ASROutput` |

### Language support

<details class="vh-language-support" markdown>
<summary>Supported language abbreviations</summary>

`en`

</details>

## Paper and GitHub

- **Paper:** [HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction](https://arxiv.org/abs/2106.07447)
- **Upstream GitHub:** [fairseq](https://github.com/facebookresearch/fairseq)
- **VoiceHub source:** [VoiceHub model implementation](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_hubert/modeling_asr_hubert.py)

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('asr_hubert')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `asr_hubert` |
| Configuration class | `HubertASRConfig` |
| Architecture class | `HubertForSpeechRecognition` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'facebook/hubert-large-ls960-ft',
    model_type='asr_hubert',
)
print(type(processor).__name__)
```

## Inference

The Usage example returns `ASROutput` through `AutoModelForSpeechRecognition`.

### Input and output contract

| Property | Value |
| --- | --- |
| Readiness | `integrated-raw` |
| Data architecture | `ctc` |
| Sample rate | 16,000 Hz |
| Contract getter | `get_asr_dataset_spec('asr_hubert')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `raw-audio` | `audio` | text / transcription / transcript | Source | at most one: text / transcription / transcript |
| `waveform-ctc` | `input_values`, `labels` | — | Prepared | — |

HuBERT waveform and CTC transcript records. See the [data workflow](../../guides/speech-data.md).

## Training and optimization

Use `available_optimization_passes()` to discover reversible public passes.
Unsupported runtime or hardware fails closed before mutation.

### Training contract

| Property | Value |
| --- | --- |
| Support | `native` |
| Family | `ctc` |
| Recipe | `single-phase` |
| Default phase | `speech_recognition` |
| Training checkpoint | `facebook/hubert-large-ls960-ft` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `speech_recognition` | objective | `model` | — | `loss` |

The integration accepts its declared source or prepared contract directly. Call `model.validate_training_support()` first, then follow the
[training workflow](../../guides/training.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | [`facebook/hubert-large-ls960-ft`](https://huggingface.co/facebook/hubert-large-ls960-ft) |
| Hugging Face ID | [`facebook/hubert-large-ls960-ft`](https://huggingface.co/facebook/hubert-large-ls960-ft)<br>Repository availability verified through the Hugging Face model API on 2026-08-11; pin a revision before production use. |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.asr_hubert.modeling_asr_hubert.HubertForSpeechRecognition` |
| Configuration | `voicehub.models.asr_hubert.configuration_asr_hubert.HubertASRConfig` |
| Source provenance | No integration-specific bundled `SOURCE.json` is declared for this registry entry. |
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

### `HubertASRConfig`

[View `HubertASRConfig` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_hubert/configuration_asr_hubert.py)

```text
HubertASRConfig(**config_kwargs)
```

### `HubertForSpeechRecognition`

[View `HubertForSpeechRecognition` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_hubert/modeling_asr_hubert.py)

```text
AutoModelForSpeechRecognition.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='asr_hubert',
    config=None,
    **model_kwargs,
)
```

```python
from voicehub import get_model_spec

spec = get_model_spec('asr_hubert')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('asr_hubert')` |
| Load and run | `AutoModelForSpeechRecognition` |
| Configure | `HubertASRConfig` |
| Process | `AutoProcessor` |
| Model implementation | `HubertForSpeechRecognition` |
| Normalized output | `ASROutput` |
| Training contract | `get_training_spec('asr_hubert')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).
