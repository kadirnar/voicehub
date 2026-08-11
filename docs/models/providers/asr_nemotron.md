---
description: Public API, checkpoint, training, and optimization guide for the asr_nemotron integration.
---

# Nemotron {.vh-model-title}

## Usage

Complete the [VoiceHub installation](../../getting-started/installation.md) once,
then run this repository-authored example. Model pages intentionally contain no
package-install command.

This example is maintained against VoiceHub's public API; it is not copied from an upstream demo or package README.

**Model-specific path:** Uses Nemotron's cache-aware native decoder and requests word timestamps.

**Inputs and controls:** Chunk geometry is owned by the checkpoint runtime; common chunk and stride overrides intentionally fail closed.

```python
from pathlib import Path

from voicehub import AutoModelForSpeechRecognition

AUDIO_FILE = Path("speech.wav")
if not AUDIO_FILE.is_file():
    raise FileNotFoundError(AUDIO_FILE)

model = AutoModelForSpeechRecognition.from_pretrained(
    'nvidia/nemotron-3.5-asr-streaming-0.6b',
    model_type='asr_nemotron',
    device="cuda",
    lazy_load=True,
)
output = model.transcribe(
    AUDIO_FILE,
    return_timestamps="word",
)
print(output.text)
for segment in output.segments:
    print(segment.start, segment.end, segment.text, segment.confidence)
```

Use authorized recordings. Verify hardware needs and pin a revision in production.

## Overview

`asr_nemotron` is a VoiceHub **automatic speech recognition**
integration. This page is generated from its registry contract. [Open the `asr_nemotron` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/asr_nemotron.ipynb).

| Property | Value |
| --- | --- |
| Task | Automatic speech recognition |
| Architecture | `nemotron-3.5-rnnt` |
| Runtime | `VoiceHub-native` |
| Languages | `en-US`, `en-GB`, `es-US`, `es-ES`, `fr-FR`, `fr-CA`, `it-IT`, `pt-BR`, `pt-PT`, `nl-NL`, `de-DE`, `tr-TR`, `ru-RU`, `ar-AR`, `hi-IN`, `ja-JP`, `ko-KR`, `vi-VN`, `uk-UA`, `pl-PL`, `sv-SE`, `cs-CZ`, `nb-NO`, `da-DK`, `bg-BG`, `fi-FI`, `hr-HR`, `sk-SK`, `zh-CN`, `hu-HU`, `ro-RO`, `et-EE`, `el-GR`, `lt-LT`, `lv-LV`, `mt-MT`, `sl-SI`, `he-IL`, `th-TH`, `nn-NO` |
| Capabilities | `automatic-speech-recognition`, `multilingual`, `language-identification`, `timestamps`, `streaming-architecture`, `safetensors`, `fine-tuning`, `voicehub-native`, `native-runtime` |
| Reusable components | — |
| Normalized output | `ASROutput` |

### Language support

<details class="vh-language-support" markdown>
<summary>Supported language abbreviations</summary>

`en-US`, `en-GB`, `es-US`, `es-ES`, `fr-FR`, `fr-CA`, `it-IT`, `pt-BR`, `pt-PT`, `nl-NL`, `de-DE`, `tr-TR`, `ru-RU`, `ar-AR`, `hi-IN`, `ja-JP`, `ko-KR`, `vi-VN`, `uk-UA`, `pl-PL`, `sv-SE`, `cs-CZ`, `nb-NO`, `da-DK`, `bg-BG`, `fi-FI`, `hr-HR`, `sk-SK`, `zh-CN`, `hu-HU`, `ro-RO`, `et-EE`, `el-GR`, `lt-LT`, `lv-LV`, `mt-MT`, `sl-SI`, `he-IL`, `th-TH`, `nn-NO`

The `el-GR`, `lt-LT`, `lv-LV`, `mt-MT`, `sl-SI`, `he-IL`, `th-TH`, and `nn-NO` locales are adaptation-ready and require in-domain fine-tuning; the other listed locales are transcription-ready or broad-coverage.

</details>

## Paper and GitHub

- **Paper:** No dedicated upstream research paper is published for this integration.
- **Upstream GitHub:** [NVIDIA NeMo](https://github.com/NVIDIA/NeMo)
- **VoiceHub source:** [VoiceHub model implementation](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_nemotron/modeling_asr_nemotron.py)

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('asr_nemotron')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `asr_nemotron` |
| Configuration class | `NemotronASRConfig` |
| Architecture class | `NemotronForSpeechRecognition` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'nvidia/nemotron-3.5-asr-streaming-0.6b',
    model_type='asr_nemotron',
)
print(type(processor).__name__)
```

## Inference

The Usage example returns `ASROutput` through `AutoModelForSpeechRecognition`.

### Input and output contract

| Property | Value |
| --- | --- |
| Readiness | `integrated-raw` |
| Data architecture | `rnnt` |
| Sample rate | 16,000 Hz |
| Contract getter | `get_asr_dataset_spec('asr_nemotron')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `raw-audio` | `audio` | text / transcription / transcript | Source | at most one: text / transcription / transcript |
| `nemotron-rnnt-model-ready` | `input_features`, `attention_mask`, `prompt_ids`, `labels`, `label_lengths`, `decoder_input_ids` | — | Prepared | — |

Language-prompted Nemotron RNN-T fine-tuning records. See the [data workflow](../../guides/speech-data.md).

## Training and optimization

Use `available_optimization_passes()` to discover reversible public passes.
Unsupported runtime or hardware fails closed before mutation.

### Training contract

| Property | Value |
| --- | --- |
| Support | `native` |
| Family | `rnnt` |
| Recipe | `single-phase` |
| Default phase | `speech_recognition` |
| Training checkpoint | `nvidia/nemotron-3.5-asr-streaming-0.6b` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `speech_recognition` | objective | `model.encoder`, `model.encoder_projector`, `model.prompt_projector`, `model.decoder`, `model.joint` | `input_features`, `attention_mask`, `prompt_ids`, `labels`, `label_lengths`, `decoder_input_ids` | `loss` |

The integration accepts its declared source or prepared contract directly. Call `model.validate_training_support()` first, then follow the
[training workflow](../../guides/training.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | [`nvidia/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) |
| Hugging Face ID | [`nvidia/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)<br>Repository availability verified through the Hugging Face model API on 2026-08-11; pin a revision before production use. |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.asr_nemotron.modeling_asr_nemotron.NemotronForSpeechRecognition` |
| Configuration | `voicehub.models.asr_nemotron.configuration_asr_nemotron.NemotronASRConfig` |
| Source provenance | `voicehub/architectures/nemotron_asr/SOURCE.json` |
| License | [OpenMDW-1.1](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) |

Use of the checkpoint and derivatives is governed by the OpenMDW-1.1 license. Commercial use: **allowed by the registered terms**.

Confirm the checkpoint revision, access terms, provenance, and license.

### Limitations

- No integration-specific checkpoint limitation is registered. Verify the selected checkpoint revision and its documented runtime requirements.
- Validate memory, precision, and optional dependencies on the target system.
- Public optimizations fail closed when the runtime or hardware cannot satisfy
  their validation contract; an unavailable pass is not reported as applied.
- Contract tests do not replace the linked released-checkpoint evidence.

## Public API

Use the stable configuration, processor, and task-model facades below.

### `NemotronASRConfig`

[View `NemotronASRConfig` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_nemotron/configuration_asr_nemotron.py)

```text
NemotronASRConfig(**config_kwargs)
```

### `NemotronForSpeechRecognition`

[View `NemotronForSpeechRecognition` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_nemotron/modeling_asr_nemotron.py)

```text
AutoModelForSpeechRecognition.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='asr_nemotron',
    config=None,
    **model_kwargs,
)
```

```python
from voicehub import get_model_spec

spec = get_model_spec('asr_nemotron')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('asr_nemotron')` |
| Load and run | `AutoModelForSpeechRecognition` |
| Configure | `NemotronASRConfig` |
| Process | `AutoProcessor` |
| Model implementation | `NemotronForSpeechRecognition` |
| Normalized output | `ASROutput` |
| Training contract | `get_training_spec('asr_nemotron')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).
