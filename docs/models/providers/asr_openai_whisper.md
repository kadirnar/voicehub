---
description: Public API, checkpoint, training, and optimization guide for the asr_openai_whisper integration.
---

# OpenAIWhisper {.vh-model-title}

## Usage

Complete the [VoiceHub installation](../../getting-started/installation.md) once,
then run this repository-authored example. Model pages intentionally contain no
package-install command.

This example is maintained against VoiceHub's public API; it is not copied from an upstream demo or package README.

**Model-specific path:** Runs the original OpenAI Whisper backend with deterministic beam decoding.

**Inputs and controls:** This integration is distinct from native Whisper and faster-whisper even when they share an HF checkpoint ID.

```python
from pathlib import Path

from voicehub import AutoModelForSpeechRecognition

AUDIO_FILE = Path("speech.wav")
if not AUDIO_FILE.is_file():
    raise FileNotFoundError(AUDIO_FILE)

model = AutoModelForSpeechRecognition.from_pretrained(
    'openai/whisper-small',
    model_type='asr_openai_whisper',
    device="cuda",
    lazy_load=True,
)
output = model.transcribe(
    AUDIO_FILE,
    language="en",
    task="transcribe",
    num_beams=5,
)
print(output.text)
for segment in output.segments:
    print(segment.start, segment.end, segment.text, segment.confidence)
```

Use authorized recordings. Verify hardware needs and pin a revision in production.

## Overview

`asr_openai_whisper` is a VoiceHub **automatic speech recognition**
integration. This page is generated from its registry contract. [Open the `asr_openai_whisper` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/asr_openai_whisper.ipynb).

| Property | Value |
| --- | --- |
| Task | Automatic speech recognition |
| Architecture | `whisper` |
| Runtime | `VoiceHub-native` |
| Languages | `en`, `zh`, `de`, `es`, `ru`, `ko`, `fr`, `ja`, `pt`, `tr`, `pl`, `ca`, `nl`, `ar`, `sv`, `it`, `id`, `hi`, `fi`, `vi`, `he`, `uk`, `el`, `ms`, `cs`, `ro`, `da`, `hu`, `ta`, `no`, `th`, `ur`, `hr`, `bg`, `lt`, `la`, `mi`, `ml`, `cy`, `sk`, `te`, `fa`, `lv`, `bn`, `sr`, `az`, `sl`, `kn`, `et`, `mk`, `br`, `eu`, `is`, `hy`, `ne`, `mn`, `bs`, `kk`, `sq`, `sw`, `gl`, `mr`, `pa`, `si`, `km`, `sn`, `yo`, `so`, `af`, `oc`, `ka`, `be`, `tg`, `sd`, `gu`, `am`, `yi`, `lo`, `uz`, `fo`, `ht`, `ps`, `tk`, `nn`, `mt`, `sa`, `lb`, `my`, `bo`, `tl`, `mg`, `as`, `tt`, `haw`, `ln`, `ha`, `ba`, `jw`, `su` |
| Capabilities | `automatic-speech-recognition`, `multilingual`, `translation`, `timestamps`, `safetensors`, `fine-tuning`, `voicehub-native` |
| Reusable components | — |
| Normalized output | `ASROutput` |

### Language support

<details class="vh-language-support" markdown>
<summary>Supported language abbreviations</summary>

`en`, `zh`, `de`, `es`, `ru`, `ko`, `fr`, `ja`, `pt`, `tr`, `pl`, `ca`, `nl`, `ar`, `sv`, `it`, `id`, `hi`, `fi`, `vi`, `he`, `uk`, `el`, `ms`, `cs`, `ro`, `da`, `hu`, `ta`, `no`, `th`, `ur`, `hr`, `bg`, `lt`, `la`, `mi`, `ml`, `cy`, `sk`, `te`, `fa`, `lv`, `bn`, `sr`, `az`, `sl`, `kn`, `et`, `mk`, `br`, `eu`, `is`, `hy`, `ne`, `mn`, `bs`, `kk`, `sq`, `sw`, `gl`, `mr`, `pa`, `si`, `km`, `sn`, `yo`, `so`, `af`, `oc`, `ka`, `be`, `tg`, `sd`, `gu`, `am`, `yi`, `lo`, `uz`, `fo`, `ht`, `ps`, `tk`, `nn`, `mt`, `sa`, `lb`, `my`, `bo`, `tl`, `mg`, `as`, `tt`, `haw`, `ln`, `ha`, `ba`, `jw`, `su`

</details>

## Paper and GitHub

- **Paper:** [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356)
- **Upstream GitHub:** [Whisper](https://github.com/openai/whisper)
- **VoiceHub source:** [VoiceHub model implementation](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_native/openai_whisper.py)

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('asr_openai_whisper')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `asr_openai_whisper` |
| Configuration class | `OpenAIWhisperConfig` |
| Architecture class | `OpenAIWhisperForSpeechRecognition` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'openai/whisper-small',
    model_type='asr_openai_whisper',
)
print(type(processor).__name__)
```

## Inference

The Usage example returns `ASROutput` through `AutoModelForSpeechRecognition`.

### Input and output contract

| Property | Value |
| --- | --- |
| Readiness | `integrated-raw` |
| Data architecture | `speech-sequence-to-sequence` |
| Sample rate | 16,000 Hz |
| Contract getter | `get_asr_dataset_spec('asr_openai_whisper')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `raw-audio` | `audio` | text / transcription / transcript | Source | at most one: text / transcription / transcript |
| `whisper-model-ready` | `input_features`, `labels` | — | Prepared | — |

OpenAI Whisper-compatible records trained through the native Whisper graph. See the [data workflow](../../guides/speech-data.md).

## Training and optimization

Use `available_optimization_passes()` to discover reversible public passes.
Unsupported runtime or hardware fails closed before mutation.

### Training contract

| Property | Value |
| --- | --- |
| Support | `native` |
| Family | `speech-sequence-to-sequence` |
| Recipe | `single-phase` |
| Default phase | `speech_recognition` |
| Training checkpoint | `openai/whisper-small` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `speech_recognition` | objective | `model` | — | `loss` |

The integration accepts its declared source or prepared contract directly. Call `model.validate_training_support()` first, then follow the
[training workflow](../../guides/training.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | [`openai/whisper-small`](https://huggingface.co/openai/whisper-small) |
| Hugging Face ID | [`openai/whisper-small`](https://huggingface.co/openai/whisper-small)<br>Repository availability verified through the Hugging Face model API on 2026-08-11; pin a revision before production use. |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.asr_native.openai_whisper.OpenAIWhisperForSpeechRecognition` |
| Configuration | `voicehub.models.asr_native.configuration.OpenAIWhisperConfig` |
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

### `OpenAIWhisperConfig`

[View `OpenAIWhisperConfig` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_native/configuration.py)

```text
OpenAIWhisperConfig(**config_kwargs)
```

### `OpenAIWhisperForSpeechRecognition`

[View `OpenAIWhisperForSpeechRecognition` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_native/openai_whisper.py)

```text
AutoModelForSpeechRecognition.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='asr_openai_whisper',
    config=None,
    **model_kwargs,
)
```

```python
from voicehub import get_model_spec

spec = get_model_spec('asr_openai_whisper')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('asr_openai_whisper')` |
| Load and run | `AutoModelForSpeechRecognition` |
| Configure | `OpenAIWhisperConfig` |
| Process | `AutoProcessor` |
| Model implementation | `OpenAIWhisperForSpeechRecognition` |
| Normalized output | `ASROutput` |
| Training contract | `get_training_spec('asr_openai_whisper')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).
