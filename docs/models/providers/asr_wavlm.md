---
description: Public API, checkpoint, training, and optimization guide for the asr_wavlm integration.
---

# WavLM {.vh-model-title}

## Usage

```bash
python -m pip install "voicehub @ git+https://github.com/kadirnar/voicehub.git@main"
```

Install from source, then choose a compatible checkpoint. Place a supported recording at `speech.wav` and inspect the transcript.

```python
from voicehub import AutoModelForSpeechRecognition

model = AutoModelForSpeechRecognition.from_pretrained(
    'patrickvonplaten/wavlm-libri-clean-100h-base-plus',
    model_type='asr_wavlm',
    device="cuda",
    lazy_load=True,
)
output = model.transcribe("speech.wav")
print(output.text)
for segment in output.segments:
    print(segment.start, segment.end, segment.text)
```

Use authorized recordings. Verify hardware needs and pin a revision in production.

## Overview

`asr_wavlm` is a VoiceHub **automatic speech recognition**
integration. This page is generated from its registry contract. [Open the `asr_wavlm` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/asr_wavlm.ipynb).

| Property | Value |
| --- | --- |
| Task | Automatic speech recognition |
| Architecture | `wavlm` |
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

- **Paper:** [WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing](https://arxiv.org/abs/2110.13900)
- **Upstream GitHub:** [UniLM / WavLM](https://github.com/microsoft/unilm)
- **VoiceHub source:** [VoiceHub model implementation](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_wavlm/modeling_asr_wavlm.py)

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('asr_wavlm')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `asr_wavlm` |
| Configuration class | `WavLMASRConfig` |
| Architecture class | `WavLMForSpeechRecognition` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'patrickvonplaten/wavlm-libri-clean-100h-base-plus',
    model_type='asr_wavlm',
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
| Contract getter | `get_asr_dataset_spec('asr_wavlm')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `raw-audio` | `audio` | text / transcription / transcript | Source | at most one: text / transcription / transcript |
| `waveform-ctc` | `input_values`, `labels` | — | Prepared | — |

WavLM waveform and CTC transcript records. See the [data workflow](../../guides/speech-data.md).

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
| Training checkpoint | `patrickvonplaten/wavlm-libri-clean-100h-base-plus` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `speech_recognition` | objective | `model` | — | `loss` |

The integration accepts its declared source or prepared contract directly. Call `model.validate_training_support()` first, then follow the
[training workflow](../../guides/training.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | [`patrickvonplaten/wavlm-libri-clean-100h-base-plus`](https://huggingface.co/patrickvonplaten/wavlm-libri-clean-100h-base-plus) |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.asr_wavlm.modeling_asr_wavlm.WavLMForSpeechRecognition` |
| Configuration | `voicehub.models.asr_wavlm.configuration_asr_wavlm.WavLMASRConfig` |
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

### `WavLMASRConfig`

[View `WavLMASRConfig` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_wavlm/configuration_asr_wavlm.py)

```text
WavLMASRConfig(**config_kwargs)
```

### `WavLMForSpeechRecognition`

[View `WavLMForSpeechRecognition` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_wavlm/modeling_asr_wavlm.py)

```text
AutoModelForSpeechRecognition.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='asr_wavlm',
    config=None,
    **model_kwargs,
)
```

```python
from voicehub import get_model_spec

spec = get_model_spec('asr_wavlm')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('asr_wavlm')` |
| Load and run | `AutoModelForSpeechRecognition` |
| Configure | `WavLMASRConfig` |
| Process | `AutoProcessor` |
| Model implementation | `WavLMForSpeechRecognition` |
| Normalized output | `ASROutput` |
| Training contract | `get_training_spec('asr_wavlm')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).
