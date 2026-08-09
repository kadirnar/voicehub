---
description: Public API, checkpoint, training, and optimization guide for the asr_vibevoice integration.
---

# VibeVoice {.vh-model-title}

## Usage

```bash
python -m pip install "voicehub @ git+https://github.com/kadirnar/voicehub.git@main"
```

Install from source, then choose a compatible checkpoint. Place a supported recording at `speech.wav` and inspect the transcript.

```python
from voicehub import AutoModelForSpeechRecognition

model = AutoModelForSpeechRecognition.from_pretrained(
    'microsoft/VibeVoice-ASR-HF',
    model_type='asr_vibevoice',
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

`asr_vibevoice` is a VoiceHub **automatic speech recognition**
integration. This page is generated from its registry contract. [Open the `asr_vibevoice` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/asr_vibevoice.ipynb).

| Property | Value |
| --- | --- |
| Task | Automatic speech recognition |
| Architecture | `vibevoice-asr` |
| Runtime | `VoiceHub-native` |
| Languages | Checkpoint-defined; not exhaustively enumerated |
| Capabilities | `automatic-speech-recognition`, `multilingual`, `speaker-attribution`, `timestamps`, `hotwords`, `long-form`, `safetensors`, `fine-tuning`, `voicehub-native`, `native-runtime` |
| Reusable components | — |
| Normalized output | `ASROutput` |

### Language support

VoiceHub does not claim one exhaustive language list across compatible checkpoints; verify the selected checkpoint card and processor metadata.

## Paper and GitHub

- **Paper:** No dedicated upstream research paper is published for this integration.
- **Upstream GitHub:** [VibeVoice](https://github.com/microsoft/VibeVoice)
- **VoiceHub source:** [VoiceHub model implementation](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_vibevoice/modeling_asr_vibevoice.py)

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('asr_vibevoice')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `asr_vibevoice` |
| Configuration class | `VibeVoiceASRConfig` |
| Architecture class | `VibeVoiceForSpeechRecognition` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'microsoft/VibeVoice-ASR-HF',
    model_type='asr_vibevoice',
)
print(type(processor).__name__)
```

## Inference

The Usage example returns `ASROutput` through `AutoModelForSpeechRecognition`.

### Input and output contract

| Property | Value |
| --- | --- |
| Readiness | `integrated-raw` |
| Data architecture | `prompted-multimodal` |
| Sample rate | 24,000 Hz |
| Contract getter | `get_asr_dataset_spec('asr_vibevoice')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `segmented-audio` | `audio`, `segments` | — | Source | forbidden: text, transcription, transcript |
| `serialized-audio` | `audio` | text / transcription / transcript | Source | at most one: text / transcription / transcript; forbidden: segments |
| `vibevoice-model-ready` | `input_ids`, `attention_mask`, `input_values`, `padding_mask`, `labels` | — | Prepared | — |

VibeVoice structured long-form ASR targets and multimodal prompt inputs. See the [data workflow](../../guides/speech-data.md).

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
| Training checkpoint | `microsoft/VibeVoice-ASR-HF` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `speech_recognition` | objective | `model.model.multi_modal_projector`, `model.model.language_model`, `model.lm_head` | `input_ids`, `attention_mask`, `input_values`, `padding_mask`, `labels` | `loss` |

The integration accepts its declared source or prepared contract directly. Call `model.validate_training_support()` first, then follow the
[training workflow](../../guides/training.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | [`microsoft/VibeVoice-ASR-HF`](https://huggingface.co/microsoft/VibeVoice-ASR-HF) |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.asr_vibevoice.modeling_asr_vibevoice.VibeVoiceForSpeechRecognition` |
| Configuration | `voicehub.models.asr_vibevoice.configuration_asr_vibevoice.VibeVoiceASRConfig` |
| Source provenance | `voicehub/architectures/vibevoice/source/SOURCE.json` |
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

### `VibeVoiceASRConfig`

[View `VibeVoiceASRConfig` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_vibevoice/configuration_asr_vibevoice.py)

```text
VibeVoiceASRConfig(**config_kwargs)
```

### `VibeVoiceForSpeechRecognition`

[View `VibeVoiceForSpeechRecognition` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_vibevoice/modeling_asr_vibevoice.py)

```text
AutoModelForSpeechRecognition.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='asr_vibevoice',
    config=None,
    **model_kwargs,
)
```

```python
from voicehub import get_model_spec

spec = get_model_spec('asr_vibevoice')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('asr_vibevoice')` |
| Load and run | `AutoModelForSpeechRecognition` |
| Configure | `VibeVoiceASRConfig` |
| Process | `AutoProcessor` |
| Model implementation | `VibeVoiceForSpeechRecognition` |
| Normalized output | `ASROutput` |
| Training contract | `get_training_spec('asr_vibevoice')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).
