---
description: Public API, checkpoint, training, and optimization guide for the asr_parakeet_tdt integration.
---

# ParakeetTDT {.vh-model-title}

## Usage

```bash
python -m pip install "voicehub @ git+https://github.com/kadirnar/voicehub.git@main"
```

Install from source, then choose a compatible checkpoint. Place a supported recording at `speech.wav` and inspect the transcript.

```python
from voicehub import AutoModelForSpeechRecognition

model = AutoModelForSpeechRecognition.from_pretrained(
    'nvidia/parakeet-tdt-0.6b-v3',
    model_type='asr_parakeet_tdt',
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

`asr_parakeet_tdt` is a VoiceHub **automatic speech recognition**
integration. This page is generated from its registry contract. [Open the `asr_parakeet_tdt` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/asr_parakeet_tdt.ipynb).

| Property | Value |
| --- | --- |
| Task | Automatic speech recognition |
| Architecture | `parakeet-tdt` |
| Runtime | `VoiceHub-native` |
| Languages | Checkpoint-defined; not exhaustively enumerated |
| Capabilities | `automatic-speech-recognition`, `multilingual`, `timestamps`, `long-form`, `safetensors`, `fine-tuning`, `voicehub-native`, `native-runtime` |
| Reusable components | — |
| Normalized output | `ASROutput` |

### Language support

VoiceHub does not claim one exhaustive language list across compatible checkpoints; verify the selected checkpoint card and processor metadata.

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('asr_parakeet_tdt')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `asr_parakeet_tdt` |
| Configuration class | `ParakeetTDTASRConfig` |
| Architecture class | `ParakeetTDTForSpeechRecognition` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'nvidia/parakeet-tdt-0.6b-v3',
    model_type='asr_parakeet_tdt',
)
print(type(processor).__name__)
```

## Inference

The Usage example returns `ASROutput` through `AutoModelForSpeechRecognition`.

### Input and output contract

| Property | Value |
| --- | --- |
| Readiness | `integrated-raw` |
| Data architecture | `tdt` |
| Sample rate | 16,000 Hz |
| Contract getter | `get_asr_dataset_spec('asr_parakeet_tdt')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `raw-audio` | `audio` | text / transcription / transcript | Source | at most one: text / transcription / transcript |
| `parakeet-tdt-model-ready` | `input_features`, `attention_mask`, `labels`, `decoder_input_ids` | — | Prepared | — |

Parakeet token-duration transducer audio and transcript records. See the [data workflow](../../guides/speech-data.md).

## Training and optimization

Use `available_optimization_passes()` to discover reversible public passes.
Unsupported runtime or hardware fails closed before mutation.

### Training contract

| Property | Value |
| --- | --- |
| Support | `native` |
| Family | `tdt` |
| Recipe | `single-phase` |
| Default phase | `speech_recognition` |
| Training checkpoint | `nvidia/parakeet-tdt-0.6b-v3` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `speech_recognition` | objective | `model.encoder`, `model.encoder_projector`, `model.decoder`, `model.joint` | `input_features`, `attention_mask`, `labels`, `decoder_input_ids` | `loss` |

The integration accepts its declared source or prepared contract directly. Call `model.validate_training_support()` first, then follow the
[training workflow](../../guides/training.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.asr_parakeet_tdt.modeling_asr_parakeet_tdt.ParakeetTDTForSpeechRecognition` |
| Configuration | `voicehub.models.asr_parakeet_tdt.configuration_asr_parakeet_tdt.ParakeetTDTASRConfig` |
| Source provenance | `voicehub/architectures/parakeet_tdt/SOURCE.json` |
| License | [CC-BY-4.0](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) |

The pinned Parakeet TDT checkpoint and derivatives require CC-BY-4.0 attribution. The VoiceHub-owned architecture port is audited against Apache-2.0 Transformers and NeMo source. Commercial use: **allowed by the registered terms**.

Confirm the checkpoint revision, access terms, provenance, and license.

### Limitations

- No integration-specific checkpoint limitation is registered. Verify the selected checkpoint revision and its documented runtime requirements.
- Validate memory, precision, and optional dependencies on the target system.
- Public optimizations fail closed when the runtime or hardware cannot satisfy
  their validation contract; an unavailable pass is not reported as applied.
- Contract tests do not replace the linked released-checkpoint evidence.

## Public API

Use the stable configuration, processor, and task-model facades below.

### `ParakeetTDTASRConfig`

[View `ParakeetTDTASRConfig` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_parakeet_tdt/configuration_asr_parakeet_tdt.py)

```text
ParakeetTDTASRConfig(**config_kwargs)
```

### `ParakeetTDTForSpeechRecognition`

[View `ParakeetTDTForSpeechRecognition` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_parakeet_tdt/modeling_asr_parakeet_tdt.py)

```text
AutoModelForSpeechRecognition.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='asr_parakeet_tdt',
    config=None,
    **model_kwargs,
)
```

```python
from voicehub import get_model_spec

spec = get_model_spec('asr_parakeet_tdt')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('asr_parakeet_tdt')` |
| Load and run | `AutoModelForSpeechRecognition` |
| Configure | `ParakeetTDTASRConfig` |
| Process | `AutoProcessor` |
| Model implementation | `ParakeetTDTForSpeechRecognition` |
| Normalized output | `ASROutput` |
| Training contract | `get_training_spec('asr_parakeet_tdt')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).
