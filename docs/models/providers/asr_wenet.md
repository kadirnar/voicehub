---
description: Public API, checkpoint, training, and optimization guide for the asr_wenet integration.
---

# WeNetASR {.vh-model-title}

## Usage

```bash
python -m pip install "voicehub @ git+https://github.com/kadirnar/voicehub.git@main"
```

Install from source, then choose a compatible checkpoint. Place a supported recording at `speech.wav` and inspect the transcript.

Checkpoint note: The registry identifier is not a Hugging Face repository and the original upstream archive endpoints are unavailable. VoiceHub verifies an immutable mirror against the published 503,845,602-byte archive's SHA-256. Convert that trust-gated pickle archive first, then replace the path below with the resulting VoiceHub-native directory containing model.safetensors, config.json, tokenizer.model, and units.txt.

```python
from voicehub import AutoModelForSpeechRecognition

model = AutoModelForSpeechRecognition.from_pretrained(
    'path/to/converted-wenet-u2pp',
    model_type='asr_wenet',
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

`asr_wenet` is a VoiceHub **automatic speech recognition**
integration. This page is generated from its registry contract.

| Property | Value |
| --- | --- |
| Task | Automatic speech recognition |
| Architecture | `wenet-asr` |
| Runtime | `VoiceHub-native` |
| Languages | `en` |
| Capabilities | `automatic-speech-recognition`, `english`, `timestamps`, `safetensors`, `fine-tuning`, `voicehub-native`, `ctc`, `attention-rescoring` |
| Reusable components | — |
| Normalized output | `ASROutput` |

### Language support

<details class="vh-language-support" markdown>
<summary>1 documented language</summary>

`en`

</details>

## Paper and GitHub

- **Paper:** [WeNet: Production Oriented Streaming and Non-Streaming End-to-End Speech Recognition Toolkit](https://arxiv.org/abs/2102.01547)
- **Upstream GitHub:** [WeNet](https://github.com/wenet-e2e/wenet)
- **VoiceHub source:** [VoiceHub model implementation](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_wenet/__init__.py)

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('asr_wenet')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `asr_wenet` |
| Configuration class | `WeNetASRConfig` |
| Architecture class | `WeNetASRForSpeechRecognition` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'path/to/converted-wenet-u2pp',
    model_type='asr_wenet',
)
print(type(processor).__name__)
```

## Inference

The Usage example returns `ASROutput` through `AutoModelForSpeechRecognition`.

### Input and output contract

| Property | Value |
| --- | --- |
| Readiness | `integrated-raw` |
| Data architecture | `hybrid-ctc-attention` |
| Sample rate | 16,000 Hz |
| Contract getter | `get_asr_dataset_spec('asr_wenet')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `raw-audio` | `audio` | text / transcription / transcript | Source | at most one: text / transcription / transcript |
| `wenet-waveform-model-ready` | `input_signal`, `input_signal_length`, `labels`, `label_lengths` | — | Prepared | — |
| `wenet-feature-model-ready` | `features`, `feature_lengths`, `labels`, `label_lengths` | — | Prepared | — |

WeNet U2++ joint CTC/attention fine-tuning records. See the [data workflow](../../guides/speech-data.md).

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
| Training checkpoint | `wenet/gigaspeech-u2pp-conformer` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `speech_recognition` | objective | `model` | `labels`, `label_lengths` | `loss`, `attention_loss`, `ctc_loss` |

The integration accepts its declared source or prepared contract directly. Call `model.validate_training_support()` first, then follow the
[training workflow](../../guides/training.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | [`wenet/gigaspeech-u2pp-conformer`](https://github.com/wenet-e2e/wenet/blob/a50d4208f13bbf3a0746e606ac29176cd2e87e6b/examples/gigaspeech/s0/README.md#conformer-u2-result) |
| Checkpoint status | Original upstream archive unavailable (HTTP 404 and TLS failures verified 2026-08-04); exact bytes are available from the immutable openspeech/wenet-models mirror at 90acd57d17169a15d5ceab462c6e7db3bd003921 |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.asr_wenet.WeNetASRForSpeechRecognition` |
| Configuration | `voicehub.models.asr_wenet.WeNetASRConfig` |
| Source provenance | `voicehub/architectures/wenet_u2pp/SOURCE.json` |
| License | [NOT DECLARED](https://github.com/wenet-e2e/wenet/blob/a50d4208f13bbf3a0746e606ac29176cd2e87e6b/examples/gigaspeech/s0/README.md#conformer-u2-result) |

The published GigaSpeech checkpoint archive does not declare a checkpoint license. The VoiceHub-owned architecture port is Apache-2.0, but that source license is not assumed for the weights. Commercial use: **review required**.

Confirm the checkpoint revision, access terms, provenance, and license.

### Limitations

- The registry identifier is not a Hugging Face repository and the original upstream archive endpoints are unavailable. VoiceHub verifies an immutable mirror against the published 503,845,602-byte archive's SHA-256. Convert that trust-gated pickle archive first, then replace the path below with the resulting VoiceHub-native directory containing model.safetensors, config.json, tokenizer.model, and units.txt.
- Validate memory, precision, and optional dependencies on the target system.
- Public optimizations fail closed when the runtime or hardware cannot satisfy
  their validation contract; an unavailable pass is not reported as applied.
- Contract tests do not replace the linked released-checkpoint evidence.

## Public API

Use the stable configuration, processor, and task-model facades below.

### `WeNetASRConfig`

[View `WeNetASRConfig` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_wenet/__init__.py)

```text
WeNetASRConfig(**config_kwargs)
```

### `WeNetASRForSpeechRecognition`

[View `WeNetASRForSpeechRecognition` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_wenet/__init__.py)

```text
AutoModelForSpeechRecognition.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='asr_wenet',
    config=None,
    **model_kwargs,
)
```

```python
from voicehub import get_model_spec

spec = get_model_spec('asr_wenet')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('asr_wenet')` |
| Load and run | `AutoModelForSpeechRecognition` |
| Configure | `WeNetASRConfig` |
| Process | `AutoProcessor` |
| Model implementation | `WeNetASRForSpeechRecognition` |
| Normalized output | `ASROutput` |
| Training contract | `get_training_spec('asr_wenet')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).
