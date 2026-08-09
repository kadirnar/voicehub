---
description: Public API, checkpoint, training, and optimization guide for the asr_speechbrain integration.
---

# SpeechBrainASR {.vh-model-title}

## Usage

```bash
python -m pip install "voicehub @ git+https://github.com/kadirnar/voicehub.git@main"
```

Install from source, then choose a compatible checkpoint. Place a supported recording at `speech.wav` and inspect the transcript.

```python
from voicehub import AutoModelForSpeechRecognition

model = AutoModelForSpeechRecognition.from_pretrained(
    'speechbrain/asr-crdnn-rnnlm-librispeech',
    model_type='asr_speechbrain',
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

`asr_speechbrain` is a VoiceHub **automatic speech recognition**
integration. This page is generated from its registry contract. [Open the `asr_speechbrain` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/asr_speechbrain.ipynb).

| Property | Value |
| --- | --- |
| Task | Automatic speech recognition |
| Architecture | `speechbrain-crdnn-asr` |
| Runtime | `VoiceHub-native` |
| Languages | `en` |
| Capabilities | `automatic-speech-recognition`, `english`, `beam-search`, `safetensors`, `fine-tuning`, `voicehub-native`, `crdnn`, `ctc-seq2seq`, `rnnlm-shallow-fusion` |
| Reusable components | — |
| Normalized output | `ASROutput` |

### Language support

<details class="vh-language-support" markdown>
<summary>1 documented language</summary>

`en`

</details>

## Paper and GitHub

- **Paper:** [SpeechBrain: A General-Purpose Speech Toolkit](https://arxiv.org/abs/2106.04624)
- **Upstream GitHub:** [SpeechBrain](https://github.com/speechbrain/speechbrain)
- **VoiceHub source:** [VoiceHub model implementation](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_native/speechbrain.py)

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('asr_speechbrain')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `asr_speechbrain` |
| Configuration class | `SpeechBrainASRConfig` |
| Architecture class | `SpeechBrainASRForSpeechRecognition` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'speechbrain/asr-crdnn-rnnlm-librispeech',
    model_type='asr_speechbrain',
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
| Contract getter | `get_asr_dataset_spec('asr_speechbrain')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `raw-audio` | — | audio / audio_path; text / transcription / transcript | Source | at most one: audio / audio_path; text / transcription / transcript |
| `speechbrain-model-ready` | `waveforms`, `waveform_lengths`, `tokens_bos`, `tokens_eos`, `token_lengths`, `ctc_tokens`, `ctc_token_lengths` | — | Prepared | — |

SpeechBrain CRDNN joint CTC/attention fine-tuning records. See the [data workflow](../../guides/speech-data.md).

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
| Training checkpoint | `speechbrain/asr-crdnn-rnnlm-librispeech` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `speech_recognition` | objective | `model` | `waveforms`, `waveform_lengths`, `tokens_bos`, `tokens_eos`, `token_lengths`, `ctc_tokens`, `ctc_token_lengths` | `loss`, `seq2seq_loss`, `ctc_loss` |

The integration accepts its declared source or prepared contract directly. Call `model.validate_training_support()` first, then follow the
[training workflow](../../guides/training.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | [`speechbrain/asr-crdnn-rnnlm-librispeech`](https://huggingface.co/speechbrain/asr-crdnn-rnnlm-librispeech) |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.asr_native.speechbrain.SpeechBrainASRForSpeechRecognition` |
| Configuration | `voicehub.models.asr_native.configuration.SpeechBrainASRConfig` |
| Source provenance | `voicehub/architectures/speechbrain_asr/SOURCE.json` |
| License | [Apache-2.0](https://huggingface.co/speechbrain/asr-crdnn-rnnlm-librispeech) |

The pinned CRDNN, RNNLM, tokenizer, and source implementation are Apache-2.0. The original pickle files cross a strict one-time conversion boundary; steady-state artifacts are Safetensors. Commercial use: **allowed by the registered terms**.

Confirm the checkpoint revision, access terms, provenance, and license.

### Limitations

- No integration-specific checkpoint limitation is registered. Verify the selected checkpoint revision and its documented runtime requirements.
- Validate memory, precision, and optional dependencies on the target system.
- Public optimizations fail closed when the runtime or hardware cannot satisfy
  their validation contract; an unavailable pass is not reported as applied.
- Contract tests do not replace the linked released-checkpoint evidence.

## Public API

Use the stable configuration, processor, and task-model facades below.

### `SpeechBrainASRConfig`

[View `SpeechBrainASRConfig` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_native/configuration.py)

```text
SpeechBrainASRConfig(**config_kwargs)
```

### `SpeechBrainASRForSpeechRecognition`

[View `SpeechBrainASRForSpeechRecognition` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_native/speechbrain.py)

```text
AutoModelForSpeechRecognition.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='asr_speechbrain',
    config=None,
    **model_kwargs,
)
```

```python
from voicehub import get_model_spec

spec = get_model_spec('asr_speechbrain')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('asr_speechbrain')` |
| Load and run | `AutoModelForSpeechRecognition` |
| Configure | `SpeechBrainASRConfig` |
| Process | `AutoProcessor` |
| Model implementation | `SpeechBrainASRForSpeechRecognition` |
| Normalized output | `ASROutput` |
| Training contract | `get_training_spec('asr_speechbrain')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).
