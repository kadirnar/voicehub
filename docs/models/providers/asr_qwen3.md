---
description: Public API, checkpoint, training, and optimization guide for the asr_qwen3 integration.
---

# Qwen3ASR {.vh-model-title}

## Usage

```bash
python -m pip install "voicehub @ git+https://github.com/kadirnar/voicehub.git@main"
```

Install from source, then choose a compatible checkpoint. Place a supported recording at `speech.wav` and inspect the transcript.

```python
from voicehub import AutoModelForSpeechRecognition

model = AutoModelForSpeechRecognition.from_pretrained(
    'Qwen/Qwen3-ASR-0.6B',
    model_type='asr_qwen3',
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

`asr_qwen3` is a VoiceHub **automatic speech recognition**
integration. This page is generated from its registry contract. [Open the `asr_qwen3` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/asr_qwen3.ipynb).

| Property | Value |
| --- | --- |
| Task | Automatic speech recognition |
| Architecture | `qwen3-asr` |
| Runtime | `VoiceHub-native` |
| Languages | `ar`, `yue`, `zh`, `cs`, `da`, `nl`, `en`, `fil`, `fi`, `fr`, `de`, `el`, `hi`, `hu`, `id`, `it`, `ja`, `ko`, `mk`, `ms`, `fa`, `pl`, `pt`, `ro`, `ru`, `es`, `sv`, `th`, `tr`, `vi` |
| Capabilities | `automatic-speech-recognition`, `multilingual`, `language-identification`, `hotwords`, `long-form`, `safetensors`, `fine-tuning`, `lora`, `voicehub-native`, `native-runtime` |
| Reusable components | — |
| Normalized output | `ASROutput` |

### Language support

<details class="vh-language-support" markdown>
<summary>Supported language abbreviations</summary>

`ar`, `yue`, `zh`, `cs`, `da`, `nl`, `en`, `fil`, `fi`, `fr`, `de`, `el`, `hi`, `hu`, `id`, `it`, `ja`, `ko`, `mk`, `ms`, `fa`, `pl`, `pt`, `ro`, `ru`, `es`, `sv`, `th`, `tr`, `vi`

The same checkpoint also names Anhui, Dongbei, Fujian, Gansu, Guizhou, Hebei, Henan, Hubei, Hunan, Jiangxi, Ningxia, Shandong, Shaanxi, Shanxi, Sichuan, Tianjin, Yunnan, Zhejiang, Cantonese (Hong Kong accent), Cantonese (Guangdong accent), Wu, and Minnan dialects.

</details>

## Paper and GitHub

- **Paper:** No dedicated upstream research paper is published for this integration.
- **Upstream GitHub:** [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)
- **VoiceHub source:** [VoiceHub model implementation](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_qwen3/modeling_asr_qwen3.py)

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('asr_qwen3')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `asr_qwen3` |
| Configuration class | `Qwen3ASRConfig` |
| Architecture class | `Qwen3ASRForSpeechRecognition` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'Qwen/Qwen3-ASR-0.6B',
    model_type='asr_qwen3',
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
| Sample rate | 16,000 Hz |
| Contract getter | `get_asr_dataset_spec('asr_qwen3')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `raw-audio` | `audio` | text / transcription / transcript | Source | at most one: text / transcription / transcript |
| `qwen3-model-ready` | `input_ids`, `attention_mask`, `input_features`, `feature_attention_mask`, `labels` | — | Prepared | — |

Qwen3-ASR completion-only multimodal fine-tuning records. See the [data workflow](../../guides/speech-data.md).

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
| Training checkpoint | `Qwen/Qwen3-ASR-0.6B` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `speech_recognition` | objective | `model` | — | `loss` |

The integration accepts its declared source or prepared contract directly. Call `model.validate_training_support()` first, then follow the
[training workflow](../../guides/training.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | [`Qwen/Qwen3-ASR-0.6B`](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.asr_qwen3.modeling_asr_qwen3.Qwen3ASRForSpeechRecognition` |
| Configuration | `voicehub.models.asr_qwen3.configuration_asr_qwen3.Qwen3ASRConfig` |
| Source provenance | `voicehub/architectures/qwen3_asr/SOURCE.json` |
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

### `Qwen3ASRConfig`

[View `Qwen3ASRConfig` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_qwen3/configuration_asr_qwen3.py)

```text
Qwen3ASRConfig(**config_kwargs)
```

### `Qwen3ASRForSpeechRecognition`

[View `Qwen3ASRForSpeechRecognition` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_qwen3/modeling_asr_qwen3.py)

```text
AutoModelForSpeechRecognition.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='asr_qwen3',
    config=None,
    **model_kwargs,
)
```

```python
from voicehub import get_model_spec

spec = get_model_spec('asr_qwen3')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('asr_qwen3')` |
| Load and run | `AutoModelForSpeechRecognition` |
| Configure | `Qwen3ASRConfig` |
| Process | `AutoProcessor` |
| Model implementation | `Qwen3ASRForSpeechRecognition` |
| Normalized output | `ASROutput` |
| Training contract | `get_training_spec('asr_qwen3')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).
