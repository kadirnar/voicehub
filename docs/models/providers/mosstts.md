---
description: Public API, checkpoint, training, and optimization guide for the mosstts integration.
---

# MossTTS {.vh-model-title}

## Usage

```bash
python -m pip install "voicehub @ git+https://github.com/kadirnar/voicehub.git@main"
```

Install from source, then choose a compatible checkpoint. Set the text and generation options, then inspect the returned audio.

```python
from pathlib import Path

from voicehub import AutoModelForTextToSpeech, TTSGenerationConfig

model = AutoModelForTextToSpeech.from_pretrained(
    'OpenMOSS-Team/MOSS-TTS-v1.5',
    model_type='mosstts',
    device="cuda",
    lazy_load=True,
)
generation_kwargs = {}
output = model.generate(
    "VoiceHub keeps model integrations consistent and easy to extend.",
    generation_config=TTSGenerationConfig(
        seed=42,
        output_file=Path("output.wav"),
    ),
    **generation_kwargs,
)
print(output.file_path, output.sample_rate)
```

Use authorized recordings. Verify hardware needs and pin a revision in production.

## Overview

`mosstts` is a VoiceHub **text to speech**
integration. This page is generated from its registry contract. [Open the `mosstts` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/mosstts.ipynb).

| Property | Value |
| --- | --- |
| Task | Text to speech |
| Architecture | `moss-tts` |
| Runtime | `VoiceHub-native` |
| Languages | Checkpoint-defined; not exhaustively enumerated |
| Capabilities | `text-to-speech`, `voice-cloning`, `multilingual`, `fine-tuning`, `safetensors`, `voicehub-native`, `native-runtime`, `delay-variant`, `local-variant`, `local-v1.5-variant`, `realtime-variant`, `raw-audio-fine-tuning`, `preencoded-rvq-fine-tuning`, `native-codec-v1`, `native-codec-v2`, `buffered-generation` |
| Reusable components | — |
| Normalized output | `TTSOutput` |

### Language support

VoiceHub does not claim one exhaustive language list across compatible checkpoints; verify the selected checkpoint card and processor metadata.

## Paper and GitHub

- **Paper:** No dedicated upstream research paper is published for this integration.
- **Upstream GitHub:** [MOSS-TTS](https://github.com/OpenMOSS/MOSS-TTS)
- **VoiceHub source:** [VoiceHub model implementation](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/mosstts/modeling_mosstts.py)

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('mosstts')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `mosstts` |
| Configuration class | `MossTTSConfig` |
| Architecture class | `MossTTSForTextToSpeech` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'OpenMOSS-Team/MOSS-TTS-v1.5',
    model_type='mosstts',
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
| Sample rate | Model/checkpoint specific |
| Contract getter | `get_tts_dataset_spec('mosstts')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `raw-audio` | `text` | audio / waveform / audio_path | Source | at most one: audio / waveform / audio_path; forbidden: speech_tokens |
| `preencoded-rvq` | `text`, `speech_tokens` | — | Prepared | forbidden: audio, waveform, audio_path |

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
| Default phase | `semantic_language_model` |
| Training checkpoint | `OpenMOSS-Team/MOSS-TTS-v1.5` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `semantic_language_model` | objective | `model` | `input_ids`, `attention_mask`, `labels` | `loss` |

The integration accepts its declared source or prepared contract directly. Call `model.validate_training_support()` first, then follow the
[training workflow](../../guides/training.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | [`OpenMOSS-Team/MOSS-TTS-v1.5`](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-v1.5) |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.mosstts.modeling_mosstts.MossTTSForTextToSpeech` |
| Configuration | `voicehub.models.mosstts.configuration_mosstts.MossTTSConfig` |
| Source provenance | `voicehub/models/mosstts/source/SOURCE.json` |
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

### `MossTTSConfig`

[View `MossTTSConfig` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/mosstts/configuration_mosstts.py)

```text
MossTTSConfig(**config_kwargs)
```

### `MossTTSForTextToSpeech`

[View `MossTTSForTextToSpeech` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/mosstts/modeling_mosstts.py)

```text
AutoModelForTextToSpeech.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='mosstts',
    config=None,
    **model_kwargs,
)
```

```python
from voicehub import get_model_spec

spec = get_model_spec('mosstts')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('mosstts')` |
| Load and run | `AutoModelForTextToSpeech` |
| Configure | `MossTTSConfig` |
| Process | `AutoProcessor` |
| Model implementation | `MossTTSForTextToSpeech` |
| Normalized output | `TTSOutput` |
| Training contract | `get_training_spec('mosstts')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).
