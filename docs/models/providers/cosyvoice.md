---
description: Public API, checkpoint, training, and optimization guide for the cosyvoice integration.
---

# CosyVoice {.vh-model-title}

## Usage

```bash
python -m pip install "voicehub @ git+https://github.com/kadirnar/voicehub.git@main"
```

Install from source, then choose a compatible checkpoint. Provide an authorized `reference.wav` and its exact transcript when requested.

```python
from pathlib import Path

REFERENCE_AUDIO = Path("reference.wav")
REFERENCE_TEXT = "This transcript must exactly match the authorized reference audio."

from voicehub import AutoModelForTextToSpeech, TTSGenerationConfig

model = AutoModelForTextToSpeech.from_pretrained(
    'FunAudioLLM/Fun-CosyVoice3-0.5B-2512',
    model_type='cosyvoice',
    device="cuda",
    lazy_load=True,
)
generation_kwargs = {
    "speaker_embedding": None,
    "speaker_audio_path": str(REFERENCE_AUDIO),
}
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

`cosyvoice` is a VoiceHub **text to speech**
integration. This page is generated from its registry contract. [Open the `cosyvoice` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/cosyvoice.ipynb).

| Property | Value |
| --- | --- |
| Task | Text to speech |
| Architecture | `cosyvoice-native` |
| Runtime | `VoiceHub-native` |
| Languages | 9 enumerated languages |
| Capabilities | `text-to-speech`, `voice-cloning`, `multilingual`, `fine-tuning`, `flow-matching`, `adversarial-vocoder-training`, `safetensors`, `voicehub-native`, `native-runtime`, `precomputed-speaker-embedding`, `preencoded-speech-token-fine-tuning` |
| Reusable components | `conformer` |
| Normalized output | `TTSOutput` |

### Language support

<details class="vh-language-support" markdown>
<summary>9 documented languages</summary>

`zh`, `en`, `ja`, `ko`, `de`, `es`, `fr`, `it`, `ru`

The registered family also documents 18 Chinese dialect variants.

</details>

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('cosyvoice')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `cosyvoice` |
| Configuration class | `CosyVoiceConfig` |
| Architecture class | `CosyVoiceForTextToSpeech` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'FunAudioLLM/Fun-CosyVoice3-0.5B-2512',
    model_type='cosyvoice',
)
print(type(processor).__name__)
```

## Inference

The Usage example returns `TTSOutput` through `AutoModelForTextToSpeech`.

### Input and output contract

| Property | Value |
| --- | --- |
| Readiness | `integrated-raw` |
| Data architecture | `hybrid` |
| Sample rate | 24,000 Hz |
| Contract getter | `get_tts_dataset_spec('cosyvoice')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `llm-raw-audio` | `text` | speech_audio / audio / waveform / audio_path | Source | at most one: speech_audio / audio / waveform / audio_path; forbidden: speech_tokens; speech_audio requires one of speech_sampling_rate, sampling_rate, sample_rate; audio requires one of speech_sampling_rate, sampling_rate, sample_rate; waveform requires one of speech_sampling_rate, sampling_rate, sample_rate |
| `llm-record` | `text`, `speech_tokens` | — | Prepared | forbidden: speech_audio, audio, waveform, audio_path |

Multi-component language-model, diffusion, acoustic, or GAN data. See the [data workflow](../../guides/data-preparation.md).

## Training and optimization

Use `available_optimization_passes()` to discover reversible public passes.
Unsupported runtime or hardware fails closed before mutation.

### Training contract

| Property | Value |
| --- | --- |
| Support | `custom` |
| Family | `composite` |
| Recipe | `adversarial` |
| Default phase | `llm` |
| Training checkpoint | `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `llm` | objective | `model.llm` | — | `language_model_loss` |
| `flow` | objective | `model.flow` | — | `flow_matching_loss` |
| `hifigan_generator` | generator | `model.hift` | — | `adversarial_loss`, `feature_matching_loss`, `pitch_loss`, `spectral_reconstruction_loss` |
| `hifigan_discriminator` | discriminator | `model.hifigan.discriminator` | — | `discriminator_loss` |

This profile uses model-specific phases; inspect and honor each phase boundary. Call `model.validate_training_support()` first, then follow the
[training workflow](../../guides/training.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | [`FunAudioLLM/Fun-CosyVoice3-0.5B-2512`](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.cosyvoice.modeling_cosyvoice.CosyVoiceForTextToSpeech` |
| Configuration | `voicehub.models.cosyvoice.configuration_cosyvoice.CosyVoiceConfig` |
| Source provenance | `voicehub/models/cosyvoice/source/SOURCE.json` |
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

### `CosyVoiceConfig`

[View `CosyVoiceConfig` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/cosyvoice/configuration_cosyvoice.py)

```text
CosyVoiceConfig(**config_kwargs)
```

### `CosyVoiceForTextToSpeech`

[View `CosyVoiceForTextToSpeech` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/cosyvoice/modeling_cosyvoice.py)

```text
AutoModelForTextToSpeech.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='cosyvoice',
    config=None,
    **model_kwargs,
)
```

```python
from voicehub import get_model_spec

spec = get_model_spec('cosyvoice')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('cosyvoice')` |
| Load and run | `AutoModelForTextToSpeech` |
| Configure | `CosyVoiceConfig` |
| Process | `AutoProcessor` |
| Model implementation | `CosyVoiceForTextToSpeech` |
| Normalized output | `TTSOutput` |
| Training contract | `get_training_spec('cosyvoice')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).
