---
description: Public API, checkpoint, training, and optimization guide for the fishtts integration.
---

# FishTTS {.vh-model-title}

## Usage

```bash
python -m pip install "voicehub @ git+https://github.com/kadirnar/voicehub.git@main"
```

Install from source, then choose a compatible checkpoint. Set the text and generation options, then inspect the returned audio.

```python
from pathlib import Path

from voicehub import AutoModelForTextToSpeech, TTSGenerationConfig

model = AutoModelForTextToSpeech.from_pretrained(
    'fishaudio/s2-pro',
    model_type='fishtts',
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

`fishtts` is a VoiceHub **text to speech**
integration. This page is generated from its registry contract. [Open the `fishtts` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/fishtts.ipynb).

| Property | Value |
| --- | --- |
| Task | Text to speech |
| Architecture | `fish-s2` |
| Runtime | `VoiceHub-native` |
| Languages | `zh`, `en`, `ja`, `ko`, `es`, `pt`, `ar`, `ru`, `fr`, `de`, `sv`, `it`, `tr`, `no`, `nl`, `cy`, `eu`, `ca`, `da`, `gl`, `ta`, `hu`, `fi`, `pl`, `et`, `hi`, `la`, `ur`, `th`, `vi`, `jw`, `bn`, `yo`, `sl`, `cs`, `sw`, `nn`, `he`, `ms`, `uk`, `id`, `kk`, `bg`, `lv`, `my`, `tl`, `sk`, `ne`, `fa`, `af`, `el`, `bo`, `hr`, `ro`, `sn`, `mi`, `yi`, `am`, `be`, `km`, `is`, `az`, `sd`, `br`, `sq`, `ps`, `mn`, `ht`, `ml`, `sr`, `sa`, `te`, `ka`, `bs`, `pa`, `lt`, `kn`, `si`, `hy`, `mr`, `as`, `gu`, `fo` |
| Capabilities | `text-to-speech`, `voice-cloning`, `multilingual`, `fine-tuning`, `safetensors`, `voicehub-native`, `native-runtime`, `preprocessed-training`, `noncommercial` |
| Reusable components | `dac` |
| Normalized output | `TTSOutput` |

### Language support

<details class="vh-language-support" markdown>
<summary>Supported language abbreviations</summary>

`zh`, `en`, `ja`, `ko`, `es`, `pt`, `ar`, `ru`, `fr`, `de`, `sv`, `it`, `tr`, `no`, `nl`, `cy`, `eu`, `ca`, `da`, `gl`, `ta`, `hu`, `fi`, `pl`, `et`, `hi`, `la`, `ur`, `th`, `vi`, `jw`, `bn`, `yo`, `sl`, `cs`, `sw`, `nn`, `he`, `ms`, `uk`, `id`, `kk`, `bg`, `lv`, `my`, `tl`, `sk`, `ne`, `fa`, `af`, `el`, `bo`, `hr`, `ro`, `sn`, `mi`, `yi`, `am`, `be`, `km`, `is`, `az`, `sd`, `br`, `sq`, `ps`, `mn`, `ht`, `ml`, `sr`, `sa`, `te`, `ka`, `bs`, `pa`, `lt`, `kn`, `si`, `hy`, `mr`, `as`, `gu`, `fo`

</details>

## Paper and GitHub

- **Paper:** [Fish-Speech: Leveraging Large Language Models for Advanced Multilingual TTS](https://arxiv.org/abs/2411.01156)
- **Upstream GitHub:** [Fish Speech](https://github.com/fishaudio/fish-speech)
- **VoiceHub source:** [VoiceHub model implementation](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/fishtts/modeling_fishtts.py)

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('fishtts')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `fishtts` |
| Configuration class | `FishTTSConfig` |
| Architecture class | `FishTTSForTextToSpeech` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'fishaudio/s2-pro',
    model_type='fishtts',
)
print(type(processor).__name__)
```

## Inference

The Usage example returns `TTSOutput` through `AutoModelForTextToSpeech`.

### Input and output contract

| Property | Value |
| --- | --- |
| Readiness | `preprocessed` |
| Data architecture | `codec-lm` |
| Sample rate | 44,100 Hz |
| Contract getter | `get_tts_dataset_spec('fishtts')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `semantic-tokens` | `labels` | tokens / inputs | Prepared | — |

Autoregressive text/audio-token or codec-language-model data. See the [data workflow](../../guides/data-preparation.md).

## Training and optimization

Use `available_optimization_passes()` to discover reversible public passes.
Unsupported runtime or hardware fails closed before mutation.

### Training contract

| Property | Value |
| --- | --- |
| Support | `preprocessed` |
| Family | `causal-lm` |
| Recipe | `single-phase` |
| Default phase | `semantic` |
| Training checkpoint | `fishaudio/s2-pro` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `semantic` | objective | `model` | `inputs`, `labels` | `loss`, `base_loss`, `semantic_loss` |

Prepare the exact tensors listed in the data contract before this step. Call `model.validate_training_support()` first, then follow the
[training workflow](../../guides/training.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | [`fishaudio/s2-pro`](https://huggingface.co/fishaudio/s2-pro) |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.fishtts.modeling_fishtts.FishTTSForTextToSpeech` |
| Configuration | `voicehub.models.fishtts.configuration_fishtts.FishTTSConfig` |
| Source provenance | `voicehub/models/fishtts/source/SOURCE.json` |
| License | [Fish-Audio-Research-License](https://github.com/fishaudio/fish-speech) |

Fine-tuned checkpoints are derivative works. Commercial use requires a separate written Fish Audio license. Distribution must include the Fish Audio Research License, retain its exact copyright notice, and prominently display “Built with Fish Audio”. The license also restricts using materials, derivatives, or outputs to create or improve non-Fish foundational generative-AI models. Commercial use: **not allowed**.

Confirm the checkpoint revision, access terms, provenance, and license.

### Limitations

- No integration-specific checkpoint limitation is registered. Verify the selected checkpoint revision and its documented runtime requirements.
- Validate memory, precision, and optional dependencies on the target system.
- Public optimizations fail closed when the runtime or hardware cannot satisfy
  their validation contract; an unavailable pass is not reported as applied.
- Contract tests do not replace the linked released-checkpoint evidence.

## Public API

Use the stable configuration, processor, and task-model facades below.

### `FishTTSConfig`

[View `FishTTSConfig` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/fishtts/configuration_fishtts.py)

```text
FishTTSConfig(**config_kwargs)
```

### `FishTTSForTextToSpeech`

[View `FishTTSForTextToSpeech` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/fishtts/modeling_fishtts.py)

```text
AutoModelForTextToSpeech.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='fishtts',
    config=None,
    **model_kwargs,
)
```

```python
from voicehub import get_model_spec

spec = get_model_spec('fishtts')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('fishtts')` |
| Load and run | `AutoModelForTextToSpeech` |
| Configure | `FishTTSConfig` |
| Process | `AutoProcessor` |
| Model implementation | `FishTTSForTextToSpeech` |
| Normalized output | `TTSOutput` |
| Training contract | `get_training_spec('fishtts')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).
