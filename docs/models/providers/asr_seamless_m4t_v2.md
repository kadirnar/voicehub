---
description: Public API, checkpoint, training, and optimization guide for the asr_seamless_m4t_v2 integration.
---

# SeamlessM4Tv2 {.vh-model-title}

## Usage

```bash
python -m pip install "voicehub @ git+https://github.com/kadirnar/voicehub.git@main"
```

Install from source, then choose a compatible checkpoint. Place a supported recording at `speech.wav` and inspect the transcript.

```python
from voicehub import AutoModelForSpeechRecognition

model = AutoModelForSpeechRecognition.from_pretrained(
    'facebook/seamless-m4t-v2-large',
    model_type='asr_seamless_m4t_v2',
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

`asr_seamless_m4t_v2` is a VoiceHub **automatic speech recognition**
integration. This page is generated from its registry contract. [Open the `asr_seamless_m4t_v2` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/asr_seamless_m4t_v2.ipynb).

| Property | Value |
| --- | --- |
| Task | Automatic speech recognition |
| Architecture | `seamless-m4t-v2-s2t` |
| Runtime | `VoiceHub-native` |
| Languages | 98 enumerated languages |
| Capabilities | `automatic-speech-recognition`, `multilingual`, `safetensors`, `fine-tuning`, `voicehub-native`, `native-runtime`, `greedy-decoding`, `full-model-training` |
| Reusable components | — |
| Normalized output | `ASROutput` |

### Language support

<details class="vh-language-support" markdown>
<summary>98 documented languages</summary>

`afr`, `amh`, `arb`, `ary`, `arz`, `asm`, `azj`, `bel`, `ben`, `bos`, `bul`, `cat`, `ceb`, `ces`, `ckb`, `cmn`, `cmn_Hant`, `cym`, `dan`, `deu`, `ell`, `eng`, `est`, `eus`, `fin`, `fra`, `fuv`, `gaz`, `gle`, `glg`, `guj`, `heb`, `hin`, `hrv`, `hun`, `hye`, `ibo`, `ind`, `isl`, `ita`, `jav`, `jpn`, `kan`, `kat`, `kaz`, `khk`, `khm`, `kir`, `kor`, `lao`, `lit`, `lug`, `luo`, `lvs`, `mai`, `mal`, `mar`, `mkd`, `mlt`, `mni`, `mya`, `nld`, `nno`, `nob`, `npi`, `nya`, `ory`, `pan`, `pbt`, `pes`, `pol`, `por`, `ron`, `rus`, `sat`, `slk`, `slv`, `sna`, `snd`, `som`, `spa`, `srp`, `swe`, `swh`, `tam`, `tel`, `tgk`, `tgl`, `tha`, `tur`, `ukr`, `urd`, `uzn`, `vie`, `yor`, `yue`, `zlm`, `zul`

These are output-language prompts supported by the audited S2T checkpoint.

</details>

## Paper and GitHub

- **Paper:** [Seamless: Multilingual Expressive and Streaming Speech Translation](https://arxiv.org/abs/2312.05187)
- **Upstream GitHub:** [Seamless Communication](https://github.com/facebookresearch/seamless_communication)
- **VoiceHub source:** [VoiceHub model implementation](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_seamless_m4t_v2/modeling_asr_seamless_m4t_v2.py)

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('asr_seamless_m4t_v2')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `asr_seamless_m4t_v2` |
| Configuration class | `SeamlessM4Tv2ASRConfig` |
| Architecture class | `SeamlessM4Tv2ForSpeechRecognition` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'facebook/seamless-m4t-v2-large',
    model_type='asr_seamless_m4t_v2',
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
| Contract getter | `get_asr_dataset_spec('asr_seamless_m4t_v2')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `raw-audio` | `audio` | text / transcription / transcript | Source | at most one: text / transcription / transcript |
| `seamless-model-ready` | `input_features`, `attention_mask`, `labels` | — | Prepared | — |

SeamlessM4T-v2 multilingual speech-to-text records. See the [data workflow](../../guides/speech-data.md).

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
| Training checkpoint | `facebook/seamless-m4t-v2-large` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `speech_recognition` | objective | `model.speech_encoder`, `model.text_decoder`, `model.shared`, `model.lm_head` | `input_features`, `attention_mask`, `labels` | `loss` |

The integration accepts its declared source or prepared contract directly. Call `model.validate_training_support()` first, then follow the
[training workflow](../../guides/training.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | [`facebook/seamless-m4t-v2-large`](https://huggingface.co/facebook/seamless-m4t-v2-large) |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.asr_seamless_m4t_v2.modeling_asr_seamless_m4t_v2.SeamlessM4Tv2ForSpeechRecognition` |
| Configuration | `voicehub.models.asr_seamless_m4t_v2.configuration_asr_seamless_m4t_v2.SeamlessM4Tv2ASRConfig` |
| Source provenance | `voicehub/architectures/seamless_m4t_v2/SOURCE.json` |
| License | [CC-BY-NC-4.0](https://huggingface.co/facebook/seamless-m4t-v2-large) |

The pinned SeamlessM4T-v2 Large checkpoint and fine-tuned derivatives are non-commercial under CC-BY-NC-4.0. The VoiceHub-native S2T architecture port is audited against Apache-2.0 Transformers source. Commercial use: **not allowed**.

Confirm the checkpoint revision, access terms, provenance, and license.

### Limitations

- No integration-specific checkpoint limitation is registered. Verify the selected checkpoint revision and its documented runtime requirements.
- Validate memory, precision, and optional dependencies on the target system.
- Public optimizations fail closed when the runtime or hardware cannot satisfy
  their validation contract; an unavailable pass is not reported as applied.
- Contract tests do not replace the linked released-checkpoint evidence.

## Public API

Use the stable configuration, processor, and task-model facades below.

### `SeamlessM4Tv2ASRConfig`

[View `SeamlessM4Tv2ASRConfig` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_seamless_m4t_v2/configuration_asr_seamless_m4t_v2.py)

```text
SeamlessM4Tv2ASRConfig(**config_kwargs)
```

### `SeamlessM4Tv2ForSpeechRecognition`

[View `SeamlessM4Tv2ForSpeechRecognition` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_seamless_m4t_v2/modeling_asr_seamless_m4t_v2.py)

```text
AutoModelForSpeechRecognition.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='asr_seamless_m4t_v2',
    config=None,
    **model_kwargs,
)
```

```python
from voicehub import get_model_spec

spec = get_model_spec('asr_seamless_m4t_v2')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('asr_seamless_m4t_v2')` |
| Load and run | `AutoModelForSpeechRecognition` |
| Configure | `SeamlessM4Tv2ASRConfig` |
| Process | `AutoProcessor` |
| Model implementation | `SeamlessM4Tv2ForSpeechRecognition` |
| Normalized output | `ASROutput` |
| Training contract | `get_training_spec('asr_seamless_m4t_v2')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).
