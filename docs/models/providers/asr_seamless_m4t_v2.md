---
description: Public API, checkpoint, training, and optimization guide for the asr_seamless_m4t_v2 integration.
hide:
  - toc
---

<div class="vh-model-detail" data-vh-model-detail data-model-type="asr_seamless_m4t_v2" data-task="automatic-speech-recognition" data-training="native" data-parameter-count="1501842240" markdown>

<header class="vh-model-detail__hero" data-vh-model-hero markdown>

<p class="vh-model-detail__namespace" aria-label="Model repository"><span class="vh-model-detail__owner-avatar" aria-hidden="true">FA</span><a href="https://huggingface.co/facebook">facebook</a><span aria-hidden="true">/</span><strong>seamless-m4t-v2-large</strong></p>

# SeamlessM4Tv2 {.vh-model-title}

<p class="vh-model-detail__summary">Selects SeamlessM4T v2 transcription rather than speech translation.</p>
<div class="vh-model-detail__tags" aria-label="Model metadata"><span class="vh-model-detail__chip" data-chip-kind="task">Automatic speech recognition</span><span class="vh-model-detail__chip" data-chip-kind="runtime">VoiceHub-native</span><span class="vh-model-detail__chip" data-chip-kind="architecture">seamless-m4t-v2-s2t</span><span class="vh-model-detail__chip" data-chip-kind="parameters" aria-describedby="vh-model-parameters-note-asr_seamless_m4t_v2">Parameters: 1.5B</span><span class="vh-model-detail__chip" data-chip-kind="language">Languages: afr, amh +96</span><span class="vh-model-detail__chip" data-chip-kind="training">Training: native</span><span class="vh-model-detail__chip" data-chip-kind="license">License: CC-BY-NC-4.0</span></div>
<p class="vh-model-detail__parameter-note" id="vh-model-parameters-note-asr_seamless_m4t_v2"><strong>Parameter metadata:</strong> Exact parameter total for the speech-to-text subset loaded from VoiceHub&#x27;s audited unified default checkpoint.</p>
<div class="vh-model-detail__actions" aria-label="Model actions">
<a class="vh-model-detail__action vh-model-detail__action--primary" href="#usage" data-vh-model-action="use">Use this model</a>
<button class="vh-model-detail__action vh-model-detail__copy" type="button" data-vh-copy-model-id data-model-id="facebook/seamless-m4t-v2-large" aria-describedby="vh-model-checkpoint-asr_seamless_m4t_v2"><span data-vh-copy-model-id-label>Copy model ID</span></button>
<a class="vh-model-detail__action" href="https://huggingface.co/facebook/seamless-m4t-v2-large" data-vh-model-action="checkpoint">Checkpoint</a>
<details class="vh-model-detail__resources">
<summary class="vh-model-detail__action">Resources</summary>
<div class="vh-model-detail__resource-menu">
<a href="https://arxiv.org/abs/2312.05187" data-vh-model-action="paper">Paper</a>
<a href="https://github.com/facebookresearch/seamless_communication" data-vh-model-action="github">Upstream GitHub</a>
<a href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_seamless_m4t_v2/modeling_asr_seamless_m4t_v2.py" data-vh-model-action="source">VoiceHub source</a>
<a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/asr_seamless_m4t_v2.ipynb" data-vh-model-action="colab">Open in Colab</a>
</div>
</details>
</div>
</header>

<nav class="vh-model-detail__tabs" aria-label="Model sections"><a href="#usage" data-vh-model-tab="usage">Usage</a><a href="#overview" data-vh-model-tab="model-card" aria-current="location">Model card</a><a href="#paper-and-github" data-vh-model-tab="sources">Sources</a><a href="#training-and-optimization" data-vh-model-tab="training">Training</a><a href="#checkpoints-provenance-license-and-limitations" data-vh-model-tab="checkpoint">Checkpoint</a><a href="#public-api" data-vh-model-tab="api">Public API</a></nav>

<div class="vh-model-detail__layout" markdown>

<aside class="vh-model-detail__sidebar" data-vh-model-facts aria-labelledby="vh-model-facts-title-asr_seamless_m4t_v2"><h2 id="vh-model-facts-title-asr_seamless_m4t_v2">Model facts</h2><details class="vh-model-detail__facts-disclosure" data-vh-model-facts-disclosure aria-labelledby="vh-model-facts-title-asr_seamless_m4t_v2" open><summary><span>Toggle model facts</span></summary><dl class="vh-model-detail__facts"><div><dt>Task</dt><dd>Automatic speech recognition</dd></div><div><dt>Parameters</dt><dd aria-describedby="vh-model-parameters-note-asr_seamless_m4t_v2">1.5B</dd></div><div><dt>Architecture</dt><dd><code>seamless-m4t-v2-s2t</code></dd></div><div><dt>Runtime</dt><dd>VoiceHub-native</dd></div><div><dt>Languages</dt><dd><details class="vh-model-detail__languages"><summary>98 documented codes</summary><span><code>afr</code> <code>amh</code> <code>arb</code> <code>ary</code> <code>arz</code> <code>asm</code> <code>azj</code> <code>bel</code> <code>ben</code> <code>bos</code> <code>bul</code> <code>cat</code> <code>ceb</code> <code>ces</code> <code>ckb</code> <code>cmn</code> <code>cmn_Hant</code> <code>cym</code> <code>dan</code> <code>deu</code> <code>ell</code> <code>eng</code> <code>est</code> <code>eus</code> <code>fin</code> <code>fra</code> <code>fuv</code> <code>gaz</code> <code>gle</code> <code>glg</code> <code>guj</code> <code>heb</code> <code>hin</code> <code>hrv</code> <code>hun</code> <code>hye</code> <code>ibo</code> <code>ind</code> <code>isl</code> <code>ita</code> <code>jav</code> <code>jpn</code> <code>kan</code> <code>kat</code> <code>kaz</code> <code>khk</code> <code>khm</code> <code>kir</code> <code>kor</code> <code>lao</code> <code>lit</code> <code>lug</code> <code>luo</code> <code>lvs</code> <code>mai</code> <code>mal</code> <code>mar</code> <code>mkd</code> <code>mlt</code> <code>mni</code> <code>mya</code> <code>nld</code> <code>nno</code> <code>nob</code> <code>npi</code> <code>nya</code> <code>ory</code> <code>pan</code> <code>pbt</code> <code>pes</code> <code>pol</code> <code>por</code> <code>ron</code> <code>rus</code> <code>sat</code> <code>slk</code> <code>slv</code> <code>sna</code> <code>snd</code> <code>som</code> <code>spa</code> <code>srp</code> <code>swe</code> <code>swh</code> <code>tam</code> <code>tel</code> <code>tgk</code> <code>tgl</code> <code>tha</code> <code>tur</code> <code>ukr</code> <code>urd</code> <code>uzn</code> <code>vie</code> <code>yor</code> <code>yue</code> <code>zlm</code> <code>zul</code></span></details></dd></div><div><dt>Capabilities</dt><dd><details class="vh-model-detail__capabilities"><summary>8 capabilities</summary><span><code>automatic-speech-recognition</code> <code>multilingual</code> <code>safetensors</code> <code>fine-tuning</code> <code>voicehub-native</code> <code>native-runtime</code> <code>greedy-decoding</code> <code>full-model-training</code></span></details></dd></div><div><dt>Training</dt><dd><code>native</code></dd></div><div><dt>License</dt><dd><a href="https://huggingface.co/facebook/seamless-m4t-v2-large">CC-BY-NC-4.0</a></dd></div><div><dt>Default checkpoint</dt><dd id="vh-model-checkpoint-asr_seamless_m4t_v2"><a href="https://huggingface.co/facebook/seamless-m4t-v2-large"><code>facebook/seamless-m4t-v2-large</code></a></dd></div></dl></details></aside>

<div class="vh-model-detail__main vh-model-detail__content" markdown>

## Usage

Complete the [VoiceHub installation](../../getting-started/installation.md) once,
then run this repository-authored example. Model pages intentionally contain no
package-install command.

This example is maintained against VoiceHub's public API; it is not copied from an upstream demo or package README.

**Model-specific path:** Selects SeamlessM4T v2 transcription rather than speech translation.

**Inputs and controls:** The native complete-waveform path is greedy and does not claim timestamp alignment.

```python
from pathlib import Path

from voicehub import AutoModelForSpeechRecognition

AUDIO_FILE = Path("speech.wav")
if not AUDIO_FILE.is_file():
    raise FileNotFoundError(AUDIO_FILE)

model = AutoModelForSpeechRecognition.from_pretrained(
    'facebook/seamless-m4t-v2-large',
    model_type='asr_seamless_m4t_v2',
    device="cuda",
    lazy_load=True,
)
output = model.transcribe(
    AUDIO_FILE,
    task="transcribe",
    num_beams=1,
    max_new_tokens=256,
)
print(output.text)
for segment in output.segments:
    print(segment.start, segment.end, segment.text, segment.confidence)
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
| Languages | `afr`, `amh`, `arb`, `ary`, … complete audited list below |
| Capabilities | `automatic-speech-recognition`, `multilingual`, `safetensors`, `fine-tuning`, `voicehub-native`, `native-runtime`, `greedy-decoding`, `full-model-training` |
| Reusable components | — |
| Normalized output | `ASROutput` |

### Language support

<details class="vh-language-support" markdown>
<summary>Supported language abbreviations</summary>

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
| Hugging Face ID | [`facebook/seamless-m4t-v2-large`](https://huggingface.co/facebook/seamless-m4t-v2-large)<br>Repository availability verified through the Hugging Face model API on 2026-08-11; pin a revision before production use. |
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

<section class="vh-model-api-card" data-vh-model-api-card="configuration" markdown>
<p class="vh-model-api-card__badge-wrap"><span class="vh-model-api-card__badge">Configuration</span></p>

### `SeamlessM4Tv2ASRConfig`

<p class="vh-model-api-card__source-wrap"><a class="vh-model-api-card__source" href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_seamless_m4t_v2/configuration_asr_seamless_m4t_v2.py">View source</a></p>
<div class="vh-model-api-card__signature" markdown>

```text
SeamlessM4Tv2ASRConfig(**config_kwargs)
```

</div>
<h4>Parameters</h4>
<div class="vh-model-api-card__parameters" markdown>
- `**config_kwargs` — Configuration fields validated by SeamlessM4Tv2ASRConfig.
</div>
</section>

<section class="vh-model-api-card" data-vh-model-api-card="model" markdown>
<p class="vh-model-api-card__badge-wrap"><span class="vh-model-api-card__badge">Model</span></p>

### `SeamlessM4Tv2ForSpeechRecognition`

<p class="vh-model-api-card__source-wrap"><a class="vh-model-api-card__source" href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_seamless_m4t_v2/modeling_asr_seamless_m4t_v2.py">View source</a></p>
<div class="vh-model-api-card__signature" markdown>

```text
AutoModelForSpeechRecognition.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='asr_seamless_m4t_v2',
    config=None,
    **model_kwargs,
)
```

</div>
<h4>Parameters</h4>
<div class="vh-model-api-card__parameters" markdown>
- `pretrained_model_name_or_path` — Hub ID or compatible local directory.
- `model_type` — Canonical model type; use 'asr_seamless_m4t_v2'.
- `config` — Optional preloaded SeamlessM4Tv2ASRConfig instance.
- `**model_kwargs` — Model-specific loading arguments.
</div>
</section>

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

</div>

</div>

</div>
