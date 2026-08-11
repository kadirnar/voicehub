---
description: Public API, checkpoint, training, and optimization guide for the omnivoice integration.
---

# OmniVoice {.vh-model-title}

## Usage

Complete the [VoiceHub installation](../../getting-started/installation.md) once,
then run this repository-authored example. Model pages intentionally contain no
package-install command.

This example is maintained against VoiceHub's public API; it is not copied from an upstream demo or package README.

**Model-specific path:** Pairs OmniVoice speaker audio with its transcript and selects the native iterative decoder controls.

**Inputs and controls:** Voice cloning requires both reference fields; external text normalization stays outside the model boundary.

```python
from pathlib import Path

from voicehub import AutoModelForTextToSpeech, TTSGenerationConfig

REFERENCE_AUDIO = Path("reference.wav")
REFERENCE_TEXT = "The reference transcript must exactly match the authorized audio."
if not REFERENCE_AUDIO.is_file():
    raise FileNotFoundError(REFERENCE_AUDIO)

model = AutoModelForTextToSpeech.from_pretrained(
    'k2-fsa/OmniVoice',
    model_type='omnivoice',
    device="cuda",
    lazy_load=True,
)
output = model.generate(
    'VoiceHub keeps model integrations explicit and reproducible.',
    generation_config=TTSGenerationConfig(
        seed=42,
        output_file=Path("output.wav"),
    ),
    language="en",
    speaker_audio_path=str(REFERENCE_AUDIO),
    reference_text=REFERENCE_TEXT,
    num_steps=8,
    guidance_scale=2.0,
)
print(output.file_path, output.sample_rate, output.metadata)
```

Use authorized recordings. Verify hardware needs and pin a revision in production.

## Overview

`omnivoice` is a VoiceHub **text to speech**
integration. This page is generated from its registry contract. [Open the `omnivoice` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/omnivoice.ipynb).

| Property | Value |
| --- | --- |
| Task | Text to speech |
| Architecture | `omnivoice` |
| Runtime | `VoiceHub-native` |
| Languages | `aae`, `aal`, `aao`, `ab`, `abb`, `abn`, `abr`, `abs`, `abv`, `acm`, `acw`, `acx`, `adf`, `adx`, `ady`, `aeb`, `aec`, `af`, `afb`, `afo`, `ahl`, `ahs`, `ajg`, `aju`, `ala`, `aln`, `alo`, `am`, `amu`, `an`, `anc`, `ank`, `anp`, `anw`, `aom`, `apc`, `apd`, `arb`, `arq`, `ars`, `ary`, `arz`, `as`, `ast`, `avl`, `awo`, `ayl`, `ayp`, `az`, `ba`, `bag`, `bas`, `bax`, `bba`, `bbj`, `bbl`, `bbu`, `bce`, `bci`, `bcs`, `bcy`, `bda`, `bde`, `bdm`, `be`, `beb`, `bew`, `bfd`, `bft`, `bg`, `bgp`, `bhb`, `bhh`, `bho`, `bhp`, `bhr`, `bjj`, `bjk`, `bjn`, `bjt`, `bkh`, `bkm`, `bky`, `bmm`, `bmq`, `bn`, `bnm`, `bnn`, `bns`, `bo`, `bou`, `bqg`, `br`, `bra`, `brh`, `bri`, `brx`, `bs`, `bsh`, `bsj`, `bsk`, `btm`, `btv`, `bug`, `bum`, `buo`, `bux`, `bwr`, `bxf`, `byc`, `bys`, `byv`, `byx`, `bzc`, `bzw`, `ca`, `ccg`, `ceb`, `cen`, `cfa`, `cgg`, `chq`, `cjk`, `ckb`, `ckl`, `ckr`, `cky`, `cnh`, `cpy`, `cs`, `cte`, `ctl`, `cut`, `cux`, `cv`, `cy`, `da`, `dag`, `dar`, `dav`, `dbd`, `dcc`, `de`, `deg`, `dgh`, `dgo`, `dje`, `dmk`, `dml`, `dru`, `dty`, `dua`, `dv`, `dyu`, `dzg`, `ebr`, `ebu`, `ego`, `eiv`, `eko`, `ekr`, `el`, `elm`, `en`, `eo`, `es`, `esu`, `et`, `eto`, `ets`, `etu`, `eu`, `ewo`, `ext`, `eyo`, `fa`, `fan`, `fat`, `ff`, `ffm`, `fi`, `fia`, `fil`, `fip`, `fkk`, `fmp`, `fr`, `fub`, `fuc`, `fue`, `fuf`, `fuh`, `fui`, `fuq`, `fuv`, `fy`, `ga`, `gbm`, `gbr`, `gby`, `gcc`, `gdf`, `gej`, `ges`, `ggg`, `gid`, `gig`, `giz`, `gjk`, `gju`, `gl`, `glw`, `gn`, `gol`, `gom`, `gsl`, `gu`, `gui`, `gur`, `guz`, `gv`, `gwc`, `gwe`, `gwt`, `gya`, `gyz`, `ha`, `hah`, `hao`, `haw`, `haz`, `hbb`, `he`, `hem`, `hi`, `hia`, `hkk`, `hla`, `hno`, `hoj`, `hr`, `hsb`, `ht`, `hu`, `hue`, `hul`, `hux`, `hwo`, `hy`, `hz`, `ia`, `ibb`, `id`, `ida`, `idu`, `ig`, `ijc`, `ijn`, `ik`, `ikw`, `is`, `ish`, `iso`, `it`, `its`, `itw`, `itz`, `ja`, `jal`, `jax`, `jgo`, `jmx`, `jns`, `jqr`, `juk`, `juo`, `jv`, `ka`, `kab`, `kai`, `kaj`, `kam`, `kbd`, `kbl`, `kbt`, `kcq`, `kdh`, `kea`, `keu`, `kfe`, `kfk`, `kfp`, `khg`, `khw`, `kj`, `kjc`, `kjk`, `kk`, `kln`, `kls`, `km`, `kmr`, `kmy`, `kn`, `kna`, `knn`, `ko`, `kol`, `koo`, `kpo`, `kqo`, `ks`, `ksd`, `ksf`, `kto`, `kuh`, `kvx`, `kw`, `kwm`, `kxp`, `ky`, `kyx`, `lag`, `lb`, `lcm`, `ldb`, `lg`, `lij`, `lir`, `lkb`, `lla`, `ln`, `lnu`, `lo`, `loa`, `lrk`, `lss`, `lt`, `ltg`, `lto`, `lua`, `luo`, `lus`, `lv`, `lwg`, `mab`, `maf`, `mai`, `mau`, `max`, `mbo`, `mcf`, `mcn`, `mcx`, `mdd`, `mde`, `mdf`, `mek`, `mer`, `meu`, `mfm`, `mfn`, `mfo`, `mfv`, `mgg`, `mgi`, `mhk`, `mhr`, `mi`, `mig`, `miu`, `mk`, `mkf`, `mki`, `ml`, `mlq`, `mn`, `mne`, `mni`, `mqy`, `mr`, `mrj`, `mrr`, `mrt`, `ms`, `mse`, `msh`, `msw`, `mt`, `mtr`, `mtu`, `mtx`, `mua`, `mug`, `mui`, `mve`, `mvy`, `mxs`, `mxu`, `mxy`, `my`, `myv`, `mzl`, `nal`, `nan`, `nap`, `nb`, `nbh`, `ncf`, `nco`, `ncx`, `ndi`, `ng`, `ngi`, `nhg`, `nhi`, `nhn`, `nhq`, `nja`, `nl`, `nla`, `nlv`, `nmg`, `nmz`, `nn`, `nnh`, `no`, `noe`, `npi`, `nso`, `ny`, `nyu`, `oc`, `odk`, `odu`, `ogo`, `om`, `orc`, `oru`, `ory`, `os`, `pa`, `pbs`, `pbt`, `pbu`, `pcm`, `pex`, `phl`, `phr`, `pip`, `piy`, `pko`, `pl`, `plk`, `plt`, `pmq`, `pms`, `pmy`, `pnb`, `poc`, `poe`, `pow`, `prq`, `ps`, `pst`, `pt`, `pua`, `pwn`, `qug`, `qum`, `qup`, `qur`, `qus`, `quv`, `qux`, `quy`, `qva`, `qvi`, `qvj`, `qvl`, `qwa`, `qws`, `qxa`, `qxp`, `qxt`, `qxu`, `qxw`, `rag`, `rm`, `ro`, `rob`, `rof`, `roo`, `rth`, `ru`, `rup`, `rw`, `sa`, `sah`, `sat`, `sau`, `say`, `sbn`, `sc`, `scl`, `scn`, `sd`, `sei`, `shu`, `si`, `sip`, `siw`, `sjr`, `sk`, `skg`, `skr`, `sl`, `sn`, `snc`, `snk`, `so`, `sol`, `sps`, `sq`, `sr`, `src`, `sro`, `ssi`, `ste`, `sua`, `sv`, `sva`, `sw`, `szy`, `ta`, `tan`, `tar`, `tay`, `tbf`, `tcf`, `tcy`, `tdn`, `tdx`, `te`, `tg`, `tgc`, `th`, `the`, `thq`, `thr`, `thv`, `ti`, `tig`, `tio`, `tk`, `tkg`, `tkt`, `tli`, `tlp`, `tn`, `tok`, `tpl`, `tpz`, `tqp`, `tr`, `trp`, `trq`, `trv`, `trw`, `tt`, `ttj`, `ttr`, `ttu`, `tui`, `tul`, `tuq`, `tuv`, `tuy`, `tvo`, `tvu`, `tw`, `twu`, `txs`, `txy`, `udl`, `ug`, `uk`, `uki`, `umb`, `ur`, `ush`, `uz`, `uzn`, `vai`, `var`, `ver`, `vi`, `vmc`, `vmj`, `vmm`, `vmp`, `vmz`, `vot`, `vro`, `wbl`, `wci`, `weo`, `wes`, `wja`, `wji`, `wo`, `wof`, `xh`, `xhe`, `xka`, `xmf`, `xmv`, `xmw`, `xpe`, `xti`, `xtu`, `yaq`, `yav`, `yay`, `ydd`, `ydg`, `yer`, `yes`, `yi`, `yo`, `yue`, `zga`, `zgh`, `zh`, `zoc`, `zoh`, `zor`, `zpv`, `zpy`, `ztg`, `ztn`, `ztp`, `zts`, `ztu`, `zu`, `zza` |
| Capabilities | `text-to-speech`, `voice-cloning`, `voice-design`, `multilingual`, `fine-tuning`, `safetensors`, `voicehub-native`, `native-runtime`, `raw-audio-fine-tuning`, `preencoded-code-fine-tuning` |
| Reusable components | — |
| Normalized output | `TTSOutput` |

### Language support

<details class="vh-language-support" markdown>
<summary>Supported language abbreviations</summary>

`aae`, `aal`, `aao`, `ab`, `abb`, `abn`, `abr`, `abs`, `abv`, `acm`, `acw`, `acx`, `adf`, `adx`, `ady`, `aeb`, `aec`, `af`, `afb`, `afo`, `ahl`, `ahs`, `ajg`, `aju`, `ala`, `aln`, `alo`, `am`, `amu`, `an`, `anc`, `ank`, `anp`, `anw`, `aom`, `apc`, `apd`, `arb`, `arq`, `ars`, `ary`, `arz`, `as`, `ast`, `avl`, `awo`, `ayl`, `ayp`, `az`, `ba`, `bag`, `bas`, `bax`, `bba`, `bbj`, `bbl`, `bbu`, `bce`, `bci`, `bcs`, `bcy`, `bda`, `bde`, `bdm`, `be`, `beb`, `bew`, `bfd`, `bft`, `bg`, `bgp`, `bhb`, `bhh`, `bho`, `bhp`, `bhr`, `bjj`, `bjk`, `bjn`, `bjt`, `bkh`, `bkm`, `bky`, `bmm`, `bmq`, `bn`, `bnm`, `bnn`, `bns`, `bo`, `bou`, `bqg`, `br`, `bra`, `brh`, `bri`, `brx`, `bs`, `bsh`, `bsj`, `bsk`, `btm`, `btv`, `bug`, `bum`, `buo`, `bux`, `bwr`, `bxf`, `byc`, `bys`, `byv`, `byx`, `bzc`, `bzw`, `ca`, `ccg`, `ceb`, `cen`, `cfa`, `cgg`, `chq`, `cjk`, `ckb`, `ckl`, `ckr`, `cky`, `cnh`, `cpy`, `cs`, `cte`, `ctl`, `cut`, `cux`, `cv`, `cy`, `da`, `dag`, `dar`, `dav`, `dbd`, `dcc`, `de`, `deg`, `dgh`, `dgo`, `dje`, `dmk`, `dml`, `dru`, `dty`, `dua`, `dv`, `dyu`, `dzg`, `ebr`, `ebu`, `ego`, `eiv`, `eko`, `ekr`, `el`, `elm`, `en`, `eo`, `es`, `esu`, `et`, `eto`, `ets`, `etu`, `eu`, `ewo`, `ext`, `eyo`, `fa`, `fan`, `fat`, `ff`, `ffm`, `fi`, `fia`, `fil`, `fip`, `fkk`, `fmp`, `fr`, `fub`, `fuc`, `fue`, `fuf`, `fuh`, `fui`, `fuq`, `fuv`, `fy`, `ga`, `gbm`, `gbr`, `gby`, `gcc`, `gdf`, `gej`, `ges`, `ggg`, `gid`, `gig`, `giz`, `gjk`, `gju`, `gl`, `glw`, `gn`, `gol`, `gom`, `gsl`, `gu`, `gui`, `gur`, `guz`, `gv`, `gwc`, `gwe`, `gwt`, `gya`, `gyz`, `ha`, `hah`, `hao`, `haw`, `haz`, `hbb`, `he`, `hem`, `hi`, `hia`, `hkk`, `hla`, `hno`, `hoj`, `hr`, `hsb`, `ht`, `hu`, `hue`, `hul`, `hux`, `hwo`, `hy`, `hz`, `ia`, `ibb`, `id`, `ida`, `idu`, `ig`, `ijc`, `ijn`, `ik`, `ikw`, `is`, `ish`, `iso`, `it`, `its`, `itw`, `itz`, `ja`, `jal`, `jax`, `jgo`, `jmx`, `jns`, `jqr`, `juk`, `juo`, `jv`, `ka`, `kab`, `kai`, `kaj`, `kam`, `kbd`, `kbl`, `kbt`, `kcq`, `kdh`, `kea`, `keu`, `kfe`, `kfk`, `kfp`, `khg`, `khw`, `kj`, `kjc`, `kjk`, `kk`, `kln`, `kls`, `km`, `kmr`, `kmy`, `kn`, `kna`, `knn`, `ko`, `kol`, `koo`, `kpo`, `kqo`, `ks`, `ksd`, `ksf`, `kto`, `kuh`, `kvx`, `kw`, `kwm`, `kxp`, `ky`, `kyx`, `lag`, `lb`, `lcm`, `ldb`, `lg`, `lij`, `lir`, `lkb`, `lla`, `ln`, `lnu`, `lo`, `loa`, `lrk`, `lss`, `lt`, `ltg`, `lto`, `lua`, `luo`, `lus`, `lv`, `lwg`, `mab`, `maf`, `mai`, `mau`, `max`, `mbo`, `mcf`, `mcn`, `mcx`, `mdd`, `mde`, `mdf`, `mek`, `mer`, `meu`, `mfm`, `mfn`, `mfo`, `mfv`, `mgg`, `mgi`, `mhk`, `mhr`, `mi`, `mig`, `miu`, `mk`, `mkf`, `mki`, `ml`, `mlq`, `mn`, `mne`, `mni`, `mqy`, `mr`, `mrj`, `mrr`, `mrt`, `ms`, `mse`, `msh`, `msw`, `mt`, `mtr`, `mtu`, `mtx`, `mua`, `mug`, `mui`, `mve`, `mvy`, `mxs`, `mxu`, `mxy`, `my`, `myv`, `mzl`, `nal`, `nan`, `nap`, `nb`, `nbh`, `ncf`, `nco`, `ncx`, `ndi`, `ng`, `ngi`, `nhg`, `nhi`, `nhn`, `nhq`, `nja`, `nl`, `nla`, `nlv`, `nmg`, `nmz`, `nn`, `nnh`, `no`, `noe`, `npi`, `nso`, `ny`, `nyu`, `oc`, `odk`, `odu`, `ogo`, `om`, `orc`, `oru`, `ory`, `os`, `pa`, `pbs`, `pbt`, `pbu`, `pcm`, `pex`, `phl`, `phr`, `pip`, `piy`, `pko`, `pl`, `plk`, `plt`, `pmq`, `pms`, `pmy`, `pnb`, `poc`, `poe`, `pow`, `prq`, `ps`, `pst`, `pt`, `pua`, `pwn`, `qug`, `qum`, `qup`, `qur`, `qus`, `quv`, `qux`, `quy`, `qva`, `qvi`, `qvj`, `qvl`, `qwa`, `qws`, `qxa`, `qxp`, `qxt`, `qxu`, `qxw`, `rag`, `rm`, `ro`, `rob`, `rof`, `roo`, `rth`, `ru`, `rup`, `rw`, `sa`, `sah`, `sat`, `sau`, `say`, `sbn`, `sc`, `scl`, `scn`, `sd`, `sei`, `shu`, `si`, `sip`, `siw`, `sjr`, `sk`, `skg`, `skr`, `sl`, `sn`, `snc`, `snk`, `so`, `sol`, `sps`, `sq`, `sr`, `src`, `sro`, `ssi`, `ste`, `sua`, `sv`, `sva`, `sw`, `szy`, `ta`, `tan`, `tar`, `tay`, `tbf`, `tcf`, `tcy`, `tdn`, `tdx`, `te`, `tg`, `tgc`, `th`, `the`, `thq`, `thr`, `thv`, `ti`, `tig`, `tio`, `tk`, `tkg`, `tkt`, `tli`, `tlp`, `tn`, `tok`, `tpl`, `tpz`, `tqp`, `tr`, `trp`, `trq`, `trv`, `trw`, `tt`, `ttj`, `ttr`, `ttu`, `tui`, `tul`, `tuq`, `tuv`, `tuy`, `tvo`, `tvu`, `tw`, `twu`, `txs`, `txy`, `udl`, `ug`, `uk`, `uki`, `umb`, `ur`, `ush`, `uz`, `uzn`, `vai`, `var`, `ver`, `vi`, `vmc`, `vmj`, `vmm`, `vmp`, `vmz`, `vot`, `vro`, `wbl`, `wci`, `weo`, `wes`, `wja`, `wji`, `wo`, `wof`, `xh`, `xhe`, `xka`, `xmf`, `xmv`, `xmw`, `xpe`, `xti`, `xtu`, `yaq`, `yav`, `yay`, `ydd`, `ydg`, `yer`, `yes`, `yi`, `yo`, `yue`, `zga`, `zgh`, `zh`, `zoc`, `zoh`, `zor`, `zpv`, `zpy`, `ztg`, `ztn`, `ztp`, `zts`, `ztu`, `zu`, `zza`

</details>

## Paper and GitHub

- **Paper:** No dedicated upstream research paper is published for this integration.
- **Upstream GitHub:** [OmniVoice](https://github.com/k2-fsa/OmniVoice)
- **VoiceHub source:** [VoiceHub model implementation](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/omnivoice/modeling_omnivoice.py)

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('omnivoice')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `omnivoice` |
| Configuration class | `OmniVoiceConfig` |
| Architecture class | `OmniVoiceForTextToSpeech` |

## Processing

Create the registered processor without allocating model weights:

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'k2-fsa/OmniVoice',
    model_type='omnivoice',
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
| Contract getter | `get_tts_dataset_spec('omnivoice')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `raw-audio` | `text` | audio / waveform | Source | at most one: audio / waveform; forbidden: audio_tokens |
| `audio-tokens` | `text`, `audio_tokens` | — | Prepared | forbidden: audio, waveform |

Multi-component language-model, diffusion, acoustic, or GAN data. See the [data workflow](../../guides/data-preparation.md).

## Training and optimization

Use `available_optimization_passes()` to discover reversible public passes.
Unsupported runtime or hardware fails closed before mutation.

### Training contract

| Property | Value |
| --- | --- |
| Support | `native` |
| Family | `composite` |
| Recipe | `single-phase` |
| Default phase | `masked_audio` |
| Training checkpoint | `k2-fsa/OmniVoice` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `masked_audio` | objective | `model` | `input_ids`, `audio_mask`, `labels` | `loss` |

The integration accepts its declared source or prepared contract directly. Call `model.validate_training_support()` first, then follow the
[training workflow](../../guides/training.md).

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | [`k2-fsa/OmniVoice`](https://huggingface.co/k2-fsa/OmniVoice) |
| Hugging Face ID | [`k2-fsa/OmniVoice`](https://huggingface.co/k2-fsa/OmniVoice)<br>Repository availability verified through the Hugging Face model API on 2026-08-11; pin a revision before production use. |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.omnivoice.modeling_omnivoice.OmniVoiceForTextToSpeech` |
| Configuration | `voicehub.models.omnivoice.configuration_omnivoice.OmniVoiceConfig` |
| Source provenance | `voicehub/models/omnivoice/source/SOURCE.json` |
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

### `OmniVoiceConfig`

[View `OmniVoiceConfig` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/omnivoice/configuration_omnivoice.py)

```text
OmniVoiceConfig(**config_kwargs)
```

### `OmniVoiceForTextToSpeech`

[View `OmniVoiceForTextToSpeech` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/omnivoice/modeling_omnivoice.py)

```text
AutoModelForTextToSpeech.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='omnivoice',
    config=None,
    **model_kwargs,
)
```

```python
from voicehub import get_model_spec

spec = get_model_spec('omnivoice')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('omnivoice')` |
| Load and run | `AutoModelForTextToSpeech` |
| Configure | `OmniVoiceConfig` |
| Process | `AutoProcessor` |
| Model implementation | `OmniVoiceForTextToSpeech` |
| Normalized output | `TTSOutput` |
| Training contract | `get_training_spec('omnivoice')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).
