---
description: Public API, checkpoint, training, and optimization guide for the omnivoice integration.
hide:
  - toc
---

<div class="vh-model-detail" data-vh-model-detail data-model-type="omnivoice" data-task="text-to-speech" data-training="native" data-parameter-count="612577280" markdown>

<header class="vh-model-detail__hero" data-vh-model-hero markdown>

<p class="vh-model-detail__namespace" aria-label="Model repository"><span class="vh-model-detail__owner-avatar" aria-hidden="true">KF</span><a href="https://huggingface.co/k2-fsa">k2-fsa</a><span aria-hidden="true">/</span><strong>OmniVoice</strong></p>

# OmniVoice {.vh-model-title}

<p class="vh-model-detail__summary">Pairs OmniVoice speaker audio with its transcript and selects the native iterative decoder controls.</p>
<div class="vh-model-detail__tags" aria-label="Model metadata"><span class="vh-model-detail__chip" data-chip-kind="task">Text to speech</span><span class="vh-model-detail__chip" data-chip-kind="runtime">VoiceHub-native</span><span class="vh-model-detail__chip" data-chip-kind="architecture">omnivoice</span><span class="vh-model-detail__chip" data-chip-kind="parameters" aria-describedby="vh-model-parameters-note-omnivoice">Parameters: 612.6M</span><span class="vh-model-detail__chip" data-chip-kind="language">Languages: aae, aal +644</span><span class="vh-model-detail__chip" data-chip-kind="training">Training: native</span><span class="vh-model-detail__chip" data-chip-kind="license">License: Checkpoint-specific</span></div>
<p class="vh-model-detail__parameter-note" id="vh-model-parameters-note-omnivoice"><strong>Parameter metadata:</strong> Exact learned-parameter total for VoiceHub&#x27;s audited native primary graph at the registered default selection; separately loaded auxiliary models are excluded.</p>
<div class="vh-model-detail__actions" aria-label="Model actions">
<a class="vh-model-detail__action vh-model-detail__action--primary" href="#usage" data-vh-model-action="use">Use this model</a>
<button class="vh-model-detail__action vh-model-detail__copy" type="button" data-vh-copy-model-id data-model-id="k2-fsa/OmniVoice" aria-describedby="vh-model-checkpoint-omnivoice"><span data-vh-copy-model-id-label>Copy model ID</span></button>
<a class="vh-model-detail__action" href="https://huggingface.co/k2-fsa/OmniVoice" data-vh-model-action="checkpoint">Checkpoint</a>
<details class="vh-model-detail__resources">
<summary class="vh-model-detail__action">Resources</summary>
<div class="vh-model-detail__resource-menu">
<a href="https://github.com/k2-fsa/OmniVoice" data-vh-model-action="github">Upstream GitHub</a>
<a href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/omnivoice/modeling_omnivoice.py" data-vh-model-action="source">VoiceHub source</a>
<a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/omnivoice.ipynb" data-vh-model-action="colab">Open in Colab</a>
</div>
</details>
</div>
</header>

<nav class="vh-model-detail__tabs" aria-label="Model sections"><a href="#usage" data-vh-model-tab="usage">Usage</a><a href="#overview" data-vh-model-tab="model-card" aria-current="location">Model card</a><a href="#paper-and-github" data-vh-model-tab="sources">Sources</a><a href="#training-and-optimization" data-vh-model-tab="training">Training</a><a href="#checkpoints-provenance-license-and-limitations" data-vh-model-tab="checkpoint">Checkpoint</a><a href="#public-api" data-vh-model-tab="api">Public API</a></nav>

<div class="vh-model-detail__layout" markdown>

<aside class="vh-model-detail__sidebar" data-vh-model-facts aria-labelledby="vh-model-facts-title-omnivoice"><h2 id="vh-model-facts-title-omnivoice">Model facts</h2><details class="vh-model-detail__facts-disclosure" data-vh-model-facts-disclosure aria-labelledby="vh-model-facts-title-omnivoice" open><summary><span>Toggle model facts</span></summary><dl class="vh-model-detail__facts"><div><dt>Task</dt><dd>Text to speech</dd></div><div><dt>Parameters</dt><dd aria-describedby="vh-model-parameters-note-omnivoice">612.6M</dd></div><div><dt>Architecture</dt><dd><code>omnivoice</code></dd></div><div><dt>Runtime</dt><dd>VoiceHub-native</dd></div><div><dt>Languages</dt><dd><details class="vh-model-detail__languages"><summary>646 documented codes</summary><span><code>aae</code> <code>aal</code> <code>aao</code> <code>ab</code> <code>abb</code> <code>abn</code> <code>abr</code> <code>abs</code> <code>abv</code> <code>acm</code> <code>acw</code> <code>acx</code> <code>adf</code> <code>adx</code> <code>ady</code> <code>aeb</code> <code>aec</code> <code>af</code> <code>afb</code> <code>afo</code> <code>ahl</code> <code>ahs</code> <code>ajg</code> <code>aju</code> <code>ala</code> <code>aln</code> <code>alo</code> <code>am</code> <code>amu</code> <code>an</code> <code>anc</code> <code>ank</code> <code>anp</code> <code>anw</code> <code>aom</code> <code>apc</code> <code>apd</code> <code>arb</code> <code>arq</code> <code>ars</code> <code>ary</code> <code>arz</code> <code>as</code> <code>ast</code> <code>avl</code> <code>awo</code> <code>ayl</code> <code>ayp</code> <code>az</code> <code>ba</code> <code>bag</code> <code>bas</code> <code>bax</code> <code>bba</code> <code>bbj</code> <code>bbl</code> <code>bbu</code> <code>bce</code> <code>bci</code> <code>bcs</code> <code>bcy</code> <code>bda</code> <code>bde</code> <code>bdm</code> <code>be</code> <code>beb</code> <code>bew</code> <code>bfd</code> <code>bft</code> <code>bg</code> <code>bgp</code> <code>bhb</code> <code>bhh</code> <code>bho</code> <code>bhp</code> <code>bhr</code> <code>bjj</code> <code>bjk</code> <code>bjn</code> <code>bjt</code> <code>bkh</code> <code>bkm</code> <code>bky</code> <code>bmm</code> <code>bmq</code> <code>bn</code> <code>bnm</code> <code>bnn</code> <code>bns</code> <code>bo</code> <code>bou</code> <code>bqg</code> <code>br</code> <code>bra</code> <code>brh</code> <code>bri</code> <code>brx</code> <code>bs</code> <code>bsh</code> <code>bsj</code> <code>bsk</code> <code>btm</code> <code>btv</code> <code>bug</code> <code>bum</code> <code>buo</code> <code>bux</code> <code>bwr</code> <code>bxf</code> <code>byc</code> <code>bys</code> <code>byv</code> <code>byx</code> <code>bzc</code> <code>bzw</code> <code>ca</code> <code>ccg</code> <code>ceb</code> <code>cen</code> <code>cfa</code> <code>cgg</code> <code>chq</code> <code>cjk</code> <code>ckb</code> <code>ckl</code> <code>ckr</code> <code>cky</code> <code>cnh</code> <code>cpy</code> <code>cs</code> <code>cte</code> <code>ctl</code> <code>cut</code> <code>cux</code> <code>cv</code> <code>cy</code> <code>da</code> <code>dag</code> <code>dar</code> <code>dav</code> <code>dbd</code> <code>dcc</code> <code>de</code> <code>deg</code> <code>dgh</code> <code>dgo</code> <code>dje</code> <code>dmk</code> <code>dml</code> <code>dru</code> <code>dty</code> <code>dua</code> <code>dv</code> <code>dyu</code> <code>dzg</code> <code>ebr</code> <code>ebu</code> <code>ego</code> <code>eiv</code> <code>eko</code> <code>ekr</code> <code>el</code> <code>elm</code> <code>en</code> <code>eo</code> <code>es</code> <code>esu</code> <code>et</code> <code>eto</code> <code>ets</code> <code>etu</code> <code>eu</code> <code>ewo</code> <code>ext</code> <code>eyo</code> <code>fa</code> <code>fan</code> <code>fat</code> <code>ff</code> <code>ffm</code> <code>fi</code> <code>fia</code> <code>fil</code> <code>fip</code> <code>fkk</code> <code>fmp</code> <code>fr</code> <code>fub</code> <code>fuc</code> <code>fue</code> <code>fuf</code> <code>fuh</code> <code>fui</code> <code>fuq</code> <code>fuv</code> <code>fy</code> <code>ga</code> <code>gbm</code> <code>gbr</code> <code>gby</code> <code>gcc</code> <code>gdf</code> <code>gej</code> <code>ges</code> <code>ggg</code> <code>gid</code> <code>gig</code> <code>giz</code> <code>gjk</code> <code>gju</code> <code>gl</code> <code>glw</code> <code>gn</code> <code>gol</code> <code>gom</code> <code>gsl</code> <code>gu</code> <code>gui</code> <code>gur</code> <code>guz</code> <code>gv</code> <code>gwc</code> <code>gwe</code> <code>gwt</code> <code>gya</code> <code>gyz</code> <code>ha</code> <code>hah</code> <code>hao</code> <code>haw</code> <code>haz</code> <code>hbb</code> <code>he</code> <code>hem</code> <code>hi</code> <code>hia</code> <code>hkk</code> <code>hla</code> <code>hno</code> <code>hoj</code> <code>hr</code> <code>hsb</code> <code>ht</code> <code>hu</code> <code>hue</code> <code>hul</code> <code>hux</code> <code>hwo</code> <code>hy</code> <code>hz</code> <code>ia</code> <code>ibb</code> <code>id</code> <code>ida</code> <code>idu</code> <code>ig</code> <code>ijc</code> <code>ijn</code> <code>ik</code> <code>ikw</code> <code>is</code> <code>ish</code> <code>iso</code> <code>it</code> <code>its</code> <code>itw</code> <code>itz</code> <code>ja</code> <code>jal</code> <code>jax</code> <code>jgo</code> <code>jmx</code> <code>jns</code> <code>jqr</code> <code>juk</code> <code>juo</code> <code>jv</code> <code>ka</code> <code>kab</code> <code>kai</code> <code>kaj</code> <code>kam</code> <code>kbd</code> <code>kbl</code> <code>kbt</code> <code>kcq</code> <code>kdh</code> <code>kea</code> <code>keu</code> <code>kfe</code> <code>kfk</code> <code>kfp</code> <code>khg</code> <code>khw</code> <code>kj</code> <code>kjc</code> <code>kjk</code> <code>kk</code> <code>kln</code> <code>kls</code> <code>km</code> <code>kmr</code> <code>kmy</code> <code>kn</code> <code>kna</code> <code>knn</code> <code>ko</code> <code>kol</code> <code>koo</code> <code>kpo</code> <code>kqo</code> <code>ks</code> <code>ksd</code> <code>ksf</code> <code>kto</code> <code>kuh</code> <code>kvx</code> <code>kw</code> <code>kwm</code> <code>kxp</code> <code>ky</code> <code>kyx</code> <code>lag</code> <code>lb</code> <code>lcm</code> <code>ldb</code> <code>lg</code> <code>lij</code> <code>lir</code> <code>lkb</code> <code>lla</code> <code>ln</code> <code>lnu</code> <code>lo</code> <code>loa</code> <code>lrk</code> <code>lss</code> <code>lt</code> <code>ltg</code> <code>lto</code> <code>lua</code> <code>luo</code> <code>lus</code> <code>lv</code> <code>lwg</code> <code>mab</code> <code>maf</code> <code>mai</code> <code>mau</code> <code>max</code> <code>mbo</code> <code>mcf</code> <code>mcn</code> <code>mcx</code> <code>mdd</code> <code>mde</code> <code>mdf</code> <code>mek</code> <code>mer</code> <code>meu</code> <code>mfm</code> <code>mfn</code> <code>mfo</code> <code>mfv</code> <code>mgg</code> <code>mgi</code> <code>mhk</code> <code>mhr</code> <code>mi</code> <code>mig</code> <code>miu</code> <code>mk</code> <code>mkf</code> <code>mki</code> <code>ml</code> <code>mlq</code> <code>mn</code> <code>mne</code> <code>mni</code> <code>mqy</code> <code>mr</code> <code>mrj</code> <code>mrr</code> <code>mrt</code> <code>ms</code> <code>mse</code> <code>msh</code> <code>msw</code> <code>mt</code> <code>mtr</code> <code>mtu</code> <code>mtx</code> <code>mua</code> <code>mug</code> <code>mui</code> <code>mve</code> <code>mvy</code> <code>mxs</code> <code>mxu</code> <code>mxy</code> <code>my</code> <code>myv</code> <code>mzl</code> <code>nal</code> <code>nan</code> <code>nap</code> <code>nb</code> <code>nbh</code> <code>ncf</code> <code>nco</code> <code>ncx</code> <code>ndi</code> <code>ng</code> <code>ngi</code> <code>nhg</code> <code>nhi</code> <code>nhn</code> <code>nhq</code> <code>nja</code> <code>nl</code> <code>nla</code> <code>nlv</code> <code>nmg</code> <code>nmz</code> <code>nn</code> <code>nnh</code> <code>no</code> <code>noe</code> <code>npi</code> <code>nso</code> <code>ny</code> <code>nyu</code> <code>oc</code> <code>odk</code> <code>odu</code> <code>ogo</code> <code>om</code> <code>orc</code> <code>oru</code> <code>ory</code> <code>os</code> <code>pa</code> <code>pbs</code> <code>pbt</code> <code>pbu</code> <code>pcm</code> <code>pex</code> <code>phl</code> <code>phr</code> <code>pip</code> <code>piy</code> <code>pko</code> <code>pl</code> <code>plk</code> <code>plt</code> <code>pmq</code> <code>pms</code> <code>pmy</code> <code>pnb</code> <code>poc</code> <code>poe</code> <code>pow</code> <code>prq</code> <code>ps</code> <code>pst</code> <code>pt</code> <code>pua</code> <code>pwn</code> <code>qug</code> <code>qum</code> <code>qup</code> <code>qur</code> <code>qus</code> <code>quv</code> <code>qux</code> <code>quy</code> <code>qva</code> <code>qvi</code> <code>qvj</code> <code>qvl</code> <code>qwa</code> <code>qws</code> <code>qxa</code> <code>qxp</code> <code>qxt</code> <code>qxu</code> <code>qxw</code> <code>rag</code> <code>rm</code> <code>ro</code> <code>rob</code> <code>rof</code> <code>roo</code> <code>rth</code> <code>ru</code> <code>rup</code> <code>rw</code> <code>sa</code> <code>sah</code> <code>sat</code> <code>sau</code> <code>say</code> <code>sbn</code> <code>sc</code> <code>scl</code> <code>scn</code> <code>sd</code> <code>sei</code> <code>shu</code> <code>si</code> <code>sip</code> <code>siw</code> <code>sjr</code> <code>sk</code> <code>skg</code> <code>skr</code> <code>sl</code> <code>sn</code> <code>snc</code> <code>snk</code> <code>so</code> <code>sol</code> <code>sps</code> <code>sq</code> <code>sr</code> <code>src</code> <code>sro</code> <code>ssi</code> <code>ste</code> <code>sua</code> <code>sv</code> <code>sva</code> <code>sw</code> <code>szy</code> <code>ta</code> <code>tan</code> <code>tar</code> <code>tay</code> <code>tbf</code> <code>tcf</code> <code>tcy</code> <code>tdn</code> <code>tdx</code> <code>te</code> <code>tg</code> <code>tgc</code> <code>th</code> <code>the</code> <code>thq</code> <code>thr</code> <code>thv</code> <code>ti</code> <code>tig</code> <code>tio</code> <code>tk</code> <code>tkg</code> <code>tkt</code> <code>tli</code> <code>tlp</code> <code>tn</code> <code>tok</code> <code>tpl</code> <code>tpz</code> <code>tqp</code> <code>tr</code> <code>trp</code> <code>trq</code> <code>trv</code> <code>trw</code> <code>tt</code> <code>ttj</code> <code>ttr</code> <code>ttu</code> <code>tui</code> <code>tul</code> <code>tuq</code> <code>tuv</code> <code>tuy</code> <code>tvo</code> <code>tvu</code> <code>tw</code> <code>twu</code> <code>txs</code> <code>txy</code> <code>udl</code> <code>ug</code> <code>uk</code> <code>uki</code> <code>umb</code> <code>ur</code> <code>ush</code> <code>uz</code> <code>uzn</code> <code>vai</code> <code>var</code> <code>ver</code> <code>vi</code> <code>vmc</code> <code>vmj</code> <code>vmm</code> <code>vmp</code> <code>vmz</code> <code>vot</code> <code>vro</code> <code>wbl</code> <code>wci</code> <code>weo</code> <code>wes</code> <code>wja</code> <code>wji</code> <code>wo</code> <code>wof</code> <code>xh</code> <code>xhe</code> <code>xka</code> <code>xmf</code> <code>xmv</code> <code>xmw</code> <code>xpe</code> <code>xti</code> <code>xtu</code> <code>yaq</code> <code>yav</code> <code>yay</code> <code>ydd</code> <code>ydg</code> <code>yer</code> <code>yes</code> <code>yi</code> <code>yo</code> <code>yue</code> <code>zga</code> <code>zgh</code> <code>zh</code> <code>zoc</code> <code>zoh</code> <code>zor</code> <code>zpv</code> <code>zpy</code> <code>ztg</code> <code>ztn</code> <code>ztp</code> <code>zts</code> <code>ztu</code> <code>zu</code> <code>zza</code></span></details></dd></div><div><dt>Capabilities</dt><dd><details class="vh-model-detail__capabilities"><summary>10 capabilities</summary><span><code>text-to-speech</code> <code>voice-cloning</code> <code>voice-design</code> <code>multilingual</code> <code>fine-tuning</code> <code>safetensors</code> <code>voicehub-native</code> <code>native-runtime</code> <code>raw-audio-fine-tuning</code> <code>preencoded-code-fine-tuning</code></span></details></dd></div><div><dt>Training</dt><dd><code>native</code></dd></div><div><dt>License</dt><dd>Checkpoint-specific</dd></div><div><dt>Default checkpoint</dt><dd id="vh-model-checkpoint-omnivoice"><a href="https://huggingface.co/k2-fsa/OmniVoice"><code>k2-fsa/OmniVoice</code></a></dd></div></dl></details></aside>

<div class="vh-model-detail__main vh-model-detail__content" markdown>

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
| Languages | `aae`, `aal`, `aao`, `ab`, … complete audited list below |
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

<section class="vh-model-api-card" data-vh-model-api-card="configuration" markdown>
<p class="vh-model-api-card__badge-wrap"><span class="vh-model-api-card__badge">Configuration</span></p>

### `OmniVoiceConfig`

<p class="vh-model-api-card__source-wrap"><a class="vh-model-api-card__source" href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/omnivoice/configuration_omnivoice.py">View source</a></p>
<div class="vh-model-api-card__signature" markdown>

```text
OmniVoiceConfig(**config_kwargs)
```

</div>
<h4>Parameters</h4>
<div class="vh-model-api-card__parameters" markdown>
- `**config_kwargs` — Configuration fields validated by OmniVoiceConfig.
</div>
</section>

<section class="vh-model-api-card" data-vh-model-api-card="model" markdown>
<p class="vh-model-api-card__badge-wrap"><span class="vh-model-api-card__badge">Model</span></p>

### `OmniVoiceForTextToSpeech`

<p class="vh-model-api-card__source-wrap"><a class="vh-model-api-card__source" href="https://github.com/kadirnar/voicehub/blob/main/voicehub/models/omnivoice/modeling_omnivoice.py">View source</a></p>
<div class="vh-model-api-card__signature" markdown>

```text
AutoModelForTextToSpeech.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='omnivoice',
    config=None,
    **model_kwargs,
)
```

</div>
<h4>Parameters</h4>
<div class="vh-model-api-card__parameters" markdown>
- `pretrained_model_name_or_path` — Hub ID or compatible local directory.
- `model_type` — Canonical model type; use 'omnivoice'.
- `config` — Optional preloaded OmniVoiceConfig instance.
- `**model_kwargs` — Model-specific loading arguments.
</div>
</section>

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

</div>

</div>

</div>
