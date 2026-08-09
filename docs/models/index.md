# TTS model catalog

For a step-by-step page with inference, the exact data contract, and training
support for each individual model, open the [model guides](providers/index.md).

This page covers text-to-speech backends. For automatic speech recognition
and voice activity detection, see the
[ASR and VAD support matrix](asr-vad-support.md).

For a compact comparison of model families, languages, voice controls, and
adaptation limits, see the
[TTS capabilities and adaptation boundaries](tts-capabilities.md).

VoiceHub ships model implementation source and every built-in inference
runtime dependency in its default installation. It never installs a separate
TTS implementation package. Model checkpoints are still downloaded lazily or
supplied as local paths.

## Choose a model

Start from the capability you need, then inspect the exact checkpoint and
conditioning contract before loading weights. Install the current VoiceHub
source first.

| Model type | Good fit |
| --- | --- |
| `orpheustts` | Expressive speech |
| `dia` | Multi-speaker dialogue |
| `vui` | Compact native 100M/Fluac synthesis |
| `chatterbox` | Voice cloning |
| `kokoro` | Lightweight multilingual speech |
| `echo` | Reference-conditioned cloning |
| `conversationtts` | Multilingual conversation |
| `llasa` | Multilingual codec-LM cloning |
| `cosyvoice` | Cloning, multilingual speech, streaming |
| `f5tts` | Flow-matching voice cloning |
| `gptsovits` | V1, V2, V2Pro, and V2ProPlus multilingual cloning |
| `melotts` | Fast multilingual synthesis |
| `openvoice` | Cross-lingual voice transfer |
| `outetts` | Native V3 speaker-profile-conditioned codec-LM synthesis |
| `parlertts` | Natural-language style control |
| `styletts2` | Style diffusion and voice cloning |
| `mosstts` | Native Delay, Local, Local v1.5, and Realtime cloning |
| `qwen3tts` | Custom voices, design, and cloning |
| `irodoritts` | Reference and caption conditioning |
| `zonos` | Native multilingual voice cloning |
| `zonos2` | Batched mixture-of-experts synthesis |
| `voxcpm` | Native voice design, cloning, and continuation |
| `omnivoice` | Multilingual cloning and voice design |
| `higgstts` | Expressive long-form generation |
| `xtts` | Multilingual voice cloning |
| `vibevoice` | Native staged realtime graph and non-streaming 1.5B fine-tuning |
| `fishtts` | Multilingual semantic-token cloning |
| `csm` | Native conversational inference and full-model fine-tuning |
| `neutts` | Compact local and multilingual variants |
| `supertonic` | Native multilingual flow inference and prepared-latent FT |
| `inflecttts` | Compact local synthesis |
| `bark` | Expressive prompt- and preset-conditioned generation |
| `speecht5` | Native speaker-embedding-conditioned synthesis |
| `vits` | VITS and 1,100+ MMS-TTS language checkpoints |

Training capability is checkpoint-aware. Check the
[training support matrix](training-support.md) before selecting an artifact or
designing a dataset.

```python
from voicehub import AutoInferenceModel, AutoModelForTextToSpeech

model = AutoModelForTextToSpeech.from_pretrained(
    "parler-tts/parler-tts-mini-v1",
    model_type="parlertts",
    device="cuda",
)
output = model(
    "VoiceHub uses one model lifecycle.",
    description="A warm speaker in a clean studio.",
    output_file="speech.wav",
)
print(output.sample_rate, output.file_path)
```

`AutoInferenceModel` remains as a compatible factory. Construction is lazy;
call `model.load()` once during service startup to warm the checkpoint.

## Source status

| Model type | Implementation used by VoiceHub | Source status |
|---|---|---|
| `orpheustts` | VoiceHub generation code + vendored SNAC | Included |
| `dia` | VoiceHub-native Dia graph + native DAC | Included |
| `vui` | VoiceHub-native pinned Vui 100M and frozen Fluac graph | Pinned MIT source/checkpoints; strict standalone model-plus-codec Safetensors export and no external model runtime |
| `chatterbox` | VoiceHub-native T3, S3Gen, S3Tokenizer, voice encoder, audio frontend, and Perth watermark | Pinned MIT source/checkpoint; dependency-free inference plus raw-audio full/LoRA fine-tuning |
| `kokoro` | VoiceHub-native PL-BERT, prosody, text encoder, iSTFTNet, checkpoint conversion, and phoneme frontend | Pinned Apache-2.0 source/checkpoint; no external model runtime |
| `echo` | VoiceHub Echo-TTS source | Included |
| `conversationtts` | VoiceHub-native Llama 3.2 conversational codec LM + bundled MimiCodec | Pinned CC BY-NC 4.0 source/checkpoint; restricted one-time legacy load and strict Safetensors export |
| `llasa` | VoiceHub-native Llama 3.2 codec LM, tokenizer, and XCodec2 graph | Pinned CC BY-NC 4.0 Safetensors; no external model runtime |
| `cosyvoice` | VoiceHub-native CosyVoice 3 LM, S3 speech-tokenizer encoder/FSQ, flow matcher, HiFT generator/discriminator, and byte-BPE tokenizer | Pinned Apache-2.0 sources/checkpoint; audited one-time conversion and dependency-free strict Safetensors runtime |
| `f5tts` | F5-TTS v1 DiT + Vocos | VoiceHub-native; vendored tree retained only as an audited reference |
| `gptsovits` | VoiceHub-native GPT-SoVITS V1/V2/V2Pro/V2ProPlus S1 semantic and classic-S2 VITS/GAN graphs | Pinned MIT source and 12 exact component inventories; restricted legacy import and variant-aware staged Safetensors export |
| `melotts` | VoiceHub-native multilingual VITS2 generator, MPD, and duration discriminator | Pinned MIT source/releases; restricted legacy conversion and strict Safetensors runtime/export |
| `openvoice` | VoiceHub-native OpenVoice V2 converter + optional native MeloTTS base | Pinned MIT source/checkpoint; exact converter, native STFT/reference encoder, restricted legacy import, strict Safetensors export, and explicit reconstructed paired-waveform FT |
| `outetts` | VoiceHub-native Llama/Qwen codec LM, byte-BPE tokenizer, V3 prompt protocol, and shared DAC | Strict pinned Safetensors; native regular/chunked generation and prepared-profile fine-tuning |
| `parlertts` | VoiceHub-native Parler decoder + FLAN-T5 + shared DAC | Pinned Apache-2.0 source/checkpoint; no external model runtime |
| `styletts2` | VoiceHub-native StyleTTS 2 diffusion, PL-BERT, HiFi-GAN/iSTFTNet, and training objectives | Pinned MIT source; strict Safetensors runtime/export and explicit restricted legacy import |
| `mosstts` | VoiceHub-native Delay, Local, Local v1.5, and Realtime semantic graphs, Qwen byte-BPE tokenizer, and MOSS Audio Tokenizer v1/v2 | Pinned Apache-2.0 source and seven immutable Safetensors repositories; raw-audio or pre-encoded full semantic-model fine-tuning |
| `qwen3tts` | VoiceHub-native Qwen3 talker, residual predictor, speaker encoder, and complete Mimi-derived speech tokenizer encoder/quantizer/decoder | Pinned Apache-2.0 source/checkpoints; raw-reference ICL and no Transformers runtime |
| `irodoritts` | VoiceHub-native Irodori RF-DiT, duration predictor, tokenizer, and frozen Semantic-DACVAE | Pinned MIT source/checkpoints; raw-audio flow/duration FT and strict Safetensors export |
| `zonos` | VoiceHub-native Zonos v0.1 dense Transformer, conditioning, and shared DAC | Pinned Apache-2.0 source/checkpoints; strict Safetensors runtime and export |
| `zonos2` | VoiceHub-native ZONOS2 dense/MoE graph, speaker encoder, and shared DAC | Pinned source/checkpoints; no fused provider runtime |
| `voxcpm` | VoiceHub-native VoxCPM2 language/flow graph, tokenizer, and AudioVAE V2 | Pinned Apache-2.0 source/checkpoint; strict Safetensors runtime and export |
| `omnivoice` | VoiceHub-native bidirectional Qwen3 graph, tokenizer, generation loop, and frozen Higgs Audio v2 codec | Pinned Apache-2.0 source/checkpoints; raw-audio full FT and strict Safetensors export |
| `higgstts` | VoiceHub-native Higgs Audio v2 dual-FFN decoder, tokenizer, and frozen native audio tokenizer | Apache-2.0 source, custom-license checkpoint; raw-audio full SFT and strict Safetensors export |
| `xtts` | VoiceHub-native XTTS v2 GPT, tokenizer, conditioning, speaker encoder, HiFi-GAN decoder, and separate full DVAE encoder/codebook/decoder | Pinned MPL-2.0 source/CPML checkpoint; explicit legacy conversion, strict Safetensors runtime, and [raw-waveform or precomputed-code GPT FT](xtts2.md) |
| `vibevoice` | VoiceHub-native ASR, 1.5B TTS, and realtime-stage graphs | Pinned MIT source/checkpoints; strict Safetensors loading and export |
| `fishtts` | VoiceHub-native Fish Speech S2 DualAR + ModifiedDAC | Pinned Fish Audio Research License source/checkpoint; strict semantic Safetensors, explicit digest-gated codec conversion, and native full semantic FT |
| `csm` | VoiceHub-native Sesame CSM + frozen native Mimi | Pinned Apache-2.0/CC-BY-4.0 Safetensors; explicit SilentCipher boundary |
| `neutts` | VoiceHub-native Qwen/Llama NeuTTS, byte-BPE tokenizers, and NeuCodec | Strict pinned Safetensors; Air is Apache-2.0, other checkpoints use the NeuTTS Open License |
| `supertonic` | Supertonic 3 reviewed graph | VoiceHub-native PyTorch |
| `inflecttts` | VoiceHub-native Inflect Micro/Nano v2 generator, posterior, and MPD | Pinned Apache-2.0 source/checkpoints; strict trusted conversion and Safetensors export |
| `bark` | VoiceHub-native semantic/coarse/fine Transformers and shared Encodec | Pinned MIT source/checkpoint; explicit restricted legacy conversion and Safetensors runtime |
| `speecht5` | VoiceHub-native SpeechT5, SentencePiece/log-mel processor, and HiFi-GAN | Pinned MIT checkpoints and Apache-2.0 source; no Transformers runtime |
| `vits` | VoiceHub-native VITS/MMS-TTS graph and declarative frontend | External Safetensors checkpoint |
| `echo` | VoiceHub-native rectified-flow DiT, Fish codec, and samplers | External non-commercial Safetensors checkpoints |

Each vendored directory contains `SOURCE.json` and `THIRD_PARTY_LICENSE`.
`scripts/vendor_tts_sources.py` rebuilds deterministic snapshots from pinned
upstream revisions. Pretrained weights are not copied except for Perth's
small runtime watermark checkpoint.

## Current-generation families

The current backends keep checkpoint variants behind one architecture key:

| Backend | Supported family variants |
|---|---|
| `kokoro` | Kokoro-82M native Safetensors runtime; released `.pth` and voice packs cross a restricted one-time `weights_only=True` conversion boundary |
| `gptsovits` | V1, V2, V2Pro, and V2ProPlus classic-S2 checkpoints. V3, V4, and LoRA fail closed because they require distinct flow-matching/vocoder or PEFT-merge graphs. |
| `mosstts` | Official MOSS-TTS/MOSS-TTS-v1.5 Delay, Local Transformer, Local Transformer v1.5, and Realtime checkpoints. Realtime uses the published buffered generation schedule; incremental queue/transport streaming is not claimed. |
| `qwen3tts` | 0.6B/1.7B Base, CustomVoice, VoiceDesign, x-vector voice cloning, and native raw-reference ICL through the published Mimi-derived speech encoder |
| `irodoritts` | v2, v3, and VoiceDesign-compatible checkpoints |
| `zonos` | Exact Zonos v0.1 dense Transformer; the structurally distinct Mamba-2 hybrid checkpoint is rejected until its graph is implemented |
| `zonos2` | ZONOS2 dense and mixture-of-experts checkpoints |
| `voxcpm` | Exact VoxCPM2 577-tensor language/flow graph and 312-tensor AudioVAE V2 codec; legacy VoxCPM1 and streaming are outside the verified contract |
| `higgstts` | Higgs Audio v2/v2.5 source architecture |
| `outetts` | Llama 1B and Qwen3 0.6B native backbones; the default Llama checkpoint is CC-BY-NC-SA-4.0, while the Qwen3 checkpoint is Apache-2.0 |
| `neutts` | Air, Nano, multilingual Nano, and 2E native backbones; author-verified fine-tuning is Air-only |
| `inflecttts` | Inflect Micro v2 and Nano v2 |
| `bark` | Bark small/large checkpoints and semantic, coarse, and fine token stages |
| `speecht5` | SpeechT5 TTS checkpoints with configurable HiFi-GAN and speaker embeddings |
| `vits` | VITS-compatible checkpoints, including Meta's multilingual MMS-TTS collection |

Model weights, cached voice prompts, preset speaker embeddings, and ONNX
graphs are not embedded in the wheel. They are resolved from a checkpoint
repository or accepted as local paths.

MOSS example:

```python
model = AutoInferenceModel.from_pretrained(
    "moss-tts",
    model_path="OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5",
    variant="local_v1_5",
    device="cuda",
)
output = model(
    "Source-integrated speech generation.",
    speaker_audio_path="reference.wav",
    output_file="moss.wav",
)
```

Qwen3-TTS voice design:

```python
model = AutoInferenceModel.from_pretrained(
    "qwen3-tts",
    model_path="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    device="cuda",
)
output = model(
    "Merhaba, VoiceHub'a hoş geldiniz.",
    mode="voice_design",
    language="Turkish",
    instruct="A calm, confident adult speaker.",
)
```

## Training-safe runtime selection

Fine-tuning starts through `load_for_training()`, which can select a different
construction path from inference and reject an incompatible backend:

- Dia inference and fine-tuning use VoiceHub's native
  `DiaForConditionalGeneration`, byte tokenizer, delay processor, and DAC.
  The converted `Dia-1.6B-0626` Safetensors namespace is loaded strictly. The
  incompatible original `Dia-1.6B` pickle/JAX layout is rejected with a
  migration message.
- Sesame CSM inference and fine-tuning use VoiceHub's checkpoint-compatible
  PyTorch graph, byte-BPE tokenizer, processor, generation loop, and two-level
  backbone/depth-decoder objective. Raw audio is encoded by the separately
  pinned, frozen native Mimi graph; pre-encoded Mimi codes skip that step.
  Watermarking is reported only when an explicitly supplied postprocessor
  declares that it watermarks audio.
- OmniVoice keeps `torch_dtype="float16"` as its inference default but uses
  `training_torch_dtype="float32"` by default for training. Its registered
  collator schema treats codebook tensors as codebook-first and time-last.
- VoxCPM2 uses VoiceHub's native conditional-flow-matching and stop-token
  objectives. Full-model SFT and the published LoRA target topology are
  supported while AudioVAE V2 remains frozen. Raw audio enters the codec at
  16 kHz, synthesized output is 48 kHz, and merged exports reload through the
  ordinary strict Safetensors path.
- OuteTTS uses the native Llama/Qwen causal graph, byte-BPE tokenizer, V3
  prompt protocol, and frozen DAC for both inference and full-LM fine-tuning.
  Prepared V3 profiles or exact token labels are required; raw audio,
  quantized settings, and external provider backends fail closed.
- NeuTTS uses VoiceHub's native causal LM, byte-BPE tokenizer, and NeuCodec.
  The Air Safetensors checkpoint supports raw-audio or preencoded completion-
  only fine-tuning with frozen NeuCodec. Nano and 2E training, GGUF, ONNX,
  pickle, and distilled-codec artifacts fail closed.
- Qwen3-TTS fine-tuning starts from
  `Qwen/Qwen3-TTS-12Hz-1.7B-Base`, exposed as
  `model.training_default_model_name_or_path`. CustomVoice and VoiceDesign
  checkpoints are inference/export targets for this recipe.
- MOSS-TTS exposes one native semantic-language-model phase for Delay, Local,
  Local v1.5, and Realtime. Records may contain text plus raw audio (with
  sampling-rate metadata for in-memory tensors) or a pre-encoded
  `[frames, n_vq]` `speech_tokens` matrix. MOSS Audio Tokenizer v1 supplies the
  24 kHz targets and v2 supplies the 48 kHz stereo Local v1.5 targets; both
  codecs are VoiceHub-owned and remain frozen while the complete semantic
  graph receives the release-specific multichannel cross-entropy. Native
  exports are strict, fresh-inference Safetensors artifacts.

Safetensors are weight containers, not proof of a trainable graph. When a
model repository also publishes GGUF or another serving artifact, select the
compatible unquantized native Safetensors checkpoint. Exact training resume
uses a VoiceHub checkpoint because optimizer, scheduler, RNG, sampler, and
recipe state are not present in a safetensors weight export.

The exact model-by-model boundary is maintained in the
[training model matrix](training-support.md).

The [current model research](../project/model-audit.md) records the dated Hugging
Face audit, download/trending signals, upstream source, licensing, and
source-only inclusion decisions.

LLaSA uses VoiceHub's native Llama 3.2, byte-BPE tokenizer, and complete
XCodec2 graph instead of the `transformers` or `xcodec2` packages. Both
checkpoint families use strict Safetensors inventories. The LLaSA/XCodec2
weights are licensed CC BY-NC 4.0, so review the non-commercial restriction
before selecting this backend:

```python
model = AutoInferenceModel.from_pretrained("llasa")
output = model(
    "VoiceHub decodes LLaSA speech tokens locally.",
    speaker_audio_path="reference.wav",
    reference_text="Transcript of the reference.",
    output_file="llasa.wav",
)
```

## Voice cloning models

F5-TTS:

```python
from voicehub import AutoInferenceModel

model = AutoInferenceModel.from_pretrained("f5tts")
output = model(
    "Flow matching makes speech generation sound natural.",
    speaker_audio_path="reference.wav",
    reference_text="Transcript of the reference audio.",
    output_file="f5.wav",
)
```

The production F5-TTS path imports neither the vendored F5 package nor
Transformers, Diffusers, Torchaudio, Librosa, torchdiffeq, Vocos, or
Safetensors. VoiceHub implements the released DiT/flow graph, log-mel frontend,
Euler and midpoint integration, Vocos decoder, Hub transport, and Safetensors
I/O with PyTorch and the standard library. The 1.35 GB flow checkpoint and
54 MB vocoder remain separate, pinned artifacts with separate licenses.
`reference_text` is required: run ASR explicitly if it is unavailable.

CosyVoice:

```python
import torch

model = AutoInferenceModel.from_pretrained("cosyvoice")
output = model(
    "Voice cloning works from a short reference.",
    speaker_embedding=torch.load(
        "speaker_embedding.pt",
        map_location="cpu",
        weights_only=True,
    ),
    prompt_speech_tokens=torch.load(
        "prompt_speech_tokens.pt",
        map_location="cpu",
        weights_only=True,
    ),
    output_file="cosyvoice.wav",
)
```

The native serving graph intentionally does not execute CAMPPlus or
`speech_tokenizer_v3`. Prepare those frozen prompt tensors offline and keep
their revisions with the dataset or deployment artifact.

OpenVoice:

```python
from voicehub import AutoInferenceModel

model = AutoInferenceModel.from_pretrained(
    "openvoice",
    model_path="myshell-ai/OpenVoiceV2",
    trust_pickle_checkpoint=True,  # One-time import of the hash-pinned release.
)
output = model(
    "The text is metadata when an explicit base waveform is supplied.",
    base_audio="source.wav",
    speaker_audio_path="reference.wav",
    output_file="openvoice.wav",
)
```

OpenVoice V2 is a tone-color converter, not a text frontend. Pass an existing
22.05 kHz base waveform as above, or configure a native MeloTTS checkpoint and
supply its exact phone, tone, language, BERT, and Japanese-BERT features.
VoiceHub does not hide an ASR, phonemizer, Silero VAD, or watermarking runtime
inside conversion. After the one-time, digest-checked `weights_only=True`
import, export the converter and load its `model.safetensors` directory with
`trust_pickle_checkpoint=False`.

GPT-SoVITS V1, V2, V2Pro, and V2ProPlus use VoiceHub's native staged runtime.
The normal artifact is a directory containing the variant-tagged S1, S2
generator, optional S2 discriminator, configuration, and integrity manifest.
Inference consumes the exact prepared features for the selected family;
VoiceHub does not substitute another multilingual phonemizer, Chinese-RoBERTa,
CN-HuBERT, or speaker-verification implementation:

```python
model = AutoInferenceModel.from_pretrained(
    "gpt-sovits",
    model_path="/models/gpt-sovits-v2-native",
    variant="v2",
)
prepared = prepared_dataset_row  # Produced by the pinned V2 data pipeline.
output = model(
    prepared["text"],
    s1_phoneme_ids=prepared["s1_phoneme_ids"],
    s1_bert_features=prepared["s1_bert_features"],
    s2_phoneme_ids=prepared["s2_phoneme_ids"],
    prompt_semantic_ids=prepared["prompt_semantic_ids"],
    reference_spectrogram=prepared["reference_spectrogram"],
    output_file="gpt-sovits.wav",
)
```

For `variant="v2Pro"` or `"v2ProPlus"`, also pass the prepared 20,480-D
ERes2NetV2 `speaker_embedding`. The official PyTorch archives are accepted
only through the explicit revision- and digest-pinned, `weights_only=True`
conversion boundary with exact inventory validation. V3, V4, and LoRA remain
rejected because they require the separately structured flow-matching,
vocoder, or PEFT-merge graphs.

StyleTTS 2 runs through VoiceHub's native graph. A normal artifact directory
contains `config.json` and `model.safetensors`; no provider runtime is
imported. The released English frontend requires explicit phonemes or
checkpoint-compatible token IDs instead of silently substituting a different
G2P implementation:

```python
model = AutoInferenceModel.from_pretrained(
    "style-tts2",
    model_path="/models/styletts2-native",
)
output = model(
    "staɪl dɪfjuːʒən kəntrəʊlz tɪmbə ənd prɒsədi",
    text_is_phonemes=True,
    speaker_audio_path="reference.wav",
    output_file="styletts2.wav",
)
```

Importing a reviewed official `.pth` file is a one-time compatibility action
and requires `trust_pickle_checkpoint=True`. Save the loaded model immediately
to obtain the strict Safetensors layout used for subsequent inference and
fine-tuning.

Only clone a voice with the speaker's permission and follow the selected
checkpoint's license and disclosure requirements.

Fish Speech and every fine-tuned derivative are governed by the Fish Audio
Research License. Commercial use requires a separate written license.
Distribution must include the full license, retain the required copyright
notice, and prominently display: **Built with Fish Audio**. The license also
restricts using Fish materials, derivatives, or outputs to create or improve
non-Fish foundational generative-AI models.

## ConversationTTS

ConversationTTS revision `b3851f7` declares its source, checkpoints, datasets,
and evaluation tools under CC BY-NC 4.0. VoiceHub therefore includes its
executable model, inference, text-tokenizer, and MimiCodec runtime source. The
license does not permit commercial use:

```python
model = AutoInferenceModel.from_pretrained(
    "conversationtts",
    model_path="AudioFoundation/SpeechFoundation",
    device="cuda",
)
output = model(
    "A source-integrated conversational model.",
    speaker_audio_path="reference.wav",
    reference_text="Transcript of the reference speaker.",
    output_file="conversation.wav",
)
```

The main checkpoint and Mimi tokenizer weights remain external, revision-pinned
Hub artifacts. VoiceHub validates the complete model namespace and tensor
shapes. The published 9.3 GB PyTorch archive is read only with
`weights_only=True`; fine-tuned and converted artifacts use Safetensors.

Training accepts either raw text plus 24 kHz PCM audio or precomputed text IDs
plus 32-codebook Mimi codes. VoiceHub constructs the released 33-stream
text-then-audio layout, preserves the source padding and EOS IDs, freezes Mimi,
and evaluates the published codebook-zero and residual losses. Serving KV
caches are removed before training and excluded from export, so the same model
instance can move between fine-tuning and inference without changing parameter
identity.
