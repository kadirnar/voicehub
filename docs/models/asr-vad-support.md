---
description: Compare VoiceHub ASR and VAD providers, architecture families, default runtime coverage, outputs, and fine-tuning boundaries.
---

# ASR and VAD support

For runnable inference, model-specific data steps, and the verified training
path for each provider, open the [model guides](providers/index.md).

VoiceHub covers speech-input models through provider and architecture families.
One registry entry can load many compatible checkpoints; the table is
therefore a runtime coverage map, not a finite list of every model repository
that may work.

Each integration provides:

- lazy discovery through a task-aware `ModelSpec`;
- a task-specific auto-model factory;
- file, array, tensor, mapping, and `AudioInput` ingestion;
- normalized `ASROutput` or `VADOutput`;
- serializable inference configuration;
- portable VoiceHub configuration and processor metadata; and
- an explicit fine-tuning boundary.

## ASR providers

| Model type | Provider and family coverage | Normalized capability | Runtime setup | Fine-tuning boundary |
| --- | --- | --- | --- | --- |
| `asr_transformers` | Closed VoiceHub-native dispatcher for verified Whisper, Wav2Vec2 CTC, HuBERT CTC, WavLM CTC, and Moonshine checkpoints | Family-native text, segment/timestamp, and language fields | Default install; Safetensors only | **VoiceHub native** inference, loss/backward, export, and reload |
| `asr_whisper` | VoiceHub-native Whisper graph, tokenizer, log-mel frontend, generation engine, and strict Safetensors adapter, defaulting to `openai/whisper-large-v3-turbo` | Multilingual transcription, translation, and timestamps | Default install; Safetensors only | **VoiceHub native** teacher-forced Whisper fine-tuning |
| `asr_tiron` | VoiceHub-native Tiron on the internal Whisper graph with the pinned public speaker-token layout and ported `speaker_blocks` constraint grammar | Per-window speaker-attributed, 20 ms timestamped segments; padded vocabulary rows are masked | Default install; Safetensors only | **VoiceHub native** grammar-validated sequence-to-sequence fine-tuning; whole-meeting speaker linking is outside the model graph |
| `asr_qwen3` | VoiceHub-native Qwen3-ASR 0.6B/1.7B audio tower, Qwen3 decoder, byte-BPE tokenizer, log-mel frontend, cached generation, and strict Safetensors adapter | Multilingual transcription, validated language forcing, language identification, hotword context, and 20-minute energy-aware chunking; word timing requires the separate Qwen forced aligner | Default install; official immutable Safetensors revisions | **VoiceHub native** completion-only causal labels, full backward, native LoRA, and merged portable export |
| `asr_vibevoice` | VoiceHub-native continuous acoustic/semantic encoders, multimodal projector, Qwen2 decoder, byte-BPE tokenizer, prompt renderer, and strict eight-shard Safetensors adapter | Long-form multilingual text plus parsed speaker/timestamp segments and prompt or terminology context; an optional language is context, not a forced decoder token | Default install; pinned MIT checkpoint | **VoiceHub native** completion-only causal fine-tuning with frozen speech encoders, trainable projector/LM/head, and portable export |
| `asr_granite_speech` | VoiceHub-native Granite Speech 4.1 Conformer, windowed Q-Former, Granite decoder, byte-BPE tokenizer, HTK log-mel frontend, generation cache, and strict sharded-Safetensors adapter | Multilingual prompt-conditioned transcription, speech translation, and keyword biasing | Default install; official immutable Safetensors revision | **VoiceHub native** IBM-style prompt/target concatenation with completion-only labels, source-compatible projector plus native-LoRA optimization, full-graph backward, and merged portable export |
| `asr_parakeet_tdt` | VoiceHub-native Parakeet TDT 0.6B v3 FastConformer, LSTM predictor, duration joint head, log-mel frontend, tokenizer, greedy decoder, and strict Safetensors adapter | Multilingual whole-waveform transcription and duration-derived word timestamps; language is automatic | Default install; pinned CC-BY-4.0 checkpoint | **VoiceHub native** raw-audio preparation, exact TDT loss, full-graph backward, gradient checkpointing, and portable export |
| `asr_nemotron` | VoiceHub-native Nemotron 3.5 FastConformer, prompt projector, LSTM predictor, RNN-T joint, log-mel frontend, tokenizer, and strict Safetensors adapter | Multilingual transcription, automatic locale tags, and token timestamps; the graph has cache-aware chunk generation, while common VoiceHub sessions remain buffered | Default install; pinned OpenMDW-1.1 checkpoint | **VoiceHub native** raw-audio preparation, exact differentiable RNN-T loss, full-graph backward, gradient checkpointing, and portable export |
| `asr_cohere` | VoiceHub-native Cohere Transcribe 03-2026 48-layer FastConformer, cross-attention decoder, log-mel frontend, byte-fallback BPE tokenizer, prompt builder, and strict Safetensors adapter | Explicitly language-conditioned transcription for ar, de, el, en, es, fr, it, ja, ko, nl, pl, pt, vi, and zh; offline quiet-boundary long-form reassembly | Default install plus gated checkpoint access/token; exact immutable Apache-2.0 checkpoint revision | **VoiceHub native** prompt-conditioned teacher forcing, full-graph backward, gradient checkpointing, and portable export/reload |
| `asr_medasr` | VoiceHub-native 17-layer LASR/Conformer CTC graph, log-mel frontend, Unigram tokenizer, greedy decoder, and strict gated Safetensors adapter | English complete-waveform medical dictation; timestamps and non-English forcing are rejected | Default install plus Health AI Developer Foundations terms/token | **VoiceHub native** raw-audio full-model CTC fine-tuning, gradient checkpointing, and portable export/reload |
| `asr_wav2vec2` | VoiceHub-native Wav2Vec2 CTC graph and declarative processor | Text and token timestamps when the tokenizer exposes offsets | Default install | **VoiceHub native** raw-waveform CTC fine-tuning |
| `asr_hubert` | VoiceHub-native HuBERT CTC graph with exact `hubert.*` checkpoint mapping and stable-layer-norm support | Text and greedy CTC word timestamps | Default install; Safetensors only | **VoiceHub native** raw-waveform CTC fine-tuning with learned SpecAugment mask embedding |
| `asr_wavlm` | VoiceHub-native WavLM CTC graph with gated bucketed relative-position attention and an exact `wavlm.*` checkpoint mapping | Text and greedy CTC word timestamps | Default install; Safetensors only | **VoiceHub native** raw-waveform CTC fine-tuning |
| `asr_moonshine` | VoiceHub-native Useful Sensors Moonshine tiny/base graph with exact raw-waveform frontend, rotary attention, tied projection, and SentencePiece BPE checkpoint mapping | Compact English greedy transcription; timestamp, hotword, sampled, and beam modes are rejected | Default install; Safetensors only | **VoiceHub native** raw-waveform teacher-forced sequence-to-sequence fine-tuning |
| `asr_seamless_m4t_v2` | VoiceHub-native SeamlessM4T-v2 S2T: stacked Kaldi-style frontend, 24-layer Conformer, 24-layer decoder, SentencePiece BPE, and strict sharded-Safetensors adapter | Multilingual complete-waveform recognition with output-language prompts afr, amh, arb, ary, arz, asm, azj, bel, ben, bos, bul, cat, ceb, ces, ckb, cmn, cmn_Hant, cym, dan, deu, ell, eng, est, eus, fin, fra, fuv, gaz, gle, glg, guj, heb, hin, hrv, hun, hye, ibo, ind, isl, ita, jav, jpn, kan, kat, kaz, khk, khm, kir, kor, lao, lit, lug, luo, lvs, mai, mal, mar, mkd, mlt, mni, mya, nld, nno, nob, npi, nya, ory, pan, pbt, pes, pol, por, ron, rus, sat, slk, slv, sna, snd, som, spa, srp, swe, swh, tam, tel, tgk, tgl, tha, tur, ukr, urd, uzn, vie, yor, yue, zlm, and zul; greedy text only | Default install; pinned CC-BY-NC-4.0 checkpoint | **VoiceHub native** target-language teacher forcing, full-model backward, gradient checkpointing, and portable S2T-only export |
| `asr_faster_whisper` | Compatibility key backed by VoiceHub-native Whisper; CTranslate2 model names are normalized to their canonical Safetensors source | Multilingual text and timestamps | Default install; no CTranslate2 runtime | **VoiceHub native** teacher-forced Whisper fine-tuning and portable export |
| `asr_whisperx` | VoiceHub-native Whisper plus language-specific native Wav2Vec2 CTC forced alignment, following pinned WhisperX trellis semantics | Multilingual transcription and aligned word intervals | Default install; Safetensors only; no upstream WhisperX runtime | **VoiceHub native** Whisper fine-tuning; the independent alignment checkpoint is fine-tuned through `asr_wav2vec2` |
| `asr_openai_whisper` | Compatibility key backed by VoiceHub-native Whisper; official OpenAI aliases resolve to canonical Safetensors checkpoints | Multilingual text and timestamps | Default install; no `openai-whisper` runtime | **VoiceHub native** teacher-forced Whisper fine-tuning and portable export |
| `asr_nemo` | VoiceHub-native NVIDIA QuartzNet15x5 character-CTC graph | English text and greedy CTC word timestamps; buffered/offline | Default install; the hash-pinned NGC `.nemo` release is converted once to Safetensors under NVIDIA NGC Terms | **VoiceHub native** raw-waveform CTC fine-tuning with the released log-mel frontend and rectangular spectrogram cutout |
| `asr_speechbrain` | VoiceHub-native LibriSpeech CRDNN: legacy Fbank/global CMVN, two VGG blocks, four-layer bidirectional LSTM, location-aware GRU decoder, and frozen two-layer RNNLM | English text and beam score; buffered/offline | Default install; the three hash-pinned upstream pickle states require explicit `trust_pickle_checkpoint=True` once, then inference/training/export use Safetensors | **VoiceHub native** raw-waveform fine-tuning with label-smoothed attention NLL, CTC for epochs 1–5, released Adadelta settings, validation-WER NewBob scheduling, and portable export |
| `asr_funasr` | VoiceHub-native SenseVoiceSmall SANM-CTC graph | Multilingual text, language ID, emotion, audio events, inverse text normalization controls, and native CTC word timestamps | Default install; Safetensors, or one explicit trust-gated conversion of the hash-pinned release pickle | **VoiceHub native** raw-audio CTC plus four-query rich-control fine-tuning, released AdamW/WarmupLR settings, and portable Safetensors export. Paraformer and other FunASR registry graphs are rejected. |
| `asr_espnet` | VoiceHub-native LibriSpeech Transformer-e18 frontend, Transformer/CTC graph, RNNLM, tokenizer, and joint beam search | English text and beam score; buffered/offline | Default install; no ESPnet or model-zoo package. The exact hash-pinned release pickle requires explicit trust for one-time restricted conversion to Safetensors. | **VoiceHub native** raw-waveform or prepared-feature hybrid 0.3 CTC / 0.7 label-smoothed sequence fine-tuning, published SpecAugment, Adam/WarmupLR settings, and portable export |
| `asr_wenet` | VoiceHub-native 20210728 GigaSpeech U2++ Conformer (conv2d6, relative attention, macaron FFN, causal convolution, dual Transformer decoder) | English text, CTC prefix search, bidirectional attention rescoring, confidence, and word timestamps; buffered/offline | Default install; the exact hash-pinned WeNet pickle archive requires explicit `trust_pickle_checkpoint=True` for one-time restricted conversion to Safetensors | **VoiceHub native** raw-waveform hybrid training with 0.3 CTC weight, 0.3 reverse-decoder weight, 0.1 label smoothing, Kaldi fbank/CMVN, and the released SpecAugment recipe |

The `asr_transformers` provider uses `architecture_family="auto"` by default.
Specify `ctc` or `speech-seq2seq` only when checkpoint metadata cannot
identify its native graph. Qwen3-ASR, VibeVoice-ASR,
Nemotron 3.5, Cohere Transcribe, Granite Speech, and Voxtral Realtime are
deliberately rejected by this generic path because they require
model-specific processor, prompt, or label semantics.

## VAD providers

| Model type | Provider and family coverage | Normalized capability | Runtime setup | Fine-tuning boundary |
| --- | --- | --- | --- | --- |
| `vad_transformers` | Compatibility key for verified VoiceHub-native Wav2Vec2 audio- and frame-classification checkpoints | Speech regions plus real frame/window probabilities when requested | Default install; no Transformers runtime | **VoiceHub native** classification path for differentiable checkpoints |
| `vad_silero` | VoiceHub-native Silero v6.2.1 graph; official JIT is accepted only for strict one-time weight conversion | Real frame probabilities and speech regions; 8/16 kHz; isolated recurrent streaming sessions | Default install | **VoiceHub native** frame BCE recipe with official timestamp records or aligned frame labels; decoder-only by default |
| `vad_webrtc` | VoiceHub-native port of the pinned WebRTC six-band fixed-point GMM | Binary frame decisions normalized to regions; 8/16/32/48 kHz; isolated adaptive state | Default install; no WebRTC package or compiled extension | **Not applicable** fixed algorithm with no differentiable parameters |
| `vad_pyannote` | VoiceHub-native PyanNet with the published pyannote segmentation weights | Segmentation regions and frame scores | Default install; no pyannote runtime | **VoiceHub native** multi-label frame fine-tuning |
| `vad_speechbrain` | VoiceHub-native SpeechBrain CRDNN VAD with the pinned LibriParty weights | Real 10 ms frame probabilities and source-compatible offline chunk segmentation | Default install; no SpeechBrain, torchaudio, HyperPyYAML, or Transformers runtime | **VoiceHub native** raw-audio frame BCE, interval data preparation, portable Safetensors export; the archived author recipe is a different GRU-only graph |
| `vad_nemo` | VoiceHub-native multilingual MarbleNet Frame-VAD using the pinned NVIDIA release | Real two-class scores on a 20 ms grid, normalized to common speech regions | Default install; no NeMo, Lightning, Hydra, OmegaConf, librosa, or torchaudio runtime | **VoiceHub native** raw-audio or aligned-frame cross-entropy with the released SGD and polynomial hold-decay recipe; portable Safetensors export |
| `vad_funasr` | VoiceHub-native FSMN graph, Kaldi-compatible fbank/LFR/CMVN frontend, and endpoint state machine for the published FunASR checkpoint | Real 10 ms frame probabilities, isolated streaming state, and 16 kHz speech boundaries normalized from milliseconds to seconds | Default install; no FunASR, ModelScope, torchaudio, or Transformers runtime | **VoiceHub native** raw-audio binary frame fine-tuning or aligned 248-PDF cross-entropy, with portable Safetensors export |
| `vad_auditok` | Auditok fixed or automatically calibrated energy detector | Dependency-light speech regions with configurable duration and silence constraints | Default install | **Inference-only** deterministic signal-processing algorithm |
| `vad_sherpa_onnx` | VoiceHub-native Silero and TEN graphs with pinned Sherpa-compatible streaming decisions | Real frame scores and isolated streaming-capable 16 kHz speech regions on CPU/CUDA; reviewed TEN ONNX weights convert once to Safetensors without ONNX execution | Default install; no Sherpa, ONNX, ONNX Runtime, Kaldi, librosa, or NumPy runtime | **VoiceHub native** Silero official decoder recipe or explicitly reconstructed TEN masked window BCE; portable Safetensors export |
| `vad_pyannote_segmentation` | VoiceHub-native PyanNet with `pyannote/segmentation-3.0` powerset weights | Frame scores and normalized speech regions | Default install; no pyannote runtime | **VoiceHub native** seven-class powerset frame fine-tuning |
| `vad_pyannote_brouhaha` | VoiceHub-native PyanNet with Brouhaha speech, signal-to-noise, and reverberation heads | Speech regions plus frame-level SNR/C50 metadata | Default install; no Brouhaha or pyannote runtime | **VoiceHub native** joint VAD/SNR/C50 fine-tuning |

Authentication for gated pyannote checkpoints is passed at runtime and is
never stored in serializable model configuration. The upstream repositories
publish Lightning pickle files rather than Safetensors. VoiceHub requires an
explicit, one-time restricted conversion and then uses only the converted
Safetensors artifact for inference, training, export, and reload.

## Installation

Install every ASR and VAD inference provider at once:

```bash
python -m pip install "voicehub @ git+https://github.com/kadirnar/voicehub.git@main"
```

Add the separate shared trainer and Weights & Biases reporting bundle:

```bash
python -m pip install "voicehub[training] @ git+https://github.com/kadirnar/voicehub.git@main"
```

This does not make every inference algorithm trainable. VoiceHub-native
profiles use the shared trainer with their architecture-specific objectives;
fixed algorithms such as WebRTC and Auditok remain inference-only because
they have no gradient-bearing parameters.

There is no ASR/VAD or provider-specific inference extra. Every built-in
speech-input `ModelSpec.install_extra` is `None`, meaning the runtime belongs to
the default installation. Package dependencies do not include checkpoint
weights or gated-repository access.

## Fine-tuning boundaries

| Boundary | What VoiceHub guarantees |
| --- | --- |
| **VoiceHub native** | `load_for_training()` retains or reconstructs a differentiable model, the training family is registered, and the model or adapter returns the intended scalar objective |
| **Inference-only** | The published or selected runtime has no verified trainable graph in VoiceHub |

These statuses describe the current integration, not what is theoretically
possible. Safetensors is only a weight container. It is suitable for
fine-tuning when the matching unfused model class, processor, objective, and
trainable parameters can be reconstructed. GGUF, ONNX, CTranslate2, JIT,
fixed-point, quantized, and other serving artifacts are not generic
fine-tuning checkpoints.

### ASR dataset boundary

Every registered ASR training profile now has an inspectable
`ASRDatasetSpec`. Query it with `get_asr_dataset_spec(model_type)`,
`get_training_spec(model_type).dataset_spec`, or
`model.validate_training_support().dataset_spec`; enumerate every ASR contract
with `list_asr_dataset_specs()`.

`ASRDataset` accepts mappings and JSON/JSONL/CSV/TSV manifests, normalizes
common audio and transcript aliases, resolves relative files, and validates
either an integrated source record or the profile's exact prepared-tensor
variant. `from_audio_folder()` pairs PCM WAV files with transcript sidecars,
while `from_kaldi()` imports materialized `wav.scp` plus `text` directories.
The [speech data guide](../guides/speech-data.md) lists the distinct CTC,
Whisper/sequence-to-sequence, prompted Qwen/Granite/VibeVoice, RNN-T/TDT,
SpeechBrain, SenseVoice, ESPnet, and WeNet records.

Cohere records require `language` and are automatically grouped by language
and punctuation mode. SeamlessM4T-v2 records are grouped by target language.
The Trainer uses the dataset's deterministic epoch-aware batch sampler so a
mixed multilingual corpus does not create an invalid model batch.

Transcript-bearing evaluation records activate the model's native
teacher-forced objective and produce `eval_loss`. WER/CER is a separate
generation-and-decoding evaluation; it is reported only when a specialized
adapter or caller-supplied metric implements the necessary decoding and text
normalization. This distinction prevents a finite loss from being presented
as recognition accuracy.

### Native ASR dispatch

The historical `asr_transformers` key is a compatibility name for a closed
dispatcher over audited VoiceHub architectures. VoiceHub registers
task-neutral adapters for:

- CTC, preserving backend-native blank and alignment semantics;
- speech sequence-to-sequence, using the checkpoint's teacher-forced native
  loss.

RNN-T and TDT are deliberately outside this dispatcher. Use the dedicated
`asr_nemotron` and `asr_parakeet_tdt` profiles so their transducer alignment
and token-duration objectives cannot be replaced with ordinary cross entropy.
The processor, label padding, lengths, blank ID, alignment topology, and
duration terms are part of the model.

### Native VAD dispatch

Audio classification accepts one class or binary/multilabel target per
window. Frame classification requires targets already aligned to the output
timebase and an explicit mask for padded frames. Native model loss is
preferred; the classification fallback runs only when the profile declares
it.

### Native PyanNet VAD recipes

The three PyanNet providers own their graph and objectives inside VoiceHub.
`vad_pyannote` consumes four-channel multi-label frame targets.
`vad_pyannote_segmentation` consumes integer powerset class IDs from 0 through
6. `vad_pyannote_brouhaha` consumes `[vad, snr_db, c50_db]` per frame; SNR loss
is evaluated on speech frames and C50 loss on valid frames. Callers remain
responsible for aligning labels to `model.frame_count(num_samples)`.

### Native FSMN VAD recipe

`vad_funasr` is a compatibility name; inference and training execute the
VoiceHub-owned `fsmn-vad` architecture. The graph reproduces the released
400→140→250→four-layer FSMN→140→248 topology, including the 16 kHz
Kaldi-compatible fbank, 5-frame LFR stacking, fixed CMVN, per-layer streaming
caches, and endpoint decoder. Raw timestamp segments are aligned to the
25 ms analysis windows on a 10 ms grid using at least 50 percent speech
coverage. Explicit `pdf_labels` select the 248-class objective; binary frame
labels optimize the decoder's grouped silence-versus-speech probability.

The public artifact contains an inference state dict and CMVN transform, but
does not publish the private corpus, acoustic-PDF target generator, or original
loss recipe. VoiceHub therefore guarantees checkpoint-compatible PDF
cross-entropy and a documented binary VAD objective, not reproduction of an
unpublished training run.

### Native SpeechBrain CRDNN VAD recipe

`vad_speechbrain` executes the exact published 40-bin
Fbank→CNN→bidirectional-GRU→DNN graph and preserves all 49 released tensors.
Raw interval annotations use the archived LibriParty recipe's 10 ms indexing;
masked binary cross-entropy slices away the final centered-STFT frame exactly
as that recipe does. Export and fresh reload use only `config.json` and
Safetensors.

The artifact points to SpeechBrain revision
`ea17d223cc7814f1027d657ed713676bbaacb608`, but the VAD recipe at that
revision builds a smaller GRU-only graph with global normalization. The
published CRDNN training program and claimed augmentation path are therefore
not author-verifiable. VoiceHub guarantees graph/checkpoint/inference parity
and a documented native fine-tuning objective, not exact reproduction of
that missing recipe.

### Native MarbleNet Frame-VAD recipe

`vad_nemo` executes the native `marblenet-vad` graph rather than importing
NeMo. It preserves the released 25 ms Hann frontend with 10 ms feature hop,
stride-two six-block MarbleNet encoder, and two-class 20 ms frame decoder.
Timestamp segments are aligned to that 20 ms output grid; callers may instead
provide explicit binary frame labels and masks.

The trainer uses frame cross-entropy and source-native SGD defaults
(`lr=0.01`, `momentum=0.9`, `weight_decay=0.001`) with five percent warmup,
fifteen percent hold, and power-two polynomial decay to `1e-8`. The published
white-noise, gain, noise-mixture, and spectrogram masking parameters are
documented in the artifact manifest. The original multilingual corpora remain
the user's responsibility.

The native ESPnet key is deliberately narrower than the upstream project. It
supports only the audited LibriSpeech Transformer-e18 release; other ESPnet
architectures are rejected. VoiceHub owns its frontend, global CMVN,
SpecAugment, hybrid loss, scheduler, decoding, checkpoint conversion, and
export. Corpus acquisition and speed perturbation remain explicit dataset
operations rather than hidden provider-runtime behavior.

## Discover support in code

```python
from voicehub import SpeechTask, list_model_specs, list_training_specs

asr_models = list_model_specs(task="asr")
vad_models = list_model_specs(task="vad")

for spec in (*asr_models, *vad_models):
    training = spec.training
    print(
        spec.model_type,
        spec.task.value,
        training.family_name,
        training.support.value,
    )

all_training_profiles = list_training_specs(task=None)
```

The historical `AutoInferenceModel` and default `list_training_specs()` views
remain TTS-oriented for compatibility. Use task-specific factories and an
explicit task filter for new ASR/VAD code.

## Adding future checkpoints and providers

Use an existing provider key when a new checkpoint conforms to that runtime
and output contract. Add a new `ModelSpec` only when loading, execution,
dependencies, or training ownership differs materially.

The [ASR/VAD provider integration guide](../project/adding-speech-provider.md)
defines the configuration, lazy wrapper, normalization, registry, training,
and test contracts for future families.
