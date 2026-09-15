---
description: Scope, measured findings, and pending work for original-repository speech comparisons.
---

# Upstream comparison status

The audit is incomplete. As of September 16, 2026, **23 of 68 integration types**
have paired execution evidence. A successful measurement is not a parity pass.
Raw artifacts are stored locally under `.cache/upstream-parity/`; they are not
part of the source contribution. See [the protocol](upstream-parity.md) for
reproduction and interpretation.

## Paired integrations

`dia`, `kokoro`, `echo`, `llasa`, `irodoritts`, `parlertts`, `asr_qwen3`, `asr_transformers`, `asr_whisper`, `asr_wav2vec2`, `asr_wavlm`, `asr_moonshine`, `asr_faster_whisper`, `asr_openai_whisper`, `asr_nemo`, `asr_wenet`, `vad_silero`, `vad_webrtc`, `vad_speechbrain`, `vad_nemo`, `vad_funasr`, `vad_auditok`, `vad_sherpa_onnx`.

## Verified corrections

- Whisper: English-only language-token handling.
- Auditok: original trailing-silence behavior.
- Silero: reduced repeated validation and state allocation.
- FunASR VAD: endpoint boundaries and frame-score iteration.
- WebRTC VAD: optional original-source C acceleration, with the Python fallback.
- Dia: delayed end-of-speech tokens and codec-safe output.
- Sherpa VAD: original overlapping-window context.
- LLaSA: checkpoint symlinks, original sampling defaults, runtime prompt date,
  and final-token slicing.
- Echo: codec layer order, legacy weight-normalization keys, BF16 rounding.
- Irodori: BF16 rounding, fixed text padding, and disabled speaker guidance for
  requests without a reference voice.
- Parler-TTS: preserve checkpoint filenames and sibling assets when loading
  symbolic links in Hugging Face snapshots.
- Hub downloads: recovery of locks belonging to dead local processes.

## Remaining differences

Kokoro's default text frontend has a measured pronunciation/intelligibility gap.
The original frontend injected through its public API matches the tested CPU
output. Irodori's original runtime adds SilentCipher watermarking; VoiceHub does
not support that stage. Its corrected BF16 latent sequence matched all 8,000
values in the recorded Japanese example, while the final waveforms differ.

Timing regressions remain for Whisper, faster-whisper, Silero, Sherpa, Dia,
LLaSA, and some FunASR, WeNet, Wav2Vec2, and NeMo VAD samples. Irodori's corrected
FP32 and BF16 latency ratios were 0.999 and 1.028 relative to upstream, below
the 10% screening allowance. These measurements apply only to the recorded
hardware, precision, samples, and decoding settings.

Orpheus, CSM, NeuTTS, and MedASR checkpoints returned HTTP 401. Per the audit instruction,
inaccessible checkpoints remain pending access and receive no pass credit.

Qwen3-ASR matched normalized transcripts on four of five public samples. On
the longest sample, upstream WER/CER were 0.0588/0.0169 and VoiceHub
WER/CER were 0.0882/0.0339. VoiceHub took 1.477–1.602 times as long,
with all five samples flagged. Input token IDs and masks matched exactly.
Substituting the original log-mel features into the native graph did not
change the longest-sample transcript, so preprocessing roundoff alone does
not explain that mismatch. These differences remain unresolved.

Parler-TTS now completes the public-transcript comparison after correcting
checkpoint-symlink loading and selecting eager attention on both sides (the
original T5 encoder rejects SDPA). Both produce 8.824 seconds at 44.1 kHz;
waveform RMSE is 1.44e-8. This greedy-decoding example does not establish
parity for sampled generation. Isolated mean inference times were 10.861 s
upstream and 18.465 s in VoiceHub (1.700×, flagged). Independent Whisper
recognized only "Mr. Quilter" on both greedy outputs: WER 0.9412 and CER
0.8767 on each side. Matching this output therefore does not establish good
intelligibility.

With checkpoint-default sampling, seed 123, and a 768-token cap, both sides
produce 5.654 seconds of audio with waveform RMSE 2.88e-8. The independent
recognizer returns the same complete sentence on both: WER 0.0588 and CER
0.0548, including the "Mister"/"Mr." normalization difference. Mean inference
times are 7.160 s upstream and 9.398 s in VoiceHub (1.313×, flagged). This is
one public transcript and one seed, not a full perceptual quality evaluation.

## Pending integrations

`orpheustts`, `vui`, `chatterbox`, `conversationtts`, `cosyvoice`, `f5tts`, `gptsovits`, `melotts`, `openvoice`, `outetts`, `styletts2`, `mosstts`, `qwen3tts`, `zonos`, `zonos2`, `voxcpm`, `omnivoice`, `higgstts`, `xtts`, `vibevoice`, `fishtts`, `csm`, `neutts`, `supertonic`, `inflecttts`, `bark`, `speecht5`, `vits`, `asr_tiron`, `asr_vibevoice`, `asr_granite_speech`, `asr_parakeet_tdt`, `asr_nemotron`, `asr_cohere`, `asr_medasr`, `asr_hubert`, `asr_seamless_m4t_v2`, `asr_whisperx`, `asr_speechbrain`, `asr_funasr`, `asr_espnet`, `vad_transformers`, `vad_pyannote`, `vad_pyannote_segmentation`, `vad_pyannote_brouhaha`.

## Validation snapshot

The last completed source validation passed 2,626 Python tests and 5,006
subtests, with 16 tests skipped. Strict documentation, DOM, wheel, source
archive, editable-install, lazy-import, and provenance checks passed on Linux.
The documentation visual run covered 60 cases, three viewports, and two palettes.
The updated Linux screenshot baseline is retained locally rather than included
in the PR. The committed baseline predates the requested welcome-page design:
its six home-page signatures fail comparison against the reviewed render; the
other 54 signatures pass. The default visual gate is therefore pending a
separate reviewed baseline update. Windows and macOS were not executed. These checks do not establish speech
quality parity or accuracy across a representative evaluation corpus.

The initial public samples comprise five LibriSpeech clean utterances from one
speaker plus a public Japanese Irodori example. Recognizer error rates are
intelligibility proxies; no human listening panel or reference VAD annotations
were used. Full corpus-level quality evaluation remains outstanding.
