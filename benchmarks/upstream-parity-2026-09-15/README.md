# Upstream comparison audit — incomplete

This audit does **not** establish quality or performance parity for all 68
VoiceHub integrations. It records real paired executions, unsuccessful
attempts, and pending work separately. See [inventory.json](inventory.json)
for every model, its original repository and checked-out SHA, and its attempts.
A cloned repository or installed environment is not a successful inference run.

Current coverage: **21 of 68 integration types have paired smoke evidence**.
The other 47 remain unverified. This count includes original repositories'
documented Hugging Face inference paths; it does not mean every original
backend, default checkpoint, model size, language, or quality metric was tested.

## Fixes verified in this working tree

- English-only Whisper accepts explicit English without adding nonexistent
  multilingual decoder tokens.
- Auditok retains trailing silence by default, matching upstream. Its new
  `drop_trailing_silence=True` option selects trimming.
- Silero validates complete input once and avoids repeated frame-state object
  construction. Whole-audio and frame-by-frame outputs remain bit-identical.
- Download locks record their host and process. On POSIX, a dead local owner
  can be recovered immediately; an old lock owned by a live process is kept.
- FunASR preserves the upstream decoder's region boundaries instead of merging
  them a second time. Its frame-score loop avoids repeated tensor indexing.
- WebRTC batches the pinned original C detector through an optional compiler
  path. All 48 rate/frame-duration/mode configurations agree with the Python
  detector, including padded tails and repeated fresh requests. Source and
  license files ship in the wheel and source distribution. Compilation occurs
  at first model load and is cached; its cost is excluded from warm inference.
  Missing compilers retain the Python fallback. The output's `execution_engine`
  metadata identifies the active path. Only Linux compilation was validated.
- Dia pads completed channels after EOS and includes their delayed tail in
  the generation budget. A real checkpoint previously failed when special
  tokens reached its DAC decoder; the corrected checkpoint run completes.
- Sherpa's Silero scorer uses overlapping 576-sample windows with a 512-sample
  shift, matching the original runtime. The prior standalone-Silero framing
  shifted speech starts and missed a pause in a public sample.
- LLaSA loads symlinked Hub snapshots, uses the original top-50 filter and
  runtime date in its prompt, and follows the published final-token slicing.
  An explicit `date_string` configuration makes the prompt date reproducible.
- Echo removes an extra codec decoder layer, maps legacy weight-normalization
  tensor names with strict inventory checks, and preserves the original BF16
  gain rounding. The latter fixes a measured waveform discrepancy.
- Irodori preserves that same BF16 gain rounding, pads text to the original
  runtime's configured width, and disables speaker guidance when no reference
  is supplied. The released Japanese checkpoint now produces identical BF16
  duration predictions and codec-input latents in a diagnostic comparison.
  SilentCipher watermarking remains unsupported by VoiceHub.

Kokoro's built-in text frontend remains a quality gap. Its supported
caller-provided frontend API can run the original Misaki English frontend;
that configuration produced identical audio in the measured sentence. This
does not make the built-in fallback equivalent to upstream.

## Measurements

The [expanded controlled run](runs/final-results/report.json) used an RTX 5070 Ti,
PyTorch 2.8.0, one CPU thread, fixed seeds, two warmups, and ten measurements
per sample. Other benchmark processes were excluded and the audit's background
download processes were paused and documentation checks had finished. Package versions,
working directories, input/checkpoint hashes, loading time, memory, and raw
latencies are attached to each execution. These are measurements of a modified
working tree, not a released commit.

Latency ratios below are **VoiceHub / upstream**. Ratios above 1 mean VoiceHub
was slower. The 95% interval screening rule flags a regression only when its
lower bound exceeds a 10% allowance. This is a small timing experiment, not a
claim that noise or clock variation has been eliminated.

| Integration / configuration                               | Output comparison                                                                | Latency ratio                          |
| --------------------------------------------------------- | -------------------------------------------------------------------------------- | -------------------------------------- |
| Silero VAD, CPU                                           | Same boundaries on all five excerpts                                             | 1.17–1.18; regression flagged          |
| WebRTC VAD, Python, CPU (before accelerator)              | Same boundaries on all five excerpts                                             | 51.88–61.76; regression flagged        |
| Auditok, CPU                                              | Boundaries agree within 1 ns on all five excerpts                                | 0.19–0.30                              |
| Native Whisper, tiny.en, FP32 CUDA                        | Same normalized text and WER/CER on all five; verbatim text differs              | 1.38–1.47; regression flagged          |
| OpenAI Whisper compatibility provider, tiny.en, FP32 CUDA | Same normalized text on two excerpts                                             | 1.40–1.46; regression flagged          |
| Transformers ASR, tiny.en, FP32 CUDA                      | Same normalized text and WER/CER on one excerpt                                  | 1.07                                   |
| NeMo QuartzNet, FP32 CUDA                                 | Same normalized text and WER/CER on five excerpts                                | 0.71–0.81                              |
| WeNet U2++, FP32 CUDA                                     | Same normalized text and WER/CER on five excerpts                                | 1.08–1.17; regression flagged on three |
| SpeechBrain VAD, CPU                                      | Same regions; boundary float representation differs by at most 0.84 microseconds | 0.89–0.98                              |
| FunASR VAD, CPU, corrected endpoints                      | Same boundaries on five excerpts                                                 | 0.99–1.15; regression flagged on one   |
| Kokoro, raw text, FP32 CUDA                               | Different audio duration and waveform                                            | 0.98                                   |
| Kokoro, identical supplied phonemes, CPU                  | Exact 93,000-sample waveform                                                     | 1.04                                   |
| Kokoro, original text frontend injected, CPU              | Exact 93,000-sample waveform                                                     | 1.05                                   |

These 13 cases cover **11 integration types**, using the specific checkpoints
recorded in their requests. The Whisper comparisons use tiny.en, not every
registry default or model size. The subsequent
[compiled WebRTC run](runs/webrtc-compiled-final-results/report.json) supersedes
the Python timing result for the compiled path; inspect its raw ratios and
`execution_engine` metadata. The Python fallback still has the measured gap.
The compiled run measured **0.73–0.86×** upstream latency with identical
boundaries on all five samples and no flagged regression.

The additional [Dia comparison](runs/dia-cross-run-results/report.json)
combines the successful original execution with the corrected VoiceHub run
of the identical request. Both generated 5.75855 seconds of audio at 44.1 kHz;
maximum waveform error was 6.56e-7 and RMSE was 4.88e-8. This brings successful
paired evidence beyond the controlled run. The 512-token budget limits this
to a smoke test. The exploratory latency ratio is **2.42×**, a flagged
regression. Setup activity overlapped portions of the runs. An intervening
original retry exited with signal 11 and is preserved as a failed attempt;
the combined report identifies the two successful evidence sources explicitly.

The [NeMo VAD run](runs/nemo-vad-restored-results/report.json) adds another
integration. Speech regions agree within 0.84 microseconds on five samples;
frame probability error is at most 2.39e-7. Exploratory latency ratios range
from 0.92 to 1.22; background compilation/download work was active. The
original NeMo 2.1 checkout's restoration permits the archive's missing
training-only `loss.weight`, while rejecting every other missing or unexpected
tensor. That compatibility step does not change inference weights.

Further exploratory runs:

| Integration                                    | Output agreement on five excerpts         | Latency ratio                  | Evidence                                                   |
| ---------------------------------------------- | ----------------------------------------- | ------------------------------ | ---------------------------------------------------------- |
| Sherpa-compatible Silero, corrected context    | Exact segment lists                       | 3.89–4.00; regression flagged  | [Report](runs/sherpa-context-fixed-results/report.json)    |
| Wav2Vec2 base 960h                             | Same normalized transcripts, WER, and CER | 1.10–1.61; two samples flagged | [Report](runs/wav2vec2-prepared-results/report.json)       |
| Moonshine tiny, 128-token budget               | Same normalized transcripts, WER, and CER | 1.04–1.10; none flagged        | [Report](runs/moonshine-prepared-results/report.json)      |
| WavLM base-plus CTC                            | Same normalized transcripts, WER, and CER | 0.86–1.02; none flagged        | [Report](runs/wavlm-prepared-results/report.json)          |
| Faster-whisper compatibility, tiny.en FP32 CPU | Same normalized transcripts, WER, and CER | 1.39–1.54; all flagged         | [Report](runs/faster-whisper-prepared-results/report.json) |

Echo's [corrected sentence run](runs/echo-rounding-fixed-results/report.json)
produced the same 3.29723-second waveform shape with RMSE 1.56e-7 and maximum
absolute error 1.76e-6. Its latency ratio was 1.03 with no flag. An independent
diagnostic found exactly equal diffusion latents after preserving BF16 gain
rounding; the remaining small waveform difference is in codec decoding.
The original checkout's unused `AudioDecoder` import was deferred until file
input, and local artifact resolution was pinned. Both sides used PyTorch 2.8;
the original project requests >=2.9.1. The reference compatibility patch is
archived, and this deviation limits the original-environment claim.

LLaSA's staged runs preserve its initial snapshot-loading failure, then the
sampling and prompt-date differences found with real checkpoints. Its reference
uses the published Hugging Face XCodec2 conversion. The 512-token smoke cap
must be considered when assessing output quality and termination behavior.
Faster-whisper uses the original CTranslate2 converter's FP32 output from the
same tiny.en tensors; VAD, timestamps, and beam search were disabled on both sides.

### Public-transcript TTS check

The repaired integrations also synthesized the first public LibriSpeech
transcript. The [public-text run](runs/tts-public-results/report.json) and
[final LLaSA retest](runs/llasa-tail-fixed-results/report.json) contain the
measurements below. The latter supersedes the earlier LLaSA output-length
difference. All four comparisons use the same text, seed, weights, and
effective generation settings on both sides.

| Integration                        | Waveform RMSE     | Latency ratio | Upstream / VoiceHub WER |
| ---------------------------------- | ----------------- | ------------- | ----------------------- |
| Echo                               | 2.19e-7           | 0.99; no flag | 5.88% / 5.88%           |
| LLaSA, 512-token cap               | 6.86e-8           | 1.36; flagged | 5.88% / 5.88%           |
| Dia, 512-token cap                 | 7.16e-8           | 2.37; flagged | 11.76% / 11.76%         |
| Kokoro, original frontend injected | 0, exact waveform | 1.04; no flag | 5.88% / 5.88%           |

One original Whisper tiny.en recognizer scored all eight listening outputs
after identical FFmpeg conversion to mono 16 kHz PCM16. WER and CER agree
between sides for each model; [raw transcripts and scoring evidence](tts-intelligibility/report.json)
are attached. “Mr.” versus “Mister” accounts for one normalized word error;
the capped Dia output also ends before “gospel.” These are bounded smoke
comparisons, not full perceptual-quality benchmarks. The Kokoro row uses an
injected frontend and does not resolve its built-in frontend gap.

The [separate default-Kokoro public-text run](runs/kokoro-raw-public-results/report.json)
confirms that gap: the same recognizer measured **88.24% WER for VoiceHub's
built-in fallback versus 5.88% upstream**. Audio duration also differed
(6.05 versus 5.90 seconds), despite a similar 1.02 latency ratio. See
[its transcripts and scoring evidence](kokoro-public-intelligibility/report.json).
This failed quality comparison must not be combined with the injected-frontend
row as a default-path pass.

LLaSA's corrected prompt and generated token sequence match exactly in a
separate diagnostic. Its final 10.22-second audio also matches upstream shape
after matching the published last-token slicing. The archived diagnostics
precede that last slicing change and retain the intermediate 511/512-code
difference. Auxiliary codec revisions and file hashes are recorded in
[auxiliary-artifacts.json](auxiliary-artifacts.json).

Sherpa was compiled from its own checkout with unrelated TTS support disabled.
Its package initializer then requested absent TTS symbols, so the worker loads
the original VAD extension directly. That reference uses Python 3.11 and ONNX
Runtime; VoiceHub uses Python 3.12 and PyTorch. Its ONNX and JIT artifacts are
sibling exports from the same pinned Silero checkout. The first whole-file
ingestion attempt was invalidated: Sherpa's example requires incremental frame
ingestion for correct buffer timestamps. All attempts remain archived.

Wav2Vec2 follows fairseq's explicit Transformers example. Moonshine and WavLM
use the Hugging Face releases linked by their original repositories, rather
than the projects' other runtime formats. For WavLM, the published pickle was
loaded with `weights_only=True` and serialized to shared Safetensors; all 250
tensor names and values were verified unchanged. Original and converted hashes
are recorded. These details constrain the coverage claim above.

The separate [expanded run](runs/expanded-results/report.json) also measured
the Transformers ASR provider on the same tiny.en checkpoint: normalized text
and WER/CER agreed on one excerpt. Other expanded attempts include download,
dependency, and checkpoint failures. Its timing measurements were exploratory
and did not pause all background work.

The [prepared run](runs/prepared-results/report.json) exercises Kokoro's
injected original frontend and additional VAD recipes. Its initial FunASR
attempt used different effective endpoint settings; that comparison is marked
invalid, with its original metrics retained for diagnosis. Do not treat those
differences as a model regression.

## Audio quality and public samples

ASR and VAD use five public LibriSpeech excerpts with transcripts, recorded in
[public-samples.json](public-samples.json), with [audio attribution](AUDIO_ATTRIBUTION.md).
These excerpts are smoke fixtures,
not a representative evaluation corpus. No human VAD boundary labels or
listening panel were available; VAD error rates and MOS remain unmeasured.

Kokoro synthesized: “Hello, world. This is a speech quality comparison.” The
same original Whisper tiny.en recognizer scored the two synthesized outputs.
It measured 0% WER for upstream audio and 75% for VoiceHub's default fallback
audio. See [the intelligibility diagnostic](kokoro-intelligibility.json).
This one-sentence recognizer proxy is not a perceptual quality score.

Listening files:

- [Original Kokoro, raw text](runs/controlled-results/kokoro/upstream-hello.wav)
- [VoiceHub Kokoro, built-in fallback](runs/controlled-results/kokoro/voicehub-hello.wav)
- [VoiceHub Kokoro, original frontend injected](runs/prepared-results/kokoro-injected-frontend/voicehub-hello.wav)
- [Original Echo, public transcript](runs/tts-public-results/echo-public-text/upstream-1272-128104-0000.wav)
- [VoiceHub Echo, public transcript](runs/tts-public-results/echo-public-text/voicehub-1272-128104-0000.wav)
- [Original LLaSA, public transcript](runs/llasa-tail-fixed-results/llasa-public-text/upstream-1272-128104-0000.wav)
- [VoiceHub LLaSA, public transcript](runs/llasa-tail-fixed-results/llasa-public-text/voicehub-1272-128104-0000.wav)
- [VoiceHub Kokoro, built-in fallback, public transcript](runs/kokoro-raw-public-results/kokoro-raw-public-text/voicehub-1272-128104-0000.wav)

## Irodori released-checkpoint comparison

The public Japanese model-card example was executed in the original Irodori
repository and VoiceHub, each with its own Python 3.11 environment and
PyTorch 2.14.0+cu130 on the same RTX 5070 Ti. The model, tokenizer, codec, and
original watermark artifacts are pinned in [irodori-artifacts.json](irodori-artifacts.json).
This single demonstration sentence is not a held-out quality corpus.

| Setting                 | Upstream seconds | VoiceHub seconds | Ratio | Measured regression |
| ----------------------- | ---------------: | ---------------: | ----: | ------------------- |
| FP32, corrected runtime |           0.8742 |           0.8732 | 0.999 | No                  |
| BF16, corrected runtime |           0.7584 |           0.7793 | 1.028 | No                  |

The initial FP32 ratio was 1.118 and flagged a regression. The corrected
[FP32 run](runs/irodori-runtime-fixed-results/report.json) and isolated
[BF16 repeat](runs/irodori-bf16-verified-results/report.json) do not exceed the
10% allowance. The first corrected BF16 timing overlapped a diagnostic process
loading on the GPU; its report retains that caveat and is superseded for timing
by the isolated repeat. Loading and hashing are excluded from warm inference.

The initial BF16 final waveform RMSE was 0.05170. After preserving fixed text
padding and omitting unused speaker guidance, it fell to 0.0004841. The
[diagnostic](irodori-diagnostics/summary.json) shows an exact duration prediction
and all 8,000 codec-input latent values matching. The captured first 100,000
samples immediately after codec decoding have RMSE 1.08e-7; this is a bounded
prefix diagnostic. Both final outputs have the same sample rate and duration
(9.96 seconds for FP32; 10.00 for BF16), with no clipped samples.

**Full waveform parity remains unverified:** the original automatically applies
SilentCipher watermarking, while VoiceHub rejects that unsupported option.
The small codec rounding difference and the watermark stage remain distinct
from the fixed synthesis differences. No original model or codec graph was
modified for these runs.

Independent multilingual Whisper-small, explicitly set to Japanese transcription,
produced identical transcripts for both repositories' corrected outputs. The
[FP32](irodori-fp32-final-intelligibility/report.json) and
[BF16](irodori-bf16-final-intelligibility/report.json) character error rates are
both 0.125 on each side. This metric includes Japanese orthographic differences;
whitespace-based WER in the raw report is not an appropriate Japanese word
segmentation metric. These are recognizer proxies, not listening-panel scores.
Earlier scorer dependency failures and an English-only recognizer attempt are
retained separately; the latter is explicitly invalidated and excluded.

The diagnostic script retains audit-machine paths. Full intermediate dumps
remain in the ignored cache; selected numeric arrays and the before/after
summary are archived. Only complete codec-input arrays support the latent
exactness claim; other recorded tensors may be truncated or differently ranked.

## Pending work

All 68 original repositories have separate checkouts. Many do not yet have a
successful independent inference recipe and checkpoint execution. The initial
generic environment setup contains failed or timed-out installations;
subsequent source-specific environment repairs are recorded in run package
metadata. An installation status alone must not be interpreted as readiness.

The registry cache sweep explicitly sets `VOICEHUB_OFFLINE=1`; it checks what
can load and execute from already available artifacts. Cache misses in that
sweep are **pending downloads**, not evidence of incorrect model outputs.
The earlier online attempts retain download timeouts and access errors.
Orpheus, CSM, and NeuTTS returned HTTP 401. Per the user's instruction,
**inaccessible checkpoints remain pending access**, with their original errors
preserved and no pass or parity credit. See [audit-policy.json](audit-policy.json).
Accessible checkpoints continue through the same comparison protocol.
Some integrations require reviewed conversion,
additional codecs or voice references, custom artifact layouts, or more model
preparation. Full per-model checkpoint, inference, quality, and performance
validation remains outstanding for the entries without paired evidence.

The measured Whisper, faster-whisper, Silero, Sherpa, Dia, and LLaSA timing
gaps remain unresolved. Some FunASR, WeNet, Wav2Vec2, and NeMo VAD samples also
exceed the regression allowance. The compiled WebRTC
path fixes the large Python-loop gap on this host.
Kokoro's default pronunciation gap remains unresolved. The audit does not
claim that every model error has been fixed.

## Reproduce and extend

Use [the comparison guide](../../docs/guides/upstream-parity.md) and
`scripts/benchmark_upstream_parity.py`. Each side has its own repository,
Python environment, log, and output file. Freeze identical requests and
checkpoint tensors before timing. The recorded commands contain paths from
this audit machine; adjust paths when replaying elsewhere.

The local preparation and run manifests remain under
`.cache/upstream-parity/`. Retest into a fresh output directory after changing
settings. `--resume` keeps prior cases, including failures, by name.

Collect completed evidence without downloading weights:

```python
import json
import subprocess
import sys
from pathlib import Path

output = Path("benchmarks/upstream-parity-2026-09-15")
runs = json.loads((output / "collection-runs.json").read_text())["runs"]
command = [
    sys.executable,
    "scripts/collect_upstream_parity_evidence.py",
    ".cache/upstream-parity",
    str(output),
]
subprocess.run(command + [arg for run in runs for arg in ("--run", run)], check=True)
```

`scripts/score_tts_intelligibility.py --help` describes how to repeat the
independent recognizer check. Supply the two public TTS reports above and the
recorded original Whisper repository, Python executable, and ASR request.
The later report supersedes the earlier LLaSA output for scoring. The
[diagnostic script](diagnose-tts.py) records intermediate token/latent arrays;
its paths are specific to this audit host and its runs are not timing evidence.

## Code and documentation validation

- Full Python suite after the library fixes: **2,625 passed, 16 skipped**,
  including 5,002 passing subtests (Linux, Python 3.12).
- Wheel, source distribution, editable install, registry, and compliance-file
  checks passed on Linux. Windows and macOS were not executed here.
- The welcome page adapts the supplied Liquid template. Strict MkDocs, DOM,
  60 screenshot/accessibility cases, 328 keyboard cases, and 4,258 focus steps
  passed across three viewports and two palettes. This evidence is separate from speech
  model parity. A successful documentation check is not model evidence.

See [validation.json](validation.json) and the archived validation logs for
the executed gates. Checkpoint prefetch was stopped for the user-requested
handoff; downloaded and partial artifacts remain in the ignored local cache.
Completing a download does not itself add a successful model comparison.
