---
description: Compare original speech repositories with VoiceHub using isolated environments, identical inputs, and explicit evidence.
---

# Upstream comparisons

Run the original model in its own repository and environment, then run VoiceHub
in a separate process. Compare the same checkpoint, audio or text, conditioning,
precision, seed, and decoding settings. A passing unit test or successful model
load does not establish end-to-end parity.

## Evaluation protocol

1. Record each upstream Git SHA and checkpoint revision, file hashes, package
   versions, hardware, and effective inference settings.
2. Prepare one shared input manifest. Use the exact same mono waveform and
   sampling rate for both implementations. Keep text normalization explicit.
3. Download and load weights before timing. Warm up both implementations, reset
   random seeds and recurrent state, and synchronize CUDA around each measurement.
4. Run inference serially on an otherwise idle device. Retain at least five
   warm measurements per sample; use longer, alternating runs to confirm timing
   regressions.
5. Save both outputs, errors, and logs. A failed, timed-out, or unconfigured
   comparison remains unverified.

### What the measurements mean

| Task | Compare | Interpretation |
| --- | --- | --- |
| TTS | Sampling rate, duration, waveform error, clipping, and listening samples | Numerical similarity alone is not perceived quality. Different phonemization or sampling can change audio even with identical weights. |
| ASR | Verbatim transcripts, normalized word error rate (WER), character error rate (CER) | Use the same ground-truth transcript and decoding policy. Report normalization alongside the scores. |
| VAD | Speech intervals and, when available, frame probabilities | Agreement with upstream is not accuracy against human speech-boundary labels. |
| All | Loading time, warm inference measurements, latency ratio | Wall-clock equality is not expected. Flag regressions and confirm them under controlled load. |

The timing report uses the difference between independent mean latencies with
an approximate 95% interval. A regression is flagged when its lower bound is
above the configured relative tolerance (10% by default). This is a screening
rule, not a guarantee of statistical significance for correlated GPU timings.
If either side records fewer than five warm timings for a sample (for example,
a slow model run once for correctness), that sample's `timing` entry is
`{"status": "insufficient-samples", ...}` with `regression_flag: null`; the
waveform, transcript, or segment comparison still runs. Such samples never set
the case-level `timing_regression_flag`.

## Run a comparison

From the VoiceHub checkout:

```bash
python scripts/benchmark_upstream_parity.py comparison.json \
  --output benchmarks/local-comparison/report.json \
  --timing-tolerance 0.10
```

The runner executes argument arrays without a shell. Each side specifies its
own working directory and Python executable. Relative paths are resolved from
the directory where you launch the runner; absolute paths are recommended.

```json
{
  "cases": [
    {
      "name": "silero-public-samples",
      "request": "/evaluation/silero-request.json",
      "timeout_seconds": 600,
      "provenance": {
        "source_revision": "record-the-checked-out-git-sha",
        "checkpoint_sha256": "record-the-shared-checkpoint-hash"
      },
      "upstream": {
        "cwd": "/evaluation/repos/silero-vad",
        "command": ["/evaluation/repos/silero-vad/.venv/bin/python", "/voicehub/scripts/upstream_parity_worker.py", "upstream"]
      },
      "voicehub": {
        "cwd": "/voicehub",
        "command": ["/voicehub/.venv/bin/python", "/voicehub/scripts/upstream_parity_worker.py", "voicehub"]
      }
    }
  ]
}
```

A request for the included Silero worker looks like this:

```json
{
  "model_type": "vad_silero",
  "task": "vad",
  "checkpoint": "/evaluation/repos/silero-vad/src/silero_vad/data/silero_vad.jit",
  "device": "cpu",
  "seed": 123,
  "threads": 1,
  "warmup": 2,
  "repeats": 10,
  "samples": [
    {"id": "speech-001", "audio": "/evaluation/speech-001.wav"}
  ]
}
```

The included worker implements original-repository recipes for Kokoro,
Whisper, Transformers ASR, Qwen3 ASR, NeMo ASR/VAD, SpeechBrain ASR/VAD,
FunASR ASR/VAD, WeNet ASR, Dia (the original repository's Transformers recipe),
LLaSA, Echo, faster-whisper's CTranslate2 runtime, Silero VAD, WebRTC VAD, and Auditok. A recipe's presence does
not mean its checkpoint execution succeeded. CTC and Moonshine recipes use
the Hugging Face releases linked from the original projects; their reports
identify that backend explicitly. `scripts/upstream_sherpa_vad_worker.py`
runs the original compiled Sherpa VAD bindings. The VoiceHub side discovers
registered models generically. Other upstream models need a source-specific
worker; missing recipes are never counted as successful comparisons.

Concurrent runner processes share an execution lock. Each case freezes its
request before either side runs; different executed request hashes invalidate
the comparison. `--resume` retains recorded cases **including failures** by
case name. Use a new output directory to retest changed requests or fixes.

### Result format

A worker receives the request and output JSON paths as its final two arguments.
It writes `status`, `task` (`tts`, `asr`, or `vad`), and a nonempty `samples`
list. Every sample includes `id` and `warm_seconds`.

- TTS adds `audio` (an absolute path to a NumPy array saved without pickle) and
  `sample_rate`. Keep WAV copies for listening.
- ASR adds `text` and the matching `reference` transcript. The runner
  compares `text` only. The generic VoiceHub worker also reports, when the
  model returns them, the detected `language`, timestamped `segments` as
  `[start_seconds, end_seconds, text]` rows (plus `segment_speakers`, one label
  per segment, when any segment has a speaker) and `words` as
  `[text, start_seconds, end_seconds]` rows (plus `word_confidences`, one score
  per word, when any word has one). Upstream recipes should write the same keys
  and shapes so the fields can be compared.
- VAD adds `segments` as `[start_seconds, end_seconds]` pairs and optionally
  `probabilities`.

### Generic VoiceHub TTS requests

The worker's `voicehub` side loads any registered model with
`request["config"]` and calls `model.generate(text, seed=request["seed"], **kwargs)`
for each TTS sample. These request and sample keys shape that call; an upstream
recipe must read the same keys and files so both sides receive identical inputs.

| Key | Scope | Effect |
| --- | --- | --- |
| `generation` | request | JSON keyword arguments for every sample. |
| `tensor_inputs` | request | Path to a NumPy `.npz` archive. Each array becomes a `torch.Tensor` keyword argument named after its entry, for every sample (for example, a speaker embedding exported by the upstream encoder). |
| `generation` | sample | JSON keyword arguments for this sample only, such as a per-sample reference voice file. |
| `tensor_inputs` | sample | `.npz` archive of tensor keyword arguments for this sample only, such as upstream frontend outputs. |
| `phonemes` | sample | Upstream G2P output for this sample; used only with one of the next two flags. |
| `matched_phonemes` | request | Pass the sample's `phonemes` as a keyword argument. |
| `phoneme_argument` | request | Name of that keyword argument (default `phonemes`, e.g. `phoneme_text`). |
| `phonemes_as_text` | request | Pass the sample's `phonemes` as the text argument instead (explicit-phoneme models). Mutually exclusive with `matched_phonemes`. |

Keyword arguments merge in this order, later entries winning: request
`generation`, request `tensor_inputs`, sample `generation`, sample
`tensor_inputs`, then phonemes. Archives are read with `allow_pickle=False`
(write them with `numpy.savez`), loaded once per process, and copied for every
call so in-place model updates cannot leak into later repeats. Tensors stay on
the CPU with their stored dtype; the model decides placement. Each report
sample records `tensor_inputs_sha256` for the archives it used. Model metadata
is made JSON-safe: scalar tensors and arrays become numbers, larger ones are
summarized as `{"shape": [...], "dtype": "..."}`.

The runner preserves loading metadata, command lines, working directories,
request hashes, exceptions, and separate logs. A `measured` status means both
sides produced comparable evidence; inspect the metric fields for differences.
It does **not** mean that quality or timing parity passed.

The included worker also records installed package versions, local checkpoint
hashes, input hashes, GPU allocation peaks, and process peak resident memory on
POSIX systems. GPU memory uses PyTorch's allocator measurements; it does not
include every allocation made by another library or driver. Process memory
includes Python, imports, loading, and inference.

## Public evaluation samples

The initial local audit uses five excerpts from the
[LibriSpeech dummy validation fixture](https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy).
These are public speech samples with transcripts, suitable for smoke comparisons.
They do not cover speakers, languages, noise conditions, or human VAD boundaries
well enough to establish corpus-level accuracy or perceptual quality.

Use representative held-out corpora and listening evaluations before making
broader quality claims. Keep raw-text TTS tests separate from tests supplied
with upstream-generated phonemes: the latter only verify the downstream graph.

## Findings from the September 2026 audit

The local audit records source revisions, individual attempts, raw measurements,
and listening outputs under the ignored `.cache/upstream-parity/` directory.
Keep generated WAV/NumPy files, JSON reports, logs, and screenshots out of pull
requests. Commit the comparison code, tests, and written findings instead.
Full quality and performance parity across the 68 integrations has **not**
been established. See [the comparison status](upstream-parity-status.md).

The audit fixed English-only Whisper language handling, Auditok's default
trailing-silence behavior, and recovery of download locks left by dead local
processes on POSIX. Silero's frame loop now avoids repeated validation and
state-object construction while preserving its measured output.

FunASR no longer merges decoded speech regions a second time using the silence
threshold. Its five measured samples now match upstream endpoints. Its decoder
also transfers frame scores to Python once, reducing scalar-indexing overhead.

WebRTC can compile the bundled, pinned original C implementation at model load
on Linux or macOS when C and C++ compilers are available. Warm inference uses
one batch call and preserves request-local detector state. The cached build's
first-load cost is excluded from warm timings. Without a working compiler,
the Python detector remains available. `VADOutput.metadata.execution_engine`
reports the selected path and any fallback reason. Linux differential checks
cover all supported rates, frame durations, and modes; macOS compilation has
not been tested here. This internal accelerator does not change the public
optimization API.

Dia's delayed generation now pads channels after their EOS token and reserves
the delayed tail within `max_new_tokens`. This fixes a released-checkpoint
failure where special tokens reached the DAC decoder. The measured 512-token
smoke output matches upstream duration with sub-micro-unit waveform error;
it is a bounded generation, not a full perceptual-quality evaluation. Its
cross-run timing comparison remains exploratory and flags a regression.

The Sherpa-compatible Silero scorer now consumes the upstream 576-sample
overlapping windows at 16 kHz with a 512-sample shift. Using standalone Silero
framing had shifted speech starts and missed a short pause. All five public
sample segment lists match after this correction. The original reference uses
compiled C++/ONNX Runtime; its build settings and Python version are retained
in the report. A substantial timing regression remains on the native path.

LLaSA now preserves Hugging Face snapshot filenames when they are symlinks to
extensionless cache blobs. Its sampler uses the original top-50 default, and
its chat template follows the runtime date instead of hard-coding July 2024.
Use `top_k=0` to disable the filter and configuration `date_string="15 Sep 2026"`
to freeze the prompt date for replay. These corrections can change previously
seeded outputs. The reference recipe uses the authors' `xcodec2-hf` release;
the original legacy XCodec2 backend is a separate, unverified configuration.
Like the published recipe, VoiceHub omits the final generated token even at
the length cap. Output metadata records whether a speech-end token was reached.

Echo's codec now follows the published decoder block order and accepts legacy
weight-normalization tensor names without relaxing checkpoint validation.
Its adaptive normalization preserves the original BF16 addition before FP32
promotion. On the measured sentence, this reduced waveform RMSE from 0.0734
to 1.56e-7; the generated diffusion latents also match exactly in a separate
diagnostic. Echo's original environment
defers its unused audio-file decoder import and uses PyTorch 2.8 to match this
host; the upstream requested version is recorded as a compatibility deviation.

Irodori additionally preserves the original runtime's fixed text padding and
omits speaker guidance when no reference voice is supplied. On the released
Japanese checkpoint, the corrected BF16 duration prediction and generated
codec-input latents match exactly in the diagnostic run. The original runtime
applies SilentCipher watermarking when available; VoiceHub does not support
that stage, so the final waveforms still differ. This is paired execution
evidence, not full output parity.

Kokoro's built-in grapheme fallback does not reproduce the upstream text
frontend. The tested sentence produced different audio and substantially worse
recognizer-based intelligibility. Identical supplied phonemes, or the original
Misaki frontend injected through the existing public `text_frontend` argument,
produced identical CPU waveforms in the tested comparison. This does not
establish parity for other texts or languages. The benchmark worker's
`"text_frontend": "misaki_en"` request exercises that injection with separately
installed `misaki[en]`, its English language model, and espeak dependencies.

Measured timing regressions remain for native Whisper, Silero, and some
FunASR/WeNet samples. The compiled WebRTC path removed the large measured
Python-loop regression on this Linux host; the Python fallback remains slower.
Checkpoints that are inaccessible, unavailable locally, or still awaiting a
working original-repository recipe remain pending. Per the audit instruction,
access-denied checkpoints are left pending access; their failures remain in
the report and receive no pass credit. Unit tests and source
checkouts do not turn those entries into passes.

## Documentation reference

The welcome page follows the structure of
[Liquid's LFM welcome page](https://docs.liquid.ai/lfm/getting-started/welcome),
retrieved September 15, 2026: a welcome heading, short introduction, branded
banner, capability summary, and four starting-point cards. VoiceHub keeps its
own content, links, branding, and existing accessible documentation shell.
This is a template adaptation; pixel-identical reproduction is not claimed.
