---
description: Transcribe file, array, and tensor audio through VoiceHub's normalized ASR lifecycle.
---

# Speech recognition

VoiceHub exposes every ASR integration through
`AutoModelForSpeechRecognition` and normalizes results as `ASROutput`. Use the
[ASR/VAD matrix](../models/asr-vad-support.md) for checkpoint-specific
languages, timestamps, decoding, licenses, and training boundaries.

## Install

```bash
python -m pip install "voicehub @ git+https://github.com/kadirnar/voicehub.git@main"
```

No ASR-specific inference extra is required. Checkpoints load only when a
selected model is used.

## Discover providers

```python
from voicehub import AutoModelForSpeechRecognition

for spec in AutoModelForSpeechRecognition.available_models():
    print(
        spec.model_type,
        spec.architecture,
        spec.default_model_path,
        spec.capabilities,
    )
```

Use a canonical `model_type`; compatibility aliases may be accepted, but the
canonical key makes recorded runs easier to compare.

## Transcribe

```python
from voicehub import AutoModelForSpeechRecognition

model = AutoModelForSpeechRecognition.from_pretrained(
    "Qwen/Qwen3-ASR-0.6B",
    model_type="asr_qwen3",
    device="cuda",
    lazy_load=True,
)
result = model.transcribe(
    "meeting.wav",
    language="English",
    hotwords=["VoiceHub"],
)
print(result.text)
```

Construction is lazy. The first transcription loads the checkpoint. Unknown
decoding options raise rather than being silently ignored.

## Audio inputs

Every ASR wrapper accepts the same input envelope:

```python
# File: sampling rate comes from the header.
file_result = model.transcribe("speech.wav")

# Tensor or array: sampling rate is required.
tensor_result = model.transcribe(waveform, sampling_rate=16_000)

# Mapping:
mapping_result = model.transcribe(
    {"array": waveform, "sampling_rate": 48_000}
)
```

`AudioInput` is also accepted. VoiceHub validates finite audio, downmixes when
required, and resamples to the selected provider's rate. Public timestamps are
always seconds on the original recording timebase.

## Decoding configuration

Use one serializable config for repeated requests:

```python
from voicehub import ASRInferenceConfig

decoding = ASRInferenceConfig(
    language="English",
    task="transcribe",
    hotwords=("VoiceHub",),
    batch_size=1,
    num_beams=1,
    max_new_tokens=256,
)
result = model.transcribe("long-form.wav", inference_config=decoding)
```

Common fields are a shared vocabulary, not a promise that every provider
implements every mode. Unsupported translation, timestamp, hotword, chunking,
or beam options fail closed.

## Output

`ASROutput` contains:

| Field | Meaning |
| --- | --- |
| `text` | Complete transcript |
| `segments` | Ordered `ASRSegment` values, when computed |
| `language` | Requested or detected language, when known |
| `duration` | Input duration, when materialized |
| `metadata` | Provider-specific details |

Segments can contain start/end times, confidence, language, speaker, and
word-level `ASRWord` values. Missing timing or confidence stays `None`; the
wrapper does not invent it.

## Buffered streaming

```python
session = model.stream(sampling_rate=16_000, language="English")
session.push(chunk_1)
session.push(chunk_2)
result = session.flush()
session.close()
```

The common ASR session buffers chunks and performs offline inference on
`flush()`. Do not describe it as low-latency incremental decoding unless the
selected integration explicitly overrides that contract.

## Fine-tuning

Install training tools:

```bash
python -m pip install "voicehub[training] @ git+https://github.com/kadirnar/voicehub.git@main"
```

Start with one step on a speaker-disjoint split:

```python
from voicehub import (
    ASRDataset,
    AutoModelForSpeechRecognition,
    Trainer,
    TrainingArguments,
)

model = AutoModelForSpeechRecognition.from_pretrained(
    "facebook/wav2vec2-base-960h",
    model_type="asr_wav2vec2",
    device="cuda",
    lazy_load=True,
)
model.validate_training_support()

corpus = ASRDataset.from_manifest(
    "data/asr.jsonl",
    model_type="asr_wav2vec2",
    validate_files=True,
)
train_source, validation_source = corpus.train_test_split(
    validation_fraction=0.1,
    seed=42,
    group_by="speaker_id",
)

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="runs/asr-smoke",
        max_steps=1,
        per_device_train_batch_size=1,
        learning_rate=3e-5,
        logging_steps=1,
        save_steps=1,
        report_to="none",
    ),
    train_dataset=model.create_training_dataset(train_source),
    eval_dataset=model.create_training_dataset(validation_source),
)
trainer.train()
trainer.save_model("runs/asr-smoke/final")
```

The model adapter owns CTC, sequence-to-sequence, RNN-T, TDT, or hybrid loss
semantics. The generic trainer never guesses an objective from arbitrary
logits. Evaluation loss is not automatically WER or CER; decoded metrics need
an explicit decoding and text-normalization policy.

These canonical ASR keys currently have registered training profiles:

| Model type | Objective family |
| --- | --- |
| `asr_transformers` | Native verified dispatcher |
| `asr_whisper` | Speech sequence-to-sequence |
| `asr_faster_whisper` | Native Whisper compatibility |
| `asr_whisperx` | Whisper plus separate CTC alignment |
| `asr_openai_whisper` | Native Whisper compatibility |
| `asr_tiron` | Speaker/time-token sequence-to-sequence |
| `asr_qwen3` | Prompted audio-language modeling |
| `asr_vibevoice` | Prompted multimodal sequence-to-sequence |
| `asr_granite_speech` | Multimodal causal language modeling |
| `asr_parakeet_tdt` | Token-and-duration transducer |
| `asr_nemotron` | RNN-T |
| `asr_cohere` | Speech sequence-to-sequence |
| `asr_medasr` | CTC |
| `asr_wav2vec2` | CTC |
| `asr_hubert` | CTC |
| `asr_wavlm` | CTC |
| `asr_moonshine` | Speech sequence-to-sequence |
| `asr_nemo` | Character CTC |
| `asr_speechbrain` | CTC plus attention |
| `asr_funasr` | SANM-CTC with control tokens |
| `asr_espnet` | Hybrid CTC plus attention |
| `asr_wenet` | Hybrid CTC plus bidirectional attention |
| `asr_seamless_m4t_v2` | Multilingual sequence-to-sequence |

Read [speech data](speech-data.md) and the
[training matrix](../models/training-support.md) before choosing records,
checkpoint formats, or trust-gated conversions.

## Safety and troubleshooting

- Never enable a legacy checkpoint trust flag for an unverified file.
- Pin revisions and review checkpoint licenses before downloading.
- A missing transcript usually means a decoding or audio contract failed;
  inspect the returned metadata and provider matrix.
- For long audio, compare chunked output against an unchunked reference before
  using it in production.
- Keep raw audio and transcript provenance; do not train on recordings without
  appropriate rights and consent.

See the [API reference](../reference/api.md) and
[inference notebook](notebook.md) for complete runnable examples.
