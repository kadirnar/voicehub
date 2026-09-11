---
description: Define source manifests and model-shaped batches for ASR and VAD training without hiding alignment semantics.
---

# ASR and VAD data

VoiceHub separates auditable source records from model-shaped training
batches. A manifest records what the sample means; a processor, dataset, or
specialized adapter owns tokenization, feature extraction, alignment, padding,
and the exact loss inputs required by one architecture.

There is no safe universal transformation from a transcript or list of speech
regions into every CTC, sequence-to-sequence, transducer, or VAD target.

## Shared audio contract

Inference and data preparation accept:

- a local audio path;
- a NumPy array or Torch tensor plus `sampling_rate`;
- a mapping containing `array`, `waveform`, `audio`, or `input_values` and a
  sample rate; or
- `AudioInput(waveform=..., sampling_rate=...)`.

`load_audio()` materializes finite float32 mono audio and optionally resamples
it:

```python
from voicehub import load_audio

audio = load_audio(
    "recordings/session-004.wav",
    target_sampling_rate=16_000,
)
print(audio.waveform.shape, audio.sampling_rate, audio.duration)
```

Keep the original file, sample rate, checksum, and provenance in the source
manifest. Resampling is a derived preprocessing step, not a replacement for
the source record.

`SpeechDataset` is a dependency-light, immutable-indexed view when records are
already in memory:

```python
from voicehub.training import SpeechDataset

dataset = SpeechDataset(
    records,
    required_fields=("audio", "text"),
    transform=processor_transform,
)
print(len(dataset), dataset.column_names)
```

It copies source mappings, validates required fields, and applies the optional
transform when an item is read. It intentionally does not decode audio or
invent model targets; keep those operations in the selected processor or
training adapter.

## ASR source records

A practical JSON Lines record is:

```json
{
  "id": "session-004-utterance-0012",
  "audio": "audio/session-004/0012.flac",
  "text": "The verified reference transcript.",
  "language": "en",
  "speaker_id": "speaker-018",
  "session_id": "session-004",
  "duration": 5.42,
  "split": "train",
  "license": "dataset-specific",
  "consent_id": "consent-018"
}
```

Required semantic values are the audio and verified transcript. Stable IDs,
language, speaker/session identity, duration, provenance, and consent make the
dataset auditable and prevent leakage.

For translation ASR, store source-language transcription and target-language
text in separate named fields. Do not overload `text` with two different
semantics.

### Build a validated `ASRDataset`

`ASRDataset` is the public corpus boundary for fine-tuning. It accepts Python
mappings directly, or reads JSON, JSON Lines, CSV, and TSV manifests without
importing a tensor framework:

```python
from voicehub.training import ASRDataset

records = ASRDataset.from_manifest(
    "data/train.jsonl",
    model_type="asr_wav2vec2",
    validate_files=True,
)
print(records.spec.architecture, records.variant_names)
```

JSON Lines is a useful default because each line is independently inspectable:

```json
{"audio": "clips/000001.wav", "text": "A verified transcript.", "speaker_id": "spk-01"}
{"audio": "clips/000002.wav", "text": "Another transcript.", "speaker_id": "spk-02"}
```

CSV and TSV use the same column names:

```csv
audio,text,speaker_id
clips/000001.wav,A verified transcript.,spk-01
clips/000002.wav,Another transcript.,spk-02
```

Relative audio paths resolve against the manifest directory unless `root=` is
provided. Common upstream names are normalized at the boundary:
`audio_path`, `audio_filepath`, `wav_path`, `wav`, `waveform`, `speech`,
`file`, or `path` become `audio`; `transcript`, `transcription`, `sentence`,
or `target_text` become `text`; `sample_rate` becomes `sampling_rate`; and
`lang` becomes `language`. Pass `aliases={"recording": "audio"}` for a
corpus-specific column. A record containing both an alias and its canonical
field is rejected instead of silently choosing one.

`validate_files=True` checks path existence. Audio decoding, resampling,
feature extraction, tokenization, decoder prompts, CTC blanks, and transducer
targets remain the selected model's responsibility.

A model accepts a manifest path directly and applies the same normalization:

```python
training_dataset = model.create_training_dataset(
    "data/train.tsv",
    validate_audio_files=True,
)
```

Use `data_root=`, `data_aliases=`, and `validate_records=` on
`create_training_dataset()` when the manifest needs non-default handling.

### Create data without writing a manifest first

For a small corpus, place a UTF-8 transcript beside every PCM WAV file:

```text
data/clips/000001.wav
data/clips/000001.txt
data/clips/000002.wav
data/clips/000002.txt
```

Then pair them by stem:

```python
from voicehub.training import ASRDataset

records = ASRDataset.from_audio_folder(
    "data/clips",
    model_type="asr_whisper",
    metadata={"language": "en"},
)
records.to_jsonl("data/train.jsonl")
```

`from_audio_folder()` scans recursively by default, requires every transcript
sidecar to be non-empty, and discovers materialized `.wav` files. Audio
decoding remains model-owned; native preprocessors currently decode PCM WAVE,
while a custom transform can decode other WAV encodings before training.
Change `transcript_extension=` when sidecars use another suffix.

Kaldi/ESPnet-style directories containing `wav.scp` and `text` can be loaded
without executing shell commands:

```python
records = ASRDataset.from_kaldi(
    "data/kaldi/train",
    model_type="asr_espnet",
    validate_files=True,
)
```

Utterance IDs must match across both files. `wav.scp` shell pipelines are
rejected; materialize them as audio files before constructing a portable
VoiceHub dataset.

### Inspect the model contract

Every ASR training profile exposes its accepted record shapes before a model
or checkpoint is loaded:

```python
from voicehub.training import ASRDataArchitecture, get_asr_dataset_spec, get_training_spec, list_asr_dataset_specs

contract = get_asr_dataset_spec("asr_qwen3")
same_contract = get_training_spec("asr_qwen3").dataset_spec

print(contract.architecture)
print(contract.sample_rate)
print([variant.name for variant in contract.raw_variants])
print([variant.name for variant in contract.preprocessed_variants])

generic_ctc = get_asr_dataset_spec(
    architecture=ASRDataArchitecture.CTC,
)
all_asr_contracts = list_asr_dataset_specs()
```

After constructing a model,
`model.validate_training_support().dataset_spec` returns the same
model-specific contract. `ASRDataReadiness` distinguishes `integrated-raw`,
`preprocessed`, `custom`, and `unavailable`; `training_support` reports the
corresponding registered training boundary.

Each `ASRRecordVariant` declares exact required fields, alternative `one_of`
fields, excluded fields, and whether it is already model-shaped. Validation
checks the portable record schema. The processor still validates tensor
shape, dtype, vocabulary, duration, and sample rate.

Some upstream manifests use architecture-specific nested records or control
tokens. Their `ASRDatasetSpec` declares `field_aliases` and a lazy
`record_normalizer`; the implementation remains in the corresponding
architecture package. Inspect those fields to audit preprocessing without
loading the normalizer or model graph.

The profile's `dataset_spec_factory` is the lazy `module:callable` boundary for
the full model-specific contract. The zero-argument callable must return an
`ASRDatasetSpec`; VoiceHub attaches the registered model type, training support,
and derived readiness. This lets an extension register a new ASR data contract
without editing a shared provider table, while keeping registry inspection free
of PyTorch and checkpoint imports.

### Architecture-specific records

The simplest raw record remains `{"audio": ..., "text": ...}`, but metadata
and cached tensor forms are architecture-specific:

| Family or provider | Integrated source record | Important model-owned preparation |
| --- | --- | --- |
| CTC (`asr_wav2vec2`, `asr_hubert`, `asr_wavlm`, `asr_medasr`, `asr_nemo`) | `audio`, `text` | Waveform or log-mel extraction, tokenizer labels, blank ID, input/label lengths |
| Whisper and speech seq2seq (`asr_whisper`, compatibility keys, Moonshine) | `audio`, `text`; optional `language` and `task` for Whisper | Encoder features or waveform values plus teacher-forced decoder labels |
| Tiron | `audio`, inline speaker/timestamp target in `text`; optional `language` | Whisper features plus grammar-validated speaker and 20 ms timestamp labels |
| Qwen3-ASR | `audio`, `text`; optional `context`, `prompt`, `language` | Multimodal prompt tokens, log-mel features, feature mask, completion-only labels |
| Granite Speech | `audio`, `text`; optional `prompt`; no `language` field | Language or translation guidance belongs in the prompt; the adapter builds prompt/completion tokens and acoustic features |
| VibeVoice-ASR | `audio`, structured `segments`; or `audio`, serialized segment `text`; optional context | 24 kHz audio, continuous speech encoders, speaker/timestamp/content serialization, completion-only labels |
| Parakeet TDT | `audio`, `text` | Log-mel inputs, blank-prefixed decoder input, native token-duration objective |
| Nemotron RNN-T | `audio`, `text`; optional `language` | Acoustic inputs, language prompt IDs, label lengths, blank-prefixed predictor input, native transducer objective |
| Cohere Transcribe | `audio`, `text`, `language`; optional `punctuation` | Language/punctuation prompt and teacher-forced decoder labels; batches are homogeneous for both controls |
| SeamlessM4T-v2 | `audio`, `text`, plus `target_language` or `language` unless configured on the model | Target-language-conditioned encoder/decoder labels; batches are homogeneous by target language |
| SpeechBrain CRDNN | `audio`, `text` | Waveforms, BOS/EOS attention targets, parallel CTC targets, staged joint objective |
| SenseVoiceSmall (`asr_funasr`) | `audio`, `text`, `language`; optional `emotion`, `event`, `use_itn` | Fbank features and four rich-control query targets plus CTC transcript labels |
| ESPnet Transformer | `audio`, `text`, or cached `features`, `text` | Joint CTC/attention labels, SpecAugment, and source-shaped hybrid objective |
| WeNet U2++ | `audio`, `text` | Waveform/frontend lengths and shared CTC/forward/reverse attention targets |

Prepared variants are also available for caching expensive preprocessing.
Inspect `contract.preprocessed_variants` instead of assuming field names:
Qwen, Granite, VibeVoice, TDT, RNN-T, SpeechBrain, SenseVoice, ESPnet, and
WeNet do not share one model-ready tensor layout.
Compatibility dispatchers can list a union of cached shapes; the selected
checkpoint delegate determines which one is valid at runtime.

SenseVoice's published JSONL names (`source`, `target`, `text_language`,
`emo_target`, `event_target`, and `with_or_wo_itn`) are accepted directly.
Published control spellings such as `<|en|>`, `<|NEUTRAL|>`, and
`<|woitn|>` are normalized to the canonical fields and values.

SeamlessM4T-v2 also accepts the original repository's nested source/target
shape and normalizes it to the flat contract:

```json
{
  "source": {
    "audio_local_path": "clips/000001.wav",
    "lang": "eng"
  },
  "target": {
    "text": "The target-language transcript.",
    "lang": "eng"
  }
}
```

### Split, export, and resume safely

Create the split before windowing or augmentation, preferably using the
strongest leakage boundary:

```python
train_records, validation_records = records.train_test_split(
    validation_fraction=0.1,
    seed=42,
    group_by="speaker_id",
)

train_records.to_jsonl("data/frozen/train.jsonl")
print(train_records.resume_fingerprint())
```

The split is deterministic. A grouped split keeps an entire speaker, session,
or source recording on one side. `resume_fingerprint()` includes normalized
record content and order; lazy transforms require an explicit stable
`transform_fingerprint`.

For Cohere, the Trainer automatically batches records with the same
`language` and `punctuation` values. For SeamlessM4T-v2, it batches the same
`target_language` together. The dataset's epoch-aware batch sampler is
deterministic and checkpointable. Do not bypass it with a custom DataLoader
unless that loader preserves the same grouping rule.

## VAD source records

Represent speech annotations in seconds on the original recording timebase:

```json
{
  "id": "meeting-007",
  "audio": "audio/meeting-007.wav",
  "duration": 184.37,
  "segments": [
    {"start": 0.82, "end": 4.19, "label": "speech"},
    {"start": 5.03, "end": 9.44, "label": "speech"}
  ],
  "session_id": "meeting-007",
  "split": "validation",
  "annotation_revision": "vad-review-2"
}
```

Validate that regions are ordered, non-overlapping, non-negative, and bounded
by the file duration. Keep ambiguous, overlapping-speaker, music, and
non-speech labels when the selected recipe uses them; reducing everything to
one binary flag too early can discard supervision.

### Clip-classification examples

A clip classifier consumes one label per extracted window:

```python
{
    "input_values": waveform_window,
    "labels": 1,  # speech class
}
```

The window duration, hop, class mapping, and boundary sampling policy are part
of the dataset recipe and must be saved with the run.

### Frame-classification examples

A frame model requires targets aligned to its output timebase:

```python
{
    "input_values": waveform,
    "labels": frame_labels,
    "frame_mask": valid_frames,
}
```

`frame_labels` cannot be padded or interpolated independently of the model's
feature stride. Use the checkpoint's feature extractor and preserve a mask for
padded frames.

## Model-shaped ASR batches

| Family | Common fields | Non-negotiable semantics |
| --- | --- | --- |
| CTC | `input_values` or `input_features`, `attention_mask`, `labels` | Processor vocabulary, blank index, reduction, input/target lengths, and the checkpoint-specific label padding/ignore value must match |
| Speech seq2seq | `input_features`, optional encoder mask, decoder `labels` | Decoder start/language/task tokens and label padding belong to the checkpoint processor |
| RNN-T | acoustic inputs and lengths, prediction-network targets and lengths | Use the backend's transducer loss and blank/alignment conventions |
| TDT | RNN-T-like inputs plus token/duration targets required by the model | Duration topology and loss weights remain backend-native |
| Upstream native | Provider task/configuration batch | Preserve the upstream data module, augmentation, tokenizer, and distributed recipe |

VoiceHub's CTC, RNN-T, and TDT adapters require a backend-native scalar loss.
They do not guess blank placement, alignment topology, or duration losses from
arbitrary logits.

## Collate variable-length audio fields

`DataCollatorForAudioTraining` stacks equal tensors and pads declared
variable-length dimensions. Use `AudioFieldSchema` when a field's time
dimension is ambiguous:

```python
from voicehub.training import AudioFieldSchema, DataCollatorForAudioTraining

collator = DataCollatorForAudioTraining(
    label_pad_token_id=-100,
    field_schemas={
        "input_values": AudioFieldSchema(
            sequence_dim=0,
            padding_value=0.0,
            length_field="input_lengths",
            mask_field="attention_mask",
            pad_to_multiple_of=320,
        ),
        "labels": AudioFieldSchema(
            sequence_dim=0,
            padding_value=-100,
            length_field="label_lengths",
        ),
    },
)
```

Dotted paths such as `model_inputs.input_features` describe nested batches.
`sequence_dim=-1` is useful for time-last features. Set `allow_missing=True`
only when a missing value genuinely means a zero-length sequence.

For frame classification, declare labels and their mask on the same padded
time dimension:

```python
frame_collator = DataCollatorForAudioTraining(
    field_schemas={
        "input_values": AudioFieldSchema(
            sequence_dim=0,
            mask_field="attention_mask",
        ),
        "labels": AudioFieldSchema(
            sequence_dim=0,
            padding_value=-100,
            mask_field="frame_mask",
        ),
    },
)
```

## Split before windowing

Split by the strongest leakage boundary before chunking:

1. speaker, conversation, recording session, or source file;
2. then train, validation, and test assignment;
3. then normalization, windowing, augmentation, and feature extraction.

Randomly splitting windows from one recording leaks noise, room acoustics,
speaker identity, and neighboring content across evaluation boundaries.

ASR evaluation has two separate paths. When the evaluation dataset contains
raw `text`/`transcript`/`transcription` references, the training adapter
preprocesses them and the Trainer reports the model's native teacher-forced
`eval_loss`. That proves the differentiable objective works; it is not WER.
WER or CER requires decoded hypotheses, a declared generation/beam policy,
and a documented reference-normalization policy. Supply a model-specific
decoding metric path or `compute_metrics` that compares generated transcripts.
The SpeechBrain recipe is a specialized exception that performs its published
validation decoding and reports corpus WER for its scheduler.

VAD evaluation should include a boundary-aware metric or false-alarm/miss
duration, not only frame accuracy on heavily imbalanced silence.

## Validate before training

For every record:

- verify the audio exists, is decodable, finite, and non-empty;
- compare recorded and decoded duration;
- reject empty or unverified ASR transcripts;
- validate VAD boundaries against duration;
- record language and annotation policy;
- retain speaker/session grouping keys;
- confirm consent and license scope; and
- freeze the manifest and split revision used by the run.

For one collated batch:

- inspect tensor shapes, dtypes, masks, and padded values;
- decode ASR labels back to text where possible;
- project VAD frame labels back onto the waveform;
- require a finite scalar loss with `requires_grad=True`; and
- confirm the intended parameters, and only those parameters, receive
  gradients.

Continue with [ASR inference](speech-recognition.md),
[VAD inference](voice-activity-detection.md), or the
[provider and training matrix](../models/asr-vad-support.md).
