---
description: Prepare auditable, leakage-resistant datasets for diverse TTS training families.
---

# Data preparation

There is no universal TTS batch. VoiceHub keeps source records simple, then
delegates semantic target construction to the selected model's dataset,
processor, collator, or specialized training adapter.

This page covers TTS manifests and targets. For transcript manifests,
speech-region annotations, CTC/seq2seq/transducer batches, and clip/frame VAD
labels, use the [ASR and VAD data guide](speech-data.md).

<ol class="vh-process vh-process--six" role="list" aria-label="Data preparation workflow">
  <li>
    <span class="vh-process__number" aria-hidden="true">01</span>
    <strong>Collect recordings</strong>
    <span class="vh-process__detail">Keep source audio with consent, license, and provenance records.</span>
  </li>
  <li>
    <span class="vh-process__number" aria-hidden="true">02</span>
    <strong>Write the manifest</strong>
    <span class="vh-process__detail">Record exact text, audio paths, speakers, sessions, and languages.</span>
  </li>
  <li>
    <span class="vh-process__number" aria-hidden="true">03</span>
    <strong>Validate and normalize</strong>
    <span class="vh-process__detail">Check samples, channels, rates, transcripts, and required metadata.</span>
  </li>
  <li>
    <span class="vh-process__number" aria-hidden="true">04</span>
    <strong>Split without leakage</strong>
    <span class="vh-process__detail">Separate speakers and recording sessions across dataset splits.</span>
  </li>
  <li>
    <span class="vh-process__number" aria-hidden="true">05</span>
    <strong>Run the model processor</strong>
    <span class="vh-process__detail">Apply the selected tokenizer, codec, mel, or conditioning path.</span>
  </li>
  <li>
    <span class="vh-process__number" aria-hidden="true">06</span>
    <strong>Produce training targets</strong>
    <span class="vh-process__detail">Build the tokens, codes, masks, or flow targets used by the recipe.</span>
  </li>
</ol>

## Start with an auditable manifest

JSON Lines keeps each source utterance independent of transient tensors:

```json
{"id":"speaker01-session01-0001","text":"[S1] This transcript matches the recording.","audio":"audio/speaker01-session01-0001.wav","speaker_id":"speaker01","session_id":"session01","language":"en","consent":true,"license":"owned"}
{"id":"speaker01-session02-0001","text":"[S1] Validation uses a different session.","audio":"audio/speaker01-session02-0001.wav","speaker_id":"speaker01","session_id":"session02","language":"en","consent":true,"license":"owned"}
```

Recommended source fields:

| Field                | Purpose                                                        |
| -------------------- | -------------------------------------------------------------- |
| `id`                 | Stable utterance identifier                                    |
| `text`               | Exact spoken transcript, including model-specific speaker tags |
| `audio`              | Path relative to the manifest, or an absolute local path       |
| `speaker_id`         | Speaker-disjoint evaluation and conditioning                   |
| `session_id`         | Prevent adjacent takes from leaking across splits              |
| `language`           | Filtering, balancing, and language-conditioned models          |
| `consent`            | Authorization for the intended voice use                       |
| `license` / `source` | Provenance and redistribution constraints                      |
| `duration`           | Positive seconds for reproducible raw-audio length batching    |
| `num_frames`         | Prepared mel/spectrogram length for VITS or diffusion budgets  |
| `num_tokens`         | Complete prepared text/audio sequence cost for codec/LLM TTS   |

!!! danger "Treat voice data as sensitive"

    Train only on voices authorized for the intended use. Do not put secrets,
    access tokens, or unnecessary personally identifying notes in a manifest.
    Preserve consent and provenance with every derived dataset version.

## Resolve relative audio safely

`TTSDataset` loads JSON, JSON Lines, CSV, and TSV manifests, normalizes common
field aliases, resolves audio paths relative to the manifest, and validates
the selected model's declared record variants:

```python
from voicehub.training import TTSDataset

records = TTSDataset.from_manifest(
    "data/manifest.jsonl",
    model_type="dia",
    validate_files=True,
)
```

Aliases such as `transcript`, `audio_path`, and `wav_path` are converted to
`text` and `audio`. Reference-audio aliases are model-aware: Qwen3-TTS
normalizes them to `ref_audio`, while Higgs Audio and MOSS-TTS preserve the
native `reference_audio` field.
`TTSDataset.from_ljspeech()` reads the common
`id|text|normalized_text` layout. Use `to_jsonl()` to persist a portable
manifest and `resume_fingerprint()` to capture its content and order.

Length fields are optional for ordinary fixed-size batches and required only
when attaching an [architecture optimization profile](tts-optimization.md).
Compute them once during preparation; the training sampler deliberately does
not decode every audio file at startup.

Keep raw recordings immutable. Write normalized audio and resolved split
manifests to a new, versioned prepared-data directory.

## Normalize for the selected processor

This guide uses Dia as a concrete raw-data route. The tutorial's prepared-data
policy is:

- mono;
- exactly 44,100 Hz;
- finite and non-empty; and
- aligned with the transcript.

The Dia adapter rejects unexpected sample rates. File-backed multichannel
audio is downmixed by the adapter, while in-memory waveforms must already be
mono rank-1 tensors or numeric sequences. Enforcing mono files during
preparation keeps the dataset explicit and consistent.

```python
from pathlib import Path

from voicehub.processing import load_native_audio, save_pcm_wave


def prepare_dia_audio(source: str | Path, destination: str | Path) -> Path:
    source = Path(source)
    destination = Path(destination)
    audio = load_native_audio(
        source,
        target_sampling_rate=44_100,
    )
    return save_pcm_wave(
        destination,
        audio.waveform,
        audio.sampling_rate,
    )
```

The native decoder accepts uncompressed PCM WAVE input, averages channels to
mono, performs a band-limited PyTorch resample, and writes a portable 16-bit
PCM WAVE file. It does not require NumPy, SoundFile, Librosa, or Torchaudio.
Decode other containers explicitly before this boundary.

Resampling does not repair clipping, background music, long silence, incorrect
transcripts, or licensing problems. Measure those separately.

## Validate the prepared records

```python
from pathlib import Path

import torch

from voicehub.processing import load_pcm_wave


def validate_dia_records(records: list[dict]) -> None:
    seen = set()
    for index, record in enumerate(records):
        record_id = str(record.get("id", index))
        if record_id in seen:
            raise ValueError(f"Duplicate record id: {record_id}")
        seen.add(record_id)

        if not str(record.get("text", "")).strip():
            raise ValueError(f"{record_id}: empty transcript")
        if record.get("consent") is not True:
            raise ValueError(f"{record_id}: consent is not recorded")

        audio_path = Path(record["audio"])
        if not audio_path.is_file():
            raise FileNotFoundError(f"{record_id}: {audio_path}")

        channels, sample_rate = load_pcm_wave(
            audio_path,
            preserve_channels=True,
        )
        if channels.shape[0] != 1 or sample_rate != 44_100:
            raise ValueError(
                f"{record_id}: expected mono 44100 Hz, received "
                f"{channels.shape[0]} channel(s) at {sample_rate} Hz"
            )
        if channels.numel() == 0 or not torch.isfinite(channels).all():
            raise ValueError(f"{record_id}: audio is empty or non-finite")
```

Add project-specific checks for duration limits, signal-to-noise ratio,
clipping, loudness, transcript normalization, language, speaker balance, and
duplicate content.

## Split by speaker or session

Randomly splitting adjacent clips leaks room tone, microphone response,
background noise, and neighboring takes into validation. Split whole speakers
or recording sessions with the dataset API:

```python
train_records, validation_records = records.train_test_split(
    group_by="session_id",
    validation_fraction=0.1,
    seed=42,
)

train_records.to_jsonl("data/splits/train.jsonl")
validation_records.to_jsonl("data/splits/validation.jsonl")
```

Persist the resulting manifests, split seed, preprocessing revision, and
content hashes.

## Inspect the architecture contract first

TTS architectures do not share one prepared batch. Inspect a generic
architecture contract when building a reusable corpus, or a model contract
before starting a run:

```python
from voicehub.training import get_tts_dataset_spec

contract = get_tts_dataset_spec("f5tts")
print(contract.architecture)       # diffusion
print(contract.readiness)          # preprocessed
print(contract.training_support)   # preprocessed

for variant in contract.variants:
    print(variant.name, variant.required_fields, variant.one_of)
```

The training profile selects this model-specific contract through the lazy
`dataset_spec_factory="module:callable"` boundary. The zero-argument callable
returns a `TTSDatasetSpec`; VoiceHub attaches the registered model type,
training support, and derived readiness. Extensions can therefore register an
exact TTS contract without editing a shared provider table or importing the
model graph.

The six architecture contracts deliberately stop at different boundaries:

| Architecture | Canonical source record | Typical prepared target |
| --- | --- | --- |
| Codec/LLM | Text/audio, conversation, or codec tokens | Packed text/audio tokens, codebook layout, causal labels, validity mask |
| Sequence-to-sequence | Text and target audio | Encoder tokens plus teacher-forced acoustic/codec labels |
| Diffusion/flow | Text/audio/reference conditioning | Clean latent, noisy state, timestep, target, and target-aligned mask |
| VITS/GAN | Text or phonemes and waveform | Text IDs, waveform/spectrogram lengths, posterior inputs, real/fake phase data |
| Acoustic | Text and audio | Mel, duration, pitch, stop, codec, or waveform targets |
| Hybrid | Conversation or text/audio | Explicit component- or phase-specific batches |

`readiness` is intentionally separate from the training objective:

- `integrated-raw` means the model adapter can create its training data from
  one of the contract's non-preprocessed variants.
- `preprocessed` means the objective is integrated but the caller must provide
  the exact tensor variant shown by the model contract.
- `custom` means source-owned preparation or orchestration is still required.
- `unavailable` means the current runtime has no verified trainable graph.

A generic architecture contract may describe a useful raw corpus shape even
when a particular model contract is preprocessed-only. VoiceHub does not
silently promote that generic shape to model support.

## Use an integrated raw-data adapter

Representative integrated raw-data routes still have model-specific
contracts:

| Model | Accepted source record |
| --- | --- |
| Dia | Non-empty `text`; 44.1 kHz audio path or mono rank-1 audio array |
| Orpheus | `text` plus an audio path resampled to 24 kHz by the helper, or SNAC `audio_codes`; optional `voice` |
| LLaSA | `text` plus `audio` or XCodec2 `audio_codes`; completion-only labels |
| Chatterbox | T3: `text` plus raw audio; flow: raw audio alone; both also accept their explicit precomputed tensors |
| ConversationTTS | Raw or tokenized text plus raw audio or Mimi `audio_codes` |
| CSM | Conversation/messages; grouped `texts`/`speaker_ids` with `audios`, concatenated `audio` plus `audio_cut_idxs`, or scalar text/audio |
| IrodoriTTS | `text` plus an in-memory waveform, or a precomputed target latent |
| MOSS-TTS | `text` plus exactly one waveform source or `speech_tokens`; checkpoint families use different codec rates |
| Parler-TTS | Description or text IDs plus an in-memory waveform, DAC codes, or delayed labels |
| Zonos2 | `text` plus 44.1 kHz audio, or undelayed cached DAC frames |
| VoxCPM2 | `text` plus an in-memory 16 kHz waveform or AudioVAE features |
| OmniVoice | `text` plus 24 kHz audio or eight-codebook audio tokens |
| Higgs Audio | `text` plus raw audio or audio codes; reference audio/codes require matching `reference_text` |
| OpenVoice | Linguistically aligned `source_audio` and `target_audio`; reference waveforms or embeddings are optional |
| NeuTTS-Air | `text` plus raw audio or native NeuCodec `audio_codes`; phoneme checkpoints need explicit phonemes or an injected phonemizer |
| SpeechT5 | `text` plus an audio path, `AudioInput`, or waveform mapping; audio is materialized and resampled to 16 kHz |
| VITS | `text` or text IDs plus waveform; the full GAN route also requires the explicit acoustic configuration described in the support matrix |

For adapters exposing `create_training_dataset()`:

```python
train_dataset = training_model.create_training_dataset(train_records)
validation_dataset = training_model.create_training_dataset(
    validation_records
)
```

The model factory accepts a manifest directly:

```python
train_dataset = training_model.create_training_dataset(
    "data/splits/train.jsonl",
    validate_audio_files=True,
)
```

Audit fields such as `id`, `session_id`, `consent`, `license`, `source`, and
nested `metadata` remain available in `TTSDataset` and the source manifest,
but are excluded from generic model keyword arguments. A model-owned prepared
dataset may intentionally project them away. Conditioning fields such as
`speaker_id` and `language` remain available to model-specific processors.

## Inspect a model-owned batch

For Dia, VoiceHub's native processor creates byte-text inputs, delayed decoder
inputs, attention masks, and channel-major masked codec labels:

```python
features = [
    train_dataset[index]
    for index in range(min(2, len(train_dataset)))
]
batch = train_dataset.collate_fn(features)

for name, value in batch.items():
    print(name, getattr(value, "shape", type(value).__name__))
```

Before a long run, confirm:

- label tensors contain trainable positions;
- masks align with their sequences;
- source audio was validated at the processor sample rate;
- codec channel and codebook order match the source implementation; and
- frozen target encoders do not receive gradients.

## Supply preprocessed tensors when required

Some integrations expose a verified objective but do not yet own raw-data
preparation. The generic collator pads structure; it does not invent semantic
targets:

OuteTTS is one such explicit boundary. Each V3 speaker profile must already
contain word timings, equal-length two-codebook DAC codes, per-word features,
and global features:

```python
outetts_records = [
    {
        "speaker_profile": {
            "interface_version": 3,
            "text": "Hello.",
            "words": [
                {
                    "word": "Hello.",
                    "duration": 0.32,
                    "c1": [101, 231],
                    "c2": [77, 912],
                    "features": {
                        "energy": 28,
                        "spectral_centroid": 42,
                        "pitch": 51,
                    },
                }
            ],
            "global_features": {
                "energy": 28,
                "spectral_centroid": 42,
                "pitch": 51,
            },
        }
    }
]

train_dataset = training_model.create_training_dataset(outetts_records)
```

Feature values must be integers in `[0, 100]`; each code must be in
`[0, 1024]`. VoiceHub validates the complete profile and constructs exact V3
completion-only labels. It rejects raw audio instead of inventing timestamps
or acoustic features that differ from the author pipeline. A preparation
service may also persist exact `input_ids` and `labels`, using `-100` only for
masked label positions.

Fish Speech S2 has a different prepared boundary. Each record contains
integer `tokens` and `labels` shaped
`[num_codebooks + 1, sequence_length]`—11 channels for S2-Pro. Channel zero
holds text/protocol/semantic token IDs; channels 1 through 10 hold aligned
ModifiedDAC IDs. Labels are aligned to the prediction at the same position,
so dataset preparation must not add another causal shift:

```python
fish_records = [
    {
        "tokens": prepared_inputs,  # integer tensor [11, time]
        "labels": prepared_labels,  # integer tensor [11, time]
    }
]
train_dataset = training_model.create_training_dataset(
    fish_records,
    max_length=4096,
)
batch = train_dataset.collate_fn([train_dataset[0]])
```

The collator pads channel zero with the checkpoint's end-of-text token, pads
codec channels with zero, pads every label channel with `-100`, and emits
`attention_masks` where `True` means padding. VoiceHub rejects raw legacy Fish
protobuf paths: convert them into this explicit channel-first contract before
training. ModifiedDAC is the frozen offline tokenizer and is not an optimizer
phase.

OpenVoice V2 uses paired waveform records rather than text labels or codec
tokens:

```python
openvoice_records = [
    {
        "source_audio": "speaker-a/line-004.wav",
        "target_audio": "speaker-b/line-004.wav",
        "source_reference_audio": "speaker-a/reference.wav",
        "target_reference_audio": "speaker-b/reference.wav",
        "sampling_rate": 22_050,
    }
]
```

The source and target utterances must carry the same words and should be
temporally aligned. Split by target speaker and recording session before
creating pairs so validation does not reuse reference identity or room
acoustics. The native collator deliberately keeps audio as variable-length
tuples; the OpenVoice processor then resamples, computes the released
513-channel magnitude spectrogram, records exact frame/sample lengths, and
right-pads. Precomputed `[256, 1]` source/target embeddings may replace
reference waveforms, but do not mix present and missing embeddings within one
batch.

```python
from voicehub.training import DataCollatorForTTSTraining, TTSFieldSchema

collator = DataCollatorForTTSTraining(
    field_schemas={
        "model_inputs.mel": TTSFieldSchema(
            sequence_dim=-1,
            padding_side="right",
            length_field="mel_lengths",
            mask_field="mel_mask",
            pad_to_multiple_of=8,
        ),
    },
)
```

A backend-shaped record may look like:

```python
record = {
    "model_inputs": {
        "text_tokens": text_tokens,
        "speaker_embedding": speaker_embedding,
        "mel": mel,
    },
    "labels": target,
}
```

The meanings, shapes, masks, and loss target come from the selected model
recipe—not from the generic field names.

## Preparation changes by family

| Family                  | Preparation boundary                                                                                                                  |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| Causal/codec LM         | Frame text and codec tokens, apply codebook delays, and mask prompt positions. Target codecs normally remain frozen.                  |
| Encoder-decoder LM      | Build encoder inputs, delayed decoder inputs, decoder masks, and teacher-forced labels through the model processor.                  |
| Flow matching/diffusion | Define clean samples, noise, sampled time, conditioning, masks, and the exact velocity or noise target expected by the source loss.   |
| VITS/GAN                | Prepare phonemes, lengths, spectrograms, waveforms, alignments, speakers/languages, and phase-specific real/fake batches.             |
| Hybrid/composite        | Prepare the union of component schemas and preserve explicit detach boundaries between independently optimized phases.                |

VoiceHub never fabricates a generic flow target or collapses a
generator/discriminator recipe into waveform regression.

## Data readiness checklist

- Consent, allowed uses, source, and license are recorded.
- Raw recordings are immutable.
- Prepared audio follows the selected processor's sample-rate and channel
  contract.
- Transcripts match the audio and preserve required speaker/control tokens.
- Train and validation groups do not share speakers or recording sessions.
- Manifest and preprocessing revisions are content-addressed.
- One collated batch has correct shapes, masks, labels, and finite values.
- Frozen codecs, vocoders, and speaker encoders are documented.

Continue with [training](training.md) once one complete batch satisfies the
selected model's contract.
