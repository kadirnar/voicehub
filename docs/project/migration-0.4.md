---
description: Breaking import changes, architecture analysis, and codec usage for VoiceHub 0.4.
---

# Migrate to VoiceHub 0.4

Version 0.4 intentionally breaks internal import paths and removes the legacy
factory. The small root API handles four tasks; specialist APIs live in their
own namespaces. See [Library architecture](../concepts/architecture.md) for the
new layout.

## What the code review found

The original tree contained 3,169 Python files and approximately 773,000 Python
lines across the library, tests, and scripts. The review parsed every Python
file and inspected the shared lifecycle, imports, registries, task wrappers,
checkpoint code, and tests in detail. This was a structural audit, not a
line-by-line mathematical review of every vendored network.

The main sources of complexity were concrete:

- TTS and audio input models duplicated pretrained loading, saving, processor
  attachment, portable state restoration, and mode transitions.
- A model's wrapper and native graph were owned by different package trees.
  Several families also had separate native wrapper packages and forwarding
  modules.
- The generic, task-specific, and legacy factories repeated dispatch behavior.
- The package root exposed 259 symbols spanning training, datasets, serving,
  optimization, internals, and ordinary inference.
- Codecs existed as shared TTS components without an independent task API.

The rewrite moves native graphs beside their model, removes forwarding layers
and duplicate native wrappers, consolidates the lifecycle and factories, and
reduces root exports to 39. All 68 existing registered integrations remain;
DAC and EnCodec bring the total to 70. The native model mathematics and upstream
reference sources are retained, with their provenance and legal notices.

## Import changes

| Previous import | New import |
| --- | --- |
| `voicehub.configuration_utils` | `voicehub.configuration` |
| `voicehub.modeling_utils.PreTrainedTTSModel` | `voicehub.PreTrainedTTSModel` |
| `voicehub.audio_modeling_utils` | `voicehub.models.audio` |
| `voicehub.modeling_outputs` | `voicehub.outputs` |
| `voicehub.processing_utils` | `voicehub.processing.processor` |
| `voicehub.generation_configuration` | `voicehub.generation.configuration` |
| `voicehub.trainer`, `voicehub.training_args` | `voicehub.training` |
| `voicehub.trainer_callback` | `voicehub.training.callbacks` |
| `voicehub.trainer_utils` | `voicehub.training.utils` |
| `voicehub.data_collator` | `voicehub.training.data_collator` |
| `voicehub.integrations` | `voicehub.training.integrations` |
| `voicehub.models.registry` | `voicehub.registry` |
| `models/<family>/configuration_<family>.py` | `models/<family>/configuration.py` |
| `models/<family>/modeling_<family>.py` | `models/<family>/modeling.py` |
| `architectures/<graph>/` | `models/<owning-family>/native/` |
| `architectures` metadata exports | `voicehub.runtime` |
| `voicehub.Trainer`, `voicehub.TrainingArguments` | `voicehub.training.Trainer`, `voicehub.training.TrainingArguments` |
| Root optimization exports | `voicehub.optimization` |
| Root dataset exports | `voicehub.training` or `voicehub.training.datasets` |

Graph and model family names need not be identical. For example, the old
`architectures/whisper/` is now `models/asr_whisper_native/native/`, and
`architectures/qwen3_tts/` is now `models/qwen3tts/native/`. CosyVoice, OmniVoice,
VoxCPM, and XTTS use their canonical model packages instead of duplicate
`<family>_native` wrapper packages.

## One loading API

Replace the removed model-type-first `AutoInferenceModel`:

```python
from voicehub import AutoModelForTextToSpeech

model = AutoModelForTextToSpeech.from_pretrained(
    "checkpoints/my-tts",
    model_type="vits",
    device="cpu",
)
```

`AutoModel.from_pretrained(...)` accepts any registered task. The four task
factories add task validation. `from_config(...)`, `load()`, and
`save_pretrained(...)` use the shared lifecycle. Use `model_type=` for model
identifiers and the first argument for a checkpoint source.

Use `AutoModel.available_models()` or `list_model_specs()` for all models and
`list_model_specs(task="tts")` for a filtered view. Aliases `stt`, `asr`, `vad`,
and `codec` are supported.

## Codec usage

```python
from voicehub import AutoModelForAudioCodec, pipeline

codec = AutoModelForAudioCodec.from_pretrained(
    "descript/dac_44khz",
    model_type="dac",
    device="cpu",
)
encoded = codec.encode("speech.wav")
waveform = codec.decode(encoded)
codec.save_pretrained("checkpoints/dac")

codec = AutoModelForAudioCodec.from_pretrained("checkpoints/dac", device="cpu")
codec_pipeline = pipeline("audio-codec", model=codec)
encoded = codec_pipeline("speech.wav")
waveform = codec_pipeline.decode(encoded)
```

Pass `sampling_rate=` with in-memory arrays. Two-dimensional input means
channels by samples; batched mono audio needs `[B, 1, T]`. Encoded payloads are
specific to a codec and its checkpoint. Do not interchange codes between
checkpoints even when their model type matches.

DAC accepts its supported Hugging Face layout and native exports. EnCodec
accepts native Safetensors exports or explicitly trusted pinned Meta artifacts;
its loader does not support arbitrary Transformers-format checkpoints.

## Checkpoint and validation boundaries

Saved VoiceHub configurations keep their original model identifiers. Standard
weight files remain unchanged. Configurations containing custom import strings
must use the new module paths; Python pickles referencing removed modules are
not a portable migration format.

Regression coverage includes existing TTS/STT/VAD contracts and actual small
DAC/EnCodec graphs, mono and stereo batches, segmented scales, task rejection,
and exact native save/reload results. Released weights, GPU execution, training
quality, and speech quality still require their model-specific gates. Historical
benchmark records retain the version that produced them; they are not new
measurements of 0.4.
