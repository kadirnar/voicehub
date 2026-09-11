---
description: Shared pretrained speech-model bases, normalized outputs, and portable lifecycle contracts.
---

# Models

VoiceHub model wrappers expose one pretrained lifecycle across text to speech,
automatic speech recognition, voice activity detection, and audio codecs. Task-specific
classes keep their input and output semantics explicit while sharing lazy
loading, saving, training validation, and runtime-state transitions.

Create an unloaded model from configuration when you want to inspect or adjust
the wrapper before allocating checkpoint-backed state:

```python
from voicehub import AutoConfig, AutoModelForTextToSpeech

config = AutoConfig.for_model(
    "parlertts",
    name_or_path="parler-tts/parler-tts-mini-v1",
)
model = AutoModelForTextToSpeech.from_config(config)
assert not model.is_loaded
```

Use the task-specific auto classes for normal discovery and construction. See
the [Auto Classes catalog](../models/providers/index.md) for every registered
model and the [full API reference](api.md) for training, optimization, serving,
and extension contracts.

## `PreTrainedSpeechModel`

[`PreTrainedSpeechModel`](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/tts.py)
is the public marker shared by every pretrained speech wrapper. It deliberately
does not implement a token-embedding or language-model utility mixin. Speech
models may own waveform processors, acoustic encoders, codecs, vocoders, and
streaming state, so those operations remain on the relevant task base or model
integration.

The task bases expose the same core lifecycle:

| Member | Contract |
| --- | --- |
| `from_pretrained(source, **kwargs)` | Restore configuration and portable VoiceHub metadata from a local path or Hub repository without loading weights unless `lazy_load=False` |
| `load()` | Allocate the checkpoint-backed runtime and enter inference mode once |
| `load_for_training()` | Validate the configured training path and restore a differentiable runtime |
| `validate_training_support()` | Check the exact backend and checkpoint before model allocation |
| `save_pretrained(directory, include_native_export=True)` | Save portable metadata and, when supported, a namespaced native artifact |
| `is_loaded` | Report whether checkpoint-backed runtime state exists |

Loading and configuration resolution are lazy at the package boundary.
Inspecting the public classes or registry does not construct a model.

## Task-specific pretrained models

The source implementations are split by input contract:

- [`PreTrainedTTSModel`](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/tts.py)
  owns text processing, `TTSGenerationConfig`, `generate()`, and `TTSOutput`.
- [`PreTrainedAudioModel`, `PreTrainedASRModel`, and `PreTrainedVADModel`](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/audio.py)
  share audio loading, inference configuration, streaming sessions, and the
  ASR/VAD lifecycle.

| Base class | Main input | Primary call | Normalized output |
| --- | --- | --- | --- |
| `PreTrainedTTSModel` | Text plus optional conditioning | `generate()` | `TTSOutput` |
| `PreTrainedASRModel` | Audio or an audio path | `transcribe()` | `ASROutput` |
| `PreTrainedVADModel` | Audio or an audio path | `detect()` | `VADOutput` |
| `PreTrainedCodecModel` | Audio or a codec payload | `encode()` / `decode()` | `CodecOutput` / `AudioOutput` |

`PreTrainedAudioModel.forward()` provides the common audio request path.
ASR and VAD keep separate task classes so a wrapper cannot silently return the
wrong output type. `PreTrainedTTSModel.forward()` similarly validates that the
model integration returned `TTSOutput`.

Use `AutoModel`, `AutoModelForTextToSpeech`,
`AutoModelForSpeechRecognition`, `AutoModelForVoiceActivityDetection`, or
`AutoModelForAudioCodec`
instead of instantiating an abstract pretrained base directly.

## Model outputs

The public output dataclasses live in
[`voicehub/outputs.py`](https://github.com/kadirnar/voicehub/blob/main/voicehub/outputs.py).
They provide named fields, integer and string indexing, `keys()`, `to_dict()`,
and a compact tuple view without making a provider-specific object public.

| Output | Required fields | Optional evidence |
| --- | --- | --- |
| `AudioOutput` / `TTSOutput` | `audio`, `sample_rate` | Output path and model metadata |
| `CodecOutput` | `codes`, `sample_rate`, `length`, `model_type` | Model-owned encoding metadata |
| `ASROutput` | `text`, ordered `segments` | Language, duration, confidence, words, speakers, and metadata |
| `VADOutput` | Ordered, non-overlapping speech `segments` | Duration, sample rate, frame probabilities, and metadata |

```python
from voicehub import TTSOutput

output = TTSOutput(audio=[0.0, 0.1, 0.0], sample_rate=24_000)
audio, sample_rate = output
assert output["sample_rate"] == sample_rate
```

Training uses `SpeechTrainingOutput` and its backward-compatible
`TTSTrainingOutput` specialization. Their first populated value is `loss`,
matching the shared Trainer contract while allowing speech-specific fields.

## Loading, saving, and sharing

`from_pretrained()` accepts a VoiceHub artifact, a direct checkpoint when the
task factory receives `model_type`, or a Hub repository. `save_pretrained()`
writes only the portable files supported by the wrapper:

| Artifact | Purpose |
| --- | --- |
| `config.json` | Serializable architecture and checkpoint identity |
| `generation_config.json` or task inference config | Saved request defaults |
| `processor_config.json` | Text or audio preprocessing metadata |
| `model_state.safetensors` | Optional portable trained state |
| `native/` | Optional provider-native export owned by the integration |

VoiceHub does not expose a public `push_to_hub()` method. Saving locally and
uploading with the Hub client are separate operations until VoiceHub can
guarantee one registry-wide upload contract for configuration, processors,
portable state, native artifacts, provenance, and license files. This is an
explicit support boundary, not an implied sharing pass.

For exact artifact and resume behavior, see
[Save, load, and resume boundaries](api.md#save-load-and-resume-boundaries).
