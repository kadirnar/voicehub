---
description: The shared model lifecycle and four task contracts in VoiceHub 0.4.
---

# Library architecture

VoiceHub has one model registry, one pretrained lifecycle, and four task contracts:
TTS, speech recognition (STT/ASR), VAD, and audio codecs. All model families use
`AutoConfig`, `AutoModel`, `AutoProcessor`, and `pipeline`.

## Package structure

```text
voicehub/
├── auto.py                 # One factory implementation, optional task constraints
├── configuration.py        # Serializable configuration
├── model.py                # Load, save, device, mode, and inference lifecycle
├── outputs.py              # Audio, transcription, activity, and encoded outputs
├── registry.py             # Lazy model registration and lookup
├── tasks.py                # Four task identifiers and aliases
├── pipelines.py            # Small task conveniences
├── models/
│   ├── catalog.py          # Built-in model declarations
│   ├── base.py             # Shared runtime and optimization interfaces
│   ├── tts.py              # Text → speech
│   ├── audio.py            # Speech → text or activity segments
│   ├── codec.py            # Audio ↔ codec-owned representation
│   └── <family>/
│       ├── __init__.py     # Lazy family exports
│       ├── configuration.py
│       ├── modeling.py     # Task adapter and checkpoint loading
│       ├── processing.py   # Only when the family needs its own processor
│       └── native/         # Network, conversion, metadata, and legal notices
├── runtime/                # Lazy graph metadata and component references
├── components/             # Reused codecs, vocoders, and network components
├── processing/             # Audio and text preparation
├── generation/             # Generation configuration and sampling
├── training/               # Trainer, arguments, callbacks, adapters, objectives
└── optimization/           # Optional runtime transformations
```

Only the small inference and registration surface is exported from `voicehub`.
Training and optimization have their own namespaces. Existing model-local
training modules and upstream reference sources stay with their model family.
Shared neural utilities and checkpoint helpers retain their existing dedicated
modules; the tree above shows the architectural boundaries, not every file.

## Task contracts

| Task | Factory | Implementation hook | Public result |
| --- | --- | --- | --- |
| TTS | `AutoModelForTextToSpeech` | `_generate(text, ...)` | `TTSOutput` |
| STT / ASR | `AutoModelForSpeechRecognition` | `_transcribe(audio, ...)` | `ASROutput` |
| VAD | `AutoModelForVoiceActivityDetection` | `_detect(audio, ...)` | `VADOutput` |
| Codec | `AutoModelForAudioCodec` | `_encode(waveform, ...)`, `_decode(codes, ...)` | `CodecOutput`, `AudioOutput` |

Each task base inherits `PreTrainedSpeechModel`. A model implements its loading
hook and task operation. Shared code owns configuration, lazy loading, device
selection, checkpoint persistence, inference settings, and mode transitions.
A new model does not copy `from_pretrained()` or `save_pretrained()`.

Codecs preserve batches and channels. Input tensors may have shapes `[T]`,
`[C, T]`, or `[B, C, T]`; arrays require a sample rate. `CodecOutput.codes` is an
opaque model-owned payload: a tensor, segmented frames with scales, or another
representation. The shared contract imposes no universal codebook layout.
Decoding restores the original encoded length and returns `[B, C, T]` audio.

## Loading and discovery

```python
from voicehub import AutoModel, list_model_specs

for spec in list_model_specs(task="codec"):
    print(spec.model_type)

model = AutoModel.from_pretrained(
    "descript/dac_44khz",
    model_type="dac",
    device="cpu",
)
```

The registry resolves a lazy import reference, the factory creates the registered
configuration, and the task adapter is constructed with `model=None`. The first
operation or explicit `load()` builds the native graph. Merely importing the
package, listing models, and constructing configurations do not import PyTorch.

`AutoModel` uses the same implementation as the four constrained factories.
A task factory rejects the wrong task before importing its model. A saved
VoiceHub `config.json` identifies the family; external checkpoints may need an
explicit `model_type`.

The model catalog records selectable integrations. Runtime metadata describes
network components, architecture capabilities, and conversion hooks. These are
different records: a multi-component TTS model may reuse a codec graph. Runtime
metadata never implements a second loading lifecycle.

## Extension rule

Keep the integration, its native graph, conversion code, and preprocessing in
one `models/<family>/` directory. Promote a component into shared code when
multiple families actually use it. Shared behavior uses capabilities and
protocols instead of branching on model names.

Register the configuration and model class with a task factory:

```python
from voicehub import AutoModelForAudioCodec

# MyCodecConfig and MyCodecModel are your importable implementation classes.
AutoModelForAudioCodec.register(
    MyCodecConfig,
    MyCodecModel,
    default_model_path="acme/my-codec",
)
```

The registry stores import paths. Model packages must keep their `__init__.py`
lazy so metadata discovery never imports a network. The model scaffold accepts
`--task tts`, `asr`, `vad`, or `codec`; see [Add a model](../project/adding-a-model.md).

## Training and optimization

Import `Trainer`, `TrainingArguments`, callbacks, and training adapters from
`voicehub.training`. Training availability remains explicit in each model's
training specification. The new DAC and EnCodec task wrappers currently expose
inference and export; they do not advertise a universal codec training recipe.

Optimization passes remain optional and validate the actual loaded runtime.
The shared lifecycle restores reversible inference transformations before
training. TTS generation settings stay in the TTS task layer; they are not
forced onto recognition, VAD, or codecs.

See the [0.4 migration guide](../project/migration-0.4.md) for old-to-new imports
and the analysis behind this layout.
