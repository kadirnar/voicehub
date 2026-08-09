---
description: Code-grounded reference for VoiceHub inference, training, artifacts, and extension registries.
---

# API reference

This page documents the public Python surface exported by `voicehub`. VoiceHub
keeps registry discovery and configuration lightweight; model runtimes and
PyTorch are imported only when the selected operation needs them.

For the generated object-by-object source, signature, summary, and lazy-export
map, see the [public exports inventory](public-api.md).

The default package installs every built-in inference runtime. Add the
independent `training` extra for fine-tuning:

```bash
python -m pip install "voicehub[training] @ git+https://github.com/kadirnar/voicehub.git@main"
```

!!! note "Training support is model and checkpoint specific"

    A registered inference backend is not automatically trainable. Inspect the
    exact [TTS training boundary](../models/training-support.md) or
    [ASR/VAD support matrix](../models/asr-vad-support.md) before selecting a
    checkpoint, backend, or dataset contract.

## Public surface at a glance

| Area | Primary API |
| --- | --- |
| Discovery | `list_model_specs()`, `SpeechTask`, `AutoInferenceModel.available_models()`, `ModelSpec` |
| Configuration | `AutoConfig`, `VoiceHubConfig`, `AutoProcessor`, `VoiceHubProcessor`, `AudioProcessor` |
| Task pipeline | `pipeline()`, `Pipeline`, `TextToSpeechPipeline`, `AutomaticSpeechRecognitionPipeline`, `VoiceActivityDetectionPipeline` |
| TTS inference | `AutoModelForTextToSpeech`, `TTSGenerationConfig`, `TTSOutput` |
| ASR inference | `AutoModelForSpeechRecognition`, `ASRInferenceConfig`, `ASROutput` |
| VAD inference | `AutoModelForVoiceActivityDetection`, `VADInferenceConfig`, `VADOutput` |
| Inference execution | `InferenceStrategy`, `EagerInferenceStrategy`, `TorchCompileInferenceStrategy` |
| LLM TTS serving | `LLMBackendConfig`, `list_llm_backend_support()`, token and Omni speech transports |
| Diffusion serving | `list_diffusion_serving_capabilities()`, fail-closed vLLM-Omni and SGLang modality resolution |
| Training discovery | `get_training_spec()`, `list_training_specs()`, `ModelTrainingSpec` |
| Training adaptation | `AutoTrainingAdapter`, `BaseTrainingAdapter`, family adapters |
| Training loop | `TrainingArguments`, `Trainer`, callbacks, trainer outputs |
| Training execution | `TrainingStrategy`, `TorchTrainingStrategy` |
| TTS datasets | `TTSDataset`, `TTSDatasetSpec`, `TTSDataArchitecture`, `TTSDataReadiness`, length-aware batching |
| TTS optimization | Universal `TTSOptimizationConfig`, sampler/NFE acceleration, diffusion caching, codec kernels, capability discovery/resolution, and source-specific training profiles |
| ASR datasets | `ASRDataset`, `ASRDatasetSpec`, `ASRDataArchitecture`, `ASRDataReadiness` |
| TTS objectives | Multi-codebook CE, diffusion/flow pair builders, VITS loss primitives |
| Collation | `default_data_collator`, `DefaultDataCollator`, `DataCollatorForTTSTraining`, `DataCollatorForAudioTraining` |
| Extensions | Inference-strategy, training-spec, adapter, and training-strategy registries |

Unless a different module is shown, names on this page can be imported directly:

```python
from voicehub import AutoModelForTextToSpeech, Trainer, TrainingArguments
```

## Model discovery

### `list_model_specs` and `SpeechTask`

```python
list_model_specs(
    *,
    task: SpeechTask | str | None = None,
) -> tuple[ModelSpec, ...]
```

Filter the shared registry by `text-to-speech`,
`automatic-speech-recognition`, or `voice-activity-detection`. Short aliases
`tts`, `asr`, `stt`, and `vad` are accepted:

```python
from voicehub import list_model_specs

for spec in list_model_specs(task="asr"):
    print(spec.model_type, spec.architecture, spec.install_extra or "default")
```

### `AutoInferenceModel.available_models`

```python
AutoInferenceModel.available_models() -> tuple[ModelSpec, ...]
```

Returns the legacy TTS-only registry view in stable display order without
loading model weights or importing a model runtime. Use
`list_model_specs(task=None)` for all speech tasks, or pass `task="asr"` /
`task="vad"` for a task-specific view.

```python
from voicehub import AutoInferenceModel

for spec in AutoInferenceModel.available_models():
    print(
        spec.model_type,
        spec.default_model_path,
        spec.install_extra or "default",
        spec.training.support.value,
    )
```

### `ModelSpec`

`ModelSpec` is immutable registry metadata.

| Attribute | Meaning |
| --- | --- |
| `model_type` | Canonical model identifier used by factories |
| `module` / `class_name` | Lazy import target for the model wrapper |
| `config_module` / `config_class` | Lazy import target for its configuration |
| `processor_module` / `processor_class` | Lazy import target for its task-default or explicitly registered processor |
| `default_model_path` | Default Hub identifier or local artifact name |
| `install_extra` | `None` for built-in inference; optional setup identifier reserved for external/future runtimes |
| `capabilities` | Open capability tokens. `fine-tuning` is family-level; `default-checkpoint-inference-only` means the training profile names a different differentiable starting checkpoint. |
| `task` | Canonical `SpeechTask` owned by the provider |
| `architecture` | Provider/runtime architecture family, when declared |
| `components` | Canonical shared codec, vocoder, or neural-component names declared by this model spec |
| `default_for_task` | Whether this is the registry's unique no-argument default for its task |
| `license` | `ModelLicenseSpec` when additional model terms are recorded, otherwise `None` |
| `training` | The model's `ModelTrainingSpec` |

`ModelLicenseSpec` contains `model_type`, `license_id`, `commercial_use`,
`upstream`, and `notice`. License metadata is a discovery aid, not legal advice.

## Configuration and processor factories

### `AutoConfig`

Create a configuration from a registry key:

```python
AutoConfig.for_model(model_type: str, **kwargs) -> VoiceHubConfig
```

Load `config.json` from a local path or Hub repository:

```python
AutoConfig.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type: str | None = None,
    **kwargs,
) -> VoiceHubConfig
```

Pass `model_type` when the source cannot identify its architecture, including a
raw checkpoint file. When `model_type` is omitted, `config.json` must contain
it. Loader options such as `subfolder`, `cache_dir`, `revision`, `token`, and
`local_files_only` apply to both automatic model-type discovery and the
concrete configuration load. Pass those options through `config_kwargs` when
using an auto model or processor factory.

```python
from voicehub import AutoConfig

config = AutoConfig.for_model(
    "parlertts",
    name_or_path="parler-tts/parler-tts-mini-v1",
    sample_rate=44_100,
)
```

### `VoiceHubConfig`

```python
VoiceHubConfig(
    *,
    sample_rate: int = 24_000,
    architectures: list[str] | None = None,
    name_or_path: str | Path = "",
    return_dict: bool = True,
    output_hidden_states: bool = False,
    output_attentions: bool = False,
    generation_config: dict[str, Any] | None = None,
    **kwargs,
)
```

Concrete integrations normally provide a subclass with a canonical
`model_type`. Additional keyword arguments are retained as attributes, but
credential-shaped fields are not configuration data. Construction rejects
top-level or nested runtime secrets, and dictionary, diff, JSON, representation,
and file serialization revalidate the final payload. A top-level Hub `token`
accepted by a concrete loader remains runtime-only and is omitted; fields such
as `pad_token_id` remain normal serializable model settings. Pass credentials
only to `from_pretrained()` or the model constructor.

Hub API responses and VoiceHub's cached file and snapshot metadata use the
same strict JSON decoder as local configuration artifacts. Duplicate object
keys and non-finite numbers are rejected before a remote commit or repository
tree is interpreted. Ambiguous cache metadata is treated as a cache miss, and
diagnostics include the source context without including discarded values or
authentication tokens.

The shared declarative Byte-BPE and SentencePiece-BPE tokenizer loaders apply
the same policy before interpreting the model graph, vocabulary, merges, or
added tokens. Their existing byte, token, merge, nesting, and node limits still
apply after strict JSON decoding.

| Method | Contract |
| --- | --- |
| `from_dict(values, **overrides)` | Construct from a mapping and apply explicit overrides |
| `from_pretrained(source, *, subfolder="", cache_dir=None, revision=None, token=None, local_files_only=False, **kwargs)` | Load `config.json` from local or Hub storage |
| `to_dict()` | Return a deep-copied, path-normalized mapping including `model_type` |
| `to_diff_dict()` | Return values differing from the common base configuration |
| `to_json_string(use_diff=False)` | Return stable, indented JSON |
| `to_json_file(path, use_diff=False)` | Write configuration to an explicit JSON file |
| `save_pretrained(directory)` | Write `config.json` and return its `Path` |
| `update(values)` | Apply mapping values in place |

### `AutoProcessor`

```python
AutoProcessor.from_config(
    config: VoiceHubConfig,
    **kwargs,
) -> VoiceHubProcessor

AutoProcessor.from_pretrained(
    pretrained_model_name_or_path="",
    *,
    model_type: str | None = None,
    config: VoiceHubConfig | None = None,
    config_kwargs: Mapping[str, object] | None = None,
    **kwargs,
) -> VoiceHubProcessor
```

`from_config()` selects the processor class registered by the model wrapper.
`from_pretrained()` delegates a local directory, direct
`processor_config.json`, or Hub source to that processor's artifact loader.
The base processor falls back to construction options when its optional
processor configuration is absent. Pass `model_type` or `config` when the
source does not provide a VoiceHub `config.json`. Use `config_kwargs` for
configuration loading or overrides such as `revision` and
`local_files_only`; supported Hub loader values are reused for processor
artifact resolution without becoming processor state. Remaining keyword
arguments are reserved for processor construction and restoration. Pass either
a complete `config` or `config_kwargs`, not both.

The base `VoiceHubProcessor` API is:

```python
processor(text: str, **conditioning) -> BatchFeature
processor.to_dict() -> dict[str, Any]
processor.save_pretrained(directory) -> Path
VoiceHubProcessor.from_pretrained(source, *, subfolder="", **kwargs)
```

The base processor rejects empty text and retains conditioning fields.
Architecture-specific processors may perform additional validation or
conversion. `BatchFeature` is a dictionary whose `.to(device)` method moves
tensor-like values in place.

Processor artifacts contain construction settings, not credentials.
Construction and untrusted `processor_config.json` loading reject top-level or
nested runtime secrets, and dictionary conversion rechecks mutable state.
`save_pretrained()` also validates the final mapping returned by a subclass,
so a rejected save creates neither the processor file nor its artifact
directory. Hub `token` values remain loader-only, while ordinary settings such
as `normalization` round-trip normally.

Audio-input ASR and VAD models use `AudioProcessor`:

```python
processor(
    audio,
    *,
    sampling_rate: int | None = None,
    **inference_options,
) -> BatchFeature
```

It validates the dependency-light input envelope. `load_audio()` performs
decoding, mono downmixing, and optional resampling lazily when inference
begins.

## Pipeline

`pipeline()` selects the task-specific auto factory and preserves the model's
normalized `TTSOutput`, `ASROutput`, or `VADOutput`:

```python
pipeline(
    task: SpeechTask | str,
    model=None,
    *,
    model_type: str | None = None,
    config: VoiceHubConfig | None = None,
    device: str | None = None,
    inference_strategy: str | InferenceStrategy | None = None,
    config_kwargs: Mapping[str, object] | None = None,
    model_kwargs: Mapping[str, object] | None = None,
) -> Pipeline
```

Pass a repository ID or local artifact as `model` to load it through
`AutoModelForTextToSpeech`, `AutoModelForSpeechRecognition`, or
`AutoModelForVoiceActivityDetection`. Passing `None` selects the registry
default for the task when one is declared. Loading remains lazy unless
`model_kwargs={"lazy_load": False}` is explicit.

An already configured model object can be wrapped without changing its device
or runtime state:

```python
from voicehub import pipeline

speech_pipeline = pipeline("tts", model=model)
output = speech_pipeline("A normalized pipeline output.")
speech_pipeline.load()
speech_pipeline.save_pretrained("artifacts/model")
```

Task aliases are normalized by `SpeechTask.coerce()`. A registry-owned model
must match the requested task and implement its callable contract:
`generate()` for TTS, `transcribe()` for ASR, or `detect()` for VAD.
`Pipeline.processor`, `.device`, and `.model_type` expose the wrapped model's
declared values. See the [Pipeline guide](../guides/inference.md) for complete
workflows and the explicit batching boundary.

## Model factories

### `AutoModel`

`AutoModel` reads the registered task and dispatches to the TTS, ASR, or VAD
factory:

```python
model = AutoModel.from_pretrained(
    checkpoint,
    model_type="asr_qwen3",
)
```

Use a task-specific factory when the task is already known. Use `AutoModel`
for tools that handle several speech tasks.

### `AutoModelForTextToSpeech`

This is the preferred checkpoint-first factory.

```python
AutoModelForTextToSpeech.from_pretrained(
    pretrained_model_name_or_path="",
    *,
    model_type: str | None = None,
    config: VoiceHubConfig | None = None,
    inference_strategy: str | InferenceStrategy | None = None,
    config_kwargs: Mapping[str, object] | None = None,
    **kwargs,
)
```

```python
AutoModelForTextToSpeech.from_config(
    config: VoiceHubConfig,
    *,
    inference_strategy: str | InferenceStrategy | None = None,
    **kwargs,
)
```

`model_type` can be omitted when a VoiceHub artifact contains `config.json`.
For a Hub repository that does not publish VoiceHub metadata, supply the
registry key explicitly.

Use `config_kwargs` for configuration fields such as `torch_dtype`, decoding
defaults, or model-family settings. Its keys must be non-empty strings. Pass
either a complete `config` object or `config_kwargs`, not both. The
`model_type` field is reserved for the top-level factory argument and cannot
be overridden inside `config_kwargs`.

```python
from voicehub import AutoModelForTextToSpeech

model = AutoModelForTextToSpeech.from_pretrained(
    "parler-tts/parler-tts-mini-v1",
    model_type="parlertts",
    device="auto",
    lazy_load=True,
)
```

### `AutoInferenceModel`

`AutoInferenceModel` is the compatibility, model-type-first factory:

```python
AutoInferenceModel.from_pretrained(
    model_type: str | None = None,
    model_path: str | Path | None = None,
    device: str = "cuda",
    inference_strategy: str | InferenceStrategy | None = None,
    **kwargs,
)
```

When `model_type` is omitted, the registry's unique TTS
`default_for_task` entry is used. When `model_path` is `None`, that entry's
`default_model_path` is used. The built-in declaration preserves the legacy
Orpheus default without embedding a provider name in the compatibility
factory.
Prefer `AutoModelForTextToSpeech` in new code because it can infer the model
type from a saved VoiceHub configuration.

### ASR and VAD factories

`AutoModelForSpeechRecognition` and
`AutoModelForVoiceActivityDetection` expose the same
`from_pretrained()` / `from_config()` construction contract while enforcing
the registry task before a model module is imported:

```python
from voicehub import (
    AutoModelForSpeechRecognition,
    AutoModelForVoiceActivityDetection,
)

asr = AutoModelForSpeechRecognition.from_pretrained(
    "openai/whisper-small",
    model_type="asr_transformers",
)
vad = AutoModelForVoiceActivityDetection.from_pretrained(
    "silero_vad",
    model_type="vad_silero",
)
```

Calling a task factory without a checkpoint uses the unique registered
`ModelSpec` whose `default_for_task` value is true. Tasks without that
declaration fail with an actionable error instead of relying on a provider name
embedded in the shared auto factory.

Audio-input pretrained models provide:

| Method | Result |
| --- | --- |
| `forward(audio, *, sampling_rate=None, inference_config=None, **kwargs)` | Validate, lazy-load, infer, and enforce the task output type |
| `transcribe(...)` | ASR alias returning `ASROutput` |
| `detect(...)` | VAD alias returning `VADOutput` |
| `stream(*, sampling_rate, **kwargs)` | Create an isolated session; the base session buffers until `flush()` |
| `load()` / `load_for_training()` | Enter the inference or differentiable lifecycle |
| `save_pretrained(directory, include_native_export=True)` | Save configuration, inference configuration, processor, and optional native export |

`ASRInferenceConfig` and `VADInferenceConfig` are safe to persist as public
task settings. Construction and untrusted checkpoint loading reject nested or
top-level credentials, and dictionary conversion rechecks mutable state.
Representation and `save_pretrained()` also validate the final serialized
payload, so an unsafe subclass fails before a task configuration file or its
artifact directory is created. Hub `token` values remain runtime-only, while
ordinary inference fields such as `max_new_tokens` round-trip normally.

See the [ASR guide](../guides/speech-recognition.md),
[VAD guide](../guides/voice-activity-detection.md), and
[provider matrix](../models/asr-vad-support.md).

### Register a model

Each auto factory provides a Transformers-style registration method:

```python
AutoModelForTextToSpeech.register(
    AuroraConfig,
    AuroraForTextToSpeech,
    default_model_path="acme/aurora-base",
    aliases=("aurora-tts",),
)
```

The config supplies `model_type`; the factory supplies the task. The registry
stores lazy import paths. See [Add a model](../project/adding-a-model.md).

### ASR and VAD outputs

`ASROutput` contains `text`, `segments`, optional `language`, optional
`duration`, and `metadata`. An `ASRSegment` may include timestamps,
confidence, language, speaker, and `ASRWord` values.

`VADOutput` contains ordered, non-overlapping `SpeechSegment` values and
optional duration, sample rate, frame/window probabilities, and metadata.
`speech_duration` sums accepted regions and `contains(timestamp)` tests a
point.

Optional timing and score values remain `None` when the provider did not
compute them.

### Common pretrained lifecycle

Models based on `PreTrainedTTSModel` provide:

| Member | Contract |
| --- | --- |
| `config` | Architecture configuration |
| `generation_config` | Saved/default `TTSGenerationConfig` |
| `processor` | Architecture processor |
| `model` | Loaded backend runtime, initially `None` for a lazy wrapper |
| `device` | Requested device; `"auto"` resolves to CUDA, MPS, or CPU during load |
| `sample_rate` | Configured sample rate; generated output still reports the runtime's actual rate |
| `is_loaded` | Whether the checkpoint-backed runtime has been constructed |
| `inference_strategy` | Active inference policy |
| `llm_backend` | `native`, `vllm`, or `sglang` |
| `llm_backend_config` | Runtime-only external connection settings, or `None` |
| `llm_backend_transport` | Resolved `auto`, `tokens`, or `speech` transport |
| `training_default_model_name_or_path` | Recommended differentiable starting checkpoint from the training spec |

```python
PreTrainedTTSModel.from_pretrained(
    pretrained_model_name_or_path="",
    *,
    config=None,
    device="auto",
    lazy_load=True,
    inference_strategy=None,
    llm_backend=None,
    llm_backend_config=None,
    optimization_config=None,
    attn_implementation=None,
    kernel_backend=None,
    torch_compile=None,
    compile_config=None,
    diffusion_cache=None,
    diffusion_cache_config=None,
    config_kwargs=None,
    **kwargs,
)
```

| Method | Result |
| --- | --- |
| `load()` | Load once and prepare the runtime for inference |
| `load_for_training()` | Validate and construct or restore a differentiable runtime |
| `validate_training_support()` | Validate the exact configured backend/checkpoint without loading weights; return `ModelTrainingSpec` |
| `set_inference_strategy(strategy)` | Select a policy before an inference runtime is active |
| `set_llm_backend(backend, config=None, **config_kwargs)` | Select and validate a vLLM/SGLang server before loading |
| `clear_llm_backend()` | Detach an idle speech backend or an unloaded token backend |
| `set_optimization_config(config)` | Schedule a universal TTS policy before the next inference load |
| `clear_optimization_config()` | Return and remove a still-pending policy, including after runtime-dependent load failure |
| `prepare_inputs_for_generation(text, **kwargs)` | Run the configured processor and return model inputs |
| `forward(text, **kwargs)` | Validate, lazy-load, synthesize, and enforce `TTSOutput` |
| `generate(text, *, generation_config=None, **kwargs)` | Merge generation defaults and call `forward()` |
| `create_training_dataset(records, **kwargs)` | Delegate raw-data construction to the model's adapter |
| `get_training_adapter()` | Create the unloaded adapter paired with this wrapper |
| `save_pretrained(directory, include_native_export=True)` | Save VoiceHub metadata and optional backend-native artifacts |

See [External LLM serving](../guides/llm-serving.md) for the capability
matrix, server launch commands, request schemas, and lifecycle constraints.

`LLMBackendSupport` is the immutable, JSON-serializable request capability for
one model/backend pair. In addition to transports and checkpoint metadata, it
declares default task types with and without reference audio, normalized task
type aliases,
`reference_format` (`flat` or `references`), and validated
`speech_string_options`. The derived `speech_input_options` property is the
recognized wrapper and client input schema, `speech_default_options` lists the
keys accepted from `GenerationConfig`, and `speech_native_only_options` lists
known native-only fields that the external endpoint must reject. Declaring a
verified extension string option updates both wrapper and client behavior
without a central allowlist edit. These properties describe request handling,
not a performance or checkpoint-support claim.

```python
list_llm_backend_support(*, backend=None, model_type=None)
get_llm_backend_support(model_type, backend, *, transport="auto")
register_llm_backend_support(support, *, exist_ok=False) -> None
unregister_llm_backend_support(
    model_type,
    backend,
    *,
    missing_ok=False,
) -> LLMBackendSupport | None

support.to_dict() -> dict[str, object]
```

`support.to_dict()` includes the three derived option lists so capability
evidence and downstream tooling can inspect the same fail-closed request
contract used at runtime.

Extension registrations are process-local. Built-in records cannot be removed,
and a duplicate pair is accepted with `exist_ok=True` only when its complete
record is identical.

There is no universal `unload()` or `release()` API. A serving-to-training
transition uses `load_for_training()`, allowing the active inference strategy
to restore a trainable representation first.

## Generation

### `TTSGenerationConfig`

```python
TTSGenerationConfig(
    *,
    output_file: str | Path | None = None,
    seed: int | None = None,
    speed: float | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    max_new_tokens: int | None = None,
    **backend_options,
)
```

| Common field | Validation |
| --- | --- |
| `output_file` | Non-empty path that is not an existing directory |
| `seed` | Integer in Torch's supported seed range |
| `speed` | Finite number greater than zero |
| `temperature` | Finite, non-negative number |
| `top_p` | Finite number in `[0, 1]` |
| `max_new_tokens` | Positive integer |

The configuration is extensible: extra keyword arguments are retained for a
backend. A common field is not a promise that every backend implements it.
Generation input validation rejects unsupported options when the backend
exposes a finite generation signature.

Credential-shaped values are runtime state, not generation defaults.
Construction and checkpoint loading reject top-level or nested secrets;
dictionary, representation, and file serialization revalidate the payload.
The Hub `token` accepted by `from_pretrained()` is used only to resolve the
artifact and is never stored. Model settings such as `pad_token_id`,
`eos_token_id`, and `max_new_tokens` remain serializable. Pass credentials to
the model or loader call instead of `TTSGenerationConfig`.

Generation values are merged in this exact order, with later sources winning:

1. defaults stored on `model.generation_config`;
2. the supplied `generation_config`; and
3. explicit keyword arguments to `generate()`.

```python
from voicehub import TTSGenerationConfig

request = TTSGenerationConfig(
    seed=42,
    temperature=0.8,
    output_file="artifacts/sample.wav",
)

output = model.generate(
    "VoiceHub applies the explicit temperature last.",
    generation_config=request,
    temperature=0.7,
    description="A clear, measured studio voice.",
)
```

| Method | Contract |
| --- | --- |
| `validate()` | Validate common fields without rejecting backend extensions |
| `to_dict()` | Deep-copy and normalize nested paths for serialization |
| `from_dict(values, **overrides)` | Construct with explicit overrides |
| `from_model_config(config)` | Read the config's `generation_config` mapping |
| `from_pretrained(source, *, subfolder="", **hub_kwargs)` | Load `generation_config.json` |
| `save_pretrained(directory)` | Write `generation_config.json` |
| `update(**kwargs)` | Apply known/existing fields and return unknown fields |

### `TTSOutput`

```python
@dataclass
class TTSOutput:
    audio: Any
    sample_rate: int
    file_path: str | Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

`audio` must be a non-empty, finite, real-valued materialized waveform.
`sample_rate` must be a positive integer. `metadata` must be a dictionary.

```python
print(output.audio)
print(output.sample_rate)
print(output.file_path)
print(output.path)       # pathlib.Path or None
print(output.metadata)

audio, sample_rate = output
same_pair = output.to_tuple()
populated = output.to_dict()
written_path = output.save("artifacts/copy.wav")
```

`keys()` returns `audio` and `sample_rate`, plus `file_path` and `metadata` only
when populated. String indexing uses these keys. Integer indexing and iteration
operate on the interoperability pair `(audio, sample_rate)`.

`save()` writes one mono waveform, creates parent directories, updates
`file_path`, and returns the written path as a string.

## Inference errors

| Exception | Base classes | Meaning |
| --- | --- | --- |
| `VoiceHubError` | `Exception` | Base class for VoiceHub-specific failures |
| `UnknownModelError` | `ValueError`, `VoiceHubError` | Registry key is unknown |
| `OptionalDependencyError` | `ImportError`, `VoiceHubError` | Selected optional runtime is missing |
| `SourceLicenseError` | `VoiceHubError` | Upstream source cannot legally be redistributed |

Standard Python exceptions are also part of validation behavior:

- explicit local paths that do not exist raise `FileNotFoundError`;
- an output target that is a directory raises `IsADirectoryError`;
- invalid values and incompatible lifecycle transitions raise `ValueError` or
  `RuntimeError`; and
- wrong API types raise `TypeError`.

Do not catch only `VoiceHubError` if local I/O and caller input errors must also
be handled.

## Inference strategies

`InferenceStrategy` separates runtime optimization from model-family generation.
The built-in `EagerInferenceStrategy` is named `"eager"` and is a no-op.
`TorchCompileInferenceStrategy` is named `"torch-compile"` and applies
reversible `torch.compile` preparation to the execution boundaries declared
by the loaded speech model.

```python
class InferenceStrategy:
    name = "base"

    def validate(self, wrapper) -> None: ...
    def prepare(self, model, *, wrapper): ...
    def restore_for_training(self, model, *, wrapper): ...
```

- `validate()` must remain side-effect free and runs before model allocation.
- `prepare()` may return the same runtime or a replacement.
- `restore_for_training()` must return the representation expected by the
  wrapper's training path.

Both returning methods must return a runtime; returning `None` is an error.

The compile strategy is opt-in and compiles lazily on the first real request:

```python
TorchCompileInferenceStrategy(
    *,
    backend: str = "inductor",
    mode: str | None = None,
    fullgraph: bool = False,
    dynamic: bool | None = True,
    options: dict[str, object] | None = None,
    requirement: str = "required",
)
```

It supports CPU and CUDA runtimes while preserving canonical state-dict keys.
The default `requirement="required"` fails when the compiler cannot be
prepared or executed. Set `requirement="auto"` only when an eager fallback is
acceptable. `runtime_metadata(wrapper)` reports whether the active runtime was
compiled or selected an eager fallback, and `restore_for_training()` restores
the original callables before a training transition.

Registry functions:

```python
list_inference_strategies() -> tuple[str, ...]
get_inference_strategy(strategy: str | InferenceStrategy | None = None)
register_inference_strategy(name, factory, *, exist_ok=False) -> None
unregister_inference_strategy(name) -> None
```

Factories must be zero-argument callables returning an `InferenceStrategy`.
Names are stripped and lowercased. The built-in `"eager"` and
`"torch-compile"` entries cannot be replaced or removed.

```python
from voicehub import (
    InferenceStrategy,
    register_inference_strategy,
    unregister_inference_strategy,
)


class AuditedEagerStrategy(InferenceStrategy):
    name = "audited-eager"

    def prepare(self, model, *, wrapper):
        return model

    def restore_for_training(self, model, *, wrapper):
        return model


register_inference_strategy("audited-eager", AuditedEagerStrategy)
try:
    model = AutoModelForTextToSpeech.from_pretrained(
        "parler-tts/parler-tts-mini-v1",
        model_type="parlertts",
        device="auto",
        inference_strategy="audited-eager",
    )
finally:
    unregister_inference_strategy("audited-eager")
```

Registry mutations are process-global. Register extensions during application
startup, not per request.

## Universal TTS optimization

`TTSOptimizationConfig` is the configuration-first optimization API for all
registered text-to-speech models:

```python
TTSOptimizationConfig(
    attn_implementation="auto",
    kernel_backend="auto",
    compile="auto",
    compile_config=None,
    diffusion_cache="disabled",
    diffusion_cache_config=None,
    diffusion_sampling="disabled",
    diffusion_sampling_config=None,
    optimization_passes=(),
)
```

Accepted attention values are `auto`, `native`, `sdpa`, and
`flash_attention_4`. Kernel values are `auto`, `native`, `torch`, `triton`,
and `cuda_extension`. Compile values are `auto`, `required`, and `disabled`;
the boolean aliases `True` and `False` mean `required` and `disabled`.
`compile_config` accepts a `TorchCompileConfig`, a mapping with the same
fields, or `None`:

```python
TorchCompileConfig(
    backend="inductor",
    mode=None,
    fullgraph=False,
    dynamic=None,
    options=None,
    requirement="auto",
)
```

Diffusion-cache values are `disabled`, `auto`, and `required`; the boolean
aliases `False` and `True` mean `disabled` and `required`. This is a separate,
approximate inference policy and therefore defaults to `disabled`.
`diffusion_cache_config` accepts a `DiffusionCacheConfig`, a mapping with the
same fields, or `None`.

Diffusion-sampling values are also `disabled`, `auto`, and `required`.
This independent, approximate inference policy rebuilds sampler schedules,
reduces CFG evaluations, predicts selected model outputs, or activates a
compatible specialized solver. `diffusion_sampling_config` accepts a
`DiffusionSamplingConfig`, a mapping with the same fields, or `None`.

The enclosing compile policy controls `requirement`, so
`TTSOptimizationConfig(compile="required")` always produces a required
`TorchCompileConfig`. Configuration serialization and resolution are:

```python
TTSOptimizationConfig.from_dict(values, **overrides)
config.to_dict()
config.to_json_string()
config.resolve(
    target,
    *,
    mode="inference",
    context=None,
    registry=None,
) -> TTSOptimizationPlan

get_tts_optimization_config(target, **overrides) -> TTSOptimizationConfig
get_tts_optimization_support(target) -> TTSOptimizationSupport
list_tts_optimization_support() -> tuple[TTSOptimizationSupport, ...]
resolve_tts_optimization(
    target,
    config=None,
    *,
    mode="inference",
    context=None,
    registry=None,
) -> TTSOptimizationPlan
```

`target` may be a registered TTS model type, a TTS architecture ID, or a
model exposing `config.model_type`. Resolution validates the model task,
canonical architecture, device, dtype, mode, streaming, and distributed
context without loading weights. `TTSOptimizationPlan` contains `config`,
`context`, `support`, ordered `passes`, and an ordered `decisions` tuple.
`plan.manifest()` records both executable passes and native/eager fallbacks.

The current 34-entry TTS registry declares the universal compile policy for
every model. Apply-time discovery still validates that the loaded execution
mode has a real target: an explicit empty target set selects eager execution
for an automatic policy and fails a required policy. `conversationtts`,
`f5tts`, and `qwen3tts` additionally declare the selectable
FlashAttention-4 protocol. Those three plus the trait-discovered VITS family
(`vits`, `melotts`, `inflecttts`, GPT-SoVITS S2, and the OpenVoice converter)
declare architecture-owned custom-kernel protocols. Capability discovery is
registry-driven, so callers should query the support functions instead of
depending on those counts.

Automatic choices are non-strict. Attention retains verified SDPA or native
semantics, custom-kernel dispatch retains a registered Torch implementation,
and recognized compiler availability or execution failures can retain eager
execution. Explicit `flash_attention_4`, `triton`, `cuda_extension`, and
required compile choices fail rather than silently selecting another
implementation. CUDA-extension compilation/loading remains an explicit
`load_tts_activation_cuda_extension()` operation.

### Diffusion sampler acceleration

`DiffusionSamplingConfig` controls the architecture-owned solver boundary:

```python
DiffusionSamplingConfig(
    target_steps=None,
    schedule="native",  # native, uniform, quadratic, or trailing
    solver="native",  # native or stork2
    stork_stages=9,
    guidance="native",  # native, limited_interval, or adaptive
    guidance_start=0.0,
    guidance_end=1.0,
    adaptive_guidance_threshold=0.01,
    adaptive_guidance_warmup_steps=4,
    adaptive_guidance_patience=2,
    prediction_cache="disabled",  # fora, teacache, smoothcache, or taylor
    cache_interval=2,
    cache_warmup_steps=2,
    cache_max_consecutive_steps=2,
    cache_rel_l1_threshold=0.08,
    cache_error_budget=0.20,
    teacache_coefficients=(),
    smoothcache_compute_step_mask=(),
    taylor_order=1,
    epsilon=1e-6,
)

DiffusionSamplingConfig.from_dict(values, **overrides)
config.to_dict()
DiffusionSamplingPass(config=None)
```

The pass is inference-only, reversible, and disabled by default. TeaCache
requires checkpoint-specific rescaling coefficients; SmoothCache requires an
explicit compute mask for the prepared schedule. STORK-2 is accepted only by
reviewed direct deterministic velocity-field adapters and cannot be combined
with guidance pruning or whole-prediction caching.

Architecture registrations expose `sampling_techniques` through
`get_diffusion_model_optimization_support()`. Required unsupported
techniques fail during resolution; automatic requests retain the native
sampler. See the diffusion optimization guide for the per-model matrix and
50-step examples.

### Diffusion block-residual cache

`DiffusionCacheConfig` configures the architecture-owned, Cache-DiT-style
middle-block residual cache:

```python
DiffusionCacheConfig(
    method="dbcache",  # dbcache or first_block
    front_blocks=1,
    back_blocks=0,
    residual_diff_threshold=0.08,
    warmup_steps=2,
    warmup_interval=1,
    max_cached_steps=-1,
    max_consecutive_cached_steps=3,
    max_accumulated_relative_error=None,
    predictor="reuse",
    taylor_order=1,
    compute_step_mask=(),
    compute_step_policy="dynamic",
    num_inference_steps=None,
    force_refresh_step_hint=None,
    force_refresh_step_policy="once",
    probe_downsample_factor=1,
    metrics_history_size=256,
    synchronize_distributed=True,
    epsilon=1e-6,
)

DiffusionCacheConfig.from_dict(values, **overrides)
config.to_dict()
DiffusionCachePass(config=None)
```

`front_blocks` and `back_blocks` are the DBCache `Fn` and `Bn` boundaries.
Predictors are `reuse` and `taylor`; the latter supports orders 1-3 using only
fully computed residuals. `compute_step_mask[index] is True` forces a full
middle-block evaluation. A `False` entry uses threshold-based reuse under the
`dynamic` policy and unconditional compatible reuse under `static`. Warm-up
cadence, total/consecutive cache-hit limits, inference-step segmentation,
forced refreshes, and the optional accumulated-error budget provide additional
safety controls. Cache-DiT spellings such as `Fn_compute_blocks`,
`Bn_compute_blocks`, `max_warmup_steps`, `steps_computation_mask`, and
`taylorseer_order` are accepted by `from_dict()`.

`TTSOptimizationConfig(diffusion_cache="auto", ...)` is an explicit,
non-strict request. It retains exact inference when the architecture does not
declare `diffusion-cache`; `"required"` raises instead. Both policies are
rejected or disabled in training, streaming, and unsupported contexts.
Gradient-enabled model calls bypass the cache even if a pass was already
applied.

The resolved pass is reversible and runs after architecture kernel/attention
selection but before `torch.compile`. Its application manifest records
`fidelity="approximate"`, the adapted module labels, and live hit/miss and
invalidation statistics. All nine registered diffusion-family model types
declare a cache surface: Chatterbox, CosyVoice, Echo, F5-TTS, Irodori-TTS,
StyleTTS 2, Supertonic, VibeVoice, and VoxCPM. Most cache repeated native
transformer/DiT blocks. Supertonic's flattened ONNX graph instead caches the
predicted `next_latent - current_latent` residual so the recurrence cannot
stall by reusing an absolute output. VibeVoice's cache remains limited to its
low-level diffusion head while its public high-level inference path is
unsupported.

Adapted modules expose the following request-control surface:

```python
module.enable_diffusion_cache(config=None) -> DiffusionCacheConfig
module.disable_diffusion_cache() -> DiffusionCacheConfig | None
module.reset_diffusion_cache(*, lane=None)
module.reset_diffusion_cache_stats()
module.diffusion_cache_session()
module.diffusion_cache_stats(*, details=False) -> dict[str, object]

diffusion_cache_request(model)
diffusion_cache_summary(model, *, details=False)
reset_diffusion_cache_metrics(model) -> int
```

Use `diffusion_cache_session()` or the architecture sampler's built-in reset
to isolate requests. Separate CFG modes use distinct lane names. Shape,
dtype, device, and block-layout mismatches bypass or invalidate cached
tensors.

The summary reports cache/miss counts and reasons, dynamic versus static hits,
residual-difference percentiles, predictor usage, executed and skipped block
evaluations, estimated block-compute reduction, active/peak cache bytes, and
per-lane totals. `details=True` adds bounded residual and step histories.

### Diffusion serving capabilities

The serving API records engine modality separately from TTS compatibility and
does not import optional engines during normal package import:

```python
list_diffusion_serving_capabilities(
    *,
    supports_tts=None,
    supports_visual_diffusion=None,
) -> tuple[DiffusionServingCapability, ...]

get_diffusion_serving_capability(backend) -> DiffusionServingCapability
resolve_diffusion_tts_backend(
    model_type,
    backend,
    *,
    plugin=None,
) -> DiffusionTTSServingPlan

detect_vllm_omni_features(
    *,
    probe_registry=True,
) -> VLLMOmniFeatureStatus

bridge_vllm_omni_tts_config(
    model_type,
    config,
) -> tuple[DiffusionTTSServingPlan, LLMBackendConfig]
```

`DiffusionServingBackend` values are `native`, `vllm-omni`,
`sglang-diffusion`, and `sglang-omni`. vLLM-Omni is verified for complete
CosyVoice and VoxCPM speech pipelines; the bridge validates an
`LLMBackendConfig(backend="vllm")` and forces its existing speech transport.
Other vLLM-Omni pairings require an explicit, experimental
`VLLMOmniDiffusionPlugin` that declares a complete pipeline and audio
post-processing:

```python
VLLMOmniDiffusionPlugin(
    model_type,
    model_arch,
    module_name,
    class_name,
    complete_tts_pipeline=False,
    pre_process_func_name=None,
    post_process_func_name=None,
    action_post_process_func_name=None,
    ir_op_priority_func_name=None,
)

plugin.registration_kwargs()
plugin.register()
```

`plugin.register()` lazily calls the installed vLLM-Omni
`register_diffusion_model` API. `detect_vllm_omni_features()` reports whether
that hook exists before registration. SGLang Diffusion is recorded as an
image/video runtime and always fails TTS resolution. SGLang-Omni is a
separate LLM-TTS backend served through `voicehub.llm_serving`, not a visual
diffusion TTS adapter.

Every `BaseTTSModel` exposes:

```python
model.resolve_optimization(
    config=None,
    *,
    mode="inference",
    context=None,
    registry=None,
) -> TTSOptimizationPlan

model.optimize(
    config=None,
    *,
    mode="inference",
    context=None,
    registry=None,
) -> TTSOptimizationResult

model.tts_optimization_result(*, mode="inference")
model.tts_optimization_manifest(*, mode=None)
model.restore_tts_optimization(*, mode="inference")
```

`model.optimize()` loads the correct execution runtime (or the architecture
training adapter in training mode), applies nonempty plans transactionally,
and also publishes a result for a successful all-native plan.
`TTSOptimizationResult.optimized` indicates that at least one executable pass
was applied; it does not claim a benchmarked speedup or that every auxiliary
codec/vocoder stage was compiled. `restore_tts_optimization()` reverses
applied passes or clears a native fallback report.

`PreTrainedTTSModel.from_pretrained()` additionally accepts:

```python
AutoModelForTextToSpeech.from_pretrained(
    checkpoint,
    model_type=model_type,
    optimization_config=config_or_mapping,
    attn_implementation=None,
    kernel_backend=None,
    torch_compile=None,
    compile_config=None,
)
```

The direct arguments override the corresponding complete configuration.
Without `optimization_config`, unspecified direct attention and kernel fields
remain native and compilation remains disabled; supplying only
`compile_config` selects automatic compilation. The resolved policy is
scheduled before a lazy inference load and applied after the native runtime
and inference strategy are prepared.

If runtime-dependent policy validation fails after weights load,
`clear_optimization_config()` returns and removes the retained pending policy;
a following `load()` reuses the native weights. A pending policy must be
loaded or cleared before calling `optimize()` or
`apply_optimization_plan()`. Within `compile_config`, PyTorch's `mode` and
`options` settings are mutually exclusive.

`Trainer` accepts the same policy through `optimization_config`:

```python
trainer = Trainer(
    model=model,
    args=training_arguments,
    train_dataset=train_dataset,
    optimization_config=TTSOptimizationConfig(
        attn_implementation="auto",
        kernel_backend="auto",
        compile="auto",
    ),
)
```

It is mutually exclusive with `optimization_plan`. Trainer resolves the
policy in training mode after device placement and before strategy wrapping
and optimizer creation. An explicit `optimization_context` must use
`mode="training"` and `persist_result=True`. The resolved plan is available as
`trainer.tts_optimization_plan`; `trainer.optimization_manifest()` combines
the resolution and application records for checkpoints.

This interface follows the configuration and registry separation of
Transformers'
[`attn_implementation`](https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.from_pretrained),
[`AttentionInterface`](https://huggingface.co/docs/transformers/main/attention_interface),
and
[`torch.compile` training configuration](https://huggingface.co/docs/transformers/torch_compile),
while remaining a VoiceHub-native implementation.

#### Multi-stage runtime optimization protocol

`OptimizationCompileTargetProvider` and `OptimizationModuleRootProvider`
separate the two optional hooks used by runtimes whose executed graph is not
one ordinary `nn.Module.forward()`. `OptimizationRuntimeProtocol` combines
both with checkpoint/device discovery:

```python
OptimizationModuleRoot(label: str, module: Any)
OptimizationCompileTarget(label: str, owner: Any, attribute: str)

OptimizationModuleRootProvider
OptimizationCompileTargetProvider
OptimizationRuntimeProtocol

runtime.optimization_module_roots()
runtime.optimization_compile_targets(mode: str)
runtime.parameters()
runtime.state_dict()
```

`optimization_module_roots()` returns the module trees searched by attention
and custom-kernel selector passes. `optimization_compile_targets()` returns
ordered bound methods that the requested inference or training path actually
invokes. An empty return is authoritative for an unsupported mode. Target
labels and owner/attribute pairs must be unique, and the target
owner/attribute must remain stable until restoration.
`parameters()` provides device and dtype discovery, while `state_dict()` must
return stable non-empty string keys so every pass can verify checkpoint
identity. Simple modules with a concrete `forward()` are discovered
automatically; inherited PyTorch `_forward_unimplemented` is never treated as
a valid compile target. Portable resolved plans do not retain instance-bound
targets; the pass discovers them from the loaded runtime when it is applied.

## Explicit optimization passes

`voicehub.optimization` provides a dependency-light pass contract that can be
used by both pretrained inference wrappers and `Trainer`:

```python
class OptimizationPass:
    pass_id: str
    pass_version: str
    optimization_kind: str | None
    requires_architecture_support: bool = False
    capabilities: OptimizationCapabilities

    def manifest_configuration(self) -> Mapping[str, Any]: ...
    def validate(self, model, context) -> None: ...
    def not_applicable_result(self, model, *, reason) -> PassResult: ...
    def apply(self, model, context) -> PassResult: ...
    def restore(self, model, state, context): ...
    def route_optimizer_parameters(
        self,
        model,
        *,
        optimizer_names,
    ) -> Mapping[str, Iterable[tuple[str, Parameter]]]: ...
    def export_portable_state(
        self,
        model,
        context,
    ) -> Mapping[str, Tensor]: ...


OptimizationPassManager.apply_plan(
    model,
    passes,  # name, pass object, or iterable mixing both
    context,
    *,
    registry=None,
) -> OptimizationResult
```

`OptimizationContext` declares `mode`, optional `architecture`, `device`,
`dtype`, streaming, distributed execution, and whether the result must be
persistable. Registered wrappers bind the canonical architecture
automatically; registered model specs with `architecture=None` remain
agnostic. Before applying any pass, the manager validates pass and architecture
device/dtype/mode/streaming constraints plus distributed-training capability.
The pass then validates any loaded runtime structure it finds. When a model has
no relevant protocol surface, a public pass returns
`not_applicable_result(model, reason=...)`; the model is unchanged and the
manifest records `outcome="not-applicable"` plus the reason. This is an explicit
universal fallback, not an acceleration claim or a silent skip. A present but
malformed protocol, an unsupported explicit backend, or incompatible hardware
still fails before mutation. A pass may set `requires_architecture_support=True`
for a manually audited compatibility kind. Architecture-bound distributed
inference is explicitly unsupported by the current schema. The manager rolls
back earlier reversible passes after a failure and returns ordered application
state. Declaring reversibility requires an actual `restore()` override.

`manifest_configuration()` is mandatory and must return every effective pass
option, including defaults, as a strict JSON string-key tree. The manager
snapshots pass ID, kind, version, capabilities, and configuration before
mutation, then snapshots result metadata. Architecture compatibility
declarations do not register executable pass factories. Use
`OptimizationResult.manifest()` for deterministic checkpoint metadata and
`OptimizationResult.restore()` only when every pass declares itself
reversible. `OptimizationResult.portable_state_dict(model=None)` returns
canonical save state, optionally from a strategy-unwrapped execution handle.

Optimization manifests are portable artifacts, not credential stores. The
shared strict-JSON boundary rejects nested credential-shaped fields in pass
configuration before mutation, in result metadata before publishing an
application, and in mutable runtime status before manifest output. Invalid
result metadata rolls the pass back. Descriptive fields such as `token_count`
remain serializable; pass credentials through a runtime-only constructor or
loader boundary.

Register extension passes globally with a function or decorator:

```python
register_optimization_pass("acme-pass", AcmePass)
unregister_optimization_pass("acme-pass")
```

The pass becomes available through every speech model's
`available_optimization_passes()` and `apply_optimization_plan()` methods.
Compatibility is still checked before mutation. See
[Add an optimization](../project/adding-an-optimization.md).

Built-in accelerator passes are available from `voicehub.optimization`:

```python
TorchCompilePass(
    backend="inductor",
    mode="max-autotune-no-cudagraphs",
    fullgraph=False,
    dynamic=True,
    requirement="auto",  # or "required"
)
CustomKernelPass(
    backend="auto",  # torch, triton, or cuda_extension
)
CodecKernelPass(
    backend="auto",  # torch, triton, cute, or cuda_extension
)
FlashAttention4Pass(
    policy="auto",  # disabled, auto, or required
)
```

All four are reversible configuration/execution passes and preserve canonical
state-dict keys. `CodecKernelPass` only visits codec-specific selectors, and
can resolve different effective backends for different codec operations.
`CustomKernelPass` never builds an extension. Call
`voicehub.kernels.load_tts_activation_cuda_extension()` explicitly before
selecting `backend="cuda_extension"`. `FlashAttention4Pass` imports the
optional `flash_attn.cute` package only for a compatible concrete attention
call.

`voicehub.kernels` exposes `KERNEL_REGISTRY`, `register_kernel()`,
`resolve_kernel()`, `dispatch_kernel()`, `KernelBackend`, and `KernelSupport`
for application-defined implementations. It also exposes
`get_kernel_capabilities()` and the explicit
`load_tts_activation_cuda_extension()` build seam, plus
`load_tts_activation_triton_kernels()` for eager activation before full-graph
capture. Importing this namespace does not import Triton, initialize CUDA, or
invoke a compiler.
`get_codec_kernel_capabilities()` additionally probes the optional
CuTe-backed CUTLASS Operator API used by DAC Euclidean VQ search.

Pretrained speech wrappers expose:

```python
model.apply_optimization_plan(
    passes,
    *,
    mode,
    context=None,
    registry=None,
) -> OptimizationResult
model.optimization_result(*, mode)
model.optimization_manifest(*, mode=None)
model.restore_optimization_plan(*, mode)
```

`Trainer` accepts `optimization_plan`, `optimization_context`, and
`optimization_pass_registry`. It exposes the applied result through
`trainer.optimization_result` and its checkpoint-safe record through
`trainer.optimization_manifest()`. These low-level APIs never select a plan
implicitly; the separate universal `optimization_config` argument is the
explicit request that asks a lazy TTS loader or Trainer to resolve one.
Trainer requires `mode="training"` and `persist_result=True`. A
topology/name-changing pass used with a separate-optimizer recipe must
implement complete routing; a topology/name-changing pass included in a
portable save must declare `portable_export=True` and return canonical state
through `export_portable_state()`.

## Training discovery and contracts

### Support levels

`TrainingSupport` is registry metadata, not a guarantee that every checkpoint
variant is trainable.

| Value | Contract |
| --- | --- |
| `TrainingSupport.NATIVE` (`"native"`) | Integrated runtime exposes a differentiable backend-native loss |
| `TrainingSupport.PREPROCESSED` (`"preprocessed"`) | Differentiable route is integrated; caller supplies backend-shaped data |
| `TrainingSupport.CUSTOM` (`"custom"`) | A model-specific adapter is required |
| `TrainingSupport.INFERENCE_ONLY` (`"inference-only"`) | Current integration has no verified gradient path |

`TrainingSupport.is_trainable` is `False` only for `INFERENCE_ONLY`. A `CUSTOM`
profile can still be gated when its required specialized adapter is absent.

```python
from voicehub import get_training_spec, list_training_specs, TrainingSupport

dia = get_training_spec("dia")
preprocessed = list_training_specs(
    support=TrainingSupport.PREPROCESSED,
)
```

```python
get_training_spec(model_type: str) -> ModelTrainingSpec
list_training_specs(
    *,
    task: SpeechTask | str | None = SpeechTask.TEXT_TO_SPEECH,
    support: TrainingSupport | str | None = None,
) -> tuple[ModelTrainingSpec, ...]
```

Omitting `task` preserves the historical TTS-only view. Pass `task=None` for
all registered speech tasks, or `task="asr"` / `task="vad"` for one
speech-input task.

### `ModelTrainingSpec`

`ModelTrainingSpec` is an immutable, framework-light recipe declaration.

| Field | Purpose |
| --- | --- |
| `model_type` | Canonical model key |
| `task` | `SpeechTask` owned by this training profile |
| `family` | A built-in `TrainingFamily` or custom non-empty family name |
| `support` | Capability boundary |
| `module_paths` | Ordered candidates for the primary trainable module |
| `component_paths` | Declared trainable component roots |
| `label_names` | Accepted target fields |
| `prediction_keys` | Output fields that can carry predictions |
| `loss_keys` / `loss_weights` | Native loss discovery and aggregation |
| `fallback_objective` | Explicit fallback objective, when allowed |
| `native_training` | Whether the source runtime owns its loss |
| `separate_optimizers` | Whether the recipe uses named optimizer routes |
| `phases` / `default_phase` | Phase declarations and default selection |
| `recipe_kind` | `single-phase`, `multi-phase`, or `adversarial` |
| `source_entrypoints` | Audited upstream training entry points |
| `allow_module_discovery` | Opt in to bounded module discovery |
| `training_default_model_name_or_path` | Recommended differentiable checkpoint |
| `field_schemas` | Dotted collator paths and their padding schemas |
| `adapter_factory` | Lazy `module:callable` path for a model-specific training adapter |
| `dataset_factory` | Lazy `module:callable` path for a source-native dataset builder |
| `tokenizer_paths` | Ordered wrapper-relative tokenizer paths used by generic exports |
| `optimization_profile_factory` | Lazy `module:callable` path for a special optimization profile |

Useful properties and methods:

| Member | Meaning |
| --- | --- |
| `family_name` | String form of the family |
| `supports_training` / `is_turnkey` | `True` for `native` or `preprocessed` |
| `has_training_recipe` | `True` for every value except `inference-only` |
| `requires_custom_adapter` | Whether support is `custom` |
| `phase_map` | Read-only phase-name mapping |
| `get_phase(name=None)` | Resolve a phase, defaulting to `default_phase` |
| `dataset_spec` | Architecture-aware TTS or ASR data contract |
| `install_extra` | `"training"` for built-in trainable profiles; otherwise an optional extension-owned setup identifier |

Built-in `TrainingFamily` values are:

```text
causal-lm
sequence-to-sequence
flow-matching
acoustic-regression
vits
composite
ctc
speech-sequence-to-sequence
rnnt
tdt
audio-classification
frame-classification
native-asr-dispatch
upstream-native
```

A custom non-empty family string is also valid when the profile declares an
`adapter_factory` or a reusable family factory is registered for it.

### `TrainingPhaseSpec`

```python
TrainingPhaseSpec(
    name: str,
    component_paths=(),
    optimizer_names=(),
    forward_component=None,
    forward_method="forward",
    label_names=("labels", "targets", "target"),
    prediction_keys=("logits", "predictions", "audio_values", "waveform"),
    loss_keys=("loss", "total_loss"),
    loss_weights=(),
    input_aliases=(),
    required_inputs=(),
    frequency=1,
    offset=0,
    fallback_objective=None,
    kind=TrainingPhaseKind.OBJECTIVE,
    detach_inputs=(),
    frozen_component_paths=(),
    optimizer_step_after_phase=False,
)
```

`frequency` and `offset` schedule a phase when
`step % frequency == offset`. Generator, discriminator, and duration
discriminator phases must declare optimizer names. Multiple optimizer names
must map one-to-one to component paths; one name may own all phase components.
With named separate optimizers, `optimizer_step_after_phase=True` creates an
immediate optimizer boundary before the next phase is recomputed. Every
scheduled phase must be routed and use the policy consistently, and the current exact
implementation requires `gradient_accumulation_steps=1`.

`TrainingPhaseKind` values are `objective`, `generator`, `discriminator`,
`duration-discriminator`, and `auxiliary`.

### `TrainingContext` and speech training outputs

`TrainingContext` carries:

```python
@dataclass(frozen=True)
class TrainingContext:
    phase: TrainingPhaseSpec
    inputs: Mapping[str, Any]
    step: int | None = None
    epoch: float | None = None
    is_training: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)
```

`inputs` and `metadata` become read-only mappings. `phase_name`,
`optimizer_names`, and `with_inputs(new_inputs)` are convenience members.

Adapters normalize a training forward into:

```python
@dataclass
class SpeechTrainingOutput:
    loss: Any | None = None
    logits: Any | None = None
    predictions: Any | None = None
    audio_values: Any | None = None
    hidden_states: Any | None = None
    attentions: Any | None = None
    training_phase: str | None = None
    optimizer_names: tuple[str, ...] = ()
    losses: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
```

Shared adapters return `TTSTrainingOutput`, a backward-compatible
`SpeechTrainingOutput` subclass, for TTS and `SpeechTrainingOutput` for ASR or
VAD. Both support populated-field `keys()`, string/integer access, iteration,
`to_tuple()`, and `to_dict()`. The `phase` property aliases
`training_phase`.

## Training adapter factory

### `AutoTrainingAdapter`

```python
AutoTrainingAdapter.from_model(
    model,
    *,
    spec: ModelTrainingSpec | None = None,
) -> BaseTrainingAdapter
```

The factory chooses, in order:

1. a process-local per-model override;
2. the profile's declarative `adapter_factory`; or
3. the adapter registered for the profile's family.

The declarative path is resolved only when that model's adapter is requested;
listing training profiles does not import adapter modules. It constructs an
unloaded adapter. `adapter.setup()` or
`adapter.build_training_graph()` performs training validation, calls the
wrapper's training lifecycle, and resolves trainable components.

Built-in family adapters:

| Family | Adapter |
| --- | --- |
| `causal-lm` | `CausalLMTrainingAdapter` |
| `sequence-to-sequence` | `Seq2SeqTrainingAdapter` |
| `flow-matching` | `FlowMatchingTrainingAdapter` |
| `acoustic-regression` | `AcousticTrainingAdapter` |
| `vits` | `VITSTrainingAdapter` |
| `composite` | `CompositeTrainingAdapter` |
| `ctc` | `CTCTrainingAdapter` |
| `speech-sequence-to-sequence` | `SpeechSeq2SeqTrainingAdapter` |
| `rnnt` | `RNNTTrainingAdapter` |
| `tdt` | `TDTTrainingAdapter` |
| `audio-classification` | `AudioClassificationTrainingAdapter` |
| `frame-classification` | `FrameClassificationTrainingAdapter` |
| `native-asr-dispatch` | Verified model-specific native ASR adapter |
| `upstream-native` | `UpstreamNativeTrainingAdapter` |

Important `BaseTrainingAdapter` extension points include:

```python
validate_support() -> None
build_training_graph()
create_dataset(records, **kwargs)
prepare_training_inputs(inputs, context)
prepare_batch(inputs, context)
execute_training_phase(context) -> SpeechTrainingOutput
execute_prediction_phase(context)
create_optimizer(name, parameters, training_args)
create_scheduler(name, optimizer, num_training_steps, training_args)
on_before_optimizer_step(*, optimizer_names, step) -> None
on_optimizer_step(*, optimizer_names, step) -> None
on_optimizer_step_skipped(*, optimizer_names, step) -> None
recipe_state_dict()
load_recipe_state_dict(state_dict, *, strict=True) -> None
save_pretrained(save_directory) -> None
```

`save_pretrained()` on an adapter writes only its optional source-native export.
Portable VoiceHub state is owned by `Trainer.save_model()`.

Adapter registry methods:

```python
AutoTrainingAdapter.register(
    model_type,
    adapter_class_or_factory,
    *,
    exist_ok=False,
) -> None

AutoTrainingAdapter.unregister(
    model_type,
    *,
    missing_ok=False,
)

AutoTrainingAdapter.register_family(
    family,
    factory,
    *,
    exist_ok=False,
) -> None

AutoTrainingAdapter.unregister_family(
    family,
    *,
    missing_ok=False,
)

AutoTrainingAdapter.available_models() -> tuple[str, ...]
AutoTrainingAdapter.available_families() -> tuple[str, ...]
```

`register_model_adapter()` and `unregister_model_adapter()` are explicit aliases
for the per-model methods.

## Training arguments

```python
TrainingArguments(output_dir="trainer_output", ...)
```

The names intentionally follow the Transformers vocabulary, while the current
built-in execution strategy is single-process PyTorch.

### Run and evaluation control

| Argument | Default | Meaning |
| --- | ---: | --- |
| `output_dir` | `"trainer_output"` | Checkpoint and default artifact root |
| `overwrite_output_dir` | `False` | Permit starting when a checkpoint already exists; does not delete it |
| `do_train` | `False` | Serialized compatibility flag; calling `train()` starts training |
| `do_eval` | `False` | Serialized compatibility flag; calling `evaluate()` starts evaluation |
| `eval_strategy` | `"no"` | Evaluation cadence: `no`, `steps`, or `epoch` |
| `evaluation_strategy` | `None` | Compatibility alias for `eval_strategy`; do not pass both |
| `prediction_loss_only` | `False` | Omit predictions and labels in the evaluation loop |
| `load_best_model_at_end` | `False` | Restore the best saved checkpoint after training |
| `metric_for_best_model` | `None` | Metric name; defaults to `loss` when best-model loading is enabled |
| `greater_is_better` | `None` | Inferred as `False` for names ending in `loss`, otherwise `True` |

### Batch and dataloader control

| Argument | Default | Meaning |
| --- | ---: | --- |
| `per_device_train_batch_size` | `8` | Training batch size |
| `per_device_eval_batch_size` | `8` | Evaluation/prediction batch size |
| `gradient_accumulation_steps` | `1` | Micro-batches per optimizer update |
| `eval_accumulation_steps` | `None` | Reserved compatibility setting |
| `dataloader_drop_last` | `False` | Drop incomplete final batches |
| `dataloader_num_workers` | `0` | DataLoader workers; exact generic mid-epoch resume requires `0` |
| `dataloader_pin_memory` | `True` | Pin DataLoader memory when the selected device is CUDA |
| `remove_unused_columns` | `True` | Filter batch keys against finite model signatures |
| `label_names` | `["labels"]` | Fields removed and passed to a custom loss function |

### Optimization

| Argument | Default | Meaning |
| --- | ---: | --- |
| `learning_rate` | `5e-5` | Default AdamW learning rate |
| `weight_decay` | `0.0` | Weight decay for non-bias, non-normalization parameters |
| `adam_beta1` | `0.9` | Adam first-moment coefficient |
| `adam_beta2` | `0.999` | Adam second-moment coefficient |
| `adam_epsilon` | `1e-8` | Adam numerical-stability value |
| `adamw_fused` | `False` | Request fused AdamW when all parameters are on CUDA and PyTorch supports it; otherwise fall back safely |
| `adamw_torch_compile` | `False` | Compile AdamW `step` with Inductor's no-CUDA-graphs mode; on CUDA this can generate fused Triton code |
| `max_grad_norm` | `1.0` | Gradient clipping norm; `0` disables effective clipping |
| `num_train_epochs` | `3.0` | Epoch target when `max_steps` is not positive |
| `max_steps` | `-1` | Positive value overrides the epoch-derived update count |
| `lr_scheduler_type` | `"linear"` | `linear`, `cosine`, `constant`, or epoch-normalized `exponential` |
| `lr_scheduler_gamma` | `1.0` | Per-epoch factor used by the exponential schedule |
| `warmup_ratio` | `0.0` | Fractional warmup when `warmup_steps` is zero |
| `warmup_steps` | `0` | Explicit warmup; takes precedence over the ratio |
| `gradient_checkpointing` | `False` | Enable only when the resolved runtime implements it |

### Logging, checkpointing, precision, and reproducibility

| Argument | Default | Meaning |
| --- | ---: | --- |
| `logging_strategy` | `"steps"` | `no`, `steps`, or `epoch` |
| `logging_steps` | `500` | Optimizer-update interval |
| `logging_first_step` | `False` | Log after the first optimizer update |
| `eval_steps` | `None` | Step interval; defaults to `logging_steps` for step evaluation |
| `save_strategy` | `"steps"` | `no`, `steps`, or `epoch` |
| `save_steps` | `500` | Optimizer-update checkpoint interval |
| `save_total_limit` | `None` | Maximum retained numeric checkpoints |
| `seed` | `42` | Python, NumPy, and framework seed |
| `data_seed` | `None` | Sampler seed; falls back to `seed` |
| `fp16` | `False` | CUDA float16 autocast and gradient scaling |
| `bf16` | `False` | bfloat16 autocast on a supported CPU or CUDA runtime |
| `use_cpu` | `False` | Force the trainer device to CPU |
| `disable_tqdm` | `True` | Compatibility flag; `False` enables the built-in printing callback |
| `report_to` | `[]` | Reporting backend name or names; supports `"wandb"`, `"all"`, and `"none"` |
| `run_name` | `None` | Human-readable reporting run name |
| `wandb_project` | `None` | W&B project; falls back to `WANDB_PROJECT`, then `"voicehub"` |
| `wandb_entity` | `None` | Optional W&B user or team |
| `wandb_group` | `None` | Optional W&B run group |
| `wandb_tags` | `[]` | Deduplicated W&B tags |
| `wandb_notes` | `None` | Optional W&B run notes |
| `wandb_mode` | `None` | `online`, `offline`, or `disabled`; `None` defers to the SDK/environment |
| `wandb_log_model` | `False` | `false`, `checkpoint`, or `end`; booleans normalize to `false`/`end` |

Important validation rules:

- batch sizes and gradient accumulation must be positive integers;
- `max_steps` is `-1` or a positive integer;
- `fp16` and `bf16` are mutually exclusive, and `fp16` training requires CUDA;
- reporting names and W&B modes/artifact policies are validated before a run;
- `load_best_model_at_end=True` requires matching non-`no` save/evaluation
  strategies; with step strategies, `save_steps` must be a multiple of
  `eval_steps`; and
- an iterable dataset without a stable length requires positive `max_steps`.

Serialization and derived properties:

```python
arguments.train_batch_size
arguments.eval_batch_size
arguments.device
arguments.get_warmup_steps(num_training_steps)
arguments.to_dict()
arguments.to_json_string()
arguments.save_json(path) -> Path
TrainingArguments.from_json_file(path)
```

Training arguments and their subclasses are portable configuration, not a
credential store. Construction, untrusted JSON loading, dictionary/string
conversion, file writes, and `Trainer.save_model()` preflight reject nested
credential-shaped fields. Keep Hub and reporting credentials in their
runtime-only loader, SDK, or environment boundary; safe metadata such as
`token_count` remains serializable. Loading also rejects duplicate object keys
and non-finite numbers before constructing the argument object.

`device` resolves to CPU when `use_cpu=True`; otherwise it selects CUDA, MPS,
then CPU.

## Trainer

### Constructor

```python
Trainer(
    model=None,
    args: TrainingArguments | None = None,
    data_collator=None,
    train_dataset=None,
    eval_dataset=None,
    processing_class=None,
    model_init=None,
    compute_loss_func=None,
    compute_metrics=None,
    callbacks=None,
    optimizers=(None, None),
    optimizer_cls_and_kwargs=None,
    preprocess_logits_for_metrics=None,
    training_adapter=None,
    optimizer_factory=None,
    scheduler_factory=None,
    training_strategy=None,
)
```

| Parameter | Contract |
| --- | --- |
| `model` | Concrete wrapper or trainable module |
| `args` | `TrainingArguments`; defaults are constructed when omitted |
| `data_collator` | Explicit callable; has highest collation precedence |
| `train_dataset` / `eval_dataset` | Sized datasets, iterable datasets, or evaluation split mapping |
| `processing_class` | Retained for saving and callbacks; does not preprocess raw records implicitly |
| `model_init` | Zero-argument model factory used instead of `model` |
| `compute_loss_func` | `(outputs, labels, num_items_in_batch) -> loss` for a single custom loss boundary |
| `compute_metrics` | `(EvalPrediction) -> dict[str, float]` |
| `callbacks` | Callback classes or instances |
| `optimizers` | Preconstructed `(optimizer, scheduler)` pair |
| `optimizer_cls_and_kwargs` | Optimizer class and constructor kwargs |
| `preprocess_logits_for_metrics` | `(logits, labels) -> processed_logits` |
| `training_adapter` | Explicit `BaseTrainingAdapter` wrapping the same model |
| `optimizer_factory` | `(name, named_parameters, args) -> optimizer` |
| `scheduler_factory` | `(name, optimizer, num_training_steps, args) -> scheduler` |
| `training_strategy` | Registered name or `TrainingStrategy` instance |

Pass exactly one of `model` and `model_init`. A concrete `training_adapter` or
preconstructed optimizer cannot be reused with `model_init`.

Collator selection order is:

1. explicit `data_collator`;
2. callable `train_dataset.collate_fn`;
3. the selected training adapter's collator; or
4. `default_data_collator`.

### Minimal loop

The dataset must already satisfy the selected model recipe, unless the
integration supplies `create_training_dataset()`.

```python
from voicehub import Trainer, TrainingArguments

args = TrainingArguments(
    output_dir="runs/voicehub",
    max_steps=1,
    per_device_train_batch_size=1,
    logging_steps=1,
    save_strategy="no",
)

trainer = Trainer(
    model=training_model,
    args=args,
    train_dataset=train_dataset,
    processing_class=training_model.processor,
)

result = trainer.train()
print(result.global_step, result.training_loss)
```

### Public methods

| Method | Return | Notes |
| --- | --- | --- |
| `train(resume_from_checkpoint=None)` | `TrainOutput` | `True` selects the newest complete checkpoint; a path selects one explicitly |
| `evaluate(eval_dataset=None, metric_key_prefix="eval")` | Metrics dictionary | A mapping of named datasets is evaluated one split at a time |
| `predict(test_dataset, metric_key_prefix="test")` | `PredictionOutput` | Returns predictions, labels, and prefixed metrics |
| `save_model(output_dir=None, include_native_export=True, portable=True)` | `Path` | Write canonical portable state by default; `portable=False` is for exact internal checkpoints |
| `save_state()` | `Path` | Write only root `trainer_state.json`; this is not an exact-resume checkpoint |
| `compute_loss(model, inputs, return_outputs=False, num_items_in_batch=None)` | Loss or `(loss, outputs)` | Override point for the scalar loss boundary |
| `training_step(model, inputs, num_items_in_batch=None, sync_gradients=True)` | Detached loss | One prepared/backpropagated micro-batch |
| `prediction_step(model, inputs, prediction_loss_only)` | `(loss, predictions, labels)` | One no-gradient batch |
| `get_train_dataloader()` | Prepared DataLoader | Deterministically shuffled for sized datasets |
| `get_eval_dataloader(eval_dataset=None)` | Prepared DataLoader | Deterministic, unshuffled loader |
| `get_test_dataloader(test_dataset)` | Prepared DataLoader | Prediction loader |
| `add_callback(callback)` | `None` | Add class or instance |
| `pop_callback(callback)` | Callback or `None` | Remove and return first matching type |
| `remove_callback(callback)` | `None` | Remove first matching type |
| `log(logs)` | `None` | Normalize, store, and dispatch metrics |
| `get_learning_rate()` | `float` | First optimizer-group learning rate |
| `get_learning_rates()` | `list[float]` | Every optimizer-group learning rate |
| `get_num_trainable_parameters()` | `int` | Count parameters with gradients enabled |

When `report_to="wandb"`, Trainer adds `WandbCallback` automatically. The
integration remains lazy and runs only on the world-primary process.
`wandb_log_model="checkpoint"` uploads after an atomic checkpoint has
completed; `"end"` writes `output_dir/final-model` and uploads that portable
artifact before a VoiceHub-owned W&B run is finished.

`TrainOutput` is `(global_step, training_loss, metrics)`.
`PredictionOutput` is `(predictions, label_ids, metrics)`.
`EvalPrediction` passed to `compute_metrics` contains `predictions`,
`label_ids`, and optional `inputs`.

Metric keys returned by `compute_metrics` receive the active prefix unless they
already have it. Evaluation always adds `<prefix>_samples` and adds
`<prefix>_loss` when loss values are available.

## Callbacks

Subclass `TrainerCallback` and override only the events needed:

```python
class TrainerCallback:
    def resume_fingerprint(self): ...
    def state_dict(self): ...
    def load_state_dict(self, state_dict) -> None: ...
    def on_init_end(self, args, state, control, **kwargs): ...
    def on_train_begin(self, args, state, control, **kwargs): ...
    def on_train_end(self, args, state, control, **kwargs): ...
    def on_train_error(self, args, state, control, **kwargs): ...
    def requires_final_model(self, args, state): ...
    def on_final_model_saved(self, args, state, control, **kwargs): ...
    def on_epoch_begin(self, args, state, control, **kwargs): ...
    def on_epoch_end(self, args, state, control, **kwargs): ...
    def on_step_begin(self, args, state, control, **kwargs): ...
    def on_substep_end(self, args, state, control, **kwargs): ...
    def on_step_end(self, args, state, control, **kwargs): ...
    def on_evaluate(self, args, state, control, **kwargs): ...
    def on_predict(self, args, state, control, **kwargs): ...
    def on_save(self, args, state, control, **kwargs): ...
    def on_checkpoint_saved(self, args, state, control, **kwargs): ...
    def on_log(self, args, state, control, **kwargs): ...
    def on_prediction_step(self, args, state, control, **kwargs): ...
```

Return the supplied or modified `TrainerControl`. Its public signals are
`should_training_stop`, `should_epoch_stop`, `should_save`,
`should_evaluate`, and `should_log`.

Stateful callbacks should return exact-continuation configuration from
`resume_fingerprint()`, mutable checkpoint state from `state_dict()`, and
restore it in `load_state_dict()`.

Exact-resume fingerprints and checkpoint manifests are portable artifacts,
not credential stores. Dataset, collator, callback, optimizer, and scheduler
fingerprints reject nested credential-shaped fields after normalization. The
complete manifest is checked again before its atomic write, including values
added by a `Trainer` subclass, and untrusted manifests are checked before any
model or runtime state is restored. Trainer-owned model, optimizer, scheduler,
random-generator, gradient-scaler, callback, sampler, and strategy state
mappings are checked at their final binary write boundary. They are checked
again after deserialization and before `load_state_dict()` or any other
checkpoint state application. Keep credentials in runtime-only model or
service configuration. Safe identity and metric fields such as `dataset_id`
and `token_count` remain valid fingerprint and checkpoint data.

Checkpoint discovery and restoration reject duplicate keys and non-finite
numbers in the checkpoint, optimization, and Trainer-state JSON documents.
Discovery ignores such an invalid candidate, and explicit resume fails with
the artifact source and duplicate key or numeric path before model or runtime
state restoration.

The post-deserialization check does not make Python pickle safe. Resume only
from a trusted VoiceHub checkpoint, and retain the versioned manifest and its
file-integrity records.

`EarlyStoppingCallback` is provided:

```python
EarlyStoppingCallback(
    early_stopping_patience: int = 1,
    early_stopping_threshold: float = 0.0,
)
```

It requires `load_best_model_at_end=True` and a
`metric_for_best_model`.

`WandbCallback` is also public and is normally registered through
`TrainingArguments(report_to="wandb")`. It lazily initializes or reuses a W&B
run, logs phase-namespaced metrics, stores its run ID in callback state,
optionally uploads complete model artifacts, and closes only runs it owns.

`TrainerState` exposes serializable progress including `epoch`, `global_step`,
`max_steps`, interval values, `log_history`, best metric/checkpoint, and exact
dataloader cursor fields. Use `save_to_json(path)` and
`TrainerState.load_from_json(path)` for state-only serialization.

Trainer logs and state files are portable artifacts, not credential stores.
`Trainer.log()`, `TrainerState` construction, untrusted state loading, and the
final state write reject nested credential-shaped fields before callback
dispatch, state mutation, or filesystem creation. Keep access tokens and
provider credentials in runtime-only model or service configuration. Metric
names such as `token_count` remain valid. State loading also rejects duplicate
keys and non-finite values before constructing `TrainerState`.

## Data collators

### `default_data_collator`

```python
default_data_collator(
    features: list[Any],
    return_tensors: str = "pt",
) -> dict[str, Any]

DefaultDataCollator(return_tensors="pt")
```

The default collator stacks already equal-shaped tensors and numeric values. It
maps `label` or `label_ids` to `labels`, preserves strings and unsupported
metadata as lists, and currently supports only PyTorch output. It does not pad
variable-length TTS sequences.

### `DataCollatorForTTSTraining`

```python
DataCollatorForTTSTraining(
    padding_value: float = 0.0,
    label_pad_token_id: int = -100,
    return_attention_mask: bool = True,
    return_input_lengths: bool = False,
    field_schemas: Mapping[str, TTSFieldSchema | Mapping] | None = None,
)
```

This collator recursively handles nested mappings and dataclasses. It stacks
equal shapes, pads unambiguous variable first/last dimensions, uses `-100` for
integer labels, and uses `padding_value` for other sequences. Strings and
unsupported ambiguous values remain lists.

`training_phase` is a batch-level control: every sample in one batch must
select the same value.

```python
TTSFieldSchema(
    sequence_dim: int = 0,
    padding_value: float | int | None = None,
    padding_side: str = "right",
    length_field: str | None = None,
    mask_field: str | None = None,
    pad_to_multiple_of: int | None = None,
    allow_missing: bool = False,
)
```

Schema paths are dotted, such as `"model_inputs.mel"`. A derived field name
without a dot is written beside its source; a dotted derived name is written
from the batch root. Masks have shape `(batch, padded_sequence_length)`.

```python
from voicehub import DataCollatorForTTSTraining, TTSFieldSchema

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

`resume_fingerprint()` returns all options that can change exact resumed
batching. The collator is structural: it does not invent codec delays, flow
targets, acoustic alignments, or adversarial pairs. Empty batches raise, and a
caller-provided derived length or mask must exactly match the value computed
from its source tensor.

### `SpeechDataset` and `DataCollatorForAudioTraining`

```python
SpeechDataset(
    records: Iterable[Mapping[str, Any]],
    *,
    required_fields: Iterable[str] = (),
    transform: Callable[[dict[str, Any]], Mapping[str, Any]] | None = None,
)

DataCollatorForAudioTraining(
    padding_value: float = 0.0,
    label_pad_token_id: int = -100,
    return_attention_mask: bool = True,
    return_input_lengths: bool = False,
    field_schemas: Mapping[str, AudioFieldSchema | Mapping] | None = None,
)
```

`SpeechDataset` validates and copies dependency-light source records without
decoding audio. Its optional transform runs at item access, and
`column_names` reports first-seen fields.

`DataCollatorForAudioTraining` uses the same recursive structural rules as the
TTS collator, with `AudioFieldSchema` declarations for waveform, feature,
token, or frame time dimensions. It does not infer CTC blanks, transducer
alignments, decoder prompts, or frame labels. See the
[ASR and VAD data guide](../guides/speech-data.md#collate-variable-length-audio-fields)
for a schema-based example.

### `ASRDataset` and ASR data contracts

```python
ASRDataset(
    records,
    *,
    model_type: str | None = None,
    architecture: ASRDataArchitecture | str | None = None,
    root=None,
    aliases=None,
    validate=True,
    validate_files=False,
    batching: TTSBatchingConfig | Mapping | None = None,
    transform=None,
    transform_fingerprint=None,
)

ASRDataset.coerce(
    records_or_manifest,
    *,
    model_type=None,
    architecture=None,
    root=None,
    aliases=None,
    validate=True,
    validate_files=False,
    transform_fingerprint=None,
) -> ASRDataset

ASRDataset.from_manifest(
    path,
    *,
    model_type=None,
    architecture=None,
    root=None,
    aliases=None,
    validate=True,
    validate_files=False,
    delimiter=None,
    transform=None,
    transform_fingerprint=None,
) -> ASRDataset

ASRDataset.from_audio_folder(
    root,
    *,
    model_type=None,
    architecture=None,
    transcript_extension=".txt",
    recursive=True,
    metadata=None,
    validate_files=True,
    transform=None,
    transform_fingerprint=None,
) -> ASRDataset

ASRDataset.from_kaldi(
    root,
    *,
    model_type=None,
    architecture=None,
    wav_scp="wav.scp",
    text_file="text",
    metadata=None,
    validate_files=False,
    transform=None,
    transform_fingerprint=None,
) -> ASRDataset

get_asr_dataset_spec(
    model_type: str | None = None,
    *,
    architecture: ASRDataArchitecture | str | None = None,
) -> ASRDatasetSpec

list_asr_dataset_specs() -> tuple[ASRDatasetSpec, ...]

ASRRecordVariant(
    name: str,
    required_fields=(),
    one_of=(),
    at_most_one_of=(),
    forbidden_fields=(),
    requires=(),
    requires_one_of=(),
    description="",
    preprocessed=False,
)

ASRDatasetSpec(
    architecture: ASRDataArchitecture,
    variants: tuple[ASRRecordVariant, ...],
    model_type=None,
    sample_rate=None,
    description="",
    readiness=None,
    training_support=None,
    homogeneous_batch_fields=(),
    field_aliases=(),
    record_normalizer=None,
    record_normalizer_phase="after-aliases",
)

variant.missing(record) -> tuple[str, ...]
variant.matches(record) -> bool

spec.match_variant(record, *, index=None) -> str
spec.raw_variants -> tuple[ASRRecordVariant, ...]
spec.preprocessed_variants -> tuple[ASRRecordVariant, ...]
spec.accepts_raw_records -> bool
spec.requires_preprocessing -> bool
spec.requires_homogeneous_batches -> bool

EpochGroupedBatchSampler(
    dataset: ASRDataset,
    *,
    batch_size: int,
    seed: int,
    shuffle: bool,
    drop_last: bool,
)

sampler.set_epoch(epoch: int) -> None
sampler.state_dict() -> dict
sampler.load_state_dict(state_dict) -> None
```

`ASRDataset` reads JSON, JSON Lines, CSV, and TSV, normalizes common audio,
transcript, language, and sample-rate aliases, resolves relative audio paths,
and validates model-specific source or cached-tensor variants. It can also
pair recursively discovered `.wav` files with same-stem transcript sidecars,
or import a simple Kaldi/ESPnet `wav.scp` plus `text` directory. Native
preprocessors decode PCM WAVE; custom transforms can materialize other
encodings. Kaldi shell pipelines are rejected.

JSON objects in JSON manifests, JSON Lines records, and JSON-shaped CSV/TSV
fields use the shared strict decoder. Duplicate keys, `NaN`, infinities, and
numeric overflow fail before dataset construction with file, line, and field
context where available. A malformed JSON-shaped tabular value remains a
string for backward-compatible scalar coercion. A `.json` file falls back to
NeMo-style JSON Lines only after a syntax error, never after a strict-decoder
violation.

ASR manifest loading and `to_jsonl()` apply the same credential boundary as
TTS. Nested credentials fail before dataset construction or destination
mutation, while descriptive fields such as `token_count` remain portable.

The dataset exposes:

| Member | Contract |
| --- | --- |
| `spec` | Resolved `ASRDatasetSpec` |
| `variant_names` | Matching source/preprocessed variant for each row |
| `train_test_split(validation_fraction=0.1, seed=42, group_by=None)` | Deterministic optional speaker/session-disjoint split |
| `to_jsonl(path, relative_to=None)` | Portable normalized manifest export |
| `resume_fingerprint()` | Stable content/order identity; transformed datasets require `transform_fingerprint` |
| `create_batch_sampler(...)` | Deterministic homogeneous grouping for models that require it |

`ASRDataArchitecture` values are `native-dispatch`, `ctc`,
`speech-sequence-to-sequence`, `prompted-multimodal`, `rnnt`, `tdt`, and
`hybrid-ctc-attention`. `ASRDataReadiness` uses the same
`integrated-raw`, `preprocessed`, `custom`, and `unavailable` meanings as the
TTS data layer.

An `ASRDatasetSpec` exposes raw and preprocessed `ASRRecordVariant` values,
sample rate, training support, readiness, and any
`homogeneous_batch_fields`. Cohere contracts group by `language` and
`punctuation`; SeamlessM4T-v2 groups by target language. The Trainer requests
the dataset's epoch-aware `EpochGroupedBatchSampler` automatically, including
for evaluation.

Architecture-specific source spellings are declarative. `field_aliases`
contains ordered source/target pairs, while `record_normalizer` is an optional
lazy `module:attribute` path. Its callable receives a copied record and
keyword-only `index`, and must return a mapping. `record_normalizer_phase`
selects `before-aliases` or `after-aliases`. Listing dataset specifications does
not import these normalizers; `ASRDataset` resolves and validates one only when
constructing records. Keep the framework-free normalizer beside the owning
architecture rather than adding a model-name branch to the shared dataset.

`ModelTrainingSpec.dataset_spec` returns a model-specific `ASRDatasetSpec` for
ASR profiles. Before weights load, use either
`get_training_spec(model_type).dataset_spec` or
`get_asr_dataset_spec(model_type)`. After model construction,
`model.validate_training_support().dataset_spec` provides the same contract.
Passing a manifest path to `PreTrainedASRModel.create_training_dataset()`
coerces it through `ASRDataset`; `data_root`, `data_aliases`,
`validate_records`, and `validate_audio_files` customize that boundary.

Transcript-bearing evaluation records are treated as references for native
teacher-forced evaluation, so the Trainer can report `eval_loss`. That value
does not imply generation WER or CER. Those metrics require model-appropriate
decoding and explicit hypothesis/reference normalization; specialized
adapters may add them.

See the [ASR and VAD data guide](../guides/speech-data.md) for portable
manifest examples and the architecture-specific record matrix.

### `TTSDataset` and TTS data contracts

```python
TTSDataset.from_manifest(
    path,
    *,
    model_type: str | None = None,
    architecture: TTSDataArchitecture | str | None = None,
    root=None,
    aliases=None,
    validate=True,
    validate_files=False,
    transform=None,
    transform_fingerprint=None,
) -> TTSDataset

get_tts_dataset_spec(
    model_type: str | None = None,
    *,
    architecture: TTSDataArchitecture | str | None = None,
) -> TTSDatasetSpec
```

`TTSDataset` reads JSON, JSON Lines, CSV, TSV, and LJSpeech metadata without
importing a tensor framework. It normalizes common text/audio aliases and the
selected specification's declarative `field_aliases`, resolves paths, validates
record variants, performs deterministic group-disjoint splits, writes portable
JSON Lines, and fingerprints normalized record content and order. Ordered
source/target pairs override shared aliases; an identity pair preserves a
model-canonical field spelling.

JSON objects in JSON manifests, JSON Lines records, and JSON-shaped CSV/TSV
fields use the shared strict decoder. Duplicate keys, `NaN`, infinities, and
numeric overflow fail before dataset construction with file, line, and field
context where available. A malformed JSON-shaped tabular value remains a
string for backward-compatible scalar coercion.

Portable dataset manifests are not credential stores. TTS manifest loading and
`to_jsonl()` reject nested credential-shaped fields; export validates and
serializes every record before creating or truncating the destination. Keep
dataset-service credentials in runtime-only loaders. Descriptive metadata such
as `token_count` remains portable.

`with_batching(config)` returns a new dataset with an immutable
`TTSBatchingConfig`. `Trainer` then requests an `EpochLengthBatchSampler`
automatically. `length-bucket` uses fixed item counts inside ordered
boundaries; `max-units` supports summed or padded token/frame budgets,
optional maximum item and sequence limits, deterministic `set_epoch()`, and
exact-resume state. Batching settings and normalized lengths are included in
dataset and sampler fingerprints.

Lazy transforms must declare a stable `transform_fingerprint` before
`resume_fingerprint()` can be used; changing that value changes the content
fingerprint. This prevents an exact resume from silently accepting changed
materialization logic.

`TTSDataArchitecture` values are `codec-lm`, `sequence-to-sequence`,
`diffusion`, `vits`, `acoustic`, and `hybrid`. A model-specific
`TTSDatasetSpec` exposes `variants`, `sample_rate`, `training_support`,
`readiness`, and normalized `field_aliases`. `TTSDataReadiness` values are:

| Value | Meaning |
| --- | --- |
| `integrated-raw` | At least one ordinary source-record preparation path is integrated |
| `preprocessed` | The caller must supply a declared backend-shaped variant |
| `custom` | A source-owned data adapter or orchestration step is still required |
| `unavailable` | The current model runtime has no verified training route |

Each `TTSRecordVariant` declares `required_fields` and alternative `one_of`
groups. It may also reject ambiguous aliases through `at_most_one_of`, exclude
incompatible source forms through `forbidden_fields`, and express dependent
metadata through `requires` or `requires_one_of`. These checks validate the
portable record boundary; the model processor remains responsible for tensor
rank, dtype, value range, and sample-rate checks.

For the six built-in TTS training families, `ModelTrainingSpec.dataset_spec`
lazily resolves the profile's `dataset_spec_factory`. A custom training-family
string can select a generic contract directly with
`get_tts_dataset_spec(architecture=...)`. Generic architecture contracts may
describe raw corpus structures; model-specific contracts do not inherit raw
support unless it is integrated.

### Source-specific TTS training profiles

```python
get_tts_training_optimization_profile(
    model_type_or_architecture,
) -> TTSTrainingOptimizationProfile

VITSOptimizationConfig().training_arguments(output_dir, **overrides)
LLMTTSOptimizationConfig().training_arguments(output_dir, **overrides)
LLMTTSOptimizationConfig.qwen3tts()
DiffusionTTSOptimizationConfig().training_arguments(output_dir, **overrides)

VITSOptimizationConfig().acceleration_plan(...)
LLMTTSOptimizationConfig().acceleration_plan(...)
DiffusionTTSOptimizationConfig().acceleration_plan(...)

vits_acceleration_plan(...)
llm_tts_acceleration_plan(...)
diffusion_tts_acceleration_plan(...)

list_vits_model_optimization_support()
get_vits_model_optimization_support(model_type)
```

`TTSTrainingOptimizationProfile` is the union of
`VITSOptimizationConfig`, `LLMTTSOptimizationConfig`, and
`DiffusionTTSOptimizationConfig`. The historical
`TTSOptimizationProfile` name is an alias for the same training union; neither
is the universal, constructible `TTSOptimizationConfig`.

Each profile exposes `batching_config()`, `prepare_dataset(dataset)`,
`techniques`, `source_url`, and `to_dict()`. The diffusion profile additionally
returns EMA and activation-checkpoint settings through
`model_config_overrides()`. `acceleration_plan()` returns custom-kernel and
attention passes followed by `TorchCompilePass`; VITS intentionally omits FA4
because its relative-position terms are not equivalent to dense scaled
dot-product attention. `VITSOptimizationConfig.acceleration_plan()` also
accepts `cuda_graphs="auto" | "disabled" | "required"` (or a boolean).
Required graphs select static shapes and `reduce-overhead`; the default
training policy retains dynamic shapes and
`max-autotune-no-cudagraphs`. Profiles are opt-in and do not mutate a model or
existing arguments. See
[VITS-family optimization](../guides/vits-optimization.md) and
[TTS optimization](../guides/tts-optimization.md) for the
pinned recipes and tradeoffs.

For model-specific lookup, the resolver reads
`ModelTrainingSpec.optimization_profile_factory`. The value is a lazy
`module:callable` import path whose zero-argument callable must return a profile
implementing the methods above. A model that shares a verified profile can
declare that factory in its training specification without adding its name to
the resolver. A model without a source-verified factory fails explicitly;
belonging to the same data architecture does not make another model's optimizer
recipe interchangeable.

### Specialized TTS objective primitives

The following framework-lazy helpers enforce exact shapes and explicit masks:

```python
multi_codebook_cross_entropy(...)
build_diffusion_training_pair(...) -> DiffusionTrainingPair
build_flow_matching_training_pair(...) -> DiffusionTrainingPair
masked_diffusion_regression_loss(...)

vits_discriminator_loss(...) -> VITSDiscriminatorLoss
vits_generator_adversarial_loss(...)
vits_feature_matching_loss(...)
vits_kl_loss(...)
```

The diffusion builder delegates alpha/sigma coefficients to the selected
recipe and supports epsilon, velocity, or clean-sample targets. The flow
builder uses a linear continuous path. VITS helpers implement multiscale
least-squares adversarial losses, detached-real feature matching, and masked
diagonal-Gaussian KL. They provide objective math, not missing tokenizers,
codecs, schedulers, posterior/alignment graphs, discriminators, or checkpoint
assets.

## Training strategies

`TrainingStrategy` owns device, precision, backward, optimizer execution,
distributed synchronization, metric gathering, and runtime state. The built-in
`TorchTrainingStrategy` is named `"torch"` and is single-process.

Custom strategies can override these exact hooks:

```python
prepare_device(model, *, device)
prepare_model(model, *, device)
prepare_training_adapter(adapter, *, device)
prepare_optimization(model, optimizer, scheduler)
prepare_dataloader(dataloader, *, training)
prepare_input(value, *, device)
autocast_context(args)
create_grad_scaler(args)
backward(loss, *, scaler=None) -> None
normalize_gradients(optimizer, microstep_counts) -> None
clip_grad_norm(
    parameters,
    max_norm,
    *,
    optimizer=None,
    scaler=None,
    optimizer_names=None,
)
optimizer_step(
    optimizer,
    *,
    scaler=None,
    optimizer_names=None,
) -> bool
scheduler_step(
    scheduler,
    *,
    optimizer_names=None,
    metric=None,
) -> None
zero_grad(optimizer, *, optimizer_names=None) -> None
no_sync(model, *, enabled)
execute_training_phase(model, adapter, context)
execute_prediction_phase(model, adapter, context)
gather_for_metrics(value)
state_dict() -> dict
load_state_dict(state_dict) -> None
resume_signature() -> dict
unwrap_model(model)
```

`optimizer_step()` returns whether the update succeeded. Mixed-precision
overflow can therefore skip scheduler and recipe-state updates.
`resume_signature()` must record topology that affects exact continuation,
such as world size and sharding layout.

Registry functions:

```python
list_training_strategies() -> tuple[str, ...]
get_training_strategy(strategy: str | TrainingStrategy | None = None)
register_training_strategy(name, factory, *, exist_ok=False) -> None
unregister_training_strategy(name) -> None
```

Factories are constructed lazily and must return `TrainingStrategy`. The
built-in `"torch"` strategy cannot be unregistered.

```python
from voicehub import (
    TorchTrainingStrategy,
    register_training_strategy,
    unregister_training_strategy,
)


class InstrumentedTorchStrategy(TorchTrainingStrategy):
    name = "instrumented-torch"


register_training_strategy(
    "instrumented-torch",
    InstrumentedTorchStrategy,
)
try:
    trainer = Trainer(
        model=training_model,
        args=training_args,
        train_dataset=train_dataset,
        training_strategy="instrumented-torch",
    )
finally:
    unregister_training_strategy("instrumented-torch")
```

`OptimizerBundle` and `SchedulerBundle` expose multiple named optimization
objects while allowing each phase to step only its declared routes. Their
`state_dict()` and strict `load_state_dict()` preserve the named topology.

## Training extension registries

### Training specifications and aliases

```python
register_training_spec(
    spec: ModelTrainingSpec,
    *,
    exist_ok: bool = False,
    aliases: Iterable[str] = (),
) -> None

unregister_training_spec(
    model_type: str,
    *,
    missing_ok: bool = False,
) -> ModelTrainingSpec | None

register_training_alias(
    alias: str,
    model_type: str,
    *,
    exist_ok: bool = False,
) -> None

unregister_training_alias(
    alias: str,
    *,
    missing_ok: bool = False,
) -> str | None
```

Registering a training specification does not register a new inference backend.
It attaches a recipe contract to a model type or supports a future
training-only integration. Aliases cannot collide with canonical model types.
Inference-alias collisions are rejected by default; `exist_ok=True` permits
only an alias that resolves to the same canonical target.

`adapter_factory`, `dataset_factory`, and `optimization_profile_factory`, when
present, must use a validated `module:callable` path. Registration stores these
paths without importing their modules. `AutoTrainingAdapter` resolves
`adapter_factory` only when that model's adapter is constructed. A codec-LM
adapter resolves `dataset_factory` only
when `create_training_dataset()` is called, then invokes it as
`factory(model, records, **kwargs)`. Keep that callable beside the model's
training implementation. Keep a specialized adapter beside the owning model or
architecture and declare it on the same profile instead of adding the model to
a shared map. The optimization resolver likewise imports and structurally
validates its factory only when the profile is requested.

`tokenizer_paths` contains validated dotted attribute paths relative to the
model wrapper. Generic codec-LM export selects the first resolved object and
calls its `save_pretrained()` method. Declare a nonstandard layout here instead
of branching on the model type in a shared adapter.

```python
from voicehub import (
    get_training_spec,
    ModelTrainingSpec,
    TrainingFamily,
    TrainingSupport,
    register_training_spec,
    unregister_training_spec,
)

profile = ModelTrainingSpec(
    model_type="exampletts",
    family=TrainingFamily.CAUSAL_LM,
    module_paths=("model",),
    support=TrainingSupport.PREPROCESSED,
)

register_training_spec(profile, aliases=("example-tts",))
try:
    resolved = get_training_spec("example-tts")
    assert resolved.model_type == "exampletts"
finally:
    unregister_training_spec("exampletts")
```

All extension registries are process-global. Use `exist_ok=True` only for an
intentional replacement, and clean up temporary registrations in tests.

## Save, load, and resume boundaries

VoiceHub deliberately separates metadata, portable model state, optional native
exports, and exact-resume checkpoints.

### Model metadata

```python
model.save_pretrained(
    save_directory,
    *,
    include_native_export=True,
) -> Path
```

The common wrapper writes task-specific request metadata:

```text
config.json
processor_config.json
generation_config.json       # TTS
transcription_config.json    # ASR
vad_config.json              # VAD
native_export/             # optional, backend-defined
```

Exactly one of the three task configuration files is written by a normal
task-specific wrapper.

JSON documents routed through the shared configuration, processor, native-
export, or Trainer writer are encoded as finite JSON before their parent
directory is created. The encoded document is flushed to a temporary sibling
and atomically replaces the destination. A serialization or replacement
failure therefore preserves an existing document and removes the temporary
file. This is a per-document guarantee; a backend that writes several native
files must still use its staging-directory or manifest contract for an atomic
multi-file export.

The shared reader rejects duplicate object keys, `NaN`, positive or negative
`Infinity`, and exponent overflow before configuration or model dispatch,
Trainer object construction, checkpoint discovery, or exact-resume state
restoration. Diagnostics identify the source and duplicate key or numeric path
without printing discarded values. Ordinary JSON syntax errors retain their
parser error type, and finite descriptive metadata such as `token_count`
continues to round-trip.

Shared JSON artifacts are also bounded to 64 MiB before decoding. Internal
loaders that own a smaller format may lower the byte ceiling explicitly. A file
that is already larger than the ceiling, or grows past it while being read,
fails with source and size-or-limit context without including document content.

The same strict decoder protects `VoiceHubManifest.load()`, bounded
Safetensors headers, and sharded Safetensors indexes. Ambiguous or non-finite
checkpoint metadata therefore fails before manifest construction, tensor
materialization, or shard lookup. Valid deterministic Safetensors files and
manifest metadata retain their existing format.

Portable `model_state.pt` restoration also validates the sibling `config.json`
through this decoder even when the caller supplies an explicit configuration.
Both the TTS and audio-input base loaders fail before wrapper construction on
an ambiguous artifact, while a valid saved `name_or_path` continues to restore
the original base-checkpoint identity.

The common method does not itself write a generic `model_state.pt`.
Backend-specific `_save_pretrained()` hooks may write native artifacts under
`native_export/`.

Native artifacts use `VoiceHubManifest` for portable architecture, checkpoint,
provenance, file-integrity, and additional metadata. Manifest `metadata` must
contain finite JSON values and cannot contain top-level or nested
credential-shaped fields. Construction, untrusted manifest loading,
`to_dict()`, and `save()` all enforce that boundary, including the final
mapping returned by a subclass. Pass credentials only at runtime; a rejected
save creates neither `voicehub_manifest.json` nor a temporary manifest file.
Legitimate descriptive fields such as `token_count` remain serializable.

### Portable trained artifact

```python
trainer.save_model(
    output_dir=None,
    *,
    include_native_export=True,
    portable=True,
) -> Path
```

Typical output:

```text
config.json
processor_config.json
generation_config.json       # TTS, or the task-specific ASR/VAD file above
model_state.pt
training_args.json
training_recipe.json
native_export/             # optional; semantics declared by the adapter
```

`model_state.pt` contains canonical state for a fresh runtime. The training
recipe manifest records model family, recipe identity, phases, base model, and
native-export semantics. It is not a credential store: the exact mapping
returned by an adapter is checked before model or native-export state is
written or the destination is created, and the final mapping is checked again
before output. On reload, every public pretrained speech-model base checks the
deserialized mapping again before adapter or runtime `load_state_dict()` can
mutate the fresh model. This credential check does not make Python pickle a
safe untrusted-input format; load only trusted VoiceHub artifacts. Safe
metadata such as `token_count` remains serializable. If an active
topology/name-changing pass has no declared canonical export, the default
portable save fails before writing the artifact. `portable=False` is reserved
for Trainer's exact checkpoint path, which may store persistent transformed
state for same-plan resume.

Reload through the matching checkpoint-first factory:

```python
from voicehub import AutoModelForSpeechRecognition

reloaded = AutoModelForSpeechRecognition.from_pretrained(
    "runs/voicehub/final",
    device="auto",
    lazy_load=True,
)
```

The saved `config.json` identifies the model type and original base checkpoint.
Loading may still require access to that base checkpoint so VoiceHub can
reconstruct the correct graph before applying portable state.

### Exact-resume checkpoint

Periodic `checkpoint-N/` directories additionally contain:

```text
model_state.pt
optimizer.pt
scheduler.pt
trainer_state.json
training_args.json
rng_state.pth
training_runtime.pt
training_recipe.json       # when an adapter is active
optimization_manifest.json # when an explicit plan is active
scaler.pt                  # when a scaler is active
checkpoint_manifest.json
.complete
```

Checkpoint format 3 records required files, byte sizes, SHA-256 digests, global
step, adapter/recipe identity, optimizer names, training strategy, and the
exact-resume signature. Explicit optimization records include immutable pass
identity, kind, version, capabilities, configuration, and result metadata. A
checkpoint with a manifest but no `.complete` marker is ignored as incomplete.

```python
trainer.train(resume_from_checkpoint=True)  # newest valid checkpoint
trainer.train(
    resume_from_checkpoint="runs/voicehub/checkpoint-1000",
)
```

`get_last_checkpoint(folder)` returns the greatest valid numeric checkpoint or
`None`. `trainer.save_state()` alone, a portable model folder, a standalone
safetensors file, GGUF, or a native inference export is not an exact-resume
artifact.

Exact generic mid-epoch resume requires a stable, sized dataset/dataloader and
`dataloader_num_workers=0`. Changes to recipe, optimizer topology, strategy,
precision, batching, dataset/collator fingerprint, callbacks, or schedule can
invalidate the resume signature.

## Utility enums and functions

```python
IntervalStrategy.NO      # "no"
IntervalStrategy.STEPS   # "steps"
IntervalStrategy.EPOCH   # "epoch"

SchedulerType.LINEAR     # "linear"
SchedulerType.COSINE     # "cosine"
SchedulerType.CONSTANT   # "constant"
```

```python
set_seed(seed: int) -> None
get_last_checkpoint(folder: str | Path) -> str | None
```

`set_seed()` seeds Python, NumPy when installed, and Torch CPU/CUDA when
installed. For request-scoped inference reproducibility, prefer the `seed`
field on `TTSGenerationConfig`, which model integrations use without
permanently changing caller random state.

For end-to-end usage, continue with the [inference](../guides/inference.md),
[data preparation](../guides/data-preparation.md), and
[training](../guides/training.md) guides.
