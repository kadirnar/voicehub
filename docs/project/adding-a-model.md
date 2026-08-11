---
description: Add a lazy TTS, ASR, or VAD model with the shared VoiceHub lifecycle.
---

# Add a model

A model integration is one vertical slice. It includes the model package,
runtime wiring, lazy registration, training and optimization declarations,
provenance and license records, CPU-safe tests, and a generated model page.
Put model-specific code in its package and use shared VoiceHub contracts for
loading, outputs, optimization, training, and serialization.

The current Transformers reference is
[Add a model with modular transformers](https://huggingface.co/docs/transformers/main/en/modular_transformers).
It reduces contribution boilerplate while generating explicit standalone files
for each model. VoiceHub maps that mental model to a manifest-driven scaffold:
reuse stable speech components through composition, then keep the generated
configuration, runtime, and task wrapper explicit and locally readable.
VoiceHub does not generate a provider runtime through inheritance because a
speech integration may own checkpoint conversion, audio processing, codecs,
or streaming state. The legacy `add_new_model` guide remains an upstream
secondary path, not the representative route for this page.

<ol class="vh-process vh-process--eight" role="list" aria-label="Model integration workflow">
  <li><span class="vh-process__number" aria-hidden="true">01</span><strong>Create</strong><span class="vh-process__detail">Create the explicit package and optional owned architecture with the scaffold.</span></li>
  <li><span class="vh-process__number" aria-hidden="true">02</span><strong>Audit</strong><span class="vh-process__detail">Pin checkpoint and source revisions, copy legal text, and state the evidence boundary.</span></li>
  <li><span class="vh-process__number" aria-hidden="true">03</span><strong>Configure</strong><span class="vh-process__detail">Define a JSON-serializable config with a unique model type.</span></li>
  <li><span class="vh-process__number" aria-hidden="true">04</span><strong>Wrap</strong><span class="vh-process__detail">Implement the task base class and keep heavyweight imports inside the load hook.</span></li>
  <li><span class="vh-process__number" aria-hidden="true">05</span><strong>Register</strong><span class="vh-process__detail">Activate one lazy manifest or an explicit extension registrar.</span></li>
  <li><span class="vh-process__number" aria-hidden="true">06</span><strong>Support</strong><span class="vh-process__detail">Declare owned capabilities plus honest training and optimization boundaries.</span></li>
  <li><span class="vh-process__number" aria-hidden="true">07</span><strong>Test</strong><span class="vh-process__detail">Cover lazy import, factory loading, normalized output, persistence, and optimization.</span></li>
  <li><span class="vh-process__number" aria-hidden="true">08</span><strong>Document</strong><span class="vh-process__detail">Generate the provider page and navigation from registry metadata.</span></li>
</ol>

## 1. Create the package

Start with the repository scaffold instead of copying an existing provider:

```bash
python scripts/scaffold_model.py create \
  --model-type auroratts \
  --class-prefix AuroraTTS \
  --task tts \
  --checkpoint acme/aurora-base \
  --source-url https://github.com/acme/aurora-tts \
  --source-revision 0123456789abcdef0123456789abcdef01234567 \
  --license-id Apache-2.0 \
  --license-file /path/to/authoritative/upstream/LICENSE \
  --alias aurora-tts
```

The command validates identifiers, rejects mutable source revisions, copies
the supplied authoritative license text, and never overwrites an existing
artifact. It deliberately creates an **incomplete** integration: the runtime
raises `NotImplementedError`, the checkpoint revision remains a required
placeholder, and the generated CPU-safe test has a completion gate.

Use this file set as the contribution template:

```text
voicehub/models/auroratts/
  __init__.py
  configuration_auroratts.py
  modeling_auroratts.py
  registration.py
  runtime.py
  model-integration.json
  source/
    SOURCE.json
    THIRD_PARTY_LICENSE
tests/test_auroratts.py
docs/models/providers/auroratts.md  # generated after registration
```

Every step owns a predictable file boundary. Generic registry, optimization,
and documentation tests discover the new integration; they do not require a
provider entry.

| Step | Owned files or generated artifacts |
| --- | --- |
| 1. Create the package | `voicehub/models/<model_type>/`; optional `voicehub/architectures/<model_type>/` |
| 2. Record provenance and license | `source/SOURCE.json`, `source/THIRD_PARTY_LICENSE`, `scripts/documentation_references.py`, and any required `NOTICE` or `COPYING` file |
| 3. Define the config | `configuration_<model_type>.py` |
| 4. Implement the task wrapper | `modeling_<model_type>.py`, `runtime.py`, and model-local processing or conversion modules |
| 5. Register once | `model-integration.json` and `registration.py`; legacy central fragments only while migrating an existing declaration |
| 6. Declare training and optimization support | The manifest, optional `voicehub/architectures/<model_type>/` registration, and model-local training or optimization factories |
| 7. Test the contract | `tests/test_<model_type>.py`; registry-wide tests discover the activated integration without a provider list edit |
| 8. Generate the model page | `docs/models/providers/<model_type>.md` and the generated navigation block in `mkdocs.yml` |

If VoiceHub owns the executable graph, put it in
`voicehub/architectures/auroratts/`. Keep reviewed upstream code and its
license notice beside the integration. Put reusable codecs, vocoders, and
layers under `voicehub/components/`.

`__init__.py` should expose config and model classes lazily. `runtime.py` owns
checkpoint resolution and executable construction. Do not import PyTorch,
optional kernels, or provider SDKs while the registry is being inspected.

The native dependency policy discovers every immediate Python file in a model
package and every literal lazy component declared by an `ArchitectureSpec`.
Adding a normal facade, config, runtime, or training module therefore requires
no provider entry in a central dependency-policy list. Its internal VoiceHub
imports join the audited fixed-point closure automatically. Only an active
vendored module reached through a runtime-generated import needs a narrow
source-root declaration; dormant upstream frontends remain outside the native
boundary.

As work progresses, run the structural validator:

```bash
python scripts/scaffold_model.py check --model-type auroratts
```

It reports every missing artifact, mutable or placeholder revision, incomplete
runtime marker, contract-test marker, model-page section, registry entry,
training profile, alias, and navigation entry separately. Built-in registry
validation parses declarations without importing VoiceHub or the model. A
quoted name, comment, alias-only entry, mismatched task, or mismatched lazy
module path cannot satisfy the completion gate. A clean structural result does
not replace the focused runtime, checkpoint, optimization, documentation, or
package tests below.

When both built-in catalog files are absent, the checker treats the scaffold as
a separately distributed extension and validates its explicit registrar. If
either built-in catalog exists, it requires the complete built-in inference,
alias, and training declarations; a half-registered built-in cannot fall back
to extension status.

## 2. Record provenance and license

Pin every source and checkpoint to an immutable revision before copying or
adapting code. `SOURCE.json` must identify the model type, upstream URL,
revision, license, checkpoint revision, and the exact verified scope. Copy the
authoritative license text into `THIRD_PARTY_LICENSE`; do not paraphrase or
invent license terms.

Add the model's official GitHub repository and dedicated primary paper to
`MODEL_REFERENCES` in `scripts/documentation_references.py`. Use an empty paper
tuple when upstream has not published one; do not substitute an unrelated
architecture paper.

```json
{
  "model_type": "auroratts",
  "upstream": "https://github.com/acme/aurora-tts",
  "revision": "0123456789abcdef0123456789abcdef01234567",
  "license": "Apache-2.0",
  "checkpoint": {
    "repository": "acme/aurora-base",
    "revision": "89abcdef0123456789abcdef0123456789abcdef",
    "license": "Apache-2.0"
  },
  "verified_scope": {
    "inference": [],
    "training": [],
    "limitations": [
      "Replace this entry with the exact unverified or hardware-limited boundary."
    ]
  }
}
```

Keep an integration unregistered until these fields and the bundled license
text have been reviewed. Record inaccessible checkpoints or hardware paths as
unverified; never turn them into a passing claim.

## 3. Define the config

Configs contain stable JSON data, not loaded modules, tensors, devices,
callables, or secrets.

```python
from voicehub import VoiceHubConfig


class AuroraTTSConfig(VoiceHubConfig):
    model_type = "auroratts"

    def __init__(self, *, sample_rate=24_000, **kwargs):
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        super().__init__(sample_rate=sample_rate, **kwargs)
```

## 4. Implement the task wrapper

Choose one base class:

| Task | Base class | Implement | Public method |
| --- | --- | --- | --- |
| TTS | `PreTrainedTTSModel` | `_load_pretrained_model`, `_generate` | `generate` |
| ASR | `PreTrainedASRModel` | `_load_pretrained_model`, `_transcribe` | `transcribe` |
| VAD | `PreTrainedVADModel` | `_load_pretrained_model`, `_detect` | `detect` |

This minimal TTS wrapper gets lazy loading, save/reload, inference strategies,
training hooks, and the generic optimization-pass API from the base class:

```python
from voicehub import PreTrainedTTSModel, TTSOutput

from .configuration_auroratts import AuroraTTSConfig


class AuroraTTSForTextToSpeech(PreTrainedTTSModel):
    config_class = AuroraTTSConfig
    default_model_name_or_path = "acme/aurora-base"

    def _load_pretrained_model(self):
        from .runtime import load_runtime

        self.model = load_runtime(self.config.name_or_path, device=self.device)

    def _generate(self, text, **kwargs):
        audio = self.model.synthesize(text, **kwargs)
        return TTSOutput(audio=audio, sample_rate=self.config.sample_rate)
```

Return `TTSOutput`, `ASROutput`, or `VADOutput`; do not return a
provider-specific object.

## 5. Register once

The auto factories use the config's `model_type` and store lazy import paths.
No central auto-factory mapping needs to change. Registration also records the
model class's `processor_class` as lazy `ModelSpec` metadata. Models that retain
the shared task-default processor need no extra declaration; an extension that
sets a custom importable processor class keeps that class without forcing model
wrapper import during `AutoProcessor` discovery.

```python
from voicehub import AutoModelForTextToSpeech

from .configuration_auroratts import AuroraTTSConfig
from .modeling_auroratts import AuroraTTSForTextToSpeech


def register_auroratts():
    return AutoModelForTextToSpeech.register(
        AuroraTTSConfig,
        AuroraTTSForTextToSpeech,
        default_model_path="acme/aurora-base",
        aliases=("aurora-tts",),
        components=(),
    )
```

Use `AutoModelForSpeechRecognition.register(...)` or
`AutoModelForVoiceActivityDetection.register(...)` for the other tasks.
Separately distributed extensions call their registration function when the
extension is imported. The generated registrar also provides a dependency-light
development smoke path.

For a standard inference-only built-in, the package-local
`model-integration.json` is the registry declaration. Complete and verify the
runtime, checkpoint revision, legal records, contract tests, capabilities,
components, and limitations first. Then change only its activation field:

```json
{
  "builtin": true,
  "capabilities": ["text-to-speech"],
  "components": [],
  "training": {
    "family": "acoustic-regression",
    "support": "inference-only"
  }
}
```

The model registry derives the lazy config/model paths and aliases from that
activated manifest. The training registry derives the explicit inference-only
profile from the same source. It imports neither the model package nor PyTorch,
and it requires no edit to `voicehub/models/registry.py` or
`voicehub/training/specs.py`. An inactive or invalid work-in-progress manifest
is never registered. A central declaration and an activated manifest for the
same model fail as a duplicate.

Activation is also a strict JSON trust boundary. An activated
`model-integration.json` and its required `source/SOURCE.json` reject duplicate
keys, `NaN`, infinities, and numeric overflow before registry or training-spec
construction. The scaffold checker and read-only catalog report the same
source-aware diagnostics without echoing a discarded value. An inactive
work-in-progress manifest remains undiscovered, including while its JSON is
temporarily incomplete or ambiguous.

The read-only legacy renderer remains available when auditing or migrating a
central declaration:

```bash
python scripts/scaffold_model.py catalog --model-type auroratts
```

Do not paste its output for a manifest-discovered model. Do not add a provider
branch to an auto factory, component registry, or optimization resolver. The
scaffold checker reports missing activation metadata, unsafe training claims,
and duplicate legacy declarations separately.

A task may have at most one no-argument default. Set `default_for_task=True`
only when the default checkpoint/provider policy has explicit project approval;
the model registry rejects ambiguous defaults. The shared auto factories derive
this choice from `ModelSpec` metadata and contain no provider-name fallback.
The compatibility `AutoInferenceModel` uses the same TTS declaration, so a
default policy must never be duplicated in that legacy surface.

If the backend reuses a registered codec, vocoder, watermark, or neural block,
list its canonical names in `components=(...)`. The resulting `ModelSpec` owns
that relationship and updates the read-only `MODEL_COMPONENTS` view; do not add
the model name to a second component map.

Users may load the result through the task factory or the task-aware factory:

```python
from voicehub import AutoModel

model = AutoModel.from_pretrained(
    "acme/aurora-base",
    model_type="auroratts",
)
```

## 6. Declare training and optimization support

Add an `ArchitectureSpec` when VoiceHub owns and can verify the runtime graph.
Its capabilities declare the model-independent protocols implemented by the
graph. Public optimization code must use those protocols, not the model type or
provider name.

Every built-in model needs one honest training profile. The activated manifest
above provides an inference-only boundary without a central edit. Use it only
when the integrated artifact has no verified differentiable path.

When a differentiable path is verified, keep the manifest inactive and replace
the legacy renderer's inference-only training fragment with an explicit
`ModelTrainingSpec` in `voicehub/training/specs.py`; use its `ModelSpec` and
alias fragments for inference discovery:

```python
_profile(
    "auroratts",
    TrainingFamily.ACOUSTIC,
    task=SpeechTask.TEXT_TO_SPEECH,
    support=TrainingSupport.NATIVE,
)
```

The checker requires either one activated inference-only manifest or a
task-matching `_profile(...)`/`ModelTrainingSpec(...)` declaration, never both.
See the
[training architecture](../concepts/trainer.md).
If the published objective needs a specialized adapter, keep that callable
beside the model or owning architecture and declare it with
`adapter_factory="module:callable"` on the same profile. Listing the registry
remains framework-lazy, and adding the model does not require a central adapter
map edit.
If a codec-LM needs source-native record construction, expose a framework-lazy
`build_training_dataset(model, records, **kwargs)` callable beside its training
implementation and declare it with
`dataset_factory="module:build_training_dataset"`. For a tokenizer stored at a
nonstandard wrapper location, extend `tokenizer_paths` in the same profile.
Do not add the model name to a shared adapter.
If the model has a source-verified special training profile, declare its lazy
`optimization_profile_factory="module:callable"` on that specification. The
callable must return the shared profile behavior; do not add the model name to
the public resolver or infer that another model's recipe applies merely because
both use the same data architecture.
For an exact TTS or ASR source-data contract, expose a zero-argument,
framework-free callable that returns `TTSDatasetSpec` or `ASRDatasetSpec` and
declare it with
`dataset_spec_factory="module:callable"` on the training profile. Keep the
callable beside the model or owning architecture. Declare manifest spellings
with `field_aliases`; an identity pair on `TTSDatasetSpec` preserves a
model-canonical field that shared aliases would otherwise rewrite. ASR
contracts may also declare a lazy `record_normalizer`. Cover normalized output
and failure behavior. Do not add the model to a shared dataset-spec map or add
a model-name conditional to `TTSDataset` or `ASRDataset`.

The wrapper must retain the shared `apply_optimization_plan`, validation,
manifest reporting, and `restore_optimization_plan` lifecycle. Add the model to
registry-wide optimization coverage and test a supported application and
deterministic restoration. An unsupported device or missing protocol must fail
before mutation with an actionable compatibility error; a silent skip is not
support.

If an external vLLM or SGLang speech pipeline is verified, register an
`LLMBackendSupport` record that declares its concrete transport, reference
format, default task types, and verified `speech_string_options`. That record
drives recognized wrapper inputs, generation defaults, and the direct HTTP
payload; do not edit a second option allowlist. Test the serialized record,
unknown-option failure, and exact request payload. Do not add the model name to
the shared HTTP client.

## 7. Test the contract

At minimum, test that:

- importing registration does not load a checkpoint or optional GPU package;
- the correct auto factory creates the wrapper and wrong-task factories fail;
- invalid config and inputs fail before expensive loading;
- inference returns the task's normalized output;
- local config and model state save and reload;
- one registered optimization pass applies, reports, and restores
  deterministically;
- unsupported optimization hardware or capabilities fail before mutation;
- training metadata matches the implemented graph.

Representative real-checkpoint execution belongs in dated evidence when the
checkpoint is accessible and practical. Otherwise add an explicit unverified
or hardware-limited record to the provider page and evidence inventory.

## 8. Generate the model page

Do not maintain a second hand-written catalog. After the registry, training,
license, and capability metadata are complete, add the model's explicit entry
to `scripts/model_documentation.py`. Record either its verified Hugging Face
repository ID or the exact reason no ID applies. The inference profile must use
the VoiceHub public wrapper, include model-specific required inputs and
controls, and contain no package-install command or copied upstream snippet.
Generation fails when either record is missing.

Then generate the provider page, focused Hub notebook when applicable, and the
generated navigation entry in `mkdocs.yml`:

```bash
python scripts/generate_model_notebooks.py
python scripts/generate_model_pages.py
python scripts/generate_model_notebooks.py --check
python scripts/generate_model_pages.py --check
```

The generated page must contain the overview, a copyable quickstart, supported
tasks and capabilities, checkpoint provenance and license, optimization and
training support, and the public API. Put shared workflows in Guides instead
of duplicating them across model pages.

## Completion evidence

| Contract | Required evidence |
| --- | --- |
| Package and lazy registry | Config, wrapper, runtime, registration, and an import test that does not load a checkpoint or heavy backend |
| Inputs, outputs, and persistence | CPU-safe construction, validation, representative execution, normalized output, save, and reload tests |
| Provenance and license | Immutable source/checkpoint revisions, authoritative license metadata, `SOURCE.json`, and bundled legal text |
| Training | A truthful `ModelTrainingSpec`, dataset/input boundary, failure behavior, and at least one CPU-safe step when supported |
| Optimization | Registry-wide capability coverage plus apply, validate, manifest, restore, and unsupported-hardware tests |
| Documentation | Explicit HF-ID status and model-specific VoiceHub inference profile; generated provider page, notebook when applicable, navigation entry, limitations, checkpoint status, provenance, and license |
| Real checkpoint | Reproducible dated evidence, or an explicit unverified or hardware-limited record |

Run:

```bash
python -m pytest -q \
  tests/test_your_model.py \
  tests/test_registry.py \
  tests/test_native_optimization.py \
  tests/test_documentation_site.py
python scripts/generate_model_pages.py --check
python -m pytest -q
pre-commit run --all-files
mkdocs build --strict --clean
python scripts/check_distribution.py
```

For ASR- and VAD-specific output examples, see
[Add an ASR or VAD provider](adding-speech-provider.md). For optimization
extensions, see [Add an optimization](adding-an-optimization.md).
