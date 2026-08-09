#!/usr/bin/env python3
"""Generate one compact, contract-aligned guide for every registered model."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(SCRIPTS_ROOT))

from documentation_references import MODEL_REFERENCES, Reference  # noqa: E402
from generate_model_notebooks import (  # noqa: E402
    TASK_LABELS,
    TASK_ORDER,
    TTS_GENERATION_OPTIONS,
    checkpoint_documentation,
)

from voicehub import list_model_specs  # noqa: E402
from voicehub.models.language_support import model_language_support  # noqa: E402

MODEL_PAGE_DIR = REPOSITORY_ROOT / "docs" / "models" / "providers"
SITE_CONFIG_PATH = REPOSITORY_ROOT / "mkdocs.yml"
GENERATOR_PATH = "scripts/generate_model_pages.py"
COLAB_ROOT = ("https://colab.research.google.com/github/kadirnar/voicehub/"
              "blob/main/notebooks/models")
NAVIGATION_START = "          # BEGIN GENERATED MODEL GUIDE NAVIGATION"
NAVIGATION_END = "          # END GENERATED MODEL GUIDE NAVIGATION"
AUTO_CLASSES_NAVIGATION_ENTRY = "          - Auto Classes: models/providers/index.md"
MODEL_GUIDES_NAVIGATION_SECTION = "      - Models:\n"
FULL_API_NAVIGATION_ENTRY = "      - Full API reference: reference/api.md"
CONTRIBUTE_NAVIGATION_SECTION = "          - Contribute:\n"
MODEL_PAGE_SECTIONS = (
    "Usage",
    "Overview",
    "Paper and GitHub",
    "Configuration",
    "Processing",
    "Inference",
    "Training and optimization",
    "Checkpoints, provenance, license, and limitations",
    "Public API",
)


def _value(value) -> str:
    if value is None:
        return "Not declared"
    if hasattr(value, "value"):
        return str(value.value)
    return str(value)


def _cell(value) -> str:
    """Escape one compact Markdown table cell."""
    text = _value(value).replace("|", "\\|").replace("\n", " ").strip()
    return text or "—"


def _code_list(values) -> str:
    items = tuple(values)
    return ", ".join(f"`{_cell(item)}`" for item in items) if items else "—"


def _language_summary(spec) -> str:
    support = model_language_support(spec)
    if support.kind == "enumerated":
        if len(support.codes) == 1:
            return f"`{support.codes[0]}`"
        return f"{len(support.codes)} enumerated languages"
    if support.kind == "not-text-conditioned":
        return "Not text-language conditioned"
    return "Checkpoint-defined; not exhaustively enumerated"


def _language_details(spec) -> str:
    support = model_language_support(spec)
    if support.kind == "enumerated":
        codes = _code_list(support.codes)
        note = f"\n\n{support.note}" if support.note else ""
        return f'''<details class="vh-language-support" markdown>
<summary>{len(support.codes)} documented language{'s' if len(support.codes) != 1 else ''}</summary>

{codes}{note}

</details>'''
    return support.note or "Language support is not declared."


def _checkpoint(spec) -> tuple[str, str]:
    documentation = checkpoint_documentation(spec)
    if documentation.identifier and documentation.url:
        rendered = f"[`{documentation.identifier}`]({documentation.url})"
    elif documentation.identifier:
        rendered = f"`{documentation.identifier}`"
    else:
        rendered = "No default; pass a compatible Hub ID or local directory."
    return documentation.example, rendered


def _install_command(spec) -> str:
    extra = f"[{spec.install_extra}]" if spec.install_extra else ""
    return (
        f'python -m pip install "voicehub{extra} @ '
        'git+https://github.com/kadirnar/voicehub.git@main"')


def _inference_code(spec) -> str:
    checkpoint, _ = _checkpoint(spec)
    if spec.task.value == "text-to-speech":
        options = TTS_GENERATION_OPTIONS.get(spec.model_type, ())
        generation_kwargs = "{}"
        if options:
            rendered = "\n".join(f"    {line}" for line in options)
            generation_kwargs = f"{{\n{rendered}\n}}"
        reference_setup = "\n"
        if any("REFERENCE_" in line for line in options):
            reference_setup = '''
REFERENCE_AUDIO = Path("reference.wav")
REFERENCE_TEXT = "This transcript must exactly match the authorized reference audio."

'''
        return f'''from pathlib import Path
{reference_setup}from voicehub import AutoModelForTextToSpeech, TTSGenerationConfig

model = AutoModelForTextToSpeech.from_pretrained(
    {checkpoint!r},
    model_type={spec.model_type!r},
    device="cuda",
    lazy_load=True,
)
generation_kwargs = {generation_kwargs}
output = model.generate(
    "VoiceHub keeps model integrations consistent and easy to extend.",
    generation_config=TTSGenerationConfig(
        seed=42,
        output_file=Path("output.wav"),
    ),
    **generation_kwargs,
)
print(output.file_path, output.sample_rate)'''
    if spec.task.value == "automatic-speech-recognition":
        return f'''from voicehub import AutoModelForSpeechRecognition

model = AutoModelForSpeechRecognition.from_pretrained(
    {checkpoint!r},
    model_type={spec.model_type!r},
    device="cuda",
    lazy_load=True,
)
output = model.transcribe("speech.wav")
print(output.text)
for segment in output.segments:
    print(segment.start, segment.end, segment.text)'''
    return f'''from voicehub import AutoModelForVoiceActivityDetection

model = AutoModelForVoiceActivityDetection.from_pretrained(
    {checkpoint!r},
    model_type={spec.model_type!r},
    device="cpu",
    lazy_load=True,
)
output = model.detect("speech.wav", threshold=0.5)
for segment in output.segments:
    print(segment.start, segment.end, segment.score)'''


def _inference_notes(spec) -> str:
    notes = ["Install from source, then choose a compatible checkpoint."]
    if spec.task.value == "text-to-speech":
        if spec.model_type in TTS_GENERATION_OPTIONS:
            notes.append("Provide an authorized `reference.wav` and its exact transcript when requested.")
        else:
            notes.append("Set the text and generation options, then inspect the returned audio.")
    elif spec.task.value == "automatic-speech-recognition":
        notes.append("Place a supported recording at `speech.wav` and inspect the transcript.")
    else:
        notes.append("Place a recording at `speech.wav`; tune the threshold on labeled audio.")
    checkpoint_note = checkpoint_documentation(spec).note
    rendered = " ".join(notes)
    if checkpoint_note:
        rendered += f"\n\nCheckpoint note: {checkpoint_note}"
    return rendered


def _factory_name(spec) -> str:
    return {
        "text-to-speech": "AutoModelForTextToSpeech",
        "automatic-speech-recognition": "AutoModelForSpeechRecognition",
        "voice-activity-detection": "AutoModelForVoiceActivityDetection",
    }[spec.task.value]


def _output_name(spec) -> str:
    return {
        "text-to-speech": "TTSOutput",
        "automatic-speech-recognition": "ASROutput",
        "voice-activity-detection": "VADOutput",
    }[spec.task.value]


def _module_source_roots(module: str) -> tuple[Path, ...]:
    """Return package roots for a lazy module path without importing it."""
    module_path = REPOSITORY_ROOT / Path(*module.split("."))
    if module_path.is_dir():
        return (module_path, )
    source_path = module_path.with_suffix(".py")
    if source_path.is_file():
        return (source_path.parent, )
    return ()


def _source_provenance(spec) -> str:
    """Describe the closest bundled source record without inventing provenance."""
    roots = [
        REPOSITORY_ROOT / "voicehub" / "models" / spec.model_type,
        *_module_source_roots(spec.module),
    ]
    if spec.is_voicehub_native:
        architecture = spec.native_architecture
        roots.extend(
            root for reference in architecture.component_references.values()
            for root in _module_source_roots(reference.module))
    roots.append(REPOSITORY_ROOT / "voicehub" / "architectures" / (spec.architecture or ""))

    candidates = []
    seen = set()
    for root in roots:
        for candidate in (root / "source" / "SOURCE.json", root / "SOURCE.json"):
            if candidate not in seen:
                candidates.append(candidate)
                seen.add(candidate)
    for candidate in candidates:
        if candidate.is_file():
            return f"`{candidate.relative_to(REPOSITORY_ROOT).as_posix()}`"
    return "No integration-specific bundled `SOURCE.json` is declared for this registry entry."


def _module_source_path(module: str) -> Path:
    """Resolve a declared lazy module to its stable repository source."""
    module_path = REPOSITORY_ROOT / Path(*module.split("."))
    source_path = module_path.with_suffix(".py")
    if source_path.is_file():
        return source_path.relative_to(REPOSITORY_ROOT)
    package_source = module_path / "__init__.py"
    if package_source.is_file():
        return package_source.relative_to(REPOSITORY_ROOT)
    raise RuntimeError(f"Declared module {module!r} has no repository source file")


def _module_source_link(module: str, label: str) -> str:
    source_path = _module_source_path(module).as_posix()
    return f"[{label}](https://github.com/kadirnar/voicehub/blob/main/{source_path})"


def _reference_links(references: tuple[Reference, ...]) -> str:
    return "; ".join(f"[{reference.title}]({reference.url})" for reference in references)


def _research_section(spec) -> str:
    references = MODEL_REFERENCES[spec.model_type]
    papers = _reference_links(references.papers) if references.papers else (
        "No dedicated upstream research paper is published for this integration.")
    github = _reference_links((references.github, ))
    implementation = _module_source_link(spec.module, "VoiceHub model implementation")
    return f'''- **Paper:** {papers}
- **Upstream GitHub:** {github}
- **VoiceHub source:** {implementation}'''


def _example_device(spec) -> str:
    return "cpu" if spec.task.value == "voice-activity-detection" else "cuda"


def _processor_code(spec) -> str:
    checkpoint, _ = _checkpoint(spec)
    return f'''from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    {checkpoint!r},
    model_type={spec.model_type!r},
)
print(type(processor).__name__)'''


def _variant_dependencies(variant) -> str:
    parts = []
    if variant.at_most_one_of:
        parts.append("at most one: " + "; ".join(" / ".join(group) for group in variant.at_most_one_of))
    if variant.forbidden_fields:
        parts.append("forbidden: " + ", ".join(variant.forbidden_fields))
    if variant.requires:
        parts.extend(f"{trigger} requires {', '.join(required)}" for trigger, required in variant.requires)
    if variant.requires_one_of:
        parts.extend(
            f"{trigger} requires one of {', '.join(required)}"
            for trigger, required in variant.requires_one_of)
    return "; ".join(parts) or "—"


def _dataset_section(spec) -> str:
    training = spec.training
    if spec.task.value in (
            "text-to-speech",
            "automatic-speech-recognition",
    ):
        dataset = training.dataset_spec
        rows = []
        for variant in dataset.variants:
            one_of = "; ".join(" / ".join(group) for group in variant.one_of)
            rows.append(
                f"| `{_cell(variant.name)}` | {_code_list(variant.required_fields)} | "
                f"{_cell(one_of)} | {'Prepared' if variant.preprocessed else 'Source'} | "
                f"{_cell(_variant_dependencies(variant))} |")
        getter = "get_tts_dataset_spec" if spec.task.value == "text-to-speech" else "get_asr_dataset_spec"
        guide = (
            "../../guides/data-preparation.md"
            if spec.task.value == "text-to-speech" else "../../guides/speech-data.md")
        sample_rate = f"{dataset.sample_rate:,} Hz" if dataset.sample_rate else "Model/checkpoint specific"
        return f'''| Property | Value |
| --- | --- |
| Readiness | `{_cell(dataset.readiness)}` |
| Data architecture | `{_cell(dataset.architecture)}` |
| Sample rate | {sample_rate} |
| Contract getter | `{getter}({spec.model_type!r})` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
{chr(10).join(rows)}

{_cell(dataset.description)} See the [data workflow]({guide}).'''

    required = tuple(dict.fromkeys(name for phase in training.phases for name in phase.required_inputs))
    fields = _code_list(required) if required else "—"
    boundary = (
        "No verified training dataset contract"
        if training.support.value == "inference-only" else "Clip-, frame-, or segment-level labels")
    return f'''| Property | Value |
| --- | --- |
| Label boundary | {boundary} |
| Required training inputs | {fields} |

Use authorized audio and preserve annotation provenance. See the
[ASR and VAD data workflow](../../guides/speech-data.md).'''


def _phase_rows(training) -> str:
    rows = []
    for phase in training.phases:
        components = phase.component_paths or ((phase.forward_component, ) if phase.forward_component else ())
        rows.append(
            f"| `{_cell(phase.name)}` | {_cell(phase.kind)} | "
            f"{_code_list(components)} | {_code_list(phase.required_inputs)} | "
            f"{_code_list(phase.loss_keys)} |")
    return "\n".join(rows)


def _training_section(spec) -> str:
    training = spec.training
    training_checkpoint = (
        training.training_default_model_name_or_path or spec.default_model_path or
        "owner/model-or-local-directory")
    summary = f'''| Property | Value |
| --- | --- |
| Support | `{training.support.value}` |
| Family | `{training.family_name}` |
| Recipe | `{training.recipe_kind.value}` |
| Default phase | `{training.default_phase}` |
| Training checkpoint | `{training_checkpoint}` |
| Native training graph | `{'yes' if training.native_training else 'no'}` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
{_phase_rows(training)}'''
    if not training.support.is_trainable:
        return f'''{summary}

This integration is **inference-only**. Choose a verified model from the
[training matrix](../training-support.md).'''

    qualifier = {
        "native": "The integration accepts its declared source or prepared contract directly.",
        "preprocessed": "Prepare the exact tensors listed in the data contract before this step.",
        "custom": "This profile uses model-specific phases; inspect and honor each phase boundary.",
    }[training.support.value]
    return f'''{summary}

{qualifier} Call `model.validate_training_support()` first, then follow the
[training workflow](../../guides/training.md).'''


def render_page(spec) -> str:
    """Render one deterministic provider guide."""
    _, checkpoint = _checkpoint(spec)
    checkpoint_metadata = checkpoint_documentation(spec)
    license_spec = spec.license
    if license_spec is None:
        license_text = (
            "No VoiceHub-specific license override is registered. Verify the "
            "checkpoint and upstream source terms before use.")
        license_value = "Checkpoint-specific"
    else:
        commercial = {
            True: "allowed by the registered terms",
            False: "not allowed",
            None: "review required",
        }[license_spec.commercial_use]
        license_value = f"[{license_spec.license_id}]({license_spec.upstream})"
        license_text = f"{license_spec.notice} Commercial use: **{commercial}**."
    notebook = ""
    if checkpoint_metadata.is_hugging_face:
        notebook = (
            f" [Open the `{spec.model_type}` Colab notebook]"
            f"({COLAB_ROOT}/{spec.model_type}.ipynb).")
    architecture = spec.architecture or "provider-owned"
    components = _code_list(spec.components)
    factory = _factory_name(spec)
    output = _output_name(spec)
    source_provenance = _source_provenance(spec)
    config_source = _module_source_link(
        spec.config_module,
        f"View `{spec.config_class}` source",
    )
    model_source = _module_source_link(
        spec.module,
        f"View `{spec.class_name}` source",
    )
    dependency_extra = (f"`voicehub[{spec.install_extra}]`" if spec.install_extra else "Core package")
    checkpoint_note = checkpoint_metadata.note or (
        "No integration-specific checkpoint limitation is registered. Verify the selected "
        "checkpoint revision and its documented runtime requirements.")
    return f'''---
description: Public API, checkpoint, training, and optimization guide for the {spec.model_type} integration.
---

# {spec.display_name} {{.vh-model-title}}

## Usage

```bash
{_install_command(spec)}
```

{_inference_notes(spec)}

```python
{_inference_code(spec)}
```

Use authorized recordings. Verify hardware needs and pin a revision in production.

## Overview

`{spec.model_type}` is a VoiceHub **{TASK_LABELS[spec.task.value].lower()}**
integration. This page is generated from its registry contract.{notebook}

| Property | Value |
| --- | --- |
| Task | {TASK_LABELS[spec.task.value]} |
| Architecture | `{architecture}` |
| Runtime | `{'VoiceHub-native' if spec.is_voicehub_native else 'provider adapter'}` |
| Languages | {_language_summary(spec)} |
| Capabilities | {_code_list(spec.capabilities)} |
| Reusable components | {components} |
| Normalized output | `{output}` |

### Language support

{_language_details(spec)}

## Paper and GitHub

{_research_section(spec)}

## Configuration

Load configuration without constructing the model:

```python
from voicehub import AutoConfig

config = AutoConfig.for_model({spec.model_type!r})
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `{spec.model_type}` |
| Configuration class | `{spec.config_class}` |
| Architecture class | `{spec.class_name}` |

## Processing

Create the registered processor without allocating model weights:

```python
{_processor_code(spec)}
```

## Inference

The Usage example returns `{output}` through `{factory}`.

### Input and output contract

{_dataset_section(spec)}

## Training and optimization

Use `available_optimization_passes()` to discover reversible public passes.
Unsupported runtime or hardware fails closed before mutation.

### Training contract

{_training_section(spec)}

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | {checkpoint} |
| Checkpoint status | {_cell(checkpoint_metadata.status)} |
| Optional dependency extra | {dependency_extra} |
| Hardware and runtime | Usage selects `{_example_device(spec)}`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `{spec.module}.{spec.class_name}` |
| Configuration | `{spec.config_module}.{spec.config_class}` |
| Source provenance | {source_provenance} |
| License | {license_value} |

{license_text}

Confirm the checkpoint revision, access terms, provenance, and license.

### Limitations

- {checkpoint_note}
- Validate memory, precision, and optional dependencies on the target system.
- Public optimizations fail closed when the runtime or hardware cannot satisfy
  their validation contract; an unavailable pass is not reported as applied.
- Contract tests do not replace the linked released-checkpoint evidence.

## Public API

Use the stable configuration, processor, and task-model facades below.

### `{spec.config_class}`

{config_source}

```text
{spec.config_class}(**config_kwargs)
```

### `{spec.class_name}`

{model_source}

```text
{factory}.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type={spec.model_type!r},
    config=None,
    **model_kwargs,
)
```

```python
from voicehub import get_model_spec

spec = get_model_spec({spec.model_type!r})
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec({spec.model_type!r})` |
| Load and run | `{factory}` |
| Configure | `{spec.config_class}` |
| Process | `AutoProcessor` |
| Model implementation | `{spec.class_name}` |
| Normalized output | `{output}` |
| Training contract | `get_training_spec({spec.model_type!r})` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

See [all model guides](index.md), [inference](../../guides/index.md), and the
[training matrix](../training-support.md).
'''


def render_index(specs) -> str:
    """Render the generated provider-guide index."""
    lines = [
        "---",
        "description: Auto configuration, processing, task dispatch, and discovery for every registered VoiceHub model.",
        "---",
        "",
        "# Auto Classes",
        "",
        "Auto classes choose a registered configuration, processor, or task-specific model",
        "from a checkpoint and its canonical `model_type`. They keep discovery lazy, so",
        "listing the registry or reading this page does not import a model runtime.",
        "",
        "## Choose an Auto class",
        "",
        "| Workflow | Public class | Normalized output |",
        "| --- | --- | --- |",
        "| Text to speech | `AutoModelForTextToSpeech` | `TTSOutput` |",
        "| Automatic speech recognition | `AutoModelForSpeechRecognition` | `ASROutput` |",
        "| Voice activity detection | `AutoModelForVoiceActivityDetection` | `VADOutput` |",
        "",
        "Use `AutoModel` only when the checkpoint configuration already identifies its task.",
        "Prefer a task-specific class when the expected output contract is known.",
        "",
        "## AutoConfig",
        "",
        "`AutoConfig` resolves the registered configuration class without constructing the",
        "model. A local `config.json` may provide `model_type`; raw checkpoint files require",
        "the explicit canonical identifier.",
        "",
        "```python",
        "from voicehub import AutoConfig",
        "",
        "config = AutoConfig.from_pretrained(",
        "    \"microsoft/speecht5_tts\",",
        "    model_type=\"speecht5\",",
        ")",
        "print(config.model_type)",
        "```",
        "",
        "## AutoProcessor",
        "",
        "`AutoProcessor` builds the processor paired with the registered model configuration",
        "without allocating the model itself.",
        "",
        "```python",
        "from voicehub import AutoProcessor",
        "",
        "processor = AutoProcessor.from_pretrained(",
        "    \"microsoft/speecht5_tts\",",
        "    model_type=\"speecht5\",",
        ")",
        "```",
        "",
        "## Task-specific AutoModel classes",
        "",
        "The task-specific factories preserve one output and failure contract across every",
        "registered TTS, ASR, or VAD integration.",
        "",
        "```python",
        "from voicehub import (",
        "    AutoModelForSpeechRecognition,",
        "    AutoModelForTextToSpeech,",
        "    AutoModelForVoiceActivityDetection,",
        ")",
        "",
        "for auto_class in (",
        "    AutoModelForTextToSpeech,",
        "    AutoModelForSpeechRecognition,",
        "    AutoModelForVoiceActivityDetection,",
        "):",
        "    for spec in auto_class.available_models():",
        "        print(spec.display_name, spec.model_type)",
        "",
        "model = AutoModelForTextToSpeech.from_pretrained(",
        "    \"microsoft/speecht5_tts\",",
        "    model_type=\"speecht5\",",
        "    device=\"cpu\",",
        "    lazy_load=True,",
        ")",
        "```",
        "",
        "Extensions register through the same auto-class contract. Follow the",
        "[model contribution workflow](../../project/adding-a-model.md) so configuration,",
        "runtime, provenance, tests, optimization support, and documentation stay complete.",
        "",
        "## Registered models",
        "",
        "Every entry below has one generated guide with the same nine required sections.",
        "Hub-backed models also link to a dedicated Colab notebook.",
        "",
        f"Generated by `{GENERATOR_PATH}` from lazy registry metadata.",
        "",
    ]
    for task in TASK_ORDER:
        task_specs = sorted(
            (spec for spec in specs if spec.task.value == task),
            key=lambda spec: (spec.display_name.casefold(), spec.model_type),
        )
        lines.extend((
            f"### {TASK_LABELS[task]}",
            "",
            "<div class=\"vh-model-catalog\" markdown>",
            "",
            "| Model | Languages | Default checkpoint | Training | Notebook |",
            "| --- | --- | --- | --- | --- |",
        ))
        for spec in task_specs:
            _, checkpoint = _checkpoint(spec)
            checkpoint_metadata = checkpoint_documentation(spec)
            notebook = (
                f"[Colab]({COLAB_ROOT}/{spec.model_type}.ipynb)"
                if checkpoint_metadata.is_hugging_face else "—")
            lines.append(
                f"| [`{spec.display_name}`]({spec.model_type}.md) | "
                f"{_language_summary(spec)} | {checkpoint} | "
                f"`{spec.training.support.value}` | {notebook} |")
        lines.extend(("", "</div>", ""))
    return "\n".join(lines)


def render_navigation(specs) -> str:
    """Render the task-grouped model links shown in the site sidebar."""
    lines = [NAVIGATION_START]
    for task in TASK_ORDER:
        task_specs = sorted(
            (spec for spec in specs if spec.task.value == task),
            key=lambda spec: (spec.display_name.casefold(), spec.model_type),
        )
        lines.append(f"          - {TASK_LABELS[task]}:")
        lines.extend(
            f'              - "{spec.display_name}": models/providers/{spec.model_type}.md'
            for spec in task_specs)
    lines.append(NAVIGATION_END)
    return "\n".join(lines)


def render_site_config(specs) -> str:
    """Render model guides in the public Models navigation hierarchy."""
    source = SITE_CONFIG_PATH.read_text(encoding="utf-8")
    if source.count(NAVIGATION_START) != 1 or source.count(NAVIGATION_END) != 1:
        raise RuntimeError("mkdocs.yml must contain exactly one generated model navigation block")
    prefix, remainder = source.split(NAVIGATION_START, 1)
    _, suffix = remainder.split(NAVIGATION_END, 1)
    if prefix.endswith(MODEL_GUIDES_NAVIGATION_SECTION):
        prefix = prefix[:-len(MODEL_GUIDES_NAVIGATION_SECTION)]
    suffix = suffix.removeprefix("\n")
    source = f"{prefix}{suffix}"

    base_prefix, base_remainder = source.split("  - Base classes:\n", 1)
    base_navigation, following_navigation = base_remainder.split("  - Inference:\n", 1)
    base_navigation = base_navigation.replace(f"{AUTO_CLASSES_NAVIGATION_ENTRY}\n", "", 1)
    base_navigation = base_navigation.replace("      - Models:\n\n", "      - Models:\n", 1)
    if base_navigation.count(CONTRIBUTE_NAVIGATION_SECTION) != 1:
        raise RuntimeError("mkdocs.yml must contain exactly one model contribution section")
    base_navigation = base_navigation.replace(
        CONTRIBUTE_NAVIGATION_SECTION,
        f"{render_navigation(specs)}\n{CONTRIBUTE_NAVIGATION_SECTION}",
        1,
    )
    source = (f"{base_prefix}  - Base classes:\n{base_navigation}"
              f"  - Inference:\n{following_navigation}")

    api_prefix, api_remainder = source.split("  - API:\n", 1)
    api_navigation, plugins = api_remainder.split("\nplugins:", 1)
    if AUTO_CLASSES_NAVIGATION_ENTRY not in api_navigation:
        main_classes = "      - Main Classes:\n"
        if api_navigation.count(main_classes) != 1:
            raise RuntimeError("mkdocs.yml must contain exactly one API Main Classes section")
        api_navigation = api_navigation.replace(
            main_classes,
            f"{main_classes}{AUTO_CLASSES_NAVIGATION_ENTRY}\n",
            1,
        )
    if api_navigation.count(FULL_API_NAVIGATION_ENTRY) != 1:
        raise RuntimeError("mkdocs.yml must contain exactly one Full API reference entry")
    rendered = f"{api_prefix}  - API:\n{api_navigation}\nplugins:{plugins}"
    if rendered.count(AUTO_CLASSES_NAVIGATION_ENTRY) != 1:
        raise RuntimeError("Auto Classes must appear exactly once under API Main Classes")
    return rendered


def generated_files() -> dict[Path, str]:
    """Return every expected generated path and its contents."""
    specs = tuple(list_model_specs(task=None))
    model_types = {spec.model_type for spec in specs}
    reference_types = set(MODEL_REFERENCES)
    if model_types != reference_types:
        missing = sorted(model_types - reference_types)
        unknown = sorted(reference_types - model_types)
        raise RuntimeError(f"Model reference coverage mismatch: missing={missing}, unknown={unknown}")
    files = {MODEL_PAGE_DIR / f"{spec.model_type}.md": render_page(spec) for spec in specs}
    files[MODEL_PAGE_DIR / "index.md"] = render_index(specs)
    files[SITE_CONFIG_PATH] = render_site_config(specs)
    return files


def check_generated_files(files: dict[Path, str]) -> tuple[Path, ...]:
    """Return generated paths that are missing or stale."""
    expected = set(files)
    stale = [
        path for path, content in files.items()
        if not path.is_file() or path.read_text(encoding="utf-8") != content
    ]
    if MODEL_PAGE_DIR.is_dir():
        stale.extend(path for path in MODEL_PAGE_DIR.glob("*.md") if path not in expected)
    return tuple(stale)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail when generated model pages are missing or stale.",
    )
    args = parser.parse_args()
    files = generated_files()
    stale = check_generated_files(files)
    if args.check:
        if stale:
            for path in stale:
                print(f"stale: {path.relative_to(REPOSITORY_ROOT)}", file=sys.stderr)
            return 1
        page_count = sum(path.parent == MODEL_PAGE_DIR and path.name != "index.md" for path in files)
        print(f"OK: {page_count} model pages and navigation are current")
        return 0

    MODEL_PAGE_DIR.mkdir(parents=True, exist_ok=True)
    expected = set(files)
    for path in tuple(MODEL_PAGE_DIR.glob("*.md")):
        if path not in expected:
            path.unlink()
            print(f"removed: {path.relative_to(REPOSITORY_ROOT)}")
    for path, content in files.items():
        if not path.is_file() or path.read_text(encoding="utf-8") != content:
            path.write_text(content, encoding="utf-8", newline="\n")
            print(f"wrote: {path.relative_to(REPOSITORY_ROOT)}")
    page_count = sum(path.parent == MODEL_PAGE_DIR and path.name != "index.md" for path in files)
    print(f"OK: {page_count} model pages and navigation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
