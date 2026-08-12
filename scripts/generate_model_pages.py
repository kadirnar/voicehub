#!/usr/bin/env python3
"""Generate one compact, contract-aligned guide for every registered model."""

from __future__ import annotations

import argparse
import html
import re
import sys
from collections import Counter
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(SCRIPTS_ROOT))

from documentation_references import MODEL_REFERENCES, Reference  # noqa: E402
from model_documentation import TASK_LABELS, TASK_ORDER, checkpoint_documentation, inference_profile  # noqa: E402

from voicehub import list_model_specs  # noqa: E402
from voicehub.models.language_support import model_language_support  # noqa: E402

MODEL_PAGE_DIR = REPOSITORY_ROOT / "docs" / "models" / "providers"
SITE_CONFIG_PATH = REPOSITORY_ROOT / "mkdocs.yml"
GENERATOR_PATH = "scripts/generate_model_pages.py"
COLAB_ROOT = ("https://colab.research.google.com/github/kadirnar/voicehub/"
              "blob/main/notebooks/models")
NAVIGATION_START = "      # BEGIN GENERATED MODEL GUIDE NAVIGATION"
NAVIGATION_END = "      # END GENERATED MODEL GUIDE NAVIGATION"
MODEL_LIST_NAVIGATION_ENTRY = "      - Model list: models/providers/index.md"
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

CATALOG_FEATURE_FILTERS = (
    ("Voice cloning", "voice-cloning"),
    ("Streaming", "streaming"),
    ("Timestamps", "timestamps"),
    ("Translation", "translation"),
    ("Voice design", "voice-design"),
    ("Expressive speech", "expressive-speech"),
    ("Long-form audio", "long-form"),
    ("Speaker attribution", "speaker-attribution"),
    ("Language identification", "language-identification"),
    ("Hotwords", "hotwords"),
    ("Frame scores", "frame-scores"),
)
CATALOG_TRAINING_LABELS = {
    "native": "Native training",
    "preprocessed": "Prepared-data training",
    "custom": "Custom training",
    "inference-only": "Inference only",
}
CATALOG_CHECKPOINT_LABELS = {
    "huggingface": "Hugging Face",
    "external-archive": "External archive",
    "local": "Local or caller-provided",
}
CATALOG_LICENSE_LABELS = {
    "commercial": "Commercial use declared",
    "noncommercial": "Non-commercial",
    "review": "Review required",
    "checkpoint-specific": "Checkpoint-specific",
}


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
        return _code_list(support.codes)
    return "Not text-language conditioned"


def _language_details(spec) -> str:
    support = model_language_support(spec)
    if support.kind == "enumerated":
        codes = _code_list(support.codes)
        note = f"\n\n{support.note}" if support.note else ""
        return f'''<details class="vh-language-support" markdown>
<summary>Supported language abbreviations</summary>

{codes}{note}

</details>'''
    return support.note or "The model is not text-language conditioned."


def _checkpoint(spec) -> tuple[str, str]:
    documentation = checkpoint_documentation(spec)
    if documentation.identifier and documentation.url:
        rendered = f"[`{documentation.identifier}`]({documentation.url})"
    elif documentation.identifier:
        rendered = f"`{documentation.identifier}`"
    else:
        rendered = "No default; pass a compatible Hub ID or local directory."
    return documentation.example, rendered


def _render_call_arguments(arguments: tuple[str, ...], *, indent: int = 4) -> str:
    prefix = " " * indent
    return "".join(f"{prefix}{argument},\n" for argument in arguments)


def _inference_code(spec) -> str:
    checkpoint, _ = _checkpoint(spec)
    profile = inference_profile(spec)
    if spec.task.value == "text-to-speech":
        if not profile.high_level_supported:
            return f'''from voicehub import AutoModelForTextToSpeech

model = AutoModelForTextToSpeech.from_pretrained(
    {checkpoint!r},
    model_type={spec.model_type!r},
    device="cuda",
    lazy_load=True,
)
model.load()
required_stages = (
    "forward_lm",
    "forward_tts_lm",
    "sample_speech_latents",
    "decode_speech_latents",
)
missing = [name for name in required_stages if not hasattr(model.model, name)]
if missing:
    raise RuntimeError(f"Missing audited VibeVoice stage(s): {{', '.join(missing)}}")
print("High-level synthesis is not verified; available native stages:", required_stages)'''

        imports = (
            "AutoModelForTextToSpeech",
            "TTSGenerationConfig",
            *profile.voicehub_imports,
        )
        setup_imports = "\nimport json" if any("json." in line for line in profile.setup) else ""
        setup = "\n".join(profile.setup)
        if setup:
            setup += "\n\n"
        load_arguments = _render_call_arguments(profile.load_arguments)
        arguments = _render_call_arguments(profile.arguments)
        rendered_imports = ", ".join(dict.fromkeys(imports))
        return f'''from pathlib import Path{setup_imports}

from voicehub import {rendered_imports}

{setup}model = AutoModelForTextToSpeech.from_pretrained(
    {checkpoint!r},
    model_type={spec.model_type!r},
    device="cuda",
    lazy_load=True,
{load_arguments})
output = model.generate(
    {profile.text!r},
    generation_config=TTSGenerationConfig(
        seed=42,
        output_file=Path("output.wav"),
    ),
{arguments})
print(output.file_path, output.sample_rate, output.metadata)'''

    setup = '''AUDIO_FILE = Path("speech.wav")
if not AUDIO_FILE.is_file():
    raise FileNotFoundError(AUDIO_FILE)

'''
    arguments = _render_call_arguments(profile.arguments)
    if spec.task.value == "automatic-speech-recognition":
        return f'''from pathlib import Path

from voicehub import AutoModelForSpeechRecognition

{setup}model = AutoModelForSpeechRecognition.from_pretrained(
    {checkpoint!r},
    model_type={spec.model_type!r},
    device="cuda",
    lazy_load=True,
)
output = model.transcribe(
    AUDIO_FILE,
{arguments})
print(output.text)
for segment in output.segments:
    print(segment.start, segment.end, segment.text, segment.confidence)'''
    return f'''from pathlib import Path

from voicehub import AutoModelForVoiceActivityDetection

{setup}model = AutoModelForVoiceActivityDetection.from_pretrained(
    {checkpoint!r},
    model_type={spec.model_type!r},
    device="cpu",
    lazy_load=True,
)
output = model.detect(
    AUDIO_FILE,
{arguments})
for segment in output.segments:
    print(segment.start, segment.end, segment.score)'''


def _inference_notes(spec) -> str:
    profile = inference_profile(spec)
    notes = [
        (
            "This example is maintained against VoiceHub's public API; it is "
            "not copied from an upstream demo or package README."),
        f"**Model-specific path:** {profile.summary}",
        f"**Inputs and controls:** {profile.input_note}",
    ]
    checkpoint_note = checkpoint_documentation(spec).note
    if checkpoint_note:
        notes.append(f"**Checkpoint note:** {checkpoint_note}")
    return "\n\n".join(notes)


def _hugging_face_checkpoint(spec, *, detailed: bool = True) -> str:
    documentation = checkpoint_documentation(spec)
    if documentation.hugging_face_id is not None:
        rendered = (f"[`{documentation.hugging_face_id}`]"
                    f"({documentation.hugging_face_url})")
    else:
        rendered = "Not published / not applicable"
    if detailed:
        return f"{rendered}<br>{_cell(documentation.hugging_face_status)}"
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

Complete the [VoiceHub installation](../../getting-started/installation.md) once,
then run this repository-authored example. Model pages intentionally contain no
package-install command.

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
| Hugging Face ID | {_hugging_face_checkpoint(spec)} |
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


def _catalog_license_kind(spec) -> str:
    """Return a compact, filterable license posture for one model."""
    license_spec = spec.license
    if license_spec is None:
        return "checkpoint-specific"
    if license_spec.commercial_use is True:
        return "commercial"
    if license_spec.commercial_use is False:
        return "noncommercial"
    return "review"


def _render_catalog_options(options) -> str:
    """Render select options with registry-derived result counts."""
    return "\n".join(
        f'<option value="{html.escape(value, quote=True)}">'
        f'{html.escape(label)} ({count})</option>' for value, label, count in options if count)


def _render_catalog_select(name: str, label: str, options: str) -> str:
    """Render one labelled model explorer select."""
    identifier = f"vh-model-{name}"
    return f'''<div class="vh-model-filter-field">
        <label for="{identifier}">{html.escape(label)}</label>
        <select id="{identifier}" name="{html.escape(name, quote=True)}" data-vh-model-select>
          {options}
        </select>
      </div>'''


def _render_catalog_checkbox(*, name: str, value: str, label: str, count: int) -> str:
    """Render one count-labelled toggle chip."""
    identifier = f"vh-model-{name}-{value}"
    return f'''<label class="vh-model-filter-chip" for="{identifier}">
            <input id="{identifier}" name="{html.escape(name, quote=True)}"
              type="checkbox" value="{html.escape(value, quote=True)}"
              data-vh-model-checkbox data-filter-label="{html.escape(label, quote=True)}">
            <span>{html.escape(label)} <small>{count}</small></span>
          </label>'''


def _render_catalog_card(spec) -> str:
    """Render one semantic, filter-ready model result card."""
    checkpoint = checkpoint_documentation(spec)
    support = model_language_support(spec)
    training = spec.training.support.value
    license_kind = _catalog_license_kind(spec)
    task = spec.task.value
    task_short = {
        "text-to-speech": "TTS",
        "automatic-speech-recognition": "ASR",
        "voice-activity-detection": "VAD",
    }[task]
    language_codes = support.codes
    preview_codes = language_codes[:5]
    language_preview = "".join(f'<code>{html.escape(code)}</code>' for code in preview_codes)
    if support.kind == "not-text-conditioned":
        language_preview = '<span class="vh-model-card__neutral-language">Language-neutral</span>'
    elif len(language_codes) > len(preview_codes):
        language_preview += (
            f'<span class="vh-model-card__more-languages">+{len(language_codes) - len(preview_codes)}</span>')

    filter_capabilities = tuple((label, capability) for label, capability in CATALOG_FEATURE_FILTERS
                                if capability in spec.capabilities)
    visible_capabilities = filter_capabilities[:3]
    capability_chips = "".join(f'<span>{html.escape(label)}</span>' for label, _ in visible_capabilities)
    if len(filter_capabilities) > len(visible_capabilities):
        capability_chips += (
            f'<span class="vh-model-card__more-features">'
            f'+{len(filter_capabilities) - len(visible_capabilities)} more</span>')
    if not capability_chips:
        capability_chips = '<span>Core speech inference</span>'

    notebook_available = checkpoint.is_hugging_face
    resources = tuple(
        resource for resource, available in (
            ("notebook", notebook_available),
            ("huggingface", checkpoint.has_hugging_face_id),
        ) if available)
    actions = [
        f'<a class="vh-model-card__primary-action" href="{html.escape(spec.model_type, quote=True)}/">'
        'View model <span aria-hidden="true">→</span></a>',
    ]
    if checkpoint.has_hugging_face_id:
        actions.append(f'<a href="{html.escape(checkpoint.hugging_face_url, quote=True)}">Hugging Face</a>')
    if notebook_available:
        actions.append(f'<a href="{COLAB_ROOT}/{html.escape(spec.model_type, quote=True)}.ipynb">Colab</a>')

    architecture = spec.architecture or "provider-owned"
    search_values = (
        spec.display_name,
        spec.model_type,
        TASK_LABELS[task],
        task_short,
        architecture,
        checkpoint.identifier,
        checkpoint.hugging_face_id or "",
        CATALOG_TRAINING_LABELS[training],
        CATALOG_CHECKPOINT_LABELS[checkpoint.provider],
        CATALOG_LICENSE_LABELS[license_kind],
    )
    summary = inference_profile(spec).summary
    training_rank = {
        "native": 0,
        "preprocessed": 1,
        "custom": 2,
        "inference-only": 3,
    }[training]
    return f'''<article class="vh-model-card"
      data-vh-model-card
      data-name="{html.escape(spec.display_name.casefold(), quote=True)}"
      data-model-type="{html.escape(spec.model_type, quote=True)}"
      data-task="{html.escape(task, quote=True)}"
      data-training="{html.escape(training, quote=True)}"
      data-training-rank="{training_rank}"
      data-checkpoint="{html.escape(checkpoint.provider, quote=True)}"
      data-license="{html.escape(license_kind, quote=True)}"
      data-architecture="{html.escape(architecture, quote=True)}"
      data-language-kind="{html.escape(support.kind, quote=True)}"
      data-language-count="{len(language_codes)}"
      data-languages="{html.escape(' '.join(language_codes), quote=True)}"
      data-capabilities="{html.escape(' '.join(spec.capabilities), quote=True)}"
      data-resources="{html.escape(' '.join(resources), quote=True)}"
      data-search="{html.escape(' '.join(search_values), quote=True)}">
      <div class="vh-model-card__topline">
        <span class="vh-model-badge vh-model-badge--{task_short.casefold()}">{task_short}</span>
        <span class="vh-model-card__training">{html.escape(CATALOG_TRAINING_LABELS[training])}</span>
      </div>
      <div class="vh-model-card__heading">
        <h2><a href="{html.escape(spec.model_type, quote=True)}/">{html.escape(spec.display_name)}</a></h2>
        <code>{html.escape(spec.model_type)}</code>
      </div>
      <p class="vh-model-card__summary">{html.escape(summary)}</p>
      <dl class="vh-model-card__metadata">
        <div><dt>Architecture</dt><dd><code>{html.escape(architecture)}</code></dd></div>
        <div><dt>Languages</dt><dd class="vh-model-card__languages">{language_preview}</dd></div>
        <div><dt>Checkpoint</dt><dd>{html.escape(CATALOG_CHECKPOINT_LABELS[checkpoint.provider])}</dd></div>
        <div><dt>License</dt><dd>{html.escape(CATALOG_LICENSE_LABELS[license_kind])}</dd></div>
      </dl>
      <div class="vh-model-card__features" aria-label="Capabilities">{capability_chips}</div>
      <footer class="vh-model-card__actions">{' '.join(actions)}</footer>
    </article>'''


def render_index(specs) -> str:
    """Render the generated, progressively enhanced model explorer."""
    specs = tuple(specs)
    task_counts = Counter(spec.task.value for spec in specs)
    training_counts = Counter(spec.training.support.value for spec in specs)
    checkpoint_counts = Counter(checkpoint_documentation(spec).provider for spec in specs)
    license_counts = Counter(_catalog_license_kind(spec) for spec in specs)
    architecture_counts = Counter(spec.architecture or "provider-owned" for spec in specs)
    language_counts = Counter(code for spec in specs for code in model_language_support(spec).codes)
    cards = "\n".join(
        _render_catalog_card(spec)
        for spec in sorted(specs, key=lambda spec: (spec.display_name.casefold(), spec.model_type)))
    feature_filters = "\n".join(
        _render_catalog_checkbox(
            name="feature",
            value=capability,
            label=label,
            count=sum(capability in spec.capabilities for spec in specs),
        ) for label, capability in CATALOG_FEATURE_FILTERS)
    resource_filters = "\n".join((
        _render_catalog_checkbox(
            name="resource",
            value="notebook",
            label="Colab notebook",
            count=sum(checkpoint_documentation(spec).is_hugging_face for spec in specs),
        ),
        _render_catalog_checkbox(
            name="resource",
            value="huggingface",
            label="Hugging Face page",
            count=sum(checkpoint_documentation(spec).has_hugging_face_id for spec in specs),
        ),
    ))
    task_options = _render_catalog_options(
        (task, TASK_LABELS[task], task_counts[task]) for task in TASK_ORDER)
    training_options = _render_catalog_options(
        (value, label, training_counts[value]) for value, label in CATALOG_TRAINING_LABELS.items())
    checkpoint_options = _render_catalog_options(
        (value, label, checkpoint_counts[value]) for value, label in CATALOG_CHECKPOINT_LABELS.items())
    license_options = _render_catalog_options(
        (value, label, license_counts[value]) for value, label in CATALOG_LICENSE_LABELS.items())
    architecture_options = _render_catalog_options(
        (value, value, count)
        for value, count in sorted(architecture_counts.items(), key=lambda item: item[0].casefold()))
    language_select = _render_catalog_select(
        "language",
        "Language",
        '<option value="">Any language</option>\n'
        '<option value="not-text-conditioned">Language-neutral (VAD)</option>',
    )
    task_select = _render_catalog_select(
        "task", "Task", '<option value="">Any task</option>\n' + task_options)
    training_select = _render_catalog_select(
        "training",
        "Training",
        '<option value="">Any training path</option>\n' + training_options,
    )
    checkpoint_select = _render_catalog_select(
        "checkpoint",
        "Checkpoint",
        '<option value="">Any checkpoint source</option>\n' + checkpoint_options,
    )
    license_select = _render_catalog_select(
        "license",
        "License",
        '<option value="">Any license status</option>\n' + license_options,
    )
    architecture_select = _render_catalog_select(
        "architecture",
        "Architecture",
        '<option value="">Any architecture</option>\n' + architecture_options,
    )
    return f'''---
description: Search and filter every registered VoiceHub TTS, ASR, and VAD model.
---

# Model list

<div class="vh-model-explorer" data-vh-model-explorer data-model-count="{len(specs)}">
  <section class="vh-model-explorer__hero" aria-labelledby="vh-model-explorer-title">
    <div class="vh-model-explorer__hero-copy">
      <p class="vh-model-explorer__eyebrow">Model discovery</p>
      <h2 id="vh-model-explorer-title">Find the right speech model</h2>
      <p>Search the complete VoiceHub catalog by language, task, training path,
      architecture, checkpoint source, license, and production capability.</p>
    </div>
    <dl class="vh-model-explorer__stats" aria-label="Catalog summary">
      <div><dt>{len(specs)}</dt><dd>models</dd></div>
      <div><dt>{len(language_counts)}</dt><dd>indexed codes</dd></div>
      <div><dt>{task_counts['text-to-speech']}</dt><dd>TTS</dd></div>
      <div><dt>{task_counts['automatic-speech-recognition']}</dt><dd>ASR</dd></div>
    </dl>
  </section>

  <form class="vh-model-filters" data-vh-model-filters>
    <div class="vh-model-filters__search">
      <label for="vh-model-query">Search models</label>
      <div class="vh-model-search-field">
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="m21 21-4.35-4.35m2.35-5.4A7.75 7.75 0 1 1 3.5 11.25a7.75 7.75 0 0 1 15.5 0Z"/></svg>
        <input id="vh-model-query" name="query" type="search" autocomplete="off"
          placeholder="Search model, architecture, feature, or Hub ID…"
          aria-describedby="vh-model-search-hint" data-vh-model-query>
        <kbd>/</kbd>
      </div>
      <span id="vh-model-search-hint" class="vh-model-filters__hint">Try “Turkish voice cloning” or “Whisper timestamps”.</span>
    </div>

    <div class="vh-model-filters__controls">
      <div class="vh-model-filters__quick" aria-label="Quick filters">
        {language_select}
        {task_select}
        {training_select}
      </div>

      <details class="vh-model-filters__advanced">
        <summary>
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 6h16M7 12h10m-7 6h4"/></svg>
          <span>More filters</span>
          <span class="vh-model-filters__advanced-count"
            data-vh-model-advanced-count hidden>0</span>
        </summary>
        <div class="vh-model-filters__advanced-body">
          <div class="vh-model-filters__secondary">
            {checkpoint_select}
            {license_select}
            {architecture_select}
          </div>
          <fieldset>
            <legend>Capabilities <span>Models must include every selected capability</span></legend>
            <div class="vh-model-filter-chips">{feature_filters}</div>
          </fieldset>
          <fieldset>
            <legend>Resources</legend>
            <div class="vh-model-filter-chips">{resource_filters}</div>
          </fieldset>
        </div>
      </details>
    </div>
  </form>

  <div class="vh-model-results__toolbar">
    <div>
      <p class="vh-model-results__count" role="status" aria-live="polite">
        <strong data-vh-model-result-count>{len(specs)}</strong>
        <span data-vh-model-result-label>models</span>
      </p>
      <div class="vh-model-active-filters" data-vh-model-active-filters hidden></div>
    </div>
    <div class="vh-model-results__actions">
      <button type="button" class="vh-model-clear" data-vh-model-clear hidden>Clear filters</button>
      <label for="vh-model-sort">Sort</label>
      <select id="vh-model-sort" name="sort" data-vh-model-sort>
        <option value="name">Name A–Z</option>
        <option value="languages">Language coverage</option>
        <option value="task">Task</option>
        <option value="training">Training readiness</option>
      </select>
    </div>
  </div>

  <div class="vh-model-results" data-vh-model-results>{cards}</div>
  <div class="vh-model-empty" data-vh-model-empty hidden>
    <span aria-hidden="true">⌕</span>
    <h2>No models match these filters</h2>
    <p>Try a broader language, remove a capability, or clear the search.</p>
    <button type="button" data-vh-model-clear>Clear all filters</button>
  </div>
  <noscript><p class="vh-model-explorer__noscript">Enable JavaScript to filter this catalog. All model cards remain available below.</p></noscript>
</div>

## Search the registry in Python

Registry discovery stays lazy and imports no model runtime.

```python
from voicehub import list_model_specs

for model in list_model_specs():
    print(model.task.value, model.display_name, model.model_type)
```

Use the [training matrix](../training-support.md) and
[optimization catalog](../../optimizations/index.md) for deeper comparisons.

Generated by `{GENERATOR_PATH}` from lazy registry metadata.
'''


def render_navigation(specs) -> str:
    """Render the task-grouped model links shown in the site sidebar."""
    lines = [NAVIGATION_START]
    for task in TASK_ORDER:
        task_specs = sorted(
            (spec for spec in specs if spec.task.value == task),
            key=lambda spec: (spec.display_name.casefold(), spec.model_type),
        )
        lines.append(f"      - {TASK_LABELS[task]}:")
        lines.extend(
            f'          - "{spec.display_name}": models/providers/{spec.model_type}.md'
            for spec in task_specs)
    lines.append(NAVIGATION_END)
    return "\n".join(lines)


def render_site_config(specs) -> str:
    """Replace the generated model list in the public Models section."""
    source = SITE_CONFIG_PATH.read_text(encoding="utf-8")
    if source.count(NAVIGATION_START) != 1 or source.count(NAVIGATION_END) != 1:
        raise RuntimeError("mkdocs.yml must contain exactly one generated model navigation block")
    prefix, remainder = source.split(NAVIGATION_START, 1)
    _, suffix = remainder.split(NAVIGATION_END, 1)
    rendered = f"{prefix}{render_navigation(specs)}{suffix}"
    if rendered.count(MODEL_LIST_NAVIGATION_ENTRY) != 1:
        raise RuntimeError("Model list must appear exactly once under Models")
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
