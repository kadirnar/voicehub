#!/usr/bin/env python3
"""Generate one compact, contract-aligned guide for every registered model."""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(SCRIPTS_ROOT))

from documentation_references import MODEL_REFERENCES, Reference  # noqa: E402
from model_documentation import (  # noqa: E402
    TASK_LABELS,
    TASK_ORDER,
    checkpoint_documentation,
    inference_profile,
    parameter_documentation,
)

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


def _language_summary(spec, *, full_list_location: str = "below") -> str:
    support = model_language_support(spec)
    if support.kind == "enumerated":
        if len(support.codes) > 8:
            preview = _code_list(support.codes[:4])
            return f"{preview}, … complete audited list {full_list_location}"
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
    if _is_weightless(spec):
        return (
            "This weightless runtime does not select a spoken language and is not "
            "text-language conditioned; validate its "
            "implementation, configuration, and recording conditions for the target speech.")
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
    source_record = _source_record_path(spec)
    if source_record is not None:
        return f"`{source_record.relative_to(REPOSITORY_ROOT).as_posix()}`"
    return "No integration-specific bundled `SOURCE.json` is declared for this registry entry."


def _source_record_path(spec) -> Path | None:
    """Resolve the closest bundled provenance record without importing a backend."""
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
            return candidate
    return None


def _source_terms(spec) -> tuple[str, str | None]:
    """Return audited source-license terms and their local evidence link when declared."""
    source_record = _source_record_path(spec)
    if source_record is None:
        return "Source terms require review", None
    try:
        record = json.loads(source_record.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return "Source terms require review", None
    upstream = record.get("upstream")
    license_name = upstream.get("license") if isinstance(upstream, dict) else None
    if not isinstance(license_name, str) or not license_name.strip():
        return "Source terms require review", None
    source_url = (
        "https://github.com/kadirnar/voicehub/blob/main/" +
        source_record.relative_to(REPOSITORY_ROOT).as_posix())
    return license_name.strip(), source_url


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
    training_artifact_label = "Runtime identifier" if _is_weightless(spec) else "Training checkpoint"
    summary = f'''| Property | Value |
| --- | --- |
| Support | `{training.support.value}` |
| Family | `{training.family_name}` |
| Recipe | `{training.recipe_kind.value}` |
| Default phase | `{training.default_phase}` |
| {training_artifact_label} | `{training_checkpoint}` |
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


def _is_weightless(spec) -> bool:
    """Whether the integration is an algorithm with no model weights."""
    return parameter_documentation(spec).count == 0


def _license_details(spec) -> tuple[str, str | None, str]:
    """Return the compact label, source URL, and complete license notice."""
    if _is_weightless(spec):
        license_label, source_url = _source_terms(spec)
        if source_url:
            notice = (
                "This weightless runtime has no checkpoint license. Its audited source "
                f"record declares **{license_label}**; verify those implementation terms.")
        else:
            notice = (
                "This weightless runtime has no checkpoint license. Review the VoiceHub "
                "and upstream implementation terms before use.")
        return license_label, source_url, notice
    license_spec = spec.license
    if license_spec is None:
        return (
            "Checkpoint-specific",
            None,
            "No VoiceHub-specific license override is registered. Verify the "
            "checkpoint and upstream source terms before use.",
        )
    commercial = {
        True: "allowed by the registered terms",
        False: "not allowed",
        None: "review required",
    }[license_spec.commercial_use]
    return (
        license_spec.license_id,
        license_spec.upstream,
        f"{license_spec.notice} Commercial use: **{commercial}**.",
    )


def _module_source_url(module: str) -> str:
    """Return the stable repository URL for a declared lazy module."""
    source_path = _module_source_path(module).as_posix()
    return f"https://github.com/kadirnar/voicehub/blob/main/{source_path}"


def _render_model_chip(label: str, kind: str, *, described_by: str | None = None) -> str:
    description = (f' aria-describedby="{html.escape(described_by, quote=True)}"' if described_by else "")
    return (
        f'<span class="vh-model-detail__chip" data-chip-kind="{html.escape(kind, quote=True)}"'
        f'{description}>{html.escape(label)}</span>')


def _model_language_chip(spec) -> str:
    support = model_language_support(spec)
    if support.kind != "enumerated":
        return "Not text-language conditioned"
    if len(support.codes) == 1:
        return f"Language: {support.codes[0]}"
    if len(support.codes) == 2:
        return f"Languages: {', '.join(support.codes)}"
    preview = ", ".join(support.codes[:2])
    return f"Languages: {preview} +{len(support.codes) - 2}"


def _render_model_namespace(spec, checkpoint_metadata) -> str:
    if (checkpoint_metadata.is_hugging_face and checkpoint_metadata.hugging_face_id and
            checkpoint_metadata.hugging_face_url):
        owner, repository = checkpoint_metadata.hugging_face_id.split("/", 1)
        owner_url = f"https://huggingface.co/{owner}"
    else:
        owner = "VoiceHub"
        owner_url = "https://github.com/kadirnar/voicehub"
        repository = spec.model_type
    initials = "".join(character for character in owner if character.isupper())[:2]
    if len(initials) < 2:
        initials = "".join(part[0] for part in re.split(r"[^A-Za-z0-9]+", owner) if part)[:2].upper()
    if len(initials) < 2:
        initials = owner[:2].upper()
    return (
        '<p class="vh-model-detail__namespace" aria-label="Model repository">'
        f'<span class="vh-model-detail__owner-avatar" aria-hidden="true">{html.escape(initials)}</span>'
        f'<a href="{html.escape(owner_url, quote=True)}">{html.escape(owner)}</a>'
        '<span aria-hidden="true">/</span>'
        f'<strong>{html.escape(repository)}</strong></p>')


def _render_model_actions(spec, checkpoint_metadata) -> str:
    references = MODEL_REFERENCES[spec.model_type]
    source_url = _module_source_url(spec.module)
    actions = [
        '<a class="vh-model-detail__action vh-model-detail__action--primary" '
        'href="#usage" data-vh-model-action="use">Use this model</a>',
    ]
    if checkpoint_metadata.identifier and not _is_weightless(spec):
        checkpoint_id = html.escape(checkpoint_metadata.identifier, quote=True)
        checkpoint_description_id = f"vh-model-checkpoint-{spec.model_type}"
        actions.append(
            '<button class="vh-model-detail__action vh-model-detail__copy" type="button" '
            f'data-vh-copy-model-id data-model-id="{checkpoint_id}" '
            f'aria-describedby="{checkpoint_description_id}">'
            '<span data-vh-copy-model-id-label>Copy model ID</span></button>')
    if checkpoint_metadata.url:
        checkpoint_label = "Runtime source" if _is_weightless(spec) else "Checkpoint"
        actions.append(
            f'<a class="vh-model-detail__action" href="{html.escape(checkpoint_metadata.url, quote=True)}" '
            f'data-vh-model-action="checkpoint">{checkpoint_label}</a>')
    resources = []
    if references.papers:
        paper = references.papers[0]
        resources.append(
            f'<a href="{html.escape(paper.url, quote=True)}" '
            'data-vh-model-action="paper">Paper</a>')
    resources.extend((
        f'<a href="{html.escape(references.github.url, quote=True)}" '
        'data-vh-model-action="github">Upstream GitHub</a>',
        f'<a href="{html.escape(source_url, quote=True)}" '
        'data-vh-model-action="source">VoiceHub source</a>',
    ))
    if checkpoint_metadata.is_hugging_face:
        notebook_url = f"{COLAB_ROOT}/{spec.model_type}.ipynb"
        resources.append(
            f'<a href="{html.escape(notebook_url, quote=True)}" '
            'data-vh-model-action="colab">Open in Colab</a>')
    actions.append(
        '<details class="vh-model-detail__resources">\n'
        '<summary class="vh-model-detail__action">Resources</summary>\n'
        '<div class="vh-model-detail__resource-menu">\n' + "\n".join(resources) + "\n</div>\n</details>")
    return (
        '<div class="vh-model-detail__actions" aria-label="Model actions">\n' + "\n".join(actions) +
        "\n</div>")


def _render_model_detail_hero(spec, checkpoint_metadata, license_label: str) -> str:
    parameter_metadata = parameter_documentation(spec)
    parameter_note_id = f"vh-model-parameters-note-{spec.model_type}"
    architecture = spec.architecture or "provider-owned"
    runtime = "VoiceHub-native" if spec.is_voicehub_native else "Provider adapter"
    chips = [
        _render_model_chip(TASK_LABELS[spec.task.value], "task"),
        _render_model_chip(runtime, "runtime"),
        _render_model_chip(architecture, "architecture"),
        _render_model_chip(
            f"Parameters: {_format_parameter_count(parameter_metadata.count)}",
            "parameters",
            described_by=parameter_note_id,
        ),
        _render_model_chip(_model_language_chip(spec), "language"),
        _render_model_chip(f"Training: {spec.training.support.value}", "training"),
        _render_model_chip(f"License: {license_label}", "license"),
    ]
    profile = inference_profile(spec)
    return f'''<header class="vh-model-detail__hero" data-vh-model-hero markdown>

{_render_model_namespace(spec, checkpoint_metadata)}

# {spec.display_name} {{.vh-model-title}}

<p class="vh-model-detail__summary">{html.escape(profile.summary)}</p>
<div class="vh-model-detail__tags" aria-label="Model metadata">{"".join(chips)}</div>
<p class="vh-model-detail__parameter-note" id="{parameter_note_id}"><strong>Parameter metadata:</strong> {html.escape(parameter_metadata.note)}</p>
{_render_model_actions(spec, checkpoint_metadata)}
</header>'''


def _render_model_detail_tabs(spec) -> str:
    tabs = (
        ("usage", "Usage", "#usage", False),
        ("model-card", "Model card", "#overview", True),
        ("sources", "Sources", "#paper-and-github", False),
        ("training", "Training", "#training-and-optimization", False),
        (
            "checkpoint",
            "Runtime" if _is_weightless(spec) else "Checkpoint",
            "#checkpoints-provenance-license-and-limitations",
            False,
        ),
        ("api", "Public API", "#public-api", False),
    )
    links = []
    for value, label, target, active in tabs:
        current = ' aria-current="location"' if active else ""
        links.append(f'<a href="{target}" data-vh-model-tab="{value}"{current}>{label}</a>')
    return ('<nav class="vh-model-detail__tabs" aria-label="Model sections">' + "".join(links) + "</nav>")


def _render_model_languages_fact(spec) -> str:
    support = model_language_support(spec)
    if support.kind != "enumerated":
        return html.escape(_language_details(spec))
    codes = " ".join(f"<code>{html.escape(code)}</code>" for code in support.codes)
    if len(support.codes) <= 4:
        return codes
    return (
        '<details class="vh-model-detail__languages">'
        f'<summary>{len(support.codes)} documented codes</summary>'
        f'<span>{codes}</span></details>')


def _render_model_capabilities_fact(spec) -> str:
    capabilities = " ".join(f"<code>{html.escape(capability)}</code>" for capability in spec.capabilities)
    if len(spec.capabilities) <= 3:
        return capabilities or "Not declared"
    return (
        '<details class="vh-model-detail__capabilities">'
        f'<summary>{len(spec.capabilities)} capabilities</summary>'
        f'<span>{capabilities}</span></details>')


def _render_model_fact(label: str, value: str, *, value_attributes: str = "") -> str:
    return (f'<div><dt>{html.escape(label)}</dt><dd{value_attributes}>{value}</dd></div>')


def _render_model_detail_facts(spec, checkpoint_metadata, license_label: str, license_url: str | None) -> str:
    parameter_metadata = parameter_documentation(spec)
    parameter_note_id = f"vh-model-parameters-note-{spec.model_type}"
    checkpoint_description_id = f"vh-model-checkpoint-{spec.model_type}"
    architecture = spec.architecture or "provider-owned"
    runtime = "VoiceHub-native" if spec.is_voicehub_native else "Provider adapter"
    parameter_count = html.escape(_format_parameter_count(parameter_metadata.count))
    if checkpoint_metadata.identifier:
        identifier = html.escape(checkpoint_metadata.identifier)
        checkpoint_value = f"<code>{identifier}</code>"
        if checkpoint_metadata.url:
            checkpoint_value = (
                f'<a href="{html.escape(checkpoint_metadata.url, quote=True)}">'
                f'{checkpoint_value}</a>')
    else:
        checkpoint_value = "Caller-provided compatible artifact"
    if license_url:
        license_value = (
            f'<a href="{html.escape(license_url, quote=True)}">'
            f'{html.escape(license_label)}</a>')
    else:
        license_value = html.escape(license_label)
    facts = (
        _render_model_fact("Task", html.escape(TASK_LABELS[spec.task.value])),
        _render_model_fact(
            "Parameters",
            parameter_count,
            value_attributes=f' aria-describedby="{parameter_note_id}"',
        ),
        _render_model_fact("Architecture", f"<code>{html.escape(architecture)}</code>"),
        _render_model_fact("Runtime", html.escape(runtime)),
        _render_model_fact("Languages", _render_model_languages_fact(spec)),
        _render_model_fact("Capabilities", _render_model_capabilities_fact(spec)),
        _render_model_fact("Training", f"<code>{html.escape(spec.training.support.value)}</code>"),
        _render_model_fact("License", license_value),
        _render_model_fact(
            "Runtime identifier" if _is_weightless(spec) else "Default checkpoint",
            checkpoint_value,
            value_attributes=f' id="{checkpoint_description_id}"',
        ),
    )
    facts_heading_id = f"vh-model-facts-title-{spec.model_type}"
    return (
        '<aside class="vh-model-detail__sidebar" data-vh-model-facts '
        f'aria-labelledby="{facts_heading_id}">'
        f'<h2 id="{facts_heading_id}">Model facts</h2>'
        '<details class="vh-model-detail__facts-disclosure" '
        f'data-vh-model-facts-disclosure aria-labelledby="{facts_heading_id}" open>'
        '<summary><span>Toggle model facts</span></summary>'
        '<dl class="vh-model-detail__facts">' + "".join(facts) + "</dl></details></aside>")


def _render_model_api_card(
        *, kind: str, badge: str, class_name: str, source_url: str, signature: str,
        parameters: tuple[tuple[str, str], ...]) -> str:
    parameter_items = "\n".join(f"- `{name}` — {description}" for name, description in parameters)
    return f'''<section class="vh-model-api-card" data-vh-model-api-card="{html.escape(kind, quote=True)}" markdown>
<p class="vh-model-api-card__badge-wrap"><span class="vh-model-api-card__badge">{html.escape(badge)}</span></p>

### `{class_name}`

<p class="vh-model-api-card__source-wrap"><a class="vh-model-api-card__source" href="{html.escape(source_url, quote=True)}">View source</a></p>
<div class="vh-model-api-card__signature" markdown>

```text
{signature}
```

</div>
<h4>Parameters</h4>
<div class="vh-model-api-card__parameters" markdown>
{parameter_items}
</div>
</section>'''


def render_page(spec) -> str:
    """Render one deterministic provider guide."""
    _, checkpoint = _checkpoint(spec)
    checkpoint_metadata = checkpoint_documentation(spec)
    license_label, license_url, license_text = _license_details(spec)
    license_value = f"[{license_label}]({license_url})" if license_url else license_label
    notebook = ""
    if checkpoint_metadata.is_hugging_face:
        notebook = (
            f" [Open the `{spec.model_type}` Colab notebook]"
            f"({COLAB_ROOT}/{spec.model_type}.ipynb).")
    architecture = spec.architecture or "provider-owned"
    components = _code_list(spec.components)
    factory = _factory_name(spec)
    output = _output_name(spec)
    parameter_metadata = parameter_documentation(spec)
    weightless = _is_weightless(spec)
    source_provenance = _source_provenance(spec)
    config_source_url = _module_source_url(spec.config_module)
    model_source_url = _module_source_url(spec.module)
    dependency_extra = (f"`voicehub[{spec.install_extra}]`" if spec.install_extra else "Core package")
    checkpoint_note = checkpoint_metadata.note or (
        parameter_metadata.note
        if weightless else "No integration-specific checkpoint limitation is registered. Verify the selected "
        "checkpoint revision and its documented runtime requirements.")
    production_note = (
        "Use authorized recordings. Version the implementation and configuration in production." if weightless
        else "Use authorized recordings. Verify hardware needs and pin a revision in production.")
    artifact_property = "Runtime identifier" if weightless else "Default checkpoint"
    checkpoint_status = (
        "Not applicable; this is a weightless algorithm with no checkpoint" if weightless else _cell(
            checkpoint_metadata.status))
    hardware_note = (
        f"Usage selects `{_example_device(spec)}`; verify implementation-specific requirements" if weightless
        else f"Usage selects `{_example_device(spec)}`; verify checkpoint-specific requirements")
    checkpoint_evidence = (
        "Not applicable; version the implementation, configuration, and source provenance" if weightless else
        "[Release evidence](../../project/release-readiness.md); a registry default alone is not "
        "execution evidence")
    confirmation = (
        "Confirm the implementation revision, source provenance, access terms, and license."
        if weightless else "Confirm the checkpoint revision, access terms, provenance, and license.")
    contract_evidence_note = (
        "Contract tests do not replace implementation and recording-condition validation."
        if weightless else "Contract tests do not replace the linked released-checkpoint evidence.")
    configuration_api = _render_model_api_card(
        kind="configuration",
        badge="Configuration",
        class_name=spec.config_class,
        source_url=config_source_url,
        signature=f"{spec.config_class}(**config_kwargs)",
        parameters=(("**config_kwargs", f"Configuration fields validated by {spec.config_class}."), ),
    )
    model_signature = f'''{factory}.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type={spec.model_type!r},
    config=None,
    **model_kwargs,
)'''
    model_api = _render_model_api_card(
        kind="model",
        badge="Model",
        class_name=spec.class_name,
        source_url=model_source_url,
        signature=model_signature,
        parameters=(
            (
                "pretrained_model_name_or_path",
                "Runtime identifier for this weightless implementation."
                if weightless else "Hub ID or compatible local directory.",
            ),
            ("model_type", f"Canonical model type; use {spec.model_type!r}."),
            ("config", f"Optional preloaded {spec.config_class} instance."),
            ("**model_kwargs", "Model-specific loading arguments."),
        ),
    )
    return f'''---
description: Public API, checkpoint, training, and optimization guide for the {spec.model_type} integration.
hide:
  - toc
---

<div class="vh-model-detail" data-vh-model-detail data-model-type="{html.escape(spec.model_type, quote=True)}" data-task="{html.escape(spec.task.value, quote=True)}" data-training="{html.escape(spec.training.support.value, quote=True)}" data-parameter-count="{parameter_metadata.count if parameter_metadata.count is not None else ''}" markdown>

{_render_model_detail_hero(spec, checkpoint_metadata, license_label)}

{_render_model_detail_tabs(spec)}

<div class="vh-model-detail__layout" markdown>

{_render_model_detail_facts(spec, checkpoint_metadata, license_label, license_url)}

<div class="vh-model-detail__main vh-model-detail__content" markdown>

## Usage

Complete the [VoiceHub installation](../../getting-started/installation.md) once,
then run this repository-authored example. Model pages intentionally contain no
package-install command.

{_inference_notes(spec)}

```python
{_inference_code(spec)}
```

{production_note}

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
| {artifact_property} | {checkpoint} |
| Hugging Face ID | {_hugging_face_checkpoint(spec)} |
| Checkpoint status | {checkpoint_status} |
| Optional dependency extra | {dependency_extra} |
| Hardware and runtime | {hardware_note} |
| Real-checkpoint evidence | {checkpoint_evidence} |
| Implementation | `{spec.module}.{spec.class_name}` |
| Configuration | `{spec.config_module}.{spec.config_class}` |
| Source provenance | {source_provenance} |
| License | {license_value} |

{license_text}

{confirmation}

### Limitations

- {checkpoint_note}
- Validate memory, precision, and optional dependencies on the target system.
- Public optimizations fail closed when the runtime or hardware cannot satisfy
  their validation contract; an unavailable pass is not reported as applied.
- {contract_evidence_note}

## Public API

Use the stable configuration, processor, and task-model facades below.

{configuration_api}

{model_api}

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

</div>

</div>

</div>
'''


def _format_parameter_count(count: int | None) -> str:
    """Format one exact count for compact model-page display."""
    if count is None:
        return "Not reported"
    if count == 0:
        return "Weightless"
    for threshold, suffix in ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K")):
        if count >= threshold:
            value = f"{count / threshold:.1f}".rstrip("0").rstrip(".")
            return f"{value}{suffix}"
    return str(count)


def render_index(specs) -> str:
    """Render the generated, task-grouped model index."""
    lines = [
        "---",
        "description: Browse every registered VoiceHub TTS, ASR, and VAD model.",
        "---",
        "",
        "# Model list",
        "",
        "Choose a model to open its dedicated usage, source, training, checkpoint,",
        "and public API documentation.",
        "",
    ]
    for task in TASK_ORDER:
        task_specs = sorted(
            (spec for spec in specs if spec.task.value == task),
            key=lambda spec: (spec.display_name.casefold(), spec.model_type),
        )
        lines.extend((
            f"## {TASK_LABELS[task]}",
            "",
            '<div class="vh-model-catalog" markdown>',
            "",
            "| Model | Languages | Hugging Face ID | Training | Notebook |",
            "| --- | --- | --- | --- | --- |",
        ))
        for spec in task_specs:
            checkpoint_metadata = checkpoint_documentation(spec)
            notebook = (
                f"[Colab]({COLAB_ROOT}/{spec.model_type}.ipynb)"
                if checkpoint_metadata.is_hugging_face else "—")
            lines.append(
                f"| [`{spec.display_name}`]({spec.model_type}.md) | "
                f"{_language_summary(spec, full_list_location='on the model page')} | "
                f"{_hugging_face_checkpoint(spec, detailed=False)} | "
                f"`{spec.training.support.value}` | {notebook} |")
        lines.extend(("", "</div>", ""))
    lines.extend((
        "## Registry access",
        "",
        "Registry discovery stays lazy and imports no model runtime.",
        "",
        "```python",
        "from voicehub import list_model_specs",
        "",
        "for model in list_model_specs():",
        "    print(model.task.value, model.display_name, model.model_type)",
        "```",
        "",
        "Use the [training matrix](../training-support.md) and",
        "[optimization catalog](../../optimizations/index.md) for compact comparisons.",
        "",
        f"Generated by `{GENERATOR_PATH}` from lazy registry metadata.",
        "",
    ))
    return "\n".join(lines)


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
