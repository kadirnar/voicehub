#!/usr/bin/env python3
"""Create, complete, and validate one explicit VoiceHub model scaffold."""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path, PurePath
from urllib.parse import urlsplit

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODEL_TYPE_PATTERN = re.compile(r"[a-z][a-z0-9_]*")
CLASS_PREFIX_PATTERN = re.compile(r"[A-Z][A-Za-z0-9]*")
MUTABLE_REVISIONS = frozenset({"dev", "head", "latest", "main", "master", "nightly", "trunk"})
IMPLEMENTATION_STATUS = "replace-me"
READY_STATUS = "ready"
MODEL_PAGE_SECTIONS = (
    "Overview",
    "Paper and GitHub",
    "Quickstart",
    "Supported tasks and capabilities",
    "Checkpoints, provenance, and license",
    "Optimization and training support",
    "Public API",
)


class _StrictJSONError(ValueError):
    """Internal error for ambiguous standalone scaffold artifacts."""


def _unique_json_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise _StrictJSONError(f"Duplicate JSON object key {key!r}.")
        value[key] = item
    return value


def _reject_json_constant(value):
    raise _StrictJSONError(f"Unsupported non-finite JSON constant {value!r}.")


def _json_path_for_key(path, key):
    return f"{path}.{key}" if key.isidentifier() else f"{path}[{key!r}]"


def _validate_finite_json_numbers(value, *, path="$"):
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_finite_json_numbers(
                item,
                path=_json_path_for_key(path, key),
            )
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_finite_json_numbers(item, path=f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise _StrictJSONError(f"{path} contains a non-finite JSON number.")


def _parse_strict_json(document, *, source):
    """Mirror the runtime JSON contract without importing VoiceHub."""
    try:
        value = json.loads(
            document,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
        _validate_finite_json_numbers(value)
    except _StrictJSONError as error:
        raise ValueError(f"Invalid JSON artifact {source}: {error}") from error
    return value


class ScaffoldError(ValueError):
    """Raised when a scaffold request is unsafe or incomplete."""


@dataclass(frozen=True)
class TaskTemplate:
    task: str
    enum_member: str
    factory: str
    base_class: str
    output_class: str
    implementation_method: str
    public_method: str
    model_suffix: str
    training_family: str
    input_name: str
    quickstart_input: str
    sample_rate: int


TASKS = {
    "tts":
    TaskTemplate(
        task="text-to-speech",
        enum_member="TEXT_TO_SPEECH",
        factory="AutoModelForTextToSpeech",
        base_class="PreTrainedTTSModel",
        output_class="TTSOutput",
        implementation_method="_generate",
        public_method="generate",
        model_suffix="ForTextToSpeech",
        training_family="ACOUSTIC",
        input_name="text",
        quickstart_input='"VoiceHub keeps integrations explicit."',
        sample_rate=24_000,
    ),
    "asr":
    TaskTemplate(
        task="automatic-speech-recognition",
        enum_member="AUTOMATIC_SPEECH_RECOGNITION",
        factory="AutoModelForSpeechRecognition",
        base_class="PreTrainedASRModel",
        output_class="ASROutput",
        implementation_method="_transcribe",
        public_method="transcribe",
        model_suffix="ForSpeechRecognition",
        training_family="CTC",
        input_name="audio",
        quickstart_input='"speech.wav"',
        sample_rate=16_000,
    ),
    "vad":
    TaskTemplate(
        task="voice-activity-detection",
        enum_member="VOICE_ACTIVITY_DETECTION",
        factory="AutoModelForVoiceActivityDetection",
        base_class="PreTrainedVADModel",
        output_class="VADOutput",
        implementation_method="_detect",
        public_method="detect",
        model_suffix="ForVoiceActivityDetection",
        training_family="AUDIO_CLASSIFICATION",
        input_name="audio",
        quickstart_input='"speech.wav"',
        sample_rate=16_000,
    ),
}
TRAINING_FAMILY_VALUES = {
    "ACOUSTIC": "acoustic-regression",
    "AUDIO_CLASSIFICATION": "audio-classification",
    "CTC": "ctc",
}


@dataclass(frozen=True, slots=True)
class BuiltinCatalogFragments:
    """Paste-ready declarations derived from one integration manifest."""

    model_spec: str
    aliases: str
    training_spec: str


def _display_relative_path(path: PurePath, root: PurePath) -> str:
    """Render one repository-relative path consistently on every platform."""
    return path.relative_to(root).as_posix()


def _nonempty(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ScaffoldError(f"{name} must be a non-empty string.")
    return value.strip()


def _model_type(value: str) -> str:
    normalized = _nonempty(value, name="model_type").lower()
    if MODEL_TYPE_PATTERN.fullmatch(normalized) is None:
        raise ScaffoldError(
            "model_type must start with a lowercase letter and contain only "
            "lowercase letters, digits, and underscores.")
    return normalized


def _class_prefix(value: str) -> str:
    normalized = _nonempty(value, name="class_prefix")
    if CLASS_PREFIX_PATTERN.fullmatch(normalized) is None:
        raise ScaffoldError(
            "class_prefix must be a single PascalCase Python identifier, "
            "for example AuroraTTS.")
    return normalized


def _revision(value: str, *, name: str) -> str:
    normalized = _nonempty(value, name=name)
    if normalized.lower() in MUTABLE_REVISIONS:
        raise ScaffoldError(f"{name} must be an immutable commit, tag, or release, not "
                            f"{normalized!r}.")
    if any(character.isspace() for character in normalized):
        raise ScaffoldError(f"{name} must not contain whitespace.")
    return normalized


def _source_url(value: str) -> str:
    normalized = _nonempty(value, name="source_url")
    parsed = urlsplit(normalized)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ScaffoldError("source_url must be an absolute HTTPS URL.")
    return normalized


def _aliases(values: tuple[str, ...], *, model_type: str) -> tuple[str, ...]:
    normalized = tuple(_nonempty(value, name="alias").lower() for value in values)
    if len(normalized) != len(set(normalized)):
        raise ScaffoldError("aliases must not contain duplicates.")
    if model_type in normalized:
        raise ScaffoldError("an alias must not equal model_type.")
    return normalized


def _task(value: str) -> TaskTemplate:
    try:
        return TASKS[value]
    except KeyError:
        raise ScaffoldError(f"task must be one of {', '.join(TASKS)}.") from None


def _render_init(model_type: str, config_class: str, model_class: str) -> str:
    return textwrap.dedent(
        f'''\
        """Lazy public exports for the {model_type} integration."""

        from __future__ import annotations

        import importlib
        from typing import Any

        _EXPORTS = {{
            {config_class!r}: __name__ + ".configuration_{model_type}",
            {model_class!r}: __name__ + ".modeling_{model_type}",
        }}

        __all__ = sorted(_EXPORTS)


        def __getattr__(name: str) -> Any:
            try:
                module_name = _EXPORTS[name]
            except KeyError:
                raise AttributeError(f"module {{__name__!r}} has no attribute {{name!r}}") from None
            value = getattr(importlib.import_module(module_name), name)
            globals()[name] = value
            return value


        def __dir__() -> list[str]:
            return sorted((*globals(), *_EXPORTS))
        ''')


def _render_config(model_type: str, config_class: str, sample_rate: int) -> str:
    return textwrap.dedent(
        f'''\
        """Configuration for the {model_type} integration."""

        from __future__ import annotations

        from typing import Any

        from voicehub import VoiceHubConfig


        class {config_class}(VoiceHubConfig):
            """Serializable {model_type} configuration."""

            model_type = {model_type!r}

            def __init__(self, *, sample_rate: int = {sample_rate}, **kwargs: Any) -> None:
                if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
                    raise ValueError("sample_rate must be a positive integer.")
                super().__init__(sample_rate=sample_rate, **kwargs)
        ''')


def _render_model(
    model_type: str,
    config_class: str,
    model_class: str,
    task: TaskTemplate,
) -> str:
    return textwrap.dedent(
        f'''\
        """Shared-contract wrapper for the {model_type} integration."""

        from __future__ import annotations

        from typing import Any

        from voicehub import {task.base_class}, {task.output_class}

        from .configuration_{model_type} import {config_class}

        IMPLEMENTATION_STATUS = {IMPLEMENTATION_STATUS!r}


        class {model_class}({task.base_class}):
            """TODO: replace the scaffold with the reviewed {model_type} runtime."""

            config_class = {config_class}

            def _load_pretrained_model(self) -> None:
                from .runtime import load_runtime

                self.model = load_runtime(
                    self.config.name_or_path,
                    device=self.device,
                )

            def {task.implementation_method}(
                self,
                {task.input_name}: Any,
                **kwargs: Any,
            ) -> {task.output_class}:
                del {task.input_name}, kwargs
                raise NotImplementedError(
                    "Implement {model_type} {task.public_method} and return "
                    "{task.output_class}."
                )
        ''')


def _render_runtime(model_type: str) -> str:
    return textwrap.dedent(
        f'''\
        """Checkpoint loading boundary for the {model_type} integration."""

        from __future__ import annotations

        from typing import Any


        def load_runtime(name_or_path: str, *, device: str) -> Any:
            del name_or_path, device
            raise NotImplementedError(
                "Implement immutable checkpoint resolution, validation, and "
                "runtime construction for {model_type}."
            )
        ''')


def _render_registration(
    model_type: str,
    checkpoint: str,
    aliases: tuple[str, ...],
    config_class: str,
    model_class: str,
    task: TaskTemplate,
) -> str:
    return textwrap.dedent(
        f'''\
        """Explicit public registration for the {model_type} integration."""

        from __future__ import annotations

        from voicehub import (
            {task.factory},
            ModelTrainingSpec,
            TrainingFamily,
            TrainingSupport,
            register_training_spec,
            unregister_training_spec,
        )

        from .configuration_{model_type} import {config_class}
        from .modeling_{model_type} import {model_class}

        _ALIASES = {aliases!r}


        def register_{model_type}():
            """Register inference and an explicit inference-only training boundary."""
            spec = {task.factory}.register(
                {config_class},
                {model_class},
                default_model_path={checkpoint!r},
                aliases=_ALIASES,
                capabilities=({task.task!r},),
                components=(),
            )
            try:
                register_training_spec(
                    ModelTrainingSpec(
                        model_type={model_type!r},
                        family=TrainingFamily.{task.training_family},
                        support=TrainingSupport.INFERENCE_ONLY,
                        task={task.task!r},
                    )
                )
            except BaseException:
                {task.factory}.unregister({model_type!r}, missing_ok=True)
                raise
            return spec


        def unregister_{model_type}():
            """Remove the integration from both public registries."""
            unregister_training_spec({model_type!r}, missing_ok=True)
            return {task.factory}.unregister({model_type!r}, missing_ok=True)
        ''')


def _render_test(
    model_type: str,
    checkpoint: str,
    config_class: str,
    model_class: str,
    task: TaskTemplate,
) -> str:
    return textwrap.dedent(
        f'''\
        """CPU-safe public contract tests for {model_type}."""

        from __future__ import annotations

        import unittest

        from voicehub import {task.factory}
        from voicehub.base_model import BaseSpeechModel
        from voicehub.models.{model_type}.configuration_{model_type} import {config_class}
        from voicehub.models.{model_type}.modeling_{model_type} import (
            IMPLEMENTATION_STATUS,
            {model_class},
        )
        from voicehub.models.{model_type}.registration import (
            register_{model_type},
            unregister_{model_type},
        )


        class {config_class}Tests(unittest.TestCase):

            def tearDown(self):
                unregister_{model_type}()

            def test_scaffold_completion_gate(self):
                self.assertEqual(
                    IMPLEMENTATION_STATUS,
                    "ready",
                    "Implement runtime loading, normalized inference, failure behavior, "
                    "serialization, real-checkpoint evidence, and optimization coverage "
                    "before marking this integration ready.",
                )

            def test_config_roundtrip_and_lazy_public_registration(self):
                config = {config_class}(sample_rate={task.sample_rate})
                self.assertEqual(
                    {config_class}.from_dict(config.to_dict()).to_dict(),
                    config.to_dict(),
                )

                spec = register_{model_type}()
                self.assertEqual(spec.model_type, {model_type!r})
                model = {task.factory}.from_pretrained(
                    {checkpoint!r},
                    model_type={model_type!r},
                    device="cpu",
                    lazy_load=True,
                )
                self.assertIsInstance(model, {model_class})
                self.assertFalse(model.is_loaded)
                self.assertIs(
                    {model_class}.apply_optimization_plan,
                    BaseSpeechModel.apply_optimization_plan,
                )


        if __name__ == "__main__":
            unittest.main()
        ''')


def _render_page(
    model_type: str,
    checkpoint: str,
    source_url: str,
    source_revision: str,
    license_id: str,
    config_class: str,
    model_class: str,
    task: TaskTemplate,
) -> str:
    return textwrap.dedent(
        f'''\
        ---
        description: Public contract and evidence boundary for the {model_type} integration.
        ---

        # `{model_type}`

        ## Overview

        `{model_type}` is a scaffolded **{task.task}** integration. Do not merge or
        publish it while `IMPLEMENTATION_STATUS` is not `ready`.

        ## Paper and GitHub

        - **Paper:** No dedicated upstream research paper is declared by this scaffold.
        - **Upstream GitHub:** [{source_url}]({source_url})
        - **VoiceHub source:** `voicehub/models/{model_type}/`

        ## Quickstart

        ```python
        from voicehub import {task.factory}

        model = {task.factory}.from_pretrained(
            {checkpoint!r},
            model_type={model_type!r},
            device="cpu",
            lazy_load=True,
        )
        output = model.{task.public_method}({task.quickstart_input})
        print(output)
        ```

        ## Supported tasks and capabilities

        | Property | Value |
        | --- | --- |
        | Task | `{task.task}` |
        | Normalized output | `{task.output_class}` |
        | Runtime status | Unverified scaffold |

        ## Checkpoints, provenance, and license

        | Property | Value |
        | --- | --- |
        | Default checkpoint | `{checkpoint}` |
        | Source | [{source_url}]({source_url}) |
        | Source revision | `{source_revision}` |
        | License | `{license_id}`; review `source/THIRD_PARTY_LICENSE` |

        Replace this scaffold boundary with exact checkpoint revisions, verified scope,
        hardware requirements, and inaccessible or hardware-limited paths.

        ## Optimization and training support

        Training is inference-only until a differentiable path is tested. The wrapper
        inherits VoiceHub's model-independent optimization lifecycle, but no pass may be
        reported as supported until application, validation, manifest reporting, and
        restoration are covered on the implemented runtime.

        ## Public API

        - `{config_class}`
        - `{model_class}`
        - `{task.factory}`
        ''')


def _manifest(
    model_type: str,
    class_prefix: str,
    checkpoint: str,
    aliases: tuple[str, ...],
    source_url: str,
    source_revision: str,
    license_id: str,
    task: TaskTemplate,
) -> str:
    payload = {
        "aliases": list(aliases),
        "architecture": None,
        "builtin": False,
        "capabilities": [task.task],
        "class_prefix": class_prefix,
        "components": [],
        "default_checkpoint": checkpoint,
        "default_for_task": False,
        "format_version": 1,
        "install_extra": None,
        "license": {
            "file": "source/THIRD_PARTY_LICENSE",
            "id": license_id,
        },
        "model_type": model_type,
        "model_page_sections": list(MODEL_PAGE_SECTIONS),
        "source": {
            "revision": source_revision,
            "url": source_url,
        },
        "task": task.task,
        "training": {
            "family": TRAINING_FAMILY_VALUES[task.training_family],
            "support": "inference-only",
        },
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _source_manifest(
    model_type: str,
    checkpoint: str,
    source_url: str,
    source_revision: str,
    license_id: str,
) -> str:
    payload = {
        "checkpoint": {
            "license": license_id,
            "repository": checkpoint,
            "revision": "REPLACE_WITH_IMMUTABLE_CHECKPOINT_REVISION",
        },
        "license": license_id,
        "model_type": model_type,
        "revision": source_revision,
        "upstream": source_url,
        "verified_scope": {
            "inference": [],
            "limitations": ["Replace with the exact unverified or hardware-limited boundary."],
            "training": [],
        },
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def scaffold_files(
        *,
        model_type: str,
        class_prefix: str,
        task: str,
        checkpoint: str,
        source_url: str,
        source_revision: str,
        license_id: str,
        license_text: str,
        aliases: tuple[str, ...] = (),
) -> dict[Path, str]:
    """Render every required scaffold file without touching the filesystem."""
    normalized_model_type = _model_type(model_type)
    normalized_prefix = _class_prefix(class_prefix)
    task_template = _task(task)
    normalized_checkpoint = _nonempty(checkpoint, name="checkpoint")
    normalized_source_url = _source_url(source_url)
    normalized_revision = _revision(source_revision, name="source_revision")
    normalized_license = _nonempty(license_id, name="license_id")
    normalized_license_text = _nonempty(license_text, name="license_text") + "\n"
    normalized_aliases = _aliases(aliases, model_type=normalized_model_type)
    config_class = normalized_prefix + "Config"
    model_class = normalized_prefix + task_template.model_suffix
    package = Path("voicehub") / "models" / normalized_model_type

    return {
        package / "__init__.py":
        _render_init(
            normalized_model_type,
            config_class,
            model_class,
        ),
        package / f"configuration_{normalized_model_type}.py":
        _render_config(
            normalized_model_type,
            config_class,
            task_template.sample_rate,
        ),
        package / f"modeling_{normalized_model_type}.py":
        _render_model(
            normalized_model_type,
            config_class,
            model_class,
            task_template,
        ),
        package / "runtime.py":
        _render_runtime(normalized_model_type),
        package / "registration.py":
        _render_registration(
            normalized_model_type,
            normalized_checkpoint,
            normalized_aliases,
            config_class,
            model_class,
            task_template,
        ),
        package / "model-integration.json":
        _manifest(
            normalized_model_type,
            normalized_prefix,
            normalized_checkpoint,
            normalized_aliases,
            normalized_source_url,
            normalized_revision,
            normalized_license,
            task_template,
        ),
        package / "source" / "SOURCE.json":
        _source_manifest(
            normalized_model_type,
            normalized_checkpoint,
            normalized_source_url,
            normalized_revision,
            normalized_license,
        ),
        package / "source" / "THIRD_PARTY_LICENSE":
        normalized_license_text,
        Path("tests") / f"test_{normalized_model_type}.py":
        _render_test(
            normalized_model_type,
            normalized_checkpoint,
            config_class,
            model_class,
            task_template,
        ),
        Path("docs") / "models" / "providers" / f"{normalized_model_type}.md":
        _render_page(
            normalized_model_type,
            normalized_checkpoint,
            normalized_source_url,
            normalized_revision,
            normalized_license,
            config_class,
            model_class,
            task_template,
        ),
    }


def create_model_scaffold(output_root: Path, files: dict[Path, str]) -> tuple[Path, ...]:
    """Write a scaffold transactionally with respect to existing files."""
    root = Path(output_root).expanduser().resolve()
    destinations = tuple(root / relative_path for relative_path in files)
    existing = tuple(path for path in destinations if path.exists())
    if existing:
        rendered = ", ".join(_display_relative_path(path, root) for path in existing)
        raise ScaffoldError("Refusing to overwrite existing scaffold paths: " + rendered)
    for destination in destinations:
        destination.parent.mkdir(parents=True, exist_ok=True)
    for relative_path, content in files.items():
        (root / relative_path).write_text(content, encoding="utf-8", newline="\n")
    return destinations


def render_builtin_catalog_fragments(
    output_root: Path,
    model_type: str,
) -> BuiltinCatalogFragments:
    """Render legacy central declarations from ``model-integration.json``.

    The renderer only reads the manifest. It does not import VoiceHub or
    edit either catalog, so contributors can review and paste each
    declaration at its named insertion point.
    """
    root = Path(output_root).expanduser().resolve()
    normalized_model_type = _model_type(model_type)
    manifest_path = (root / "voicehub" / "models" / normalized_model_type / "model-integration.json")
    if not manifest_path.is_file():
        raise ScaffoldError(
            f"{normalized_model_type}: missing {_display_relative_path(manifest_path, root)}; "
            "run scaffold_model.py create first.")
    try:
        manifest = _parse_strict_json(
            manifest_path.read_bytes(),
            source=manifest_path,
        )
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise ScaffoldError(f"{normalized_model_type}: invalid model-integration.json: {error}.") from error
    if not isinstance(manifest, dict):
        raise ScaffoldError(f"{normalized_model_type}: model-integration.json must contain an object.")
    if manifest.get("model_type") != normalized_model_type:
        raise ScaffoldError(
            f"{normalized_model_type}: model-integration.json model_type must match its directory.")

    class_prefix = _class_prefix(manifest.get("class_prefix"))
    checkpoint = _nonempty(
        manifest.get("default_checkpoint"),
        name="model-integration.json default_checkpoint",
    )
    task_value = manifest.get("task")
    task = next((item for item in TASKS.values() if item.task == task_value), None)
    if task is None:
        supported = ", ".join(item.task for item in TASKS.values())
        raise ScaffoldError(f"{normalized_model_type}: task must be one of {supported}.")
    aliases_value = manifest.get("aliases")
    if not isinstance(aliases_value, list):
        raise ScaffoldError(
            f"{normalized_model_type}: model-integration.json aliases must be a list of strings.")
    aliases = _aliases(tuple(aliases_value), model_type=normalized_model_type)

    config_class = class_prefix + "Config"
    model_class = class_prefix + task.model_suffix
    model_spec = textwrap.dedent(
        f'''\
        ModelSpec(
            {normalized_model_type!r},
            {f"voicehub.models.{normalized_model_type}.modeling_{normalized_model_type}"!r},
            {model_class!r},
            {checkpoint!r},
            capabilities={(task.task,)!r},
            config_module={f"voicehub.models.{normalized_model_type}.configuration_{normalized_model_type}"!r},
            config_class={config_class!r},
            task=SpeechTask.{task.enum_member},
        ),
        ''')
    alias_entries = "".join(f"    {alias!r}: {normalized_model_type!r},\n" for alias in aliases)
    training_spec = textwrap.dedent(
        f'''\
        _profile(
            {normalized_model_type!r},
            TrainingFamily.{task.training_family},
            task=SpeechTask.{task.enum_member},
            support=TrainingSupport.INFERENCE_ONLY,
        ),
        ''')
    return BuiltinCatalogFragments(
        model_spec=model_spec,
        aliases=alias_entries,
        training_spec=training_spec,
    )


def _string_assignment(tree: ast.Module, name: str) -> str | None:
    for statement in tree.body:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        targets = statement.targets if isinstance(statement, ast.Assign) else (statement.target, )
        if not any(isinstance(target, ast.Name) and target.id == name for target in targets):
            continue
        try:
            value = ast.literal_eval(statement.value)
        except (TypeError, ValueError):
            return None
        return value if isinstance(value, str) else None
    return None


_UNRESOLVED = object()
_MODEL_SPEC_POSITIONS = {
    "model_type": 0,
    "module": 1,
    "class_name": 2,
    "default_model_path": 3,
    "config_module": 6,
    "config_class": 7,
    "task": 8,
}
_TASK_BY_ENUM_MEMBER = {item.enum_member: item.task for item in TASKS.values()}
_TASK_BY_LITERAL = {
    "tts": TASKS["tts"].task,
    "text-to-speech": TASKS["tts"].task,
    "asr": TASKS["asr"].task,
    "stt": TASKS["asr"].task,
    "speech-to-text": TASKS["asr"].task,
    "automatic-speech-recognition": TASKS["asr"].task,
    "vad": TASKS["vad"].task,
    "voice-activity-detection": TASKS["vad"].task,
}


def _callable_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _call_argument(
    call: ast.Call,
    name: str,
    *,
    position: int | None,
) -> ast.expr | None:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    if position is not None and position < len(call.args):
        return call.args[position]
    return None


def _literal_value(node: ast.expr | None):
    if node is None:
        return _UNRESOLVED
    try:
        return ast.literal_eval(node)
    except (TypeError, ValueError):
        return _UNRESOLVED


def _assignment_value(tree: ast.Module, name: str):
    for statement in tree.body:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        targets = statement.targets if isinstance(statement, ast.Assign) else (statement.target, )
        if not any(isinstance(target, ast.Name) and target.id == name for target in targets):
            continue
        return _literal_value(statement.value)
    return _UNRESOLVED


def _task_value(
    node: ast.expr | None,
    *,
    default: str,
) -> str | None:
    if node is None:
        return default
    literal = _literal_value(node)
    if isinstance(literal, str):
        return _TASK_BY_LITERAL.get(literal.strip().lower())
    if isinstance(node, ast.Attribute):
        return _TASK_BY_ENUM_MEMBER.get(node.attr)
    return None


def _model_calls(
    tree: ast.Module,
    *,
    call_names: tuple[str, ...],
    model_type: str,
) -> tuple[ast.Call, ...]:
    matches = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _callable_name(node.func) not in call_names:
            continue
        model_node = _call_argument(node, "model_type", position=0)
        if _literal_value(model_node) == model_type:
            matches.append(node)
    return tuple(matches)


def _render_static_value(value: object) -> str:
    return "a non-literal expression" if value is _UNRESOLVED else repr(value)


def _validate_builtin_model_spec(
    tree: ast.Module,
    *,
    model_type: str,
    config_class: str,
    model_class: str,
    checkpoint: str,
    task: TaskTemplate,
) -> list[str]:
    errors = []
    calls = _model_calls(
        tree,
        call_names=("ModelSpec", ),
        model_type=model_type,
    )
    if not calls:
        return [
            f"{model_type}: built-in registry discovery is missing; add one "
            "ModelSpec or keep the integration in a separately imported extension package."
        ]
    if len(calls) > 1:
        errors.append(
            f"{model_type}: built-in registry declares {len(calls)} ModelSpec entries; "
            "keep exactly one canonical declaration.")
    call = calls[0]
    expected_fields = {
        "module": f"voicehub.models.{model_type}.modeling_{model_type}",
        "class_name": model_class,
        "default_model_path": checkpoint,
        "config_module": f"voicehub.models.{model_type}.configuration_{model_type}",
        "config_class": config_class,
    }
    for field_name, expected in expected_fields.items():
        node = _call_argument(
            call,
            field_name,
            position=_MODEL_SPEC_POSITIONS[field_name],
        )
        actual = _literal_value(node)
        if actual != expected:
            errors.append(
                f"{model_type}: built-in ModelSpec {field_name} must be {expected!r}; "
                f"found {_render_static_value(actual)}.")
    task_node = _call_argument(
        call,
        "task",
        position=_MODEL_SPEC_POSITIONS["task"],
    )
    actual_task = _task_value(
        task_node,
        default=TASKS["tts"].task,
    )
    if actual_task != task.task:
        rendered = "an unsupported expression" if actual_task is None else repr(actual_task)
        errors.append(f"{model_type}: built-in ModelSpec task must be {task.task!r}; found {rendered}.")
    return errors


def _validate_builtin_aliases(
    tree: ast.Module,
    *,
    model_type: str,
    aliases: tuple[str, ...],
) -> list[str]:
    errors = []
    mapping = _assignment_value(tree, "_BUILTIN_MODEL_ALIASES")
    if not isinstance(mapping, dict):
        if aliases:
            errors.append(
                f"{model_type}: _BUILTIN_MODEL_ALIASES must be a literal mapping "
                "containing every manifest alias.")
        mapping = {}
    for alias in aliases:
        target = mapping.get(alias)
        if target != model_type:
            errors.append(
                f"{model_type}: built-in alias {alias!r} must target {model_type!r}; "
                f"found {target!r}.")
    undeclared = sorted(
        alias for alias, target in mapping.items() if target == model_type and alias not in aliases)
    if undeclared:
        errors.append(
            f"{model_type}: built-in aliases {undeclared!r} are missing from "
            "model-integration.json.")
    return errors


def _validate_builtin_training_spec(
    tree: ast.Module,
    *,
    model_type: str,
    task: TaskTemplate,
) -> list[str]:
    calls = _model_calls(
        tree,
        call_names=("_profile", "ModelTrainingSpec"),
        model_type=model_type,
    )
    if not calls:
        return [
            f"{model_type}: built-in training profile is missing; add one "
            "_profile(...) or ModelTrainingSpec(...) entry to voicehub/training/specs.py."
        ]
    errors = []
    if len(calls) > 1:
        errors.append(
            f"{model_type}: built-in training registry declares {len(calls)} profiles; "
            "keep exactly one canonical declaration.")
    call = calls[0]
    position = 20 if _callable_name(call.func) == "ModelTrainingSpec" else None
    actual_task = _task_value(
        _call_argument(call, "task", position=position),
        default=TASKS["tts"].task,
    )
    if actual_task != task.task:
        rendered = "an unsupported expression" if actual_task is None else repr(actual_task)
        errors.append(
            f"{model_type}: built-in training profile task must be {task.task!r}; "
            f"found {rendered}.")
    return errors


def check_model_scaffold(output_root: Path, model_type: str) -> tuple[str, ...]:
    """Return actionable structural omissions for one scaffolded model."""
    root = Path(output_root).expanduser().resolve()
    normalized_model_type = _model_type(model_type)
    package = root / "voicehub" / "models" / normalized_model_type
    manifest_path = package / "model-integration.json"
    errors: list[str] = []

    if not manifest_path.is_file():
        return (
            f"{normalized_model_type}: missing {_display_relative_path(manifest_path, root)}; "
            "run scaffold_model.py create first.", )
    try:
        manifest = _parse_strict_json(
            manifest_path.read_bytes(),
            source=manifest_path,
        )
    except (OSError, UnicodeDecodeError, ValueError) as error:
        return (f"{normalized_model_type}: invalid model-integration.json: {error}.", )
    if not isinstance(manifest, dict):
        return (f"{normalized_model_type}: model-integration.json must contain an object.", )

    class_prefix = manifest.get("class_prefix")
    task_value = manifest.get("task")
    task_key = next((name for name, item in TASKS.items() if item.task == task_value), None)
    if manifest.get("model_type") != normalized_model_type:
        errors.append(f"{normalized_model_type}: model-integration.json model_type must match its directory.")
    try:
        normalized_prefix = _class_prefix(class_prefix)
    except ScaffoldError as error:
        errors.append(f"{normalized_model_type}: {error}")
        normalized_prefix = "Invalid"
    if task_key is None:
        errors.append(
            f"{normalized_model_type}: task must be one of "
            f"{', '.join(item.task for item in TASKS.values())}.")
        task_template = TASKS["tts"]
    else:
        task_template = TASKS[task_key]
    builtin_value = manifest.get("builtin", False)
    if not isinstance(builtin_value, bool):
        errors.append(f"{normalized_model_type}: model-integration.json builtin must be a boolean.")
        manifest_builtin = False
    else:
        manifest_builtin = builtin_value
    capabilities = manifest.get("capabilities")
    if (not isinstance(capabilities, list) or
            any(not isinstance(value, str) or not value.strip() for value in capabilities)):
        errors.append(
            f"{normalized_model_type}: model-integration.json capabilities must be a list of strings.")
    elif task_template.task not in capabilities:
        errors.append(
            f"{normalized_model_type}: model-integration.json capabilities must include "
            f"{task_template.task!r}.")
    components = manifest.get("components")
    if (not isinstance(components, list) or
            any(not isinstance(value, str) or not value.strip() for value in components)):
        errors.append(
            f"{normalized_model_type}: model-integration.json components must be a list of strings.")
    training = manifest.get("training")
    if not isinstance(training, dict):
        errors.append(f"{normalized_model_type}: model-integration.json training must be an object.")
    else:
        family = training.get("family")
        support = training.get("support")
        if not isinstance(family, str) or not family.strip():
            errors.append(
                f"{normalized_model_type}: model-integration.json training.family must be "
                "a non-empty string.")
        if support != "inference-only":
            errors.append(
                f"{normalized_model_type}: manifest discovery requires training.support "
                "to be 'inference-only'; register richer training metadata explicitly.")
    aliases_value = manifest.get("aliases")
    try:
        if not isinstance(aliases_value, list):
            raise ScaffoldError("aliases must be a list of strings.")
        manifest_aliases = _aliases(
            tuple(aliases_value),
            model_type=normalized_model_type,
        )
    except (ScaffoldError, TypeError) as error:
        errors.append(f"{normalized_model_type}: model-integration.json {error}")
        manifest_aliases = ()
    checkpoint_value = manifest.get("default_checkpoint")
    try:
        default_checkpoint = _nonempty(
            checkpoint_value,
            name="model-integration.json default_checkpoint",
        )
    except ScaffoldError as error:
        errors.append(f"{normalized_model_type}: {error}")
        default_checkpoint = ""

    config_class = normalized_prefix + "Config"
    model_class = normalized_prefix + task_template.model_suffix
    required = (
        package / "__init__.py",
        package / f"configuration_{normalized_model_type}.py",
        package / f"modeling_{normalized_model_type}.py",
        package / "runtime.py",
        package / "registration.py",
        package / "source" / "SOURCE.json",
        package / "source" / "THIRD_PARTY_LICENSE",
        root / "tests" / f"test_{normalized_model_type}.py",
        root / "docs" / "models" / "providers" / f"{normalized_model_type}.md",
    )
    for path in required:
        if not path.is_file():
            errors.append(
                f"{normalized_model_type}: missing {_display_relative_path(path, root)}; "
                "restore the scaffold artifact and complete it.")

    python_paths = tuple(path for path in required if path.suffix == ".py" and path.is_file())
    parsed: dict[Path, ast.Module] = {}
    for path in python_paths:
        try:
            parsed[path] = ast.parse(
                path.read_text(encoding="utf-8"),
                filename=str(path),
            )
        except SyntaxError as error:
            errors.append(
                f"{normalized_model_type}: {_display_relative_path(path, root)} "
                f"does not compile: {error}.")

    config_path = package / f"configuration_{normalized_model_type}.py"
    if config_path in parsed:
        class_names = {node.name for node in parsed[config_path].body if isinstance(node, ast.ClassDef)}
        if config_class not in class_names:
            errors.append(
                f"{normalized_model_type}: {_display_relative_path(config_path, root)} must define "
                f"{config_class}.")

    model_path = package / f"modeling_{normalized_model_type}.py"
    if model_path in parsed:
        class_names = {node.name for node in parsed[model_path].body if isinstance(node, ast.ClassDef)}
        if model_class not in class_names:
            errors.append(
                f"{normalized_model_type}: {_display_relative_path(model_path, root)} must define "
                f"{model_class}.")
        status = _string_assignment(parsed[model_path], "IMPLEMENTATION_STATUS")
        if status != READY_STATUS:
            errors.append(
                f"{normalized_model_type}: IMPLEMENTATION_STATUS is {status!r}; set it to "
                f"{READY_STATUS!r} only after runtime, normalized output, failure, "
                "serialization, checkpoint, and optimization tests pass.")

    registration_path = package / "registration.py"
    if registration_path.is_file():
        registration = registration_path.read_text(encoding="utf-8")
        for fragment in (
                f"{task_template.factory}.register(",
                "register_training_spec(",
                f"def register_{normalized_model_type}(",
                f"def unregister_{normalized_model_type}(",
        ):
            if fragment not in registration:
                errors.append(f"{normalized_model_type}: registration.py is missing {fragment!r}.")

    source_path = package / "source" / "SOURCE.json"
    if source_path.is_file():
        try:
            source = _parse_strict_json(
                source_path.read_bytes(),
                source=source_path,
            )
        except (OSError, UnicodeDecodeError, ValueError) as error:
            errors.append(f"{normalized_model_type}: SOURCE.json is invalid: {error}.")
        else:
            if not isinstance(source, dict):
                errors.append(f"{normalized_model_type}: SOURCE.json must contain an object.")
            else:
                for field in ("model_type", "upstream", "revision", "license", "checkpoint",
                              "verified_scope"):
                    if field not in source:
                        errors.append(f"{normalized_model_type}: SOURCE.json is missing {field!r}.")
                revision = source.get("revision")
                try:
                    _revision(revision, name="SOURCE.json revision")
                except ScaffoldError as error:
                    errors.append(f"{normalized_model_type}: {error}")
                source_checkpoint = source.get("checkpoint")
                if (not isinstance(source_checkpoint, dict) or not source_checkpoint.get("revision") or
                        str(source_checkpoint.get("revision")).startswith("REPLACE_")):
                    errors.append(
                        f"{normalized_model_type}: SOURCE.json checkpoint.revision must be "
                        "replaced with an immutable revision.")

    license_path = package / "source" / "THIRD_PARTY_LICENSE"
    if license_path.is_file() and not license_path.read_text(encoding="utf-8").strip():
        errors.append(
            f"{normalized_model_type}: THIRD_PARTY_LICENSE must contain the authoritative "
            "license text.")

    test_path = root / "tests" / f"test_{normalized_model_type}.py"
    if test_path.is_file():
        test_source = test_path.read_text(encoding="utf-8")
        for fragment in (
                "test_scaffold_completion_gate",
                "from_dict(config.to_dict())",
                "lazy_load=True",
                "apply_optimization_plan",
        ):
            if fragment not in test_source:
                errors.append(
                    f"{normalized_model_type}: {_display_relative_path(test_path, root)} is missing "
                    f"contract coverage marker {fragment!r}.")

    page_path = root / "docs" / "models" / "providers" / f"{normalized_model_type}.md"
    if page_path.is_file():
        page = page_path.read_text(encoding="utf-8")
        if normalized_model_type not in page:
            errors.append(f"{normalized_model_type}: provider page does not identify the model type.")
        headings = tuple(f"## {section}" for section in MODEL_PAGE_SECTIONS)
        positions = tuple(page.find(heading) for heading in headings)
        if any(position < 0 for position in positions) or positions != tuple(sorted(positions)):
            errors.append(
                f"{normalized_model_type}: provider page must use the common section order: " +
                ", ".join(MODEL_PAGE_SECTIONS) + ".")

    navigation_path = f"models/providers/{normalized_model_type}.md"
    site_config = root / "mkdocs.yml"
    if not site_config.is_file() or navigation_path not in site_config.read_text(encoding="utf-8"):
        errors.append(
            f"{normalized_model_type}: mkdocs.yml is missing {navigation_path!r}; "
            "register the built-in model, run scripts/generate_model_pages.py, and rerun --check.")

    registry_path = root / "voicehub" / "models" / "registry.py"
    training_path = root / "voicehub" / "training" / "specs.py"
    uses_builtin_catalogs = registry_path.is_file() or training_path.is_file()
    if uses_builtin_catalogs:
        registry_tree = None
        if not registry_path.is_file():
            errors.append(
                f"{normalized_model_type}: missing {_display_relative_path(registry_path, root)}; "
                "restore the shared built-in registry implementation.")
        else:
            try:
                registry_tree = ast.parse(
                    registry_path.read_text(encoding="utf-8"),
                    filename=str(registry_path),
                )
            except SyntaxError as error:
                errors.append(
                    f"{normalized_model_type}: {_display_relative_path(registry_path, root)} does not "
                    f"compile: {error}.")
        if registry_tree is not None:
            if manifest_builtin:
                central_specs = _model_calls(
                    registry_tree,
                    call_names=("ModelSpec", ),
                    model_type=normalized_model_type,
                )
                if central_specs:
                    errors.append(
                        f"{normalized_model_type}: activated manifest discovery duplicates "
                        "a central ModelSpec; remove the legacy catalog entry.")
                alias_mapping = _assignment_value(registry_tree, "_BUILTIN_MODEL_ALIASES")
                central_aliases = [] if not isinstance(alias_mapping, dict) else sorted(
                    alias for alias, target in alias_mapping.items() if target == normalized_model_type)
                if central_aliases:
                    errors.append(
                        f"{normalized_model_type}: activated manifest discovery duplicates "
                        f"central aliases {central_aliases!r}; remove the legacy entries.")
            else:
                errors.extend(
                    _validate_builtin_model_spec(
                        registry_tree,
                        model_type=normalized_model_type,
                        config_class=config_class,
                        model_class=model_class,
                        checkpoint=default_checkpoint,
                        task=task_template,
                    ))
                errors.extend(
                    _validate_builtin_aliases(
                        registry_tree,
                        model_type=normalized_model_type,
                        aliases=manifest_aliases,
                    ))

        training_tree = None
        if not training_path.is_file():
            errors.append(
                f"{normalized_model_type}: missing {_display_relative_path(training_path, root)}; "
                "restore the shared training registry implementation.")
        else:
            try:
                training_tree = ast.parse(
                    training_path.read_text(encoding="utf-8"),
                    filename=str(training_path),
                )
            except SyntaxError as error:
                errors.append(
                    f"{normalized_model_type}: {_display_relative_path(training_path, root)} does not "
                    f"compile: {error}.")
        if training_tree is not None:
            if manifest_builtin:
                central_profiles = _model_calls(
                    training_tree,
                    call_names=("_profile", "ModelTrainingSpec"),
                    model_type=normalized_model_type,
                )
                if central_profiles:
                    errors.append(
                        f"{normalized_model_type}: activated manifest discovery duplicates "
                        "a central training profile; remove the legacy catalog entry.")
            else:
                errors.extend(
                    _validate_builtin_training_spec(
                        training_tree,
                        model_type=normalized_model_type,
                        task=task_template,
                    ))
    elif manifest_builtin:
        errors.append(
            f"{normalized_model_type}: builtin manifest discovery requires the shared "
            "VoiceHub registry and training catalog modules.")

    return tuple(errors)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create", help="Create a new incomplete scaffold.")
    create.add_argument("--model-type", required=True)
    create.add_argument("--class-prefix", required=True)
    create.add_argument("--task", choices=tuple(TASKS), required=True)
    create.add_argument("--checkpoint", required=True)
    create.add_argument("--source-url", required=True)
    create.add_argument("--source-revision", required=True)
    create.add_argument("--license-id", required=True)
    create.add_argument("--license-file", type=Path, required=True)
    create.add_argument("--alias", action="append", default=[])
    create.add_argument("--output-root", type=Path, default=REPOSITORY_ROOT)

    check = subparsers.add_parser("check", help="Report incomplete scaffold contracts.")
    check.add_argument("--model-type", required=True)
    check.add_argument("--output-root", type=Path, default=REPOSITORY_ROOT)

    catalog = subparsers.add_parser(
        "catalog",
        help="Render legacy central catalog declarations without editing files.",
    )
    catalog.add_argument("--model-type", required=True)
    catalog.add_argument("--output-root", type=Path, default=REPOSITORY_ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "create":
            try:
                license_text = args.license_file.read_text(encoding="utf-8")
            except OSError as error:
                raise ScaffoldError(
                    f"Could not read authoritative license file {args.license_file}: {error}") from error
            files = scaffold_files(
                model_type=args.model_type,
                class_prefix=args.class_prefix,
                task=args.task,
                checkpoint=args.checkpoint,
                source_url=args.source_url,
                source_revision=args.source_revision,
                license_id=args.license_id,
                license_text=license_text,
                aliases=tuple(args.alias),
            )
            paths = create_model_scaffold(args.output_root, files)
            for path in paths:
                print(f"created: {_display_relative_path(path, args.output_root.resolve())}")
            print(
                "INCOMPLETE: implement the runtime, pin the checkpoint revision, "
                "register the built-in ModelSpec, aliases, and training profile, "
                "generate navigation, and run --check.")
            return 0

        if args.command == "catalog":
            fragments = render_builtin_catalog_fragments(
                args.output_root,
                args.model_type,
            )
            sections = (
                (
                    "voicehub/models/registry.py :: _MODEL_SPECS",
                    fragments.model_spec,
                ),
                (
                    "voicehub/models/registry.py :: _BUILTIN_MODEL_ALIASES",
                    fragments.aliases or "    # No aliases declared.\n",
                ),
                (
                    "voicehub/training/specs.py :: _BUILTIN_TRAINING_SPECS",
                    fragments.training_spec,
                ),
            )
            for index, (target, source) in enumerate(sections):
                if index:
                    print()
                print(f"# {target}")
                print(source, end="" if source.endswith("\n") else "\n")
            return 0

        errors = check_model_scaffold(args.output_root, args.model_type)
        if errors:
            for error in errors:
                print(f"ERROR: {error}", file=sys.stderr)
            return 1
        print(f"OK: {args.model_type} scaffold structure is complete")
        return 0
    except ScaffoldError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
