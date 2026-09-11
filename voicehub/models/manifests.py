"""Source-only discovery for completed built-in model integrations."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from voicehub.json_utils import parse_json_value
from voicehub.tasks import SpeechTask

_CLASS_PREFIX_PATTERN = re.compile(r"[A-Z][A-Za-z0-9]*")
_MODEL_TYPE_PATTERN = re.compile(r"[a-z][a-z0-9_]*")
_SUPPORTED_FORMAT_VERSION = 1


@dataclass(frozen=True, slots=True)
class BuiltinModelManifest:
    """Validated registry metadata from one activated integration manifest."""

    source_path: Path
    model_type: str
    class_prefix: str
    default_checkpoint: str
    aliases: tuple[str, ...]
    task: SpeechTask
    capabilities: tuple[str, ...]
    architecture: str | None
    components: tuple[str, ...]
    install_extra: str | None
    default_for_task: bool
    training_family: str
    training_support: str

    @property
    def config_class(self) -> str:
        return self.class_prefix + "Config"

    @property
    def model_class(self) -> str:
        suffixes = {
            SpeechTask.TEXT_TO_SPEECH: "ForTextToSpeech",
            SpeechTask.AUTOMATIC_SPEECH_RECOGNITION: "ForSpeechRecognition",
            SpeechTask.VOICE_ACTIVITY_DETECTION: "ForVoiceActivityDetection",
            SpeechTask.AUDIO_CODEC: "ForAudioCodec",
        }
        return self.class_prefix + suffixes[self.task]


def _nonempty_string(value: Any, *, field: str, source: Path) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{source}: {field} must be a non-empty string.")
    return value.strip()


def _string_list(value: Any, *, field: str, source: Path) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{source}: {field} must be a list of strings.")
    normalized = tuple(_nonempty_string(item, field=field, source=source).lower() for item in value)
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{source}: {field} must not contain duplicates.")
    return normalized


def _literal_assignment(path: Path, name: str) -> Any:
    try:
        tree = ast.parse(
            path.read_text(encoding="utf-8"),
            filename=str(path),
        )
    except (OSError, SyntaxError) as error:
        raise ValueError(f"{path}: activated model source is not inspectable: {error}.") from error
    for statement in tree.body:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        targets = statement.targets if isinstance(statement, ast.Assign) else (statement.target, )
        if not any(isinstance(target, ast.Name) and target.id == name for target in targets):
            continue
        try:
            return ast.literal_eval(statement.value)
        except (TypeError, ValueError):
            return None
    return None


def _validate_activation_artifacts(path: Path, model_type: str) -> None:
    package = path.parent
    required = (
        package / "__init__.py",
        package / "configuration.py",
        package / "modeling.py",
        package / "registration.py",
        package / "runtime.py",
        package / "source" / "SOURCE.json",
        package / "source" / "THIRD_PARTY_LICENSE",
    )
    missing = [item.relative_to(package).as_posix() for item in required if not item.is_file()]
    if missing:
        raise ValueError(f"{path}: activated built-in is missing required package artifacts: {missing!r}.")

    modeling_path = package / "modeling.py"
    status = _literal_assignment(modeling_path, "IMPLEMENTATION_STATUS")
    if status != "ready":
        raise ValueError(
            f"{path}: activated built-in requires IMPLEMENTATION_STATUS = 'ready'; found {status!r}.")

    source_path = package / "source" / "SOURCE.json"
    try:
        source = parse_json_value(
            source_path.read_bytes(),
            source=source_path,
        )
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise ValueError(f"{source_path}: activated source manifest is invalid: {error}.") from error
    checkpoint = source.get("checkpoint") if isinstance(source, dict) else None
    revision = checkpoint.get("revision") if isinstance(checkpoint, dict) else None
    if (not isinstance(revision, str) or not revision.strip() or revision.startswith("REPLACE_")):
        raise ValueError(f"{source_path}: activated built-in requires an immutable checkpoint revision.")

    license_path = package / "source" / "THIRD_PARTY_LICENSE"
    try:
        license_text = license_path.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"{license_path}: activated license is not readable: {error}.") from error
    if not license_text.strip():
        raise ValueError(f"{license_path}: activated built-in requires bundled license text.")


def _parse_active_manifest(path: Path) -> BuiltinModelManifest | None:
    """Parse one manifest only after its explicit built-in activation flag."""
    try:
        document = path.read_bytes()
        payload = json.loads(document)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        # Work-in-progress scaffolds must not break normal VoiceHub imports.
        # The scaffold checker reports their precise syntax or I/O error.
        return None
    if not isinstance(payload, dict) or payload.get("builtin") is not True:
        return None
    try:
        payload = parse_json_value(document, source=path)
    except (UnicodeDecodeError, ValueError) as error:
        raise ValueError(f"{path}: activated model manifest is invalid: {error}.") from error
    if payload.get("format_version") != _SUPPORTED_FORMAT_VERSION:
        raise ValueError(
            f"{path}: format_version must be {_SUPPORTED_FORMAT_VERSION} for built-in discovery.")

    model_type = _nonempty_string(payload.get("model_type"), field="model_type", source=path).lower()
    if _MODEL_TYPE_PATTERN.fullmatch(model_type) is None:
        raise ValueError(f"{path}: model_type must use lowercase letters, digits, and underscores.")
    if path.parent.name != model_type:
        raise ValueError(f"{path}: model_type must match its package directory {path.parent.name!r}.")

    class_prefix = _nonempty_string(payload.get("class_prefix"), field="class_prefix", source=path)
    if _CLASS_PREFIX_PATTERN.fullmatch(class_prefix) is None:
        raise ValueError(f"{path}: class_prefix must be one PascalCase identifier.")
    checkpoint = _nonempty_string(
        payload.get("default_checkpoint"),
        field="default_checkpoint",
        source=path,
    )
    aliases = _string_list(payload.get("aliases"), field="aliases", source=path)
    if model_type in aliases:
        raise ValueError(f"{path}: an alias must not equal model_type.")

    task_value = _nonempty_string(payload.get("task"), field="task", source=path)
    task = SpeechTask.coerce(task_value)
    if task_value != task.value:
        raise ValueError(f"{path}: task must use canonical value {task.value!r}.")
    capabilities = _string_list(payload.get("capabilities"), field="capabilities", source=path)
    if task.value not in capabilities:
        raise ValueError(f"{path}: capabilities must include the registered task {task.value!r}.")

    architecture_value = payload.get("architecture")
    architecture = None
    if architecture_value is not None:
        architecture = _nonempty_string(
            architecture_value,
            field="architecture",
            source=path,
        ).lower()
    if "voicehub-native" in capabilities and architecture is None:
        raise ValueError(f"{path}: voicehub-native capability requires an architecture identifier.")
    components = _string_list(payload.get("components"), field="components", source=path)

    install_extra_value = payload.get("install_extra")
    install_extra = None
    if install_extra_value is not None:
        install_extra = _nonempty_string(
            install_extra_value,
            field="install_extra",
            source=path,
        )
    default_for_task = payload.get("default_for_task", False)
    if not isinstance(default_for_task, bool):
        raise ValueError(f"{path}: default_for_task must be a boolean.")

    training = payload.get("training")
    if not isinstance(training, dict):
        raise ValueError(f"{path}: training must be an object.")
    training_family = _nonempty_string(
        training.get("family"),
        field="training.family",
        source=path,
    ).lower()
    training_support = _nonempty_string(
        training.get("support"),
        field="training.support",
        source=path,
    ).lower()
    if training_support != "inference-only":
        raise ValueError(
            f"{path}: manifest discovery supports only an explicit "
            "'inference-only' training boundary; register richer training metadata explicitly.")

    _validate_activation_artifacts(path, model_type)

    return BuiltinModelManifest(
        source_path=path,
        model_type=model_type,
        class_prefix=class_prefix,
        default_checkpoint=checkpoint,
        aliases=aliases,
        task=task,
        capabilities=capabilities,
        architecture=architecture,
        components=components,
        install_extra=install_extra,
        default_for_task=default_for_task,
        training_family=training_family,
        training_support=training_support,
    )


def discover_builtin_model_manifests(
        models_root: str | Path | None = None) -> tuple[BuiltinModelManifest, ...]:
    """Discover activated manifests without importing their model packages."""
    root = Path(__file__).resolve().parent if models_root is None else Path(models_root)
    manifests = tuple(
        manifest for path in sorted(root.glob("*/model-integration.json"))
        if (manifest := _parse_active_manifest(path)) is not None)
    model_types = [manifest.model_type for manifest in manifests]
    if len(model_types) != len(set(model_types)):
        raise ValueError("Built-in model manifests must not declare duplicate model types.")
    aliases = [alias for manifest in manifests for alias in manifest.aliases]
    if len(aliases) != len(set(aliases)):
        raise ValueError("Built-in model manifests must not declare duplicate aliases.")
    collisions = sorted(set(model_types) & set(aliases))
    if collisions:
        raise ValueError(f"Built-in model aliases collide with model types: {collisions!r}.")
    return manifests


__all__ = [
    "BuiltinModelManifest",
    "discover_builtin_model_manifests",
]
