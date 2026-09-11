#!/usr/bin/env python3
"""Generate the object-by-object VoiceHub root API reference."""

from __future__ import annotations

import argparse
import ast
import html
import importlib
import inspect
import sys
import types
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
PACKAGE_ROOT = REPOSITORY_ROOT / "voicehub"
ROOT_INIT_PATH = PACKAGE_ROOT / "__init__.py"
OUTPUT_PATH = REPOSITORY_ROOT / "docs" / "reference" / "public-api.md"
SOURCE_BASE_URL = "https://github.com/kadirnar/voicehub/blob/main"

_CONSTANT_SUMMARIES = {
    "ALL_MODEL_TRAINING_SPECS": "Immutable catalog of every built-in model training specification.",
    "CODEC_CATALOG": "Immutable catalog of the registered shared audio codecs.",
    "MODEL_CATALOG": "Read-only facade over the built-in model registry.",
    "__version__": "Installed VoiceHub package version.",
}

_TYPE_ALIAS_SUMMARIES = {
    "CodecCodeBatch": "Dense or ragged codec-token batch accepted by shared codec helpers.",
    "TTSOptimizationProfile": "Union of the public TTS training optimization profiles.",
    "TTSTrainingOptimizationProfile": "Union of model-specific TTS training optimization profiles.",
}

_CATEGORY_ORDER = (
    "Package metadata",
    "Configuration, factories, and models",
    "Inputs and normalized outputs",
    "Inference and serving",
    "Training",
    "Optimization and codecs",
    "Policies, errors, and utilities",
)


@dataclass(frozen=True)
class PublicAPIRecord:
    """One generated root-export reference entry."""

    name: str
    category: str
    kind: str
    source_module: str
    source_path: str
    source_line: int
    signature: str
    summary: str
    lazy: bool


def _literal_assignment(tree: ast.Module, name: str):
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"{ROOT_INIT_PATH} does not define a literal {name} assignment.")


def _root_export_metadata() -> tuple[tuple[str, ...], dict[str, str], set[str]]:
    tree = ast.parse(ROOT_INIT_PATH.read_text(encoding="utf-8"), filename=str(ROOT_INIT_PATH))
    lazy_exports = dict(_literal_assignment(tree, "_EXPORTS"))
    exports = tuple(sorted(lazy_exports))
    source_modules: dict[str, str] = {}

    for node in tree.body:
        if not isinstance(node, ast.ImportFrom) or node.module is None:
            continue
        if not node.module.startswith("voicehub"):
            continue
        for alias in node.names:
            source_modules[alias.asname or alias.name] = node.module
    source_modules.update(lazy_exports)

    duplicate_exports = sorted({name for name in exports if exports.count(name) > 1})
    if duplicate_exports:
        raise ValueError(f"Duplicate root exports: {duplicate_exports!r}")
    missing_sources = sorted(set(exports) - source_modules.keys())
    extra_sources = sorted(source_modules.keys() - set(exports))
    if missing_sources or extra_sources:
        raise ValueError(
            "Root export/source metadata differs: "
            f"missing={missing_sources!r}, extra={extra_sources!r}.")
    return exports, source_modules, set(lazy_exports)


def _category_for(module_name: str, export_name: str) -> str:
    if export_name == "__version__":
        return "Package metadata"
    if module_name.startswith((
            "voicehub.auto",
            "voicehub.configuration",
            "voicehub.models.tts",
            "voicehub.registry",
            "voicehub.processing.processor",
            "voicehub.registry",
            "voicehub.pipelines",
            "voicehub.tasks",
    )):
        return "Configuration, factories, and models"
    if module_name.startswith((
            "voicehub.audio",
            "voicehub.training.data_collator",
            "voicehub.generation.configuration",
            "voicehub.inference_configuration",
            "voicehub.outputs",
    )):
        return "Inputs and normalized outputs"
    if module_name.startswith((
            "voicehub.diffusion_serving",
            "voicehub.inference_strategy",
            "voicehub.llm_serving",
    )):
        return "Inference and serving"
    if module_name.startswith((
            "voicehub.training.integrations",
            "voicehub.training.trainer",
            "voicehub.training",
    )):
        return "Training"
    if module_name.startswith((
            "voicehub.components.audio.codecs",
            "voicehub.kernels",
            "voicehub.optimization",
    )):
        return "Optimization and codecs"
    return "Policies, errors, and utilities"


def _kind_for(value) -> str:
    if inspect.isclass(value):
        if issubclass(value, BaseException):
            return "exception"
        if issubclass(value, Enum):
            return "enum"
        return "class"
    if isinstance(value, types.UnionType):
        return "type alias"
    if callable(value):
        return "callable"
    return "constant"


def _signature_for(value, kind: str) -> str:
    if kind == "constant":
        return "constant"
    if kind == "type alias":
        return "type alias"
    if kind == "enum":
        # EnumMeta's introspected signature changed between Python 3.10 and
        # 3.11. Public enum members are consistently recovered by value.
        return "(value)"
    try:
        return str(inspect.signature(inspect.unwrap(value)))
    except (TypeError, ValueError):
        if kind in {"class", "exception"}:
            return "inherited constructor"
        return "signature unavailable"


def _summary_for(name: str, value, kind: str) -> str:
    if kind == "constant":
        summary = _CONSTANT_SUMMARIES.get(name)
    elif kind == "type alias":
        summary = _TYPE_ALIAS_SUMMARIES.get(name)
    else:
        documentation = inspect.getdoc(value) or ""
        summary = documentation.split("\n\n", 1)[0].strip() if documentation else None
    if not summary:
        raise ValueError(f"Public export {name!r} has no reference summary.")
    return " ".join(summary.split())


def _module_source_path(module_name: str) -> Path:
    module = importlib.import_module(module_name)
    raw_path = getattr(module, "__file__", None)
    if raw_path is None:
        raise ValueError(f"Public source module {module_name!r} has no source file.")
    path = Path(raw_path).resolve()
    if path.suffix == ".pyc":
        path = path.with_suffix(".py")
    return path


def _definition_target(module_name: str, name: str) -> tuple[str, str]:
    visited = set()
    while (module_name, name) not in visited:
        visited.add((module_name, name))
        source_path = _module_source_path(module_name)
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                return module_name, name
            if isinstance(node, ast.Assign):
                if any(name in _assigned_names(target) for target in node.targets):
                    return module_name, name
            if isinstance(node, ast.AnnAssign) and isinstance(node.target,
                                                              ast.Name) and node.target.id == name:
                return module_name, name

        module = importlib.import_module(module_name)
        export_modules = getattr(module, "_EXPORT_MODULES", {})
        if isinstance(export_modules, dict) and isinstance(export_modules.get(name), str):
            module_name = export_modules[name]
            continue
        reexport = next(((node.module, alias.name)
                         for node in tree.body if isinstance(node, ast.ImportFrom) and node.module is not None
                         for alias in node.names if (alias.asname or alias.name) == name), None)
        if reexport is None:
            return module_name, name
        module_name, name = reexport
    raise ValueError(f"Cyclic public re-export while resolving {name!r} from {module_name!r}.")


def _assigned_names(target: ast.expr) -> set[str]:
    return {node.id for node in ast.walk(target) if isinstance(node, ast.Name)}


def _declaration_line(source_path: Path, name: str) -> int:
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node.lineno
        if isinstance(node, ast.Assign):
            if any(name in _assigned_names(target) for target in node.targets):
                return node.lineno
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
            return node.lineno
        if isinstance(node, ast.ImportFrom):
            if any((alias.asname or alias.name) == name for alias in node.names):
                return node.lineno
    raise ValueError(f"Could not locate {name!r} in {source_path.relative_to(REPOSITORY_ROOT)}.")


def _source_location(value, source_module: str, export_name: str) -> tuple[str, int]:
    target = inspect.unwrap(value) if callable(value) else value
    try:
        raw_source_path = inspect.getsourcefile(target)
        _, source_line = inspect.getsourcelines(target)
    except (OSError, TypeError):
        raw_source_path = None
        source_line = 0

    source_path = Path(raw_source_path).resolve() if raw_source_path else None
    if source_path is None or REPOSITORY_ROOT not in source_path.parents:
        source_path = _module_source_path(source_module)
        source_line = _declaration_line(source_path, export_name)
    if REPOSITORY_ROOT not in source_path.parents:
        raise ValueError(f"Public export {export_name!r} resolves outside this repository: {source_path}")
    return source_path.relative_to(REPOSITORY_ROOT).as_posix(), source_line


def build_public_api_inventory() -> tuple[PublicAPIRecord, ...]:
    """Resolve and validate every object exported from :mod:`voicehub`."""
    exports, source_modules, lazy_exports = _root_export_metadata()
    voicehub = importlib.import_module("voicehub")
    records = []
    for name in exports:
        try:
            value = getattr(voicehub, name)
        except Exception as error:
            raise RuntimeError(f"Could not resolve public export {name!r}.") from error
        kind = _kind_for(value)
        value_module = getattr(value, "__module__", None)
        value_name = getattr(value, "__name__", name)
        if isinstance(value_module, str) and value_module.startswith("voicehub"):
            source_module, source_name = _definition_target(value_module, value_name)
        else:
            source_module, source_name = _definition_target(source_modules[name], name)
        source_path, source_line = _source_location(value, source_module, source_name)
        records.append(
            PublicAPIRecord(
                name=name,
                category=_category_for(source_module, name),
                kind=kind,
                source_module=source_module,
                source_path=source_path,
                source_line=source_line,
                signature=_signature_for(value, kind),
                summary=_summary_for(name, value, kind),
                lazy=name in lazy_exports,
            ))
    category_positions = {name: index for index, name in enumerate(_CATEGORY_ORDER)}
    return tuple(
        sorted(records, key=lambda record: (category_positions[record.category], record.name.lower())))


def _table_cell(value: str) -> str:
    return html.escape(" ".join(value.split()), quote=False).replace("|", "&#124;")


def render_public_api_reference() -> str:
    """Return the complete generated Markdown reference."""
    records = build_public_api_inventory()
    lines = [
        "---",
        "description: Generated inventory of every object exported by the VoiceHub package root.",
        "---",
        "",
        "# Public exports",
        "",
        "<!-- Generated by scripts/generate_public_api.py. Do not edit by hand. -->",
        "",
        "VoiceHub exposes one explicit package-root surface for discovery, loading,",
        "inference, training, optimization, serving, and serialization. This page is",
        "generated from `voicehub.__all__` and complements the task-oriented",
        "[full API reference](api.md). Its grouping follows the role of Transformers'",
        "[Main Classes](https://huggingface.co/docs/transformers/main_classes) while",
        "keeping speech-specific contracts and names.",
        "",
        "Every row resolves from `voicehub`, points to repository source, records the",
        "canonical defining or re-export module, and includes a callable signature or",
        "an explicit constant/type-alias marker. Generation fails on duplicate,",
        "unresolved, undocumented, source-less, or stale exports.",
        "",
        f"Current inventory: **{len(records)} public exports**.",
        "",
    ]
    for category in _CATEGORY_ORDER:
        category_records = [record for record in records if record.category == category]
        if not category_records:
            continue
        lines.extend((
            f"## {category}",
            "",
            "| Export | Kind | Canonical module | Signature | Summary | Lazy |",
            "| --- | --- | --- | --- | --- | --- |",
        ))
        for record in category_records:
            source_url = f"{SOURCE_BASE_URL}/{record.source_path}#L{record.source_line}"
            lines.append(
                f"| [`{record.name}`]({source_url}) | {record.kind} | "
                f"`{record.source_module}` | <code>{_table_cell(record.signature)}</code> | "
                f"{_table_cell(record.summary)} | {'yes' if record.lazy else 'no'} |")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if the generated reference is stale.")
    arguments = parser.parse_args(argv)
    rendered = render_public_api_reference()
    export_count = len(build_public_api_inventory())
    if arguments.check:
        if not OUTPUT_PATH.is_file() or OUTPUT_PATH.read_text(encoding="utf-8") != rendered:
            print(f"ERROR: {OUTPUT_PATH.relative_to(REPOSITORY_ROOT)} is missing or stale.", file=sys.stderr)
            return 1
        print(f"OK: {export_count} public exports are current")
        return 0
    OUTPUT_PATH.write_text(rendered, encoding="utf-8")
    print(f"Wrote {export_count} public exports to {OUTPUT_PATH.relative_to(REPOSITORY_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
