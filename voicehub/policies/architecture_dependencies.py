"""Import boundary for VoiceHub-owned architecture and training code.

The native runtime may use Python's standard library, VoiceHub itself,
and PyTorch as the tensor/autograd substrate.  Model frameworks,
upstream provider packages, optimization engines, and convenience DSP
libraries are forbidden in this layer.  They may only appear behind
explicit optional execution strategies outside the architecture
implementation.
"""

from __future__ import annotations

import ast
import sys
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from importlib.util import resolve_name
from pathlib import Path

ALLOWED_NATIVE_IMPORT_ROOTS = frozenset({"torch", "voicehub"})
_DYNAMIC_IMPORT_INFRASTRUCTURE = frozenset({
    "auto.py",
    'runtime/catalog.py',
    'runtime/specifications.py',
    "dependencies.py",
})
_PER_FILE_ALLOWED_IMPORT_ROOTS = {
    'training/integrations.py': frozenset({"wandb"}),
    "kernels/capabilities.py": frozenset({"cutlass", "triton"}),
    "kernels/cute_codecs.py": frozenset({"cutlass"}),
    "kernels/triton_activations.py": frozenset({"triton"}),
    "neural/backends/flash_attention4.py": frozenset({"flash_attn"}),
}
_CORE_NATIVE_RUNTIME_DIRECTORIES = (
    'runtime',
    "audio.py",
    'models/base.py',
    "checkpointing",
    "components/neural/conformer",
    "components/audio/watermarking/wavmark",
    "components/audio/vocoders/vocos",
    "components/audio/codecs/dac/model",
    "components/audio/codecs/dac/nn",
    "components/audio/codecs/dac/utils",
    "components/audio/codecs/dac/compare",
    "components/audio/codecs/encodec",
    'training/data_collator.py',
    "generation",
    "neural",
    "objectives",
    "optimization",
    "processing",
    "streaming.py",
    "tokenization",
    'training/trainer.py',
    'training/utils.py',
    "training",
)
_DYNAMIC_NATIVE_SOURCE_DIRECTORIES = (
    "models/asr_native/_wenet",
    "models/chatterbox/models",
    "models/melotts/source/melo/monotonic_align",
)


def _discover_architecture_reference_files(root: Path) -> tuple[str, ...]:
    """Resolve literal lazy-component modules from registration source."""
    package_prefix = f"{root.name}."
    references = set()
    for registration_path in sorted((root / "models").glob("*/native/registration.py")):
        tree = ast.parse(
            registration_path.read_text(encoding="utf-8"),
            filename=str(registration_path),
        )
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            module_name, separator, _ = node.value.partition(":")
            if not separator or not module_name.startswith(package_prefix):
                continue
            relative_parts = module_name[len(package_prefix):].split(".")
            module_path = root.joinpath(*relative_parts).with_suffix(".py")
            if not module_path.is_file():
                module_path = root.joinpath(*relative_parts, "__init__.py")
            if module_path.is_file():
                references.add(module_path.relative_to(root).as_posix())
    return tuple(sorted(references))


def discover_native_runtime_directories(package_root: str | Path, ) -> tuple[str, ...]:
    """Derive native-policy seeds without importing model packages.

    Immediate Python files under each model package are public facades,
    configuration, runtime, and training boundaries. Internal imports
    expand those seeds to a fixed-point closure. The small explicit
    source set covers active vendored modules reached through runtime-
    generated imports.
    """
    root = Path(package_root)
    model_root = root / "models"
    model_facades = tuple(path.relative_to(root).as_posix() for path in sorted(model_root.glob("*/*.py")))
    native_networks = tuple(path.relative_to(root).as_posix() for path in sorted(model_root.glob("*/native")))
    architecture_references = _discover_architecture_reference_files(root)
    return tuple(
        dict.fromkeys((
            *_CORE_NATIVE_RUNTIME_DIRECTORIES,
            *model_facades,
            *native_networks,
            *architecture_references,
            *_DYNAMIC_NATIVE_SOURCE_DIRECTORIES,
        )))


NATIVE_RUNTIME_DIRECTORIES = discover_native_runtime_directories(Path(__file__).resolve().parents[1], )


@dataclass(frozen=True, order=True)
class ImportPolicyViolation:
    """One external import found inside the native runtime boundary."""

    path: Path
    line: int
    module: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: external import {self.module!r}"


def _module_root(module: str) -> str:
    return module.partition(".")[0]


def _is_allowed(module: str, *, allowed_roots: frozenset[str]) -> bool:
    root = _module_root(module)
    return (root in allowed_roots or root in sys.stdlib_module_names or root == "__future__")


def _literal_dynamic_import(node: ast.Call) -> tuple[str, int] | None:
    function_name = None
    if isinstance(node.func, ast.Name):
        function_name = node.func.id
    elif isinstance(node.func, ast.Attribute):
        function_name = node.func.attr
    if function_name not in {"__import__", "import_module", "import_optional"}:
        return None
    if not node.args or not isinstance(node.args[0], ast.Constant):
        return None
    module = node.args[0].value
    if not isinstance(module, str) or not module:
        return None
    return module, node.lineno


def _dynamic_import_function_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        function_name = node.func.id
    elif isinstance(node.func, ast.Attribute):
        function_name = node.func.attr
    else:
        return None
    if function_name in {"__import__", "import_module", "import_optional"}:
        return function_name
    return None


def _parse_source(path: Path) -> ast.Module:
    try:
        return ast.parse(
            path.read_text(encoding="utf-8"),
            filename=str(path),
        )
    except (OSError, UnicodeDecodeError, SyntaxError) as error:
        raise ValueError(f"Could not inspect Python source {path}: {error}.") from error


class _LazyNamespaceImportVisitor(ast.NodeVisitor):
    """Recognize unresolved imports used only by a lazy package namespace."""

    def __init__(self) -> None:
        self._function_stack: list[str] = []
        self.unresolved_imports: list[bool] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._function_stack.append(node.name)
        self.generic_visit(node)
        self._function_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._function_stack.append(node.name)
        self.generic_visit(node)
        self._function_stack.pop()

    def visit_Call(self, node: ast.Call) -> None:
        if (_dynamic_import_function_name(node) is not None and _literal_dynamic_import(node) is None):
            self.unresolved_imports.append(
                bool(self._function_stack) and self._function_stack[-1] == "__getattr__", )
        self.generic_visit(node)


def _is_lazy_namespace_initializer(path: Path) -> bool:
    """Return whether unresolved imports are confined to package
    ``__getattr__``."""
    if path.name != "__init__.py":
        return False
    visitor = _LazyNamespaceImportVisitor()
    visitor.visit(_parse_source(path))
    return bool(visitor.unresolved_imports) and all(visitor.unresolved_imports)


def inspect_native_imports(
    path: str | Path,
    *,
    allowed_roots: Iterable[str] = ALLOWED_NATIVE_IMPORT_ROOTS,
    allow_unresolved_dynamic_imports: bool = False,
) -> tuple[ImportPolicyViolation, ...]:
    """Inspect one Python file without importing it."""
    source_path = Path(path)
    normalized_roots = frozenset(allowed_roots)
    tree = _parse_source(source_path)
    violations: set[ImportPolicyViolation] = set()
    for node in ast.walk(tree):
        modules: tuple[tuple[str, int], ...] = ()
        if isinstance(node, ast.Import):
            modules = tuple((alias.name, node.lineno) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules = ((node.module, node.lineno), )
        elif isinstance(node, ast.Call):
            dynamic = _literal_dynamic_import(node)
            if dynamic is not None:
                modules = (dynamic, )
            elif not allow_unresolved_dynamic_imports:
                function_name = _dynamic_import_function_name(node)
                if function_name is not None:
                    modules = ((f"<dynamic:{function_name}>", node.lineno), )
        for module, line in modules:
            if not _is_allowed(module, allowed_roots=normalized_roots):
                violations.add(ImportPolicyViolation(
                    path=source_path,
                    line=line,
                    module=module,
                ))
    return tuple(sorted(violations))


def _module_name_for_path(package_root: Path, path: Path) -> str:
    relative = path.relative_to(package_root)
    if relative.name == "__init__.py":
        parts = relative.parent.parts
    else:
        parts = relative.with_suffix("").parts
    return ".".join((package_root.name, *parts))


def _resolve_internal_module_path(
    package_root: Path,
    module_name: str,
) -> Path | None:
    package_name = package_root.name
    if module_name == package_name:
        initializer = package_root / "__init__.py"
        return initializer if initializer.is_file() else None
    prefix = f"{package_name}."
    if not module_name.startswith(prefix):
        return None
    relative_parts = module_name[len(prefix):].split(".")
    module_path = package_root.joinpath(*relative_parts).with_suffix(".py")
    if module_path.is_file():
        return module_path
    initializer = package_root.joinpath(*relative_parts, "__init__.py")
    return initializer if initializer.is_file() else None


def _absolute_import_name(
    *,
    imported_name: str,
    level: int,
    package_name: str,
) -> str | None:
    if level == 0:
        return imported_name
    relative_name = f"{'.' * level}{imported_name}"
    try:
        return resolve_name(relative_name, package_name)
    except (ImportError, ValueError):
        return None


def _iter_internal_import_names(
    tree: ast.Module,
    *,
    module_name: str,
    is_package: bool,
    package_root_name: str,
) -> Iterable[str]:
    package_name = module_name if is_package else module_name.rpartition(".")[0]
    internal_prefix = f"{package_root_name}."

    def is_internal(name: str) -> bool:
        return name == package_root_name or name.startswith(internal_prefix)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if is_internal(alias.name):
                    yield alias.name
            continue
        if isinstance(node, ast.ImportFrom):
            imported_name = _absolute_import_name(
                imported_name=node.module or "",
                level=node.level,
                package_name=package_name,
            )
            if imported_name is None or not is_internal(imported_name):
                continue
            yield imported_name
            for alias in node.names:
                if alias.name != "*":
                    yield f"{imported_name}.{alias.name}"
            continue
        if not isinstance(node, ast.Call):
            continue
        dynamic = _literal_dynamic_import(node)
        if dynamic is None:
            continue
        imported_name = dynamic[0]
        if imported_name.startswith("."):
            imported_name = _absolute_import_name(
                imported_name=imported_name.lstrip("."),
                level=len(imported_name) - len(imported_name.lstrip(".")),
                package_name=package_name,
            )
        if imported_name is not None and is_internal(imported_name):
            yield imported_name


def inspect_native_runtime(
    package_root: str | Path,
    *,
    directories: Iterable[str] | None = None,
    allowed_roots: Iterable[str] = ALLOWED_NATIVE_IMPORT_ROOTS,
) -> tuple[ImportPolicyViolation, ...]:
    """Inspect all present native runtime files and their package initializers.

    Importing ``voicehub.a.b.module`` executes every package
    ``__init__.py`` between ``voicehub`` and ``module``.  Auditing only
    ``module.py`` would therefore allow an eager dependency in an
    ancestor package to bypass the native boundary.
    """
    violations: list[ImportPolicyViolation] = []
    root = Path(package_root)
    for path in collect_native_import_closure(
            package_root,
            directories=directories,
    ):
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            relative = ""
        allow_dynamic = (relative in _DYNAMIC_IMPORT_INFRASTRUCTURE or _is_lazy_namespace_initializer(path))
        path_allowed_roots = (
            frozenset(allowed_roots)
            | _PER_FILE_ALLOWED_IMPORT_ROOTS.get(relative, frozenset()))
        violations.extend(
            inspect_native_imports(
                path,
                allowed_roots=path_allowed_roots,
                allow_unresolved_dynamic_imports=allow_dynamic,
            ))
    return tuple(sorted(violations))


def collect_native_runtime_paths(
    package_root: str | Path,
    *,
    directories: Iterable[str] | None = None,
) -> tuple[Path, ...]:
    """Resolve the complete, auditable file set for a native boundary."""
    root = Path(package_root)
    resolved_directories = (
        discover_native_runtime_directories(root) if directories is None else tuple(directories))
    root_init = root / "__init__.py"
    paths: set[Path] = set()

    def add_package_initializers(path: Path) -> None:
        parent = path.parent
        if root_init.is_file():
            paths.add(root_init)
        try:
            relative_parent = parent.relative_to(root)
        except ValueError:
            return
        current = root
        for part in relative_parent.parts:
            current /= part
            initializer = current / "__init__.py"
            if initializer.is_file():
                paths.add(initializer)

    for directory in resolved_directories:
        runtime_path = root / directory
        if runtime_path.is_file():
            runtime_files = (runtime_path, )
        elif runtime_path.is_dir():
            runtime_files = tuple(sorted(runtime_path.rglob("*.py")))
        else:
            continue
        for path in runtime_files:
            add_package_initializers(path)
            paths.add(path)
    return tuple(sorted(paths))


def collect_native_import_closure(
    package_root: str | Path,
    *,
    directories: Iterable[str] | None = None,
) -> tuple[Path, ...]:
    """Resolve the fixed-point VoiceHub import closure of the native boundary.

    The explicit native paths are seeds, not an exemption list. Every
    statically discoverable internal import is recursively inspected,
    including relative imports, package initializers, and literal
    ``import_module``/``__import__``/``import_optional`` calls.
    """
    root = Path(package_root)
    root_init = root / "__init__.py"
    paths = set(collect_native_runtime_paths(
        root,
        directories=directories,
    ))
    pending = deque(sorted(paths))

    def add_path(path: Path) -> None:
        if path not in paths:
            paths.add(path)
            pending.append(path)

        if root_init.is_file() and root_init not in paths:
            paths.add(root_init)
            pending.append(root_init)
        try:
            relative_parent = path.parent.relative_to(root)
        except ValueError:
            return
        current = root
        for part in relative_parent.parts:
            current /= part
            initializer = current / "__init__.py"
            if initializer.is_file() and initializer not in paths:
                paths.add(initializer)
                pending.append(initializer)

    while pending:
        path = pending.popleft()
        module_name = _module_name_for_path(root, path)
        tree = _parse_source(path)
        for imported_name in _iter_internal_import_names(
                tree,
                module_name=module_name,
                is_package=path.name == "__init__.py",
                package_root_name=root.name,
        ):
            imported_path = _resolve_internal_module_path(root, imported_name)
            if imported_path is not None:
                add_path(imported_path)

    return tuple(sorted(paths))


def require_native_runtime_independence(
    package_root: str | Path,
    *,
    directories: Iterable[str] | None = None,
    allowed_roots: Iterable[str] = ALLOWED_NATIVE_IMPORT_ROOTS,
) -> None:
    """Raise with every violation instead of failing on only the first."""
    violations = inspect_native_runtime(
        package_root,
        directories=directories,
        allowed_roots=allowed_roots,
    )
    if violations:
        details = "\n".join(f"- {violation}" for violation in violations)
        raise RuntimeError("VoiceHub native runtime imports external architecture code:\n"
                           f"{details}")


__all__ = [
    "ALLOWED_NATIVE_IMPORT_ROOTS",
    "NATIVE_RUNTIME_DIRECTORIES",
    "ImportPolicyViolation",
    "collect_native_import_closure",
    "collect_native_runtime_paths",
    "discover_native_runtime_directories",
    "inspect_native_imports",
    "inspect_native_runtime",
    "require_native_runtime_independence",
]
