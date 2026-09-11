"""Static policy for capability-driven behavior in shared VoiceHub layers."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

_MODEL_LOCAL_DIRECTORIES = frozenset({"models"})


@dataclass(frozen=True, order=True)
class ProviderBranchViolation:
    """One registered provider literal used by a shared behavior branch."""

    path: str
    line: int
    column: int
    provider: str
    construct: str

    def __str__(self) -> str:
        return (
            f"{self.path}:{self.line}:{self.column}: shared {self.construct} "
            f"branches on registered provider {self.provider!r}")


class _ProviderBranchVisitor(ast.NodeVisitor):
    """Find provider literals in conditions without inspecting declarations."""

    def __init__(self, path: str, provider_names: frozenset[str]) -> None:
        self.path = path
        self.provider_names = provider_names
        self.violations: dict[tuple[int, int, str], ProviderBranchViolation] = {}

    def _record(self, node: ast.AST | None, construct: str) -> None:
        if node is None:
            return
        for value in ast.walk(node):
            if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
                continue
            if value.value not in self.provider_names:
                continue
            key = (value.lineno, value.col_offset, value.value)
            self.violations.setdefault(
                key,
                ProviderBranchViolation(
                    path=self.path,
                    line=value.lineno,
                    column=value.col_offset + 1,
                    provider=value.value,
                    construct=construct,
                ),
            )

    def visit_Assert(self, node: ast.Assert) -> None:
        self._record(node.test, "assertion")
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        self._record(node, "comparison")
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        self._record(node.test, "if condition")
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self._record(node.test, "conditional expression")
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self._record(node.test, "while condition")
        self.generic_visit(node)

    def visit_comprehension(self, node: ast.comprehension) -> None:
        for condition in node.ifs:
            self._record(condition, "comprehension condition")
        self.generic_visit(node)

    def visit_Match(self, node: ast.Match) -> None:
        for case in node.cases:
            self._record(case.pattern, "match case")
            self._record(case.guard, "match guard")
        self.generic_visit(node)


def _registered_provider_names() -> frozenset[str]:
    """Read canonical names and live aliases from the dependency-light
    registry."""
    from voicehub.registry import MODEL_ALIASES, list_model_specs

    return frozenset({
        *(spec.model_type for spec in list_model_specs(task=None)),
        *MODEL_ALIASES,
    })


def collect_shared_python_paths(
    package_root: str | Path,
    *,
    excluded_directories: Iterable[str] = _MODEL_LOCAL_DIRECTORIES,
) -> tuple[Path, ...]:
    """Return shared Python files, excluding model-local implementation
    roots."""
    root = Path(package_root)
    excluded = frozenset(excluded_directories)
    return tuple(
        path for path in sorted(root.rglob("*.py")) if path.relative_to(root).parts[0] not in excluded)


def inspect_shared_provider_branches(
    package_root: str | Path,
    *,
    provider_names: Iterable[str] | None = None,
    excluded_directories: Iterable[str] = _MODEL_LOCAL_DIRECTORIES,
) -> tuple[ProviderBranchViolation, ...]:
    """Inspect shared conditions for canonical model names and aliases."""
    root = Path(package_root)
    names = (_registered_provider_names() if provider_names is None else frozenset(provider_names))
    violations = []
    for path in collect_shared_python_paths(
            root,
            excluded_directories=excluded_directories,
    ):
        relative = path.relative_to(root).as_posix()
        try:
            tree = ast.parse(
                path.read_text(encoding="utf-8"),
                filename=str(path),
            )
        except (OSError, SyntaxError) as error:
            raise ValueError(f"Could not inspect Python source {path}: {error}.") from error
        visitor = _ProviderBranchVisitor(relative, names)
        visitor.visit(tree)
        violations.extend(visitor.violations.values())
    return tuple(sorted(violations))


def require_shared_provider_independence(
    package_root: str | Path,
    *,
    provider_names: Iterable[str] | None = None,
    excluded_directories: Iterable[str] = _MODEL_LOCAL_DIRECTORIES,
) -> None:
    """Raise with every shared provider-name branch found by the policy."""
    violations = inspect_shared_provider_branches(
        package_root,
        provider_names=provider_names,
        excluded_directories=excluded_directories,
    )
    if violations:
        details = "\n".join(f"- {violation}" for violation in violations)
        raise RuntimeError(f"Shared VoiceHub behavior must use capabilities, not provider names:\n{details}")


__all__ = [
    "ProviderBranchViolation",
    "collect_shared_python_paths",
    "inspect_shared_provider_branches",
    "require_shared_provider_independence",
]
