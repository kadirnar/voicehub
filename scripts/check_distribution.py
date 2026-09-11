#!/usr/bin/env python3
"""Build and install-check VoiceHub's wheel, sdist, and editable source tree.

The default check skips runtime dependencies so it can validate
packaging without downloading PyTorch. Pass ``--with-dependencies`` for
a complete dependency installation on a release machine.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from zipfile import ZipFile

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REPRESENTATIVE_PACKAGE_FILES = (
    "voicehub/py.typed",
    "voicehub/models/outetts/native/default_speaker.json",
    "voicehub/models/conversationtts/source/conversationtts/llama3_2/tokenizer.json",
    ("voicehub/models/chatterbox/source/perth/perth_net/pretrained/implicit/"
     "perth_net_250000.pth.tar"),
    "voicehub/kernels/csrc/activations.cpp",
)
COMPLIANCE_NAME_TOKENS = ("LICENSE", "LICENCE", "NOTICE", "COPYING")


def compliance_package_files(repository_root: Path = REPOSITORY_ROOT) -> tuple[str, ...]:
    """List every provenance manifest and legal notice shipped in VoiceHub."""
    package_root = repository_root / "voicehub"
    return tuple(
        sorted(
            path.relative_to(repository_root).as_posix() for path in package_root.rglob("*")
            if path.is_file() and (
                path.name == "SOURCE.json" or any(
                    token in path.name.upper() for token in COMPLIANCE_NAME_TOKENS))))


COMPLIANCE_PACKAGE_FILES = compliance_package_files()
REQUIRED_PACKAGE_FILES = tuple(dict.fromkeys((*REPRESENTATIVE_PACKAGE_FILES, *COMPLIANCE_PACKAGE_FILES)))


def _manifest_values(value: object, key_tokens: tuple[str, ...]) -> list[object]:
    values: list[object] = []
    if isinstance(value, dict):
        for key, child in value.items():
            normalized_key = str(key).lower()
            if (any(token in normalized_key for token in key_tokens) and child not in (None, "", [], {})):
                values.append(child)
            values.extend(_manifest_values(child, key_tokens))
    elif isinstance(value, list):
        for child in value:
            values.extend(_manifest_values(child, key_tokens))
    return values


def validate_provenance_manifests(repository_root: Path = REPOSITORY_ROOT) -> int:
    """Require every retained source manifest to pin origin and license
    terms."""
    manifest_paths = sorted((repository_root / "voicehub").rglob("SOURCE.json"))
    if not manifest_paths:
        raise RuntimeError("No source provenance manifests were found.")
    for path in manifest_paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Could not read provenance manifest {path}: {error}") from error
        if not isinstance(payload, dict) or not payload:
            raise RuntimeError(f"Provenance manifest {path} must contain a JSON object.")
        if not _manifest_values(payload, ("revision", "release", "commit", "tag")):
            raise RuntimeError(f"Provenance manifest {path} does not pin an upstream source.")
        if not _manifest_values(payload, ("license", "licence")):
            raise RuntimeError(f"Provenance manifest {path} does not record license terms.")
    return len(manifest_paths)


def run(*command: str | Path, cwd: Path | None = None) -> None:
    rendered = [str(item) for item in command]
    print("+", " ".join(rendered), flush=True)
    completed = subprocess.run(
        rendered,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        if completed.stdout:
            print(completed.stdout)
        if completed.stderr:
            print(completed.stderr, file=sys.stderr)
        completed.check_returncode()


def wheel_members(path: Path) -> set[str]:
    with ZipFile(path) as archive:
        return {member.filename for member in archive.infolist()}


def sdist_members(path: Path) -> set[str]:
    with tarfile.open(path) as archive:
        names = set()
        for member in archive.getmembers():
            parts = Path(member.name).parts
            if member.isfile() and len(parts) > 1:
                names.add(Path(*parts[1:]).as_posix())
        return names


def require_members(kind: str, members: set[str]) -> None:
    missing = sorted(set(REQUIRED_PACKAGE_FILES) - members)
    if missing:
        raise RuntimeError(f"{kind} is missing required package data: {missing}")


def require_project_license(kind: str, members: set[str]) -> None:
    """Require the project license in both standard distribution layouts."""
    if kind == "wheel":
        licenses = sorted(member for member in members if member.endswith(".dist-info/licenses/LICENSE"))
        if len(licenses) != 1:
            raise RuntimeError(f"wheel must contain one dist-info project LICENSE; found {licenses}")
    elif kind == "sdist":
        if "LICENSE" not in members:
            raise RuntimeError("sdist is missing the project LICENSE")
    else:
        raise ValueError(f"Unknown distribution kind: {kind}")


def create_venv(path: Path) -> Path:
    run(sys.executable, "-m", "venv", path)
    executable = "python.exe" if sys.platform == "win32" else "python"
    scripts = "Scripts" if sys.platform == "win32" else "bin"
    return path / scripts / executable


def install_and_probe(
    name: str,
    source: Path,
    *,
    root: Path,
    editable: bool,
    with_dependencies: bool,
) -> dict[str, object]:
    environment = root / f"{name}-venv"
    python = create_venv(environment)
    command: list[str | Path] = [python, "-m", "pip", "install"]
    if not with_dependencies:
        command.append("--no-deps")
    if editable:
        command.append("-e")
    command.append(source)
    run(*command)

    probe = """
import json
import sys
from pathlib import Path
from importlib.resources import files

import voicehub
from voicehub.policies.architecture_dependencies import (
    collect_native_runtime_paths,
    inspect_native_runtime,
)

required = {
    "py.typed": files("voicehub").joinpath("py.typed").is_file(),
    "default_speaker.json": files("voicehub").joinpath(
        "models", "outetts", "native", "default_speaker.json"
    ).is_file(),
    "tokenizer.json": files("voicehub").joinpath(
        "models", "conversationtts", "source", "conversationtts",
        "llama3_2", "tokenizer.json"
    ).is_file(),
    "watermark_checkpoint": files("voicehub").joinpath(
        "models", "chatterbox", "source", "perth", "perth_net",
        "pretrained", "implicit", "perth_net_250000.pth.tar"
    ).is_file(),
    "kernel_source": files("voicehub").joinpath(
        "kernels", "csrc", "activations.cpp"
    ).is_file(),
}
if not all(required.values()):
    raise RuntimeError(f"Missing installed package data: {required}")

package_root = Path(str(files("voicehub")))
compliance_files = sorted(
    path
    for path in package_root.rglob("*")
    if path.is_file()
    and (
        path.name == "SOURCE.json"
        or any(
            token in path.name.upper()
            for token in ("LICENSE", "LICENCE", "NOTICE", "COPYING")
        )
    )
)
expected_compliance_files = int(sys.argv[1])
if len(compliance_files) != expected_compliance_files:
    raise RuntimeError(
        "Installed compliance inventory is incomplete: "
        f"{len(compliance_files)} files, expected {expected_compliance_files}"
    )
violations = inspect_native_runtime(package_root)
if violations:
    raise RuntimeError(
        "Registered runtime imports undeclared dependencies: "
        + "; ".join(str(item) for item in violations)
    )
covered = set(collect_native_runtime_paths(package_root))
uncovered = {}
for spec in voicehub.list_model_specs(task=None):
    paths = set()
    for module_name in (spec.module, spec.config_module):
        relative = Path(*module_name.split(".")[1:])
        module_file = (package_root / relative).with_suffix(".py")
        package_file = package_root / relative / "__init__.py"
        paths.add(module_file if module_file.is_file() else package_file)
    missing = sorted(str(path.relative_to(package_root)) for path in paths - covered)
    if missing:
        uncovered[spec.model_type] = missing
if uncovered:
    raise RuntimeError(f"Registered runtimes outside dependency audit: {uncovered}")

print(json.dumps({
    "version": voicehub.__version__,
    "models": len(voicehub.list_model_specs(task=None)),
    "torch_imported": "torch" in sys.modules,
    "required_data": required,
    "runtime_dependency_violations": len(violations),
    "compliance_files": len(compliance_files),
}))
"""
    completed = subprocess.run(
        [str(python), "-c", probe, str(len(COMPLIANCE_PACKAGE_FILES))],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    print(f"{name}: {json.dumps(result, sort_keys=True)}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-dependencies",
        action="store_true",
        help="Install declared runtime dependencies in each test environment.",
    )
    return parser.parse_args()


def main() -> None:
    options = parse_args()
    provenance_manifests = validate_provenance_manifests()
    with tempfile.TemporaryDirectory(prefix="voicehub-distribution-") as directory:
        root = Path(directory)
        dist = root / "dist"
        run(
            sys.executable,
            "-m",
            "build",
            "--outdir",
            dist,
            REPOSITORY_ROOT,
            # Avoid a repository-local ``build/`` output directory shadowing
            # the installed PyPA ``build`` package.
            cwd=root,
        )

        wheels = sorted(dist.glob("voicehub-*.whl"))
        sdists = sorted(dist.glob("voicehub-*.tar.gz"))
        if len(wheels) != 1 or len(sdists) != 1:
            raise RuntimeError(f"Expected one wheel and one sdist, found {wheels!r} and {sdists!r}")

        wheel_files = wheel_members(wheels[0])
        sdist_files = sdist_members(sdists[0])
        require_members("wheel", wheel_files)
        require_members("sdist", sdist_files)
        require_project_license("wheel", wheel_files)
        require_project_license("sdist", sdist_files)

        results = {
            "wheel":
            install_and_probe(
                "wheel",
                wheels[0],
                root=root,
                editable=False,
                with_dependencies=options.with_dependencies,
            ),
            "sdist":
            install_and_probe(
                "sdist",
                sdists[0],
                root=root,
                editable=False,
                with_dependencies=options.with_dependencies,
            ),
            "editable":
            install_and_probe(
                "editable",
                REPOSITORY_ROOT,
                root=root,
                editable=True,
                with_dependencies=options.with_dependencies,
            ),
        }
        versions = {str(result["version"]) for result in results.values()}
        model_counts = {int(result["models"]) for result in results.values()}
        compliance_counts = {int(result["compliance_files"]) for result in results.values()}
        if len(versions) != 1 or len(model_counts) != 1 or len(compliance_counts) != 1:
            raise RuntimeError(f"Installation modes disagree: {results}")

        print(
            "PASS:",
            f"version={versions.pop()}",
            f"models={model_counts.pop()}",
            f"provenance_manifests={provenance_manifests}",
            f"compliance_files={compliance_counts.pop()}",
            f"wheel_bytes={wheels[0].stat().st_size}",
            f"sdist_bytes={sdists[0].stat().st_size}",
        )


if __name__ == "__main__":
    main()
