#!/usr/bin/env python3
"""Validate VoiceHub source, evidence, artifacts, tags, and PyPI release
state."""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import tarfile
import urllib.request
from email.parser import Parser
from pathlib import Path
from typing import Any
from zipfile import ZipFile

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PYPI_JSON_URL = "https://pypi.org/pypi/voicehub/json"
PYPI_DEFAULT_FILE_LIMIT = 100_000_000
VERSION_PATTERN = re.compile(r"(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)")
WHEEL_VERSION_PATTERN = re.compile(r"voicehub-(\d+\.\d+\.\d+)-py3-none-any\.whl")


class ReleaseCheckError(RuntimeError):
    """Raised when release evidence contradicts the candidate contract."""


def parse_version(value: str) -> tuple[int, int, int]:
    """Parse the stable semantic version format used by VoiceHub releases."""
    if VERSION_PATTERN.fullmatch(value) is None:
        raise ReleaseCheckError(
            f"VoiceHub release versions must use stable X.Y.Z syntax; received {value!r}.")
    major, minor, patch = value.split(".")
    return int(major), int(minor), int(patch)


def source_version(repository_root: Path = REPOSITORY_ROOT) -> str:
    """Read ``voicehub.__version__`` without importing the package."""
    init_path = repository_root / "voicehub" / "__init__.py"
    module = ast.parse(init_path.read_text(encoding="utf-8"), filename=str(init_path))
    for statement in module.body:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        targets = statement.targets if isinstance(statement, ast.Assign) else (statement.target, )
        if not any(isinstance(target, ast.Name) and target.id == "__version__" for target in targets):
            continue
        value = ast.literal_eval(statement.value)
        if not isinstance(value, str):
            break
        parse_version(value)
        return value
    raise ReleaseCheckError(f"Could not read a string __version__ assignment from {init_path}.")


def validate_source_metadata(version: str, repository_root: Path = REPOSITORY_ROOT) -> None:
    """Require packaging metadata to resolve its version from the public
    package."""
    pyproject = (repository_root / "pyproject.toml").read_text(encoding="utf-8")
    required_fragments = (
        'name = "voicehub"',
        'dynamic = ["version"]',
        'description = "A unified inference and training interface for TTS, ASR, and VAD models"',
        'version = { attr = "voicehub.__version__" }',
        'requires-python = ">=3.10"',
    )
    missing = [fragment for fragment in required_fragments if fragment not in pyproject]
    if missing:
        raise ReleaseCheckError(f"pyproject.toml is missing release metadata: {missing}")
    if f'__version__ = "{version}"' not in (repository_root / "voicehub" /
                                            "__init__.py").read_text(encoding="utf-8"):
        raise ReleaseCheckError("The parsed source version does not have one canonical assignment.")


def _voicehub_versions(value: Any) -> list[str]:
    versions: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "voicehub_version":
                versions.append(str(child))
            versions.extend(_voicehub_versions(child))
    elif isinstance(value, list):
        for child in value:
            versions.extend(_voicehub_versions(child))
    return versions


def validate_benchmark_versions(version: str, repository_root: Path = REPOSITORY_ROOT) -> int:
    """Require every retained JSON evidence file to identify the source
    version."""
    benchmark_paths = tuple(sorted((repository_root / "benchmarks").glob("*.json")))
    if not benchmark_paths:
        raise ReleaseCheckError("No benchmark evidence files were found.")
    for path in benchmark_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        versions = _voicehub_versions(payload)
        if not versions:
            raise ReleaseCheckError(f"{path.name} does not record voicehub_version.")
        mismatched = sorted(set(versions) - {version})
        if mismatched:
            raise ReleaseCheckError(f"{path.name} records versions {mismatched}, expected only {version}.")
    return len(benchmark_paths)


def _front_matter_value(source: str, key: str) -> str | None:
    if not source.startswith("---\n"):
        return None
    front_matter = source.split("---\n", 2)[1]
    match = re.search(rf"^{re.escape(key)}:\s*[\"']?([^\n\"']+)[\"']?\s*$", front_matter, re.MULTILINE)
    return match.group(1).strip() if match is not None else None


def validate_documentation_version(version: str, repository_root: Path = REPOSITORY_ROOT) -> None:
    """Keep the release report and versioned install examples aligned."""
    for relative_path in (
            "docs/project/release-readiness.md",
            "docs/project/roadmap.md",
    ):
        path = repository_root / relative_path
        documented_version = _front_matter_value(path.read_text(encoding="utf-8"), "release")
        if documented_version != version:
            raise ReleaseCheckError(
                f"{relative_path} records release {documented_version!r}, expected {version!r}.")

    readme = (repository_root / "README.md").read_text(encoding="utf-8")
    readme_contract = (
        "Unified Inference, Training, and Optimization for TTS, ASR, and VAD",
        "VoiceHub supports Python 3.10 through 3.12.",
    )
    missing_contract = [fragment for fragment in readme_contract if fragment not in readme]
    if missing_contract:
        raise ReleaseCheckError(f"README.md is missing the 0.3 product contract: {missing_contract}")

    documented_wheel_versions: set[str] = set()
    for root in (repository_root / "README.md", repository_root / "docs"):
        paths = (root, ) if root.is_file() else tuple(root.rglob("*.md"))
        for path in paths:
            documented_wheel_versions.update(WHEEL_VERSION_PATTERN.findall(path.read_text(encoding="utf-8")))
    if documented_wheel_versions - {version}:
        raise ReleaseCheckError(
            "Versioned wheel examples disagree with the source version: "
            f"{sorted(documented_wheel_versions)} versus {version}.")


def _read_evidence(repository_root: Path, filename: str) -> dict[str, Any]:
    path = repository_root / "benchmarks" / filename
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReleaseCheckError(f"Could not read benchmark evidence {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ReleaseCheckError(f"Benchmark evidence {path} must contain a JSON object.")
    return payload


def validate_layered_evidence(repository_root: Path = REPOSITORY_ROOT) -> dict[str, int]:
    """Prove the all-provider contract matrix and representative real runs
    coexist."""
    tts = _read_evidence(repository_root, "tts_optimization_rtx4090_2026-07-31.json")
    speech_input = _read_evidence(repository_root, "asr_vad_rtx4090_2026-07-31.json")
    vits = _read_evidence(repository_root, "tts_vits_rtx4090_2026-07-31.json")
    rejected_vui = _read_evidence(repository_root, "tts_vui_rtx4090_rejected_2026-07-31.json")

    tts_providers = tts.get("providers")
    if not isinstance(tts_providers, list) or len(tts_providers) != 34:
        raise ReleaseCheckError("TTS evidence must contain exactly 34 provider rows.")
    tts_types = {str(row.get("model_type", "")) for row in tts_providers if isinstance(row, dict)}
    if len(tts_types) != 34 or "" in tts_types:
        raise ReleaseCheckError("TTS evidence contains missing or duplicate model types.")
    incomplete_tts = []
    for row in tts_providers:
        if not isinstance(row, dict):
            incomplete_tts.append(repr(row))
        elif (row.get("static_plan_status") != "passed" or not row.get("evidence") or not row.get("note")):
            incomplete_tts.append(str(row.get("model_type")))
    incomplete_tts.sort()
    if incomplete_tts:
        raise ReleaseCheckError(f"TTS provider rows lack contract evidence: {incomplete_tts}")

    coverage = speech_input.get("coverage")
    if not isinstance(coverage, list) or len(coverage) != 34:
        raise ReleaseCheckError("ASR/VAD evidence must contain exactly 34 provider rows.")
    speech_input_types = {str(row.get("model_type", "")) for row in coverage if isinstance(row, dict)}
    if len(speech_input_types) != 34 or "" in speech_input_types:
        raise ReleaseCheckError("ASR/VAD evidence contains missing or duplicate model types.")
    task_counts = {
        "asr": sum(model_type.startswith("asr_") for model_type in speech_input_types),
        "vad": sum(model_type.startswith("vad_") for model_type in speech_input_types),
    }
    if task_counts != {"asr": 23, "vad": 11}:
        raise ReleaseCheckError(f"ASR/VAD evidence has unexpected task counts: {task_counts}")

    allowed_verification = {
        "external-checkpoint-blocker",
        "real-algorithm",
        "real-checkpoint",
        "tiny-native-graph",
    }
    invalid_rows = []
    for row in coverage:
        if not isinstance(row, dict) or row.get("verification") not in allowed_verification:
            invalid_rows.append(str(row.get("model_type") if isinstance(row, dict) else row))
        elif (row["verification"] == "external-checkpoint-blocker" and not row.get("blocker")):
            invalid_rows.append(str(row.get("model_type")))
    if invalid_rows:
        raise ReleaseCheckError(f"ASR/VAD evidence has invalid verification rows: {invalid_rows}")
    for prefix in ("asr_", "vad_"):
        if not any(str(row["model_type"]).startswith(prefix) and
                   row["verification"] in {"real-checkpoint", "real-algorithm"} for row in coverage):
            raise ReleaseCheckError(f"No representative real evidence exists for task prefix {prefix!r}.")

    documented_types = {
        path.stem
        for path in (repository_root / "docs" / "models" / "providers").glob("*.md")
        if path.name != "index.md"
    }
    evidence_types = tts_types | speech_input_types
    if documented_types != evidence_types:
        raise ReleaseCheckError(
            "Provider guides and layered evidence disagree: "
            f"missing guides={sorted(evidence_types - documented_types)}, "
            f"missing evidence={sorted(documented_types - evidence_types)}")

    checkpoint = vits.get("checkpoint")
    coverage_summary = vits.get("coverage_summary")
    if (not isinstance(checkpoint, dict) or checkpoint.get("model_type") != "vits" or
            not checkpoint.get("resolved_revision") or not isinstance(coverage_summary, dict) or
            coverage_summary.get("real_checkpoint_inference_models") != 1):
        raise ReleaseCheckError("VITS evidence does not prove a pinned real-checkpoint TTS run.")
    policy = rejected_vui.get("policy")
    if (rejected_vui.get("status") != "rejected" or not isinstance(policy, dict) or
            policy.get("inference_torch_compile") != "rejected"):
        raise ReleaseCheckError("Rejected VUI optimization evidence is not fail-closed.")

    return {
        "tts_providers": len(tts_types),
        "asr_providers": task_counts["asr"],
        "vad_providers": task_counts["vad"],
        "documented_providers": len(documented_types),
    }


def validate_tag(tag: str, version: str) -> None:
    expected = f"v{version}"
    if tag != expected:
        raise ReleaseCheckError(f"Release tag {tag!r} must exactly match {expected!r}.")


def _git_output(repository_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ReleaseCheckError(f"git {' '.join(arguments)} failed: {detail}")
    return completed.stdout.strip()


def validate_tag_at_head(tag: str, repository_root: Path = REPOSITORY_ROOT) -> None:
    """Reject publishing from a branch or from a tag that points elsewhere."""
    tag_commit = _git_output(repository_root, "rev-parse", "--verify", f"refs/tags/{tag}^{{commit}}")
    head_commit = _git_output(repository_root, "rev-parse", "HEAD")
    if tag_commit != head_commit:
        raise ReleaseCheckError(
            f"Tag {tag!r} points to {tag_commit}, but the checked-out commit is {head_commit}.")


def _metadata(text: str, *, source: str) -> dict[str, str]:
    message = Parser().parsestr(text)
    values = {name: str(message.get(name, "")).strip() for name in ("Name", "Version")}
    if not all(values.values()):
        raise ReleaseCheckError(f"{source} is missing Name or Version metadata.")
    return values


def _validate_distribution_metadata(values: dict[str, str], version: str, *, source: str) -> None:
    if values != {"Name": "voicehub", "Version": version}:
        raise ReleaseCheckError(f"{source} metadata is {values}, expected VoiceHub {version}.")


def validate_distributions(
    dist_dir: Path,
    version: str,
) -> dict[str, int]:
    """Verify the exact wheel/sdist pair that the publish job will upload."""
    wheel = dist_dir / f"voicehub-{version}-py3-none-any.whl"
    sdist = dist_dir / f"voicehub-{version}.tar.gz"
    distributions = tuple(sorted(path for path in dist_dir.iterdir() if path.is_file()))
    if len(distributions) != 2 or set(distributions) != {sdist, wheel}:
        raise ReleaseCheckError(
            "The release directory must contain only the expected wheel and sdist; "
            f"found {[path.name for path in distributions]}.")

    sizes = {path.name: path.stat().st_size for path in distributions}
    oversized = {name: size for name, size in sizes.items() if size > PYPI_DEFAULT_FILE_LIMIT}
    if oversized:
        raise ReleaseCheckError(f"Distribution files exceed PyPI's default 100 MB limit: {oversized}")

    with ZipFile(wheel) as archive:
        metadata_path = f"voicehub-{version}.dist-info/METADATA"
        try:
            values = _metadata(
                archive.read(metadata_path).decode("utf-8"),
                source=f"{wheel.name}:{metadata_path}",
            )
        except KeyError as error:
            raise ReleaseCheckError(f"{wheel.name} is missing {metadata_path}.") from error
        _validate_distribution_metadata(values, version, source=wheel.name)

    with tarfile.open(sdist) as archive:
        metadata_path = f"voicehub-{version}/PKG-INFO"
        try:
            member = archive.getmember(metadata_path)
            extracted = archive.extractfile(member)
        except KeyError as error:
            raise ReleaseCheckError(f"{sdist.name} is missing {metadata_path}.") from error
        if extracted is None:
            raise ReleaseCheckError(f"Could not read {metadata_path} from {sdist.name}.")
        values = _metadata(
            extracted.read().decode("utf-8"),
            source=f"{sdist.name}:{metadata_path}",
        )
        _validate_distribution_metadata(values, version, source=sdist.name)
    return sizes


def fetch_pypi_payload(url: str = PYPI_JSON_URL) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "voicehub-release-check/0.3"},
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.load(response)
    except Exception as error:
        raise ReleaseCheckError(f"Could not read the PyPI release state: {error}") from error
    if not isinstance(payload, dict):
        raise ReleaseCheckError("PyPI returned a non-object JSON payload.")
    return payload


def validate_pypi_payload(
    payload: dict[str, Any],
    version: str,
    policy: str,
) -> str:
    """Validate either a not-yet-published candidate or a completed release."""
    try:
        latest = str(payload["info"]["version"])
        releases = payload["releases"]
    except (KeyError, TypeError) as error:
        raise ReleaseCheckError("PyPI payload is missing info.version or releases.") from error
    if not isinstance(releases, dict):
        raise ReleaseCheckError("PyPI releases must be a JSON object.")

    if policy == "candidate":
        if version in releases and releases[version]:
            raise ReleaseCheckError(f"VoiceHub {version} is already present on PyPI.")
        if parse_version(latest) >= parse_version(version):
            raise ReleaseCheckError(f"PyPI latest version {latest} is not older than candidate {version}.")
    elif policy == "published":
        if latest != version or not releases.get(version):
            raise ReleaseCheckError(
                f"PyPI latest version is {latest}; published verification expected {version}.")
    else:
        raise ValueError(f"Unknown PyPI policy: {policy}")
    return latest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", help="Expected release tag, for example v0.3.0.")
    parser.add_argument(
        "--require-tag-at-head",
        action="store_true",
        help="Require --tag to exist and point at the checked-out commit.",
    )
    parser.add_argument(
        "--dist-dir",
        type=Path,
        help="Directory containing the exact wheel and sdist pair to validate.",
    )
    parser.add_argument(
        "--pypi-policy",
        choices=("candidate", "published"),
        help="Validate PyPI before publishing or after the release is visible.",
    )
    return parser.parse_args()


def main() -> None:
    options = parse_args()
    version = source_version()
    report: dict[str, Any] = {"version": version}

    validate_source_metadata(version)
    report["benchmark_files"] = validate_benchmark_versions(version)
    report["layered_evidence"] = validate_layered_evidence()
    validate_documentation_version(version)
    report["source_metadata"] = "passed"
    report["documentation_version"] = "passed"

    if options.require_tag_at_head and not options.tag:
        raise ReleaseCheckError("--require-tag-at-head also requires --tag.")
    if options.tag:
        validate_tag(options.tag, version)
        report["tag"] = options.tag
        if options.require_tag_at_head:
            validate_tag_at_head(options.tag)
            report["tag_at_head"] = "passed"
    if options.dist_dir:
        report["distribution_bytes"] = validate_distributions(options.dist_dir, version)
    if options.pypi_policy:
        payload = fetch_pypi_payload()
        report["pypi_latest"] = validate_pypi_payload(payload, version, options.pypi_policy)
        report["pypi_policy"] = options.pypi_policy

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
