#!/usr/bin/env python3
"""Run the complete visual documentation contract in concurrent viewport
shards."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
VISUAL_CHECK_PATH = REPOSITORY_ROOT / "scripts" / "check_documentation_visual.py"
VIEWPORT_NAMES = ("desktop", "tablet", "mobile")
PALETTE_NAMES = ("default", "slate")
SHARED_SUMMARY_FIELDS = ("axe_core", "palettes", "representative_routes")
EXPECTED_TOTALS = {
    "accessibility_cases": 60,
    "cases": 60,
    "contribution_cases": 6,
    "contribution_interaction_cases": 6,
    "focus_cycle_cases": 60,
    "home_cases": 6,
    "home_interaction_cases": 6,
    "installation_cases": 6,
    "installation_code_interaction_cases": 6,
    "installation_page_interaction_cases": 6,
    "interactive_accessibility_cases": 30,
    "keyboard_activation_cases": 2,
    "keyboard_cases": 348,
    "language_activation_cases": 40,
    "language_interaction_accessibility_cases": 40,
    "language_keyboard_activation_cases": 20,
    "language_pointer_activation_cases": 20,
    "model_api_cases": 6,
    "model_api_interaction_cases": 6,
    "model_index_cases": 6,
    "model_index_interaction_cases": 6,
    "nested_branch_activation_cases": 36,
    "nested_branch_interaction_accessibility_cases": 36,
    "nested_branch_keyboard_activation_cases": 18,
    "nested_branch_pointer_activation_cases": 18,
    "optimization_cases": 6,
    "optimization_interaction_cases": 6,
    "page_action_back_to_top_activations": 60,
    "page_action_cases": 60,
    "page_action_edit_activations": 60,
    "page_action_footer_activations": 114,
    "page_action_interaction_accessibility_cases": 60,
    "page_action_keyboard_cases": 30,
    "page_action_pointer_cases": 30,
    "pipeline_cases": 6,
    "pipeline_interaction_cases": 6,
    "quickstart_cases": 6,
    "quickstart_interaction_cases": 6,
    "quickstart_page_interaction_cases": 6,
    "root_branch_activation_cases": 32,
    "root_branch_interaction_accessibility_cases": 32,
    "root_branch_keyboard_activation_cases": 16,
    "root_branch_pointer_activation_cases": 16,
    "search_activation_cases": 60,
    "search_interaction_accessibility_cases": 60,
    "search_keyboard_activation_cases": 40,
    "search_pointer_activation_cases": 20,
    "screenshot_cases": 60,
    "speecht5_cases": 6,
    "speecht5_interaction_cases": 6,
    "source_activation_cases": 40,
    "source_interaction_accessibility_cases": 40,
    "source_keyboard_activation_cases": 20,
    "source_pointer_activation_cases": 20,
    "theme_activation_cases": 40,
    "theme_interaction_accessibility_cases": 40,
    "theme_keyboard_activation_cases": 20,
    "theme_pointer_activation_cases": 20,
    "toc_activation_cases": 40,
    "toc_interaction_accessibility_cases": 40,
    "toc_keyboard_activation_cases": 20,
    "toc_pointer_activation_cases": 20,
    "trainer_cases": 6,
    "trainer_interaction_cases": 6,
    "version_activation_cases": 60,
    "version_interaction_accessibility_cases": 60,
    "version_keyboard_activation_cases": 30,
    "version_pointer_activation_cases": 30,
    "viewports": 3,
}
VIEWPORT_SPECIFIC_FIELDS = (
    "keyboard_activation_cases",
    "keyboard_cases",
    "language_activation_cases",
    "language_interaction_accessibility_cases",
    "language_keyboard_activation_cases",
    "language_pointer_activation_cases",
    "nested_branch_activation_cases",
    "nested_branch_interaction_accessibility_cases",
    "nested_branch_keyboard_activation_cases",
    "nested_branch_pointer_activation_cases",
    "root_branch_activation_cases",
    "root_branch_interaction_accessibility_cases",
    "root_branch_keyboard_activation_cases",
    "root_branch_pointer_activation_cases",
    "search_keyboard_activation_cases",
    "search_pointer_activation_cases",
    "source_activation_cases",
    "source_interaction_accessibility_cases",
    "source_keyboard_activation_cases",
    "source_pointer_activation_cases",
    "theme_activation_cases",
    "theme_interaction_accessibility_cases",
    "theme_keyboard_activation_cases",
    "theme_pointer_activation_cases",
    "toc_activation_cases",
    "toc_interaction_accessibility_cases",
    "toc_keyboard_activation_cases",
    "toc_pointer_activation_cases",
)
NON_MOBILE_SPECIFIC_EXPECTATIONS = {
    "keyboard_activation_cases": 0,
    "language_activation_cases": 20,
    "language_interaction_accessibility_cases": 20,
    "language_keyboard_activation_cases": 10,
    "language_pointer_activation_cases": 10,
    "nested_branch_activation_cases": 18,
    "nested_branch_interaction_accessibility_cases": 18,
    "nested_branch_keyboard_activation_cases": 9,
    "nested_branch_pointer_activation_cases": 9,
    "root_branch_activation_cases": 16,
    "root_branch_interaction_accessibility_cases": 16,
    "root_branch_keyboard_activation_cases": 8,
    "root_branch_pointer_activation_cases": 8,
    "search_keyboard_activation_cases": 20,
    "search_pointer_activation_cases": 0,
    "source_activation_cases": 20,
    "source_interaction_accessibility_cases": 20,
    "source_keyboard_activation_cases": 10,
    "source_pointer_activation_cases": 10,
    "theme_activation_cases": 20,
    "theme_interaction_accessibility_cases": 20,
    "theme_keyboard_activation_cases": 10,
    "theme_pointer_activation_cases": 10,
}
VIEWPORT_SPECIFIC_EXPECTATIONS = {
    "desktop": {
        **NON_MOBILE_SPECIFIC_EXPECTATIONS,
        "keyboard_cases": 151,
        "toc_activation_cases": 40,
        "toc_interaction_accessibility_cases": 40,
        "toc_keyboard_activation_cases": 20,
        "toc_pointer_activation_cases": 20,
    },
    "tablet": {
        **NON_MOBILE_SPECIFIC_EXPECTATIONS,
        "keyboard_cases": 131,
        "toc_activation_cases": 0,
        "toc_interaction_accessibility_cases": 0,
        "toc_keyboard_activation_cases": 0,
        "toc_pointer_activation_cases": 0,
    },
    "mobile": {
        "keyboard_activation_cases": 2,
        "keyboard_cases": 66,
        "language_activation_cases": 0,
        "language_interaction_accessibility_cases": 0,
        "language_keyboard_activation_cases": 0,
        "language_pointer_activation_cases": 0,
        "nested_branch_activation_cases": 0,
        "nested_branch_interaction_accessibility_cases": 0,
        "nested_branch_keyboard_activation_cases": 0,
        "nested_branch_pointer_activation_cases": 0,
        "root_branch_activation_cases": 0,
        "root_branch_interaction_accessibility_cases": 0,
        "root_branch_keyboard_activation_cases": 0,
        "root_branch_pointer_activation_cases": 0,
        "search_keyboard_activation_cases": 0,
        "search_pointer_activation_cases": 20,
        "source_activation_cases": 0,
        "source_interaction_accessibility_cases": 0,
        "source_keyboard_activation_cases": 0,
        "source_pointer_activation_cases": 0,
        "theme_activation_cases": 0,
        "theme_interaction_accessibility_cases": 0,
        "theme_keyboard_activation_cases": 0,
        "theme_pointer_activation_cases": 0,
        "toc_activation_cases": 0,
        "toc_interaction_accessibility_cases": 0,
        "toc_keyboard_activation_cases": 0,
        "toc_pointer_activation_cases": 0,
    },
}
MINIMUM_FOCUS_STEPS_BY_VIEWPORT = {
    "desktop": 1700,
    "tablet": 1550,
    "mobile": 1250,
}
PALETTE_METHOD_CASE_FIELDS = (
    ("language_keyboard_activation_cases", "language_pointer_activation_cases"),
    ("nested_branch_keyboard_activation_cases", "nested_branch_pointer_activation_cases"),
    ("page_action_keyboard_cases", "page_action_pointer_cases"),
    ("root_branch_keyboard_activation_cases", "root_branch_pointer_activation_cases"),
    ("source_keyboard_activation_cases", "source_pointer_activation_cases"),
    ("theme_keyboard_activation_cases", "theme_pointer_activation_cases"),
    ("version_keyboard_activation_cases", "version_pointer_activation_cases"),
)
KEYBOARD_CASE_FIELDS = (
    "contribution_interaction_cases",
    "focus_cycle_cases",
    "home_interaction_cases",
    "installation_code_interaction_cases",
    "installation_page_interaction_cases",
    "keyboard_activation_cases",
    "language_keyboard_activation_cases",
    "model_api_interaction_cases",
    "model_index_interaction_cases",
    "nested_branch_keyboard_activation_cases",
    "optimization_interaction_cases",
    "page_action_keyboard_cases",
    "pipeline_interaction_cases",
    "quickstart_interaction_cases",
    "quickstart_page_interaction_cases",
    "root_branch_keyboard_activation_cases",
    "search_keyboard_activation_cases",
    "source_keyboard_activation_cases",
    "speecht5_interaction_cases",
    "theme_keyboard_activation_cases",
    "toc_keyboard_activation_cases",
    "trainer_interaction_cases",
    "version_keyboard_activation_cases",
)
MINIMUM_FOCUS_STEPS_BY_VIEWPORT_PALETTE = {
    "desktop": {
        "default": 850,
        "slate": 850,
    },
    "tablet": {
        "default": 775,
        "slate": 775,
    },
    "mobile": {
        "default": 626,
        "slate": 624,
    },
}


class DocumentationVisualShardError(RuntimeError):
    """Raised when a shard fails or the aggregate loses contract coverage."""


@dataclass(frozen=True, slots=True)
class ShardResult:
    """Capture one viewport process and its elapsed wall time."""

    viewport: str
    returncode: int
    elapsed_seconds: float
    stdout: str
    stderr: str


def _run_shard(
    viewport: str,
    site_directory: Path,
    screenshot_baselines_path: Path | None,
    palette: str | None = None,
) -> ShardResult:
    command = [
        sys.executable,
        str(VISUAL_CHECK_PATH),
        str(site_directory),
        "--viewport",
        viewport,
    ]
    if palette is not None:
        command.extend(("--palette", palette))
    if screenshot_baselines_path is not None:
        command.extend(("--screenshot-baselines", str(screenshot_baselines_path)))
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return ShardResult(
        viewport=viewport,
        returncode=completed.returncode,
        elapsed_seconds=time.monotonic() - started,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _parse_summaries(results: tuple[ShardResult, ...]) -> dict[str, dict[str, Any]]:
    summaries = {}
    failures = []
    for result in results:
        if result.returncode:
            failures.append(
                f"{result.viewport} exited {result.returncode}\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
            continue
        try:
            summary = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            failures.append(
                f"{result.viewport} returned invalid JSON: {error}\nstdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}")
            continue
        if summary.get("viewports") != 1:
            failures.append(
                f"{result.viewport} reported {summary.get('viewports')!r} viewports instead of 1.")
            continue
        summaries[result.viewport] = summary
    if failures:
        raise DocumentationVisualShardError("\n\n".join(failures))
    return summaries


def _aggregate_summaries(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if set(summaries) != set(VIEWPORT_NAMES):
        raise DocumentationVisualShardError(f"Viewport shard inventory differs: {sorted(summaries)!r}.")

    first = summaries[VIEWPORT_NAMES[0]]
    shared = {field: first[field] for field in SHARED_SUMMARY_FIELDS}
    for viewport, summary in summaries.items():
        for field, expected in shared.items():
            if summary.get(field) != expected:
                raise DocumentationVisualShardError(
                    f"{viewport} reported {field}={summary.get(field)!r}; expected {expected!r}.")

    totals = {}
    numeric_fields = set().union(*(summary.keys() for summary in summaries.values()))
    numeric_fields.difference_update(SHARED_SUMMARY_FIELDS)
    for field in sorted(numeric_fields):
        values = [summaries[viewport].get(field) for viewport in VIEWPORT_NAMES]
        if not all(isinstance(value, int) and not isinstance(value, bool) for value in values):
            raise DocumentationVisualShardError(f"Cannot aggregate non-integer field {field!r}: {values!r}.")
        totals[field] = sum(values)

    mismatches = {
        field: {
            "actual": totals.get(field),
            "expected": expected
        }
        for field, expected in EXPECTED_TOTALS.items() if totals.get(field) != expected
    }
    if mismatches:
        raise DocumentationVisualShardError(f"Aggregated visual contract coverage differs: {mismatches!r}.")
    if totals.get("focus_steps", 0) < 4500:
        raise DocumentationVisualShardError(
            f"Aggregated native focus coverage is unexpectedly low: {totals.get('focus_steps')!r}.")

    return {
        **shared,
        "totals": totals,
    }


def _expected_viewport_summary(viewport: str) -> dict[str, int]:
    specific_expectations = VIEWPORT_SPECIFIC_EXPECTATIONS[viewport]
    if set(specific_expectations) != set(VIEWPORT_SPECIFIC_FIELDS):
        raise DocumentationVisualShardError(
            f"{viewport} viewport-specific coverage fields differ: "
            f"{sorted(set(specific_expectations) ^ set(VIEWPORT_SPECIFIC_FIELDS))!r}.")
    for field in VIEWPORT_SPECIFIC_FIELDS:
        total = sum(expectations[field] for expectations in VIEWPORT_SPECIFIC_EXPECTATIONS.values())
        if total != EXPECTED_TOTALS[field]:
            raise DocumentationVisualShardError(
                f"Viewport-specific coverage for {field!r} totals {total!r}; "
                f"expected {EXPECTED_TOTALS[field]!r}.")

    expected = {}
    for field, total in EXPECTED_TOTALS.items():
        if field in specific_expectations:
            expected[field] = specific_expectations[field]
            continue
        if total % len(VIEWPORT_NAMES):
            raise DocumentationVisualShardError(
                f"Shared coverage field {field!r} cannot be divided across viewports: {total!r}.")
        expected[field] = total // len(VIEWPORT_NAMES)
    return expected


def _validate_shared_summary(label: str, summary: dict[str, Any], *, palettes: int) -> None:
    shared_mismatches = {
        "palettes": {
            "actual": summary.get("palettes"),
            "expected": palettes,
        },
        "representative_routes": {
            "actual": summary.get("representative_routes"),
            "expected": 10,
        },
    }
    shared_mismatches = {
        field: values
        for field, values in shared_mismatches.items() if values["actual"] != values["expected"]
    }
    if not isinstance(summary.get("axe_core"), str) or summary["axe_core"] in ("", "unknown"):
        shared_mismatches["axe_core"] = {
            "actual": summary.get("axe_core"),
            "expected": "a detected Axe engine version",
        }
    if shared_mismatches:
        raise DocumentationVisualShardError(
            f"{label} shared visual contract coverage differs: {shared_mismatches!r}.")


def _validate_viewport_summary(viewport: str, summary: dict[str, Any]) -> dict[str, Any]:
    _validate_shared_summary(viewport, summary, palettes=len(PALETTE_NAMES))

    expected = _expected_viewport_summary(viewport)
    mismatches = {
        field: {
            "actual": summary.get(field),
            "expected": value,
        }
        for field, value in expected.items() if summary.get(field) != value
    }
    if mismatches:
        raise DocumentationVisualShardError(f"{viewport} visual contract coverage differs: {mismatches!r}.")

    minimum_focus_steps = MINIMUM_FOCUS_STEPS_BY_VIEWPORT[viewport]
    if summary.get("focus_steps", 0) < minimum_focus_steps:
        raise DocumentationVisualShardError(
            f"{viewport} native focus coverage is unexpectedly low: "
            f"{summary.get('focus_steps')!r}; expected at least {minimum_focus_steps}.")
    return summary


def _expected_viewport_palette_summary(viewport: str, palette: str) -> dict[str, int]:
    viewport_expected = _expected_viewport_summary(viewport)
    expected = {}
    method_fields = {field for pair in PALETTE_METHOD_CASE_FIELDS for field in pair}
    for field, value in viewport_expected.items():
        if field == "viewports":
            expected[field] = value
            continue
        if field == "keyboard_cases" or field in method_fields:
            continue
        if value % len(PALETTE_NAMES):
            raise DocumentationVisualShardError(
                f"{viewport} coverage field {field!r} cannot be divided across palettes: {value!r}.")
        expected[field] = value // len(PALETTE_NAMES)

    for keyboard_field, pointer_field in PALETTE_METHOD_CASE_FIELDS:
        cases = (viewport_expected[keyboard_field] + viewport_expected[pointer_field]) // len(PALETTE_NAMES)
        expected[keyboard_field] = cases if palette == "default" else 0
        expected[pointer_field] = cases if palette == "slate" else 0
    expected["keyboard_cases"] = sum(expected[field] for field in KEYBOARD_CASE_FIELDS)
    return expected


def _validate_viewport_palette_summary(
    viewport: str,
    palette: str,
    summary: dict[str, Any],
) -> dict[str, Any]:
    label = f"{viewport}/{palette}"
    _validate_shared_summary(label, summary, palettes=1)
    expected = _expected_viewport_palette_summary(viewport, palette)
    mismatches = {
        field: {
            "actual": summary.get(field),
            "expected": value,
        }
        for field, value in expected.items() if summary.get(field) != value
    }
    if mismatches:
        raise DocumentationVisualShardError(f"{label} visual contract coverage differs: {mismatches!r}.")

    minimum_focus_steps = MINIMUM_FOCUS_STEPS_BY_VIEWPORT_PALETTE[viewport][palette]
    if summary.get("focus_steps", 0) < minimum_focus_steps:
        raise DocumentationVisualShardError(
            f"{label} native focus coverage is unexpectedly low: "
            f"{summary.get('focus_steps')!r}; expected at least {minimum_focus_steps}.")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "site_directory",
        nargs="?",
        type=Path,
        default=Path("site"),
        help="MkDocs output directory (default: site)",
    )
    parser.add_argument(
        "--screenshot-baselines",
        type=Path,
        help="Screenshot signature manifest passed to every viewport shard",
    )
    parser.add_argument(
        "--viewport",
        choices=VIEWPORT_NAMES,
        help="Validate one fail-closed viewport shard (default: run and aggregate all viewports)",
    )
    parser.add_argument(
        "--palette",
        choices=PALETTE_NAMES,
        help="Validate one fail-closed palette within a selected viewport",
    )
    args = parser.parse_args()
    if args.palette and not args.viewport:
        parser.error("--palette requires --viewport")
    site_directory = args.site_directory.resolve()
    screenshot_baselines_path = (args.screenshot_baselines.resolve() if args.screenshot_baselines else None)
    selected_viewports = (args.viewport, ) if args.viewport else VIEWPORT_NAMES
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=len(selected_viewports)) as executor:
        futures = {
            executor.submit(
                _run_shard,
                viewport,
                site_directory,
                screenshot_baselines_path,
                args.palette,
            ):
            viewport
            for viewport in selected_viewports
        }
        results_by_viewport = {futures[future]: future.result() for future in as_completed(futures)}
    results = tuple(results_by_viewport[viewport] for viewport in selected_viewports)
    try:
        summaries = _parse_summaries(results)
        if args.viewport and args.palette:
            summary = _validate_viewport_palette_summary(
                args.viewport,
                args.palette,
                summaries[args.viewport],
            )
            aggregate = {field: summary[field] for field in SHARED_SUMMARY_FIELDS}
            aggregate["totals"] = {field: summary[field] for field in EXPECTED_TOTALS}
        elif args.viewport:
            summary = _validate_viewport_summary(args.viewport, summaries[args.viewport])
            aggregate = {field: summary[field] for field in SHARED_SUMMARY_FIELDS}
            aggregate["totals"] = {field: summary[field] for field in EXPECTED_TOTALS}
        else:
            aggregate = _aggregate_summaries(summaries)
    except DocumentationVisualShardError as error:
        print(str(error), file=sys.stderr)
        return 1

    aggregate["elapsed_seconds"] = round(time.monotonic() - started, 3)
    aggregate["shards"] = {
        result.viewport: {
            "elapsed_seconds": round(result.elapsed_seconds, 3),
            "summary": summaries[result.viewport],
        }
        for result in results
    }
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
