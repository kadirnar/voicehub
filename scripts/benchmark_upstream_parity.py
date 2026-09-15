#!/usr/bin/env python3
"""Run upstream/VoiceHub commands in separate repositories and compare
evidence.

Commands receive a request JSON path and an output JSON path as their
last arguments. They must report warmed, synchronized inference timings
separately from loading and return one result per input sample. No
missing measurement is treated as a pass. See docs/guides/upstream-
parity.md for the wire format.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def execution_lock(path: Path):
    """Serialize benchmark processes on this host before starting timeouts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if os.name == "nt":
            import msvcrt

            handle.write(b"0")
            handle.flush()
            handle.seek(0)
            while True:
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(0.1)
        else:
            import fcntl

            fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def error_rate(reference: list[str], hypothesis: list[str]) -> float | None:
    if not reference:
        return None
    previous = list(range(len(hypothesis) + 1))
    for row, expected in enumerate(reference, 1):
        current = [row]
        for column, actual in enumerate(hypothesis, 1):
            current.append(
                min(current[-1] + 1, previous[column] + 1, previous[column - 1] + (expected != actual)))
        previous = current
    return previous[-1] / len(reference)


def latency_comparison(upstream: list[float], voicehub: list[float], *, tolerance: float) -> dict:
    """Report a conservative independent-sample 95% normal-approximation
    interval.

    This is a screening flag, not a significance claim for noisy,
    autocorrelated GPU runs. Retain all samples so a longer alternating
    run can confirm it.
    """
    for values in (upstream, voicehub):
        if len(values) < 5 or any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError("At least five finite, positive warm timings are required per side.")
    original, wrapped = statistics.mean(upstream), statistics.mean(voicehub)
    uncertainty = 1.96 * math.sqrt(
        statistics.variance(upstream) / len(upstream) + statistics.variance(voicehub) / len(voicehub))
    return {
        "upstream_mean_seconds": original,
        "voicehub_mean_seconds": wrapped,
        "ratio": wrapped / original,
        "difference_95pct_lower_seconds": wrapped - original - uncertainty,
        "difference_95pct_upper_seconds": wrapped - original + uncertainty,
        "relative_tolerance": tolerance,
        "regression_flag": wrapped - original - uncertainty > tolerance * original,
        "method": "independent means, normal approximation; screening only",
    }


def compare_results(upstream: dict, voicehub: dict, *, timing_tolerance: float = 0.10) -> dict:
    import numpy as np

    if upstream.get("status") != "ok" or voicehub.get("status") != "ok":
        return {"status": "not-comparable", "reason": "Both executions must succeed."}
    if (upstream.get("request_sha256") and voicehub.get("request_sha256") and
            upstream["request_sha256"] != voicehub["request_sha256"]):
        return {"status": "not-comparable", "reason": "The executed requests differ."}
    if upstream["task"] != voicehub["task"]:
        raise ValueError("Task mismatch.")
    left, right = upstream["samples"], voicehub["samples"]
    if not left or [s["id"] for s in left] != [s["id"] for s in right]:
        raise ValueError("Sample IDs/order must match and cannot be empty.")
    rows = []
    for baseline, candidate in zip(left, right):
        row = {
            "id":
            baseline["id"],
            "timing":
            latency_comparison(
                baseline["warm_seconds"], candidate["warm_seconds"], tolerance=timing_tolerance)
        }
        task = upstream["task"]
        if task == "tts":
            a = np.load(baseline["audio"], allow_pickle=False)
            b = np.load(candidate["audio"], allow_pickle=False)
            same_shape = a.shape == b.shape
            finite = bool(np.isfinite(a).all() and np.isfinite(b).all() and a.size and b.size)
            row.update({
                "sample_rate_equal":
                baseline["sample_rate"] == candidate["sample_rate"],
                "shape_equal":
                same_shape,
                "finite_audio":
                finite,
                "upstream_duration_seconds":
                a.size / baseline["sample_rate"],
                "voicehub_duration_seconds":
                b.size / candidate["sample_rate"],
                "waveform_exact":
                bool(finite and np.array_equal(a, b)),
                "waveform_max_absolute_error":
                float(np.max(np.abs(a - b))) if same_shape and finite else None,
                "waveform_rmse":
                float(np.sqrt(np.mean((a.astype(float) - b)**2))) if same_shape and finite else None,
                "upstream_clipping_fraction":
                float(np.mean(np.abs(a) >= 1)) if finite else None,
                "voicehub_clipping_fraction":
                float(np.mean(np.abs(b) >= 1)) if finite else None,
                "perceptual_quality":
                "unmeasured; waveform error is not a listening score",
            })
        elif task == "asr":
            # Deliberately expose both verbatim and normalized equality.
            import re
            import unicodedata

            def normalize(value):
                return " ".join(
                    re.findall(r"[^\W_]+(?:['’][^\W_]+)*",
                               unicodedata.normalize("NFKC", value).casefold()))

            if baseline["reference"] != candidate["reference"]:
                raise ValueError("Reference transcript mismatch.")
            reference = normalize(baseline["reference"])
            row["text_exact"] = baseline["text"] == candidate["text"]
            row["normalized_text_equal"] = normalize(baseline["text"]) == normalize(candidate["text"])
            for name, result in (("upstream", baseline), ("voicehub", candidate)):
                hypothesis = normalize(result["text"])
                row[name + "_wer"] = error_rate(reference.split(), hypothesis.split())
                row[name + "_cer"] = error_rate(
                    list(reference.replace(" ", "")), list(hypothesis.replace(" ", "")))
                row[name + "_text"] = result["text"]
        elif task == "vad":
            row["segments_equal"] = baseline["segments"] == candidate["segments"]
            row["upstream_segments"] = baseline["segments"]
            row["voicehub_segments"] = candidate["segments"]
            a, b = np.asarray(baseline["segments"]), np.asarray(candidate["segments"])
            if not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError("VAD segment boundaries must be finite.")
            same_shape = a.shape == b.shape
            boundary_error = float(np.max(
                np.abs(a - b))) if same_shape and a.size else (0.0 if same_shape else None)
            row["boundary_max_absolute_error_seconds"] = boundary_error
            row["boundaries_equal_within_1ns"] = boundary_error is not None and boundary_error <= 1e-9
            row["ground_truth_error_rate"] = None
            row["ground_truth_note"] = "No human speech-boundary labels; agreement is not accuracy."
            if "probabilities" in baseline and "probabilities" in candidate:
                a, b = np.asarray(baseline["probabilities"]), np.asarray(candidate["probabilities"])
                if not np.isfinite(a).all() or not np.isfinite(b).all():
                    raise ValueError("VAD frame probabilities must be finite.")
                row["probability_shapes_equal"] = a.shape == b.shape
                row["probability_max_absolute_error"] = (
                    float(np.max(np.abs(a - b))) if a.size and a.shape == b.shape else None)
        else:
            raise ValueError(f"Unknown task: {task}")
        rows.append(row)
    return {
        "status": "measured",
        "samples": rows,
        "timing_regression_flag": any(row["timing"]["regression_flag"] for row in rows)
    }


def run_side(spec: dict, request: Path, output: Path, *, timeout: float) -> dict:
    cwd = Path(spec["cwd"]).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    snapshot = output.with_suffix(".request.json")
    snapshot.write_bytes(request.read_bytes())
    command = [*spec["command"], str(snapshot.resolve()), str(output.resolve())]
    # Remove inherited import paths so upstream cannot accidentally resolve the
    # editable VoiceHub runtime. Each command chooses its own environment.
    env = {k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "PYTHONHOME"}}
    env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false"})
    output.unlink(missing_ok=True)
    started = time.perf_counter()
    try:
        with output.with_suffix(".log").open("w") as log:
            result = subprocess.run(
                command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout, check=False)
        record = read_json(output) if output.exists() else {"status": "error", "reason": "No result JSON."}
        if result.returncode:
            record["status"] = "error"
        record["returncode"] = result.returncode
    except (OSError, subprocess.TimeoutExpired, ValueError) as error:
        record = {"status": "error", "reason": str(error)}
    record.update({
        "cwd": str(cwd),
        "command": command,
        "wall_seconds": time.perf_counter() - started,
        "log": str(output.with_suffix(".log")),
        "request_sha256": sha256(snapshot),
        "request": read_json(snapshot)
    })
    write_json(output, record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timing-tolerance", type=float, default=0.10)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep recorded cases, including failures; execute remaining cases.")
    parser.add_argument(
        "--lock-file",
        type=Path,
        default=Path(__file__).resolve().parents[1] / ".cache/upstream-parity/execution.lock")
    args = parser.parse_args()
    if not math.isfinite(args.timing_tolerance) or args.timing_tolerance < 0:
        parser.error("--timing-tolerance must be finite and non-negative")
    manifest = read_json(args.manifest)
    records = []
    previous = ({
        r["name"]: r
        for r in read_json(args.output)["cases"]
    } if args.resume and args.output.exists() else {})
    names = [case["name"] for case in manifest["cases"]]
    if len(names) != len(set(names)) or any(not n or Path(n).name != n or n in {".", ".."} for n in names):
        parser.error("Case names must be unique path basenames")
    for case in manifest["cases"]:
        if case["name"] in previous:
            records.append(previous[case["name"]])
            continue
        record = {"name": case["name"], "provenance": case.get("provenance", {})}
        # Freeze once for both sides, even if a checkpoint preparation process
        # updates the manifest's request while a benchmark waits for the lock.
        request = args.output.parent / case["name"] / "request.json"
        request.parent.mkdir(parents=True, exist_ok=True)
        request.write_bytes(Path(case["request"]).read_bytes())
        for side in ("upstream", "voicehub"):
            if side not in case:
                record[side] = {"status": "not-run", "reason": case.get("reason", "No command configured.")}
                continue
            with execution_lock(args.lock_file):
                record[side] = run_side(
                    case[side],
                    request,
                    args.output.parent / case["name"] / (side + ".json"),
                    timeout=case.get("timeout_seconds", 600))
        try:
            record["comparison"] = compare_results(
                record["upstream"], record["voicehub"], timing_tolerance=args.timing_tolerance)
        except (KeyError, ValueError, OSError) as error:
            record["comparison"] = {"status": "not-comparable", "reason": str(error)}
        records.append(record)
        write_json(args.output, {"schema_version": 1, "cases": records})
        print(case["name"], record["comparison"]["status"], flush=True)
    return 0 if records and all(r["comparison"]["status"] == "measured" for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
