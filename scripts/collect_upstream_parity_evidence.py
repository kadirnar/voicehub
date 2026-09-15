#!/usr/bin/env python3
"""Collect completed audit attempts without promoting missing evidence to
passes."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--run", action="append", required=True, help="Run directory relative to cache")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    inventory = read(args.cache / "checkouts.json")
    policy_path = args.cache / "audit-policy.json"
    policy = read(policy_path) if policy_path.exists() else {}
    pending_access = {row["model_type"]: row for row in policy.get("pending_access", [])}
    if policy:
        write(args.output / "audit-policy.json", policy)
    attempts = {row["model_type"]: [] for row in inventory}
    for name in args.run:
        path = args.cache / name / "report.json"
        if not path.exists():
            continue
        report = read(path)
        target = args.output / "runs" / name
        target.mkdir(parents=True, exist_ok=True)
        replay = []
        # Preserve raw execution metadata, even when its paths are specific to
        # this audit host. Copy listening outputs to durable relative paths.
        for case in report["cases"]:
            request = case["voicehub"].get("request", case["upstream"].get("request"))
            if request is not None:
                request_path = target / case["name"] / "request.json"
                write(request_path, request)
                replay.append({
                    "name": case["name"],
                    "request": str(request_path.resolve()),
                    "provenance": case.get("provenance", {}),
                    **{
                        side: {
                            "cwd": case[side]["cwd"],
                            "command": case[side]["command"][:-2]
                        }
                        for side in ("upstream", "voicehub") if "command" in case[side]
                    },
                })
            model_type = case.get("provenance", {}).get("model_type", case["name"])
            if model_type in attempts:
                attempts[model_type].append({
                    "run": name,
                    "case": case["name"],
                    "comparison": case["comparison"],
                    **{
                        side: {
                            key: case[side][key]
                            for key in ("status", "error", "reason", "returncode", "error_type") if key in case[side]
                        }
                        for side in ("upstream", "voicehub")
                    },
                })
            for side in ("upstream", "voicehub"):
                run = case[side]
                for sample in run.get("samples", []):
                    if "audio" not in sample:
                        continue
                    original = Path(sample["audio"])
                    for suffix in (".npy", ".wav"):
                        source = original.with_suffix(suffix)
                        if source.exists():
                            destination = target / case["name"] / source.name
                            destination.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copyfile(source, destination)
                            sample["archived_" + suffix[1:]] = str(destination.relative_to(args.output))
                if run.get("log") and Path(run["log"]).is_file():
                    destination = target / case["name"] / (side + ".txt")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    # Keep bounded diagnostic tails; full logs remain in cache.
                    destination.write_text(Path(run["log"]).read_text(errors="replace")[-20000:])
            write(target / "report.json", report)
        write(target / "replay.json", {"cases": replay})
    rows = []
    for original in inventory:
        row = dict(original)
        row["attempts"] = attempts[row["model_type"]]
        row["paired_measurement_available"] = any(
            a["comparison"]["status"] == "measured" for a in row["attempts"])
        row["full_quality_parity_established"] = False
        if row["model_type"] in pending_access:
            row["checkpoint_access"] = pending_access[row["model_type"]]
        rows.append(row)
    root = Path(__file__).resolve().parents[1]
    source_hash = hashlib.sha256()
    for path in sorted(p for p in (root / "voicehub").rglob("*") if p.suffix in {".py", ".c", ".cc", ".h"}):
        source_hash.update(str(path.relative_to(root)).encode() + b"\0")
        source_hash.update(path.read_bytes())
    result = {
        "schema_version": 1,
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "voicehub_source_tree_sha256": source_hash.hexdigest(),
        "working_tree_modified": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root)),
        "scope": "Partial execution evidence; a checkout, install, or failed attempt is not model parity.",
        "registered_models": len(rows),
        "models_with_paired_measurements": sum(r["paired_measurement_available"] for r in rows),
        "models": rows,
    }
    write(args.output / "inventory.json", result)
    for filename in ("environments.json", "prefetch.json"):
        source = args.cache / filename
        if source.exists():
            shutil.copyfile(source, args.output / filename)
    public_samples = read(args.cache / "samples/manifest.json")
    public_samples["license"] = "CC-BY-4.0"
    public_samples["corpus_source"] = "https://www.openslr.org/12/"
    for sample in public_samples["samples"]:
        source = Path(sample["audio"])
        destination = args.output / "audio" / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        sample["archived_audio"] = str(destination.relative_to(args.output))
        pcm_source = source.with_name(source.stem + "-pcm16.wav")
        if pcm_source.exists():
            pcm_destination = destination.with_name(pcm_source.name)
            shutil.copyfile(pcm_source, pcm_destination)
            sample["archived_pcm16_audio"] = str(pcm_destination.relative_to(args.output))
            sample["pcm16_sha256"] = hashlib.sha256(pcm_destination.read_bytes()).hexdigest()
    write(args.output / "public-samples.json", public_samples)
    print(json.dumps({key: result[key] for key in ("registered_models", "models_with_paired_measurements")}))


if __name__ == "__main__":
    main()
