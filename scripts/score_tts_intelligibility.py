#!/usr/bin/env python3
"""Score paired TTS listening files with one independent upstream ASR recipe.

This is a recognizer-based smoke diagnostic, not a perceptual quality or
MOS benchmark. Later --report inputs supersede earlier outputs for the
same model, sample, and side. The ASR recipe must be an
upstream_parity_worker request.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import unicodedata
from pathlib import Path

from benchmark_upstream_parity import error_rate, execution_lock, read_json, run_side, write_json


def normalized(text):
    return " ".join(re.sub(r"[^\w\s]", " ", unicodedata.normalize("NFKC", text).casefold()).split())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--asr-request", type=Path, required=True)
    parser.add_argument("--asr-repository", type=Path, required=True)
    parser.add_argument("--asr-python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selected = {}
    for report_path in args.report:
        for case in read_json(report_path)["cases"]:
            request = case["upstream"].get("request", {})
            if request.get("task") != "tts":
                continue
            texts = {sample["id"]: sample["text"] for sample in request["samples"]}
            for side in ("upstream", "voicehub"):
                result = case[side]
                if result.get("status") != "ok":
                    continue
                for sample in result["samples"]:
                    key = (request["model_type"], sample["id"], side)
                    audio = Path(sample["audio"]).with_suffix(".wav")
                    if not audio.is_file() and sample.get("archived_wav"):
                        audio = next((
                            parent / sample["archived_wav"] for parent in report_path.resolve().parents
                            if (parent / sample["archived_wav"]).is_file()), audio)
                    text = texts[sample["id"]]
                    if request["model_type"] == "dia":
                        text = re.sub(r"\[S\d+\]\s*", "", text)
                    selected[key] = (audio, text, report_path.resolve())
    if not selected:
        parser.error("No successful TTS outputs were found.")
    request = read_json(args.asr_request)
    request.update(device="cpu", threads=1, warmup=0, repeats=1, samples=[])
    request.pop("diagnostics_after_seconds", None)
    root = args.output.resolve().parent
    root.mkdir(parents=True, exist_ok=True)
    sources = {}
    for (model, sample_id, side), (audio, text, report) in selected.items():
        if any(re.fullmatch(r"[A-Za-z0-9_.-]+", value) is None for value in (model, sample_id)):
            raise ValueError("Model and sample identifiers must be safe filename components.")
        identifier = f"{model}--{sample_id}--{side}"
        destination = root / "asr-inputs" / (identifier + ".wav")
        destination.parent.mkdir(parents=True, exist_ok=True)
        resample_command = [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(audio),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(destination),
        ]
        subprocess.run(resample_command, check=True)
        request["samples"].append({
            "id": identifier,
            "audio": str(destination),
            "reference": text,
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        })
        sources[identifier] = {
            "model_type": model,
            "side": side,
            "sample_id": sample_id,
            "report": str(report),
            "source_audio": str(audio),
            "source_audio_sha256": hashlib.sha256(audio.read_bytes()).hexdigest(),
        }
    request_path = root / "asr-request.json"
    write_json(request_path, request)
    project = Path(__file__).resolve().parents[1]
    reference_command = [
        str(args.asr_python.absolute()),
        str(project / "scripts/upstream_parity_worker.py"),
        "upstream",
    ]
    reference = {"cwd": str(args.asr_repository.resolve()), "command": reference_command}
    with execution_lock(project / ".cache/upstream-parity/execution.lock"):
        result = run_side(reference, request_path, root / "asr-results.json", timeout=600)
    scores = []
    for sample in result.get("samples", []):
        reference = normalized(sample["reference"])
        hypothesis = normalized(sample["text"])
        scores.append({
            **sources[sample["id"]],
            "reference": sample["reference"],
            "transcript": sample["text"],
            "wer": error_rate(reference.split(), hypothesis.split()),
            "cer": error_rate(list(reference.replace(" ", "")), list(hypothesis.replace(" ", ""))),
        })
    write_json(
        args.output, {
            "status": result["status"],
            "scope":
            "Single-recognizer intelligibility proxy; not MOS, a listening panel, or full quality parity.",
            "resampling": "FFmpeg mono 16 kHz PCM16, identical conversion for both sides",
            "ffmpeg_version": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0],
            "recognizer_evidence": "asr-results.json",
            "scores": scores,
        })
    return 0 if result["status"] == "ok" and len(scores) == len(selected) else 1


if __name__ == "__main__":
    raise SystemExit(main())
