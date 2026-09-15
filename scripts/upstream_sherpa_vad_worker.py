#!/usr/bin/env python3
"""Run the original Sherpa C++ detector without adding PyTorch to its
runtime."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path


def main():
    request_path, output_path = map(Path, sys.argv[1:])
    request = json.loads(request_path.read_text())
    report = {"status": "error", "task": "vad", "samples": []}
    try:
        import numpy as np
        import soundfile as sf

        # The VAD-only upstream build omits TTS symbols that its package
        # __init__ imports unconditionally. Load the original compiled VAD
        # bindings directly, without modifying any detector implementation.
        package = importlib.util.find_spec("sherpa_onnx")
        libraries = list((Path(package.origin).parent / "lib").glob("_sherpa_onnx*.so"))
        if len(libraries) != 1:
            raise RuntimeError("Expected one original compiled Sherpa extension.")
        extension = importlib.util.spec_from_file_location("_sherpa_onnx", libraries[0])
        sherpa_onnx = importlib.util.module_from_spec(extension)
        extension.loader.exec_module(sherpa_onnx)
        options = request["generation"]
        config = sherpa_onnx.VadModelConfig()
        config.silero_vad.model = request["upstream_checkpoint"]
        config.silero_vad.threshold = options["threshold"]
        config.silero_vad.min_silence_duration = options["min_silence_duration_ms"] / 1000
        config.silero_vad.min_speech_duration = options["min_speech_duration_ms"] / 1000
        config.silero_vad.max_speech_duration = options["max_speech_duration_s"]
        config.sample_rate = 16000
        config.num_threads = request["threads"]
        config.provider = "cpu"
        if options["speech_pad_ms"] != 0:
            raise ValueError("This original Sherpa recipe requires zero external speech padding.")
        report["runtime"] = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "device": "cpu",
            "threads": request["threads"],
            "worker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "repository_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "packages": {
                d.metadata["Name"]: d.version
                for d in importlib.metadata.distributions()
            },
            "backend": "original Sherpa C++ with ONNX Runtime; no Torch backend",
            "build_configuration": "TTS, tests, and standalone binaries disabled; direct VAD bindings",
            "extension_sha256": hashlib.sha256(libraries[0].read_bytes()).hexdigest(),
        }
        checkpoint = Path(request["upstream_checkpoint"])
        report["checkpoint_sha256"] = {str(checkpoint): hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
        started = time.perf_counter()
        detector = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=60)
        report["load_seconds"] = time.perf_counter() - started
        for sample in request["samples"]:
            audio_path = Path(sample["audio"])
            digest = hashlib.sha256(audio_path.read_bytes()).hexdigest()
            if digest != sample["sha256"]:
                raise ValueError("Public audio fixture hash mismatch.")
            waveform, sample_rate = sf.read(audio_path, dtype="float32")
            if sample_rate != 16000 or waveform.ndim != 1:
                raise ValueError("Expected the shared mono 16 kHz evaluation waveform.")
            waveform = np.ascontiguousarray(waveform)
            timings = []
            for index in range(request["warmup"] + request["repeats"]):
                started = time.perf_counter()
                detector.reset()
                # Follow the original file-inference example's window-sized
                # ingestion. The C++ detector's segment timestamps depend on
                # incremental buffer positions, not one whole-file push.
                window = config.silero_vad.window_size
                for start in range(0, waveform.size, window):
                    detector.accept_waveform(waveform[start:start + window])
                detector.flush()
                segments = []
                while not detector.empty():
                    segment = detector.front
                    segments.append(
                        [segment.start / sample_rate, (segment.start + len(segment.samples)) / sample_rate])
                    detector.pop()
                elapsed = time.perf_counter() - started
                if index >= request["warmup"]:
                    timings.append(elapsed)
            report["samples"].append({
                "id": sample["id"],
                "input_sha256": digest,
                "warm_seconds": timings,
                "segments": segments
            })
        report["status"] = "ok"
    except Exception as error:
        report.update(error_type=type(error).__name__, error=str(error))
        traceback.print_exc()
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
