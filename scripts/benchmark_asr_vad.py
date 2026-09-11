#!/usr/bin/env python3
"""Reproducible ASR/VAD registry audit and isolated inference benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import time
import unicodedata
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_ASR_PROFILES = {
    "fp32": ("float32", None),
    "fp16": ("float16", None),
    "bf16": ("bfloat16", None),
    "compile": ("float32", "torch-compile"),
}
_VAD_PROFILES = {
    "eager": (None, None),
    "eager-1-thread": (None, 1),
    "compile": ("torch-compile", None),
    "compile-1-thread": ("torch-compile", 1),
}
_HIGH_LATENCY_VARIABILITY_CV = 0.10


def _json_dump(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )


def _word_tokens(text: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return tuple(re.findall(
        r"[^\W_]+(?:['’][^\W_]+)*",
        normalized,
        flags=re.UNICODE,
    ))


def _character_tokens(text: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return tuple(character for character in normalized if unicodedata.category(character)[0] in {"L", "N"})


def _error_rate(
    expected: tuple[str, ...],
    actual: tuple[str, ...],
) -> float | None:
    if not expected:
        return None
    previous = list(range(len(actual) + 1))
    for row, expected_word in enumerate(expected, 1):
        current = [row]
        for column, actual_word in enumerate(actual, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (expected_word != actual_word),
                ))
        previous = current
    return previous[-1] / len(expected)


def _word_error_rate(reference: str, hypothesis: str) -> float | None:
    return _error_rate(
        _word_tokens(reference),
        _word_tokens(hypothesis),
    )


def _character_error_rate(reference: str, hypothesis: str) -> float | None:
    return _error_rate(
        _character_tokens(reference),
        _character_tokens(hypothesis),
    )


def _latency_statistics(values: list[float]) -> dict[str, Any]:
    mean = statistics.fmean(values)
    deviation = statistics.pstdev(values)
    coefficient = deviation / mean if mean else 0.0
    return {
        "high_latency_variability": (coefficient >= _HIGH_LATENCY_VARIABILITY_CV),
        "warm_latency_coefficient_of_variation": coefficient,
        "warm_latency_max_seconds": max(values),
        "warm_latency_mean_seconds": mean,
        "warm_latency_min_seconds": min(values),
        "warm_latency_stdev_seconds": deviation,
    }


def _sync_device(device: str) -> None:
    if device.partition(":")[0].lower() != "cuda":
        return
    import torch

    torch.cuda.synchronize(device)


def _peak_memory_mib(device: str) -> float:
    if device.partition(":")[0].lower() == "cuda":
        import torch

        return torch.cuda.max_memory_allocated(device) / 2**20
    if os.name == "nt":  # pragma: no cover - exercised by Windows CI
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = (
                ("cb", wintypes.DWORD),
                ("page_fault_count", wintypes.DWORD),
                ("peak_working_set_size", ctypes.c_size_t),
                ("working_set_size", ctypes.c_size_t),
                ("quota_peak_paged_pool_usage", ctypes.c_size_t),
                ("quota_paged_pool_usage", ctypes.c_size_t),
                ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
                ("quota_non_paged_pool_usage", ctypes.c_size_t),
                ("pagefile_usage", ctypes.c_size_t),
                ("peak_pagefile_usage", ctypes.c_size_t),
            )

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = (
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        )
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(
                kernel32.GetCurrentProcess(),
                ctypes.byref(counters),
                counters.cb,
        ):
            error = ctypes.get_last_error()
            raise OSError(error, "GetProcessMemoryInfo failed")
        return counters.peak_working_set_size / 2**20
    try:
        import resource
    except ImportError as error:  # pragma: no cover - Unix CI/runtime
        raise RuntimeError(
            "CPU peak-memory measurement requires Python's Unix "
            "`resource` module on this platform.") from error

    maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB; macOS reports bytes.
    divisor = 2**10 if sys.platform.startswith("linux") else 2**20
    return maximum / divisor


def _memory_metric(device: str) -> str:
    if device.partition(":")[0].lower() == "cuda":
        return "cuda-max-memory-allocated"
    if os.name == "nt":
        return "process-peak-working-set"
    return "process-max-rss"


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reset_peak_memory(device: str) -> None:
    if device.partition(":")[0].lower() != "cuda":
        return
    import torch

    torch.cuda.reset_peak_memory_stats(device)


def _runtime_metadata() -> dict[str, Any]:
    import torch

    from voicehub import __version__

    values: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "voicehub_version": __version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        values.update({
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        })
    return values


def _revision_from(value: Any) -> str | None:
    if value is None:
        return None
    revision = (value.get("revision") if isinstance(value, Mapping) else getattr(value, "revision", None))
    if revision is None:
        return None
    normalized = str(revision).strip()
    return normalized or None


def _resolved_checkpoint_revision(model: Any, output: Any) -> str | None:
    delegate = getattr(model, "_delegate", None)
    for owner in (
            model,
            delegate,
            getattr(model, "model", None),
            getattr(delegate, "model", None),
    ):
        if owner is None:
            continue
        for name in ("artifacts", "artifact", "runtime"):
            revision = _revision_from(getattr(owner, name, None))
            if revision is not None:
                return revision
    metadata = getattr(output, "metadata", {})
    if isinstance(metadata, Mapping):
        revision = metadata.get("checkpoint_revision")
        if revision is not None and str(revision).strip():
            return str(revision).strip()
    return None


def audit_registry() -> dict[str, Any]:
    """Lazy-construct every public ASR/VAD provider."""
    from voicehub import AutoConfig, AutoModelForSpeechRecognition, AutoModelForVoiceActivityDetection, list_model_specs
    from voicehub.tasks import SpeechTask

    records = []
    factories = {
        SpeechTask.AUTOMATIC_SPEECH_RECOGNITION: AutoModelForSpeechRecognition,
        SpeechTask.VOICE_ACTIVITY_DETECTION: AutoModelForVoiceActivityDetection,
    }
    for task, factory in factories.items():
        for spec in list_model_specs(task=task):
            record: dict[str, Any] = {
                "architecture": spec.architecture,
                "capabilities": list(spec.capabilities),
                "checkpoint": spec.default_model_path,
                "class": f"{spec.module}:{spec.class_name}",
                "model_type": spec.model_type,
                "status": "failed",
                "task": task.value,
            }
            try:
                config = AutoConfig.for_model(
                    spec.model_type,
                    name_or_path=spec.default_model_path,
                )
                model = factory.from_config(
                    config,
                    device="cpu",
                    lazy_load=True,
                )
                method_name = ("transcribe" if task is SpeechTask.AUTOMATIC_SPEECH_RECOGNITION else "detect")
                if model.model is not None:
                    raise RuntimeError("lazy construction allocated a model runtime")
                if not callable(getattr(model, method_name, None)):
                    raise TypeError(f"provider does not expose {method_name}()")
                record.update({
                    "config_class": (f"{type(config).__module__}:"
                                     f"{type(config).__qualname__}"),
                    "lazy_runtime_allocated": False,
                    "status": "lazy-contract-passed",
                })
            except Exception as error:  # noqa: BLE001 - retain every provider
                record["error"] = (f"{type(error).__name__}: {error}")
            records.append(record)
    passed = sum(record["status"] == "lazy-contract-passed" for record in records)
    return {
        "kind": "voicehub-asr-vad-registry-audit",
        "provider_count": len(records),
        "passed": passed,
        "failed": len(records) - passed,
        "providers": records,
        "runtime": _runtime_metadata(),
    }


def _load_audio(path: str):
    from voicehub import load_audio

    audio = load_audio(path)
    if audio.duration < 10.0:
        raise ValueError(
            "ASR/VAD benchmarks require at least 10.0 seconds of audio; "
            f"received {audio.duration:.3f} seconds.")
    return audio


def _strategy(args: argparse.Namespace, strategy_name: str | None):
    if strategy_name is None:
        return None
    from voicehub.inference_strategy import TorchCompileInferenceStrategy

    return TorchCompileInferenceStrategy(
        backend=args.compile_backend,
        mode=args.compile_mode,
        dynamic=True,
        requirement="required",
    )


def _run_asr_worker(args: argparse.Namespace) -> dict[str, Any]:
    from voicehub import AutoConfig, AutoModelForSpeechRecognition

    dtype, strategy_name = _ASR_PROFILES[args.profile]
    audio = _load_audio(args.audio)
    strategy = _strategy(args, strategy_name)
    config = AutoConfig.for_model(
        args.model_type,
        name_or_path=args.model_path,
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
    )
    _reset_peak_memory(args.device)
    started = time.perf_counter()
    model = AutoModelForSpeechRecognition.from_config(
        config,
        device=args.device,
        lazy_load=False,
        inference_strategy=strategy,
    )
    _sync_device(args.device)
    load_seconds = time.perf_counter() - started
    loaded_memory_mib = _peak_memory_mib(args.device)

    latencies = []
    peak_memory = []
    transcripts = []
    output = None
    for _ in range(args.runs + args.warmup_runs):
        _reset_peak_memory(args.device)
        started = time.perf_counter()
        output = model.transcribe(
            audio.waveform,
            sampling_rate=audio.sampling_rate,
            language=args.language,
        )
        _sync_device(args.device)
        latencies.append(time.perf_counter() - started)
        peak_memory.append(_peak_memory_mib(args.device))
        transcripts.append(output.text)
    assert output is not None
    warm = latencies[args.warmup_runs:]
    warm_peak_memory = peak_memory[args.warmup_runs:]
    warm_median = statistics.median(warm)
    result: dict[str, Any] = {
        "audio": str(Path(args.audio).expanduser().resolve()),
        "checkpoint": args.model_path,
        "checkpoint_request": {
            "checkpoint_filename": getattr(
                config,
                "checkpoint_filename",
                None,
            ),
            "local_files_only": bool(getattr(config, "local_files_only", False)),
            "revision": getattr(config, "revision", None),
            "source": args.model_path,
        },
        "cer": _character_error_rate(
            args.reference,
            transcripts[-1],
        ),
        "cold_inference_seconds": latencies[0],
        "cold_peak_memory_mib": peak_memory[0],
        "compile_backend": (args.compile_backend if strategy is not None else None),
        "deterministic_transcript": len(set(transcripts)) == 1,
        "device": args.device,
        "duration_seconds": audio.duration,
        "load_seconds": load_seconds,
        "loaded_peak_memory_mib": loaded_memory_mib,
        "memory_metric": _memory_metric(args.device),
        "model_type": args.model_type,
        "peak_memory_mib": max(warm_peak_memory),
        "profile": args.profile,
        "real_time_factor": warm_median / audio.duration,
        "real_time_factor_mean": (statistics.fmean(warm) / audio.duration),
        "reference": args.reference,
        "resolved_checkpoint_revision": (_resolved_checkpoint_revision(model, output)),
        "speed_x_realtime": audio.duration / warm_median,
        "speed_x_realtime_mean": (audio.duration / statistics.fmean(warm)),
        "stabilization_seconds": sum(latencies[:args.warmup_runs]),
        "status": "passed",
        "transcript": transcripts[-1],
        "warmup_latency_seconds": latencies[:args.warmup_runs],
        "warmup_runs": args.warmup_runs,
        "warm_latency_median_seconds": warm_median,
        "warm_latency_seconds": warm,
        "wer": _word_error_rate(args.reference, transcripts[-1]),
    }
    result.update(_latency_statistics(warm))
    if strategy is not None:
        result["compile_runtime"] = strategy.runtime_metadata(model)
    return result


def _segments(output: Any) -> list[dict[str, Any]]:
    return [{
        "end": segment.end,
        "score": segment.score,
        "start": segment.start,
    } for segment in output.segments]


def _run_vad_worker(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from voicehub import AutoConfig, AutoModelForVoiceActivityDetection

    strategy_name, configured_threads = _VAD_PROFILES[args.profile]
    audio = _load_audio(args.audio)
    strategy = _strategy(args, strategy_name)
    config = AutoConfig.for_model(
        args.model_type,
        name_or_path=args.model_path,
        local_files_only=args.local_files_only,
    )
    _reset_peak_memory(args.device)
    started = time.perf_counter()
    model = AutoModelForVoiceActivityDetection.from_config(
        config,
        device=args.device,
        lazy_load=False,
        inference_strategy=strategy,
    )
    _sync_device(args.device)
    load_seconds = time.perf_counter() - started
    loaded_memory_mib = _peak_memory_mib(args.device)

    latencies = []
    peak_memory = []
    segment_runs = []
    metadata = None
    output = None
    for _ in range(args.runs + args.warmup_runs):
        _reset_peak_memory(args.device)
        started = time.perf_counter()
        output = model.detect(
            audio.waveform,
            sampling_rate=audio.sampling_rate,
        )
        _sync_device(args.device)
        latencies.append(time.perf_counter() - started)
        peak_memory.append(_peak_memory_mib(args.device))
        segment_runs.append(_segments(output))
        metadata = dict(output.metadata)
    assert output is not None
    warm = latencies[args.warmup_runs:]
    warm_peak_memory = peak_memory[args.warmup_runs:]
    warm_median = statistics.median(warm)
    result: dict[str, Any] = {
        "audio": str(Path(args.audio).expanduser().resolve()),
        "checkpoint": args.model_path,
        "checkpoint_request": {
            "checkpoint_filename":
            getattr(
                config,
                "checkpoint_filename",
                getattr(config, "model_filename", None),
            ),
            "local_files_only":
            bool(getattr(config, "local_files_only", False)),
            "revision":
            getattr(config, "revision", None),
            "source":
            args.model_path,
        },
        "cold_inference_seconds": latencies[0],
        "cold_peak_memory_mib": peak_memory[0],
        "compile_backend": (args.compile_backend if strategy is not None else None),
        "configured_cpu_threads": configured_threads,
        "deterministic_segments": all(segments == segment_runs[0] for segments in segment_runs[1:]),
        "device": args.device,
        "duration_seconds": audio.duration,
        "load_seconds": load_seconds,
        "loaded_peak_memory_mib": loaded_memory_mib,
        "memory_metric": _memory_metric(args.device),
        "metadata": metadata,
        "model_type": args.model_type,
        "peak_memory_mib": max(warm_peak_memory),
        "profile": args.profile,
        "real_time_factor": warm_median / audio.duration,
        "real_time_factor_mean": (statistics.fmean(warm) / audio.duration),
        "resolved_checkpoint_revision": (_resolved_checkpoint_revision(model, output)),
        "segments": segment_runs[-1],
        "speech_duration_seconds": output.speech_duration,
        "speed_x_realtime": audio.duration / warm_median,
        "speed_x_realtime_mean": (audio.duration / statistics.fmean(warm)),
        "stabilization_seconds": sum(latencies[:args.warmup_runs]),
        "status": "passed",
        "torch_intraop_threads": torch.get_num_threads(),
        "warmup_latency_seconds": latencies[:args.warmup_runs],
        "warmup_runs": args.warmup_runs,
        "warm_latency_median_seconds": warm_median,
        "warm_latency_seconds": warm,
    }
    result.update(_latency_statistics(warm))
    if strategy is not None:
        result["compile_runtime"] = strategy.runtime_metadata(model)
    return result


def _run_worker(args: argparse.Namespace) -> dict[str, Any]:
    if args.task == "asr":
        return _run_asr_worker(args)
    return _run_vad_worker(args)


def _percent_reduction(baseline: float, candidate: float) -> float:
    if baseline == 0:
        return 0.0
    return (baseline - candidate) / baseline * 100.0


def _profile_comparisons(
    task: str,
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    passed = [result for result in results if result.get("status") == "passed"]
    if not passed:
        return []
    preferred = "fp32" if task == "asr" else "eager"
    baseline = next(
        (result for result in passed if result.get("profile") == preferred),
        passed[0],
    )
    comparisons = []
    for candidate in passed:
        baseline_latency = baseline["warm_latency_median_seconds"]
        candidate_latency = candidate["warm_latency_median_seconds"]
        comparison: dict[str, Any] = {
            "baseline_profile":
            baseline["profile"],
            "candidate_profile":
            candidate["profile"],
            "mean_latency_reduction_percent":
            _percent_reduction(
                baseline["warm_latency_mean_seconds"],
                candidate["warm_latency_mean_seconds"],
            ),
            "mean_speedup_ratio":
            (baseline["warm_latency_mean_seconds"] / candidate["warm_latency_mean_seconds"]),
            "median_latency_reduction_percent":
            _percent_reduction(
                baseline_latency,
                candidate_latency,
            ),
            "peak_memory_reduction_percent":
            _percent_reduction(
                baseline["peak_memory_mib"],
                candidate["peak_memory_mib"],
            ),
            "median_speedup_ratio":
            baseline_latency / candidate_latency,
        }
        if task == "asr":
            comparison.update({
                "transcript_equal": (baseline["transcript"] == candidate["transcript"]),
                "wer_delta": (
                    None if baseline["wer"] is None or candidate["wer"] is None else candidate["wer"] -
                    baseline["wer"]),
                "cer_delta": (
                    None if baseline["cer"] is None or candidate["cer"] is None else candidate["cer"] -
                    baseline["cer"]),
            })
        else:
            baseline_segments = baseline["segments"]
            candidate_segments = candidate["segments"]
            comparison["segment_boundaries_equal"] = ([
                (segment["start"], segment["end"]) for segment in baseline_segments
            ] == [(segment["start"], segment["end"]) for segment in candidate_segments])
            paired_scores = [(left["score"], right["score"]) for left, right in zip(
                baseline_segments,
                candidate_segments,
                strict=False,
            ) if left["score"] is not None and right["score"] is not None]
            comparison["maximum_score_absolute_difference"] = (
                None if not paired_scores else max(abs(left - right) for left, right in paired_scores))
        comparisons.append(comparison)
    return comparisons


def _worker_command(
    args: argparse.Namespace,
    profile: str,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_worker",
        "--task",
        args.task,
        "--audio",
        args.audio,
        "--reference",
        args.reference,
        "--model-type",
        args.model_type,
        "--model-path",
        args.model_path,
        "--device",
        args.device,
        "--profile",
        profile,
        "--runs",
        str(args.runs),
        "--warmup-runs",
        str(args.warmup_runs),
        "--compile-backend",
        args.compile_backend,
    ]
    if args.language is not None:
        command.extend(("--language", args.language))
    if args.compile_mode is not None:
        command.extend(("--compile-mode", args.compile_mode))
    if args.local_files_only:
        command.append("--local-files-only")
    return command


def _worker_environment(
    profile: str,
    *,
    compile_cache: Path,
) -> dict[str, str]:
    environment = dict(os.environ)
    environment["TORCHINDUCTOR_CACHE_DIR"] = str(compile_cache / "torchinductor")
    environment["TRITON_CACHE_DIR"] = str(compile_cache / "triton")
    environment["CUDA_CACHE_PATH"] = str(compile_cache / "cuda")
    if profile in _VAD_PROFILES:
        _, threads = _VAD_PROFILES[profile]
        if threads is not None:
            value = str(threads)
            environment["OMP_NUM_THREADS"] = value
            environment["MKL_NUM_THREADS"] = value
            environment["OPENBLAS_NUM_THREADS"] = value
    return environment


def _run_matrix(args: argparse.Namespace) -> dict[str, Any]:
    # Validate duration before starting any isolated model process.
    audio = _load_audio(args.audio)
    profile_definitions = (_ASR_PROFILES if args.task == "asr" else _VAD_PROFILES)
    requested_profiles = args.profiles
    if requested_profiles is None:
        requested_profiles = (
            "fp32,fp16,bf16,compile" if args.task == "asr" else "eager,eager-1-thread,compile-1-thread")
    profiles = tuple(profile.strip().lower() for profile in requested_profiles.split(",") if profile.strip())
    unknown = sorted(set(profiles) - set(profile_definitions))
    if not profiles or unknown:
        expected = ", ".join(profile_definitions)
        raise ValueError(f"`--profiles` must select from {expected}; unknown: {unknown}.")
    results = []
    for profile in profiles:
        with tempfile.TemporaryDirectory(prefix=f"voicehub-{args.task}-{profile}-", ) as cache_directory:
            try:
                completed = subprocess.run(
                    _worker_command(args, profile),
                    cwd=PROJECT_ROOT,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=args.worker_timeout_seconds,
                    env=_worker_environment(
                        profile,
                        compile_cache=Path(cache_directory),
                    ),
                )
            except subprocess.TimeoutExpired as error:
                results.append({
                    "profile": profile,
                    "status": "timed-out",
                    "timeout_seconds": args.worker_timeout_seconds,
                    "error": str(error),
                })
                continue
        if completed.returncode:
            results.append({
                "profile": profile,
                "status": "failed",
                "returncode": completed.returncode,
                "error": completed.stderr.strip() or completed.stdout.strip(),
            })
            continue
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        try:
            results.append(json.loads(lines[-1]))
        except (IndexError, json.JSONDecodeError) as error:
            results.append({
                "profile":
                profile,
                "status":
                "failed",
                "error": ("Worker did not emit valid JSON: "
                          f"{error}; stdout={completed.stdout!r}"),
            })
    return {
        "audio_sha256": _sha256(args.audio),
        "audio_duration_seconds": audio.duration,
        "comparisons": _profile_comparisons(args.task, results),
        "compiler_cache_isolated_per_profile": True,
        "error_rate_metrics": {
            "cer": "Unicode alphanumeric code points",
            "normalization": ("NFKC plus casefold; whitespace and punctuation excluded"),
            "wer": "Unicode words",
        },
        "high_latency_variability_cv_threshold": (_HIGH_LATENCY_VARIABILITY_CV),
        "kind": f"voicehub-{args.task}-profile-matrix",
        "profiles": results,
        "registry_audit": audit_registry(),
        "runtime": _runtime_metadata(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit every ASR/VAD provider or benchmark inference profiles "
            "in isolated processes."))
    parser.add_argument("--audit-registry", action="store_true")
    parser.add_argument("--task", choices=("asr", "vad"), default="asr")
    parser.add_argument("--audio")
    parser.add_argument("--reference", default="")
    parser.add_argument("--model-type")
    parser.add_argument("--model-path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--language",
        help="Optional decoding language; omit for checkpoint defaults.",
    )
    parser.add_argument("--profiles")
    parser.add_argument(
        "--profile",
        choices=(*_ASR_PROFILES, *_VAD_PROFILES),
        default="fp32",
    )
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--compile-backend", default="inductor")
    parser.add_argument("--compile-mode")
    parser.add_argument(
        "--worker-timeout-seconds",
        type=float,
        default=900.0,
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output")
    parser.add_argument(
        "--_worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.model_type is None:
        args.model_type = ("asr_moonshine" if args.task == "asr" else "vad_silero")
    if args.model_path is None:
        args.model_path = ("UsefulSensors/moonshine-tiny" if args.task == "asr" else "safestack/silero-vad")
    if args.runs < 1:
        raise ValueError("`--runs` must be at least one.")
    if args.warmup_runs < 1:
        raise ValueError("`--warmup-runs` must be at least one.")
    if args.worker_timeout_seconds <= 0:
        raise ValueError("`--worker-timeout-seconds` must be positive.")
    if args._worker:
        if not args.audio:
            raise ValueError("Worker mode requires `--audio`.")
        valid_profiles = (_ASR_PROFILES if args.task == "asr" else _VAD_PROFILES)
        if args.profile not in valid_profiles:
            expected = ", ".join(valid_profiles)
            raise ValueError(f"Worker profile for {args.task} must be one of: {expected}.")
        # One compact line makes parent-process parsing robust to logs.
        print(json.dumps(_run_worker(args), allow_nan=False, sort_keys=True))
        return 0
    if args.audit_registry:
        result = audit_registry()
    else:
        if not args.audio:
            raise ValueError(
                "Pass `--audio` for a profile matrix or "
                "`--audit-registry` for a dependency-free audit.")
        result = _run_matrix(args)
    encoded = _json_dump(result)
    if args.output:
        destination = Path(args.output).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(encoded + os.linesep, encoding="utf-8")
    print(encoded)
    failed = bool(result.get("failed", 0))
    failed = failed or any(profile.get("status") != "passed" for profile in result.get("profiles", ()))
    registry_audit = result.get("registry_audit", {})
    failed = failed or bool(registry_audit.get("failed", 0))
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
