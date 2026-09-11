#!/usr/bin/env python3
"""Benchmark real VoiceHub TTS checkpoints in isolated processes.

Preset execution profiles keep the model, generation arguments, text,
and seed fixed, then test waveform equivalence. Explicitly named float16
and bfloat16 profiles measure their memory trade-off under the same
quality gate. Custom JSON profile specifications can vary model
configuration, generation arguments, and optimization policy for
heterogeneous TTS providers. Every effective profile is recorded so
those intentional differences remain auditable.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any

DEFAULT_TEXT = (
    "VoiceHub makes speech model inference easier to understand and reproduce. "
    "This sample is intentionally long enough to produce more than ten seconds "
    "of clear spoken audio. It measures the complete text tokenizer, acoustic "
    "model, normalizing flow, and neural vocoder pipeline on one graphics "
    "processor. The same sentence and random seed are used for every benchmark "
    "so that baseline and optimized runs remain directly comparable.")

PROFILE_NAMES = (
    "baseline",
    "weight-norm-cache",
    "float16-cache",
    "bfloat16-cache",
    "triton",
    "compile",
    "triton-compile",
)
DEFAULT_PROFILE_NAMES = (
    "baseline",
    "compile",
)
CACHE_PROFILE_NAMES = frozenset({
    "weight-norm-cache",
    "float16-cache",
    "bfloat16-cache",
})
PROFILE_SPEC_FIELDS = frozenset({
    "config_kwargs",
    "generation_kwargs",
    "optimization_config",
    "weight_norm_cache",
})
PROTECTED_GENERATION_KWARGS = frozenset({
    "output_file",
    "seed",
})
_SAFE_PROFILE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SENSITIVE_KEYS = frozenset({
    "access_token",
    "api_key",
    "apikey",
    "auth_token",
    "authorization",
    "credential",
    "credentials",
    "hf_token",
    "huggingface_token",
    "password",
    "secret",
    "token",
    "use_auth_token",
})
_UNSET = object()


def _is_sensitive_key(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return normalized in _SENSITIVE_KEYS


def _secret_strings(value: Any) -> frozenset[str]:
    secrets: set[str] = set()

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            for key, nested in item.items():
                if (_is_sensitive_key(key) and isinstance(nested, str) and nested):
                    secrets.add(nested)
                else:
                    visit(nested)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                visit(nested)

    visit(value)
    return frozenset(secrets)


def _redact_sensitive(
        value: Any,
        *,
        secrets: frozenset[str] = frozenset(),
) -> Any:
    if isinstance(value, Mapping):
        return {
            key: ("<redacted>" if _is_sensitive_key(key) else _redact_sensitive(nested, secrets=secrets))
            for key, nested in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_redact_sensitive(item, secrets=secrets) for item in value)
    if isinstance(value, list):
        return [_redact_sensitive(item, secrets=secrets) for item in value]
    if isinstance(value, str):
        redacted = value
        for secret in sorted(secrets, key=len, reverse=True):
            redacted = redacted.replace(secret, "<redacted>")
        redacted = re.sub(
            r"(?i)(https?://)[^/@\s]+@",
            r"\1<redacted>@",
            redacted,
        )
        return re.sub(
            r'(?i)(https?://[^\s?#"\']+)\?[^\s#"\']*',
            r"\1?<redacted>",
            redacted,
        )
    return value


def _json_mapping(value: str, *, name: str) -> dict[str, Any]:
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(f"{name} must be valid JSON: {error}") from error
    if not isinstance(decoded, Mapping):
        raise argparse.ArgumentTypeError(f"{name} must decode to a JSON object.")
    if any(not isinstance(key, str) or not key for key in decoded):
        raise argparse.ArgumentTypeError(f"{name} keys must be non-empty strings.")
    return dict(decoded)


def _profiles(value: str) -> tuple[str, ...]:
    items = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    if not items:
        raise argparse.ArgumentTypeError("At least one benchmark profile is required.")
    unknown = tuple(item for item in items if item not in PROFILE_NAMES)
    if unknown:
        raise argparse.ArgumentTypeError(
            "Unknown profile(s) "
            f"{', '.join(unknown)}; expected: {', '.join(PROFILE_NAMES)}.")
    if len(items) != len(set(items)):
        raise argparse.ArgumentTypeError("Benchmark profiles cannot contain duplicates.")
    return items


def _profile_name(value: str, *, option: str = "Profile") -> str:
    if not isinstance(value, str) or not _SAFE_PROFILE_NAME.fullmatch(value):
        raise argparse.ArgumentTypeError(
            f"{option} names must match {_SAFE_PROFILE_NAME.pattern!r}; "
            "use letters, digits, dots, underscores, or hyphens without "
            "leading punctuation.")
    return value


def _profile_mapping(
    value: Any,
    *,
    profile: str,
    field: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise argparse.ArgumentTypeError(
            f"--profile-specs profile {profile!r} field {field!r} must "
            "be a JSON object.")
    if any(not isinstance(key, str) or not key for key in value):
        raise argparse.ArgumentTypeError(
            f"--profile-specs profile {profile!r} field {field!r} must "
            "use non-empty string keys.")
    return copy.deepcopy(dict(value))


def _validate_generation_kwargs(
    value: Mapping[str, Any],
    *,
    location: str,
) -> None:
    protected = sorted(PROTECTED_GENERATION_KWARGS.intersection(value))
    if protected:
        joined = ", ".join(repr(name) for name in protected)
        raise argparse.ArgumentTypeError(
            f"{location} cannot set benchmark-managed generation "
            f"argument(s) {joined}. Use --seed for deterministic seeding; "
            "`output_file` is managed by the benchmark.")


def _normalize_profile_spec(
    value: Any,
    *,
    profile: str,
) -> dict[str, Any]:
    _profile_name(profile, option="Custom profile")
    if not isinstance(value, Mapping):
        raise argparse.ArgumentTypeError(f"--profile-specs profile {profile!r} must be a JSON object.")
    unknown = sorted(set(value) - PROFILE_SPEC_FIELDS)
    if unknown:
        raise argparse.ArgumentTypeError(
            f"--profile-specs profile {profile!r} has unknown field(s) "
            f"{', '.join(unknown)}; expected only "
            f"{', '.join(sorted(PROFILE_SPEC_FIELDS))}.")
    config_kwargs = _profile_mapping(
        value.get("config_kwargs", {}),
        profile=profile,
        field="config_kwargs",
    )
    generation_kwargs = _profile_mapping(
        value.get("generation_kwargs", {}),
        profile=profile,
        field="generation_kwargs",
    )
    _validate_generation_kwargs(
        generation_kwargs,
        location=(f"--profile-specs profile {profile!r} generation_kwargs"),
    )
    optimization_value = value.get("optimization_config")
    if optimization_value is None:
        optimization_config = None
    else:
        optimization_config = _profile_mapping(
            optimization_value,
            profile=profile,
            field="optimization_config",
        )
    weight_norm_cache = value.get("weight_norm_cache", False)
    if not isinstance(weight_norm_cache, bool):
        raise argparse.ArgumentTypeError(
            f"--profile-specs profile {profile!r} field "
            "`weight_norm_cache` must be true or false.")
    return {
        "config_kwargs": config_kwargs,
        "generation_kwargs": generation_kwargs,
        "optimization_config": optimization_config,
        "weight_norm_cache": weight_norm_cache,
    }


def _profile_specs(value: str) -> dict[str, dict[str, Any]]:
    decoded = _json_mapping(value, name="--profile-specs")
    if not decoded:
        raise argparse.ArgumentTypeError("--profile-specs must contain at least one named profile.")
    return {
        _profile_name(profile, option="Custom profile"): _normalize_profile_spec(
            spec,
            profile=profile,
        )
        for profile, spec in decoded.items()
    }


def _worker_profile_spec(value: str) -> dict[str, Any]:
    decoded = _json_mapping(value, name="--worker-profile-spec")
    return _normalize_profile_spec(
        decoded,
        profile="worker-profile",
    )


def _generation_mapping(value: str) -> dict[str, Any]:
    decoded = _json_mapping(value, name="--generation-kwargs")
    _validate_generation_kwargs(
        decoded,
        location="--generation-kwargs",
    )
    return decoded


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("Value must be a non-negative integer.")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be finite and greater than zero.")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("Value must be finite and non-negative.")
    return parsed


def profile_optimization_config(
    profile: str,
    *,
    compile_backend: str,
    compile_mode: str | None,
    compile_dynamic: bool | None,
) -> dict[str, Any] | None:
    """Return a quality-candidate execution policy for *profile*."""
    if profile not in PROFILE_NAMES:
        raise ValueError(f"Unknown benchmark profile {profile!r}.")
    if profile in {
            "baseline",
            "weight-norm-cache",
            "float16-cache",
            "bfloat16-cache",
    }:
        return None
    compile_enabled = profile in {"compile", "triton-compile"}
    kernel_backend = "triton" if profile in {"triton", "triton-compile"} else "native"
    values: dict[str, Any] = {
        "attn_implementation": "native",
        "kernel_backend": kernel_backend,
        "compile": compile_enabled,
        "diffusion_cache": False,
        "diffusion_sampling": False,
    }
    if compile_enabled:
        values["compile_config"] = {
            "backend": compile_backend,
            "mode": compile_mode,
            "dynamic": compile_dynamic,
            "fullgraph": False,
        }
    return values


def profile_uses_weight_norm_cache(profile: str) -> bool:
    """Return whether *profile* opts in to the VITS inference cache."""
    if profile not in PROFILE_NAMES:
        raise ValueError(f"Unknown benchmark profile {profile!r}.")
    return profile in CACHE_PROFILE_NAMES


def effective_profile_spec(
    profile: str,
    *,
    config_kwargs: Mapping[str, Any],
    generation_kwargs: Mapping[str, Any],
    compile_backend: str,
    compile_mode: str | None,
    compile_dynamic: bool | None,
    custom_spec: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve one preset or custom profile into the worker-facing schema."""
    _profile_name(profile)
    if not isinstance(config_kwargs, Mapping):
        raise TypeError("`config_kwargs` must be a mapping.")
    if not isinstance(generation_kwargs, Mapping):
        raise TypeError("`generation_kwargs` must be a mapping.")
    base_config = copy.deepcopy(dict(config_kwargs))
    base_generation = copy.deepcopy(dict(generation_kwargs))
    if any(not isinstance(key, str) or not key for key in base_config):
        raise ValueError("`config_kwargs` keys must be non-empty strings.")
    if any(not isinstance(key, str) or not key for key in base_generation):
        raise ValueError("`generation_kwargs` keys must be non-empty strings.")
    _validate_generation_kwargs(
        base_generation,
        location="--generation-kwargs",
    )

    if custom_spec is None:
        profile_config: dict[str, Any] = {}
        if profile == "float16-cache":
            profile_config["torch_dtype"] = "float16"
        elif profile == "bfloat16-cache":
            profile_config["torch_dtype"] = "bfloat16"
        normalized = {
            "config_kwargs":
            profile_config,
            "generation_kwargs": {},
            "optimization_config":
            profile_optimization_config(
                profile,
                compile_backend=compile_backend,
                compile_mode=compile_mode,
                compile_dynamic=compile_dynamic,
            ),
            "weight_norm_cache":
            profile_uses_weight_norm_cache(profile),
        }
    else:
        normalized = _normalize_profile_spec(
            custom_spec,
            profile=profile,
        )

    base_config.update(normalized["config_kwargs"])
    base_generation.update(normalized["generation_kwargs"])
    return {
        "config_kwargs": base_config,
        "generation_kwargs": base_generation,
        "optimization_config": copy.deepcopy(normalized["optimization_config"]),
        "weight_norm_cache": normalized["weight_norm_cache"],
    }


def benchmark_profile_specs(args: argparse.Namespace, ) -> dict[str, dict[str, Any]]:
    """Return ordered, effective profiles for one parent benchmark process."""
    custom_specs = getattr(args, "profile_specs", None)
    if custom_specs is None:
        selected = (
            args.profiles if args.profiles is not None else _profiles(",".join(DEFAULT_PROFILE_NAMES)))
        return {
            profile:
            effective_profile_spec(
                profile,
                config_kwargs=args.config_kwargs,
                generation_kwargs=args.generation_kwargs,
                compile_backend=args.compile_backend,
                compile_mode=args.compile_mode,
                compile_dynamic=args.compile_dynamic,
            )
            for profile in selected
        }
    return {
        profile:
        effective_profile_spec(
            profile,
            config_kwargs=args.config_kwargs,
            generation_kwargs=args.generation_kwargs,
            compile_backend=args.compile_backend,
            compile_mode=args.compile_mode,
            compile_dynamic=args.compile_dynamic,
            custom_spec=spec,
        )
        for profile, spec in custom_specs.items()
    }


def _effective_worker_profile_spec(args: argparse.Namespace, ) -> dict[str, Any]:
    supplied = getattr(args, "worker_profile_spec", None)
    if supplied is not None:
        return _normalize_profile_spec(
            supplied,
            profile=args.worker_profile,
        )
    return effective_profile_spec(
        args.worker_profile,
        config_kwargs=args.config_kwargs,
        generation_kwargs=args.generation_kwargs,
        compile_backend=args.compile_backend,
        compile_mode=args.compile_mode,
        compile_dynamic=args.compile_dynamic,
    )


def _load_worker_input(args: argparse.Namespace) -> None:
    path = getattr(args, "worker_input", None)
    if path is None:
        raise ValueError("Worker mode requires --worker-input.")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError(
            f"Could not read worker input {path}: "
            f"{type(error).__name__}: {error}") from error
    if not isinstance(payload, Mapping):
        raise TypeError("Worker input must contain one JSON object.")
    expected = {
        "config_kwargs",
        "generation_kwargs",
        "model",
        "profile",
        "profile_spec",
        "text",
    }
    if set(payload) != expected:
        raise ValueError("Worker input fields must be exactly: " + ", ".join(sorted(expected)) + ".")
    if payload["profile"] != args.worker_profile:
        raise ValueError("Worker input profile does not match --worker-profile.")
    if payload["model"] is not None and not isinstance(payload["model"], str):
        raise ValueError("Worker input model must be a string or null.")
    if not isinstance(payload["text"], str) or not payload["text"]:
        raise ValueError("Worker input text must be a non-empty string.")
    args.model = payload["model"]
    args.text = payload["text"]
    args.config_kwargs = _profile_mapping(
        payload["config_kwargs"],
        profile=args.worker_profile,
        field="config_kwargs",
    )
    args.generation_kwargs = _profile_mapping(
        payload["generation_kwargs"],
        profile=args.worker_profile,
        field="generation_kwargs",
    )
    _validate_generation_kwargs(
        args.generation_kwargs,
        location="Worker input generation_kwargs",
    )
    args.worker_profile_spec = _normalize_profile_spec(
        payload["profile_spec"],
        profile=args.worker_profile,
    )


def _synchronize(torch: Any, device: Any) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("Latency percentiles require at least one value.")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _memory_snapshot(torch: Any, device: Any) -> dict[str, int] | None:
    if device.type != "cuda":
        return None
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def _reset_peak_memory(torch: Any, device: Any) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _waveform(output: Any, torch: Any) -> Any:
    audio = getattr(output, "audio", None)
    if audio is None:
        raise RuntimeError("TTS output does not expose an `audio` waveform.")
    audio = audio if isinstance(audio, torch.Tensor) else torch.as_tensor(audio)
    audio = audio.detach().float().cpu().squeeze().contiguous()
    if audio.ndim != 1 or audio.numel() < 1:
        raise RuntimeError(
            "TTS output must contain one non-empty mono waveform; "
            f"received shape {tuple(audio.shape)}.")
    if not bool(torch.isfinite(audio).all().item()):
        raise RuntimeError("TTS output contains NaN or infinite samples.")
    return audio


def _waveform_digest(audio: Any) -> str:
    return hashlib.sha256(audio.numpy().tobytes(order="C")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hugging_face_snapshot_anchor(source: str | Path, ) -> tuple[str, str] | None:
    """Resolve an immutable local Hugging Face snapshot directory."""
    try:
        path = Path(source).expanduser()
    except TypeError:
        return None
    if not path.is_dir():
        return None
    path = path.resolve()
    revision = path.name
    if (path.parent.name != "snapshots" or re.fullmatch(r"[0-9a-fA-F]{40,64}", revision) is None):
        return None
    return str(path.parent.parent), revision


def checkpoint_identity(
    model: Any,
    *,
    requested_source: str | Path,
    requested_revision: object = _UNSET,
) -> dict[str, Any]:
    """Describe a resolved checkpoint without guessing provider internals."""
    config = getattr(model, "config", None)
    if requested_revision is _UNSET:
        requested_revision = getattr(config, "revision", None)
    artifacts = None
    for owner in (
            model,
            getattr(model, "model", None),
            getattr(model, "runtime", None),
    ):
        candidate = getattr(owner, "artifacts", None)
        if candidate is not None:
            artifacts = candidate
            break

    resolved_source = getattr(artifacts, "source", None)
    resolved_revision = getattr(artifacts, "revision", None)
    snapshot_anchor = _hugging_face_snapshot_anchor(requested_source)
    if snapshot_anchor is not None:
        snapshot_source, snapshot_revision = snapshot_anchor
        if resolved_source is None:
            resolved_source = snapshot_source
        if resolved_revision is None:
            resolved_revision = snapshot_revision
    revision_was_explicit = requested_revision is not None
    if (requested_revision is None and getattr(config, "model_type", None) == "vits" and
            resolved_revision is not None):
        requested_revision = "main"
    checkpoint_value = getattr(artifacts, "checkpoint", None)
    checkpoint_path = None
    if isinstance(checkpoint_value, (str, os.PathLike)):
        candidate_path = Path(checkpoint_value).expanduser()
        if candidate_path.is_file():
            checkpoint_path = candidate_path.resolve()
    weight_path = (
        checkpoint_path
        if checkpoint_path is not None and not checkpoint_path.name.endswith(".index.json") else None)
    return {
        "requested_source":
        str(requested_source),
        "requested_revision":
        requested_revision,
        "requested_revision_was_explicit":
        revision_was_explicit,
        "resolved_source": (None if resolved_source is None else str(resolved_source)),
        "resolved_revision":
        resolved_revision,
        "local_checkpoint_path": (None if checkpoint_path is None else str(checkpoint_path)),
        "local_weight_sha256": (None if weight_path is None else _sha256_file(weight_path)),
        "weight_digest_status": (
            "sha256" if weight_path is not None else
            "checkpoint-index-only" if checkpoint_path is not None else "not-exposed"),
    }


def _audio_result(
    output: Any,
    torch: Any,
    *,
    minimum_audio_seconds: float,
) -> tuple[Any, dict[str, Any]]:
    audio = _waveform(output, torch)
    sample_rate = getattr(output, "sample_rate", None)
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise RuntimeError("TTS output sample rate must be a positive integer.")
    duration = audio.numel() / sample_rate
    if duration < minimum_audio_seconds:
        raise RuntimeError(
            f"Generated audio is {duration:.3f} seconds, below the required "
            f"{minimum_audio_seconds:.3f} seconds.")
    return audio, {
        "samples": int(audio.numel()),
        "sample_rate": sample_rate,
        "duration_seconds": duration,
        "sha256_float32": _waveform_digest(audio),
        "minimum": float(audio.min().item()),
        "maximum": float(audio.max().item()),
        "rms": float(audio.square().mean().sqrt().item()),
    }


def _timed_generate(
    model: Any,
    torch: Any,
    device: Any,
    text: str,
    generation_kwargs: Mapping[str, Any],
    *,
    minimum_audio_seconds: float,
) -> tuple[Any, dict[str, Any], float]:
    _synchronize(torch, device)
    started = time.perf_counter()
    output = model.generate(text, **generation_kwargs)
    _synchronize(torch, device)
    elapsed = time.perf_counter() - started
    audio, metadata = _audio_result(
        output,
        torch,
        minimum_audio_seconds=minimum_audio_seconds,
    )
    return audio, metadata, elapsed


def _model_statistics(model: Any) -> dict[str, Any]:
    parameters_method = getattr(model.model, "parameters", None)
    if not callable(parameters_method):
        return {
            "parameters": None,
            "trainable_parameters": None,
            "parameter_bytes": None,
            "dtypes": [],
        }
    parameters = tuple(parameters_method())
    return {
        "parameters": sum(parameter.numel() for parameter in parameters),
        "trainable_parameters": sum(parameter.numel() for parameter in parameters if parameter.requires_grad),
        "parameter_bytes": sum(parameter.numel() * parameter.element_size() for parameter in parameters),
        "dtypes": sorted({str(parameter.dtype).removeprefix("torch.")
                          for parameter in parameters}),
    }


def _benchmark_worker(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from voicehub import AutoModelForTextToSpeech
    from voicehub import __version__ as voicehub_version
    from voicehub import get_model_spec
    from voicehub.optimization import diffusion_cache_summary, reset_diffusion_cache_metrics

    effective_spec = _effective_worker_profile_spec(args)
    config_kwargs = copy.deepcopy(effective_spec["config_kwargs"])
    profile_generation_kwargs = copy.deepcopy(effective_spec["generation_kwargs"])
    _validate_generation_kwargs(
        profile_generation_kwargs,
        location=f"Profile {args.worker_profile!r}",
    )
    optimization_config = copy.deepcopy(effective_spec["optimization_config"])
    weight_norm_cache = effective_spec["weight_norm_cache"]
    spec = get_model_spec(args.model_type)
    source = args.model or spec.default_model_path
    if not source:
        raise ValueError(f"Model type {args.model_type!r} has no default checkpoint; pass `--model`.")
    generation_kwargs = copy.deepcopy(profile_generation_kwargs)
    generation_kwargs["seed"] = args.seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    load_started = time.perf_counter()
    model = AutoModelForTextToSpeech.from_pretrained(
        source,
        model_type=args.model_type,
        device=args.device,
        lazy_load=True,
        config_kwargs=config_kwargs,
        optimization_config=optimization_config,
    )
    model.load()
    device = torch.device(model.device)
    _synchronize(torch, device)
    load_seconds = time.perf_counter() - load_started
    cache_method = getattr(model.model, "cache_weight_norm_for_inference", None)
    clear_cache_method = getattr(model.model, "clear_weight_norm_cache", None)
    cache_bytes = None
    cache_enabled = None
    if weight_norm_cache:
        if not callable(cache_method) or not callable(clear_cache_method):
            raise ValueError(
                f"Profile {args.worker_profile!r} requested "
                "`weight_norm_cache=true`, but the loaded model does not "
                "expose both public weight-normalization cache methods.")
        cache_bytes = int(cache_method())
        cache_enabled = True
    elif callable(clear_cache_method):
        clear_cache_method()
        cache_enabled = False
    model_memory = _memory_snapshot(torch, device)
    model_statistics = _model_statistics(model)
    optimization = model.tts_optimization_result(mode="inference")

    reset_diffusion_cache_metrics(model.model)
    _reset_peak_memory(torch, device)
    cold_audio, cold_metadata, cold_seconds = _timed_generate(
        model,
        torch,
        device,
        args.text,
        generation_kwargs,
        minimum_audio_seconds=args.minimum_audio_seconds,
    )
    cold_memory = _memory_snapshot(torch, device)
    cold_diffusion_cache = diffusion_cache_summary(
        model.model,
        details=True,
    )
    torch.save(cold_audio, args.worker_waveform)
    if args.worker_audio is not None:
        model.save_audio(
            args.worker_audio,
            cold_audio,
            cold_metadata["sample_rate"],
        )

    reset_diffusion_cache_metrics(model.model)
    warmup_hashes = []
    for _ in range(args.warmup):
        _audio, metadata, _elapsed = _timed_generate(
            model,
            torch,
            device,
            args.text,
            generation_kwargs,
            minimum_audio_seconds=args.minimum_audio_seconds,
        )
        warmup_hashes.append(metadata["sha256_float32"])
        if metadata["sample_rate"] != cold_metadata["sample_rate"]:
            raise RuntimeError("Warm-up generation changed the output sample rate.")
        if metadata["samples"] != cold_metadata["samples"]:
            raise RuntimeError("Warm-up generation changed the output waveform length.")
    warmup_diffusion_cache = diffusion_cache_summary(
        model.model,
        details=True,
    )

    reset_diffusion_cache_metrics(model.model)
    _reset_peak_memory(torch, device)
    latencies = []
    repeat_hashes = []
    for _ in range(args.repeats):
        _audio, metadata, elapsed = _timed_generate(
            model,
            torch,
            device,
            args.text,
            generation_kwargs,
            minimum_audio_seconds=args.minimum_audio_seconds,
        )
        if metadata["sample_rate"] != cold_metadata["sample_rate"]:
            raise RuntimeError("Measured generation changed the output sample rate.")
        if metadata["samples"] != cold_metadata["samples"]:
            raise RuntimeError("Measured generation changed the output waveform length.")
        latencies.append(elapsed)
        repeat_hashes.append(metadata["sha256_float32"])
    steady_memory = _memory_snapshot(torch, device)
    steady_diffusion_cache = diffusion_cache_summary(
        model.model,
        details=True,
    )
    mean_seconds = statistics.fmean(latencies)
    latency_stdev = statistics.stdev(latencies) if len(latencies) > 1 else 0.0

    gpu = None
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        gpu = {
            "name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "compute_capability": [
                int(properties.major),
                int(properties.minor),
            ],
        }
    identity = checkpoint_identity(
        model,
        requested_source=source,
        requested_revision=config_kwargs.get("revision"),
    )
    return {
        "status": "ok",
        "profile": args.worker_profile,
        "model_type": spec.model_type,
        "architecture": spec.architecture,
        "checkpoint": source,
        "coverage": "real-checkpoint-end-to-end",
        "device": str(device),
        "gpu": gpu,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "voicehub_version": voicehub_version,
        "checkpoint_identity": identity,
        "effective_profile_spec": effective_spec,
        "config_kwargs": config_kwargs,
        "generation_kwargs": generation_kwargs,
        "optimization_config": optimization_config,
        "optimization_manifest": (None if optimization is None else optimization.manifest()),
        "weight_norm_cache": {
            "supported": callable(cache_method) and callable(clear_cache_method),
            "enabled": cache_enabled,
            "cache_bytes": cache_bytes,
        },
        "model": model_statistics,
        "load_seconds": load_seconds,
        "model_memory": model_memory,
        "cold": {
            **cold_metadata,
            "latency_seconds": cold_seconds,
            "end_to_end_latency_seconds": cold_seconds,
            "time_to_first_audio_seconds": cold_seconds,
            "real_time_factor": cold_seconds / cold_metadata["duration_seconds"],
            "audio_seconds_per_second": cold_metadata["duration_seconds"] / cold_seconds,
            "memory": cold_memory,
            "diffusion_cache": cold_diffusion_cache,
        },
        "warmup_diffusion_cache": warmup_diffusion_cache,
        "steady": {
            "warmup_runs_after_cold":
            args.warmup,
            "repeats":
            args.repeats,
            "latency_seconds":
            latencies,
            "mean_latency_seconds":
            mean_seconds,
            "median_latency_seconds":
            statistics.median(latencies),
            "p95_latency_seconds":
            _percentile(latencies, 0.95),
            "latency_stdev_seconds":
            latency_stdev,
            "latency_coefficient_of_variation": (latency_stdev / mean_seconds if mean_seconds else 0.0),
            "minimum_latency_seconds":
            min(latencies),
            "maximum_latency_seconds":
            max(latencies),
            "mean_time_to_first_audio_seconds":
            mean_seconds,
            "mean_real_time_factor":
            mean_seconds / cold_metadata["duration_seconds"],
            "mean_audio_seconds_per_second":
            cold_metadata["duration_seconds"] / mean_seconds,
            "memory":
            steady_memory,
            "diffusion_cache":
            steady_diffusion_cache,
            "deterministic_across_repeats":
            (all(value == cold_metadata["sha256_float32"] for value in (*warmup_hashes, *repeat_hashes))),
            "sha256_float32":
            repeat_hashes,
        },
    }


def compare_waveforms(
    reference: Any,
    candidate: Any,
    *,
    sample_rate: int,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, Any]:
    """Compare deterministic waveforms without making a perceptual claim."""
    import torch

    reference = reference.detach().float().cpu().reshape(-1)
    candidate = candidate.detach().float().cpu().reshape(-1)
    same_length = reference.numel() == candidate.numel()
    result: dict[str, Any] = {
        "same_length": same_length,
        "reference_samples": int(reference.numel()),
        "candidate_samples": int(candidate.numel()),
        "duration_delta_seconds": ((candidate.numel() - reference.numel()) / sample_rate),
        "exact": False,
        "within_tolerance": False,
        "max_absolute_error": None,
        "mean_absolute_error": None,
        "root_mean_square_error": None,
        "signal_to_noise_db": None,
    }
    if not same_length:
        return result
    difference = candidate - reference
    absolute = difference.abs()
    noise_power = difference.double().square().mean()
    signal_power = reference.double().square().mean()
    exact = bool(torch.equal(reference, candidate))
    within_tolerance = bool(
        torch.allclose(
            reference,
            candidate,
            atol=absolute_tolerance,
            rtol=relative_tolerance,
        ))
    if noise_power.item() > 0 and signal_power.item() > 0:
        snr = float(10.0 * torch.log10(signal_power / noise_power).item())
    else:
        snr = None
    result.update({
        "exact": exact,
        "within_tolerance": within_tolerance,
        "max_absolute_error": float(absolute.max().item()),
        "mean_absolute_error": float(absolute.mean().item()),
        "root_mean_square_error": float(noise_power.sqrt().item()),
        "signal_to_noise_db": snr,
    })
    return result


def _candidate_minus_baseline(
    baseline: float | None,
    candidate: float | None,
) -> float | None:
    if baseline is None or candidate is None:
        return None
    return candidate - baseline


def _percent_reduction(
    baseline: float | None,
    candidate: float | None,
) -> float | None:
    if baseline is None or candidate is None or baseline == 0:
        return None
    return float((baseline - candidate) / baseline * 100.0)


def _speedup_ratio(
    baseline_seconds: float | None,
    candidate_seconds: float | None,
) -> float | None:
    if (baseline_seconds is None or candidate_seconds is None or candidate_seconds == 0):
        return None
    return float(baseline_seconds / candidate_seconds)


def _latency_comparison(
    baseline_seconds: float | None,
    candidate_seconds: float | None,
) -> dict[str, float | None]:
    return {
        "baseline_seconds": baseline_seconds,
        "candidate_seconds": candidate_seconds,
        "candidate_minus_baseline_seconds": _candidate_minus_baseline(
            baseline_seconds,
            candidate_seconds,
        ),
        "latency_reduction_percent": _percent_reduction(
            baseline_seconds,
            candidate_seconds,
        ),
        "speedup_ratio": _speedup_ratio(
            baseline_seconds,
            candidate_seconds,
        ),
    }


def _memory_comparison(
    baseline_bytes: int | None,
    candidate_bytes: int | None,
) -> dict[str, int | float | None]:
    delta = _candidate_minus_baseline(baseline_bytes, candidate_bytes)
    return {
        "baseline_bytes": baseline_bytes,
        "candidate_bytes": candidate_bytes,
        "candidate_minus_baseline_bytes": delta,
        "reduction_bytes": None if delta is None else -delta,
        "reduction_percent": _percent_reduction(
            baseline_bytes,
            candidate_bytes,
        ),
    }


def _peak_memory_value(
    result: Mapping[str, Any],
    *,
    phase: str,
    key: str,
) -> int | None:
    phase_result = result.get(phase)
    if not isinstance(phase_result, Mapping):
        return None
    memory = phase_result.get("memory")
    if not isinstance(memory, Mapping):
        return None
    value = memory.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def performance_comparisons(
    results: list[dict[str, Any]],
    waveform_comparisons: Mapping[str, Mapping[str, Any]],
    *,
    reference_profile: str,
) -> dict[str, dict[str, Any]]:
    """Calculate candidate deltas against one successful reference profile."""
    successful = {result["profile"]: result for result in results if result.get("status") == "ok"}
    baseline = successful.get(reference_profile)
    if baseline is None:
        return {}

    comparisons = {}
    for profile, candidate in successful.items():
        waveform = waveform_comparisons.get(profile)
        if waveform is None:
            continue
        comparisons[profile] = {
            "reference_profile":
            reference_profile,
            "candidate_profile":
            profile,
            "waveform_equivalence_passed":
            bool(waveform.get("within_tolerance", False)),
            "waveform_exact":
            bool(waveform.get("exact", False)),
            "cold_latency":
            _latency_comparison(
                baseline["cold"]["latency_seconds"],
                candidate["cold"]["latency_seconds"],
            ),
            "steady_mean_latency":
            _latency_comparison(
                baseline["steady"]["mean_latency_seconds"],
                candidate["steady"]["mean_latency_seconds"],
            ),
            "steady_median_latency":
            _latency_comparison(
                baseline["steady"]["median_latency_seconds"],
                candidate["steady"]["median_latency_seconds"],
            ),
            "steady_peak_allocated_memory":
            _memory_comparison(
                _peak_memory_value(
                    baseline,
                    phase="steady",
                    key="peak_allocated_bytes",
                ),
                _peak_memory_value(
                    candidate,
                    phase="steady",
                    key="peak_allocated_bytes",
                ),
            ),
            "steady_peak_reserved_memory":
            _memory_comparison(
                _peak_memory_value(
                    baseline,
                    phase="steady",
                    key="peak_reserved_bytes",
                ),
                _peak_memory_value(
                    candidate,
                    phase="steady",
                    key="peak_reserved_bytes",
                ),
            ),
        }
    return comparisons


def checkpoint_identity_comparison_error(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> str | None:
    """Return why two profiles lack one shared immutable checkpoint anchor."""
    if not isinstance(reference, Mapping) or not isinstance(candidate, Mapping):
        return "Checkpoint identity must be a JSON object for every profile."

    reference_digest = reference.get("local_weight_sha256")
    candidate_digest = candidate.get("local_weight_sha256")
    if reference_digest is not None and candidate_digest is not None:
        digest_pattern = re.compile(r"^[0-9a-fA-F]{64}$")
        if (not isinstance(reference_digest, str) or not digest_pattern.fullmatch(reference_digest) or
                not isinstance(candidate_digest, str) or not digest_pattern.fullmatch(candidate_digest)):
            return "Checkpoint weight digests must be 64 hexadecimal characters."
        if reference_digest != candidate_digest:
            return "Checkpoint identity differs from baseline in local_weight_sha256."
        return None

    reference_source = reference.get("resolved_source")
    candidate_source = candidate.get("resolved_source")
    reference_revision = reference.get("resolved_revision")
    candidate_revision = candidate.get("resolved_revision")
    revision_pattern = re.compile(r"^[0-9a-fA-F]{40,64}$")
    reference_has_revision = (
        isinstance(reference_source, str) and bool(reference_source) and
        isinstance(reference_revision, str) and revision_pattern.fullmatch(reference_revision) is not None)
    candidate_has_revision = (
        isinstance(candidate_source, str) and bool(candidate_source) and
        isinstance(candidate_revision, str) and revision_pattern.fullmatch(candidate_revision) is not None)
    if reference_has_revision and candidate_has_revision:
        if (reference_source != candidate_source or reference_revision != candidate_revision):
            return ("Checkpoint resolved source or immutable revision differs "
                    "from baseline.")
        return None

    return (
        "Checkpoint comparison requires a shared SHA-256 weight digest or "
        "an identical resolved source and immutable 40–64 character revision.")


def audit_registry() -> dict[str, Any]:
    """Exercise lazy construction and static optimization for every TTS
    provider."""
    from voicehub import AutoModelForTextToSpeech
    from voicehub import __version__ as voicehub_version
    from voicehub.optimization import OptimizationContext
    from voicehub.registry import list_model_specs
    from voicehub.tasks import SpeechTask

    providers = []
    for spec in list_model_specs(task=SpeechTask.TEXT_TO_SPEECH):
        item: dict[str, Any] = {
            "model_type": spec.model_type,
            "architecture": spec.architecture,
            "default_checkpoint": spec.default_model_path,
            "coverage": "lazy-construction-and-static-plan",
            "real_weights": {
                "status":
                "not-attempted",
                "reason": (
                    "The registry audit is intentionally offline and does not "
                    "download or allocate checkpoint weights."),
            },
            "lazy_construction": None,
            "baseline_plan": None,
            "optimized_plan": None,
        }
        try:
            model = AutoModelForTextToSpeech.from_pretrained(
                "",
                model_type=spec.model_type,
                lazy_load=True,
            )
            item["lazy_construction"] = {
                "status": "ok",
                "class": type(model).__name__,
                "loaded": bool(model.is_loaded),
            }
        except Exception as error:  # noqa: BLE001 - the audit records every provider failure
            item["lazy_construction"] = {
                "status": "error",
                "error_type": type(error).__name__,
                "error": str(error),
            }
            providers.append(item)
            continue
        try:
            baseline = model.resolve_optimization(
                {
                    "attn_implementation": "native",
                    "kernel_backend": "native",
                    "compile": False,
                    "diffusion_cache": False,
                    "diffusion_sampling": False,
                },
                context=OptimizationContext(
                    mode="inference",
                    device="cpu",
                ),
            )
            item["baseline_plan"] = {
                "status": "ok",
                "passes": [entry.qualified_id for entry in baseline.passes],
                "decisions": [entry.to_dict() for entry in baseline.decisions],
            }
        except Exception as error:  # noqa: BLE001 - the audit must retain plan failures
            item["baseline_plan"] = {
                "status": "error",
                "error_type": type(error).__name__,
                "error": str(error),
            }
        try:
            optimized = model.resolve_optimization(
                context=OptimizationContext(
                    mode="inference",
                    device="cpu",
                ))
            item["optimized_plan"] = {
                "status": "ok",
                "passes": [entry.qualified_id for entry in optimized.passes],
                "decisions": [entry.to_dict() for entry in optimized.decisions],
            }
        except Exception as error:  # noqa: BLE001 - the audit must retain plan failures
            item["optimized_plan"] = {
                "status": "error",
                "error_type": type(error).__name__,
                "error": str(error),
            }
        providers.append(item)
    return {
        "audit":
        "tts-registry",
        "voicehub_version":
        voicehub_version,
        "provider_count":
        len(providers),
        "coverage_definition": (
            "Lazy wrapper construction plus baseline and automatic static "
            "optimization-plan resolution. This does not claim real-weight inference."),
        "providers":
        providers,
    }


def _load_tensor(path: Path) -> Any:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )


def _write_private_json(path: Path, value: Mapping[str, Any]) -> None:
    """Create a credential-bearing worker payload with owner-only access."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _worker_entry(args: argparse.Namespace) -> int:
    _load_worker_input(args)
    secrets = _secret_strings({
        "config_kwargs": args.config_kwargs,
        "generation_kwargs": args.generation_kwargs,
        "profile_spec": args.worker_profile_spec,
    })
    try:
        effective_spec = _effective_worker_profile_spec(args)
    except Exception:  # noqa: BLE001 - preserve the primary worker validation error
        effective_spec = None
    try:
        result = _benchmark_worker(args)
    except Exception as error:  # noqa: BLE001 - worker errors belong in the JSON report
        from voicehub import __version__ as voicehub_version

        result = {
            "status": "error",
            "profile": args.worker_profile,
            "model_type": args.model_type,
            "checkpoint": args.model,
            "voicehub_version": voicehub_version,
            "effective_profile_spec": effective_spec,
            "checkpoint_identity": {
                "requested_source":
                args.model,
                "requested_revision": (
                    args.config_kwargs.get("revision")
                    if effective_spec is None else effective_spec["config_kwargs"].get("revision")),
                "requested_revision_was_explicit": ((
                    args.config_kwargs.get("revision") if effective_spec is None else
                    effective_spec["config_kwargs"].get("revision")) is not None),
                "resolved_source":
                None,
                "resolved_revision":
                None,
                "local_checkpoint_path":
                None,
                "local_weight_sha256":
                None,
                "weight_digest_status":
                "unresolved",
            },
            "coverage": "real-checkpoint-attempted",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        _write_json(
            args.worker_result,
            _redact_sensitive(result, secrets=secrets),
        )
        return 1
    _write_json(
        args.worker_result,
        _redact_sensitive(result, secrets=secrets),
    )
    return 0


def _main_benchmark(args: argparse.Namespace) -> int:
    from voicehub import __version__ as voicehub_version

    if args.model_type is None:
        raise ValueError("`--model-type` is required unless `--audit-registry` is used.")
    text = args.text
    if args.text_file is not None:
        text = args.text_file.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError("Benchmark text must be non-empty.")
    args.text = text
    effective_specs = benchmark_profile_specs(args)
    secrets = _secret_strings({
        "config_kwargs": args.config_kwargs,
        "generation_kwargs": args.generation_kwargs,
        "profile_specs": effective_specs,
    })
    baseline_spec = effective_specs.get("baseline")
    if baseline_spec is None:
        raise ValueError(
            "TTS benchmarks require a `baseline` profile so optimized "
            "profiles cannot become their own reference.")
    if (baseline_spec["optimization_config"] is not None or baseline_spec["weight_norm_cache"]):
        raise ValueError(
            "The `baseline` profile must use eager inference without an "
            "optimization policy or weight-normalization cache.")
    profile_mode = ("custom" if getattr(args, "profile_specs", None) is not None else "preset")
    command_path = Path(__file__).resolve()
    artifact_context = (
        tempfile.TemporaryDirectory(prefix="voicehub-tts-benchmark-") if args.artifact_dir is None else None)
    artifact_root = (Path(artifact_context.name) if artifact_context is not None else args.artifact_dir)
    artifact_root.mkdir(parents=True, exist_ok=True)
    artifact_run_root = (
        artifact_root if artifact_context is not None else Path(
            tempfile.mkdtemp(prefix="run-", dir=artifact_root)))
    if args.audio_dir is not None:
        args.audio_dir.mkdir(parents=True, exist_ok=True)

    results = []
    waveform_paths: dict[str, Path] = {}
    try:
        for profile, effective_spec in effective_specs.items():
            result_path = artifact_run_root / f"{profile}.json"
            waveform_path = artifact_run_root / f"{profile}.pt"
            worker_input_path = artifact_run_root / f"{profile}.input.json"
            _write_private_json(
                worker_input_path,
                {
                    "config_kwargs": args.config_kwargs,
                    "generation_kwargs": args.generation_kwargs,
                    "model": args.model,
                    "profile": profile,
                    "profile_spec": effective_spec,
                    "text": text,
                },
            )
            audio_path = (
                None if args.audio_dir is None else (args.audio_dir / f"{args.model_type}-{profile}.wav"))
            worker_command = [
                sys.executable,
                str(command_path),
                "--_worker",
                "--worker-profile",
                profile,
                "--worker-result",
                str(result_path),
                "--worker-waveform",
                str(waveform_path),
                "--worker-input",
                str(worker_input_path),
                "--model-type",
                args.model_type,
                "--device",
                args.device,
                "--seed",
                str(args.seed),
                "--warmup",
                str(args.warmup),
                "--repeats",
                str(args.repeats),
                "--minimum-audio-seconds",
                str(args.minimum_audio_seconds),
                "--compile-backend",
                args.compile_backend,
            ]
            if args.compile_mode is not None:
                worker_command.extend(("--compile-mode", args.compile_mode))
            if args.compile_dynamic is True:
                worker_command.append("--compile-dynamic")
            elif args.compile_dynamic is False:
                worker_command.append("--no-compile-dynamic")
            if audio_path is not None:
                worker_command.extend(("--worker-audio", str(audio_path)))
            environment = dict(os.environ)
            environment.setdefault("PYTHONHASHSEED", str(args.seed))
            with tempfile.TemporaryDirectory(prefix=f"voicehub-{profile}-compiler-cache-") as compiler_cache:
                compiler_cache_path = Path(compiler_cache)
                environment["TORCHINDUCTOR_CACHE_DIR"] = str(compiler_cache_path / "torchinductor")
                environment["TRITON_CACHE_DIR"] = str(compiler_cache_path / "triton")
                environment["CUDA_CACHE_PATH"] = str(compiler_cache_path / "cuda")
                try:
                    completed = subprocess.run(
                        worker_command,
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=args.timeout,
                        env=environment,
                    )
                except subprocess.TimeoutExpired as error:
                    result = {
                        "status": "error",
                        "profile": profile,
                        "model_type": args.model_type,
                        "checkpoint": args.model,
                        "coverage": "real-checkpoint-attempted",
                        "error_type": "TimeoutExpired",
                        "error": str(error),
                    }
                else:
                    if not result_path.is_file():
                        result = {
                            "status":
                            "error",
                            "profile":
                            profile,
                            "model_type":
                            args.model_type,
                            "checkpoint":
                            args.model,
                            "coverage":
                            "real-checkpoint-attempted",
                            "error_type":
                            "WorkerProcessError",
                            "error": (
                                f"Worker exited with status {completed.returncode} "
                                "without writing a result."),
                        }
                    else:
                        try:
                            result = json.loads(result_path.read_text(encoding="utf-8"))
                        except (json.JSONDecodeError, OSError) as error:
                            result = {
                                "status":
                                "error",
                                "profile":
                                profile,
                                "model_type":
                                args.model_type,
                                "checkpoint":
                                args.model,
                                "coverage":
                                "real-checkpoint-attempted",
                                "error_type":
                                "InvalidWorkerResult",
                                "error":
                                ("Worker wrote an unreadable result: "
                                 f"{type(error).__name__}: {error}"),
                            }
                        else:
                            if (not isinstance(result, Mapping) or result.get("profile") != profile or
                                    result.get("status") not in {"ok", "error"}):
                                result = {
                                    "status":
                                    "error",
                                    "profile":
                                    profile,
                                    "model_type":
                                    args.model_type,
                                    "checkpoint":
                                    args.model,
                                    "coverage":
                                    "real-checkpoint-attempted",
                                    "error_type":
                                    "InvalidWorkerResult",
                                    "error": (
                                        "Worker result must be an object with "
                                        "the requested profile and status "
                                        "'ok' or 'error'."),
                                }
                    if completed.returncode != 0 and result["status"] == "ok":
                        result = {
                            "status":
                            "error",
                            "profile":
                            profile,
                            "model_type":
                            args.model_type,
                            "checkpoint":
                            args.model,
                            "coverage":
                            "real-checkpoint-attempted",
                            "error_type":
                            "WorkerProcessError",
                            "error": (
                                f"Worker exited with status "
                                f"{completed.returncode} after writing an "
                                "apparently successful result."),
                            "discarded_worker_result":
                            result,
                        }
                    if completed.stderr.strip():
                        result["worker_stderr"] = completed.stderr.strip()
                    result["worker_exit_code"] = completed.returncode
                finally:
                    worker_input_path.unlink(missing_ok=True)
                # The parent resolved and transmitted this exact spec. Record
                # that source of truth even if a failed worker emitted a
                # partial or stale error payload.
                result["effective_profile_spec"] = effective_spec
                result["cold_compiler_cache"] = "fresh-per-profile"
                result = _redact_sensitive(result, secrets=secrets)
                _write_json(result_path, result)
            results.append(result)
            if result["status"] == "ok":
                waveform_paths[profile] = waveform_path

        successful = [result for result in results if result["status"] == "ok"]
        reference_result = next(
            (result for result in successful if result["profile"] == "baseline"),
            None,
        )
        comparisons = {}
        comparison_errors: dict[str, str] = {}
        if reference_result is None and successful:
            comparison_errors["baseline"] = (
                "The baseline worker did not complete successfully; "
                "candidate profiles were not compared.")
        if reference_result is not None:
            reference_profile = reference_result["profile"]
            try:
                reference = _load_tensor(waveform_paths[reference_profile])
                sample_rate = int(reference_result["cold"]["sample_rate"])
            except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
                comparison_errors[reference_profile] = (
                    "Could not load the baseline comparison data: "
                    f"{type(error).__name__}: {error}")
            else:
                reference_identity = reference_result.get(
                    "checkpoint_identity",
                    {},
                )
                for result in successful:
                    profile = result["profile"]
                    candidate_rate = result.get("cold", {}).get("sample_rate")
                    if candidate_rate != sample_rate:
                        comparison_errors[profile] = (
                            "Sample-rate mismatch: baseline uses "
                            f"{sample_rate} Hz and candidate uses "
                            f"{candidate_rate!r}.")
                        continue
                    candidate_identity = result.get("checkpoint_identity", {})
                    identity_error = checkpoint_identity_comparison_error(
                        reference_identity,
                        candidate_identity,
                    )
                    if identity_error is not None:
                        comparison_errors[profile] = identity_error
                        continue
                    try:
                        candidate = _load_tensor(waveform_paths[profile])
                    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
                        comparison_errors[profile] = (
                            "Could not load candidate comparison data: "
                            f"{type(error).__name__}: {error}")
                        continue
                    comparison = compare_waveforms(
                        reference,
                        candidate,
                        sample_rate=sample_rate,
                        absolute_tolerance=args.waveform_atol,
                        relative_tolerance=args.waveform_rtol,
                    )
                    comparison["reference_profile"] = reference_profile
                    comparison["candidate_profile"] = profile
                    comparisons[profile] = comparison

        reference_profile = (None if reference_result is None else reference_result["profile"])
        measured_comparisons = ({} if reference_profile is None else performance_comparisons(
            results,
            comparisons,
            reference_profile=reference_profile,
        ))
        aggregate_identity = (
            reference_result.get("checkpoint_identity", {}) if reference_result is not None else {
                "requested_source":
                args.model,
                "requested_revision":
                next(iter(effective_specs.values()), )["config_kwargs"].get("revision"),
                "requested_revision_was_explicit":
                (next(iter(effective_specs.values()), )["config_kwargs"].get("revision") is not None),
                "resolved_source":
                None,
                "resolved_revision":
                None,
                "local_checkpoint_path":
                None,
                "local_weight_sha256":
                None,
                "weight_digest_status":
                "unresolved",
            })
        benchmark = {
            "benchmark":
            "voicehub-tts-real-checkpoint",
            "command":
            sys.argv,
            "model_type":
            args.model_type,
            "checkpoint":
            args.model,
            "voicehub_version":
            voicehub_version,
            "checkpoint_identity":
            aggregate_identity,
            "text":
            text,
            "text_sha256":
            hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "minimum_audio_seconds":
            args.minimum_audio_seconds,
            "profile_mode":
            profile_mode,
            "profile_specs":
            effective_specs,
            "profiles":
            results,
            "performance_comparisons":
            measured_comparisons,
            "waveform_comparisons":
            comparisons,
            "comparison_errors":
            comparison_errors,
            "quality_scope": (
                "Waveform equality/tolerance at fixed text and benchmark-managed "
                "seed. Presets keep generation arguments fixed; custom profiles "
                "may intentionally vary recorded configuration, generation, and "
                "optimization values. No perceptual quality claim is inferred "
                "from timing alone."),
        }
        public_benchmark = _redact_sensitive(benchmark, secrets=secrets)
        if args.output is not None:
            _write_json(args.output, public_benchmark)
        print(json.dumps(
            public_benchmark,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        ))
        failed = any(result["status"] != "ok" for result in results)
        nondeterministic = any(not result["steady"]["deterministic_across_repeats"] for result in successful)
        inequivalent = any(
            not comparison["within_tolerance"] for profile, comparison in comparisons.items()
            if profile != reference_result["profile"]) if reference_result is not None else False
        return int(
            failed or bool(comparison_errors) or nondeterministic or
            (args.require_waveform_equivalence and inequivalent))
    finally:
        if artifact_context is not None:
            artifact_context.cleanup()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark one real TTS checkpoint with isolated preset or "
            "custom execution profiles."), )
    parser.add_argument("--audit-registry", action="store_true")
    parser.add_argument("--model-type")
    parser.add_argument("--model")
    text_group = parser.add_mutually_exclusive_group()
    text_group.add_argument("--text", default=DEFAULT_TEXT)
    text_group.add_argument("--text-file", type=Path)
    profile_group = parser.add_mutually_exclusive_group()
    profile_group.add_argument(
        "--profiles",
        type=_profiles,
        default=_profiles(",".join(DEFAULT_PROFILE_NAMES)),
        help=("Comma-separated preset profiles. Defaults to "
              f"{','.join(DEFAULT_PROFILE_NAMES)}."),
    )
    profile_group.add_argument(
        "--profile-specs",
        type=_profile_specs,
        help=(
            "JSON object mapping safe profile names to optional "
            "config_kwargs, generation_kwargs, optimization_config, and "
            "weight_norm_cache fields. Cannot be combined with --profiles."),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--warmup", type=_non_negative_int, default=1)
    parser.add_argument("--repeats", type=_positive_int, default=3)
    parser.add_argument("--minimum-audio-seconds", type=_positive_float, default=10.0)
    parser.add_argument("--waveform-atol", type=_non_negative_float, default=1e-5)
    parser.add_argument("--waveform-rtol", type=_non_negative_float, default=1e-4)
    parser.add_argument(
        "--require-waveform-equivalence",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--compile-backend", default="inductor")
    parser.add_argument("--compile-mode")
    parser.add_argument(
        "--compile-dynamic",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--config-kwargs",
        type=lambda value: _json_mapping(value, name="--config-kwargs"),
        default={},
    )
    parser.add_argument(
        "--generation-kwargs",
        type=_generation_mapping,
        default={},
    )
    parser.add_argument("--timeout", type=_positive_float, default=1_800.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--audio-dir", type=Path)

    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-profile",
        type=lambda value: _profile_name(value, option="Worker profile"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--worker-profile-spec",
        type=_worker_profile_spec,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--worker-input", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-result", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-waveform", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-audio", type=Path, help=argparse.SUPPRESS)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args._worker:
        required = {
            "--worker-profile": args.worker_profile,
            "--worker-input": args.worker_input,
            "--worker-result": args.worker_result,
            "--worker-waveform": args.worker_waveform,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError("Worker mode requires " + ", ".join(missing) + ".")
        return _worker_entry(args)
    if args.audit_registry:
        result = audit_registry()
        if args.output is not None:
            _write_json(args.output, result)
        print(json.dumps(result, allow_nan=False, indent=2, sort_keys=True))
        return int(
            any(
                provider["lazy_construction"]["status"] != "ok" or provider["baseline_plan"] is None or
                provider["baseline_plan"]["status"] != "ok" or provider["optimized_plan"] is None or
                provider["optimized_plan"]["status"] != "ok" for provider in result["providers"]))
    return _main_benchmark(args)


if __name__ == "__main__":
    raise SystemExit(main())
