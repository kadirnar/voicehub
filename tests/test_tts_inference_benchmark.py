from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

import voicehub

PROJECT_ROOT = Path(__file__).parents[1]
RESULTS = (PROJECT_ROOT / "benchmarks" / "tts_vits_rtx4090_2026-07-31.json")
VUI_REJECTED_RESULTS = (PROJECT_ROOT / "benchmarks" / "tts_vui_rtx4090_rejected_2026-07-31.json")
REPORT = (PROJECT_ROOT / "docs" / "guides" / "rtx-4090-speech-benchmarks.md")


def _load_benchmark_module():
    path = PROJECT_ROOT / "scripts" / "benchmark_tts_inference.py"
    spec = importlib.util.spec_from_file_location(
        "voicehub_test_tts_inference_benchmark",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the TTS inference benchmark module.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


benchmark = _load_benchmark_module()


class TTSInferenceBenchmarkTests(unittest.TestCase):

    def test_all_credential_aliases_and_authenticated_urls_are_redacted(self):
        aliases = (
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
        )
        payload = {key: f"private-{key}" for key in aliases}
        payload["source"] = "https://alice:secret@example.com/private/model"
        payload["query_source"] = (
            "https://example.com/model?X-Amz-Credential=user"
            "&X-Amz-Signature=private-signature")
        secrets = benchmark._secret_strings(payload)

        redacted = benchmark._redact_sensitive(
            payload,
            secrets=secrets,
        )

        for key in aliases:
            with self.subTest(key=key):
                self.assertEqual(redacted[key], "<redacted>")
                self.assertNotIn(payload[key], json.dumps(redacted))
        self.assertEqual(
            redacted["source"],
            "https://<redacted>@example.com/private/model",
        )
        self.assertEqual(
            redacted["query_source"],
            "https://example.com/model?<redacted>",
        )

    def test_vui_compile_speedups_remain_rejected_by_the_quality_gate(self):
        result = json.loads(VUI_REJECTED_RESULTS.read_text(encoding="utf-8"), )
        profiles = {profile["profile"]: profile for profile in result["results"]}

        self.assertEqual(result["voicehub_version"], "0.3.0")
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(
            set(profiles),
            {
                "compile-dynamic",
                "compile-specialized",
            },
        )
        for profile in profiles.values():
            with self.subTest(profile=profile["profile"]):
                self.assertGreater(
                    profile["candidate"]["speedup_ratio"],
                    1.0,
                )
                self.assertNotEqual(
                    profile["candidate"]["samples"],
                    profile["baseline"]["samples"],
                )
                self.assertGreater(
                    profile["candidate"]["whisper_wer"],
                    profile["baseline"]["whisper_wer"],
                )
                self.assertFalse(profile["waveform_equivalent"])
                self.assertFalse(profile["accepted"])
        self.assertEqual(
            result["policy"],
            {
                "inference_torch_compile":
                "rejected",
                "training_torch_compile":
                "available",
                "reason": (
                    "Measured inference speedups are not reported as usable "
                    "because both tested shape policies failed the quality gate."),
            },
        )

    def test_checked_in_evidence_is_pinned_and_matches_report(self):
        result = json.loads(RESULTS.read_text(encoding="utf-8"))
        report = REPORT.read_text(encoding="utf-8")
        checkpoint = result["checkpoint"]
        matrix = {profile["profile"]: profile for profile in result["clean_candidate_matrix"]}
        accepted = result["accepted_weight_norm_cache"]["primary"]

        self.assertEqual(result["voicehub_version"], "0.3.0")
        self.assertEqual(checkpoint["requested_revision"], "main")
        self.assertEqual(
            checkpoint["resolved_revision"],
            "c71de0fe7204c83f1c10820a7d696d0b450048ba",
        )
        self.assertEqual(len(checkpoint["weight_sha256"]), 64)
        self.assertEqual(
            set(matrix),
            {
                "baseline",
                "triton",
                "compile",
                "triton-compile",
                "float16-cache",
                "bfloat16-cache",
            },
        )
        self.assertTrue(all(profile["deterministic_across_repeats"] for profile in matrix.values()))
        baseline = matrix["baseline"]
        for profile in matrix.values():
            with self.subTest(profile=profile["profile"]):
                self.assertAlmostEqual(
                    baseline["mean_latency_seconds"] / profile["mean_latency_seconds"],
                    profile["mean_speedup_ratio"],
                )
                self.assertAlmostEqual(
                    baseline["median_latency_seconds"] / profile["median_latency_seconds"],
                    profile["median_speedup_ratio"],
                )
                self.assertIn(
                    (
                        f"{profile['mean_latency_seconds'] * 1000:.3f} / "
                        f"{profile['median_latency_seconds'] * 1000:.3f} ms"),
                    report,
                )
        self.assertTrue(accepted["quality"]["exact"])
        self.assertAlmostEqual(
            (
                accepted["baseline"]["median_latency_seconds"] /
                accepted["candidate"]["median_latency_seconds"]),
            accepted["comparison"]["median_speedup_ratio"],
        )
        self.assertIn(RESULTS.relative_to(PROJECT_ROOT).as_posix(), report)
        self.assertIn(
            f"{accepted['candidate']['median_latency_seconds'] * 1000:.3f} ms",
            report,
        )
        self.assertIn(
            f"{accepted['candidate']['median_real_time_factor']:.5f}",
            report,
        )
        self.assertIn(checkpoint["resolved_revision"], report)
        self.assertIn(checkpoint["weight_sha256"], report)
        self.assertIn(
            f"{result['fine_tuning_smoke']['loss_before']:.6f}",
            report,
        )

    def test_checkpoint_identity_records_revision_and_explicit_weight_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.safetensors"
            checkpoint.write_bytes(b"reproducible-weights")
            model = SimpleNamespace(
                config=SimpleNamespace(revision="config-revision"),
                model=SimpleNamespace(
                    artifacts=SimpleNamespace(
                        source="resolved/repository",
                        revision="a" * 40,
                        checkpoint=checkpoint,
                    )),
            )

            result = benchmark.checkpoint_identity(
                model,
                requested_source="requested/repository",
                requested_revision="requested-tag",
            )

        self.assertEqual(result["requested_source"], "requested/repository")
        self.assertEqual(result["requested_revision"], "requested-tag")
        self.assertTrue(result["requested_revision_was_explicit"])
        self.assertEqual(result["resolved_source"], "resolved/repository")
        self.assertEqual(result["resolved_revision"], "a" * 40)
        self.assertEqual(result["local_checkpoint_path"], str(checkpoint.resolve()))
        self.assertEqual(
            result["local_weight_sha256"],
            hashlib.sha256(b"reproducible-weights").hexdigest(),
        )
        self.assertEqual(result["weight_digest_status"], "sha256")

        model.config = SimpleNamespace(model_type="vits", revision=None)
        defaulted = benchmark.checkpoint_identity(
            model,
            requested_source="requested/repository",
            requested_revision=None,
        )
        self.assertEqual(defaulted["requested_revision"], "main")
        self.assertFalse(defaulted["requested_revision_was_explicit"])

    def test_checkpoint_identity_recognizes_local_hugging_face_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            model_root = Path(directory) / "models--owner--model"
            snapshot = model_root / "snapshots" / ("a" * 40)
            snapshot.mkdir(parents=True)

            result = benchmark.checkpoint_identity(
                SimpleNamespace(config=SimpleNamespace(revision=None)),
                requested_source=snapshot,
            )

        self.assertEqual(result["resolved_source"], str(model_root.resolve()))
        self.assertEqual(result["resolved_revision"], "a" * 40)
        self.assertIsNone(benchmark.checkpoint_identity_comparison_error(result, result), )

    def test_profiles_only_select_quality_candidate_execution_changes(self):
        self.assertIsNone(
            benchmark.profile_optimization_config(
                "baseline",
                compile_backend="inductor",
                compile_mode=None,
                compile_dynamic=None,
            ))
        self.assertIsNone(
            benchmark.profile_optimization_config(
                "weight-norm-cache",
                compile_backend="inductor",
                compile_mode=None,
                compile_dynamic=None,
            ))
        self.assertIsNone(
            benchmark.profile_optimization_config(
                "float16-cache",
                compile_backend="inductor",
                compile_mode=None,
                compile_dynamic=None,
            ))
        self.assertIsNone(
            benchmark.profile_optimization_config(
                "bfloat16-cache",
                compile_backend="inductor",
                compile_mode=None,
                compile_dynamic=None,
            ))
        triton = benchmark.profile_optimization_config(
            "triton",
            compile_backend="inductor",
            compile_mode=None,
            compile_dynamic=None,
        )
        compiled = benchmark.profile_optimization_config(
            "compile",
            compile_backend="inductor",
            compile_mode="reduce-overhead",
            compile_dynamic=True,
        )
        combined = benchmark.profile_optimization_config(
            "triton-compile",
            compile_backend="inductor",
            compile_mode=None,
            compile_dynamic=None,
        )

        self.assertEqual(triton["kernel_backend"], "triton")
        self.assertFalse(triton["compile"])
        self.assertFalse(triton["diffusion_cache"])
        self.assertFalse(triton["diffusion_sampling"])
        self.assertEqual(compiled["kernel_backend"], "native")
        self.assertTrue(compiled["compile"])
        self.assertEqual(compiled["compile_config"]["mode"], "reduce-overhead")
        self.assertTrue(compiled["compile_config"]["dynamic"])
        self.assertEqual(combined["kernel_backend"], "triton")
        self.assertTrue(combined["compile"])

    def test_only_explicit_cache_profiles_enable_weight_norm_cache(self):
        self.assertFalse(benchmark.profile_uses_weight_norm_cache("baseline"))
        self.assertFalse(benchmark.profile_uses_weight_norm_cache("triton"))
        self.assertFalse(benchmark.profile_uses_weight_norm_cache("compile"))
        self.assertFalse(benchmark.profile_uses_weight_norm_cache("triton-compile"))
        self.assertTrue(benchmark.profile_uses_weight_norm_cache("weight-norm-cache"))
        self.assertTrue(benchmark.profile_uses_weight_norm_cache("float16-cache"))
        self.assertTrue(benchmark.profile_uses_weight_norm_cache("bfloat16-cache"))
        with self.assertRaisesRegex(ValueError, "Unknown benchmark profile"):
            benchmark.profile_uses_weight_norm_cache("unknown")

    def test_custom_profile_specs_merge_global_values_and_preserve_presets(self):
        parser = benchmark._parser()
        defaults = parser.parse_args([])
        default_specs = benchmark.benchmark_profile_specs(defaults)
        self.assertEqual(
            tuple(default_specs),
            benchmark.DEFAULT_PROFILE_NAMES,
        )
        self.assertIsNone(default_specs["baseline"]["optimization_config"])
        self.assertTrue(default_specs["compile"]["optimization_config"]["compile"])

        profile_specs = {
            "quality-baseline": {
                "config_kwargs": {
                    "language": "en",
                },
                "generation_kwargs": {
                    "voice": "M1",
                },
                "optimization_config": None,
                "weight_norm_cache": False,
            },
            "bf16.compile": {
                "config_kwargs": {
                    "torch_dtype": "bfloat16",
                },
                "generation_kwargs": {
                    "total_steps": 8,
                },
                "optimization_config": {
                    "attn_implementation": "sdpa",
                    "compile": True,
                },
                "weight_norm_cache": True,
            },
        }
        args = parser.parse_args([
            "--config-kwargs",
            '{"revision":"pinned","torch_dtype":"float32"}',
            "--generation-kwargs",
            '{"speed":1.0}',
            "--profile-specs",
            json.dumps(profile_specs),
        ])
        effective = benchmark.benchmark_profile_specs(args)

        self.assertEqual(tuple(effective), tuple(profile_specs))
        self.assertEqual(
            effective["quality-baseline"]["config_kwargs"],
            {
                "language": "en",
                "revision": "pinned",
                "torch_dtype": "float32",
            },
        )
        self.assertEqual(
            effective["quality-baseline"]["generation_kwargs"],
            {
                "speed": 1.0,
                "voice": "M1",
            },
        )
        self.assertIsNone(effective["quality-baseline"]["optimization_config"])
        self.assertEqual(
            effective["bf16.compile"]["config_kwargs"]["torch_dtype"],
            "bfloat16",
        )
        self.assertEqual(
            effective["bf16.compile"]["generation_kwargs"],
            {
                "speed": 1.0,
                "total_steps": 8,
            },
        )
        self.assertEqual(
            effective["bf16.compile"]["optimization_config"],
            {
                "attn_implementation": "sdpa",
                "compile": True,
            },
        )
        self.assertTrue(effective["bf16.compile"]["weight_norm_cache"])

    def test_custom_profile_schema_fails_closed(self):
        invalid_specs = (
            ({}, "at least one"),
            ({
                "../escape": {}
            }, "names must match"),
            ({
                "candidate": []
            }, "must be a JSON object"),
            ({
                "candidate": {
                    "unknown": True
                }
            }, "unknown field"),
            (
                {
                    "candidate": {
                        "config_kwargs": []
                    }
                },
                "config_kwargs",
            ),
            (
                {
                    "candidate": {
                        "optimization_config": []
                    }
                },
                "optimization_config",
            ),
            (
                {
                    "candidate": {
                        "weight_norm_cache": 1
                    }
                },
                "true or false",
            ),
            (
                {
                    "candidate": {
                        "generation_kwargs": {
                            "seed": 4
                        }
                    }
                },
                "benchmark-managed",
            ),
            (
                {
                    "candidate": {
                        "generation_kwargs": {
                            "output_file": "unmanaged.wav",
                        },
                    },
                },
                "benchmark-managed",
            ),
        )
        for value, message in invalid_specs:
            with (
                    self.subTest(value=value),
                    self.assertRaisesRegex(
                        argparse.ArgumentTypeError,
                        message,
                    ),
            ):
                benchmark._profile_specs(json.dumps(value))

        with self.assertRaisesRegex(
                argparse.ArgumentTypeError,
                "benchmark-managed",
        ):
            benchmark._generation_mapping('{"seed":12}')

    def test_profiles_and_profile_specs_are_mutually_exclusive(self):
        parser = benchmark._parser()
        with (
                patch("sys.stderr", new=io.StringIO()),
                self.assertRaises(SystemExit),
        ):
            parser.parse_args([
                "--profiles",
                "baseline",
                "--profile-specs",
                '{"candidate":{}}',
            ])

    def test_custom_profiles_are_isolated_and_record_effective_specs(self):
        calls = []

        def fake_run(command, **kwargs):
            profile = command[command.index("--worker-profile") + 1]
            result_path = Path(command[command.index("--worker-result") + 1])
            worker_input_path = Path(command[command.index("--worker-input") + 1])
            worker_input = json.loads(worker_input_path.read_text(encoding="utf-8"))
            effective = worker_input["profile_spec"]
            environment = kwargs["env"]
            calls.append({
                "profile": profile,
                "effective": effective,
                "worker_input": worker_input_path,
                "worker_command": command,
                "compiler_cache": environment["TORCHINDUCTOR_CACHE_DIR"],
                "triton_cache": environment["TRITON_CACHE_DIR"],
                "cuda_cache": environment["CUDA_CACHE_PATH"],
            })
            result_path.write_text(
                json.dumps({
                    "status": "error",
                    "profile": profile,
                    "model_type": "vits",
                    "coverage": "real-checkpoint-attempted",
                    "error_type": "DeliberateTestStop",
                    "error": "No checkpoint was loaded.",
                }),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(
                command,
                1,
                stdout="",
                stderr="",
            )

        profile_specs = {
            "baseline": {
                "generation_kwargs": {
                    "speaking_rate": 1.0,
                },
            },
            "compile-fast": {
                "config_kwargs": {
                    "torch_dtype": "bfloat16",
                },
                "optimization_config": {
                    "compile": True,
                },
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "result.json"
            args = benchmark._parser().parse_args([
                "--model-type",
                "vits",
                "--device",
                "cpu",
                "--profile-specs",
                json.dumps(profile_specs),
                "--artifact-dir",
                str(root / "artifacts"),
                "--output",
                str(output),
            ])
            with (
                    patch.object(
                        benchmark.subprocess,
                        "run",
                        side_effect=fake_run,
                    ),
                    patch("builtins.print"),
            ):
                return_code = benchmark._main_benchmark(args)
            result = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(return_code, 1)
        self.assertEqual(
            [call["profile"] for call in calls],
            list(profile_specs),
        )
        compiler_roots = {str(Path(call["compiler_cache"]).parent) for call in calls}
        self.assertEqual(len(compiler_roots), len(profile_specs))
        for call in calls:
            root = Path(call["compiler_cache"]).parent
            self.assertNotIn("--worker-profile-spec", call["worker_command"])
            self.assertNotIn("--config-kwargs", call["worker_command"])
            self.assertNotIn("--generation-kwargs", call["worker_command"])
            self.assertFalse(call["worker_input"].exists())
            self.assertEqual(
                Path(call["triton_cache"]).parent,
                root,
            )
            self.assertEqual(
                Path(call["cuda_cache"]).parent,
                root,
            )
        self.assertEqual(result["profile_mode"], "custom")
        self.assertEqual(
            result["profile_specs"]["compile-fast"]["config_kwargs"]["torch_dtype"],
            "bfloat16",
        )
        self.assertTrue(result["profile_specs"]["compile-fast"]["optimization_config"]["compile"], )
        for profile in result["profiles"]:
            self.assertEqual(
                profile["effective_profile_spec"],
                result["profile_specs"][profile["profile"]],
            )
            self.assertEqual(
                profile["cold_compiler_cache"],
                "fresh-per-profile",
            )

    def test_optimized_only_profile_cannot_be_its_own_reference(self):
        args = benchmark._parser().parse_args([
            "--model-type",
            "vits",
            "--profiles",
            "compile",
        ])

        with self.assertRaisesRegex(
                ValueError,
                "require a `baseline` profile",
        ):
            benchmark._main_benchmark(args)

    def test_nonzero_worker_exit_discards_apparently_successful_payload(self):

        def fake_run(command, **_kwargs):
            result_path = Path(command[command.index("--worker-result") + 1])
            waveform_path = Path(command[command.index("--worker-waveform") + 1])
            result_path.write_text(
                json.dumps({
                    "status": "ok",
                    "profile": "baseline",
                    "steady": {
                        "deterministic_across_repeats": True,
                    },
                }),
                encoding="utf-8",
            )
            torch.save(torch.zeros(8), waveform_path)
            return subprocess.CompletedProcess(
                command,
                139,
                stdout="",
                stderr="segmentation fault",
            )

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            args = benchmark._parser().parse_args([
                "--model-type",
                "vits",
                "--profiles",
                "baseline",
                "--artifact-dir",
                str(Path(directory) / "artifacts"),
                "--output",
                str(output),
            ])
            with (
                    patch.object(
                        benchmark.subprocess,
                        "run",
                        side_effect=fake_run,
                    ),
                    patch("builtins.print"),
            ):
                return_code = benchmark._main_benchmark(args)
            result = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(return_code, 1)
        profile = result["profiles"][0]
        self.assertEqual(profile["status"], "error")
        self.assertEqual(profile["error_type"], "WorkerProcessError")
        self.assertEqual(profile["worker_exit_code"], 139)
        self.assertEqual(
            profile["discarded_worker_result"]["status"],
            "ok",
        )

    def test_credentials_are_private_in_transit_and_redacted_at_rest(self):
        secret = "hf_voicehub_private_test_token"
        observed_input = None

        def fake_run(command, **_kwargs):
            nonlocal observed_input
            self.assertNotIn(secret, " ".join(command))
            input_path = Path(command[command.index("--worker-input") + 1])
            if os.name != "nt":
                self.assertEqual(input_path.stat().st_mode & 0o777, 0o600)
            observed_input = json.loads(input_path.read_text(encoding="utf-8"))
            result_path = Path(command[command.index("--worker-result") + 1])
            result_path.write_text(
                json.dumps({
                    "status": "error",
                    "profile": "baseline",
                    "model_type": "vits",
                    "error_type": "FixtureError",
                    "error": f"credential was {secret}",
                    "effective_profile_spec": observed_input["profile_spec"],
                }),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(
                command,
                1,
                stdout="",
                stderr=f"credential was {secret}",
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "result.json"
            artifact_root = root / "artifacts"
            args = benchmark._parser().parse_args([
                "--model-type",
                "vits",
                "--profiles",
                "baseline",
                "--config-kwargs",
                json.dumps({"token": secret}),
                "--artifact-dir",
                str(artifact_root),
                "--output",
                str(output),
            ])
            with (
                    patch.object(
                        benchmark.subprocess,
                        "run",
                        side_effect=fake_run,
                    ),
                    patch("builtins.print"),
            ):
                return_code = benchmark._main_benchmark(args)
            output_text = output.read_text(encoding="utf-8")
            artifact_text = "\n".join(
                path.read_text(encoding="utf-8") for path in artifact_root.rglob("*.json"))

        self.assertEqual(return_code, 1)
        self.assertIsNotNone(observed_input)
        self.assertEqual(
            observed_input["config_kwargs"]["token"],
            secret,
        )
        self.assertNotIn(secret, output_text)
        self.assertNotIn(secret, artifact_text)
        self.assertIn("<redacted>", output_text)

    def test_nondeterministic_repeats_fail_the_benchmark(self):

        def fake_run(command, **_kwargs):
            result_path = Path(command[command.index("--worker-result") + 1])
            waveform_path = Path(command[command.index("--worker-waveform") + 1])
            torch.save(torch.zeros(160_000), waveform_path)
            result_path.write_text(
                json.dumps({
                    "status": "ok",
                    "profile": "baseline",
                    "model_type": "vits",
                    "checkpoint": "fixture",
                    "coverage": "real-checkpoint-end-to-end",
                    "checkpoint_identity": {},
                    "cold": {
                        "sample_rate": 16_000,
                        "latency_seconds": 1.0,
                        "memory": {
                            "peak_allocated_bytes": 100,
                            "peak_reserved_bytes": 200,
                        },
                    },
                    "steady": {
                        "mean_latency_seconds": 1.0,
                        "median_latency_seconds": 1.0,
                        "memory": {
                            "peak_allocated_bytes": 100,
                            "peak_reserved_bytes": 200,
                        },
                        "deterministic_across_repeats": False,
                    },
                }),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="",
                stderr="",
            )

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            args = benchmark._parser().parse_args([
                "--model-type",
                "vits",
                "--profiles",
                "baseline",
                "--artifact-dir",
                str(Path(directory) / "artifacts"),
                "--output",
                str(output),
            ])
            with (
                    patch.object(
                        benchmark.subprocess,
                        "run",
                        side_effect=fake_run,
                    ),
                    patch("builtins.print"),
            ):
                return_code = benchmark._main_benchmark(args)

        self.assertEqual(return_code, 1)

    def test_cross_profile_sample_rate_and_checkpoint_mismatches_fail(self):
        identities = {
            "baseline": "a" * 64,
            "rate-mismatch": "a" * 64,
            "checkpoint-mismatch": "b" * 64,
        }

        def fake_run(command, **_kwargs):
            profile = command[command.index("--worker-profile") + 1]
            result_path = Path(command[command.index("--worker-result") + 1])
            waveform_path = Path(command[command.index("--worker-waveform") + 1])
            sample_rate = 24_000 if profile == "rate-mismatch" else 16_000
            torch.save(torch.zeros(sample_rate * 10), waveform_path)
            result_path.write_text(
                json.dumps({
                    "status": "ok",
                    "profile": profile,
                    "model_type": "vits",
                    "checkpoint": "fixture",
                    "coverage": "real-checkpoint-end-to-end",
                    "checkpoint_identity": {
                        "requested_source": "fixture",
                        "requested_revision": "revision",
                        "resolved_source": "fixture",
                        "resolved_revision": "revision",
                        "local_weight_sha256": identities[profile],
                    },
                    "cold": {
                        "sample_rate": sample_rate,
                        "latency_seconds": 1.0,
                        "memory": {
                            "peak_allocated_bytes": 100,
                            "peak_reserved_bytes": 200,
                        },
                    },
                    "steady": {
                        "mean_latency_seconds": 1.0,
                        "median_latency_seconds": 1.0,
                        "memory": {
                            "peak_allocated_bytes": 100,
                            "peak_reserved_bytes": 200,
                        },
                        "deterministic_across_repeats": True,
                    },
                }),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="",
                stderr="",
            )

        specs = {
            "baseline": {},
            "rate-mismatch": {},
            "checkpoint-mismatch": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            args = benchmark._parser().parse_args([
                "--model-type",
                "vits",
                "--profile-specs",
                json.dumps(specs),
                "--artifact-dir",
                str(Path(directory) / "artifacts"),
                "--output",
                str(output),
            ])
            with (
                    patch.object(
                        benchmark.subprocess,
                        "run",
                        side_effect=fake_run,
                    ),
                    patch("builtins.print"),
            ):
                return_code = benchmark._main_benchmark(args)
            result = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(return_code, 1)
        self.assertIn(
            "Sample-rate mismatch",
            result["comparison_errors"]["rate-mismatch"],
        )
        self.assertIn(
            "local_weight_sha256",
            result["comparison_errors"]["checkpoint-mismatch"],
        )
        self.assertNotIn(
            "rate-mismatch",
            result["performance_comparisons"],
        )
        self.assertNotIn(
            "checkpoint-mismatch",
            result["performance_comparisons"],
        )

    def test_checkpoint_comparisons_require_one_immutable_shared_anchor(self):
        mutable = {
            "requested_source": "mutable-local-dir",
            "resolved_source": None,
            "resolved_revision": None,
            "local_weight_sha256": None,
        }
        self.assertIn(
            "shared SHA-256",
            benchmark.checkpoint_identity_comparison_error(
                mutable,
                mutable,
            ),
        )

        digest = {
            **mutable,
            "local_weight_sha256": "a" * 64,
        }
        self.assertIsNone(benchmark.checkpoint_identity_comparison_error(digest, digest), )

        revision = {
            **mutable,
            "resolved_source": "owner/model",
            "resolved_revision": "b" * 40,
        }
        self.assertIsNone(benchmark.checkpoint_identity_comparison_error(
            revision,
            revision,
        ), )

    def test_waveform_comparison_distinguishes_exact_tolerated_and_shape_change(self):
        reference = torch.tensor([0.0, 0.25, -0.5, 1.0])
        exact = benchmark.compare_waveforms(
            reference,
            reference.clone(),
            sample_rate=2,
            absolute_tolerance=1e-5,
            relative_tolerance=1e-4,
        )
        tolerated = benchmark.compare_waveforms(
            reference,
            reference + 1e-6,
            sample_rate=2,
            absolute_tolerance=1e-5,
            relative_tolerance=1e-4,
        )
        changed = benchmark.compare_waveforms(
            reference,
            reference[:-1],
            sample_rate=2,
            absolute_tolerance=1e-5,
            relative_tolerance=1e-4,
        )

        self.assertTrue(exact["exact"])
        self.assertTrue(exact["within_tolerance"])
        self.assertFalse(tolerated["exact"])
        self.assertTrue(tolerated["within_tolerance"])
        self.assertFalse(changed["same_length"])
        self.assertFalse(changed["within_tolerance"])
        self.assertEqual(changed["duration_delta_seconds"], -0.5)

    def test_performance_comparisons_report_speed_memory_and_quality(self):
        baseline = {
            "status": "ok",
            "profile": "baseline",
            "cold": {
                "latency_seconds": 4.0,
                "memory": {
                    "peak_allocated_bytes": 1_000,
                    "peak_reserved_bytes": 2_000,
                },
            },
            "steady": {
                "mean_latency_seconds": 2.0,
                "median_latency_seconds": 1.5,
                "memory": {
                    "peak_allocated_bytes": 800,
                    "peak_reserved_bytes": 1_200,
                },
            },
        }
        candidate = {
            "status": "ok",
            "profile": "candidate",
            "cold": {
                "latency_seconds": 2.0,
                "memory": {
                    "peak_allocated_bytes": 700,
                    "peak_reserved_bytes": 1_600,
                },
            },
            "steady": {
                "mean_latency_seconds": 1.0,
                "median_latency_seconds": 1.0,
                "memory": {
                    "peak_allocated_bytes": 600,
                    "peak_reserved_bytes": 1_500,
                },
            },
        }

        result = benchmark.performance_comparisons(
            [baseline, candidate, {
                "status": "error",
                "profile": "failed"
            }],
            {
                "baseline": {
                    "within_tolerance": True,
                    "exact": True
                },
                "candidate": {
                    "within_tolerance": True,
                    "exact": False
                },
            },
            reference_profile="baseline",
        )

        comparison = result["candidate"]
        self.assertTrue(comparison["waveform_equivalence_passed"])
        self.assertFalse(comparison["waveform_exact"])
        self.assertEqual(
            comparison["steady_mean_latency"]["speedup_ratio"],
            2.0,
        )
        self.assertEqual(
            comparison["steady_mean_latency"]["latency_reduction_percent"],
            50.0,
        )
        self.assertEqual(
            comparison["steady_peak_allocated_memory"]["candidate_minus_baseline_bytes"],
            -200,
        )
        self.assertEqual(
            comparison["steady_peak_allocated_memory"]["reduction_percent"],
            25.0,
        )
        self.assertEqual(
            comparison["steady_peak_reserved_memory"]["reduction_percent"],
            -25.0,
        )
        self.assertNotIn("failed", result)

    def test_performance_comparison_guards_missing_and_zero_metrics(self):
        baseline = {
            "status": "ok",
            "profile": "baseline",
            "cold": {
                "latency_seconds": 0.0,
                "memory": None
            },
            "steady": {
                "mean_latency_seconds": 0.0,
                "median_latency_seconds": 0.0,
                "memory": None,
            },
        }
        result = benchmark.performance_comparisons(
            [baseline],
            {"baseline": {
                "within_tolerance": True,
                "exact": True
            }},
            reference_profile="baseline",
        )["baseline"]

        self.assertIsNone(result["cold_latency"]["speedup_ratio"])
        self.assertIsNone(result["steady_mean_latency"]["latency_reduction_percent"])
        self.assertIsNone(result["steady_peak_allocated_memory"]["reduction_percent"])
        self.assertIsNone(result["steady_peak_reserved_memory"]["candidate_minus_baseline_bytes"])

    def test_audio_validation_enforces_duration_and_finite_samples(self):

        class Output:
            sample_rate = 4

            def __init__(self, audio):
                self.audio = audio

        audio, metadata = benchmark._audio_result(
            Output(torch.ones(40)),
            torch,
            minimum_audio_seconds=10.0,
        )
        self.assertEqual(audio.numel(), 40)
        self.assertEqual(metadata["duration_seconds"], 10.0)

        with self.assertRaisesRegex(RuntimeError, "below the required"):
            benchmark._audio_result(
                Output(torch.ones(39)),
                torch,
                minimum_audio_seconds=10.0,
            )
        with self.assertRaisesRegex(RuntimeError, "NaN or infinite"):
            benchmark._audio_result(
                Output(torch.tensor([float("nan")] * 40)),
                torch,
                minimum_audio_seconds=10.0,
            )

    def test_registry_audit_covers_every_registered_tts_provider(self):
        result = benchmark.audit_registry()

        self.assertEqual(result["voicehub_version"], voicehub.__version__)
        self.assertEqual(result["provider_count"], 34)
        self.assertEqual(len(result["providers"]), 34)
        self.assertEqual(
            len({provider["model_type"]
                 for provider in result["providers"]}),
            34,
        )
        for provider in result["providers"]:
            with self.subTest(model_type=provider["model_type"]):
                self.assertEqual(provider["lazy_construction"]["status"], "ok")
                self.assertEqual(provider["baseline_plan"]["status"], "ok")
                self.assertEqual(provider["optimized_plan"]["status"], "ok")
                self.assertFalse(provider["lazy_construction"]["loaded"])
                self.assertEqual(
                    provider["coverage"],
                    "lazy-construction-and-static-plan",
                )
                self.assertEqual(
                    provider["real_weights"]["status"],
                    "not-attempted",
                )


if __name__ == "__main__":
    unittest.main()
