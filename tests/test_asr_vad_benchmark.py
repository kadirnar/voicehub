from __future__ import annotations

import json
import math
import struct
import subprocess
import sys
import tempfile
import unittest
import wave
from array import array
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "benchmark_asr_vad.py"
RESULTS = (PROJECT_ROOT / "benchmarks" / "asr_vad_rtx4090_2026-07-31.json")


class ASRVADBenchmarkScriptTests(unittest.TestCase):

    @staticmethod
    def _script_module():
        spec = spec_from_file_location(
            "voicehub_asr_vad_benchmark_test_module",
            SCRIPT,
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("Could not import ASR/VAD benchmark script.")
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def _write_long_voice_sample(path: Path) -> None:
        sample_rate = 16_000
        samples = array("h")
        for index in range(sample_rate * 12):
            second = index / sample_rate
            value = (
                0 if second < 1 or second >= 11 else round(12_000 * math.sin(2 * math.pi * 220 * second)))
            samples.append(value)
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(sample_rate)
            output.writeframes(samples.tobytes())

    def test_registry_audit_lazy_constructs_all_public_providers(self):
        completed = subprocess.run(
            [sys.executable, str(SCRIPT), "--audit-registry"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        result = json.loads(completed.stdout)
        self.assertEqual(result["provider_count"], 34)
        self.assertEqual(result["runtime"]["voicehub_version"], "0.4.0")
        self.assertEqual(result["passed"], 34)
        self.assertEqual(result["failed"], 0)
        self.assertEqual(
            {record["task"]
             for record in result["providers"]},
            {
                "automatic-speech-recognition",
                "voice-activity-detection",
            },
        )
        self.assertTrue(
            all(
                record["status"] == "lazy-contract-passed" and record["lazy_runtime_allocated"] is False
                for record in result["providers"]))

    def test_checked_in_results_cover_every_provider(self):
        result = json.loads(RESULTS.read_text(encoding="utf-8"))
        coverage = result["coverage"]

        self.assertEqual(result["voicehub_version"], "0.3.0")
        self.assertEqual(len(coverage), 34)
        self.assertEqual(
            len({record["model_type"]
                 for record in coverage}),
            34,
        )
        self.assertEqual(
            sum(
                result["coverage_summary"][name] for name in (
                    "real_checkpoint_or_algorithm",
                    "tiny_native_graph",
                    "external_checkpoint_blocker",
                )),
            34,
        )
        sherpa = next(
            measurement for measurement in result["vad_measurements"]
            if measurement["model_type"] == "vad_sherpa_onnx")
        self.assertEqual(
            sherpa["profiles"][0]["mean_seconds"],
            1.6402040142255525,
        )

    def test_worker_environment_isolates_all_compiler_caches(self):
        module = self._script_module()
        cache = Path("/tmp/voicehub-test-compiler-cache")
        environment = module._worker_environment(
            "compile",
            compile_cache=cache,
        )

        self.assertEqual(
            environment["TORCHINDUCTOR_CACHE_DIR"],
            str(cache / "torchinductor"),
        )
        self.assertEqual(
            environment["TRITON_CACHE_DIR"],
            str(cache / "triton"),
        )
        self.assertEqual(
            environment["CUDA_CACHE_PATH"],
            str(cache / "cuda"),
        )

    def test_error_rates_are_unicode_aware(self):
        module = self._script_module()

        self.assertEqual(
            module._word_error_rate("你好 世界", "你好 世界"),
            0.0,
        )
        self.assertEqual(
            module._word_error_rate("你好 世界", "你坏 世界"),
            0.5,
        )
        self.assertEqual(
            module._character_error_rate("你好世界", "你坏世界"),
            0.25,
        )

    def test_resolved_revision_supports_remote_and_local_artifacts(self):
        module = self._script_module()

        self.assertEqual(
            module._resolved_checkpoint_revision(
                SimpleNamespace(
                    model=SimpleNamespace(artifacts=SimpleNamespace(revision="commit-123", ), ), ),
                SimpleNamespace(metadata={}),
            ),
            "commit-123",
        )
        self.assertEqual(
            module._resolved_checkpoint_revision(
                SimpleNamespace(),
                SimpleNamespace(metadata={"checkpoint_revision": "main"}, ),
            ),
            "main",
        )
        self.assertIsNone(
            module._resolved_checkpoint_revision(
                SimpleNamespace(),
                SimpleNamespace(metadata={}),
            ))

    def test_worker_rejects_short_audio_before_model_allocation(self):
        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "short.wav"
            with wave.open(str(audio_path), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(16_000)
                output.writeframes(struct.pack("<16000h", *([0] * 16_000)))

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--_worker",
                    "--audio",
                    str(audio_path),
                    "--device",
                    "cpu",
                    "--local-files-only",
                ],
                cwd=PROJECT_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn(
            "at least 10.0 seconds",
            completed.stderr,
        )

    def test_real_algorithmic_vad_matrix_runs_end_to_end(self):
        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "voice.wav"
            self._write_long_voice_sample(audio_path)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--task",
                    "vad",
                    "--audio",
                    str(audio_path),
                    "--model-type",
                    "vad_auditok",
                    "--model-path",
                    "auditok-energy-vad",
                    "--device",
                    "cpu",
                    "--profiles",
                    "eager",
                    "--runs",
                    "2",
                    "--worker-timeout-seconds",
                    "30",
                    "--local-files-only",
                ],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

        result = json.loads(completed.stdout)
        self.assertEqual(result["kind"], "voicehub-vad-profile-matrix")
        self.assertTrue(result["compiler_cache_isolated_per_profile"])
        self.assertEqual(len(result["audio_sha256"]), 64)
        self.assertEqual(len(result["profiles"]), 1)
        profile = result["profiles"][0]
        self.assertEqual(profile["status"], "passed")
        self.assertTrue(profile["deterministic_segments"])
        self.assertGreater(profile["speed_x_realtime"], 0)
        self.assertGreater(profile["speed_x_realtime_mean"], 0)
        self.assertIn("warm_latency_stdev_seconds", profile)
        self.assertTrue(profile["segments"])
        self.assertEqual(
            result["comparisons"][0]["mean_speedup_ratio"],
            1.0,
        )

    def test_timed_out_worker_makes_matrix_command_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "voice.wav"
            self._write_long_voice_sample(audio_path)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--task",
                    "vad",
                    "--audio",
                    str(audio_path),
                    "--model-type",
                    "vad_auditok",
                    "--model-path",
                    "auditok-energy-vad",
                    "--device",
                    "cpu",
                    "--profiles",
                    "eager",
                    "--worker-timeout-seconds",
                    "0.000001",
                    "--local-files-only",
                ],
                cwd=PROJECT_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 1)
        result = json.loads(completed.stdout)
        self.assertEqual(result["profiles"][0]["status"], "timed-out")


if __name__ == "__main__":
    unittest.main()
