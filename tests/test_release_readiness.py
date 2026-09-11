from __future__ import annotations

import io
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "check_release.py"
WORKFLOW = PROJECT_ROOT / ".github" / "workflows" / "release.yml"
PACKAGE_WORKFLOW = PROJECT_ROOT / ".github" / "workflows" / "package_testing.yml"


class ReleaseReadinessTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        spec = spec_from_file_location("voicehub_release_check_test_module", SCRIPT)
        if spec is None or spec.loader is None:
            raise RuntimeError("Could not import the release check script.")
        cls.release = module_from_spec(spec)
        spec.loader.exec_module(cls.release)

    @staticmethod
    def _write_distributions(dist_dir: Path, version: str) -> None:
        metadata = f"Metadata-Version: 2.4\nName: voicehub\nVersion: {version}\n\n"
        wheel = dist_dir / f"voicehub-{version}-py3-none-any.whl"
        with ZipFile(wheel, "w", compression=ZIP_DEFLATED) as archive:
            archive.writestr(f"voicehub-{version}.dist-info/METADATA", metadata)

        sdist = dist_dir / f"voicehub-{version}.tar.gz"
        encoded = metadata.encode("utf-8")
        with tarfile.open(sdist, "w:gz") as archive:
            member = tarfile.TarInfo(f"voicehub-{version}/PKG-INFO")
            member.size = len(encoded)
            archive.addfile(member, io.BytesIO(encoded))

    def test_rewrite_preserves_historical_evidence_and_requires_new_release_gates(self):
        version = self.release.source_version(PROJECT_ROOT)
        self.assertEqual(version, "0.4.0")
        self.release.validate_source_metadata(version, PROJECT_ROOT)
        self.release.validate_documentation_version(version, PROJECT_ROOT)
        # Do not relabel old measurements or silently claim codec checkpoint coverage.
        with self.assertRaisesRegex(self.release.ReleaseCheckError, "records versions"):
            self.release.validate_benchmark_versions(version, PROJECT_ROOT)
        with self.assertRaisesRegex(self.release.ReleaseCheckError, "missing evidence"):
            self.release.validate_layered_evidence(PROJECT_ROOT)
        self.assertEqual(self.release.validate_benchmark_versions("0.3.0", PROJECT_ROOT), 5)

    def test_layered_evidence_requires_explicit_external_blockers(self):
        filenames = (
            "tts_optimization_rtx4090_2026-07-31.json",
            "asr_vad_rtx4090_2026-07-31.json",
            "tts_vits_rtx4090_2026-07-31.json",
            "tts_vui_rtx4090_rejected_2026-07-31.json",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark_dir = root / "benchmarks"
            benchmark_dir.mkdir()
            for filename in filenames:
                shutil.copy2(PROJECT_ROOT / "benchmarks" / filename, benchmark_dir / filename)
            shutil.copytree(
                PROJECT_ROOT / "docs" / "models" / "providers",
                root / "docs" / "models" / "providers",
            )
            path = benchmark_dir / "asr_vad_rtx4090_2026-07-31.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            blocker = next(
                row for row in payload["coverage"] if row["verification"] == "external-checkpoint-blocker")
            blocker.pop("blocker")
            path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(
                    self.release.ReleaseCheckError,
                    "invalid verification rows",
            ):
                self.release.validate_layered_evidence(root)

    def test_distribution_pair_embeds_the_candidate_version(self):
        with tempfile.TemporaryDirectory() as directory:
            dist_dir = Path(directory)
            self._write_distributions(dist_dir, "0.3.0")

            sizes = self.release.validate_distributions(dist_dir, "0.3.0")

        self.assertEqual(
            set(sizes),
            {
                "voicehub-0.3.0-py3-none-any.whl",
                "voicehub-0.3.0.tar.gz",
            },
        )
        self.assertTrue(all(size > 0 for size in sizes.values()))

    def test_distribution_pair_rejects_an_unexpected_file(self):
        with tempfile.TemporaryDirectory() as directory:
            dist_dir = Path(directory)
            self._write_distributions(dist_dir, "0.3.0")
            (dist_dir / "unreviewed.bin").write_bytes(b"not a distribution")

            with self.assertRaisesRegex(
                    self.release.ReleaseCheckError,
                    "only the expected wheel and sdist",
            ):
                self.release.validate_distributions(dist_dir, "0.3.0")

    def test_pypi_candidate_must_be_newer_and_unpublished(self):
        payload = {
            "info": {
                "version": "0.1.6"
            },
            "releases": {
                "0.1.6": [{
                    "filename": "voicehub-0.1.6.tar.gz"
                }]
            },
        }

        self.assertEqual(
            self.release.validate_pypi_payload(payload, "0.3.0", "candidate"),
            "0.1.6",
        )
        payload["releases"]["0.3.0"] = [{"filename": "voicehub-0.3.0.tar.gz"}]
        with self.assertRaisesRegex(
                self.release.ReleaseCheckError,
                "already present on PyPI",
        ):
            self.release.validate_pypi_payload(payload, "0.3.0", "candidate")

    def test_pypi_post_publish_requires_exact_external_parity(self):
        payload = {
            "info": {
                "version": "0.3.0"
            },
            "releases": {
                "0.3.0": [{
                    "filename": "voicehub-0.3.0-py3-none-any.whl"
                }]
            },
        }

        self.assertEqual(
            self.release.validate_pypi_payload(payload, "0.3.0", "published"),
            "0.3.0",
        )

    def test_release_workflow_separates_verification_from_oidc_publish(self):
        source = WORKFLOW.read_text(encoding="utf-8")

        self.assertIn("workflow_dispatch:", source)
        self.assertIn("confirm_publish:", source)
        self.assertIn("default: false", source)
        self.assertIn("tagged-cross-platform-tests:", source)
        self.assertIn(
            "operating-system: [ubuntu-latest, windows-latest, macos-latest]",
            source,
        )
        self.assertIn('python-version: ["3.10", "3.11", "3.12"]', source)
        self.assertIn("needs: tagged-cross-platform-tests", source)
        self.assertIn("--require-tag-at-head", source)
        self.assertIn("--pypi-policy candidate", source)
        self.assertIn("- tagged-cross-platform-tests\n      - verify-and-build", source)
        self.assertEqual(source.count('VOICEHUB_FULL_RUNTIME_TEST: "1"'), 2)
        self.assertIn("Verify the pinned release assets", source)
        self.assertIn('VOICEHUB_TEST_RELEASE_ASSETS: "1"', source)
        self.assertIn("tests/test_native_sensevoice.py", source)
        self.assertIn("tests/test_native_speechbrain_asr.py", source)
        self.assertIn("release_tokenizer_matches_published_sentencepiece", source)
        self.assertIn("Verify the TEN VAD ONNX differential oracle", source)
        self.assertIn('onnxruntime==1.22.1', source)
        self.assertIn("22a3bcd4509d0faaa8eef4881e8af5f39c178950", source)
        self.assertIn("VOICEHUB_TEN_VAD_ONNX", source)
        self.assertIn("Verify the NVIDIA QuartzNet checkpoint conversion", source)
        self.assertIn("stt_en_quartznet15x5/versions/1.0.0rc1", source)
        self.assertIn("VOICEHUB_TEST_NEMO_QUARTZNET_CHECKPOINT", source)
        self.assertIn("Verify the WeNet GigaSpeech checkpoint and tokenizer", source)
        self.assertIn("openspeech/wenet-models/resolve/90acd57d17169a15d5ceab462c6e7db3bd003921", source)
        self.assertIn("061ccfa51d64ebe7ea091a5a13ae31e37d9c36f4eface5c7bafc80bd4a06b26e", source)
        self.assertIn("VOICEHUB_TEST_WENET_CHECKPOINT", source)
        self.assertIn("VOICEHUB_TEST_WENET_ASSETS", source)
        self.assertIn("real_tokenizer_uses_sorted_wenet_unit_ids", source)
        self.assertIn("name: pypi", source)
        self.assertIn("id-token: write", source)
        self.assertIn(
            "pypa/gh-action-pypi-publish@"
            "ec4db0b4ddc65acdf4bff5fa45ac92d78b56bdf0",
            source,
        )
        self.assertNotIn("PYPI_API_TOKEN", source)
        self.assertNotIn("password:", source)

    def test_package_ci_runs_the_complete_distribution_contract(self):
        source = PACKAGE_WORKFLOW.read_text(encoding="utf-8")

        self.assertIn("python scripts/check_release.py --dist-dir dist", source)
        self.assertIn("python scripts/check_distribution.py", source)
        self.assertIn("Verify the pinned release assets", source)
        self.assertIn('VOICEHUB_TEST_RELEASE_ASSETS: "1"', source)
        self.assertIn("tests/test_native_sensevoice.py", source)
        self.assertIn("tests/test_native_speechbrain_asr.py", source)
        self.assertIn("release_tokenizer_matches_published_sentencepiece", source)
        self.assertIn("Verify the TEN VAD ONNX differential oracle", source)
        self.assertIn('onnxruntime==1.22.1', source)
        self.assertIn("22a3bcd4509d0faaa8eef4881e8af5f39c178950", source)
        self.assertIn("VOICEHUB_TEN_VAD_ONNX", source)
        self.assertIn("Verify the NVIDIA QuartzNet checkpoint conversion", source)
        self.assertIn("stt_en_quartznet15x5/versions/1.0.0rc1", source)
        self.assertIn("VOICEHUB_TEST_NEMO_QUARTZNET_CHECKPOINT", source)
        self.assertIn("Verify the WeNet GigaSpeech checkpoint and tokenizer", source)
        self.assertIn("openspeech/wenet-models/resolve/90acd57d17169a15d5ceab462c6e7db3bd003921", source)
        self.assertIn("061ccfa51d64ebe7ea091a5a13ae31e37d9c36f4eface5c7bafc80bd4a06b26e", source)
        self.assertIn("VOICEHUB_TEST_WENET_CHECKPOINT", source)
        self.assertIn("VOICEHUB_TEST_WENET_ASSETS", source)
        self.assertIn("real_tokenizer_uses_sorted_wenet_unit_ids", source)
        self.assertIn("Import every integration from the installed wheel", source)

    def test_release_script_rejects_stale_evidence_for_the_rewrite(self):
        completed = subprocess.run(
            [sys.executable, str(SCRIPT)],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("records versions", completed.stderr)
        self.assertIn("expected only 0.4.0", completed.stderr)

    def test_development_metadata_check_does_not_claim_checkpoint_evidence(self):
        completed = subprocess.run(
            [sys.executable, str(SCRIPT), "--metadata-only"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        report = json.loads(completed.stdout)
        self.assertEqual(report["version"], "0.4.0")
        self.assertEqual(report["source_metadata"], "passed")
        self.assertEqual(report["checkpoint_evidence"], "not evaluated (metadata-only)")
        rejected = subprocess.run(
            [sys.executable, str(SCRIPT), "--metadata-only", "--tag", "v0.4.0"],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("cannot be used for tag or publication checks", rejected.stderr)


if __name__ == "__main__":
    unittest.main()
