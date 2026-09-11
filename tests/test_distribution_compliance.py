from __future__ import annotations

import json
import tempfile
import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "check_distribution.py"


class DistributionComplianceTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        spec = spec_from_file_location("voicehub_distribution_check_test_module", SCRIPT)
        if spec is None or spec.loader is None:
            raise RuntimeError("Could not import the distribution check script.")
        cls.distribution = module_from_spec(spec)
        spec.loader.exec_module(cls.distribution)

    def test_every_source_manifest_is_pinned_and_licensed(self):
        manifests = tuple(sorted((PROJECT_ROOT / "voicehub").rglob("SOURCE.json")))

        self.assertGreater(len(manifests), 0)
        self.assertEqual(
            self.distribution.validate_provenance_manifests(PROJECT_ROOT),
            len(manifests),
        )

    def test_compliance_inventory_covers_manifests_licenses_and_notices(self):
        files = set(self.distribution.compliance_package_files(PROJECT_ROOT))

        self.assertIn("voicehub/models/bark/native/SOURCE.json", files)
        self.assertIn("voicehub/models/bark/native/THIRD_PARTY_LICENSE", files)
        self.assertIn("voicehub/models/asr_medasr/native/MODEL_TERMS_NOTICE", files)
        self.assertIn("voicehub/models/vibevoice/native/source/THIRD_PARTY_NOTICES.md", files)
        self.assertEqual(
            {path
             for path in files if path.endswith("SOURCE.json")},
            {
                path.relative_to(PROJECT_ROOT).as_posix()
                for path in (PROJECT_ROOT / "voicehub").rglob("SOURCE.json")
            },
        )

    def test_manifest_validation_rejects_missing_license_terms(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "voicehub" / "models" / "example" / "SOURCE.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(
                json.dumps({
                    "upstream": "https://example.invalid",
                    "revision": "abc123"
                }),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "does not record license terms"):
                self.distribution.validate_provenance_manifests(root)

    def test_archive_contract_requires_project_license(self):
        self.distribution.require_project_license(
            "wheel",
            {"voicehub-0.3.0.dist-info/licenses/LICENSE"},
        )
        self.distribution.require_project_license("sdist", {"LICENSE"})

        with self.assertRaisesRegex(RuntimeError, "project LICENSE"):
            self.distribution.require_project_license("wheel", set())
        with self.assertRaisesRegex(RuntimeError, "project LICENSE"):
            self.distribution.require_project_license("sdist", set())


if __name__ == "__main__":
    unittest.main()
