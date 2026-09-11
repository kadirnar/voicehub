from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from voicehub.models.asr_whisper_native.native.artifacts import resolve_whisper_artifacts


class WhisperArtifactTests(unittest.TestCase):

    def test_local_sharded_artifact_is_complete_and_ordered(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(
                json.dumps({"model_type": "whisper"}),
                encoding="utf-8",
            )
            (root / "tokenizer.json").write_text("{}", encoding="utf-8")
            for name in ("part-1.safetensors", "part-2.safetensors"):
                (root / name).write_bytes(b"checkpoint")
            index = root / "model.safetensors.index.json"
            index.write_text(
                json.dumps({"weight_map": {
                    "z": "part-2.safetensors",
                    "a": "part-1.safetensors",
                }}),
                encoding="utf-8",
            )

            artifacts = resolve_whisper_artifacts(root)

        self.assertTrue(artifacts.is_sharded)
        self.assertEqual(artifacts.checkpoint, index.resolve())
        self.assertEqual(artifacts.revision, None)

    def test_missing_or_unsafe_shards_are_rejected(self):
        cases = ("../outside.safetensors", "nested/part.safetensors")
        for shard in cases:
            with self.subTest(shard=shard), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "config.json").write_text("{}", encoding="utf-8")
                (root / "tokenizer.json").write_text("{}", encoding="utf-8")
                (root / "model.safetensors.index.json").write_text(
                    json.dumps({"weight_map": {
                        "weight": shard
                    }}),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, "Unsafe"):
                    resolve_whisper_artifacts(root)

    def test_remote_assets_are_pinned_after_config_resolution(self):
        requests = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {}
            for filename in (
                    "config.json",
                    "tokenizer.json",
                    "model.safetensors",
                    "generation_config.json",
                    "preprocessor_config.json",
            ):
                path = root / filename
                path.write_text("{}", encoding="utf-8")
                paths[filename] = path

            def resolve(repo_id, filename, **kwargs):
                requests.append((repo_id, filename, kwargs["revision"]))
                return paths[filename]

            with (
                    patch(
                        "voicehub.models.asr_whisper_native.native.artifacts."
                        "resolve_pretrained_file",
                        side_effect=resolve,
                    ),
                    patch(
                        "voicehub.models.asr_whisper_native.native.artifacts."
                        "get_cached_hugging_face_commit",
                        return_value="a" * 40,
                    ),
            ):
                artifacts = resolve_whisper_artifacts("owner/whisper")

        self.assertEqual(artifacts.revision, "a" * 40)
        self.assertEqual(requests[0], ("owner/whisper", "config.json", "main"))
        self.assertTrue(all(revision == "a" * 40 for _, _, revision in requests[1:]))


if __name__ == "__main__":
    unittest.main()
