from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from voicehub.path_utils import normalize_model_source


def _hub_cache_snapshot(root: Path, files: dict[str, bytes]) -> Path:
    """Build a huggingface_hub-style snapshot of blob symlinks."""
    blobs = root / "blobs"
    snapshot = root / "snapshots" / ("0" * 40)
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    for index, (name, content) in enumerate(files.items()):
        blob = blobs / f"{index:064x}"
        blob.write_bytes(content)
        try:
            (snapshot / name).symlink_to(blob)
        except OSError as error:
            raise unittest.SkipTest(f"Symlinks are unavailable: {error}") from None
    return snapshot


class NormalizeModelSourceTests(unittest.TestCase):

    def test_hub_cache_snapshot_file_keeps_its_name_and_siblings(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = _hub_cache_snapshot(
                Path(directory),
                {
                    "model.pth": b"weights",
                    "config.yml": b"config",
                },
            )
            checkpoint = snapshot / "model.pth"

            normalized = Path(normalize_model_source(str(checkpoint)))

            self.assertEqual(normalized, checkpoint)
            self.assertEqual(normalized.suffix, ".pth")
            self.assertEqual((normalized.parent / "config.yml").read_bytes(), b"config")

    def test_relative_local_path_becomes_absolute_without_resolving(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = _hub_cache_snapshot(root, {"model.pt": b"weights"})
            previous = Path.cwd()
            os.chdir(root)
            try:
                normalized = normalize_model_source(f"./{snapshot.relative_to(root).as_posix()}/model.pt")
                expected = Path.cwd() / snapshot.relative_to(root) / "model.pt"
            finally:
                os.chdir(previous)

            self.assertEqual(Path(normalized), expected)
            self.assertEqual(Path(normalized).name, "model.pt")

    def test_hub_identifiers_pass_through_and_missing_paths_fail(self):
        self.assertEqual(normalize_model_source("organization/model"), "organization/model")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "was not found"):
                normalize_model_source(Path(directory) / "missing")

    def test_vui_snapshot_checkpoint_finds_its_sibling_codec(self):
        from voicehub.models.vui.artifacts import VUI_CODEC_FILENAME, VUI_MODEL_FILENAME, resolve_vui_artifacts
        from voicehub.models.vui.inference import VuiForTextToSpeech

        with tempfile.TemporaryDirectory() as directory:
            snapshot = _hub_cache_snapshot(
                Path(directory),
                {
                    VUI_MODEL_FILENAME: b"model",
                    VUI_CODEC_FILENAME: b"codec",
                },
            )
            model = VuiForTextToSpeech(model_path=str(snapshot / VUI_MODEL_FILENAME), device="cpu")

            artifacts = resolve_vui_artifacts(
                model.config.name_or_path,
                model_filename=model.config.checkpoint_filename,
                codec_filename=model.config.codec_filename,
            )

            self.assertEqual(artifacts.model_checkpoint.read_bytes(), b"model")
            self.assertEqual(artifacts.codec_checkpoint.read_bytes(), b"codec")


if __name__ == "__main__":
    unittest.main()
