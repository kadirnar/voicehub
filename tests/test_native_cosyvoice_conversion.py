from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from voicehub.architectures.cosyvoice_native.artifacts import resolve_cosyvoice_artifacts
from voicehub.architectures.cosyvoice_native.configuration import CosyVoiceArchitectureConfig
from voicehub.architectures.cosyvoice_native.metadata import COSYVOICE3_MODEL_ID, COSYVOICE3_MODEL_REVISION
from voicehub.hub import read_json_file

_CONVERTER = "voicehub.converters.cosyvoice"


def _official_snapshot(root: Path) -> Path:
    """Lay out placeholder files with the official snapshot's names."""
    root.mkdir()
    for filename in ("llm.pt", "flow.pt", "hift.pt", "speech_tokenizer_v3.onnx"):
        (root / filename).write_bytes(filename.encode())
    tokenizer = root / "CosyVoice-BlankEN"
    tokenizer.mkdir()
    for filename in ("vocab.json", "merges.txt", "tokenizer_config.json"):
        (tokenizer / filename).write_text(filename, encoding="utf-8")
    return root


def _fake_legacy(calls):

    def convert(module, source, destination, *, component):
        calls.append((component, Path(source).name))
        Path(destination).write_bytes(b"native")
        return Path(destination)

    return convert


def _fake_speech_tokenizer(calls):

    def convert(source, destination):
        calls.append(("speech_tokenizer", Path(source).name))
        Path(destination).write_bytes(b"native")
        return Path(destination)

    return convert


class CosyVoiceConversionEntryPointTests(unittest.TestCase):

    def test_default_hub_error_names_the_packaged_conversion_command(self):
        with mock.patch(
                "voicehub.architectures.cosyvoice_native.artifacts.resolve_pretrained_file",
                side_effect=FileNotFoundError("llm.safetensors"),
        ):
            with self.assertRaises(FileNotFoundError) as raised:
                resolve_cosyvoice_artifacts(COSYVOICE3_MODEL_ID)

        self.assertIn("python -m voicehub.converters.cosyvoice OUTPUT_DIR", str(raised.exception))
        self.assertIn("voicehub[conversion]", str(raised.exception))
        self.assertIsNotNone(importlib.util.find_spec(_CONVERTER))

    def test_hub_conversion_builds_one_complete_native_artifact(self):
        from voicehub.converters.cosyvoice import convert_audited_cosyvoice3_checkpoint

        calls = []
        revisions = set()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = _official_snapshot(root / "snapshot")

            def resolve(repo_id, filename, *, subfolder="", revision=None, **kwargs):
                self.assertEqual(repo_id, COSYVOICE3_MODEL_ID)
                revisions.add(revision)
                return snapshot / subfolder / filename

            with (
                    mock.patch(f"{_CONVERTER}.resolve_pretrained_file", side_effect=resolve),
                    mock.patch(f"{_CONVERTER}.find_spec", return_value=object()),
                    mock.patch(f"{_CONVERTER}.convert_audited_cosyvoice_legacy_checkpoint", _fake_legacy(calls)),
                    mock.patch(
                        f"{_CONVERTER}.convert_audited_cosyvoice_speech_tokenizer",
                        _fake_speech_tokenizer(calls),
                    ),
            ):
                output = convert_audited_cosyvoice3_checkpoint(root / "native")

            artifacts = resolve_cosyvoice_artifacts(output)
            config = CosyVoiceArchitectureConfig.from_dict(read_json_file(artifacts.config))
            self.assertEqual(config, CosyVoiceArchitectureConfig())
            self.assertIsNotNone(artifacts.speech_tokenizer)
            self.assertIsNotNone(artifacts.speech_tokenizer_config)
            self.assertEqual(artifacts.vocab.read_text(encoding="utf-8"), "vocab.json")

        self.assertEqual(revisions, {COSYVOICE3_MODEL_REVISION})
        self.assertEqual(
            calls,
            [
                ("llm", "llm.pt"),
                ("flow", "flow.pt"),
                ("hift", "hift.pt"),
                ("speech_tokenizer", "speech_tokenizer_v3.onnx"),
            ],
        )

    def test_missing_onnx_fails_before_converting_and_can_be_skipped(self):
        from voicehub.converters.cosyvoice import main

        calls = []
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = _official_snapshot(root / "snapshot")
            output = root / "native"
            with (
                    mock.patch(f"{_CONVERTER}.find_spec", return_value=None),
                    mock.patch(f"{_CONVERTER}.convert_audited_cosyvoice_legacy_checkpoint", _fake_legacy(calls)),
                    mock.patch("builtins.print"),
            ):
                with self.assertRaisesRegex(RuntimeError, r"voicehub\[conversion\]"):
                    main([str(output), "--source", str(snapshot)])
                self.assertEqual(calls, [])
                self.assertFalse(output.exists())

                self.assertEqual(
                    main([str(output), "--source", str(snapshot), "--skip-speech-tokenizer"]),
                    0,
                )

            artifacts = resolve_cosyvoice_artifacts(output)
            self.assertIsNone(artifacts.speech_tokenizer)
            self.assertEqual([component for component, _ in calls], ["llm", "flow", "hift"])

    def test_conversion_refuses_a_non_empty_destination(self):
        from voicehub.converters.cosyvoice import convert_audited_cosyvoice3_checkpoint

        with tempfile.TemporaryDirectory() as temporary:
            (Path(temporary) / "llm.safetensors").write_bytes(b"existing")
            with self.assertRaisesRegex(FileExistsError, "must be empty"):
                convert_audited_cosyvoice3_checkpoint(temporary, speech_tokenizer=False)


if __name__ == "__main__":
    unittest.main()
