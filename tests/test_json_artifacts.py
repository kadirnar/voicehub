import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from voicehub.hub import read_json_file, write_json_file
from voicehub.training.utils import write_json


class JSONArtifactWriterTest(unittest.TestCase):

    def test_writers_preflight_before_filesystem_mutation(self):
        writers = (
            ("model artifact", write_json_file),
            ("trainer artifact", write_json),
        )
        invalid_payloads = (
            ("non-serializable", {
                "metadata": {
                    "value": object()
                }
            }, TypeError),
            ("non-finite", {
                "metadata": {
                    "loss": float("nan")
                }
            }, ValueError),
        )

        for writer_name, writer in writers:
            for payload_name, payload, error_type in invalid_payloads:
                for existing in (False, True):
                    with self.subTest(
                            writer=writer_name,
                            payload=payload_name,
                            existing=existing,
                    ):
                        with tempfile.TemporaryDirectory() as temporary:
                            destination = Path(temporary) / "nested" / "artifact.json"
                            if existing:
                                destination.parent.mkdir()
                                destination.write_text("existing\n", encoding="utf-8")

                            with self.assertRaises(error_type):
                                writer(destination, payload)

                            if existing:
                                self.assertEqual(
                                    destination.read_text(encoding="utf-8"),
                                    "existing\n",
                                )
                            else:
                                self.assertFalse(destination.parent.exists())

    def test_writers_replace_atomically_and_cleanup_failed_temporary_files(self):
        writers = (
            ("model artifact", write_json_file),
            ("trainer artifact", write_json),
        )

        for writer_name, writer in writers:
            with self.subTest(writer=writer_name):
                with tempfile.TemporaryDirectory() as temporary:
                    destination = Path(temporary) / "artifact.json"
                    destination.write_text("existing\n", encoding="utf-8")

                    with patch(
                            "voicehub.hub.os.replace",
                            side_effect=OSError("replacement failed"),
                    ):
                        with self.assertRaisesRegex(OSError, "replacement failed"):
                            writer(destination, {"status": "new"})

                    self.assertEqual(
                        destination.read_text(encoding="utf-8"),
                        "existing\n",
                    )
                    self.assertEqual(
                        list(destination.parent.glob(f".{destination.name}.*.tmp")),
                        [],
                    )

    def test_writers_emit_the_same_deterministic_document(self):
        writers = (
            ("model artifact", write_json_file),
            ("trainer artifact", write_json),
        )
        payload = {
            "zeta": [2, 1],
            "alpha": {
                "token_count": 3,
            },
        }
        expected = json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ) + "\n"

        for writer_name, writer in writers:
            with self.subTest(writer=writer_name):
                with tempfile.TemporaryDirectory() as temporary:
                    destination = Path(temporary) / "nested" / "artifact.json"
                    result = writer(destination, payload)

                    self.assertEqual(
                        destination.read_text(encoding="utf-8"),
                        expected,
                    )
                    self.assertEqual(read_json_file(destination), payload)
                    if writer is write_json:
                        self.assertEqual(result, destination)
                    else:
                        self.assertIsNone(result)


class JSONArtifactReaderTest(unittest.TestCase):

    def test_reader_rejects_invalid_byte_limits(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "artifact.json"
            source.write_text("{}", encoding="utf-8")

            for value, error_type, diagnostic in (
                (True, TypeError, "integer"),
                (1.5, TypeError, "integer"),
                (0, ValueError, "greater than zero"),
                (-1, ValueError, "greater than zero"),
            ):
                with self.subTest(max_bytes=value):
                    with self.assertRaisesRegex(error_type, diagnostic):
                        read_json_file(source, max_bytes=value)

    def test_reader_rejects_oversized_documents_without_exposing_values(self):
        document = '{"secret": "discarded-value"}'
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "artifact.json"
            source.write_text(document, encoding="utf-8")

            with self.assertRaises(ValueError) as captured:
                read_json_file(source, max_bytes=len(document.encode("utf-8")) - 1)

            rendered_error = str(captured.exception)
            self.assertIn(str(source), rendered_error)
            self.assertIn(str(len(document.encode("utf-8"))), rendered_error)
            self.assertIn("configured limit", rendered_error)
            self.assertNotIn("discarded-value", rendered_error)

    def test_reader_accepts_a_document_at_the_exact_byte_limit(self):
        document = '{"token_count": 3}'
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "artifact.json"
            source.write_text(document, encoding="utf-8")

            self.assertEqual(
                read_json_file(
                    source,
                    max_bytes=len(document.encode("utf-8")),
                ),
                {"token_count": 3},
            )

    def test_reader_rejects_a_document_that_grows_past_the_byte_limit(self):
        document = '{"secret": "discarded-value"}'
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "artifact.json"
            source.write_text(document, encoding="utf-8")

            with patch.object(Path, "stat") as mocked_stat:
                mocked_stat.return_value.st_size = 2
                with self.assertRaises(ValueError) as captured:
                    read_json_file(source, max_bytes=2)

            rendered_error = str(captured.exception)
            self.assertIn(str(source), rendered_error)
            self.assertIn("configured 2-byte limit", rendered_error)
            self.assertNotIn("discarded-value", rendered_error)

    def test_reader_rejects_duplicate_keys_without_exposing_values(self):
        documents = (
            (
                "top-level",
                '{"model_type": "first-secret", "model_type": "second-secret"}',
                "model_type",
            ),
            (
                "nested",
                '{"metadata": {"api_key": "first-secret", "api_key": "second-secret"}}',
                "api_key",
            ),
        )

        for name, document, duplicate_key in documents:
            with self.subTest(document=name):
                with tempfile.TemporaryDirectory() as temporary:
                    source = Path(temporary) / "artifact.json"
                    source.write_text(document, encoding="utf-8")

                    with self.assertRaises(ValueError) as captured:
                        read_json_file(source)

                    rendered_error = str(captured.exception)
                    self.assertIn(str(source), rendered_error)
                    self.assertIn(
                        f"Duplicate JSON object key {duplicate_key!r}",
                        rendered_error,
                    )
                    self.assertNotIn("first-secret", rendered_error)
                    self.assertNotIn("second-secret", rendered_error)

    def test_reader_rejects_non_finite_numbers(self):
        documents = (
            ("nan", '{"metric": NaN}', "NaN"),
            ("positive-infinity", '{"metric": Infinity}', "Infinity"),
            ("negative-infinity", '{"metric": -Infinity}', "-Infinity"),
            ("overflow", '{"metadata": {"loss": 1e10000}}', "$.metadata.loss"),
        )

        for name, document, expected_detail in documents:
            with self.subTest(document=name):
                with tempfile.TemporaryDirectory() as temporary:
                    source = Path(temporary) / "artifact.json"
                    source.write_text(document, encoding="utf-8")

                    with self.assertRaises(ValueError) as captured:
                        read_json_file(source)

                    rendered_error = str(captured.exception)
                    self.assertIn(str(source), rendered_error)
                    self.assertIn(expected_detail, rendered_error)
                    self.assertIn("non-finite JSON", rendered_error)

    def test_auto_config_rejects_ambiguous_model_type_before_dispatch(self):
        from voicehub import AutoConfig

        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "config.json"
            source.write_text(
                '{"model_type": "vits", "model_type": "bark"}',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                    ValueError,
                    "Duplicate JSON object key 'model_type'",
            ):
                AutoConfig.from_pretrained(source)

    def test_reader_keeps_safe_nested_metadata(self):
        payload = {
            "model_type": "vits",
            "metadata": {
                "loss": 0.125,
                "token_count": 3,
            },
        }

        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "artifact.json"
            write_json_file(source, payload)

            self.assertEqual(read_json_file(source), payload)


if __name__ == "__main__":
    unittest.main()
