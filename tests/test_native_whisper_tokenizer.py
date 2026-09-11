from __future__ import annotations

import ast
import base64
import json
import tempfile
import unittest
from pathlib import Path

from voicehub.models.asr_whisper_native.native.tokenization import (
    LANGUAGES,
    TIMESTAMP_COUNT,
    WhisperTokenizer,
    WhisperTokenizerFormatError,
    build_openai_whisper_special_tokens,
    discover_whisper_special_tokens,
)
from voicehub.tokenization import encode_gpt2_token


def _miniature_vocabulary(*, reserve_empty_token: bool = True) -> dict[bytes, int]:
    vocabulary = {bytes((value, )): value for value in range(256)}
    vocabulary.update({
        b"he": 256,
        b"hel": 257,
        b"hell": 258,
        b"hello": 259,
    })
    if reserve_empty_token:
        vocabulary[b""] = 260
    return vocabulary


def _write_tiktoken(path: Path) -> None:
    records = []
    for token, rank in sorted(_miniature_vocabulary().items(), key=lambda item: item[1]):
        encoded = b"=" if not token else base64.b64encode(token)
        records.append(encoded + b" " + str(rank).encode("ascii"))
    path.write_bytes(b"\n".join(records) + b"\n")


def _whisper_tokenizer_document() -> dict[str, object]:
    mergeable = _miniature_vocabulary()
    vocabulary = {encode_gpt2_token(token): token_id for token, token_id in mergeable.items()}
    special_tokens = dict(
        build_openai_whisper_special_tokens(
            len(mergeable),
            num_languages=3,
            timestamp_count=4,
        ))
    no_speech_id = special_tokens.pop("<|nospeech|>")
    special_tokens["<|nocaptions|>"] = no_speech_id
    vocabulary["<|endoftext|>"] = special_tokens["<|endoftext|>"]
    added_tokens = []
    for token, token_id in sorted(special_tokens.items(), key=lambda item: item[1]):
        added_tokens.append({
            "id": token_id,
            "content": token,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": token.startswith("<|0."),
            # Official HF assets intentionally do not flag timestamps as
            # special, even though Whisper gives them reserved token IDs.
            "special": not token.startswith("<|0."),
        })
    return {
        "version": "1.0",
        "added_tokens": added_tokens,
        "normalizer": None,
        "pre_tokenizer": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": True,
            "use_regex": True,
        },
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": True,
            "trim_offsets": True,
            "use_regex": True,
        },
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": False,
            "vocab": vocabulary,
            "merges": [
                ["h", "e"],
                ["he", "l"],
                ["hel", "l"],
                ["hell", "o"],
            ],
        },
    }


def _write_tokenizer_json(path: Path) -> None:
    path.write_text(
        json.dumps(_whisper_tokenizer_document()),
        encoding="utf-8",
    )


class WhisperSpecialTokenTests(unittest.TestCase):

    def test_openai_layout_matches_known_multilingual_checkpoint_ids(self):
        tokens = build_openai_whisper_special_tokens(
            50_257,
            num_languages=99,
        )

        self.assertEqual(tokens["<|endoftext|>"], 50_257)
        self.assertEqual(tokens["<|startoftranscript|>"], 50_258)
        self.assertEqual(tokens["<|en|>"], 50_259)
        self.assertEqual(tokens["<|tr|>"], 50_268)
        self.assertEqual(tokens["<|translate|>"], 50_358)
        self.assertEqual(tokens["<|transcribe|>"], 50_359)
        self.assertEqual(tokens["<|notimestamps|>"], 50_363)
        self.assertEqual(tokens["<|0.00|>"], 50_364)
        self.assertEqual(tokens["<|30.00|>"], 51_864)
        self.assertEqual(len(LANGUAGES), 100)
        self.assertEqual(TIMESTAMP_COUNT, 1_501)

    def test_large_v3_layout_includes_cantonese_before_control_tokens(self):
        tokens = build_openai_whisper_special_tokens(
            50_257,
            num_languages=100,
        )

        self.assertEqual(tokens["<|yue|>"], 50_358)
        self.assertEqual(tokens["<|translate|>"], 50_359)
        self.assertEqual(tokens["<|notimestamps|>"], 50_364)
        self.assertEqual(tokens["<|0.00|>"], 50_365)
        self.assertEqual(tokens["<|30.00|>"], 51_865)

    def test_discovery_accepts_legacy_no_speech_alias_and_rejects_gaps(self):
        tokens = dict(build_openai_whisper_special_tokens(
            261,
            num_languages=3,
            timestamp_count=4,
        ))
        no_speech_id = tokens.pop("<|nospeech|>")
        tokens["<|nocaptions|>"] = no_speech_id

        discovered = discover_whisper_special_tokens(
            tokens,
            num_languages=3,
            timestamp_count=4,
        )

        self.assertEqual(discovered.no_speech, no_speech_id)
        self.assertEqual(discovered.language_codes, ("en", "zh", "de"))
        broken = dict(tokens)
        del broken["<|zh|>"]
        with self.assertRaisesRegex(WhisperTokenizerFormatError, "contiguous prefix"):
            discover_whisper_special_tokens(
                broken,
                timestamp_count=4,
            )
        broken = dict(tokens)
        del broken["<|0.04|>"]
        with self.assertRaisesRegex(WhisperTokenizerFormatError, "missing timestamp"):
            discover_whisper_special_tokens(
                broken,
                num_languages=3,
                timestamp_count=4,
            )

    def test_discovery_normalizes_legacy_huggingface_hebrew_token(self):
        tokens = dict(build_openai_whisper_special_tokens(
            261,
            num_languages=21,
            timestamp_count=1,
        ))
        hebrew_id = tokens.pop("<|he|>")
        tokens["<|iw|>"] = hebrew_id

        discovered = discover_whisper_special_tokens(
            tokens,
            num_languages=21,
            timestamp_count=1,
        )

        self.assertEqual(discovered.id_for("<|he|>"), hebrew_id)
        self.assertEqual(discovered.id_for("<|iw|>"), hebrew_id)


class OpenAIWhisperTokenizerTests(unittest.TestCase):

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "mini.tiktoken"
        _write_tiktoken(self.path)
        self.tokenizer = WhisperTokenizer.from_tiktoken_file(
            self.path,
            multilingual=True,
            num_languages=3,
            timestamp_count=4,
        )

    def tearDown(self):
        self.directory.cleanup()

    def test_official_reserved_empty_token_preserves_all_checkpoint_ids(self):
        tokenizer = self.tokenizer

        self.assertEqual(tokenizer.vocabulary_size, 276)
        self.assertEqual(tokenizer.eot, 261)
        self.assertEqual(tokenizer.sot, 262)
        self.assertEqual(tokenizer.no_speech, 270)
        self.assertEqual(tokenizer.no_timestamps, 271)
        self.assertEqual(tokenizer.timestamp_begin, 272)
        self.assertEqual(tokenizer.timestamp_end, 275)
        self.assertEqual(tokenizer.decode_with_timestamps((260, )), "")

    def test_prompt_construction_is_request_local_and_matches_upstreams(self):
        tokenizer = self.tokenizer

        self.assertEqual(tokenizer.sot_sequence, (262, 263, 267))
        self.assertEqual(
            tokenizer.sot_sequence_including_notimestamps,
            (262, 263, 267, 271),
        )
        self.assertEqual(tokenizer.prefix_tokens, (262, 263, 267, 271))
        self.assertEqual(
            tokenizer.get_decoder_prompt_ids(),
            ((1, 263), (2, 267), (3, 271)),
        )
        self.assertEqual(
            tokenizer.prompt_tokens(
                language="Chinese",
                task="translate",
            ),
            (262, 264, 266, 271),
        )
        self.assertEqual(tokenizer.prefix_tokens, (262, 263, 267, 271))
        with self.assertRaisesRegex(KeyError, "not available"):
            tokenizer.to_language_token("Turkish")

    def test_english_only_mode_ignores_multilingual_prompt_options(self):
        tokenizer = WhisperTokenizer.from_tiktoken_file(
            self.path,
            multilingual=False,
            num_languages=3,
            language="Chinese",
            task="translate",
            timestamp_count=4,
        )

        self.assertIsNone(tokenizer.language)
        self.assertIsNone(tokenizer.task)
        self.assertEqual(tokenizer.sot_sequence, (262, ))
        self.assertEqual(tokenizer.prefix_tokens, (262, 271))

    def test_training_labels_frame_text_and_preserve_special_masks(self):
        plain = self.tokenizer.encode("hello")
        framed = self.tokenizer.encode("hello", add_special_tokens=True)

        self.assertEqual(plain.input_ids, (259, ))
        self.assertEqual(framed.input_ids, (262, 263, 267, 271, 259, 261))
        self.assertEqual(framed.attention_mask, (1, 1, 1, 1, 1, 1))
        self.assertEqual(framed.special_tokens_mask, (1, 1, 1, 1, 0, 1))
        self.assertEqual(self.tokenizer.decode(framed, skip_special_tokens=True), "hello")

    def test_batch_padding_and_content_only_truncation_are_explicit(self):
        batch = self.tokenizer.encode_batch(
            ["hello", "hello!"],
            add_special_tokens=True,
            padding=True,
            pad_to_multiple_of=4,
        )

        self.assertEqual(tuple(map(len, batch.input_ids)), (8, 8))
        self.assertEqual(batch.attention_mask[0], (1, 1, 1, 1, 1, 1, 0, 0))
        truncated = self.tokenizer.encode(
            "hello!",
            add_special_tokens=True,
            max_length=6,
            truncation=True,
        )
        self.assertEqual(truncated.input_ids, (262, 263, 267, 271, 259, 261))
        with self.assertRaisesRegex(ValueError, "cannot fit"):
            self.tokenizer.encode(
                "hello",
                add_special_tokens=True,
                max_length=4,
                truncation=True,
            )

    def test_timestamp_helpers_keep_precise_twenty_millisecond_semantics(self):
        timestamp_id = self.tokenizer.token_for_timestamp(0.04)

        self.assertEqual(timestamp_id, 274)
        self.assertTrue(self.tokenizer.is_timestamp(timestamp_id))
        self.assertEqual(self.tokenizer.timestamp_seconds(timestamp_id), 0.04)
        self.assertEqual(
            self.tokenizer.decode_with_timestamps((259, timestamp_id)),
            "hello<|0.04|>",
        )
        self.assertEqual(self.tokenizer.decode((259, timestamp_id)), "hello")
        self.assertEqual(
            self.tokenizer.iter_timestamps((259, timestamp_id))[0].text,
            "<|0.04|>",
        )
        with self.assertRaisesRegex(ValueError, "not aligned"):
            self.tokenizer.token_for_timestamp(0.03)
        self.assertEqual(
            self.tokenizer.token_for_timestamp(0.03, rounding="nearest"),
            274,
        )
        self.assertEqual(
            self.tokenizer.token_for_timestamp(20.0, clamp=True),
            275,
        )


class HuggingFaceWhisperTokenizerTests(unittest.TestCase):

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "tokenizer.json"
        _write_tokenizer_json(self.path)

    def tearDown(self):
        self.directory.cleanup()

    def test_official_tokenizer_json_shape_needs_no_transformers_runtime(self):
        tokenizer = WhisperTokenizer.from_tokenizer_json(
            self.path,
            multilingual=True,
            num_languages=3,
            timestamp_count=4,
        )

        self.assertEqual(tokenizer.encode("hello").input_ids, (259, ))
        self.assertEqual(tokenizer.no_speech, 270)
        self.assertEqual(tokenizer.timestamp_begin, 272)
        self.assertEqual(tokenizer.vocabulary_size, 276)
        self.assertEqual(tokenizer.decode_with_timestamps((260, )), "")

    def test_false_special_timestamp_records_are_still_reserved(self):
        tokenizer = WhisperTokenizer.from_tokenizer_json(
            self.path,
            multilingual=True,
            num_languages=3,
            timestamp_count=4,
        )

        encoded = tokenizer.encoding.encode(
            "<|0.04|>",
            allowed_special={"<|0.04|>"},
        )

        self.assertEqual(encoded.input_ids, (274, ))
        self.assertEqual(encoded.special_tokens_mask, (1, ))

    def test_unsupported_normalization_and_added_tokens_fail_closed(self):
        document = _whisper_tokenizer_document()
        document["normalizer"] = {"type": "NFC"}
        self.path.write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaisesRegex(WhisperTokenizerFormatError, "normalizer"):
            WhisperTokenizer.from_tokenizer_json(
                self.path,
                timestamp_count=4,
            )

        document = _whisper_tokenizer_document()
        document["added_tokens"].append({
            "id": 999,
            "content": "<unsupported>",
            "lstrip": False,
            "rstrip": False,
            "special": False,
        })
        self.path.write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaisesRegex(WhisperTokenizerFormatError, "Unsupported non-special"):
            WhisperTokenizer.from_tokenizer_json(
                self.path,
                timestamp_count=4,
            )


class WhisperTokenizerDependencyTests(unittest.TestCase):

    def test_module_imports_only_standard_library_and_voicehub(self):
        module_path = (
            Path(__file__).parents[1] / "voicehub/models/asr_whisper_native/native/tokenization.py")
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        external = set()
        standard_roots = {
            "__future__",
            "base64",
            "binascii",
            "collections",
            "dataclasses",
            "json",
            "math",
            "numbers",
            "pathlib",
            "string",
            "types",
            "typing",
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = {alias.name.split(".", 1)[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots = {node.module.split(".", 1)[0]}
            else:
                continue
            external.update(roots - standard_roots - {"voicehub"})

        self.assertEqual(external, set())


if __name__ == "__main__":
    unittest.main()
