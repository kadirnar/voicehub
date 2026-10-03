from __future__ import annotations

import base64
import json
import struct
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from voicehub.tokenization import (
    ByteBPETokenizer,
    Encoding,
    SentencePieceModelBPETokenizer,
    SentencePieceUnigramTokenizer,
    SpecialTokenError,
    Tokenizer,
    TokenizerAssetError,
    encode_gpt2_token,
    load_huggingface_byte_bpe,
    load_sentencepiece_model_bpe,
    load_sentencepiece_unigram,
    load_tiktoken_ranks,
    pretokenize,
    read_bounded_asset,
)


def _miniature_vocabulary() -> dict[bytes, int]:
    vocabulary = {bytes((value, )): value for value in range(256)}
    vocabulary.update({
        b"he": 256,
        b"hel": 257,
        b"hell": 258,
        b"hello": 259,
    })
    return vocabulary


def _write_tiktoken(path: Path, vocabulary: dict[bytes, int]) -> None:
    records = [
        base64.b64encode(token) + b" " + str(rank).encode("ascii")
        for token, rank in sorted(vocabulary.items(), key=lambda item: item[1])
    ]
    path.write_bytes(b"\n".join(records) + b"\n")


def _write_tokenizer_json(path: Path) -> None:
    vocabulary = {encode_gpt2_token(token): token_id for token, token_id in _miniature_vocabulary().items()}
    document = {
        "version":
        "1.0",
        "added_tokens": [
            {
                "id": 300,
                "content": "<|end|>",
                "special": True,
                "lstrip": False,
                "rstrip": False,
            },
            {
                "id": 301,
                "content": "<|pad|>",
                "special": True,
                "lstrip": False,
                "rstrip": False,
            },
        ],
        "normalizer": {
            "type": "NFC"
        },
        "pre_tokenizer": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": True,
            "use_regex": True,
        },
        "model": {
            "type": "BPE",
            "vocab": vocabulary,
            "merges": [
                ["h", "e"],
                ["he", "l"],
                ["hel", "l"],
                ["hell", "o"],
            ],
            "unk_token": None,
        },
    }
    path.write_text(json.dumps(document), encoding="utf-8")


def _protobuf_varint(value: int) -> bytes:
    value &= (1 << 64) - 1
    output = bytearray()
    while value >= 0x80:
        output.append((value & 0x7F) | 0x80)
        value >>= 7
    output.append(value)
    return bytes(output)


def _protobuf_integer(field: int, value: int) -> bytes:
    return _protobuf_varint(field << 3) + _protobuf_varint(value)


def _protobuf_bytes(field: int, value: bytes) -> bytes:
    return (_protobuf_varint((field << 3) | 2) + _protobuf_varint(len(value)) + value)


def _sentencepiece_piece(
    text: str,
    score: float,
    piece_type: int = 1,
) -> bytes:
    return b"".join((
        _protobuf_bytes(1, text.encode("utf-8")),
        _protobuf_varint((2 << 3) | 5),
        struct.pack("<f", score),
        _protobuf_integer(3, piece_type),
    ))


def _write_sentencepiece_bpe(path: Path) -> None:
    pieces = (
        ("<unk>", 0.0, 2),
        ("<s>", 0.0, 3),
        ("</s>", 0.0, 3),
        ("\u2581", -10.0, 1),
        ("h", -11.0, 1),
        ("e", -12.0, 1),
        ("l", -13.0, 1),
        ("o", -14.0, 1),
        ("\u2581h", 5.0, 1),
        ("\u2581he", 4.0, 1),
        ("\u2581hel", 3.0, 1),
        ("\u2581hell", 2.0, 1),
        ("\u2581hello", 1.0, 1),
    )
    trainer = b"".join((
        _protobuf_integer(3, 2),
        _protobuf_integer(40, 0),
        _protobuf_integer(41, 1),
        _protobuf_integer(42, 2),
        _protobuf_integer(43, -1),
        _protobuf_bytes(44, b" <unk> "),
    ))
    normalizer = b"".join((
        _protobuf_bytes(1, b"nmt_nfkc"),
        _protobuf_integer(3, 1),
        _protobuf_integer(4, 1),
        _protobuf_integer(5, 1),
    ))
    payload = b"".join((
        *(
            _protobuf_bytes(
                1,
                _sentencepiece_piece(text, score, piece_type),
            ) for text, score, piece_type in pieces),
        _protobuf_bytes(2, trainer),
        _protobuf_bytes(3, normalizer),
    ))
    path.write_bytes(payload)


# Precompiled charsmap that SentencePiece built from the rules
# U+000B -> "", U+200B -> " ", U+FB01 -> "fi", U+0065 U+0301 -> U+00E9, U+3000 -> " ".
_TINY_CHARSMAP = base64.b64decode(
    "AAQAAAAoAAALDQAAAAAAgMwYAgCBBQAABgAAgIAoAgCLPQAAAQAAgIAMAgCACQAArBgCAIEFAAADAACADwAAAA4AAAARAAAA"
    "EAAAABMAAAASAAAAFQAAABQAAAAXAAAAFgAAABkAAAAYAAAAGwAAABoAAAAdAAAAHAAAAB8AAAAeAAAAIQAAACAAAAAjAAAA"
    "IgAAACUAAAAkAAAAJwAAACYAAAApAAAAKAAAACsAAAAqAAAALQAAACwAAAAvAAAALgAAADEAAAAwAAAAMwAAADIAAAA1AAAA"
    "NAAAADcAAAA2AAAAOQAAADgAAAA7AAAAOgAAAD0AAAA8AAAAPwAAAD4AAABBAAAAQAAAAEMAAABCAAAARQAAAEQAAABHAAAA"
    "RgAAAEkAAABIAAAASwAAAEoAAABNAAAATAAAAE8AAABOAAAAUQAAAFAAAABTAAAAUgAAAFUAAABUAAAAVwAAAFYAAABZAAAA"
    "WAAAAFsAAABaAAAAXQAAAFwAAABfAAAAXgAAAGEAAABgAAAAYwAAAGIAAABlAAAAZAAAAGcAAABmAAAAaQAAAGgAAABrAAAA"
    "agAAAG0AAABsAAAAbwAAAGWAAgBxAAAAcAAAAHMAAAByAAAAdQAAAHQAAAB3AAAAdgAAAHkAAAB4AAAAewAAAHoAAAB9AAAA"
    "fAAAAH8AAAB+AAAAgQAAAIAAAACDAAAAggAAAIUAAACEAAAAhwAAAIYAAACJAAAAiAAAAIsAAACKAAAAjQAAAIwAAACPAAAA"
    "jgAAAJEAAACQAAAAkwAAAJIAAACVAAAAlAAAAJcAAACWAAAAmQAAAJgAAACbAAAAmgAAAJ0AAACcAAAAnwAAAJ4AAAChAAAA"
    "oAAAAKMAAACiAAAApQAAAKQAAACnAAAApgAAAKkAAACoAAAAqwAAAKoAAACtAAAArAAAAK8AAACuAAAAsQAAALAAAACzAAAA"
    "sgAAALUAAAC0AAAAtwAAALYAAAC5AAAAuAAAALsAAAC6AAAAvQAAALwAAAC/AAAAvgAAAMEAAADAAAAAwwAAAMIAAADFAAAA"
    "xAAAAMcAAADGAAAAyQAAAMgAAADLAAAAygAAAM0AAADMAAAAzwAAAM4AAADRAAAA0AAAANMAAADSAAAA1QAAANQAAADXAAAA"
    "1gAAANkAAADYAAAA2wAAANoAAADdAAAA3AAAAN8AAADeAAAA4QAAAOAAAADjAAAA4gAAAOUAAADvCAEA5wAAAOYAAADiuAEA"
    "44ABAOsAAADqAAAA7QAAAOwAAADvAAAA7gAAAPEAAADwAAAA8wAAAPIAAAD1AAAA9AAAAPcAAAD2AAAA+QAAAPgAAAD7AAAA"
    "+gAAAP0AAAD8AAAA/wAAAP4AAAAAIABmaQDDqQA=")


def _write_sentencepiece_char(
        path: Path,
        extra_pieces: tuple[str, ...] = (),
        *,
        charsmap: bytes = b"",
) -> None:
    pieces = (
        ("<s>", 0.0, 3),
        ("<pad>", 0.0, 3),
        ("</s>", 0.0, 3),
        ("<unk>", 0.0, 2),
        ("\u2581", -1.0, 1),
        ("i", -2.0, 1),
        ("f", -2.0, 1),
        ("c", -2.0, 1),
        ("a", -2.0, 1),
        ("b", -3.0, 1),
        ("e", -3.0, 1),
        ("d", -3.0, 1),
        *((text, -4.0, 1) for text in extra_pieces),
    )
    trainer = b"".join((
        _protobuf_integer(3, 4),
        _protobuf_integer(40, 3),
        _protobuf_integer(41, 0),
        _protobuf_integer(42, 2),
        _protobuf_integer(43, 1),
    ))
    normalizer = b"".join((
        _protobuf_bytes(1, b"nmt_nfkc"),
        *((_protobuf_bytes(2, charsmap), ) if charsmap else ()),
        _protobuf_integer(3, 1),
        _protobuf_integer(4, 1),
        _protobuf_integer(5, 1),
    ))
    path.write_bytes(
        b"".join((
            *(
                _protobuf_bytes(1, _sentencepiece_piece(text, score, piece_type))
                for text, score, piece_type in pieces),
            _protobuf_bytes(2, trainer),
            _protobuf_bytes(3, normalizer),
        )))


class EncodingContractTests(unittest.TestCase):

    def test_encoding_is_immutable_and_builds_aligned_default_masks(self):
        encoding = Encoding([4, 7])

        self.assertEqual(encoding.ids, (4, 7))
        self.assertEqual(encoding.attention_mask, (1, 1))
        self.assertEqual(encoding.special_tokens_mask, (0, 0))
        with self.assertRaises(FrozenInstanceError):
            encoding.input_ids = (1, )

    def test_encoding_rejects_invalid_or_misaligned_masks(self):
        with self.assertRaisesRegex(ValueError, "length"):
            Encoding((1, 2), attention_mask=(1, ))
        with self.assertRaisesRegex(ValueError, "zeros and ones"):
            Encoding((1, ), special_tokens_mask=(2, ))
        with self.assertRaisesRegex(TypeError, "integer token IDs"):
            Encoding((True, ))


class PretokenizationTests(unittest.TestCase):

    def test_unicode_words_contractions_and_whitespace_preserve_source_text(self):
        text = "We're naïve! 你好\n"

        pieces = pretokenize(text)

        self.assertEqual(pieces, ("We", "'re", " naïve", "!", " 你好", "\n"))
        self.assertEqual("".join(pieces), text)

    def test_only_whisper_compatible_ascii_lowercase_contractions_are_split(self):
        self.assertEqual(
            pretokenize("I’LL we're"),
            ("I", "’", "LL", " we", "'re"),
        )

    def test_internal_whitespace_follows_the_whisper_regex_boundaries(self):
        self.assertEqual(pretokenize("a \nb"), ("a", " ", "\n", "b"))
        self.assertEqual(pretokenize("a \n b"), ("a", " \n", " b"))


class TikTokenAssetTests(unittest.TestCase):

    def test_base64_rank_file_loads_and_encodes_known_miniature_vocabulary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mini.tiktoken"
            _write_tiktoken(path, _miniature_vocabulary())

            tokenizer = ByteBPETokenizer.from_tiktoken_file(path)

        self.assertIsInstance(tokenizer, Tokenizer)
        self.assertEqual(tokenizer.encode("hello").input_ids, (259, ))
        self.assertEqual(tokenizer.decode((259, )), "hello")

    def test_rank_loader_rejects_duplicate_ranks_and_invalid_base64(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            duplicate = root / "duplicate.tiktoken"
            duplicate.write_bytes(b"YQ== 0\nYg== 0\n")
            malformed = root / "malformed.tiktoken"
            malformed.write_bytes(b"%%% 0\n")

            with self.assertRaisesRegex(TokenizerAssetError, "Duplicate rank"):
                load_tiktoken_ranks(duplicate)
            with self.assertRaisesRegex(TokenizerAssetError, "base64"):
                load_tiktoken_ranks(malformed)

    def test_asset_reader_enforces_byte_limit_before_parsing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "large.asset"
            path.write_bytes(b"x" * 20)

            with self.assertRaisesRegex(TokenizerAssetError, "limit"):
                read_bounded_asset(path, max_bytes=10)


class HuggingFaceAssetTests(unittest.TestCase):

    def test_tokenizer_json_loads_without_a_tokenizer_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            _write_tokenizer_json(path)

            assets = load_huggingface_byte_bpe(path)
            tokenizer = ByteBPETokenizer.from_tokenizer_json(
                path,
                pad_token_id=301,
            )

        self.assertEqual(assets.vocabulary[b"hello"], 259)
        self.assertEqual(assets.special_tokens["<|end|>"], 300)
        self.assertEqual(assets.normalization, "NFC")
        self.assertTrue(assets.use_regex)
        self.assertEqual(tokenizer.encode("hello").input_ids, (259, ))

    def test_tokenizer_json_rejects_ambiguous_json_before_model_interpretation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            _write_tokenizer_json(path)
            valid = path.read_text(encoding="utf-8")
            cases = (
                (
                    "duplicate",
                    valid.replace('"model": {', '"model": "discarded-secret", "model": {', 1),
                    r"Duplicate JSON object key 'model'",
                ),
                (
                    "constant",
                    valid[:-1] + ', "metadata": {"score": NaN}}',
                    r"non-finite JSON constant 'NaN'",
                ),
                (
                    "overflow",
                    valid[:-1] + ', "metadata": {"score": 1e400}}',
                    r"\$\.metadata\.score.*non-finite",
                ),
            )
            for name, document, diagnostic in cases:
                with self.subTest(name=name):
                    path.write_text(document, encoding="utf-8")
                    with self.assertRaises(TokenizerAssetError) as captured:
                        load_huggingface_byte_bpe(path)
                    rendered = str(captured.exception)
                    self.assertIn(str(path), rendered)
                    self.assertRegex(rendered, diagnostic)
                    self.assertNotIn("discarded-secret", rendered)

    def test_tokenizer_json_rejects_merges_missing_from_vocabulary(self):
        document = {
            "model": {
                "type": "BPE",
                "vocab": {
                    "a": 0,
                    "b": 1
                },
                "merges": [["a", "b"]],
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            path.write_text(json.dumps(document), encoding="utf-8")

            with self.assertRaisesRegex(TokenizerAssetError, "absent"):
                load_huggingface_byte_bpe(path)


class SentencePieceModelBPETests(unittest.TestCase):

    def test_binary_bpe_model_loads_merges_and_round_trips_without_plugins(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.model"
            _write_sentencepiece_bpe(path)

            assets = load_sentencepiece_model_bpe(path)
            tokenizer = SentencePieceModelBPETokenizer(assets)
            encoded = tokenizer.encode("ｈｅｌｌｏ")

        self.assertEqual(encoded.input_ids, (12, ))
        self.assertEqual(tokenizer.decode(encoded.input_ids), "hello")
        self.assertEqual(tokenizer.encode_as_pieces("hello"), ["▁hello"])
        self.assertEqual(tokenizer.get_piece_size(), 13)

    def test_unigram_and_bpe_loaders_reject_the_wrong_model_family(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.model"
            _write_sentencepiece_bpe(path)

            with self.assertRaisesRegex(TokenizerAssetError, "declares BPE"):
                load_sentencepiece_unigram(path)

    def test_char_model_splits_code_points_and_fuses_unknowns(self):
        # SentencePiece CHAR models (e.g. microsoft/speecht5_tts spm_char.model)
        # split normalized text into code points; adjacent unknowns fuse.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "spm_char.model"
            _write_sentencepiece_char(path)

            tokenizer = SentencePieceUnigramTokenizer.from_model_file(path)

        self.assertEqual(tokenizer.encode_as_ids("ab  ba"), [4, 8, 9, 4, 9, 8])
        self.assertEqual(tokenizer.encode_as_ids("axyb"), [4, 8, 3, 9])
        self.assertEqual(tokenizer.encode_as_pieces("axyb"), ["\u2581", "a", "xy", "b"])
        self.assertEqual(tokenizer.encode_as_ids("ａｂ"), [4, 8, 9])

    def test_precompiled_charsmap_normalizes_like_sentencepiece(self):
        # Expected IDs come from sentencepiece.SentencePieceProcessor on the
        # same model. Python NFKC would keep U+200B (unknown) and split on
        # U+000B instead of removing it.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "spm_char.model"
            _write_sentencepiece_char(path, charsmap=_TINY_CHARSMAP)

            tokenizer = SentencePieceUnigramTokenizer.from_model_file(path)

        cases = {
            "  ab\u200b\u200bc  ": [4, 8, 9, 4, 7],
            "a\x0bb": [4, 8, 9],
            "\ufb01e": [4, 6, 5, 10],
            "e\u0301f": [4, 3, 6],
            "\u3000\u3000": [],
            "a\u3000 \u200bb": [4, 8, 4, 9],
            "\x0b": [],
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(tokenizer.encode_as_ids(text), expected)

    def test_char_model_with_multi_character_pieces_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "spm_char.model"
            _write_sentencepiece_char(path, extra_pieces=("ab", ))

            with self.assertRaisesRegex(TokenizerAssetError, "multi-character"):
                load_sentencepiece_unigram(path)
            with self.assertRaisesRegex(TokenizerAssetError, "UNIGRAM or CHAR"):
                load_sentencepiece_model_bpe(path)

    def test_model_proto_does_not_invent_a_task_postprocessor(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.model"
            _write_sentencepiece_bpe(path)
            tokenizer = SentencePieceModelBPETokenizer.from_model_file(path)

            with self.assertRaisesRegex(ValueError, "postprocessor"):
                tokenizer.encode("hello", add_special_tokens=True)


class ByteBPEBehaviorTests(unittest.TestCase):

    def setUp(self):
        self.tokenizer = ByteBPETokenizer(
            _miniature_vocabulary(),
            special_tokens={
                "<|end|>": 300,
                "<|pad|>": 301
            },
            pad_token_id=301,
        )

    def test_multilingual_and_invalid_boundary_bytes_round_trip(self):
        values = (
            "VoiceHub — Türkçe, 日本語, عربى 🚀",
            "e\u0301 and é",
            "\x00line\nbreak",
        )
        for text in values:
            with self.subTest(text=text):
                encoded = self.tokenizer.encode(text)
                self.assertEqual(self.tokenizer.decode(encoded), text)

    def test_special_tokens_require_an_explicit_policy(self):
        text = "a<|end|>b"

        with self.assertRaisesRegex(SpecialTokenError, r"<\|end\|>"):
            self.tokenizer.encode(text)
        encoded = self.tokenizer.encode(
            text,
            allowed_special={"<|end|>"},
        )
        self.assertEqual(encoded.input_ids, (ord("a"), 300, ord("b")))
        self.assertEqual(encoded.special_tokens_mask, (0, 1, 0))
        self.assertEqual(self.tokenizer.decode(encoded), text)
        self.assertEqual(
            self.tokenizer.decode(encoded, skip_special_tokens=True),
            "ab",
        )

    def test_disallowed_none_treats_special_spelling_as_ordinary_bytes(self):
        encoded = self.tokenizer.encode(
            "<|end|>",
            disallowed_special="none",
        )

        self.assertNotIn(300, encoded.input_ids)
        self.assertEqual(self.tokenizer.decode(encoded), "<|end|>")

    def test_batch_padding_and_masks_are_directly_collatable(self):
        batch = self.tokenizer.encode_batch(
            ["a", "hello!"],
            padding=True,
            pad_to_multiple_of=4,
        )

        self.assertEqual(batch.input_ids[0], (ord("a"), 301, 301, 301))
        self.assertEqual(batch.input_ids[1], (259, ord("!"), 301, 301))
        self.assertEqual(batch.attention_mask, ((1, 0, 0, 0), (1, 1, 0, 0)))
        self.assertEqual(batch.special_tokens_mask, ((0, 1, 1, 1), (0, 0, 1, 1)))
        self.assertTrue(batch.is_padded)

    def test_truncation_is_never_implicit(self):
        with self.assertRaisesRegex(ValueError, "enable truncation"):
            self.tokenizer.encode("abc", max_length=2)

        right = self.tokenizer.encode("abc", max_length=2, truncation=True)
        left = self.tokenizer.encode("abc", max_length=2, truncation="left")
        self.assertEqual(right.input_ids, (ord("a"), ord("b")))
        self.assertEqual(left.input_ids, (ord("b"), ord("c")))

    def test_missing_byte_fails_clearly_instead_of_corrupting_text(self):
        tokenizer = ByteBPETokenizer({b"a": 0})

        with self.assertRaisesRegex(ValueError, "0x62"):
            tokenizer.encode("b")

    def test_regex_splitting_can_be_disabled_for_matching_json_assets(self):
        vocabulary = {
            **{
                bytes((value, )): value
                for value in range(256)
            },
            b"a!": 256,
        }
        merge = ((b"a", b"!"), )

        split = ByteBPETokenizer(vocabulary, merges=merge)
        unsplit = ByteBPETokenizer(vocabulary, merges=merge, use_regex=False)

        self.assertEqual(split.encode("a!").input_ids, (ord("a"), ord("!")))
        self.assertEqual(unsplit.encode("a!").input_ids, (256, ))


class SentencePieceUnigramDecodeTests(unittest.TestCase):

    @staticmethod
    def _tokenizer(*, add_dummy_prefix, remove_extra_whitespaces):
        from voicehub.tokenization.sentencepiece_unigram import (
            SentencePieceUnigramAssets,
            SentencePieceUnigramPiece,
            SentencePieceUnigramTokenizer,
        )

        pieces = (
            SentencePieceUnigramPiece("<unk>", 0.0, 2),
            SentencePieceUnigramPiece("<s>", 0.0, 3),
            SentencePieceUnigramPiece("</s>", 0.0, 3),
            SentencePieceUnigramPiece("\u2581", -1.0),
            SentencePieceUnigramPiece("\u2581the", -1.0),
            SentencePieceUnigramPiece("lo", -1.0),
            SentencePieceUnigramPiece("<|en|>", 0.0, 4),
        )
        return SentencePieceUnigramTokenizer(
            SentencePieceUnigramAssets(
                pieces=pieces,
                unk_token_id=0,
                bos_token_id=1,
                eos_token_id=2,
                pad_token_id=-1,
                unk_surface=" \u2047 ",
                byte_fallback=False,
                normalizer_name="identity",
                add_dummy_prefix=add_dummy_prefix,
                remove_extra_whitespaces=remove_extra_whitespaces,
                escape_whitespaces=True,
                has_precompiled_normalizer=False,
                original_model=b"",
            ))

    def test_decode_strips_only_the_first_emitted_whitespace_marker(self):
        # Expected strings are sentencepiece.SentencePieceProcessor.decode
        # outputs for the same flags (SenseVoiceSmall: dummy prefix off,
        # extra-whitespace removal on).
        for add_dummy_prefix, remove_extra_whitespaces in ((False, True), (True, False), (True, True)):
            tokenizer = self._tokenizer(
                add_dummy_prefix=add_dummy_prefix,
                remove_extra_whitespaces=remove_extra_whitespaces,
            )
            self.assertEqual(tokenizer.decode([4, 5]), "thelo")
            self.assertEqual(
                tokenizer.decode([1, 3, 4, 5]), "thelo" if remove_extra_whitespaces else " thelo")
            self.assertEqual(tokenizer.decode([6, 4]), "<|en|> the")
            self.assertEqual(tokenizer.decode([0, 4]), " \u2047  the")
            self.assertEqual(tokenizer.decode([4, 2, 5]), "thelo")
            # Only extra-whitespace removal keeps stripping bare markers
            # while the text is still empty.
            self.assertEqual(tokenizer.decode([1, 3, 3, 4]), "the" if remove_extra_whitespaces else "  the")
        plain = self._tokenizer(add_dummy_prefix=False, remove_extra_whitespaces=False)
        self.assertEqual(plain.decode([4, 5]), " thelo")


if __name__ == "__main__":
    unittest.main()
