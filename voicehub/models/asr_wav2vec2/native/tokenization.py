"""Dependency-free character tokenization and CTC decoding for Wav2Vec2.

Semantics were reviewed against Hugging Face Transformers revision
``ebea912f0bb6f9e28ad2df04acd9b4df035933a9``.  The implementation reads
declarative vocabulary data only and does not import ``transformers``,
``tokenizers``, or NumPy.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from types import MappingProxyType
from typing import Any

from voicehub.tokenization import (
    BatchEncoding,
    Encoding,
    PaddingStrategy,
    SpecialTokenSelection,
    TruncationStrategy,
    pad_encodings,
    read_bounded_asset,
)

_MAX_VOCABULARY_SIZE = 2_000_000


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Vocabulary JSON contains invalid constant {value!r}.")


def _json_object(path: str | Path) -> dict[str, Any]:
    payload = read_bounded_asset(path)

    def reject_duplicates(pairs: list[tuple[str, Any]], ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, value in pairs:
            if name in result:
                raise ValueError(f"Vocabulary JSON contains duplicate key {name!r}.")
            result[name] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid vocabulary JSON: {error}.") from error
    if not isinstance(value, dict):
        raise ValueError("Wav2Vec2 vocabulary JSON must be an object.")
    return value


def _validated_vocabulary(
    values: Mapping[str, Any],
    *,
    context: str,
) -> Mapping[str, int]:
    if not isinstance(values, Mapping) or not values:
        raise ValueError(f"{context} must contain a non-empty vocabulary.")
    if len(values) > _MAX_VOCABULARY_SIZE:
        raise ValueError(f"{context} contains more than {_MAX_VOCABULARY_SIZE} tokens.")
    vocabulary: dict[str, int] = {}
    seen_ids: set[int] = set()
    for token, token_id in values.items():
        if not isinstance(token, str) or not token:
            raise ValueError(f"{context} token strings must be non-empty.")
        if (isinstance(token_id, bool) or not isinstance(token_id, Integral) or token_id < 0):
            raise ValueError(f"{context} token IDs must be non-negative integers.")
        normalized_id = int(token_id)
        if normalized_id in seen_ids:
            raise ValueError(f"{context} contains duplicate token ID {normalized_id}.")
        vocabulary[token] = normalized_id
        seen_ids.add(normalized_id)
    expected_ids = set(range(len(vocabulary)))
    if seen_ids != expected_ids:
        raise ValueError(
            f"{context} token IDs must be contiguous from zero through "
            f"{len(vocabulary) - 1}.")
    return MappingProxyType(vocabulary)


def _special_selection(
    value: SpecialTokenSelection,
    *,
    name: str,
    available: frozenset[str],
) -> frozenset[str]:
    if value == "all":
        return available
    if value == "none":
        return frozenset()
    if isinstance(value, str) or not isinstance(value, Collection):
        raise TypeError(f"`{name}` must be 'all', 'none', or a collection of tokens.")
    selected = frozenset(value)
    if any(not isinstance(token, str) for token in selected):
        raise TypeError(f"`{name}` collections must contain strings.")
    unknown = selected - available
    if unknown:
        names = ", ".join(sorted(repr(token) for token in unknown))
        raise ValueError(f"`{name}` contains unknown special tokens: {names}.")
    return selected


@dataclass(frozen=True, slots=True)
class CTCCharacterOffset:
    """One collapsed CTC token and its half-open logit-frame interval."""

    token: str
    start_offset: int
    end_offset: int

    def __post_init__(self) -> None:
        if not isinstance(self.token, str) or not self.token:
            raise ValueError("A CTC character offset requires a token.")
        for name in ("start_offset", "end_offset"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"`{name}` must be an integer.")
            if value < 0:
                raise ValueError(f"`{name}` cannot be negative.")
        if self.end_offset <= self.start_offset:
            raise ValueError("A CTC character interval must have positive width.")


@dataclass(frozen=True, slots=True)
class CTCWordOffset:
    """One decoded word and its half-open logit-frame interval."""

    word: str
    start_offset: int
    end_offset: int

    def __post_init__(self) -> None:
        if not isinstance(self.word, str) or not self.word.strip():
            raise ValueError("A CTC word offset requires non-empty text.")
        object.__setattr__(self, "word", self.word.strip())
        if (isinstance(self.start_offset, bool) or not isinstance(self.start_offset, int) or
                isinstance(self.end_offset, bool) or not isinstance(self.end_offset, int)):
            raise TypeError("CTC word offsets must be integers.")
        if self.start_offset < 0 or self.end_offset <= self.start_offset:
            raise ValueError("A CTC word interval must be positive.")


@dataclass(frozen=True, slots=True)
class Wav2Vec2CTCDecodeOutput:
    """Decoded text plus optional character and word frame offsets."""

    text: str
    char_offsets: tuple[CTCCharacterOffset, ...] = ()
    word_offsets: tuple[CTCWordOffset, ...] = ()


class Wav2Vec2CTCTokenizer:
    """Validated character tokenizer with native greedy CTC decoding."""

    def __init__(
        self,
        vocabulary: Mapping[str, Any],
        *,
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        word_delimiter_token: str = "|",
        replace_word_delimiter_char: str = " ",
        do_lower_case: bool = False,
        target_language: str | None = None,
    ) -> None:
        if not isinstance(vocabulary, Mapping) or not vocabulary:
            raise ValueError("Wav2Vec2 vocabulary must be a non-empty mapping.")
        nested = all(isinstance(value, Mapping) for value in vocabulary.values())
        if nested:
            languages = {
                str(language):
                _validated_vocabulary(
                    value,
                    context=f"Wav2Vec2 vocabulary language {language!r}",
                )
                for language, value in vocabulary.items()
            }
            if target_language is None:
                if len(languages) != 1:
                    choices = ", ".join(sorted(languages))
                    raise ValueError(
                        "A nested Wav2Vec2 vocabulary requires "
                        f"`target_language`; available languages: {choices}.")
                target_language = next(iter(languages))
            if target_language not in languages:
                choices = ", ".join(sorted(languages))
                raise ValueError(
                    f"Unknown target language {target_language!r}; choose "
                    f"one of: {choices}.")
            self._language_vocabularies = MappingProxyType(languages)
            resolved_vocabulary = languages[target_language]
        elif any(isinstance(value, Mapping) for value in vocabulary.values()):
            raise ValueError("Wav2Vec2 vocabulary cannot mix tokens and language maps.")
        else:
            self._language_vocabularies = None
            resolved_vocabulary = _validated_vocabulary(
                vocabulary,
                context="Wav2Vec2 vocabulary",
            )
            if target_language is not None:
                raise ValueError("`target_language` requires a nested multilingual "
                                 "vocabulary.")

        for name, token in (
            ("bos_token", bos_token),
            ("eos_token", eos_token),
            ("unk_token", unk_token),
            ("pad_token", pad_token),
            ("word_delimiter_token", word_delimiter_token),
            ("replace_word_delimiter_char", replace_word_delimiter_char),
        ):
            if not isinstance(token, str) or not token:
                raise ValueError(f"`{name}` must be a non-empty string.")
        if not isinstance(do_lower_case, bool):
            raise TypeError("`do_lower_case` must be a boolean.")

        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.word_delimiter_token = word_delimiter_token
        self.replace_word_delimiter_char = replace_word_delimiter_char
        self.do_lower_case = do_lower_case
        self.target_language = target_language
        self._set_vocabulary(resolved_vocabulary)

    def _set_vocabulary(self, vocabulary: Mapping[str, int]) -> None:
        required = {
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "unk_token": self.unk_token,
            "pad_token": self.pad_token,
            "word_delimiter_token": self.word_delimiter_token,
        }
        missing = [f"{name}={token!r}" for name, token in required.items() if token not in vocabulary]
        if missing:
            raise ValueError("Wav2Vec2 vocabulary is missing required token(s): " + ", ".join(missing) + ".")
        self._vocabulary = MappingProxyType(dict(vocabulary))
        self._decoder = MappingProxyType({token_id: token for token, token_id in vocabulary.items()})
        self._special_tokens = frozenset({
            self.bos_token,
            self.eos_token,
            self.unk_token,
            self.pad_token,
        })
        self._special_ids = frozenset(self._vocabulary[token] for token in self._special_tokens)
        self._multi_tokens = tuple(
            sorted(
                (token for token in self._vocabulary if len(token) > 1),
                key=lambda token: (-len(token), token),
            ))

    @classmethod
    def from_vocab_file(
        cls,
        path: str | Path,
        **options: Any,
    ) -> Wav2Vec2CTCTokenizer:
        """Load a flat or nested Hugging Face ``vocab.json``."""
        return cls(_json_object(path), **options)

    @property
    def vocabulary(self) -> Mapping[str, int]:
        """Read-only token-to-ID mapping for the selected language."""
        return self._vocabulary

    @property
    def vocabulary_size(self) -> int:
        return len(self._vocabulary)

    @property
    def available_languages(self) -> tuple[str, ...]:
        """Return selectable vocabulary languages, if the asset is nested."""
        if self._language_vocabularies is None:
            return ()
        return tuple(sorted(self._language_vocabularies))

    @property
    def pad_token_id(self) -> int:
        return self._vocabulary[self.pad_token]

    @property
    def unk_token_id(self) -> int:
        return self._vocabulary[self.unk_token]

    @property
    def bos_token_id(self) -> int:
        return self._vocabulary[self.bos_token]

    @property
    def eos_token_id(self) -> int:
        return self._vocabulary[self.eos_token]

    @property
    def word_delimiter_token_id(self) -> int:
        return self._vocabulary[self.word_delimiter_token]

    def set_target_language(self, language: str) -> None:
        """Select one vocabulary from a nested multilingual asset."""
        if self._language_vocabularies is None:
            raise ValueError("This Wav2Vec2 tokenizer has a single-language vocabulary.")
        if not isinstance(language, str) or language not in self._language_vocabularies:
            choices = ", ".join(sorted(self._language_vocabularies))
            raise ValueError(f"Unknown target language {language!r}; choose one of: "
                             f"{choices}.")
        self.target_language = language
        self._set_vocabulary(self._language_vocabularies[language])

    def _tokens(self, text: str) -> tuple[str, ...]:
        normalized = text.upper() if self.do_lower_case else text
        normalized = normalized.replace(" ", self.word_delimiter_token)
        tokens: list[str] = []
        cursor = 0
        while cursor < len(normalized):
            match = next(
                (token for token in self._multi_tokens if normalized.startswith(token, cursor)),
                None,
            )
            if match is None:
                match = normalized[cursor]
            tokens.append(match)
            cursor += len(match)
        return tuple(tokens)

    def encode(
        self,
        text: str,
        *,
        allowed_special: SpecialTokenSelection = "none",
        disallowed_special: SpecialTokenSelection = "all",
        max_length: int | None = None,
        truncation: TruncationStrategy = False,
    ) -> Encoding:
        """Encode one transcript into CTC target IDs."""
        if not isinstance(text, str):
            raise TypeError("Wav2Vec2 text must be a string.")
        allowed = _special_selection(
            allowed_special,
            name="allowed_special",
            available=self._special_tokens,
        )
        disallowed = _special_selection(
            disallowed_special,
            name="disallowed_special",
            available=self._special_tokens,
        ) - allowed
        present = tuple(token for token in disallowed if token and token in text)
        if present:
            names = ", ".join(sorted(repr(token) for token in present))
            raise ValueError(f"Text contains disallowed special token(s): {names}.")
        ids = tuple(self._vocabulary.get(token, self.unk_token_id) for token in self._tokens(text))
        if max_length is not None:
            if (isinstance(max_length, bool) or not isinstance(max_length, Integral) or max_length < 0):
                raise ValueError("`max_length` must be a non-negative integer.")
            maximum = int(max_length)
            if len(ids) > maximum:
                if truncation in (True, "right"):
                    ids = ids[:maximum]
                elif truncation == "left":
                    ids = ids[-maximum:] if maximum else ()
                else:
                    raise ValueError("Tokenized text exceeds `max_length`; enable "
                                     "truncation explicitly.")
        elif truncation not in (False, ):
            raise ValueError("Truncation requires `max_length`.")
        special_mask = tuple(int(token_id in self._special_ids) for token_id in ids)
        return Encoding(
            input_ids=ids,
            special_tokens_mask=special_mask,
        )

    def encode_batch(
        self,
        texts: Iterable[str],
        *,
        padding: PaddingStrategy = False,
        max_length: int | None = None,
        truncation: TruncationStrategy = False,
        pad_to_multiple_of: int | None = None,
        allowed_special: SpecialTokenSelection = "none",
        disallowed_special: SpecialTokenSelection = "all",
    ) -> BatchEncoding:
        """Encode transcripts and optionally right-pad them."""
        encodings = tuple(
            self.encode(
                text,
                allowed_special=allowed_special,
                disallowed_special=disallowed_special,
                max_length=max_length,
                truncation=truncation,
            ) for text in texts)
        if padding in (True, "longest"):
            return pad_encodings(
                encodings,
                pad_token_id=self.pad_token_id,
                pad_to_multiple_of=pad_to_multiple_of,
            )
        if padding == "max_length":
            if max_length is None:
                raise ValueError("`padding='max_length'` requires `max_length`.")
            return pad_encodings(
                encodings,
                pad_token_id=self.pad_token_id,
                length=max_length,
                pad_to_multiple_of=pad_to_multiple_of,
            )
        if padding is not False:
            raise ValueError("`padding` must be False, True, 'longest', or 'max_length'.")
        return BatchEncoding(
            input_ids=tuple(encoding.input_ids for encoding in encodings),
            attention_mask=tuple(encoding.attention_mask for encoding in encodings),
            special_tokens_mask=tuple(encoding.special_tokens_mask for encoding in encodings),
        )

    @staticmethod
    def _integer_ids(token_ids: Iterable[int] | Encoding, ) -> tuple[int, ...]:
        values = (token_ids.input_ids if isinstance(token_ids, Encoding) else tuple(token_ids))
        result = []
        for token_id in values:
            if isinstance(token_id, bool) or not isinstance(token_id, Integral):
                raise TypeError("CTC token IDs must be integers.")
            result.append(int(token_id))
        return tuple(result)

    def decode_ctc(
        self,
        token_ids: Iterable[int] | Encoding,
        *,
        group_tokens: bool = True,
        skip_special_tokens: bool = False,
        output_char_offsets: bool = False,
        output_word_offsets: bool = False,
    ) -> Wav2Vec2CTCDecodeOutput:
        """Collapse CTC repetitions/blanks and optionally return frame
        offsets."""
        if not isinstance(group_tokens, bool):
            raise TypeError("`group_tokens` must be a boolean.")
        if not isinstance(skip_special_tokens, bool):
            raise TypeError("`skip_special_tokens` must be a boolean.")
        ids = self._integer_ids(token_ids)
        groups: list[tuple[int, int, int]] = []
        for index, token_id in enumerate(ids):
            if group_tokens and groups and groups[-1][0] == token_id:
                previous_id, start, _ = groups[-1]
                groups[-1] = (previous_id, start, index + 1)
            else:
                groups.append((token_id, index, index + 1))

        characters: list[CTCCharacterOffset] = []
        for token_id, start, end in groups:
            if token_id == self.pad_token_id:
                continue
            token = self._decoder.get(token_id, self.unk_token)
            if skip_special_tokens and token in self._special_tokens:
                continue
            rendered = (self.replace_word_delimiter_char if token == self.word_delimiter_token else token)
            characters.append(CTCCharacterOffset(
                token=rendered,
                start_offset=start,
                end_offset=end,
            ))

        text = "".join(item.token for item in characters).strip()
        if self.do_lower_case:
            text = text.lower()

        words: list[CTCWordOffset] = []
        word_parts: list[str] = []
        word_start: int | None = None
        word_end: int | None = None
        for character in characters:
            if character.token == self.replace_word_delimiter_char:
                if word_parts and word_start is not None and word_end is not None:
                    words.append(
                        CTCWordOffset(
                            word="".join(word_parts),
                            start_offset=word_start,
                            end_offset=word_end,
                        ))
                word_parts = []
                word_start = None
                word_end = None
                continue
            if word_start is None:
                word_start = character.start_offset
            word_end = character.end_offset
            word_parts.append(character.token)
        if word_parts and word_start is not None and word_end is not None:
            words.append(
                CTCWordOffset(
                    word="".join(word_parts),
                    start_offset=word_start,
                    end_offset=word_end,
                ))
        if self.do_lower_case:
            words = [
                CTCWordOffset(
                    word=word.word.lower(),
                    start_offset=word.start_offset,
                    end_offset=word.end_offset,
                ) for word in words
            ]

        return Wav2Vec2CTCDecodeOutput(
            text=text,
            char_offsets=(tuple(characters) if output_char_offsets else ()),
            word_offsets=tuple(words) if output_word_offsets else (),
        )

    def decode(
        self,
        token_ids: Iterable[int] | Encoding,
        *,
        skip_special_tokens: bool = False,
        errors: str = "replace",
        group_tokens: bool = True,
    ) -> str:
        """Return text from one CTC token sequence."""
        if errors != "replace":
            raise ValueError("Character CTC decoding supports only `errors='replace'`.")
        return self.decode_ctc(
            token_ids,
            group_tokens=group_tokens,
            skip_special_tokens=skip_special_tokens,
        ).text

    def to_config(self) -> dict[str, Any]:
        """Serialize tokenizer behavior without embedding credentials."""
        return {
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "unk_token": self.unk_token,
            "pad_token": self.pad_token,
            "word_delimiter_token": self.word_delimiter_token,
            "replace_word_delimiter_char": (self.replace_word_delimiter_char),
            "do_lower_case": self.do_lower_case,
            "target_lang": self.target_language,
            "tokenizer_class": "Wav2Vec2CTCTokenizer",
        }

    def save_pretrained(self, directory: str | Path) -> None:
        """Write ``vocab.json`` and ``tokenizer_config.json``."""
        destination = Path(directory).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        vocabulary: Mapping[str, Any]
        if self._language_vocabularies is None:
            vocabulary = dict(self._vocabulary)
        else:
            vocabulary = {
                language: dict(language_vocabulary)
                for language, language_vocabulary in (self._language_vocabularies.items())
            }
        for path, value in (
            (destination / "vocab.json", vocabulary),
            (destination / "tokenizer_config.json", self.to_config()),
            (
                destination / "special_tokens_map.json",
                {
                    "bos_token": self.bos_token,
                    "eos_token": self.eos_token,
                    "unk_token": self.unk_token,
                    "pad_token": self.pad_token,
                },
            ),
        ):
            with path.open("w", encoding="utf-8") as handle:
                json.dump(
                    value,
                    handle,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                handle.write("\n")


__all__ = [
    "CTCCharacterOffset",
    "CTCWordOffset",
    "Wav2Vec2CTCDecodeOutput",
    "Wav2Vec2CTCTokenizer",
]
