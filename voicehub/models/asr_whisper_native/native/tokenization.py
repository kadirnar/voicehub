"""VoiceHub-native tokenization and prompt semantics for Whisper.

The behavior in this module was checked against OpenAI Whisper revision
``04f449b8a437f1bbd3dba5c9f826aca972e7709a`` and Hugging Face Transformers
revision ``ebea912f0bb6f9e28ad2df04acd9b4df035933a9``. It is an independent,
standard-library implementation built on :mod:`voicehub.tokenization`; neither
upstream tokenizer runtime is imported or executed.
"""

from __future__ import annotations

import base64
import binascii
import json
import math
import string
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from voicehub.tokenization import (
    BatchEncoding,
    ByteBPETokenizer,
    Encoding,
    PaddingStrategy,
    SpecialTokenSelection,
    TokenizerAssetError,
    TruncationStrategy,
    decode_gpt2_token,
    pad_encodings,
    read_bounded_asset,
)

OPENAI_WHISPER_REVISION = "04f449b8a437f1bbd3dba5c9f826aca972e7709a"
TRANSFORMERS_WHISPER_REVISION = "ebea912f0bb6f9e28ad2df04acd9b4df035933a9"
TIMESTAMP_PRECISION = 0.02
TIMESTAMP_COUNT = 1501
MAX_TIMESTAMP_SECONDS = (TIMESTAMP_COUNT - 1) * TIMESTAMP_PRECISION
DEFAULT_MAX_TOKENIZER_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_VOCABULARY = 2_000_000
DEFAULT_MAX_MERGES = 2_000_000
DEFAULT_MAX_TOKEN_BYTES = 16 * 1024
DEFAULT_MAX_JSON_DEPTH = 64
DEFAULT_MAX_JSON_NODES = 4_000_000

LANGUAGES: Mapping[str, str] = MappingProxyType({
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "pl": "polish",
    "ca": "catalan",
    "nl": "dutch",
    "ar": "arabic",
    "sv": "swedish",
    "it": "italian",
    "id": "indonesian",
    "hi": "hindi",
    "fi": "finnish",
    "vi": "vietnamese",
    "he": "hebrew",
    "uk": "ukrainian",
    "el": "greek",
    "ms": "malay",
    "cs": "czech",
    "ro": "romanian",
    "da": "danish",
    "hu": "hungarian",
    "ta": "tamil",
    "no": "norwegian",
    "th": "thai",
    "ur": "urdu",
    "hr": "croatian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "sk": "slovak",
    "te": "telugu",
    "fa": "persian",
    "lv": "latvian",
    "bn": "bengali",
    "sr": "serbian",
    "az": "azerbaijani",
    "sl": "slovenian",
    "kn": "kannada",
    "et": "estonian",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
    "is": "icelandic",
    "hy": "armenian",
    "ne": "nepali",
    "mn": "mongolian",
    "bs": "bosnian",
    "kk": "kazakh",
    "sq": "albanian",
    "sw": "swahili",
    "gl": "galician",
    "mr": "marathi",
    "pa": "punjabi",
    "si": "sinhala",
    "km": "khmer",
    "sn": "shona",
    "yo": "yoruba",
    "so": "somali",
    "af": "afrikaans",
    "oc": "occitan",
    "ka": "georgian",
    "be": "belarusian",
    "tg": "tajik",
    "sd": "sindhi",
    "gu": "gujarati",
    "am": "amharic",
    "yi": "yiddish",
    "lo": "lao",
    "uz": "uzbek",
    "fo": "faroese",
    "ht": "haitian creole",
    "ps": "pashto",
    "tk": "turkmen",
    "nn": "nynorsk",
    "mt": "maltese",
    "sa": "sanskrit",
    "lb": "luxembourgish",
    "my": "myanmar",
    "bo": "tibetan",
    "tl": "tagalog",
    "mg": "malagasy",
    "as": "assamese",
    "tt": "tatar",
    "haw": "hawaiian",
    "ln": "lingala",
    "ha": "hausa",
    "ba": "bashkir",
    "jw": "javanese",
    "su": "sundanese",
    "yue": "cantonese",
})

TO_LANGUAGE_CODE: Mapping[str, str] = MappingProxyType({
    **{
        language: code
        for code, language in LANGUAGES.items()
    },
    "burmese": "my",
    "valencian": "ca",
    "flemish": "nl",
    "haitian": "ht",
    "letzeburgesch": "lb",
    "pushto": "ps",
    "panjabi": "pa",
    "moldavian": "ro",
    "moldovan": "ro",
    "sinhalese": "si",
    "castilian": "es",
    "mandarin": "zh",
})

TASKS = frozenset({"transcribe", "translate"})
_REQUIRED_CONTROL_TOKENS = (
    "<|endoftext|>",
    "<|startoftranscript|>",
    "<|translate|>",
    "<|transcribe|>",
    "<|startoflm|>",
    "<|startofprev|>",
    "<|notimestamps|>",
)
_NO_SPEECH_ALIASES = ("<|nospeech|>", "<|nocaptions|>")
_LANGUAGE_TOKEN_ALIASES: Mapping[str, tuple[str, ...]] = MappingProxyType({
    # Original Hugging Face English-only assets used this legacy ISO code.
    "he": ("<|iw|>", ),
})


class WhisperTokenizerFormatError(TokenizerAssetError):
    """Raised when tokenizer assets violate Whisper's fixed token layout."""


@dataclass(frozen=True, slots=True)
class TimestampToken:
    """One Whisper timestamp token and its exact 20 ms position."""

    token_id: int
    index: int
    seconds: float
    text: str


@dataclass(frozen=True, slots=True)
class WhisperSpecialTokens:
    """Validated Whisper control, language, and timestamp token layout."""

    tokens: Mapping[str, int]
    language_codes: tuple[str, ...]
    timestamp_count: int
    no_speech_text: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "tokens", MappingProxyType(dict(self.tokens)))
        object.__setattr__(self, "language_codes", tuple(self.language_codes))

    def id_for(self, token: str) -> int:
        """Return a required token ID with a useful error for unknown names."""
        try:
            return self.tokens[token]
        except KeyError as error:
            raise KeyError(f"Unknown Whisper special token: {token!r}.") from error

    @property
    def eot(self) -> int:
        return self.id_for("<|endoftext|>")

    @property
    def sot(self) -> int:
        return self.id_for("<|startoftranscript|>")

    @property
    def translate(self) -> int:
        return self.id_for("<|translate|>")

    @property
    def transcribe(self) -> int:
        return self.id_for("<|transcribe|>")

    @property
    def sot_lm(self) -> int:
        return self.id_for("<|startoflm|>")

    @property
    def sot_prev(self) -> int:
        return self.id_for("<|startofprev|>")

    @property
    def no_speech(self) -> int:
        return self.id_for(self.no_speech_text)

    @property
    def no_timestamps(self) -> int:
        return self.id_for("<|notimestamps|>")

    @property
    def timestamp_begin(self) -> int:
        return self.id_for("<|0.00|>")

    @property
    def timestamp_end(self) -> int:
        return self.timestamp_begin + self.timestamp_count - 1


def build_openai_whisper_special_tokens(
    mergeable_vocabulary_size: int,
    *,
    num_languages: int = 99,
    timestamp_count: int = TIMESTAMP_COUNT,
) -> Mapping[str, int]:
    """Build the exact sequential special-token table used by OpenAI."""
    vocabulary_size = _positive_integer(
        mergeable_vocabulary_size,
        name="mergeable_vocabulary_size",
    )
    language_count = _language_count(num_languages)
    timestamps = _positive_integer(timestamp_count, name="timestamp_count")
    tokens = [
        "<|endoftext|>",
        "<|startoftranscript|>",
        *(f"<|{code}|>" for code in tuple(LANGUAGES)[:language_count]),
        "<|translate|>",
        "<|transcribe|>",
        "<|startoflm|>",
        "<|startofprev|>",
        "<|nospeech|>",
        "<|notimestamps|>",
        *(_timestamp_text(index) for index in range(timestamps)),
    ]
    return MappingProxyType({token: vocabulary_size + offset for offset, token in enumerate(tokens)})


def discover_whisper_special_tokens(
    tokens: Mapping[str, int],
    *,
    num_languages: int | None = None,
    timestamp_count: int = TIMESTAMP_COUNT,
) -> WhisperSpecialTokens:
    """Validate and describe an OpenAI-compatible Whisper token layout."""
    if not isinstance(tokens, Mapping):
        raise TypeError("Whisper special tokens must be a mapping.")
    normalized: dict[str, int] = {}
    seen_ids: dict[int, str] = {}
    for token, token_id in tokens.items():
        if not isinstance(token, str) or not token:
            raise WhisperTokenizerFormatError("Whisper special-token names must be non-empty strings.")
        normalized_id = _token_id(token_id, name=f"ID for {token!r}")
        previous = seen_ids.get(normalized_id)
        if previous is not None and previous != token:
            if {previous, token} <= set(_NO_SPEECH_ALIASES):
                normalized[token] = normalized_id
                continue
            previous_language = _language_code_for_token(previous)
            current_language = _language_code_for_token(token)
            if (previous_language is not None and previous_language == current_language):
                normalized[token] = normalized_id
                continue
            raise WhisperTokenizerFormatError(
                f"Whisper tokens {previous!r} and {token!r} share ID "
                f"{normalized_id}.")
        normalized[token] = normalized_id
        seen_ids[normalized_id] = token

    missing = tuple(token for token in _REQUIRED_CONTROL_TOKENS if token not in normalized)
    if missing:
        raise WhisperTokenizerFormatError(
            "Whisper tokenizer is missing required tokens: "
            f"{', '.join(missing)}.")

    no_speech_tokens = tuple(token for token in _NO_SPEECH_ALIASES if token in normalized)
    if not no_speech_tokens:
        raise WhisperTokenizerFormatError(
            "Whisper tokenizer requires `<|nospeech|>` or "
            "`<|nocaptions|>`.")
    no_speech_ids = {normalized[token] for token in no_speech_tokens}
    if len(no_speech_ids) != 1:
        raise WhisperTokenizerFormatError("Whisper no-speech aliases must resolve to the same token ID.")
    no_speech_text = ("<|nospeech|>" if "<|nospeech|>" in normalized else "<|nocaptions|>")

    present_language_indices = []
    for index, code in enumerate(LANGUAGES):
        candidates = (f"<|{code}|>", ) + _LANGUAGE_TOKEN_ALIASES.get(code, ())
        present = {normalized[token] for token in candidates if token in normalized}
        if len(present) > 1:
            raise WhisperTokenizerFormatError(
                f"Whisper language aliases for {code!r} have conflicting "
                "token IDs.")
        if present:
            normalized.setdefault(f"<|{code}|>", next(iter(present)))
            present_language_indices.append(index)
    present_language_indices = tuple(present_language_indices)
    if not present_language_indices:
        raise WhisperTokenizerFormatError("Whisper tokenizer does not contain language tokens.")
    discovered_count = present_language_indices[-1] + 1
    if present_language_indices != tuple(range(discovered_count)):
        raise WhisperTokenizerFormatError(
            "Whisper language tokens must be a contiguous prefix of the "
            "official language order.")
    if num_languages is not None:
        expected_count = _language_count(num_languages)
        if discovered_count != expected_count:
            raise WhisperTokenizerFormatError(
                f"Tokenizer contains {discovered_count} language tokens; "
                f"expected {expected_count}.")
    language_codes = tuple(LANGUAGES)[:discovered_count]

    expected_ids: list[tuple[str, int]] = [
        ("<|startoftranscript|>", normalized["<|endoftext|>"] + 1),
    ]
    expected_ids.extend((f"<|{code}|>", normalized["<|startoftranscript|>"] + index + 1)
                        for index, code in enumerate(language_codes))
    translate_id = normalized["<|startoftranscript|>"] + discovered_count + 1
    expected_ids.extend((
        ("<|translate|>", translate_id),
        ("<|transcribe|>", translate_id + 1),
        ("<|startoflm|>", translate_id + 2),
        ("<|startofprev|>", translate_id + 3),
        (no_speech_text, translate_id + 4),
        ("<|notimestamps|>", translate_id + 5),
    ))
    for token, expected_id in expected_ids:
        if normalized[token] != expected_id:
            raise WhisperTokenizerFormatError(
                f"Whisper token {token!r} has ID {normalized[token]}, "
                f"expected {expected_id}.")

    timestamps = _positive_integer(timestamp_count, name="timestamp_count")
    timestamp_begin = normalized["<|notimestamps|>"] + 1
    for index in range(timestamps):
        token = _timestamp_text(index)
        expected_id = timestamp_begin + index
        if token not in normalized:
            raise WhisperTokenizerFormatError(f"Whisper tokenizer is missing timestamp token {token!r}.")
        if normalized[token] != expected_id:
            raise WhisperTokenizerFormatError(
                f"Whisper timestamp {token!r} has ID {normalized[token]}, "
                f"expected {expected_id}.")

    return WhisperSpecialTokens(
        tokens=normalized,
        language_codes=language_codes,
        timestamp_count=timestamps,
        no_speech_text=no_speech_text,
    )


class WhisperTokenizer:
    """Native Whisper text tokenizer with request-local prompt semantics."""

    def __init__(
            self,
            encoding: ByteBPETokenizer,
            *,
            multilingual: bool | None = None,
            language: str | None = None,
            task: Literal["transcribe", "translate"] | None = None,
            predict_timestamps: bool = False,
            num_languages: int | None = None,
            timestamp_count: int = TIMESTAMP_COUNT,
            empty_token_ids: Collection[int] = (),
    ) -> None:
        if not isinstance(encoding, ByteBPETokenizer):
            raise TypeError("`encoding` must be a native ByteBPETokenizer.")
        if multilingual is not None and not isinstance(multilingual, bool):
            raise TypeError("`multilingual` must be a boolean or None.")
        if not isinstance(predict_timestamps, bool):
            raise TypeError("`predict_timestamps` must be a boolean.")
        self._encoding = encoding
        self._special = discover_whisper_special_tokens(
            encoding.special_tokens,
            num_languages=num_languages,
            timestamp_count=timestamp_count,
        )
        empty_ids = frozenset(_token_id(token_id, name="empty token ID") for token_id in empty_token_ids)
        assigned_ids = (set(encoding.vocabulary.values()) | set(encoding.special_tokens.values()))
        overlap = empty_ids & assigned_ids
        if overlap:
            raise ValueError(
                "Empty Whisper token IDs cannot also have byte or special "
                f"values: {sorted(overlap)!r}.")
        if any(token_id >= self._special.eot for token_id in empty_ids):
            raise ValueError("Empty Whisper token IDs must precede `<|endoftext|>`.")
        self._empty_token_ids = empty_ids
        self.multilingual = multilingual
        if multilingual is False:
            resolved_language = None
            resolved_task = None
        else:
            resolved_language = _normalize_language(language)
            resolved_task = _normalize_task(task)
            if multilingual is True:
                resolved_language = resolved_language or "en"
                resolved_task = resolved_task or "transcribe"
        if (resolved_language is not None and resolved_language not in self._special.language_codes):
            raise ValueError(
                f"Language {resolved_language!r} is not present in this "
                f"{len(self._special.language_codes)}-language tokenizer.")
        self.language = resolved_language
        self.task = resolved_task
        self.predict_timestamps = predict_timestamps

    @classmethod
    def from_tiktoken_file(
        cls,
        path: str | Path,
        *,
        multilingual: bool,
        num_languages: int = 99,
        language: str | None = None,
        task: Literal["transcribe", "translate"] | None = None,
        predict_timestamps: bool = False,
        timestamp_count: int = TIMESTAMP_COUNT,
        max_asset_bytes: int = DEFAULT_MAX_TOKENIZER_BYTES,
        max_tokens: int = DEFAULT_MAX_VOCABULARY,
        max_token_bytes: int = DEFAULT_MAX_TOKEN_BYTES,
    ) -> WhisperTokenizer:
        """Load an official OpenAI Whisper ``*.tiktoken`` vocabulary."""
        if not isinstance(multilingual, bool):
            raise TypeError("`multilingual` must be a boolean.")
        ranks = _load_openai_whisper_ranks(
            path,
            max_bytes=max_asset_bytes,
            max_tokens=max_tokens,
            max_token_bytes=max_token_bytes,
        )
        _validate_mergeable_vocabulary(ranks)
        empty_token_id = ranks.get(b"")
        mergeable_ranks = {token: token_id for token, token_id in ranks.items() if token}
        special_tokens = build_openai_whisper_special_tokens(
            len(ranks),
            num_languages=num_languages,
            timestamp_count=timestamp_count,
        )
        encoding = ByteBPETokenizer(
            mergeable_ranks,
            special_tokens=special_tokens,
            pad_token_id=special_tokens["<|endoftext|>"],
        )
        return cls(
            encoding,
            multilingual=multilingual,
            language=language,
            task=task,
            predict_timestamps=predict_timestamps,
            num_languages=num_languages,
            timestamp_count=timestamp_count,
            empty_token_ids=(() if empty_token_id is None else (empty_token_id, )),
        )

    @classmethod
    def from_tiktoken(
        cls,
        path: str | Path,
        **options: object,
    ) -> WhisperTokenizer:
        """Alias for :meth:`from_tiktoken_file`."""
        return cls.from_tiktoken_file(path, **options)

    @classmethod
    def from_tokenizer_json(
        cls,
        path: str | Path,
        *,
        multilingual: bool | None = None,
        num_languages: int | None = None,
        language: str | None = None,
        task: Literal["transcribe", "translate"] | None = None,
        predict_timestamps: bool = False,
        timestamp_count: int = TIMESTAMP_COUNT,
        max_asset_bytes: int = DEFAULT_MAX_TOKENIZER_BYTES,
        max_tokens: int = DEFAULT_MAX_VOCABULARY,
        max_merges: int = DEFAULT_MAX_MERGES,
    ) -> WhisperTokenizer:
        """Load an official Hugging Face Whisper ``tokenizer.json``."""
        encoding, empty_token_ids = _load_huggingface_whisper_tokenizer(
            path,
            max_asset_bytes=max_asset_bytes,
            max_tokens=max_tokens,
            max_merges=max_merges,
        )
        return cls(
            encoding,
            multilingual=multilingual,
            language=language,
            task=task,
            predict_timestamps=predict_timestamps,
            num_languages=num_languages,
            timestamp_count=timestamp_count,
            empty_token_ids=empty_token_ids,
        )

    @classmethod
    def from_huggingface_tokenizer_json(
        cls,
        path: str | Path,
        **options: object,
    ) -> WhisperTokenizer:
        """Explicit alias for :meth:`from_tokenizer_json`."""
        return cls.from_tokenizer_json(path, **options)

    @property
    def encoding(self) -> ByteBPETokenizer:
        return self._encoding

    @property
    def special_tokens(self) -> Mapping[str, int]:
        return self._special.tokens

    @property
    def vocabulary_size(self) -> int:
        # Whisper checkpoints size their token embeddings by ID range. The
        # multilingual OpenAI vocabulary reserves one empty byte token which
        # can never be emitted, so counting stored entries would be one short.
        return self.timestamp_end + 1

    @property
    def token_id_space_size(self) -> int:
        """Exclusive upper bound of every token declared by the asset.

        Standard Whisper tokenizers end at the timestamp range. Derived
        checkpoints may append architecture-specific tokens after it
        while retaining padded, unused embedding rows. Callers that
        support those formats should compare this bound with the model
        vocabulary and mask any undeclared rows during generation.
        """
        return self._encoding.token_id_space_size

    @property
    def num_languages(self) -> int:
        return len(self._special.language_codes)

    @property
    def eot(self) -> int:
        return self._special.eot

    @property
    def bos_token_id(self) -> int:
        return self.eot

    @property
    def eos_token_id(self) -> int:
        return self.eot

    @property
    def pad_token_id(self) -> int:
        return self.eot

    @property
    def sot(self) -> int:
        return self._special.sot

    @property
    def transcribe(self) -> int:
        return self._special.transcribe

    @property
    def translate(self) -> int:
        return self._special.translate

    @property
    def sot_lm(self) -> int:
        return self._special.sot_lm

    @property
    def sot_prev(self) -> int:
        return self._special.sot_prev

    @property
    def no_speech(self) -> int:
        return self._special.no_speech

    @property
    def no_timestamps(self) -> int:
        return self._special.no_timestamps

    @property
    def timestamp_begin(self) -> int:
        return self._special.timestamp_begin

    @property
    def timestamp_end(self) -> int:
        return self._special.timestamp_end

    @property
    def all_language_codes(self) -> tuple[str, ...]:
        return self._special.language_codes

    @property
    def all_language_tokens(self) -> tuple[int, ...]:
        return tuple(self._special.id_for(f"<|{code}|>") for code in self._special.language_codes)

    @property
    def language_token(self) -> int:
        if self.language is None:
            raise ValueError("This Whisper tokenizer has no configured language.")
        return self.to_language_token(self.language)

    def to_language_token(self, language: str) -> int:
        """Resolve a language code, canonical name, or official alias."""
        code = _normalize_language(language)
        if code not in self._special.language_codes:
            raise KeyError(f"Language {language!r} is not available in this tokenizer.")
        return self._special.id_for(f"<|{code}|>")

    @property
    def sot_sequence(self) -> tuple[int, ...]:
        return self.prompt_tokens(include_no_timestamps=False)

    @property
    def sot_sequence_including_notimestamps(self) -> tuple[int, ...]:
        return self.sot_sequence + (self.no_timestamps, )

    @property
    def prefix_tokens(self) -> tuple[int, ...]:
        return self.prompt_tokens(include_no_timestamps=not self.predict_timestamps)

    def prompt_tokens(
        self,
        *,
        language: str | None = None,
        task: Literal["transcribe", "translate"] | None = None,
        include_no_timestamps: bool | None = None,
    ) -> tuple[int, ...]:
        """Build a Whisper decoder prefix without mutating shared state."""
        resolved_language = (self.language if language is None else _normalize_language(language))
        resolved_task = self.task if task is None else _normalize_task(task)
        if resolved_language is not None:
            language_id = self.to_language_token(resolved_language)
        tokens = [self.sot]
        if resolved_language is not None:
            tokens.append(language_id)
        if resolved_task is not None:
            tokens.append(self.transcribe if resolved_task == "transcribe" else self.translate)
        include_no_timestamps = (
            not self.predict_timestamps if include_no_timestamps is None else _boolean(
                include_no_timestamps, name="include_no_timestamps"))
        if include_no_timestamps:
            tokens.append(self.no_timestamps)
        return tuple(tokens)

    def get_decoder_prompt_ids(
        self,
        task: Literal["transcribe", "translate"] | None = None,
        language: str | None = None,
        no_timestamps: bool = True,
    ) -> tuple[tuple[int, int], ...]:
        """Return Hugging Face-compatible forced decoder positions."""
        prefix = self.prompt_tokens(
            language=language,
            task=task,
            include_no_timestamps=_boolean(no_timestamps, name="no_timestamps"),
        )
        return tuple((position, token) for position, token in enumerate(prefix[1:], start=1))

    def get_prompt_ids(self, text: str) -> tuple[int, ...]:
        """Encode carry-over text behind OpenAI's ``startofprev`` token."""
        if not isinstance(text, str):
            raise TypeError("Whisper prompt text must be a string.")
        prompt_text = " " + text.strip()
        encoded = self._encoding.encode(prompt_text)
        return (self.sot_prev, ) + encoded.input_ids

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        allowed_special: SpecialTokenSelection = "none",
        disallowed_special: SpecialTokenSelection = "all",
        max_length: int | None = None,
        truncation: TruncationStrategy = False,
    ) -> Encoding:
        """Encode text, optionally framing it as a Whisper training label."""
        add_special_tokens = _boolean(add_special_tokens, name="add_special_tokens")
        content = self._encoding.encode(
            text,
            allowed_special=allowed_special,
            disallowed_special=disallowed_special,
        )
        prefix = self.prefix_tokens if add_special_tokens else ()
        suffix = (self.eot, ) if add_special_tokens else ()
        content = _truncate_content(
            content,
            max_length=max_length,
            truncation=truncation,
            reserved=len(prefix) + len(suffix),
        )
        return Encoding(
            input_ids=prefix + content.input_ids + suffix,
            attention_mask=(1, ) * (len(prefix) + len(content) + len(suffix)),
            special_tokens_mask=(1, ) * len(prefix) + content.special_tokens_mask + (1, ) * len(suffix),
        )

    def encode_batch(
        self,
        texts: Iterable[str],
        *,
        add_special_tokens: bool = False,
        padding: PaddingStrategy = False,
        max_length: int | None = None,
        truncation: TruncationStrategy = False,
        pad_to_multiple_of: int | None = None,
        allowed_special: SpecialTokenSelection = "none",
        disallowed_special: SpecialTokenSelection = "all",
    ) -> BatchEncoding:
        """Encode training or inference text and optionally pad with EOT."""
        try:
            values = tuple(texts)
        except TypeError as error:
            raise TypeError("`texts` must be an iterable of strings.") from error
        encodings = tuple(
            self.encode(
                text,
                add_special_tokens=add_special_tokens,
                allowed_special=allowed_special,
                disallowed_special=disallowed_special,
                max_length=max_length,
                truncation=truncation,
            ) for text in values)
        if padding is False:
            if pad_to_multiple_of is not None:
                raise ValueError("`pad_to_multiple_of` requires an enabled padding strategy.")
            return BatchEncoding(
                input_ids=tuple(item.input_ids for item in encodings),
                attention_mask=tuple(item.attention_mask for item in encodings),
                special_tokens_mask=tuple(item.special_tokens_mask for item in encodings),
            )
        if padding not in (True, "longest", "max_length"):
            raise ValueError("`padding` must be False, True, 'longest', or 'max_length'.")
        target_length: int | None = None
        if padding == "max_length":
            if max_length is None:
                raise ValueError("`padding='max_length'` requires `max_length`.")
            target_length = _nonnegative_integer(max_length, name="max_length")
        return pad_encodings(
            encodings,
            pad_token_id=self.eot,
            length=target_length,
            pad_to_multiple_of=pad_to_multiple_of,
        )

    def decode(
        self,
        token_ids: Iterable[int] | Encoding,
        *,
        skip_special_tokens: bool = False,
        errors: str = "replace",
    ) -> str:
        """Decode text while omitting timestamps, as OpenAI Whisper does."""
        values = _coerce_token_ids(token_ids)
        filtered = tuple(
            token_id for token_id in values
            if (token_id < self.timestamp_begin and token_id not in self._empty_token_ids))
        return self._encoding.decode(
            filtered,
            skip_special_tokens=skip_special_tokens,
            errors=errors,
        )

    def decode_with_timestamps(
        self,
        token_ids: Iterable[int] | Encoding,
        *,
        skip_special_tokens: bool = False,
        errors: str = "replace",
    ) -> str:
        """Decode timestamp IDs to their canonical ``<|seconds|>`` labels."""
        return self._encoding.decode(
            tuple(
                token_id for token_id in _coerce_token_ids(token_ids)
                if token_id not in self._empty_token_ids),
            skip_special_tokens=skip_special_tokens,
            errors=errors,
        )

    def is_timestamp(self, token_id: int) -> bool:
        """Whether ``token_id`` is in this tokenizer's timestamp range."""
        normalized = _token_id(token_id, name="token_id")
        return self.timestamp_begin <= normalized <= self.timestamp_end

    def timestamp_for_token(self, token_id: int) -> TimestampToken | None:
        """Interpret one token as a timestamp, returning ``None`` for text."""
        normalized = _token_id(token_id, name="token_id")
        if not self.timestamp_begin <= normalized <= self.timestamp_end:
            return None
        index = normalized - self.timestamp_begin
        return TimestampToken(
            token_id=normalized,
            index=index,
            seconds=index * TIMESTAMP_PRECISION,
            text=_timestamp_text(index),
        )

    def timestamp_seconds(self, token_id: int) -> float:
        """Return seconds for a timestamp token or raise for a text token."""
        timestamp = self.timestamp_for_token(token_id)
        if timestamp is None:
            raise ValueError(f"Token ID {token_id} is not a Whisper timestamp.")
        return timestamp.seconds

    def token_for_timestamp(
        self,
        seconds: float,
        *,
        rounding: Literal["exact", "nearest", "floor", "ceil"] = "exact",
        clamp: bool = False,
    ) -> int:
        """Map seconds to a 20 ms token with explicit rounding behavior."""
        if isinstance(seconds, bool) or not isinstance(seconds, Real):
            raise TypeError("`seconds` must be a real number.")
        value = float(seconds)
        if not math.isfinite(value):
            raise ValueError("`seconds` must be finite.")
        if rounding not in ("exact", "nearest", "floor", "ceil"):
            raise ValueError("`rounding` must be 'exact', 'nearest', 'floor', or 'ceil'.")
        position = value / TIMESTAMP_PRECISION
        if rounding == "exact":
            index = round(position)
            if not math.isclose(position, index, rel_tol=0.0, abs_tol=1e-7):
                raise ValueError(
                    f"Timestamp {value} is not aligned to "
                    f"{TIMESTAMP_PRECISION:.2f}-second Whisper frames.")
        elif rounding == "nearest":
            index = math.floor(position + 0.5)
        elif rounding == "floor":
            index = math.floor(position)
        else:
            index = math.ceil(position)
        if clamp:
            index = min(max(index, 0), self._special.timestamp_count - 1)
        elif not 0 <= index < self._special.timestamp_count:
            raise ValueError(
                f"Timestamp must be in [0.00, "
                f"{(self._special.timestamp_count - 1) * TIMESTAMP_PRECISION:.2f}] "
                "seconds.")
        return self.timestamp_begin + index

    def iter_timestamps(
        self,
        token_ids: Iterable[int] | Encoding,
    ) -> tuple[TimestampToken, ...]:
        """Return all timestamp tokens in sequence order."""
        return tuple(
            timestamp for token_id in _coerce_token_ids(token_ids)
            if (timestamp := self.timestamp_for_token(token_id)) is not None)

    @property
    def non_speech_tokens(self) -> tuple[int, ...]:
        """Return OpenAI's punctuation and annotation suppression set."""
        symbols = list('"#()*+/:;<=>@[\\]^_`{|}~「」『』')
        symbols += ("<< >> <<< >>> -- --- -( -[ (' (\" (( )) ((( ))) [[ ]] {{ }} "
                    "♪♪ ♪♪♪").split()
        miscellaneous = set("♩♪♫♬♭♮♯")
        result = {
            self.encode(" -").input_ids[0],
            self.encode(" '").input_ids[0],
        }
        for symbol in symbols + list(miscellaneous):
            for value in (symbol, " " + symbol):
                encoded = self.encode(value).input_ids
                if len(encoded) == 1 or symbol in miscellaneous:
                    result.add(encoded[0])
        return tuple(sorted(result))

    def split_tokens_on_unicode(
        self,
        token_ids: Iterable[int],
    ) -> tuple[tuple[str, ...], tuple[tuple[int, ...], ...]]:
        """Split after each sequence that decodes to complete Unicode."""
        values = _coerce_token_ids(token_ids)
        decoded_full = self.decode_with_timestamps(values)
        replacement = "\ufffd"
        words: list[str] = []
        word_tokens: list[tuple[int, ...]] = []
        current: list[int] = []
        unicode_offset = 0
        for token_id in values:
            current.append(token_id)
            decoded = self.decode_with_timestamps(current)
            replacement_index = decoded.find(replacement)
            complete = replacement_index < 0
            if not complete:
                full_index = unicode_offset + replacement_index
                complete = (full_index < len(decoded_full) and decoded_full[full_index] == replacement)
            if complete:
                words.append(decoded)
                word_tokens.append(tuple(current))
                current = []
                unicode_offset += len(decoded)
        if current:
            words.append(self.decode_with_timestamps(current))
            word_tokens.append(tuple(current))
        return tuple(words), tuple(word_tokens)

    def split_tokens_on_spaces(
        self,
        token_ids: Iterable[int],
    ) -> tuple[tuple[str, ...], tuple[tuple[int, ...], ...]]:
        """Combine Unicode-safe pieces using OpenAI's word boundaries."""
        subwords, subword_tokens = self.split_tokens_on_unicode(token_ids)
        words: list[str] = []
        word_tokens: list[list[int]] = []
        for subword, tokens in zip(subwords, subword_tokens):
            special = tokens[0] >= self.eot
            with_space = subword.startswith(" ")
            punctuation = subword.strip() in string.punctuation
            if special or with_space or punctuation or not words:
                words.append(subword)
                word_tokens.append(list(tokens))
            else:
                words[-1] += subword
                word_tokens[-1].extend(tokens)
        return tuple(words), tuple(tuple(tokens) for tokens in word_tokens)

    def split_to_word_tokens(
        self,
        token_ids: Iterable[int],
    ) -> tuple[tuple[str, ...], tuple[tuple[int, ...], ...]]:
        """Split text appropriately for the configured Whisper language."""
        if self.language in {"zh", "ja", "th", "lo", "my", "yue"}:
            return self.split_tokens_on_unicode(token_ids)
        return self.split_tokens_on_spaces(token_ids)


def _load_openai_whisper_ranks(
    path: str | Path,
    *,
    max_bytes: int,
    max_tokens: int,
    max_token_bytes: int,
) -> dict[bytes, int]:
    """Load OpenAI ranks, including multilingual's reserved empty token.

    The official ``multilingual.tiktoken`` ends in ``= 50256``. OpenAI's
    permissive base64 decoder interprets that record as ``b""`` and uses
    it only to reserve an embedding ID before the special-token range.
    Accepting precisely that legacy spelling preserves official IDs
    without making the generic tokenizer asset parser permissive.
    """
    token_limit = _positive_integer(max_tokens, name="max_tokens")
    byte_limit = _positive_integer(max_token_bytes, name="max_token_bytes")
    payload = read_bounded_asset(path, max_bytes=max_bytes)
    ranks: dict[bytes, int] = {}
    seen_ranks: set[int] = set()
    for line_number, raw_line in enumerate(payload.splitlines(), start=1):
        if not raw_line.strip():
            continue
        if len(ranks) >= token_limit:
            raise WhisperTokenizerFormatError(
                f"Whisper TikToken asset contains more than {token_limit} "
                "tokens.")
        fields = raw_line.split()
        if len(fields) != 2:
            raise WhisperTokenizerFormatError(
                f"Invalid Whisper TikToken record on line {line_number}: "
                "expected two fields.")
        encoded_token, encoded_rank = fields
        try:
            token = (b"" if encoded_token == b"=" else base64.b64decode(encoded_token, validate=True))
        except (binascii.Error, ValueError) as error:
            raise WhisperTokenizerFormatError(f"Invalid base64 token on line {line_number}.") from error
        if not token and encoded_token != b"=":
            raise WhisperTokenizerFormatError(
                f"Whisper TikToken record on line {line_number} contains an "
                "unsupported empty token.")
        if len(token) > byte_limit:
            raise WhisperTokenizerFormatError(f"Token on line {line_number} exceeds {byte_limit} bytes.")
        try:
            rank = int(encoded_rank.decode("ascii"))
        except (UnicodeDecodeError, ValueError) as error:
            raise WhisperTokenizerFormatError(f"Invalid token rank on line {line_number}.") from error
        rank = _token_id(rank, name=f"rank on line {line_number}")
        if token in ranks:
            raise WhisperTokenizerFormatError(
                f"Duplicate token in Whisper TikToken asset on line "
                f"{line_number}.")
        if rank in seen_ranks:
            raise WhisperTokenizerFormatError(
                f"Duplicate rank {rank} in Whisper TikToken asset on line "
                f"{line_number}.")
        ranks[token] = rank
        seen_ranks.add(rank)
    if not ranks:
        raise WhisperTokenizerFormatError("Whisper TikToken asset does not contain any tokens.")
    return ranks


def _load_huggingface_whisper_tokenizer(
    path: str | Path,
    *,
    max_asset_bytes: int,
    max_tokens: int,
    max_merges: int,
) -> tuple[ByteBPETokenizer, frozenset[int]]:
    payload = read_bounded_asset(path, max_bytes=max_asset_bytes)
    try:
        document = json.loads(
            payload.decode("utf-8"),
            parse_constant=lambda value: _reject_json_constant(value),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise WhisperTokenizerFormatError(f"Invalid Whisper tokenizer JSON: {error}.") from error
    _validate_json_bounds(
        document,
        max_depth=DEFAULT_MAX_JSON_DEPTH,
        max_nodes=DEFAULT_MAX_JSON_NODES,
    )
    if not isinstance(document, dict):
        raise WhisperTokenizerFormatError("Whisper tokenizer JSON root must be an object.")
    if document.get("normalizer") is not None:
        raise WhisperTokenizerFormatError("Whisper tokenizer JSON with a text normalizer is unsupported.")

    model = document.get("model")
    if not isinstance(model, dict) or model.get("type") != "BPE":
        raise WhisperTokenizerFormatError("Whisper tokenizer JSON must use a BPE model.")
    if model.get("dropout") not in (None, 0, 0.0):
        raise WhisperTokenizerFormatError("Whisper BPE dropout is not deterministic and is unsupported.")
    if model.get("continuing_subword_prefix") not in (None, ""):
        raise WhisperTokenizerFormatError("Whisper does not support a continuing-subword prefix.")
    if model.get("end_of_word_suffix") not in (None, ""):
        raise WhisperTokenizerFormatError("Whisper does not support an end-of-word suffix.")
    if model.get("byte_fallback") is True:
        raise WhisperTokenizerFormatError("SentencePiece byte fallback is not a Whisper ByteLevel tokenizer.")
    if model.get("ignore_merges") is True:
        raise WhisperTokenizerFormatError("Whisper tokenizer JSON cannot ignore its BPE merges.")

    raw_vocabulary = model.get("vocab")
    if not isinstance(raw_vocabulary, dict) or not raw_vocabulary:
        raise WhisperTokenizerFormatError("Whisper BPE vocabulary must be a non-empty object.")
    if len(raw_vocabulary) > _positive_integer(max_tokens, name="max_tokens"):
        raise WhisperTokenizerFormatError(f"Whisper vocabulary exceeds the {max_tokens}-token limit.")
    vocabulary: dict[bytes, int] = {}
    seen_ids: set[int] = set()
    empty_token_ids: set[int] = set()
    vocabulary_strings: dict[str, int] = {}
    for token_text, token_id in raw_vocabulary.items():
        if not isinstance(token_text, str):
            raise WhisperTokenizerFormatError("Whisper vocabulary keys must be strings.")
        token = decode_gpt2_token(token_text)
        normalized_id = _token_id(token_id, name="vocabulary token ID")
        if normalized_id in seen_ids:
            raise WhisperTokenizerFormatError("Whisper vocabulary contains duplicate tokens or IDs.")
        if not token:
            if token_text or empty_token_ids:
                raise WhisperTokenizerFormatError("Whisper vocabulary contains an invalid empty byte token.")
            empty_token_ids.add(normalized_id)
            vocabulary_strings[token_text] = normalized_id
            seen_ids.add(normalized_id)
            continue
        if token in vocabulary:
            raise WhisperTokenizerFormatError("Whisper vocabulary contains duplicate tokens or IDs.")
        vocabulary[token] = normalized_id
        vocabulary_strings[token_text] = normalized_id
        seen_ids.add(normalized_id)
    _validate_byte_alphabet(vocabulary)

    raw_merges = model.get("merges", [])
    if not isinstance(raw_merges, list):
        raise WhisperTokenizerFormatError("Whisper BPE merges must be an array.")
    if len(raw_merges) > _positive_integer(max_merges, name="max_merges"):
        raise WhisperTokenizerFormatError(f"Whisper tokenizer exceeds the {max_merges}-merge limit.")
    merges: list[tuple[bytes, bytes]] = []
    seen_merges: set[tuple[bytes, bytes]] = set()
    for index, raw_merge in enumerate(raw_merges):
        left_text, right_text = _parse_merge(raw_merge, index=index)
        pair = (decode_gpt2_token(left_text), decode_gpt2_token(right_text))
        if not pair[0] or not pair[1]:
            raise WhisperTokenizerFormatError(f"Whisper BPE merge {index} contains an empty token.")
        if pair in seen_merges:
            raise WhisperTokenizerFormatError(f"Whisper BPE merge {index} is duplicated.")
        if pair[0] + pair[1] not in vocabulary:
            raise WhisperTokenizerFormatError(f"Whisper BPE merge {index} produces an unknown token.")
        merges.append(pair)
        seen_merges.add(pair)

    raw_added_tokens = document.get("added_tokens")
    if not isinstance(raw_added_tokens, list) or not raw_added_tokens:
        raise WhisperTokenizerFormatError("Whisper tokenizer JSON must contain added control tokens.")
    if len(raw_added_tokens) > _positive_integer(max_tokens, name="max_tokens"):
        raise WhisperTokenizerFormatError(f"Whisper added tokens exceed the {max_tokens}-token limit.")
    special_tokens: dict[str, int] = {}
    for index, record in enumerate(raw_added_tokens):
        if not isinstance(record, dict):
            raise WhisperTokenizerFormatError(f"Whisper added token {index} must be an object.")
        content = record.get("content")
        token_id = _token_id(record.get("id"), name=f"added token {index} ID")
        if not isinstance(content, str) or not content:
            raise WhisperTokenizerFormatError(f"Whisper added token {index} has invalid content.")
        if record.get("lstrip") or record.get("rstrip"):
            raise WhisperTokenizerFormatError("Whitespace-stripping Whisper added tokens are unsupported.")
        is_timestamp = _timestamp_index_from_text(content) is not None
        if record.get("special") is True or is_timestamp:
            previous = special_tokens.get(content)
            if previous is not None and previous != token_id:
                raise WhisperTokenizerFormatError(f"Whisper added token {content!r} has conflicting IDs.")
            special_tokens[content] = token_id
        elif vocabulary_strings.get(content) != token_id:
            raise WhisperTokenizerFormatError(f"Unsupported non-special Whisper added token: {content!r}.")

    pre_tokenizer = document.get("pre_tokenizer")
    if not isinstance(pre_tokenizer, dict) or pre_tokenizer.get("type") != "ByteLevel":
        raise WhisperTokenizerFormatError("Whisper tokenizer JSON must use a ByteLevel pre-tokenizer.")
    add_prefix_space = pre_tokenizer.get("add_prefix_space", False)
    use_regex = pre_tokenizer.get("use_regex", True)
    if not isinstance(add_prefix_space, bool) or not isinstance(use_regex, bool):
        raise WhisperTokenizerFormatError("Whisper ByteLevel options must be booleans.")
    decoder = document.get("decoder")
    if decoder is not None and (not isinstance(decoder, dict) or decoder.get("type") != "ByteLevel"):
        raise WhisperTokenizerFormatError("Whisper tokenizer JSON must use a ByteLevel decoder.")

    eot = special_tokens.get("<|endoftext|>")
    if eot is None:
        raise WhisperTokenizerFormatError("Whisper tokenizer is missing `<|endoftext|>`.")
    if empty_token_ids and empty_token_ids != {eot - 1}:
        raise WhisperTokenizerFormatError(
            "Whisper's reserved empty token must immediately precede "
            "`<|endoftext|>`.")
    return (
        ByteBPETokenizer(
            vocabulary,
            merges=tuple(merges),
            special_tokens=special_tokens,
            pad_token_id=eot,
            add_prefix_space=add_prefix_space,
            use_regex=use_regex,
        ),
        frozenset(empty_token_ids),
    )


def _validate_mergeable_vocabulary(ranks: Mapping[bytes, int]) -> None:
    _validate_byte_alphabet(ranks)
    expected_ids = set(range(len(ranks)))
    actual_ids = set(ranks.values())
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)[:5]
        unexpected = sorted(actual_ids - expected_ids)[:5]
        raise WhisperTokenizerFormatError(
            "OpenAI Whisper mergeable ranks must be contiguous from zero; "
            f"missing={missing}, unexpected={unexpected}.")
    empty_token_id = ranks.get(b"")
    if empty_token_id is not None and empty_token_id != len(ranks) - 1:
        raise WhisperTokenizerFormatError(
            "OpenAI Whisper's reserved empty token must have the final "
            "mergeable rank.")


def _validate_byte_alphabet(vocabulary: Mapping[bytes, int]) -> None:
    missing = tuple(value for value in range(256) if bytes((value, )) not in vocabulary)
    if missing:
        preview = ", ".join(f"0x{value:02x}" for value in missing[:8])
        raise WhisperTokenizerFormatError(
            "Whisper byte-BPE vocabulary is incomplete; missing "
            f"{preview}.")


def _parse_merge(value: Any, *, index: int) -> tuple[str, str]:
    fields = value.split(" ") if isinstance(value, str) else value
    if (not isinstance(fields, (list, tuple)) or len(fields) != 2 or
            not all(isinstance(field, str) for field in fields)):
        raise WhisperTokenizerFormatError(f"Whisper BPE merge {index} must contain two token strings.")
    return fields[0], fields[1]


def _language_code_for_token(token: str) -> str | None:
    for code in LANGUAGES:
        if token == f"<|{code}|>" or token in _LANGUAGE_TOKEN_ALIASES.get(code, ()):
            return code
    return None


def _normalize_language(language: str | None) -> str | None:
    if language is None:
        return None
    if not isinstance(language, str) or not language.strip():
        raise TypeError("Whisper `language` must be a non-empty string or None.")
    normalized = language.strip().lower()
    if normalized in LANGUAGES:
        return normalized
    code = TO_LANGUAGE_CODE.get(normalized)
    if code is None:
        raise ValueError(f"Unsupported Whisper language: {language!r}.")
    return code


def _normalize_task(task: str | None) -> str | None:
    if task is None:
        return None
    if not isinstance(task, str):
        raise TypeError("Whisper `task` must be a string or None.")
    normalized = task.strip().lower()
    if normalized not in TASKS:
        raise ValueError(f"Unsupported Whisper task {task!r}; expected 'transcribe' or "
                         "'translate'.")
    return normalized


def _timestamp_text(index: int) -> str:
    return f"<|{index * TIMESTAMP_PRECISION:.2f}|>"


def _timestamp_index_from_text(value: str) -> int | None:
    if not (value.startswith("<|") and value.endswith("|>")):
        return None
    number = value[2:-2]
    try:
        seconds = float(number)
    except ValueError:
        return None
    if not math.isfinite(seconds) or seconds < 0:
        return None
    position = seconds / TIMESTAMP_PRECISION
    index = round(position)
    if not math.isclose(position, index, rel_tol=0.0, abs_tol=1e-7):
        return None
    if value != _timestamp_text(index):
        return None
    return index


def _truncate_content(
    encoding: Encoding,
    *,
    max_length: int | None,
    truncation: TruncationStrategy,
    reserved: int,
) -> Encoding:
    if truncation not in (False, True, "left", "right"):
        raise ValueError("`truncation` must be False, True, 'left', or 'right'.")
    if max_length is None:
        return encoding
    total_limit = _nonnegative_integer(max_length, name="max_length")
    if total_limit < reserved:
        raise ValueError(
            f"`max_length={total_limit}` cannot fit {reserved} required "
            "Whisper special tokens.")
    content_limit = total_limit - reserved
    if len(encoding) <= content_limit:
        return encoding
    if truncation is False:
        raise ValueError(
            f"Whisper encoding length {len(encoding) + reserved} exceeds "
            f"`max_length={total_limit}`; enable truncation explicitly.")
    selection = (
        slice(len(encoding) - content_limit, None) if truncation == "left" else slice(None, content_limit))
    return Encoding(
        input_ids=encoding.input_ids[selection],
        attention_mask=encoding.attention_mask[selection],
        special_tokens_mask=encoding.special_tokens_mask[selection],
    )


def _coerce_token_ids(values: Iterable[int] | Encoding) -> tuple[int, ...]:
    source = values.input_ids if isinstance(values, Encoding) else values
    try:
        result = tuple(source)
    except TypeError as error:
        raise TypeError("Whisper token IDs must be an iterable.") from error
    return tuple(_token_id(value, name="token ID") for value in result)


def _token_id(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"`{name}` must be an integer.")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"`{name}` must be non-negative.")
    return normalized


def _positive_integer(value: object, *, name: str) -> int:
    normalized = _token_id(value, name=name)
    if normalized == 0:
        raise ValueError(f"`{name}` must be greater than zero.")
    return normalized


def _nonnegative_integer(value: object, *, name: str) -> int:
    return _token_id(value, name=name)


def _language_count(value: object) -> int:
    count = _positive_integer(value, name="num_languages")
    if count > len(LANGUAGES):
        raise ValueError(f"`num_languages` cannot exceed {len(LANGUAGES)}.")
    return count


def _boolean(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"`{name}` must be a boolean.")
    return value


def _validate_json_bounds(value: Any, *, max_depth: int, max_nodes: int) -> None:
    stack: list[tuple[Any, int]] = [(value, 1)]
    nodes = 0
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if nodes > max_nodes:
            raise WhisperTokenizerFormatError(f"Whisper tokenizer JSON exceeds {max_nodes} values.")
        if depth > max_depth:
            raise WhisperTokenizerFormatError(f"Whisper tokenizer JSON exceeds nesting depth {max_depth}.")
        if isinstance(item, dict):
            stack.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            stack.extend((child, depth + 1) for child in item)


def _reject_json_constant(value: str) -> None:
    raise WhisperTokenizerFormatError(f"Non-finite JSON constant {value!r} is not allowed.")


__all__ = [
    "DEFAULT_MAX_TOKENIZER_BYTES",
    "LANGUAGES",
    "MAX_TIMESTAMP_SECONDS",
    "OPENAI_WHISPER_REVISION",
    "TASKS",
    "TIMESTAMP_COUNT",
    "TIMESTAMP_PRECISION",
    "TO_LANGUAGE_CODE",
    "TRANSFORMERS_WHISPER_REVISION",
    "TimestampToken",
    "WhisperSpecialTokens",
    "WhisperTokenizer",
    "WhisperTokenizerFormatError",
    "build_openai_whisper_special_tokens",
    "discover_whisper_special_tokens",
]
