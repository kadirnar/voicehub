"""Declarative, dependency-free text frontend for VITS and MMS-TTS.

Language-specific normalization, romanization, and phonemization are
explicit protocols. Loading a model asset never imports or executes
frontend code from a remote repository.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from numbers import Integral
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from voicehub.tokenization import (
    BatchEncoding,
    Encoding,
    PaddingStrategy,
    TruncationStrategy,
    pad_encodings,
    read_bounded_asset,
)

DEFAULT_MAX_FRONTEND_BYTES = 4 * 1024 * 1024
DEFAULT_MAX_VOCABULARY = 100_000
_WHITESPACE = re.compile(r"\s+")


class VitsFrontendError(ValueError):
    """Base error for invalid or unavailable language frontend behavior."""


class VitsFrontendAssetError(VitsFrontendError):
    """Raised when a declarative VITS frontend asset is malformed."""


class VitsFrontendCapabilityError(VitsFrontendError):
    """Raised when requested language processing has no supplied provider."""


@runtime_checkable
class TextNormalizer(Protocol):
    """Language-aware text normalizer supplied by the application."""

    def normalize(self, text: str, *, language: str | None) -> str:
        """Return normalized text without changing token semantics."""


@runtime_checkable
class TextRomanizer(Protocol):
    """Language-aware romanization contract."""

    def romanize(self, text: str, *, language: str | None) -> str:
        """Convert text to the script expected by the checkpoint."""


@runtime_checkable
class TextPhonemizer(Protocol):
    """Language-aware phonemization contract."""

    def phonemize(self, text: str, *, language: str | None) -> str:
        """Convert normalized text into checkpoint-compatible phonemes."""


@dataclass(frozen=True, slots=True)
class VitsFrontendConfig:
    """Serializable frontend policy read from ``tokenizer_config.json``."""

    language: str | None = None
    add_blank: bool = True
    normalize: bool = True
    phonemize: bool = True
    romanize: bool = False
    pad_token: str = "<pad>"
    unk_token: str = "<unk>"
    extra_config: Mapping[str, Any] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.language is not None:
            if not isinstance(self.language, str) or not self.language.strip():
                raise ValueError("`language` must be a non-empty string or None.")
            object.__setattr__(self, "language", self.language.strip())
        for name in ("add_blank", "normalize", "phonemize", "romanize"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"`{name}` must be a boolean.")
        for name in ("pad_token", "unk_token"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"`{name}` must be a non-empty string.")
        if not isinstance(self.extra_config, Mapping):
            raise TypeError("`extra_config` must be a mapping.")
        object.__setattr__(
            self,
            "extra_config",
            MappingProxyType(copy.deepcopy(dict(self.extra_config))),
        )

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> VitsFrontendConfig:
        """Read the stable subset of Hugging Face VITS tokenizer metadata."""
        if not isinstance(values, Mapping):
            raise TypeError("Frontend configuration values must be a mapping.")
        source = copy.deepcopy(dict(values))
        known = {
            "language",
            "add_blank",
            "normalize",
            "phonemize",
            "romanize",
            "is_uroman",
            "pad_token",
            "unk_token",
        }
        romanize = source.get("romanize", source.get("is_uroman", False))
        return cls(
            language=source.get("language"),
            add_blank=source.get("add_blank", True),
            normalize=source.get("normalize", True),
            phonemize=source.get("phonemize", True),
            romanize=romanize,
            pad_token=source.get("pad_token", "<pad>"),
            unk_token=source.get("unk_token", "<unk>"),
            extra_config={
                key: value
                for key, value in source.items() if key not in known
            },
        )

    def to_dict(self) -> dict[str, Any]:
        result = copy.deepcopy(dict(self.extra_config))
        result.update({
            "language": self.language,
            "add_blank": self.add_blank,
            "normalize": self.normalize,
            "phonemize": self.phonemize,
            "is_uroman": self.romanize,
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
        })
        return result


class VitsTokenizer:
    """Character tokenizer for declarative VITS/MMS frontend assets."""

    def __init__(
        self,
        vocabulary: Mapping[str, int],
        *,
        config: VitsFrontendConfig | Mapping[str, Any] | None = None,
        normalizer: TextNormalizer | None = None,
        romanizer: TextRomanizer | None = None,
        phonemizer: TextPhonemizer | None = None,
    ) -> None:
        self._vocabulary = _validate_vocabulary(vocabulary)
        self._id_to_token = MappingProxyType({
            token_id: token
            for token, token_id in self._vocabulary.items()
        })
        self.config = (
            VitsFrontendConfig() if config is None else
            config if isinstance(config, VitsFrontendConfig) else VitsFrontendConfig.from_mapping(config))
        _validate_provider(normalizer, TextNormalizer, name="normalizer")
        _validate_provider(romanizer, TextRomanizer, name="romanizer")
        _validate_provider(phonemizer, TextPhonemizer, name="phonemizer")
        self.normalizer = normalizer
        self.romanizer = romanizer
        self.phonemizer = phonemizer

        try:
            self.pad_token_id = self._vocabulary[self.config.pad_token]
        except KeyError:
            if self.config.add_blank and 0 in self._id_to_token:
                self.pad_token_id = 0
            else:
                raise VitsFrontendAssetError(
                    f"Pad token {self.config.pad_token!r} is absent from the "
                    "VITS vocabulary.") from None
        self.unk_token_id = self._vocabulary.get(self.config.unk_token)
        if self.config.add_blank and 0 not in self._id_to_token:
            raise VitsFrontendAssetError("Blank interspersion requires vocabulary token ID zero.")

    @classmethod
    def from_files(
        cls,
        vocab_file: str | Path,
        *,
        tokenizer_config_file: str | Path | None = None,
        max_asset_bytes: int = DEFAULT_MAX_FRONTEND_BYTES,
        max_vocabulary: int = DEFAULT_MAX_VOCABULARY,
        **providers: object,
    ) -> VitsTokenizer:
        """Load declarative MMS ``vocab.json`` and tokenizer metadata."""
        vocabulary = _read_json_mapping(
            vocab_file,
            max_asset_bytes=max_asset_bytes,
            description="VITS vocabulary",
        )
        if len(vocabulary) > _positive_integer(max_vocabulary, name="max_vocabulary"):
            raise VitsFrontendAssetError(f"VITS vocabulary exceeds {max_vocabulary} entries.")
        config: Mapping[str, Any] | None = None
        if tokenizer_config_file is not None:
            config = _read_json_mapping(
                tokenizer_config_file,
                max_asset_bytes=max_asset_bytes,
                description="VITS tokenizer configuration",
            )
        return cls(vocabulary, config=config, **providers)

    @property
    def vocabulary(self) -> Mapping[str, int]:
        return self._vocabulary

    @property
    def vocab_size(self) -> int:
        return max(self._id_to_token) + 1

    def prepare_text(self, text: str) -> str:
        """Apply only configured and explicitly available frontend stages."""
        if not isinstance(text, str):
            raise TypeError("VITS input text must be a string.")
        prepared = text
        if self.config.normalize:
            prepared = (
                _provider_text(
                    "TextNormalizer",
                    self.normalizer.normalize(
                        prepared,
                        language=self.config.language,
                    ),
                ) if self.normalizer is not None else self._vocabulary_aware_lowercase(prepared))
        if self.config.language == "ron":
            prepared = prepared.replace("ț", "ţ")
        if self.config.romanize and _contains_non_ascii(prepared):
            if self.romanizer is None:
                raise VitsFrontendCapabilityError(
                    "This checkpoint requires romanization; supply a "
                    "TextRomanizer implementation.")
            prepared = _provider_text(
                "TextRomanizer",
                self.romanizer.romanize(
                    prepared,
                    language=self.config.language,
                ),
            )
        if self.config.phonemize:
            if self.phonemizer is None:
                raise VitsFrontendCapabilityError(
                    "This checkpoint requires phonemization; supply a "
                    "TextPhonemizer implementation.")
            prepared = _provider_text(
                "TextPhonemizer",
                self.phonemizer.phonemize(
                    prepared,
                    language=self.config.language,
                ),
            )
            prepared = _WHITESPACE.sub(" ", prepared).strip()
        elif self.config.normalize:
            prepared = "".join(character for character in prepared if character in self._vocabulary).strip()
        if not isinstance(prepared, str):
            raise TypeError("Frontend providers must return strings.")
        return prepared

    def encode(
        self,
        text: str,
        *,
        max_length: int | None = None,
        truncation: TruncationStrategy = False,
    ) -> Encoding:
        prepared = self.prepare_text(text)
        token_ids: list[int] = []
        position = 0
        while position < len(prepared):
            special_position = prepared.find(
                self.config.pad_token,
                position,
            )
            if special_position < 0:
                token_ids.extend(self._encode_text_segment(prepared[position:]))
                break
            token_ids.extend(self._encode_text_segment(prepared[position:special_position]))
            token_ids.append(self.pad_token_id)
            position = special_position + len(self.config.pad_token)
        encoding = Encoding(tuple(token_ids))
        return _truncate(encoding, max_length=max_length, truncation=truncation)

    def _encode_text_segment(self, text: str) -> list[int]:
        if not text:
            return []
        token_ids = []
        for token in text:
            token_id = self._vocabulary.get(token, self.unk_token_id)
            if token_id is None:
                raise VitsFrontendError(
                    f"Token {token!r} is absent and this vocabulary has no "
                    "unknown token.")
            token_ids.append(token_id)
        if not self.config.add_blank:
            return token_ids
        blanked = [0] * (len(token_ids) * 2 + 1)
        blanked[1::2] = token_ids
        return blanked

    def encode_batch(
        self,
        texts: Iterable[str],
        *,
        padding: PaddingStrategy = False,
        max_length: int | None = None,
        truncation: TruncationStrategy = False,
        pad_to_multiple_of: int | None = None,
    ) -> BatchEncoding:
        try:
            values = tuple(texts)
        except TypeError as error:
            raise TypeError("`texts` must be an iterable of strings.") from error
        encodings = tuple(
            self.encode(
                text,
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
        target = None
        if padding == "max_length":
            if max_length is None:
                raise ValueError("`padding='max_length'` requires `max_length`.")
            target = _nonnegative_integer(max_length, name="max_length")
        return pad_encodings(
            encodings,
            pad_token_id=self.pad_token_id,
            length=target,
            pad_to_multiple_of=pad_to_multiple_of,
        )

    def decode(
        self,
        token_ids: Iterable[int] | Encoding,
        *,
        skip_padding: bool = False,
    ) -> str:
        source = token_ids.input_ids if isinstance(token_ids, Encoding) else token_ids
        try:
            values = tuple(source)
        except TypeError as error:
            raise TypeError("VITS token IDs must be iterable.") from error
        if self.config.add_blank and not skip_padding:
            values = values[1::2]
        pieces = []
        for value in values:
            token_id = _nonnegative_integer(value, name="token ID")
            if skip_padding and token_id == self.pad_token_id:
                continue
            try:
                pieces.append(self._id_to_token[token_id])
            except KeyError:
                raise VitsFrontendError(f"Unknown VITS token ID {token_id}.") from None
        return "".join(pieces)

    def _vocabulary_aware_lowercase(self, text: str) -> str:
        """Match upstream normalization while preserving exact vocab tokens."""
        output = []
        position = 0
        vocabulary_tokens = tuple(self._vocabulary)
        while position < len(text):
            match = next(
                (token for token in vocabulary_tokens if text.startswith(token, position)),
                None,
            )
            if match is None:
                output.append(text[position].lower())
                position += 1
            else:
                output.append(match)
                position += len(match)
        return "".join(output)


def _validate_vocabulary(value: Mapping[str, int]) -> Mapping[str, int]:
    if not isinstance(value, Mapping) or not value:
        raise VitsFrontendAssetError("VITS vocabulary must be a non-empty mapping.")
    normalized: dict[str, int] = {}
    seen_ids: set[int] = set()
    for token, token_id in value.items():
        if not isinstance(token, str) or not token:
            raise VitsFrontendAssetError("VITS vocabulary tokens must be non-empty strings.")
        normalized_id = _nonnegative_integer(
            token_id,
            name=f"ID for {token!r}",
        )
        if normalized_id in seen_ids:
            raise VitsFrontendAssetError(f"VITS vocabulary repeats token ID {normalized_id}.")
        normalized[token] = normalized_id
        seen_ids.add(normalized_id)
    expected = set(range(max(seen_ids) + 1))
    if seen_ids != expected:
        raise VitsFrontendAssetError("VITS vocabulary IDs must be contiguous from zero.")
    return MappingProxyType(normalized)


def _read_json_mapping(
    path: str | Path,
    *,
    max_asset_bytes: int,
    description: str,
) -> Mapping[str, Any]:
    payload = read_bounded_asset(path, max_bytes=max_asset_bytes)
    try:
        value = json.loads(
            payload.decode("utf-8"),
            parse_constant=lambda constant: _reject_constant(constant),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise VitsFrontendAssetError(f"Invalid {description} JSON: {error}.") from error
    if not isinstance(value, dict):
        raise VitsFrontendAssetError(f"{description} must be a JSON object.")
    return value


def _reject_constant(value: str) -> None:
    raise VitsFrontendAssetError(f"Non-finite JSON constant {value!r} is not allowed.")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]], ) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in pairs:
        if name in result:
            raise VitsFrontendAssetError(f"VITS frontend JSON repeats key {name!r}.")
        result[name] = value
    return result


def _validate_provider(value: object, protocol: type, *, name: str) -> None:
    if value is not None and not isinstance(value, protocol):
        raise TypeError(f"`{name}` does not implement {protocol.__name__}.")


def _provider_text(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} providers must return strings.")
    return value


def _contains_non_ascii(value: str) -> bool:
    return any(ord(character) > 127 for character in value)


def _positive_integer(value: object, *, name: str) -> int:
    normalized = _nonnegative_integer(value, name=name)
    if normalized == 0:
        raise ValueError(f"`{name}` must be greater than zero.")
    return normalized


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"`{name}` must be an integer.")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"`{name}` must be non-negative.")
    return normalized


def _truncate(
    encoding: Encoding,
    *,
    max_length: int | None,
    truncation: TruncationStrategy,
) -> Encoding:
    if truncation not in (False, True, "left", "right"):
        raise ValueError("`truncation` must be False, True, 'left', or 'right'.")
    if max_length is None:
        return encoding
    limit = _nonnegative_integer(max_length, name="max_length")
    if len(encoding) <= limit:
        return encoding
    if truncation is False:
        raise ValueError(
            f"VITS encoding length {len(encoding)} exceeds "
            f"`max_length={limit}`; enable truncation explicitly.")
    selection = (slice(len(encoding) - limit, None) if truncation == "left" else slice(None, limit))
    return Encoding(
        input_ids=encoding.input_ids[selection],
        attention_mask=encoding.attention_mask[selection],
        special_tokens_mask=encoding.special_tokens_mask[selection],
    )


__all__ = [
    "DEFAULT_MAX_FRONTEND_BYTES",
    "DEFAULT_MAX_VOCABULARY",
    "TextNormalizer",
    "TextPhonemizer",
    "TextRomanizer",
    "VitsFrontendAssetError",
    "VitsFrontendCapabilityError",
    "VitsFrontendConfig",
    "VitsFrontendError",
    "VitsTokenizer",
]
