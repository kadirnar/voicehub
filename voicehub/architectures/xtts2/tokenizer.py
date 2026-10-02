"""Small standard-library BPE reader for the published XTTS vocabulary."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from voicehub.tokenization.assets import read_bounded_asset

_WHITESPACE_PATTERN = re.compile(r"\s+")
# Oniguruma's Unicode ``\w`` (used by the Hugging Face ``Whitespace``
# pre-tokenizer) also matches combining marks, letter numbers and joiners,
# unlike Python's ``\w``; Devanagari vowel signs and Arabic harakat must stay
# inside their word.
_JOIN_CONTROLS = frozenset({"\u200c", "\u200d"})
_LANGUAGE_ALIASES = {"zh": "zh-cn"}
_TRANSCRIPTION_REQUIRED = frozenset({"ja", "ko", "zh", "zh-cn"})
_ABBREVIATIONS = {
    "en": (
        ("mrs", "misess"),
        ("mr", "mister"),
        ("dr", "doctor"),
        ("st", "saint"),
        ("co", "company"),
        ("jr", "junior"),
        ("maj", "major"),
        ("gen", "general"),
        ("drs", "doctors"),
        ("rev", "reverend"),
        ("lt", "lieutenant"),
        ("hon", "honorable"),
        ("sgt", "sergeant"),
        ("capt", "captain"),
        ("esq", "esquire"),
        ("ltd", "limited"),
        ("col", "colonel"),
        ("ft", "fort"),
    ),
    "es": (
        ("sra", "señora"),
        ("sr", "señor"),
        ("dr", "doctor"),
        ("dra", "doctora"),
        ("st", "santo"),
        ("co", "compañía"),
        ("jr", "junior"),
        ("ltd", "limitada"),
    ),
    "fr": (
        ("mme", "madame"),
        ("mr", "monsieur"),
        ("dr", "docteur"),
        ("st", "saint"),
        ("co", "compagnie"),
        ("jr", "junior"),
        ("ltd", "limitée"),
    ),
    "de": (
        ("fr", "frau"),
        ("dr", "doktor"),
        ("st", "sankt"),
        ("co", "firma"),
        ("jr", "junior"),
    ),
    "pt": (
        ("sra", "senhora"),
        ("sr", "senhor"),
        ("dr", "doutor"),
        ("dra", "doutora"),
        ("st", "santo"),
        ("co", "companhia"),
        ("jr", "júnior"),
        ("ltd", "limitada"),
    ),
    "it": (
        ("sig", "signore"),
        ("dr", "dottore"),
        ("st", "santo"),
        ("co", "compagnia"),
        ("jr", "junior"),
        ("ltd", "limitata"),
    ),
    "pl": (
        ("p", "pani"),
        ("m", "pan"),
        ("dr", "doktor"),
        ("sw", "święty"),
        ("jr", "junior"),
    ),
    "cs": (
        ("dr", "doktor"),
        ("ing", "inženýr"),
        ("p", "pan"),
    ),
    "nl": (
        ("dhr", "de heer"),
        ("mevr", "mevrouw"),
        ("dr", "dokter"),
        ("jhr", "jonkheer"),
    ),
    "tr": (
        ("b", "bay"),
        ("byk", "büyük"),
        ("dr", "doktor"),
    ),
    "hu": (
        ("dr", "doktor"),
        ("b", "bácsi"),
        ("nőv", "nővér"),
    ),
}
# Russian abbreviations carry no trailing dot in the source tables.
_WORD_ABBREVIATIONS = {
    "ru": (
        ("г-жа", "госпожа"),
        ("г-н", "господин"),
        ("д-р", "доктор"),
    ),
}
_SYMBOLS = {
    "en": (" and ", " at ", " percent ", " hash ", " dollar ", " pound ", " degree "),
    "es": (" y ", " arroba ", " por ciento ", " numeral ", " dolar ", " libra ", " grados "),
    "fr": (" et ", " arobase ", " pour cent ", " dièse ", " dollar ", " livre ", " degrés "),
    "de": (" und ", " at ", " prozent ", " raute ", " dollar ", " pfund ", " grad "),
    "pt": (" e ", " arroba ", " por cento ", " cardinal ", " dólar ", " libra ", " graus "),
    "it": (" e ", " chiocciola ", " per cento ", " cancelletto ", " dollaro ", " sterlina ", " gradi "),
    "pl": (" i ", " małpa ", " procent ", " krzyżyk ", " dolar ", " funt ", " stopnie "),
    "ar": (" و ", " على ", " في المئة ", " رقم ", " دولار ", " جنيه ", " درجة "),
    "zh": (" 和 ", " 在 ", " 百分之 ", " 号 ", " 美元 ", " 英镑 ", " 度 "),
    "cs": (" a ", " na ", " procento ", " křížek ", " dolar ", " libra ", " stupně "),
    "ru": (" и ", " собака ", " процентов ", " номер ", " доллар ", " фунт ", " градус "),
    "nl": (" en ", " bij ", " procent ", " hekje ", " dollar ", " pond ", " graden "),
    "tr": (" ve ", " at ", " yüzde ", " diyez ", " dolar ", " sterlin ", " derece "),
    "hu": (" és ", " kukac ", " százalék ", " kettőskereszt ", " dollár ", " font ", " fok "),
    "ko": (" 그리고 ", " 에 ", " 퍼센트 ", " 번호 ", " 달러 ", " 파운드 ", " 도 "),
}
_SYMBOL_CHARACTERS = ("&", "@", "%", "#", "$", "£", "°")
_ABBREVIATION_PATTERNS = {
    language: tuple((re.compile(r"\b%s\." % source, re.IGNORECASE), target) for source, target in table)
    for language, table in _ABBREVIATIONS.items()
}
for _language, _table in _WORD_ABBREVIATIONS.items():
    _ABBREVIATION_PATTERNS[_language] = tuple(
        (re.compile(r"\b%s\b" % source, re.IGNORECASE), target) for source, target in _table)
# The source cleaners only run for these codes; Hindi only lowercases and
# collapses whitespace.
_MULTILINGUAL_CLEANER_LANGUAGES = frozenset(
    {"ar", "cs", "de", "en", "es", "fr", "hu", "it", "nl", "pl", "pt", "ru", "tr", "zh", "ko"})
_SOURCE_SPECIALS = {
    "[STOP]",
    "[UNK]",
    "[SPACE]",
    "[START]",
}


class XTTS2Tokenizer:
    """Execute the BPE graph stored in XTTS's immutable ``vocab.json``."""

    def __init__(self, vocabulary: dict[str, int], merges: list[str]) -> None:
        if (not vocabulary or any(not isinstance(token, str) or not token or isinstance(token_id, bool) or
                                  not isinstance(token_id, int) or token_id < 0
                                  for token, token_id in vocabulary.items())):
            raise ValueError("XTTS v2 vocabulary contains an invalid token record.")
        if len(set(vocabulary.values())) != len(vocabulary):
            raise ValueError("XTTS v2 vocabulary contains duplicate token IDs.")
        if set(vocabulary.values()) != set(range(len(vocabulary))):
            raise ValueError("XTTS v2 vocabulary IDs must be contiguous from zero.")
        merge_pairs = []
        for merge in merges:
            if not isinstance(merge, str):
                raise ValueError("XTTS v2 BPE merges must be strings.")
            pair = tuple(merge.split(" "))
            if len(pair) != 2 or not all(pair):
                raise ValueError(f"Invalid XTTS v2 BPE merge: {merge!r}.")
            merge_pairs.append(pair)
        self.vocabulary = dict(vocabulary)
        # The released vocabulary repeats some merge pairs. The reference
        # tokenizer resolves them to the final occurrence.
        self.merge_ranks = {pair: rank for rank, pair in enumerate(merge_pairs)}
        self.unknown_id = self.vocabulary["[UNK]"]
        self.start_id = self.vocabulary["[START]"]
        self.stop_id = self.vocabulary["[STOP]"]
        self.space_id = self.vocabulary["[SPACE]"]
        self._cache: dict[str, tuple[str, ...]] = {}

    @classmethod
    def from_file(cls, path: str | Path) -> XTTS2Tokenizer:
        source = Path(path).expanduser().resolve()
        try:
            value = json.loads(read_bounded_asset(
                source,
                max_bytes=4 * 1024 * 1024,
            ).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
            raise ValueError(f"Invalid XTTS v2 vocabulary JSON: {error}.") from error
        if not isinstance(value, dict):
            raise ValueError("XTTS v2 vocabulary must contain a JSON object.")
        if value.get("normalizer") is not None or value.get("decoder") is not None:
            raise ValueError("XTTS v2 vocabulary has an unsupported tokenizer pipeline.")
        if value.get("pre_tokenizer") != {"type": "Whitespace"}:
            raise ValueError("XTTS v2 requires the released Whitespace pre-tokenizer.")
        model = value.get("model")
        if (not isinstance(model, dict) or model.get("type") != "BPE" or model.get("unk_token") != "[UNK]" or
                model.get("continuing_subword_prefix") is not None or
                model.get("end_of_word_suffix") is not None or model.get("fuse_unk") is not False):
            raise ValueError("XTTS v2 vocabulary must contain a BPE model.")
        vocabulary = model.get("vocab")
        merges = model.get("merges")
        if not isinstance(vocabulary, dict) or not isinstance(merges, list):
            raise ValueError("XTTS v2 BPE vocabulary is incomplete.")
        missing_specials = sorted(_SOURCE_SPECIALS - set(vocabulary))
        if missing_specials:
            raise ValueError("XTTS v2 vocabulary is missing required tokens: " + ", ".join(missing_specials))
        return cls(vocabulary, merges)

    @staticmethod
    def _preprocess_text(
        text: str,
        *,
        language: str,
        preprocessed: bool,
    ) -> str:
        if preprocessed:
            normalized = unicodedata.normalize("NFKC", text).strip()
            return _WHITESPACE_PATTERN.sub(" ", normalized)
        base_language = language.split("-", 1)[0]
        if base_language in _TRANSCRIPTION_REQUIRED:
            raise ValueError(
                f"XTTS {language!r} requires author-compatible "
                "transliteration. Pass source-normalized romanized text with "
                "`text_is_normalized=True`.")
        # The source inference loop strips and lowercases every sentence
        # before the language cleaners run (so its Turkish capital-letter
        # replacements never see a capital).
        normalized = text.strip().lower()
        if any(character.isdigit() for character in normalized):
            raise ValueError(
                "XTTS numeric text requires language-specific verbalization. "
                "Spell numbers out or pass source-normalized text with "
                "`text_is_normalized=True`.")
        if base_language not in _MULTILINGUAL_CLEANER_LANGUAGES:
            return _WHITESPACE_PATTERN.sub(" ", normalized)
        normalized = normalized.replace('"', "")
        if base_language == "tr":
            normalized = (normalized.replace("İ", "i").replace("Ö", "ö").replace("Ü", "ü"))
        normalized = normalized.lower()
        for pattern, replacement in _ABBREVIATION_PATTERNS.get(base_language, ()):
            normalized = pattern.sub(replacement, normalized)
        for symbol, replacement in zip(_SYMBOL_CHARACTERS, _SYMBOLS[base_language]):
            normalized = normalized.replace(symbol, replacement).replace("  ", " ")
        return _WHITESPACE_PATTERN.sub(" ", normalized.strip())

    def encode(
        self,
        text: str,
        *,
        language: str,
        preprocessed: bool = False,
    ) -> list[int]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("XTTS v2 text must be non-empty.")
        if not isinstance(preprocessed, bool):
            raise TypeError("`preprocessed` must be a boolean.")
        language = str(language).strip().lower()
        language = _LANGUAGE_ALIASES.get(language, language)
        language_token = f"[{language}]"
        if language_token not in self.vocabulary:
            raise ValueError(f"XTTS v2 vocabulary has no {language_token!r} token.")
        normalized = self._preprocess_text(
            text,
            language=language,
            preprocessed=preprocessed,
        )
        result = [self.vocabulary[language_token]]
        words = normalized.split(" ")
        for index, word in enumerate(words):
            if index:
                result.append(self.space_id)
            for token in _pretokenize(word):
                result.extend(self.vocabulary.get(piece, self.unknown_id) for piece in self._bpe(token))
        return result

    def _bpe(self, token: str) -> tuple[str, ...]:
        cached = self._cache.get(token)
        if cached is not None:
            return cached
        pieces = tuple(token)
        while len(pieces) > 1:
            pairs = tuple(zip(pieces, pieces[1:]))
            ranked = [(self.merge_ranks[pair], pair) for pair in pairs if pair in self.merge_ranks]
            if not ranked:
                break
            _rank, selected = min(ranked)
            merged = []
            index = 0
            while index < len(pieces):
                if index + 1 < len(pieces) and pieces[index:index + 2] == selected:
                    merged.append("".join(selected))
                    index += 2
                else:
                    merged.append(pieces[index])
                    index += 1
            pieces = tuple(merged)
        self._cache[token] = pieces
        return pieces

    def token_to_id(self, token: str) -> int | None:
        return self.vocabulary.get(token)

    def __len__(self) -> int:
        return len(self.vocabulary)


def _is_word_character(character: str) -> bool:
    category = unicodedata.category(character)
    return (category[0] in "LM" or category in {"Nd", "Nl", "Pc"} or character in _JOIN_CONTROLS)


def _pretokenize(text: str) -> list[str]:
    """Split like the ``Whitespace`` pre-tokenizer: ``\\w+|[^\\w\\s]+``."""
    pieces = []
    start = None
    kind = None
    for index, character in enumerate(text):
        current = (None if character.isspace() else _is_word_character(character))
        if current != kind:
            if kind is not None:
                pieces.append(text[start:index])
            start = index
            kind = current
    if kind is not None:
        pieces.append(text[start:])
    return pieces


__all__ = ["XTTS2Tokenizer"]
