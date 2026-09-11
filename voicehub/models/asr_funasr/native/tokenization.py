"""SenseVoice control semantics over VoiceHub's SentencePiece runtime."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from voicehub.tokenization import SentencePieceUnigramTokenizer

LANGUAGE_TOKENS = {
    "zh": ("<|zh|>", 24_884),
    "en": ("<|en|>", 24_885),
    "yue": ("<|yue|>", 24_888),
    "ja": ("<|ja|>", 24_892),
    "ko": ("<|ko|>", 24_896),
    "nospeech": ("<|nospeech|>", 24_992),
}
EMOTION_TOKENS = {
    "happy": ("<|HAPPY|>", 25_001),
    "sad": ("<|SAD|>", 25_002),
    "angry": ("<|ANGRY|>", 25_003),
    "neutral": ("<|NEUTRAL|>", 25_004),
    "fearful": ("<|FEARFUL|>", 25_005),
    "disgusted": ("<|DISGUSTED|>", 25_006),
    "surprised": ("<|SURPRISED|>", 25_007),
    "other": ("<|OTHER|>", 25_008),
    "unknown": ("<|EMO_UNKNOWN|>", 25_009),
}
EVENT_TOKENS = {
    "speech": ("<|Speech|>", 24_993),
    "bgm": ("<|BGM|>", 24_995),
    "laughter": ("<|Laughter|>", 24_997),
    "applause": ("<|Applause|>", 24_999),
    "cry": ("<|Cry|>", 25_010),
    "sneeze": ("<|Sneeze|>", 25_011),
    "breath": ("<|Breath|>", 25_012),
    "cough": ("<|Cough|>", 25_013),
    "sing": ("<|Sing|>", 25_014),
    "speech_noise": ("<|Speech_Noise|>", 25_015),
    "unknown": ("<|Event_UNK|>", 25_019),
}
TEXT_NORMALIZATION_TOKENS = {
    "withitn": ("<|withitn|>", 25_016),
    "woitn": ("<|woitn|>", 25_017),
}

_EMOTION_EMOJI = {
    "<|HAPPY|>": "😊",
    "<|SAD|>": "😔",
    "<|ANGRY|>": "😡",
    "<|NEUTRAL|>": "",
    "<|FEARFUL|>": "😰",
    "<|DISGUSTED|>": "🤢",
    "<|SURPRISED|>": "😮",
}
_EVENT_EMOJI = {
    "<|BGM|>": "🎼",
    "<|Speech|>": "",
    "<|Applause|>": "👏",
    "<|Laughter|>": "😀",
    "<|Cry|>": "😭",
    "<|Sneeze|>": "🤧",
    "<|Breath|>": "",
    "<|Cough|>": "😷",
}
_EMOTION_EMOJIS = frozenset(value for value in _EMOTION_EMOJI.values() if value)
_EVENT_EMOJIS = frozenset(value for value in _EVENT_EMOJI.values() if value)
_LANGUAGE_MARKERS = frozenset(spelling for spelling, _ in LANGUAGE_TOKENS.values())
_ALL_CONTROL_SPELLINGS = frozenset(
    spelling for values in (
        LANGUAGE_TOKENS,
        EMOTION_TOKENS,
        EVENT_TOKENS,
        TEXT_NORMALIZATION_TOKENS,
    ) for spelling, _ in values.values())


@dataclass(frozen=True, slots=True)
class SenseVoiceSemantics:
    language: str | None
    emotion: str | None
    events: tuple[str, ...]
    text_normalization: str | None
    control_tokens: tuple[str, ...]


def _name_for_id(
    token_id: int,
    values: dict[str, tuple[str, int]],
) -> str | None:
    return next(
        (name for name, (_, expected_id) in values.items() if token_id == expected_id),
        None,
    )


def _format_rich_section(value: str) -> str:
    counts = {
        spelling: value.count(spelling)
        for spelling in (
            *_EMOTION_EMOJI,
            *_EVENT_EMOJI,
        )
    }
    for spelling in _ALL_CONTROL_SPELLINGS:
        value = value.replace(spelling, "")
    emotion = "<|NEUTRAL|>"
    for spelling in _EMOTION_EMOJI:
        if counts.get(spelling, 0) > counts.get(emotion, 0):
            emotion = spelling
    for spelling, emoji in _EVENT_EMOJI.items():
        if counts.get(spelling, 0) > 0:
            value = emoji + value
    value += _EMOTION_EMOJI[emotion]
    for emoji in _EMOTION_EMOJIS | _EVENT_EMOJIS:
        value = value.replace(" " + emoji, emoji)
        value = value.replace(emoji + " ", emoji)
    return value.strip()


def rich_transcription_postprocess(value: str) -> str:
    """Preserve the published event/emotion emoji rendering."""
    if not isinstance(value, str):
        raise TypeError("SenseVoice transcription must be a string.")
    value = value.replace("<|nospeech|><|Event_UNK|>", "❓")
    for marker in _LANGUAGE_MARKERS:
        value = value.replace(marker, "<|lang|>")
    sections = [_format_rich_section(section).strip() for section in value.split("<|lang|>")]
    if not sections:
        return ""
    merged = " " + sections[0]
    current_event = (merged[0] if merged and merged[0] in _EVENT_EMOJIS else None)
    for section in sections[1:]:
        if not section:
            continue
        event = section[0] if section[0] in _EVENT_EMOJIS else None
        if event == current_event and event is not None:
            section = section[1:]
        if not section:
            continue
        current_event = (section[0] if section[0] in _EVENT_EMOJIS else None)
        emotion = (section[-1] if section[-1] in _EMOTION_EMOJIS else None)
        merged_emotion = (merged[-1] if merged and merged[-1] in _EMOTION_EMOJIS else None)
        if emotion is not None and emotion == merged_emotion:
            merged = merged[:-1]
        merged += section.strip().lstrip()
    return merged.replace("The.", " ").strip()


class SenseVoiceTokenizer:
    """Validate and expose the released 25,055-piece tokenizer."""

    def __init__(
        self,
        sentencepiece: SentencePieceUnigramTokenizer,
        *,
        strict_release: bool = True,
    ) -> None:
        if not isinstance(sentencepiece, SentencePieceUnigramTokenizer):
            raise TypeError("`sentencepiece` must use VoiceHub's native unigram runtime.")
        self.sentencepiece = sentencepiece
        if strict_release:
            self._validate_release()

    @classmethod
    def from_model_file(
        cls,
        path: str | Path,
        *,
        strict_release: bool = True,
    ) -> SenseVoiceTokenizer:
        return cls(
            SentencePieceUnigramTokenizer.from_model_file(path),
            strict_release=strict_release,
        )

    def _validate_release(self) -> None:
        if self.sentencepiece.vocabulary_size != 25_055:
            raise ValueError("SenseVoiceSmall requires the audited 25,055-piece tokenizer.")
        expected = {
            spelling: token_id
            for values in (
                LANGUAGE_TOKENS,
                EMOTION_TOKENS,
                EVENT_TOKENS,
                TEXT_NORMALIZATION_TOKENS,
            )
            for spelling, token_id in values.values()
        }
        mismatches = {
            token_id: (
                self.sentencepiece.id_to_piece(token_id),
                spelling,
            )
            for spelling, token_id in expected.items() if self.sentencepiece.id_to_piece(token_id) != spelling
        }
        if mismatches:
            raise ValueError("SenseVoice tokenizer control-token mismatch: "
                             f"{mismatches}.")
        if (
                self.sentencepiece.unk_token_id,
                self.sentencepiece.bos_token_id,
                self.sentencepiece.eos_token_id,
        ) != (0, 1, 2):
            raise ValueError("SenseVoice tokenizer requires UNK/BOS/EOS IDs 0/1/2.")

    @property
    def vocabulary_size(self) -> int:
        return self.sentencepiece.vocabulary_size

    def encode_text(self, text: str) -> tuple[int, ...]:
        return tuple(self.sentencepiece.encode_as_ids(text))

    def prepare_training_labels(
        self,
        text: str,
        *,
        language: str,
        emotion: str = "neutral",
        event: str = "speech",
        use_itn: bool = False,
    ) -> tuple[int, ...]:
        try:
            language_id = LANGUAGE_TOKENS[language][1]
        except KeyError as error:
            raise ValueError(
                "SenseVoiceSmall training language must be one of: " + ", ".join(LANGUAGE_TOKENS) +
                ".") from error
        try:
            emotion_id = EMOTION_TOKENS[emotion][1]
        except KeyError as error:
            raise ValueError("Unsupported SenseVoice emotion label: "
                             f"{emotion!r}.") from error
        try:
            event_id = EVENT_TOKENS[event][1]
        except KeyError as error:
            raise ValueError(f"Unsupported SenseVoice event label: {event!r}.") from error
        style = "withitn" if use_itn else "woitn"
        return (
            language_id,
            emotion_id,
            event_id,
            TEXT_NORMALIZATION_TOKENS[style][1],
            *self.encode_text(text),
        )

    def decode_raw(self, token_ids: Iterable[int]) -> str:
        return self.sentencepiece.decode(
            token_ids,
            skip_special_tokens=False,
        )

    def decode_text(self, token_ids: Iterable[int]) -> str:
        return rich_transcription_postprocess(self.decode_raw(token_ids))

    def token_pieces(self, token_ids: Iterable[int]) -> tuple[str, ...]:
        return tuple(self.sentencepiece.id_to_piece(int(token_id)) for token_id in token_ids)

    def semantics(
        self,
        token_ids: Iterable[int],
    ) -> SenseVoiceSemantics:
        ids = tuple(int(token_id) for token_id in token_ids)
        spellings = self.token_pieces(ids)
        return SenseVoiceSemantics(
            language=next(
                (name for token_id in ids if (name := _name_for_id(token_id, LANGUAGE_TOKENS)) is not None),
                None,
            ),
            emotion=next(
                (name for token_id in ids if (name := _name_for_id(token_id, EMOTION_TOKENS)) is not None),
                None,
            ),
            events=tuple(
                name for token_id in ids if (name := _name_for_id(token_id, EVENT_TOKENS)) is not None),
            text_normalization=next(
                (
                    name for token_id in ids if (name := _name_for_id(
                        token_id,
                        TEXT_NORMALIZATION_TOKENS,
                    )) is not None),
                None,
            ),
            control_tokens=tuple(spelling for spelling in spellings if spelling in _ALL_CONTROL_SPELLINGS),
        )

    def save_pretrained(
        self,
        directory: str | Path,
        *,
        filename: str = "tokenizer.model",
    ) -> Path:
        return self.sentencepiece.save_pretrained(
            directory,
            filename=filename,
        )


__all__ = [
    "EMOTION_TOKENS",
    "EVENT_TOKENS",
    "LANGUAGE_TOKENS",
    "TEXT_NORMALIZATION_TOKENS",
    "SenseVoiceSemantics",
    "SenseVoiceTokenizer",
    "rich_transcription_postprocess",
]
