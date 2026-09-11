"""Checkpoint-compatible Zonos phoneme tokenization and conditioning.

Zonos v0.1 was trained on eSpeak phonemes.  A grapheme-to-phoneme engine
is a linguistic runtime, not part of the acoustic checkpoint.  VoiceHub
therefore keeps that boundary explicit: callers may provide phonemes
directly or inject a frontend implementing
:class:`ZonosPhonemeFrontend`.  Raw text is never silently treated as
phonemes, which would produce valid tensor shapes but degrade synthesis
quality.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3
SPECIAL_TOKEN_IDS = (PAD_ID, UNK_ID, BOS_ID, EOS_ID)

PUNCTUATION = ';:,.!?¡¿—…"«»“”() *~-/\\&'
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
IPA_LETTERS = (
    "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸ"
    "θœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷ"
    "ˠˤ˞↓↑→↗↘'̩'ᵻ")
PHONEME_SYMBOLS = tuple((*PUNCTUATION, *LETTERS, *IPA_LETTERS))
PHONEME_SYMBOL_TO_ID = {
    symbol: index
    for index, symbol in enumerate(
        PHONEME_SYMBOLS,
        start=len(SPECIAL_TOKEN_IDS),
    )
}

SUPPORTED_LANGUAGE_CODES = (
    "af",
    "am",
    "an",
    "ar",
    "as",
    "az",
    "ba",
    "bg",
    "bn",
    "bpy",
    "bs",
    "ca",
    "cmn",
    "cs",
    "cy",
    "da",
    "de",
    "el",
    "en-029",
    "en-gb",
    "en-gb-scotland",
    "en-gb-x-gbclan",
    "en-gb-x-gbcwmd",
    "en-gb-x-rp",
    "en-us",
    "eo",
    "es",
    "es-419",
    "et",
    "eu",
    "fa",
    "fa-latn",
    "fi",
    "fr-be",
    "fr-ch",
    "fr-fr",
    "ga",
    "gd",
    "gn",
    "grc",
    "gu",
    "hak",
    "hi",
    "hr",
    "ht",
    "hu",
    "hy",
    "hyw",
    "ia",
    "id",
    "is",
    "it",
    "ja",
    "jbo",
    "ka",
    "kk",
    "kl",
    "kn",
    "ko",
    "kok",
    "ku",
    "ky",
    "la",
    "lfn",
    "lt",
    "lv",
    "mi",
    "mk",
    "ml",
    "mr",
    "ms",
    "mt",
    "my",
    "nb",
    "nci",
    "ne",
    "nl",
    "om",
    "or",
    "pa",
    "pap",
    "pl",
    "pt",
    "pt-br",
    "py",
    "quc",
    "ro",
    "ru",
    "ru-lv",
    "sd",
    "shn",
    "si",
    "sk",
    "sl",
    "sq",
    "sr",
    "sv",
    "sw",
    "ta",
    "te",
    "tn",
    "tr",
    "tt",
    "ur",
    "uz",
    "vi",
    "vi-vn-x-central",
    "vi-vn-x-south",
    "yue",
)
LANGUAGE_CODE_TO_ID = {language: index for index, language in enumerate(SUPPORTED_LANGUAGE_CODES)}

DEFAULT_EMOTION = (
    0.3077,
    0.0256,
    0.0256,
    0.0256,
    0.0256,
    0.0256,
    0.2564,
    0.3077,
)


@runtime_checkable
class ZonosPhonemeFrontend(Protocol):
    """Language-aware raw-text to eSpeak-compatible phoneme boundary."""

    frontend_id: str

    def phonemize(self, text: str, *, language: str) -> str:
        """Return one non-empty string in the published Zonos symbol set."""


class CallableZonosPhonemeFrontend:
    """Adapt a plain callable to :class:`ZonosPhonemeFrontend`."""

    def __init__(
        self,
        function: Callable[..., str],
        *,
        frontend_id: str = "caller-supplied",
    ) -> None:
        if not callable(function):
            raise TypeError("Zonos phoneme frontend must be callable.")
        if not isinstance(frontend_id, str) or not frontend_id.strip():
            raise ValueError("`frontend_id` must be a non-empty string.")
        self.function = function
        self.frontend_id = frontend_id.strip()

    def phonemize(self, text: str, *, language: str) -> str:
        try:
            value = self.function(text, language=language)
        except TypeError:
            value = self.function(text, language)
        return validate_phonemes(value)


class PrecomputedPhonemeFrontend:
    """Explicitly interpret the public text field as precomputed phonemes."""

    frontend_id = "precomputed-phonemes"

    def phonemize(self, text: str, *, language: str) -> str:
        del language
        return validate_phonemes(text)


def normalize_language_code(language: str) -> str:
    if not isinstance(language, str) or not language.strip():
        raise ValueError("Zonos `language` must be a non-empty string.")
    normalized = language.strip().lower().replace("_", "-")
    if normalized not in LANGUAGE_CODE_TO_ID:
        raise ValueError(
            f"Unsupported Zonos language {language!r}. Supported language "
            f"codes: {', '.join(SUPPORTED_LANGUAGE_CODES)}.")
    return normalized


def validate_phonemes(phonemes: str) -> str:
    if not isinstance(phonemes, str):
        raise TypeError("Zonos phonemes must be a string.")
    normalized = phonemes.strip()
    if not normalized:
        raise ValueError("Zonos phonemes cannot be empty.")
    unsupported = sorted(set(normalized) - set(PHONEME_SYMBOL_TO_ID))
    if unsupported:
        raise ValueError(
            "Zonos phonemes contain symbols outside the published "
            f"vocabulary: {unsupported!r}.")
    return normalized


def resolve_phonemes(
    text: str,
    *,
    language: str,
    phonemes: str | None = None,
    frontend: ZonosPhonemeFrontend | None = None,
) -> tuple[str, str]:
    """Resolve one explicit, checkpoint-compatible phoneme sequence."""
    normalized_language = normalize_language_code(language)
    if phonemes is not None:
        return validate_phonemes(phonemes), "precomputed-phonemes"
    if frontend is None:
        raise RuntimeError(
            "Zonos v0.1 requires eSpeak-compatible phonemes. Pass "
            "`phonemes=...`, configure `PrecomputedPhonemeFrontend` when the "
            "text argument already contains phonemes, or inject a "
            "`ZonosPhonemeFrontend`. VoiceHub does not silently substitute "
            "raw graphemes for the checkpoint's phoneme input.")
    if not isinstance(frontend, ZonosPhonemeFrontend):
        raise TypeError("Zonos `frontend` must implement phonemize(text, *, language).")
    return (
        validate_phonemes(frontend.phonemize(text, language=normalized_language), ),
        frontend.frontend_id,
    )


def tokenize_phonemes(
    phonemes: str,
    *,
    device: torch.device | str | None = None,
) -> Tensor:
    """Tokenize one sequence with the published BOS/EOS convention."""
    value = validate_phonemes(phonemes)
    ids = [
        BOS_ID,
        *(PHONEME_SYMBOL_TO_ID[symbol] for symbol in value),
        EOS_ID,
    ]
    return torch.tensor(ids, dtype=torch.long, device=device)


def batch_phoneme_ids(
    phonemes: Sequence[str],
    *,
    device: torch.device | str | None = None,
) -> tuple[Tensor, Tensor]:
    """Left-pad phoneme IDs exactly like the released Zonos frontend."""
    if isinstance(phonemes, (str, bytes)) or not isinstance(
            phonemes,
            Sequence,
    ):
        raise TypeError("Zonos phoneme batch must be a sequence of strings.")
    if not phonemes:
        raise ValueError("Zonos phoneme batch cannot be empty.")
    rows = [tokenize_phonemes(value) for value in phonemes]
    lengths = torch.tensor(
        [row.numel() for row in rows],
        dtype=torch.long,
        device=device,
    )
    longest = int(lengths.max().item())
    batch = torch.full(
        (len(rows), longest),
        PAD_ID,
        dtype=torch.long,
        device=device,
    )
    for index, row in enumerate(rows):
        batch[index, -row.numel():] = row.to(device=device)
    return batch, lengths


def _phoneme_id_batch(
    value: Tensor | Sequence[Sequence[int]] | Sequence[int],
    *,
    device: torch.device | str,
) -> Tensor:
    if isinstance(value, Tensor):
        ids = value
    else:
        ids = torch.as_tensor(value)
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    if ids.ndim != 2:
        raise ValueError("Zonos phoneme IDs must have shape [batch, time].")
    if ids.dtype == torch.bool or ids.is_floating_point():
        raise TypeError("Zonos phoneme IDs must use an integer dtype.")
    ids = ids.to(device=device, dtype=torch.long)
    if ids.shape[-1] == 0:
        raise ValueError("Zonos phoneme IDs cannot have an empty time axis.")
    if bool(((ids < PAD_ID) | (ids >= len(SPECIAL_TOKEN_IDS) + len(PHONEME_SYMBOLS))).any()):
        raise ValueError("Zonos phoneme IDs are outside the published vocabulary.")
    return ids


def _batch_languages(
    language: str | Sequence[str],
    *,
    batch_size: int,
) -> tuple[str, ...]:
    if isinstance(language, str):
        values = (normalize_language_code(language), ) * batch_size
    else:
        if isinstance(language, bytes) or not isinstance(language, Sequence):
            raise TypeError("Zonos language must be a string or string sequence.")
        values = tuple(normalize_language_code(item) for item in language)
        if len(values) != batch_size:
            raise ValueError("Zonos language batch size must match phoneme batch size.")
    return values


def _batch_float_feature(
    value: float | Sequence[float] | Tensor,
    *,
    name: str,
    batch_size: int,
    width: int,
    minimum: float,
    maximum: float,
    device: torch.device | str,
) -> Tensor:
    tensor = value if isinstance(value, Tensor) else torch.as_tensor(value)
    tensor = tensor.to(device=device, dtype=torch.float32)
    if tensor.ndim == 0:
        tensor = tensor.reshape(1, 1, 1).expand(batch_size, 1, width)
    elif tensor.ndim == 1:
        if width == 1 and tensor.numel() == batch_size:
            tensor = tensor.reshape(batch_size, 1, 1)
        elif tensor.numel() == width:
            tensor = tensor.reshape(1, 1, width).expand(batch_size, -1, -1)
        else:
            raise ValueError(f"Zonos `{name}` cannot be broadcast to "
                             f"[{batch_size}, 1, {width}].")
    elif tensor.ndim == 2:
        if tensor.shape == (batch_size, width):
            tensor = tensor.unsqueeze(1)
        elif tensor.shape == (1, width):
            tensor = tensor.unsqueeze(1).expand(batch_size, -1, -1)
        else:
            raise ValueError(f"Zonos `{name}` must have shape [batch, {width}].")
    elif tensor.ndim == 3:
        if tensor.shape[1:] != (1, width) or tensor.shape[0] not in {
                1,
                batch_size,
        }:
            raise ValueError(f"Zonos `{name}` must have shape [batch, 1, {width}].")
        tensor = tensor.expand(batch_size, -1, -1)
    else:
        raise ValueError(f"Zonos `{name}` must have at most three dimensions.")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"Zonos `{name}` must contain finite values.")
    if bool(((tensor < minimum) | (tensor > maximum)).any()):
        raise ValueError(f"Zonos `{name}` values must be in [{minimum:g}, {maximum:g}].")
    return tensor


def _speaker_feature(
    speaker_embedding: Tensor | None,
    *,
    batch_size: int,
    device: torch.device | str,
) -> Tensor | None:
    if speaker_embedding is None:
        return None
    if not isinstance(speaker_embedding, Tensor):
        raise TypeError("Zonos speaker embedding must be a PyTorch tensor.")
    speaker = speaker_embedding
    if speaker.ndim == 1:
        speaker = speaker.reshape(1, 1, -1)
    elif speaker.ndim == 2:
        speaker = speaker.unsqueeze(1)
    if speaker.ndim != 3 or speaker.shape[1:] != (1, 128):
        raise ValueError(
            "Zonos speaker embedding must have shape [128], [batch, 128], "
            "or [batch, 1, 128].")
    if speaker.shape[0] not in {1, batch_size}:
        raise ValueError("Zonos speaker embedding batch size must be one or match the "
                         "phoneme batch.")
    speaker = speaker.to(device=device, dtype=torch.float32)
    if not bool(torch.isfinite(speaker).all()):
        raise ValueError("Zonos speaker embedding must contain finite values.")
    return speaker.expand(batch_size, -1, -1)


def make_condition_dict(
    phoneme_ids: Tensor | Sequence[Sequence[int]] | Sequence[int],
    *,
    language: str | Sequence[str] = "en-us",
    speaker_embedding: Tensor | None = None,
    emotion: Tensor | Sequence[float] = DEFAULT_EMOTION,
    fmax: float | Sequence[float] | Tensor = 22_050.0,
    pitch_std: float | Sequence[float] | Tensor = 20.0,
    speaking_rate: float | Sequence[float] | Tensor = 15.0,
    device: torch.device | str = "cpu",
) -> dict[str, Tensor | None]:
    """Build source-shaped conditioning without a third-party G2P runtime."""
    ids = _phoneme_id_batch(phoneme_ids, device=device)
    batch_size = ids.shape[0]
    languages = _batch_languages(language, batch_size=batch_size)
    emotion_tensor = _batch_float_feature(
        emotion,
        name="emotion",
        batch_size=batch_size,
        width=8,
        minimum=0.0,
        maximum=1.0,
        device=device,
    )
    emotion_sum = emotion_tensor.sum(dim=-1, keepdim=True)
    if bool((emotion_sum <= 0).any()):
        raise ValueError("Every Zonos emotion vector must contain a positive value.")
    emotion_tensor = emotion_tensor / emotion_sum
    return {
        "espeak":
        ids,
        "speaker":
        _speaker_feature(
            speaker_embedding,
            batch_size=batch_size,
            device=device,
        ),
        "emotion":
        emotion_tensor,
        "fmax":
        _batch_float_feature(
            fmax,
            name="fmax",
            batch_size=batch_size,
            width=1,
            minimum=0.0,
            maximum=24_000.0,
            device=device,
        ),
        "pitch_std":
        _batch_float_feature(
            pitch_std,
            name="pitch_std",
            batch_size=batch_size,
            width=1,
            minimum=0.0,
            maximum=400.0,
            device=device,
        ),
        "speaking_rate":
        _batch_float_feature(
            speaking_rate,
            name="speaking_rate",
            batch_size=batch_size,
            width=1,
            minimum=0.0,
            maximum=40.0,
            device=device,
        ),
        "language_id":
        torch.tensor(
            [LANGUAGE_CODE_TO_ID[item] for item in languages],
            dtype=torch.long,
            device=device,
        ).reshape(batch_size, 1, 1),
    }


__all__ = [
    "BOS_ID",
    "CallableZonosPhonemeFrontend",
    "DEFAULT_EMOTION",
    "EOS_ID",
    "IPA_LETTERS",
    "LANGUAGE_CODE_TO_ID",
    "LETTERS",
    "PAD_ID",
    "PHONEME_SYMBOLS",
    "PHONEME_SYMBOL_TO_ID",
    "PUNCTUATION",
    "PrecomputedPhonemeFrontend",
    "SPECIAL_TOKEN_IDS",
    "SUPPORTED_LANGUAGE_CODES",
    "UNK_ID",
    "ZonosPhonemeFrontend",
    "batch_phoneme_ids",
    "make_condition_dict",
    "normalize_language_code",
    "resolve_phonemes",
    "tokenize_phonemes",
    "validate_phonemes",
]
