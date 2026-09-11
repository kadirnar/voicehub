"""Dependency-free Kokoro phoneme pipeline and voice-pack loader."""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass
from inspect import signature
from pathlib import Path
from typing import Any

import torch

from voicehub.hub import resolve_pretrained_file
from voicehub.models.kokoro.native.checkpoint import (
    KOKORO_CHECKPOINT_REVISION,
    import_legacy_kokoro_checkpoint,
    import_legacy_kokoro_voice,
    load_native_kokoro_checkpoint,
    load_native_kokoro_voice,
)

from .artifacts import KokoroArtifacts, resolve_kokoro_artifacts
from .model import KModel

_LOGGER = logging.getLogger(__name__)

ALIASES = {
    "en-us": "a",
    "en-gb": "b",
    "es": "e",
    "fr-fr": "f",
    "hi": "h",
    "it": "i",
    "pt-br": "p",
    "ja": "j",
    "zh": "z",
}

LANG_CODES = {
    "a": "American English",
    "b": "British English",
    "e": "Spanish",
    "f": "French",
    "h": "Hindi",
    "i": "Italian",
    "p": "Brazilian Portuguese",
    "j": "Japanese",
    "z": "Mandarin Chinese",
}

_PUNCTUATION_TRANSLATION = str.maketrans({
    "\u00a0": " ",
    "\t": " ",
    "\r": "\n",
    "–": "—",
    "―": "—",
    "‘": "'",
    "’": "'",
    "„": '"',
})


class KokoroFrontendError(ValueError):
    """Raised when a text frontend cannot produce supported phonemes."""


class GraphemeFallbackFrontend:
    """Minimal built-in text normalizer with explicit non-parity semantics.

    The Kokoro repository delegates G2P to Misaki/espeak and does not
    release its linguistic tables. VoiceHub therefore cannot reproduce
    multilingual G2P without another runtime. This fallback only
    normalizes Unicode, lowercases Latin text, and retains characters
    present in the released phoneme vocabulary. It keeps raw-text
    inference usable, but callers that need source-quality pronunciation
    should pass phonemes or a frontend callable.
    """

    frontend_id = "voicehub-grapheme-fallback-v1"
    source_equivalent = False

    def __init__(self, vocabulary: dict[str, int]) -> None:
        self.vocabulary = frozenset(vocabulary)

    def __call__(self, text: str, *, language_code: str) -> str:
        del language_code
        if not isinstance(text, str) or not text.strip():
            raise KokoroFrontendError("Kokoro text must be non-empty.")
        normalized = unicodedata.normalize(
            "NFKC",
            text.translate(_PUNCTUATION_TRANSLATION),
        ).lower()
        output = []
        previous_space = False
        for character in normalized:
            if character.isspace():
                if output and not previous_space and " " in self.vocabulary:
                    output.append(" ")
                previous_space = True
                continue
            previous_space = False
            if character in self.vocabulary:
                output.append(character)
        phonemes = "".join(output).strip()
        if not phonemes:
            raise KokoroFrontendError(
                "The built-in Kokoro fallback could not map this text to the "
                "released symbol vocabulary. Pass `phonemes=` or inject a "
                "frontend callable.")
        return phonemes


class PhonemeFrontend:
    """Validate caller-supplied phonemes without linguistic rewriting."""

    frontend_id = "caller-supplied-phonemes"
    source_equivalent = True

    def __init__(self, vocabulary: dict[str, int]) -> None:
        self.vocabulary = frozenset(vocabulary)

    def __call__(self, text: str, *, language_code: str) -> str:
        del language_code
        if not isinstance(text, str) or not text:
            raise KokoroFrontendError("Kokoro phonemes must be non-empty.")
        unknown = sorted(set(text) - self.vocabulary)
        if unknown:
            display = ", ".join(repr(item) for item in unknown)
            raise KokoroFrontendError("Kokoro phonemes contain unsupported symbols: "
                                      f"{display}.")
        return text


def _call_frontend(
    frontend: Callable[..., str],
    text: str,
    *,
    language_code: str,
) -> str:
    try:
        accepts_language = True
        signature(frontend).bind(text, language_code=language_code)
    except TypeError:
        accepts_language = False
    except (ValueError, AttributeError):
        # Some extension callables do not expose a Python signature. Invoke
        # their documented two-argument form once so an internal TypeError is
        # never hidden by an accidental retry.
        accepts_language = True
    if accepts_language:
        output = frontend(text, language_code=language_code)
    else:
        output = frontend(text)
    if not isinstance(output, str) or not output:
        raise KokoroFrontendError("Kokoro frontend callables must return a non-empty phoneme string.")
    return output


def _split_text(text: str, pattern: str | None) -> list[str]:
    if pattern is None:
        return [text]
    return [segment for segment in re.split(pattern, text.strip()) if segment.strip()]


def _segments(
    value: str | Sequence[str],
    *,
    split_pattern: str | None,
    name: str,
) -> list[str]:
    if isinstance(value, str):
        segments = _split_text(value, split_pattern)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        segments = list(value)
    else:
        raise TypeError(f"Kokoro {name} must be a string or string sequence.")
    if not segments or any(not isinstance(segment, str) or not segment for segment in segments):
        raise ValueError(f"Kokoro {name} segments must be non-empty strings.")
    return segments


class KPipeline:
    """Language/front-end orchestration around one native :class:`KModel`."""

    def __init__(
        self,
        lang_code: str,
        repo_id: str | Path = "hexgrad/Kokoro-82M",
        model: KModel | bool = True,
        *,
        frontend: Callable[..., str] | None = None,
        device: str | torch.device = "cpu",
        checkpoint_filename: str | None = None,
        revision: str | None = None,
        cache_dir: str | None = None,
        token: str | bool | None = None,
        local_files_only: bool = False,
        allow_legacy_checkpoint_conversion: bool = False,
    ) -> None:
        normalized_language = ALIASES.get(
            lang_code.lower(),
            lang_code.lower(),
        )
        if normalized_language not in LANG_CODES:
            choices = ", ".join(sorted(LANG_CODES))
            raise ValueError(f"Unknown Kokoro language code {lang_code!r}; choose {choices}.")
        self.lang_code = normalized_language
        self.repo_id = repo_id
        self.device = torch.device(device)
        self.cache_dir = cache_dir
        self.revision = revision
        self.token = token
        self.local_files_only = local_files_only
        self.allow_legacy_checkpoint_conversion = (allow_legacy_checkpoint_conversion)
        self.artifacts: KokoroArtifacts | None = None
        self.voices: dict[str, torch.Tensor] = {}

        if isinstance(model, KModel):
            self.model = model
        elif model is False:
            self.model = None
        elif model is True:
            self.model = self._load_model(checkpoint_filename)
        else:
            raise TypeError("`model` must be a KModel or boolean.")
        vocabulary = (
            self.model.vocab
            if self.model is not None else self._resolve_artifacts(checkpoint_filename).config.vocab)
        self.phoneme_frontend = PhonemeFrontend(dict(vocabulary))
        self.frontend = (GraphemeFallbackFrontend(dict(vocabulary)) if frontend is None else frontend)

    def _resolve_artifacts(
        self,
        checkpoint_filename: str | None = None,
    ) -> KokoroArtifacts:
        if self.artifacts is None:
            self.artifacts = resolve_kokoro_artifacts(
                self.repo_id,
                checkpoint_filename=checkpoint_filename,
                revision=self.revision,
                cache_dir=self.cache_dir,
                token=self.token,
                local_files_only=self.local_files_only,
            )
            self.revision = self.artifacts.revision
        return self.artifacts

    def _load_model(self, checkpoint_filename: str | None) -> KModel:
        artifacts = self._resolve_artifacts(checkpoint_filename)
        model = KModel(artifacts.config)
        checkpoint = artifacts.checkpoint
        if artifacts.legacy_pytorch:
            native_checkpoint = checkpoint.with_suffix(".voicehub.safetensors")
            if not native_checkpoint.is_file():
                if (not artifacts.official_legacy_checkpoint and not self.allow_legacy_checkpoint_conversion):
                    raise ValueError(
                        "A local/custom Kokoro .pth requires explicit "
                        "`allow_legacy_checkpoint_conversion=True`. The "
                        "restricted weights-only importer will write a "
                        "portable Safetensors file.")
                # Use a meta graph for inventory validation so one-time
                # conversion does not duplicate the initialized 82M graph.
                with torch.device("meta"):
                    shape_model = KModel(artifacts.config)
                import_legacy_kokoro_checkpoint(
                    shape_model,
                    checkpoint,
                    output_path=native_checkpoint,
                    verify_official_hash=(artifacts.official_legacy_checkpoint),
                )
            checkpoint = native_checkpoint
        load_native_kokoro_checkpoint(
            model,
            checkpoint,
            device=self.device,
        )
        return model.to(self.device).eval()

    def _voice_paths(self, voice: str) -> tuple[Path | None, Path | None]:
        explicit = Path(voice).expanduser()
        if explicit.is_file():
            if explicit.suffix == ".safetensors":
                return explicit.resolve(), None
            if explicit.suffix == ".pt":
                return None, explicit.resolve()
            raise ValueError("Explicit Kokoro voice files must use .safetensors or .pt.")
        source = Path(self.repo_id).expanduser()
        if source.is_dir():
            native_candidates = (
                source / "voices" / f"{voice}.safetensors",
                source / "voices" / f"{voice}.voicehub.safetensors",
            )
            for candidate in native_candidates:
                if candidate.is_file():
                    return candidate.resolve(), None
            legacy = source / "voices" / f"{voice}.pt"
            if legacy.is_file():
                return None, legacy.resolve()
            raise FileNotFoundError(f"Kokoro voice {voice!r} was not found under "
                                    f"{source / 'voices'}.")
        legacy = resolve_pretrained_file(
            self.repo_id,
            f"voices/{voice}.pt",
            cache_dir=self.cache_dir,
            revision=self.revision,
            token=self.token,
            local_files_only=self.local_files_only,
        )
        return None, legacy

    def load_single_voice(self, voice: str) -> torch.Tensor:
        if not isinstance(voice, str) or not voice.strip():
            raise ValueError("Kokoro voice names must be non-empty.")
        voice = voice.strip()
        if voice in self.voices:
            return self.voices[voice]
        native_path, legacy_path = self._voice_paths(voice)
        if native_path is None:
            if legacy_path is None:  # pragma: no cover - invariant
                raise RuntimeError("Kokoro voice resolver returned no file.")
            native_path = legacy_path.with_suffix(".voicehub.safetensors")
            if not native_path.is_file():
                source_is_official = (
                    str(self.repo_id) in KModel.MODEL_NAMES and self.revision == KOKORO_CHECKPOINT_REVISION)
                if (not source_is_official and not self.allow_legacy_checkpoint_conversion):
                    raise ValueError(
                        "A local/custom Kokoro voice .pt requires explicit "
                        "`allow_legacy_checkpoint_conversion=True`.")
                import_legacy_kokoro_voice(
                    legacy_path,
                    output_path=native_path,
                )
        value = load_native_kokoro_voice(
            native_path,
            device=self.device,
            dtype=(self.model.dtype if self.model is not None else torch.float32),
        )
        self.voices[voice] = value
        return value

    def load_voice(
        self,
        voice: str | torch.Tensor,
        delimiter: str = ",",
    ) -> torch.Tensor:
        """Load or average one or more validated voice style tables."""
        if torch.is_tensor(voice):
            if (voice.ndim != 3 or voice.shape[1:] != (1, 256) or voice.shape[0] < 1):
                raise ValueError("Kokoro tensor voices must have shape [length, 1, 256].")
            return voice.to(
                device=self.device,
                dtype=(self.model.dtype if self.model is not None else torch.float32),
            )
        if not isinstance(voice, str) or not voice.strip():
            raise ValueError("Kokoro `voice` must be a name or style tensor.")
        if voice in self.voices:
            return self.voices[voice]
        names = [item.strip() for item in voice.split(delimiter)]
        if not names or any(not item for item in names):
            raise ValueError("Kokoro voice mixtures contain an empty name.")
        packs = [self.load_single_voice(name) for name in names]
        lengths = {pack.shape[0] for pack in packs}
        if len(lengths) != 1:
            raise ValueError("Kokoro voice packs must have equal length before mixing.")
        mixed = packs[0] if len(packs) == 1 else torch.stack(packs).mean(0)
        self.voices[voice] = mixed
        return mixed

    @staticmethod
    def _style_for_phonemes(
        model: KModel,
        phonemes: str,
        pack: torch.Tensor,
    ) -> torch.Tensor:
        token_count = len(model.tokenize_phonemes(phonemes))
        index = token_count - 1
        if index >= pack.shape[0]:
            raise ValueError(
                "Kokoro voice pack does not cover this phoneme sequence "
                f"length ({token_count} > {pack.shape[0]}).")
        return pack[index]

    @staticmethod
    def infer(
        model: KModel,
        phonemes: str,
        pack: torch.Tensor,
        speed: float | Callable[[int], float] = 1.0,
    ) -> KModel.Output:
        resolved_speed = speed(len(phonemes)) if callable(speed) else speed
        style = KPipeline._style_for_phonemes(model, phonemes, pack)
        return model(
            phonemes,
            style,
            float(resolved_speed),
            return_output=True,
        )

    @dataclass
    class Result:
        """One normalized text/phoneme/audio segment."""

        graphemes: str
        phonemes: str
        output: KModel.Output | None = None
        text_index: int | None = None
        frontend_id: str | None = None

        @property
        def audio(self) -> torch.Tensor | None:
            return None if self.output is None else self.output.audio

        @property
        def pred_dur(self) -> torch.Tensor | None:
            return None if self.output is None else self.output.pred_dur

        def __iter__(self):
            yield self.graphemes
            yield self.phonemes
            yield self.audio

        def __getitem__(self, index: int):
            return [self.graphemes, self.phonemes, self.audio][index]

        def __len__(self) -> int:
            return 3

    def _chunks(self, phonemes: str) -> list[str]:
        if self.model is None:
            limit = 510
        else:
            limit = self.model.context_length - 2
        if len(phonemes) <= limit:
            return [phonemes]
        chunks: list[str] = []
        current = ""
        for item in re.split(r"(?<=[.!?…])\s+|\s+", phonemes):
            if not item:
                continue
            candidate = f"{current} {item}".strip()
            if len(candidate) <= limit:
                current = candidate
                continue
            if current:
                chunks.append(current)
            while len(item) > limit:
                chunks.append(item[:limit])
                item = item[limit:]
            current = item
        if current:
            chunks.append(current)
        return chunks

    def generate_from_tokens(
        self,
        tokens: str | Sequence[Any],
        voice: str | torch.Tensor,
        speed: float | Callable[[int], float] = 1.0,
        model: KModel | None = None,
    ) -> Generator[Result]:
        """Generate from a phoneme string.

        Legacy token-object lists depended on Misaki classes and are
        rejected with a precise migration message.
        """
        if not isinstance(tokens, str):
            raise TypeError(
                "Native Kokoro accepts a phoneme string here; Misaki token "
                "objects are not part of the VoiceHub runtime.")
        runtime = model or self.model
        if runtime is None:
            raise RuntimeError("Kokoro generation requires a loaded KModel.")
        phonemes = _call_frontend(
            self.phoneme_frontend,
            tokens,
            language_code=self.lang_code,
        )
        pack = self.load_voice(voice)
        for chunk in self._chunks(phonemes):
            output = self.infer(runtime, chunk, pack, speed)
            yield self.Result(
                graphemes="",
                phonemes=chunk,
                output=output,
                frontend_id=self.phoneme_frontend.frontend_id,
            )

    def __call__(
        self,
        text: str | Sequence[str],
        voice: str | torch.Tensor | None = None,
        speed: float | Callable[[int], float] = 1.0,
        split_pattern: str | None = r"\n+",
        model: KModel | None = None,
        *,
        phonemes: str | Sequence[str] | None = None,
    ) -> Generator[Result]:
        runtime = model or self.model
        if runtime is not None and voice is None:
            raise ValueError("Kokoro generation requires `voice`.")
        grapheme_segments = _segments(
            text,
            split_pattern=split_pattern,
            name="text",
        )
        if phonemes is None:
            frontend = self.frontend
            frontend_segments = grapheme_segments
        else:
            frontend = self.phoneme_frontend
            frontend_segments = _segments(
                phonemes,
                split_pattern=split_pattern,
                name="phoneme",
            )
            if len(frontend_segments) != len(grapheme_segments):
                raise ValueError(
                    "Kokoro explicit phoneme segments must match the number "
                    "of text segments.")
        pack = self.load_voice(voice) if runtime is not None else None
        frontend_id = getattr(
            frontend,
            "frontend_id",
            frontend.__class__.__name__,
        )
        for text_index, (graphemes, frontend_input) in enumerate(zip(grapheme_segments, frontend_segments)):
            normalized = _call_frontend(
                frontend,
                frontend_input,
                language_code=self.lang_code,
            )
            for chunk in self._chunks(normalized):
                output = (self.infer(runtime, chunk, pack, speed) if runtime is not None else None)
                yield self.Result(
                    graphemes=graphemes,
                    phonemes=chunk,
                    output=output,
                    text_index=text_index,
                    frontend_id=frontend_id,
                )


__all__ = [
    "ALIASES",
    "LANG_CODES",
    "GraphemeFallbackFrontend",
    "KPipeline",
    "KokoroFrontendError",
    "PhonemeFrontend",
]
