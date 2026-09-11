"""Explicit checkpoint-compatible feature boundary for native MeloTTS."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from voicehub.models.melotts.native.configuration import MeloTTSArchitectureConfig


@dataclass(frozen=True, slots=True)
class MeloTTSFeatureBatch:
    input_ids: Tensor
    input_lengths: Tensor
    tone_ids: Tensor
    language_ids: Tensor
    bert_features: Tensor
    ja_bert_features: Tensor
    speaker_ids: Tensor


class NativeMeloTTSFrontend:
    """Validate already prepared upstream-compatible linguistic features.

    MeloTTS checkpoints consume outputs from language-specific G2P and
    BERT models. Those pretrained frontends are not encoded in the
    acoustic checkpoint, so VoiceHub does not silently replace them.
    Callers provide the exact phone, tone, language, 1024-channel BERT,
    and 768-channel Japanese-BERT sequences used by their checkpoint.
    """

    def __init__(self, config: MeloTTSArchitectureConfig) -> None:
        if not isinstance(config, MeloTTSArchitectureConfig):
            raise TypeError("`config` must be a MeloTTSArchitectureConfig.")
        self.config = config

    @staticmethod
    def _integer_sequence(
        value: Any,
        *,
        name: str,
        device: torch.device | str | None,
    ) -> Tensor:
        tensor = value if isinstance(value, Tensor) else torch.as_tensor(value)
        if tensor.dtype == torch.bool or tensor.is_floating_point():
            raise TypeError(f"MeloTTS `{name}` must use integer dtype.")
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2 or tensor.shape[0] != 1 or tensor.shape[1] < 1:
            raise ValueError(f"MeloTTS `{name}` must have shape [text] or [1, text].")
        return tensor.to(device=device, dtype=torch.long)

    @staticmethod
    def _features(
        value: Any,
        *,
        name: str,
        channels: int,
        text_steps: int,
        device: torch.device | str | None,
        dtype: torch.dtype,
    ) -> Tensor:
        tensor = value if isinstance(value, Tensor) else torch.as_tensor(value)
        if tensor.is_complex() or tensor.dtype == torch.bool:
            raise TypeError(f"MeloTTS `{name}` must contain real features.")
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 3 or tuple(tensor.shape) != (
                1,
                channels,
                text_steps,
        ):
            raise ValueError(
                f"MeloTTS `{name}` must have shape "
                f"[{channels}, text] or [1, {channels}, text].")
        tensor = tensor.to(device=device, dtype=dtype)
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"MeloTTS `{name}` contains NaN or infinity.")
        return tensor

    def resolve_speaker(
        self,
        speaker: str | int | None,
    ) -> int:
        speakers = self.config.data.speakers
        if not speakers:
            if speaker is None and self.config.data.n_speakers == 1:
                return 0
            raise RuntimeError("The MeloTTS configuration does not declare public speaker IDs.")
        if speaker is None:
            return next(iter(speakers.values()))
        if isinstance(speaker, int) and not isinstance(speaker, bool):
            if speaker not in speakers.values():
                available = ", ".join(str(item) for item in speakers.values())
                raise ValueError(f"Unknown MeloTTS speaker ID {speaker}. "
                                 f"Available IDs: {available}.")
            return speaker
        if not isinstance(speaker, str) or not speaker.strip():
            raise TypeError("`speaker` must be a speaker name, integer ID, or None.")
        try:
            return speakers[speaker]
        except KeyError as error:
            available = ", ".join(speakers)
            raise ValueError(
                f"Unknown MeloTTS speaker {speaker!r}. "
                f"Available speakers: {available}.") from error

    def prepare(
        self,
        *,
        input_ids: Tensor | Sequence[int] | Sequence[Sequence[int]],
        tone_ids: Tensor | Sequence[int] | Sequence[Sequence[int]],
        language_ids: Tensor | Sequence[int] | Sequence[Sequence[int]],
        bert_features: Any,
        ja_bert_features: Any,
        speaker: str | int | None,
        device: torch.device | str | None,
        dtype: torch.dtype,
    ) -> MeloTTSFeatureBatch:
        input_ids = self._integer_sequence(
            input_ids,
            name="input_ids",
            device=device,
        )
        tone_ids = self._integer_sequence(
            tone_ids,
            name="tone_ids",
            device=device,
        )
        language_ids = self._integer_sequence(
            language_ids,
            name="language_ids",
            device=device,
        )
        if tone_ids.shape != input_ids.shape:
            raise ValueError("MeloTTS tone IDs must align with input IDs.")
        if language_ids.shape != input_ids.shape:
            raise ValueError("MeloTTS language IDs must align with input IDs.")
        if bool(((input_ids < 0) | (input_ids >= self.config.vocab_size)).any()):
            raise ValueError("MeloTTS input IDs are outside the checkpoint vocabulary.")
        if bool(((tone_ids < 0) | (tone_ids >= self.config.num_tones)).any()):
            raise ValueError("MeloTTS tone IDs are outside the checkpoint inventory.")
        if bool(((language_ids < 0) | (language_ids >= self.config.num_languages)).any()):
            raise ValueError("MeloTTS language IDs are outside the checkpoint inventory.")
        text_steps = input_ids.shape[1]
        bert_features = self._features(
            bert_features,
            name="bert_features",
            channels=1024,
            text_steps=text_steps,
            device=device,
            dtype=dtype,
        )
        ja_bert_features = self._features(
            ja_bert_features,
            name="ja_bert_features",
            channels=768,
            text_steps=text_steps,
            device=device,
            dtype=dtype,
        )
        speaker_id = self.resolve_speaker(speaker)
        return MeloTTSFeatureBatch(
            input_ids=input_ids,
            input_lengths=torch.tensor(
                [text_steps],
                dtype=torch.long,
                device=device,
            ),
            tone_ids=tone_ids,
            language_ids=language_ids,
            bert_features=bert_features,
            ja_bert_features=ja_bert_features,
            speaker_ids=torch.tensor(
                [speaker_id],
                dtype=torch.long,
                device=device,
            ),
        )


def validate_feature_mapping(values: Mapping[str, Any], ) -> None:
    """Fail before model loading when raw-text-only requests are attempted."""
    required = (
        "input_ids",
        "tone_ids",
        "language_ids",
        "bert_features",
        "ja_bert_features",
    )
    missing = [name for name in required if values.get(name) is None]
    if missing:
        raise ValueError(
            "Native MeloTTS requires checkpoint-compatible precomputed "
            "linguistic features. Missing: " + ", ".join(missing) + ".")


__all__ = [
    "MeloTTSFeatureBatch",
    "NativeMeloTTSFrontend",
    "validate_feature_mapping",
]
