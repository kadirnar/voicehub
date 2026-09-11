"""Prepared-input boundary for native GPT-SoVITS classic S2 variants.

The released pipeline delegates multilingual normalization/phonemization
to language-specific packages and derives 1,024-dimensional features
from a separate Chinese RoBERTa checkpoint.  VoiceHub does not
approximate either operation: callers provide the exact IDs/features
produced for their training corpus, or preprocessing fails closed.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from voicehub.models.gptsovits.native.configuration import GPTSoVITSS1Config, GPTSoVITSS2Config


class GPTSoVITSFrontendError(ValueError):
    """Raised when raw or checkpoint-incompatible frontend input is
    supplied."""


def reject_raw_text(text: str) -> None:
    if not isinstance(text, str):
        raise TypeError("GPT-SoVITS text must be a string.")
    raise GPTSoVITSFrontendError(
        "Native GPT-SoVITS does not guess multilingual normalization, "
        "phonemes, or Chinese-RoBERTa features. Supply prepared S1 phoneme "
        "IDs and 1,024-dimensional BERT features plus target-only S2 phoneme "
        "IDs. This preserves checkpoint accuracy instead of substituting an "
        "incompatible frontend.")


def validate_prepared_inference(
    *,
    s1_phoneme_ids: Any,
    s1_bert_features: Any,
    s2_phoneme_ids: Any,
    prompt_semantic_ids: Any | None,
    reference_spectrogram: Any,
    speaker_embedding: Any | None = None,
    semantic_codes: Any | None = None,
    s1_config: GPTSoVITSS1Config | None = None,
    s2_config: GPTSoVITSS2Config | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> dict[str, Tensor | None]:
    s1_config = s1_config or GPTSoVITSS1Config()
    s2_config = s2_config or GPTSoVITSS2Config()
    s1_ids = torch.as_tensor(s1_phoneme_ids, device=device, dtype=torch.long)
    s2_ids = torch.as_tensor(s2_phoneme_ids, device=device, dtype=torch.long)
    bert = torch.as_tensor(s1_bert_features, device=device, dtype=dtype)
    reference = torch.as_tensor(
        reference_spectrogram,
        device=device,
        dtype=dtype,
    )
    if s1_ids.ndim == 1:
        s1_ids = s1_ids.unsqueeze(0)
    if s2_ids.ndim == 1:
        s2_ids = s2_ids.unsqueeze(0)
    if s1_ids.ndim != 2 or s1_ids.shape[0] != 1:
        raise GPTSoVITSFrontendError("S1 phoneme IDs must have shape [1, time].")
    if s2_ids.ndim != 2 or s2_ids.shape[0] != 1:
        raise GPTSoVITSFrontendError("S2 phoneme IDs must have shape [1, time].")
    if bert.shape != (
            1,
            s1_config.bert_feature_dim,
            s1_ids.shape[1],
    ):
        raise GPTSoVITSFrontendError(
            "S1 BERT features must have shape "
            f"[1, {s1_config.bert_feature_dim}, s1_phoneme_time].")
    if reference.ndim == 2:
        reference = reference.unsqueeze(0)
    if reference.ndim != 3 or reference.shape[:2] != (
            1,
            s2_config.spectrogram_channels,
    ):
        raise GPTSoVITSFrontendError(
            "Reference spectrogram must have shape "
            f"[1, {s2_config.spectrogram_channels}, frames].")
    if reference.shape[2] < 1:
        raise GPTSoVITSFrontendError("Reference spectrogram cannot be empty.")
    for name, ids, vocabulary in (
        ("S1", s1_ids, s1_config.phoneme_vocabulary_size),
        ("S2", s2_ids, s2_config.phoneme_vocabulary_size),
    ):
        if bool(((ids < 0) | (ids >= vocabulary)).any()):
            raise GPTSoVITSFrontendError(
                f"{name} phoneme IDs are outside the "
                f"{s2_config.version} checkpoint vocabulary.")
    prepared_speaker = None
    if s2_config.requires_speaker_embedding:
        if speaker_embedding is None:
            raise GPTSoVITSFrontendError(
                f"GPT-SoVITS {s2_config.version} requires a prepared "
                f"{s2_config.speaker_embedding_dim}-dimensional speaker embedding.")
        prepared_speaker = torch.as_tensor(
            speaker_embedding,
            device=device,
            dtype=dtype,
        )
        if prepared_speaker.ndim == 1:
            prepared_speaker = prepared_speaker.unsqueeze(0)
        expected = (1, s2_config.speaker_embedding_dim)
        if tuple(prepared_speaker.shape) != expected:
            raise GPTSoVITSFrontendError(f"Speaker embedding must have shape {expected}.")
    elif speaker_embedding is not None:
        raise GPTSoVITSFrontendError(f"GPT-SoVITS {s2_config.version} does not consume speaker embeddings.")
    prompt = None
    if prompt_semantic_ids is not None:
        prompt = torch.as_tensor(
            prompt_semantic_ids,
            device=device,
            dtype=torch.long,
        )
        if prompt.ndim == 1:
            prompt = prompt.unsqueeze(0)
        if prompt.ndim != 2 or prompt.shape[0] != 1:
            raise GPTSoVITSFrontendError("Prompt semantic IDs must have shape [1, time].")
        if bool(((prompt < 0) | (prompt >= s1_config.eos_token_id)).any()):
            raise GPTSoVITSFrontendError("Prompt semantic IDs are outside the codebook.")
    prepared_codes = None
    if semantic_codes is not None:
        prepared_codes = torch.as_tensor(
            semantic_codes,
            device=device,
            dtype=torch.long,
        )
        if prepared_codes.ndim == 1:
            prepared_codes = prepared_codes.view(1, 1, -1)
        elif prepared_codes.ndim == 2:
            prepared_codes = prepared_codes.unsqueeze(0)
        if prepared_codes.ndim != 3 or prepared_codes.shape[:2] != (1, 1):
            raise GPTSoVITSFrontendError("Prepared semantic codes must have shape [1, 1, time].")
        if bool(((prepared_codes < 0) | (prepared_codes >= s1_config.eos_token_id)).any()):
            raise GPTSoVITSFrontendError("Prepared semantic codes are outside the codebook.")
    return {
        "s1_phoneme_ids": s1_ids,
        "s1_bert_features": bert,
        "s2_phoneme_ids": s2_ids,
        "prompt_semantic_ids": prompt,
        "reference_spectrogram": reference,
        "speaker_embedding": prepared_speaker,
        "semantic_codes": prepared_codes,
    }


__all__ = [
    "GPTSoVITSFrontendError",
    "reject_raw_text",
    "validate_prepared_inference",
]
