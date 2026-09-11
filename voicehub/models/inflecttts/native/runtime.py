"""Request-local native inference runtime for Inflect v2."""

from __future__ import annotations

import re
from collections.abc import Sequence
from numbers import Real
from typing import Any

import torch
from torch import Tensor, nn

from voicehub.models.inflecttts.native.configuration import InflectV2Config
from voicehub.models.inflecttts.native.frontend import (
    batch_token_ids,
    phonemes_to_ids,
    require_preprocessed_phonemes,
    validate_token_ids,
)
from voicehub.models.inflecttts.native.modeling import SynthesizerTrn
from voicehub.models.vits.native.weight_norm import (
    LegacyWeightNormInferenceCache,
    enable_legacy_weight_norm_inference_cache,
)
from voicehub.optimization.protocols import OptimizationCompileTarget, OptimizationModuleRoot


def split_phoneme_text(text: str, limit: int = 280) -> list[str]:
    """Split long preprocessed input without discarding punctuation."""
    normalized = " ".join(text.split())
    if not normalized:
        raise ValueError("Inflect phoneme text cannot be empty.")
    sentences = [part.strip() for part in re.split(r"(?<=[.!?;:])\s+", normalized) if part.strip()]
    chunks: list[str] = []
    for sentence in sentences or [normalized]:
        while len(sentence) > limit:
            search = sentence[:limit + 1]
            punctuation = max(search.rfind(mark) for mark in (",", ";", ":"))
            split_at = (punctuation + 1 if punctuation >= limit // 2 else sentence.rfind(" ", 0, limit + 1))
            if split_at < limit // 2:
                split_at = limit
            chunks.append(sentence[:split_at].strip())
            sentence = sentence[split_at:].strip()
        if sentence:
            chunks.append(sentence)
    return chunks


def boundary_pause_seconds(chunk: str) -> float:
    ending = chunk.rstrip()[-1:] if chunk.strip() else ""
    return {
        "?": 0.28,
        "!": 0.24,
        ".": 0.22,
        ";": 0.16,
        ":": 0.13,
        ",": 0.09,
    }.get(ending, 0.08)


def edge_fade(
    waveform: Tensor,
    sample_rate: int,
    *,
    milliseconds: float = 5.0,
) -> Tensor:
    """Apply the published short linear edge fade using PyTorch only."""
    frames = min(
        round(sample_rate * milliseconds / 1000.0),
        waveform.numel() // 2,
    )
    if frames <= 0:
        return waveform
    output = waveform.clone()
    ramp = torch.linspace(
        0.0,
        1.0,
        frames,
        device=output.device,
        dtype=output.dtype,
    )
    output[:frames] *= ramp
    output[-frames:] *= ramp.flip(0)
    return output


class InflectV2Runtime(nn.Module):
    """Exact generator graph plus explicit checkpoint-native preprocessing."""

    def __init__(
        self,
        model: SynthesizerTrn,
        config: InflectV2Config,
    ) -> None:
        super().__init__()
        if not isinstance(model, SynthesizerTrn):
            raise TypeError("`model` must be a native SynthesizerTrn.")
        if not isinstance(config, InflectV2Config):
            raise TypeError("`config` must be an InflectV2Config.")
        self.generator = model
        self.config = config
        self._weight_norm_cache: LegacyWeightNormInferenceCache | None = None
        if not model.training:
            self._weight_norm_cache = (enable_legacy_weight_norm_inference_cache(model))

    def train(self, mode: bool = True) -> InflectV2Runtime:
        super().train(mode)
        if mode:
            if self._weight_norm_cache is not None:
                self._weight_norm_cache.restore()
                self._weight_norm_cache = None
        elif self._weight_norm_cache is None:
            self._weight_norm_cache = (enable_legacy_weight_norm_inference_cache(self.generator))
        return self

    @property
    def sample_rate(self) -> int:
        return self.config.sample_rate

    @property
    def device(self) -> torch.device:
        return next(self.generator.parameters()).device

    def optimization_module_roots(self):
        """Expose the checkpoint-native generator owned by this runtime."""
        return (OptimizationModuleRoot("generator", self.generator), )

    def optimization_compile_targets(self, mode: str):
        """Compile the generator boundary used by training or synthesis."""
        attribute = "forward" if mode == "training" else "infer"
        if mode not in {"inference", "training"}:
            raise ValueError(f"Unsupported optimization mode {mode!r}.")
        return (OptimizationCompileTarget(
            f"generator.{attribute}",
            self.generator,
            attribute,
        ), )

    @staticmethod
    def _controls(
        *,
        speed: object,
        variation: object,
        seed: object,
    ) -> tuple[float, float, int]:
        if (isinstance(speed, bool) or not isinstance(speed, Real) or not 0.5 <= float(speed) <= 2.0):
            raise ValueError("`speed` must be a finite number in [0.5, 2.0].")
        if (isinstance(variation, bool) or not isinstance(variation, Real) or
                not 0.0 <= float(variation) <= 1.0):
            raise ValueError("`variation` must be a finite number in [0.0, 1.0].")
        if (isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63):
            raise ValueError("`seed` must be an integer in [0, 2**63).")
        return float(speed), float(variation), seed

    def _infer_ids(
        self,
        input_ids: Sequence[int] | Tensor,
        *,
        speed: float,
        variation: float,
        seed: int,
    ) -> Tensor:
        item = validate_token_ids(
            input_ids,
            vocabulary_size=self.config.vocabulary_size,
        )
        batch = batch_token_ids(
            [item],
            vocabulary_size=self.config.vocabulary_size,
            device=self.device,
        )
        torch.manual_seed(seed)
        waveform = self.generator.infer(
            batch.input_ids,
            batch.input_lengths,
            noise_scale=variation,
            noise_scale_w=0.8,
            length_scale=1.0 / speed,
            max_len=4_000,
        )[0][0, 0]
        return waveform.float()

    @torch.inference_mode()
    def synthesize(
        self,
        text: str = "",
        *,
        phoneme_text: str | None = None,
        input_ids: Sequence[int] | Tensor | None = None,
        input_is_phonemes: bool = False,
        speed: float = 1.0,
        variation: float = 0.667,
        seed: int = 0,
    ) -> tuple[int, Tensor]:
        """Synthesize from explicit phonemes or exact checkpoint token IDs."""
        speed, variation, seed = self._controls(
            speed=speed,
            variation=variation,
            seed=seed,
        )
        if input_ids is not None:
            if phoneme_text is not None or input_is_phonemes:
                raise ValueError("`input_ids` cannot be combined with phoneme inputs.")
            waveform = self._infer_ids(
                input_ids,
                speed=speed,
                variation=variation,
                seed=seed,
            )
            return self.sample_rate, edge_fade(waveform, self.sample_rate).cpu()

        phonemes = require_preprocessed_phonemes(
            text,
            phoneme_text=phoneme_text,
            input_is_phonemes=input_is_phonemes,
        )
        pieces: list[Tensor] = []
        chunks = split_phoneme_text(phonemes)
        for index, chunk in enumerate(chunks):
            if index:
                pieces.append(
                    torch.zeros(
                        round(self.sample_rate * boundary_pause_seconds(chunks[index - 1])),
                        device=self.device,
                        dtype=torch.float32,
                    ))
            sequence = phonemes_to_ids(
                chunk,
                add_blank=self.config.add_blank,
            )
            chunk_seed = (seed + index) % (2**63)
            waveform = self._infer_ids(
                sequence,
                speed=speed,
                variation=variation,
                seed=chunk_seed,
            )
            pieces.append(edge_fade(waveform, self.sample_rate))
        return (
            self.sample_rate,
            torch.cat(pieces).clamp(-1.0, 1.0).cpu(),
        )


__all__ = [
    "InflectV2Runtime",
    "boundary_pause_seconds",
    "edge_fade",
    "split_phoneme_text",
]
