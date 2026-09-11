"""Native inference lifecycle for MeloTTS VITS checkpoints."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from voicehub.models.melotts.native.artifacts import MeloTTSArtifacts, resolve_melotts_artifacts
from voicehub.models.melotts.native.checkpoint import load_melotts_checkpoint, read_legacy_melotts_checkpoint
from voicehub.models.melotts.native.frontend import NativeMeloTTSFrontend
from voicehub.models.melotts.native.modeling import build_melotts_model
from voicehub.models.vits.native.weight_norm import (
    LegacyWeightNormInferenceCache,
    enable_legacy_weight_norm_inference_cache,
)
from voicehub.optimization.protocols import OptimizationCompileTarget, OptimizationModuleRoot


class MeloTTSRuntime:
    """Strict native execution over caller-supplied linguistic features."""

    def __init__(
        self,
        source: str | Path,
        *,
        config_path: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        checkpoint_filename: str | None = None,
        revision: str | None = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        trust_pickle_checkpoint: bool = False,
    ) -> None:
        self.device = torch.device(device)
        self.artifacts = resolve_melotts_artifacts(
            source,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            checkpoint_filename=checkpoint_filename,
            revision=revision,
        )
        self.config = self.artifacts.config
        self.sample_rate = self.config.data.sample_rate
        self.model = build_melotts_model(self.config)
        self._load_checkpoint(
            self.artifacts,
            dtype=dtype,
            trust_pickle_checkpoint=trust_pickle_checkpoint,
        )
        parameter = next(self.model.parameters())
        self.dtype = parameter.dtype
        self.frontend = NativeMeloTTSFrontend(self.config)
        self._weight_norm_cache: LegacyWeightNormInferenceCache | None = None
        self.eval()

    def optimization_compile_targets(
        self,
        mode: str,
    ) -> tuple[OptimizationCompileTarget, ...]:
        """Return the generator method used by the selected runtime mode."""
        if mode == "inference":
            attribute = "infer"
        elif mode == "training":
            attribute = "forward"
        else:
            raise ValueError(f"Unsupported optimization mode {mode!r}.")
        return (OptimizationCompileTarget(
            f"model.{attribute}",
            self.model,
            attribute,
        ), )

    def optimization_module_roots(self) -> tuple[OptimizationModuleRoot, ...]:
        """Expose the checkpoint graph to reversible selector passes."""
        return (OptimizationModuleRoot("model", self.model), )

    def state_dict(self):
        """Proxy canonical checkpoint state without wrapping the graph."""
        return self.model.state_dict()

    def _load_checkpoint(
        self,
        artifacts: MeloTTSArtifacts,
        *,
        dtype: torch.dtype | None,
        trust_pickle_checkpoint: bool,
    ) -> None:
        if artifacts.legacy_checkpoint:
            state = read_legacy_melotts_checkpoint(
                self.model,
                artifacts.checkpoint_path,
                trust_pickle_checkpoint=trust_pickle_checkpoint,
                expected_sha256=artifacts.expected_checkpoint_sha256,
            )
            incompatible = self.model.load_state_dict(state, strict=True)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise RuntimeError("MeloTTS legacy checkpoint failed strict assignment.")
        else:
            load_melotts_checkpoint(
                self.model,
                artifacts.checkpoint_path,
                device=self.device,
                dtype=dtype,
            )
        if dtype is None:
            self.model.to(device=self.device)
        else:
            self.model.to(device=self.device, dtype=dtype)

    def train(self, mode: bool = True) -> MeloTTSRuntime:
        self.model.train(mode)
        if mode:
            if self._weight_norm_cache is not None:
                self._weight_norm_cache.restore()
                self._weight_norm_cache = None
        elif self._weight_norm_cache is None:
            self._weight_norm_cache = (enable_legacy_weight_norm_inference_cache(self.model))
        return self

    def eval(self) -> MeloTTSRuntime:
        return self.train(False)

    @property
    def speakers(self) -> tuple[str, ...]:
        return tuple(self.config.data.speakers)

    @staticmethod
    def _validate_controls(
        *,
        speed: float,
        sdp_ratio: float,
        noise_scale: float,
        noise_scale_w: float,
        max_frames: int | None,
    ) -> None:
        for name, value in (
            ("speed", speed),
            ("sdp_ratio", sdp_ratio),
            ("noise_scale", noise_scale),
            ("noise_scale_w", noise_scale_w),
        ):
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                    not math.isfinite(float(value))):
                raise ValueError(f"MeloTTS `{name}` must be finite.")
            if name == "speed" and value <= 0:
                raise ValueError("MeloTTS `speed` must be positive.")
            if name != "speed" and value < 0:
                raise ValueError(f"MeloTTS `{name}` must be non-negative.")
        if sdp_ratio > 1:
            raise ValueError("MeloTTS `sdp_ratio` must be in [0, 1].")
        if max_frames is not None and (isinstance(max_frames, bool) or not isinstance(max_frames, int) or
                                       max_frames < 1):
            raise ValueError("MeloTTS `max_frames` must be a positive integer.")

    def generate(
        self,
        *,
        input_ids: Any,
        tone_ids: Any,
        language_ids: Any,
        bert_features: Any,
        ja_bert_features: Any,
        speaker: str | int | None = None,
        speed: float = 1.0,
        sdp_ratio: float = 0.2,
        noise_scale: float = 0.6,
        noise_scale_w: float = 0.8,
        max_frames: int | None = 4_096,
    ) -> Tensor:
        self._validate_controls(
            speed=speed,
            sdp_ratio=sdp_ratio,
            noise_scale=noise_scale,
            noise_scale_w=noise_scale_w,
            max_frames=max_frames,
        )
        features = self.frontend.prepare(
            input_ids=input_ids,
            tone_ids=tone_ids,
            language_ids=language_ids,
            bert_features=bert_features,
            ja_bert_features=ja_bert_features,
            speaker=speaker,
            device=self.device,
            dtype=self.dtype,
        )
        with torch.inference_mode():
            waveform = self.model.infer(
                features.input_ids,
                features.input_lengths,
                features.speaker_ids,
                features.tone_ids,
                features.language_ids,
                features.bert_features,
                features.ja_bert_features,
                noise_scale=float(noise_scale),
                length_scale=1.0 / float(speed),
                noise_scale_w=float(noise_scale_w),
                max_len=max_frames,
                sdp_ratio=float(sdp_ratio),
            )[0][0, 0]
        waveform = waveform.detach().float().cpu().contiguous()
        if waveform.numel() == 0:
            raise RuntimeError("MeloTTS produced an empty waveform.")
        if not bool(torch.isfinite(waveform).all()):
            raise RuntimeError("MeloTTS produced NaN or infinite waveform values.")
        return waveform


__all__ = ["MeloTTSRuntime"]
