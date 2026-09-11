"""Backward-compatible facade over the VoiceHub-native Dia runtime."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import torch

from voicehub.models.base import BaseSpeechModel
from voicehub.models.dia.native.runtime import DiaRuntime, load_dia_runtime

DEFAULT_SAMPLE_RATE = 44_100
SAMPLE_RATE_RATIO = 512


class ComputeDtype(str, Enum):
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"

    def to_dtype(self) -> torch.dtype:
        return getattr(torch, self.value)


def _get_default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Dia:
    """Compatibility API backed by :class:`DiaRuntime`.

    The historical ``dia-v1.pth`` implementation is intentionally not
    retained. ``from_pretrained`` accepts the strict 0626 Safetensors
    artifact and exposes the familiar ``generate`` and ``save_audio``
    methods.
    """

    def __init__(self, runtime: DiaRuntime) -> None:
        if not isinstance(runtime, DiaRuntime):
            raise TypeError(
                "Construct Dia with `Dia.from_pretrained(...)`; direct legacy "
                "configuration construction is no longer supported.")
        self.runtime = runtime
        self.model = runtime.model
        self.dac_model = runtime.processor.audio_tokenizer
        self.config = runtime.model.config
        self.device = runtime.model.device
        self.compute_dtype = next(runtime.model.parameters()).dtype

    @classmethod
    def from_pretrained(
        cls,
        model_name: str | Path = "nari-labs/Dia-1.6B-0626",
        compute_dtype: str | ComputeDtype = ComputeDtype.FLOAT32,
        device: str | torch.device | None = None,
        load_dac: bool = True,
        **kwargs: Any,
    ) -> Dia:
        if not load_dac:
            raise ValueError("Native Dia requires its frozen DAC tokenizer for decoding.")
        dtype_name = (compute_dtype.value if isinstance(compute_dtype, ComputeDtype) else str(compute_dtype))
        runtime = load_dia_runtime(
            model_name,
            device=device or _get_default_device(),
            compute_dtype=dtype_name,
            **kwargs,
        )
        return cls(runtime)

    @classmethod
    def from_local(
        cls,
        config_path: str | Path,
        checkpoint_path: str | Path,
        **kwargs: Any,
    ) -> Dia:
        config = Path(config_path).expanduser().resolve()
        checkpoint = Path(checkpoint_path).expanduser().resolve()
        if config.name != "config.json":
            raise ValueError("Native Dia configuration must be named `config.json`.")
        if checkpoint.suffix != ".safetensors" and not checkpoint.name.endswith(".safetensors.index.json"):
            raise ValueError("Native Dia local checkpoints must use Safetensors.")
        if config.parent != checkpoint.parent:
            raise ValueError("Dia config and checkpoint must be in one coherent directory.")
        return cls.from_pretrained(config.parent, **kwargs)

    def generate(
        self,
        text: str,
        *,
        audio_prompt: Any | None = None,
        max_tokens: int = 256,
        cfg_scale: float = 3.0,
        cfg_filter_top_k: int | None = 50,
        temperature: float = 1.8,
        top_p: float = 0.9,
        **kwargs: Any,
    ) -> torch.Tensor:
        inputs = self.runtime.processor(
            text=[text],
            audio=audio_prompt,
            generation=True,
        ).to(self.device)
        prompt_length = (
            self.runtime.processor.get_audio_prompt_len(inputs["decoder_attention_mask"])
            if audio_prompt is not None else None)
        tokens = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            guidance_scale=cfg_scale,
            top_k=cfg_filter_top_k,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )
        return self.runtime.processor.decode(
            tokens,
            audio_prompt_len=prompt_length,
        )

    @staticmethod
    def save_audio(
        path: str | Path,
        audio: Any,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ) -> str:
        return BaseSpeechModel.save_audio(path, audio, sample_rate)


__all__ = [
    "ComputeDtype",
    "DEFAULT_SAMPLE_RATE",
    "Dia",
    "SAMPLE_RATE_RATIO",
]
