"""Serializable configuration for the dia model family."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from voicehub.configuration import VoiceHubConfig


class DiaConfig(VoiceHubConfig):
    """Serializable configuration for the VoiceHub-owned Dia runtime.

    ``backend="auto"`` is retained as a compatibility spelling, but both
    accepted values select the native implementation. The original
    ``Dia-1.6B`` pickle/JAX layout and framework-owned Dia backends are
    deliberately rejected because they do not share the strict,
    trainable 0626 Safetensors namespace.
    """

    model_type = "dia"

    def __init__(
        self,
        *,
        backend: str = "native",
        compute_dtype: str = "bfloat16",
        use_torch_compile: bool = False,
        revision: str | None = None,
        cache_dir: str | None = None,
        local_files_only: bool = False,
        sample_rate: int = 44_100,
        generation_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        defaults = {
            "do_sample": True,
            "guidance_scale": 3.0,
            "max_new_tokens": 256,
            "temperature": 1.8,
            "top_k": 50,
            "top_p": 0.9,
        }
        defaults.update(dict(generation_config or {}))
        super().__init__(
            sample_rate=sample_rate,
            generation_config=defaults,
            **kwargs,
        )
        self.backend = str(backend).strip().lower()
        self.compute_dtype = compute_dtype
        self.use_torch_compile = use_torch_compile
        self.revision = revision
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self.validate()

    def validate(self) -> None:
        if self.backend not in {"auto", "native"}:
            raise ValueError(
                "Dia backend must be 'native' or 'auto'. Provider-owned "
                "legacy/Transformers runtimes are no longer used.")
        if not isinstance(self.compute_dtype, str) or not self.compute_dtype.strip():
            raise ValueError("`compute_dtype` must be a non-empty string.")
        if not isinstance(self.use_torch_compile, bool):
            raise TypeError("`use_torch_compile` must be a boolean.")
        if self.use_torch_compile:
            raise ValueError(
                "`use_torch_compile=True` is no longer a Dia-specific "
                "runtime switch. Register a reversible VoiceHub "
                "InferenceStrategy and pass it through "
                "`from_pretrained(..., inference_strategy=...)` instead.")
        if self.revision is not None and (not isinstance(self.revision, str) or not self.revision.strip()):
            raise ValueError("`revision` must be a non-empty string or None.")
        if self.cache_dir is not None and (not isinstance(self.cache_dir,
                                                          (str, Path)) or not str(self.cache_dir).strip()):
            raise ValueError("`cache_dir` must be a non-empty path or None.")
        if not isinstance(self.local_files_only, bool):
            raise TypeError("`local_files_only` must be a boolean.")
        if (isinstance(self.sample_rate, bool) or not isinstance(self.sample_rate, int) or
                self.sample_rate < 1):
            raise ValueError("`sample_rate` must be a positive integer.")


__all__ = ['DiaConfig']
