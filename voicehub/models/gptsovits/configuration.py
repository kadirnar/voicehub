"""Configuration for native GPT-SoVITS classic-S2 variants."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicehub.configuration import VoiceHubConfig, reject_serialized_secrets
from voicehub.models.gptsovits.native.configuration import normalize_gptsovits_variant


class GPTSoVITSConfig(VoiceHubConfig):
    """Configure one audited S1/classic-S2 checkpoint set."""

    model_type = "gptsovits"

    def __init__(
        self,
        *,
        runtime_config: dict[str, Any] | None = None,
        variant: str = "v2",
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        trust_pickle_checkpoint: bool = False,
        enable_native_finetuning: bool = False,
        training_enable_s2_discriminator: bool = True,
        sample_rate: int = 32_000,
        **kwargs: Any,
    ) -> None:
        reject_serialized_secrets(kwargs, owner=self.__class__.__name__)
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.runtime_config = runtime_config
        self.variant = normalize_gptsovits_variant(variant)
        self.revision = revision
        self.cache_dir = (None if cache_dir is None else str(Path(cache_dir).expanduser()))
        self.local_files_only = local_files_only
        self.trust_pickle_checkpoint = trust_pickle_checkpoint
        self.enable_native_finetuning = enable_native_finetuning
        self.training_enable_s2_discriminator = (training_enable_s2_discriminator)
        self._validate()

    def _validate(self) -> None:
        if self.runtime_config is not None and not isinstance(
                self.runtime_config,
                Mapping,
        ):
            raise TypeError("`runtime_config` must be a mapping or None.")
        if self.revision is not None:
            if not isinstance(self.revision, str) or not self.revision.strip():
                raise ValueError("`revision` must be non-empty or None.")
            self.revision = self.revision.strip()
        for name in (
                "local_files_only",
                "trust_pickle_checkpoint",
                "enable_native_finetuning",
                "training_enable_s2_discriminator",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"`{name}` must be a boolean.")
        if self.sample_rate != 32_000:
            raise ValueError("The supported GPT-SoVITS classic S2 decoders synthesize at 32 kHz.")


__all__ = ["GPTSoVITSConfig"]
