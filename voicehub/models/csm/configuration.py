"""Dependency-light configuration for VoiceHub-native CSM."""

from voicehub.configuration import VoiceHubConfig
from voicehub.models.csm.native.configuration import CSMArchitectureConfig, CSMTransformerConfig


class CSMConfig(VoiceHubConfig):
    """Configuration for VoiceHub's source-faithful native CSM runtime."""

    model_type = "csm"

    def __init__(
        self,
        *,
        torch_dtype: str = "bfloat16",
        sample_rate: int = 24_000,
        revision: str | None = None,
        cache_dir: str | None = None,
        local_files_only: bool = False,
        load_codec: bool = True,
        codec_path: str | None = None,
        verify_integrity: bool = False,
        verify_checkpoint_integrity: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.torch_dtype = torch_dtype
        self.revision = revision
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self.load_codec = load_codec
        self.codec_path = codec_path
        self.verify_integrity = verify_integrity
        self.verify_checkpoint_integrity = verify_checkpoint_integrity


__all__ = [
    "CSMArchitectureConfig",
    "CSMConfig",
    "CSMTransformerConfig",
]
