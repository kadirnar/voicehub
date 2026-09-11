"""Public configuration for the standalone Descript audio codec."""

from voicehub.configuration import VoiceHubConfig


class DacCodecConfig(VoiceHubConfig):
    model_type = "dac"

    def __init__(self, *, sample_rate=44100, native_config=None, **kwargs):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.native_config = dict(native_config or {})
        if self.native_config:
            from .native.configuration import DacConfig

            config = DacConfig.from_dict(self.native_config)
            self.native_config = config.to_dict()
            self.sample_rate = config.sampling_rate
