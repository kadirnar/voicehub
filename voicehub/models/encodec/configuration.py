"""Public configuration for standalone Encodec audio coding."""

from voicehub.configuration import VoiceHubConfig


class EncodecCodecConfig(VoiceHubConfig):
    model_type = "encodec"

    def __init__(self, *, sample_rate=24000, native_config=None, bandwidth=None, **kwargs):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.native_config = dict(native_config or {})
        self.bandwidth = bandwidth
        if self.native_config:
            from voicehub.components.audio.codecs.encodec.configuration import EncodecConfig

            config = EncodecConfig.from_dict(self.native_config)
            self.native_config = config.to_dict()
            self.sample_rate = config.sample_rate
