"""Configuration for the WebRTC fixed-point VAD."""

from collections.abc import Mapping

from voicehub.configuration import VoiceHubConfig

_SECRET_OPTIONS = frozenset({
    "access_token",
    "api_key",
    "auth_token",
    "fetch_config",
    "hf_token",
    "token",
    "use_auth_token",
})


def _contains_secret(value) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(name).strip().lower() in _SECRET_OPTIONS or _contains_secret(nested)
            for name, nested in value.items())
    if isinstance(value, (tuple, list)):
        return any(_contains_secret(nested) for nested in value)
    return False


class WebRTCVADConfig(VoiceHubConfig):
    model_type = "vad_webrtc"

    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        aggressiveness: int = 2,
        frame_duration_ms: int = 30,
        inference_config=None,
        **kwargs,
    ):
        if _contains_secret(kwargs) or _contains_secret(inference_config):
            raise ValueError("WebRTC VAD does not accept serialized authentication state.")
        if sample_rate not in (8_000, 16_000, 32_000, 48_000):
            raise ValueError("WebRTC VAD supports 8, 16, 32, or 48 kHz audio.")
        if aggressiveness not in (0, 1, 2, 3):
            raise ValueError("WebRTC `aggressiveness` must be 0, 1, 2, or 3.")
        if frame_duration_ms not in (10, 20, 30):
            raise ValueError("WebRTC frames must be 10, 20, or 30 milliseconds.")
        super().__init__(
            sample_rate=sample_rate,
            inference_config=inference_config or {},
            aggressiveness=aggressiveness,
            frame_duration_ms=frame_duration_ms,
            **kwargs,
        )
