"""Configuration aliases for the historical ASR preset namespace."""

from voicehub.models.asr_cohere.configuration import CohereASRConfig
from voicehub.models.asr_hubert.configuration import HubertASRConfig
from voicehub.models.asr_medasr.configuration import MedASRConfig
from voicehub.models.asr_moonshine.configuration import MoonshineASRConfig
from voicehub.models.asr_nemotron.configuration import NemotronASRConfig
from voicehub.models.asr_parakeet_tdt.configuration import ParakeetTDTASRConfig
from voicehub.models.asr_seamless_m4t_v2.configuration import SeamlessM4Tv2ASRConfig
from voicehub.models.asr_wav2vec2.configuration import Wav2Vec2ASRConfig
from voicehub.models.asr_wavlm.configuration import WavLMASRConfig
from voicehub.models.asr_whisper_native.configuration import WhisperASRConfig

__all__ = [
    "CohereASRConfig",
    "HubertASRConfig",
    "MedASRConfig",
    "MoonshineASRConfig",
    "NemotronASRConfig",
    "ParakeetTDTASRConfig",
    "SeamlessM4Tv2ASRConfig",
    "Wav2Vec2ASRConfig",
    "WavLMASRConfig",
    "WhisperASRConfig",
]
