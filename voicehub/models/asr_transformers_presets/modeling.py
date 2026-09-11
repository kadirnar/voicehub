"""Model aliases for the historical ASR preset namespace.

The former shared upstream runtime has been retired.  Each exported
class is the canonical VoiceHub-native implementation for its
architecture family.
"""

from voicehub.models.asr_cohere.modeling import CohereForSpeechRecognition
from voicehub.models.asr_hubert.modeling import HubertForSpeechRecognition
from voicehub.models.asr_medasr.modeling import MedASRForSpeechRecognition
from voicehub.models.asr_moonshine.modeling import MoonshineForSpeechRecognition
from voicehub.models.asr_nemotron.modeling import NemotronForSpeechRecognition
from voicehub.models.asr_parakeet_tdt.modeling import ParakeetTDTForSpeechRecognition
from voicehub.models.asr_seamless_m4t_v2.modeling import SeamlessM4Tv2ForSpeechRecognition
from voicehub.models.asr_wav2vec2.modeling import Wav2Vec2ForSpeechRecognition
from voicehub.models.asr_wavlm.modeling import WavLMForSpeechRecognition
from voicehub.models.asr_whisper_native.modeling import WhisperForSpeechRecognition

__all__ = [
    "CohereForSpeechRecognition",
    "HubertForSpeechRecognition",
    "MedASRForSpeechRecognition",
    "MoonshineForSpeechRecognition",
    "NemotronForSpeechRecognition",
    "ParakeetTDTForSpeechRecognition",
    "SeamlessM4Tv2ForSpeechRecognition",
    "Wav2Vec2ForSpeechRecognition",
    "WavLMForSpeechRecognition",
    "WhisperForSpeechRecognition",
]
