"""Compatibility names for the VoiceHub-owned Brouhaha graph.

Older converted metadata may refer to this module. Runtime checkpoint
loading does not import or emulate ``brouhaha.models`` and never
deserializes its class objects.
"""

from voicehub.models.vad_pyannote.native.modeling import BrouhahaActivation as CustomActivation
from voicehub.models.vad_pyannote.native.modeling import BrouhahaClassifier as CustomClassifier
from voicehub.models.vad_pyannote.native.modeling import ParametricSigmoid
from voicehub.models.vad_pyannote.native.modeling import PyanNet as CustomPyanNetModel

__all__ = [
    "CustomActivation",
    "CustomClassifier",
    "CustomPyanNetModel",
    "ParametricSigmoid",
]
