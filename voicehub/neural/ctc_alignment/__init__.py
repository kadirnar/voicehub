"""Native CTC forced alignment used by the WhisperX compatibility provider."""

from voicehub.neural.ctc_alignment.alignment import (
    AlignedCharacter,
    AlignedWord,
    CTCAlignment,
    align_ctc_transcript,
    build_trellis,
)
from voicehub.neural.ctc_alignment.metadata import DEFAULT_ALIGNMENT_MODELS, WHISPERX_REVISION

__all__ = [
    "DEFAULT_ALIGNMENT_MODELS",
    "WHISPERX_REVISION",
    "AlignedCharacter",
    "AlignedWord",
    "CTCAlignment",
    "align_ctc_transcript",
    "build_trellis",
]
