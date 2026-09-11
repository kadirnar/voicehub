"""Stable lazy model imports for native OuteTTS."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from voicehub.models.outetts.inference import OuteTTSForTextToSpeech
    from voicehub.models.outetts.native.modeling import OuteTTSForCausalLM


def __getattr__(name: str):
    if name == "OuteTTSForTextToSpeech":
        from voicehub.models.outetts.inference import OuteTTSForTextToSpeech

        return OuteTTSForTextToSpeech
    if name == "OuteTTSForCausalLM":
        from voicehub.models.outetts.native.modeling import OuteTTSForCausalLM

        return OuteTTSForCausalLM
    raise AttributeError(name)


__all__ = ["OuteTTSForCausalLM", "OuteTTSForTextToSpeech"]
