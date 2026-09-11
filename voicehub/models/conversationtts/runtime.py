"""Compatibility entry points for native ConversationTTS checkpoints."""

from __future__ import annotations

from pathlib import Path

from voicehub.models.conversationtts.native.checkpoint import (
    ConversationTTSCheckpointReport,
    load_conversationtts_checkpoint,
)


def resume_for_inference(
    checkpoint: str | Path,
    experiment_directory: str | None,
    model,
    device: str,
) -> ConversationTTSCheckpointReport:
    """Load a strict native or restricted legacy checkpoint."""
    del experiment_directory
    return load_conversationtts_checkpoint(
        model,
        checkpoint,
        device=device,
    )


__all__ = [
    "ConversationTTSCheckpointReport",
    "resume_for_inference",
]
