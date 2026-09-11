"""Backward-compatible training imports for VoiceHub-native VoxCPM2."""

from voicehub.models.voxcpm.training_voxcpm import VoxCPMTrainingAdapter, VoxCPMTrainingCollator

__all__ = ["VoxCPMTrainingAdapter", "VoxCPMTrainingCollator"]
