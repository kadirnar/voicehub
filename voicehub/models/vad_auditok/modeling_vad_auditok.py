"""VoiceHub-native Auditok-compatible energy VAD wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from voicehub.audio_modeling_utils import PreTrainedVADModel
from voicehub.modeling_outputs import SpeechSegment, VADOutput
from voicehub.models.vad_auditok.configuration_vad_auditok import AuditokVADConfig
from voicehub.vad_utils import merge_speech_segments


class AuditokVADForVoiceActivityDetection(PreTrainedVADModel):
    """Detect energetic regions without allocating neural weights.

    The public configuration preserves Auditok's established energy and
    calibration semantics, but the complete algorithm executes inside
    VoiceHub. It is an audio-activity detector rather than a calibrated
    speech classifier, so frame probabilities are deliberately
    unavailable.
    """

    config_class = AuditokVADConfig
    default_model_name_or_path = "auditok-energy-vad"
    training_support = "inference-only"
    supports_generic_finetuning = False

    def __init__(
        self,
        config: AuditokVADConfig | str | Path | None = None,
        *,
        model_path: str | Path | None = None,
        device: str = "cpu",
        lazy_load: bool = True,
        **kwargs,
    ):
        config = self._coerce_config(config, model_path=model_path, **kwargs)
        super().__init__(config, device=device, lazy_load=lazy_load)

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device not in {"auto", "cpu"}:
            raise ValueError("Auditok is CPU-only; use `device='cpu'`.")
        return "cpu"

    def _load_pretrained_model(self) -> None:
        from voicehub.architectures.energy_vad import EnergyVoiceActivityDetector

        self.model = EnergyVoiceActivityDetector()

    def _detect(
        self,
        audio: Any,
        *,
        sampling_rate: int | None = None,
        threshold: float = 0.5,
        onset: float | None = None,
        offset: float | None = None,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30,
        max_speech_duration_s: float | None = None,
        window_size_samples: int | None = None,
        return_frames: bool = False,
    ) -> VADOutput:
        unsupported = []
        if threshold != 0.5:
            unsupported.append("threshold")
        if onset is not None:
            unsupported.append("onset")
        if offset is not None:
            unsupported.append("offset")
        if return_frames:
            unsupported.append("return_frames")
        if unsupported:
            formatted = ", ".join(f"`{name}`" for name in unsupported)
            raise ValueError(
                "Auditok exposes an energy threshold in decibels, not "
                f"calibrated speech probabilities, and cannot honor: {formatted}.")

        from voicehub.processing.waveform import load_native_audio

        materialized = load_native_audio(
            audio,
            sampling_rate=sampling_rate,
            target_sampling_rate=self.sample_rate,
        )
        analysis_window_s = self.config.analysis_window_s
        if window_size_samples is not None:
            analysis_window_s = window_size_samples / materialized.sampling_rate
            if not 0.01 <= analysis_window_s <= 0.1:
                raise ValueError(
                    "`window_size_samples` must resolve to an Auditok analysis "
                    "window between 0.01 and 0.1 seconds.")

        minimum_duration = max(
            min_speech_duration_ms / 1000,
            analysis_window_s,
        )
        if (max_speech_duration_s is not None and max_speech_duration_s < minimum_duration):
            raise ValueError(
                "`max_speech_duration_s` cannot be shorter than Auditok's "
                "effective minimum speech duration.")
        detection = self.model.detect(
            materialized.waveform,
            sampling_rate=materialized.sampling_rate,
            energy_threshold_db=self.config.energy_threshold_db,
            threshold_method=self.config.threshold_method,
            analysis_window_s=analysis_window_s,
            minimum_energy_threshold_db=(self.config.minimum_energy_threshold_db),
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
            max_speech_duration_s=max_speech_duration_s,
            strict_min_duration=self.config.strict_min_duration,
            drop_trailing_silence=self.config.drop_trailing_silence,
            window_size_samples=window_size_samples,
        )
        segments = []
        for region in detection.regions:
            start = region.start_sample / materialized.sampling_rate
            end = region.end_sample / materialized.sampling_rate
            if end <= start:
                continue
            segments.append(
                SpeechSegment(
                    start=start,
                    end=end,
                    score=None,
                    metadata={
                        "decision": "energy",
                        "threshold_method": self.config.threshold_method,
                        "threshold_db": detection.threshold_db,
                    },
                ))

        return VADOutput(
            segments=merge_speech_segments(segments),
            duration=materialized.duration,
            sample_rate=materialized.sampling_rate,
            probabilities=None,
            metadata={
                "backend":
                "voicehub-native",
                "algorithm":
                "short-term-energy",
                "threshold_method":
                self.config.threshold_method,
                "energy_threshold_db":
                (self.config.energy_threshold_db if self.config.threshold_method == "fixed" else None),
                "analysis_window_s":
                analysis_window_s,
                "effective_min_speech_duration_s":
                minimum_duration,
                "resolved_energy_threshold_db":
                detection.threshold_db,
                "frame_count":
                detection.frame_energies_db.numel(),
                "frame_scores_available":
                False,
            },
        )

    def _validate_training_runtime(self) -> None:
        raise ValueError(
            "Auditok is an algorithmic energy detector with no trainable "
            "checkpoint. Fine-tuning is not applicable.")
