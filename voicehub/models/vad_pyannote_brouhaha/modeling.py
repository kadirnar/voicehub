"""VoiceHub-native Brouhaha VAD, SNR, and C50 provider."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from voicehub.inference_configuration import VADInferenceConfig
from voicehub.models.vad_pyannote.modeling import PyannoteVADForVoiceActivityDetection
from voicehub.outputs import VADOutput
from voicehub.vad_utils import frame_probabilities_to_segments

from .configuration import PyannoteBrouhahaVADConfig


class PyannoteBrouhahaVADForVoiceActivityDetection(PyannoteVADForVoiceActivityDetection):
    """Expose native Brouhaha speech, SNR, and C50 frame predictions."""

    config_class = PyannoteBrouhahaVADConfig
    default_model_name_or_path = "pyannote/brouhaha"
    native_variant = "brouhaha"

    def __init__(
        self,
        config: PyannoteBrouhahaVADConfig | str | Path | None = None,
        *,
        model_path: str | Path | None = None,
        device: str = "auto",
        lazy_load: bool = True,
        token: str | bool | None = None,
        trust_pickle_checkpoint: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            config,
            model_path=model_path,
            device=device,
            lazy_load=lazy_load,
            token=token,
            trust_pickle_checkpoint=trust_pickle_checkpoint,
            **kwargs,
        )

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
        import torch

        from voicehub.processing.waveform import load_native_audio

        if window_size_samples is not None:
            raise ValueError(
                "Brouhaha chunk and frame geometry are fixed by its "
                "checkpoint; `window_size_samples` is not supported.")
        materialized = load_native_audio(
            audio,
            sampling_rate=sampling_rate,
            target_sampling_rate=self.sample_rate,
        )
        with torch.inference_mode():
            output = self._frame_output(materialized.waveform)
        values = output.scores.detach().float().cpu()
        if values.ndim != 2 or values.shape[-1] != 3:
            raise RuntimeError("Native Brouhaha must emit VAD, SNR, and C50 per frame.")
        vad_scores = values[:, 0]
        snr_scores = values[:, 1]
        c50_scores = values[:, 2]
        postprocessing = VADInferenceConfig(
            threshold=threshold,
            onset=onset,
            offset=offset,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
            max_speech_duration_s=max_speech_duration_s,
        )
        segments = frame_probabilities_to_segments(
            vad_scores.tolist(),
            sampling_rate=self.sample_rate,
            frame_hop_samples=output.frame_hop_samples,
            frame_length_samples=output.frame_length_samples,
            duration_samples=materialized.waveform.numel(),
            config=postprocessing,
        )
        frame_snr = tuple(float(item) for item in snr_scores.tolist())
        frame_c50 = tuple(float(item) for item in c50_scores.tolist())
        return VADOutput(
            segments=segments,
            duration=materialized.duration,
            sample_rate=self.sample_rate,
            probabilities=(tuple(float(item) for item in vad_scores.tolist()) if return_frames else None),
            metadata={
                "backend":
                "voicehub-native",
                "architecture":
                "pyannet",
                "variant":
                self.native_variant,
                "checkpoint_format":
                self.native_checkpoint_format,
                "checkpoint_adapter":
                self.checkpoint_adapter,
                "checkpoint_revision": (None if self.artifacts is None else self.artifacts.revision),
                "converted_from_pickle":
                (False if self.artifacts is None else self.artifacts.converted_from_pickle),
                "frame_hop_samples":
                output.frame_hop_samples,
                "frame_length_samples":
                output.frame_length_samples,
                "frame_scores_available":
                True,
                "mean_snr_db": (float(snr_scores.mean().item()) if snr_scores.numel() else None),
                "mean_c50_db": (float(c50_scores.mean().item()) if c50_scores.numel() else None),
                "frame_snr_db":
                frame_snr if return_frames else None,
                "frame_c50_db":
                frame_c50 if return_frames else None,
                "auxiliary_outputs": ("snr_db", "c50_db"),
            },
        )


__all__ = ["PyannoteBrouhahaVADForVoiceActivityDetection"]
