"""VoiceHub-native WebRTC VAD wrapper with explicit PCM framing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from voicehub.architectures.webrtc_vad.detector import NativeWebRTCVAD
from voicehub.audio import load_audio
from voicehub.audio_modeling_utils import PreTrainedVADModel
from voicehub.inference_configuration import VADInferenceConfig
from voicehub.modeling_outputs import SpeechSegment, VADOutput
from voicehub.models.vad_webrtc.configuration_vad_webrtc import WebRTCVADConfig
from voicehub.vad_utils import frame_probabilities_to_segments


def _pcm16_tensor(waveform: Any) -> Any:
    """Convert normalized audio with the reference scalar rounding rules."""
    import torch

    values = waveform.detach().to(
        device="cpu",
        dtype=torch.float64,
    )
    values = torch.nan_to_num(
        values,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    return values.clamp(-1.0, 1.0).mul(32767.0).round().to(torch.int16).contiguous()


def _pcm16_samples(waveform: Any) -> list[int]:
    return _pcm16_tensor(waveform).tolist()


class WebRTCVADForVoiceActivityDetection(PreTrainedVADModel):
    """Streaming-compatible WebRTC GMM VAD exposed as normalized segments."""

    config_class = WebRTCVADConfig
    default_model_name_or_path = "webrtc-vad"

    def __init__(
        self,
        config: WebRTCVADConfig | str | Path | None = None,
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
        if device not in ("auto", "cpu"):
            raise ValueError("WebRTC VAD is CPU-only; use `device='cpu'`.")
        return "cpu"

    def _load_pretrained_model(self) -> None:
        from voicehub.architectures.webrtc_vad.acceleration import get_accelerator

        self.model = NativeWebRTCVAD(self.config.aggressiveness)
        self._accelerator, self._acceleration_status = get_accelerator()

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
                "WebRTC exposes binary decisions configured by "
                "`aggressiveness` and cannot honor inference option(s): "
                f"{formatted}.")
        materialized = load_audio(
            audio,
            sampling_rate=sampling_rate,
            target_sampling_rate=self.sample_rate,
        )
        frame_samples = round(materialized.sampling_rate * self.config.frame_duration_ms / 1000)
        if window_size_samples is not None and window_size_samples != frame_samples:
            raise ValueError(
                "WebRTC requires the configured 10/20/30 ms frame size; "
                f"expected {frame_samples} samples.")
        accelerator = getattr(self, "_accelerator", None)
        if accelerator is not None:
            flags = accelerator(
                _pcm16_tensor(materialized.waveform), materialized.sampling_rate, frame_samples,
                self.config.aggressiveness)
        else:
            pcm = _pcm16_samples(materialized.waveform)
            vad = NativeWebRTCVAD(self.config.aggressiveness)
            flags = []
            for start in range(0, len(pcm), frame_samples):
                frame = pcm[start:start + frame_samples]
                if len(frame) < frame_samples:
                    frame.extend([0] * (frame_samples - len(frame)))
                flags.append(1.0 if vad.is_speech(
                    frame,
                    materialized.sampling_rate,
                ) else 0.0)
        postprocessing = VADInferenceConfig(
            threshold=0.5,
            onset=0.5,
            offset=0.5,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
            max_speech_duration_s=max_speech_duration_s,
        )
        detected = frame_probabilities_to_segments(
            flags,
            sampling_rate=materialized.sampling_rate,
            frame_hop_samples=frame_samples,
            frame_length_samples=frame_samples,
            config=postprocessing,
            duration_samples=len(materialized.waveform),
        )
        segments = tuple(
            SpeechSegment(
                start=segment.start,
                end=segment.end,
                score=None,
                metadata={"decision": "binary"},
            ) for segment in detected)
        return VADOutput(
            segments=segments,
            duration=materialized.duration,
            sample_rate=materialized.sampling_rate,
            probabilities=None,
            metadata={
                "backend": "voicehub-native-webrtc",
                "aggressiveness": self.config.aggressiveness,
                "frame_duration_ms": self.config.frame_duration_ms,
                "frame_scores_available": False,
                "execution_engine": getattr(self, "_acceleration_status", "python"),
            },
        )

    def _validate_training_runtime(self) -> None:
        raise ValueError(
            "WebRTC VAD is a fixed-point GMM detector, not a differentiable "
            "checkpoint. Fine-tuning is not applicable.")
