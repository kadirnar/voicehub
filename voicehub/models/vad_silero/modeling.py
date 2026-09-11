"""Native Silero VAD inference and fine-tuning wrapper."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from voicehub.hub import write_json_file
from voicehub.models.audio import PreTrainedVADModel
from voicehub.models.native_utils import resolve_cpu_cuda_device
from voicehub.models.vad_silero.configuration import SileroVADConfig
from voicehub.outputs import VADOutput, VADSegment


class SileroVADForVoiceActivityDetection(PreTrainedVADModel):
    """Run official Silero weights through VoiceHub's differentiable graph.

    The upstream package is never imported. Official Safetensors or JIT
    weights are converted strictly into a request-stateless native
    model, so offline calls and concurrent streams cannot leak recurrent
    state.
    """

    config_class = SileroVADConfig
    default_model_name_or_path = "safestack/silero-vad"
    architecture_family = "frame-classification"

    def __init__(
        self,
        config: SileroVADConfig | str | Path | None = None,
        *,
        model_path: str | Path | None = None,
        device: str = "auto",
        lazy_load: bool = True,
        token: str | bool | None = None,
        **kwargs: Any,
    ) -> None:
        if token is not None and not isinstance(token, (str, bool)):
            raise TypeError("`token` must be a string, boolean, or None.")
        if isinstance(token, str) and not token.strip():
            raise ValueError("String `token` values must be non-empty.")
        self._hub_token = token
        self.artifact: Any | None = None
        self.native_config: Any | None = None
        self.checkpoint_format: str | None = None
        self.checkpoint_adapter: str | None = None
        config = self._coerce_config(
            config,
            model_path=model_path,
            **kwargs,
        )
        super().__init__(
            config,
            device=device,
            lazy_load=lazy_load,
        )

    @staticmethod
    def _resolve_device(device: str) -> str:
        return resolve_cpu_cuda_device(device, provider="silero-vad")

    def _load_pretrained_model(self) -> None:
        from voicehub.models.vad_silero.artifacts import load_silero_vad_checkpoint, resolve_silero_vad_artifact
        from voicehub.models.vad_silero.native.configuration import SileroVADConfig as NativeSileroVADConfig
        from voicehub.models.vad_silero.native.modeling import SileroVADModel

        # Base lifecycle calls resolve this before loading, while the explicit
        # guard keeps the provider hook correct when exercised directly.
        self.device = self._resolve_device(self.device)
        source = self.config.name_or_path or self.default_model_name_or_path
        native_config = NativeSileroVADConfig(sampling_rate=self.sample_rate, )
        artifact = resolve_silero_vad_artifact(
            source,
            sample_rate=self.sample_rate,
            checkpoint_filename=self.config.checkpoint_filename,
            cache_dir=self.config.cache_dir,
            revision=self.config.revision,
            token=self._hub_token,
            local_files_only=self.config.local_files_only,
        )
        model = SileroVADModel(native_config)
        checkpoint_format, adapter = load_silero_vad_checkpoint(
            model,
            artifact,
            native_config,
        )
        model.to(device=self.device)

        self.artifact = artifact
        self.native_config = native_config
        self.checkpoint_format = checkpoint_format
        self.checkpoint_adapter = adapter
        self.model = model

    def _segmentation_config(
        self,
        *,
        threshold: float,
        onset: float | None,
        offset: float | None,
        min_speech_duration_ms: int,
        min_silence_duration_ms: int,
        speech_pad_ms: int,
        max_speech_duration_s: float | None,
        window_size_samples: int | None,
    ) -> Any:
        from voicehub.models.vad_silero.native.segmentation import SileroVADSegmentationConfig

        if self.native_config is None:
            raise RuntimeError("Silero VAD must be loaded before inference.")
        if (window_size_samples is not None and window_size_samples != self.native_config.frame_size):
            raise ValueError(
                "The native Silero graph has a fixed window of "
                f"{self.native_config.frame_size} samples at "
                f"{self.sample_rate} Hz; received "
                f"`window_size_samples={window_size_samples}`.")
        return SileroVADSegmentationConfig(
            threshold=threshold if onset is None else onset,
            negative_threshold=offset,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
            max_speech_duration_s=(math.inf if max_speech_duration_s is None else max_speech_duration_s),
        )

    def _probabilities_to_output(
        self,
        probabilities: Any,
        *,
        valid_samples: int,
        segmentation_config: Any,
        return_frames: bool,
        streaming: bool,
    ) -> VADOutput:
        from voicehub.models.vad_silero.native.segmentation import segment_speech_probabilities

        if self.native_config is None:
            raise RuntimeError("Silero VAD must be loaded before inference.")
        values = probabilities.detach().float().cpu().reshape(-1)
        native_segments = segment_speech_probabilities(
            values,
            audio_length_samples=valid_samples,
            model_config=self.native_config,
            config=segmentation_config,
        )
        frame_size = self.native_config.frame_size
        normalized_segments = []
        for segment in native_segments:
            first_frame = segment.start // frame_size
            final_frame = min(
                values.numel(),
                math.ceil(segment.end / frame_size),
            )
            segment_values = values[first_frame:final_frame]
            score = (None if segment_values.numel() == 0 else float(segment_values.mean().item()))
            normalized_segments.append(
                VADSegment(
                    start=segment.start / self.sample_rate,
                    end=segment.end / self.sample_rate,
                    score=score,
                ))
        return VADOutput(
            segments=tuple(normalized_segments),
            duration=valid_samples / self.sample_rate,
            sample_rate=self.sample_rate,
            probabilities=(tuple(float(value) for value in values.tolist()) if return_frames else None),
            metadata={
                "architecture": "silero-vad",
                "backend": "voicehub-native",
                "runtime": "pytorch",
                "checkpoint_format": self.checkpoint_format,
                "checkpoint_adapter": self.checkpoint_adapter,
                "checkpoint_revision": (None if self.artifact is None else self.artifact.revision),
                "frame_scores_available": True,
                "frame_size_samples": frame_size,
                "streaming": streaming,
            },
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
        from voicehub.processing.waveform import load_native_audio

        segmentation_config = self._segmentation_config(
            threshold=threshold,
            onset=onset,
            offset=offset,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
            max_speech_duration_s=max_speech_duration_s,
            window_size_samples=window_size_samples,
        )
        materialized = load_native_audio(
            audio,
            sampling_rate=sampling_rate,
            target_sampling_rate=self.sample_rate,
        )
        waveform = materialized.waveform.unsqueeze(0).to(device=self.device, )

        import torch

        with torch.inference_mode():
            output = self.model.frame_probabilities(
                waveform,
                pad_final_frame=True,
            )
        return self._probabilities_to_output(
            output.probabilities,
            valid_samples=output.valid_samples,
            segmentation_config=segmentation_config,
            return_frames=return_frames,
            streaming=False,
        )

    def stream(
        self,
        *,
        sampling_rate: int,
        **inference_kwargs: Any,
    ) -> Any:
        """Create a true incremental session with independent recurrent
        state."""
        from voicehub.models.vad_silero.streaming import SileroVADStreamingSession

        if sampling_rate != self.sample_rate:
            raise ValueError(
                "Native Silero streaming requires chunks already sampled at "
                f"{self.sample_rate} Hz. Offline inference can resample a "
                "complete waveform safely.")
        defaults = self.inference_config.to_dict()
        defaults.update(inference_kwargs)
        self.load()
        segmentation_config = self._segmentation_config(
            threshold=defaults.pop("threshold", 0.5),
            onset=defaults.pop("onset", None),
            offset=defaults.pop("offset", None),
            min_speech_duration_ms=defaults.pop(
                "min_speech_duration_ms",
                250,
            ),
            min_silence_duration_ms=defaults.pop(
                "min_silence_duration_ms",
                100,
            ),
            speech_pad_ms=defaults.pop("speech_pad_ms", 30),
            max_speech_duration_s=defaults.pop(
                "max_speech_duration_s",
                None,
            ),
            window_size_samples=defaults.pop(
                "window_size_samples",
                None,
            ),
        )
        return_frames = defaults.pop("return_frames", False)
        if defaults:
            unknown = ", ".join(sorted(defaults))
            raise ValueError(f"Unsupported Silero streaming option(s): {unknown}.")
        return SileroVADStreamingSession(
            self,
            sampling_rate=sampling_rate,
            segmentation_config=segmentation_config,
            return_frames=return_frames,
        )

    def prepare_training_inputs(
        self,
        inputs: dict[str, Any],
        *,
        phase: str,
    ) -> dict[str, Any]:
        from voicehub.models.vad_silero.training_vad_silero import prepare_silero_vad_training_batch

        del phase
        if self.model is None:
            self.load_for_training()
        return prepare_silero_vad_training_batch(self, inputs)

    def _validate_training_runtime(self) -> None:
        """Every supported checkpoint is converted to a differentiable
        graph."""

    def _save_pretrained(self, save_directory: Path) -> None:
        from voicehub.checkpointing import save_safetensors
        from voicehub.models.vad_silero.artifacts import NATIVE_SILERO_VAD_FILENAME, NATIVE_SILERO_VAD_FORMAT

        if self.model is None or self.native_config is None:
            self.load()
        save_directory.mkdir(parents=True, exist_ok=True)
        save_safetensors(
            self.model.state_dict(),
            save_directory / NATIVE_SILERO_VAD_FILENAME,
            metadata={
                "format": NATIVE_SILERO_VAD_FORMAT,
                "sample_rate": str(self.sample_rate),
                "architecture": "silero-vad",
            },
        )
        config_values = self.config.to_dict()
        config_values.update({
            'architectures': [self.__class__.__name__],
            "voicehub_checkpoint_format": NATIVE_SILERO_VAD_FORMAT,
        })
        write_json_file(
            save_directory / "config.json",
            config_values,
        )

    def export_native_pretrained(
        self,
        save_directory: str | Path,
    ) -> Path:
        """Write a directly loadable native artifact without nesting."""
        destination = Path(save_directory).expanduser()
        self._save_pretrained(destination)
        return destination


__all__ = ["SileroVADForVoiceActivityDetection"]
