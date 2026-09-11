"""Native Silero/TEN execution with Sherpa-compatible streaming semantics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any

from voicehub.hub import read_json_file, write_json_file
from voicehub.models.audio import PreTrainedVADModel
from voicehub.models.native_utils import resolve_cpu_cuda_device
from voicehub.models.vad_sherpa_onnx.configuration import SherpaONNXVADConfig
from voicehub.outputs import SpeechSegment, VADOutput
from voicehub.vad_utils import merge_speech_segments


@dataclass(frozen=True, slots=True)
class _NativeVADRuntime:
    architecture: str
    checkpoint: Path
    checkpoint_format: str
    checkpoint_adapter: str
    revision: str | None
    converted_from_onnx: bool = False


def _drain_segments(
    detector: Any,
    *,
    sample_rate: int,
) -> tuple[SpeechSegment, ...]:
    segments = []
    while not detector.empty:
        native = detector.front
        start = round(float(native.start) / sample_rate, 12)
        end = round(start + len(native.samples) / sample_rate, 12)
        if end > start:
            segments.append(
                SpeechSegment(
                    start=max(0.0, start),
                    end=end,
                    score=None,
                    metadata={
                        "decision": "sherpa-compatible-native",
                        "start_sample": int(native.start),
                        "num_samples": len(native.samples),
                    },
                ))
        detector.pop()
    return tuple(segments)


def _finalize_segments(
    values: Any,
    *,
    duration: float,
    speech_pad_ms: int,
    max_speech_duration_s: float | None,
) -> tuple[SpeechSegment, ...]:
    padding = speech_pad_ms / 1_000
    padded = []
    for segment in values:
        start = max(0.0, segment.start - padding)
        end = min(duration, segment.end + padding)
        if end > start:
            padded.append(
                SpeechSegment(
                    start=start,
                    end=end,
                    score=segment.score,
                    label=segment.label,
                    channel=segment.channel,
                    metadata=dict(segment.metadata),
                ))
    merged = merge_speech_segments(padded)
    if max_speech_duration_s is None:
        return merged
    split = []
    for segment in merged:
        cursor = segment.start
        while segment.end - cursor > max_speech_duration_s + 1e-12:
            end = round(cursor + max_speech_duration_s, 12)
            split.append(
                SpeechSegment(
                    start=cursor,
                    end=end,
                    score=segment.score,
                    label=segment.label,
                    channel=segment.channel,
                    metadata=dict(segment.metadata),
                ))
            cursor = end
        if segment.end - cursor > 1e-12:
            split.append(
                SpeechSegment(
                    start=cursor,
                    end=segment.end,
                    score=segment.score,
                    label=segment.label,
                    channel=segment.channel,
                    metadata=dict(segment.metadata),
                ))
    return tuple(split)


class SherpaONNXVADSession:
    """One isolated native scorer and pinned Sherpa segmentation state."""

    def __init__(
        self,
        model: SherpaONNXVADForVoiceActivityDetection,
        *,
        sampling_rate: int,
        inference_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if sampling_rate != model.sample_rate:
            raise ValueError(
                f"Native Sherpa-compatible streaming requires "
                f"{model.sample_rate} Hz chunks.")
        self.model = model.load()
        self.sampling_rate = sampling_rate
        defaults = self.model.inference_config.to_dict()
        defaults.update(inference_kwargs or {})
        self.inference_kwargs = defaults
        self._validate_options()
        self._detector = self.model._create_detector(**self.inference_kwargs)
        self._window_shift = self.model._window_shift()
        self._pending = None
        self._segments: list[SpeechSegment] = []
        self._probabilities: list[float] = []
        self._sample_count = 0
        self._result: VADOutput | None = None
        self._closed = False
        self._lock = RLock()

    def _validate_options(self) -> None:
        allowed = {
            "threshold",
            "onset",
            "offset",
            "min_speech_duration_ms",
            "min_silence_duration_ms",
            "speech_pad_ms",
            "max_speech_duration_s",
            "window_size_samples",
            "return_frames",
        }
        unknown = sorted(set(self.inference_kwargs) - allowed)
        if unknown:
            raise ValueError("Unsupported native Sherpa streaming option(s): "
                             f"{', '.join(unknown)}.")
        # Reuse the public inference config for strict ranges and types.
        from voicehub.inference_configuration import VADInferenceConfig

        self.inference_kwargs = VADInferenceConfig.from_dict(self.inference_kwargs).to_dict()

    @property
    def is_closed(self) -> bool:
        return self._closed

    def _feed(self, waveform: Any) -> tuple[SpeechSegment, ...]:
        import torch

        if self._pending is None:
            buffered = waveform
        else:
            buffered = torch.cat((self._pending, waveform))
        cursor = 0
        while buffered.numel() - cursor >= self._window_shift:
            frame = buffered[cursor:cursor + self._window_shift]
            self._probabilities.extend(self._detector.accept_waveform(frame))
            cursor += self._window_shift
        self._pending = buffered[cursor:].clone()
        emitted = _drain_segments(
            self._detector,
            sample_rate=self.sampling_rate,
        )
        self._segments.extend(emitted)
        return emitted

    def push(self, audio_chunk: Any) -> tuple[SpeechSegment, ...]:
        """Feed one chunk and return newly completed speech segments."""
        from voicehub.processing.waveform import load_native_audio

        with self._lock:
            if self._closed:
                raise RuntimeError("Cannot push audio to a closed stream.")
            if self._result is not None:
                raise RuntimeError("Reset the stream before pushing after flush().")
            chunk = load_native_audio(
                audio_chunk,
                sampling_rate=self.sampling_rate,
                target_sampling_rate=self.sampling_rate,
            )
            self._sample_count += chunk.waveform.numel()
            return self._feed(chunk.waveform)

    def flush(self) -> VADOutput:
        """Pad the final shift, finalize once, and cache the result."""
        import torch

        with self._lock:
            if self._closed:
                raise RuntimeError("Cannot flush a closed stream.")
            if self._result is not None:
                return self._result
            if self._sample_count == 0:
                raise ValueError("Cannot flush a stream that has no audio.")
            if self._pending is not None and self._pending.numel():
                padded = torch.nn.functional.pad(
                    self._pending,
                    (0, self._window_shift - self._pending.numel()),
                )
                self._probabilities.extend(self._detector.accept_waveform(padded))
                self._pending = None
            self._detector.flush()
            self._segments.extend(_drain_segments(
                self._detector,
                sample_rate=self.sampling_rate,
            ))
            duration = self._sample_count / self.sampling_rate
            segments = _finalize_segments(
                self._segments,
                duration=duration,
                speech_pad_ms=self.inference_kwargs["speech_pad_ms"],
                max_speech_duration_s=self.inference_kwargs.get("max_speech_duration_s"),
            )
            self._result = VADOutput(
                segments=segments,
                duration=duration,
                sample_rate=self.sampling_rate,
                probabilities=(
                    tuple(self._probabilities) if self.inference_kwargs["return_frames"] else None),
                metadata=self.model._output_metadata(window_size_samples=self._window_shift),
            )
            return self._result

    def reset(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("Cannot reset a closed stream.")
            self._detector.reset()
            self._pending = None
            self._segments.clear()
            self._probabilities.clear()
            self._sample_count = 0
            self._result = None

    def close(self) -> None:
        with self._lock:
            self._detector.reset()
            self._pending = None
            self._segments.clear()
            self._probabilities.clear()
            self._closed = True

    def __enter__(self) -> SherpaONNXVADSession:
        if self._closed:
            raise RuntimeError("Cannot re-enter a closed stream.")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


class SherpaONNXVADForVoiceActivityDetection(PreTrainedVADModel):
    """Native TEN/Silero graph with the historical provider API."""

    config_class = SherpaONNXVADConfig
    default_model_name_or_path = "safestack/silero-vad"
    architecture_family = "frame-classification"
    native_checkpoint_format = "voicehub-native-ten-vad-v1"
    training_support = "native"
    supports_generic_finetuning = True

    def __init__(
        self,
        config: SherpaONNXVADConfig | str | Path | None = None,
        *,
        model_path: str | Path | None = None,
        device: str = "auto",
        lazy_load: bool = True,
        token: str | bool | None = None,
        trust_onnx_checkpoint: bool = False,
        **kwargs: Any,
    ) -> None:
        if token is not None and not isinstance(token, (str, bool)):
            raise TypeError("`token` must be a string, boolean, or None.")
        if isinstance(token, str) and not token.strip():
            raise ValueError("String `token` values must be non-empty.")
        if not isinstance(trust_onnx_checkpoint, bool):
            raise TypeError("`trust_onnx_checkpoint` must be a boolean.")
        self._hub_token = token
        self._trust_onnx_checkpoint = trust_onnx_checkpoint
        self.native_config: Any | None = None
        self.runtime: _NativeVADRuntime | None = None
        config = self._coerce_config(config, model_path=model_path, **kwargs)
        lifecycle_device = "cuda" if config.provider == "cuda" else "cpu"
        if device == "auto":
            device = lifecycle_device
        if device.partition(":")[0].lower() != lifecycle_device:
            raise ValueError(
                "`device` is incompatible with the native provider: "
                f"provider={config.provider!r}, device={device!r}.")
        super().__init__(config, device=device, lazy_load=lazy_load)

    @staticmethod
    def _resolve_device(device: str) -> str:
        return resolve_cpu_cuda_device(
            device,
            provider="native Sherpa-compatible VAD",
        )

    def _load_silero(self) -> None:
        from voicehub.models.vad_silero.artifacts import (
            DEFAULT_SILERO_VAD_REPOSITORY,
            load_silero_vad_checkpoint,
            resolve_silero_vad_artifact,
        )
        from voicehub.models.vad_silero.configuration import DEFAULT_SILERO_VAD_REVISION
        from voicehub.models.vad_silero.native.configuration import SileroVADConfig as NativeSileroVADConfig
        from voicehub.models.vad_silero.native.modeling import SileroVADModel

        source = self.config.name_or_path or DEFAULT_SILERO_VAD_REPOSITORY
        if source in {"csukuangfj/vad", "sherpa-onnx-vad", "sherpa-vad"}:
            source = DEFAULT_SILERO_VAD_REPOSITORY
        filename = self.config.model_filename
        checkpoint_filename = (
            None if filename in {"silero_vad.onnx", "silero_vad_v5.onnx"} else "/".join(
                part for part in (self.config.subfolder, filename) if part))
        source_path = Path(source).expanduser()
        if (source_path.is_file() and source_path.suffix.lower() == ".onnx"):
            raise ValueError(
                "Sherpa Silero ONNX graphs are no longer executed. Select "
                "the verified native Safetensors/JIT weight artifact instead.")
        revision = self.config.revision
        if revision is None and source == DEFAULT_SILERO_VAD_REPOSITORY:
            revision = DEFAULT_SILERO_VAD_REVISION
        native_config = NativeSileroVADConfig(sampling_rate=self.sample_rate)
        artifact = resolve_silero_vad_artifact(
            source,
            sample_rate=self.sample_rate,
            checkpoint_filename=checkpoint_filename,
            cache_dir=self.config.cache_dir,
            revision=revision,
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
        self.native_config = native_config
        self.runtime = _NativeVADRuntime(
            architecture="silero-vad",
            checkpoint=artifact.checkpoint,
            checkpoint_format=checkpoint_format,
            checkpoint_adapter=adapter,
            revision=artifact.revision,
        )
        self.model = model

    def _load_ten(self) -> None:
        import torch

        from voicehub.checkpointing import SafeTensorReader
        from voicehub.models.vad_sherpa_onnx.artifacts import resolve_ten_vad_artifacts
        from voicehub.models.vad_ten.native.checkpoint import NATIVE_TEN_VAD_FORMAT, TENVADSafeTensorsCheckpointAdapter
        from voicehub.models.vad_ten.native.configuration import TENVADConfig
        from voicehub.models.vad_ten.native.modeling import TENVADModel

        source = self.config.name_or_path
        if not source or source == self.default_model_name_or_path:
            raise ValueError(
                "TEN VAD requires a local/native artifact or reviewed "
                "`ten-vad.onnx` source; no TEN checkpoint is bundled with "
                "the default Silero repository.")
        artifacts = resolve_ten_vad_artifacts(
            source,
            model_filename=self.config.model_filename,
            subfolder=self.config.subfolder,
            revision=self.config.revision,
            cache_dir=self.config.cache_dir,
            token=self._hub_token,
            local_files_only=self.config.local_files_only,
            trust_onnx_checkpoint=self._trust_onnx_checkpoint,
            window_size=self.config.window_size_samples,
        )
        values = read_json_file(artifacts.config)
        native_config = TENVADConfig.from_dict(values)
        model = TENVADModel(native_config)
        adapter = TENVADSafeTensorsCheckpointAdapter()
        with SafeTensorReader(artifacts.checkpoint) as reader:
            declared = reader.metadata.get("format")
            if declared != NATIVE_TEN_VAD_FORMAT:
                raise ValueError(
                    "TEN Safetensors must declare the native "
                    f"{NATIVE_TEN_VAD_FORMAT!r} format.")
            report = adapter.load_streaming(
                model,
                reader,
                values,
                strict=True,
            )
        model.to(device=self.device, dtype=torch.float32)
        self.native_config = native_config
        self.runtime = _NativeVADRuntime(
            architecture="ten-vad",
            checkpoint=artifacts.checkpoint,
            checkpoint_format=NATIVE_TEN_VAD_FORMAT,
            checkpoint_adapter=report.adapter,
            revision=artifacts.revision,
            converted_from_onnx=artifacts.converted_from_onnx,
        )
        self.model = model

    def _load_pretrained_model(self) -> None:
        if self.config.model_family == "silero":
            self._load_silero()
        else:
            self._load_ten()

    def _window_shift(self) -> int:
        if self.native_config is None:
            raise RuntimeError("VAD must be loaded before creating a stream.")
        return (
            self.native_config.frame_size
            if self.config.model_family == "silero" else self.native_config.window_size)

    def _create_detector(
        self,
        *,
        threshold: float = 0.5,
        onset: float | None = None,
        offset: float | None = None,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30,
        max_speech_duration_s: float | None = None,
        window_size_samples: int | None = None,
        return_frames: bool = False,
    ) -> Any:
        del speech_pad_ms, return_frames
        from voicehub.models.vad_sherpa_onnx.streaming import (
            NativeSherpaVoiceActivityDetector,
            NativeSileroScorer,
            NativeTENScorer,
        )

        expected = self._window_shift()
        if window_size_samples is not None and window_size_samples != expected:
            raise ValueError(
                f"The native {self.config.model_family} graph requires a "
                f"{expected}-sample shift.")
        if self.config.model_family == "ten" and offset is not None:
            raise ValueError("TEN VAD does not expose a separate offset threshold.")
        scorer = (
            NativeSileroScorer(self.model) if self.config.model_family == "silero" else NativeTENScorer(
                self.model))
        return NativeSherpaVoiceActivityDetector(
            scorer,
            family=self.config.model_family,
            sample_rate=self.sample_rate,
            threshold=threshold if onset is None else onset,
            negative_threshold=offset,
            min_speech_duration=min_speech_duration_ms / 1_000,
            min_silence_duration=min_silence_duration_ms / 1_000,
            max_speech_duration=(
                self.config.buffer_size_s if max_speech_duration_s is None else max_speech_duration_s),
        )

    def _output_metadata(
        self,
        *,
        window_size_samples: int,
    ) -> dict[str, Any]:
        if self.runtime is None:
            raise RuntimeError("VAD runtime metadata is unavailable before load.")
        score_window = (
            self.native_config.frame_size + self.native_config.context_size
            if self.config.model_family == "silero" else self.native_config.window_size)
        return {
            "backend": "voicehub-native",
            "compatibility": "sherpa-onnx-segmentation",
            "model_family": self.config.model_family,
            "architecture": self.runtime.architecture,
            "checkpoint_path": str(self.runtime.checkpoint),
            "checkpoint_format": self.runtime.checkpoint_format,
            "checkpoint_adapter": self.runtime.checkpoint_adapter,
            "checkpoint_revision": self.runtime.revision,
            "converted_from_onnx": self.runtime.converted_from_onnx,
            "provider": self.config.provider,
            "window_size_samples": window_size_samples,
            "score_window_size_samples": score_window,
            "frame_scores_available": True,
            "streaming": True,
        }

    def _detect(
        self,
        audio: Any,
        *,
        sampling_rate: int | None = None,
        **options: Any,
    ) -> VADOutput:
        from voicehub.processing.waveform import load_native_audio

        materialized = load_native_audio(
            audio,
            sampling_rate=sampling_rate,
            target_sampling_rate=self.sample_rate,
        )
        session = SherpaONNXVADSession(
            self,
            sampling_rate=materialized.sampling_rate,
            inference_kwargs=options,
        )
        with session:
            session.push(materialized.waveform)
            return session.flush()

    def stream(
        self,
        *,
        sampling_rate: int,
        **inference_kwargs: Any,
    ) -> SherpaONNXVADSession:
        return SherpaONNXVADSession(
            self,
            sampling_rate=sampling_rate,
            inference_kwargs=inference_kwargs,
        )

    def prepare_training_inputs(
        self,
        inputs: dict[str, Any],
        *,
        phase: str,
    ) -> dict[str, Any]:
        del phase
        if self.model is None:
            self.load_for_training()
        if self.config.model_family == "silero":
            from voicehub.models.vad_silero.training_vad_silero import prepare_silero_vad_training_batch

            return prepare_silero_vad_training_batch(self, inputs)
        from voicehub.models.vad_sherpa_onnx.training_vad_sherpa_onnx import prepare_ten_vad_training_batch

        return prepare_ten_vad_training_batch(self, inputs)

    def _validate_training_runtime(self) -> None:
        return None

    def _save_pretrained(self, save_directory: Path) -> None:
        from voicehub.checkpointing import save_safetensors

        if self.model is None or self.native_config is None:
            self.load()
        save_directory.mkdir(parents=True, exist_ok=True)
        if self.config.model_family == "ten":
            from voicehub.models.vad_ten.native.checkpoint import NATIVE_TEN_VAD_FILENAME, NATIVE_TEN_VAD_FORMAT

            filename = NATIVE_TEN_VAD_FILENAME
            checkpoint_format = NATIVE_TEN_VAD_FORMAT
            architecture = "ten-vad"
        else:
            from voicehub.models.vad_silero.artifacts import NATIVE_SILERO_VAD_FILENAME, NATIVE_SILERO_VAD_FORMAT

            filename = NATIVE_SILERO_VAD_FILENAME
            checkpoint_format = NATIVE_SILERO_VAD_FORMAT
            architecture = "silero-vad"
        save_safetensors(
            self.model.state_dict(),
            save_directory / filename,
            metadata={
                "format": checkpoint_format,
                "architecture": architecture,
                "sample_rate": str(self.sample_rate),
            },
        )
        # TEN's frame shift is a native graph/frontend setting rather than a
        # provider-only option. Preserve the complete native configuration so
        # non-default reviewed window sizes survive an export/reload cycle.
        values = (self.native_config.to_dict() if self.config.model_family == "ten" else {})
        values.update(self.config.to_dict())
        values.update({
            "name_or_path": str(save_directory),
            "model_filename": filename,
            "checkpoint_format": checkpoint_format,
            "architecture": architecture,
            'architectures': [self.__class__.__name__],
        })
        write_json_file(save_directory / "config.json", values)

    def export_native_pretrained(self, save_directory: str | Path) -> Path:
        destination = Path(save_directory).expanduser()
        self._save_pretrained(destination)
        return destination


SherpaNativeVADSession = SherpaONNXVADSession

__all__ = [
    "SherpaNativeVADSession",
    "SherpaONNXVADForVoiceActivityDetection",
    "SherpaONNXVADSession",
]
