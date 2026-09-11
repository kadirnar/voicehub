"""Request-local incremental streaming for native FSMN VAD."""

from __future__ import annotations

import math
from threading import RLock
from typing import Any


class FSMNVADStreamingSession:
    """Own frontend progress, four FSMN caches, and endpoint state."""

    def __init__(
        self,
        wrapper: Any,
        *,
        sampling_rate: int,
        inference_kwargs: dict[str, Any],
    ) -> None:
        import torch

        from voicehub.models.vad_funasr.native.inference import FSMNVADDecoder

        if wrapper.model is None or wrapper.native_config is None:
            raise RuntimeError("FSMN VAD must be loaded before streaming.")
        if sampling_rate != wrapper.sample_rate:
            raise ValueError("FSMN streaming sample rate must match the model.")
        options = dict(inference_kwargs)
        threshold = options.pop("threshold", 0.5)
        onset = options.pop("onset", None)
        offset = options.pop("offset", None)
        effective_threshold = threshold if onset is None else onset
        if offset is not None and not math.isclose(
                offset,
                effective_threshold,
                rel_tol=0.0,
                abs_tol=1e-12,
        ):
            raise ValueError("FSMN streaming cannot use independent onset/offset thresholds.")
        self.min_speech_duration_ms = options.pop(
            "min_speech_duration_ms",
            250,
        )
        self.min_silence_duration_ms = options.pop(
            "min_silence_duration_ms",
            100,
        )
        self.speech_pad_ms = options.pop("speech_pad_ms", 30)
        self.max_speech_duration_s = options.pop(
            "max_speech_duration_s",
            None,
        )
        window_size = options.pop("window_size_samples", None)
        if (window_size is not None and window_size != wrapper.native_config.frame_length_samples):
            raise ValueError("FSMN streaming uses a fixed 400-sample analysis window.")
        self.return_frames = options.pop("return_frames", False)
        if options:
            raise ValueError("Unsupported FSMN streaming option(s): "
                             f"{', '.join(sorted(options))}.")
        self.wrapper = wrapper
        self.sampling_rate = sampling_rate
        self._threshold = effective_threshold
        self._decoder = FSMNVADDecoder(
            wrapper.native_config,
            speech_noise_threshold=effective_threshold,
            max_end_silence_ms=self.min_silence_duration_ms,
            max_single_segment_ms=(
                None if self.max_speech_duration_s is None else round(self.max_speech_duration_s * 1_000)),
        )
        self._waveform = torch.empty(0, dtype=torch.float32)
        self._encoder_cache: dict[str, Any] = {}
        self._emitted_frames = 0
        self._probabilities: list[Any] = []
        self._result: Any | None = None
        self._closed = False
        self._lock = RLock()

    @property
    def is_closed(self) -> bool:
        with self._lock:
            return self._closed

    def _process(self, *, final: bool) -> tuple[float, ...]:
        import torch

        from voicehub.models.vad_funasr.native.inference import frame_decibels

        model = self.wrapper.model
        features = model.frontend(
            self._waveform.unsqueeze(0),
            final=final,
        )
        if features.shape[1] < self._emitted_frames:
            raise RuntimeError("FSMN frontend frame count moved backwards.")
        new_features = features[:, self._emitted_frames:]
        emitted: list[float] = []
        if new_features.shape[1]:
            parameter = next(model.parameters())
            with torch.inference_mode():
                output = model(
                    features=new_features.to(
                        device=parameter.device,
                        dtype=parameter.dtype,
                    ),
                    cache=self._encoder_cache,
                )
            speech = output.speech_probabilities[0].detach().float().cpu()
            all_pdf = output.probabilities[0].detach().float().cpu()
            silence = all_pdf[
                :,
                list(self.wrapper.native_config.silence_pdf_ids),
            ].sum(dim=-1)
            all_decibels = frame_decibels(
                self._waveform,
                config=self.wrapper.native_config,
                frame_count=features.shape[1],
            )
            decibels = all_decibels[self._emitted_frames:]
            self._decoder.process(
                speech,
                silence_probabilities=silence,
                decibels=decibels,
                final=final,
            )
            for value in speech:
                item = value.reshape(1)
                self._probabilities.append(item)
                emitted.append(float(item.item()))
            self._emitted_frames = features.shape[1]
        elif final:
            self._decoder.process(
                torch.empty(0),
                final=True,
            )
        return tuple(emitted)

    def push(self, audio_chunk: Any) -> tuple[float, ...]:
        """Consume stable frames and retain only unresolved LFR right
        context."""
        import torch

        from voicehub.processing.waveform import load_native_audio

        with self._lock:
            if self._closed:
                raise RuntimeError("Cannot push audio to a closed stream.")
            if self._result is not None:
                raise RuntimeError("Reset the stream before pushing after flush().")
            materialized = load_native_audio(
                audio_chunk,
                sampling_rate=self.sampling_rate,
                target_sampling_rate=self.sampling_rate,
            )
            self._waveform = torch.cat((self._waveform, materialized.waveform.detach().float().cpu()), )
            if (self._waveform.numel() < self.wrapper.native_config.frame_length_samples):
                return ()
            return self._process(final=False)

    def flush(self):
        """Finalize right-context padding and return one normalized output."""
        import torch

        from voicehub.models.vad_funasr.modeling import _postprocess_segments
        from voicehub.outputs import SpeechSegment, VADOutput

        with self._lock:
            if self._result is not None:
                return self._result
            if self._waveform.numel() == 0:
                raise ValueError("Cannot flush a stream that has no audio.")
            if (self._waveform.numel() < self.wrapper.native_config.frame_length_samples):
                raise ValueError("FSMN streaming requires at least 25 ms of audio.")
            self._process(final=True)
            duration = self._waveform.numel() / self.sampling_rate
            raw = tuple(
                SpeechSegment(
                    start=min(duration, boundary.start_ms / 1_000.0),
                    end=min(duration, boundary.end_ms / 1_000.0),
                ) for boundary in self._decoder.boundaries if boundary.end_ms > boundary.start_ms)
            segments = _postprocess_segments(
                raw,
                duration=duration,
                min_speech_duration_ms=self.min_speech_duration_ms,
                min_silence_duration_ms=self.min_silence_duration_ms,
                speech_pad_ms=self.speech_pad_ms,
                max_speech_duration_s=self.max_speech_duration_s,
            )
            probabilities = torch.cat(self._probabilities, ) if self._probabilities else torch.empty(0)
            self._result = VADOutput(
                segments=segments,
                duration=duration,
                sample_rate=self.sampling_rate,
                probabilities=(
                    tuple(float(item) for item in probabilities.tolist()) if self.return_frames else None),
                metadata={
                    "backend": "voicehub-native",
                    "architecture": "fsmn-vad",
                    "streaming": True,
                    "frame_scores_available": True,
                    "frame_hop_samples": 160,
                    "frame_length_samples": 400,
                    "checkpoint_adapter": self.wrapper.checkpoint_adapter,
                },
            )
            return self._result

    def reset(self) -> None:
        import torch

        from voicehub.models.vad_funasr.native.inference import FSMNVADDecoder

        with self._lock:
            if self._closed:
                raise RuntimeError("Cannot reset a closed stream.")
            self._decoder = FSMNVADDecoder(
                self.wrapper.native_config,
                speech_noise_threshold=self._threshold,
                max_end_silence_ms=self.min_silence_duration_ms,
                max_single_segment_ms=(
                    None if self.max_speech_duration_s is None else round(self.max_speech_duration_s *
                                                                          1_000)),
            )
            self._waveform = torch.empty(0, dtype=torch.float32)
            self._encoder_cache.clear()
            self._emitted_frames = 0
            self._probabilities.clear()
            self._result = None

    def close(self) -> None:
        with self._lock:
            self._waveform = self._waveform.new_empty(0)
            self._encoder_cache.clear()
            self._probabilities.clear()
            self._closed = True

    def __enter__(self):
        if self.is_closed:
            raise RuntimeError("Cannot enter a closed FSMN stream.")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


__all__ = ["FSMNVADStreamingSession"]
