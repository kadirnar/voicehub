"""WhisperX-compatible transcription with VoiceHub-owned model graphs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from voicehub.modeling_outputs import ASROutput, ASRSegment, ASRWord
from voicehub.models.asr_native.configuration import WhisperXConfig
from voicehub.models.asr_native.whisper_compat import normalize_whisper_source
from voicehub.models.asr_whisper_native.modeling_asr_whisper_native import WhisperForSpeechRecognition


class WhisperXForSpeechRecognition(WhisperForSpeechRecognition):
    """Compose native Whisper transcription and native Wav2Vec2 alignment.

    WhisperX is a pipeline rather than a separately trainable acoustic
    model. The primary graph is VoiceHub's Whisper implementation and
    remains fully fine-tunable.  When word alignment is requested, a
    separately versioned VoiceHub Wav2Vec2 CTC graph supplies emissions
    to the pinned WhisperX dynamic-programming algorithm.
    """

    config_class = WhisperXConfig
    default_model_name_or_path = "openai/whisper-small"

    def __init__(
        self,
        config: WhisperXConfig | str | Path | None = None,
        *,
        model_path: str | Path | None = None,
        device: str = "auto",
        lazy_load: bool = True,
        token: str | bool | None = None,
        **kwargs: Any,
    ) -> None:
        if isinstance(config, WhisperXConfig):
            values = config.to_dict()
            values["name_or_path"] = normalize_whisper_source(values.get("name_or_path", ""))
            config = WhisperXConfig.from_dict(values)
        elif isinstance(config, (str, Path)):
            config = normalize_whisper_source(config)
        if model_path is not None:
            model_path = normalize_whisper_source(model_path)
        self._alignment_runtime: Any | None = None
        self._alignment_source: str | None = None
        super().__init__(
            config,
            model_path=model_path,
            device=device,
            lazy_load=lazy_load,
            token=token,
            **kwargs,
        )

    def _alignment_model_source(self, language: str) -> str:
        from voicehub.architectures.ctc_alignment.metadata import DEFAULT_ALIGNMENT_MODELS

        configured = self.config.alignment_model_path
        if configured is not None:
            return configured
        try:
            return DEFAULT_ALIGNMENT_MODELS[language]
        except KeyError as error:
            raise ValueError(
                "No declarative native CTC alignment checkpoint is known for "
                f"language {language!r}. Set `alignment_model_path` to a "
                "compatible Wav2Vec2 CTC Safetensors artifact.") from error

    def _load_alignment_model(self, language: str):
        from voicehub.models.asr_wav2vec2 import Wav2Vec2ASRConfig, Wav2Vec2ForSpeechRecognition

        source = self._alignment_model_source(language)
        if (self._alignment_runtime is not None and self._alignment_source == source):
            return self._alignment_runtime
        runtime = Wav2Vec2ForSpeechRecognition(
            Wav2Vec2ASRConfig(
                name_or_path=source,
                revision=self.config.alignment_revision,
                cache_dir=self.config.alignment_cache_dir,
                local_files_only=self.config.alignment_local_files_only,
                torch_dtype=self.config.alignment_torch_dtype,
            ),
            device=self.device,
            lazy_load=False,
            token=self._hub_token,
        )
        self._alignment_runtime = runtime
        self._alignment_source = source
        return runtime

    @staticmethod
    def _alignment_language(
        output: ASROutput,
        requested: str | None,
        *,
        multilingual: bool,
    ) -> str:
        language = output.language or requested
        if language is None and not multilingual:
            language = "en"
        if language is None:
            raise ValueError("CTC word alignment requires a detected or requested "
                             "language.")
        return language.lower()

    @staticmethod
    def _ctc_emission(runtime: Any, waveform: Any):
        import torch

        if (runtime.model is None or runtime.native_config is None or runtime.ctc_processor is None):
            raise RuntimeError("Native Wav2Vec2 alignment runtime is not loaded.")
        if (not isinstance(waveform, torch.Tensor) or waveform.ndim != 1 or waveform.numel() == 0 or
                not waveform.is_floating_point()):
            raise ValueError("Alignment audio must be a non-empty floating-point mono tensor.")
        # WhisperX feeds the alignment model the raw segment samples: neither
        # its torchaudio bundles nor its Transformers path apply the
        # processor's zero-mean/unit-variance normalization.
        input_values = waveform.float().reshape(1, -1)
        minimum = runtime.native_config.minimum_input_samples
        if input_values.shape[-1] < minimum:
            # Like WhisperX, zero-pad segments shorter than the convolutional
            # receptive field and align against every emitted frame.
            input_values = torch.nn.functional.pad(
                input_values,
                (0, minimum - input_values.shape[-1]),
            )
        attention_mask = torch.ones_like(input_values, dtype=torch.bool)
        parameter = next(runtime.model.parameters())
        with torch.inference_mode():
            result = runtime.model(
                input_values.to(
                    device=parameter.device,
                    dtype=parameter.dtype,
                ),
                attention_mask=attention_mask.to(parameter.device),
            )
        return result.logits[0].float().log_softmax(dim=-1)

    def _align_segments(
        self,
        output: ASROutput,
        *,
        waveform: Any,
        language: str,
    ) -> tuple[ASRSegment, ...]:
        from voicehub.architectures.ctc_alignment import align_ctc_transcript

        runtime = self._load_alignment_model(language)
        tokenizer = runtime.ctc_processor.tokenizer
        segments: list[ASRSegment] = []
        for segment in output.segments:
            if (not segment.text or segment.start is None or segment.end is None or
                    segment.end <= segment.start):
                segments.append(segment)
                continue
            # WhisperX truncates segment times to sample indices.
            start_sample = max(0, int(segment.start * 16_000))
            end_sample = min(
                waveform.numel(),
                int(segment.end * 16_000),
            )
            if end_sample <= start_sample:
                segments.append(segment)
                continue
            emission = self._ctc_emission(
                runtime,
                waveform[start_sample:end_sample],
            )
            alignment = align_ctc_transcript(
                emission,
                segment.text,
                tokenizer.vocabulary,
                blank_id=tokenizer.pad_token_id,
                word_delimiter_token=tokenizer.word_delimiter_token,
                language=language,
                segment_start=segment.start,
                segment_end=segment.end,
            )
            words = tuple(
                ASRWord(
                    text=word.text,
                    start=word.start,
                    end=word.end,
                    confidence=word.confidence,
                ) for word in alignment.words)
            metadata = dict(segment.metadata)
            metadata["ctc_aligned"] = bool(words)
            # WhisperX reports aligned segments from their first aligned
            # character to their last one.
            segments.append(
                ASRSegment(
                    text=segment.text,
                    start=words[0].start if words else segment.start,
                    end=words[-1].end if words else segment.end,
                    confidence=segment.confidence,
                    language=segment.language,
                    speaker=segment.speaker,
                    words=words,
                    metadata=metadata,
                ))
        return tuple(segments)

    def _transcribe_alignment_windows(
        self,
        materialized: Any,
        **options: Any,
    ) -> ASROutput:
        """Decode Whisper windows the way WhisperX does before alignment.

        WhisperX transcribes each speech chunk of at most 30 seconds
        without timestamp tokens and aligns the decoded text against the
        whole chunk; Whisper's own timestamps are never used. Without a
        VAD, each decoding window is one such chunk, so requesting word
        alignment does not change the transcript.
        """
        if self.native_config is None:
            raise RuntimeError("Whisper runtime is not loaded.")
        maximum_chunk_seconds = (self.native_config.expected_input_frames * 160 / 16_000)
        chunk_length_s = options.get("chunk_length_s")
        resolved_chunk_seconds = (maximum_chunk_seconds if chunk_length_s is None else float(chunk_length_s))
        if resolved_chunk_seconds > maximum_chunk_seconds:
            raise ValueError(f"Whisper chunks cannot exceed {maximum_chunk_seconds:g} "
                             "seconds.")
        chunk_samples = max(1, round(resolved_chunk_seconds * 16_000))
        waveform = materialized.waveform
        texts: list[str] = []
        segments: list[ASRSegment] = []
        detected_language: str | None = None
        metadata: dict[str, Any] = {}
        for start in range(0, waveform.numel(), chunk_samples):
            stop = min(start + chunk_samples, waveform.numel())
            window = super()._transcribe(
                waveform[start:stop],
                sampling_rate=16_000,
                return_timestamps=False,
                **options,
            )
            if detected_language is None:
                detected_language = window.language
            metadata = metadata or dict(window.metadata)
            texts.append(window.text)
            if window.text.strip():
                segments.append(
                    ASRSegment(
                        text=window.text.strip(),
                        start=start / 16_000,
                        end=stop / 16_000,
                        language=window.language,
                    ))
        return ASROutput(
            text=self._join_chunk_text(texts),
            segments=tuple(segments),
            language=detected_language,
            duration=materialized.duration,
            metadata=metadata,
        )

    def _transcribe(
        self,
        audio: Any,
        *,
        sampling_rate: int | None = None,
        language: str | None = None,
        task: str = "transcribe",
        return_timestamps: bool | str = False,
        chunk_length_s: float | None = None,
        stride_length_s: float | tuple[float, float] | None = None,
        batch_size: int | None = None,
        num_beams: int | None = None,
        max_new_tokens: int | None = None,
        hotwords: str | tuple[str, ...] | list[str] | None = None,
    ) -> ASROutput:
        from voicehub.architectures.ctc_alignment.metadata import WHISPERX_REVISION
        from voicehub.processing.waveform import load_native_audio

        should_align = self.config.align_output or return_timestamps == "word"
        options = {
            "language": language,
            "task": task,
            "chunk_length_s": chunk_length_s,
            "stride_length_s": stride_length_s,
            "batch_size": batch_size,
            "num_beams": num_beams,
            "max_new_tokens": max_new_tokens,
            "hotwords": hotwords,
        }
        if should_align:
            materialized = load_native_audio(
                audio,
                sampling_rate=sampling_rate,
                target_sampling_rate=16_000,
            )
            base_output = self._transcribe_alignment_windows(
                materialized,
                **options,
            )
        else:
            base_output = super()._transcribe(
                audio,
                sampling_rate=sampling_rate,
                return_timestamps=return_timestamps,
                **options,
            )
        metadata = dict(base_output.metadata)
        metadata.update({
            "alignment_requested": should_align,
            "alignment_revision": WHISPERX_REVISION,
            "pipeline": "voicehub-native-whisperx",
        })
        if not should_align:
            metadata["aligned"] = False
            return ASROutput(
                text=base_output.text,
                segments=base_output.segments,
                language=base_output.language,
                duration=base_output.duration,
                metadata=metadata,
            )
        if self.generation_adapter is None:
            raise RuntimeError("Native Whisper generation runtime is not loaded.")
        resolved_language = self._alignment_language(
            base_output,
            language,
            multilingual=self.generation_adapter.token_set.is_multilingual,
        )
        segments = self._align_segments(
            base_output,
            waveform=materialized.waveform,
            language=resolved_language,
        )
        aligned_count = sum(bool(segment.words) for segment in segments)
        metadata.update({
            "aligned": aligned_count > 0,
            "aligned_segments": aligned_count,
            "alignment_backend": "voicehub-native-wav2vec2-ctc",
            "alignment_checkpoint": self._alignment_source,
        })
        return ASROutput(
            text=base_output.text,
            segments=segments,
            language=base_output.language or resolved_language,
            duration=base_output.duration,
            metadata=metadata,
        )


__all__ = ["WhisperXForSpeechRecognition"]
