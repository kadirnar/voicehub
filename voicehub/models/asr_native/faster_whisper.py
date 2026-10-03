"""Faster-whisper compatibility provider using VoiceHub's native graph."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from voicehub.modeling_outputs import ASROutput, ASRSegment
from voicehub.models.asr_native.configuration import FasterWhisperConfig
from voicehub.models.asr_native.whisper_compat import normalize_whisper_source
from voicehub.models.asr_whisper_native.modeling_asr_whisper_native import WhisperForSpeechRecognition

# The CTranslate2 repositories faster-whisper (``faster_whisper/utils.py``
# ``_MODELS``) downloads for its size names, mapped to the Safetensors
# checkpoints they were converted from. Distil-Whisper releases are omitted:
# the native tokenizer rejects their whitespace-stripping added tokens.
FASTER_WHISPER_MODEL_ALIASES = {
    "systran/faster-whisper-tiny.en": "openai/whisper-tiny.en",
    "systran/faster-whisper-tiny": "openai/whisper-tiny",
    "systran/faster-whisper-base.en": "openai/whisper-base.en",
    "systran/faster-whisper-base": "openai/whisper-base",
    "systran/faster-whisper-small.en": "openai/whisper-small.en",
    "systran/faster-whisper-small": "openai/whisper-small",
    "systran/faster-whisper-medium.en": "openai/whisper-medium.en",
    "systran/faster-whisper-medium": "openai/whisper-medium",
    "systran/faster-whisper-large-v1": "openai/whisper-large",
    "systran/faster-whisper-large-v2": "openai/whisper-large-v2",
    "systran/faster-whisper-large-v3": "openai/whisper-large-v3",
    "mobiuslabsgmbh/faster-whisper-large-v3-turbo": "openai/whisper-large-v3-turbo",
}


def normalize_faster_whisper_source(value: str | Path) -> str | Path:
    """Resolve faster-whisper size names and CTranslate2 repository IDs."""
    value = normalize_whisper_source(value)
    if isinstance(value, str) and not Path(value.strip()).expanduser().exists():
        return FASTER_WHISPER_MODEL_ALIASES.get(value.strip().lower(), value)
    return value


# Whisper's encoder consumes two mel frames per output position and emits
# timestamps at 20 ms resolution; one mel frame is 10 ms at 16 kHz.
_INPUT_STRIDE = 2
_FRAMES_PER_SECOND = 100
_TIME_PER_FRAME = 1 / _FRAMES_PER_SECOND


class FasterWhisperForSpeechRecognition(WhisperForSpeechRecognition):
    """Preserve legacy model names without depending on CTranslate2.

    This compatibility provider intentionally uses the canonical native
    Whisper graph. Runtime quantization belongs to VoiceHub's
    optimization layer rather than a second model implementation.

    Transcription follows faster-whisper's ``WhisperModel.transcribe``
    framing (SYSTRAN/faster-whisper ``transcribe.py``): one log-mel
    spectrogram over the whole recording, windows sliced from it and
    zero-padded in the log-mel domain, and, with timestamps, decoding
    resumed from the last complete segment of each window.
    """

    config_class = FasterWhisperConfig
    default_model_name_or_path = "openai/whisper-small"

    def __init__(
        self,
        config: FasterWhisperConfig | str | Path | None = None,
        *,
        model_path: str | Path | None = None,
        device: str = "auto",
        lazy_load: bool = True,
        token: str | bool | None = None,
        **kwargs: Any,
    ) -> None:
        if isinstance(config, FasterWhisperConfig):
            values = config.to_dict()
            values["name_or_path"] = normalize_faster_whisper_source(values.get("name_or_path", ""))
            config = FasterWhisperConfig.from_dict(values)
        elif isinstance(config, (str, Path)):
            config = normalize_faster_whisper_source(config)
        if model_path is not None:
            model_path = normalize_faster_whisper_source(model_path)
        super().__init__(
            config,
            model_path=model_path,
            device=device,
            lazy_load=lazy_load,
            token=token,
            **kwargs,
        )

    def _recording_features(self, waveform: Any) -> Any:
        """Return faster-whisper's ``FeatureExtractor`` log-mel for a recording.

        The waveform is padded with one hop of zeros and normalized with
        one maximum over the whole recording, as upstream does.
        """
        import torch

        padded = torch.nn.functional.pad(waveform.float(), (0, 160))
        return self._feature_operation().process({"waveform": padded})["input_features"]

    def _window_features(self, features: Any, seek: int, frames: int) -> Any:
        """Slice ``frames`` log-mel frames and zero-pad them like ``pad_or_trim``."""
        import torch

        window = features[:, seek:seek + frames]
        window = torch.nn.functional.pad(
            window,
            (0, self.native_config.expected_input_frames - window.shape[-1]),
        )
        return window.unsqueeze(0).to(
            device=self.model.device,
            dtype=next(self.model.parameters()).dtype,
        )

    def _split_window(
        self,
        tokens: Sequence[int],
        *,
        seek: int,
        segment_size: int,
    ) -> tuple[list[tuple[float, float, list[int]]], int]:
        """Split one window's tokens and advance ``seek`` like faster-whisper.

        Returns ``(start, end, tokens)`` segments and the next seek frame.
        """
        timestamp_begin = self.tokenizer.timestamp_begin
        time_precision = _INPUT_STRIDE * _TIME_PER_FRAME
        time_offset = seek * _TIME_PER_FRAME
        single_timestamp_ending = (len(tokens) >= 2 and tokens[-2] < timestamp_begin <= tokens[-1])
        consecutive = [
            index for index in range(1, len(tokens))
            if tokens[index] >= timestamp_begin and tokens[index - 1] >= timestamp_begin
        ]
        segments = []
        if consecutive:
            slices = consecutive + ([len(tokens)] if single_timestamp_ending else [])
            last_slice = 0
            for current_slice in slices:
                sliced = list(tokens[last_slice:current_slice])
                segments.append((
                    time_offset + (sliced[0] - timestamp_begin) * time_precision,
                    time_offset + (sliced[-1] - timestamp_begin) * time_precision,
                    sliced,
                ))
                last_slice = current_slice
            if single_timestamp_ending:
                next_seek = seek + segment_size
            else:
                # Drop the unfinished segment and decode again from its start.
                next_seek = seek + (tokens[last_slice - 1] - timestamp_begin) * _INPUT_STRIDE
        else:
            duration = segment_size * _TIME_PER_FRAME
            timestamps = [token for token in tokens if token >= timestamp_begin]
            if timestamps and timestamps[-1] != timestamp_begin:
                duration = (timestamps[-1] - timestamp_begin) * time_precision
            segments.append((time_offset, time_offset + duration, list(tokens)))
            next_seek = seek + segment_size
        # Upstream would loop forever on a window that does not advance.
        return segments, (next_seek if next_seek > seek else seek + segment_size)

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
        import torch

        from voicehub.architectures.whisper.decoding import WhisperDecodingConfig
        from voicehub.generation import GenerationConfig
        from voicehub.processing.waveform import load_native_audio

        if stride_length_s is not None:
            raise ValueError("faster-whisper framing has no `stride_length_s` option.")
        if batch_size not in (None, 1):
            raise ValueError("One public audio request requires `batch_size=1`.")
        if num_beams not in (None, 1):
            raise ValueError(
                "Native Whisper currently provides greedy decoding and "
                "requires `num_beams=1`.")
        if hotwords is not None:
            raise ValueError(
                "Native Whisper does not expose `hotwords` biasing through "
                "the common ASR option.")
        if return_timestamps == "word":
            raise ValueError(
                "Native Whisper provides segment timestamps; word timestamps "
                "require a separate alignment architecture.")
        if (self.native_config is None or self.tokenizer is None or self.generation_adapter is None):
            raise RuntimeError("Whisper runtime is not loaded.")

        materialized = load_native_audio(
            audio,
            sampling_rate=sampling_rate,
            target_sampling_rate=16_000,
        )
        maximum_frames = self.native_config.expected_input_frames
        maximum_seconds = maximum_frames * _TIME_PER_FRAME
        chunk_seconds = (maximum_seconds if chunk_length_s is None else float(chunk_length_s))
        if not 0 < chunk_seconds <= maximum_seconds:
            raise ValueError(f"Whisper chunks must last between 0 and {maximum_seconds:g} "
                             "seconds.")
        window_frames = max(1, round(chunk_seconds * _FRAMES_PER_SECOND))
        generated_limit = (
            self.native_config.max_target_positions - 4 if max_new_tokens is None else max_new_tokens)
        multilingual = self.generation_adapter.token_set.is_multilingual
        resolved_language = self._normalized_language(language)

        features = self._recording_features(materialized.waveform)
        content_frames = features.shape[-1] - 1
        if resolved_language is None and multilingual:
            # Like faster-whisper, detect the language once, on the first
            # window of the recording, and reuse it for every window.
            with torch.inference_mode():
                encoded = self.model.encode(self._window_features(features, 0, window_frames))
                language_id = self.generation_adapter._detect_languages(
                    encoded,
                    encoder_attention_mask=None,
                )[0]
            resolved_language = next(
                code for code, token_id in self.generation_adapter.token_set.language_tokens.items()
                if token_id == language_id)

        texts: list[str] = []
        segments: list[ASRSegment] = []
        seek = 0
        while seek < content_frames:
            segment_size = min(window_frames, content_frames - seek)
            decoding = WhisperDecodingConfig(
                generation=GenerationConfig(
                    max_new_tokens=generated_limit,
                    do_sample=False,
                    eos_token_id=self.tokenizer.eot,
                    pad_token_id=self.tokenizer.eot,
                    use_cache=True,
                ),
                task=task,
                language=resolved_language if multilingual else None,
                return_timestamps=bool(return_timestamps),
                suppress_tokens=tuple(self._generation_values.get("suppress_tokens", ())),
            )
            generated = self.generation_adapter.generate(
                self._window_features(features, seek, segment_size),
                config=decoding,
            )
            tokens = generated.generated_sequences[0].tolist()
            if self.tokenizer.eot in tokens:
                tokens = tokens[:tokens.index(self.tokenizer.eot)]
            window_segments, seek = self._split_window(
                tokens,
                seek=seek,
                segment_size=segment_size,
            )
            for start, end, segment_tokens in window_segments:
                text = self.tokenizer.decode(segment_tokens, skip_special_tokens=True)
                if start == end or not text.strip():
                    continue
                texts.append(text)
                if return_timestamps:
                    segments.append(ASRSegment(text=text, start=start, end=end, language=resolved_language))

        return ASROutput(
            text="".join(texts).strip(),
            segments=tuple(segments),
            language=resolved_language,
            duration=materialized.duration,
            metadata={
                "architecture": "whisper",
                "backend": "voicehub-native",
                "framing": "faster-whisper",
                "checkpoint_revision": (None if self.artifacts is None else self.artifacts.revision),
            },
        )


__all__ = [
    "FASTER_WHISPER_MODEL_ALIASES",
    "FasterWhisperForSpeechRecognition",
    "normalize_faster_whisper_source",
]
