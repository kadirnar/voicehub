"""VoiceHub-native FunASR FSMN voice activity detection provider."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from voicehub.hub import read_json_file, write_json_file
from voicehub.models.audio import PreTrainedVADModel
from voicehub.models.native_utils import resolve_native_device
from voicehub.models.vad_funasr.configuration import FunASRVADConfig
from voicehub.outputs import SpeechSegment, VADOutput
from voicehub.vad_utils import merge_speech_segments


def _postprocess_segments(
    segments: tuple[SpeechSegment, ...],
    *,
    duration: float,
    min_speech_duration_ms: int,
    min_silence_duration_ms: int,
    speech_pad_ms: int,
    max_speech_duration_s: float | None,
) -> tuple[SpeechSegment, ...]:
    minimum_speech = min_speech_duration_ms / 1_000.0
    retained = tuple(segment for segment in segments if segment.end - segment.start >= minimum_speech)
    joined = merge_speech_segments(
        retained,
        max_gap=min_silence_duration_ms / 1_000.0,
    )
    padding = speech_pad_ms / 1_000.0
    padded = tuple(
        SpeechSegment(
            start=max(0.0, segment.start - padding),
            end=min(duration, segment.end + padding),
            score=segment.score,
        ) for segment in joined if min(duration, segment.end + padding) > max(0.0, segment.start - padding))
    merged = merge_speech_segments(padded)
    if max_speech_duration_s is None:
        return merged
    split = []
    for segment in merged:
        cursor = segment.start
        while segment.end - cursor > max_speech_duration_s + 1e-12:
            end = round(cursor + max_speech_duration_s, 12)
            split.append(SpeechSegment(
                start=cursor,
                end=end,
                score=segment.score,
            ), )
            cursor = end
        if segment.end - cursor > 1e-12:
            split.append(SpeechSegment(
                start=cursor,
                end=segment.end,
                score=segment.score,
            ), )
    return tuple(split)


class FunASRVADForVoiceActivityDetection(PreTrainedVADModel):
    """Run and fine-tune FSMN VAD without importing FunASR or torchaudio."""

    config_class = FunASRVADConfig
    default_model_name_or_path = "funasr/fsmn-vad"
    architecture_family = "frame-classification"
    training_support = "native"
    supports_generic_finetuning = True

    def __init__(
        self,
        config: FunASRVADConfig | str | Path | None = None,
        *,
        model_path: str | Path | None = None,
        device: str = "auto",
        lazy_load: bool = True,
        token: str | bool | None = None,
        trust_pickle_checkpoint: bool = False,
        **kwargs: Any,
    ) -> None:
        if token is not None and not isinstance(token, (str, bool)):
            raise TypeError("`token` must be a string, boolean, or None.")
        if isinstance(token, str) and not token.strip():
            raise ValueError("String `token` values must be non-empty.")
        if not isinstance(trust_pickle_checkpoint, bool):
            raise TypeError("`trust_pickle_checkpoint` must be a boolean.")
        self._hub_token = token
        self._trust_pickle_checkpoint = trust_pickle_checkpoint
        self.artifacts: Any | None = None
        self.native_config: Any | None = None
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
        return resolve_native_device(
            device,
            provider="native FunASR FSMN VAD",
            supported_types=("cpu", "cuda", "mps"),
        )

    def _load_pretrained_model(self) -> None:
        import torch

        from voicehub.checkpointing import SafeTensorReader
        from voicehub.models.vad_funasr.artifacts import resolve_fsmn_vad_artifacts
        from voicehub.models.vad_funasr.native.checkpoint import FSMNVADSafeTensorsCheckpointAdapter
        from voicehub.models.vad_funasr.native.configuration import FSMNVADConfig
        from voicehub.models.vad_funasr.native.modeling import FSMNVADModel

        source = self.config.name_or_path or self.default_model_name_or_path
        artifacts = resolve_fsmn_vad_artifacts(
            source,
            revision=self.config.revision,
            cache_dir=self.config.cache_dir,
            token=self._hub_token,
            local_files_only=self.config.local_files_only,
            trust_pickle_checkpoint=self._trust_pickle_checkpoint,
        )
        values = read_json_file(artifacts.config)
        native_config = FSMNVADConfig.from_dict(values)
        model = FSMNVADModel(native_config)
        adapter = FSMNVADSafeTensorsCheckpointAdapter()
        with SafeTensorReader(artifacts.checkpoint) as reader:
            declared_format = reader.metadata.get("format")
            if (declared_format is not None and declared_format != "voicehub-fsmn-vad-v1"):
                raise ValueError("FSMN VAD Safetensors declares unsupported format "
                                 f"{declared_format!r}.")
            adapter.load_streaming(
                model,
                reader,
                values,
                strict=True,
            )
        model.to(device=self.device, dtype=torch.float32)
        self.config.sample_rate = native_config.sampling_rate
        self.artifacts = artifacts
        self.native_config = native_config
        self.checkpoint_adapter = adapter.qualified_id
        self.model = model

    def _native_inference(
        self,
        waveform: Any,
        *,
        threshold: float,
        min_silence_duration_ms: int,
        max_speech_duration_s: float | None,
    ) -> tuple[Any, tuple[Any, ...]]:
        import torch

        from voicehub.models.vad_funasr.native.inference import FSMNVADDecoder, frame_decibels

        if self.model is None or self.native_config is None:
            raise RuntimeError("Native FSMN VAD runtime is not loaded.")
        parameter = next(self.model.parameters())
        values = waveform.unsqueeze(0).to(
            device=parameter.device,
            dtype=parameter.dtype,
        )
        with torch.inference_mode():
            output = self.model(values)
        speech = output.speech_probabilities[0].detach().float().cpu()
        probabilities = output.probabilities[0].detach().float().cpu()
        silence = probabilities[
            :,
            list(self.native_config.silence_pdf_ids),
        ].sum(dim=-1)
        decibels = frame_decibels(
            waveform,
            config=self.native_config,
            frame_count=speech.numel(),
        )
        decoder = FSMNVADDecoder(
            self.native_config,
            speech_noise_threshold=threshold,
            max_end_silence_ms=min_silence_duration_ms,
            max_single_segment_ms=(
                None if max_speech_duration_s is None else round(max_speech_duration_s * 1_000)),
        )
        boundaries = decoder.process(
            speech,
            silence_probabilities=silence,
            decibels=decibels,
            final=True,
        )
        return speech, boundaries

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

        if window_size_samples is not None:
            expected = (None if self.native_config is None else self.native_config.frame_length_samples)
            if expected is None or window_size_samples != expected:
                raise ValueError(
                    "Native FSMN VAD uses a fixed 25 ms analysis window; "
                    f"received `window_size_samples={window_size_samples}`.")
        effective_threshold = threshold if onset is None else onset
        if offset is not None and not math.isclose(
                offset,
                effective_threshold,
                rel_tol=0.0,
                abs_tol=1e-12,
        ):
            raise ValueError(
                "FSMN VAD exposes one speech/noise threshold and cannot "
                "apply independent `onset` and `offset` values.")
        materialized = load_native_audio(
            audio,
            sampling_rate=sampling_rate,
            target_sampling_rate=self.sample_rate,
        )
        if materialized.waveform.numel() < 400:
            raise ValueError("Native FSMN VAD requires at least 25 ms (400 samples) of audio.")
        speech, boundaries = self._native_inference(
            materialized.waveform,
            threshold=effective_threshold,
            min_silence_duration_ms=min_silence_duration_ms,
            max_speech_duration_s=max_speech_duration_s,
        )
        raw_segments = tuple(
            SpeechSegment(
                start=min(
                    materialized.duration,
                    boundary.start_ms / 1_000.0,
                ),
                end=min(
                    materialized.duration,
                    boundary.end_ms / 1_000.0,
                ),
            ) for boundary in boundaries if boundary.end_ms > boundary.start_ms)
        segments = _postprocess_segments(
            raw_segments,
            duration=materialized.duration,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
            max_speech_duration_s=max_speech_duration_s,
        )
        return VADOutput(
            segments=segments,
            duration=materialized.duration,
            sample_rate=self.sample_rate,
            probabilities=(tuple(float(item) for item in speech.tolist()) if return_frames else None),
            metadata={
                "backend":
                "voicehub-native",
                "architecture":
                "fsmn-vad",
                "source":
                self.config.name_or_path,
                "checkpoint_format":
                "voicehub-fsmn-vad-v1",
                "checkpoint_adapter":
                self.checkpoint_adapter,
                "checkpoint_revision":
                None if self.artifacts is None else self.artifacts.revision,
                "converted_from_pickle":
                (False if self.artifacts is None else self.artifacts.converted_from_pickle),
                "frame_scores_available":
                True,
                "frame_hop_samples":
                160,
                "frame_length_samples":
                400,
                "native_timestamp_unit":
                "milliseconds",
            },
        )

    def stream(
        self,
        *,
        sampling_rate: int,
        **inference_kwargs: Any,
    ) -> Any:
        """Create an incremental session with isolated frontend/FSMN state."""
        from voicehub.models.vad_funasr.streaming import FSMNVADStreamingSession

        if sampling_rate != self.sample_rate:
            raise ValueError("Native FSMN streaming requires 16 kHz input chunks.")
        defaults = self.inference_config.to_dict()
        defaults.update(inference_kwargs)
        self.load()
        return FSMNVADStreamingSession(
            self,
            sampling_rate=sampling_rate,
            inference_kwargs=defaults,
        )

    def prepare_training_inputs(
        self,
        inputs: dict[str, Any],
        *,
        phase: str,
    ) -> dict[str, Any]:
        from voicehub.models.vad_funasr.training_vad_funasr import prepare_fsmn_vad_training_batch

        del phase
        if self.model is None:
            self.load_for_training()
        return prepare_fsmn_vad_training_batch(self, inputs)

    def _validate_training_runtime(self) -> None:
        return None

    def _save_pretrained(self, save_directory: Path) -> None:
        from voicehub.checkpointing import save_safetensors
        from voicehub.models.vad_funasr.native.checkpoint import NATIVE_FSMN_VAD_FILENAME, NATIVE_FSMN_VAD_FORMAT

        if self.model is None or self.native_config is None:
            self.load()
        save_directory.mkdir(parents=True, exist_ok=True)
        save_safetensors(
            self.model.state_dict(),
            save_directory / NATIVE_FSMN_VAD_FILENAME,
            metadata={
                "format": NATIVE_FSMN_VAD_FORMAT,
                "architecture": "fsmn-vad",
                "sample_rate": str(self.sample_rate),
            },
        )
        values = self.native_config.to_dict()
        values.update({
            "model_type": self.config.model_type,
            'architectures': [self.__class__.__name__],
            "name_or_path": str(save_directory),
            "checkpoint_format": NATIVE_FSMN_VAD_FORMAT,
        })
        write_json_file(save_directory / "config.json", values)

    def export_native_pretrained(
        self,
        save_directory: str | Path,
    ) -> Path:
        destination = Path(save_directory).expanduser()
        self._save_pretrained(destination)
        return destination


__all__ = ["FunASRVADForVoiceActivityDetection"]
