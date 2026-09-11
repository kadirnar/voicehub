"""VoiceHub-native SpeechBrain CRDNN voice activity detection provider."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from voicehub.hub import read_json_file, write_json_file
from voicehub.models.audio import PreTrainedVADModel
from voicehub.models.native_utils import resolve_cpu_cuda_device
from voicehub.models.vad_speechbrain.configuration import SpeechBrainVADConfig
from voicehub.outputs import SpeechSegment, VADOutput
from voicehub.vad_utils import merge_speech_segments, normalize_backend_segments


def _finalize_segments(
    values: Any,
    *,
    duration: float,
    sample_rate: int,
    speech_pad_ms: int,
    max_speech_duration_s: float | None,
) -> tuple[SpeechSegment, ...]:
    """Normalize, pad, merge, and safely split provider intervals."""
    normalized = normalize_backend_segments(values, sampling_rate=sample_rate)
    padding = speech_pad_ms / 1_000.0
    padded = []
    for segment in normalized:
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
        start = segment.start
        while segment.end - start > max_speech_duration_s + 1e-12:
            end = round(start + max_speech_duration_s, 12)
            split.append(
                SpeechSegment(
                    start=start,
                    end=end,
                    score=segment.score,
                    label=segment.label,
                    channel=segment.channel,
                    metadata=dict(segment.metadata),
                ))
            start = end
        if segment.end - start > 1e-12:
            split.append(
                SpeechSegment(
                    start=start,
                    end=segment.end,
                    score=segment.score,
                    label=segment.label,
                    channel=segment.channel,
                    metadata=dict(segment.metadata),
                ))
    return tuple(split)


class SpeechBrainVADForVoiceActivityDetection(PreTrainedVADModel):
    """Run and fine-tune the published CRDNN without importing SpeechBrain."""

    config_class = SpeechBrainVADConfig
    default_model_name_or_path = "speechbrain/vad-crdnn-libriparty"
    native_checkpoint_format = "voicehub-speechbrain-crdnn-vad-v1"
    architecture_family = "frame-classification"
    training_support = "native"
    supports_generic_finetuning = True

    def __init__(
        self,
        config: SpeechBrainVADConfig | str | Path | None = None,
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
        config = self._coerce_config(config, model_path=model_path, **kwargs)
        super().__init__(config, device=device, lazy_load=lazy_load)

    @staticmethod
    def _resolve_device(device: str) -> str:
        return resolve_cpu_cuda_device(device, provider="native SpeechBrain VAD")

    def _load_pretrained_model(self) -> None:
        import torch

        from voicehub.checkpointing import SafeTensorReader
        from voicehub.models.vad_speechbrain.artifacts import resolve_speechbrain_vad_artifacts
        from voicehub.models.vad_speechbrain.native.checkpoint import (
            NATIVE_SPEECHBRAIN_VAD_FORMAT,
            SpeechBrainVADSafeTensorsCheckpointAdapter,
        )
        from voicehub.models.vad_speechbrain.native.configuration import SpeechBrainCRDNNVADConfig
        from voicehub.models.vad_speechbrain.native.modeling import SpeechBrainCRDNNVADModel

        source = self.config.name_or_path or self.default_model_name_or_path
        artifacts = resolve_speechbrain_vad_artifacts(
            source,
            revision=self.config.revision,
            cache_dir=self.config.cache_dir,
            token=self._hub_token,
            local_files_only=self.config.local_files_only,
            trust_pickle_checkpoint=self._trust_pickle_checkpoint,
        )
        values = read_json_file(artifacts.config)
        native_config = SpeechBrainCRDNNVADConfig.from_dict(values)
        model = SpeechBrainCRDNNVADModel(native_config)
        adapter = SpeechBrainVADSafeTensorsCheckpointAdapter()
        with SafeTensorReader(artifacts.checkpoint) as reader:
            declared_format = reader.metadata.get("format")
            if (declared_format is not None and declared_format != NATIVE_SPEECHBRAIN_VAD_FORMAT):
                raise ValueError(
                    "SpeechBrain VAD Safetensors declares unsupported format "
                    f"{declared_format!r}.")
            adapter.load_streaming(model, reader, values, strict=True)
        model.to(device=self.device, dtype=torch.float32)
        self.config.sample_rate = native_config.sampling_rate
        self.artifacts = artifacts
        self.native_config = native_config
        self.checkpoint_adapter = adapter.qualified_id
        self.model = model

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

        from voicehub.models.vad_speechbrain.native.inference import SpeechBrainVADInference
        from voicehub.processing.waveform import load_native_audio

        if window_size_samples is not None:
            raise ValueError(
                "SpeechBrain CRDNN uses fixed 25 ms analysis windows and "
                "source-compatible chunking; `window_size_samples` is not supported.")
        materialized = load_native_audio(
            audio,
            sampling_rate=sampling_rate,
            target_sampling_rate=self.sample_rate,
        )
        if materialized.waveform.numel() < 160:
            raise ValueError("SpeechBrain VAD requires at least 10 ms of audio.")
        inference = SpeechBrainVADInference(
            self.model,
            large_chunk_size=self.config.large_chunk_size,
            small_chunk_size=self.config.small_chunk_size,
            overlap_small_chunk=self.config.overlap_small_chunk,
        )
        activation = threshold if onset is None else onset
        deactivation = (min(self.config.deactivation_threshold, activation) if offset is None else offset)
        with torch.inference_mode():
            probabilities, boundaries = inference.segment(
                materialized.waveform.to(
                    device=next(self.model.parameters()).device,
                    dtype=torch.float32,
                ),
                activation_threshold=activation,
                deactivation_threshold=deactivation,
                minimum_speech_duration=min_speech_duration_ms / 1_000.0,
                maximum_silence_duration=min_silence_duration_ms / 1_000.0,
                apply_energy_vad=self.config.apply_energy_vad,
                double_check=self.config.double_check,
                speech_threshold=threshold,
            )
        raw = tuple({
            "start": boundary.start,
            "end": boundary.end,
            "score": boundary.score,
        } for boundary in boundaries)
        segments = _finalize_segments(
            raw,
            duration=materialized.duration,
            sample_rate=materialized.sampling_rate,
            speech_pad_ms=speech_pad_ms,
            max_speech_duration_s=max_speech_duration_s,
        )
        scores = probabilities.detach().float().cpu()
        return VADOutput(
            segments=segments,
            duration=materialized.duration,
            sample_rate=materialized.sampling_rate,
            probabilities=(tuple(float(value) for value in scores.tolist()) if return_frames else None),
            metadata={
                "backend":
                "voicehub-native",
                "architecture":
                "speechbrain-crdnn-vad",
                "source": (self.config.name_or_path or self.default_model_name_or_path),
                "checkpoint_format":
                self.native_checkpoint_format,
                "checkpoint_adapter":
                self.checkpoint_adapter,
                "checkpoint_revision": (None if self.artifacts is None else self.artifacts.revision),
                "converted_from_pickle":
                (False if self.artifacts is None else self.artifacts.converted_from_pickle),
                "frame_hop_samples":
                160,
                "frame_length_samples":
                400,
                "frame_scores_available":
                True,
                "causal_streaming":
                False,
            },
        )

    def prepare_training_inputs(
        self,
        inputs: dict[str, Any],
        *,
        phase: str,
    ) -> dict[str, Any]:
        from voicehub.models.vad_speechbrain.training_vad_speechbrain import prepare_speechbrain_vad_training_batch

        del phase
        if self.model is None:
            self.load_for_training()
        return prepare_speechbrain_vad_training_batch(self, inputs)

    def _validate_training_runtime(self) -> None:
        return None

    def _save_pretrained(self, save_directory: Path) -> None:
        from voicehub.checkpointing import save_safetensors
        from voicehub.models.vad_speechbrain.native.checkpoint import (
            NATIVE_SPEECHBRAIN_VAD_FILENAME,
            NATIVE_SPEECHBRAIN_VAD_FORMAT,
        )

        if self.model is None or self.native_config is None:
            self.load()
        save_directory.mkdir(parents=True, exist_ok=True)
        save_safetensors(
            self.model.state_dict(),
            save_directory / NATIVE_SPEECHBRAIN_VAD_FILENAME,
            metadata={
                "format": NATIVE_SPEECHBRAIN_VAD_FORMAT,
                "architecture": "speechbrain-crdnn-vad",
                "sample_rate": str(self.sample_rate),
            },
        )
        values = self.native_config.to_dict()
        values.update({
            "model_type": self.config.model_type,
            'architectures': [self.__class__.__name__],
            "name_or_path": str(save_directory),
            "checkpoint_format": NATIVE_SPEECHBRAIN_VAD_FORMAT,
        })
        write_json_file(save_directory / "config.json", values)

    def export_native_pretrained(self, save_directory: str | Path) -> Path:
        destination = Path(save_directory).expanduser()
        self._save_pretrained(destination)
        return destination


__all__ = ["SpeechBrainVADForVoiceActivityDetection"]
