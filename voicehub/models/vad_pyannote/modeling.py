"""VoiceHub-native PyanNet voice activity detection provider."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral, Real
from pathlib import Path
from typing import Any

from voicehub.hub import read_json_file, write_json_file
from voicehub.inference_configuration import VADInferenceConfig
from voicehub.models.audio import PreTrainedVADModel
from voicehub.models.native_utils import resolve_cpu_cuda_device
from voicehub.models.vad_pyannote.configuration import PyannoteVADConfig
from voicehub.outputs import SpeechSegment, VADOutput
from voicehub.vad_utils import frame_probabilities_to_segments, merge_speech_segments, normalize_backend_segments


def _finalize_segments(
    values: Any,
    *,
    duration: float,
    sample_rate: int,
    speech_pad_ms: int,
    max_speech_duration_s: float | None,
) -> tuple[SpeechSegment, ...]:
    """Backward-compatible normalization helper with no provider dependency."""
    normalized = normalize_backend_segments(values, sampling_rate=sample_rate)
    padding = speech_pad_ms / 1000.0
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


class PyannoteVADForVoiceActivityDetection(PreTrainedVADModel):
    """Run and fine-tune the published PyanNet graph without pyannote.audio."""

    config_class = PyannoteVADConfig
    default_model_name_or_path = "pyannote/voice-activity-detection"
    native_checkpoint_format = "voicehub-pyannet-v1"
    native_variant = "segmentation"
    architecture_family = "frame-classification"
    training_support = "native"
    supports_generic_finetuning = True

    def __init__(
        self,
        config: PyannoteVADConfig | str | Path | None = None,
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
        super().__init__(config, device=device, lazy_load=lazy_load)

    @staticmethod
    def _resolve_device(device: str) -> str:
        return resolve_cpu_cuda_device(device, provider="native PyanNet VAD")

    def _load_pretrained_model(self) -> None:
        import torch

        from voicehub.checkpointing import SafeTensorReader
        from voicehub.models.vad_pyannote.artifacts import resolve_pyannet_artifacts
        from voicehub.models.vad_pyannote.native.checkpoint import PyanNetSafeTensorsCheckpointAdapter
        from voicehub.models.vad_pyannote.native.configuration import PyanNetConfig
        from voicehub.models.vad_pyannote.native.modeling import PyanNet

        source = self.config.name_or_path or self.default_model_name_or_path
        artifacts = resolve_pyannet_artifacts(
            source,
            variant=self.native_variant,
            cache_dir=self.config.cache_dir,
            revision=self.config.revision,
            subfolder=self.config.subfolder,
            token=self._hub_token,
            local_files_only=self.config.local_files_only,
            trust_pickle_checkpoint=self._trust_pickle_checkpoint,
        )
        values = read_json_file(artifacts.config)
        native_config = PyanNetConfig.from_dict(values)
        if native_config.variant != self.native_variant:
            raise ValueError(
                f"Provider {self.config.model_type!r} requires PyanNet variant "
                f"{self.native_variant!r}, found {native_config.variant!r}.")
        model = PyanNet(native_config)
        adapter = PyanNetSafeTensorsCheckpointAdapter()
        with SafeTensorReader(artifacts.checkpoint) as reader:
            adapter.load_streaming(model, reader, values, strict=True)
        model.to(device=self.device, dtype=torch.float32)
        self.config.sample_rate = native_config.sampling_rate
        self.artifacts = artifacts
        self.native_config = native_config
        self.checkpoint_adapter = adapter.qualified_id
        self.model = model

    def _frame_output(self, waveform: Any) -> Any:
        from voicehub.models.vad_pyannote.native.inference import PyanNetFrameInference

        if self.model is None or self.native_config is None:
            raise RuntimeError("Native PyanNet runtime is not loaded.")
        inference = PyanNetFrameInference(
            self.model,
            batch_size=self.config.batch_size,
            duration_s=getattr(self.config, "inference_duration_s", None),
            step_s=getattr(self.config, "inference_step_s", None),
        )
        return inference(waveform)

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
                "PyanNet chunk and frame geometry are fixed by the converted "
                "checkpoint; `window_size_samples` is not supported.")
        materialized = load_native_audio(
            audio,
            sampling_rate=sampling_rate,
            target_sampling_rate=self.sample_rate,
        )
        with torch.inference_mode():
            output = self._frame_output(materialized.waveform)
        scores = output.scores[:, 0].detach().float().cpu()
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
            scores.tolist(),
            sampling_rate=self.sample_rate,
            frame_hop_samples=output.frame_hop_samples,
            frame_length_samples=output.frame_length_samples,
            duration_samples=materialized.waveform.numel(),
            config=postprocessing,
        )
        return VADOutput(
            segments=segments,
            duration=materialized.duration,
            sample_rate=self.sample_rate,
            probabilities=(tuple(float(item) for item in scores.tolist()) if return_frames else None),
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
            },
        )

    @staticmethod
    def _audio_batch(value: Any) -> list[Any]:
        shape = getattr(value, "shape", None)
        rank = None if shape is None else len(shape)
        if rank == 2:
            values = [value[index] for index in range(int(shape[0]))]
        elif rank == 1 or (rank is None and not isinstance(value, (list, tuple))):
            values = [value]
        elif rank is not None:
            raise ValueError("Training audio must have shape [samples] or "
                             "[batch, samples].")
        elif value and all(isinstance(item, Real) for item in value):
            values = [value]
        else:
            values = list(value)
        if not values:
            raise ValueError("Training audio batches cannot be empty.")
        return values

    @staticmethod
    def _batch_values(
        value: Any,
        *,
        batch_size: int,
        name: str,
        broadcast: bool,
    ) -> list[Any]:
        import torch

        if value is None:
            return [None] * batch_size
        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                values = [value.item()]
            elif value.ndim == 1:
                values = value.tolist()
            else:
                raise ValueError(f"`{name}` must be scalar or one-dimensional.")
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            values = list(value)
        else:
            values = [value]
        if broadcast and len(values) == 1 and batch_size > 1:
            values *= batch_size
        if len(values) != batch_size:
            raise ValueError(f"Batched audio and `{name}` fields must have equal lengths.")
        return values

    def prepare_training_inputs(
        self,
        inputs: dict[str, Any],
        *,
        phase: str,
    ) -> dict[str, Any]:
        """Build a padded native waveform batch and preserve frame targets."""
        del phase
        import torch

        from voicehub.processing.waveform import load_native_audio

        if self.model is None:
            self.load_for_training()
        if self.native_config is None:
            raise RuntimeError("Native PyanNet training runtime is not loaded.")
        if "waveforms" in inputs or "input_values" in inputs:
            prepared = dict(inputs)
            if "labels" not in prepared and "y" in prepared:
                prepared["labels"] = prepared.pop("y")
            return prepared
        audio = inputs.get("audio", inputs.get("X"))
        if audio is None:
            return dict(inputs)
        values = self._audio_batch(audio)
        lengths = self._batch_values(
            inputs.get("audio_lengths"),
            batch_size=len(values),
            name="audio_lengths",
            broadcast=False,
        )
        rates = self._batch_values(
            inputs.get("sampling_rate", inputs.get("sample_rate")),
            batch_size=len(values),
            name="sampling_rate",
            broadcast=True,
        )
        waveforms = []
        for value, length, rate in zip(values, lengths, rates):
            if length is not None:
                if (isinstance(length, bool) or not isinstance(length, Integral) or length < 1):
                    raise ValueError("`audio_lengths` must contain positive integers.")
                value = torch.as_tensor(value)
                if value.ndim != 1 or int(length) > value.shape[-1]:
                    raise ValueError("`audio_lengths` exceeds a waveform's sample count.")
                value = value[:int(length)]
            waveform = load_native_audio(
                value,
                sampling_rate=rate,
                target_sampling_rate=self.native_config.sampling_rate,
            ).waveform
            if waveform.numel() < self.model.sincnet.minimum_samples:
                waveform = torch.nn.functional.pad(
                    waveform,
                    (
                        0,
                        self.model.sincnet.minimum_samples - waveform.numel(),
                    ),
                )
            waveforms.append(waveform)
        maximum = max(item.numel() for item in waveforms)
        prepared: dict[str, Any] = {
            "waveforms":
            torch.stack(
                tuple(torch.nn.functional.pad(
                    item,
                    (0, maximum - item.numel()),
                ) for item in waveforms))
        }
        ignored = {
            "audio",
            "X",
            "audio_lengths",
            "sample_rate",
            "sampling_rate",
            "y",
        }
        for name, value in inputs.items():
            if name in ignored:
                continue
            prepared[name] = value
        labels = inputs.get("labels", inputs.get("y"))
        if labels is not None:
            prepared["labels"] = (labels if isinstance(labels, torch.Tensor) else torch.as_tensor(labels))
        for name in ("snr_loss_scale", "c50_loss_scale"):
            value = getattr(self.config, name, None)
            if value is not None:
                prepared.setdefault(name, value)
        return prepared

    def _save_pretrained(self, save_directory: Path) -> None:
        from voicehub.checkpointing import save_safetensors

        if self.model is None or self.native_config is None:
            self.load()
        save_directory.mkdir(parents=True, exist_ok=True)
        save_safetensors(
            self.model.state_dict(),
            save_directory / "model.safetensors",
            metadata={
                "format": self.native_checkpoint_format,
                "variant": self.native_variant,
            },
        )
        values = self.native_config.to_dict()
        values.update({
            "voicehub_checkpoint_format": self.native_checkpoint_format,
            "voicehub_provider": self.config.model_type,
        })
        write_json_file(save_directory / "config.json", values)

    def _validate_training_runtime(self) -> None:
        """PyanNet Safetensors graphs are fully differentiable."""
        return None


__all__ = ["PyannoteVADForVoiceActivityDetection"]
