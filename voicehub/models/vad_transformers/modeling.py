"""Native Wav2Vec2 checkpoint provider for neural VAD."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from numbers import Integral, Real
from pathlib import Path
from re import findall
from typing import Any

from voicehub.hub import read_json_file, write_json_file
from voicehub.inference_configuration import VADInferenceConfig
from voicehub.models.audio import PreTrainedVADModel
from voicehub.models.vad_transformers.configuration import TransformersVADConfig
from voicehub.outputs import VADOutput
from voicehub.vad_utils import frame_probabilities_to_segments

_SEQUENCE_ARCHITECTURES = frozenset({
    "Wav2Vec2ForAudioClassification",
    "Wav2Vec2ForSequenceClassification",
})
_FRAME_ARCHITECTURES = frozenset({
    "Wav2Vec2ForAudioFrameClassification",
})
_NON_VAD_ARCHITECTURE_MARKERS = (
    "forctc",
    "forrnnt",
    "fortdt",
    "forspeechseq2seq",
    "speechencoderdecoder",
)
_NEGATIVE_SPEECH_LABEL_TOKENS = frozenset({
    "background",
    "inactive",
    "music",
    "no",
    "noise",
    "non",
    "not",
    "silence",
    "silent",
})
_RAW_TRAINING_FIELDS = frozenset({
    "audio",
    "audio_lengths",
    "sample_rate",
    "sampling_rate",
})


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _architecture_names(config: Any) -> tuple[str, ...]:
    architectures = _config_value(config, 'architectures', ()) or ()
    if isinstance(architectures, str):
        architectures = (architectures, )
    if not isinstance(architectures, Sequence):
        raise TypeError("Checkpoint `architectures` must be a sequence.")
    return tuple(str(name) for name in architectures)


class TransformersVADForVoiceActivityDetection(PreTrainedVADModel):
    """Run and fine-tune native Wav2Vec2 VAD classifiers.

    The historical provider name remains stable for API compatibility. No
    Transformers code is imported or executed: configuration, waveform
    processing, architecture, checkpoint mapping, inference, and training
    are all implemented by VoiceHub.
    """

    config_class = TransformersVADConfig
    default_model_name_or_path = ""
    native_checkpoint_format = "voicehub-wav2vec2-vad-v1"

    def __init__(
        self,
        config: TransformersVADConfig | str | Path | None = None,
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
        self.artifacts: Any | None = None
        self.native_config: Any | None = None
        self.feature_extractor: Any | None = None
        self.architecture_family: str | None = None
        config = self._coerce_config(
            config,
            model_path=model_path,
            **kwargs,
        )
        config.validate()
        super().__init__(
            config,
            device=device,
            lazy_load=lazy_load,
        )

    @staticmethod
    def _infer_architecture_family(native_config: Any) -> str:
        """Resolve a declared VAD head without task-ambiguous guessing."""
        architectures = _architecture_names(native_config)
        for architecture in architectures:
            normalized = architecture.replace("_", "").replace("-", "").lower()
            if architecture in _FRAME_ARCHITECTURES:
                return "frame-classification"
            if architecture in _SEQUENCE_ARCHITECTURES:
                return "audio-classification"
            if any(marker in normalized for marker in _NON_VAD_ARCHITECTURE_MARKERS):
                raise ValueError(
                    f"Checkpoint architecture {architecture!r} is an ASR "
                    "head, not a VAD classifier.")
        model_type = str(_config_value(native_config, "model_type", "")).strip().lower()
        raise ValueError(
            "VoiceHub could not determine whether this checkpoint uses "
            "audio- or frame-classification" + (f" from model_type {model_type!r}" if model_type else "") +
            ". Shared encoders such as Wav2Vec2 are task-ambiguous; "
            "publish an explicit classification architecture or set "
            "`architecture_family`.")

    @classmethod
    def _validate_architecture(
        cls,
        values: Mapping[str, Any],
        requested_family: str,
    ) -> str:
        model_type = str(values.get("model_type", "")).strip().lower()
        if model_type not in {"vad_transformers", "wav2vec2"}:
            raise ValueError(
                "The native VAD provider currently supports Wav2Vec2 "
                f"classifiers; received model type {model_type or '<missing>'!r}.")
        declared = cls._infer_architecture_family(values)
        if requested_family == "auto":
            return declared
        normalized = (
            "audio-classification" if requested_family == "audio-classification" else "frame-classification")
        if declared != normalized:
            raise ValueError(
                f"Configured architecture family {normalized!r} conflicts "
                f"with checkpoint architecture family {declared!r}.")
        return normalized

    def _model_dtype(self) -> Any:
        import torch

        configured = self.config.torch_dtype
        if configured == "auto":
            return (torch.float16 if torch.device(self.device).type in {"cuda", "mps"} else torch.float32)
        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[configured]
        if torch.device(self.device).type == "cpu" and dtype == torch.float16:
            raise ValueError(
                "Native Wav2Vec2 VAD does not support float16 on CPU; "
                "use float32 or bfloat16.")
        return dtype

    def _load_pretrained_model(self) -> None:
        from voicehub.checkpointing import SafeTensorReader, ShardedSafeTensorReader
        from voicehub.models.asr_wav2vec2.native.artifacts import resolve_wav2vec2_classification_artifacts
        from voicehub.models.asr_wav2vec2.native.checkpoint import HuggingFaceWav2Vec2ClassificationCheckpointAdapter
        from voicehub.models.asr_wav2vec2.native.configuration import Wav2Vec2Config
        from voicehub.models.asr_wav2vec2.native.modeling import (
            Wav2Vec2ForAudioFrameClassification,
            Wav2Vec2ForSequenceClassification,
        )
        from voicehub.models.asr_wav2vec2.native.processing import Wav2Vec2FeatureExtractor

        source = self.config.name_or_path or self.default_model_name_or_path
        if not source:
            raise ValueError(
                "`vad_transformers` is a native checkpoint-family provider. "
                "Pass a Wav2Vec2 audio- or frame-classification Safetensors "
                "checkpoint to from_pretrained().")
        artifacts = resolve_wav2vec2_classification_artifacts(
            source,
            checkpoint_filename=self.config.checkpoint_filename,
            cache_dir=self.config.cache_dir,
            revision=self.config.revision,
            token=self._hub_token,
            local_files_only=self.config.local_files_only,
        )
        values = read_json_file(artifacts.config)
        family = self._validate_architecture(
            values,
            self.config.architecture_family,
        )
        values["_classification_family"] = family
        native_config = Wav2Vec2Config.from_dict(values)
        feature_extractor = Wav2Vec2FeatureExtractor.from_preprocessor_config(
            artifacts.preprocessor_config,
            default_sampling_rate=native_config.sampling_rate,
        )
        if feature_extractor.sampling_rate != native_config.sampling_rate:
            raise ValueError(
                "Wav2Vec2 processor/model sampling-rate mismatch: "
                f"{feature_extractor.sampling_rate} != "
                f"{native_config.sampling_rate}.")
        model = (
            Wav2Vec2ForAudioFrameClassification(native_config)
            if family == "frame-classification" else Wav2Vec2ForSequenceClassification(native_config))
        reader_type = (ShardedSafeTensorReader if artifacts.is_sharded else SafeTensorReader)
        with reader_type(artifacts.checkpoint) as reader:
            (
                HuggingFaceWav2Vec2ClassificationCheckpointAdapter().load_streaming(
                    model,
                    reader,
                    values,
                    strict=True,
                ))
        model.to(
            device=self.device,
            dtype=self._model_dtype(),
        )
        self.artifacts = artifacts
        self.native_config = native_config
        self.feature_extractor = feature_extractor
        self.architecture_family = family
        self.config.sample_rate = native_config.sampling_rate
        self.model = model

    def _id2label(self) -> Mapping[Any, Any]:
        if self.native_config is None:
            return {}
        values = getattr(self.native_config, "extra_config", {})
        labels = values.get("id2label", {}) if isinstance(values, Mapping) else {}
        return labels if isinstance(labels, Mapping) else {}

    def _speech_class_id(self, class_count: int) -> int:
        if (isinstance(class_count, bool) or not isinstance(class_count, Integral) or class_count <= 0):
            raise ValueError("VAD logits must expose at least one class.")
        class_count = int(class_count)
        configured = self.config.speech_class_id
        if configured is not None:
            if configured >= class_count:
                raise ValueError(f"`speech_class_id` {configured} is outside "
                                 f"{class_count} classes.")
            return configured
        if class_count == 1:
            return 0
        for class_id, label in self._id2label().items():
            normalized = str(label).strip().lower()
            tokens = frozenset(findall(r"[a-z0-9]+", normalized))
            exact = normalized in self.config.speech_labels
            positive = (
                not tokens.intersection(_NEGATIVE_SPEECH_LABEL_TOKENS) and
                any(token in tokens for token in self.config.speech_labels))
            if exact or positive:
                resolved = int(class_id)
                if not 0 <= resolved < class_count:
                    raise ValueError(
                        "Checkpoint `id2label` contains a class index outside "
                        "the logits dimension.")
                return resolved
        if class_count == 2:
            return 1
        raise ValueError(
            "Could not identify the speech class from checkpoint labels. "
            "Set `speech_class_id` in TransformersVADConfig.")

    def _processor_sample_rate(self) -> int:
        value = (
            self.config.sample_rate
            if self.feature_extractor is None else self.feature_extractor.sampling_rate)
        if (isinstance(value, bool) or not isinstance(value, Real) or not isfinite(float(value)) or
                float(value) <= 0 or not float(value).is_integer()):
            raise ValueError("The native VAD processor reported an invalid sampling rate.")
        return int(value)

    def _frame_geometry(
        self,
        *,
        frame_count: int,
        waveform_samples: int,
    ) -> tuple[int, int]:
        if (isinstance(frame_count, bool) or not isinstance(frame_count, Integral) or frame_count <= 0):
            raise ValueError("Frame-classification logits must contain at least one frame.")
        ratio = (None if self.native_config is None else self.native_config.inputs_to_logits_ratio)
        if ratio is not None:
            if (isinstance(ratio, bool) or not isinstance(ratio, Real) or not isfinite(float(ratio)) or
                    ratio <= 0):
                raise ValueError(
                    "Checkpoint `inputs_to_logits_ratio` must be finite and "
                    "greater than zero.")
            frame_hop = max(1, round(float(ratio)))
            return frame_hop, frame_hop
        frame_hop = max(1, round(waveform_samples / int(frame_count)))
        return frame_hop, frame_hop

    @staticmethod
    def _probabilities(logits: Any) -> Any:
        import torch

        if logits.shape[-1] == 1:
            return torch.sigmoid(logits)
        return torch.softmax(logits, dim=-1)

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

        if (self.model is None or self.native_config is None or self.feature_extractor is None or
                self.architecture_family is None):
            raise RuntimeError("Native Wav2Vec2 VAD runtime is not loaded.")
        target_rate = self._processor_sample_rate()
        materialized = load_native_audio(
            audio,
            sampling_rate=sampling_rate,
            target_sampling_rate=target_rate,
        )
        waveform = materialized.waveform
        parameter = next(self.model.parameters())

        if self.architecture_family == "frame-classification":
            if waveform.numel() < self.native_config.minimum_input_samples:
                waveform = torch.nn.functional.pad(
                    waveform,
                    (
                        0,
                        self.native_config.minimum_input_samples - waveform.numel(),
                    ),
                )
            batch = self.feature_extractor.prepare_audio_batch((waveform, ))
            with torch.inference_mode():
                outputs = self.model(
                    batch["input_values"].to(
                        device=parameter.device,
                        dtype=parameter.dtype,
                    ),
                    attention_mask=batch["attention_mask"].to(device=parameter.device, ),
                )
            probabilities = self._probabilities(outputs.logits)[0]
            valid_frames = int(outputs.feature_attention_mask[0].sum().item())
            probabilities = probabilities[:valid_frames]
            speech_id = self._speech_class_id(probabilities.shape[-1])
            frame_scores = probabilities[..., speech_id]
            frame_hop, frame_length = self._frame_geometry(
                frame_count=valid_frames,
                waveform_samples=materialized.waveform.numel(),
            )
        else:
            frame_length = (
                window_size_samples if window_size_samples is not None else round(
                    self.config.window_duration_s * target_rate))
            frame_hop = round(self.config.hop_duration_s * target_rate)
            if frame_length < self.native_config.minimum_input_samples:
                raise ValueError(
                    "The VAD window is shorter than the Wav2Vec2 "
                    "convolutional frontend minimum.")
            windows = []
            for start in range(0, waveform.numel(), frame_hop):
                window = waveform[start:start + frame_length]
                if window.numel() < frame_length:
                    window = torch.nn.functional.pad(
                        window,
                        (0, frame_length - window.numel()),
                    )
                windows.append(window)
                if start + frame_length >= waveform.numel():
                    break
            batch = self.feature_extractor.prepare_audio_batch(tuple(windows))
            with torch.inference_mode():
                outputs = self.model(
                    batch["input_values"].to(
                        device=parameter.device,
                        dtype=parameter.dtype,
                    ),
                    attention_mask=batch["attention_mask"].to(device=parameter.device, ),
                )
            probabilities = self._probabilities(outputs.logits)
            speech_id = self._speech_class_id(probabilities.shape[-1])
            frame_scores = probabilities[..., speech_id]

        cpu_scores = frame_scores.detach().float().cpu()
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
            cpu_scores.tolist(),
            sampling_rate=target_rate,
            frame_hop_samples=frame_hop,
            frame_length_samples=frame_length,
            duration_samples=materialized.waveform.numel(),
            config=postprocessing,
        )
        return VADOutput(
            segments=segments,
            duration=materialized.duration,
            sample_rate=target_rate,
            probabilities=cpu_scores if return_frames else None,
            metadata={
                "backend": "voicehub-native",
                "architecture": "wav2vec2",
                "architecture_family": self.architecture_family,
                "checkpoint_revision": (None if self.artifacts is None else self.artifacts.revision),
                "speech_class_id": speech_id,
                "frame_hop_samples": frame_hop,
                "frame_length_samples": frame_length,
            },
        )

    @staticmethod
    def _array_rank(value: Any) -> int | None:
        shape = getattr(value, "shape", None)
        if shape is None:
            return None
        try:
            return len(shape)
        except TypeError:
            ndim = getattr(value, "ndim", None)
            return int(ndim) if isinstance(ndim, Integral) else None

    @classmethod
    def _audio_batch(cls, value: Any) -> list[Any]:
        rank = cls._array_rank(value)
        if rank == 2:
            values = [value[index] for index in range(int(value.shape[0]))]
        elif rank == 1 or (rank is None and not isinstance(value, (list, tuple))):
            values = [value]
        elif rank is not None:
            raise ValueError(
                "Raw VAD training audio must be rank 1, or rank 2 with "
                "shape (batch, samples).")
        elif value and all(isinstance(item, Real) for item in value):
            values = [value]
        else:
            values = list(value)
        if not values:
            raise ValueError("Raw VAD training audio batches cannot be empty.")
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
        """Build a native waveform batch while preserving classifier labels."""
        import torch

        from voicehub.processing.waveform import load_native_audio

        del phase
        if "input_values" in inputs:
            return dict(inputs)
        if self.model is None:
            self.load_for_training()
        if self.native_config is None or self.feature_extractor is None:
            raise RuntimeError("Native VAD training processor is not loaded.")
        audio = inputs.get("audio")
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
                if (isinstance(length, bool) or not isinstance(length, Integral) or length <= 0):
                    raise ValueError("`audio_lengths` must contain positive integers.")
                value = torch.as_tensor(value)
                if value.ndim != 1 or length > value.shape[-1]:
                    raise ValueError("`audio_lengths` exceeds a waveform's sample count.")
                value = value[:int(length)]
            waveform = load_native_audio(
                value,
                sampling_rate=rate,
                target_sampling_rate=self.native_config.sampling_rate,
            ).waveform
            minimum = self.native_config.minimum_input_samples
            if waveform.numel() < minimum:
                waveform = torch.nn.functional.pad(
                    waveform,
                    (0, minimum - waveform.numel()),
                )
            waveforms.append(waveform)
        prepared = self.feature_extractor.prepare_audio_batch(tuple(waveforms))
        for name, value in inputs.items():
            if name in _RAW_TRAINING_FIELDS:
                continue
            if name == "labels" and not isinstance(value, torch.Tensor):
                value = torch.as_tensor(value)
            prepared[name] = value
        return prepared

    def _save_pretrained(self, save_directory: Path) -> None:
        from voicehub.checkpointing import save_safetensors

        if (self.model is None or self.native_config is None or self.feature_extractor is None or
                self.architecture_family is None):
            self.load()
        save_directory.mkdir(parents=True, exist_ok=True)
        save_safetensors(
            self.model.state_dict(),
            save_directory / "model.safetensors",
            metadata={"format": self.native_checkpoint_format},
        )
        architecture = (
            "Wav2Vec2ForAudioFrameClassification"
            if self.architecture_family == "frame-classification" else "Wav2Vec2ForSequenceClassification")
        values = self.native_config.to_dict()
        values.update({
            'architectures': [architecture],
            "model_type": "wav2vec2",
            "voicehub_checkpoint_format": self.native_checkpoint_format,
            "voicehub_provider": self.config.model_type,
        })
        write_json_file(save_directory / "config.json", values)
        self.feature_extractor.save_pretrained(save_directory)

    def export_native_pretrained(
        self,
        save_directory: str | Path,
    ) -> Path:
        """Write a self-contained native VAD artifact."""
        destination = Path(save_directory).expanduser()
        self._save_pretrained(destination)
        return destination

    def _validate_training_runtime(self) -> None:
        self.config.validate()


__all__ = ["TransformersVADForVoiceActivityDetection"]
