"""Native Parakeet TDT inference, fine-tuning, and portable export."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral
from pathlib import Path
from typing import Any

from voicehub.models.asr_parakeet_tdt.configuration import ParakeetTDTASRConfig
from voicehub.models.audio import PreTrainedASRModel
from voicehub.outputs import ASROutput, ASRSegment, ASRWord

_RAW_FIELDS = frozenset({
    "audio",
    "audio_lengths",
    "sample_rate",
    "sampling_rate",
    "text",
    "transcript",
    "transcription",
})


def _batch_values(
    value: Any,
    *,
    batch_size: int,
    name: str,
) -> tuple[Any, ...]:
    if value is None or isinstance(value, (str, bytes)):
        return (value, ) * batch_size
    try:
        import torch
    except ModuleNotFoundError:  # pragma: no cover
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return (value.item(), ) * batch_size
        if value.ndim != 1:
            raise ValueError(f"`{name}` must be scalar or one-dimensional.")
        values = tuple(value.tolist())
    elif isinstance(value, Sequence):
        values = tuple(value)
    else:
        return (value, ) * batch_size
    if len(values) != batch_size:
        raise ValueError(f"`{name}` contains {len(values)} values for batch {batch_size}.")
    return values


class ParakeetTDTForSpeechRecognition(PreTrainedASRModel):
    """Run and fine-tune Parakeet TDT with VoiceHub and PyTorch only."""

    config_class = ParakeetTDTASRConfig
    default_model_name_or_path = "nvidia/parakeet-tdt-0.6b-v3"
    architecture_family = "tdt"
    native_checkpoint_format = "native-parakeet-tdt-v1"
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: ParakeetTDTASRConfig | str | Path | None = None,
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
        self.runtime: Any | None = None
        self.artifacts: Any | None = None
        self.native_config: Any | None = None
        self.parakeet_processor: Any | None = None
        self.training_processor: Any | None = None
        self.transformers_processor: Any | None = None
        config = self._coerce_config(
            config,
            model_path=model_path,
            **kwargs,
        )
        super().__init__(config, device=device, lazy_load=lazy_load)

    def prepare_inputs_for_inference(
        self,
        audio: Any,
        *,
        sampling_rate: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return {"audio": audio, "sampling_rate": sampling_rate, **kwargs}

    def _load_pretrained_model(self) -> None:
        from voicehub.models.asr_parakeet_tdt.native.runtime import load_parakeet_tdt_runtime

        source = self.config.name_or_path or self.default_model_name_or_path
        runtime = load_parakeet_tdt_runtime(
            source,
            device=self.device,
            compute_dtype=self.config.torch_dtype,
            revision=self.config.revision,
            cache_dir=self.config.cache_dir,
            token=self._hub_token,
            local_files_only=self.config.local_files_only,
            for_training=self.is_training_load,
        )
        self.runtime = runtime
        self.artifacts = runtime.artifacts
        self.native_config = runtime.config
        self.parakeet_processor = runtime.processor
        self.training_processor = runtime.processor
        self.transformers_processor = runtime.processor
        self.model = runtime.model

    @staticmethod
    def _validate_request(
        *,
        language: str | None,
        task: str,
        return_timestamps: bool | str,
        chunk_length_s: float | None,
        stride_length_s: float | tuple[float, float] | None,
        batch_size: int | None,
        num_beams: int | None,
        max_new_tokens: int | None,
        hotwords: str | tuple[str, ...] | list[str] | None,
    ) -> bool:
        if language is not None:
            raise ValueError(
                "Parakeet TDT v3 auto-detects language; a forced `language` "
                "value is unsupported.")
        if task != "transcribe":
            raise ValueError("Parakeet TDT does not implement speech translation.")
        if return_timestamps not in (False, True, "word"):
            raise ValueError("Parakeet TDT timestamps accept False, True, or 'word'.")
        if chunk_length_s is not None or stride_length_s is not None:
            raise ValueError(
                "Native Parakeet TDT currently processes one complete waveform; "
                "explicit chunk/stride controls are unsupported.")
        if batch_size not in (None, 1):
            raise ValueError("One public audio request requires `batch_size=1`.")
        if num_beams not in (None, 1):
            raise ValueError("Native Parakeet TDT implements greedy decoding only.")
        if max_new_tokens is not None:
            raise ValueError(
                "Parakeet TDT sizes its safety bound from encoder frames; "
                "`max_new_tokens` is unsupported.")
        if hotwords is not None:
            raise ValueError("Native Parakeet TDT has no validated hotword decoder.")
        return return_timestamps in (True, "word")

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

        from voicehub.models.asr_parakeet_tdt.native.decoding import decode_tdt_sequence
        from voicehub.processing.waveform import load_native_audio

        timestamps = self._validate_request(
            language=language,
            task=task,
            return_timestamps=return_timestamps,
            chunk_length_s=chunk_length_s,
            stride_length_s=stride_length_s,
            batch_size=batch_size,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            hotwords=hotwords,
        )
        if (self.model is None or self.native_config is None or self.parakeet_processor is None):
            raise RuntimeError("Parakeet TDT runtime is not loaded.")
        materialized = load_native_audio(
            audio,
            sampling_rate=sampling_rate,
            target_sampling_rate=16_000,
        )
        waveform = materialized.waveform
        minimum_samples = (self.parakeet_processor.feature_extractor.hop_length * 2)
        if waveform.numel() < minimum_samples:
            waveform = torch.nn.functional.pad(
                waveform,
                (0, minimum_samples - waveform.numel()),
            )
        parameter = next(self.model.parameters())
        prepared = self.parakeet_processor(
            waveform,
            sampling_rate=16_000,
            device=parameter.device,
        )
        input_features = prepared["input_features"].to(dtype=parameter.dtype)
        attention_mask = prepared["attention_mask"]
        with torch.inference_mode():
            generated = self.model.generate(
                input_features,
                attention_mask,
            )
        decoded = decode_tdt_sequence(
            self.parakeet_processor.tokenizer,
            generated.sequences[0],
            generated.durations[0],
            frame_seconds=self.parakeet_processor.frame_seconds,
        )
        segments: tuple[ASRSegment, ...] = ()
        if timestamps and decoded.words:
            words = tuple(
                ASRWord(
                    text=value.text,
                    start=min(materialized.duration, value.start),
                    end=min(materialized.duration, value.end),
                ) for value in decoded.words)
            segments = (
                ASRSegment(
                    text=decoded.text,
                    start=words[0].start,
                    end=words[-1].end,
                    words=words,
                ), )
        return ASROutput(
            text=decoded.text,
            segments=segments,
            language=None,
            duration=materialized.duration,
            metadata={
                "architecture": "parakeet-tdt",
                "architecture_family": "tdt",
                "backend": "voicehub-native",
                "checkpoint_revision": (None if self.artifacts is None else self.artifacts.revision),
                "decoder": "greedy-tdt",
                "emissions": int(generated.sequences.shape[1] - 1),
                "frame_seconds": self.parakeet_processor.frame_seconds,
            },
        )

    @staticmethod
    def _audio_batch(
        audio: Any,
        *,
        text_is_batch: bool,
    ) -> tuple[tuple[Any, ...], bool]:
        import torch

        if isinstance(audio, torch.Tensor):
            if audio.ndim == 1:
                return (audio, ), False
            if audio.ndim == 2:
                return tuple(audio[index] for index in range(audio.shape[0])), True
            raise ValueError("Parakeet training audio must be rank one or two.")
        if text_is_batch:
            if isinstance(audio, (str, bytes)) or not isinstance(audio, Sequence):
                raise ValueError("Batched Parakeet transcripts require waveform sequence.")
            return tuple(audio), True
        return (audio, ), False

    def prepare_training_inputs(
        self,
        inputs: dict[str, Any],
        *,
        phase: str,
    ) -> dict[str, Any]:
        """Build native log-mels, labels, and blank-prefixed decoder IDs."""
        import torch

        from voicehub.processing.waveform import load_native_audio

        del phase
        required = {"input_features", "attention_mask", "labels", "decoder_input_ids"}
        if required.issubset(inputs):
            return dict(inputs)
        if self.model is None:
            self.load_for_training()
        if self.parakeet_processor is None:
            raise RuntimeError("Parakeet TDT training processor is not loaded.")
        audio = inputs.get("audio")
        text = inputs.get(
            "text",
            inputs.get("transcription", inputs.get("transcript")),
        )
        if isinstance(text, str):
            texts = (text, )
            text_is_batch = False
        elif isinstance(text, Sequence) and not isinstance(text, (str, bytes)):
            texts = tuple(text)
            text_is_batch = True
        else:
            raise ValueError("Parakeet training records require `text`/`transcription`.")
        if not texts or any(not isinstance(value, str) or not value.strip() for value in texts):
            raise ValueError("Parakeet training transcripts must be non-empty strings.")
        if audio is None:
            raise ValueError("Parakeet training records require `audio`.")
        audio_values, was_batched = self._audio_batch(
            audio,
            text_is_batch=text_is_batch,
        )
        if len(audio_values) != len(texts):
            raise ValueError("Parakeet training requires one transcript per waveform.")
        rates = _batch_values(
            inputs.get("sampling_rate", inputs.get("sample_rate")),
            batch_size=len(audio_values),
            name="sampling_rate",
        )
        lengths = _batch_values(
            inputs.get("audio_lengths"),
            batch_size=len(audio_values),
            name="audio_lengths",
        )
        for length in lengths:
            if length is None:
                continue
            if (isinstance(length, bool) or not isinstance(length, Integral) or length <= 0):
                raise ValueError("`audio_lengths` must contain positive integers.")
        target_rate = self.parakeet_processor.feature_extractor.sampling_rate
        waveforms = tuple(
            load_native_audio(
                waveform,
                sampling_rate=rate,
                target_sampling_rate=target_rate,
                num_samples=length,
            ).waveform for waveform, rate, length in zip(
                audio_values,
                rates,
                lengths,
            ))
        minimum = self.parakeet_processor.feature_extractor.hop_length * 2
        waveforms = tuple(
            torch.nn.functional.pad(value, (0, minimum - value.numel())) if value.numel() < minimum else value
            for value in waveforms)
        prepared = self.parakeet_processor(
            waveforms,
            text=texts,
            sampling_rate=target_rate,
        )
        for name, value in inputs.items():
            if name not in _RAW_FIELDS and name not in prepared:
                prepared[name] = value
        if not was_batched:
            return {
                name: (
                    value[0]
                    if isinstance(value, torch.Tensor) and value.ndim > 1 and value.shape[0] == 1 else value)
                for name, value in prepared.items()
            }
        return prepared

    def _save_pretrained(self, save_directory: Path) -> None:
        if self.runtime is None:
            self.load()
        from voicehub.models.asr_parakeet_tdt.native.runtime import save_parakeet_tdt_runtime

        save_parakeet_tdt_runtime(self.runtime, save_directory)

    def export_native_pretrained(self, save_directory: str | Path) -> Path:
        destination = Path(save_directory).expanduser()
        self._save_pretrained(destination)
        return destination


__all__ = ["ParakeetTDTForSpeechRecognition"]
