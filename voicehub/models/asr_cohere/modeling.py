"""Native Cohere Transcribe inference and fine-tuning wrapper."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral
from pathlib import Path
from typing import Any

from voicehub.models.asr_cohere.configuration import CohereASRConfig
from voicehub.models.audio import PreTrainedASRModel
from voicehub.outputs import ASROutput

_RAW_FIELDS = frozenset({
    "audio",
    "audio_lengths",
    "language",
    "punctuation",
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


class CohereForSpeechRecognition(PreTrainedASRModel):
    """Run and fine-tune Cohere Transcribe with VoiceHub and PyTorch only."""

    config_class = CohereASRConfig
    default_model_name_or_path = ("CohereLabs/cohere-transcribe-03-2026")
    architecture_family = "speech-seq2seq"
    native_checkpoint_format = "native-cohere-asr-v1"
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: CohereASRConfig | str | Path | None = None,
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
        self.cohere_processor: Any | None = None
        self.training_processor: Any | None = None
        self.transformers_processor: Any | None = None
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

    def prepare_inputs_for_inference(
        self,
        audio: Any,
        *,
        sampling_rate: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return {
            "audio": audio,
            "sampling_rate": sampling_rate,
            **kwargs,
        }

    def _load_pretrained_model(self) -> None:
        from voicehub.models.asr_cohere.native.runtime import load_cohere_asr_runtime

        source = (self.config.name_or_path or self.default_model_name_or_path)
        runtime = load_cohere_asr_runtime(
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
        self.cohere_processor = runtime.processor
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
        punctuation: bool,
    ) -> tuple[str, int]:
        from voicehub.models.asr_cohere.native.configuration import SUPPORTED_LANGUAGES

        if language is None:
            raise ValueError(
                "Cohere Transcribe requires an explicit language; it does "
                "not expose verified language detection.")
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported Cohere ASR language {language!r}.")
        if task != "transcribe":
            raise ValueError("Cohere Transcribe does not implement speech translation.")
        if return_timestamps is not False:
            raise ValueError("Native Cohere ASR has no verified timestamp decoder.")
        if chunk_length_s is not None or stride_length_s is not None:
            raise ValueError(
                "Cohere ASR uses its source-verified quiet-boundary long-form "
                "splitter; manual chunk/stride controls are unsupported.")
        if batch_size not in (None, 1):
            raise ValueError("One public Cohere ASR request requires `batch_size=1`.")
        if num_beams not in (None, 1):
            raise ValueError("Native Cohere ASR implements greedy decoding only.")
        if max_new_tokens is None:
            resolved_maximum = 256
        elif (isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or
              not 1 <= max_new_tokens <= 256):
            raise ValueError("Cohere ASR `max_new_tokens` must be in 1..256.")
        else:
            resolved_maximum = max_new_tokens
        if hotwords is not None:
            raise ValueError("Native Cohere ASR has no validated hotword decoder.")
        if not isinstance(punctuation, bool):
            raise TypeError("`punctuation` must be a boolean.")
        return language, resolved_maximum

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
        punctuation: bool = True,
    ) -> ASROutput:
        import torch

        from voicehub.processing.waveform import load_native_audio

        language, maximum = self._validate_request(
            language=language,
            task=task,
            return_timestamps=return_timestamps,
            chunk_length_s=chunk_length_s,
            stride_length_s=stride_length_s,
            batch_size=batch_size,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            hotwords=hotwords,
            punctuation=punctuation,
        )
        if (self.model is None or self.native_config is None or self.cohere_processor is None):
            raise RuntimeError("Cohere ASR runtime is not loaded.")
        materialized = load_native_audio(
            audio,
            sampling_rate=sampling_rate,
            target_sampling_rate=16_000,
        )
        waveform = materialized.waveform
        minimum = self.native_config.hop_length * 2
        if waveform.numel() < minimum:
            waveform = torch.nn.functional.pad(
                waveform,
                (0, minimum - waveform.numel()),
            )
        parameter = next(self.model.parameters())
        prepared = self.cohere_processor(
            waveform,
            language=language,
            punctuation=punctuation,
            sampling_rate=16_000,
            device=parameter.device,
        )
        input_features = prepared["input_features"].to(dtype=parameter.dtype)
        with torch.inference_mode():
            generated = self.model.generate(
                input_features,
                prepared["attention_mask"],
                prepared["decoder_input_ids"],
                decoder_attention_mask=prepared["decoder_attention_mask"],
                max_new_tokens=maximum,
            )
        prompt_length = prepared["decoder_input_ids"].shape[1]
        decoded = []
        for row in generated.sequences[:, prompt_length:].tolist():
            if self.native_config.eos_token_id in row:
                row = row[:row.index(self.native_config.eos_token_id)]
            row = [token_id for token_id in row if token_id != self.native_config.pad_token_id]
            decoded.append(self.cohere_processor.tokenizer.decode(
                row,
                skip_special_tokens=True,
            ).strip())
        texts = self.cohere_processor.reassemble_chunk_texts(
            decoded,
            prepared["audio_chunk_index"],
            language=language,
        )
        if len(texts) != 1:
            raise RuntimeError("One Cohere ASR request produced an invalid sample count.")
        return ASROutput(
            text=texts[0],
            segments=(),
            language=language,
            duration=materialized.duration,
            metadata={
                "architecture": "cohere-asr",
                "architecture_family": "speech-seq2seq",
                "backend": "voicehub-native",
                "checkpoint_revision": (None if self.artifacts is None else self.artifacts.revision),
                "decoder": "greedy-autoregressive",
                "long_form_segments": len(decoded),
                "punctuation": punctuation,
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
            raise ValueError("Cohere ASR training audio must be rank one or two.")
        if text_is_batch:
            if (isinstance(audio, (str, bytes)) or not isinstance(audio, Sequence)):
                raise ValueError("Batched Cohere transcripts require waveform sequence.")
            return tuple(audio), True
        return (audio, ), False

    def prepare_training_inputs(
        self,
        inputs: dict[str, Any],
        *,
        phase: str,
    ) -> dict[str, Any]:
        """Build prompt-conditioned teacher-forcing tensors."""
        import torch

        from voicehub.processing.waveform import load_native_audio

        del phase
        required = {
            "input_features",
            "attention_mask",
            "decoder_input_ids",
            "decoder_attention_mask",
            "labels",
        }
        if required.issubset(inputs):
            return dict(inputs)
        if self.model is None:
            self.load_for_training()
        if self.cohere_processor is None:
            raise RuntimeError("Cohere ASR training processor is not loaded.")
        audio = inputs.get("audio")
        text = inputs.get(
            "text",
            inputs.get(
                "transcription",
                inputs.get("transcript"),
            ),
        )
        if isinstance(text, str):
            texts = (text, )
            text_is_batch = False
        elif isinstance(text, Sequence) and not isinstance(text, (str, bytes)):
            texts = tuple(text)
            text_is_batch = True
        else:
            raise ValueError("Cohere ASR training records require a transcript.")
        if not texts or any(not isinstance(value, str) or not value.strip() for value in texts):
            raise ValueError("Cohere ASR transcripts must be non-empty strings.")
        if audio is None:
            raise ValueError("Cohere ASR training records require `audio`.")
        audio_values, was_batched = self._audio_batch(
            audio,
            text_is_batch=text_is_batch,
        )
        if len(audio_values) != len(texts):
            raise ValueError("Cohere ASR training requires one transcript per waveform.")
        rates = _batch_values(
            inputs.get(
                "sampling_rate",
                inputs.get("sample_rate"),
            ),
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
        target_rate = self.cohere_processor.feature_extractor.sampling_rate
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
        minimum = self.cohere_processor.feature_extractor.hop_length * 2
        waveforms = tuple(
            torch.nn.functional.pad(
                value,
                (0, minimum - value.numel()),
            ) if value.numel() < minimum else value for value in waveforms)
        language_values = _batch_values(
            inputs.get("language"),
            batch_size=len(waveforms),
            name="language",
        )
        if any(not isinstance(value, str) or not value.strip() for value in language_values):
            raise ValueError("Cohere ASR training requires an explicit language string "
                             "for every sample.")
        if len(set(language_values)) != 1:
            raise ValueError("One Cohere ASR training batch requires one explicit shared "
                             "language.")
        punctuation_values = _batch_values(
            inputs.get("punctuation", True),
            batch_size=len(waveforms),
            name="punctuation",
        )
        if any(not isinstance(value, bool) for value in punctuation_values):
            raise TypeError("Cohere ASR training punctuation values must be booleans.")
        if len(set(punctuation_values)) != 1:
            raise ValueError("One Cohere ASR training batch requires one punctuation mode.")
        prepared = self.cohere_processor(
            waveforms,
            language=language_values[0],
            text=texts,
            punctuation=punctuation_values[0],
            sampling_rate=target_rate,
            device=next(self.model.parameters()).device,
        )
        for name, value in inputs.items():
            if name not in _RAW_FIELDS and name not in prepared:
                prepared[name] = value
        prepared.pop("audio_chunk_index", None)
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
        from voicehub.models.asr_cohere.native.runtime import save_cohere_asr_runtime

        save_cohere_asr_runtime(self.runtime, save_directory)

    def export_native_pretrained(
        self,
        save_directory: str | Path,
    ) -> Path:
        destination = Path(save_directory).expanduser()
        self._save_pretrained(destination)
        return destination


__all__ = ["CohereForSpeechRecognition"]
