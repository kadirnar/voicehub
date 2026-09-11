"""Native SeamlessM4T-v2 inference, fine-tuning, and export wrapper."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral
from pathlib import Path
from typing import Any

from voicehub.models.asr_seamless_m4t_v2.configuration import SeamlessM4Tv2ASRConfig
from voicehub.models.audio import PreTrainedASRModel
from voicehub.outputs import ASROutput

_RAW_FIELDS = frozenset({
    "audio",
    "audio_lengths",
    "language",
    "sample_rate",
    "sampling_rate",
    "target_language",
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
    except ModuleNotFoundError:  # pragma: no cover - package invariant
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return (value.item(), ) * batch_size
        if value.ndim != 1:
            raise ValueError(f"`{name}` must be scalar or one-dimensional.")
        result = tuple(value.detach().cpu().tolist())
    elif isinstance(value, Sequence):
        result = tuple(value)
    else:
        return (value, ) * batch_size
    if len(result) != batch_size:
        raise ValueError(f"`{name}` contains {len(result)} values for batch size "
                         f"{batch_size}.")
    return result


class SeamlessM4Tv2ForSpeechRecognition(PreTrainedASRModel):
    """Run Facebook SeamlessM4T-v2 S2T with VoiceHub and PyTorch only."""

    config_class = SeamlessM4Tv2ASRConfig
    default_model_name_or_path = "facebook/seamless-m4t-v2-large"
    architecture_family = "speech-seq2seq"
    training_support = "native"
    supports_generic_finetuning = True
    supports_gradient_checkpointing = True
    native_checkpoint_format = "native-seamless-m4t-v2-s2t-v1"

    def __init__(
        self,
        config: SeamlessM4Tv2ASRConfig | str | Path | None = None,
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
        self.seamless_processor: Any | None = None
        self.training_processor: Any | None = None
        self.transformers_processor: Any | None = None
        normalized = self._coerce_config(
            config,
            model_path=model_path,
            **kwargs,
        )
        super().__init__(
            normalized,
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
        from voicehub.models.asr_seamless_m4t_v2.native.runtime import load_seamless_m4t_v2_runtime

        source = self.config.name_or_path or self.default_model_name_or_path
        runtime = load_seamless_m4t_v2_runtime(
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
        self.seamless_processor = runtime.processor
        self.training_processor = runtime.processor
        # Compatibility alias during migration; this is a VoiceHub processor.
        self.transformers_processor = runtime.processor
        self.model = runtime.model

    def _validate_request(
        self,
        *,
        language: str | None,
        task: str,
        return_timestamps: bool | str,
        chunk_length_s: float | None,
        stride_length_s: Any,
        batch_size: int | None,
        num_beams: int | None,
        max_new_tokens: int | None,
        hotwords: Any,
    ) -> tuple[str, int | None]:
        if task != "transcribe":
            raise ValueError(
                "Native SeamlessM4T-v2 currently implements recognition, "
                "not speech translation.")
        if return_timestamps is not False:
            raise ValueError("Native SeamlessM4T-v2 does not claim timestamp alignment.")
        unsupported = {
            "chunk_length_s": chunk_length_s,
            "stride_length_s": stride_length_s,
            "hotwords": hotwords,
        }
        active = [name for name, value in unsupported.items() if value is not None]
        if active:
            raise ValueError(
                "Native SeamlessM4T-v2 performs complete-waveform greedy "
                "recognition; unsupported option(s): " + ", ".join(active))
        if batch_size not in (None, 1):
            raise ValueError("One public S2T request requires `batch_size=1`.")
        if num_beams not in (None, 1):
            raise ValueError("Native SeamlessM4T-v2 implements greedy generation only.")
        if max_new_tokens is not None and (isinstance(max_new_tokens, bool) or
                                           not isinstance(max_new_tokens, int) or
                                           not 1 <= max_new_tokens <= 256):
            raise ValueError("`max_new_tokens` must be in [1, 256].")
        target = self.config.target_language if language is None else language
        if not isinstance(target, str) or not target.strip():
            raise ValueError("Recognition language must be a non-empty code.")
        return target.strip(), max_new_tokens

    def _transcribe(
        self,
        audio: Any,
        *,
        sampling_rate: int | None = None,
        language: str | None = None,
        task: str = "transcribe",
        return_timestamps: bool | str = False,
        chunk_length_s: float | None = None,
        stride_length_s: Any = None,
        batch_size: int | None = None,
        num_beams: int | None = None,
        max_new_tokens: int | None = None,
        hotwords: Any = None,
    ) -> ASROutput:
        import torch

        from voicehub.processing.waveform import load_native_audio

        target, maximum = self._validate_request(
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
        if self.model is None or self.seamless_processor is None:
            raise RuntimeError("SeamlessM4T-v2 runtime is not loaded.")
        materialized = load_native_audio(
            audio,
            sampling_rate=sampling_rate,
            target_sampling_rate=16_000,
        )
        batch = self.seamless_processor.process_audio(
            materialized.waveform,
            sampling_rate=16_000,
        )
        parameter = next(self.model.parameters())
        input_features = batch.input_features.to(
            device=parameter.device,
            dtype=parameter.dtype,
        )
        attention_mask = batch.attention_mask.to(device=parameter.device)
        language_id = self.seamless_processor.generation_language_id(target)
        with torch.inference_mode():
            generated = self.model.generate(
                input_features,
                attention_mask=attention_mask,
                language_token_id=language_id,
                max_new_tokens=maximum,
            )
        text = self.seamless_processor.decode(generated[0].detach().cpu().tolist(), )
        return ASROutput(
            text=text,
            language=target,
            duration=materialized.duration,
            metadata={
                "architecture": "seamless-m4t-v2-s2t",
                "architecture_family": "speech-seq2seq",
                "backend": "voicehub-native",
                "checkpoint_revision": (None if self.artifacts is None else self.artifacts.revision),
                "decoding": "greedy",
                "model": "seamless-m4t-v2-large",
            },
        )

    @staticmethod
    def _audio_batch(
        audio: Any,
        *,
        text_is_batch: bool,
    ) -> tuple[Any, ...]:
        import torch

        if isinstance(audio, torch.Tensor):
            if audio.ndim == 1:
                return (audio, )
            if audio.ndim == 2:
                return tuple(audio[index] for index in range(audio.shape[0]))
            raise ValueError("Training audio must be rank one or rank two.")
        if text_is_batch:
            if (isinstance(audio, (str, bytes)) or not isinstance(audio, Sequence)):
                raise ValueError("Batched transcripts require a sequence of waveforms.")
            return tuple(audio)
        return (audio, )

    def prepare_training_inputs(
        self,
        inputs: dict[str, Any],
        *,
        phase: str,
    ) -> dict[str, Any]:
        """Build stacked features and language-conditioned sequence labels."""
        from voicehub.processing.waveform import load_native_audio

        del phase
        if "input_features" in inputs and "labels" in inputs:
            return dict(inputs)
        if self.model is None:
            self.load_for_training()
        if self.seamless_processor is None:
            raise RuntimeError("SeamlessM4T-v2 training processor is not loaded.")
        audio = inputs.get("audio")
        text = inputs.get(
            "text",
            inputs.get("transcription", inputs.get("transcript")),
        )
        if isinstance(text, str):
            texts = (text, )
            text_is_batch = False
        elif (isinstance(text, Sequence) and not isinstance(text, (str, bytes))):
            texts = tuple(text)
            text_is_batch = True
        else:
            raise ValueError("Training records require `text`, `transcription`, or "
                             "`transcript`.")
        if (not texts or any(not isinstance(value, str) or not value.strip() for value in texts)):
            raise ValueError("Training transcripts must be non-empty strings.")
        if audio is None:
            raise ValueError("Training records require `audio`.")
        audio_values = self._audio_batch(
            audio,
            text_is_batch=text_is_batch,
        )
        if len(audio_values) != len(texts):
            raise ValueError("Training requires one transcript per waveform.")
        lengths = _batch_values(
            inputs.get("audio_lengths"),
            batch_size=len(audio_values),
            name="audio_lengths",
        )
        if any(length is not None and
               (isinstance(length, bool) or not isinstance(length, Integral) or length <= 0)
               for length in lengths):
            raise ValueError("`audio_lengths` must contain positive integers.")
        rates = _batch_values(
            inputs.get("sampling_rate", inputs.get("sample_rate")),
            batch_size=len(audio_values),
            name="sampling_rate",
        )
        waveforms = tuple(
            load_native_audio(
                value,
                sampling_rate=rate,
                target_sampling_rate=16_000,
                num_samples=(None if length is None else int(length)),
            ).waveform for value, rate, length in zip(
                audio_values,
                rates,
                lengths,
            ))
        requested_languages = _batch_values(
            inputs.get(
                "target_language",
                inputs.get("language", self.config.target_language),
            ),
            batch_size=len(audio_values),
            name="target_language",
        )
        if any(not isinstance(value, str) or not value.strip() for value in requested_languages):
            raise TypeError("Every training row requires a non-empty `target_language`.")
        requested_language = requested_languages[0].strip()
        if any(value.strip() != requested_language for value in requested_languages[1:]):
            raise ValueError(
                "One SeamlessM4T-v2 training batch requires a homogeneous "
                "`target_language`.")
        feature_batch = self.seamless_processor.process_audio(
            waveforms,
            sampling_rate=16_000,
        )
        labels = self.seamless_processor.encode_labels(
            texts,
            target_language=requested_language,
        )
        prepared = {
            "attention_mask": feature_batch.attention_mask,
            "decoder_attention_mask": labels.ne(-100),
            "input_features": feature_batch.input_features,
            "labels": labels,
        }
        for name, value in inputs.items():
            if name not in _RAW_FIELDS and name not in prepared:
                prepared[name] = value
        return prepared

    def _validate_training_runtime(self) -> None:
        return None

    def _save_pretrained(self, save_directory: Path) -> None:
        from voicehub.models.asr_seamless_m4t_v2.native.runtime import save_seamless_m4t_v2_runtime

        if self.runtime is None:
            self.load()
        if self.runtime is None:
            raise RuntimeError("SeamlessM4T-v2 runtime is not loaded.")
        save_seamless_m4t_v2_runtime(
            self.runtime,
            save_directory,
        )

    def export_native_pretrained(
        self,
        save_directory: str | Path,
    ) -> Path:
        from voicehub.models.asr_seamless_m4t_v2.native.runtime import save_seamless_m4t_v2_runtime

        if self.runtime is None:
            self.load_for_training()
        if self.runtime is None:
            raise RuntimeError("SeamlessM4T-v2 runtime is not loaded.")
        return save_seamless_m4t_v2_runtime(
            self.runtime,
            Path(save_directory).expanduser(),
        )


__all__ = ["SeamlessM4Tv2ForSpeechRecognition"]
