"""Public ZONOS2 API backed entirely by VoiceHub-owned PyTorch code."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from numbers import Integral
from pathlib import Path
from typing import Any

import torch

from voicehub.models._shared import finish_audio_output, seeded_inference
from voicehub.models.tts import PreTrainedTTSModel
from voicehub.models.zonos2.configuration import Zonos2Config
from voicehub.models.zonos2.native.metadata import ZONOS2_OFFICIAL_CHECKPOINT
from voicehub.models.zonos2.native.runtime import NativeZonos2Runtime
from voicehub.models.zonos2.native.sampling import Zonos2SamplingOptions
from voicehub.outputs import TTSOutput


def _pop_unique_alias(
    values: dict[str, Any],
    names: tuple[str, ...],
    *,
    description: str,
) -> Any:
    present = [name for name in names if name in values]
    if len(present) > 1:
        raise ValueError(f"Provide only one {description} field; received {present!r}.")
    return values.pop(present[0]) if present else None


def _batch_lengths(
    value: Any,
    *,
    batch_size: int,
    name: str,
) -> tuple[int, ...] | None:
    if value is None:
        return None
    lengths = torch.as_tensor(value)
    if lengths.ndim == 0 and batch_size == 1:
        lengths = lengths.unsqueeze(0)
    if lengths.ndim != 1 or lengths.numel() != batch_size:
        raise ValueError(f"`{name}` must contain one length per example.")
    if lengths.dtype == torch.bool or lengths.is_floating_point():
        raise TypeError(f"`{name}` must use an integer dtype.")
    result = tuple(int(item) for item in lengths.detach().cpu().tolist())
    if any(item <= 0 for item in result):
        raise ValueError(f"`{name}` values must be positive.")
    return result


def _batch_sampling_rates(
    value: Any,
    *,
    batch_size: int,
) -> tuple[int | None, ...]:
    if value is None:
        return (None, ) * batch_size
    if isinstance(value, Integral) and not isinstance(value, bool):
        rates = (int(value), ) * batch_size
    else:
        tensor = torch.as_tensor(value)
        if tensor.ndim == 0:
            tensor = tensor.repeat(batch_size)
        if tensor.ndim != 1 or tensor.numel() != batch_size:
            raise ValueError("`sampling_rate` must be scalar or contain one rate per example.")
        if tensor.dtype == torch.bool or tensor.is_floating_point():
            raise TypeError("`sampling_rate` must use an integer dtype.")
        rates = tuple(int(item) for item in tensor.detach().cpu().tolist())
    if any(item <= 0 for item in rates):
        raise ValueError("`sampling_rate` values must be positive.")
    return rates


def _split_audio_codes(
    value: Any,
    *,
    batch_size: int,
    lengths: tuple[int, ...] | None,
) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        if value.ndim == 2 and batch_size == 1:
            examples = [value]
        elif value.ndim == 3 and value.shape[0] == batch_size:
            examples = list(value.unbind(0))
        else:
            raise ValueError(
                "ZONOS2 audio_codes must be [frames, codebooks] or "
                "[batch, frames, codebooks].")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        examples = list(value)
        if len(examples) != batch_size:
            raise ValueError("ZONOS2 audio-code batch size must match the text batch.")
    else:
        raise TypeError("ZONOS2 audio_codes must be a tensor or tensor sequence.")
    if any(not isinstance(item, torch.Tensor) for item in examples):
        raise TypeError("Every ZONOS2 audio-code example must be a tensor.")
    if lengths is None:
        return examples
    trimmed = []
    for index, (item, length) in enumerate(zip(examples, lengths)):
        if length > item.shape[0]:
            raise ValueError(f"`audio_code_lengths[{index}]` exceeds its padded tensor.")
        trimmed.append(item[:length])
    return trimmed


def _split_audio_examples(
    value: Any,
    *,
    batch_size: int,
) -> list[Any]:
    if isinstance(value, torch.Tensor):
        if batch_size == 1:
            if value.ndim >= 2 and value.shape[0] == 1:
                return [value[0]]
            return [value]
        if value.ndim < 2 or value.shape[0] != batch_size:
            raise ValueError("Batched ZONOS2 audio must have one leading row per text.")
        return list(value.unbind(0))
    if isinstance(value, Mapping):
        if batch_size == 1:
            return [value]
        waveform_name = next(
            (name for name in ("array", "waveform", "audio", "input_values") if name in value),
            None,
        )
        if waveform_name is None:
            raise ValueError("Batched audio mappings require a waveform-like field.")
        waveform = value[waveform_name]
        if not isinstance(waveform, torch.Tensor) or waveform.shape[0] != batch_size:
            raise ValueError("Batched audio mapping waveforms must have one leading row "
                             "per text.")
        examples = []
        for index in range(batch_size):
            example = {}
            for name, item in value.items():
                if (isinstance(item, torch.Tensor) and item.ndim > 0 and item.shape[0] == batch_size):
                    example[name] = item[index]
                elif (isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and
                      len(item) == batch_size):
                    example[name] = item[index]
                else:
                    example[name] = item
            examples.append(example)
        return examples
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, Path)):
        examples = list(value)
        if batch_size == 1:
            if len(examples) == 1 and not isinstance(examples[0], (int, float, bool)):
                return examples
            return [value]
        if len(examples) != batch_size:
            raise ValueError("ZONOS2 audio batch size must match the text batch.")
        return examples
    if batch_size == 1:
        return [value]
    raise TypeError("Batched ZONOS2 audio must be a tensor, mapping, or sequence.")


def _trim_audio_example(
    value: Any,
    *,
    length: int,
    index: int,
) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim == 0 or length > value.shape[-1]:
            raise ValueError(f"`audio_lengths[{index}]` exceeds its padded waveform.")
        return value[..., :length]
    if isinstance(value, Mapping):
        result = dict(value)
        waveform_name = next(
            (name for name in ("array", "waveform", "audio", "input_values") if name in result),
            None,
        )
        if waveform_name is None:
            raise ValueError(
                "Audio mappings require a waveform-like field when "
                "`audio_lengths` is provided.")
        result[waveform_name] = _trim_audio_example(
            result[waveform_name],
            length=length,
            index=index,
        )
        return result
    raise TypeError("`audio_lengths` can only trim tensor or mapping waveforms.")


class Zonos2ForTextToSpeech(PreTrainedTTSModel):
    """Cross-device ZONOS2 synthesis and differentiable fine-tuning."""

    config_class = Zonos2Config
    default_model_name_or_path = ZONOS2_OFFICIAL_CHECKPOINT

    def __init__(
        self,
        config: Zonos2Config | str | None = None,
        *,
        model_path: str | None = None,
        device: str = "auto",
        lazy_load: bool = True,
        **config_overrides: Any,
    ) -> None:
        config = self._coerce_config(
            config,
            model_path=model_path,
            **config_overrides,
        )
        if not str(config.name_or_path).strip():
            config.name_or_path = self.default_model_name_or_path
        config.validate()
        self._runtime: NativeZonos2Runtime | None = None
        self.artifacts = None
        self.architecture_config = None
        super().__init__(config, device=device, lazy_load=lazy_load)

    def _load_pretrained_model(self) -> None:
        runtime = NativeZonos2Runtime.from_pretrained(
            self.config.name_or_path,
            architecture=(self.config.architecture if self.config.architecture else None),
            revision=self.config.revision,
            cache_dir=self.config.cache_dir,
            token=self.config.token,
            local_files_only=self.config.local_files_only,
            verify_artifacts=self.config.verify_artifacts,
            device=self.device,
            dtype=self.config.torch_dtype,
        )
        self._runtime = runtime
        self.model = runtime.model
        self.artifacts = runtime.artifacts
        self.architecture_config = runtime.config
        self.config.sample_rate = 44_100
        if self.config.compile_model and not self.is_training_load:
            compiled = torch.compile(self.model)
            self.model = compiled
            runtime.model = compiled

    def _validate_training_runtime(self) -> None:
        if self.config.compile_model:
            # Training load deliberately bypasses compilation. The flag remains
            # valid for the later transition back to serving.
            return

    def _prepare_for_training(self) -> None:
        if self.model is not None:
            self.model.train()

    def _prepare_for_inference(self) -> None:
        if self.model is not None:
            self.model.eval()

    @staticmethod
    def _finite_number(
        value: Any,
        *,
        name: str,
        minimum: float | None = None,
        maximum: float | None = None,
    ) -> None:
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value)):
            raise ValueError(f"`{name}` must be a finite number.")
        if minimum is not None and value < minimum:
            raise ValueError(f"`{name}` must be at least {minimum}.")
        if maximum is not None and value > maximum:
            raise ValueError(f"`{name}` must be at most {maximum}.")

    def _validate_generation_inputs(self, model_inputs: dict[str, Any]) -> None:
        language = model_inputs.get("language", "en_us")
        if not isinstance(language, str) or not language.strip():
            raise ValueError("`language` must be a non-empty language code.")
        speaker_audio_path = model_inputs.get("speaker_audio_path")
        speaker_embedding = model_inputs.get("speaker_embedding")
        if speaker_audio_path is not None and speaker_embedding is not None:
            raise ValueError("Provide either `speaker_audio_path` or `speaker_embedding`, "
                             "not both.")
        if speaker_audio_path is not None:
            if not isinstance(speaker_audio_path, (str, Path)):
                raise TypeError("`speaker_audio_path` must be a local path.")
            reference = Path(speaker_audio_path).expanduser()
            if not reference.is_file():
                raise FileNotFoundError(f"ZONOS2 reference audio was not found: {reference}.")
        self._finite_number(
            model_inputs.get("temperature", 1.15),
            name="temperature",
            minimum=0.0,
        )
        self._finite_number(
            model_inputs.get("repetition_penalty", 1.2),
            name="repetition_penalty",
            minimum=0.000_001,
        )
        for name, default in (("top_p", 0.0), ("min_p", 0.18)):
            self._finite_number(
                model_inputs.get(name, default),
                name=name,
                minimum=0.0,
                maximum=1.0,
            )
        top_k = model_inputs.get("top_k", 106)
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
            raise ValueError("`top_k` must be a non-negative integer.")
        max_new_tokens = model_inputs.get("max_new_tokens", 1_024)
        if (isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens <= 0):
            raise ValueError("`max_new_tokens` must be a positive integer.")
        speed = model_inputs.get("speed")
        if speed is not None:
            self._finite_number(speed, name="speed", minimum=0.000_001)
        seed = model_inputs.get("seed")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise TypeError("`seed` must be an integer or None.")
        if (not self.config.decode_audio or model_inputs.get("decode_audio") is False):
            raise ValueError(
                "VoiceHub TTS output requires decoded audio. Set "
                "`decode_audio=True` for ZONOS2 generation.")

    def prepare_training_inputs(
        self,
        inputs: Mapping[str, Any],
        *,
        phase: Any,
    ) -> dict[str, Any]:
        """Prepare raw text plus cached or on-the-fly frozen DAC codebooks."""
        del phase
        if self._runtime is None:
            raise RuntimeError("ZONOS2 runtime is not loaded.")
        batch = dict(inputs)
        if "input_ids" in batch:
            return batch
        texts = _pop_unique_alias(
            batch,
            ("texts", "text"),
            description="ZONOS2 text",
        )
        if texts is None:
            raise ValueError("ZONOS2 training requires `text` or `texts`.")
        if isinstance(texts, str):
            batch_size = 1
        else:
            texts = list(texts)
            batch_size = len(texts)
            if batch_size == 0:
                raise ValueError("ZONOS2 training text batches cannot be empty.")
        audio_codes = batch.pop("audio_codes", None)
        if audio_codes is None:
            audio = _pop_unique_alias(
                batch,
                ("audio", "audio_values"),
                description="ZONOS2 audio",
            )
            if audio is None:
                raise ValueError("ZONOS2 training requires `audio_codes` or raw `audio`.")
            root_audio_lengths = batch.pop("audio_lengths", None)
            nested_audio_lengths = None
            if isinstance(audio, Mapping) and "audio_lengths" in audio:
                audio = dict(audio)
                nested_audio_lengths = audio.pop("audio_lengths")
            if (root_audio_lengths is not None and nested_audio_lengths is not None):
                raise ValueError(
                    "Provide `audio_lengths` either beside the audio field or "
                    "inside the audio mapping, not both.")
            audio_lengths = _batch_lengths(
                (root_audio_lengths if root_audio_lengths is not None else nested_audio_lengths),
                batch_size=batch_size,
                name="audio_lengths",
            )
            audio_examples = _split_audio_examples(
                audio,
                batch_size=batch_size,
            )
            if audio_lengths is not None:
                audio_examples = [
                    _trim_audio_example(
                        item,
                        length=length,
                        index=index,
                    ) for index, (item, length) in enumerate(zip(audio_examples, audio_lengths))
                ]
            rate_value = _pop_unique_alias(
                batch,
                (
                    "sampling_rate",
                    "sampling_rates",
                    "audio_sampling_rate",
                    "audio_sampling_rates",
                ),
                description="audio sampling-rate",
            )
            sampling_rates = _batch_sampling_rates(
                rate_value,
                batch_size=batch_size,
            )
            audio_codes = [
                self._runtime.encode_audio(
                    item,
                    sampling_rate=sampling_rate,
                ) for item, sampling_rate in zip(
                    audio_examples,
                    sampling_rates,
                )
            ]
        else:
            code_lengths = _batch_lengths(
                batch.pop("audio_code_lengths", None),
                batch_size=batch_size,
                name="audio_code_lengths",
            )
            audio_codes = _split_audio_codes(
                audio_codes,
                batch_size=batch_size,
                lengths=code_lengths,
            )
        speaker_embeddings = batch.pop("speaker_embeddings", None)
        prepared = self._runtime.prepare_training_batch(
            texts,
            audio_codes,
            speaker_embeddings=speaker_embeddings,
        )
        prepared.update(batch)
        return prepared

    def _generate(
        self,
        text: str,
        *,
        output_file: str | None = None,
        speaker_audio_path: str | None = None,
        speaker_embedding: torch.Tensor | None = None,
        language: str = "en_us",
        speed: float | None = None,
        speaking_rate_bucket: int | None = None,
        quality_buckets=None,
        temperature: float = 1.15,
        top_k: int = 106,
        top_p: float = 0.0,
        min_p: float = 0.18,
        max_new_tokens: int = 1_024,
        repetition_window: int = 50,
        repetition_penalty: float = 1.2,
        repetition_codebooks: int = 8,
        seed: int | None = None,
        accurate_mode: bool = True,
        clean_speaker_background: bool = False,
        text_normalization: bool = True,
        decode_audio: bool = True,
    ) -> TTSOutput:
        if self._runtime is None:
            raise RuntimeError("ZONOS2 runtime is not loaded.")
        normalized_language = language.strip().lower().replace("-", "_")
        with seeded_inference(
                seed,
                device=self.device,
                model_type="zonos2",
        ) as effective_seed:
            result = self._runtime.generate(
                text,
                options=Zonos2SamplingOptions(
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    min_p=min_p,
                    repetition_window=repetition_window,
                    repetition_penalty=repetition_penalty,
                    repetition_codebooks=repetition_codebooks,
                    seed=effective_seed,
                ),
                speaker_audio=speaker_audio_path,
                speaker_embedding=speaker_embedding,
                speed=speed,
                speaking_rate_bucket=speaking_rate_bucket,
                quality_buckets=quality_buckets,
                clean_speaker_background=clean_speaker_background,
                accurate_mode=accurate_mode,
                text_normalization=text_normalization,
                decode_audio=decode_audio,
            )
        return finish_audio_output(
            result.audio,
            result.sample_rate,
            output_file=output_file,
            metadata={
                "language":
                normalized_language,
                "eos_frame":
                result.eos_frame,
                "voice_cloned": (speaker_audio_path is not None or speaker_embedding is not None),
                "seed":
                effective_seed,
                "requested_seed":
                seed,
                "text_frontend":
                result.text_frontend,
                "checkpoint_representation": (
                    "pinned-community-safetensors-conversion"
                    if self.artifacts.safe_conversion else "safetensors"),
            },
        )


Zonos2TTS = Zonos2ForTextToSpeech

__all__ = [
    "Zonos2Config",
    "Zonos2ForTextToSpeech",
    "Zonos2TTS",
]
