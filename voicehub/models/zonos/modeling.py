"""Public Zonos v0.1 API backed by VoiceHub-owned PyTorch code."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Integral
from pathlib import Path
from typing import Any, Callable

import torch
from torch import Tensor

from voicehub.models._shared import finish_audio_output, seeded_inference
from voicehub.models.tts import PreTrainedTTSModel
from voicehub.models.zonos.configuration import ZonosConfig
from voicehub.models.zonos.native.checkpoint import save_zonos_pretrained
from voicehub.models.zonos.native.codec import ZonosCodec
from voicehub.models.zonos.native.frontend import ZonosPhonemeFrontend
from voicehub.models.zonos.native.metadata import ZONOS_SPEAKER_SAFE_CHECKPOINT_PUBLISHED, ZONOS_TRANSFORMER_REPOSITORY
from voicehub.models.zonos.native.runtime import NativeZonosRuntime
from voicehub.models.zonos.native.sampling import ZonosSamplingOptions
from voicehub.outputs import TTSOutput
from voicehub.processing.waveform import load_pcm_wave


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
    resolved = tuple(int(item) for item in lengths.detach().cpu().tolist())
    if any(item <= 0 for item in resolved):
        raise ValueError(f"`{name}` values must be positive.")
    return resolved


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
            raise ValueError("`sampling_rate` must be scalar or contain one rate per "
                             "example.")
        if tensor.dtype == torch.bool or tensor.is_floating_point():
            raise TypeError("`sampling_rate` must use an integer dtype.")
        rates = tuple(int(item) for item in tensor.detach().cpu().tolist())
    if any(item <= 0 for item in rates):
        raise ValueError("`sampling_rate` values must be positive.")
    return rates


def _split_audio_examples(
    value: Any,
    *,
    batch_size: int,
) -> list[Any]:
    if isinstance(value, Tensor):
        if batch_size == 1:
            if value.ndim >= 2 and value.shape[0] == 1:
                return [value[0]]
            return [value]
        if value.ndim < 2 or value.shape[0] != batch_size:
            raise ValueError("Batched Zonos audio must have one leading row per text.")
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
        if (not isinstance(waveform, Tensor) or waveform.ndim == 0 or waveform.shape[0] != batch_size):
            raise ValueError("Batched audio mapping waveforms must have one leading row "
                             "per text.")
        examples = []
        for index in range(batch_size):
            example = {}
            for name, item in value.items():
                if (isinstance(item, Tensor) and item.ndim > 0 and item.shape[0] == batch_size):
                    example[name] = item[index]
                elif (isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and
                      len(item) == batch_size):
                    example[name] = item[index]
                else:
                    example[name] = item
            examples.append(example)
        return examples
    is_sequence = isinstance(value, Sequence)
    is_path_value = isinstance(value, (str, bytes, Path))
    if is_sequence and not is_path_value:
        examples = list(value)
        if batch_size == 1:
            if (len(examples) == 1 and not isinstance(examples[0], (int, float, bool))):
                return examples
            return [value]
        if len(examples) != batch_size:
            raise ValueError("Zonos audio batch size must match the text batch.")
        return examples
    if batch_size == 1:
        return [value]
    raise TypeError("Batched Zonos audio must be a tensor, mapping, or sequence.")


def _trim_audio_example(
    value: Any,
    *,
    length: int,
    index: int,
) -> Any:
    if isinstance(value, Tensor):
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


def _stack_audio_codes(
    values: Any,
    *,
    batch_size: int,
    lengths: tuple[int, ...] | None = None,
) -> tuple[Tensor, Tensor]:
    is_sequence = isinstance(values, Sequence)
    is_text_value = isinstance(values, (str, bytes))
    if isinstance(values, Tensor):
        if values.ndim == 2 and batch_size == 1:
            examples = [values]
        elif values.ndim == 3 and values.shape[0] == batch_size:
            examples = list(values.unbind(0))
        else:
            raise ValueError(
                "Zonos audio codes must have shape [codebook, time] or "
                "[batch, codebook, time].")
    elif is_sequence and not is_text_value:
        examples = list(values)
        if len(examples) != batch_size:
            raise ValueError("Zonos audio-code batch size must match the text batch.")
    else:
        raise TypeError("Zonos audio codes must be a tensor or tensor sequence.")

    normalized: list[Tensor] = []
    for index, example in enumerate(examples):
        if not isinstance(example, Tensor):
            raise TypeError("Every Zonos audio-code example must be a tensor.")
        if example.ndim == 3 and example.shape[0] == 1:
            example = example[0]
        if example.ndim != 2 or example.shape[0] != 9:
            raise ValueError(f"Zonos audio_codes[{index}] must have shape [9, time].")
        if example.dtype == torch.bool or example.is_floating_point():
            raise TypeError("Zonos audio codes must use an integer dtype.")
        length = example.shape[-1] if lengths is None else lengths[index]
        if length > example.shape[-1]:
            raise ValueError(f"`audio_code_lengths[{index}]` exceeds its padded tensor.")
        normalized.append(example[..., :length].long())

    maximum = max(example.shape[-1] for example in normalized)
    device = normalized[0].device
    padded = torch.zeros(
        batch_size,
        9,
        maximum,
        dtype=torch.long,
        device=device,
    )
    resolved_lengths = torch.empty(
        batch_size,
        dtype=torch.long,
        device=device,
    )
    for index, example in enumerate(normalized):
        example = example.to(device=device)
        padded[index, :, :example.shape[-1]] = example
        resolved_lengths[index] = example.shape[-1]
    return padded, resolved_lengths


class ZonosForTextToSpeech(PreTrainedTTSModel):
    """Dense-Transformer Zonos inference and fine-tuning lifecycle."""

    config_class = ZonosConfig
    default_model_name_or_path = ZONOS_TRANSFORMER_REPOSITORY

    def __init__(
        self,
        config: ZonosConfig | str | None = None,
        *,
        model_path: str | None = None,
        device: str = "auto",
        lazy_load: bool = True,
        **config_overrides: Any,
    ) -> None:
        # Runtime components are intentionally accepted through the common
        # provider override channel. This preserves VoiceHub's stable model
        # constructor signature while still allowing applications to inject
        # trusted phonemization, codec, and speaker-embedding boundaries.
        phoneme_frontend = config_overrides.pop("phoneme_frontend", None)
        codec = config_overrides.pop("codec", None)
        speaker_encoder = config_overrides.pop("speaker_encoder", None)
        config = self._coerce_config(
            config,
            model_path=model_path,
            **config_overrides,
        )
        if not str(config.name_or_path).strip():
            config.name_or_path = self.default_model_name_or_path
        config.validate()
        if phoneme_frontend is not None and not isinstance(
                phoneme_frontend,
                ZonosPhonemeFrontend,
        ):
            raise TypeError("`phoneme_frontend` must implement the native Zonos "
                            "frontend protocol.")
        if codec is not None and not isinstance(codec, ZonosCodec):
            raise TypeError("`codec` must implement the ZonosCodec protocol.")
        if speaker_encoder is not None and not callable(speaker_encoder):
            raise TypeError("`speaker_encoder` must be callable or None.")
        self.phoneme_frontend = phoneme_frontend
        self._codec = codec
        self._speaker_encoder = speaker_encoder
        self._runtime: NativeZonosRuntime | None = None
        self.artifacts = None
        self.architecture_config = None
        super().__init__(config, device=device, lazy_load=lazy_load)

    def _runtime_device(self) -> str:
        # The original integration explicitly excluded MPS. The exact native
        # checkpoint is presently guaranteed on CPU and CUDA only.
        if str(self.device).split(":", 1)[0].lower() == "mps":
            self.device = "cpu"
        return self.device

    def _load_pretrained_model(self) -> None:
        runtime = NativeZonosRuntime.from_pretrained(
            self.config.name_or_path,
            revision=self.config.revision,
            cache_dir=self.config.cache_dir,
            token=self.config.token,
            local_files_only=self.config.local_files_only,
            verify_artifacts=self.config.verify_artifacts,
            device=self._runtime_device(),
            dtype=self.config.torch_dtype,
            codec=self._codec,
            phoneme_frontend=self.phoneme_frontend,
        )
        self._runtime = runtime
        self.model = runtime.model
        self.artifacts = runtime.artifacts
        self.architecture_config = runtime.config
        self.config.sample_rate = runtime.config.sample_rate

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
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
            raise ValueError(f"`{name}` must be a finite number.")
        if minimum is not None and value < minimum:
            raise ValueError(f"`{name}` must be at least {minimum}.")
        if maximum is not None and value > maximum:
            raise ValueError(f"`{name}` must be at most {maximum}.")

    def _validate_generation_inputs(
        self,
        model_inputs: dict[str, Any],
    ) -> None:
        language = model_inputs.get("language", "en-us")
        if not isinstance(language, str) or not language.strip():
            raise ValueError("`language` must be a non-empty eSpeak language code.")
        phonemes = model_inputs.get("phonemes")
        if phonemes is not None and (not isinstance(phonemes, str) or not phonemes.strip()):
            raise ValueError("`phonemes` must be a non-empty string or None.")
        speaker_audio_path = model_inputs.get("speaker_audio_path")
        speaker_embedding = model_inputs.get("speaker_embedding")
        if speaker_audio_path is not None and speaker_embedding is not None:
            raise ValueError("Provide either `speaker_audio_path` or `speaker_embedding`, "
                             "not both.")
        if speaker_audio_path is not None:
            if (not isinstance(speaker_audio_path, (str, Path)) or not str(speaker_audio_path).strip()):
                raise ValueError("`speaker_audio_path` must be a local audio path or None.")
            reference = Path(speaker_audio_path).expanduser()
            if not reference.is_file():
                raise FileNotFoundError(f"Zonos reference audio was not found: {reference}.")
        if speaker_embedding is not None:
            if not isinstance(speaker_embedding, Tensor):
                raise TypeError("`speaker_embedding` must be a PyTorch tensor or None.")
            if speaker_embedding.shape[-1:] != (128, ):
                raise ValueError("Zonos `speaker_embedding` must end in 128 features.")
        for name, default, maximum in (
            ("speaking_rate", 15.0, 40.0),
            ("pitch_std", 20.0, 400.0),
            ("fmax", 22_050.0, 24_000.0),
        ):
            self._finite_number(
                model_inputs.get(name, default),
                name=name,
                minimum=0.0,
                maximum=maximum,
            )
        cfg_scale = model_inputs.get("cfg_scale", 2.0)
        self._finite_number(
            cfg_scale,
            name="cfg_scale",
            minimum=0.0,
        )
        # Preserve the stable public contract. The lower-level native sampler
        # can execute guidance-free batches, but changing this wrapper default
        # would make existing saved generation configurations ambiguous.
        if cfg_scale == 1:
            raise ValueError("`cfg_scale` cannot be 1 in the public Zonos wrapper.")
        self._finite_number(
            model_inputs.get("temperature", 1.0),
            name="temperature",
            minimum=0.0,
        )
        for name, default in (("top_p", 0.0), ("min_p", 0.1)):
            self._finite_number(
                model_inputs.get(name, default),
                name=name,
                minimum=0.0,
                maximum=1.0,
            )
        top_k = model_inputs.get("top_k", 0)
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
            raise ValueError("`top_k` must be a non-negative integer.")
        max_new_tokens = model_inputs.get("max_new_tokens", 2_580)
        if (isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens <= 0):
            raise ValueError("`max_new_tokens` must be a positive integer.")
        seed = model_inputs.get("seed")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise TypeError("`seed` must be an integer or None.")
        emotion = model_inputs.get("emotion")
        if emotion is not None:
            if not isinstance(emotion, (list, tuple)) or len(emotion) != 8:
                raise ValueError("`emotion` must contain eight values in the released "
                                 "Zonos order.")
            if any((not isinstance(value, (int, float)) or isinstance(value, bool) or
                    not math.isfinite(value) or not 0 <= value <= 1) for value in emotion):
                raise ValueError("Every `emotion` value must be finite and in [0, 1].")
            if sum(emotion) <= 0:
                raise ValueError("At least one `emotion` value must be positive.")
        decode_audio = model_inputs.get(
            "decode_audio",
            self.config.decode_audio,
        )
        if decode_audio is not True:
            raise ValueError("VoiceHub TTS output requires decoded audio. Set "
                             "`decode_audio=True`.")

    def _speaker_embedding(self, speaker_audio_path: str | None):
        if speaker_audio_path is None:
            return None
        waveform, sample_rate = load_pcm_wave(
            Path(speaker_audio_path).expanduser(),
            preserve_channels=True,
        )
        if waveform.numel() == 0:
            raise ValueError("Zonos reference audio contains no samples.")
        if self._speaker_encoder is not None:
            embedding = self._speaker_encoder(waveform, sample_rate)
        else:
            # Retain compatibility with an explicitly injected historical
            # runtime while keeping the new default free of unsafe pickle
            # loading.
            make_embedding = getattr(
                self.model,
                "make_speaker_embedding",
                None,
            )
            if not callable(make_embedding):
                boundary = (
                    "No official safe speaker-encoder checkpoint is "
                    "published."
                    if not ZONOS_SPEAKER_SAFE_CHECKPOINT_PUBLISHED else "No speaker encoder was configured.")
                raise RuntimeError(
                    "Native Zonos cannot derive a speaker embedding from "
                    f"audio automatically. {boundary} Pass a precomputed "
                    "`speaker_embedding` or inject a trusted "
                    "`speaker_encoder`.")
            embedding = make_embedding(waveform, sample_rate)
        if not isinstance(embedding, Tensor):
            raise TypeError("Zonos speaker encoder must return a tensor.")
        return embedding

    def prepare_training_inputs(
        self,
        inputs: Mapping[str, Any],
        *,
        phase: Any,
    ) -> dict[str, Any]:
        del phase
        batch = dict(inputs)
        if self._runtime is None:
            raise RuntimeError("Zonos runtime is not loaded.")
        prefix = batch.pop("prefix_conditioning", None)
        texts = _pop_unique_alias(
            batch,
            ("texts", "text"),
            description="Zonos text",
        )
        if prefix is not None:
            if not isinstance(prefix, Tensor) or prefix.ndim != 3:
                raise ValueError(
                    "Zonos `prefix_conditioning` must have shape "
                    "[batch, prefix_time, hidden_size].")
            batch_size = prefix.shape[0]
        elif texts is None:
            raise ValueError("Zonos training requires `prefix_conditioning` or text.")
        elif isinstance(texts, str):
            batch_size = 1
        elif isinstance(texts, Sequence) and not isinstance(texts, bytes):
            texts = tuple(texts)
            batch_size = len(texts)
            if batch_size == 0:
                raise ValueError("Zonos training text batches cannot be empty.")
        else:
            raise TypeError("Zonos training text must be a string or string sequence.")

        audio_codes = _pop_unique_alias(
            batch,
            ("audio_codes", "codes"),
            description="Zonos audio-code",
        )
        code_lengths = _batch_lengths(
            batch.pop("audio_code_lengths", None),
            batch_size=batch_size,
            name="audio_code_lengths",
        )
        if audio_codes is None:
            if code_lengths is not None:
                raise ValueError("`audio_code_lengths` requires precomputed `audio_codes`.")
            audio = _pop_unique_alias(
                batch,
                ("audio", "audio_values"),
                description="Zonos audio",
            )
            if audio is None:
                raise ValueError("Zonos training requires `audio_codes` or raw `audio`.")
            root_audio_lengths = batch.pop("audio_lengths", None)
            nested_audio_lengths = None
            if isinstance(audio, Mapping) and "audio_lengths" in audio:
                audio = dict(audio)
                nested_audio_lengths = audio.pop("audio_lengths")
            if (root_audio_lengths is not None and nested_audio_lengths is not None):
                raise ValueError(
                    "Provide `audio_lengths` either beside the audio field "
                    "or inside the audio mapping, not both.")
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
                        example,
                        length=length,
                        index=index,
                    ) for index, (example, length) in enumerate(zip(audio_examples, audio_lengths))
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
            encoded = [
                self._runtime.encode_audio(
                    example,
                    sampling_rate=sampling_rate,
                ) for example, sampling_rate in zip(
                    audio_examples,
                    sampling_rates,
                )
            ]
            audio_codes, code_length_tensor = _stack_audio_codes(
                encoded,
                batch_size=batch_size,
            )
        else:
            audio_codes, code_length_tensor = _stack_audio_codes(
                audio_codes,
                batch_size=batch_size,
                lengths=code_lengths,
            )

        if prefix is not None:
            prepared = {
                "prefix_conditioning": prefix,
                "audio_codes": audio_codes,
                "audio_code_lengths": code_length_tensor,
            }
        else:
            prepared = self._runtime.prepare_training_batch(
                texts,
                audio_codes,
                phonemes=batch.pop("phonemes", None),
                language=batch.pop("language", "en-us"),
                speaker_embedding=batch.pop("speaker_embedding", None),
                emotion=batch.pop(
                    "emotion",
                    (
                        0.3077,
                        0.0256,
                        0.0256,
                        0.0256,
                        0.0256,
                        0.0256,
                        0.2564,
                        0.3077,
                    ),
                ),
                fmax=batch.pop("fmax", 22_050.0),
                pitch_std=batch.pop("pitch_std", 20.0),
                speaking_rate=batch.pop("speaking_rate", 15.0),
                audio_code_lengths=code_length_tensor,
            )
        prepared.update(batch)
        return prepared

    def _generate(
        self,
        text: str,
        *,
        output_file: str | None = None,
        phonemes: str | None = None,
        speaker_audio_path: str | None = None,
        speaker_embedding: Tensor | None = None,
        language: str = "en-us",
        emotion: Sequence[float] | None = None,
        fmax: float = 22_050.0,
        speaking_rate: float = 15.0,
        pitch_std: float = 20.0,
        cfg_scale: float = 2.0,
        max_new_tokens: int = 2_580,
        temperature: float = 1.0,
        top_p: float = 0.0,
        top_k: int = 0,
        min_p: float = 0.1,
        linear: float = 0.0,
        confidence: float = 0.0,
        quadratic: float = 0.0,
        repetition_penalty: float = 3.0,
        repetition_penalty_window: int = 2,
        seed: int | None = None,
        decode_audio: bool = True,
    ) -> TTSOutput:
        if self._runtime is None:
            raise RuntimeError("Zonos runtime is not loaded.")
        if speaker_audio_path is not None:
            speaker_embedding = self._speaker_embedding(speaker_audio_path)
        if emotion is None:
            emotion = (
                0.3077,
                0.0256,
                0.0256,
                0.0256,
                0.0256,
                0.0256,
                0.2564,
                0.3077,
            )
        with seeded_inference(
                seed,
                device=self.device,
                model_type="zonos",
        ) as effective_seed:
            result = self._runtime.generate(
                text,
                phonemes=phonemes,
                language=language,
                speaker_embedding=speaker_embedding,
                emotion=emotion,
                fmax=fmax,
                pitch_std=pitch_std,
                speaking_rate=speaking_rate,
                options=ZonosSamplingOptions(
                    max_new_tokens=max_new_tokens,
                    cfg_scale=cfg_scale,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                    linear=linear,
                    confidence=confidence,
                    quadratic=quadratic,
                    repetition_penalty=repetition_penalty,
                    repetition_penalty_window=repetition_penalty_window,
                ),
                decode_audio=decode_audio,
            )
        if result.audio is None:
            raise RuntimeError("Zonos generation did not decode audio.")
        return finish_audio_output(
            result.audio.cpu()[0],
            result.sample_rate,
            output_file=output_file,
            metadata={
                "language": language.strip().lower().replace("_", "-"),
                "seed": effective_seed,
                "requested_seed": seed,
                "voice_cloned": speaker_embedding is not None,
                "text_frontend": result.text_frontend,
                "architecture": "voicehub-native-zonos-transformer",
            },
        )

    def _save_pretrained(self, save_directory: Path) -> None:
        self.load_for_training()
        save_zonos_pretrained(self.model, save_directory)


ZonosTTS = ZonosForTextToSpeech

__all__ = [
    "ZonosConfig",
    "ZonosForTextToSpeech",
    "ZonosTTS",
]
