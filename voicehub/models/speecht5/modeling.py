"""VoiceHub-native SpeechT5 inference and supervised fine-tuning."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from inspect import signature
from math import isfinite
from numbers import Integral, Real
from pathlib import Path
from typing import Any

from voicehub.audio import AudioInput, load_audio
from voicehub.checkpointing import SafeTensorReader, load_numpy_tensor
from voicehub.dependencies import import_optional
from voicehub.hub import read_json_file, write_json_file
from voicehub.models._shared import finish_audio_output, resolve_torch_dtype, seeded_inference
from voicehub.models.speecht5.metadata import SPEECHT5_HIFIGAN_REPOSITORY, SPEECHT5_REPOSITORY
from voicehub.models.tts import PreTrainedTTSModel
from voicehub.outputs import TTSOutput
from voicehub.training.utils import NATIVE_EXPORT_DIR

from .configuration import SpeechT5Config


class SpeechT5ForTextToSpeech(PreTrainedTTSModel):
    """SpeechT5 synthesis and full acoustic-model fine-tuning."""

    config_class = SpeechT5Config
    default_model_name_or_path = SPEECHT5_REPOSITORY
    passthrough_generation_options = frozenset()
    transformers_model_class = "SpeechT5ForTextToSpeech"
    transformers_processor_class = "SpeechT5Processor"

    def __init__(
        self,
        config: SpeechT5Config | str | Path | None = None,
        *,
        model_path: str | Path | None = None,
        device: str = "auto",
        lazy_load: bool = True,
        token: str | bool | None = None,
        **config_overrides: Any,
    ) -> None:
        config = self._coerce_config(
            config,
            model_path=model_path,
            **config_overrides,
        )
        if not isinstance(config, SpeechT5Config):
            raise TypeError("SpeechT5 requires a SpeechT5Config.")
        config.validate()
        if token is not None and (not isinstance(token, (str, bool)) or
                                  isinstance(token, str) and not token.strip()):
            raise ValueError("`token` must be a non-empty string, boolean, or None.")
        self._token = token.strip() if isinstance(token, str) else token
        self.native_config = None
        self.native_vocoder_config = None
        self.transformers_processor = None
        self.training_model = None
        self.vocoder = None
        self._torch = None
        self._resolved_model_artifacts = None
        self._resolved_vocoder_artifacts = None
        super().__init__(config, device=device, lazy_load=lazy_load)

    @property
    def training_processor(self):
        return self.transformers_processor

    def _hub_kwargs(self) -> dict[str, Any]:
        """Compatibility view of runtime-only Hub transport options."""
        return {
            key: value
            for key, value in {
                "revision": self.config.revision,
                "cache_dir": self.config.cache_dir,
                "local_files_only": self.config.local_files_only,
                "token": self._token,
            }.items() if value is not None
        }

    def _model_source(self) -> str:
        configured = str(self.config.name_or_path or self.default_model_name_or_path)
        source = Path(configured).expanduser()
        native_export = source / NATIVE_EXPORT_DIR
        has_root_checkpoint = any(
            (source / filename).is_file() for filename in ("model.safetensors", "pytorch_model.bin"))
        has_native_checkpoint = (native_export / "model.safetensors").is_file()
        if (source.is_dir() and not has_root_checkpoint and has_native_checkpoint):
            return str(native_export.resolve())
        return configured

    def _vocoder_source(self) -> str:
        configured = Path(self.config.vocoder_name_or_path).expanduser()
        model_source = Path(self._model_source()).expanduser()
        bundled_vocoder = model_source / "vocoder"
        if (model_source.is_dir() and bundled_vocoder.is_dir() and
                self.config.vocoder_name_or_path == SPEECHT5_HIFIGAN_REPOSITORY):
            return str(bundled_vocoder.resolve())
        if (not configured.is_absolute() and model_source.is_dir() and (model_source / configured).exists()):
            return str((model_source / configured).resolve())
        return self.config.vocoder_name_or_path

    @staticmethod
    def _native_model_values(values: Mapping[str, Any]) -> Mapping[str, Any]:
        nested = values.get("native_model_config")
        return nested if isinstance(nested, Mapping) else values

    @staticmethod
    def _native_vocoder_values(values: Mapping[str, Any]) -> Mapping[str, Any]:
        nested = values.get("native_vocoder_config")
        return nested if isinstance(nested, Mapping) else values

    def _load_pretrained_model(self) -> None:
        torch = import_optional(
            "torch",
            model_type=self.config.model_type,
            install_extra=None,
        )
        from voicehub.models.speecht5.artifacts import resolve_speecht5_artifacts, resolve_speecht5_hifigan_artifacts
        from voicehub.models.speecht5.checkpoint import load_speecht5_checkpoint
        from voicehub.models.speecht5.native_configuration import NativeSpeechT5Config, NativeSpeechT5HifiGanConfig
        from voicehub.models.speecht5.native_modeling import SpeechT5ForTextToSpeechModel, SpeechT5HifiGan
        from voicehub.models.speecht5.processing import SpeechT5Processor

        model_artifacts = resolve_speecht5_artifacts(
            self._model_source(),
            revision=self.config.revision,
            cache_dir=self.config.cache_dir,
            token=self._token,
            local_files_only=self.config.local_files_only,
            use_safetensors=self.config.use_safetensors,
            verify_official_integrity=self.config.verify_official_integrity,
        )
        vocoder_artifacts = resolve_speecht5_hifigan_artifacts(
            self._vocoder_source(),
            revision=self.config.vocoder_revision,
            cache_dir=self.config.cache_dir,
            token=self._token,
            local_files_only=self.config.local_files_only,
            use_safetensors=self.config.use_safetensors,
            verify_official_integrity=self.config.verify_official_integrity,
        )
        model_values = (
            self.config.native_model_config or
            self._native_model_values(read_json_file(model_artifacts.config)))
        vocoder_values = (
            self.config.native_vocoder_config or
            self._native_vocoder_values(read_json_file(vocoder_artifacts.config)))
        native_config = NativeSpeechT5Config.from_mapping(model_values)
        native_vocoder_config = NativeSpeechT5HifiGanConfig.from_mapping(vocoder_values)
        if native_config.num_mel_bins != native_vocoder_config.model_in_dim:
            raise ValueError(
                "SpeechT5 acoustic and vocoder mel dimensions do not match: "
                f"{native_config.num_mel_bins} and "
                f"{native_vocoder_config.model_in_dim}.")
        processor = SpeechT5Processor.from_pretrained(model_artifacts.root)
        if processor.feature_extractor.num_mel_bins != native_config.num_mel_bins:
            raise ValueError("SpeechT5 processor and acoustic model mel dimensions differ.")

        model = SpeechT5ForTextToSpeechModel(native_config)
        vocoder = SpeechT5HifiGan(native_vocoder_config)
        load_speecht5_checkpoint(
            model,
            model_artifacts.checkpoint,
            require_official_inventory=model_artifacts.official,
        )
        load_speecht5_checkpoint(
            vocoder,
            vocoder_artifacts.checkpoint,
            vocoder=True,
            require_official_inventory=vocoder_artifacts.official,
        )
        dtype = (
            torch.float32 if self.config.torch_dtype is None else resolve_torch_dtype(
                torch,
                self.config.torch_dtype,
                self.device,
            ))
        model = model.to(device=self.device, dtype=dtype)
        vocoder = vocoder.to(device=self.device, dtype=dtype)
        vocoder.requires_grad_(False)
        self._torch = torch
        self.native_config = native_config
        self.native_vocoder_config = native_vocoder_config
        self.transformers_processor = processor
        self.processor = processor
        self.model = model
        self.vocoder = vocoder
        self.config.sample_rate = native_vocoder_config.sampling_rate
        self.config.native_model_config = native_config.to_dict()
        self.config.native_vocoder_config = native_vocoder_config.to_dict()
        self._resolved_model_artifacts = model_artifacts
        self._resolved_vocoder_artifacts = vocoder_artifacts

    @staticmethod
    def _move_to_device(value: Any, device: str):
        move = getattr(value, "to", None)
        if callable(move):
            moved = move(device)
            return value if moved is None else moved
        if isinstance(value, Mapping):
            return {key: SpeechT5ForTextToSpeech._move_to_device(item, device) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(SpeechT5ForTextToSpeech._move_to_device(item, device) for item in value)
        if isinstance(value, list):
            return [SpeechT5ForTextToSpeech._move_to_device(item, device) for item in value]
        return value

    def _processor_inputs(self, text: str, **kwargs: Any) -> dict[str, Any]:
        encoded = self.transformers_processor(
            text=text,
            return_tensors="pt",
            **kwargs,
        )
        if not isinstance(encoded, Mapping):
            raise TypeError("Native SpeechT5Processor must return a mapping.")
        return dict(self._move_to_device(encoded, self.device))

    @staticmethod
    def _positive_real(
        value: Any,
        *,
        name: str,
        allow_zero: bool = False,
    ) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"`{name}` must be a real number.")
        value = float(value)
        valid = value >= 0.0 if allow_zero else value > 0.0
        if not isfinite(value) or not valid:
            qualifier = "non-negative" if allow_zero else "greater than zero"
            raise ValueError(f"`{name}` must be finite and {qualifier}.")
        return value

    def _normalize_waveform(
        self,
        audio: Any,
        *,
        output_length: Any | None = None,
    ):
        if (isinstance(audio, (tuple, list)) and not hasattr(audio, "shape") and audio and
                not isinstance(audio[0], Real)):
            audio = audio[0]
        waveform = self._torch.as_tensor(
            audio,
            dtype=self._torch.float32,
            device="cpu",
        ).detach()
        if waveform.ndim == 2:
            if waveform.shape[0] != 1:
                raise RuntimeError(
                    "VoiceHub requested one utterance, but native SpeechT5 "
                    f"returned a batch of {waveform.shape[0]}.")
            waveform = waveform[0]
        waveform = waveform.squeeze()
        if waveform.ndim == 0:
            waveform = waveform.reshape(1)
        if waveform.ndim != 1:
            raise RuntimeError(
                "Native SpeechT5 returned non-mono audio with shape "
                f"{tuple(waveform.shape)}.")
        if output_length is not None:
            if hasattr(output_length, "detach"):
                output_length = output_length.detach().cpu()
            if isinstance(output_length, (tuple, list)):
                output_length = output_length[0]
            if hasattr(output_length, "item"):
                output_length = output_length.item()
            if (isinstance(output_length, bool) or not isinstance(output_length, Integral) or
                    output_length <= 0):
                raise RuntimeError("Native SpeechT5 returned an invalid output length.")
            waveform = waveform[:int(output_length)]
        if waveform.numel() == 0:
            raise RuntimeError("Native SpeechT5 returned empty audio.")
        if not bool(self._torch.isfinite(waveform).all().item()):
            raise RuntimeError("Native SpeechT5 returned non-finite audio.")
        return waveform

    @staticmethod
    def _single_tensor_from_mapping(
        values: Mapping[str, Any],
        *,
        source: Path,
    ) -> Any:
        preferred = (
            "speaker_embeddings",
            "speaker_embedding",
            "xvector",
            "embedding",
        )
        for name in preferred:
            if name in values:
                return values[name]
        if len(values) == 1:
            return next(iter(values.values()))
        available = ", ".join(sorted(map(str, values)))
        raise ValueError(
            f"Speaker embedding file {source} must contain one tensor or one "
            f"of {preferred}; found: {available}.")

    def _load_speaker_embedding_file(self, value: str | Path):
        path = Path(value).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"SpeechT5 speaker embedding was not found: {path}.")
        suffix = path.suffix.lower()
        if suffix == ".safetensors":
            with SafeTensorReader(path) as reader:
                values = reader.state_dict(device="cpu")
            return self._single_tensor_from_mapping(values, source=path)
        if suffix == ".npy":
            # VoiceHub implements the bounded NPY tensor format directly; no
            # NumPy package is imported by this path.
            return load_numpy_tensor(path)
        if suffix in {".bin", ".pt", ".pth"}:
            parameters = signature(self._torch.load).parameters
            if "weights_only" not in parameters:
                raise RuntimeError(
                    "This PyTorch version cannot safely load speaker "
                    "embeddings. Use Safetensors or NPY.")
            values = self._torch.load(
                path,
                map_location="cpu",
                weights_only=True,
            )
            if isinstance(values, Mapping):
                return self._single_tensor_from_mapping(values, source=path)
            return values
        raise ValueError("SpeechT5 speaker embeddings must use .safetensors, .npy, .bin, "
                         ".pt, or .pth.")

    def _coerce_speaker_embeddings(
        self,
        speaker_embeddings: Any | None,
        *,
        speaker_embedding_path: str | Path | None = None,
        device: str | None = None,
        batch_size: int | None = None,
    ):
        if speaker_embeddings is not None and speaker_embedding_path is not None:
            raise ValueError("Pass `speaker_embeddings` or `speaker_embedding_path`, not both.")
        if speaker_embeddings is None:
            source = (speaker_embedding_path or self.config.default_speaker_embedding_path)
            if source is not None:
                speaker_embeddings = self._load_speaker_embedding_file(source)
        embedding_dim = int(getattr(
            getattr(self.model, "config", None),
            "speaker_embedding_dim",
            512,
        ))
        if speaker_embeddings is None:
            tensor = self._torch.zeros((1, embedding_dim))
        elif hasattr(speaker_embeddings, "detach"):
            tensor = speaker_embeddings
        else:
            tensor = self._torch.as_tensor(speaker_embeddings)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2 or tensor.shape[-1] != embedding_dim:
            raise ValueError(
                "SpeechT5 speaker embeddings must have shape "
                f"(batch, {embedding_dim}); received {tuple(tensor.shape)}.")
        if batch_size is not None and tensor.shape[0] != batch_size:
            if tensor.shape[0] == 1:
                tensor = tensor.expand(batch_size, -1)
            else:
                raise ValueError(
                    "SpeechT5 speaker embedding batch must be one or match "
                    f"the text batch ({batch_size}).")
        model_dtype = None
        parameters = getattr(self.model, "parameters", None)
        if callable(parameters):
            try:
                model_dtype = next(parameters()).dtype
            except (StopIteration, TypeError):
                pass
        destination = self.device if device is None else device
        return tensor.to(
            device=destination,
            **({} if model_dtype is None else {
                "dtype": model_dtype
            }),
        )

    @staticmethod
    def _rates(value: Any, *, count: int) -> list[int]:
        if hasattr(value, "detach"):
            value = value.detach().reshape(-1).tolist()
        if isinstance(value, Integral) and not isinstance(value, bool):
            values = [int(value)] * count
        elif (isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))):
            values = list(value)
            if len(values) == 1 and count > 1:
                values *= count
        else:
            raise ValueError("Raw SpeechT5 training audio requires positive sampling rates.")
        normalized = []
        for rate in values:
            if hasattr(rate, "detach"):
                rate = rate.detach()
                if rate.numel() == 1:
                    rate = rate.item()
            if isinstance(rate, bool) or not isinstance(rate, Integral) or rate <= 0:
                normalized = []
                break
            normalized.append(int(rate))
        if len(values) != count or len(normalized) != count:
            raise ValueError(
                "SpeechT5 training sampling rates must match the audio batch "
                "and contain positive integers.")
        return normalized

    @staticmethod
    def _batch_size(value: Any) -> int:
        shape = getattr(value, "shape", None)
        if shape is not None:
            if len(shape) == 0:
                raise ValueError("SpeechT5 input IDs must include a token dimension.")
            return 1 if len(shape) == 1 else int(shape[0])
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            if not value:
                raise ValueError("SpeechT5 input IDs cannot be empty.")
            first = value[0]
            return 1 if isinstance(first, Integral) else len(value)
        raise TypeError("SpeechT5 input IDs must be a tensor or sequence.")

    def _training_audio_batch(
        self,
        value: Any,
        sampling_rate: Any,
    ) -> tuple[Any, int, int]:
        mapped_rates = None
        if isinstance(value, Mapping):
            mapped_rates = value.get(
                "sampling_rate",
                value.get("sample_rate"),
            )
            source = None
            for name in ("array", "waveform", "audio", "input_values", "path"):
                if name in value:
                    source = value[name]
                    break
            if source is None:
                raise ValueError(
                    "SpeechT5 audio mappings require array, waveform, audio, "
                    "input_values, or path.")
            value = source

        if isinstance(value, (str, Path, AudioInput)):
            items = [value]
            batched = False
        elif hasattr(value, "ndim") and value.ndim > 1:
            unbind = getattr(value, "unbind", None)
            items = (
                list(unbind(0))
                if callable(unbind) else [value[index] for index in range(int(value.shape[0]))])
            batched = True
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            if not value:
                raise ValueError("SpeechT5 training audio cannot be empty.")
            first = value[0]
            is_batch = (
                isinstance(first, (str, Path, AudioInput, Mapping)) or hasattr(first, "ndim") or
                (isinstance(first, Sequence) and not isinstance(first, (str, bytes, bytearray))))
            items = list(value) if is_batch else [value]
            batched = is_batch
        else:
            items = [value]
            batched = False

        rate_source = mapped_rates if mapped_rates is not None else sampling_rate
        rates = ([None] * len(items) if rate_source is None else self._rates(rate_source, count=len(items)))
        waveforms = []
        for item, rate in zip(items, rates):
            if isinstance(item, (str, Path, AudioInput)):
                loaded = load_audio(
                    item,
                    target_sampling_rate=16_000,
                )
            elif isinstance(item, Mapping):
                item_rate = item.get(
                    "sampling_rate",
                    item.get("sample_rate"),
                )
                effective_rate = (rate if item_rate is None else self._rates(item_rate, count=1)[0])
                path = item.get("path")
                if path is not None and not any(name in item for name in (
                        "array",
                        "waveform",
                        "audio",
                        "input_values",
                )):
                    loaded = load_audio(
                        path,
                        **({} if effective_rate is None else {
                            "sampling_rate": effective_rate
                        }),
                        target_sampling_rate=16_000,
                    )
                else:
                    loaded = load_audio(
                        item,
                        sampling_rate=(None if item_rate is not None else effective_rate),
                        target_sampling_rate=16_000,
                    )
            else:
                if rate is None:
                    raise ValueError("Raw SpeechT5 waveform arrays require positive "
                                     "sampling rates.")
                loaded = load_audio(
                    item,
                    sampling_rate=rate,
                    target_sampling_rate=16_000,
                )
            waveforms.append(loaded.waveform)
        return (
            waveforms if batched else waveforms[0],
            16_000,
            len(waveforms),
        )

    def _training_audio(
        self,
        value: Any,
        sampling_rate: Any,
    ) -> tuple[Any, int]:
        audio, rate, _ = self._training_audio_batch(
            value,
            sampling_rate,
        )
        return audio, rate

    def prepare_training_inputs(
        self,
        inputs: dict[str, Any],
        *,
        phase: str,
    ) -> dict[str, Any]:
        """Create token IDs and masked log-mel labels from raw examples."""
        if phase != "spectrogram":
            raise ValueError(f"Unknown SpeechT5 training phase {phase!r}.")
        prepared = dict(inputs)
        embedding_path = prepared.pop("speaker_embedding_path", None)
        if "input_ids" not in prepared or "labels" not in prepared:
            text = prepared.get("text")
            audio_target = prepared.get(
                "audio_target",
                prepared.get("audio", prepared.get("audio_values")),
            )
            if text is None or audio_target is None:
                raise ValueError(
                    "SpeechT5 fine-tuning requires prepared input_ids/labels "
                    "or raw text/audio.")
            audio_target, sampling_rate, audio_batch = (
                self._training_audio_batch(
                    audio_target,
                    prepared.get("sampling_rate"),
                ))
            text_batch = 1 if isinstance(text, str) else len(text)
            if text_batch != audio_batch:
                raise ValueError(
                    "SpeechT5 text and audio training batches must have the "
                    f"same size; found {text_batch} and {audio_batch}.")
            encoded = self.transformers_processor(
                text=text,
                audio_target=audio_target,
                sampling_rate=sampling_rate,
                padding=True,
                return_tensors="pt",
            )
            if not isinstance(encoded, Mapping):
                raise TypeError("Native SpeechT5Processor must return a training mapping.")
            prepared = dict(encoded)
        batch_size = self._batch_size(prepared["input_ids"])
        if ("speaker_embeddings" in inputs or embedding_path is not None or
                self.config.default_speaker_embedding_path is not None):
            prepared["speaker_embeddings"] = self._coerce_speaker_embeddings(
                inputs.get("speaker_embeddings"),
                speaker_embedding_path=embedding_path,
                device="cpu",
                batch_size=batch_size,
            )
        return prepared

    def _generate(
        self,
        text: str,
        *,
        speaker_embeddings: Any | None = None,
        speaker_embedding_path: str | Path | None = None,
        threshold: float = 0.5,
        minlenratio: float = 0.0,
        maxlenratio: float = 20.0,
        output_file: str | Path | None = None,
        seed: int | None = None,
        **generation_options: Any,
    ) -> TTSOutput:
        if generation_options:
            names = ", ".join(sorted(generation_options))
            raise ValueError(f"Unsupported native SpeechT5 generation option(s): {names}.")
        threshold = self._positive_real(
            threshold,
            name="threshold",
            allow_zero=True,
        )
        if threshold > 1.0:
            raise ValueError("`threshold` must be between 0 and 1.")
        minlenratio = self._positive_real(
            minlenratio,
            name="minlenratio",
            allow_zero=True,
        )
        maxlenratio = self._positive_real(maxlenratio, name="maxlenratio")
        if minlenratio > maxlenratio:
            raise ValueError("`minlenratio` cannot exceed `maxlenratio`.")
        inputs = self._processor_inputs(text)
        input_ids = inputs["input_ids"]
        batch_size = (
            self._batch_size(input_ids) if (
                hasattr(input_ids, "shape") or isinstance(input_ids, Sequence) and not isinstance(
                    input_ids,
                    (str, bytes, bytearray),
                )) else None)
        speaker = self._coerce_speaker_embeddings(
            speaker_embeddings,
            speaker_embedding_path=speaker_embedding_path,
            batch_size=batch_size,
        )
        with seeded_inference(
                seed,
                device=self.device,
                model_type=self.config.model_type,
        ) as effective_seed:
            with self._torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    speaker_embeddings=speaker,
                    threshold=threshold,
                    minlenratio=minlenratio,
                    maxlenratio=maxlenratio,
                    vocoder=self.vocoder,
                    return_output_lengths=True,
                )
        if not isinstance(generated, tuple) or len(generated) < 2:
            raise RuntimeError("Native SpeechT5 did not return requested waveform lengths.")
        waveform = self._normalize_waveform(
            generated[0],
            output_length=generated[1],
        )
        return finish_audio_output(
            waveform,
            self.sample_rate,
            output_file=output_file,
            metadata={
                "backend":
                "voicehub-native",
                "vocoder":
                self.config.vocoder_name_or_path,
                "speaker_embedding": (
                    "provided" if (
                        speaker_embeddings is not None or speaker_embedding_path is not None or
                        self.config.default_speaker_embedding_path is not None) else "zero"),
                "seed":
                effective_seed,
                "requested_seed":
                seed,
            },
        )

    def _prepare_for_training(self) -> None:
        self.model.train()
        self.vocoder.eval()
        self.vocoder.requires_grad_(False)

    def _prepare_for_inference(self) -> None:
        self.model.eval()
        self.vocoder.eval()

    def _save_pretrained(self, save_directory: Path) -> None:
        if self.model is None or self.vocoder is None:
            raise RuntimeError("Load SpeechT5 before exporting native artifacts.")
        from voicehub.models.speecht5.checkpoint import save_speecht5_checkpoint

        save_directory.mkdir(parents=True, exist_ok=True)
        vocoder_directory = save_directory / "vocoder"
        vocoder_directory.mkdir(parents=True, exist_ok=True)
        runtime_config = self.config.to_dict()
        runtime_config.update({
            "name_or_path": ".",
            "vocoder_name_or_path": "vocoder",
            "native_model_config": self.native_config.to_dict(),
            "native_vocoder_config": self.native_vocoder_config.to_dict(),
        })
        write_json_file(save_directory / "config.json", runtime_config)
        write_json_file(
            vocoder_directory / "config.json",
            self.native_vocoder_config.to_dict(),
        )
        self.transformers_processor.save_pretrained(save_directory)
        save_speecht5_checkpoint(
            self.model,
            save_directory / "model.safetensors",
        )
        save_speecht5_checkpoint(
            self.vocoder,
            vocoder_directory / "model.safetensors",
            vocoder=True,
        )


SpeechT5TTS = SpeechT5ForTextToSpeech

__all__ = [
    "SpeechT5Config",
    "SpeechT5ForTextToSpeech",
    "SpeechT5TTS",
]
