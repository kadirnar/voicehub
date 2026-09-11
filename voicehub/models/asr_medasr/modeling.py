"""Native MedASR raw-audio inference, fine-tuning, and export wrapper."""

from __future__ import annotations

import shutil
from collections.abc import Mapping, Sequence
from numbers import Integral
from pathlib import Path
from typing import Any

from voicehub.hub import read_json_file, write_json_file
from voicehub.models.asr_medasr.configuration import MedASRASRConfig
from voicehub.models.audio import PreTrainedASRModel
from voicehub.outputs import ASROutput

_ENGLISH = frozenset({"en", "eng", "english"})
_RAW_TRAINING_FIELDS = frozenset({
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
    except ModuleNotFoundError:  # pragma: no cover - package invariant
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
        raise ValueError(f"`{name}` contains {len(values)} values for a batch of "
                         f"{batch_size}.")
    return values


class MedASRForSpeechRecognition(PreTrainedASRModel):
    """Run and fine-tune Google MedASR with VoiceHub and PyTorch only."""

    config_class = MedASRASRConfig
    default_model_name_or_path = "google/medasr"
    architecture_family = "ctc"
    training_support = "native"
    supports_generic_finetuning = True
    supports_gradient_checkpointing = True
    native_checkpoint_format = "voicehub-native-medasr-ctc-v1"

    def __init__(
        self,
        config: MedASRASRConfig | str | Path | None = None,
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
        self.medasr_processor: Any | None = None
        self.training_processor: Any | None = None
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
        if (torch.device(self.device).type == "cpu" and dtype == torch.float16):
            raise ValueError(
                "Native MedASR does not support float16 execution on CPU; "
                "use float32 or bfloat16.")
        return dtype

    @staticmethod
    def _validate_architecture(values: Mapping[str, Any]) -> None:
        model_type = str(values.get("model_type", ""), ).strip().lower()
        if model_type not in {"asr_medasr", "lasr_ctc"}:
            raise ValueError(
                "Native MedASR requires a LASR CTC checkpoint; received "
                f"{model_type or '<missing>'!r}.")
        if "auto_map" in values:
            raise ValueError("Native MedASR rejects remote-code `auto_map` metadata.")
        architectures = values.get('architectures', ())
        if isinstance(architectures, str):
            architectures = (architectures, )
        if (not isinstance(architectures, Sequence) or isinstance(architectures, (str, bytes))):
            raise TypeError("MedASR `architectures` must be a sequence.")
        supported = {
            "LasrForCTC",
            "MedASRForCTC",
            "MedASRForSpeechRecognition",
        }
        if architectures and not any(str(name) in supported for name in architectures):
            raise ValueError(
                "Native MedASR supports the LASR CTC graph only; received: " +
                ", ".join(str(name) for name in architectures))

    def _load_pretrained_model(self) -> None:
        import torch

        from voicehub.checkpointing import SafeTensorReader
        from voicehub.models.asr_medasr.native.artifacts import resolve_medasr_artifacts
        from voicehub.models.asr_medasr.native.checkpoint import (
            MedASRCheckpointAdapter,
            validate_published_medasr_inventory,
        )
        from voicehub.models.asr_medasr.native.configuration import MedASRConfig
        from voicehub.models.asr_medasr.native.modeling import MedASRForCTC
        from voicehub.models.asr_medasr.native.processing import MedASRProcessor

        source = (self.config.name_or_path or self.default_model_name_or_path)
        artifacts = resolve_medasr_artifacts(
            source,
            checkpoint_filename=self.config.checkpoint_filename,
            revision=self.config.revision,
            cache_dir=self.config.cache_dir,
            token=self._hub_token,
            local_files_only=self.config.local_files_only,
        )
        values = read_json_file(artifacts.config)
        self._validate_architecture(values)
        native_config = MedASRConfig.from_dict(values)
        if native_config.sampling_rate != self.config.sample_rate:
            raise ValueError("MedASR provider/model sample-rate mismatch.")
        processor = MedASRProcessor.from_artifacts(
            config=native_config,
            tokenizer_json=artifacts.tokenizer,
            tokenizer_config=artifacts.tokenizer_config,
            preprocessor_config=artifacts.preprocessor_config,
            processor_config=artifacts.processor_config,
        )
        model = MedASRForCTC(
            native_config,
            initialize=False,
        )
        adapter = MedASRCheckpointAdapter()
        dtype = self._model_dtype()
        with SafeTensorReader(artifacts.checkpoint) as reader:
            validate_published_medasr_inventory(
                reader,
                source=artifacts.source,
                revision=artifacts.revision,
            )
            adapter.load_assign_streaming(
                model,
                reader,
                native_config,
                device=self.device,
                dtype=dtype,
                strict=True,
            )
        model.to(device=self.device)
        self.artifacts = artifacts
        self.native_config = native_config
        self.medasr_processor = processor
        self.training_processor = processor
        self.checkpoint_adapter = adapter.qualified_id
        self.model = model

    @staticmethod
    def _validate_inference_request(
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
    ) -> str:
        if task != "transcribe":
            raise ValueError("MedASR is a transcription-only CTC model.")
        if language is None:
            resolved_language = "en"
        elif (isinstance(language, str) and language.strip().lower() in _ENGLISH):
            resolved_language = "en"
        else:
            raise ValueError("The released MedASR checkpoint is English-only.")
        if return_timestamps is not False:
            raise ValueError(
                "Native MedASR does not claim timestamp alignment; "
                "`return_timestamps` must be False.")
        unsupported = {
            "chunk_length_s": chunk_length_s,
            "stride_length_s": stride_length_s,
            "max_new_tokens": max_new_tokens,
            "hotwords": hotwords,
        }
        active = [name for name, value in unsupported.items() if value is not None]
        if active:
            raise ValueError(
                "Native MedASR currently supports complete-waveform greedy "
                "CTC only; unsupported option(s): " + ", ".join(active))
        if batch_size not in (None, 1):
            raise ValueError("One MedASR request requires `batch_size=1`.")
        if num_beams not in (None, 1):
            raise ValueError("MedASR uses greedy CTC decoding and requires "
                             "`num_beams=1`.")
        return resolved_language

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

        resolved_language = self._validate_inference_request(
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
        if (self.model is None or self.native_config is None or self.medasr_processor is None):
            raise RuntimeError("Native MedASR runtime is not loaded.")
        materialized = load_native_audio(
            audio,
            sampling_rate=sampling_rate,
            target_sampling_rate=self.native_config.sampling_rate,
        )
        prepared = self.medasr_processor.prepare_audio_batch((materialized.waveform, ), )
        parameter = next(self.model.parameters())
        input_features = prepared["input_features"].to(
            device=parameter.device,
            dtype=parameter.dtype,
        )
        attention_mask = prepared["attention_mask"].to(device=parameter.device, )
        with torch.inference_mode():
            outputs = self.model(
                input_features,
                attention_mask=attention_mask,
            )
        valid_frames = int(outputs.encoded_lengths[0].item())
        token_ids = outputs.logits[
            0,
            :valid_frames,
        ].argmax(dim=-1).tolist()
        decoded = self.medasr_processor.tokenizer.decode_ctc(
            token_ids,
            skip_special_tokens=True,
        )
        return ASROutput(
            text=decoded.text,
            language=resolved_language,
            duration=materialized.duration,
            metadata={
                "architecture": "lasr-ctc",
                "architecture_family": "ctc",
                "backend": "voicehub-native",
                "checkpoint_adapter": self.checkpoint_adapter,
                "checkpoint_revision": (None if self.artifacts is None else self.artifacts.revision),
                "domain": "medical-dictation",
                "logit_frames": valid_frames,
                "model": "medasr",
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
            raise ValueError("MedASR training audio must be rank one or rank two.")
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
        """Create differentiable LASR inputs and padded CTC labels."""
        from voicehub.processing.waveform import load_native_audio

        del phase
        if "input_features" in inputs and "labels" in inputs:
            return dict(inputs)
        if self.model is None:
            self.load_for_training()
        if (self.native_config is None or self.medasr_processor is None):
            raise RuntimeError("Native MedASR training runtime is not loaded.")
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
        elif (isinstance(text, Sequence) and not isinstance(text, (str, bytes))):
            texts = tuple(text)
            text_is_batch = True
        else:
            raise ValueError("MedASR training records require `text`, "
                             "`transcription`, or `transcript`.")
        if (not texts or any(not isinstance(value, str) or not value.strip() for value in texts)):
            raise ValueError("MedASR training transcripts must be non-empty strings.")
        if audio is None:
            raise ValueError("MedASR training records require `audio`.")
        audio_values = self._audio_batch(
            audio,
            text_is_batch=text_is_batch,
        )
        if len(audio_values) != len(texts):
            raise ValueError("MedASR training requires one transcript per waveform.")
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
        waveforms = tuple(
            load_native_audio(
                value,
                sampling_rate=rate,
                target_sampling_rate=self.native_config.sampling_rate,
                num_samples=length,
            ).waveform for value, rate, length in zip(
                audio_values,
                rates,
                lengths,
            ))
        prepared = self.medasr_processor(
            waveforms,
            text=texts,
        )
        for name, value in inputs.items():
            if (name not in _RAW_TRAINING_FIELDS and name not in prepared):
                prepared[name] = value
        return prepared

    def _validate_training_runtime(self) -> None:
        return None

    def _portable_state_dict(self) -> dict[str, Any]:
        import torch

        from voicehub.models.asr_medasr.native.checkpoint import (
            native_medasr_tensor_dtypes,
            native_medasr_tensor_shapes,
        )

        if self.model is None or self.native_config is None:
            raise RuntimeError("Native MedASR runtime is not loaded.")
        state = dict(self.model.state_dict())
        expected_shapes = native_medasr_tensor_shapes(self.native_config)
        expected_dtypes = native_medasr_tensor_dtypes(self.native_config)
        expected = set(expected_shapes)
        actual = set(state)
        if actual != expected:
            raise ValueError(
                "MedASR export requires the exact model namespace; "
                f"missing={sorted(expected - actual)[:5]!r}, "
                f"extra={sorted(actual - expected)[:5]!r}.")
        for name, value in state.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"MedASR export value {name!r} is not a tensor.")
            if tuple(value.shape) != expected_shapes[name]:
                raise ValueError(
                    f"MedASR export tensor {name!r} has shape "
                    f"{tuple(value.shape)}, expected {expected_shapes[name]}.")
            if value.device.type == "meta":
                raise ValueError(f"MedASR export tensor {name!r} is not materialized.")
            if value.layout != torch.strided:
                raise ValueError(f"MedASR export tensor {name!r} must use strided layout.")
            if value.is_quantized:
                raise ValueError(f"MedASR export tensor {name!r} cannot be quantized.")
            expected_dtype = expected_dtypes[name]
            if expected_dtype == "I64":
                if value.dtype != torch.int64:
                    raise TypeError(f"MedASR export buffer {name!r} must use torch.int64.")
            elif not value.is_floating_point() or value.is_complex():
                raise TypeError(f"MedASR export weight {name!r} must be real floating-point.")
        return state

    def _save_pretrained(self, save_directory: Path) -> None:
        from voicehub.checkpointing import save_safetensors
        from voicehub.models.asr_medasr.native.metadata import MEDASR_RECIPE_REVISION, MEDASR_SOURCE_REVISION

        if (self.model is None or self.native_config is None or self.medasr_processor is None):
            self.load()
        state = self._portable_state_dict()
        save_directory.mkdir(parents=True, exist_ok=True)
        save_safetensors(
            state,
            save_directory / "model.safetensors",
            metadata={
                "architecture": "lasr-ctc",
                "format": self.native_checkpoint_format,
                "source_code_revision": MEDASR_SOURCE_REVISION,
            },
        )
        values = self.native_config.to_dict()
        values.update({
            'architectures': ["MedASRForCTC"],
            "model_type":
            self.config.model_type,
            "source_artifact_revision": (None if self.artifacts is None else self.artifacts.revision),
            "source_code_revision":
            MEDASR_SOURCE_REVISION,
            "source_recipe_revision":
            MEDASR_RECIPE_REVISION,
            "voicehub_checkpoint_format":
            self.native_checkpoint_format,
            "voicehub_provider":
            self.config.model_type,
        })
        write_json_file(save_directory / "config.json", values)
        self.medasr_processor.save_pretrained(save_directory)
        source_root = (Path(__file__).parents[2] / 'models/asr_medasr/native')
        for filename in (
                "MODEL_TERMS_NOTICE",
                "NOTICE",
                "THIRD_PARTY_LICENSE",
        ):
            source = source_root / filename
            if source.is_file():
                shutil.copy2(source, save_directory / filename)

    def export_native_pretrained(
        self,
        save_directory: str | Path,
    ) -> Path:
        destination = Path(save_directory).expanduser()
        self._save_pretrained(destination)
        return destination


__all__ = ["MedASRForSpeechRecognition"]
