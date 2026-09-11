"""Native QuartzNet CTC inference and fine-tuning provider."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral
from pathlib import Path
from typing import Any

from voicehub.hub import read_json_file, write_json_file
from voicehub.models.asr_native.configuration import NeMoASRConfig
from voicehub.models.audio import PreTrainedASRModel
from voicehub.models.native_utils import resolve_cpu_cuda_device
from voicehub.outputs import ASROutput, ASRSegment, ASRWord

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
        raise ValueError(f"`{name}` contains {len(values)} values for a batch of {batch_size}.")
    return values


class NeMoASRForSpeechRecognition(PreTrainedASRModel):
    """Run and fine-tune audited QuartzNet weights without importing NeMo."""

    config_class = NeMoASRConfig
    default_model_name_or_path = "nvidia/nemo/stt_en_quartznet15x5"
    architecture_family = "ctc"
    training_support = "native"
    supports_generic_finetuning = True

    def __init__(
        self,
        config: NeMoASRConfig | str | Path | None = None,
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
        # Kept as a source-compatible runtime argument. The known NGC archive
        # is hash-pinned and loaded through torch's restricted weights reader;
        # arbitrary pickle checkpoints are rejected even when this is true.
        self._trust_pickle_checkpoint = trust_pickle_checkpoint
        self.artifacts: Any | None = None
        self.native_config: Any | None = None
        self.ctc_tokenizer: Any | None = None
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

    @staticmethod
    def _resolve_device(device: str) -> str:
        return resolve_cpu_cuda_device(device, provider="native NeMo ASR")

    def _model_dtype(self) -> Any:
        import torch

        configured = self.config.torch_dtype
        if configured == "auto":
            return (torch.float16 if torch.device(self.device).type == "cuda" else torch.float32)
        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[configured]
        if torch.device(self.device).type == "cpu" and dtype == torch.float16:
            raise ValueError(
                "Native NeMo ASR does not support float16 execution on CPU; "
                "use float32 or bfloat16.")
        return dtype

    @staticmethod
    def _validate_architecture(values: Mapping[str, Any]) -> None:
        model_type = str(values.get("model_type", "")).strip().lower()
        if model_type not in {
                "asr_nemo",
                "nemo-quartznet-ctc",
        }:
            raise ValueError(
                "Native NeMo ASR requires a VoiceHub QuartzNet CTC artifact; "
                f"received model type {model_type or '<missing>'!r}.")
        architectures = values.get('architectures', ())
        if isinstance(architectures, str):
            architectures = (architectures, )
        if architectures and not any(str(name) in {
                "NeMoASRForSpeechRecognition",
                "NeMoQuartzNetForCTC",
        } for name in architectures):
            names = ", ".join(str(name) for name in architectures)
            raise ValueError(
                "Native NeMo ASR supports the QuartzNet character-CTC graph "
                f"only; received architectures: {names}.")

    def _load_pretrained_model(self) -> None:
        import torch

        from voicehub.checkpointing import SafeTensorReader
        from voicehub.models.asr_nemo.artifacts import resolve_nemo_ctc_artifacts
        from voicehub.models.asr_nemo.native.checkpoint import NeMoCTCSafeTensorsCheckpointAdapter
        from voicehub.models.asr_nemo.native.configuration import NeMoQuartzNetCTCConfig
        from voicehub.models.asr_nemo.native.modeling import NeMoQuartzNetForCTC
        from voicehub.models.asr_nemo.native.tokenization import NeMoCharacterTokenizer

        source = self.config.name_or_path or self.default_model_name_or_path
        artifacts = resolve_nemo_ctc_artifacts(
            source,
            checkpoint_filename=self.config.checkpoint_filename,
            revision=self.config.revision,
            cache_dir=self.config.cache_dir,
            token=self._hub_token,
            local_files_only=self.config.local_files_only,
        )
        values = read_json_file(artifacts.config)
        self._validate_architecture(values)
        native_config = NeMoQuartzNetCTCConfig.from_dict(values)
        if native_config.sampling_rate != self.config.sample_rate:
            raise ValueError(
                "NeMo provider/model sample-rate mismatch: provider uses "
                f"{self.config.sample_rate}, model expects "
                f"{native_config.sampling_rate}.")
        model = NeMoQuartzNetForCTC(native_config)
        adapter = NeMoCTCSafeTensorsCheckpointAdapter()
        with SafeTensorReader(artifacts.checkpoint) as reader:
            adapter.load_streaming(
                model,
                reader,
                values,
                strict=True,
            )
        model.to(
            device=self.device,
            dtype=self._model_dtype(),
        )
        self.artifacts = artifacts
        self.native_config = native_config
        self.ctc_tokenizer = NeMoCharacterTokenizer(native_config.vocabulary, )
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
    ) -> tuple[str, bool]:
        if task != "transcribe":
            raise ValueError("Native NeMo QuartzNet supports `task='transcribe'` only.")
        if language is None:
            resolved_language = "en"
        elif isinstance(language, str) and language.strip().lower() in {
                "en",
                "eng",
                "english",
        }:
            resolved_language = "en"
        else:
            raise ValueError("The audited QuartzNet15x5 checkpoint is English-only.")
        if return_timestamps not in {False, True, "word"}:
            raise ValueError("`return_timestamps` must be false, true, or 'word' for "
                             "native NeMo CTC.")
        unsupported = {
            "chunk_length_s": chunk_length_s,
            "stride_length_s": stride_length_s,
            "num_beams": num_beams,
            "max_new_tokens": max_new_tokens,
            "hotwords": hotwords,
        }
        active = [name for name, value in unsupported.items() if value is not None]
        if active:
            raise ValueError(
                "Native NeMo QuartzNet does not support inference option(s): "
                f"{', '.join(active)}.")
        if batch_size is not None:
            if (isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size <= 0):
                raise ValueError("`batch_size` must be a positive integer or None.")
        return resolved_language, bool(return_timestamps)

    def _timestamped_segment(
        self,
        *,
        decoded: Any,
        log_probabilities: Any,
        duration: float,
        language: str,
    ) -> tuple[ASRSegment, ...]:
        import torch

        if self.native_config is None:
            raise RuntimeError("Native NeMo configuration is not loaded.")
        frame_seconds = (self.native_config.output_frame_hop_samples / self.native_config.sampling_rate)
        probabilities = log_probabilities.exp().amax(dim=-1)
        words = []
        for span in decoded.words:
            start_frame = min(span.start_offset, probabilities.shape[0])
            end_frame = min(span.end_offset, probabilities.shape[0])
            confidence = None
            if end_frame > start_frame:
                confidence = float(probabilities[start_frame:end_frame].mean().item())
            words.append(
                ASRWord(
                    text=span.word,
                    start=min(duration, span.start_offset * frame_seconds),
                    end=min(duration, span.end_offset * frame_seconds),
                    confidence=confidence,
                ))
        if not words:
            return ()
        scores = [word.confidence for word in words if word.confidence is not None]
        confidence = (float(torch.tensor(scores).mean().item()) if scores else None)
        return (
            ASRSegment(
                text=decoded.text,
                start=words[0].start,
                end=words[-1].end,
                confidence=confidence,
                language=language,
                words=tuple(words),
            ), )

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

        resolved_language, word_timestamps = self._validate_inference_request(
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
        if (self.model is None or self.native_config is None or self.ctc_tokenizer is None):
            raise RuntimeError("Native NeMo ASR runtime is not loaded.")
        materialized = load_native_audio(
            audio,
            sampling_rate=sampling_rate,
            target_sampling_rate=self.native_config.sampling_rate,
        )
        waveform = materialized.waveform
        minimum = self.native_config.minimum_input_samples
        if waveform.numel() < minimum:
            waveform = torch.nn.functional.pad(
                waveform,
                (0, minimum - waveform.numel()),
            )
        parameter = next(self.model.parameters())
        input_signal = waveform.unsqueeze(0).to(
            device=parameter.device,
            dtype=parameter.dtype,
        )
        input_length = torch.tensor(
            [waveform.numel()],
            dtype=torch.long,
            device=parameter.device,
        )
        with torch.inference_mode():
            outputs = self.model(
                input_signal=input_signal,
                input_signal_length=input_length,
            )
        valid_frames = int(outputs.encoded_lengths[0].item())
        log_probabilities = outputs.log_probabilities[0, :valid_frames]
        decoded = self.ctc_tokenizer.decode_ctc(outputs.predictions[0, :valid_frames].tolist(), )
        segments = (
            self._timestamped_segment(
                decoded=decoded,
                log_probabilities=log_probabilities,
                duration=materialized.duration,
                language=resolved_language,
            ) if word_timestamps else ())
        return ASROutput(
            text=decoded.text,
            segments=segments,
            language=resolved_language,
            duration=materialized.duration,
            metadata={
                "architecture": "nemo-quartznet-ctc",
                "architecture_family": "ctc",
                "backend": "voicehub-native",
                "checkpoint_adapter": self.checkpoint_adapter,
                "checkpoint_revision": (None if self.artifacts is None else self.artifacts.revision),
                "converted_from_nemo":
                (False if self.artifacts is None else self.artifacts.converted_from_nemo),
                "logit_frames": valid_frames,
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
            raise ValueError("NeMo training audio must be rank one or rank two.")
        if text_is_batch:
            if isinstance(audio, (str, bytes)) or not isinstance(audio, Sequence):
                raise ValueError("Batched transcripts require a sequence of waveforms.")
            return tuple(audio), True
        return (audio, ), False

    def prepare_training_inputs(
        self,
        inputs: dict[str, Any],
        *,
        phase: str,
    ) -> dict[str, Any]:
        """Create raw waveform tensors and strict character CTC labels."""
        import torch

        from voicehub.processing.waveform import load_native_audio

        del phase
        raw_ready = {
            "input_signal",
            "input_signal_length",
            "labels",
            "label_lengths",
        }
        feature_ready = {
            "processed_signal",
            "processed_signal_length",
            "labels",
            "label_lengths",
        }
        if raw_ready <= set(inputs) or feature_ready <= set(inputs):
            return dict(inputs)
        if self.model is None:
            self.load_for_training()
        if self.native_config is None or self.ctc_tokenizer is None:
            raise RuntimeError("Native NeMo ASR training runtime is not loaded.")

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
            raise ValueError("NeMo training records require `text`, `transcription`, or "
                             "`transcript`.")
        if not texts or any(not isinstance(value, str) or not value.strip() for value in texts):
            raise ValueError("NeMo training transcripts must be non-empty strings.")
        if audio is None:
            raise ValueError("NeMo training records require `audio`.")
        audio_values, was_batched = self._audio_batch(
            audio,
            text_is_batch=text_is_batch,
        )
        if len(audio_values) != len(texts):
            raise ValueError("NeMo training requires one transcript per waveform.")

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
                target_sampling_rate=self.native_config.sampling_rate,
                num_samples=(None if length is None else int(length)),
            ).waveform for value, rate, length in zip(
                audio_values,
                rates,
                lengths,
            ))
        minimum = self.native_config.minimum_input_samples
        waveforms = tuple(
            torch.nn.functional.pad(
                waveform,
                (0, minimum - waveform.numel()),
            ) if waveform.numel() < minimum else waveform for waveform in waveforms)
        waveform_lengths = torch.tensor(
            [waveform.numel() for waveform in waveforms],
            dtype=torch.long,
        )
        max_samples = int(waveform_lengths.max().item())
        input_signal = torch.stack([
            torch.nn.functional.pad(
                waveform,
                (0, max_samples - waveform.numel()),
            ) for waveform in waveforms
        ])

        encoded_labels = tuple(self.ctc_tokenizer.encode(value) for value in texts)
        label_lengths = torch.tensor(
            [len(value) for value in encoded_labels],
            dtype=torch.long,
        )
        labels = torch.full(
            (len(encoded_labels), int(label_lengths.max().item())),
            -1,
            dtype=torch.long,
        )
        for index, token_ids in enumerate(encoded_labels):
            labels[index, :len(token_ids)] = torch.tensor(
                token_ids,
                dtype=torch.long,
            )
        prepared = {
            "input_signal": input_signal,
            "input_signal_length": waveform_lengths,
            "labels": labels,
            "label_lengths": label_lengths,
        }
        for name, value in inputs.items():
            if name not in _RAW_TRAINING_FIELDS and name not in prepared:
                prepared[name] = value
        if was_batched:
            return prepared
        return {
            name: (
                value[0]
                if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == 1 else value)
            for name, value in prepared.items()
        }

    def _validate_training_runtime(self) -> None:
        return None

    def _save_pretrained(self, save_directory: Path) -> None:
        from voicehub.checkpointing import save_safetensors
        from voicehub.models.asr_nemo.native.checkpoint import NATIVE_NEMO_CTC_FILENAME, NATIVE_NEMO_CTC_FORMAT

        if self.model is None or self.native_config is None:
            self.load()
        save_directory.mkdir(parents=True, exist_ok=True)
        save_safetensors(
            self.model.state_dict(),
            save_directory / NATIVE_NEMO_CTC_FILENAME,
            metadata={
                "architecture": "nemo-quartznet-ctc",
                "format": NATIVE_NEMO_CTC_FORMAT,
                "sample_rate": str(self.sample_rate),
            },
        )
        values = self.native_config.to_dict()
        values.update({
            'architectures': ["NeMoQuartzNetForCTC"],
            "checkpoint_format": NATIVE_NEMO_CTC_FORMAT,
            "model_type": self.config.model_type,
            "name_or_path": str(save_directory),
            "voicehub_provider": self.config.model_type,
        })
        write_json_file(save_directory / "config.json", values)

    def export_native_pretrained(self, save_directory: str | Path) -> Path:
        destination = Path(save_directory).expanduser()
        self._save_pretrained(destination)
        return destination


__all__ = ["NeMoASRForSpeechRecognition"]
