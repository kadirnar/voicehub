"""Native WeNet GigaSpeech U2++ inference and fine-tuning provider."""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from numbers import Integral
from pathlib import Path
from typing import Any

from voicehub.hub import read_json_file, write_json_file
from voicehub.models.asr_native.configuration import WeNetASRConfig
from voicehub.models.audio import PreTrainedASRModel
from voicehub.models.native_utils import resolve_cpu_cuda_device
from voicehub.outputs import ASROutput, ASRSegment, ASRWord


def _batch_values(
    value: Any,
    *,
    batch_size: int,
    name: str,
) -> tuple[Any, ...]:
    import torch

    if value is None or isinstance(value, (str, bytes)):
        return (value, ) * batch_size
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return (value.item(), ) * batch_size
        if value.ndim != 1:
            raise ValueError(f"`{name}` must be scalar or one-dimensional.")
        values = tuple(value.detach().cpu().tolist())
    elif isinstance(value, Sequence):
        values = tuple(value)
    else:
        return (value, ) * batch_size
    if len(values) != batch_size:
        raise ValueError(f"`{name}` contains {len(values)} values for batch size "
                         f"{batch_size}.")
    return values


class WeNetASRForSpeechRecognition(PreTrainedASRModel):
    """Run the exact trainable U2++ graph without importing WeNet."""

    config_class = WeNetASRConfig
    default_model_name_or_path = "wenet/gigaspeech-u2pp-conformer"
    architecture_family = "speech-seq2seq"
    training_support = "native"
    supports_generic_finetuning = True

    def __init__(
        self,
        config: WeNetASRConfig | str | Path | None = None,
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
        self.tokenizer: Any | None = None
        self.checkpoint_adapter: str | None = None
        config = self._coerce_config(
            config,
            model_path=model_path,
            **kwargs,
        )
        super().__init__(config, device=device, lazy_load=lazy_load)

    @staticmethod
    def _resolve_device(device: str) -> str:
        return resolve_cpu_cuda_device(device, provider="native WeNet ASR")

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
            raise ValueError("Native WeNet does not support float16 execution on CPU.")
        return dtype

    def _load_pretrained_model(self) -> None:
        from voicehub.checkpointing import SafeTensorReader
        from voicehub.models.asr_wenet.artifacts import resolve_wenet_u2pp_artifacts
        from voicehub.models.asr_wenet.native.checkpoint import WeNetU2PPSafeTensorsCheckpointAdapter
        from voicehub.models.asr_wenet.native.configuration import WeNetU2PPConfig
        from voicehub.models.asr_wenet.native.modeling import WeNetU2PPForASR
        from voicehub.models.asr_wenet.native.tokenization import WeNetGigaSpeechTokenizer

        source = self.config.name_or_path or self.default_model_name_or_path
        artifacts = resolve_wenet_u2pp_artifacts(
            source,
            checkpoint_filename=self.config.checkpoint_filename,
            tokenizer_filename=self.config.tokenizer_filename,
            units_filename=self.config.units_filename,
            revision=self.config.revision,
            cache_dir=self.config.cache_dir,
            token=self._hub_token,
            local_files_only=self.config.local_files_only,
            trust_pickle_checkpoint=self._trust_pickle_checkpoint,
        )
        values = read_json_file(artifacts.config)
        if values.get("model_type") not in {
                "asr_wenet",
                "wenet-gigaspeech-u2pp",
        }:
            raise ValueError("Native WeNet requires a VoiceHub U2++ artifact.")
        native_config = WeNetU2PPConfig.from_dict(values)
        if native_config.sampling_rate != self.config.sample_rate:
            raise ValueError("WeNet provider/model sample rates do not match.")
        model = WeNetU2PPForASR(native_config)
        adapter = WeNetU2PPSafeTensorsCheckpointAdapter()
        with SafeTensorReader(artifacts.checkpoint) as reader:
            adapter.load_streaming(model, reader, values, strict=True)
        model.to(device=self.device, dtype=self._model_dtype())
        self.artifacts = artifacts
        self.native_config = native_config
        self.tokenizer = WeNetGigaSpeechTokenizer.from_files(
            artifacts.tokenizer,
            artifacts.units,
        )
        self.checkpoint_adapter = adapter.qualified_id
        self.model = model

    @staticmethod
    def _validate_request(
        *,
        language: str | None,
        task: str,
        return_timestamps: bool | str,
        chunk_length_s: float | None,
        stride_length_s: Any,
        batch_size: int | None,
        max_new_tokens: int | None,
        hotwords: Any,
    ) -> tuple[str, bool]:
        if task != "transcribe":
            raise ValueError("Native WeNet supports transcription only.")
        if language is None:
            resolved_language = "en"
        elif not isinstance(language, str):
            raise TypeError("`language` must be a string or None.")
        elif language.strip().lower() in {"en", "eng", "english"}:
            resolved_language = "en"
        else:
            raise ValueError("The audited GigaSpeech U2++ checkpoint is English-only.")
        if return_timestamps not in {False, True, "word"}:
            raise ValueError("`return_timestamps` must be false, true, or 'word'.")
        active = [
            name for name, value in {
                "chunk_length_s": chunk_length_s,
                "stride_length_s": stride_length_s,
                "max_new_tokens": max_new_tokens,
                "hotwords": hotwords,
            }.items() if value is not None
        ]
        if active:
            raise ValueError("Native WeNet does not support inference option(s): " + ", ".join(active) + ".")
        if batch_size is not None and (isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or
                                       batch_size <= 0):
            raise ValueError("`batch_size` must be a positive integer or None.")
        return resolved_language, bool(return_timestamps)

    def _decode(self, outputs: Any, *, beam_size: int) -> Any:
        from voicehub.models.asr_wenet.native.decoding import (
            attention_rescore,
            ctc_greedy_decode,
            ctc_prefix_beam_search,
        )

        strategy = self.config.decoding_strategy
        if strategy == "ctc_greedy_search":
            return ctc_greedy_decode(
                outputs.log_probabilities,
                outputs.encoded_lengths,
            )[0]
        nbest = ctc_prefix_beam_search(
            outputs.log_probabilities,
            outputs.encoded_lengths,
            beam_size=beam_size,
        )[0]
        if strategy == "ctc_prefix_beam_search":
            return nbest[0]
        return attention_rescore(
            self.model,
            nbest,
            outputs.encoder_output,
            ctc_weight=self.config.ctc_weight,
            reverse_weight=self.config.reverse_weight,
        )

    def _segments(
        self,
        hypothesis: Any,
        *,
        text: str,
        duration: float,
        language: str,
    ) -> tuple[ASRSegment, ...]:
        if not hypothesis.token_ids:
            return ()
        frame_seconds = (self.native_config.subsampling_rate * self.native_config.frame_shift_ms / 1000.0)
        words: list[ASRWord] = []
        pieces = [self.tokenizer.units[token] for token in hypothesis.token_ids]
        starts = list(hypothesis.token_frames)
        confidences = list(hypothesis.token_confidences)
        current_pieces: list[str] = []
        current_frames: list[int] = []
        current_confidences: list[float] = []

        def flush(end_frame: int | None = None) -> None:
            if not current_pieces:
                return
            word = "".join(current_pieces).replace("\u2581", "").strip()
            if not word:
                current_pieces.clear()
                current_frames.clear()
                current_confidences.clear()
                return
            start = min(duration, current_frames[0] * frame_seconds)
            if end_frame is None:
                end = duration
            else:
                end = min(duration, max(end_frame * frame_seconds, start))
            confidence = (
                sum(current_confidences) / len(current_confidences) if current_confidences else None)
            words.append(ASRWord(
                text=word,
                start=start,
                end=end,
                confidence=confidence,
            ))
            current_pieces.clear()
            current_frames.clear()
            current_confidences.clear()

        for index, piece in enumerate(pieces):
            frame = starts[index] if index < len(starts) else 0
            if piece.startswith("\u2581") and current_pieces:
                flush(frame)
            current_pieces.append(piece)
            current_frames.append(frame)
            if index < len(confidences):
                current_confidences.append(confidences[index])
        flush()
        return (
            ASRSegment(
                text=text,
                start=words[0].start if words else 0.0,
                end=words[-1].end if words else duration,
                confidence=hypothesis.confidence,
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

        resolved_language, timestamps = self._validate_request(
            language=language,
            task=task,
            return_timestamps=return_timestamps,
            chunk_length_s=chunk_length_s,
            stride_length_s=stride_length_s,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            hotwords=hotwords,
        )
        if self.model is None or self.native_config is None or self.tokenizer is None:
            raise RuntimeError("Native WeNet runtime is not loaded.")
        beam_size = self.config.beam_size if num_beams is None else num_beams
        if (isinstance(beam_size, bool) or not isinstance(beam_size, Integral) or beam_size <= 0):
            raise ValueError("`num_beams` must be a positive integer or None.")
        materialized = load_native_audio(
            audio,
            sampling_rate=sampling_rate,
            target_sampling_rate=self.native_config.sampling_rate,
        )
        waveform = materialized.waveform
        minimum = int(self.native_config.sampling_rate * self.native_config.frame_length_ms / 1000.0)
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
                decoding_chunk_size=-1,
            )
            hypothesis = self._decode(outputs, beam_size=int(beam_size))
        text = self.tokenizer.decode_ids(hypothesis.token_ids)
        segments = (
            self._segments(
                hypothesis,
                text=text,
                duration=materialized.duration,
                language=resolved_language,
            ) if timestamps else ())
        return ASROutput(
            text=text,
            segments=segments,
            language=resolved_language,
            duration=materialized.duration,
            metadata={
                "architecture":
                "wenet-gigaspeech-u2pp",
                "architecture_family":
                "speech-seq2seq",
                "backend":
                "voicehub-native",
                "checkpoint_adapter":
                self.checkpoint_adapter,
                "checkpoint_revision": (None if self.artifacts is None else self.artifacts.revision),
                "confidence":
                hypothesis.confidence,
                "converted_from_pickle":
                (False if self.artifacts is None else self.artifacts.converted_from_pickle),
                "decoding_strategy":
                self.config.decoding_strategy,
                "logit_frames":
                int(outputs.encoded_lengths[0].item()),
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
            raise ValueError("WeNet training audio must be rank one or two.")
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
        import torch

        from voicehub.processing.waveform import load_native_audio

        del phase
        waveform_ready = {
            "input_signal",
            "input_signal_length",
            "labels",
            "label_lengths",
        }
        feature_ready = {
            "features",
            "feature_lengths",
            "labels",
            "label_lengths",
        }
        if waveform_ready <= set(inputs) or feature_ready <= set(inputs):
            return dict(inputs)
        if self.model is None:
            self.load_for_training()
        if self.native_config is None or self.tokenizer is None:
            raise RuntimeError("Native WeNet training runtime is not loaded.")
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
            raise ValueError("WeNet training records require a transcript.")
        if not texts or any(not isinstance(value, str) or not value.strip() for value in texts):
            raise ValueError("WeNet transcripts must be non-empty strings.")
        if inputs.get("audio") is None:
            raise ValueError("WeNet training records require `audio`.")
        audio_values, was_batched = self._audio_batch(
            inputs["audio"],
            text_is_batch=text_is_batch,
        )
        if len(audio_values) != len(texts):
            raise ValueError("WeNet requires one transcript per waveform.")
        rate_values = _batch_values(
            inputs.get("sampling_rate", inputs.get("sample_rate")),
            batch_size=len(audio_values),
            name="sampling_rate",
        )
        raw_lengths = _batch_values(
            inputs.get("audio_lengths"),
            batch_size=len(audio_values),
            name="audio_lengths",
        )
        from voicehub.processing.waveform import NativeAudio

        waveforms = []
        for value, rate, raw_length in zip(
                audio_values,
                rate_values,
                raw_lengths,
        ):
            materialized = load_native_audio(
                value,
                sampling_rate=rate,
            )
            waveform = materialized.waveform
            if raw_length is not None:
                if (isinstance(raw_length, bool) or not isinstance(raw_length, Integral) or raw_length <= 0):
                    raise ValueError("`audio_lengths` must contain positive integers.")
                if int(raw_length) > waveform.numel():
                    raise ValueError("`audio_lengths` exceeds a waveform's sample count.")
                waveform = waveform[:int(raw_length)]
            resampled = load_native_audio(
                NativeAudio(
                    waveform=waveform,
                    sampling_rate=materialized.sampling_rate,
                    path=materialized.path,
                ),
                target_sampling_rate=self.native_config.sampling_rate,
            )
            waveforms.append(resampled.waveform)
        waveforms = tuple(waveforms)
        lengths = torch.tensor(
            [waveform.numel() for waveform in waveforms],
            dtype=torch.long,
        )
        maximum = int(lengths.max().item())
        input_signal = torch.stack(
            [torch.nn.functional.pad(
                waveform,
                (0, maximum - waveform.numel()),
            ) for waveform in waveforms])
        encoded = tuple(self.tokenizer.encode_as_ids(value) for value in texts)
        if any(not value for value in encoded):
            raise ValueError("WeNet transcripts must produce at least one token.")
        label_lengths = torch.tensor(
            [len(value) for value in encoded],
            dtype=torch.long,
        )
        labels = torch.full(
            (len(encoded), int(label_lengths.max().item())),
            self.native_config.ignore_token_id,
            dtype=torch.long,
        )
        for index, token_ids in enumerate(encoded):
            labels[index, :len(token_ids)] = torch.tensor(token_ids)
        prepared = {
            "input_signal": input_signal,
            "input_signal_length": lengths,
            "labels": labels,
            "label_lengths": label_lengths,
        }
        if was_batched:
            return prepared
        return {
            name:
            value[0] if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == 1 else value
            for name, value in prepared.items()
        }

    def _validate_training_runtime(self) -> None:
        return None

    def _save_pretrained(self, save_directory: Path) -> None:
        from voicehub.checkpointing import save_safetensors
        from voicehub.models.asr_wenet.native.checkpoint import (
            NATIVE_WENET_FILENAME,
            NATIVE_WENET_FORMAT,
            WENET_TOKENIZER_FILENAME,
            WENET_UNITS_FILENAME,
        )

        if self.model is None or self.native_config is None or self.artifacts is None:
            self.load()
        save_directory.mkdir(parents=True, exist_ok=True)
        save_safetensors(
            self.model.state_dict(),
            save_directory / NATIVE_WENET_FILENAME,
            metadata={
                "architecture": "wenet-gigaspeech-u2pp",
                "format": NATIVE_WENET_FORMAT,
            },
        )
        shutil.copyfile(
            self.artifacts.tokenizer,
            save_directory / WENET_TOKENIZER_FILENAME,
        )
        shutil.copyfile(
            self.artifacts.units,
            save_directory / WENET_UNITS_FILENAME,
        )
        values = self.native_config.to_dict()
        values.update({
            'architectures': ["WeNetU2PPForASR"],
            "checkpoint_format": NATIVE_WENET_FORMAT,
            "model_type": "asr_wenet",
            "voicehub_provider": "asr_wenet",
        })
        write_json_file(save_directory / "config.json", values)

    def export_native_pretrained(self, save_directory: str | Path) -> Path:
        destination = Path(save_directory).expanduser()
        self._save_pretrained(destination)
        return destination


__all__ = ["WeNetASRForSpeechRecognition"]
