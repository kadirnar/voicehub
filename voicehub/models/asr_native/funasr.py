"""VoiceHub-native SenseVoiceSmall inference and fine-tuning provider."""

from __future__ import annotations

import shutil
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from pathlib import Path
from typing import Any

from voicehub.audio_modeling_utils import PreTrainedASRModel
from voicehub.hub import read_json_file, write_json_file
from voicehub.modeling_outputs import ASROutput, ASRSegment
from voicehub.models.asr_native.configuration import FunASRConfig
from voicehub.models.native_utils import resolve_cpu_cuda_device

_LANGUAGE_ALIASES = {
    "auto": "auto",
    "cantonese": "yue",
    "chinese": "zh",
    "en": "en",
    "eng": "en",
    "english": "en",
    "ja": "ja",
    "japanese": "ja",
    "ko": "ko",
    "korean": "ko",
    "mandarin": "zh",
    "nospeech": "nospeech",
    "yue": "yue",
    "zh": "zh",
}


def _batch_strings(
    value: Any,
    *,
    batch_size: int,
    name: str,
    default: str | bool | None = None,
) -> tuple[Any, ...]:
    if value is None:
        if default is None:
            raise ValueError(f"SenseVoice raw training requires `{name}`.")
        return (default, ) * batch_size
    if isinstance(value, (str, bool)):
        return (value, ) * batch_size
    to_list = getattr(value, "tolist", None)
    if callable(to_list):
        value = to_list()
        if not isinstance(value, list):
            value = [value]
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        raise TypeError(f"`{name}` must be a scalar or sequence.")
    values = tuple(value)
    if len(values) != batch_size:
        raise ValueError(f"`{name}` must contain {batch_size} values, found {len(values)}.")
    return values


def _numeric_waveform(value: Any) -> bool:
    return (
        isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) and bool(value) and
        all(isinstance(item, Real) and not isinstance(item, bool) for item in value))


def _batch_values(
    value: Any,
    *,
    batch_size: int,
    name: str,
) -> tuple[Any, ...]:
    import torch

    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return (value.item(), ) * batch_size
        if value.ndim != 1:
            raise ValueError(f"`{name}` must be scalar or one-dimensional.")
        values = tuple(value.tolist())
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values = tuple(value)
    else:
        return (value, ) * batch_size
    if len(values) != batch_size:
        raise ValueError(f"`{name}` must contain {batch_size} values, found {len(values)}.")
    return values


def _training_batch_size_hint(values: Mapping[str, Any]) -> int | None:
    import torch

    transcript = values.get("text", values.get("transcript"))
    if isinstance(transcript, str):
        return 1
    if isinstance(transcript, Sequence) and not isinstance(transcript, (str, bytes, bytearray)):
        return len(transcript)
    labels = values.get("labels")
    if labels is None:
        return None
    label_tensor = labels if isinstance(labels, torch.Tensor) else torch.as_tensor(labels)
    if label_tensor.ndim == 1:
        return 1
    if label_tensor.ndim == 2:
        return int(label_tensor.shape[0])
    return None


def _audio_batch(
    value: Any,
    *,
    expected_batch_size: int | None = None,
) -> tuple[Any, ...]:
    import torch

    if isinstance(value, torch.Tensor):
        if value.ndim == 1:
            return (value, )
        if value.ndim == 2:
            return tuple(value[index] for index in range(value.shape[0]))
        raise ValueError("SenseVoice raw audio must have shape [samples] or [batch, samples].")
    if isinstance(value, Mapping):
        payload_name = next(
            (name for name in ("array", "waveform", "audio", "input_values") if name in value),
            None,
        )
        if payload_name is None:
            return (value, )
        payloads = _audio_batch(value[payload_name])
        if expected_batch_size not in (None, 1) and len(payloads) == expected_batch_size:
            rates = _batch_values(
                value.get("sampling_rate", value.get("sample_rate")),
                batch_size=expected_batch_size,
                name="sampling_rate",
            )
            rows = []
            for payload, rate in zip(payloads, rates):
                row = dict(value)
                row[payload_name] = payload
                row.pop("sample_rate", None)
                if rate is None:
                    row.pop("sampling_rate", None)
                else:
                    row["sampling_rate"] = rate
                rows.append(row)
            return tuple(rows)
        return (value, )
    if _numeric_waveform(value):
        return (value, )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if not value:
            raise ValueError("SenseVoice raw audio batches cannot be empty.")
        return tuple(value)
    if isinstance(value, (str, Path)):
        return (value, )
    try:
        tensor = torch.as_tensor(value)
    except (TypeError, ValueError, RuntimeError):
        return (value, )
    if tensor.ndim == 1:
        return (value, )
    if tensor.ndim == 2:
        return tuple(tensor[index] for index in range(tensor.shape[0]))
    return (value, )


def _source_declares_sampling_rate(value: Any) -> bool:
    from voicehub.processing.waveform import NativeAudio

    if isinstance(value, (NativeAudio, str, Path)):
        return True
    if isinstance(value, Mapping):
        return value.get("sampling_rate", value.get("sample_rate")) is not None
    return False


class FunASRForSpeechRecognition(PreTrainedASRModel):
    """Run the audited SenseVoiceSmall SANM-CTC graph without FunASR.

    The provider deliberately recognizes only SenseVoiceSmall.
    Paraformer, Fun-ASR-Nano, and other models distributed through the
    FunASR ecosystem use different graphs and fail with an architecture-
    specific error instead of being guessed from a repository name.
    """

    config_class = FunASRConfig
    default_model_name_or_path = "FunAudioLLM/SenseVoiceSmall"
    architecture_family = "ctc"
    native_checkpoint_format = "voicehub-sensevoice-small-v1"
    training_support = "native"
    supports_generic_finetuning = True
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: FunASRConfig | str | Path | None = None,
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
        self.frontend: Any | None = None
        self.checkpoint_adapter: str | None = None
        resolved = self._coerce_config(
            config,
            model_path=model_path,
            **kwargs,
        )
        super().__init__(resolved, device=device, lazy_load=lazy_load)

    @staticmethod
    def _resolve_device(device: str) -> str:
        return resolve_cpu_cuda_device(
            device,
            provider="native SenseVoice ASR",
        )

    def _model_dtype(self):
        import torch

        configured = str(getattr(self.config, "torch_dtype", "float32"))
        aliases = {
            "auto": (torch.float16 if torch.device(self.device).type == "cuda" else torch.float32),
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }
        try:
            dtype = aliases[configured.strip().lower()]
        except KeyError as error:
            raise ValueError(
                "Native SenseVoice `torch_dtype` must be auto, float32, "
                "float16, or bfloat16.") from error
        if torch.device(self.device).type == "cpu" and dtype == torch.float16:
            raise ValueError("Native SenseVoice does not support float16 execution on CPU.")
        return dtype

    def _validate_composed_options(self) -> None:
        composed = {
            "vad_model": self.config.vad_model,
            "punc_model": self.config.punc_model,
            "spk_model": self.config.spk_model,
        }
        active = [name for name, value in composed.items() if value is not None]
        if active:
            raise ValueError(
                "SenseVoiceSmall contains ASR/LID/SER/AED, but VAD, "
                "punctuation, and speaker models are separate architectures. "
                "Compose the corresponding VoiceHub providers explicitly; "
                f"unsupported embedded option(s): {', '.join(active)}.")
        if self.config.model_kwargs:
            raise ValueError(
                "Native SenseVoice does not delegate FunASR `model_kwargs`; "
                "unsupported option(s): " + ", ".join(sorted(self.config.model_kwargs)) + ".")
        allowed_generate = {"ban_emo_unk", "use_itn"}
        unsupported = set(self.config.generate_kwargs) - allowed_generate
        if unsupported:
            raise ValueError(
                "Native SenseVoice does not delegate arbitrary FunASR "
                "generation options; unsupported option(s): " + ", ".join(sorted(unsupported)) + ".")

    @staticmethod
    def _validate_architecture(values: Mapping[str, Any]) -> None:
        model_type = str(values.get("model_type", "")).strip().lower()
        if model_type not in {"asr_funasr", "sensevoice-small"}:
            raise ValueError(
                "Native FunASR compatibility supports only a VoiceHub "
                "SenseVoiceSmall artifact; found model type "
                f"{model_type or '<missing>'!r}.")
        architectures = values.get("architectures", ())
        if isinstance(architectures, str):
            architectures = (architectures, )
        supported = {
            "FunASRForSpeechRecognition",
            "SenseVoiceSmall",
            "SenseVoiceSmallForCTC",
        }
        if architectures and not any(str(name) in supported for name in architectures):
            raise ValueError(
                "Native FunASR compatibility does not support the declared "
                f"architecture(s): {', '.join(map(str, architectures))}.")

    def _load_pretrained_model(self) -> None:
        from voicehub.architectures.sensevoice.artifacts import resolve_sensevoice_artifacts
        from voicehub.architectures.sensevoice.checkpoint import (
            SenseVoiceSafeTensorsCheckpointAdapter,
            load_native_sensevoice_model,
        )
        from voicehub.architectures.sensevoice.configuration import SenseVoiceSmallConfig
        from voicehub.architectures.sensevoice.frontend import SenseVoiceFrontend
        from voicehub.architectures.sensevoice.tokenization import SenseVoiceTokenizer

        self._validate_composed_options()
        source = self.config.name_or_path or self.default_model_name_or_path
        artifacts = resolve_sensevoice_artifacts(
            source,
            revision=getattr(self.config, "revision", None),
            cache_dir=getattr(self.config, "cache_dir", None),
            token=self._hub_token,
            local_files_only=bool(getattr(self.config, "local_files_only", False)),
            trust_pickle_checkpoint=self._trust_pickle_checkpoint,
        )
        values = read_json_file(artifacts.config)
        self._validate_architecture(values)
        native_config = SenseVoiceSmallConfig.from_dict(values)
        if native_config.sampling_rate != self.config.sample_rate:
            raise ValueError(
                "SenseVoice provider/model sample rates do not match: "
                f"{self.config.sample_rate} and "
                f"{native_config.sampling_rate}.")
        model = load_native_sensevoice_model(
            artifacts.checkpoint,
            native_config,
            device=self.device,
            dtype=self._model_dtype(),
        )
        tokenizer = SenseVoiceTokenizer.from_model_file(
            artifacts.tokenizer,
            strict_release=native_config.variant == "sensevoice-small",
        )
        if tokenizer.vocabulary_size != native_config.vocabulary_size:
            raise ValueError(
                "SenseVoice tokenizer/model vocabulary mismatch: "
                f"{tokenizer.vocabulary_size} and "
                f"{native_config.vocabulary_size}.")
        frontend = SenseVoiceFrontend.from_cmvn_file(
            native_config,
            artifacts.cmvn,
        ).to(device=self.device)
        self.artifacts = artifacts
        self.native_config = native_config
        self.tokenizer = tokenizer
        self.frontend = frontend
        self.checkpoint_adapter = (SenseVoiceSafeTensorsCheckpointAdapter().qualified_id)
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
        num_beams: int | None,
        max_new_tokens: int | None,
        hotwords: Any,
    ) -> tuple[str, bool]:
        if task != "transcribe":
            raise ValueError("SenseVoiceSmall supports `task='transcribe'` only.")
        if language is None:
            resolved_language = "auto"
        elif not isinstance(language, str):
            raise TypeError("`language` must be a string or None.")
        else:
            try:
                resolved_language = _LANGUAGE_ALIASES[language.strip().lower()]
            except KeyError as error:
                raise ValueError(
                    "SenseVoiceSmall language must be auto, zh, en, yue, ja, "
                    "ko, or nospeech.") from error
        if return_timestamps not in {False, True, "word"}:
            raise ValueError("`return_timestamps` must be false, true, or 'word'.")
        unsupported = {
            "chunk_length_s": chunk_length_s,
            "stride_length_s": stride_length_s,
            "max_new_tokens": max_new_tokens,
            "hotwords": hotwords,
        }
        active = [name for name, value in unsupported.items() if value is not None]
        if active:
            raise ValueError(
                "Native SenseVoice does not support inference option(s): " + ", ".join(active) + ".")
        if batch_size not in (None, 1):
            raise ValueError(
                "One SenseVoice request requires `batch_size=1`; use dataset "
                "batching for fine-tuning.")
        if num_beams not in (None, 1):
            raise ValueError(
                "SenseVoiceSmall publishes greedy non-autoregressive CTC "
                "decoding; `num_beams` must be 1 or None.")
        return resolved_language, bool(return_timestamps)

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

        from voicehub.architectures.sensevoice.decoding import ctc_greedy_tokens, sensevoice_word_timestamps
        from voicehub.processing.waveform import load_native_audio

        resolved_language, timestamps = self._validate_request(
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
        if (self.model is None or self.native_config is None or self.tokenizer is None or
                self.frontend is None):
            raise RuntimeError("Native SenseVoice runtime is not loaded.")
        materialized = load_native_audio(
            audio,
            sampling_rate=sampling_rate,
            target_sampling_rate=self.native_config.sampling_rate,
        )
        maximum_duration = float(getattr(self.config, "max_duration_s", 30.0))
        if materialized.duration > maximum_duration:
            raise ValueError(
                "The audited direct SenseVoiceSmall path is limited to "
                f"{maximum_duration:g}s clips. Segment long audio with a "
                "VoiceHub VAD provider before recognition.")
        parameter = next(self.model.parameters())
        waveform = materialized.waveform.unsqueeze(0).to(
            device=parameter.device,
            dtype=torch.float32,
        )
        waveform_lengths = torch.tensor(
            [materialized.waveform.numel()],
            dtype=torch.long,
            device=parameter.device,
        )
        use_itn = bool(self.config.generate_kwargs.get("use_itn", False))
        ban_unknown = bool(self.config.generate_kwargs.get("ban_emo_unk", False))
        self.frontend.eval()
        with torch.inference_mode():
            features, feature_lengths = self.frontend(
                waveform,
                waveform_lengths,
                training=False,
            )
            features = features.to(
                device=parameter.device,
                dtype=parameter.dtype,
            )
            output = self.model.infer(
                features,
                feature_lengths,
                language=resolved_language,
                use_itn=use_itn,
                ban_unknown_emotion=ban_unknown,
            )
        encoded_length = int(output.encoded_lengths[0].item())
        log_probabilities = output.log_probabilities[0, :encoded_length]
        token_ids = ctc_greedy_tokens(
            log_probabilities,
            encoded_length,
            blank_token_id=self.native_config.blank_token_id,
        )
        raw_text = self.tokenizer.decode_raw(token_ids)
        text = self.tokenizer.decode_text(token_ids)
        semantics = self.tokenizer.semantics(token_ids)
        words = ()
        if timestamps and len(token_ids) > 4 and encoded_length > 4:
            content_ids = token_ids[4:]
            pieces = self.tokenizer.token_pieces(content_ids)
            filtered = tuple((token_id, piece) for token_id, piece in zip(content_ids, pieces)
                             if not (piece.startswith("<|") and piece.endswith("|>")))
            if filtered:
                words = sensevoice_word_timestamps(
                    log_probabilities[4:],
                    tuple(item[0] for item in filtered),
                    tuple(item[1] for item in filtered),
                    duration=materialized.duration,
                    blank_token_id=self.native_config.blank_token_id,
                    frame_seconds=(0.010 * self.native_config.lfr_stride),
                    center_offset_seconds=0.030,
                )
        detected_language = (
            semantics.language if semantics.language is not None else
            (None if resolved_language in {"auto", "nospeech"} else resolved_language))
        segments = ()
        if words:
            segments = (
                ASRSegment(
                    text=text,
                    start=words[0].start,
                    end=words[-1].end,
                    language=detected_language,
                    words=words,
                    metadata={
                        "emotion": semantics.emotion,
                        "events": semantics.events,
                    },
                ), )
        return ASROutput(
            text=text,
            segments=segments,
            language=detected_language,
            duration=materialized.duration,
            metadata={
                "architecture":
                "sensevoice-small",
                "architecture_family":
                self.architecture_family,
                "backend":
                "voicehub-native",
                "checkpoint_adapter":
                self.checkpoint_adapter,
                "checkpoint_format":
                self.native_checkpoint_format,
                "checkpoint_revision": (None if self.artifacts is None else self.artifacts.revision),
                "control_tokens":
                semantics.control_tokens,
                "converted_from_pickle":
                (False if self.artifacts is None else self.artifacts.converted_from_pickle),
                "emotion":
                semantics.emotion,
                "events":
                semantics.events,
                "raw_text":
                raw_text,
                "requested_language":
                resolved_language,
                "text_normalization":
                semantics.text_normalization,
                "token_ids":
                token_ids,
            },
        )

    def prepare_training_inputs(
        self,
        inputs: dict[str, Any],
        *,
        phase: str,
    ) -> dict[str, Any]:
        del phase
        import torch
        from torch.nn.utils.rnn import pad_sequence

        from voicehub.processing.waveform import NativeAudio, load_native_audio

        if self.model is None:
            self.load_for_training()
        if (self.model is None or self.frontend is None or self.tokenizer is None or
                self.native_config is None):
            raise RuntimeError("Native SenseVoice training runtime is not loaded.")
        values = dict(inputs)
        features = values.get("features")
        if features is not None:
            features = torch.as_tensor(features)
            if features.ndim == 2:
                features = features.unsqueeze(0)
            if features.ndim != 3:
                raise ValueError("SenseVoice `features` must have shape "
                                 "[batch, frames, bins].")
            feature_lengths = values.get("feature_lengths")
            if feature_lengths is None:
                feature_lengths = torch.full(
                    (features.shape[0], ),
                    features.shape[1],
                    dtype=torch.long,
                )
        else:
            audio = next(
                (
                    values[name] for name in (
                        "audio_values",
                        "input_signal",
                        "audio",
                        "audio_path",
                    ) if values.get(name) is not None),
                None,
            )
            if audio is None:
                raise ValueError("SenseVoice training requires `features` or raw "
                                 "`audio`/`audio_values`.")
            sources = _audio_batch(
                audio,
                expected_batch_size=_training_batch_size_hint(values),
            )
            raw_lengths = _batch_values(
                values.get(
                    "audio_lengths",
                    values.get("input_signal_length"),
                ),
                batch_size=len(sources),
                name="audio_lengths",
            )
            rates = _batch_values(
                values.get(
                    "sampling_rates",
                    values.get(
                        "sampling_rate",
                        values.get("sample_rate"),
                    ),
                ),
                batch_size=len(sources),
                name="sampling_rate",
            )
            waveforms = []
            for source, raw_length, rate in zip(sources, raw_lengths, rates):
                source_rate = rate
                if source_rate is None and not _source_declares_sampling_rate(source):
                    source_rate = self.native_config.sampling_rate
                materialized = load_native_audio(
                    source,
                    sampling_rate=source_rate,
                )
                waveform = materialized.waveform
                if raw_length is not None:
                    if (isinstance(raw_length, bool) or not isinstance(raw_length, Integral) or
                            raw_length <= 0):
                        raise ValueError("`audio_lengths` must contain positive integers.")
                    if int(raw_length) > waveform.numel():
                        raise ValueError("`audio_lengths` exceeds a waveform's sample count.")
                    waveform = waveform[:int(raw_length)]
                materialized = load_native_audio(
                    NativeAudio(
                        waveform=waveform,
                        sampling_rate=materialized.sampling_rate,
                        path=materialized.path,
                    ),
                    target_sampling_rate=self.native_config.sampling_rate,
                )
                waveforms.append(materialized.waveform)
            audio_lengths = torch.tensor(
                [waveform.numel() for waveform in waveforms],
                dtype=torch.long,
            )
            audio = pad_sequence(
                waveforms,
                batch_first=True,
                padding_value=0.0,
            )
            expected_batch_size = _training_batch_size_hint(values)
            if (expected_batch_size is not None and expected_batch_size != audio.shape[0]):
                raise ValueError(
                    "SenseVoice raw training requires one waveform per transcript "
                    f"or label row; found {audio.shape[0]} waveforms for "
                    f"{expected_batch_size} targets.")
            device = next(self.model.parameters()).device
            audio = audio.to(
                device=device,
                dtype=torch.float32,
            )
            audio_lengths = audio_lengths.to(device=device)
            self.frontend.train()
            features, feature_lengths = self.frontend(
                audio,
                audio_lengths,
                training=True,
            )
        batch_size = features.shape[0]
        labels = values.get("labels")
        label_lengths = values.get("label_lengths")
        if labels is None:
            texts = _batch_strings(
                values.get("text", values.get("transcript")),
                batch_size=batch_size,
                name="text",
            )
            languages = _batch_strings(
                values.get("language"),
                batch_size=batch_size,
                name="language",
            )
            emotions = _batch_strings(
                values.get("emotion"),
                batch_size=batch_size,
                name="emotion",
                default="neutral",
            )
            events = _batch_strings(
                values.get("event"),
                batch_size=batch_size,
                name="event",
                default="speech",
            )
            itn_values = _batch_strings(
                values.get("use_itn"),
                batch_size=batch_size,
                name="use_itn",
                default=False,
            )
            rows = [
                torch.tensor(
                    self.tokenizer.prepare_training_labels(
                        str(text),
                        language=str(language).strip().lower(),
                        emotion=str(emotion).strip().lower(),
                        event=str(event).strip().lower(),
                        use_itn=bool(use_itn),
                    ),
                    dtype=torch.long,
                ) for text, language, emotion, event, use_itn in zip(
                    texts,
                    languages,
                    emotions,
                    events,
                    itn_values,
                )
            ]
            label_lengths = torch.tensor(
                [row.numel() for row in rows],
                dtype=torch.long,
            )
            labels = pad_sequence(
                rows,
                batch_first=True,
                padding_value=self.native_config.ignore_token_id,
            )
        elif label_lengths is None:
            label_tensor = torch.as_tensor(labels)
            if label_tensor.ndim == 1:
                label_tensor = label_tensor.unsqueeze(0)
            label_lengths = label_tensor.ne(self.native_config.ignore_token_id).sum(dim=1)
        parameter = next(self.model.parameters())
        return {
            "features": torch.as_tensor(features).to(
                device=parameter.device,
                dtype=parameter.dtype,
            ),
            "feature_lengths": torch.as_tensor(
                feature_lengths,
                dtype=torch.long,
                device=parameter.device,
            ),
            "labels": torch.as_tensor(
                labels,
                dtype=torch.long,
                device=parameter.device,
            ),
            "label_lengths": torch.as_tensor(
                label_lengths,
                dtype=torch.long,
                device=parameter.device,
            ),
        }

    def _validate_training_runtime(self) -> None:
        return None

    def _save_pretrained(self, save_directory: Path) -> None:
        from voicehub.architectures.sensevoice.checkpoint import (
            NATIVE_SENSEVOICE_CMVN,
            NATIVE_SENSEVOICE_FILENAME,
            NATIVE_SENSEVOICE_FORMAT,
            NATIVE_SENSEVOICE_TOKENIZER,
        )
        from voicehub.architectures.sensevoice.metadata import (
            FUNASR_SOURCE_REVISION,
            SENSEVOICE_MODEL_LICENSE,
            SENSEVOICE_REVISION,
        )
        from voicehub.checkpointing import save_safetensors

        if (self.model is None or self.native_config is None or self.tokenizer is None or
                self.frontend is None):
            self.load()
        save_directory.mkdir(parents=True, exist_ok=True)
        save_safetensors(
            self.model.state_dict(),
            save_directory / NATIVE_SENSEVOICE_FILENAME,
            metadata={
                "architecture": "sensevoice-small",
                "format": NATIVE_SENSEVOICE_FORMAT,
                "model_license": SENSEVOICE_MODEL_LICENSE,
                "sample_rate": str(self.sample_rate),
            },
        )
        self.tokenizer.save_pretrained(
            save_directory,
            filename=NATIVE_SENSEVOICE_TOKENIZER,
        )
        if self.artifacts is not None:
            shutil.copy2(
                self.artifacts.cmvn,
                save_directory / NATIVE_SENSEVOICE_CMVN,
            )
        else:
            shift = " ".join(format(float(value), ".9g") for value in self.frontend.cmvn_shift.tolist())
            scale = " ".join(format(float(value), ".9g") for value in self.frontend.cmvn_scale.tolist())
            (save_directory / NATIVE_SENSEVOICE_CMVN).write_text(
                "<Nnet>\n"
                f"<AddShift> {self.native_config.input_dimension} "
                f"{self.native_config.input_dimension}\n"
                f"<LearnRateCoef> 0 [ {shift} ]\n"
                f"<Rescale> {self.native_config.input_dimension} "
                f"{self.native_config.input_dimension}\n"
                f"<LearnRateCoef> 0 [ {scale} ]\n"
                "</Nnet>\n",
                encoding="utf-8",
            )
        values = self.native_config.to_dict()
        values.update({
            "architectures": [
                "FunASRForSpeechRecognition",
                "SenseVoiceSmallForCTC",
            ],
            "checkpoint_format": NATIVE_SENSEVOICE_FORMAT,
            "model_type": self.config.model_type,
            "name_or_path": str(save_directory),
            "source_artifact_revision": SENSEVOICE_REVISION,
            "source_training_revision": FUNASR_SOURCE_REVISION,
            "voicehub_provider": self.config.model_type,
        })
        write_json_file(save_directory / "config.json", values)

    def export_native_pretrained(
        self,
        save_directory: str | Path,
    ) -> Path:
        destination = Path(save_directory).expanduser()
        self._save_pretrained(destination)
        return destination


__all__ = ["FunASRForSpeechRecognition"]
