"""VoiceHub-native fine-tuning support for Sesame CSM.

The public objective consumes Mimi codebook labels and optimizes both
the backbone first-codebook loss and the depth-decoder remaining-
codebook loss. Mimi stays frozen and acts only as an optional raw-
waveform preprocessor.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from voicehub.training.adapters import CausalLMTrainingAdapter

_PREPARED_INPUT_KEYS = frozenset({
    "tokens",
    "tokens_mask",
    "labels",
})
_LEGACY_PREPARED_INPUT_KEYS = frozenset({
    "input_ids",
    "inputs_embeds",
})


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    )


def _unwrap_audio(audio: Any, *, sample_rate: int) -> Any:
    if not isinstance(audio, Mapping):
        return audio
    source_rate = audio.get("sampling_rate")
    if source_rate is not None and int(source_rate) != sample_rate:
        raise ValueError(
            "CSM training audio must be resampled to "
            f"{sample_rate} Hz before collation; received {source_rate} Hz.")
    if "array" in audio:
        return audio["array"]
    if "path" in audio:
        return audio["path"]
    raise ValueError("CSM audio mappings require an `array` or `path` field.")


def _legacy_conversation(
    record: Mapping[str, Any],
    *,
    sample_rate: int,
) -> list[dict[str, Any]]:
    if "texts" in record or "speaker_ids" in record:
        texts = record.get("texts")
        speakers = record.get("speaker_ids")
        if (not _is_sequence(texts) or not _is_sequence(speakers) or not texts or
                len(texts) != len(speakers)):
            raise ValueError("CSM grouped texts and speaker_ids must have equal, "
                             "non-zero lengths.")
        if "audios" in record:
            audios = record["audios"]
            if not _is_sequence(audios) or len(audios) != len(texts):
                raise ValueError("CSM grouped records require one audio per text.")
            audios = [_unwrap_audio(audio, sample_rate=sample_rate) for audio in audios]
        else:
            audio = _unwrap_audio(
                record.get("audio"),
                sample_rate=sample_rate,
            )
            cuts = record.get("audio_cut_idxs")
            if not _is_sequence(cuts) or len(cuts) != len(texts):
                raise ValueError("Grouped CSM audio requires one `(start, end)` cut per "
                                 "text.")
            audios = [audio[int(start):int(end)] for start, end in cuts]
        return [{
            "role": str(speaker),
            "content": [
                {
                    "type": "text",
                    "text": str(text),
                },
                {
                    "type": "audio",
                    "audio": audio,
                },
            ],
        } for speaker, text, audio in zip(speakers, texts, audios)]
    if "text" not in record or "audio" not in record:
        raise ValueError("CSM training records require text and audio.")
    return [{
        "role":
        str(record.get("speaker_id", record.get("speaker", 0))),
        "content": [
            {
                "type": "text",
                "text": str(record["text"]),
            },
            {
                "type": "audio",
                "audio": _unwrap_audio(
                    record["audio"],
                    sample_rate=sample_rate,
                ),
            },
        ],
    }]


def freeze_csm_codec(value: Any) -> Any:
    """Freeze an attached Mimi codec and keep it in evaluation mode."""
    codec = getattr(value, "codec", None)
    if codec is None:
        runtime = getattr(value, "runtime", None)
        codec = getattr(runtime, "codec", None)
    if codec is None:
        codec = getattr(value, "codec_model", None)
    if codec is None:
        return None
    requires_grad = getattr(codec, "requires_grad_", None)
    if callable(requires_grad):
        requires_grad(False)
    else:
        parameters = getattr(codec, "parameters", None)
        if not callable(parameters):
            raise TypeError("CSM codec does not expose parameters().")
        for parameter in parameters():
            parameter.requires_grad = False
    evaluate = getattr(codec, "eval", None)
    if callable(evaluate):
        evaluate()
    return codec


@dataclass
class CSMTrainingCollator:
    """Collate source-layout codebook examples for native CSM."""

    processor: Any
    runtime: Any | None = None
    depth_decoder_labels_ratio: float = 1.0

    def __post_init__(self) -> None:
        ratio = float(self.depth_decoder_labels_ratio)
        if not 0.0 <= ratio <= 1.0:
            raise ValueError("`depth_decoder_labels_ratio` must be between zero and one.")
        self.depth_decoder_labels_ratio = ratio
        self.sample_rate = int(getattr(self.processor, "sample_rate", 24_000))

    def __call__(
        self,
        records: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        if not _is_sequence(records) or not records:
            raise ValueError("CSMTrainingCollator requires at least one record.")
        if any(not isinstance(record, Mapping) for record in records):
            raise TypeError("Every CSM training record must be a mapping.")
        prepared = list(records)
        training_batch = getattr(self.processor, "training_batch", None)
        if not callable(training_batch):
            apply_template = getattr(
                self.processor,
                "apply_chat_template",
                None,
            )
            if not callable(apply_template):
                raise TypeError("CSM processors must implement `training_batch()`.")
            conversations = [
                _legacy_conversation(
                    record,
                    sample_rate=self.sample_rate,
                ) for record in records
            ]
            output = apply_template(
                conversations,
                tokenize=True,
                return_dict=True,
                output_labels=True,
                depth_decoder_labels_ratio=(self.depth_decoder_labels_ratio),
            )
            if not isinstance(output, Mapping):
                raise TypeError("CSM processor output must be a mapping.")
            return dict(output)
        if self.runtime is not None:
            prepared = self.runtime.encode_training_records(records)
        elif any("audio_codes" not in record and "segments" not in record for record in records):
            raise RuntimeError(
                "Raw-audio CSM collation requires the frozen native Mimi "
                "runtime. Supply pre-encoded `audio_codes` otherwise.")
        options = {
            "depth_decoder_labels_ratio": self.depth_decoder_labels_ratio,
        }
        if self.runtime is not None:
            options["device"] = getattr(self.runtime, "device", None)
        return training_batch(prepared, **options)

    def resume_fingerprint(self) -> dict[str, Any]:
        return {
            "depth_decoder_labels_ratio": self.depth_decoder_labels_ratio,
            "sample_rate": self.sample_rate,
            "preprocessing": "native-mimi-codebooks",
        }


def _columnar_records(inputs: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    records = inputs.get("records")
    if records is not None:
        if not _is_sequence(records):
            raise TypeError("CSM `records` must be a sequence of mappings.")
        return list(records)
    texts = inputs.get("text")
    if not _is_sequence(texts):
        return [inputs]
    length = len(texts)
    output = []
    for index in range(length):
        record = {}
        for key in (
                "text",
                "audio",
                "audio_codes",
                "sampling_rate",
                "sample_rate",
                "speaker",
                "speaker_id",
        ):
            value = inputs.get(key)
            shape = getattr(value, "shape", ()) or ()
            minimum_rank = {
                "audio": 2,
                "audio_codes": 3,
            }.get(key, 1)
            is_tensor_column = (
                hasattr(value, "__getitem__") and len(shape) >= minimum_rank and int(shape[0]) == length)
            if ((_is_sequence(value) and len(value) == length) or is_tensor_column):
                record[key] = value[index]
            elif value is not None:
                record[key] = value
        output.append(record)
    return output


def prepare_csm_training_inputs(
    processor: Any,
    inputs: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    runtime: Any | None = None,
    depth_decoder_labels_ratio: float = 1.0,
) -> dict[str, Any]:
    """Pass through prepared tensors or collate codebook/raw-audio records."""
    if isinstance(inputs, Mapping):
        if _PREPARED_INPUT_KEYS.issubset(inputs):
            return dict(inputs)
        if ("labels" in inputs and _LEGACY_PREPARED_INPUT_KEYS.intersection(inputs)):
            return dict(inputs)
        records = _columnar_records(inputs)
    elif _is_sequence(inputs):
        records = list(inputs)
    else:
        raise TypeError("CSM training inputs must be a mapping or sequence.")
    return CSMTrainingCollator(
        processor,
        runtime=runtime,
        depth_decoder_labels_ratio=depth_decoder_labels_ratio,
    )(records)


@dataclass
class CSMTrainingBackend:
    """Native CSM model, processor, frozen codec, and objective helpers."""

    model: Any
    processor: Any
    sample_rate: int
    runtime: Any | None = None
    # Retained only to read artifacts produced by the earlier integration.
    # The native backend never imports or requires Transformers.
    transformers_major_version: int | None = None

    @classmethod
    def from_runtime(cls, runtime: Any) -> CSMTrainingBackend:
        backend = cls(
            model=runtime.model,
            processor=runtime.processor,
            sample_rate=int(runtime.sample_rate),
            runtime=runtime,
        )
        backend.freeze_codec()
        return backend

    def freeze_codec(self) -> Any:
        return freeze_csm_codec(self.runtime or self)

    def create_collator(
        self,
        *,
        depth_decoder_labels_ratio: float = 1.0,
    ) -> CSMTrainingCollator:
        return CSMTrainingCollator(
            self.processor,
            runtime=self.runtime,
            depth_decoder_labels_ratio=depth_decoder_labels_ratio,
        )

    def prepare_inputs(
        self,
        inputs: Mapping[str, Any] | Sequence[Mapping[str, Any]],
        *,
        depth_decoder_labels_ratio: float = 1.0,
    ) -> dict[str, Any]:
        return prepare_csm_training_inputs(
            self.processor,
            inputs,
            runtime=self.runtime,
            depth_decoder_labels_ratio=depth_decoder_labels_ratio,
        )

    @staticmethod
    def scalar_loss(outputs: Any) -> Any:
        if isinstance(outputs, Mapping):
            loss = outputs.get("loss")
        else:
            loss = getattr(outputs, "loss", None)
        if loss is None:
            raise RuntimeError("Native CSM returned no loss. Supply pre-encoded codebook "
                               "`labels`.")
        numel = getattr(loss, "numel", None)
        if not callable(numel) or int(numel()) != 1:
            raise ValueError("Native CSM must return exactly one loss value.")
        reshape = getattr(loss, "reshape", None)
        return reshape(()) if callable(reshape) else loss

    def forward_loss(
        self,
        inputs: Mapping[str, Any] | None = None,
        **model_inputs: Any,
    ) -> Any:
        if inputs is not None:
            if model_inputs:
                raise ValueError("Pass CSM inputs as a mapping or keywords, not both.")
            model_inputs = dict(inputs)
        self.freeze_codec()
        outputs = self.model(**model_inputs)
        return self.scalar_loss(outputs)

    def save_pretrained(self, save_directory: str | Path) -> Path:
        if self.runtime is None:
            save_model = getattr(self.model, "save_pretrained", None)
            save_processor = getattr(self.processor, "save_pretrained", None)
            if not callable(save_model) or not callable(save_processor):
                raise RuntimeError(
                    "A complete CSM export requires its native runtime and "
                    "tokenizer artifacts.")
            output = Path(save_directory).expanduser()
            output.mkdir(parents=True, exist_ok=True)
            save_model(output, safe_serialization=True)
            save_processor(output)
            return output
        return self.runtime.save_pretrained(
            save_directory,
            include_codec=self.runtime.codec is not None,
        )


class _LazyCSMTrainingCollator:

    def __init__(self, wrapper: Any) -> None:
        self.wrapper = wrapper

    def __call__(self, records):
        backend = getattr(self.wrapper, "training_backend", None)
        if backend is None:
            raise RuntimeError("CSM must be loaded for training before collation.")
        return backend.create_collator()(records)

    def resume_fingerprint(self) -> dict[str, Any]:
        config = getattr(self.wrapper, "config", None)
        return {
            "depth_decoder_labels_ratio": 1.0,
            "sample_rate": getattr(config, "sample_rate", None),
            'base_model': getattr(config, "name_or_path", None),
            "runtime": "voicehub-native",
        }


class CSMTrainingAdapter(CausalLMTrainingAdapter):
    """Route Trainer through CSM's native two-level objective."""

    native_export_semantics = "inference-export"

    def __init__(self, model: Any, spec: Any):
        super().__init__(model, spec)
        self.data_collator = _LazyCSMTrainingCollator(model)

    def save_pretrained(self, save_directory) -> None:
        self.setup()
        backend = getattr(self.model, "training_backend", None)
        if backend is None:
            raise RuntimeError("CSM cannot export before its native backend is loaded.")
        backend.save_pretrained(save_directory)


def load_csm_training_backend(
    model_name_or_path: str,
    *,
    device: str,
    torch_dtype: str = "bfloat16",
    codec: Any | None = None,
    codec_path: str | Path | None = None,
    include_codec: bool = True,
    revision: str | None = None,
    cache_dir: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
    verify_integrity: bool = False,
    verify_checkpoint_integrity: bool = False,
) -> CSMTrainingBackend:
    """Load the VoiceHub-native CSM graph without Transformers."""
    from voicehub.models.csm.native.runtime import load_csm_runtime

    runtime = load_csm_runtime(
        model_name_or_path,
        device=device,
        dtype=torch_dtype,
        codec=codec,
        codec_path=codec_path,
        include_codec=include_codec,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
        verify_integrity=verify_integrity,
        verify_checkpoint_integrity=verify_checkpoint_integrity,
    )
    runtime.model.train()
    return CSMTrainingBackend.from_runtime(runtime)


__all__ = [
    "CSMTrainingAdapter",
    "CSMTrainingBackend",
    "CSMTrainingCollator",
    "freeze_csm_codec",
    "load_csm_training_backend",
    "prepare_csm_training_inputs",
]
