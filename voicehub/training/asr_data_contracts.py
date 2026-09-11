"""Inspectable source-data contracts for ASR fine-tuning.

The contracts in this module describe both the human-authored records
that VoiceHub can preprocess and the cached tensor layouts accepted by
each ASR runtime.  They are deliberately framework-free so applications
can inspect and validate dataset shapes without importing PyTorch or
loading a model.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from enum import Enum
from types import MappingProxyType
from typing import Any

from voicehub.dependencies import normalize_import_path, resolve_import_path


def _field_names(values: Iterable[str], *, owner: str) -> tuple[str, ...]:
    if isinstance(values, str):
        values = (values, )
    normalized = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{owner} must contain non-empty field names.")
        normalized.append(value.strip())
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{owner} must not contain duplicate field names.")
    return tuple(normalized)


def _field_present(record: Mapping[str, Any], name: str) -> bool:
    if name not in record or record[name] is None:
        return False
    value = record[name]
    return not isinstance(value, (str, bytes, bytearray)) or bool(value.strip())


def _field_aliases(values: Mapping[str, str] | Iterable[tuple[str, str]], ) -> tuple[tuple[str, str], ...]:
    items = tuple(values.items()) if isinstance(values, Mapping) else tuple(values)
    normalized = []
    seen = set()
    for item in items:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise TypeError("ASR field_aliases entries must be source/target pairs.")
        source, target = item
        if not isinstance(source, str) or not source.strip():
            raise ValueError("ASR field alias sources must be non-empty strings.")
        if not isinstance(target, str) or not target.strip():
            raise ValueError("ASR field alias targets must be non-empty strings.")
        source = source.strip()
        target = target.strip()
        if source == target:
            raise ValueError(f"ASR field alias {source!r} cannot target itself.")
        if source in seen:
            raise ValueError(f"ASR field_aliases repeats source {source!r}.")
        seen.add(source)
        normalized.append((source, target))
    return tuple(normalized)


class ASRDataArchitecture(str, Enum):
    """Canonical source-data layouts used by ASR fine-tuning recipes."""

    NATIVE_DISPATCH = "native-dispatch"
    CTC = "ctc"
    SPEECH_SEQUENCE_TO_SEQUENCE = "speech-sequence-to-sequence"
    # Shorter spelling retained as an ergonomic enum alias.
    SEQUENCE_TO_SEQUENCE = "speech-sequence-to-sequence"
    PROMPTED_MULTIMODAL = "prompted-multimodal"
    RNNT = "rnnt"
    TDT = "tdt"
    HYBRID_CTC_ATTENTION = "hybrid-ctc-attention"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def coerce(cls, value: ASRDataArchitecture | str) -> ASRDataArchitecture:
        if isinstance(value, cls):
            return value
        if not isinstance(value, str) or not value.strip():
            raise TypeError("ASR data architecture must be a non-empty string or enum value.")
        normalized = value.strip().lower().replace("_", "-")
        aliases = {
            "dispatch": cls.NATIVE_DISPATCH.value,
            "native-asr-dispatch": cls.NATIVE_DISPATCH.value,
            "connectionist-temporal-classification": cls.CTC.value,
            "seq2seq": cls.SPEECH_SEQUENCE_TO_SEQUENCE.value,
            "speech-seq2seq": cls.SPEECH_SEQUENCE_TO_SEQUENCE.value,
            "sequence-to-sequence": cls.SPEECH_SEQUENCE_TO_SEQUENCE.value,
            "multimodal": cls.PROMPTED_MULTIMODAL.value,
            "prompted-seq2seq": cls.PROMPTED_MULTIMODAL.value,
            "rnn-t": cls.RNNT.value,
            "transducer": cls.RNNT.value,
            "token-duration-transducer": cls.TDT.value,
            "ctc-attention": cls.HYBRID_CTC_ATTENTION.value,
            "hybrid": cls.HYBRID_CTC_ATTENTION.value,
        }
        normalized = aliases.get(normalized, normalized)
        try:
            return cls(normalized)
        except ValueError as exc:
            choices = ", ".join(member.value for member in cls)
            raise ValueError(
                f"Unknown ASR data architecture {value!r}. Expected one of: "
                f"{choices}.") from exc


class ASRDataReadiness(str, Enum):
    """How far VoiceHub owns a model's ASR dataset preparation path."""

    INTEGRATED = "integrated-raw"
    PREPROCESSED = "preprocessed"
    CUSTOM = "custom"
    UNAVAILABLE = "unavailable"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def coerce(cls, value: ASRDataReadiness | str) -> ASRDataReadiness:
        if isinstance(value, cls):
            return value
        if not isinstance(value, str) or not value.strip():
            raise TypeError("ASR data readiness must be a non-empty string or enum value.")
        normalized = value.strip().lower().replace("_", "-")
        try:
            return cls(normalized)
        except ValueError as exc:
            choices = ", ".join(member.value for member in cls)
            raise ValueError(
                f"Unknown ASR data readiness {value!r}. Expected one of: "
                f"{choices}.") from exc


@dataclass(frozen=True)
class ASRRecordVariant:
    """One accepted raw or preprocessed ASR record shape.

    Every field in ``required_fields`` must be present. Each tuple in
    ``one_of`` is a group of aliases for which at least one field must
    be present. ``at_most_one_of`` rejects ambiguous aliases, while
    dependency fields describe metadata that becomes mandatory when a
    trigger is used. Values that are ``None`` or empty strings count as
    absent.
    """

    name: str
    required_fields: tuple[str, ...] = ()
    one_of: tuple[tuple[str, ...], ...] = ()
    at_most_one_of: tuple[tuple[str, ...], ...] = ()
    forbidden_fields: tuple[str, ...] = ()
    requires: tuple[tuple[str, tuple[str, ...]], ...] = ()
    requires_one_of: tuple[tuple[str, tuple[str, ...]], ...] = ()
    description: str = ""
    preprocessed: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("ASR record variant names must be non-empty strings.")
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(
            self,
            "required_fields",
            _field_names(self.required_fields, owner="required_fields"),
        )
        groups = []
        for group in self.one_of:
            normalized = _field_names(group, owner="one_of")
            if not normalized:
                raise ValueError("ASR record one_of groups cannot be empty.")
            groups.append(normalized)
        object.__setattr__(self, "one_of", tuple(groups))
        exclusive_groups = []
        for group in self.at_most_one_of:
            normalized = _field_names(group, owner="at_most_one_of")
            if len(normalized) < 2:
                raise ValueError("ASR record at_most_one_of groups require at least two fields.")
            exclusive_groups.append(normalized)
        object.__setattr__(self, "at_most_one_of", tuple(exclusive_groups))
        object.__setattr__(
            self,
            "forbidden_fields",
            _field_names(self.forbidden_fields, owner="forbidden_fields"),
        )
        object.__setattr__(
            self,
            "requires",
            self._normalize_dependencies(self.requires, owner="requires"),
        )
        object.__setattr__(
            self,
            "requires_one_of",
            self._normalize_dependencies(
                self.requires_one_of,
                owner="requires_one_of",
            ),
        )
        if not isinstance(self.description, str):
            raise TypeError("ASR record variant descriptions must be strings.")
        if not isinstance(self.preprocessed, bool):
            raise TypeError("ASR record variant preprocessed must be a boolean.")

    @staticmethod
    def _normalize_dependencies(
        values: Iterable[tuple[str, Iterable[str]]],
        *,
        owner: str,
    ) -> tuple[tuple[str, tuple[str, ...]], ...]:
        normalized = []
        for value in values:
            if not isinstance(value, (tuple, list)) or len(value) != 2:
                raise TypeError(f"{owner} entries must be (trigger, required_fields) pairs.")
            trigger, required = value
            trigger_fields = _field_names((trigger, ), owner=f"{owner} trigger")
            dependencies = _field_names(required, owner=owner)
            if not dependencies:
                raise ValueError(f"{owner} dependency groups cannot be empty.")
            normalized.append((trigger_fields[0], dependencies))
        triggers = [trigger for trigger, _ in normalized]
        if len(set(triggers)) != len(triggers):
            raise ValueError(f"{owner} must not repeat trigger fields.")
        return tuple(normalized)

    def missing(self, record: Mapping[str, Any]) -> tuple[str, ...]:
        """Return human-readable contract issues found in ``record``."""
        missing = [name for name in self.required_fields if not _field_present(record, name)]
        for group in self.one_of:
            if not any(_field_present(record, name) for name in group):
                missing.append("(" + " or ".join(group) + ")")
        for group in self.at_most_one_of:
            present = [name for name in group if _field_present(record, name)]
            if len(present) > 1:
                missing.append("at most one of (" + ", ".join(group) + ")")
        for name in self.forbidden_fields:
            if _field_present(record, name):
                missing.append(f"forbidden field {name}")
        for trigger, required in self.requires:
            if not _field_present(record, trigger):
                continue
            absent = [name for name in required if not _field_present(record, name)]
            if absent:
                missing.append(f"{trigger} requires " + ", ".join(absent))
        for trigger, alternatives in self.requires_one_of:
            if (_field_present(record, trigger) and
                    not any(_field_present(record, name) for name in alternatives)):
                missing.append(f"{trigger} requires one of (" + " or ".join(alternatives) + ")")
        return tuple(missing)

    def matches(self, record: Mapping[str, Any]) -> bool:
        return not self.missing(record)


@dataclass(frozen=True)
class ASRDatasetSpec:
    """Inspectable source-data contract for one ASR architecture or model."""

    architecture: ASRDataArchitecture
    variants: tuple[ASRRecordVariant, ...]
    model_type: str | None = None
    sample_rate: int | None = None
    description: str = ""
    readiness: ASRDataReadiness | None = None
    training_support: str | None = None
    homogeneous_batch_fields: tuple[tuple[str, ...], ...] = ()
    field_aliases: Mapping[str, str] | tuple[tuple[str, str], ...] = ()
    record_normalizer: str | None = None
    record_normalizer_phase: str = "after-aliases"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "architecture",
            ASRDataArchitecture.coerce(self.architecture),
        )
        variants = tuple(self.variants)
        if not variants or any(not isinstance(variant, ASRRecordVariant) for variant in variants):
            raise ValueError("ASRDatasetSpec.variants must contain ASRRecordVariant values.")
        names = [variant.name for variant in variants]
        if len(set(names)) != len(names):
            raise ValueError("ASR record variant names must be unique within a dataset spec.")
        object.__setattr__(self, "variants", variants)
        if self.model_type is not None:
            if not isinstance(self.model_type, str) or not self.model_type.strip():
                raise ValueError("ASRDatasetSpec.model_type must be a non-empty string or None.")
            object.__setattr__(self, "model_type", self.model_type.strip().lower())
        if self.sample_rate is not None:
            if isinstance(self.sample_rate, bool) or not isinstance(self.sample_rate, int):
                raise TypeError("ASRDatasetSpec.sample_rate must be an integer or None.")
            if self.sample_rate <= 0:
                raise ValueError("ASRDatasetSpec.sample_rate must be positive.")
        if not isinstance(self.description, str):
            raise TypeError("ASRDatasetSpec.description must be a string.")
        if self.readiness is not None:
            object.__setattr__(
                self,
                "readiness",
                ASRDataReadiness.coerce(self.readiness),
            )
        if self.training_support is not None:
            if (not isinstance(self.training_support, str) or not self.training_support.strip()):
                raise ValueError("ASRDatasetSpec.training_support must be a non-empty string or None.")
            object.__setattr__(
                self,
                "training_support",
                self.training_support.strip().lower().replace("_", "-"),
            )
        homogeneous_groups = []
        for group in self.homogeneous_batch_fields:
            normalized = _field_names(group, owner="homogeneous_batch_fields")
            if not normalized:
                raise ValueError("ASR homogeneous batch field groups cannot be empty.")
            homogeneous_groups.append(normalized)
        if len(set(homogeneous_groups)) != len(homogeneous_groups):
            raise ValueError("ASR homogeneous batch field groups must be unique.")
        object.__setattr__(
            self,
            "homogeneous_batch_fields",
            tuple(homogeneous_groups),
        )
        object.__setattr__(
            self,
            "field_aliases",
            _field_aliases(self.field_aliases),
        )
        normalizer = self.record_normalizer
        if normalizer is not None:
            normalizer = normalize_import_path(
                normalizer,
                name="ASR record_normalizer",
            )
        object.__setattr__(self, "record_normalizer", normalizer)
        phase = self.record_normalizer_phase
        if phase not in {"before-aliases", "after-aliases"}:
            raise ValueError("ASR record_normalizer_phase must be 'before-aliases' or "
                             "'after-aliases'.")
        if normalizer is None and phase != "after-aliases":
            raise ValueError("ASR record_normalizer_phase requires record_normalizer.")

    def match_variant(self, record: Mapping[str, Any], *, index: int | None = None) -> str:
        """Validate a record and return its most specific matching variant.

        A model-ready row can intentionally retain its source ``audio``
        and ``text`` for traceability. Prefer preprocessed variants so
        those rows are dispatched through the cached-input path instead
        of being needlessly processed again.
        """
        if not isinstance(record, Mapping):
            location = "" if index is None else f" {index}"
            raise TypeError(f"ASR record{location} must be a mapping, received "
                            f"{type(record).__name__}.")
        ordered_variants = self.preprocessed_variants + self.raw_variants
        for variant in ordered_variants:
            if variant.matches(record):
                return variant.name
        location = "" if index is None else f" {index}"
        alternatives = "; ".join(
            f"{variant.name}: {', '.join(variant.missing(record)) or 'valid'}"
            for variant in ordered_variants)
        target = f" for {self.model_type!r}" if self.model_type else ""
        raise ValueError(
            f"ASR record{location} does not match any {self.architecture.value} "
            f"dataset variant{target}. Missing requirements — {alternatives}.")

    @property
    def raw_variants(self) -> tuple[ASRRecordVariant, ...]:
        return tuple(variant for variant in self.variants if not variant.preprocessed)

    @property
    def preprocessed_variants(self) -> tuple[ASRRecordVariant, ...]:
        return tuple(variant for variant in self.variants if variant.preprocessed)

    @property
    def accepts_raw_records(self) -> bool:
        """Whether this contract includes model-owned raw-audio
        preprocessing."""
        return bool(self.raw_variants)

    @property
    def requires_preprocessing(self) -> bool:
        return self.readiness is ASRDataReadiness.PREPROCESSED

    @property
    def requires_homogeneous_batches(self) -> bool:
        """Whether one or more metadata values must be uniform per batch."""
        return bool(self.homogeneous_batch_fields)


def _variant(
    name: str,
    *,
    required: tuple[str, ...] = (),
    one_of: tuple[tuple[str, ...], ...] = (),
    at_most_one_of: tuple[tuple[str, ...], ...] = (),
    forbidden: tuple[str, ...] = (),
    requires: tuple[tuple[str, tuple[str, ...]], ...] = (),
    requires_one_of: tuple[tuple[str, tuple[str, ...]], ...] = (),
    description: str = "",
    preprocessed: bool = False,
) -> ASRRecordVariant:
    return ASRRecordVariant(
        name=name,
        required_fields=required,
        one_of=one_of,
        at_most_one_of=at_most_one_of,
        forbidden_fields=forbidden,
        requires=requires,
        requires_one_of=requires_one_of,
        description=description,
        preprocessed=preprocessed,
    )


_TRANSCRIPT_FIELDS = ("text", "transcription", "transcript")


def _raw_audio(
    *,
    name: str = "raw-audio",
    required: tuple[str, ...] = (),
    audio_fields: tuple[str, ...] = ("audio", ),
    forbidden: tuple[str, ...] = (),
    description: str = "Audio paired with a non-empty transcript.",
) -> ASRRecordVariant:
    required_fields = required
    one_of = (_TRANSCRIPT_FIELDS, )
    exclusive = (_TRANSCRIPT_FIELDS, )
    if len(audio_fields) == 1:
        required_fields = (*audio_fields, *required_fields)
    else:
        one_of = (audio_fields, *one_of)
        exclusive = (audio_fields, *exclusive)
    return _variant(
        name,
        required=required_fields,
        one_of=one_of,
        at_most_one_of=exclusive,
        forbidden=forbidden,
        description=description,
    )


_FEATURE_SEQ2SEQ = _variant(
    "feature-seq2seq",
    required=("input_features", "labels"),
    description="Cached acoustic features and teacher-forced text labels.",
    preprocessed=True,
)
_WAVEFORM_SEQ2SEQ = _variant(
    "waveform-seq2seq",
    required=("input_values", "labels"),
    description="Cached waveform values and teacher-forced text labels.",
    preprocessed=True,
)
_WAVEFORM_CTC = _variant(
    "waveform-ctc",
    required=("input_values", "labels"),
    description="Cached waveform values and padded CTC labels.",
    preprocessed=True,
)
_FEATURE_CTC = _variant(
    "feature-ctc",
    required=("input_features", "labels"),
    description="Cached acoustic features and padded CTC labels.",
    preprocessed=True,
)

_ARCHITECTURE_SPECS: Mapping[ASRDataArchitecture, ASRDatasetSpec] = MappingProxyType({
    ASRDataArchitecture.NATIVE_DISPATCH:
    ASRDatasetSpec(
        architecture=ASRDataArchitecture.NATIVE_DISPATCH,
        sample_rate=16_000,
        description=(
            "Raw audio/transcript records or cached inputs for the selected "
            "native Whisper, CTC, or Moonshine delegate."),
        variants=(
            _raw_audio(),
            _FEATURE_SEQ2SEQ,
            _WAVEFORM_SEQ2SEQ,
        ),
    ),
    ASRDataArchitecture.CTC:
    ASRDatasetSpec(
        architecture=ASRDataArchitecture.CTC,
        sample_rate=16_000,
        description="Audio/transcript pairs or cached CTC inputs and token labels.",
        variants=(
            _raw_audio(),
            _WAVEFORM_CTC,
            _FEATURE_CTC,
        ),
    ),
    ASRDataArchitecture.SPEECH_SEQUENCE_TO_SEQUENCE:
    ASRDatasetSpec(
        architecture=ASRDataArchitecture.SPEECH_SEQUENCE_TO_SEQUENCE,
        sample_rate=16_000,
        description=(
            "Audio/transcript pairs or cached encoder inputs with "
            "teacher-forced decoder labels."),
        variants=(
            _raw_audio(),
            _FEATURE_SEQ2SEQ,
            _WAVEFORM_SEQ2SEQ,
        ),
    ),
    ASRDataArchitecture.PROMPTED_MULTIMODAL:
    ASRDatasetSpec(
        architecture=ASRDataArchitecture.PROMPTED_MULTIMODAL,
        sample_rate=16_000,
        description=(
            "Audio, transcript, and optional prompt metadata converted to "
            "completion-only multimodal language-model supervision."),
        variants=(
            _raw_audio(),
            _variant(
                "multimodal-model-ready",
                required=("input_ids", "attention_mask", "labels"),
                one_of=(("input_features", "input_values"), ),
                description="Tokenized prompt/completion plus encoded audio.",
                preprocessed=True,
            ),
        ),
    ),
    ASRDataArchitecture.RNNT:
    ASRDatasetSpec(
        architecture=ASRDataArchitecture.RNNT,
        sample_rate=16_000,
        description="Audio/transcript pairs or model-ready RNN-T graph inputs.",
        variants=(
            _raw_audio(),
            _variant(
                "rnnt-model-ready",
                required=(
                    "input_features",
                    "attention_mask",
                    "prompt_ids",
                    "labels",
                    "label_lengths",
                    "decoder_input_ids",
                ),
                preprocessed=True,
            ),
        ),
    ),
    ASRDataArchitecture.TDT:
    ASRDatasetSpec(
        architecture=ASRDataArchitecture.TDT,
        sample_rate=16_000,
        description="Audio/transcript pairs or model-ready token-duration transducer inputs.",
        variants=(
            _raw_audio(),
            _variant(
                "tdt-model-ready",
                required=(
                    "input_features",
                    "attention_mask",
                    "labels",
                    "decoder_input_ids",
                ),
                preprocessed=True,
            ),
        ),
    ),
    ASRDataArchitecture.HYBRID_CTC_ATTENTION:
    ASRDatasetSpec(
        architecture=ASRDataArchitecture.HYBRID_CTC_ATTENTION,
        sample_rate=16_000,
        description=(
            "Audio/transcript pairs or cached inputs for joint CTC and "
            "attention-decoder objectives."),
        variants=(
            _raw_audio(),
            _variant(
                "hybrid-model-ready",
                required=("labels", "label_lengths"),
                one_of=(("features", "waveforms", "input_signal"), ),
                preprocessed=True,
            ),
        ),
    ),
})


def _model_spec(
    architecture: ASRDataArchitecture,
    variants: tuple[ASRRecordVariant, ...],
    description: str,
    *,
    sample_rate: int = 16_000,
    homogeneous_batch_fields: tuple[tuple[str, ...], ...] = (),
    field_aliases: tuple[tuple[str, str], ...] = (),
    record_normalizer: str | None = None,
    record_normalizer_phase: str = "after-aliases",
) -> ASRDatasetSpec:
    return ASRDatasetSpec(
        architecture=architecture,
        variants=variants,
        description=description,
        sample_rate=sample_rate,
        homogeneous_batch_fields=homogeneous_batch_fields,
        field_aliases=field_aliases,
        record_normalizer=record_normalizer,
        record_normalizer_phase=record_normalizer_phase,
    )


_WHISPER_VARIANTS = (
    _raw_audio(
        description=(
            "A waveform or audio path paired with a transcript; language and "
            "task metadata are optional."), ),
    _variant(
        "whisper-model-ready",
        required=("input_features", "labels"),
        description="Whisper log-mel frames and decoder token labels.",
        preprocessed=True,
    ),
)

_CTC_WAVEFORM_VARIANTS = (
    _raw_audio(),
    _WAVEFORM_CTC,
)

_NEMO_CTC_VARIANTS = (
    _raw_audio(),
    _variant(
        "nemo-ctc-waveform-model-ready",
        required=(
            "input_signal",
            "input_signal_length",
            "labels",
            "label_lengths",
        ),
        description="Padded waveform signals, lengths, and CTC targets.",
        preprocessed=True,
    ),
    _variant(
        "nemo-ctc-feature-model-ready",
        required=(
            "processed_signal",
            "processed_signal_length",
            "labels",
            "label_lengths",
        ),
        description="Cached log-mel features, lengths, and CTC targets.",
        preprocessed=True,
    ),
)


def build_asr_transformers_dataset_spec() -> ASRDatasetSpec:
    """Return the source-data contract for the checkpoint-dispatch ASR
    backend."""
    return _model_spec(
        ASRDataArchitecture.NATIVE_DISPATCH,
        (
            _raw_audio(),
            _variant(
                "feature-model-ready",
                required=("input_features", "labels"),
                preprocessed=True,
            ),
            _variant(
                "waveform-model-ready",
                required=("input_values", "labels"),
                preprocessed=True,
            ),
        ),
        "Checkpoint-dispatched raw and cached inputs for native Transformers ASR families.",
    )


def build_asr_whisper_dataset_spec() -> ASRDatasetSpec:
    """Return the native Whisper source-data contract."""
    return _model_spec(
        ASRDataArchitecture.SPEECH_SEQUENCE_TO_SEQUENCE,
        _WHISPER_VARIANTS,
        "Native Whisper transcription or translation fine-tuning records.",
    )


def build_asr_tiron_dataset_spec() -> ASRDatasetSpec:
    """Return the Tiron source-data contract."""
    return _model_spec(
        ASRDataArchitecture.SPEECH_SEQUENCE_TO_SEQUENCE,
        (
            _raw_audio(
                description=(
                    "Audio plus the native inline speaker/timestamp target; "
                    "language may come from the record or model configuration."), ),
            _variant(
                "tiron-model-ready",
                required=("input_features", "labels"),
                description="Tiron Whisper features and grammar-token labels.",
                preprocessed=True,
            ),
        ),
        "Speaker-aware Whisper fine-tuning with Tiron's inline timestamp grammar.",
    )


def build_asr_qwen3_dataset_spec() -> ASRDatasetSpec:
    """Return the Qwen3-ASR source-data contract."""
    return _model_spec(
        ASRDataArchitecture.PROMPTED_MULTIMODAL,
        (
            _raw_audio(
                description=(
                    "Audio/transcript pair with optional context, prompt, and "
                    "language metadata."), ),
            _variant(
                "qwen3-model-ready",
                required=(
                    "input_ids",
                    "attention_mask",
                    "input_features",
                    "feature_attention_mask",
                    "labels",
                ),
                description="Completion-only Qwen token labels plus log-mel features.",
                preprocessed=True,
            ),
        ),
        "Qwen3-ASR completion-only multimodal fine-tuning records.",
    )


def build_asr_vibevoice_dataset_spec() -> ASRDatasetSpec:
    """Return the VibeVoice ASR source-data contract."""
    return _model_spec(
        ASRDataArchitecture.PROMPTED_MULTIMODAL,
        (
            _variant(
                "segmented-audio",
                required=("audio", "segments"),
                forbidden=_TRANSCRIPT_FIELDS,
                description=(
                    "Audio plus structured speaker/timestamp/content segments "
                    "and optional context."),
            ),
            _raw_audio(
                name="serialized-audio",
                forbidden=("segments", ),
                description=("Audio plus a serialized VibeVoice segment target and "
                             "optional context."),
            ),
            _variant(
                "vibevoice-model-ready",
                required=(
                    "input_ids",
                    "attention_mask",
                    "input_values",
                    "padding_mask",
                    "labels",
                ),
                description="Completion-only token labels and 24 kHz audio values.",
                preprocessed=True,
            ),
        ),
        "VibeVoice structured long-form ASR targets and multimodal prompt inputs.",
        sample_rate=24_000,
    )


def build_asr_granite_speech_dataset_spec() -> ASRDatasetSpec:
    """Return the Granite Speech source-data contract."""
    return _model_spec(
        ASRDataArchitecture.PROMPTED_MULTIMODAL,
        (
            _raw_audio(
                forbidden=("language", ),
                description=(
                    "Audio/transcript pair with optional prompt text; language "
                    "guidance belongs in the prompt."),
            ),
            _variant(
                "granite-model-ready",
                required=(
                    "input_ids",
                    "attention_mask",
                    "input_features",
                    "input_features_mask",
                    "labels",
                ),
                description="Granite prompt/completion tokens and acoustic features.",
                preprocessed=True,
            ),
        ),
        "Prompt-conditioned Granite Speech multimodal fine-tuning records.",
    )


def build_asr_parakeet_tdt_dataset_spec() -> ASRDatasetSpec:
    """Return the Parakeet TDT source-data contract."""
    return _model_spec(
        ASRDataArchitecture.TDT,
        (
            _raw_audio(),
            _variant(
                "parakeet-tdt-model-ready",
                required=(
                    "input_features",
                    "attention_mask",
                    "labels",
                    "decoder_input_ids",
                ),
                description="Log-mel inputs and blank-prefixed TDT targets.",
                preprocessed=True,
            ),
        ),
        "Parakeet token-duration transducer audio and transcript records.",
    )


def build_asr_nemotron_dataset_spec() -> ASRDatasetSpec:
    """Return the Nemotron ASR source-data contract."""
    return _model_spec(
        ASRDataArchitecture.RNNT,
        (
            _raw_audio(description="Audio/transcript pair with optional language prompting.", ),
            _variant(
                "nemotron-rnnt-model-ready",
                required=(
                    "input_features",
                    "attention_mask",
                    "prompt_ids",
                    "labels",
                    "label_lengths",
                    "decoder_input_ids",
                ),
                description="Acoustic features, prompt IDs, and blank-prefixed RNN-T targets.",
                preprocessed=True,
            ),
        ),
        "Language-prompted Nemotron RNN-T fine-tuning records.",
    )


def build_asr_cohere_dataset_spec() -> ASRDatasetSpec:
    """Return the Cohere ASR source-data contract."""
    return _model_spec(
        ASRDataArchitecture.SPEECH_SEQUENCE_TO_SEQUENCE,
        (
            _raw_audio(
                required=("language", ),
                description=(
                    "Audio/transcript pair with an explicit language and "
                    "optional punctuation mode."),
            ),
            _variant(
                "cohere-model-ready",
                required=(
                    "input_features",
                    "attention_mask",
                    "decoder_input_ids",
                    "decoder_attention_mask",
                    "labels",
                ),
                description="Acoustic inputs and language-prompted decoder targets.",
                preprocessed=True,
            ),
        ),
        "Language- and punctuation-conditioned Cohere ASR records.",
        homogeneous_batch_fields=(("language", ), ("punctuation", )),
    )


def build_asr_medasr_dataset_spec() -> ASRDatasetSpec:
    """Return the MedASR source-data contract."""
    return _model_spec(
        ASRDataArchitecture.CTC,
        (
            _raw_audio(),
            _variant(
                "medasr-model-ready",
                required=("input_features", "attention_mask", "labels"),
                description="LASR acoustic features, mask, and padded CTC labels.",
                preprocessed=True,
            ),
        ),
        "MedASR native LASR feature and CTC transcript records.",
    )


def build_asr_wav2vec2_dataset_spec() -> ASRDatasetSpec:
    """Return the Wav2Vec2 source-data contract."""
    return _model_spec(
        ASRDataArchitecture.CTC,
        _CTC_WAVEFORM_VARIANTS,
        "Wav2Vec2 waveform and CTC transcript records.",
    )


def build_asr_hubert_dataset_spec() -> ASRDatasetSpec:
    """Return the HuBERT source-data contract."""
    return _model_spec(
        ASRDataArchitecture.CTC,
        _CTC_WAVEFORM_VARIANTS,
        "HuBERT waveform and CTC transcript records.",
    )


def build_asr_wavlm_dataset_spec() -> ASRDatasetSpec:
    """Return the WavLM source-data contract."""
    return _model_spec(
        ASRDataArchitecture.CTC,
        _CTC_WAVEFORM_VARIANTS,
        "WavLM waveform and CTC transcript records.",
    )


def build_asr_moonshine_dataset_spec() -> ASRDatasetSpec:
    """Return the Moonshine source-data contract."""
    return _model_spec(
        ASRDataArchitecture.SPEECH_SEQUENCE_TO_SEQUENCE,
        (
            _raw_audio(),
            _variant(
                "moonshine-model-ready",
                required=("input_values", "labels"),
                description="Waveform encoder values and teacher-forced decoder labels.",
                preprocessed=True,
            ),
        ),
        "Moonshine waveform-to-sequence fine-tuning records.",
    )


def build_asr_seamless_m4t_v2_dataset_spec() -> ASRDatasetSpec:
    """Return the SeamlessM4T-v2 source-data contract."""
    return _model_spec(
        ASRDataArchitecture.SPEECH_SEQUENCE_TO_SEQUENCE,
        (
            _raw_audio(
                description=(
                    "Audio/transcript pair with target language supplied by "
                    "the record or model configuration."), ),
            _variant(
                "seamless-model-ready",
                required=("input_features", "attention_mask", "labels"),
                description="Stacked acoustic features and language-conditioned labels.",
                preprocessed=True,
            ),
        ),
        "SeamlessM4T-v2 multilingual speech-to-text records.",
        homogeneous_batch_fields=(("target_language", "language"), ),
        field_aliases=(("target_lang", "target_language"), ),
        record_normalizer=("voicehub.models.asr_seamless_m4t_v2.native.data:normalize_record"),
        record_normalizer_phase="before-aliases",
    )


def build_asr_faster_whisper_dataset_spec() -> ASRDatasetSpec:
    """Return the Faster-Whisper-compatible source-data contract."""
    return _model_spec(
        ASRDataArchitecture.SPEECH_SEQUENCE_TO_SEQUENCE,
        _WHISPER_VARIANTS,
        "Faster-Whisper-compatible records trained through the native Whisper graph.",
    )


def build_asr_whisperx_dataset_spec() -> ASRDatasetSpec:
    """Return the WhisperX-compatible source-data contract."""
    return _model_spec(
        ASRDataArchitecture.SPEECH_SEQUENCE_TO_SEQUENCE,
        _WHISPER_VARIANTS,
        "WhisperX-compatible records trained through the native Whisper graph.",
    )


def build_asr_openai_whisper_dataset_spec() -> ASRDatasetSpec:
    """Return the OpenAI Whisper-compatible source-data contract."""
    return _model_spec(
        ASRDataArchitecture.SPEECH_SEQUENCE_TO_SEQUENCE,
        _WHISPER_VARIANTS,
        "OpenAI Whisper-compatible records trained through the native Whisper graph.",
    )


def build_asr_nemo_dataset_spec() -> ASRDatasetSpec:
    """Return the NeMo CTC source-data contract."""
    return _model_spec(
        ASRDataArchitecture.CTC,
        _NEMO_CTC_VARIANTS,
        "NeMo QuartzNet waveform and CTC transcript records.",
    )


def build_asr_speechbrain_dataset_spec() -> ASRDatasetSpec:
    """Return the SpeechBrain ASR source-data contract."""
    return _model_spec(
        ASRDataArchitecture.HYBRID_CTC_ATTENTION,
        (
            _raw_audio(
                audio_fields=("audio", "audio_path"),
                description="Audio or audio path paired with a transcript.",
            ),
            _variant(
                "speechbrain-model-ready",
                required=(
                    "waveforms",
                    "waveform_lengths",
                    "tokens_bos",
                    "tokens_eos",
                    "token_lengths",
                    "ctc_tokens",
                    "ctc_token_lengths",
                ),
                description="Waveforms plus BOS/EOS and parallel CTC targets.",
                preprocessed=True,
            ),
        ),
        "SpeechBrain CRDNN joint CTC/attention fine-tuning records.",
    )


def build_asr_funasr_dataset_spec() -> ASRDatasetSpec:
    """Return the FunASR SenseVoice source-data contract."""
    return _model_spec(
        ASRDataArchitecture.CTC,
        (
            _raw_audio(
                required=("language", ),
                audio_fields=("audio", "audio_values", "input_signal"),
                description=(
                    "A 16 kHz waveform plus transcript, language, and optional "
                    "emotion/event/ITN metadata."),
            ),
            _variant(
                "sensevoice-feature-transcript",
                required=("features", "language"),
                one_of=(_TRANSCRIPT_FIELDS, ),
                at_most_one_of=(_TRANSCRIPT_FIELDS, ),
                description="Precomputed frontend features with rich transcript metadata.",
                preprocessed=True,
            ),
            _variant(
                "sensevoice-model-ready",
                required=(
                    "features",
                    "feature_lengths",
                    "labels",
                    "label_lengths",
                ),
                description="SenseVoice frontend features and rich CTC labels.",
                preprocessed=True,
            ),
        ),
        "SenseVoice CTC records with language, emotion, event, and ITN control.",
        field_aliases=(
            ("emo_target", "emotion"),
            ("event_target", "event"),
            ("source", "audio"),
            ("target", "text"),
            ("text_language", "language"),
            ("with_or_wo_itn", "use_itn"),
        ),
        record_normalizer=("voicehub.models.asr_funasr.native.data:normalize_record"),
    )


def build_asr_espnet_dataset_spec() -> ASRDatasetSpec:
    """Return the ESPnet source-data contract."""
    return _model_spec(
        ASRDataArchitecture.HYBRID_CTC_ATTENTION,
        (
            _raw_audio(
                audio_fields=("audio", "audio_path"),
                description="Audio or audio path paired with a transcript.",
            ),
            _variant(
                "espnet-feature-transcript",
                required=("features", ),
                one_of=(_TRANSCRIPT_FIELDS, ),
                at_most_one_of=(_TRANSCRIPT_FIELDS, ),
                description="Cached acoustic features awaiting ESPnet label encoding.",
                preprocessed=True,
            ),
            _variant(
                "espnet-waveform-model-ready",
                required=(
                    "waveforms",
                    "waveform_lengths",
                    "labels",
                    "label_lengths",
                ),
                preprocessed=True,
            ),
            _variant(
                "espnet-feature-model-ready",
                required=(
                    "features",
                    "feature_lengths",
                    "labels",
                    "label_lengths",
                ),
                preprocessed=True,
            ),
        ),
        "ESPnet Transformer joint CTC/attention raw and cached records.",
    )


def build_asr_wenet_dataset_spec() -> ASRDatasetSpec:
    """Return the WeNet source-data contract."""
    return _model_spec(
        ASRDataArchitecture.HYBRID_CTC_ATTENTION,
        (
            _raw_audio(),
            _variant(
                "wenet-waveform-model-ready",
                required=(
                    "input_signal",
                    "input_signal_length",
                    "labels",
                    "label_lengths",
                ),
                description="Padded waveforms and shared CTC/attention token targets.",
                preprocessed=True,
            ),
            _variant(
                "wenet-feature-model-ready",
                required=(
                    "features",
                    "feature_lengths",
                    "labels",
                    "label_lengths",
                ),
                description=("Cached frontend features and shared CTC/attention "
                             "token targets."),
                preprocessed=True,
            ),
        ),
        "WeNet U2++ joint CTC/attention fine-tuning records.",
    )


_TRAINING_FAMILY_TO_DATA_ARCHITECTURE = MappingProxyType({
    "native-asr-dispatch": ASRDataArchitecture.NATIVE_DISPATCH,
    "ctc": ASRDataArchitecture.CTC,
    "speech-sequence-to-sequence": ASRDataArchitecture.SPEECH_SEQUENCE_TO_SEQUENCE,
    "rnnt": ASRDataArchitecture.RNNT,
    "tdt": ASRDataArchitecture.TDT,
})


def _load_model_dataset_spec(training_spec: Any) -> ASRDatasetSpec | None:
    factory_path = training_spec.dataset_spec_factory
    if factory_path is None:
        return None
    try:
        factory = resolve_import_path(factory_path)
    except (AttributeError, ImportError) as exc:
        raise ImportError(
            f"Could not resolve ASR dataset spec factory {factory_path!r} "
            f"for {training_spec.model_type!r}.") from exc
    if not callable(factory):
        raise TypeError(
            f"ASR dataset spec factory {factory_path!r} for "
            f"{training_spec.model_type!r} must be callable.")
    spec = factory()
    if not isinstance(spec, ASRDatasetSpec):
        raise TypeError(
            f"ASR dataset spec factory {factory_path!r} for "
            f"{training_spec.model_type!r} returned {type(spec).__name__}; "
            "expected ASRDatasetSpec.")
    if spec.model_type not in (None, training_spec.model_type):
        raise ValueError(
            f"ASR dataset spec factory {factory_path!r} returned a contract for "
            f"{spec.model_type!r}, not {training_spec.model_type!r}.")
    if spec.training_support not in (None, training_spec.support.value):
        raise ValueError(
            f"ASR dataset spec factory {factory_path!r} declares training support "
            f"{spec.training_support!r}, not {training_spec.support.value!r}.")
    return spec


def get_asr_dataset_spec(
    model_type: str | None = None,
    *,
    architecture: ASRDataArchitecture | str | None = None,
) -> ASRDatasetSpec:
    """Return the inspectable ASR dataset contract for a model or
    architecture."""
    canonical_model_type = None
    training_support = None
    model_spec = None
    if model_type is not None:
        if not isinstance(model_type, str) or not model_type.strip():
            raise ValueError("model_type must be a non-empty string or None.")
        from voicehub.training.specs import get_training_spec

        training_spec = get_training_spec(model_type)
        canonical_model_type = training_spec.model_type
        training_support = training_spec.support.value
        if training_spec.task.value != "automatic-speech-recognition":
            raise ValueError(
                f"{canonical_model_type!r} is registered for "
                f"{training_spec.task.value}, not ASR.")
        model_spec = _load_model_dataset_spec(training_spec)
        if model_spec is not None:
            resolved_architecture = model_spec.architecture
        else:
            try:
                resolved_architecture = _TRAINING_FAMILY_TO_DATA_ARCHITECTURE[training_spec.family_name]
            except KeyError as exc:
                raise ValueError(
                    f"ASR training family {training_spec.family_name!r} has no "
                    "registered source-data architecture.") from exc
        if (architecture is not None and
                ASRDataArchitecture.coerce(architecture) is not resolved_architecture):
            raise ValueError(
                f"{canonical_model_type!r} uses {resolved_architecture.value!r} "
                f"data, not {ASRDataArchitecture.coerce(architecture).value!r}.")
    elif architecture is None:
        raise ValueError("Pass either model_type or architecture.")
    else:
        resolved_architecture = ASRDataArchitecture.coerce(architecture)

    base = model_spec or _ARCHITECTURE_SPECS[resolved_architecture]
    if canonical_model_type is None:
        variants = base.variants
        readiness = None
    else:
        variants = base.variants
        if training_support == "inference-only":
            readiness = ASRDataReadiness.UNAVAILABLE
        elif base.readiness is not None:
            readiness = base.readiness
        elif any(not variant.preprocessed for variant in variants):
            readiness = ASRDataReadiness.INTEGRATED
        elif training_support == "custom":
            readiness = ASRDataReadiness.CUSTOM
        else:
            readiness = ASRDataReadiness.PREPROCESSED
    return replace(
        base,
        variants=variants,
        model_type=canonical_model_type,
        readiness=readiness,
        training_support=training_support,
    )


def list_asr_dataset_specs() -> tuple[ASRDatasetSpec, ...]:
    """Return one model-specific dataset contract for every ASR profile."""
    from voicehub.tasks import SpeechTask
    from voicehub.training.specs import list_training_specs

    return tuple(
        get_asr_dataset_spec(spec.model_type)
        for spec in list_training_specs(task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION, ))


__all__ = [
    "ASRDataArchitecture",
    "ASRDataReadiness",
    "ASRDatasetSpec",
    "ASRRecordVariant",
    "get_asr_dataset_spec",
    "list_asr_dataset_specs",
]
