"""Declarative, framework-lazy training profiles for audio runtime."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from threading import RLock
from types import MappingProxyType
from typing import Any

from voicehub.dependencies import normalize_import_path
from voicehub.errors import UnknownModelError
from voicehub.models.manifests import BuiltinModelManifest, discover_builtin_model_manifests
from voicehub.registry import get_model_spec, normalize_model_type
from voicehub.tasks import SpeechTask
from voicehub.training.contracts import (
    TrainingContext,
    TrainingPhaseKind,
    TrainingPhaseSpec,
    TrainingRecipeKind,
    TrainingSupport,
)


class TrainingFamily(str, Enum):
    """Built-in objective and optimization shapes.

    ``ModelTrainingSpec.family`` also accepts a non-empty string. This
    enum is therefore a convenience for the built-ins, not a closed
    extension point.
    """

    CAUSAL_LM = "causal-lm"
    SEQ2SEQ = "sequence-to-sequence"
    FLOW_MATCHING = "flow-matching"
    ACOUSTIC = "acoustic-regression"
    VITS = "vits"
    COMPOSITE = "composite"
    CTC = "ctc"
    SPEECH_SEQ2SEQ = "speech-sequence-to-sequence"
    RNNT = "rnnt"
    TDT = "tdt"
    AUDIO_CLASSIFICATION = "audio-classification"
    AUDIO_CODEC = "audio-codec"
    FRAME_CLASSIFICATION = "frame-classification"
    NATIVE_ASR_DISPATCH = "native-asr-dispatch"
    UPSTREAM_NATIVE = "upstream-native"

    def __str__(self) -> str:
        return self.value


def _strings(value: Iterable[str] | str, *, name: str) -> tuple[str, ...]:
    values = (value, ) if isinstance(value, str) else tuple(value)
    normalized = []
    for item in values:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{name} must contain non-empty strings.")
        normalized.append(item.strip())
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicate values.")
    return tuple(normalized)


def _pairs(value: Mapping[str, Any] | Iterable[tuple[str, Any]], *, name: str):
    items = tuple(value.items()) if isinstance(value, Mapping) else tuple(value)
    normalized = []
    seen = set()
    for item in items:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError(f"{name} entries must be two-item pairs.")
        key, item_value = item
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{name} keys must be non-empty strings.")
        key = key.strip()
        if key in seen:
            raise ValueError(f"{name} contains duplicate key {key!r}.")
        seen.add(key)
        normalized.append((key, item_value))
    return tuple(normalized)


@dataclass(frozen=True)
class ModelTrainingSpec:
    """Everything a training adapter needs to expose a source runtime.

    The original, single-objective fields remain supported. When
    ``phases`` is omitted they are projected into one phase named
    ``default``. New backends can instead declare multiple independently
    scheduled component phases.
    """

    model_type: str
    family: TrainingFamily | str
    module_paths: tuple[str, ...] = ("model", "model.model")
    component_paths: tuple[str, ...] = ()
    label_names: tuple[str, ...] = (
        "labels",
        "targets",
        "target",
    )
    prediction_keys: tuple[str, ...] = (
        "logits",
        "predictions",
        "audio_values",
        "waveform",
    )
    loss_keys: tuple[str, ...] = (
        "loss",
        "total_loss",
    )
    loss_weights: tuple[tuple[str, float], ...] = ()
    regression_loss: str = "mse"
    source_entrypoints: tuple[str, ...] = ()
    native_training: bool = False
    separate_optimizers: bool = False
    support: TrainingSupport = TrainingSupport.PREPROCESSED
    phases: tuple[TrainingPhaseSpec, ...] = ()
    default_phase: str | None = None
    fallback_objective: str | None = None
    recipe_kind: TrainingRecipeKind = TrainingRecipeKind.SINGLE_PHASE
    allow_module_discovery: bool = False
    training_default_model_name_or_path: str | None = None
    field_schemas: Mapping[str, Any] = field(default_factory=dict)
    task: SpeechTask | str = SpeechTask.TEXT_TO_SPEECH
    adapter_factory: str | None = None
    dataset_factory: str | None = None
    dataset_spec_factory: str | None = None
    tokenizer_paths: tuple[str, ...] = (
        "tokenizer",
        "model.tokenizer",
    )
    optimization_profile_factory: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model_type, str) or not self.model_type.strip():
            raise ValueError("model_type must be a non-empty string.")
        object.__setattr__(
            self,
            "model_type",
            normalize_model_type(self.model_type),
        )
        object.__setattr__(
            self,
            "task",
            SpeechTask.coerce(self.task),
        )

        for field_name in (
                "adapter_factory",
                "dataset_factory",
                "dataset_spec_factory",
                "optimization_profile_factory",
        ):
            import_path = getattr(self, field_name)
            if import_path is None:
                continue
            object.__setattr__(
                self,
                field_name,
                normalize_import_path(
                    import_path,
                    name=field_name,
                ),
            )

        family = self.family
        if isinstance(family, str) and not isinstance(family, TrainingFamily):
            family = family.strip().lower()
            if not family:
                raise ValueError("family must be a TrainingFamily or non-empty string.")
            try:
                family = TrainingFamily(family)
            except ValueError:
                pass
        elif not isinstance(family, TrainingFamily):
            raise TypeError("family must be a TrainingFamily or non-empty string.")
        object.__setattr__(self, "family", family)

        for field_name in (
                "module_paths",
                "component_paths",
                "label_names",
                "prediction_keys",
                "loss_keys",
                "source_entrypoints",
                "tokenizer_paths",
        ):
            object.__setattr__(
                self,
                field_name,
                _strings(getattr(self, field_name), name=field_name),
            )
        if not self.module_paths:
            raise ValueError("module_paths must contain at least one candidate path.")
        if any(any(not segment.isidentifier() for segment in path.split("."))
               for path in self.tokenizer_paths):
            raise ValueError("tokenizer_paths must contain dotted attribute paths.")

        weights = _pairs(self.loss_weights, name="loss_weights")
        normalized_weights = []
        for loss_name, weight in weights:
            if isinstance(weight, bool) or not isinstance(weight, (int, float)):
                raise TypeError("loss_weights values must be real numbers.")
            normalized_weights.append((loss_name, float(weight)))
        object.__setattr__(self, "loss_weights", tuple(normalized_weights))

        if self.regression_loss not in ("l1", "mse"):
            raise ValueError("regression_loss must be either 'l1' or 'mse'.")
        if not isinstance(self.native_training, bool):
            raise TypeError("native_training must be a boolean.")
        if not isinstance(self.separate_optimizers, bool):
            raise TypeError("separate_optimizers must be a boolean.")
        if not isinstance(self.allow_module_discovery, bool):
            raise TypeError("allow_module_discovery must be a boolean.")
        training_default = self.training_default_model_name_or_path
        if training_default is not None:
            if not isinstance(training_default, str) or not training_default.strip():
                raise ValueError(
                    "training_default_model_name_or_path must be a non-empty "
                    "string or None.")
            object.__setattr__(
                self,
                "training_default_model_name_or_path",
                training_default.strip(),
            )
        if not isinstance(self.field_schemas, Mapping):
            raise TypeError("field_schemas must be a mapping of dotted paths to schemas.")
        normalized_schemas = {}
        for path, schema in self.field_schemas.items():
            if not isinstance(path, str) or not path.strip():
                raise ValueError("field_schemas keys must be non-empty dotted paths.")
            if isinstance(schema, Mapping):
                schema = MappingProxyType(dict(schema))
            normalized_schemas[path.strip()] = schema
        object.__setattr__(
            self,
            "field_schemas",
            MappingProxyType(normalized_schemas),
        )
        object.__setattr__(
            self,
            "support",
            TrainingSupport.coerce(self.support),
        )
        object.__setattr__(
            self,
            "recipe_kind",
            TrainingRecipeKind.coerce(self.recipe_kind),
        )

        fallback = self.fallback_objective
        if fallback is None:
            fallback = self._legacy_fallback_objective()
        elif not isinstance(fallback, str) or not fallback.strip():
            raise ValueError("fallback_objective must be a non-empty string or None.")
        if fallback is not None:
            fallback = fallback.strip().lower().replace("-", "_")
        object.__setattr__(self, "fallback_objective", fallback)

        phases = tuple(self.phases)
        if any(not isinstance(phase, TrainingPhaseSpec) for phase in phases):
            raise TypeError("phases must contain TrainingPhaseSpec instances.")
        phase_names = [phase.name for phase in phases]
        if len(set(phase_names)) != len(phase_names):
            raise ValueError("Training phase names must be unique within a model profile.")

        default_phase = self.default_phase
        if default_phase is not None:
            if not isinstance(default_phase, str) or not default_phase.strip():
                raise ValueError("default_phase must be a non-empty phase name or None.")
            default_phase = default_phase.strip()
        if not phases:
            default_phase = default_phase or "default"
            phases = (
                TrainingPhaseSpec(
                    name=default_phase,
                    label_names=self.label_names,
                    prediction_keys=self.prediction_keys,
                    loss_keys=self.loss_keys,
                    loss_weights=self.loss_weights,
                    fallback_objective=fallback,
                ), )
        else:
            default_phase = default_phase or phases[0].name
            if default_phase not in phase_names:
                raise ValueError(f"default_phase {default_phase!r} is not present in phases.")
            if (len(phases) > 1 and self.recipe_kind is TrainingRecipeKind.SINGLE_PHASE):
                object.__setattr__(
                    self,
                    "recipe_kind",
                    TrainingRecipeKind.MULTI_PHASE,
                )
        object.__setattr__(self, "phases", phases)
        object.__setattr__(self, "default_phase", default_phase)
        self._validate_phase_schedule(phases)

    @staticmethod
    def _validate_phase_schedule(phases: tuple[TrainingPhaseSpec, ...]) -> None:
        """Reject schedules that leave the Trainer unable to advance a step."""
        period = 1
        for phase in phases:
            period = math.lcm(period, phase.frequency)
            if period > 100_000:
                raise ValueError(
                    "The combined training phase schedule is too large to "
                    "validate; reduce phase frequencies.")
        uncovered = [step for step in range(period) if not any(phase.is_scheduled(step) for phase in phases)]
        if uncovered:
            preview = ", ".join(str(step) for step in uncovered[:8])
            raise ValueError(
                "Training phases must cover every recipe step. "
                f"No phase is scheduled at period position(s): {preview}.")

    def _legacy_fallback_objective(self) -> str | None:
        if self.family is TrainingFamily.CAUSAL_LM:
            return "causal_cross_entropy"
        if self.family in (
                TrainingFamily.SEQ2SEQ,
                TrainingFamily.SPEECH_SEQ2SEQ,
        ):
            return "cross_entropy"
        if self.family is TrainingFamily.ACOUSTIC:
            return self.regression_loss
        if self.family in (
                TrainingFamily.AUDIO_CLASSIFICATION,
                TrainingFamily.FRAME_CLASSIFICATION,
        ):
            return "classification"
        if self.family is TrainingFamily.COMPOSITE:
            return "auto"
        # Flow, CTC, RNN-T, TDT, and source-native objectives cannot be
        # reconstructed safely from an arbitrary prediction/label pair.
        return None

    @property
    def family_name(self) -> str:
        return self.family.value if isinstance(self.family, TrainingFamily) else self.family

    @property
    def supports_training(self) -> bool:
        """Whether the built-in family adapter can be used directly."""
        return self.is_turnkey

    @property
    def has_training_recipe(self) -> bool:
        """Whether a generic or source-native recipe is represented."""
        return self.support.is_trainable

    @property
    def requires_custom_adapter(self) -> bool:
        return self.support is TrainingSupport.CUSTOM

    @property
    def is_turnkey(self) -> bool:
        return self.support in (
            TrainingSupport.NATIVE,
            TrainingSupport.PREPROCESSED,
        )

    @property
    def phase_map(self) -> Mapping[str, TrainingPhaseSpec]:
        return MappingProxyType({phase.name: phase for phase in self.phases})

    def get_phase(self, name: str | None = None) -> TrainingPhaseSpec:
        """Resolve a phase by name, defaulting to ``default_phase``."""
        selected = self.default_phase if name is None else name
        if not isinstance(selected, str):
            raise TypeError("Training phase names must be strings.")
        try:
            return self.phase_map[selected]
        except KeyError as exc:
            available = ", ".join(self.phase_map)
            raise ValueError(
                f"Unknown training phase {selected!r} for {self.model_type!r}. "
                f"Available phases: {available}.") from exc

    @property
    def install_extra(self) -> str:
        """Return the one dependency extra used by every training profile."""
        return "training"

    @property
    def dataset_spec(self):
        """Return this training profile's model-specific data contract.

        The optional ``dataset_spec_factory`` and task-specific contract
        modules are resolved lazily to keep the framework-free training
        registry free of import cycles. VAD continues to use
        :class:`SpeechDataset`.
        """
        if self.task is SpeechTask.TEXT_TO_SPEECH:
            from voicehub.training.datasets import get_tts_dataset_spec

            return get_tts_dataset_spec(self.model_type)
        if self.task is SpeechTask.AUTOMATIC_SPEECH_RECOGNITION:
            from voicehub.training.datasets import get_asr_dataset_spec

            return get_asr_dataset_spec(self.model_type)
        raise AttributeError(
            f"{self.model_type!r} is a {self.task.value} profile and has "
            "no architecture dataset spec.")


def training_spec_from_manifest(manifest: BuiltinModelManifest, ) -> ModelTrainingSpec:
    """Project one activated manifest into an honest inference-only profile."""
    if not isinstance(manifest, BuiltinModelManifest):
        raise TypeError("`manifest` must be a BuiltinModelManifest.")
    return ModelTrainingSpec(
        model_type=manifest.model_type,
        family=manifest.training_family,
        support=manifest.training_support,
        task=manifest.task,
    )


_TrainingSpecTuple = tuple[ModelTrainingSpec, ...]


def discover_manifest_training_specs(models_root: str | Path | None = None, ) -> _TrainingSpecTuple:
    """Discover inference-only profiles from activated package manifests."""
    return tuple(
        training_spec_from_manifest(manifest) for manifest in discover_builtin_model_manifests(models_root))


_COMMON_LM_PATHS = (
    "model",
    "model.model",
    "model.llm",
    "model.gpt",
    "model.language_model",
    "model.generator",
)
_COMMON_ACOUSTIC_PATHS = (
    "model",
    "model.model",
    "model.tts_model",
    "model.generator",
    "model.synthesizer",
    "model.decoder",
)
_COMMON_COMPOSITE_PATHS = (
    "model",
    "model.model",
    "model.tts_model",
    "model.generator",
    "model.llm",
)
_MELOTTS_REQUIRED_INPUTS = (
    "input_ids",
    "input_lengths",
    "tone_ids",
    "language_ids",
    "bert_features",
    "ja_bert_features",
    "spectrogram",
    "spectrogram_lengths",
    "audio_values",
    "audio_lengths",
    "speaker_ids",
)


def _phase(name: str, **kwargs) -> TrainingPhaseSpec:
    return TrainingPhaseSpec(name=name, **kwargs)


def _profile(
    model_type: str,
    family: TrainingFamily,
    *,
    task: SpeechTask | str = SpeechTask.TEXT_TO_SPEECH,
    module_paths: tuple[str, ...] | None = None,
    component_paths: tuple[str, ...] = (),
    label_names: tuple[str, ...] | None = None,
    loss_weights: tuple[tuple[str, float], ...] = (),
    regression_loss: str = "mse",
    source_entrypoints: tuple[str, ...] = (),
    adapter_factory: str | None = None,
    dataset_factory: str | None = None,
    dataset_spec_factory: str | None = None,
    tokenizer_paths: tuple[str, ...] = ModelTrainingSpec.tokenizer_paths,
    optimization_profile_factory: str | None = None,
    native_training: bool = False,
    separate_optimizers: bool | None = None,
    support: TrainingSupport = TrainingSupport.PREPROCESSED,
    phases: tuple[TrainingPhaseSpec, ...] = (),
    default_phase: str | None = None,
    fallback_objective: str | None = None,
    recipe_kind: TrainingRecipeKind = TrainingRecipeKind.SINGLE_PHASE,
    allow_module_discovery: bool = False,
    training_default_model_name_or_path: str | None = None,
    field_schemas: Mapping[str, Any] | None = None,
) -> ModelTrainingSpec:
    if module_paths is None:
        module_paths = (
            _COMMON_LM_PATHS if family is TrainingFamily.CAUSAL_LM else
            _COMMON_COMPOSITE_PATHS if family is TrainingFamily.COMPOSITE else _COMMON_ACOUSTIC_PATHS)
    return ModelTrainingSpec(
        model_type=model_type,
        family=family,
        task=task,
        module_paths=module_paths,
        component_paths=component_paths,
        label_names=label_names or ModelTrainingSpec.label_names,
        loss_weights=loss_weights,
        regression_loss=regression_loss,
        source_entrypoints=source_entrypoints,
        adapter_factory=adapter_factory,
        dataset_factory=dataset_factory,
        dataset_spec_factory=dataset_spec_factory,
        tokenizer_paths=tokenizer_paths,
        optimization_profile_factory=optimization_profile_factory,
        native_training=native_training,
        separate_optimizers=(
            family in (TrainingFamily.COMPOSITE,
                       TrainingFamily.VITS) if separate_optimizers is None else separate_optimizers),
        support=support,
        phases=phases,
        default_phase=default_phase,
        fallback_objective=fallback_objective,
        recipe_kind=recipe_kind,
        allow_module_discovery=allow_module_discovery,
        training_default_model_name_or_path=(training_default_model_name_or_path),
        field_schemas=field_schemas or {},
    )


_BUILTIN_TRAINING_SPECS = (
    _profile(
        "orpheustts",
        TrainingFamily.CAUSAL_LM,
        dataset_spec_factory="voicehub.training.data_contracts:build_orpheustts_dataset_spec",
        adapter_factory="voicehub.models.orpheustts.training:OrpheusTrainingAdapter",
        module_paths=("model", ),
        component_paths=("model", ),
        source_entrypoints=("voicehub.models.causal_lm.native.modeling:"
                            "CausalLMForCausalLM.forward", ),
        dataset_factory=("voicehub.models.orpheustts.training:"
                         "build_training_dataset"),
        native_training=True,
        support=TrainingSupport.NATIVE,
        phases=(
            _phase(
                "codec_language_model",
                component_paths=("model", ),
                optimizer_names=("model", ),
                label_names=("labels", ),
                prediction_keys=("logits", ),
                loss_keys=("loss", ),
                fallback_objective="causal_cross_entropy",
            ), ),
        default_phase="codec_language_model",
    ),
    _profile(
        "dia",
        TrainingFamily.SEQ2SEQ,
        dataset_spec_factory="voicehub.training.data_contracts:build_dia_dataset_spec",
        adapter_factory="voicehub.models.dia.training:DiaTrainingAdapter",
        module_paths=("model", ),
        component_paths=("model", ),
        source_entrypoints=("voicehub.models.dia.native.modeling:"
                            "DiaForConditionalGeneration.forward", ),
        native_training=True,
        support=TrainingSupport.NATIVE,
        phases=(
            _phase(
                "codec_language_model",
                component_paths=("model", ),
                optimizer_names=("model", ),
                label_names=("labels", ),
                prediction_keys=("logits", ),
                loss_keys=("loss", ),
                required_inputs=(
                    "input_ids",
                    "attention_mask",
                    "decoder_input_ids",
                    "decoder_attention_mask",
                    "labels",
                ),
            ), ),
        default_phase="codec_language_model",
    ),
    _profile(
        "vui",
        TrainingFamily.CAUSAL_LM,
        dataset_spec_factory="voicehub.training.data_contracts:build_vui_dataset_spec",
        adapter_factory="voicehub.models.vui.training:VuiTrainingAdapter",
        module_paths=("model", ),
        component_paths=("model", ),
        source_entrypoints=("voicehub.models.vui.training.VuiTrainingAdapter", ),
        native_training=True,
        support=TrainingSupport.PREPROCESSED,
        phases=(
            _phase(
                "codec_language_model",
                component_paths=("model", ),
                optimizer_names=("model", ),
                label_names=("audio_codes", ),
                prediction_keys=("logits", ),
                loss_keys=("loss", "codec_ce_loss"),
                required_inputs=("input_ids", "audio_codes"),
            ), ),
        default_phase="codec_language_model",
        field_schemas={
            "input_ids": {
                "sequence_dim": -1,
                "padding_value": 0,
                "mask_field": "text_attention_mask",
            },
            "audio_codes": {
                "sequence_dim": -1,
                "padding_value": 0,
                "length_field": "audio_code_lengths",
            },
        },
    ),
    _profile(
        "chatterbox",
        TrainingFamily.COMPOSITE,
        dataset_spec_factory="voicehub.training.data_contracts:build_chatterbox_dataset_spec",
        adapter_factory="voicehub.models.chatterbox.training:ChatterboxTrainingAdapter",
        component_paths=("model.t3", "model.s3gen.flow"),
        source_entrypoints=("voicehub.models.chatterbox.training:"
                            "ChatterboxTrainingAdapter", ),
        native_training=True,
        separate_optimizers=True,
        support=TrainingSupport.CUSTOM,
        phases=(
            _phase(
                "language_model",
                component_paths=("model.t3", ),
                optimizer_names=("language_model", ),
                loss_keys=("loss", "text_loss", "speech_token_loss"),
            ),
            _phase(
                "flow",
                component_paths=("model.s3gen.flow", ),
                optimizer_names=("flow", ),
                label_names=("labels", "mel_spec", "mel_labels", "target"),
                loss_keys=("loss", "flow_loss", "diffusion_loss"),
            ),
        ),
        default_phase="language_model",
        field_schemas={
            "audio": {
                "sequence_dim": -1,
                "padding_value": 0.0,
                "length_field": "audio_lengths",
            },
            "text_tokens": {
                "sequence_dim": 0,
                "padding_value": 0,
                "length_field": "text_token_lens",
            },
            "speech_tokens": {
                "sequence_dim": 0,
                "padding_value": 0,
                "length_field": "speech_token_lens",
            },
            "prompt_tokens": {
                "sequence_dim": 0,
                "padding_value": 0,
                "length_field": "prompt_lens",
                "allow_missing": True,
            },
            "speech_token": {
                "sequence_dim": 0,
                "padding_value": 0,
                "length_field": "speech_token_len",
            },
            "speech_feat": {
                "sequence_dim": -1,
                "padding_value": 0.0,
                "length_field": "speech_feat_len",
            },
        },
    ),
    _profile(
        "kokoro",
        TrainingFamily.ACOUSTIC,
        dataset_spec_factory="voicehub.training.data_contracts:build_kokoro_dataset_spec",
        adapter_factory="voicehub.models.kokoro.training:KokoroTrainingAdapter",
        module_paths=("training_model", ),
        component_paths=(
            "model.bert",
            "model.bert_encoder",
            "model.predictor",
            "model.text_encoder",
            "model.decoder",
        ),
        source_entrypoints=("voicehub.models.kokoro.training:KokoroTrainingAdapter", ),
        native_training=True,
        separate_optimizers=False,
        support=TrainingSupport.PREPROCESSED,
        phases=(
            _phase(
                "duration",
                component_paths=(
                    "model.bert",
                    "model.bert_encoder",
                    "model.predictor",
                ),
                optimizer_names=("model", ),
                forward_component="training_model",
                forward_method="duration_objective",
                label_names=("durations", ),
                prediction_keys=("logits", ),
                loss_keys=("loss", ),
                required_inputs=(
                    "input_ids",
                    "ref_s",
                    "durations",
                ),
                frequency=2,
                offset=0,
            ),
            _phase(
                "acoustic",
                component_paths=(
                    "model.bert",
                    "model.bert_encoder",
                    "model.predictor",
                    "model.text_encoder",
                    "model.decoder",
                ),
                optimizer_names=("model", ),
                forward_component="training_model",
                forward_method="acoustic_objective",
                label_names=("audio_values", ),
                prediction_keys=("logits", "waveform"),
                loss_keys=("loss", ),
                required_inputs=(
                    "input_ids",
                    "ref_s",
                    "durations",
                    "audio_values",
                ),
                frequency=2,
                offset=1,
            ),
        ),
        default_phase="acoustic",
        recipe_kind=TrainingRecipeKind.MULTI_PHASE,
    ),
    _profile(
        "echo",
        TrainingFamily.FLOW_MATCHING,
        dataset_spec_factory="voicehub.training.data_contracts:build_echo_dataset_spec",
        adapter_factory="voicehub.models.echo.training:EchoTrainingAdapter",
        module_paths=("model", ),
        component_paths=("model", ),
        source_entrypoints=("voicehub.models.echo.training.EchoTrainingAdapter", ),
        support=TrainingSupport.PREPROCESSED,
        phases=(
            _phase(
                "flow",
                component_paths=("model", ),
                optimizer_names=("model", ),
                label_names=("target_latents", ),
                prediction_keys=("logits", ),
                loss_keys=("loss", "flow_loss"),
                required_inputs=(
                    "target_latents",
                    "text_input_ids",
                    "text_mask",
                    "speaker_latents",
                    "speaker_mask",
                ),
            ), ),
        default_phase="flow",
    ),
    _profile(
        "conversationtts",
        TrainingFamily.CAUSAL_LM,
        dataset_spec_factory=("voicehub.training.data_contracts:build_conversationtts_dataset_spec"),
        adapter_factory=("voicehub.models.conversationtts.training:"
                         "ConversationTTSTrainingAdapter"),
        module_paths=("model", ),
        component_paths=("model", ),
        source_entrypoints=(
            "voicehub.models.conversationtts.native.modeling:"
            "ConversationTTSModel.forward",
            "voicehub.models.conversationtts.native.processing:"
            "build_conversationtts_sequence",
            "voicehub.models.conversationtts.training:"
            "ConversationTTSTrainingAdapter",
        ),
        optimization_profile_factory=("voicehub.training.tts_optimization:LLMTTSOptimizationConfig"),
        native_training=True,
        support=TrainingSupport.NATIVE,
        training_default_model_name_or_path=("AudioFoundation/SpeechFoundation"),
        phases=(
            _phase(
                "codec_language_model",
                component_paths=("model", ),
                optimizer_names=("model", ),
                label_names=("labels", ),
                prediction_keys=("logits", ),
                loss_keys=(
                    "loss",
                    "codebook0_loss",
                    "residual_loss",
                ),
                required_inputs=(
                    "tokens",
                    "labels",
                    "tokens_mask",
                ),
            ), ),
        default_phase="codec_language_model",
        field_schemas={
            "text_token_ids": {
                "sequence_dim": -1,
                "padding_value": 128_002,
                "length_field": "text_token_lengths",
            },
            "audio_codes": {
                "sequence_dim": -1,
                "padding_value": 2_050,
                "length_field": "audio_code_lengths",
            },
            "audio": {
                "sequence_dim": -1,
                "padding_value": 0.0,
                "length_field": "audio_lengths",
            },
            "audio.array": {
                "sequence_dim": -1,
                "padding_value": 0.0,
                "length_field": "audio_lengths",
            },
            "audio.waveform": {
                "sequence_dim": -1,
                "padding_value": 0.0,
                "length_field": "audio_lengths",
            },
            "audio.audio": {
                "sequence_dim": -1,
                "padding_value": 0.0,
                "length_field": "audio_lengths",
            },
            "audio.input_values": {
                "sequence_dim": -1,
                "padding_value": 0.0,
                "length_field": "audio_lengths",
            },
            "audio_values": {
                "sequence_dim": -1,
                "padding_value": 0.0,
                "length_field": "audio_lengths",
            },
        },
    ),
    _profile(
        "llasa",
        TrainingFamily.CAUSAL_LM,
        dataset_spec_factory="voicehub.training.data_contracts:build_llasa_dataset_spec",
        adapter_factory="voicehub.models.llasa.training:LlasaTrainingAdapter",
        module_paths=("model", ),
        component_paths=("model", ),
        label_names=("labels", ),
        source_entrypoints=(
            "voicehub.models.causal_lm.native.modeling:"
            "CausalLMForCausalLM.forward",
            "voicehub.models.llasa.training:LlasaSFTDataset",
            "voicehub.models.llasa.training:LlasaTrainingAdapter",
        ),
        dataset_factory=("voicehub.models.llasa.training:"
                         "build_training_dataset"),
        native_training=True,
        support=TrainingSupport.NATIVE,
        phases=(
            _phase(
                "codec_language_model",
                component_paths=("model", ),
                optimizer_names=("model", ),
                label_names=("labels", ),
                prediction_keys=("logits", ),
                loss_keys=("loss", ),
                required_inputs=(
                    "input_ids",
                    "attention_mask",
                    "labels",
                ),
                frozen_component_paths=("codec", ),
            ), ),
        default_phase="codec_language_model",
        field_schemas={
            "input_ids": {
                "sequence_dim": -1,
                "padding_value": 128_009,
                "mask_field": "attention_mask",
            },
            "attention_mask": {
                "sequence_dim": -1,
                "padding_value": 0,
            },
            "labels": {
                "sequence_dim": -1,
                "padding_value": -100,
            },
        },
    ),
    _profile(
        "cosyvoice",
        TrainingFamily.COMPOSITE,
        dataset_spec_factory="voicehub.training.data_contracts:build_cosyvoice_dataset_spec",
        adapter_factory="voicehub.models.cosyvoice.training:CosyVoiceTrainingAdapter",
        module_paths=("model", ),
        component_paths=(
            "model.llm",
            "model.flow",
            "model.hift",
            "model.hifigan.discriminator",
        ),
        source_entrypoints=(
            "voicehub.models.cosyvoice.native.modeling:"
            "CosyVoiceNativeModel.forward",
            "voicehub.models.cosyvoice.training_cosyvoice:"
            "CosyVoiceTrainingAdapter",
        ),
        native_training=True,
        support=TrainingSupport.CUSTOM,
        phases=(
            _phase(
                "llm",
                component_paths=("model.llm", ),
                optimizer_names=("llm", ),
                loss_keys=("language_model_loss", ),
            ),
            _phase(
                "flow",
                component_paths=("model.flow", ),
                optimizer_names=("flow", ),
                label_names=("speech_features", ),
                loss_keys=("flow_matching_loss", ),
            ),
            _phase(
                "hifigan_generator",
                component_paths=("model.hift", ),
                optimizer_names=("hifigan_generator", ),
                label_names=("waveform", "pitch"),
                loss_keys=(
                    "adversarial_loss",
                    "feature_matching_loss",
                    "pitch_loss",
                    "spectral_reconstruction_loss",
                ),
                kind=TrainingPhaseKind.GENERATOR,
                frozen_component_paths=("model.hifigan.discriminator", ),
            ),
            _phase(
                "hifigan_discriminator",
                component_paths=("model.hifigan.discriminator", ),
                optimizer_names=("hifigan_discriminator", ),
                label_names=("waveform", ),
                loss_keys=("discriminator_loss", ),
                kind=TrainingPhaseKind.DISCRIMINATOR,
                frozen_component_paths=("model.hift", ),
            ),
        ),
        default_phase="llm",
        recipe_kind=TrainingRecipeKind.ADVERSARIAL,
    ),
    _profile(
        "f5tts",
        TrainingFamily.FLOW_MATCHING,
        dataset_spec_factory="voicehub.training.data_contracts:build_f5tts_dataset_spec",
        adapter_factory="voicehub.models.f5tts.training:F5TTSTrainingAdapter",
        module_paths=("model.ema_model", "model.model", "model"),
        component_paths=("model.ema_model", ),
        label_names=("labels", "mel_spec", "mel_labels", "target"),
        source_entrypoints=("f5_tts/train/train.py", ),
        optimization_profile_factory=("voicehub.training.tts_optimization:DiffusionTTSOptimizationConfig"),
        native_training=True,
        support=TrainingSupport.PREPROCESSED,
        phases=(
            _phase(
                "flow",
                component_paths=("model.ema_model", ),
                optimizer_names=("model", ),
                label_names=("mel", "mel_spec", "input_values"),
                prediction_keys=("predictions", ),
                loss_keys=("loss", ),
                required_inputs=("inp", "text"),
            ), ),
    ),
    _profile(
        "gptsovits",
        TrainingFamily.COMPOSITE,
        dataset_spec_factory="voicehub.training.data_contracts:build_gptsovits_dataset_spec",
        adapter_factory="voicehub.models.gptsovits.training:GPTSoVITSTrainingAdapter",
        module_paths=("training_model", ),
        component_paths=(
            "training_model.s1",
            "training_model.s2.generator",
            "training_model.s2.discriminator",
        ),
        source_entrypoints=(
            "voicehub.models.gptsovits.native.training:"
            "GPTSoVITSStagedTrainingModel",
            "voicehub.models.gptsovits.training:"
            "GPTSoVITSTrainingAdapter",
        ),
        native_training=True,
        separate_optimizers=True,
        support=TrainingSupport.PREPROCESSED,
        phases=(
            _phase(
                "s1",
                component_paths=("training_model.s1", ),
                optimizer_names=("s1", ),
                forward_component="training_model",
                forward_method="s1_objective",
                loss_keys=("loss", ),
                required_inputs=(
                    "phoneme_ids",
                    "phoneme_lengths",
                    "semantic_ids",
                    "semantic_lengths",
                    "bert_features",
                ),
            ),
            _phase(
                "s2_generator",
                component_paths=("training_model.s2.generator", ),
                optimizer_names=("s2_generator", ),
                forward_component="training_model",
                forward_method="s2_generator_objective",
                loss_keys=("loss", ),
                required_inputs=(
                    "ssl_features",
                    "spectrogram",
                    "spectrogram_lengths",
                    "audio_values",
                    "phoneme_ids",
                    "phoneme_lengths",
                ),
                kind=TrainingPhaseKind.GENERATOR,
                frozen_component_paths=("training_model.s2.discriminator", ),
            ),
            _phase(
                "s2_discriminator",
                component_paths=("training_model.s2.discriminator", ),
                optimizer_names=("s2_discriminator", ),
                forward_component="training_model",
                forward_method="s2_discriminator_objective",
                loss_keys=("loss", ),
                required_inputs=(
                    "ssl_features",
                    "spectrogram",
                    "spectrogram_lengths",
                    "audio_values",
                    "phoneme_ids",
                    "phoneme_lengths",
                ),
                kind=TrainingPhaseKind.DISCRIMINATOR,
                frozen_component_paths=("training_model.s2.generator", ),
            ),
        ),
        default_phase="s1",
        recipe_kind=TrainingRecipeKind.ADVERSARIAL,
    ),
    _profile(
        "melotts",
        TrainingFamily.VITS,
        dataset_spec_factory="voicehub.training.data_contracts:build_melotts_dataset_spec",
        adapter_factory="voicehub.models.melotts.training:MeloTTSTrainingAdapter",
        module_paths=("training_model", ),
        component_paths=(
            "training_model.model",
            "training_model.mpd",
            "training_model.duration_discriminator",
        ),
        source_entrypoints=(
            "voicehub.models.melotts.native.training:"
            "MeloTTSTrainingModel",
            "voicehub.models.melotts.training:"
            "MeloTTSTrainingAdapter",
        ),
        native_training=True,
        separate_optimizers=True,
        support=TrainingSupport.PREPROCESSED,
        phases=(
            _phase(
                "generator",
                component_paths=("training_model.model", ),
                optimizer_names=("generator", ),
                forward_component="training_model",
                loss_keys=("loss", ),
                required_inputs=_MELOTTS_REQUIRED_INPUTS,
                kind=TrainingPhaseKind.GENERATOR,
                frozen_component_paths=(
                    "training_model.mpd",
                    "training_model.duration_discriminator",
                ),
            ),
            _phase(
                "discriminator",
                component_paths=("training_model.mpd", ),
                optimizer_names=("discriminator", ),
                forward_component="training_model",
                loss_keys=("loss", ),
                required_inputs=_MELOTTS_REQUIRED_INPUTS,
                kind=TrainingPhaseKind.DISCRIMINATOR,
                frozen_component_paths=(
                    "training_model.model",
                    "training_model.duration_discriminator",
                ),
            ),
            _phase(
                "duration_discriminator",
                component_paths=("training_model.duration_discriminator", ),
                optimizer_names=("duration_discriminator", ),
                forward_component="training_model",
                loss_keys=("loss", ),
                required_inputs=_MELOTTS_REQUIRED_INPUTS,
                kind=TrainingPhaseKind.DURATION_DISCRIMINATOR,
                frozen_component_paths=(
                    "training_model.model",
                    "training_model.mpd",
                ),
            ),
        ),
        default_phase="generator",
        recipe_kind=TrainingRecipeKind.ADVERSARIAL,
        field_schemas={
            "input_ids": {
                "sequence_dim": -1,
                "padding_value": 0,
                "length_field": "input_lengths",
            },
            "tone_ids": {
                "sequence_dim": -1,
                "padding_value": 0,
            },
            "language_ids": {
                "sequence_dim": -1,
                "padding_value": 0,
            },
            "bert_features": {
                "sequence_dim": -1,
                "padding_value": 0.0,
            },
            "ja_bert_features": {
                "sequence_dim": -1,
                "padding_value": 0.0,
            },
            "spectrogram": {
                "sequence_dim": -1,
                "padding_value": 0.0,
                "length_field": "spectrogram_lengths",
            },
            "audio_values": {
                "sequence_dim": -1,
                "padding_value": 0.0,
                "length_field": "audio_lengths",
            },
        },
    ),
    _profile(
        "openvoice",
        TrainingFamily.VITS,
        dataset_spec_factory="voicehub.training.data_contracts:build_openvoice_dataset_spec",
        adapter_factory="voicehub.models.openvoice.training:OpenVoiceTrainingAdapter",
        module_paths=("model", ),
        component_paths=(
            "model.enc_q",
            "model.flow",
            "model.dec",
            "model.ref_enc",
        ),
        regression_loss="l1",
        source_entrypoints=(
            "voicehub.models.openvoice.native.modeling:"
            "OpenVoiceToneColorConverter.forward",
            "voicehub.models.openvoice.training:"
            "OpenVoiceTrainingAdapter",
        ),
        native_training=True,
        separate_optimizers=False,
        support=TrainingSupport.CUSTOM,
        phases=(
            _phase(
                "generator",
                component_paths=(
                    "model.enc_q",
                    "model.flow",
                    "model.dec",
                    "model.ref_enc",
                ),
                optimizer_names=("generator", ),
                label_names=("target_waveform", ),
                prediction_keys=("waveform", ),
                loss_keys=("loss", ),
                required_inputs=(
                    "source_spectrogram",
                    "source_lengths",
                    "target_waveform",
                    "target_lengths",
                ),
                kind=TrainingPhaseKind.GENERATOR,
            ), ),
        default_phase="generator",
        recipe_kind=TrainingRecipeKind.SINGLE_PHASE,
    ),
    _profile(
        "outetts",
        TrainingFamily.CAUSAL_LM,
        dataset_spec_factory="voicehub.training.data_contracts:build_outetts_dataset_spec",
        adapter_factory="voicehub.models.outetts.training:OuteTTSTrainingAdapter",
        module_paths=("model.language_model", ),
        component_paths=("model.language_model", ),
        label_names=("labels", ),
        source_entrypoints=(
            "voicehub.models.causal_lm.native.modeling:"
            "CausalLMForCausalLM.forward",
            "voicehub.models.outetts.training:OuteTTSSFTDataset",
        ),
        dataset_factory=("voicehub.models.outetts.training:"
                         "build_training_dataset"),
        tokenizer_paths=(
            "tokenizer",
            "model.tokenizer",
            "model.prompt_processor.tokenizer",
        ),
        native_training=True,
        support=TrainingSupport.PREPROCESSED,
        phases=(
            _phase(
                "codec_language_model",
                component_paths=("model.language_model", ),
                optimizer_names=("codec_language_model", ),
                label_names=("labels", ),
                prediction_keys=("logits", ),
                loss_keys=("loss", ),
                required_inputs=("input_ids", "labels"),
            ), ),
        default_phase="codec_language_model",
    ),
    _profile(
        "parlertts",
        TrainingFamily.SEQ2SEQ,
        dataset_spec_factory="voicehub.training.data_contracts:build_parlertts_dataset_spec",
        adapter_factory="voicehub.models.parlertts.training:ParlerTTSTrainingAdapter",
        module_paths=("model", ),
        component_paths=(
            "model.decoder",
            "model.embed_prompts",
        ),
        label_names=(
            "labels",
            "audio_codes",
            "audio_values",
        ),
        field_schemas={
            "audio_codes": {
                "sequence_dim": -1,
                "padding_value": 0,
                "length_field": "audio_code_lengths",
            },
            "audio_values": {
                "sequence_dim": -1,
                "padding_value": 0.0,
                "length_field": "audio_lengths",
            },
        },
        source_entrypoints=(
            "voicehub.models.parlertts.native.modeling:"
            "ParlerTTSForConditionalGeneration.forward", ),
        native_training=True,
        support=TrainingSupport.NATIVE,
        training_default_model_name_or_path=("parler-tts/parler-tts-mini-v1"),
    ),
    _profile(
        "styletts2",
        TrainingFamily.VITS,
        dataset_spec_factory="voicehub.training.data_contracts:build_styletts2_dataset_spec",
        adapter_factory="voicehub.models.styletts2.training:StyleTTS2TrainingAdapter",
        module_paths=("training_model", ),
        component_paths=(
            "training_model.model",
            "training_model.mpd",
            "training_model.msd",
        ),
        source_entrypoints=(
            "voicehub.models.styletts2.native.training:"
            "StyleTTS2TrainingModel.forward",
            "voicehub.models.styletts2.training:"
            "StyleTTS2TrainingAdapter",
        ),
        native_training=True,
        support=TrainingSupport.PREPROCESSED,
        phases=(
            _phase(
                "generator",
                component_paths=("training_model.model", ),
                optimizer_names=("generator", ),
                loss_keys=("loss", ),
                required_inputs=(
                    "input_ids",
                    "input_lengths",
                    "alignments",
                    "alignment_lengths",
                    "normalized_mel",
                    "normalized_mel_lengths",
                    "reference_mel",
                    "reference_mel_lengths",
                    "f0_targets",
                    "noise_targets",
                    "audio_values",
                    "audio_lengths",
                ),
                kind=TrainingPhaseKind.GENERATOR,
                frozen_component_paths=(
                    "training_model.mpd",
                    "training_model.msd",
                ),
            ),
            _phase(
                "discriminator",
                component_paths=(
                    "training_model.mpd",
                    "training_model.msd",
                ),
                optimizer_names=("discriminator", ),
                loss_keys=("loss", ),
                required_inputs=(
                    "input_ids",
                    "input_lengths",
                    "alignments",
                    "alignment_lengths",
                    "normalized_mel",
                    "normalized_mel_lengths",
                    "reference_mel",
                    "reference_mel_lengths",
                    "f0_targets",
                    "noise_targets",
                    "audio_values",
                    "audio_lengths",
                ),
                kind=TrainingPhaseKind.DISCRIMINATOR,
                frozen_component_paths=("training_model.model", ),
            ),
        ),
        default_phase="generator",
        recipe_kind=TrainingRecipeKind.ADVERSARIAL,
    ),
    _profile(
        "mosstts",
        TrainingFamily.CAUSAL_LM,
        dataset_spec_factory="voicehub.training.data_contracts:build_mosstts_dataset_spec",
        adapter_factory="voicehub.models.mosstts.native.training:NativeMossTTSTrainingAdapter",
        module_paths=("model", ),
        component_paths=("model", ),
        label_names=("labels", ),
        source_entrypoints=(
            "voicehub.models.mosstts.native.modeling:"
            "MossDelayModel.forward",
            "voicehub.models.mosstts.native.modeling:"
            "MossOldLocalModel.forward",
            "voicehub.models.mosstts.native.modeling:"
            "MossLocalV15Model.forward",
            "voicehub.models.mosstts.native.modeling:"
            "MossRealtimeModel.forward",
            "voicehub.models.mosstts.native.processing:"
            "MossTTSProcessor.build_training_record",
            "voicehub.models.mosstts.native.runtime:"
            "MossTTSRuntime.prepare_training_batch",
            "voicehub.models.mosstts.native.training:"
            "NativeMossTTSTrainingAdapter",
        ),
        native_training=True,
        support=TrainingSupport.NATIVE,
        phases=(
            _phase(
                "semantic_language_model",
                component_paths=("model", ),
                optimizer_names=("model", ),
                label_names=("labels", ),
                prediction_keys=("logits", ),
                loss_keys=("loss", ),
                required_inputs=(
                    "input_ids",
                    "attention_mask",
                    "labels",
                ),
                frozen_component_paths=("training_backend.codec", ),
            ), ),
        default_phase="semantic_language_model",
        field_schemas={
            "input_ids": {
                "sequence_dim": -2,
                "padding_value": 0,
                "mask_field": "attention_mask",
            },
            "attention_mask": {
                "sequence_dim": -1,
                "padding_value": False,
            },
            "labels": {
                "sequence_dim": -2,
                "padding_value": -100,
            },
            "audio": {
                "sequence_dim": -1,
                "padding_value": 0.0,
                "length_field": "audio_lengths",
                "allow_missing": True,
            },
            "waveform": {
                "sequence_dim": -1,
                "padding_value": 0.0,
                "length_field": "audio_lengths",
                "allow_missing": True,
            },
            "reference_audio": {
                "sequence_dim": -1,
                "padding_value": 0.0,
                "length_field": "reference_audio_lengths",
                "allow_missing": True,
            },
        },
    ),
    _profile(
        "qwen3tts",
        TrainingFamily.CAUSAL_LM,
        dataset_spec_factory="voicehub.training.data_contracts:build_qwen3tts_dataset_spec",
        adapter_factory=("voicehub.models.qwen3tts.training_adapter:"
                         "Qwen3TTSTrainingAdapter"),
        module_paths=("model.model", ),
        component_paths=("model.model.talker", ),
        support=TrainingSupport.PREPROCESSED,
        native_training=True,
        training_default_model_name_or_path=("Qwen/Qwen3-TTS-12Hz-1.7B-Base"),
        source_entrypoints=(
            "voicehub.models.qwen3tts.native.modeling:"
            "Qwen3TTSForConditionalGeneration.forward",
            "voicehub.models.qwen3tts.training:Qwen3TTSSFTDataset",
        ),
        optimization_profile_factory=("voicehub.training.tts_optimization:LLMTTSOptimizationConfig.qwen3tts"),
        phases=(
            _phase(
                "codec_language_model",
                component_paths=("model.model.talker", ),
                optimizer_names=("model", ),
                label_names=("codec_0_labels", ),
                prediction_keys=("logits", ),
                loss_keys=(
                    "loss",
                    "talker_loss",
                    "sub_talker_loss",
                ),
                required_inputs=(
                    "input_ids",
                    "codec_ids",
                    "ref_mels",
                    "text_embedding_mask",
                    "codec_embedding_mask",
                    "attention_mask",
                    "codec_0_labels",
                    "codec_mask",
                ),
            ), ),
    ),
    _profile(
        "irodoritts",
        TrainingFamily.FLOW_MATCHING,
        dataset_spec_factory="voicehub.training.data_contracts:build_irodoritts_dataset_spec",
        adapter_factory="voicehub.models.irodoritts.training:NativeIrodoriTrainingAdapter",
        module_paths=("model.model", ),
        component_paths=("model.model", ),
        label_names=("target_latent", "duration_target"),
        source_entrypoints=(
            "voicehub.models.irodoritts.native.modeling:"
            "TextToLatentRFDiT.forward",
            "voicehub.models.irodoritts.native.training:"
            "irodori_training_step",
            "voicehub.models.irodoritts.training:"
            "NativeIrodoriTrainingAdapter",
        ),
        native_training=True,
        support=TrainingSupport.NATIVE,
        training_default_model_name_or_path="Aratako/Irodori-TTS-500M-v3",
        phases=(
            _phase(
                "flow",
                component_paths=("model.model", ),
                optimizer_names=("model", ),
                label_names=("target_latent", "duration_target"),
                prediction_keys=("velocity", "duration_prediction"),
                loss_keys=("loss", "flow_loss", "duration_loss"),
            ), ),
    ),
    _profile(
        "zonos",
        TrainingFamily.CAUSAL_LM,
        dataset_spec_factory="voicehub.training.data_contracts:build_zonos_dataset_spec",
        adapter_factory="voicehub.models.zonos.training:ZonosTrainingAdapter",
        module_paths=("model", ),
        component_paths=("model", ),
        label_names=("audio_codes", ),
        source_entrypoints=("voicehub.models.zonos.training.ZonosTrainingAdapter", ),
        native_training=True,
        support=TrainingSupport.PREPROCESSED,
        training_default_model_name_or_path="Zyphra/Zonos-v0.1-transformer",
        phases=(
            _phase(
                "reconstructed_codec_language_model",
                component_paths=("model", ),
                optimizer_names=("model", ),
                label_names=("audio_codes", ),
                prediction_keys=("logits", ),
                loss_keys=("loss", "codec_ce_loss"),
                required_inputs=("prefix_conditioning", "audio_codes"),
            ), ),
        default_phase="reconstructed_codec_language_model",
        field_schemas={
            "audio_codes": {
                "sequence_dim": -1,
                "padding_value": 1025,
                "length_field": "audio_code_lengths",
            },
            "prefix_conditioning": {
                "sequence_dim": 0,
                "padding_value": 0.0,
            },
            "audio": {
                "sequence_dim": -1,
                "padding_value": 0.0,
                "length_field": "audio_lengths",
            },
            "audio.array": {
                "sequence_dim": -1,
                "padding_value": 0.0,
                "length_field": "audio_lengths",
            },
            "audio.waveform": {
                "sequence_dim": -1,
                "padding_value": 0.0,
                "length_field": "audio_lengths",
            },
            "audio.audio": {
                "sequence_dim": -1,
                "padding_value": 0.0,
                "length_field": "audio_lengths",
            },
            "audio.input_values": {
                "sequence_dim": -1,
                "padding_value": 0.0,
                "length_field": "audio_lengths",
            },
            "audio_values": {
                "sequence_dim": -1,
                "padding_value": 0.0,
                "length_field": "audio_lengths",
            },
        },
    ),
    _profile(
        "zonos2",
        TrainingFamily.CAUSAL_LM,
        dataset_spec_factory="voicehub.training.data_contracts:build_zonos2_dataset_spec",
        adapter_factory="voicehub.models.zonos2.training:Zonos2TrainingAdapter",
        module_paths=("model", ),
        component_paths=("model", ),
        label_names=("labels", "audio_codes", "audio_values"),
        source_entrypoints=("voicehub.models.zonos2.native.modeling:"
                            "Zonos2ForCausalLM.forward", ),
        native_training=True,
        support=TrainingSupport.NATIVE,
        training_default_model_name_or_path="Zyphra/ZONOS2",
        phases=(
            _phase(
                "reconstructed_codec_language_model",
                component_paths=("model", ),
                optimizer_names=("model", ),
                label_names=("labels", ),
                prediction_keys=("logits", ),
                loss_keys=("loss", ),
                required_inputs=("input_ids", "labels"),
            ), ),
        default_phase="reconstructed_codec_language_model",
        field_schemas={
            "audio_codes": {
                "sequence_dim": 0,
                "padding_value": 1025,
                "length_field": "audio_code_lengths",
            },
            "audio": {
                "sequence_dim": -1,
                "padding_value": 0.0,
                "length_field": "audio_lengths",
            },
            "audio.array": {
                "sequence_dim": -1,
                "padding_value": 0.0,
                "length_field": "audio_lengths",
            },
            "audio.waveform": {
                "sequence_dim": -1,
                "padding_value": 0.0,
                "length_field": "audio_lengths",
            },
            "audio.audio": {
                "sequence_dim": -1,
                "padding_value": 0.0,
                "length_field": "audio_lengths",
            },
            "audio.input_values": {
                "sequence_dim": -1,
                "padding_value": 0.0,
                "length_field": "audio_lengths",
            },
            "audio_values": {
                "sequence_dim": -1,
                "padding_value": 0.0,
                "length_field": "audio_lengths",
            },
        },
    ),
    _profile(
        "voxcpm",
        TrainingFamily.FLOW_MATCHING,
        dataset_spec_factory="voicehub.training.data_contracts:build_voxcpm_dataset_spec",
        adapter_factory="voicehub.models.voxcpm.training:VoxCPMTrainingAdapter",
        module_paths=("model", ),
        component_paths=(
            "model",
            "model.base_lm",
            "model.residual_lm",
            "model.feat_encoder",
            "model.feat_decoder",
        ),
        source_entrypoints=("voicehub.models.voxcpm.native.modeling:"
                            "VoxCPM2Model.forward", ),
        native_training=True,
        support=TrainingSupport.NATIVE,
        phases=(
            _phase(
                "source_flow_and_stop",
                component_paths=("model", ),
                optimizer_names=("model", ),
                prediction_keys=("target_features", ),
                loss_keys=("diffusion_loss", "stop_loss"),
                required_inputs=(
                    "text_tokens",
                    "text_mask",
                    "audio_feats",
                    "audio_mask",
                    "loss_mask",
                    "position_ids",
                    "labels",
                ),
            ), ),
        default_phase="source_flow_and_stop",
    ),
    _profile(
        "omnivoice",
        TrainingFamily.COMPOSITE,
        dataset_spec_factory="voicehub.training.data_contracts:build_omnivoice_dataset_spec",
        adapter_factory="voicehub.models.omnivoice.training:OmniVoiceTrainingAdapter",
        module_paths=("model", ),
        component_paths=("model", ),
        source_entrypoints=("voicehub.models.omnivoice.native.modeling:"
                            "OmniVoiceModel.forward", ),
        native_training=True,
        separate_optimizers=False,
        support=TrainingSupport.NATIVE,
        phases=(
            _phase(
                "masked_audio",
                component_paths=("model", ),
                optimizer_names=("model", ),
                label_names=("labels", ),
                prediction_keys=("logits", ),
                loss_keys=("loss", ),
                required_inputs=("input_ids", "audio_mask", "labels"),
            ), ),
        default_phase="masked_audio",
        field_schemas={
            "input_ids": {
                "sequence_dim": -1,
                "padding_value": 0,
            },
            "audio_mask": {
                "sequence_dim": -1,
                "padding_value": False,
            },
            "labels": {
                "sequence_dim": -1,
                "padding_value": -100,
            },
        },
    ),
    _profile(
        "higgstts",
        TrainingFamily.CAUSAL_LM,
        dataset_spec_factory="voicehub.training.data_contracts:build_higgstts_dataset_spec",
        adapter_factory="voicehub.models.higgstts.training:HiggsTrainingAdapter",
        module_paths=("model", ),
        component_paths=("model", ),
        source_entrypoints=(
            "voicehub.models.higgstts.native.modeling:"
            "HiggsAudioV2ForConditionalGeneration.forward", ),
        native_training=True,
        separate_optimizers=False,
        support=TrainingSupport.NATIVE,
        training_default_model_name_or_path=("bosonai/higgs-tts-2-3b-base"),
        phases=(
            _phase(
                "codec_language_model",
                component_paths=("model", ),
                optimizer_names=("model", ),
                label_names=("labels", "audio_labels"),
                prediction_keys=("logits", "text_logits"),
                loss_keys=("loss", "text_loss", "audio_loss"),
                required_inputs=(
                    "input_ids",
                    "attention_mask",
                    "audio_input_ids",
                    "audio_input_ids_mask",
                    "labels",
                    "audio_labels",
                ),
            ), ),
        default_phase="codec_language_model",
        field_schemas={
            "input_ids": {
                "sequence_dim": -1,
                "padding_value": 128_001,
            },
            "attention_mask": {
                "sequence_dim": -1,
                "padding_value": False,
            },
            "labels": {
                "sequence_dim": -1,
                "padding_value": -100,
            },
            "audio_input_ids": {
                "sequence_dim": -2,
                "padding_value": 1_025,
            },
            "audio_input_ids_mask": {
                "sequence_dim": -1,
                "padding_value": False,
            },
            "audio_labels": {
                "sequence_dim": -2,
                "padding_value": -100,
            },
        },
    ),
    _profile(
        "xtts",
        TrainingFamily.COMPOSITE,
        dataset_spec_factory="voicehub.training.data_contracts:build_xtts_dataset_spec",
        adapter_factory="voicehub.models.xtts.training_xtts:XTTSTrainingAdapter",
        module_paths=("model.gpt", ),
        component_paths=("model.gpt", ),
        loss_weights=(
            ("text_ce", 0.01),
            ("mel_ce", 1.0),
        ),
        source_entrypoints=(
            "voicehub.models.xtts.native.gpt:XTTS2GPT.forward",
            "voicehub.models.xtts.training_xtts:"
            "XTTSTrainingAdapter",
            "TTS/tts/layers/xtts/trainer/gpt_trainer.py",
        ),
        native_training=True,
        support=TrainingSupport.PREPROCESSED,
        training_default_model_name_or_path="coqui/XTTS-v2",
        separate_optimizers=False,
        phases=(
            _phase(
                "language_model",
                component_paths=("model.gpt", ),
                optimizer_names=("language_model", ),
                label_names=("text_inputs", "audio_codes"),
                prediction_keys=("logits", ),
                loss_keys=("loss", "loss_text_ce", "loss_mel_ce"),
                required_inputs=(
                    "text_inputs",
                    "text_lengths",
                    "audio_codes",
                    "wav_lengths",
                ),
            ), ),
        default_phase="language_model",
        field_schemas={
            "text_inputs": {
                "sequence_dim": -1,
                "padding_value": 0,
                "length_field": "text_lengths",
            },
            "audio_codes": {
                "sequence_dim": -1,
                "padding_value": 1_025,
            },
            "wav": {
                "sequence_dim": -1,
                "padding_value": 0.0,
                "length_field": "wav_lengths",
            },
            "audio_values": {
                "sequence_dim": -1,
                "padding_value": 0.0,
                "length_field": "wav_lengths",
            },
            "cond_mels": {
                "sequence_dim": -1,
                "padding_value": 0.0,
            },
            "cond_latents": {
                "sequence_dim": 0,
                "padding_value": 0.0,
            },
        },
    ),
    _profile(
        "vibevoice",
        TrainingFamily.COMPOSITE,
        dataset_spec_factory="voicehub.training.data_contracts:build_vibevoice_dataset_spec",
        adapter_factory="voicehub.models.vibevoice.training:VibeVoiceTrainingAdapter",
        module_paths=("model", ),
        component_paths=("model", ),
        source_entrypoints=(
            "voicehub.models.vibevoice.training.VibeVoiceTrainingAdapter",
            "microsoft/VibeVoice/finetuning",
        ),
        native_training=True,
        separate_optimizers=False,
        support=TrainingSupport.PREPROCESSED,
        training_default_model_name_or_path="microsoft/VibeVoice-1.5B",
        phases=(
            _phase(
                "lm_diffusion",
                component_paths=("model", ),
                optimizer_names=("model", ),
                label_names=("input_ids", "speeches_loss_input"),
                prediction_keys=("logits", ),
                loss_keys=("loss", "ce_loss", "diffusion_loss"),
                required_inputs=(
                    "input_ids",
                    "attention_mask",
                    "speech_tensors",
                    "speech_masks",
                    "speeches_loss_input",
                    "speech_semantic_tensors",
                    "acoustic_input_mask",
                    "acoustic_loss_mask",
                ),
            ), ),
        default_phase="lm_diffusion",
    ),
    _profile(
        "fishtts",
        TrainingFamily.CAUSAL_LM,
        dataset_spec_factory="voicehub.training.data_contracts:build_fishtts_dataset_spec",
        adapter_factory="voicehub.models.fishtts.training:FishSpeechTrainingAdapter",
        module_paths=("model", ),
        component_paths=("model", ),
        source_entrypoints=(
            "voicehub.models.fishtts.native.modeling:"
            "FishS2ForConditionalGeneration.forward",
            "voicehub.models.fishtts.training:"
            "FishSpeechTrainingAdapter.compute_source_losses",
            "voicehub.models.fishtts.training:FishTextDataCollator",
        ),
        native_training=True,
        support=TrainingSupport.PREPROCESSED,
        training_default_model_name_or_path="fishaudio/s2-pro",
        phases=(
            _phase(
                "semantic",
                component_paths=("model", ),
                optimizer_names=("semantic", ),
                label_names=("labels", ),
                prediction_keys=("token_logits", "codebook_logits"),
                loss_keys=("loss", "base_loss", "semantic_loss"),
                required_inputs=("inputs", "labels"),
            ), ),
        default_phase="semantic",
    ),
    _profile(
        "csm",
        TrainingFamily.CAUSAL_LM,
        dataset_spec_factory="voicehub.training.data_contracts:build_csm_dataset_spec",
        adapter_factory="voicehub.models.csm.training:CSMTrainingAdapter",
        module_paths=("model", ),
        component_paths=("model", ),
        label_names=("labels", ),
        source_entrypoints=(
            "voicehub.models.csm.native.modeling:CSMModel.forward",
            "voicehub.models.csm.native.processing:"
            "CSMProcessor.training_batch",
            "voicehub.models.csm.native.mimi:load_mimi",
        ),
        native_training=True,
        support=TrainingSupport.NATIVE,
        phases=(
            _phase(
                "codec_language_model",
                component_paths=("model", ),
                optimizer_names=("codec_language_model", ),
                label_names=("labels", ),
                prediction_keys=("logits", ),
                loss_keys=("loss", "backbone_loss", "depth_decoder_loss"),
            ), ),
        default_phase="codec_language_model",
    ),
    _profile(
        "neutts",
        TrainingFamily.CAUSAL_LM,
        dataset_spec_factory="voicehub.training.data_contracts:build_neutts_dataset_spec",
        adapter_factory="voicehub.models.neutts.training:NeuTTSTrainingAdapter",
        module_paths=("model.backbone", ),
        component_paths=("model.backbone", ),
        label_names=("labels", ),
        source_entrypoints=(
            "voicehub.models.neutts.native.modeling:"
            "NeuTTSBackbone.forward",
            "voicehub.models.neutts.training:NeuTTSSFTDataset",
        ),
        dataset_factory=("voicehub.models.neutts.training:"
                         "build_training_dataset"),
        native_training=True,
        support=TrainingSupport.NATIVE,
        phases=(
            _phase(
                "codec_language_model",
                component_paths=("model.backbone", ),
                optimizer_names=("codec_language_model", ),
                label_names=("labels", ),
                prediction_keys=("logits", ),
                loss_keys=("loss", ),
                required_inputs=("input_ids", "labels"),
            ), ),
        default_phase="codec_language_model",
        training_default_model_name_or_path="neuphonic/neutts-air",
    ),
    _profile(
        "supertonic",
        TrainingFamily.FLOW_MATCHING,
        dataset_spec_factory="voicehub.training.data_contracts:build_supertonic_dataset_spec",
        adapter_factory="voicehub.models.supertonic.training:SupertonicTrainingAdapter",
        module_paths=("model", ),
        component_paths=("model", ),
        source_entrypoints=(
            "voicehub.models.supertonic.native.runtime:"
            "NativeSupertonicRuntime.fine_tuning_loss", ),
        native_training=True,
        support=TrainingSupport.PREPROCESSED,
        phases=(
            _phase(
                "published_graph",
                component_paths=("model", ),
                optimizer_names=("published_graph", ),
                forward_component="model",
                forward_method="fine_tuning_loss",
                label_names=(
                    "target_latent",
                    "target_duration",
                    "target_audio",
                ),
                prediction_keys=(
                    "next_latent",
                    "duration",
                    "waveform",
                ),
                loss_keys=(
                    "loss",
                    "duration_loss",
                    "flow_step_loss",
                    "vocoder_l1_loss",
                ),
                required_inputs=(
                    "text_ids",
                    "text_mask",
                    "style_ttl",
                    "style_dp",
                ),
            ), ),
        default_phase="published_graph",
        field_schemas={
            "text_ids": {
                "sequence_dim": -1,
                "padding_value": 0,
            },
            "text_mask": {
                "sequence_dim": -1,
                "padding_value": 0.0,
            },
            "target_latent": {
                "sequence_dim": -1,
                "padding_value": 0.0,
            },
            "latent_mask": {
                "sequence_dim": -1,
                "padding_value": 0.0,
            },
            "target_audio": {
                "sequence_dim": -1,
                "padding_value": 0.0,
            },
        },
    ),
    _profile(
        "inflecttts",
        TrainingFamily.VITS,
        dataset_spec_factory="voicehub.training.data_contracts:build_inflecttts_dataset_spec",
        adapter_factory="voicehub.models.inflecttts.training:InflectTTSTrainingAdapter",
        module_paths=("training_model", ),
        component_paths=(
            "training_model.generator",
            "training_model.discriminator",
        ),
        source_entrypoints=(
            "voicehub.models.inflecttts.native.training:"
            "InflectV2TrainingModel.generator_objective",
            "voicehub.models.inflecttts.native.training:"
            "InflectV2TrainingModel.discriminator_objective",
            "voicehub.models.inflecttts.training:"
            "InflectTTSTrainingAdapter",
        ),
        native_training=True,
        separate_optimizers=True,
        support=TrainingSupport.PREPROCESSED,
        training_default_model_name_or_path="owensong/Inflect-Micro-v2",
        phases=(
            _phase(
                "generator",
                component_paths=("training_model.generator", ),
                optimizer_names=("generator", ),
                forward_component="training_model",
                forward_method="generator_objective",
                label_names=("audio_values", "spectrogram"),
                prediction_keys=("waveform", ),
                loss_keys=(
                    "loss",
                    "mel_loss",
                    "kl_loss",
                    "duration_loss",
                    "adversarial_loss",
                    "feature_matching_loss",
                    "waveform_loss",
                ),
                required_inputs=(
                    "input_ids",
                    "input_lengths",
                    "spectrogram",
                    "spectrogram_lengths",
                    "audio_values",
                ),
                kind=TrainingPhaseKind.GENERATOR,
                frozen_component_paths=("training_model.discriminator", ),
            ),
            _phase(
                "discriminator",
                component_paths=("training_model.discriminator", ),
                optimizer_names=("discriminator", ),
                forward_component="training_model",
                forward_method="discriminator_objective",
                label_names=("audio_values", ),
                prediction_keys=("waveform", ),
                loss_keys=("loss", ),
                required_inputs=(
                    "input_ids",
                    "input_lengths",
                    "spectrogram",
                    "spectrogram_lengths",
                    "audio_values",
                ),
                kind=TrainingPhaseKind.DISCRIMINATOR,
                frozen_component_paths=("training_model.generator", ),
            ),
        ),
        default_phase="generator",
        recipe_kind=TrainingRecipeKind.ADVERSARIAL,
        field_schemas={
            "input_ids": {
                "sequence_dim": 0,
                "padding_value": 0,
                "length_field": "input_lengths",
            },
            "spectrogram": {
                "sequence_dim": -1,
                "padding_value": 0.0,
                "length_field": "spectrogram_lengths",
            },
            "audio_values": {
                "sequence_dim": -1,
                "padding_value": 0.0,
            },
        },
    ),
    _profile(
        "bark",
        TrainingFamily.COMPOSITE,
        dataset_spec_factory="voicehub.training.data_contracts:build_bark_dataset_spec",
        adapter_factory="voicehub.models.bark.native.training:BarkTrainingAdapter",
        module_paths=("training_model.semantic", ),
        component_paths=(
            "training_model.semantic",
            "training_model.coarse",
            "training_model.fine",
        ),
        label_names=("labels", ),
        source_entrypoints=(
            "voicehub.models.bark.native.modeling:BarkModel",
            "voicehub.models.bark.native.training:"
            "BarkTrainingAdapter",
            "voicehub.models.bark.native.training:"
            "BarkTokenObjective",
        ),
        native_training=True,
        separate_optimizers=True,
        support=TrainingSupport.PREPROCESSED,
        phases=(
            _phase(
                "semantic",
                component_paths=("training_model.semantic", ),
                optimizer_names=("semantic", ),
                forward_component="training_model.semantic",
                label_names=("labels", ),
                prediction_keys=("logits", ),
                loss_keys=("loss", ),
                required_inputs=("input_ids", "labels"),
            ),
            _phase(
                "coarse",
                component_paths=("training_model.coarse", ),
                optimizer_names=("coarse", ),
                forward_component="training_model.coarse",
                label_names=("labels", ),
                prediction_keys=("logits", ),
                loss_keys=("loss", ),
                required_inputs=("input_ids", "labels"),
            ),
            _phase(
                "fine",
                component_paths=("training_model.fine", ),
                optimizer_names=("fine", ),
                forward_component="training_model.fine",
                label_names=("labels", ),
                prediction_keys=("logits", ),
                loss_keys=("loss", ),
                required_inputs=("input_ids", "labels", "codebook_idx"),
            ),
        ),
        default_phase="semantic",
        recipe_kind=TrainingRecipeKind.MULTI_PHASE,
        field_schemas={
            "semantic_input_ids": {
                "sequence_dim": -1,
                "padding_value": 0,
            },
            "semantic_labels": {
                "sequence_dim": -1,
                "padding_value": -100,
            },
            "coarse_input_ids": {
                "sequence_dim": -1,
                "padding_value": 0,
            },
            "coarse_labels": {
                "sequence_dim": -1,
                "padding_value": -100,
            },
            "fine_input_ids": {
                "sequence_dim": -2,
                "padding_value": 0,
            },
            "fine_labels": {
                "sequence_dim": -1,
                "padding_value": -100,
            },
        },
    ),
    _profile(
        "speecht5",
        TrainingFamily.SEQ2SEQ,
        dataset_spec_factory="voicehub.training.data_contracts:build_speecht5_dataset_spec",
        adapter_factory="voicehub.models.speecht5.training:NativeSpeechT5TrainingAdapter",
        module_paths=("model", ),
        component_paths=("model", ),
        label_names=("labels", ),
        source_entrypoints=(
            "voicehub.models.speecht5.native_modeling:"
            "SpeechT5ForTextToSpeechModel.forward",
            "voicehub.models.speecht5.processing:SpeechT5Processor",
            "voicehub.models.speecht5.training:"
            "NativeSpeechT5TrainingAdapter",
        ),
        native_training=True,
        separate_optimizers=False,
        support=TrainingSupport.NATIVE,
        phases=(
            _phase(
                "spectrogram",
                component_paths=("model", ),
                optimizer_names=("model", ),
                label_names=("labels", ),
                prediction_keys=("spectrogram", ),
                loss_keys=("loss", ),
                required_inputs=(
                    "input_ids",
                    "attention_mask",
                    "labels",
                ),
                frozen_component_paths=("vocoder", ),
            ), ),
        default_phase="spectrogram",
        field_schemas={
            "input_ids": {
                "sequence_dim": -1,
                "padding_value": 1,
                "mask_field": "attention_mask",
            },
            "attention_mask": {
                "sequence_dim": -1,
                "padding_value": 0,
            },
            "labels": {
                "sequence_dim": -2,
                "padding_value": -100.0,
                "mask_field": "decoder_attention_mask",
            },
            "decoder_attention_mask": {
                "sequence_dim": -1,
                "padding_value": 0,
            },
            "decoder_input_values": {
                "sequence_dim": -2,
                "padding_value": 0.0,
            },
            "speaker_embeddings": {
                "sequence_dim": -1,
                "padding_value": 0.0,
            },
        },
    ),
    _profile(
        "vits",
        TrainingFamily.VITS,
        dataset_spec_factory="voicehub.training.data_contracts:build_vits_dataset_spec",
        adapter_factory="voicehub.models.vits.training:NativeVitsGeneratorTrainingAdapter",
        module_paths=("training_model", ),
        component_paths=(
            "training_model.native_model",
            "training_model.discriminator",
        ),
        label_names=("audio_values", ),
        source_entrypoints=(
            "voicehub.models.vits.native.modeling.VitsModel",
            "voicehub.models.vits.native.frontend.VitsTokenizer",
            "voicehub.models.vits.native.training.VitsAcousticFrontend",
            "voicehub.models.vits.native.training.VitsAdversarialTrainingModel",
            "voicehub.models.vits.training.NativeVitsGeneratorTrainingAdapter",
        ),
        optimization_profile_factory=("voicehub.training.tts_optimization:VITSOptimizationConfig"),
        # MMS-TTS checkpoints omit the source FFT, hop, window, mel, and
        # segment settings. Full raw-waveform training is therefore available
        # only after the caller supplies that exact acoustic configuration.
        native_training=True,
        separate_optimizers=True,
        support=TrainingSupport.PREPROCESSED,
        phases=(
            _phase(
                "discriminator",
                component_paths=("training_model.discriminator", ),
                optimizer_names=("discriminator", ),
                forward_component="training_model",
                forward_method="discriminator_step",
                label_names=("audio_values", ),
                prediction_keys=("audio_values", ),
                loss_keys=("loss", ),
                required_inputs=("input_ids", "audio_values"),
                kind=TrainingPhaseKind.DISCRIMINATOR,
                frozen_component_paths=("training_model.native_model", ),
            ),
            _phase(
                "generator",
                component_paths=("training_model.native_model", ),
                optimizer_names=("generator", ),
                forward_component="training_model",
                forward_method="generator_step",
                label_names=("audio_values", ),
                prediction_keys=("audio_values", ),
                loss_keys=("loss", ),
                required_inputs=("input_ids", "audio_values"),
                kind=TrainingPhaseKind.GENERATOR,
                frozen_component_paths=("training_model.discriminator", ),
            ),
        ),
        default_phase="generator",
        recipe_kind=TrainingRecipeKind.ADVERSARIAL,
        field_schemas={
            "input_ids": {
                "sequence_dim": -1,
                "padding_value": 0,
                "length_field": "input_lengths",
            },
            "attention_mask": {
                "sequence_dim": -1,
                "padding_value": 0,
            },
            "spectrogram": {
                "sequence_dim": -1,
                "padding_value": 0.0,
                "length_field": "spectrogram_lengths",
            },
            "spectrogram_attention_mask": {
                "sequence_dim": -1,
                "padding_value": 0,
            },
            "audio_values": {
                "sequence_dim": -1,
                "padding_value": 0.0,
                "length_field": "audio_lengths",
            },
        },
    ),
)

_RAW_AUDIO_FIELD_SCHEMAS = {
    "audio": {
        "sequence_dim": -1,
        "padding_value": 0.0,
        "length_field": "audio_lengths",
    },
    "input_values": {
        "sequence_dim": -1,
        "padding_value": 0.0,
        "length_field": "input_lengths",
        "mask_field": "attention_mask",
    },
    "input_features": {
        "sequence_dim": -1,
        "padding_value": 0.0,
        "length_field": "feature_lengths",
    },
}

_WAVEFORM_ASR_FIELD_SCHEMAS = {
    "audio": _RAW_AUDIO_FIELD_SCHEMAS["audio"],
    "input_values": _RAW_AUDIO_FIELD_SCHEMAS["input_values"],
}

_NEMO_CTC_FIELD_SCHEMAS = {
    "audio": _RAW_AUDIO_FIELD_SCHEMAS["audio"],
    "input_signal": {
        "sequence_dim": -1,
        "padding_value": 0.0,
        "length_field": "input_signal_length",
    },
    "processed_signal": {
        "sequence_dim": -1,
        "padding_value": 0.0,
        "length_field": "processed_signal_length",
    },
    "labels": {
        "sequence_dim": -1,
        "padding_value": -1,
        "length_field": "label_lengths",
    },
}

_WENET_U2PP_FIELD_SCHEMAS = {
    "audio": _RAW_AUDIO_FIELD_SCHEMAS["audio"],
    "input_signal": {
        "sequence_dim": -1,
        "padding_value": 0.0,
        "length_field": "input_signal_length",
    },
    "features": {
        "sequence_dim": -2,
        "padding_value": 0.0,
        "length_field": "feature_lengths",
    },
    "labels": {
        "sequence_dim": -1,
        "padding_value": -1,
        "length_field": "label_lengths",
    },
}

_SPEECHBRAIN_ASR_FIELD_SCHEMAS = {
    "audio": _RAW_AUDIO_FIELD_SCHEMAS["audio"],
    "waveforms": {
        "sequence_dim": -1,
        "padding_value": 0.0,
        "length_field": "waveform_lengths",
    },
    "tokens_bos": {
        "sequence_dim": -1,
        "padding_value": 0,
    },
    "tokens_eos": {
        "sequence_dim": -1,
        "padding_value": 0,
        "length_field": "token_lengths",
    },
    "ctc_tokens": {
        "sequence_dim": -1,
        "padding_value": 0,
        "length_field": "ctc_token_lengths",
    },
}

_CHANNEL_FIRST_ASR_FIELD_SCHEMAS = {
    **_WAVEFORM_ASR_FIELD_SCHEMAS,
    "input_features": {
        "sequence_dim": -1,
        "padding_value": 0.0,
        "length_field": "feature_lengths",
        "mask_field": "attention_mask",
    },
}

_QWEN3_ASR_FIELD_SCHEMAS = {
    **_WAVEFORM_ASR_FIELD_SCHEMAS,
    "input_ids": {
        "sequence_dim": -1,
        "padding_value": 151_643,
        "mask_field": "attention_mask",
    },
    "attention_mask": {
        "sequence_dim": -1,
        "padding_value": 0,
    },
    "input_features": {
        "sequence_dim": -1,
        "padding_value": 0.0,
        "length_field": "feature_lengths",
        "mask_field": "feature_attention_mask",
    },
    "feature_attention_mask": {
        "sequence_dim": -1,
        "padding_value": 0,
    },
    "labels": {
        "sequence_dim": -1,
        "padding_value": -100,
    },
}

_TIME_MAJOR_ASR_FIELD_SCHEMAS = {
    **_WAVEFORM_ASR_FIELD_SCHEMAS,
    "input_features": {
        "sequence_dim": -2,
        "padding_value": 0.0,
        "length_field": "feature_lengths",
        "mask_field": "attention_mask",
    },
}

_PARAKEET_TDT_FIELD_SCHEMAS = {
    **_TIME_MAJOR_ASR_FIELD_SCHEMAS,
    "labels": {
        "sequence_dim": -1,
        "padding_value": 2,
    },
    "decoder_input_ids": {
        "sequence_dim": -1,
        "padding_value": 2,
    },
}

_NEMOTRON_RNNT_FIELD_SCHEMAS = {
    **_TIME_MAJOR_ASR_FIELD_SCHEMAS,
    "labels": {
        "sequence_dim": -1,
        "padding_value": 13_087,
        "length_field": "label_lengths",
    },
    "decoder_input_ids": {
        "sequence_dim": -1,
        "padding_value": 13_087,
    },
}

_COHERE_ASR_FIELD_SCHEMAS = {
    **_TIME_MAJOR_ASR_FIELD_SCHEMAS,
    "decoder_input_ids": {
        "sequence_dim": -1,
        "padding_value": 2,
        "mask_field": "decoder_attention_mask",
    },
    "decoder_attention_mask": {
        "sequence_dim": -1,
        "padding_value": 0,
    },
    "labels": {
        "sequence_dim": -1,
        "padding_value": -100,
    },
}

_MEDASR_FIELD_SCHEMAS = {
    **_TIME_MAJOR_ASR_FIELD_SCHEMAS,
    # LASR masks tokenizer pad ID 0 and also uses it as the CTC blank.
    "labels": {
        "sequence_dim": -1,
        "padding_value": 0,
    },
}

_VIBEVOICE_ASR_FIELD_SCHEMAS = {
    "audio": _RAW_AUDIO_FIELD_SCHEMAS["audio"],
    "input_ids": {
        "sequence_dim": -1,
        "padding_value": 151_643,
        "mask_field": "attention_mask",
    },
    "attention_mask": {
        "sequence_dim": -1,
        "padding_value": 0,
    },
    "input_values": {
        "sequence_dim": -1,
        "padding_value": 0.0,
        "length_field": "input_lengths",
        "mask_field": "padding_mask",
    },
    "padding_mask": {
        "sequence_dim": -1,
        "padding_value": False,
    },
    "labels": {
        "sequence_dim": -1,
        "padding_value": -100,
    },
}

_GRANITE_SPEECH_ASR_FIELD_SCHEMAS = {
    **_WAVEFORM_ASR_FIELD_SCHEMAS,
    "input_ids": {
        "sequence_dim": -1,
        "padding_value": 100_256,
        "mask_field": "attention_mask",
    },
    "attention_mask": {
        "sequence_dim": -1,
        "padding_value": 0,
    },
    "input_features": {
        "sequence_dim": -2,
        "padding_value": 0.0,
        "length_field": "feature_lengths",
    },
    # Granite's encoder mask is downsampled relative to input_features, so it
    # must retain its processor-native timebase instead of being regenerated
    # from the feature tensor's length.
    "input_features_mask": {
        "sequence_dim": -1,
        "padding_value": False,
    },
    "labels": {
        "sequence_dim": -1,
        "padding_value": -100,
    },
}


def _transformers_asr_preset_profile(
    model_type: str,
    family: TrainingFamily,
    entrypoint: str,
    *,
    adapter_factory: str,
    dataset_spec_factory: str,
    field_schemas: Mapping[str, Any] = _RAW_AUDIO_FIELD_SCHEMAS,
) -> ModelTrainingSpec:
    """Create one locked preset around the shared native ASR trainer."""
    return _profile(
        model_type,
        family,
        task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
        module_paths=("model", ),
        component_paths=("model", ),
        label_names=("labels", ),
        source_entrypoints=(entrypoint, ),
        adapter_factory=adapter_factory,
        dataset_spec_factory=dataset_spec_factory,
        native_training=True,
        support=TrainingSupport.NATIVE,
        phases=(
            _phase(
                "speech_recognition",
                component_paths=("model", ),
                optimizer_names=("model", ),
                label_names=("labels", ),
                prediction_keys=("logits", ),
                loss_keys=("loss", ),
            ), ),
        default_phase="speech_recognition",
        field_schemas=field_schemas,
    )


_BUILTIN_AUDIO_INPUT_TRAINING_SPECS = (
    _profile(
        "asr_transformers",
        TrainingFamily.NATIVE_ASR_DISPATCH,
        adapter_factory=(
            "voicehub.models.asr_transformers.training_asr_transformers:TransformersASRTrainingAdapter"),
        dataset_spec_factory=("voicehub.training.asr_data_contracts:build_asr_transformers_dataset_spec"),
        task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
        module_paths=("model", ),
        component_paths=("model", ),
        label_names=("labels", ),
        source_entrypoints=(
            "voicehub.models.asr_whisper_native.native.WhisperModel",
            "voicehub.models.asr_wav2vec2.native.Wav2Vec2ForCTC",
            "voicehub.models.asr_hubert.native.HubertForCTC",
            "voicehub.models.asr_wavlm.native.WavLMForCTC",
            ("voicehub.models.asr_moonshine.native."
             "MoonshineForConditionalGeneration"),
        ),
        native_training=True,
        support=TrainingSupport.NATIVE,
        phases=(
            _phase(
                "speech_recognition",
                component_paths=("model", ),
                optimizer_names=("model", ),
                label_names=("labels", ),
                prediction_keys=("logits", ),
                loss_keys=("loss", ),
            ), ),
        default_phase="speech_recognition",
        field_schemas=_RAW_AUDIO_FIELD_SCHEMAS,
    ),
    _transformers_asr_preset_profile(
        "asr_whisper",
        TrainingFamily.SPEECH_SEQ2SEQ,
        "voicehub.models.asr_whisper_native.native.WhisperModel",
        adapter_factory=(
            "voicehub.models.asr_whisper_native.training_asr_whisper_native:NativeWhisperTrainingAdapter"),
        dataset_spec_factory=("voicehub.training.asr_data_contracts:build_asr_whisper_dataset_spec"),
        field_schemas=_CHANNEL_FIRST_ASR_FIELD_SCHEMAS,
    ),
    _transformers_asr_preset_profile(
        "asr_tiron",
        TrainingFamily.SPEECH_SEQ2SEQ,
        "voicehub.models.asr_whisper_native.native.WhisperModel",
        adapter_factory=(
            "voicehub.models.asr_whisper_native.training_asr_whisper_native:NativeWhisperTrainingAdapter"),
        dataset_spec_factory=("voicehub.training.asr_data_contracts:build_asr_tiron_dataset_spec"),
        field_schemas=_CHANNEL_FIRST_ASR_FIELD_SCHEMAS,
    ),
    _transformers_asr_preset_profile(
        "asr_qwen3",
        TrainingFamily.SPEECH_SEQ2SEQ,
        ("voicehub.models.asr_qwen3.native.modeling."
         "Qwen3ASRForConditionalGeneration"),
        adapter_factory="voicehub.models.asr_qwen3.training_asr_qwen3:NativeQwen3ASRTrainingAdapter",
        dataset_spec_factory=("voicehub.training.asr_data_contracts:build_asr_qwen3_dataset_spec"),
        field_schemas=_QWEN3_ASR_FIELD_SCHEMAS,
    ),
    _profile(
        "asr_vibevoice",
        TrainingFamily.SPEECH_SEQ2SEQ,
        adapter_factory=(
            "voicehub.models.asr_vibevoice.training_asr_vibevoice:NativeVibeVoiceASRTrainingAdapter"),
        dataset_spec_factory=("voicehub.training.asr_data_contracts:build_asr_vibevoice_dataset_spec"),
        task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
        module_paths=("model", ),
        component_paths=(
            "model.model.multi_modal_projector",
            "model.model.language_model",
            "model.lm_head",
        ),
        label_names=("labels", ),
        source_entrypoints=(
            "voicehub.models.vibevoice.native.modeling:"
            "VibeVoiceASRForConditionalGeneration",
            "voicehub.models.asr_vibevoice.training_asr_vibevoice:"
            "NativeVibeVoiceASRTrainingAdapter",
        ),
        native_training=True,
        support=TrainingSupport.NATIVE,
        phases=(
            _phase(
                "speech_recognition",
                component_paths=(
                    "model.model.multi_modal_projector",
                    "model.model.language_model",
                    "model.lm_head",
                ),
                optimizer_names=("model", ),
                forward_component="model",
                label_names=("labels", ),
                prediction_keys=("logits", ),
                loss_keys=("loss", ),
                required_inputs=(
                    "input_ids",
                    "attention_mask",
                    "input_values",
                    "padding_mask",
                    "labels",
                ),
                frozen_component_paths=(
                    "model.model.acoustic_tokenizer_encoder",
                    "model.model.semantic_tokenizer_encoder",
                ),
            ), ),
        default_phase="speech_recognition",
        training_default_model_name_or_path="microsoft/VibeVoice-ASR-HF",
        field_schemas=_VIBEVOICE_ASR_FIELD_SCHEMAS,
    ),
    _profile(
        "asr_granite_speech",
        TrainingFamily.SPEECH_SEQ2SEQ,
        adapter_factory=(
            "voicehub.models.asr_granite_speech.training_asr_granite_speech:NativeGraniteSpeechTrainingAdapter"
        ),
        dataset_spec_factory=("voicehub.training.asr_data_contracts:build_asr_granite_speech_dataset_spec"),
        task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
        module_paths=("model", ),
        component_paths=(
            "model.encoder",
            "model.projector",
            "model.language_model",
        ),
        label_names=("labels", ),
        source_entrypoints=(
            "voicehub.models.asr_granite_speech.native.modeling:"
            "GraniteSpeechForConditionalGeneration",
            "voicehub.models.asr_granite_speech."
            "training_asr_granite_speech:"
            "NativeGraniteSpeechTrainingAdapter",
        ),
        native_training=True,
        support=TrainingSupport.NATIVE,
        phases=(
            _phase(
                "speech_recognition",
                component_paths=(
                    "model.encoder",
                    "model.projector",
                    "model.language_model",
                ),
                optimizer_names=("model", ),
                forward_component="model",
                label_names=("labels", ),
                prediction_keys=("logits", ),
                loss_keys=("loss", ),
                required_inputs=(
                    "input_ids",
                    "attention_mask",
                    "input_features",
                    "input_features_mask",
                    "labels",
                ),
            ), ),
        default_phase="speech_recognition",
        field_schemas=_GRANITE_SPEECH_ASR_FIELD_SCHEMAS,
    ),
    _profile(
        "asr_parakeet_tdt",
        TrainingFamily.TDT,
        adapter_factory=(
            "voicehub.models.asr_parakeet_tdt.training_asr_parakeet_tdt:NativeParakeetTDTTrainingAdapter"),
        dataset_spec_factory=("voicehub.training.asr_data_contracts:build_asr_parakeet_tdt_dataset_spec"),
        task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
        module_paths=("model", ),
        component_paths=(
            "model.encoder",
            "model.encoder_projector",
            "model.decoder",
            "model.joint",
        ),
        label_names=("labels", ),
        source_entrypoints=(
            "voicehub.models.asr_parakeet_tdt.native.modeling:ParakeetForTDT",
            "voicehub.models.asr_parakeet_tdt."
            "training_asr_parakeet_tdt:"
            "NativeParakeetTDTTrainingAdapter",
        ),
        native_training=True,
        support=TrainingSupport.NATIVE,
        phases=(
            _phase(
                "speech_recognition",
                component_paths=(
                    "model.encoder",
                    "model.encoder_projector",
                    "model.decoder",
                    "model.joint",
                ),
                optimizer_names=("model", ),
                forward_component="model",
                label_names=("labels", ),
                prediction_keys=("logits", ),
                loss_keys=("loss", ),
                required_inputs=(
                    "input_features",
                    "attention_mask",
                    "labels",
                    "decoder_input_ids",
                ),
            ), ),
        default_phase="speech_recognition",
        field_schemas=_PARAKEET_TDT_FIELD_SCHEMAS,
    ),
    _profile(
        "asr_nemotron",
        TrainingFamily.RNNT,
        adapter_factory="voicehub.models.asr_nemotron.training_asr_nemotron:NativeNemotronASRTrainingAdapter",
        dataset_spec_factory=("voicehub.training.asr_data_contracts:build_asr_nemotron_dataset_spec"),
        task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
        module_paths=("model", ),
        component_paths=(
            "model.encoder",
            "model.encoder_projector",
            "model.prompt_projector",
            "model.decoder",
            "model.joint",
        ),
        label_names=("labels", ),
        source_entrypoints=(
            "voicehub.models.asr_nemotron.native.modeling:"
            "Nemotron3_5ASRForRNNT",
            "voicehub.models.asr_nemotron.training_asr_nemotron:"
            "NativeNemotronASRTrainingAdapter",
        ),
        native_training=True,
        support=TrainingSupport.NATIVE,
        phases=(
            _phase(
                "speech_recognition",
                component_paths=(
                    "model.encoder",
                    "model.encoder_projector",
                    "model.prompt_projector",
                    "model.decoder",
                    "model.joint",
                ),
                optimizer_names=("model", ),
                forward_component="model",
                label_names=("labels", ),
                prediction_keys=("logits", ),
                loss_keys=("loss", ),
                required_inputs=(
                    "input_features",
                    "attention_mask",
                    "prompt_ids",
                    "labels",
                    "label_lengths",
                    "decoder_input_ids",
                ),
            ), ),
        default_phase="speech_recognition",
        field_schemas=_NEMOTRON_RNNT_FIELD_SCHEMAS,
    ),
    _profile(
        "asr_cohere",
        TrainingFamily.SPEECH_SEQ2SEQ,
        adapter_factory="voicehub.models.asr_cohere.training_asr_cohere:NativeCohereASRTrainingAdapter",
        dataset_spec_factory=("voicehub.training.asr_data_contracts:build_asr_cohere_dataset_spec"),
        task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
        module_paths=("model", ),
        component_paths=(
            "model.encoder",
            "model.encoder_decoder_proj",
            "model.transf_decoder",
            "model.log_softmax",
        ),
        label_names=("labels", ),
        source_entrypoints=(
            "voicehub.models.asr_cohere.native.modeling:"
            "CohereAsrForConditionalGeneration",
            "voicehub.models.asr_cohere.training_asr_cohere:"
            "NativeCohereASRTrainingAdapter",
        ),
        native_training=True,
        support=TrainingSupport.NATIVE,
        phases=(
            _phase(
                "speech_recognition",
                component_paths=(
                    "model.encoder",
                    "model.encoder_decoder_proj",
                    "model.transf_decoder",
                    "model.log_softmax",
                ),
                optimizer_names=("model", ),
                forward_component="model",
                label_names=("labels", ),
                prediction_keys=("logits", ),
                loss_keys=("loss", ),
                required_inputs=(
                    "input_features",
                    "attention_mask",
                    "decoder_input_ids",
                    "decoder_attention_mask",
                    "labels",
                ),
            ), ),
        default_phase="speech_recognition",
        field_schemas=_COHERE_ASR_FIELD_SCHEMAS,
    ),
    _profile(
        "asr_medasr",
        TrainingFamily.CTC,
        adapter_factory="voicehub.models.asr_medasr.training_asr_medasr:NativeMedASRTrainingAdapter",
        dataset_spec_factory=("voicehub.training.asr_data_contracts:build_asr_medasr_dataset_spec"),
        task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
        module_paths=("model", ),
        component_paths=(
            "model.encoder",
            "model.ctc_head",
        ),
        label_names=("labels", ),
        source_entrypoints=(
            "voicehub.models.asr_medasr.native.modeling:MedASRForCTC",
            "voicehub.models.asr_medasr.training_asr_medasr:"
            "NativeMedASRTrainingAdapter",
        ),
        native_training=True,
        support=TrainingSupport.NATIVE,
        phases=(
            _phase(
                "speech_recognition",
                component_paths=(
                    "model.encoder",
                    "model.ctc_head",
                ),
                optimizer_names=("model", ),
                forward_component="model",
                label_names=("labels", ),
                prediction_keys=("logits", ),
                loss_keys=("loss", ),
                required_inputs=(
                    "input_features",
                    "attention_mask",
                    "labels",
                ),
            ), ),
        default_phase="speech_recognition",
        field_schemas=_MEDASR_FIELD_SCHEMAS,
    ),
    _transformers_asr_preset_profile(
        "asr_wav2vec2",
        TrainingFamily.CTC,
        "voicehub.models.asr_wav2vec2.native.Wav2Vec2ForCTC",
        adapter_factory="voicehub.models.asr_wav2vec2.training_asr_wav2vec2:NativeWav2Vec2TrainingAdapter",
        dataset_spec_factory=("voicehub.training.asr_data_contracts:build_asr_wav2vec2_dataset_spec"),
        field_schemas=_WAVEFORM_ASR_FIELD_SCHEMAS,
    ),
    _transformers_asr_preset_profile(
        "asr_hubert",
        TrainingFamily.CTC,
        "voicehub.models.asr_hubert.native.HubertForCTC",
        adapter_factory="voicehub.models.asr_hubert.training_asr_hubert:NativeHubertTrainingAdapter",
        dataset_spec_factory=("voicehub.training.asr_data_contracts:build_asr_hubert_dataset_spec"),
        field_schemas=_WAVEFORM_ASR_FIELD_SCHEMAS,
    ),
    _transformers_asr_preset_profile(
        "asr_wavlm",
        TrainingFamily.CTC,
        "voicehub.models.asr_wavlm.native.WavLMForCTC",
        adapter_factory="voicehub.models.asr_wavlm.training_asr_wavlm:NativeWavLMTrainingAdapter",
        dataset_spec_factory=("voicehub.training.asr_data_contracts:build_asr_wavlm_dataset_spec"),
        field_schemas=_WAVEFORM_ASR_FIELD_SCHEMAS,
    ),
    _transformers_asr_preset_profile(
        "asr_moonshine",
        TrainingFamily.SPEECH_SEQ2SEQ,
        "voicehub.models.asr_moonshine.native.MoonshineForConditionalGeneration",
        adapter_factory="voicehub.models.asr_moonshine.training_asr_moonshine:NativeMoonshineTrainingAdapter",
        dataset_spec_factory=("voicehub.training.asr_data_contracts:build_asr_moonshine_dataset_spec"),
        field_schemas=_WAVEFORM_ASR_FIELD_SCHEMAS,
    ),
    _profile(
        "asr_seamless_m4t_v2",
        TrainingFamily.SPEECH_SEQ2SEQ,
        adapter_factory=(
            "voicehub.models.asr_seamless_m4t_v2.training_asr_seamless_m4t_v2:NativeSeamlessM4Tv2TrainingAdapter"
        ),
        dataset_spec_factory=("voicehub.training.asr_data_contracts:build_asr_seamless_m4t_v2_dataset_spec"),
        task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
        module_paths=("model", ),
        component_paths=(
            "model.speech_encoder",
            "model.text_decoder",
            "model.shared",
            "model.lm_head",
        ),
        label_names=("labels", ),
        source_entrypoints=(
            "voicehub.models.asr_seamless_m4t_v2.native.modeling:"
            "SeamlessM4Tv2ForSpeechToText",
            "voicehub.models.asr_seamless_m4t_v2."
            "training_asr_seamless_m4t_v2:"
            "NativeSeamlessM4Tv2TrainingAdapter",
        ),
        native_training=True,
        separate_optimizers=False,
        support=TrainingSupport.NATIVE,
        phases=(
            _phase(
                "speech_recognition",
                component_paths=(
                    "model.speech_encoder",
                    "model.text_decoder",
                    "model.shared",
                    "model.lm_head",
                ),
                optimizer_names=("model", ),
                forward_component="model",
                label_names=("labels", ),
                prediction_keys=("logits", ),
                loss_keys=("loss", ),
                required_inputs=(
                    "input_features",
                    "attention_mask",
                    "labels",
                ),
            ), ),
        default_phase="speech_recognition",
        field_schemas=_TIME_MAJOR_ASR_FIELD_SCHEMAS,
    ),
    _transformers_asr_preset_profile(
        "asr_faster_whisper",
        TrainingFamily.SPEECH_SEQ2SEQ,
        "voicehub.models.asr_whisper_native.native.WhisperModel",
        adapter_factory=(
            "voicehub.models.asr_whisper_native.training_asr_whisper_native:NativeWhisperTrainingAdapter"),
        dataset_spec_factory=("voicehub.training.asr_data_contracts:build_asr_faster_whisper_dataset_spec"),
        field_schemas=_CHANNEL_FIRST_ASR_FIELD_SCHEMAS,
    ),
    _transformers_asr_preset_profile(
        "asr_whisperx",
        TrainingFamily.SPEECH_SEQ2SEQ,
        "voicehub.models.asr_whisper_native.native.WhisperModel",
        adapter_factory=(
            "voicehub.models.asr_whisper_native.training_asr_whisper_native:NativeWhisperTrainingAdapter"),
        dataset_spec_factory=("voicehub.training.asr_data_contracts:build_asr_whisperx_dataset_spec"),
        field_schemas=_CHANNEL_FIRST_ASR_FIELD_SCHEMAS,
    ),
    _transformers_asr_preset_profile(
        "asr_openai_whisper",
        TrainingFamily.SPEECH_SEQ2SEQ,
        "voicehub.models.asr_whisper_native.native.WhisperModel",
        adapter_factory=(
            "voicehub.models.asr_whisper_native.training_asr_whisper_native:NativeWhisperTrainingAdapter"),
        dataset_spec_factory=("voicehub.training.asr_data_contracts:build_asr_openai_whisper_dataset_spec"),
        field_schemas=_CHANNEL_FIRST_ASR_FIELD_SCHEMAS,
    ),
    _transformers_asr_preset_profile(
        "asr_nemo",
        TrainingFamily.CTC,
        "voicehub.models.asr_nemo.native.modeling.NeMoQuartzNetForCTC",
        adapter_factory="voicehub.models.asr_nemo.training_asr_nemo:NativeNeMoCTCTrainingAdapter",
        dataset_spec_factory=("voicehub.training.asr_data_contracts:build_asr_nemo_dataset_spec"),
        field_schemas=_NEMO_CTC_FIELD_SCHEMAS,
    ),
    _profile(
        "asr_speechbrain",
        TrainingFamily.SPEECH_SEQ2SEQ,
        adapter_factory="voicehub.models.asr_native.speechbrain_training:NativeSpeechBrainASRTrainingAdapter",
        dataset_spec_factory=("voicehub.training.asr_data_contracts:build_asr_speechbrain_dataset_spec"),
        task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
        module_paths=("model", ),
        component_paths=("model", ),
        label_names=("tokens_eos", "ctc_tokens"),
        source_entrypoints=("voicehub.models.asr_speechbrain.native.modeling."
                            "SpeechBrainCRDNNForASR", ),
        native_training=True,
        support=TrainingSupport.NATIVE,
        phases=(
            _phase(
                "speech_recognition",
                component_paths=("model", ),
                optimizer_names=("model", ),
                label_names=("tokens_eos", "ctc_tokens"),
                prediction_keys=("sequence_logits", "ctc_logits"),
                loss_keys=("loss", "seq2seq_loss", "ctc_loss"),
                required_inputs=(
                    "waveforms",
                    "waveform_lengths",
                    "tokens_bos",
                    "tokens_eos",
                    "token_lengths",
                    "ctc_tokens",
                    "ctc_token_lengths",
                ),
            ), ),
        default_phase="speech_recognition",
        field_schemas=_SPEECHBRAIN_ASR_FIELD_SCHEMAS,
    ),
    _profile(
        "asr_funasr",
        TrainingFamily.CTC,
        adapter_factory="voicehub.models.asr_funasr.native.training:NativeSenseVoiceTrainingAdapter",
        dataset_spec_factory=("voicehub.training.asr_data_contracts:build_asr_funasr_dataset_spec"),
        task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
        module_paths=("model", ),
        component_paths=("model", ),
        label_names=("labels", ),
        source_entrypoints=("voicehub.models.asr_funasr.native.modeling:"
                            "SenseVoiceSmallForCTC.forward", ),
        native_training=True,
        support=TrainingSupport.NATIVE,
        phases=(
            _phase(
                "speech_recognition",
                component_paths=("model", ),
                optimizer_names=("model", ),
                label_names=("labels", ),
                prediction_keys=("logits", "log_probabilities"),
                loss_keys=("loss", "ctc", "rich"),
                required_inputs=(
                    "features",
                    "feature_lengths",
                    "labels",
                    "label_lengths",
                ),
            ), ),
        default_phase="speech_recognition",
        field_schemas={
            "audio_values": {
                "sequence_dim": -1,
                "padding_value": 0.0,
                "length_field": "audio_lengths",
            },
            "input_signal": {
                "sequence_dim": -1,
                "padding_value": 0.0,
                "length_field": "input_signal_length",
            },
            "audio": {
                "sequence_dim": -1,
                "padding_value": 0.0,
                "length_field": "audio_lengths",
            },
            "features": {
                "sequence_dim": 0,
                "padding_value": 0.0,
                "length_field": "feature_lengths",
            },
            "labels": {
                "sequence_dim": 0,
                "padding_value": -1,
                "length_field": "label_lengths",
            },
        },
    ),
    _profile(
        "asr_espnet",
        TrainingFamily.SPEECH_SEQ2SEQ,
        adapter_factory="voicehub.models.asr_espnet.native.training:NativeESPnetASRTrainingAdapter",
        dataset_spec_factory=("voicehub.training.asr_data_contracts:build_asr_espnet_dataset_spec"),
        task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
        module_paths=("model", ),
        component_paths=("model", ),
        label_names=("labels", ),
        source_entrypoints=(
            "voicehub.models.asr_espnet.native.modeling:"
            "ESPnetLibriSpeechTransformerForASR.forward",
            "voicehub.models.asr_espnet.native.training:"
            "prepare_espnet_training_batch",
            "voicehub.models.asr_espnet.native.training:"
            "NativeESPnetASRTrainingAdapter",
        ),
        native_training=True,
        support=TrainingSupport.NATIVE,
        phases=(
            _phase(
                "speech_recognition",
                component_paths=("model", ),
                optimizer_names=("model", ),
                label_names=("labels", ),
                prediction_keys=("logits", "ctc_logits"),
                loss_keys=("loss", "ctc_loss", "attention_loss"),
                required_inputs=("labels", "label_lengths"),
                frozen_component_paths=("language_model", ),
            ), ),
        default_phase="speech_recognition",
        field_schemas={
            "audio": {
                "sequence_dim": -1,
                "padding_value": 0.0,
                "length_field": "audio_lengths",
            },
            "waveforms": {
                "sequence_dim": -1,
                "padding_value": 0.0,
                "length_field": "waveform_lengths",
            },
            "features": {
                "sequence_dim": -2,
                "padding_value": 0.0,
                "length_field": "feature_lengths",
            },
            "labels": {
                "sequence_dim": -1,
                "padding_value": -1,
                "length_field": "label_lengths",
            },
        },
    ),
    _profile(
        "asr_wenet",
        TrainingFamily.SPEECH_SEQ2SEQ,
        adapter_factory="voicehub.models.asr_wenet.training_asr_wenet:NativeWeNetU2PPTrainingAdapter",
        dataset_spec_factory=("voicehub.training.asr_data_contracts:build_asr_wenet_dataset_spec"),
        task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
        module_paths=("model", ),
        component_paths=("model", ),
        label_names=("labels", ),
        source_entrypoints=("voicehub.models.asr_wenet.native.modeling.WeNetU2PPForASR", ),
        native_training=True,
        support=TrainingSupport.NATIVE,
        phases=(
            _phase(
                "speech_recognition",
                component_paths=("model", ),
                optimizer_names=("model", ),
                label_names=("labels", ),
                prediction_keys=("log_probabilities", ),
                loss_keys=("loss", "attention_loss", "ctc_loss"),
                # The native wrapper accepts either waveform inputs or cached
                # frontend features and validates that modality-specific pair.
                required_inputs=("labels", "label_lengths"),
            ), ),
        default_phase="speech_recognition",
        field_schemas=_WENET_U2PP_FIELD_SCHEMAS,
    ),
    _profile(
        "vad_transformers",
        TrainingFamily.AUDIO_CLASSIFICATION,
        adapter_factory=(
            "voicehub.models.vad_transformers.training_vad_transformers:TransformersVADTrainingAdapter"),
        task=SpeechTask.VOICE_ACTIVITY_DETECTION,
        module_paths=("model", ),
        component_paths=("model", ),
        label_names=("labels", ),
        source_entrypoints=(
            "voicehub.models.asr_wav2vec2.native."
            "Wav2Vec2ForSequenceClassification",
            "voicehub.models.asr_wav2vec2.native."
            "Wav2Vec2ForAudioFrameClassification",
        ),
        native_training=True,
        support=TrainingSupport.NATIVE,
        phases=(
            _phase(
                "voice_activity_detection",
                component_paths=("model", ),
                optimizer_names=("model", ),
                label_names=("labels", ),
                prediction_keys=("logits", ),
                loss_keys=("loss", ),
                fallback_objective="classification",
            ), ),
        default_phase="voice_activity_detection",
        field_schemas=_RAW_AUDIO_FIELD_SCHEMAS,
    ),
    _profile(
        "vad_silero",
        TrainingFamily.FRAME_CLASSIFICATION,
        adapter_factory="voicehub.models.vad_silero.training_vad_silero:NativeSileroVADTrainingAdapter",
        task=SpeechTask.VOICE_ACTIVITY_DETECTION,
        module_paths=("model", ),
        component_paths=("model", ),
        label_names=("labels", ),
        source_entrypoints=("voicehub.models.vad_silero.native.SileroVADModel", ),
        native_training=True,
        support=TrainingSupport.NATIVE,
        phases=(
            _phase(
                "voice_activity_detection",
                component_paths=("model", ),
                optimizer_names=("model", ),
                forward_component="model",
                forward_method="frame_probabilities",
                label_names=("labels", ),
                prediction_keys=("logits", ),
                loss_keys=("loss", ),
                required_inputs=("input_values", "labels"),
                fallback_objective="binary_cross_entropy_with_logits",
            ), ),
        default_phase="voice_activity_detection",
        training_default_model_name_or_path="safestack/silero-vad",
        field_schemas=_RAW_AUDIO_FIELD_SCHEMAS,
    ),
    _profile(
        "vad_webrtc",
        TrainingFamily.UPSTREAM_NATIVE,
        task=SpeechTask.VOICE_ACTIVITY_DETECTION,
        support=TrainingSupport.INFERENCE_ONLY,
    ),
    _profile(
        "vad_pyannote",
        TrainingFamily.FRAME_CLASSIFICATION,
        adapter_factory="voicehub.models.vad_pyannote.training_vad_pyannote:NativePyanNetTrainingAdapter",
        task=SpeechTask.VOICE_ACTIVITY_DETECTION,
        module_paths=("model", ),
        component_paths=("model", ),
        label_names=("labels", "y"),
        source_entrypoints=("voicehub.models.vad_pyannote.native.PyanNet", ),
        native_training=True,
        support=TrainingSupport.NATIVE,
        phases=(
            _phase(
                "segmentation",
                component_paths=("model", ),
                optimizer_names=("model", ),
                forward_component="model",
                label_names=("labels", "y"),
                prediction_keys=("logits", "probabilities"),
                loss_keys=("loss", ),
                required_inputs=("waveforms", "labels"),
                fallback_objective="binary_cross_entropy",
            ), ),
        default_phase="segmentation",
        training_default_model_name_or_path=("pyannote/voice-activity-detection"),
        field_schemas=_RAW_AUDIO_FIELD_SCHEMAS,
    ),
    _profile(
        "vad_speechbrain",
        TrainingFamily.FRAME_CLASSIFICATION,
        adapter_factory=(
            "voicehub.models.vad_speechbrain.training_vad_speechbrain:NativeSpeechBrainVADTrainingAdapter"),
        task=SpeechTask.VOICE_ACTIVITY_DETECTION,
        module_paths=("model", ),
        component_paths=("model", ),
        label_names=("labels", ),
        source_entrypoints=("voicehub.models.vad_speechbrain.native."
                            "SpeechBrainCRDNNVADModel", ),
        native_training=True,
        support=TrainingSupport.NATIVE,
        phases=(
            _phase(
                "voice_activity_detection",
                component_paths=("model", ),
                optimizer_names=("model", ),
                forward_component="model",
                label_names=("labels", ),
                prediction_keys=("logits", "speech_probabilities"),
                loss_keys=("loss", ),
                required_inputs=("waveforms", "labels"),
                fallback_objective="binary_cross_entropy_with_logits",
            ), ),
        default_phase="voice_activity_detection",
        training_default_model_name_or_path=("speechbrain/vad-crdnn-libriparty"),
        field_schemas=_RAW_AUDIO_FIELD_SCHEMAS,
    ),
    _profile(
        "vad_nemo",
        TrainingFamily.FRAME_CLASSIFICATION,
        adapter_factory="voicehub.models.vad_nemo.training_vad_nemo:NativeMarbleNetVADTrainingAdapter",
        task=SpeechTask.VOICE_ACTIVITY_DETECTION,
        module_paths=("model", ),
        component_paths=("model", ),
        label_names=("labels", ),
        source_entrypoints=("voicehub.models.vad_nemo.native.MarbleNetVADModel", ),
        native_training=True,
        support=TrainingSupport.NATIVE,
        phases=(
            _phase(
                "voice_activity_detection",
                component_paths=("model", ),
                optimizer_names=("model", ),
                forward_component="model",
                label_names=("labels", ),
                prediction_keys=("logits", "speech_probabilities"),
                loss_keys=("loss", ),
                required_inputs=("waveforms", "labels"),
                fallback_objective="classification",
            ), ),
        default_phase="voice_activity_detection",
        training_default_model_name_or_path=("nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0"),
        field_schemas=_RAW_AUDIO_FIELD_SCHEMAS,
    ),
    _profile(
        "vad_funasr",
        TrainingFamily.FRAME_CLASSIFICATION,
        adapter_factory="voicehub.models.vad_funasr.training_vad_funasr:NativeFSMNVADTrainingAdapter",
        task=SpeechTask.VOICE_ACTIVITY_DETECTION,
        module_paths=("model", ),
        component_paths=("model", ),
        label_names=("labels", ),
        source_entrypoints=("voicehub.models.vad_funasr.native.FSMNVADModel", ),
        native_training=True,
        support=TrainingSupport.NATIVE,
        phases=(
            _phase(
                "voice_activity_detection",
                component_paths=("model", ),
                optimizer_names=("model", ),
                forward_component="model",
                label_names=("labels", ),
                prediction_keys=("logits", "speech_probabilities"),
                loss_keys=("loss", ),
                required_inputs=("waveforms", "labels"),
            ), ),
        default_phase="voice_activity_detection",
        training_default_model_name_or_path="funasr/fsmn-vad",
        field_schemas=_RAW_AUDIO_FIELD_SCHEMAS,
    ),
    _profile(
        "vad_auditok",
        TrainingFamily.UPSTREAM_NATIVE,
        task=SpeechTask.VOICE_ACTIVITY_DETECTION,
        support=TrainingSupport.INFERENCE_ONLY,
    ),
    _profile(
        "vad_sherpa_onnx",
        TrainingFamily.FRAME_CLASSIFICATION,
        adapter_factory=(
            "voicehub.models.vad_sherpa_onnx.training_vad_sherpa_onnx:create_sherpa_native_vad_training_adapter"
        ),
        task=SpeechTask.VOICE_ACTIVITY_DETECTION,
        module_paths=("model", ),
        component_paths=("model", ),
        label_names=("labels", ),
        source_entrypoints=(
            "voicehub.models.vad_ten.native.TENVADModel",
            "voicehub.models.vad_silero.native.SileroVADModel",
        ),
        native_training=True,
        support=TrainingSupport.NATIVE,
        phases=(
            _phase(
                "voice_activity_detection",
                component_paths=("model", ),
                optimizer_names=("model", ),
                forward_component="model",
                label_names=("labels", ),
                prediction_keys=("logits", ),
                loss_keys=("loss", ),
                required_inputs=("labels", ),
                fallback_objective="binary_cross_entropy_with_logits",
            ), ),
        default_phase="voice_activity_detection",
        training_default_model_name_or_path="safestack/silero-vad",
        field_schemas=_RAW_AUDIO_FIELD_SCHEMAS,
    ),
    _profile(
        "vad_pyannote_segmentation",
        TrainingFamily.FRAME_CLASSIFICATION,
        adapter_factory="voicehub.models.vad_pyannote.training_vad_pyannote:NativePyanNetTrainingAdapter",
        task=SpeechTask.VOICE_ACTIVITY_DETECTION,
        module_paths=("model", ),
        component_paths=("model", ),
        label_names=("labels", "y"),
        source_entrypoints=("voicehub.models.vad_pyannote.native.PyanNet", ),
        native_training=True,
        support=TrainingSupport.NATIVE,
        phases=(
            _phase(
                "powerset_segmentation",
                component_paths=("model", ),
                optimizer_names=("model", ),
                forward_component="model",
                label_names=("labels", "y"),
                prediction_keys=("logits", "probabilities"),
                loss_keys=("loss", ),
                required_inputs=("waveforms", "labels"),
                fallback_objective="classification",
            ), ),
        default_phase="powerset_segmentation",
        training_default_model_name_or_path=("pyannote/segmentation-3.0"),
        field_schemas=_RAW_AUDIO_FIELD_SCHEMAS,
    ),
    _profile(
        "vad_pyannote_brouhaha",
        TrainingFamily.COMPOSITE,
        adapter_factory="voicehub.models.vad_pyannote.training_vad_pyannote:NativePyanNetTrainingAdapter",
        task=SpeechTask.VOICE_ACTIVITY_DETECTION,
        module_paths=("model", ),
        component_paths=("model", ),
        label_names=("labels", "y"),
        source_entrypoints=(
            "voicehub.models.vad_pyannote.native.PyanNet",
            "voicehub.models.vad_pyannote.native.objective.pyannet_loss",
        ),
        native_training=True,
        separate_optimizers=False,
        support=TrainingSupport.NATIVE,
        phases=(
            _phase(
                "vad_snr_c50",
                component_paths=("model", ),
                optimizer_names=("model", ),
                forward_component="model",
                label_names=("labels", "y"),
                prediction_keys=("probabilities", "logits"),
                loss_keys=(
                    "loss",
                    "loss_vad",
                    "loss_snr",
                    "loss_c50",
                ),
                required_inputs=("waveforms", "labels"),
                fallback_objective="auto",
            ), ),
        default_phase="vad_snr_c50",
        training_default_model_name_or_path="pyannote/brouhaha",
        field_schemas=_RAW_AUDIO_FIELD_SCHEMAS,
    ),
)

_BUILTIN_TRAINING_SPECS += _BUILTIN_AUDIO_INPUT_TRAINING_SPECS

# Standalone codec inference is public. Neural codec training remains available
# through the native graphs; no Trainer recipe is implied by registration.
_BUILTIN_TRAINING_SPECS += tuple(
    ModelTrainingSpec(
        model_type=model_type,
        family="audio-codec",
        task=SpeechTask.AUDIO_CODEC,
        support=TrainingSupport.INFERENCE_ONLY,
    ) for model_type in ("dac", "encodec"))

_DISCOVERED_MANIFEST_TRAINING_SPECS = discover_manifest_training_specs()
_CENTRAL_TRAINING_MODEL_TYPES = {spec.model_type for spec in _BUILTIN_TRAINING_SPECS}
_DUPLICATE_MANIFEST_TRAINING_MODELS = sorted(
    spec.model_type for spec in _DISCOVERED_MANIFEST_TRAINING_SPECS
    if spec.model_type in _CENTRAL_TRAINING_MODEL_TYPES)
if _DUPLICATE_MANIFEST_TRAINING_MODELS:
    raise ValueError(
        "Manifest-discovered training profiles duplicate legacy central declarations: "
        f"{_DUPLICATE_MANIFEST_TRAINING_MODELS!r}.")
_BUILTIN_TRAINING_SPECS += _DISCOVERED_MANIFEST_TRAINING_SPECS

_TRAINING_SPEC_REGISTRY: dict[str, ModelTrainingSpec] = {
    spec.model_type: spec
    for spec in _BUILTIN_TRAINING_SPECS
}
_TTS_TRAINING_SPEC_REGISTRY: dict[str, ModelTrainingSpec] = {
    model_type: spec
    for model_type, spec in _TRAINING_SPEC_REGISTRY.items() if spec.task is SpeechTask.TEXT_TO_SPEECH
}
_TRAINING_ALIASES: dict[str, str] = {}
_TRAINING_REGISTRY_LOCK = RLock()
# Historical public view: keep this TTS-only even as ASR and VAD profiles are
# registered. New task-aware callers can use ``ALL_MODEL_TRAINING_SPECS`` or
# ``list_training_specs(task=None)``.
MODEL_TRAINING_SPECS: Mapping[str, ModelTrainingSpec] = MappingProxyType(_TTS_TRAINING_SPEC_REGISTRY)
ALL_MODEL_TRAINING_SPECS: Mapping[str, ModelTrainingSpec] = MappingProxyType(_TRAINING_SPEC_REGISTRY)


def _normalize_training_identifier(model_type: str) -> str:
    if not isinstance(model_type, str) or not model_type.strip():
        raise ValueError("Training model identifiers must be non-empty strings.")
    raw = model_type.strip().lower()
    with _TRAINING_REGISTRY_LOCK:
        if raw in _TRAINING_ALIASES:
            return _TRAINING_ALIASES[raw]
    canonical = normalize_model_type(raw)
    with _TRAINING_REGISTRY_LOCK:
        return _TRAINING_ALIASES.get(canonical, canonical)


def register_training_spec(
        spec: ModelTrainingSpec,
        *,
        exist_ok: bool = False,
        aliases: Iterable[str] = (),
) -> None:
    """Register or explicitly replace a training profile.

    The read-only ``MODEL_TRAINING_SPECS`` view updates immediately,
    allowing third-party families to participate without editing
    VoiceHub's enums.
    """
    if not isinstance(spec, ModelTrainingSpec):
        raise TypeError("Training profiles must be ModelTrainingSpec instances.")
    aliases = tuple(aliases)
    if any(not isinstance(alias, str) for alias in aliases):
        raise TypeError("Training aliases must be strings.")
    normalized_aliases = [alias.strip().lower() for alias in aliases]
    if len(set(normalized_aliases)) != len(normalized_aliases):
        raise ValueError("Training aliases must not contain duplicates.")
    try:
        inference_spec = get_model_spec(spec.model_type)
    except UnknownModelError:
        inference_spec = None
    if inference_spec is not None and inference_spec.task is not spec.task:
        raise ValueError(
            f"Training profile {spec.model_type!r} declares task "
            f"{spec.task.value!r}, but its inference backend is registered "
            f"for {inference_spec.task.value!r}.")

    with _TRAINING_REGISTRY_LOCK:
        if spec.model_type in _TRAINING_ALIASES:
            target = _TRAINING_ALIASES[spec.model_type]
            raise ValueError(
                f"Training model type {spec.model_type!r} collides with an "
                f"alias for {target!r}.")
        if spec.model_type in _TRAINING_SPEC_REGISTRY and not exist_ok:
            raise ValueError(f"A training profile is already registered for "
                             f"{spec.model_type!r}.")
        validated_aliases = tuple(
            _validate_training_alias(
                alias,
                spec.model_type,
                exist_ok=exist_ok,
            ) for alias in aliases)
        _TRAINING_SPEC_REGISTRY[spec.model_type] = spec
        if spec.task is SpeechTask.TEXT_TO_SPEECH:
            _TTS_TRAINING_SPEC_REGISTRY[spec.model_type] = spec
        else:
            _TTS_TRAINING_SPEC_REGISTRY.pop(spec.model_type, None)
        for alias in validated_aliases:
            _TRAINING_ALIASES[alias] = spec.model_type


def _validate_training_alias(
    alias: str,
    canonical: str,
    *,
    exist_ok: bool,
) -> str:
    if not isinstance(alias, str) or not alias.strip():
        raise ValueError("Training aliases must be non-empty strings.")
    normalized = alias.strip().lower()
    if normalized == canonical:
        raise ValueError(f"Training alias {alias!r} is identical to its canonical model "
                         "type.")
    inference_target = normalize_model_type(normalized)
    if normalized in _TRAINING_SPEC_REGISTRY:
        raise ValueError(f"Training alias {alias!r} collides with a registered model type.")
    try:
        inference_spec = get_model_spec(normalized)
    except UnknownModelError:
        inference_spec = None
    if inference_spec is not None and inference_spec.model_type == normalized:
        raise ValueError(f"Training alias {alias!r} collides with a registered inference "
                         "model type.")
    if inference_target != normalized:
        existing_target = _TRAINING_ALIASES.get(normalized, inference_target)
        if existing_target != canonical or not exist_ok:
            raise ValueError(
                f"Training alias {alias!r} collides with an inference alias "
                f"for {inference_target!r}.")
    existing = _TRAINING_ALIASES.get(normalized)
    if existing is not None and (existing != canonical or not exist_ok):
        raise ValueError(f"Training alias {alias!r} is already registered for {existing!r}.")
    return normalized


def register_training_alias(
    alias: str,
    model_type: str,
    *,
    exist_ok: bool = False,
) -> None:
    """Register a training-only alias for a canonical profile."""
    with _TRAINING_REGISTRY_LOCK:
        canonical = _normalize_training_identifier(model_type)
        if canonical not in _TRAINING_SPEC_REGISTRY:
            raise KeyError(f"No training profile is registered for {model_type!r}.")
        normalized = _validate_training_alias(
            alias,
            canonical,
            exist_ok=exist_ok,
        )
        _TRAINING_ALIASES[normalized] = canonical


def unregister_training_alias(
    alias: str,
    *,
    missing_ok: bool = False,
) -> str | None:
    """Remove an alias and return its former canonical target."""
    if not isinstance(alias, str) or not alias.strip():
        raise ValueError("Training aliases must be non-empty strings.")
    normalized = alias.strip().lower()
    with _TRAINING_REGISTRY_LOCK:
        try:
            return _TRAINING_ALIASES.pop(normalized)
        except KeyError:
            if missing_ok:
                return None
            raise KeyError(f"No training alias is registered for {alias!r}.") from None


def unregister_training_spec(
    model_type: str,
    *,
    missing_ok: bool = False,
) -> ModelTrainingSpec | None:
    """Remove and return a dynamically registered (or built-in) profile."""
    with _TRAINING_REGISTRY_LOCK:
        canonical = _normalize_training_identifier(model_type)
        try:
            removed = _TRAINING_SPEC_REGISTRY.pop(canonical)
        except KeyError:
            if missing_ok:
                return None
            raise KeyError(f"No training profile is registered for {model_type!r}.") from None
        stale_aliases = [alias for alias, target in _TRAINING_ALIASES.items() if target == canonical]
        for alias in stale_aliases:
            del _TRAINING_ALIASES[alias]
        _TTS_TRAINING_SPEC_REGISTRY.pop(canonical, None)
        return removed


def get_training_spec(model_type: str) -> ModelTrainingSpec:
    """Resolve inference aliases and return one registered training profile."""
    canonical = _normalize_training_identifier(model_type)
    with _TRAINING_REGISTRY_LOCK:
        spec = _TRAINING_SPEC_REGISTRY.get(canonical)
    if spec is not None:
        return spec
    # Preserve the inference registry's informative error for known public
    # lookup paths while still permitting training-only future profiles.
    try:
        canonical = get_model_spec(model_type).model_type
    except UnknownModelError:
        raise KeyError(f"No training profile is registered for {model_type!r}.") from None
    with _TRAINING_REGISTRY_LOCK:
        try:
            return _TRAINING_SPEC_REGISTRY[canonical]
        except KeyError:
            raise KeyError(f"No training profile is registered for {model_type!r}.") from None


def list_training_specs(
    *,
    task: SpeechTask | str | None = SpeechTask.TEXT_TO_SPEECH,
    support: TrainingSupport | str | None = None,
) -> tuple[ModelTrainingSpec, ...]:
    """List profiles by task and, optionally, support boundary.

    Omitting ``task`` preserves the historical TTS-only result. Pass
    ``task=None`` to inspect every registered speech task.
    """
    resolved_task = None if task is None else SpeechTask.coerce(task)
    with _TRAINING_REGISTRY_LOCK:
        if resolved_task is None:
            specs = tuple(_TRAINING_SPEC_REGISTRY.values())
        elif resolved_task is SpeechTask.TEXT_TO_SPEECH:
            specs = tuple(_TTS_TRAINING_SPEC_REGISTRY.values())
        else:
            specs = tuple(spec for spec in _TRAINING_SPEC_REGISTRY.values() if spec.task is resolved_task)
    if support is None:
        return specs
    support = TrainingSupport.coerce(support)
    return tuple(spec for spec in specs if spec.support is support)
