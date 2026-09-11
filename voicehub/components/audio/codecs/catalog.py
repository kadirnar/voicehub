"""Graph-free inventory of codecs used by LLM-style TTS models.

The catalogue is deliberately declarative.  It contains immutable metadata and
``module:attribute`` import-path strings only; importing this module never
imports PyTorch or any codec/model graph.  Runtime code can therefore inspect
codec geometry, native stage coverage, and optimization boundaries before it
decides which implementation to resolve.

``CodecOptimizationManifest`` records *structural eligibility*, not a blanket
performance guarantee.  In particular, CUDA Graph entries still require the
fixed shapes and explicit stochastic inputs documented by each manifest.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping

LLM_TTS_CODEC_FEATURE = "llm-tts-codec"


class CodecRepresentation(str, Enum):
    """The tensor representation crossing the TTS/codec boundary."""

    DENSE_DISCRETE = "dense-discrete"
    HIERARCHICAL_DISCRETE = "hierarchical-discrete"
    CONTINUOUS_VAE = "continuous-vae"


class CodecStageAvailability(str, Enum):
    """How one encoder, quantizer, or decoder stage is implemented."""

    NATIVE = "native"
    NATIVE_INTEGRATED = "native-integrated"
    NATIVE_SPLIT_PIPELINE = "native-split-pipeline"
    PRECOMPUTED_ONLY = "precomputed-only"
    LEGACY_INACTIVE = "legacy-inactive"
    NOT_APPLICABLE = "not-applicable"


class CodecIntegration(str, Enum):
    """Where the codec graph sits relative to its owning TTS architecture."""

    SHARED_NATIVE_CODEC = "shared-native-codec"
    ARCHITECTURE_LOCAL_CODEC = "architecture-local-codec"
    INTEGRATED_TTS_GRAPH = "integrated-tts-graph"
    SPLIT_TTS_PIPELINE = "split-tts-pipeline"
    PRECOMPUTED_BOUNDARY = "precomputed-boundary"


class CodecStage(str, Enum):
    """Logical stage that an optimizer may target."""

    ENCODER = "encoder"
    QUANTIZER = "quantizer"
    DECODER = "decoder"
    FLOW = "flow"
    VOCODER = "vocoder"
    FULL_PIPELINE = "full-pipeline"


class CodecOptimizationSurface(str, Enum):
    """Optimization mechanisms structurally exposed by a codec graph."""

    TORCH_COMPILE = "torch-compile"
    CUDA_GRAPH = "cuda-graph"
    SNAKE = "snake"
    SNAKE_BETA = "snake-beta"


class CodecPrimitive(str, Enum):
    """Reusable graph primitives present in native codec implementations."""

    CAUSAL_CONV1D = "causal-conv1d"
    CONV_TRANSPOSE1D = "conv-transpose1d"
    CONVNEXT = "convnext"
    CONDITIONED_VITS = "conditioned-vits"
    CONDITIONAL_FLOW_MATCHING = "conditional-flow-matching"
    EUCLIDEAN_CODEBOOK = "euclidean-codebook"
    FINITE_SCALAR_QUANTIZER = "finite-scalar-quantizer"
    GAUSSIAN_POSTERIOR = "gaussian-posterior"
    HIFIGAN = "hifigan"
    HIFT = "hift"
    HUBERT_SEMANTIC_ENCODER = "hubert-semantic-encoder"
    ISTFT = "istft"
    LSTM = "lstm"
    LOOKUP_FREE_QUANTIZER = "lookup-free-quantizer"
    RESIDUAL_FINITE_SCALAR_QUANTIZER = "residual-finite-scalar-quantizer"
    RESIDUAL_VECTOR_QUANTIZER = "residual-vector-quantizer"
    SEANET = "seanet"
    SNAKE = "snake"
    SNAKE_BETA = "snake-beta"
    SPLIT_RESIDUAL_VECTOR_QUANTIZER = "split-residual-vector-quantizer"
    SSL_FEATURE_FRONTEND = "ssl-feature-frontend"
    STREAMING_CACHE = "streaming-cache"
    TRANSFORMER = "transformer"


@dataclass(frozen=True, slots=True)
class CodecOwnerBinding:
    """One active model-registry owner of a codec family."""

    model_type: str
    architecture_id: str
    variant: str

    def __post_init__(self) -> None:
        for name in ("model_type", "architecture_id", "variant"):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise TypeError(f"{name} must be a string.")
            value = value.strip()
            if not value:
                raise ValueError(f"{name} must be a non-empty string.")
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class CodecStageManifest:
    """Native implementation status for the three canonical codec stages."""

    encoder: CodecStageAvailability
    quantizer: CodecStageAvailability
    decoder: CodecStageAvailability

    def __post_init__(self) -> None:
        for name in ("encoder", "quantizer", "decoder"):
            value = getattr(self, name)
            if not isinstance(value, CodecStageAvailability):
                try:
                    value = CodecStageAvailability(value)
                except (TypeError, ValueError) as error:
                    raise TypeError(
                        f"{name} must be a CodecStageAvailability value.") from error
                object.__setattr__(self, name, value)

    @property
    def has_native_encoder_decoder(self) -> bool:
        """Whether encoder and decoder graphs are both natively implemented."""
        return (
            self.encoder is CodecStageAvailability.NATIVE
            and self.decoder is CodecStageAvailability.NATIVE
        )

    @property
    def is_full_native_codec(self) -> bool:
        """Whether every applicable canonical stage is a standalone native graph."""
        return (
            self.has_native_encoder_decoder
            and self.quantizer in (
                CodecStageAvailability.NATIVE,
                CodecStageAvailability.NOT_APPLICABLE,
            )
        )


@dataclass(frozen=True, slots=True)
class CodecPrimitiveManifest:
    """Stage-specific primitives, suitable for graph-free capability discovery."""

    encoder: tuple[CodecPrimitive, ...] = ()
    quantizer: tuple[CodecPrimitive, ...] = ()
    decoder: tuple[CodecPrimitive, ...] = ()
    auxiliary: tuple[CodecPrimitive, ...] = ()

    def __post_init__(self) -> None:
        for name in ("encoder", "quantizer", "decoder", "auxiliary"):
            values = tuple(getattr(self, name))
            if any(not isinstance(value, CodecPrimitive) for value in values):
                raise TypeError(
                    f"{name} primitives must be CodecPrimitive values.")
            if len(values) != len(set(values)):
                raise ValueError(f"{name} primitives must not contain duplicates.")
            object.__setattr__(self, name, values)

    @property
    def all(self) -> tuple[CodecPrimitive, ...]:
        """Return every primitive once, preserving stage order."""
        return tuple(
            dict.fromkeys(
                (*self.encoder, *self.quantizer, *self.decoder, *self.auxiliary)))


@dataclass(frozen=True, slots=True)
class CodecOptimizationManifest:
    """Structurally valid optimization surfaces and their safe stage targets."""

    surfaces: tuple[CodecOptimizationSurface, ...]
    compile_targets: tuple[CodecStage, ...] = ()
    cuda_graph_targets: tuple[CodecStage, ...] = ()
    cuda_graph_constraints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        surfaces = tuple(self.surfaces)
        compile_targets = tuple(self.compile_targets)
        cuda_graph_targets = tuple(self.cuda_graph_targets)
        constraints = tuple(self.cuda_graph_constraints)
        if any(
            not isinstance(value, CodecOptimizationSurface)
            for value in surfaces
        ):
            raise TypeError(
                "surfaces must contain CodecOptimizationSurface values.")
        if any(not isinstance(value, CodecStage) for value in compile_targets):
            raise TypeError("compile_targets must contain CodecStage values.")
        if any(not isinstance(value, CodecStage) for value in cuda_graph_targets):
            raise TypeError("cuda_graph_targets must contain CodecStage values.")
        for name, values in (
            ("surfaces", surfaces),
            ("compile_targets", compile_targets),
            ("cuda_graph_targets", cuda_graph_targets),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must not contain duplicates.")
        if compile_targets and CodecOptimizationSurface.TORCH_COMPILE not in surfaces:
            raise ValueError(
                "compile_targets require the torch-compile optimization surface.")
        if cuda_graph_targets and CodecOptimizationSurface.CUDA_GRAPH not in surfaces:
            raise ValueError(
                "cuda_graph_targets require the cuda-graph optimization surface.")
        if any(not isinstance(value, str) or not value.strip()
               for value in constraints):
            raise ValueError(
                "cuda_graph_constraints must contain non-empty strings.")
        object.__setattr__(self, "surfaces", surfaces)
        object.__setattr__(self, "compile_targets", compile_targets)
        object.__setattr__(self, "cuda_graph_targets", cuda_graph_targets)
        object.__setattr__(
            self,
            "cuda_graph_constraints",
            tuple(value.strip() for value in constraints),
        )


@dataclass(frozen=True, slots=True)
class CodecCatalogEntry:
    """One canonical codec family/variant and all active TTS owners."""

    codec_id: str
    family: str
    variant: str
    owners: tuple[CodecOwnerBinding, ...]
    representation: CodecRepresentation
    integration: CodecIntegration
    stages: CodecStageManifest
    stochastic_vae: bool
    separable_autoencoder: bool
    implementation_paths: tuple[str, ...]
    primitives: CodecPrimitiveManifest
    optimization: CodecOptimizationManifest
    aliases: tuple[str, ...] = ()
    gaps: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        codec_id = _normalize_public_id(self.codec_id, name="codec_id")
        family = self.family.strip() if isinstance(self.family, str) else ""
        variant = self.variant.strip() if isinstance(self.variant, str) else ""
        if not family:
            raise ValueError("family must be a non-empty string.")
        if not variant:
            raise ValueError("variant must be a non-empty string.")
        owners = tuple(self.owners)
        if not owners or any(not isinstance(owner, CodecOwnerBinding)
                             for owner in owners):
            raise TypeError(
                "owners must contain one or more CodecOwnerBinding values.")
        owner_types = tuple(owner.model_type for owner in owners)
        if len(owner_types) != len(set(owner_types)):
            raise ValueError("owners must not repeat a model type.")
        try:
            representation = CodecRepresentation(self.representation)
            integration = CodecIntegration(self.integration)
        except (TypeError, ValueError) as error:
            raise TypeError(
                "representation and integration must be codec enum values.") from error
        if not isinstance(self.stages, CodecStageManifest):
            raise TypeError("stages must be a CodecStageManifest.")
        for name in ("stochastic_vae", "separable_autoencoder"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean.")
        if self.stochastic_vae and representation is not CodecRepresentation.CONTINUOUS_VAE:
            raise ValueError(
                "stochastic_vae is only valid for continuous-VAE codecs.")
        if self.separable_autoencoder and not self.stages.has_native_encoder_decoder:
            raise ValueError(
                "separable_autoencoder requires native encoder and decoder stages.")
        paths = tuple(self.implementation_paths)
        if not paths:
            raise ValueError("implementation_paths must not be empty.")
        for path in paths:
            _validate_import_path(path)
        if len(paths) != len(set(paths)):
            raise ValueError("implementation_paths must not contain duplicates.")
        if not isinstance(self.primitives, CodecPrimitiveManifest):
            raise TypeError("primitives must be a CodecPrimitiveManifest.")
        if not isinstance(self.optimization, CodecOptimizationManifest):
            raise TypeError("optimization must be a CodecOptimizationManifest.")
        aliases = tuple(
            _normalize_public_id(alias, name="alias") for alias in self.aliases)
        if codec_id in aliases:
            raise ValueError("aliases must not repeat codec_id.")
        if len(aliases) != len(set(aliases)):
            raise ValueError("aliases must not contain duplicates.")
        gaps = tuple(self.gaps)
        if any(not isinstance(gap, str) or not gap.strip() for gap in gaps):
            raise ValueError("gaps must contain non-empty strings.")
        object.__setattr__(self, "codec_id", codec_id)
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "variant", variant)
        object.__setattr__(self, "owners", owners)
        object.__setattr__(self, "representation", representation)
        object.__setattr__(self, "integration", integration)
        object.__setattr__(self, "implementation_paths", paths)
        object.__setattr__(self, "aliases", aliases)
        object.__setattr__(
            self,
            "gaps",
            tuple(gap.strip() for gap in gaps),
        )

    @property
    def owner_model_types(self) -> tuple[str, ...]:
        """Return the active ``voicehub.registry`` model types using this codec."""
        return tuple(owner.model_type for owner in self.owners)

    @property
    def has_explicit_gap(self) -> bool:
        """Whether the inventory records an unresolved native boundary."""
        return bool(self.gaps)


_PUBLIC_ID_PATTERN = re.compile(r"[^a-z0-9]+")


def _normalize_public_id(value: str, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    normalized = _PUBLIC_ID_PATTERN.sub("-", value.strip().lower()).strip("-")
    if not normalized:
        raise ValueError(f"{name} must be a non-empty identifier.")
    return normalized


def _validate_import_path(path: str) -> None:
    if not isinstance(path, str):
        raise TypeError("implementation paths must be strings.")
    module, separator, attribute = path.strip().partition(":")
    if not separator or not module or not attribute:
        raise ValueError(
            "implementation paths must use 'module:attribute' syntax.")
    if any(not segment.isidentifier() for segment in module.split(".")):
        raise ValueError(f"Invalid implementation module in {path!r}.")
    if any(not segment.isidentifier() for segment in attribute.split(".")):
        raise ValueError(f"Invalid implementation attribute in {path!r}.")


def _owner(
    model_type: str,
    architecture_id: str,
    variant: str,
) -> CodecOwnerBinding:
    return CodecOwnerBinding(model_type, architecture_id, variant)


def _stages(
    encoder: CodecStageAvailability,
    quantizer: CodecStageAvailability,
    decoder: CodecStageAvailability,
) -> CodecStageManifest:
    return CodecStageManifest(encoder, quantizer, decoder)


def _primitives(
    *,
    encoder: tuple[CodecPrimitive, ...] = (),
    quantizer: tuple[CodecPrimitive, ...] = (),
    decoder: tuple[CodecPrimitive, ...] = (),
    auxiliary: tuple[CodecPrimitive, ...] = (),
) -> CodecPrimitiveManifest:
    return CodecPrimitiveManifest(encoder, quantizer, decoder, auxiliary)


def _optimization(
    *surfaces: CodecOptimizationSurface,
    compile_targets: tuple[CodecStage, ...],
    cuda_graph_targets: tuple[CodecStage, ...] = (),
    cuda_graph_constraints: tuple[str, ...] = (),
) -> CodecOptimizationManifest:
    return CodecOptimizationManifest(
        surfaces=surfaces,
        compile_targets=compile_targets,
        cuda_graph_targets=cuda_graph_targets,
        cuda_graph_constraints=cuda_graph_constraints,
    )


_ENTRIES = (
    CodecCatalogEntry(
        codec_id="snac-24khz",
        family="SNAC",
        variant="24-kHz hierarchical Orpheus tokenizer",
        owners=(
            _owner(
                "orpheustts",
                "causal-lm",
                "SNAC 24-kHz three-level / seven-code LM protocol",
            ),
        ),
        representation=CodecRepresentation.HIERARCHICAL_DISCRETE,
        integration=CodecIntegration.ARCHITECTURE_LOCAL_CODEC,
        stages=_stages(
            CodecStageAvailability.NATIVE,
            CodecStageAvailability.NATIVE,
            CodecStageAvailability.NATIVE,
        ),
        stochastic_vae=False,
        separable_autoencoder=True,
        implementation_paths=(
            "voicehub.models.orpheustts.source.snac.snac:SNAC",
        ),
        primitives=_primitives(
            encoder=(
                CodecPrimitive.CAUSAL_CONV1D,
                CodecPrimitive.SNAKE,
            ),
            quantizer=(CodecPrimitive.RESIDUAL_VECTOR_QUANTIZER, ),
            decoder=(
                CodecPrimitive.CONV_TRANSPOSE1D,
                CodecPrimitive.SNAKE,
            ),
        ),
        optimization=_optimization(
            CodecOptimizationSurface.TORCH_COMPILE,
            CodecOptimizationSurface.CUDA_GRAPH,
            CodecOptimizationSurface.SNAKE,
            compile_targets=(
                CodecStage.ENCODER,
                CodecStage.QUANTIZER,
                CodecStage.DECODER,
            ),
            cuda_graph_targets=(
                CodecStage.ENCODER,
                CodecStage.DECODER,
            ),
            cuda_graph_constraints=(
                "Capture requires fixed waveform and hierarchy-level lengths.",
                "Whole-pipeline capture must keep hierarchical list structure static.",
            ),
        ),
        aliases=("orpheus-snac", "snac"),
        gaps=(
            "Orpheus is active in MODEL_REGISTRY through the generic causal-lm "
            "architecture rather than a dedicated architecture declaration.",
        ),
    ),
    CodecCatalogEntry(
        codec_id="dac-native",
        family="DAC",
        variant="checkpoint-configured Descript Audio Codec",
        owners=(
            _owner("dia", "dia", "checkpoint-configured DAC"),
            _owner("outetts", "outetts", "24-kHz two-codebook DAC"),
            _owner(
                "parlertts",
                "parlertts",
                "checkpoint-configured delayed-codebook DAC",
            ),
            _owner("zonos", "zonos", "44.1-kHz nine-codebook DAC"),
            _owner("zonos2", "zonos2", "44.1-kHz nine-codebook DAC"),
        ),
        representation=CodecRepresentation.DENSE_DISCRETE,
        integration=CodecIntegration.SHARED_NATIVE_CODEC,
        stages=_stages(
            CodecStageAvailability.NATIVE,
            CodecStageAvailability.NATIVE,
            CodecStageAvailability.NATIVE,
        ),
        stochastic_vae=False,
        separable_autoencoder=True,
        implementation_paths=(
            "voicehub.components.audio.codecs.dac.model.dac:DAC",
            "voicehub.models.dac.native.modeling:DacModel",
            "voicehub.models.parlertts.native.modeling:ParlerDacAudioEncoder",
            "voicehub.models.zonos.native.codec:ZonosDACCodec",
        ),
        primitives=_primitives(
            encoder=(
                CodecPrimitive.CAUSAL_CONV1D,
                CodecPrimitive.SNAKE,
            ),
            quantizer=(CodecPrimitive.RESIDUAL_VECTOR_QUANTIZER, ),
            decoder=(
                CodecPrimitive.CONV_TRANSPOSE1D,
                CodecPrimitive.SNAKE,
            ),
        ),
        optimization=_optimization(
            CodecOptimizationSurface.TORCH_COMPILE,
            CodecOptimizationSurface.CUDA_GRAPH,
            CodecOptimizationSurface.SNAKE,
            compile_targets=(
                CodecStage.ENCODER,
                CodecStage.QUANTIZER,
                CodecStage.DECODER,
            ),
            cuda_graph_targets=(
                CodecStage.ENCODER,
                CodecStage.DECODER,
            ),
            cuda_graph_constraints=(
                "Capture is per checkpoint configuration and fixed frame geometry.",
                "Codebook count must not change after capture.",
            ),
        ),
        aliases=("dac", "descript-audio-codec"),
    ),
    CodecCatalogEntry(
        codec_id="encodec-bark",
        family="EnCodec",
        variant="Bark 24-kHz tokenizer",
        owners=(
            _owner("bark", "bark", "Bark semantic/coarse/fine EnCodec boundary"),
        ),
        representation=CodecRepresentation.DENSE_DISCRETE,
        integration=CodecIntegration.SHARED_NATIVE_CODEC,
        stages=_stages(
            CodecStageAvailability.NATIVE,
            CodecStageAvailability.NATIVE,
            CodecStageAvailability.NATIVE,
        ),
        stochastic_vae=False,
        separable_autoencoder=True,
        implementation_paths=(
            "voicehub.components.audio.codecs.encodec.model:EncodecModel",
        ),
        primitives=_primitives(
            encoder=(
                CodecPrimitive.SEANET,
                CodecPrimitive.LSTM,
            ),
            quantizer=(CodecPrimitive.RESIDUAL_VECTOR_QUANTIZER, ),
            decoder=(
                CodecPrimitive.SEANET,
                CodecPrimitive.LSTM,
            ),
        ),
        optimization=_optimization(
            CodecOptimizationSurface.TORCH_COMPILE,
            CodecOptimizationSurface.CUDA_GRAPH,
            compile_targets=(
                CodecStage.ENCODER,
                CodecStage.QUANTIZER,
                CodecStage.DECODER,
            ),
            cuda_graph_targets=(CodecStage.DECODER, ),
            cuda_graph_constraints=(
                "Capture is decoder-only through the validation-free tensor "
                "reconstruction boundary; validate encoded frames before capture.",
                "Codebook count, frame count, scale presence, dtype, and device "
                "must remain fixed across replay.",
                "Encoder bandwidth selection and segmented overlap-add remain "
                "outside CUDA Graph capture.",
            ),
        ),
        aliases=("bark-encodec", "encodec"),
        gaps=(
            "Bark's runtime optimization target list does not yet expose codec "
            "decode explicitly; CUDA Graph support is limited to the codec's "
            "inner decoder tensor boundary.",
        ),
    ),
    CodecCatalogEntry(
        codec_id="mimi-native",
        family="Mimi",
        variant="CSM / ConversationTTS architecture-local graphs",
        owners=(
            _owner("csm", "csm", "Moshi Mimi split-RVQ codec"),
            _owner(
                "conversationtts",
                "conversationtts",
                "ConversationTTS Mimi codec",
            ),
        ),
        representation=CodecRepresentation.DENSE_DISCRETE,
        integration=CodecIntegration.ARCHITECTURE_LOCAL_CODEC,
        stages=_stages(
            CodecStageAvailability.NATIVE,
            CodecStageAvailability.NATIVE,
            CodecStageAvailability.NATIVE,
        ),
        stochastic_vae=False,
        separable_autoencoder=True,
        implementation_paths=(
            "voicehub.models.csm.source.moshi.models.compression:MimiModel",
            "voicehub.models.conversationtts.source.conversationtts.tools."
            "tokenizer.MimiCodec.mimi_tokenizer:MimiTokenizer",
            "voicehub.models.conversationtts.source.conversationtts.tools."
            "tokenizer.MimiCodec.model.models.MimiCodec:MimiCodec",
        ),
        primitives=_primitives(
            encoder=(
                CodecPrimitive.SEANET,
                CodecPrimitive.TRANSFORMER,
            ),
            quantizer=(CodecPrimitive.SPLIT_RESIDUAL_VECTOR_QUANTIZER, ),
            decoder=(
                CodecPrimitive.TRANSFORMER,
                CodecPrimitive.SEANET,
            ),
        ),
        optimization=_optimization(
            CodecOptimizationSurface.TORCH_COMPILE,
            CodecOptimizationSurface.CUDA_GRAPH,
            compile_targets=(
                CodecStage.ENCODER,
                CodecStage.QUANTIZER,
                CodecStage.DECODER,
            ),
            cuda_graph_targets=(
                CodecStage.ENCODER,
                CodecStage.DECODER,
            ),
            cuda_graph_constraints=(
                "Capture requires a fixed streaming chunk and cache geometry.",
            ),
        ),
        aliases=("mimi", "mimi-codec"),
        gaps=(
            "CSM and ConversationTTS retain separate architecture-local Mimi "
            "graphs instead of one shared codec implementation.",
        ),
    ),
    CodecCatalogEntry(
        codec_id="xcodec2-llasa",
        family="XCodec2",
        variant="LLaSA single-codebook semantic/acoustic tokenizer",
        owners=(
            _owner("llasa", "llasa", "native frozen XCodec2"),
        ),
        representation=CodecRepresentation.DENSE_DISCRETE,
        integration=CodecIntegration.ARCHITECTURE_LOCAL_CODEC,
        stages=_stages(
            CodecStageAvailability.NATIVE,
            CodecStageAvailability.NATIVE,
            CodecStageAvailability.NATIVE,
        ),
        stochastic_vae=False,
        separable_autoencoder=True,
        implementation_paths=(
            "voicehub.models.llasa.xcodec2:XCodec2Model",
        ),
        primitives=_primitives(
            encoder=(
                CodecPrimitive.CAUSAL_CONV1D,
                CodecPrimitive.SNAKE_BETA,
                CodecPrimitive.TRANSFORMER,
            ),
            quantizer=(CodecPrimitive.FINITE_SCALAR_QUANTIZER, ),
            decoder=(
                CodecPrimitive.TRANSFORMER,
                CodecPrimitive.ISTFT,
                CodecPrimitive.SNAKE_BETA,
            ),
        ),
        optimization=_optimization(
            CodecOptimizationSurface.TORCH_COMPILE,
            CodecOptimizationSurface.CUDA_GRAPH,
            CodecOptimizationSurface.SNAKE_BETA,
            compile_targets=(
                CodecStage.ENCODER,
                CodecStage.QUANTIZER,
                CodecStage.DECODER,
            ),
            cuda_graph_targets=(
                CodecStage.ENCODER,
                CodecStage.DECODER,
            ),
            cuda_graph_constraints=(
                "Capture requires fixed semantic-feature and acoustic frame lengths.",
            ),
        ),
        aliases=("llasa-xcodec2", "xcodec2"),
    ),
    CodecCatalogEntry(
        codec_id="neucodec",
        family="NeuCodec",
        variant="NeuTTS XCodec2-derived tokenizer",
        owners=(
            _owner("neutts", "neutts", "native frozen NeuCodec"),
        ),
        representation=CodecRepresentation.DENSE_DISCRETE,
        integration=CodecIntegration.ARCHITECTURE_LOCAL_CODEC,
        stages=_stages(
            CodecStageAvailability.NATIVE,
            CodecStageAvailability.NATIVE,
            CodecStageAvailability.NATIVE,
        ),
        stochastic_vae=False,
        separable_autoencoder=True,
        implementation_paths=(
            "voicehub.models.neutts.native.neucodec:NeuCodecModel",
        ),
        primitives=_primitives(
            encoder=(
                CodecPrimitive.CAUSAL_CONV1D,
                CodecPrimitive.SNAKE_BETA,
                CodecPrimitive.TRANSFORMER,
            ),
            quantizer=(CodecPrimitive.FINITE_SCALAR_QUANTIZER, ),
            decoder=(
                CodecPrimitive.TRANSFORMER,
                CodecPrimitive.ISTFT,
                CodecPrimitive.SNAKE_BETA,
            ),
        ),
        optimization=_optimization(
            CodecOptimizationSurface.TORCH_COMPILE,
            CodecOptimizationSurface.CUDA_GRAPH,
            CodecOptimizationSurface.SNAKE_BETA,
            compile_targets=(
                CodecStage.ENCODER,
                CodecStage.QUANTIZER,
                CodecStage.DECODER,
            ),
            cuda_graph_targets=(
                CodecStage.ENCODER,
                CodecStage.DECODER,
            ),
            cuda_graph_constraints=(
                "Capture requires fixed semantic-feature and acoustic frame lengths.",
            ),
        ),
        aliases=("neutts-neucodec", "neu-codec"),
    ),
    CodecCatalogEntry(
        codec_id="moss-audio-tokenizer",
        family="MOSS Audio Tokenizer",
        variant="native v1 and v2 streaming tokenizers",
        owners=(
            _owner("mosstts", "moss-tts", "MOSS tokenizer v1 / v2"),
        ),
        representation=CodecRepresentation.DENSE_DISCRETE,
        integration=CodecIntegration.ARCHITECTURE_LOCAL_CODEC,
        stages=_stages(
            CodecStageAvailability.NATIVE,
            CodecStageAvailability.NATIVE,
            CodecStageAvailability.NATIVE,
        ),
        stochastic_vae=False,
        separable_autoencoder=True,
        implementation_paths=(
            "voicehub.models.mosstts.native.codec:NativeMossAudioCodec",
            "voicehub.models.mosstts.native.codec_modeling_v1:"
            "MossAudioTokenizerV1Model",
            "voicehub.models.mosstts.native.codec_modeling:"
            "MossAudioTokenizerV2Model",
        ),
        primitives=_primitives(
            encoder=(
                CodecPrimitive.TRANSFORMER,
                CodecPrimitive.STREAMING_CACHE,
            ),
            quantizer=(
                CodecPrimitive.RESIDUAL_VECTOR_QUANTIZER,
                CodecPrimitive.LOOKUP_FREE_QUANTIZER,
            ),
            decoder=(
                CodecPrimitive.TRANSFORMER,
                CodecPrimitive.STREAMING_CACHE,
            ),
        ),
        optimization=_optimization(
            CodecOptimizationSurface.TORCH_COMPILE,
            compile_targets=(
                CodecStage.ENCODER,
                CodecStage.QUANTIZER,
                CodecStage.DECODER,
            ),
        ),
        aliases=("moss-codec", "mosstts-codec"),
        gaps=(
            "CUDA Graph capture is not advertised because the active streaming "
            "boundaries materialize CUDA values on the host.",
        ),
    ),
    CodecCatalogEntry(
        codec_id="qwen3-tts-tokenizer-12hz",
        family="Qwen3-TTS Speech Tokenizer",
        variant="12-Hz tokenizer v2",
        owners=(
            _owner(
                "qwen3tts",
                "qwen3-tts",
                "Qwen3-TTS 12-Hz 16-codebook tokenizer",
            ),
        ),
        representation=CodecRepresentation.DENSE_DISCRETE,
        integration=CodecIntegration.ARCHITECTURE_LOCAL_CODEC,
        stages=_stages(
            CodecStageAvailability.NATIVE,
            CodecStageAvailability.NATIVE,
            CodecStageAvailability.NATIVE,
        ),
        stochastic_vae=False,
        separable_autoencoder=True,
        implementation_paths=(
            "voicehub.models.qwen3tts.native.encoder:Qwen3TTSSpeechEncoder",
            "voicehub.models.qwen3tts.native.codec:Qwen3TTSSpeechDecoder",
        ),
        primitives=_primitives(
            encoder=(
                CodecPrimitive.SEANET,
                CodecPrimitive.TRANSFORMER,
            ),
            quantizer=(
                CodecPrimitive.SPLIT_RESIDUAL_VECTOR_QUANTIZER,
                CodecPrimitive.EUCLIDEAN_CODEBOOK,
            ),
            decoder=(
                CodecPrimitive.TRANSFORMER,
                CodecPrimitive.CONV_TRANSPOSE1D,
                CodecPrimitive.SNAKE_BETA,
            ),
        ),
        optimization=_optimization(
            CodecOptimizationSurface.TORCH_COMPILE,
            CodecOptimizationSurface.CUDA_GRAPH,
            CodecOptimizationSurface.SNAKE_BETA,
            compile_targets=(
                CodecStage.ENCODER,
                CodecStage.QUANTIZER,
                CodecStage.DECODER,
            ),
            cuda_graph_targets=(CodecStage.DECODER, ),
            cuda_graph_constraints=(
                "Capture is decoder-only through the validation-free tensor "
                "boundary; validate code range before capture.",
                "Capture requires fixed code length, 16 codebooks, dtype, and device.",
                "Dynamic chunk lists remain outside whole-decoder capture.",
                "Encoder frame-length materialization remains outside CUDA Graph capture.",
            ),
        ),
        aliases=(
            "qwen3-tts-codec",
            "qwen3-tts-speech-tokenizer",
            "qwen3tts-codec",
            "qwen-codec",
        ),
    ),
    CodecCatalogEntry(
        codec_id="vibevoice-acoustic-tokenizer",
        family="VibeVoice Acoustic Tokenizer",
        variant="continuous stochastic acoustic VAE",
        owners=(
            _owner(
                "vibevoice",
                "vibevoice-tts",
                "full non-streaming codec / realtime decoder-only variant",
            ),
        ),
        representation=CodecRepresentation.CONTINUOUS_VAE,
        integration=CodecIntegration.ARCHITECTURE_LOCAL_CODEC,
        stages=_stages(
            CodecStageAvailability.NATIVE,
            CodecStageAvailability.NOT_APPLICABLE,
            CodecStageAvailability.NATIVE,
        ),
        stochastic_vae=True,
        separable_autoencoder=True,
        implementation_paths=(
            "voicehub.models.vibevoice.native.codec:VibeVoiceAcousticTokenizer",
            "voicehub.models.vibevoice.native.codec:VibeVoiceSemanticTokenizer",
        ),
        primitives=_primitives(
            encoder=(
                CodecPrimitive.CAUSAL_CONV1D,
                CodecPrimitive.TRANSFORMER,
                CodecPrimitive.GAUSSIAN_POSTERIOR,
            ),
            decoder=(
                CodecPrimitive.TRANSFORMER,
                CodecPrimitive.CONV_TRANSPOSE1D,
            ),
            auxiliary=(CodecPrimitive.STREAMING_CACHE, ),
        ),
        optimization=_optimization(
            CodecOptimizationSurface.TORCH_COMPILE,
            CodecOptimizationSurface.CUDA_GRAPH,
            compile_targets=(
                CodecStage.ENCODER,
                CodecStage.DECODER,
            ),
            cuda_graph_targets=(
                CodecStage.ENCODER,
                CodecStage.DECODER,
            ),
            cuda_graph_constraints=(
                "Provide explicit posterior noise for exact repeatable capture.",
                "Streaming cache shapes and chunk sizes must remain fixed.",
            ),
        ),
        aliases=("vibevoice-codec", "vibevoice-vae"),
        gaps=(
            "The realtime checkpoint exposes decoder-side acoustic reconstruction "
            "only; full native encode/decode belongs to non-realtime variants.",
        ),
    ),
    CodecCatalogEntry(
        codec_id="voxcpm-audiovae-v2",
        family="VoxCPM AudioVAE",
        variant="AudioVAE V2 continuous tokenizer",
        owners=(
            _owner("voxcpm", "voxcpm2", "AudioVAE V2"),
        ),
        representation=CodecRepresentation.CONTINUOUS_VAE,
        integration=CodecIntegration.ARCHITECTURE_LOCAL_CODEC,
        stages=_stages(
            CodecStageAvailability.NATIVE,
            CodecStageAvailability.NOT_APPLICABLE,
            CodecStageAvailability.NATIVE,
        ),
        stochastic_vae=True,
        separable_autoencoder=True,
        implementation_paths=(
            "voicehub.models.voxcpm.native.codec:VoxCPMAudioVAE",
        ),
        primitives=_primitives(
            encoder=(
                CodecPrimitive.CAUSAL_CONV1D,
                CodecPrimitive.GAUSSIAN_POSTERIOR,
                CodecPrimitive.SNAKE,
            ),
            decoder=(
                CodecPrimitive.CONV_TRANSPOSE1D,
                CodecPrimitive.SNAKE,
            ),
        ),
        optimization=_optimization(
            CodecOptimizationSurface.TORCH_COMPILE,
            CodecOptimizationSurface.CUDA_GRAPH,
            CodecOptimizationSurface.SNAKE,
            compile_targets=(
                CodecStage.ENCODER,
                CodecStage.DECODER,
            ),
            cuda_graph_targets=(
                CodecStage.ENCODER,
                CodecStage.DECODER,
            ),
            cuda_graph_constraints=(
                "Use deterministic posterior means or supply explicit posterior noise.",
                "Decoder noise tensors must be explicit and fixed during capture.",
            ),
        ),
        aliases=("voxcpm-codec", "voxcpm-vae", "audiovae-v2"),
        gaps=(
            "The official codec weights require one digest-pinned legacy-pickle "
            "conversion before the steady-state Safetensors path.",
        ),
    ),
    CodecCatalogEntry(
        codec_id="higgs-audio-tokenizer-v2",
        family="Higgs Audio Tokenizer",
        variant="v2 semantic/acoustic fused tokenizer",
        owners=(
            _owner("omnivoice", "omnivoice", "OmniVoice tokenizer integration"),
            _owner(
                "higgstts",
                "higgs_audio_v2",
                "Higgs Audio v2 tokenizer integration",
            ),
        ),
        representation=CodecRepresentation.DENSE_DISCRETE,
        integration=CodecIntegration.ARCHITECTURE_LOCAL_CODEC,
        stages=_stages(
            CodecStageAvailability.NATIVE,
            CodecStageAvailability.NATIVE,
            CodecStageAvailability.NATIVE,
        ),
        stochastic_vae=False,
        separable_autoencoder=True,
        implementation_paths=(
            "voicehub.models.omnivoice.native.codec:HiggsAudioV2Tokenizer",
            "voicehub.models.higgstts.native.tokenizer:"
            "HiggsAudioV2TokenizerModel",
        ),
        primitives=_primitives(
            encoder=(
                CodecPrimitive.HUBERT_SEMANTIC_ENCODER,
                CodecPrimitive.CAUSAL_CONV1D,
                CodecPrimitive.SNAKE,
            ),
            quantizer=(
                CodecPrimitive.RESIDUAL_VECTOR_QUANTIZER,
                CodecPrimitive.EUCLIDEAN_CODEBOOK,
            ),
            decoder=(
                CodecPrimitive.CONV_TRANSPOSE1D,
                CodecPrimitive.SNAKE,
            ),
        ),
        optimization=_optimization(
            CodecOptimizationSurface.TORCH_COMPILE,
            CodecOptimizationSurface.SNAKE,
            compile_targets=(
                CodecStage.ENCODER,
                CodecStage.QUANTIZER,
                CodecStage.DECODER,
            ),
        ),
        aliases=("higgs-codec", "higgs-tokenizer", "omnivoice-codec"),
        gaps=(
            "OmniVoice and Higgs TTS retain duplicate native tokenizer graphs "
            "instead of one shared implementation.",
            "CUDA Graph capture is not advertised because public code validation "
            "extracts CUDA min/max values on the host.",
        ),
    ),
    CodecCatalogEntry(
        codec_id="fish-modified-dac",
        family="ModifiedDAC",
        variant="Fish Speech S2 causal ten-codebook tokenizer",
        owners=(
            _owner("fishtts", "fish-s2", "S2 ModifiedDAC"),
        ),
        representation=CodecRepresentation.DENSE_DISCRETE,
        integration=CodecIntegration.ARCHITECTURE_LOCAL_CODEC,
        stages=_stages(
            CodecStageAvailability.NATIVE,
            CodecStageAvailability.NATIVE,
            CodecStageAvailability.NATIVE,
        ),
        stochastic_vae=False,
        separable_autoencoder=True,
        implementation_paths=(
            "voicehub.models.fishtts.native.codec:FishModifiedDAC",
        ),
        primitives=_primitives(
            encoder=(
                CodecPrimitive.CAUSAL_CONV1D,
                CodecPrimitive.CONVNEXT,
                CodecPrimitive.TRANSFORMER,
                CodecPrimitive.SNAKE,
            ),
            quantizer=(CodecPrimitive.RESIDUAL_VECTOR_QUANTIZER, ),
            decoder=(
                CodecPrimitive.TRANSFORMER,
                CodecPrimitive.CONV_TRANSPOSE1D,
                CodecPrimitive.SNAKE,
            ),
        ),
        optimization=_optimization(
            CodecOptimizationSurface.TORCH_COMPILE,
            CodecOptimizationSurface.CUDA_GRAPH,
            CodecOptimizationSurface.SNAKE,
            compile_targets=(
                CodecStage.ENCODER,
                CodecStage.QUANTIZER,
                CodecStage.DECODER,
            ),
            cuda_graph_targets=(
                CodecStage.ENCODER,
                CodecStage.DECODER,
            ),
            cuda_graph_constraints=(
                "Capture requires fixed causal windows and ten-codebook geometry.",
            ),
        ),
        aliases=("fish-codec", "fishtts-codec", "modified-dac"),
        gaps=(
            "The released codec checkpoint is not published as Safetensors and "
            "therefore uses the audited one-time legacy converter.",
        ),
    ),
    CodecCatalogEntry(
        codec_id="fluac-vui",
        family="Fluac",
        variant="VUI residual-FSQ tokenizer",
        owners=(
            _owner("vui", "vui", "Fluac residual-FSQ codec"),
        ),
        representation=CodecRepresentation.DENSE_DISCRETE,
        integration=CodecIntegration.ARCHITECTURE_LOCAL_CODEC,
        stages=_stages(
            CodecStageAvailability.NATIVE,
            CodecStageAvailability.NATIVE,
            CodecStageAvailability.NATIVE,
        ),
        stochastic_vae=False,
        separable_autoencoder=True,
        implementation_paths=(
            "voicehub.models.vui.fluac:Fluac",
        ),
        primitives=_primitives(
            encoder=(
                CodecPrimitive.CAUSAL_CONV1D,
                CodecPrimitive.SNAKE,
            ),
            quantizer=(CodecPrimitive.RESIDUAL_FINITE_SCALAR_QUANTIZER, ),
            decoder=(
                CodecPrimitive.CONV_TRANSPOSE1D,
                CodecPrimitive.SNAKE,
            ),
        ),
        optimization=_optimization(
            CodecOptimizationSurface.TORCH_COMPILE,
            CodecOptimizationSurface.CUDA_GRAPH,
            CodecOptimizationSurface.SNAKE,
            compile_targets=(
                CodecStage.ENCODER,
                CodecStage.QUANTIZER,
                CodecStage.DECODER,
            ),
            cuda_graph_targets=(
                CodecStage.ENCODER,
                CodecStage.DECODER,
            ),
            cuda_graph_constraints=(
                "Capture requires fixed waveform and residual-FSQ frame lengths.",
            ),
        ),
        aliases=("fluac", "vui-codec"),
    ),
    CodecCatalogEntry(
        codec_id="chatterbox-s3gen",
        family="Chatterbox S3",
        variant="S3Tokenizer v2 plus S3Gen flow/HiFT decoder",
        owners=(
            _owner(
                "chatterbox",
                "chatterbox",
                "S3Tokenizer v2 / conditional-flow / HiFT pipeline",
            ),
        ),
        representation=CodecRepresentation.DENSE_DISCRETE,
        integration=CodecIntegration.SPLIT_TTS_PIPELINE,
        stages=_stages(
            CodecStageAvailability.NATIVE,
            CodecStageAvailability.NATIVE,
            CodecStageAvailability.NATIVE_SPLIT_PIPELINE,
        ),
        stochastic_vae=False,
        separable_autoencoder=False,
        implementation_paths=(
            "voicehub.models.chatterbox.models.s3tokenizer.model_v2:"
            "S3TokenizerV2",
            "voicehub.models.chatterbox.models.s3gen.s3gen:S3Token2Wav",
        ),
        primitives=_primitives(
            encoder=(CodecPrimitive.TRANSFORMER, ),
            quantizer=(CodecPrimitive.FINITE_SCALAR_QUANTIZER, ),
            decoder=(
                CodecPrimitive.CONDITIONAL_FLOW_MATCHING,
                CodecPrimitive.HIFT,
                CodecPrimitive.HIFIGAN,
                CodecPrimitive.SNAKE,
            ),
        ),
        optimization=_optimization(
            CodecOptimizationSurface.TORCH_COMPILE,
            CodecOptimizationSurface.CUDA_GRAPH,
            CodecOptimizationSurface.SNAKE,
            compile_targets=(
                CodecStage.ENCODER,
                CodecStage.QUANTIZER,
                CodecStage.FLOW,
                CodecStage.VOCODER,
            ),
            cuda_graph_targets=(
                CodecStage.FLOW,
                CodecStage.VOCODER,
            ),
            cuda_graph_constraints=(
                "Capture denoiser and vocoder stages separately with fixed lengths.",
                "Solver loops, random initial noise, and dynamic prompt lengths stay outside capture.",
            ),
        ),
        aliases=("chatterbox-codec", "s3gen", "s3tokenizer"),
        gaps=(
            "Waveform reconstruction is a split conditional-flow and HiFT "
            "pipeline, not a standalone codec decoder API.",
        ),
    ),
    CodecCatalogEntry(
        codec_id="gpt-sovits-s2",
        family="GPT-SoVITS S2",
        variant="conditioned RVQ/VITS acoustic stage",
        owners=(
            _owner("gptsovits", "gptsovits", "integrated semantic S2 codec"),
        ),
        representation=CodecRepresentation.DENSE_DISCRETE,
        integration=CodecIntegration.INTEGRATED_TTS_GRAPH,
        stages=_stages(
            CodecStageAvailability.NATIVE_INTEGRATED,
            CodecStageAvailability.NATIVE_INTEGRATED,
            CodecStageAvailability.NATIVE_INTEGRATED,
        ),
        stochastic_vae=False,
        separable_autoencoder=False,
        implementation_paths=(
            "voicehub.models.gptsovits.native.modeling:GPTSoVITSSynthesizer",
            "voicehub.models.gptsovits.native.quantizer:"
            "ResidualVectorQuantizer",
        ),
        primitives=_primitives(
            encoder=(CodecPrimitive.SSL_FEATURE_FRONTEND, ),
            quantizer=(CodecPrimitive.RESIDUAL_VECTOR_QUANTIZER, ),
            decoder=(
                CodecPrimitive.CONDITIONED_VITS,
                CodecPrimitive.HIFIGAN,
            ),
        ),
        optimization=_optimization(
            CodecOptimizationSurface.TORCH_COMPILE,
            CodecOptimizationSurface.CUDA_GRAPH,
            compile_targets=(
                CodecStage.QUANTIZER,
                CodecStage.DECODER,
            ),
            cuda_graph_targets=(CodecStage.DECODER, ),
            cuda_graph_constraints=(
                "Capture requires fixed phoneme, reference-spectrogram, and SSL-feature lengths.",
            ),
        ),
        aliases=("gptsovits-codec", "gpt-sovits-codec", "sovits-s2"),
        gaps=(
            "The native S2 graph extracts codes from prepared SSL features; it "
            "does not expose a standalone raw-waveform encoder.",
        ),
    ),
    CodecCatalogEntry(
        codec_id="xtts2-dvae",
        family="XTTS2 DVAE",
        variant="native single-codebook DVAE plus HiFiGAN runtime decoder",
        owners=(
            _owner(
                "xtts",
                "xtts2",
                "DVAE tokenizer / separate GPT-conditioned HiFiGAN runtime",
            ),
        ),
        representation=CodecRepresentation.DENSE_DISCRETE,
        integration=CodecIntegration.SPLIT_TTS_PIPELINE,
        stages=_stages(
            CodecStageAvailability.NATIVE,
            CodecStageAvailability.NATIVE,
            CodecStageAvailability.NATIVE,
        ),
        stochastic_vae=False,
        separable_autoencoder=True,
        implementation_paths=(
            "voicehub.models.xtts.native.dvae:XTTS2DVAE",
            "voicehub.models.xtts.native.decoder:HifiDecoder",
        ),
        primitives=_primitives(
            encoder=(CodecPrimitive.CAUSAL_CONV1D, ),
            quantizer=(CodecPrimitive.EUCLIDEAN_CODEBOOK, ),
            decoder=(
                CodecPrimitive.CONV_TRANSPOSE1D,
                CodecPrimitive.HIFIGAN,
            ),
        ),
        optimization=_optimization(
            CodecOptimizationSurface.TORCH_COMPILE,
            CodecOptimizationSurface.CUDA_GRAPH,
            compile_targets=(
                CodecStage.ENCODER,
                CodecStage.QUANTIZER,
                CodecStage.DECODER,
                CodecStage.VOCODER,
            ),
            cuda_graph_targets=(
                CodecStage.ENCODER,
                CodecStage.DECODER,
                CodecStage.VOCODER,
            ),
            cuda_graph_constraints=(
                "Capture requires fixed mel/code lengths and reference conditioning.",
                "DVAE decode and GPT-conditioned HiFiGAN are separate capture targets.",
            ),
        ),
        aliases=("xtts-codec", "xtts-dvae", "xtts2-codec"),
        gaps=(
            "Official DVAE weights and mel statistics are standalone legacy "
            "pickle artifacts and require explicit one-time Safetensors conversion.",
        ),
    ),
    CodecCatalogEntry(
        codec_id="cosyvoice-speech-tokenizer",
        family="CosyVoice Speech Tokenizer",
        variant="native S3Tokenizer v3 plus flow/HiFT decoder",
        owners=(
            _owner(
                "cosyvoice",
                "cosyvoice-native",
                "S3Tokenizer v3 / flow-matching / HiFT pipeline",
            ),
        ),
        representation=CodecRepresentation.DENSE_DISCRETE,
        integration=CodecIntegration.SPLIT_TTS_PIPELINE,
        stages=_stages(
            CodecStageAvailability.NATIVE,
            CodecStageAvailability.NATIVE,
            CodecStageAvailability.NATIVE_SPLIT_PIPELINE,
        ),
        stochastic_vae=False,
        separable_autoencoder=False,
        implementation_paths=(
            "voicehub.models.cosyvoice.native.speech_tokenizer:"
            "CosyVoiceSpeechTokenizer",
            "voicehub.models.cosyvoice.native.flow:"
            "CosyVoiceFlowMatchingModel",
            "voicehub.models.cosyvoice.native.vocoder:"
            "CosyVoiceHiFTGenerator",
        ),
        primitives=_primitives(
            encoder=(
                CodecPrimitive.CAUSAL_CONV1D,
                CodecPrimitive.TRANSFORMER,
            ),
            quantizer=(CodecPrimitive.FINITE_SCALAR_QUANTIZER, ),
            decoder=(
                CodecPrimitive.CONDITIONAL_FLOW_MATCHING,
                CodecPrimitive.HIFT,
                CodecPrimitive.HIFIGAN,
                CodecPrimitive.SNAKE,
            ),
        ),
        optimization=_optimization(
            CodecOptimizationSurface.TORCH_COMPILE,
            CodecOptimizationSurface.CUDA_GRAPH,
            CodecOptimizationSurface.SNAKE,
            compile_targets=(
                CodecStage.ENCODER,
                CodecStage.QUANTIZER,
                CodecStage.FLOW,
                CodecStage.VOCODER,
            ),
            cuda_graph_targets=(
                CodecStage.ENCODER,
                CodecStage.FLOW,
                CodecStage.VOCODER,
            ),
            cuda_graph_constraints=(
                "Capture tokenization with fixed mel-frame and padding geometry.",
                "Capture flow and HiFT stages separately with fixed token/mel lengths.",
                "Solver loops and any random initial conditions stay outside capture.",
            ),
        ),
        aliases=("cosyvoice-codec", "cosyvoice-tokenizer"),
        gaps=(
            "The published speech tokenizer is an immutable ONNX artifact and "
            "requires an audited one-time conversion to strict Safetensors; "
            "steady-state execution is fully native PyTorch.",
            "Waveform decode is a split conditional-flow and HiFT pipeline rather "
            "than a standalone codec decoder.",
        ),
    ),
)


REGISTERED_LLM_TTS_MODEL_TYPES = tuple(
    owner.model_type
    for entry in _ENTRIES
    for owner in entry.owners
)
"""Exact active model-registry inventory covered by this catalogue."""


def _build_catalog() -> tuple[
    Mapping[str, CodecCatalogEntry],
    Mapping[str, str],
    Mapping[str, str],
]:
    catalog: dict[str, CodecCatalogEntry] = {}
    aliases: dict[str, str] = {}
    owners: dict[str, str] = {}

    def add_alias(alias: str, codec_id: str) -> None:
        normalized = _normalize_public_id(alias, name="alias")
        previous = aliases.get(normalized)
        if previous is not None and previous != codec_id:
            raise RuntimeError(
                f"Codec alias {alias!r} maps to both {previous!r} and "
                f"{codec_id!r}.")
        aliases[normalized] = codec_id

    for entry in _ENTRIES:
        if entry.codec_id in catalog:
            raise RuntimeError(f"Duplicate codec ID {entry.codec_id!r}.")
        catalog[entry.codec_id] = entry
        add_alias(entry.codec_id, entry.codec_id)
        add_alias(entry.family, entry.codec_id)
        for alias in entry.aliases:
            add_alias(alias, entry.codec_id)
        for owner in entry.owners:
            if owner.model_type in owners:
                raise RuntimeError(
                    f"Model type {owner.model_type!r} owns more than one codec "
                    "catalogue entry.")
            owners[owner.model_type] = entry.codec_id
            add_alias(owner.model_type, entry.codec_id)
            add_alias(owner.architecture_id, entry.codec_id)

    expected = set(REGISTERED_LLM_TTS_MODEL_TYPES)
    actual = set(owners)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise RuntimeError(
            "Codec catalogue owner inventory does not match the active LLM-TTS "
            f"model inventory; missing={missing}, extra={extra}.")

    return (
        MappingProxyType(catalog),
        MappingProxyType(aliases),
        MappingProxyType(owners),
    )


CODEC_CATALOG, CODEC_ALIASES, _MODEL_TYPE_TO_CODEC = _build_catalog()
"""Immutable canonical inventory and alias map."""


def normalize_codec_id(codec_id: str) -> str:
    """Normalize a codec identifier or alias to its canonical catalogue ID."""
    normalized = _normalize_public_id(codec_id, name="codec_id")
    return CODEC_ALIASES.get(normalized, normalized)


def get_codec_entry(codec_id: str) -> CodecCatalogEntry:
    """Return one codec entry by canonical ID, family, alias, or owner model type."""
    canonical = normalize_codec_id(codec_id)
    try:
        return CODEC_CATALOG[canonical]
    except KeyError:
        choices = ", ".join(CODEC_CATALOG)
        raise KeyError(
            f"Unknown codec {codec_id!r}. Available codecs: {choices}.") from None


def list_codec_entries(
    *,
    model_type: str | None = None,
    representation: CodecRepresentation | str | None = None,
    integration: CodecIntegration | str | None = None,
    has_gap: bool | None = None,
) -> tuple[CodecCatalogEntry, ...]:
    """List entries in stable order with optional graph-free filters."""
    entries = tuple(CODEC_CATALOG.values())
    if model_type is not None:
        normalized_model_type = _normalize_public_id(
            model_type,
            name="model_type",
        )
        codec_id = _MODEL_TYPE_TO_CODEC.get(normalized_model_type)
        entries = () if codec_id is None else (CODEC_CATALOG[codec_id], )
    if representation is not None:
        representation = CodecRepresentation(representation)
        entries = tuple(
            entry for entry in entries
            if entry.representation is representation)
    if integration is not None:
        integration = CodecIntegration(integration)
        entries = tuple(
            entry for entry in entries if entry.integration is integration)
    if has_gap is not None:
        if not isinstance(has_gap, bool):
            raise TypeError("has_gap must be a boolean or None.")
        entries = tuple(
            entry for entry in entries
            if entry.has_explicit_gap is has_gap)
    return entries


def get_codec_entries_for_model(model_type: str) -> tuple[CodecCatalogEntry, ...]:
    """Return the single codec-family entry owned by an active LLM-TTS model."""
    return list_codec_entries(model_type=model_type)


def get_codec_primitive_manifest(codec_id: str) -> CodecPrimitiveManifest:
    """Return the stage primitive manifest for one codec."""
    return get_codec_entry(codec_id).primitives


def list_codec_primitive_manifests(
) -> tuple[tuple[str, CodecPrimitiveManifest], ...]:
    """Return ``(codec_id, manifest)`` pairs in canonical catalogue order."""
    return tuple(
        (entry.codec_id, entry.primitives)
        for entry in CODEC_CATALOG.values())


def list_registered_llm_tts_codec_model_types() -> tuple[str, ...]:
    """Derive the active codec-LM inventory from architecture traits."""
    from voicehub.runtime import get_architecture_spec
    from voicehub.registry import list_model_specs
    from voicehub.tasks import SpeechTask

    output = []
    for model in list_model_specs(task=SpeechTask.TEXT_TO_SPEECH):
        if model.architecture is None:
            continue
        architecture = get_architecture_spec(model.architecture)
        if architecture.capabilities.has_feature(LLM_TTS_CODEC_FEATURE):
            output.append(model.model_type)
    return tuple(output)


def validate_codec_catalog_registry_coverage() -> None:
    """Fail when live codec-LM traits and catalog owner bindings diverge."""
    expected = set(list_registered_llm_tts_codec_model_types())
    actual = {
        owner.model_type
        for entry in CODEC_CATALOG.values()
        for owner in entry.owners
    }
    if actual != expected:
        raise RuntimeError(
            "Codec catalog does not match the active llm-tts-codec "
            f"architecture inventory; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}."
        )


# Explicit aliases keep the public API concise while retaining searchable names.
get_codec_catalog_entry = get_codec_entry
list_codec_catalog_entries = list_codec_entries


__all__ = [
    "CODEC_ALIASES",
    "CODEC_CATALOG",
    "LLM_TTS_CODEC_FEATURE",
    "REGISTERED_LLM_TTS_MODEL_TYPES",
    "CodecCatalogEntry",
    "CodecIntegration",
    "CodecOptimizationManifest",
    "CodecOptimizationSurface",
    "CodecOwnerBinding",
    "CodecPrimitive",
    "CodecPrimitiveManifest",
    "CodecRepresentation",
    "CodecStage",
    "CodecStageAvailability",
    "CodecStageManifest",
    "get_codec_catalog_entry",
    "get_codec_entries_for_model",
    "get_codec_entry",
    "get_codec_primitive_manifest",
    "list_codec_catalog_entries",
    "list_codec_entries",
    "list_codec_primitive_manifests",
    "list_registered_llm_tts_codec_model_types",
    "normalize_codec_id",
    "validate_codec_catalog_registry_coverage",
]
