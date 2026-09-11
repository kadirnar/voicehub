"""Opt-in, source-backed optimization profiles for major TTS families.

Profiles translate architecture-specific recipes into ordinary
``TrainingArguments`` and deterministic dataset batching.  They do not
mutate models or silently override an existing training run: callers
choose a profile, inspect it, and pass the returned objects to
``Trainer``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, Any, cast

from voicehub.dependencies import resolve_import_path
from voicehub.training.arguments import TrainingArguments
from voicehub.training.data_contracts import TTSDataArchitecture, get_tts_dataset_spec
from voicehub.training.specs import get_training_spec
from voicehub.training.tts_batching import TTSBatchingConfig, TTSBatchingStrategy
from voicehub.training.tts_datasets import TTSDataset

if TYPE_CHECKING:
    from voicehub.optimization import OptimizationPass

_VITS_SOURCE = ("https://github.com/jaywalnut310/vits/tree/"
                "2e561ba58618d021b5b8323d3765880f7e0ecfdb")
_CONVERSATIONTTS_SOURCE = (
    "https://github.com/Audio-Foundation-Models/ConversationTTS/tree/"
    "b3851f70c2dc0d35ba609734b08915637fe2a733")
_QWEN3TTS_SOURCE = ("https://github.com/QwenLM/Qwen3-TTS/tree/"
                    "022e286b98fbec7e1e916cb940cdf532cd9f488e")
_F5TTS_SOURCE = ("https://github.com/SWivid/F5-TTS/tree/"
                 "9c614e9657089213efc6a7421b30630be138a3f5")


def _arguments(
    defaults: dict[str, Any],
    output_dir: str,
    overrides: dict[str, Any],
) -> TrainingArguments:
    if not isinstance(output_dir, str) or not output_dir.strip():
        raise ValueError("`output_dir` must be a non-empty string.")
    values = dict(defaults)
    values.update(overrides)
    values["output_dir"] = output_dir
    return TrainingArguments(**values)


def _batching(
    config: TTSBatchingConfig,
    overrides: dict[str, Any],
) -> TTSBatchingConfig:
    if not overrides:
        return config
    return replace(config, **overrides)


@dataclass(frozen=True, slots=True)
class VITSOptimizationConfig:
    """Original-VITS optimizer, segment, and length-bucket profile."""

    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.8
    adam_beta2: float = 0.99
    adam_epsilon: float = 1e-9
    lr_decay_per_epoch: float = 0.999875
    max_grad_norm: float = 1.0
    per_device_train_batch_size: int = 64
    length_field: str = "num_frames"
    bucket_boundaries: tuple[int, ...] = (
        32,
        300,
        400,
        500,
        600,
        700,
        800,
        900,
        1_000,
    )
    use_fp16: bool = True
    fused_adamw: bool = True
    compile_adamw: bool = False

    source_url: str = _VITS_SOURCE
    techniques: tuple[str, ...] = (
        "posterior-latent segment decoding",
        "sequential discriminator and generator updates",
        "spectrogram-length bucket batching",
        "separate fused AdamW optimizers when CUDA supports them",
        "optional torch.compile optimizer-step fusion",
        "epoch-normalized exponential learning-rate decay",
        "automatic mixed precision",
    )

    def training_arguments(
        self,
        output_dir: str,
        **overrides: Any,
    ) -> TrainingArguments:
        """Build source-style arguments.

        Gradient accumulation is one because the native adversarial
        recipe steps the discriminator before recomputing the generator
        phase.
        """
        defaults = {
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": 1,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "adam_beta1": self.adam_beta1,
            "adam_beta2": self.adam_beta2,
            "adam_epsilon": self.adam_epsilon,
            "max_grad_norm": self.max_grad_norm,
            "lr_scheduler_type": "exponential",
            "lr_scheduler_gamma": self.lr_decay_per_epoch,
            "fp16": self.use_fp16,
            "adamw_fused": self.fused_adamw,
            "adamw_torch_compile": self.compile_adamw,
        }
        return _arguments(defaults, output_dir, overrides)

    def batching_config(self, **overrides: Any) -> TTSBatchingConfig:
        return _batching(
            TTSBatchingConfig(
                strategy=TTSBatchingStrategy.LENGTH_BUCKET,
                length_field=self.length_field,
                bucket_boundaries=self.bucket_boundaries,
            ),
            overrides,
        )

    def prepare_dataset(
        self,
        dataset: TTSDataset,
        **batching_overrides: Any,
    ) -> TTSDataset:
        if not isinstance(dataset, TTSDataset):
            raise TypeError("VITS optimization requires a TTSDataset.")
        if dataset.architecture is not TTSDataArchitecture.VITS:
            raise ValueError("VITS optimization requires a VITS-architecture dataset.")
        return dataset.with_batching(self.batching_config(**batching_overrides))

    def acceleration_plan(
        self,
        *,
        kernel_backend: str = "auto",
        use_torch_compile: bool = True,
        compile_backend: str = "inductor",
        compile_mode: str | None = None,
        compile_fullgraph: bool = False,
        compile_dynamic: bool | None = None,
        compile_requirement: str = "auto",
        cuda_graphs: str | bool = "disabled",
    ) -> tuple[OptimizationPass, ...]:
        """Build the explicit VITS kernel and compile plan."""
        from voicehub.training.tts_acceleration import vits_acceleration_plan

        return vits_acceleration_plan(
            kernel_backend=kernel_backend,
            use_torch_compile=use_torch_compile,
            compile_backend=compile_backend,
            compile_mode=compile_mode,
            compile_fullgraph=compile_fullgraph,
            compile_dynamic=compile_dynamic,
            compile_requirement=compile_requirement,
            cuda_graphs=cuda_graphs,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LLMTTSOptimizationConfig:
    """Codec-language-model optimization profile.

    The default values reproduce ConversationTTS's published pretraining
    policy. ``qwen3tts()`` returns the distinct official Qwen3-TTS SFT
    settings instead of conflating the two recipes.
    """

    recipe: str = "conversationtts"
    learning_rate: float = 1e-5
    weight_decay: float = 0.05
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    per_device_train_batch_size: int = 64
    token_budget: int = 7_500
    max_sequence_length: int = 2_048
    length_field: str = "num_tokens"
    use_bf16: bool = True
    gradient_checkpointing: bool = False
    fused_adamw: bool = True
    lr_scheduler_type: str = "cosine"
    source_url: str = _CONVERSATIONTTS_SOURCE
    techniques: tuple[str, ...] = (
        "offline frozen-codec tokenization",
        "token-budget length batching",
        "scaled dot-product attention",
        "norm-excluded weight decay",
        "fused AdamW when CUDA supports it",
        "cosine decay with warmup",
        "bfloat16 mixed precision",
    )

    @classmethod
    def qwen3tts(cls) -> LLMTTSOptimizationConfig:
        """Return the official 12 Hz Qwen3-TTS single-speaker SFT profile."""
        return cls(
            recipe="qwen3tts",
            learning_rate=2e-6,
            weight_decay=0.01,
            adam_beta1=0.9,
            adam_beta2=0.999,
            warmup_ratio=0.0,
            gradient_accumulation_steps=4,
            per_device_train_batch_size=32,
            token_budget=7_500,
            max_sequence_length=2_048,
            lr_scheduler_type="constant",
            source_url=_QWEN3TTS_SOURCE,
            techniques=(
                "offline 12 Hz multi-codebook targets",
                "scaled dot-product attention",
                "last-hidden-state-only auxiliary prediction",
                "fused AdamW when CUDA supports it",
                "bfloat16 mixed precision",
                "gradient accumulation",
            ),
        )

    def training_arguments(
        self,
        output_dir: str,
        **overrides: Any,
    ) -> TrainingArguments:
        defaults = {
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "adam_beta1": self.adam_beta1,
            "adam_beta2": self.adam_beta2,
            "adam_epsilon": self.adam_epsilon,
            "max_grad_norm": self.max_grad_norm,
            "lr_scheduler_type": self.lr_scheduler_type,
            "warmup_ratio": self.warmup_ratio,
            "bf16": self.use_bf16,
            "gradient_checkpointing": self.gradient_checkpointing,
            "adamw_fused": self.fused_adamw,
        }
        return _arguments(defaults, output_dir, overrides)

    def batching_config(self, **overrides: Any) -> TTSBatchingConfig:
        return _batching(
            TTSBatchingConfig(
                strategy=TTSBatchingStrategy.MAX_UNITS,
                length_field=self.length_field,
                max_batch_units=self.token_budget,
                max_samples=self.per_device_train_batch_size,
                max_sequence_length=self.max_sequence_length,
                budget_mode="sum",
            ),
            overrides,
        )

    def prepare_dataset(
        self,
        dataset: TTSDataset,
        **batching_overrides: Any,
    ) -> TTSDataset:
        if not isinstance(dataset, TTSDataset):
            raise TypeError("LLM-TTS optimization requires a TTSDataset.")
        if dataset.architecture is not TTSDataArchitecture.CODEC_LM:
            raise ValueError("LLM-TTS optimization requires a codec-LM dataset.")
        return dataset.with_batching(self.batching_config(**batching_overrides))

    def acceleration_plan(
        self,
        *,
        kernel_backend: str = "auto",
        attention_policy: str = "auto",
        use_torch_compile: bool = True,
        compile_backend: str = "inductor",
        compile_mode: str | None = "max-autotune-no-cudagraphs",
        compile_fullgraph: bool = False,
        compile_dynamic: bool | None = True,
        compile_requirement: str = "auto",
    ) -> tuple[OptimizationPass, ...]:
        """Build the explicit LLM-TTS kernel, FA4, and compile plan."""
        from voicehub.training.tts_acceleration import llm_tts_acceleration_plan

        return llm_tts_acceleration_plan(
            kernel_backend=kernel_backend,
            attention_policy=attention_policy,
            use_torch_compile=use_torch_compile,
            compile_backend=compile_backend,
            compile_mode=compile_mode,
            compile_fullgraph=compile_fullgraph,
            compile_dynamic=compile_dynamic,
            compile_requirement=compile_requirement,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DiffusionTTSOptimizationConfig:
    """F5-style diffusion/flow-matching optimization profile."""

    learning_rate: float = 7.5e-5
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    warmup_steps: int = 20_000
    max_grad_norm: float = 1.0
    per_device_train_batch_size: int = 64
    frame_budget: int = 38_400
    length_field: str = "num_frames"
    use_bf16: bool = True
    gradient_checkpointing: bool = True
    fused_adamw: bool = True
    ema_decay: float = 0.9999
    ema_update_after_step: int = 0
    ema_update_every: int = 1
    source_url: str = _F5TTS_SOURCE
    techniques: tuple[str, ...] = (
        "mel-frame-budget batching",
        "activation checkpointing",
        "scaled dot-product attention",
        "fused AdamW when CUDA supports it",
        "linear warmup and decay",
        "optimizer-update-coupled EMA",
        "bfloat16 mixed precision",
    )

    def training_arguments(
        self,
        output_dir: str,
        **overrides: Any,
    ) -> TrainingArguments:
        defaults = {
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "adam_beta1": self.adam_beta1,
            "adam_beta2": self.adam_beta2,
            "adam_epsilon": self.adam_epsilon,
            "max_grad_norm": self.max_grad_norm,
            "lr_scheduler_type": "linear",
            "warmup_steps": self.warmup_steps,
            "bf16": self.use_bf16,
            "gradient_checkpointing": self.gradient_checkpointing,
            "adamw_fused": self.fused_adamw,
        }
        return _arguments(defaults, output_dir, overrides)

    def batching_config(self, **overrides: Any) -> TTSBatchingConfig:
        return _batching(
            TTSBatchingConfig(
                strategy=TTSBatchingStrategy.MAX_UNITS,
                length_field=self.length_field,
                max_batch_units=self.frame_budget,
                max_samples=self.per_device_train_batch_size,
                budget_mode="sum",
            ),
            overrides,
        )

    def prepare_dataset(
        self,
        dataset: TTSDataset,
        **batching_overrides: Any,
    ) -> TTSDataset:
        if not isinstance(dataset, TTSDataset):
            raise TypeError("Diffusion-TTS optimization requires a TTSDataset.")
        if dataset.architecture is not TTSDataArchitecture.DIFFUSION:
            raise ValueError("Diffusion-TTS optimization requires a diffusion dataset.")
        return dataset.with_batching(self.batching_config(**batching_overrides))

    def model_config_overrides(self) -> dict[str, Any]:
        """Return the F5 runtime settings coupled to this training profile."""
        return {
            "use_ema": True,
            "ema_decay": self.ema_decay,
            "ema_update_after_step": self.ema_update_after_step,
            "ema_update_every": self.ema_update_every,
            "architecture": {
                "checkpoint_activations": self.gradient_checkpointing,
            },
        }

    def acceleration_plan(
        self,
        *,
        kernel_backend: str = "auto",
        attention_policy: str = "auto",
        use_torch_compile: bool = True,
        compile_backend: str = "inductor",
        compile_mode: str | None = "max-autotune-no-cudagraphs",
        compile_fullgraph: bool = False,
        compile_dynamic: bool | None = True,
        compile_requirement: str = "auto",
    ) -> tuple[OptimizationPass, ...]:
        """Build the explicit diffusion kernel, FA4, and compile plan."""
        from voicehub.training.tts_acceleration import diffusion_tts_acceleration_plan

        return diffusion_tts_acceleration_plan(
            kernel_backend=kernel_backend,
            attention_policy=attention_policy,
            use_torch_compile=use_torch_compile,
            compile_backend=compile_backend,
            compile_mode=compile_mode,
            compile_fullgraph=compile_fullgraph,
            compile_dynamic=compile_dynamic,
            compile_requirement=compile_requirement,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


TTSTrainingOptimizationProfile = (
    VITSOptimizationConfig | LLMTTSOptimizationConfig | DiffusionTTSOptimizationConfig)
TTSOptimizationProfile = TTSTrainingOptimizationProfile

_ARCHITECTURE_PROFILE_FACTORIES = {
    TTSDataArchitecture.VITS: "voicehub.training.tts_optimization:VITSOptimizationConfig",
    TTSDataArchitecture.CODEC_LM: "voicehub.training.tts_optimization:LLMTTSOptimizationConfig",
    TTSDataArchitecture.DIFFUSION: "voicehub.training.tts_optimization:DiffusionTTSOptimizationConfig",
}
_PROFILE_METHODS = (
    "acceleration_plan",
    "batching_config",
    "prepare_dataset",
    "to_dict",
    "training_arguments",
)


def _instantiate_profile(factory_path: str) -> TTSTrainingOptimizationProfile:
    try:
        target = resolve_import_path(factory_path)
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        raise ImportError(
            f"Could not resolve TTS training optimization profile factory "
            f"{factory_path!r}: {exc}") from exc
    if not callable(target):
        raise TypeError(
            f"TTS training optimization profile factory {factory_path!r} "
            "must resolve to a callable.")
    profile = target()
    missing = tuple(name for name in _PROFILE_METHODS if not callable(getattr(profile, name, None)))
    if missing:
        raise TypeError(
            f"TTS training optimization profile factory {factory_path!r} "
            "returned an incompatible object; missing callable(s): " + ", ".join(missing))
    return cast(TTSTrainingOptimizationProfile, profile)


def get_tts_training_optimization_profile(
        model_type_or_architecture: str | TTSDataArchitecture) -> TTSTrainingOptimizationProfile:
    """Resolve the applicable special profile for a model or architecture."""
    model_type = None
    if isinstance(model_type_or_architecture, TTSDataArchitecture):
        architecture = model_type_or_architecture
        factory_path = _ARCHITECTURE_PROFILE_FACTORIES.get(architecture)
    elif isinstance(model_type_or_architecture, str):
        normalized = model_type_or_architecture.strip().lower().replace("_", "-")
        if normalized in {"llm", "codec-lm"}:
            architecture = TTSDataArchitecture.CODEC_LM
            factory_path = _ARCHITECTURE_PROFILE_FACTORIES[architecture]
        elif normalized in {"diffusion", "flow", "flow-matching"}:
            architecture = TTSDataArchitecture.DIFFUSION
            factory_path = _ARCHITECTURE_PROFILE_FACTORIES[architecture]
        elif normalized == "gan":
            architecture = TTSDataArchitecture.VITS
            factory_path = _ARCHITECTURE_PROFILE_FACTORIES[architecture]
        else:
            dataset_spec = get_tts_dataset_spec(model_type_or_architecture)
            architecture = dataset_spec.architecture
            model_type = dataset_spec.model_type
            training_spec = get_training_spec(model_type or model_type_or_architecture)
            factory_path = training_spec.optimization_profile_factory
    else:
        raise TypeError("TTS optimization target must be a model type or data architecture.")

    if factory_path is None and model_type is not None:
        raise ValueError(
            f"No source-verified optimization profile is registered for "
            f"{model_type!r}. Its {architecture.value!r} data architecture "
            "does not make another model's optimizer recipe interchangeable.")
    if factory_path is None:
        raise ValueError(
            "Special optimization profiles currently cover VITS, codec/LLM, "
            f"and diffusion TTS; received {architecture.value!r}.")
    return _instantiate_profile(factory_path)


__all__ = [
    "DiffusionTTSOptimizationConfig",
    "LLMTTSOptimizationConfig",
    "TTSOptimizationProfile",
    "TTSTrainingOptimizationProfile",
    "VITSOptimizationConfig",
    "get_tts_training_optimization_profile",
]
