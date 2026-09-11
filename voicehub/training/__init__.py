"""Lazy public imports; importing this package never loads a model."""

from importlib import import_module

_EXPORTS = {
    **dict.fromkeys(
        "AcousticTrainingAdapter AudioClassificationTrainingAdapter BaseTrainingAdapter CTCTrainingAdapter CausalLMTrainingAdapter CompositeTrainingAdapter FlowMatchingTrainingAdapter FrameClassificationTrainingAdapter RNNTTrainingAdapter Seq2SeqTrainingAdapter SpeechSeq2SeqTrainingAdapter TDTTrainingAdapter UpstreamNativeTrainingAdapter VITSTrainingAdapter".split(
        ),
        "voicehub.training.adapters",
    ),
    **dict.fromkeys(
        "TrainingArguments".split(),
        "voicehub.training.arguments",
    ),
    **dict.fromkeys(
        "AutoTrainingAdapter".split(),
        "voicehub.training.auto",
    ),
    **dict.fromkeys(
        "TrainerCallback".split(),
        "voicehub.training.callbacks",
    ),
    **dict.fromkeys(
        "AudioFieldSchema DataCollatorForAudioTraining DataCollatorForTTSTraining TTSFieldSchema".split(),
        "voicehub.training.collators",
    ),
    **dict.fromkeys(
        "TrainingContext TrainingPhaseKind TrainingPhaseSpec TrainingRecipeKind TrainingSupport".split(),
        "voicehub.training.contracts",
    ),
    **dict.fromkeys(
        "ASRDataArchitecture ASRDataReadiness ASRDataset ASRDatasetSpec ASRRecordVariant EpochGroupedBatchSampler EpochLengthBatchSampler SpeechDataset TTSBatchingConfig TTSBatchingStrategy TTSDataArchitecture TTSDataReadiness TTSDataset TTSDatasetSpec TTSRecordVariant get_asr_dataset_spec get_tts_dataset_spec list_asr_dataset_specs list_tts_dataset_specs".split(
        ),
        "voicehub.training.datasets",
    ),
    **dict.fromkeys(
        "WandbCallback".split(),
        "voicehub.training.integrations",
    ),
    **dict.fromkeys(
        "OptimizerBundle SchedulerBundle".split(),
        "voicehub.training.optimization",
    ),
    **dict.fromkeys(
        "ALL_MODEL_TRAINING_SPECS MODEL_TRAINING_SPECS ModelTrainingSpec TrainingFamily get_training_spec list_training_specs register_training_alias register_training_spec unregister_training_alias unregister_training_spec".split(
        ),
        "voicehub.training.specs",
    ),
    **dict.fromkeys(
        "TorchTrainingStrategy TrainingStrategy get_training_strategy list_training_strategies register_training_strategy unregister_training_strategy".split(
        ),
        "voicehub.training.strategy",
    ),
    **dict.fromkeys(
        "Trainer".split(),
        "voicehub.training.trainer",
    ),
    **dict.fromkeys(
        "VITSCUDAGraphPolicy diffusion_tts_acceleration_plan llm_tts_acceleration_plan vits_acceleration_plan".split(
        ),
        "voicehub.training.tts_acceleration",
    ),
    **dict.fromkeys(
        "DiffusionTrainingPair VITSDiscriminatorLoss build_diffusion_training_pair build_flow_matching_training_pair masked_diffusion_regression_loss multi_codebook_cross_entropy vits_discriminator_loss vits_feature_matching_loss vits_generator_adversarial_loss vits_kl_loss".split(
        ),
        "voicehub.training.tts_objectives",
    ),
    **dict.fromkeys(
        "DiffusionTTSOptimizationConfig LLMTTSOptimizationConfig TTSOptimizationProfile TTSTrainingOptimizationProfile VITSOptimizationConfig get_tts_training_optimization_profile".split(
        ),
        "voicehub.training.tts_optimization",
    ),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
