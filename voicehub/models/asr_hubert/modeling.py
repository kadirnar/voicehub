"""Native HuBERT CTC inference and fine-tuning wrapper."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from voicehub.hub import read_json_file
from voicehub.models.asr_hubert.configuration import HubertASRConfig
from voicehub.models.asr_wav2vec2.modeling import Wav2Vec2ForSpeechRecognition


def _architecture_names(values: Mapping[str, Any]) -> tuple[str, ...]:
    architectures = values.get('architectures', ())
    if isinstance(architectures, str):
        architectures = (architectures, )
    if not isinstance(architectures, Sequence):
        raise TypeError("HuBERT checkpoint `architectures` must be a sequence.")
    return tuple(str(value) for value in architectures)


class HubertForSpeechRecognition(Wav2Vec2ForSpeechRecognition):
    """Run and fine-tune HuBERT CTC using only VoiceHub and PyTorch."""

    config_class = HubertASRConfig
    default_model_name_or_path = "facebook/hubert-large-ls960-ft"
    architecture_family = "ctc"
    runtime_name = "HuBERT"
    metadata_architecture = "hubert-ctc"
    native_checkpoint_format = "native-hubert-ctc-v1"
    native_model_architecture = "HubertForCTC"

    def __init__(
        self,
        config: HubertASRConfig | str | Path | None = None,
        *,
        model_path: str | Path | None = None,
        device: str = "auto",
        lazy_load: bool = True,
        token: str | bool | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            config,
            model_path=model_path,
            device=device,
            lazy_load=lazy_load,
            token=token,
            **kwargs,
        )

    @staticmethod
    def _validate_architecture(values: Mapping[str, Any]) -> None:
        model_type = str(values.get("model_type", "")).strip().lower()
        if model_type not in {"hubert", "asr_hubert"}:
            raise ValueError(
                "Native HuBERT requires a HuBERT checkpoint; received model "
                f"type {model_type or '<missing>'!r}.")
        architectures = _architecture_names(values)
        if architectures and not any(name in {"HubertForCTC", "HubertForSpeechRecognition"}
                                     for name in architectures):
            names = ", ".join(architectures)
            raise ValueError("Native HuBERT requires a CTC checkpoint architecture; "
                             f"received: {names}.")

    def _load_pretrained_model(self) -> None:
        from voicehub.checkpointing import SafeTensorReader, ShardedSafeTensorReader
        from voicehub.models.asr_hubert.native.artifacts import resolve_hubert_artifacts
        from voicehub.models.asr_hubert.native.checkpoint import (
            FACEBOOK_HUBERT_LARGE_LS960_FT_SAFETENSORS_REVISION,
            HuggingFaceHubertCheckpointAdapter,
        )
        from voicehub.models.asr_hubert.native.configuration import HubertConfig
        from voicehub.models.asr_hubert.native.modeling import HubertForCTC
        from voicehub.models.asr_hubert.processing import HubertProcessor

        source = self.config.name_or_path or self.default_model_name_or_path
        revision = self.config.revision
        if (revision is None and str(source) == self.default_model_name_or_path):
            # The official main revision only contains a legacy pickle. Use
            # Hugging Face's tensor-equivalent conversion commit, whose parent
            # is the immutable official checkpoint revision.
            revision = (FACEBOOK_HUBERT_LARGE_LS960_FT_SAFETENSORS_REVISION)
        artifacts = resolve_hubert_artifacts(
            source,
            checkpoint_filename=self.config.checkpoint_filename,
            vocabulary_filename=self.config.vocabulary_filename,
            cache_dir=self.config.cache_dir,
            revision=revision,
            token=self._hub_token,
            local_files_only=self.config.local_files_only,
        )
        architecture_values = read_json_file(artifacts.config)
        self._validate_architecture(architecture_values)
        native_config = HubertConfig.from_dict(architecture_values)
        processor = HubertProcessor.from_artifacts(
            vocabulary=artifacts.vocabulary,
            tokenizer_config=artifacts.tokenizer_config,
            special_tokens_map=artifacts.special_tokens_map,
            preprocessor_config=artifacts.preprocessor_config,
            target_language=self.config.target_language,
        )
        self._validate_processor(processor, native_config)

        model = HubertForCTC(native_config)
        reader_type = (ShardedSafeTensorReader if artifacts.is_sharded else SafeTensorReader)
        with reader_type(artifacts.checkpoint) as reader:
            HuggingFaceHubertCheckpointAdapter().load_streaming(
                model,
                reader,
                architecture_values,
                strict=True,
            )
        model.to(
            device=self.device,
            dtype=self._model_dtype(),
        )

        self.artifacts = artifacts
        self.native_config = native_config
        self.ctc_processor = processor
        self.training_processor = processor
        self.transformers_processor = processor
        self.model = model


__all__ = ["HubertForSpeechRecognition"]
