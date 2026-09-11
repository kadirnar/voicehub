"""Native WavLM CTC inference and fine-tuning wrapper."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from voicehub.hub import read_json_file
from voicehub.models.asr_wav2vec2.modeling import Wav2Vec2ForSpeechRecognition
from voicehub.models.asr_wavlm.configuration import WavLMASRConfig


def _architecture_names(values: Mapping[str, Any]) -> tuple[str, ...]:
    architectures = values.get('architectures', ())
    if isinstance(architectures, str):
        architectures = (architectures, )
    if not isinstance(architectures, Sequence):
        raise TypeError("WavLM checkpoint `architectures` must be a sequence.")
    return tuple(str(value) for value in architectures)


class WavLMForSpeechRecognition(Wav2Vec2ForSpeechRecognition):
    """Run and fine-tune WavLM CTC using only VoiceHub and PyTorch."""

    config_class = WavLMASRConfig
    default_model_name_or_path = ("patrickvonplaten/wavlm-libri-clean-100h-base-plus")
    architecture_family = "ctc"
    runtime_name = "WavLM"
    metadata_architecture = "wavlm-ctc"
    native_checkpoint_format = "native-wavlm-ctc-v1"
    native_model_architecture = "WavLMForCTC"

    def __init__(
        self,
        config: WavLMASRConfig | str | Path | None = None,
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
        if model_type not in {"wavlm", "asr_wavlm"}:
            raise ValueError(
                "Native WavLM requires a WavLM checkpoint; received model "
                f"type {model_type or '<missing>'!r}.")
        architectures = _architecture_names(values)
        if architectures and not any(name in {"WavLMForCTC", "WavLMForSpeechRecognition"}
                                     for name in architectures):
            names = ", ".join(architectures)
            raise ValueError("Native WavLM requires a CTC checkpoint architecture; "
                             f"received: {names}.")

    @classmethod
    def _validate_processor(cls, processor: Any, config: Any) -> None:
        """Validate fields that participate in WavLM CTC computation.

        The published checkpoint retains pretraining-era
        ``bos_token_id`` and ``eos_token_id`` values that point at
        ordinary character IDs. Its tokenizer stores the unused BOS/EOS
        strings as added tokens. CTC uses only the vocabulary size and
        blank/pad ID, so treating those generation-only IDs as a loader
        invariant would reject the official artifact.
        """
        if processor.sampling_rate != config.sampling_rate:
            raise ValueError(
                "WavLM processor/model sampling-rate mismatch: processor "
                f"uses {processor.sampling_rate}, model expects "
                f"{config.sampling_rate}.")
        tokenizer = processor.tokenizer
        if tokenizer.vocabulary_size != config.vocab_size:
            raise ValueError(
                "WavLM tokenizer/model vocabulary mismatch: tokenizer has "
                f"{tokenizer.vocabulary_size} IDs, model expects "
                f"{config.vocab_size}.")
        if tokenizer.pad_token_id != config.pad_token_id:
            raise ValueError(
                "WavLM tokenizer/model pad_token_id mismatch: tokenizer "
                f"uses {tokenizer.pad_token_id}, model expects "
                f"{config.pad_token_id}.")

    def _load_pretrained_model(self) -> None:
        from voicehub.checkpointing import SafeTensorReader, ShardedSafeTensorReader
        from voicehub.models.asr_wavlm.native.artifacts import resolve_wavlm_artifacts
        from voicehub.models.asr_wavlm.native.checkpoint import (
            WAVLM_BASE_PLUS_CTC_SAFETENSORS_REVISION,
            HuggingFaceWavLMCheckpointAdapter,
        )
        from voicehub.models.asr_wavlm.native.configuration import WavLMConfig
        from voicehub.models.asr_wavlm.native.modeling import WavLMForCTC
        from voicehub.models.asr_wavlm.processing import WavLMProcessor

        source = self.config.name_or_path or self.default_model_name_or_path
        revision = self.config.revision
        if revision is None and str(source) == self.default_model_name_or_path:
            # Main contains only a legacy pickle. The pinned Safetensors
            # conversion commit has the immutable main checkpoint as its sole
            # parent and preserves its tensor inventory.
            revision = WAVLM_BASE_PLUS_CTC_SAFETENSORS_REVISION
        artifacts = resolve_wavlm_artifacts(
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
        native_config = WavLMConfig.from_dict(architecture_values)
        processor = WavLMProcessor.from_artifacts(
            vocabulary=artifacts.vocabulary,
            added_tokens=artifacts.added_tokens,
            tokenizer_config=artifacts.tokenizer_config,
            special_tokens_map=artifacts.special_tokens_map,
            preprocessor_config=artifacts.preprocessor_config,
            target_language=self.config.target_language,
        )
        self._validate_processor(processor, native_config)

        model = WavLMForCTC(native_config)
        reader_type = (ShardedSafeTensorReader if artifacts.is_sharded else SafeTensorReader)
        with reader_type(artifacts.checkpoint) as reader:
            HuggingFaceWavLMCheckpointAdapter().load_streaming(
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


__all__ = ["WavLMForSpeechRecognition"]
