"""Task-neutral pretrained lifecycle for audio-input speech models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from inspect import Parameter, signature
from pathlib import Path
from typing import Any

from voicehub.configuration import VoiceHubConfig
from voicehub.inference_configuration import ASRInferenceConfig, SpeechInferenceConfig, VADInferenceConfig
from voicehub.model import PreTrainedSpeechModel
from voicehub.outputs import ASROutput, VADOutput
from voicehub.processing.processor import AudioProcessor


class PreTrainedAudioModel(
        PreTrainedSpeechModel,
        ABC,
):
    """Shared lazy lifecycle for ASR and VAD wrappers.

    The common pretrained model owns the lifecycle:
    model allocation is lazy, inference strategy transitions are
    reversible for training, and portable trainer state is attached
    before first inference. Task subclasses own the actual input/output
    semantics.
    """

    config_class = VoiceHubConfig
    processor_class = AudioProcessor
    inference_config_class: type[SpeechInferenceConfig] = SpeechInferenceConfig
    output_type: type = object
    default_model_name_or_path = ""
    main_input_name = "audio"
    base_model_prefix = "speech_model"
    supports_gradient_checkpointing = False
    passthrough_inference_options: frozenset[str] | None = None

    def __init__(self, config, *, device="auto", lazy_load=True):
        self.inference_config = self.inference_config_class.from_model_config(config)
        super().__init__(config, device=device, lazy_load=lazy_load)

    def _restore_inference_settings(self, directory: Path) -> None:
        if (directory / self.inference_config_class.config_name).is_file():
            self.inference_config = self.inference_config_class.from_pretrained(directory)

    def _save_inference_settings(self, directory: Path) -> None:
        self.inference_config.save_pretrained(directory)

    def prepare_inputs_for_inference(
        self,
        audio: Any,
        *,
        sampling_rate: int | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        return dict(self.processor(
            audio,
            sampling_rate=sampling_rate,
            **kwargs,
        ))

    def _validate_inference_inputs(self, model_inputs: dict[str, Any]) -> None:
        """Dependency-free task/backend input validation hook."""

    def _inference_callable(self):
        return self._run_inference

    def _validate_model_kwargs(self, model_kwargs: dict[str, Any]) -> None:
        parameters = signature(self._inference_callable()).parameters
        has_passthrough = any(parameter.kind is Parameter.VAR_KEYWORD for parameter in parameters.values())
        if has_passthrough and self.passthrough_inference_options is None:
            return
        supported = {
            name
            for name, parameter in parameters.items()
            if name != "self" and parameter.kind not in (Parameter.VAR_KEYWORD, Parameter.VAR_POSITIONAL)
        }
        if self.passthrough_inference_options is not None:
            supported.update(self.passthrough_inference_options)
        unknown = sorted(set(model_kwargs) - supported)
        if unknown:
            raise ValueError(
                "Unsupported inference option(s): "
                f"{', '.join(unknown)}. {self.__class__.__name__} accepts: "
                f"{', '.join(sorted(supported))}.")

    def forward(
        self,
        audio: Any,
        *,
        sampling_rate: int | None = None,
        inference_config: SpeechInferenceConfig | None = None,
        **kwargs,
    ):
        defaults = self.inference_config.to_dict()
        if inference_config is not None:
            if not isinstance(inference_config, self.inference_config_class):
                raise TypeError(
                    "`inference_config` must be an instance of "
                    f"{self.inference_config_class.__name__}.")
            defaults.update(inference_config.to_dict())
        defaults.update(kwargs)
        options = self.inference_config_class.from_dict(defaults).to_dict()
        model_inputs = self.prepare_inputs_for_inference(
            audio,
            sampling_rate=sampling_rate,
            **options,
        )
        self._validate_model_kwargs(model_inputs)
        self._validate_inference_inputs(model_inputs)
        with self._lifecycle_lock:
            self.load()
            output = self._inference_callable()(**model_inputs)
        if not isinstance(output, self.output_type):
            raise TypeError(
                f"{self.__class__.__name__} inference must return "
                f"{self.output_type.__name__}, received {type(output).__name__}.")
        return output

    def __call__(
        self,
        audio: Any,
        *,
        sampling_rate: int | None = None,
        inference_config: SpeechInferenceConfig | None = None,
        **kwargs,
    ):
        return self.forward(
            audio,
            sampling_rate=sampling_rate,
            inference_config=inference_config,
            **kwargs,
        )

    def stream(
        self,
        *,
        sampling_rate: int,
        **inference_kwargs,
    ):
        """Create an isolated streaming session.

        The default session buffers chunks and invokes offline inference on
        ``flush``. Cache-aware backends can override this method without
        changing the public session contract.
        """
        from voicehub.streaming import BufferedSpeechSession

        return BufferedSpeechSession(
            self,
            sampling_rate=sampling_rate,
            inference_kwargs=inference_kwargs,
        )

    def create_training_dataset(self, records, **kwargs):
        return self.get_training_adapter().create_dataset(records, **kwargs)

    @abstractmethod
    def _run_inference(self, audio: Any, **kwargs):
        ...


class PreTrainedASRModel(PreTrainedAudioModel, ABC):
    """Base class for speech-recognition models."""

    inference_config_class = ASRInferenceConfig
    output_type = ASROutput

    def create_training_dataset(self, records, **kwargs):
        """Build a model-aware ASR fine-tuning dataset.

        ``records`` may be mappings, an :class:`ASRDataset`, or a
        JSON/JSONL/CSV/TSV manifest path. Common upstream fields such as
        ``audio_filepath``, ``wav``, ``transcription``, and ``sentence``
        are normalized before the model-specific adapter constructs
        features and targets.
        """
        from voicehub.training.datasets import ASRDataset, SpeechDataset

        data_root = kwargs.pop("data_root", None)
        data_aliases = kwargs.pop("data_aliases", None)
        validate_records = kwargs.pop("validate_records", None)
        validate_audio_files = kwargs.pop("validate_audio_files", False)
        should_coerce = not isinstance(records, SpeechDataset)
        should_coerce = should_coerce or isinstance(records, ASRDataset)
        should_coerce = should_coerce or validate_records is not None
        should_coerce = should_coerce or data_root is not None
        should_coerce = should_coerce or data_aliases is not None
        should_coerce = should_coerce or bool(validate_audio_files)
        if should_coerce:
            records = ASRDataset.coerce(
                records,
                model_type=self.config.model_type,
                root=data_root,
                aliases=data_aliases,
                validate=(True if validate_records is None else bool(validate_records)),
                validate_files=bool(validate_audio_files),
            )
        return self.get_training_adapter().create_dataset(records, **kwargs)

    base_model_prefix = "asr_model"

    @abstractmethod
    def _transcribe(self, audio: Any, **kwargs) -> ASROutput:
        ...

    def _run_inference(self, audio: Any, **kwargs) -> ASROutput:
        return self._transcribe(audio, **kwargs)

    def _inference_callable(self):
        return self._transcribe

    def transcribe(
        self,
        audio: Any,
        *,
        sampling_rate: int | None = None,
        inference_config: ASRInferenceConfig | None = None,
        **kwargs,
    ) -> ASROutput:
        return self.forward(
            audio,
            sampling_rate=sampling_rate,
            inference_config=inference_config,
            **kwargs,
        )

    @classmethod
    def can_transcribe(cls) -> bool:
        return cls._transcribe is not PreTrainedASRModel._transcribe


class PreTrainedVADModel(PreTrainedAudioModel, ABC):
    """Base class for voice-activity-detection models."""

    inference_config_class = VADInferenceConfig
    output_type = VADOutput
    base_model_prefix = "vad_model"

    @abstractmethod
    def _detect(self, audio: Any, **kwargs) -> VADOutput:
        ...

    def _run_inference(self, audio: Any, **kwargs) -> VADOutput:
        return self._detect(audio, **kwargs)

    def _inference_callable(self):
        return self._detect

    def detect(
        self,
        audio: Any,
        *,
        sampling_rate: int | None = None,
        inference_config: VADInferenceConfig | None = None,
        **kwargs,
    ) -> VADOutput:
        return self.forward(
            audio,
            sampling_rate=sampling_rate,
            inference_config=inference_config,
            **kwargs,
        )

    @classmethod
    def can_detect(cls) -> bool:
        return cls._detect is not PreTrainedVADModel._detect
