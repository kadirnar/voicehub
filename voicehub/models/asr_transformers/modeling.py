"""Closed native ASR dispatch under the historical Transformers model key."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from voicehub.hub import read_json_file, resolve_pretrained_file
from voicehub.hub_transport import get_cached_hugging_face_commit
from voicehub.models.asr_transformers.configuration import TransformersASRConfig
from voicehub.models.audio import PreTrainedASRModel
from voicehub.outputs import ASROutput
from voicehub.path_utils import is_explicit_local_path

_SERVING_ONLY_MARKERS = (
    ".gguf",
    "-gguf",
    "/gguf",
    "llama.cpp",
    "llama_cpp",
    ".onnx",
    ".ort",
    ".engine",
    ".plan",
    ".tflite",
    ".mlmodel",
)
_UNSAFE_CHECKPOINT_SUFFIXES = frozenset({
    ".bin",
    ".ckpt",
    ".pt",
    ".pth",
})
_NON_ASR_ARCHITECTURE_MARKERS = (
    "foraudioframeclassification",
    "foraudioclassification",
    "fortexttoaudio",
    "fortexttospeech",
)
_NATIVE_MODEL_TYPE_ALIASES = {
    "asr_hubert": "hubert",
    "asr_moonshine": "moonshine",
    "asr_wav2vec2": "wav2vec2",
    "asr_wavlm": "wavlm",
    "asr_whisper": "whisper",
    "hubert": "hubert",
    "moonshine": "moonshine",
    "wav2vec2": "wav2vec2",
    "wavlm": "wavlm",
    "whisper": "whisper",
}
_NATIVE_ARCHITECTURE_FAMILIES = {
    "hubert": "ctc",
    "moonshine": "speech-seq2seq",
    "wav2vec2": "ctc",
    "wavlm": "ctc",
    "whisper": "speech-seq2seq",
}
_NATIVE_ARCHITECTURE_NAMES = {
    "hubert": frozenset({
        "HubertForCTC",
        "HubertForSpeechRecognition",
    }),
    "moonshine": frozenset({
        "MoonshineForConditionalGeneration",
        "MoonshineForSpeechRecognition",
    }),
    "wav2vec2": frozenset({
        "Wav2Vec2ForCTC",
        "Wav2Vec2ForSpeechRecognition",
    }),
    "wavlm": frozenset({
        "WavLMForCTC",
        "WavLMForSpeechRecognition",
    }),
    "whisper": frozenset({
        "WhisperForConditionalGeneration",
        "WhisperForSpeechRecognition",
        "WhisperModel",
    }),
}


class TransformersASRForSpeechRecognition(PreTrainedASRModel):
    """Dispatch a verified checkpoint to a VoiceHub-owned ASR graph.

    ``TransformersASRForSpeechRecognition`` and the ``asr_transformers``
    key are retained as compatibility identifiers. They do not describe an
    executable dependency: checkpoint metadata is resolved by VoiceHub and
    dispatched through a closed table of native Whisper, Wav2Vec2, HuBERT,
    WavLM, and Moonshine implementations.
    """

    config_class = TransformersASRConfig
    default_model_name_or_path = "openai/whisper-small"

    def __init__(
        self,
        config: TransformersASRConfig | str | Path | None = None,
        *,
        model_path: str | Path | None = None,
        device: str = "auto",
        lazy_load: bool = True,
        token: str | bool | None = None,
        **config_overrides: Any,
    ) -> None:
        config = self._coerce_config(
            config,
            model_path=model_path,
            **config_overrides,
        )
        config.validate()
        if token is not None and (not isinstance(token, (str, bool)) or
                                  isinstance(token, str) and not token.strip()):
            raise ValueError("`token` must be a non-empty string, boolean, or None.")

        self.native_config: Any | None = None
        self.native_processor: Any | None = None
        # Historical attribute retained for callers that inspect the paired
        # processor. Its value is always a VoiceHub-owned processor.
        self.transformers_processor: Any | None = None
        self.architecture_family: str | None = None
        self.native_model_type: str | None = None
        self.artifacts: Any | None = None
        self._delegate: PreTrainedASRModel | None = None
        self._token = token
        super().__init__(
            config,
            device=device,
            lazy_load=lazy_load,
        )

    @property
    def training_processor(self) -> Any | None:
        """Return the native processor paired with the trainable graph."""
        return self.native_processor

    @staticmethod
    def _is_native_checkpoint_filename(filename: str) -> bool:
        normalized = filename.lower()
        return (normalized.endswith(".safetensors") or normalized.endswith(".safetensors.index.json"))

    def _validate_checkpoint_selector(self) -> None:
        filename = self.config.checkpoint_filename
        if (filename is not None and not self._is_native_checkpoint_filename(filename)):
            raise ValueError(
                "Native ASR dispatch accepts Safetensors checkpoints only; "
                f"received `checkpoint_filename={filename!r}`.")

        source = Path(self.config.name_or_path or self.default_model_name_or_path).expanduser()
        if source.is_file() and not self._is_native_checkpoint_filename(source.name):
            raise ValueError(
                "A direct native ASR checkpoint must be a `.safetensors` "
                f"file; received {source.name!r}.")

    def _load_pretrained_model(self) -> None:
        self._validate_checkpoint_selector()
        source, values, revision = self._resolve_dispatch_configuration()
        native_model_type = self._native_model_type_from_config(values)
        architecture_family = _NATIVE_ARCHITECTURE_FAMILIES[native_model_type]
        requested_family = self.config.architecture_family
        if (requested_family != "auto" and requested_family != architecture_family):
            raise ValueError(
                f"Checkpoint model type {native_model_type!r} belongs to "
                f"{architecture_family!r}, not requested family "
                f"{requested_family!r}.")

        delegate = self._build_native_delegate(
            native_model_type,
            source=source,
            revision=revision,
        )
        delegate.device = self.device
        delegate._loading_for_training = self.is_training_load
        if self.is_training_load:
            delegate._validate_training_runtime()
        delegate._load_pretrained_model()
        if delegate.model is None:
            raise RuntimeError(f"The native {native_model_type} loader returned no model.")

        native_processor = getattr(delegate, "training_processor", None)
        self._delegate = delegate
        self.native_model_type = native_model_type
        self.architecture_family = architecture_family
        self.native_config = getattr(delegate, "native_config", None)
        self.native_processor = native_processor
        self.transformers_processor = native_processor
        self.artifacts = getattr(delegate, "artifacts", None)
        self.model = delegate.model
        self.config.sample_rate = delegate.sample_rate

    def _resolve_dispatch_configuration(self, ) -> tuple[str | Path, dict[str, Any], str | None]:
        """Resolve and revision-pin the declarative dispatch configuration."""
        from voicehub.training.utils import NATIVE_EXPORT_DIR

        source: str | Path = (self.config.name_or_path or self.default_model_name_or_path)
        source_path = Path(source).expanduser()
        if source_path.exists():
            root = source_path.parent if source_path.is_file() else source_path
            config_path = root / "config.json"
            if not config_path.is_file():
                raise FileNotFoundError(
                    "Native ASR dispatch requires 'config.json' beside the "
                    f"checkpoint: {root}.")
            values = read_json_file(config_path)
            if (str(values.get("model_type", "")).strip().lower() == self.config.model_type):
                native_root = root / NATIVE_EXPORT_DIR
                native_config = native_root / "config.json"
                if native_config.is_file():
                    root = native_root
                    values = read_json_file(native_config)
            resolved_source: str | Path = (
                source_path.resolve()
                if source_path.is_file() and root == source_path.parent else root.resolve())
            return resolved_source, values, None
        if is_explicit_local_path(source):
            raise FileNotFoundError(f"Native ASR checkpoint path was not found: {source_path}.")

        requested_revision = self.config.revision or "main"
        config_path = resolve_pretrained_file(
            str(source),
            "config.json",
            cache_dir=self.config.cache_dir,
            revision=requested_revision,
            token=self._token,
            local_files_only=self.config.local_files_only,
        )
        pinned_revision = get_cached_hugging_face_commit(
            str(source),
            "config.json",
            cache_dir=self.config.cache_dir,
            revision=requested_revision,
        )
        return (
            str(source),
            read_json_file(config_path),
            pinned_revision or requested_revision,
        )

    @staticmethod
    def _native_model_type_from_config(values: Mapping[str, Any], ) -> str:
        """Return a native model type only for a verified closed-table
        entry."""
        architectures = values.get('architectures', ())
        if isinstance(architectures, str):
            architectures = (architectures, )
        if not isinstance(architectures, Sequence):
            raise TypeError("ASR checkpoint `architectures` must be a sequence.")
        if any(not isinstance(name, str) for name in architectures):
            raise TypeError("ASR checkpoint `architectures` entries must be strings.")
        for architecture in architectures:
            normalized = architecture.replace("_", "").replace("-", "").lower()
            if any(marker in normalized for marker in _NON_ASR_ARCHITECTURE_MARKERS):
                raise ValueError(f"Checkpoint architecture {architecture!r} is not an ASR "
                                 "head.")

        declared = values.get(
            "voicehub_provider",
            values.get("model_type", ""),
        )
        normalized_type = str(declared).strip().lower().replace("-", "_")
        native_model_type = _NATIVE_MODEL_TYPE_ALIASES.get(normalized_type)
        if native_model_type is not None:
            expected_architectures = _NATIVE_ARCHITECTURE_NAMES[native_model_type]
            if (architectures and
                    not any(architecture in expected_architectures for architecture in architectures)):
                names = ", ".join(architectures)
                raise ValueError(
                    f"Checkpoint model type {normalized_type!r} does not "
                    "declare a verified native ASR head; received "
                    f"architecture(s): {names}.")
            return native_model_type

        supported = ", ".join(sorted({
            "Whisper",
            "Wav2Vec2 CTC",
            "HuBERT CTC",
            "WavLM CTC",
            "Moonshine",
        }))
        raise ValueError(
            "The generic VoiceHub ASR provider cannot dispatch checkpoint "
            f"model type {normalized_type or '<missing>'!r}. Verified native "
            f"families are: {supported}. RNN-T, TDT, repository-code, "
            "audio-text-to-text, and other architectures require a "
            "dedicated native provider.")

    def _build_native_delegate(
        self,
        native_model_type: str,
        *,
        source: str | Path,
        revision: str | None,
    ) -> PreTrainedASRModel:
        common = {
            "name_or_path": source,
            "revision": revision,
            "cache_dir": self.config.cache_dir,
            "local_files_only": self.config.local_files_only,
            "checkpoint_filename": self.config.checkpoint_filename,
            "torch_dtype": self.config.torch_dtype,
        }
        if native_model_type == "whisper":
            from voicehub.models.asr_whisper_native import WhisperASRConfig, WhisperForSpeechRecognition

            native_config = WhisperASRConfig(
                **common,
                tokenizer_filename=self.config.tokenizer_filename,
            )
            wrapper_type = WhisperForSpeechRecognition
        elif native_model_type == "wav2vec2":
            from voicehub.models.asr_wav2vec2 import Wav2Vec2ASRConfig, Wav2Vec2ForSpeechRecognition

            native_config = Wav2Vec2ASRConfig(
                **common,
                vocabulary_filename=self.config.vocabulary_filename,
                target_language=self.config.target_language,
            )
            wrapper_type = Wav2Vec2ForSpeechRecognition
        elif native_model_type == "hubert":
            from voicehub.models.asr_hubert import HubertASRConfig, HubertForSpeechRecognition

            native_config = HubertASRConfig(
                **common,
                vocabulary_filename=self.config.vocabulary_filename,
                target_language=self.config.target_language,
            )
            wrapper_type = HubertForSpeechRecognition
        elif native_model_type == "wavlm":
            from voicehub.models.asr_wavlm import WavLMASRConfig, WavLMForSpeechRecognition

            native_config = WavLMASRConfig(
                **common,
                vocabulary_filename=self.config.vocabulary_filename,
                target_language=self.config.target_language,
            )
            wrapper_type = WavLMForSpeechRecognition
        elif native_model_type == "moonshine":
            from voicehub.models.asr_moonshine import MoonshineASRConfig, MoonshineForSpeechRecognition

            native_config = MoonshineASRConfig(
                **common,
                tokenizer_filename=self.config.tokenizer_filename,
            )
            wrapper_type = MoonshineForSpeechRecognition
        else:  # pragma: no cover - protected by the closed dispatch table
            raise AssertionError(f"Unhandled native ASR model type {native_model_type!r}.")
        return wrapper_type(
            native_config,
            device=self.device,
            lazy_load=True,
            token=self._token,
        )

    def _require_delegate(self) -> PreTrainedASRModel:
        delegate = self._delegate
        if delegate is None:
            raise RuntimeError("Native ASR dispatch has not loaded a model.")
        return delegate

    def _synchronize_delegate_model(self) -> PreTrainedASRModel:
        delegate = self._require_delegate()
        delegate.model = self.model
        return delegate

    def _transcribe(
        self,
        audio: Any,
        *,
        sampling_rate: int | None = None,
        language: str | None = None,
        task: str = "transcribe",
        return_timestamps: bool | str = False,
        chunk_length_s: float | None = None,
        stride_length_s: float | tuple[float, float] | None = None,
        batch_size: int | None = None,
        num_beams: int | None = None,
        max_new_tokens: int | None = None,
        hotwords: str | tuple[str, ...] | list[str] | None = None,
        **kwargs: Any,
    ) -> ASROutput:
        if kwargs:
            names = ", ".join(sorted(str(name) for name in kwargs))
            raise ValueError("Native ASR dispatch received unsupported inference "
                             f"option(s): {names}.")
        delegate = self._synchronize_delegate_model()
        return delegate._transcribe(
            audio,
            sampling_rate=sampling_rate,
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

    def prepare_training_inputs(
        self,
        inputs: dict[str, Any],
        *,
        phase: str,
    ) -> dict[str, Any]:
        """Delegate raw batch construction to the selected native family."""
        delegate = self._synchronize_delegate_model()
        return delegate.prepare_training_inputs(
            inputs,
            phase=phase,
        )

    def _validate_training_runtime(self) -> None:
        identifier = str(self.config.name_or_path or self.default_model_name_or_path).lower()
        if (any(marker in identifier for marker in _SERVING_ONLY_MARKERS) or
                Path(identifier).suffix in _UNSAFE_CHECKPOINT_SUFFIXES):
            raise ValueError(
                "Native ASR fine-tuning requires a differentiable "
                "Safetensors checkpoint; pickle, GGUF, ONNX, TensorRT, and "
                "Core ML artifacts are unsupported.")
        self._validate_checkpoint_selector()
        if self._delegate is not None:
            self._delegate._validate_training_runtime()

    def _prepare_for_training(self) -> None:
        delegate = self._synchronize_delegate_model()
        delegate._prepare_for_training()
        self.model = delegate.model

    def _prepare_for_inference(self) -> None:
        delegate = self._synchronize_delegate_model()
        delegate._prepare_for_inference()
        self.model = delegate.model

    def _save_pretrained(self, save_directory: Path) -> None:
        delegate = self._synchronize_delegate_model()
        delegate._save_pretrained(save_directory)


__all__ = ["TransformersASRForSpeechRecognition"]
