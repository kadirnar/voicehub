"""One pretrained lifecycle for synthesis, transcription, VAD, and codecs.

Task classes implement input/output semantics and the runtime-loading
hook. Weight loading, device selection, retry behavior, saving, and mode
transitions are shared here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from importlib import import_module
from pathlib import Path
from threading import RLock
from typing import Any

from voicehub.configuration import VoiceHubConfig
from voicehub.hub import read_json_file
from voicehub.inference_strategy import InferenceStrategy, get_inference_strategy
from voicehub.models.base import BaseSpeechModel
from voicehub.path_utils import normalize_model_source
from voicehub.processing.processor import VoiceHubProcessor
from voicehub.serialization_utils import reject_serialized_secrets
from voicehub.training.utils import MODEL_STATE_NAME, NATIVE_EXPORT_DIR


class PreTrainedSpeechModel(BaseSpeechModel, ABC):
    """Lazy, retryable runtime ownership shared by every audio task."""

    config_class = VoiceHubConfig
    processor_class = VoiceHubProcessor
    default_model_name_or_path = ""

    def __init__(
        self,
        config: VoiceHubConfig,
        *,
        device: str = "auto",
        lazy_load: bool = True,
    ):
        config.name_or_path = normalize_model_source(config.name_or_path)
        super().__init__(model_path=config.name_or_path, device=device)
        self.config = config
        if not self.config.architectures:
            self.config.architectures = [self.__class__.__name__]
        self.processor = self.processor_class()
        self.model = None
        self._lifecycle_lock = RLock()
        self._inference_strategy = get_inference_strategy()
        self._inference_strategy_validated = False
        self._inference_strategy_applied = False
        self._inference_ready = False
        self._training_ready = False
        self._loading_for_training = False
        self._pending_model_state_path: Path | None = None
        self._pending_training_recipe_state: dict[str, Any] | None = None
        if not lazy_load:
            self.load()

    @staticmethod
    def _validated_portable_model_state(state: Any) -> Any:
        """Reject credentials before applying deserialized portable state."""
        reject_serialized_secrets(
            state,
            owner="VoiceHub portable model state",
        )
        return state

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path = "",
        *,
        config: VoiceHubConfig | None = None,
        device: str = "auto",
        lazy_load: bool = True,
        inference_strategy: str | InferenceStrategy | None = None,
        config_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ):
        source_name = normalize_model_source(pretrained_model_name_or_path)
        source = Path(source_name).expanduser()
        is_direct_checkpoint = source.is_file() and source.suffix.lower() != ".json"
        if config is None:
            config_values = dict(config_kwargs or {})
            if is_direct_checkpoint:
                config = cls.config_class(name_or_path=source_name, **config_values)
            elif source_name:
                try:
                    config = cls.config_class.from_pretrained(
                        source_name,
                        **config_values,
                    )
                except FileNotFoundError:
                    config = cls.config_class(
                        name_or_path=source_name,
                        **config_values,
                    )
            else:
                config = cls.config_class(**config_values)
        elif source_name:
            config.name_or_path = source_name

        model_state_path = source / MODEL_STATE_NAME
        has_voicehub_state = bool(source_name) and source.is_dir() and model_state_path.is_file()
        if has_voicehub_state:
            config_path = source / "config.json"
            if config_path.is_file():
                saved_config = read_json_file(config_path)
                base_model = saved_config.get("name_or_path")
                if isinstance(base_model, str) and base_model.strip():
                    config.name_or_path = base_model

        model = cls(
            config,
            device=device,
            # Artifact-local processor and inference settings must be restored
            # before a provider allocates its runtime.
            lazy_load=True,
            **kwargs,
        )
        if inference_strategy is not None:
            model.set_inference_strategy(inference_strategy)
        if has_voicehub_state:
            model._pending_model_state_path = model_state_path

        if source_name and source.is_dir():
            model._restore_inference_settings(source)
            processor_path = source / "processor_config.json"
            if processor_path.is_file():
                model.processor = cls.processor_class.from_pretrained(source)
        if not lazy_load:
            model.load()
        return model

    def load(self):
        """Load weights and prepare the runtime for inference exactly once.

        Loading and mode transitions are guarded because source TTS
        runtimes commonly allocate mutable caches. Failed loads remain
        retryable.
        """
        with self._lifecycle_lock:
            requested_training = self._loading_for_training
            self._validate_optimization_transition("training" if requested_training else "inference")
            if not requested_training:
                self._validate_inference_strategy()
            model_loaded_in_this_call = False
            if self.model is None:
                self._inference_ready = False
                self._training_ready = False
                self.device = self._resolve_device(self.device)
                restore_training_state = self._pending_model_state_path is not None
                previous_mode = self._loading_for_training
                if restore_training_state:
                    self._validate_training_runtime()
                    self._loading_for_training = True
                try:
                    from voicehub.models._shared import preserve_inference_state

                    with preserve_inference_state(
                            device=self.device,
                            model_type=self.config.model_type,
                    ):
                        self._load_pretrained_model()
                except BaseException:
                    self.model = None
                    self._inference_ready = False
                    self._training_ready = False
                    raise
                finally:
                    self._loading_for_training = previous_mode
                if self.model is None:
                    raise RuntimeError(
                        f"{self.__class__.__name__}._load_pretrained_model() "
                        "completed without assigning `self.model`.")
                model_loaded_in_this_call = True
            if self._pending_model_state_path is not None:
                self._restore_voicehub_model_state(training=requested_training, )
            if not requested_training and not self._inference_ready:
                self._training_ready = False
                native_prepared = False
                strategy_prepare_started = False
                try:
                    self._prepare_for_inference()
                    native_prepared = True
                    if not self._inference_strategy_applied:
                        strategy_prepare_started = True
                        prepared_model = self._inference_strategy.prepare(
                            self.model,
                            wrapper=self,
                        )
                        if prepared_model is None:
                            raise TypeError(
                                "InferenceStrategy.prepare() must return the "
                                "prepared model runtime.")
                        self.model = prepared_model
                        self._inference_strategy_applied = True
                    self._after_prepare_for_inference()
                    self._inference_ready = True
                except BaseException:
                    # A newly allocated runtime may have been partially
                    # mutated by its native preparation hook and must be
                    # rebuilt. Preserve a pre-existing training runtime (and
                    # its optimizer-owned parameters) so serving auxiliaries
                    # such as a codec can be attached again safely.
                    discard_runtime = (not native_prepared and model_loaded_in_this_call)
                    if self._inference_strategy_applied:
                        try:
                            restored_model = self._inference_strategy.restore_for_training(
                                self.model,
                                wrapper=self,
                            )
                            self.model = restored_model
                        # Preserve the original preparation failure even if a
                        # third-party strategy cannot restore its runtime.
                        except BaseException:  # noqa: BLE001
                            discard_runtime = True
                    elif strategy_prepare_started:
                        # A strategy that raised before returning may have
                        # partially mutated its input without providing a
                        # restorable result.
                        discard_runtime = True
                    if discard_runtime:
                        self.model = None
                    self._inference_strategy_applied = False
                    self._inference_ready = False
                    self._training_ready = False
                    raise
        return self

    def save_pretrained(
        self,
        save_directory: str | Path,
        *,
        include_native_export: bool = True,
    ) -> Path:
        with self._lifecycle_lock:
            output_directory = Path(save_directory).expanduser()
            output_directory.mkdir(parents=True, exist_ok=True)
            if include_native_export:
                self._save_pretrained(output_directory / NATIVE_EXPORT_DIR)
            # Loading for export can resolve checkpoint-specific configuration.
            self.config.save_pretrained(output_directory)
            self._save_inference_settings(output_directory)
            self.processor.save_pretrained(output_directory)
        return output_directory

    @classmethod
    def _coerce_config(
        cls,
        config: VoiceHubConfig | str | Path | None = None,
        *,
        model_path: str | Path | None = None,
        **overrides,
    ) -> VoiceHubConfig:
        reject_serialized_secrets(
            overrides,
            owner=f"{cls.__name__} config overrides",
        )
        if isinstance(config, VoiceHubConfig):
            values = config.to_dict()
            configured_model_type = values.get("model_type")
            expected_model_type = cls.config_class.model_type
            if configured_model_type not in {
                    "voicehub",
                    expected_model_type,
            }:
                raise TypeError(
                    f"{cls.__name__} requires {expected_model_type!r} config, "
                    f"not {configured_model_type!r}.")
            if model_path is not None:
                values["name_or_path"] = normalize_model_source(model_path)
            values.update(overrides)
            normalized = cls.config_class.from_dict(values)
            normalized.name_or_path = normalize_model_source(normalized.name_or_path)
            return normalized
        if isinstance(config, (str, Path)):
            if model_path is not None:
                raise TypeError("Pass a path either as `config` or `model_path`, not both.")
            model_path = config
        source = (
            cls.default_model_name_or_path if model_path is None else normalize_model_source(model_path))
        return cls.config_class(name_or_path=source, **overrides)

    @property
    def sample_rate(self) -> int:
        return int(self.config.sample_rate)

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    @property
    def is_training_load(self) -> bool:
        return self._loading_for_training

    @property
    def inference_strategy(self) -> InferenceStrategy:
        return self._inference_strategy

    def set_inference_strategy(
        self,
        strategy: str | InferenceStrategy | None,
    ):
        resolved = get_inference_strategy(strategy)
        with self._lifecycle_lock:
            if self._inference_ready or self._inference_strategy_applied:
                raise RuntimeError(
                    "Cannot replace the inference strategy on an active "
                    "runtime. Call load_for_training() first.")
            self._inference_strategy = resolved
            self._inference_strategy_validated = False
        return self

    def _validate_inference_strategy(self) -> None:
        if self._inference_strategy_validated:
            return
        self._inference_strategy.validate(self)
        self._inference_strategy_validated = True

    def _restore_voicehub_model_state(self, *, training: bool) -> None:
        state_path = self._pending_model_state_path
        if state_path is None:
            return
        self._pending_model_state_path = None
        try:
            torch = import_module("torch")
            try:
                state = torch.load(
                    state_path,
                    map_location=self.device,
                    weights_only=True,
                )
            except TypeError:
                state = torch.load(state_path, map_location=self.device)
            state = self._validated_portable_model_state(state)
            if isinstance(state, Mapping) and "__voicehub_training_adapter__" in state:
                adapter = self.get_training_adapter()
                adapter.setup()
                adapter.load_state_dict(
                    state,
                    strict=True,
                    load_recipe_state=False,
                )
                recipe_state = state.get("recipe_state", {})
                if recipe_state:
                    self._pending_training_recipe_state = {
                        "model_type": state["__voicehub_training_adapter__"],
                        "recipe_id": adapter.recipe_id,
                        "state": recipe_state,
                    }
                if not training:
                    adapter.eval()
                return
            if not hasattr(self.model, "load_state_dict"):
                raise TypeError("The restored runtime does not implement load_state_dict().")
            self.model.load_state_dict(state)
            if not training and hasattr(self.model, "eval"):
                self.model.eval()
        except BaseException:
            self._pending_model_state_path = state_path
            raise

    def load_for_training(self):
        """Load or restore the differentiable runtime."""
        with self._lifecycle_lock:
            self._validate_optimization_transition("training")
            self._validate_training_runtime()
            if self.model is None:
                self._loading_for_training = True
                try:
                    self.load()
                finally:
                    self._loading_for_training = False
            self._inference_ready = False
            if not self._training_ready:
                if self._inference_strategy_applied:
                    restored = self._inference_strategy.restore_for_training(
                        self.model,
                        wrapper=self,
                    )
                    if restored is None:
                        raise TypeError(
                            "InferenceStrategy.restore_for_training() must "
                            "return the trainable runtime.")
                    self.model = restored
                    self._inference_strategy_applied = False
                self._prepare_for_training()
                self._training_ready = True
            if self._pending_model_state_path is not None:
                self._restore_voicehub_model_state(training=True)
        return self

    def export_native_pretrained(
        self,
        save_directory: str | Path,
    ) -> Path:
        """Write a flat provider-native artifact for trainer interop."""
        destination = Path(save_directory).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        self._save_pretrained(destination)
        return destination

    def get_training_adapter(self):
        from voicehub.training.auto import AutoTrainingAdapter

        return AutoTrainingAdapter.from_model(self)

    def validate_training_support(self):
        adapter = self.get_training_adapter()
        adapter.validate_support()
        return adapter.spec

    @property
    def training_default_model_name_or_path(self) -> str:
        spec = self.get_training_adapter().spec
        return spec.training_default_model_name_or_path or self.config.name_or_path

    def prepare_training_inputs(
        self,
        inputs: dict[str, Any],
        *,
        phase: str,
    ) -> dict[str, Any]:
        return dict(inputs)

    def _prepare_for_inference(self) -> None:
        evaluate = getattr(self.model, "eval", None)
        if callable(evaluate):
            evaluate()

    def _prepare_for_training(self) -> None:
        train = getattr(self.model, "train", None)
        if callable(train):
            train()

    def _validate_training_runtime(self) -> None:
        """Validate checkpoint/runtime training compatibility before load."""

    def _save_pretrained(self, save_directory: Path) -> None:
        """Optionally export a backend-native inference artifact."""

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        try:
            torch = import_module("torch")
        except ModuleNotFoundError:
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"

    def _after_prepare_for_inference(self) -> None:
        """Optional task hook after the shared runtime is ready."""

    def _restore_inference_settings(self, directory: Path) -> None:
        """Restore optional task settings before an eager runtime load."""

    def _save_inference_settings(self, directory: Path) -> None:
        """Save optional task settings next to the model configuration."""

    def _set_training_device(self, device: str) -> None:
        self.device = str(device)

    @abstractmethod
    def _load_pretrained_model(self) -> None:
        """Assign the checkpoint-backed runtime to ``self.model``."""
