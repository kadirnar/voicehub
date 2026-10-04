"""Transformers-style pretrained model lifecycle for speech models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from importlib import import_module
from inspect import Parameter, signature
from pathlib import Path
from threading import RLock
from typing import Any

from voicehub.base_model import BaseTTSModel
from voicehub.configuration_utils import VoiceHubConfig
from voicehub.generation_configuration import TTSGenerationConfig
from voicehub.hub import read_json_file
from voicehub.inference_strategy import InferenceStrategy, get_inference_strategy
from voicehub.modeling_outputs import TTSOutput
from voicehub.path_utils import normalize_model_source
from voicehub.processing_utils import VoiceHubProcessor
from voicehub.serialization_utils import reject_serialized_secrets
from voicehub.trainer_utils import NATIVE_EXPORT_DIR


class PreTrainedSpeechModel(ABC):
    """Marker base shared by task-specific pretrained speech wrappers."""

    @staticmethod
    def _validated_portable_model_state(state: Any) -> Any:
        """Reject credentials before applying deserialized portable state."""
        reject_serialized_secrets(
            state,
            owner="VoiceHub portable model state",
        )
        return state


class PreTrainedTTSModel(BaseTTSModel, PreTrainedSpeechModel, ABC):
    """Base lifecycle shared by source-integrated VoiceHub architectures."""

    config_class = VoiceHubConfig
    generation_config_class = TTSGenerationConfig
    processor_class = VoiceHubProcessor
    default_model_name_or_path = ""
    main_input_name = "text"
    base_model_prefix = "tts_model"
    supports_gradient_checkpointing = False
    passthrough_generation_options: frozenset[str] | None = None

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
        self.generation_config = self.generation_config_class.from_model_config(config)
        self.model = None
        self.processor = self.processor_class()
        # Most native TTS runtimes own mutable KV caches, vocoders, and
        # temporary conditioning state. Serialize their lifecycle and
        # synthesis. External servers instead receive overlapping requests
        # after their immutable client/runtime snapshot is reserved here.
        self._lifecycle_lock = RLock()
        self._inference_strategy = get_inference_strategy()
        self._inference_strategy_validated = False
        self._inference_strategy_applied = False
        self._inference_ready = False
        self._training_ready = False
        self._loading_for_training = False
        self._pending_model_state_path: Path | None = None
        self._pending_training_recipe_state: dict[str, Any] | None = None
        self._pending_tts_optimization_config = None
        self._llm_backend_config = None
        self._llm_backend_client = None
        self._active_generation_requests = 0
        self._active_llm_requests = 0
        if not lazy_load:
            self.load()

    @classmethod
    def _coerce_config(
        cls,
        config: VoiceHubConfig | str | Path | None = None,
        *,
        model_path: str | Path | None = None,
        **overrides,
    ) -> VoiceHubConfig:
        """Normalize legacy constructor arguments into a typed
        configuration."""
        if isinstance(config, VoiceHubConfig):
            normalized = config
            if model_path is not None:
                normalized.name_or_path = normalize_model_source(model_path)
            else:
                normalized.name_or_path = normalize_model_source(normalized.name_or_path)
            normalized.update(overrides)
            normalized.name_or_path = normalize_model_source(normalized.name_or_path)
            return normalized

        if isinstance(config, (str, Path)):
            if model_path is not None:
                raise TypeError("Pass a path either as `config` or `model_path`, not both.")
            model_path = normalize_model_source(config)
        if model_path is not None:
            model_path = normalize_model_source(model_path)
        return cls.config_class(
            name_or_path=(cls.default_model_name_or_path if model_path is None else str(model_path)),
            **overrides,
        )

    @property
    def sample_rate(self) -> int:
        return self.config.sample_rate

    def _tts_optimization_runtime(self, mode):
        """Use the architecture's complete differentiable graph in training."""
        normalized_mode = self._optimization_mode(mode)
        if normalized_mode.value != "training":
            return super()._tts_optimization_runtime(normalized_mode)

        adapter = self.get_training_adapter()
        build_training_graph = getattr(adapter, "build_training_graph", None)
        if not callable(build_training_graph):
            raise TypeError(
                f"{type(self).__name__}.get_training_adapter() returned "
                f"{type(adapter).__name__}, which does not implement "
                "build_training_graph().")
        build_training_graph()
        return adapter

    @property
    def is_loaded(self) -> bool:
        """Whether checkpoint-backed runtime objects have been created."""
        return self.model is not None

    @property
    def inference_strategy(self) -> InferenceStrategy:
        """Return the runtime policy used for inference preparation."""
        return self._inference_strategy

    def set_inference_strategy(
        self,
        strategy: str | InferenceStrategy | None,
    ):
        """Select an inference runtime policy before serving begins.

        A model loaded only for training may still select a strategy; it
        is applied lazily on the next transition to inference. An active
        serving runtime must first transition through
        :meth:`load_for_training` so its current strategy can undo
        inference-only transformations safely.
        """
        resolved = get_inference_strategy(strategy)
        with self._lifecycle_lock:
            if self._llm_backend_config is not None and resolved.name != "eager":
                raise RuntimeError(
                    "External LLM serving cannot be combined with a second "
                    "inference strategy. Use the eager strategy for the "
                    "wrapper-side tokenizer/codec work.")
            if self._inference_ready or self._inference_strategy_applied:
                raise RuntimeError(
                    "Cannot replace the inference strategy on an active "
                    "serving runtime. Call load_for_training() first.")
            self._inference_strategy = resolved
            self._inference_strategy_validated = False
        return self

    def set_optimization_config(self, optimization_config):
        """Schedule one universal TTS policy for the next inference load.

        This mirrors Transformers' configuration-first backend selection
        while keeping VoiceHub's graph transformations explicit and
        reversible.
        """
        from voicehub.optimization import validate_tts_optimization_config

        resolved = validate_tts_optimization_config(
            self,
            optimization_config,
            mode="inference",
        )
        with self._lifecycle_lock:
            if self._llm_backend_config is not None:
                raise RuntimeError(
                    "External LLM serving and in-process TTS optimization "
                    "policies are separate execution modes. Configure kernels "
                    "on the vLLM/SGLang server, or use a native wrapper.")
            if self._inference_ready or self._training_ready:
                raise RuntimeError(
                    "Cannot schedule an optimization configuration on an "
                    "active runtime. Call optimize(...) directly, or restore "
                    "the current execution mode first.")
            if self._pending_tts_optimization_config is not None:
                raise RuntimeError("A TTS optimization configuration is already pending.")
            if self.tts_optimization_result(mode="inference") is not None:
                raise RuntimeError(
                    "An inference TTS optimization policy is already active. "
                    "Restore it before selecting another.")
            self._pending_tts_optimization_config = resolved
        return self

    def clear_optimization_config(self):
        """Clear and return a policy that has not been applied yet."""
        with self._lifecycle_lock:
            pending = self._pending_tts_optimization_config
            self._pending_tts_optimization_config = None
            return pending

    @property
    def llm_backend(self):
        """Return the selected language-model serving backend."""
        from voicehub.llm_serving import LLMBackend

        if self._llm_backend_config is None:
            return LLMBackend.NATIVE
        return self._llm_backend_config.backend

    @property
    def llm_backend_config(self):
        """Return the runtime-only external backend configuration, if any."""
        return self._llm_backend_config

    @property
    def llm_backend_transport(self):
        """Return the concrete external protocol selected for this model."""
        from voicehub.llm_serving import LLMBackendTransport

        if self._llm_backend_config is None:
            return LLMBackendTransport.AUTO
        return self._llm_backend_config.transport

    @property
    def uses_llm_token_backend(self) -> bool:
        from voicehub.llm_serving import LLMBackendTransport

        return (
            self._llm_backend_config is not None and
            self._llm_backend_config.transport is LLMBackendTransport.TOKENS)

    @property
    def uses_llm_speech_backend(self) -> bool:
        from voicehub.llm_serving import LLMBackendTransport

        return (
            self._llm_backend_config is not None and
            self._llm_backend_config.transport is LLMBackendTransport.SPEECH)

    def set_llm_backend(
        self,
        backend,
        config=None,
        **config_kwargs,
    ):
        """Select a vLLM/SGLang server before allocating native weights.

        ``config_kwargs`` is a convenience for ``set_llm_backend("vllm",
        endpoint=..., transport=...)``. Credentials remain attached only
        to this live wrapper.
        """
        from dataclasses import replace

        from voicehub.llm_serving import LLMBackend, LLMBackendConfig, LLMServingClient, get_llm_backend_support

        resolved_backend = LLMBackend.coerce(backend)
        if resolved_backend is LLMBackend.NATIVE:
            if config is not None or config_kwargs:
                raise ValueError("The native backend does not accept external connection "
                                 "settings.")
            self.clear_llm_backend()
            return self
        if config is not None and config_kwargs:
            raise TypeError(
                "Pass backend settings through either `config` or keyword "
                "arguments, not both.")
        config_value = config if config is not None else config_kwargs
        resolved = LLMBackendConfig.from_value(
            config_value,
            backend=resolved_backend,
        )
        support, transport = get_llm_backend_support(
            self.config.model_type,
            resolved.backend,
            transport=resolved.transport,
        )
        server_model = (resolved.model or self.config.name_or_path or self.default_model_name_or_path or None)
        from voicehub.llm_serving import LLMBackendTransport

        if server_model is None and transport is LLMBackendTransport.TOKENS:
            raise ValueError(
                "External token generation requires a server `model` ID. "
                "Set `LLMBackendConfig.model` or configure a model checkpoint "
                "on the wrapper.")
        resolved = replace(
            resolved,
            model=server_model,
            transport=transport,
        )
        del support
        with self._lifecycle_lock:
            if self._active_generation_requests:
                raise RuntimeError("Cannot replace the LLM backend while synthesis "
                                   "requests are active.")
            if self.model is not None or self._training_ready or self._inference_ready:
                raise RuntimeError("Select the LLM backend before loading or serving the "
                                   "wrapper.")
            if self._pending_model_state_path is not None:
                raise RuntimeError(
                    "A portable VoiceHub trainer state is pending. Serve that "
                    "fine-tuned artifact in the external engine, or load it "
                    "with the native backend.")
            if self._pending_tts_optimization_config is not None:
                raise RuntimeError(
                    "External LLM serving cannot be combined with a pending "
                    "in-process optimization policy.")
            if self._inference_strategy.name != "eager":
                raise RuntimeError(
                    "External LLM serving requires the eager wrapper-side "
                    "inference strategy.")
            self._llm_backend_config = resolved
            self._llm_backend_client = LLMServingClient(resolved)
        return self

    def clear_llm_backend(self):
        """Restore native selection before external token assets are loaded."""
        with self._lifecycle_lock:
            if self._active_llm_requests:
                raise RuntimeError(
                    "Cannot detach an external LLM backend while synthesis "
                    "requests are active.")
            if self.uses_llm_token_backend and self.model is not None:
                raise RuntimeError(
                    "Cannot detach a token backend after its tokenizer/codec "
                    "runtime has loaded. Create a fresh native wrapper.")
            previous = self._llm_backend_config
            self._llm_backend_config = None
            self._llm_backend_client = None
            self._inference_ready = False
            return previous

    def _reserve_llm_backend(self):
        """Snapshot one external backend for the complete request lifecycle."""
        with self._lifecycle_lock:
            config = self._llm_backend_config
            client = self._llm_backend_client
            self._active_generation_requests += 1
            if config is not None:
                self._active_llm_requests += 1
            return config, client

    def _release_llm_backend(self, config) -> None:
        with self._lifecycle_lock:
            if self._active_generation_requests <= 0:
                raise RuntimeError("Synthesis request accounting underflow.")
            self._active_generation_requests -= 1
            if config is None:
                return
            if self._active_llm_requests <= 0:
                raise RuntimeError("External LLM request accounting underflow.")
            self._active_llm_requests -= 1

    def _create_remote_causal_lm_proxy(self):
        if not self.uses_llm_token_backend or self._llm_backend_client is None:
            raise RuntimeError("No external token backend is configured.")
        from voicehub.llm_serving import RemoteCausalLMProxy

        return RemoteCausalLMProxy(
            self._llm_backend_client,
            model_type=self.config.model_type,
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path = "",
        *,
        config: VoiceHubConfig | None = None,
        device: str = "auto",
        lazy_load: bool = True,
        inference_strategy: str | InferenceStrategy | None = None,
        llm_backend=None,
        llm_backend_config=None,
        optimization_config=None,
        attn_implementation: str | None = None,
        kernel_backend: str | None = None,
        torch_compile: bool | str | None = None,
        compile_config=None,
        diffusion_cache: bool | str | None = None,
        diffusion_cache_config=None,
        diffusion_sampling: bool | str | None = None,
        diffusion_sampling_config=None,
        config_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ):
        """Construct a model from local files or a Hub repository
        identifier."""
        pretrained_model_name_or_path = normalize_model_source(pretrained_model_name_or_path)
        source = Path(pretrained_model_name_or_path).expanduser()
        is_direct_checkpoint_file = (source.is_file() and source.suffix.lower() != ".json")
        if config is None:
            config_values = dict(config_kwargs or {})
            if is_direct_checkpoint_file:
                config = cls.config_class(
                    name_or_path=pretrained_model_name_or_path,
                    **config_values,
                )
            elif pretrained_model_name_or_path:
                try:
                    config = cls.config_class.from_pretrained(
                        pretrained_model_name_or_path,
                        **config_values,
                    )
                except FileNotFoundError:
                    config = cls.config_class(
                        name_or_path=pretrained_model_name_or_path,
                        **config_values,
                    )
            else:
                config = cls.config_class(**config_values)
        elif pretrained_model_name_or_path:
            config.name_or_path = pretrained_model_name_or_path

        model_state_path = source / "model_state.pt"
        has_voicehub_state = source.is_dir() and model_state_path.is_file()
        if has_voicehub_state:
            saved_config = read_json_file(source / "config.json")
            base_model = saved_config.get("name_or_path")
            if isinstance(base_model, str) and base_model.strip():
                config.name_or_path = base_model
        from voicehub.optimization import tts_optimization_config_from_options

        resolved_optimization_config = (
            tts_optimization_config_from_options(
                optimization_config,
                attn_implementation=attn_implementation,
                kernel_backend=kernel_backend,
                torch_compile=torch_compile,
                compile_config=compile_config,
                diffusion_cache=diffusion_cache,
                diffusion_cache_config=diffusion_cache_config,
                diffusion_sampling=diffusion_sampling,
                diffusion_sampling_config=diffusion_sampling_config,
            ))

        external_backend_requested = (llm_backend is not None or llm_backend_config is not None)
        if external_backend_requested and has_voicehub_state:
            raise ValueError(
                "External LLM serving does not restore a local VoiceHub "
                "trainer state. Point the engine at the exported fine-tuned "
                "checkpoint instead.")
        if external_backend_requested and resolved_optimization_config is not None:
            raise ValueError(
                "Choose either an external LLM backend or an in-process TTS "
                "optimization configuration.")
        if (external_backend_requested and inference_strategy is not None and
                get_inference_strategy(inference_strategy).name != "eager"):
            raise ValueError("External LLM serving requires the eager wrapper-side "
                             "inference strategy.")
        defer_initial_load = (
            has_voicehub_state or inference_strategy is not None or
            resolved_optimization_config is not None or external_backend_requested)
        model = cls(
            config,
            device=device,
            # A Trainer artifact must be attached before the first runtime load
            # and an inference strategy must be selected before the runtime is
            # prepared. Both require deferring an eager constructor load.
            lazy_load=(True if defer_initial_load else lazy_load),
            **kwargs,
        )
        if inference_strategy is not None:
            model.set_inference_strategy(inference_strategy)
        if resolved_optimization_config is not None:
            model.set_optimization_config(resolved_optimization_config)
        if external_backend_requested:
            configured_backend = llm_backend
            if configured_backend is None and isinstance(llm_backend_config, Mapping):
                configured_backend = llm_backend_config.get("backend")
            if configured_backend is None:
                from voicehub.llm_serving import LLMBackendConfig

                if isinstance(llm_backend_config, LLMBackendConfig):
                    configured_backend = llm_backend_config.backend
            if configured_backend is None:
                raise ValueError(
                    "`llm_backend` is required when "
                    "`llm_backend_config` does not declare a backend.")
            model.set_llm_backend(
                configured_backend,
                config=llm_backend_config,
            )
        if has_voicehub_state:
            model._pending_model_state_path = model_state_path
        if not lazy_load and defer_initial_load:
            model.load()
        if source.is_dir():
            generation_path = source / "generation_config.json"
            processor_path = source / "processor_config.json"
            if generation_path.is_file():
                model.generation_config = (cls.generation_config_class.from_pretrained(source))
            if processor_path.is_file():
                model.processor = cls.processor_class.from_pretrained(source)
        return model

    def load(self):
        """Load weights and prepare the runtime for inference exactly once.

        Loading and mode transitions are guarded because source TTS
        runtimes commonly allocate mutable caches. Failed loads remain
        retryable.
        """
        with self._lifecycle_lock:
            requested_training = self._loading_for_training
            if self.uses_llm_speech_backend:
                if requested_training:
                    raise RuntimeError(
                        "A complete external speech backend cannot be loaded "
                        "for training. Create a native wrapper.")
                self._inference_ready = True
                return self
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
                    if self._pending_tts_optimization_config is not None:
                        pending_config = self._pending_tts_optimization_config
                        try:
                            self._optimize_loaded_tts_runtime(
                                pending_config,
                                mode="inference",
                            )
                        except BaseException:
                            self._pending_tts_optimization_config = (pending_config)
                            raise
                        self._pending_tts_optimization_config = None
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

    def _validate_inference_strategy(self) -> None:
        """Validate the selected runtime policy once, before allocation."""
        if self._inference_strategy_validated:
            return
        self._inference_strategy.validate(self)
        self._inference_strategy_validated = True

    def _restore_voicehub_model_state(self, *, training: bool) -> None:
        """Restore a portable state written by :class:`voicehub.Trainer`."""
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

            if (isinstance(state, Mapping) and "__voicehub_training_adapter__" in state):
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
                raise TypeError("The restored VoiceHub runtime does not implement "
                                "load_state_dict().")
            self.model.load_state_dict(state)
            if not training and hasattr(self.model, "eval"):
                self.model.eval()
        except BaseException:
            self._pending_model_state_path = state_path
            raise

    @property
    def is_training_load(self) -> bool:
        """Whether the current load is constructing a training runtime."""
        return self._loading_for_training

    def load_for_training(self):
        """Load a differentiable runtime without inference-only pruning.

        Architectures that compile, quantize, fuse, or delete modules
        during inference can inspect :attr:`is_training_load` in their
        loader and restore any additional state in
        :meth:`_prepare_for_training`.
        """
        with self._lifecycle_lock:
            if self._llm_backend_config is not None:
                raise RuntimeError(
                    "External vLLM/SGLang backends are inference-only. Create "
                    "a new native wrapper for fine-tuning.")
            self._validate_optimization_transition("training")
            self._validate_training_runtime()
            if self.model is None:
                self._loading_for_training = True
                try:
                    self.load()
                finally:
                    self._loading_for_training = False
            # Invalidate the serving state before the transition starts. A
            # backend hook may clear caches or detach serving auxiliaries
            # before it discovers an error; the next inference call must then
            # rebuild those resources instead of trusting a partially changed
            # runtime.
            self._inference_ready = False
            if not self._training_ready:
                if self._inference_strategy_applied:
                    restored_model = self._inference_strategy.restore_for_training(
                        self.model,
                        wrapper=self,
                    )
                    if restored_model is None:
                        raise TypeError(
                            "InferenceStrategy.restore_for_training() must "
                            "return the trainable model runtime.")
                    self.model = restored_model
                    self._inference_strategy_applied = False
                self._prepare_for_training()
                self._training_ready = True
            # A specialized adapter can construct its differentiable graph
            # before delegating here (Fish Speech does this for its
            # semantic-only graph). In that case ``model`` is already
            # populated and ``load()`` did not get an opportunity to consume
            # a lazy portable state.
            if self._pending_model_state_path is not None:
                self._restore_voicehub_model_state(training=True)
        return self

    def validate_training_support(self):
        """Validate this exact backend/checkpoint configuration without
        loading.

        The returned profile describes the family contract. Unsupported
        quantized, fused, GGUF, ONNX, or custom-recipe variants raise
        before allocating model weights.
        """
        adapter = self.get_training_adapter()
        adapter.validate_support()
        return adapter.spec

    @property
    def training_default_model_name_or_path(self) -> str:
        """Return the recommended differentiable checkpoint for this family."""
        spec = self.get_training_adapter().spec
        return (spec.training_default_model_name_or_path or self.config.name_or_path)

    def _validate_training_runtime(self) -> None:
        """Optional pre-load validation for inference-only variants."""

    def _prepare_for_training(self) -> None:
        """Optional hook that validates or restores a trainable runtime."""

    def _prepare_for_inference(self) -> None:
        """Optional hook that restores an inference-capable runtime.

        Portable Trainer artifacts are initially loaded through the
        differentiable graph so their component topology cannot be
        pruned or fused before state restoration. Architectures whose
        training graph omits serving-only tokenizers, codecs, caches, or
        processors can rebuild those objects here after the trained
        state has been applied.
        """
        evaluate = getattr(self.model, "eval", None)
        if callable(evaluate):
            evaluate()

    def _set_training_device(self, device: str) -> None:
        """Synchronize wrapper/runtime device metadata after strategy moves."""
        self.device = str(device)

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Resolve ``auto`` only when loading, keeping package imports
        cheap."""
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

    @abstractmethod
    def _load_pretrained_model(self) -> None:
        """Build the local source architecture and load its checkpoint."""
        ...

    @abstractmethod
    def _generate(self, text: str, **kwargs) -> TTSOutput:
        """Backend generation hook called by the uniform public API."""
        ...

    def prepare_inputs_for_generation(
        self,
        text: str,
        **kwargs,
    ) -> dict[str, Any]:
        """Normalize raw inputs through the configured processor."""
        return dict(self.processor(text, **kwargs))

    def _validate_generation_inputs(
        self,
        model_inputs: dict[str, Any],
    ) -> None:
        """Validate one request before allocating the model runtime.

        Backends should override this hook for dependency-free checks
        such as required reference audio, mutually exclusive options,
        and supported modes. Tensor- or model-dependent validation
        remains in :meth:`_generate`.
        """

    def _validate_common_generation_inputs(
        self,
        model_inputs: dict[str, Any],
    ) -> None:
        """Validate common options even when callers invoke ``forward``."""
        common_options = {
            name: model_inputs[name]
            for name in self.generation_config_class._COMMON_FIELDS if name in model_inputs
        }
        self.generation_config_class(**common_options)

    def forward(self, text: str, **kwargs) -> TTSOutput:
        """Run text-to-speech using the backend implementation hook."""
        backend_config, backend_client = self._reserve_llm_backend()
        try:
            return self._forward_with_llm_backend(
                text,
                kwargs,
                backend_config=backend_config,
                backend_client=backend_client,
            )
        finally:
            self._release_llm_backend(backend_config)

    def _llm_speech_support(self, backend_config):
        """Resolve the immutable speech capability for one backend snapshot."""
        if backend_config is None:
            return None
        from voicehub.llm_serving import LLMBackendTransport, get_llm_backend_support

        support, transport = get_llm_backend_support(
            self.config.model_type,
            backend_config.backend,
            transport=backend_config.transport,
        )
        return support if transport is LLMBackendTransport.SPEECH else None

    def _forward_with_llm_backend(
        self,
        text: str,
        kwargs: Mapping[str, Any],
        *,
        backend_config,
        backend_client,
    ) -> TTSOutput:
        """Generate against the immutable backend snapshot reserved by
        caller."""
        from voicehub.llm_serving import LLMBackendTransport

        transport = (LLMBackendTransport.AUTO if backend_config is None else backend_config.transport)
        uses_speech_backend = transport is LLMBackendTransport.SPEECH
        uses_token_backend = transport is LLMBackendTransport.TOKENS
        speech_support = (self._llm_speech_support(backend_config) if uses_speech_backend else None)
        model_inputs = self.prepare_inputs_for_generation(text, **kwargs)
        with self._lifecycle_lock:
            self._validate_model_kwargs(
                model_inputs,
                uses_llm_speech_backend=uses_speech_backend,
                llm_backend_support=speech_support,
            )
            self._validate_common_generation_inputs(model_inputs)
            if not uses_speech_backend:
                self._validate_generation_inputs(model_inputs)
            if uses_speech_backend:
                if backend_client is None:
                    raise RuntimeError("The configured external speech client is missing.")
                self._inference_ready = True
            elif uses_token_backend:
                self.load()
            else:
                self.load()
                from voicehub.optimization import diffusion_cache_request

                with diffusion_cache_request(self.model):
                    output = self._generate(**model_inputs)
        if uses_speech_backend:
            output = backend_client.synthesize(
                self.config.model_type,
                model_inputs,
                default_sample_rate=self.sample_rate,
            )
        elif uses_token_backend:
            from voicehub.optimization import diffusion_cache_request

            with diffusion_cache_request(self.model):
                output = self._generate(**model_inputs)
        if not isinstance(output, TTSOutput):
            raise TypeError(
                f"{self.__class__.__name__}._generate() must return a "
                f"TTSOutput, received {type(output).__name__}.")
        return output

    def _validate_model_kwargs(
        self,
        model_kwargs: dict[str, Any],
        *,
        uses_llm_speech_backend: bool | None = None,
        llm_backend_support=None,
    ) -> None:
        """Reject misspelled generation options with an actionable error."""
        if uses_llm_speech_backend is None:
            uses_llm_speech_backend = self.uses_llm_speech_backend
        if uses_llm_speech_backend:
            support = llm_backend_support or self._llm_speech_support(self._llm_backend_config)
            if support is None:
                raise RuntimeError("The configured external speech capability is missing.")
            recognized_options = set(support.speech_input_options)
            unknown = sorted(set(model_kwargs) - recognized_options - {"text"})
            if unknown:
                recognized = ", ".join(support.speech_input_options)
                invalid = ", ".join(unknown)
                raise ValueError(
                    f"Unsupported external speech option(s): {invalid}. "
                    f"Recognized options: {recognized}.")
            return
        parameters = signature(self._generate).parameters
        has_passthrough = any(parameter.kind is Parameter.VAR_KEYWORD for parameter in parameters.values())
        if has_passthrough and self.passthrough_generation_options is None:
            return
        supported_options = {
            name
            for name, parameter in parameters.items() if name != "self" and parameter.kind not in {
                Parameter.VAR_KEYWORD,
                Parameter.VAR_POSITIONAL,
            }
        }
        if self.passthrough_generation_options is not None:
            supported_options.update(self.passthrough_generation_options)
        unknown = sorted(set(model_kwargs) - supported_options)
        if unknown:
            supported = ", ".join(sorted(supported_options))
            invalid = ", ".join(unknown)
            raise ValueError(
                f"Unsupported generation option(s): {invalid}. "
                f"{self.__class__.__name__} accepts: {supported}.")

    def generate(
        self,
        text: str,
        *,
        generation_config: TTSGenerationConfig | None = None,
        **kwargs,
    ) -> TTSOutput:
        """Generate speech with one signature shared by every architecture."""
        backend_config, backend_client = self._reserve_llm_backend()
        try:
            defaults = self.generation_config.to_dict()
            speech_support = self._llm_speech_support(backend_config)
            if speech_support is not None:
                # Architecture-native defaults frequently contain controls for
                # local vocoders or custom logits processors. The external server
                # owns those defaults; carry only fields represented by the
                # shared speech protocol. Explicit per-call options still fail
                # closed in the backend adapter when they cannot be preserved.
                defaults = {
                    name: value
                    for name, value in defaults.items() if name in speech_support.speech_default_options
                }
            if generation_config is not None:
                if not isinstance(generation_config, TTSGenerationConfig):
                    raise TypeError("`generation_config` must be a TTSGenerationConfig.")
                defaults.update(generation_config.to_dict())
            defaults.update(kwargs)
            generation_options = self.generation_config_class.from_dict(defaults).to_dict()
            # Generation configs store only set values. Re-apply explicit
            # per-call ``None`` overrides so they disable a default instead of
            # being silently replaced by it.
            generation_options.update({name: None for name, value in kwargs.items() if value is None})
            return self._forward_with_llm_backend(
                text,
                generation_options,
                backend_config=backend_config,
                backend_client=backend_client,
            )
        finally:
            self._release_llm_backend(backend_config)

    def __call__(
        self,
        text: str,
        *,
        generation_config: TTSGenerationConfig | None = None,
        **kwargs,
    ) -> TTSOutput:
        return self.generate(
            text,
            generation_config=generation_config,
            **kwargs,
        )

    @classmethod
    def can_generate(cls) -> bool:
        """Return whether this architecture implements generation."""
        return cls._generate is not PreTrainedTTSModel._generate

    def save_pretrained(
        self,
        save_directory: str | Path,
        *,
        include_native_export: bool = True,
    ) -> Path:
        """Save VoiceHub metadata and optionally namespaced native
        artifacts."""
        if include_native_export and self._llm_backend_config is not None:
            raise RuntimeError(
                "The external LLM server owns the active model weights. Use "
                "`include_native_export=False` to save only VoiceHub metadata, "
                "or create a native wrapper for a self-contained export.")
        output_directory = Path(save_directory).expanduser()
        output_directory.mkdir(parents=True, exist_ok=True)
        self.config.save_pretrained(output_directory)
        self.generation_config.save_pretrained(output_directory)
        self.processor.save_pretrained(output_directory)
        if include_native_export:
            self._save_pretrained(output_directory / NATIVE_EXPORT_DIR)
        return output_directory

    def export_native_pretrained(
        self,
        save_directory: str | Path,
    ) -> Path:
        """Write a flat runtime-native artifact for trainer interop.

        Unlike :meth:`save_pretrained`, this method does not write
        VoiceHub's portable wrapper metadata or introduce another
        ``native_export`` directory. Specialized training adapters can
        therefore export directly into the namespace allocated by
        :class:`voicehub.Trainer`.
        """
        if self._llm_backend_config is not None:
            raise RuntimeError(
                "External vLLM/SGLang backends cannot export native weights. "
                "Export from the serving engine or create a native wrapper.")
        destination = Path(save_directory).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        self._save_pretrained(destination)
        return destination

    def get_training_adapter(self):
        """Return the unloaded model-family adapter paired with this model."""
        from voicehub.training.auto import AutoTrainingAdapter

        return AutoTrainingAdapter.from_model(self)

    def create_training_dataset(self, records, **kwargs):
        """Build this architecture's source-native fine-tuning dataset.

        The returned dataset exposes ``collate_fn`` when its token or
        acoustic layout needs model-specific batching. ``records`` may
        also be a JSON/JSONL/CSV/TSV manifest path. Manifest audio paths
        are resolved relative to the manifest before the selected model
        adapter constructs tokens, codec codes, or acoustic targets.
        """
        from voicehub.training.datasets import TTSDataset

        data_root = kwargs.pop("data_root", None)
        data_aliases = kwargs.pop("data_aliases", None)
        validate_records = kwargs.pop("validate_records", None)
        validate_audio_files = kwargs.pop("validate_audio_files", False)
        should_coerce = isinstance(records, (str, Path, TTSDataset))
        should_coerce = should_coerce or validate_records is not None
        should_coerce = should_coerce or data_root is not None
        should_coerce = should_coerce or data_aliases is not None
        should_coerce = should_coerce or bool(validate_audio_files)
        if should_coerce:
            records = TTSDataset.coerce(
                records,
                model_type=self.config.model_type,
                root=data_root,
                aliases=data_aliases,
                validate=(True if validate_records is None else bool(validate_records)),
                validate_files=bool(validate_audio_files),
            )
        return self.get_training_adapter().create_dataset(records, **kwargs)

    def prepare_training_inputs(
        self,
        inputs: dict[str, Any],
        *,
        phase: str,
    ) -> dict[str, Any]:
        """Map a canonical TTS batch to one native training phase.

        Architectures with tokenizer, codec, or acoustic feature
        preparation can override this hook.  Preprocessed datasets pass
        through unchanged.
        """
        return dict(inputs)

    def _save_pretrained(self, save_directory: Path) -> None:
        """Optionally serialize native artifacts in their own namespace."""
