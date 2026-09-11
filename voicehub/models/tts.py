"""Transformers-style pretrained model lifecycle for speech models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from inspect import Parameter, signature
from pathlib import Path
from typing import Any

from voicehub.configuration import VoiceHubConfig
from voicehub.generation.configuration import TTSGenerationConfig
from voicehub.inference_strategy import InferenceStrategy, get_inference_strategy
from voicehub.model import PreTrainedSpeechModel
from voicehub.models.base import BaseTTSModel
from voicehub.outputs import TTSOutput
from voicehub.processing.processor import VoiceHubProcessor


class PreTrainedTTSModel(BaseTTSModel, PreTrainedSpeechModel, ABC):
    """Base lifecycle shared by source-integrated VoiceHub runtime."""

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
        super().__init__(config, device=device, lazy_load=True)
        self.generation_config = self.generation_config_class.from_model_config(config)
        self._pending_tts_optimization_config = None
        self._llm_backend_config = None
        self._llm_backend_client = None
        self._active_generation_requests = 0
        self._active_llm_requests = 0
        if not lazy_load:
            self.load()

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
        from voicehub.optimization import tts_optimization_config_from_options

        options = tts_optimization_config_from_options(
            optimization_config,
            attn_implementation=attn_implementation,
            kernel_backend=kernel_backend,
            torch_compile=torch_compile,
            compile_config=compile_config,
            diffusion_cache=diffusion_cache,
            diffusion_cache_config=diffusion_cache_config,
            diffusion_sampling=diffusion_sampling,
            diffusion_sampling_config=diffusion_sampling_config,
        )
        external = llm_backend is not None or llm_backend_config is not None
        if external and options is not None:
            raise ValueError(
                "Choose either an external LLM backend or an in-process TTS optimization configuration.")
        if external and inference_strategy is not None and get_inference_strategy(
                inference_strategy).name != "eager":
            raise ValueError("External LLM serving requires the eager wrapper-side inference strategy.")
        model = super().from_pretrained(
            pretrained_model_name_or_path,
            config=config,
            device=device,
            lazy_load=True,
            inference_strategy=inference_strategy,
            config_kwargs=config_kwargs,
            **kwargs,
        )
        if options is not None:
            model.set_optimization_config(options)
        if external:
            if model._pending_model_state_path is not None:
                raise ValueError("External LLM serving does not restore a local VoiceHub trainer state.")
            backend = llm_backend
            if backend is None:
                backend = llm_backend_config.get("backend") if isinstance(
                    llm_backend_config, Mapping) else getattr(llm_backend_config, "backend", None)
            if backend is None:
                raise ValueError(
                    "`llm_backend` is required when `llm_backend_config` does not declare a backend.")
            model.set_llm_backend(backend, config=llm_backend_config)
        if not lazy_load:
            model.load()
        return model

    def load(self):
        with self._lifecycle_lock:
            if self.uses_llm_speech_backend:
                if self.is_training_load:
                    raise RuntimeError("An external speech backend cannot load for training.")
                self._inference_ready = True
                return self
            return super().load()

    def load_for_training(self):
        with self._lifecycle_lock:
            if self._llm_backend_config is not None:
                raise RuntimeError(
                    "External backends are inference-only. Create a native wrapper for training.")
            return super().load_for_training()

    def _after_prepare_for_inference(self) -> None:
        pending = self._pending_tts_optimization_config
        if pending is not None:
            self._optimize_loaded_tts_runtime(pending, mode="inference")
            self._pending_tts_optimization_config = None

    def _restore_inference_settings(self, directory: Path) -> None:
        if (directory / "generation_config.json").is_file():
            self.generation_config = self.generation_config_class.from_pretrained(directory)

    def _save_inference_settings(self, directory: Path) -> None:
        self.generation_config.save_pretrained(directory)

    def save_pretrained(self, save_directory, *, include_native_export=True):
        if include_native_export and self._llm_backend_config is not None:
            raise RuntimeError("The external server owns the weights; use include_native_export=False.")
        return super().save_pretrained(save_directory, include_native_export=include_native_export)

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
            generation_options = self.generation_config_class.from_dict(defaults)
            return self._forward_with_llm_backend(
                text,
                generation_options.to_dict(),
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
