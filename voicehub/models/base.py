from abc import ABC, abstractmethod
from collections.abc import Sequence
from contextlib import nullcontext
from importlib import import_module
from math import isfinite
from numbers import Integral, Number, Real
from pathlib import Path
from typing import Any


class BaseSpeechModel(ABC):
    """Framework-independent base for every VoiceHub speech model.

    The class owns task-neutral waveform validation, persistence, and
    explicit optimization-plan state. Task semantics such as synthesis,
    transcription, and speech detection belong to the corresponding
    pretrained model subclasses.
    """

    def __init__(self, model_path: str = "", device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self._optimization_results_by_mode: dict[str, Any] = {}
        self._tts_optimization_results_by_mode: dict[str, Any] = {}
        self._tts_optimization_attaches_model_by_mode: dict[str, bool] = {}

    def _optimization_lifecycle_lock(self):
        lock = getattr(self, "_lifecycle_lock", None)
        return lock if lock is not None else nullcontext()

    @staticmethod
    def _optimization_mode(mode):
        from voicehub.optimization import OptimizationMode

        return OptimizationMode.coerce(mode)

    def _validate_optimization_transition(self, target_mode) -> None:
        """Reject implicit transitions across active optimization plans."""
        normalized = self._optimization_mode(target_mode)
        explicit_modes = tuple(
            mode for mode, result in self._optimization_results_by_mode.items()
            if result is not None and mode != normalized.value)
        universal_modes = tuple(
            mode for mode, result in self._tts_optimization_results_by_mode.items()
            if result is not None and mode != normalized.value)
        if explicit_modes:
            raise RuntimeError(
                f"Cannot enter {normalized.value!r} mode while an explicit "
                f"{explicit_modes[0]!r} optimization plan is active. Restore "
                "that plan explicitly with restore_optimization_plan() first.")
        if universal_modes:
            raise RuntimeError(
                f"Cannot enter {normalized.value!r} mode while a universal "
                f"{universal_modes[0]!r} TTS optimization policy is active. "
                "Restore that policy explicitly with "
                "restore_tts_optimization() first.")

    def _default_optimization_context(self, mode):
        """Describe the loaded runtime without importing optimization tools."""
        from voicehub.optimization import OptimizationContext

        runtime = getattr(self, "model", None)
        device = str(getattr(self, "device", "cpu"))
        dtype = "float32"
        parameters = getattr(runtime, "parameters", None)
        if callable(parameters):
            try:
                parameter = next(iter(parameters()))
            except (StopIteration, TypeError):
                parameter = None
            if parameter is not None:
                parameter_device = getattr(parameter, "device", None)
                parameter_dtype = getattr(parameter, "dtype", None)
                if parameter_device is not None:
                    device = str(parameter_device)
                if parameter_dtype is not None:
                    dtype = str(parameter_dtype).removeprefix("torch.")
        return OptimizationContext(
            mode=mode,
            device=device,
            dtype=dtype,
        )

    @classmethod
    def available_optimization_passes(cls) -> tuple[str, ...]:
        """List every pass registered for speech-model optimization.

        Availability is global and lazy. Compatibility with one loaded
        model is checked transactionally by the concrete pass before any
        mutation.
        """
        del cls
        from voicehub.optimization import OPTIMIZATION_PASSES

        return OPTIMIZATION_PASSES.list()

    def apply_optimization_plan(
        self,
        passes,
        *,
        mode,
        context=None,
        registry=None,
    ):
        """Explicitly apply named or configured passes to the model runtime.

        No plan is selected automatically. The returned
        :class:`~voicehub.optimization.OptimizationResult` owns rollback
        state and a deterministic manifest for the exact application.
        """
        from voicehub.optimization import OptimizationContext, OptimizationPassManager

        normalized_mode = self._optimization_mode(mode)
        if context is not None:
            if not isinstance(context, OptimizationContext):
                raise TypeError("`context` must be an OptimizationContext.")
            if context.mode is not normalized_mode:
                raise ValueError(
                    "Optimization context mode does not match the requested "
                    f"mode ({context.mode.value!r} != "
                    f"{normalized_mode.value!r}).")

        manager = OptimizationPassManager()
        resolved_passes = manager.resolve(passes, registry=registry)
        if not resolved_passes:
            raise ValueError("An optimization plan must contain at least one pass.")

        with self._optimization_lifecycle_lock():
            if getattr(
                    self,
                    "_pending_tts_optimization_config",
                    None,
            ) is not None:
                raise RuntimeError(
                    "A universal TTS optimization configuration is pending "
                    "for the next inference load. Call load() to apply it or "
                    "clear_optimization_config() before applying an explicit "
                    "plan.")
            self._validate_optimization_transition(normalized_mode)
            if normalized_mode.value in self._optimization_results_by_mode:
                raise RuntimeError(
                    f"An explicit {normalized_mode.value!r} optimization plan "
                    "is already active. Restore it before applying another.")
            if normalized_mode.value in self._tts_optimization_results_by_mode:
                raise RuntimeError(
                    f"A universal {normalized_mode.value!r} TTS optimization "
                    "policy is already active. Restore it before applying an "
                    "explicit plan.")

            loader_name = ("load_for_training" if normalized_mode.value == "training" else "load")
            loader = getattr(self, loader_name, None)
            if not callable(loader):
                raise TypeError(
                    f"{type(self).__name__} does not implement "
                    f"{loader_name}(), which is required for "
                    f"{normalized_mode.value} optimization.")
            loader()
            return self._apply_resolved_optimization_plan(
                resolved_passes,
                mode=normalized_mode,
                context=context,
                manager=manager,
            )

    def _apply_resolved_optimization_plan(
        self,
        resolved_passes,
        *,
        mode,
        context=None,
        manager=None,
        declaration_snapshots=None,
        runtime=None,
        attach_to_model=True,
    ):
        """Apply pre-resolved passes to an already loaded runtime."""
        from voicehub.optimization import OptimizationPassManager, bind_registered_architecture

        normalized_mode = self._optimization_mode(mode)
        if runtime is None:
            runtime = getattr(self, "model", None)
        if runtime is None:
            raise RuntimeError(
                f"{type(self).__name__} did not expose a loaded model "
                "runtime for optimization.")
        resolved_context = (
            context if context is not None else self._default_optimization_context(normalized_mode))
        resolved_context = bind_registered_architecture(
            resolved_context,
            self,
        )
        pass_manager = (OptimizationPassManager() if manager is None else manager)
        result = pass_manager.apply(
            runtime,
            resolved_passes,
            resolved_context,
            declaration_snapshots=declaration_snapshots,
        )
        # Validate manifest safety before attaching the transformed graph.
        result.manifest()
        if attach_to_model:
            self.model = result.model
        self._optimization_results_by_mode[normalized_mode.value] = result
        return result

    def optimization_result(self, *, mode):
        """Return the active result for one mode, if a plan was applied."""
        normalized = self._optimization_mode(mode)
        return self._optimization_results_by_mode.get(normalized.value)

    def optimization_manifest(self, *, mode=None):
        """Return checkpoint-safe metadata for active explicit plans."""
        if mode is not None:
            result = self.optimization_result(mode=mode)
            return None if result is None else result.manifest()
        return {
            active_mode: result.manifest()
            for active_mode, result in sorted(self._optimization_results_by_mode.items())
        }

    def restore_optimization_plan(self, *, mode):
        """Restore one reversible plan and return the original runtime."""
        normalized = self._optimization_mode(mode)
        with self._optimization_lifecycle_lock():
            result = self._optimization_results_by_mode.get(normalized.value)
            if result is None:
                raise RuntimeError(f"No {normalized.value!r} optimization plan is active.")
            restored = result.restore()
            attaches_model = self._tts_optimization_attaches_model_by_mode.get(
                normalized.value,
                True,
            )
            if attaches_model:
                self.model = restored
            del self._optimization_results_by_mode[normalized.value]
            self._tts_optimization_results_by_mode.pop(
                normalized.value,
                None,
            )
            self._tts_optimization_attaches_model_by_mode.pop(
                normalized.value,
                None,
            )
            return restored

    @abstractmethod
    def __call__(self, *args, **kwargs):
        ...

    @staticmethod
    def validate_audio(audio_data: Any) -> None:
        """Validate a materialized waveform without changing its public
        type."""
        if audio_data is None:
            raise ValueError("`audio_data` cannot be None.")

        numel = getattr(audio_data, "numel", None)
        tensor_is_finite = getattr(audio_data, "isfinite", None)
        if callable(numel) and callable(tensor_is_finite):
            if int(numel()) == 0:
                raise ValueError("`audio_data` cannot be empty.")
            if str(getattr(audio_data, "dtype", "")) in {
                    "bool",
                    "torch.bool",
            }:
                raise TypeError("`audio_data` must contain real numeric samples.", )
            is_complex = getattr(audio_data, "is_complex", None)
            if callable(is_complex) and is_complex():
                raise TypeError("`audio_data` must contain real numeric samples.", )
            finite = tensor_is_finite().all()
            if hasattr(finite, "item"):
                finite = finite.item()
            if not bool(finite):
                raise ValueError("`audio_data` contains NaN or infinite samples.", )
            return

        def validate_sequence(value: Any) -> int | None:
            if isinstance(value, Number):
                if isinstance(value, bool) or not isinstance(value, Real):
                    raise TypeError("`audio_data` must contain real numeric samples.", )
                if not isfinite(value):
                    raise ValueError("`audio_data` contains NaN or infinite samples.", )
                return 1
            if isinstance(value, (str, bytes, bytearray, dict)):
                raise TypeError("`audio_data` must contain real numeric samples.", )
            if not isinstance(value, Sequence):
                return None
            sample_count = 0
            for sample in value:
                nested_count = validate_sequence(sample)
                if nested_count is None:
                    return None
                sample_count += nested_count
            return sample_count

        sample_count = validate_sequence(audio_data)
        if sample_count is not None:
            if sample_count == 0:
                raise ValueError("`audio_data` cannot be empty.")
            return

        # NumPy-style arrays are intentionally supported through their public
        # array protocol without importing NumPy. This keeps waveform
        # validation independent from both NumPy and whichever tensor module a
        # provider happens to expose in a test or adapter process.
        array_size = getattr(audio_data, "size", None)
        flat_samples = getattr(audio_data, "flat", None)
        if (isinstance(array_size, Integral) and flat_samples is not None and not callable(flat_samples)):
            if array_size == 0:
                raise ValueError("`audio_data` cannot be empty.")
            dtype_kind = getattr(
                getattr(audio_data, "dtype", None),
                "kind",
                None,
            )
            if dtype_kind is not None and dtype_kind not in {"i", "u", "f"}:
                raise TypeError("`audio_data` must contain real numeric samples.")
            validated_count = 0
            for sample in flat_samples:
                item = getattr(sample, "item", None)
                if callable(item):
                    sample = item()
                nested_count = validate_sequence(sample)
                if nested_count is None:
                    raise TypeError("`audio_data` must contain numeric samples.")
                validated_count += nested_count
            if validated_count != int(array_size):
                raise ValueError("`audio_data` reported an inconsistent array size.")
            return

        torch = import_module("torch")
        try:
            audio_tensor = torch.as_tensor(audio_data)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise TypeError("`audio_data` must contain numeric samples.", ) from exc
        if audio_tensor.numel() == 0:
            raise ValueError("`audio_data` cannot be empty.")
        if audio_tensor.dtype == torch.bool or audio_tensor.is_complex():
            raise TypeError("`audio_data` must contain real numeric samples.", )
        if not torch.isfinite(audio_tensor).all():
            raise ValueError("`audio_data` contains NaN or infinite samples.")

    @staticmethod
    def save_audio(
        file_path: str | Path,
        audio_data: Any,
        sample_rate: int,
    ) -> str:
        """Write one mono waveform as portable 16-bit PCM WAVE audio."""
        if not isinstance(file_path, (str, Path)) or not str(file_path).strip():
            raise ValueError("`file_path` must be a non-empty path.")
        if (isinstance(sample_rate, bool) or not isinstance(sample_rate, Integral) or sample_rate <= 0):
            raise ValueError("`sample_rate` must be a positive integer.")

        BaseSpeechModel.validate_audio(audio_data)
        torch = import_module("torch")
        try:
            waveform = torch.as_tensor(audio_data).detach()
        except (TypeError, ValueError, RuntimeError) as exc:
            raise TypeError("`audio_data` must contain numeric samples.") from exc
        waveform = waveform.squeeze()
        if waveform.ndim == 0:
            waveform = waveform.reshape(1)
        if waveform.ndim != 1:
            raise ValueError(
                "`audio_data` must contain one mono waveform; "
                f"received shape {tuple(waveform.shape)}.")

        output_path = Path(file_path).expanduser()
        if output_path.exists() and output_path.is_dir():
            raise IsADirectoryError(f"Audio output path is a directory: {output_path}.")
        if output_path.suffix.lower() not in {".wav", ".wave"}:
            raise ValueError(
                "VoiceHub's dependency-free audio writer supports PCM WAVE "
                "output. Use a `.wav` or `.wave` file name.")

        from voicehub.processing.waveform import normalize_waveform, save_pcm_wave

        waveform = normalize_waveform(waveform)
        save_pcm_wave(output_path, waveform, int(sample_rate))
        return str(output_path)


class BaseTTSModel(BaseSpeechModel):
    """Backward-compatible base class for text-to-speech models."""

    def _tts_optimization_runtime(self, mode):
        """Return the execution object transformed by a universal policy."""
        del mode
        return self.model

    def resolve_optimization(
        self,
        config=None,
        *,
        mode="inference",
        context=None,
        registry=None,
    ):
        """Resolve a universal TTS policy without loading model weights."""
        from voicehub.optimization import resolve_tts_optimization

        return resolve_tts_optimization(
            self,
            config,
            mode=mode,
            context=context,
            registry=registry,
        )

    def _optimize_loaded_tts_runtime(
        self,
        config=None,
        *,
        mode="inference",
        context=None,
        registry=None,
    ):
        """Apply one universal policy after the native runtime is loaded."""
        from voicehub.optimization import OptimizationPassManager, TTSOptimizationResult

        normalized_mode = self._optimization_mode(mode)
        if normalized_mode.value in self._tts_optimization_results_by_mode:
            raise RuntimeError(
                f"A universal {normalized_mode.value!r} TTS optimization "
                "policy is already active. Restore it before applying "
                "another.")
        if normalized_mode.value in self._optimization_results_by_mode:
            raise RuntimeError(
                f"An explicit {normalized_mode.value!r} optimization plan "
                "is already active. Restore it before applying a universal "
                "TTS policy.")
        plan = self.resolve_optimization(
            config,
            mode=normalized_mode,
            context=context,
            registry=registry,
        )
        # Validate requested/resolved metadata before the first graph
        # transformation so a non-serializable policy cannot leave a partial
        # low-level optimization active.
        plan.manifest()
        application = None
        attaches_model = True
        if plan.passes:
            manager = OptimizationPassManager()
            runtime = self._tts_optimization_runtime(normalized_mode)
            if runtime is None:
                raise RuntimeError(
                    f"{type(self).__name__} did not expose a loaded "
                    f"{normalized_mode.value} runtime for optimization.")
            attaches_model = runtime is self.model
            application = self._apply_resolved_optimization_plan(
                plan.passes,
                mode=normalized_mode,
                context=plan.context,
                manager=manager,
                declaration_snapshots=(plan.pass_declaration_snapshots),
                runtime=runtime,
                attach_to_model=attaches_model,
            )
        result = TTSOptimizationResult(
            plan=plan,
            model=(self.model if application is None or attaches_model else application.model),
            application=application,
        )
        # Validate strict JSON before publishing even a native fallback.
        result.manifest()
        self._tts_optimization_results_by_mode[normalized_mode.value] = result
        self._tts_optimization_attaches_model_by_mode[normalized_mode.value] = attaches_model
        return result

    def optimize(
        self,
        config=None,
        *,
        mode="inference",
        context=None,
        registry=None,
    ):
        """Load and optimize any TTS model through the universal resolver."""
        from voicehub.optimization import validate_tts_optimization_config

        normalized_mode = self._optimization_mode(mode)
        with self._optimization_lifecycle_lock():
            if getattr(
                    self,
                    "_pending_tts_optimization_config",
                    None,
            ) is not None:
                raise RuntimeError(
                    "A universal TTS optimization configuration is pending "
                    "for the next inference load. Call load() to apply it or "
                    "clear_optimization_config() before calling optimize().")
            self._validate_optimization_transition(normalized_mode)
            if (normalized_mode.value in self._tts_optimization_results_by_mode):
                raise RuntimeError(
                    f"A universal {normalized_mode.value!r} TTS "
                    "optimization policy is already active. Restore it "
                    "before applying another.")
            resolved_config = validate_tts_optimization_config(
                self,
                config,
                mode=normalized_mode,
            )
            loader_name = ("load_for_training" if normalized_mode.value == "training" else "load")
            loader = getattr(self, loader_name, None)
            if not callable(loader):
                raise TypeError(
                    f"{type(self).__name__} does not implement "
                    f"{loader_name}(), which is required for "
                    f"{normalized_mode.value} optimization.")
            loader()
            return self._optimize_loaded_tts_runtime(
                resolved_config,
                mode=normalized_mode,
                context=context,
                registry=registry,
            )

    def tts_optimization_result(self, *, mode="inference"):
        """Return the universal policy result for one execution mode."""
        normalized = self._optimization_mode(mode)
        return self._tts_optimization_results_by_mode.get(normalized.value)

    def tts_optimization_manifest(self, *, mode=None):
        """Return requested, resolved, and applied universal TTS settings."""
        if mode is not None:
            result = self.tts_optimization_result(mode=mode)
            return None if result is None else result.manifest()
        return {
            active_mode: result.manifest()
            for active_mode, result in sorted(self._tts_optimization_results_by_mode.items())
        }

    def restore_tts_optimization(self, *, mode="inference"):
        """Restore a transformed policy or clear a native fallback report."""
        normalized = self._optimization_mode(mode)
        with self._optimization_lifecycle_lock():
            result = self._tts_optimization_results_by_mode.get(normalized.value)
            if result is None:
                raise RuntimeError(
                    f"No universal {normalized.value!r} TTS optimization "
                    "policy is active.")
            application = result.application
            low_level = self._optimization_results_by_mode.get(normalized.value)
            if application is None and low_level is not None:
                raise RuntimeError(
                    "A native TTS optimization policy cannot own an active "
                    "low-level optimization result.")
            if application is not None and low_level is not application:
                raise RuntimeError(
                    "The universal TTS optimization policy no longer owns "
                    "its low-level application result.")
            restored = result.restore()
            attaches_model = self._tts_optimization_attaches_model_by_mode.get(
                normalized.value,
                True,
            )
            if attaches_model:
                self.model = restored
            self._tts_optimization_results_by_mode.pop(
                normalized.value,
                None,
            )
            self._tts_optimization_attaches_model_by_mode.pop(
                normalized.value,
                None,
            )
            if application is not None:
                self._optimization_results_by_mode.pop(
                    normalized.value,
                    None,
                )
            return restored
