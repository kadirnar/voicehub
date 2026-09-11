"""Trainable-module discovery and phase-aware objective adapters."""

from __future__ import annotations

import inspect
from collections import deque
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from voicehub.dependencies import import_optional
from voicehub.outputs import SpeechTrainingOutput, TTSTrainingOutput
from voicehub.tasks import SpeechTask
from voicehub.training.collators import DataCollatorForAudioTraining
from voicehub.training.contracts import TrainingContext, TrainingPhaseKind, TrainingPhaseSpec, TrainingSupport
from voicehub.training.specs import ModelTrainingSpec


class BaseTrainingAdapter:
    """Expose an inference wrapper's source modules through a training API.

    Backends with unusual preprocessing should override
    :meth:`prepare_training_inputs`. Backends with non-standard orchestration
    can override :meth:`execute_training_phase` while retaining phase planning,
    parameter routing, and checkpoint behavior.
    """

    ADAPTER_STATE_VERSION = 2
    SUPPORTED_ADAPTER_STATE_VERSIONS = (1, 2)
    RECIPE_VERSION = 1
    supports_custom_recipe = False
    supports_quantized_training = False
    supports_compiled_training = False
    native_export_semantics = "component-weight-warm-start"
    SOURCE_METADATA_FIELDS = frozenset({
        "consent",
        "id",
        "license",
        "metadata",
        "session_id",
        "source",
    })

    def __init__(self, model, spec: ModelTrainingSpec):
        if not isinstance(spec, ModelTrainingSpec):
            raise TypeError("Training adapters require a ModelTrainingSpec.")
        self.model = model
        self.spec = spec
        self.primary_model = None
        self.primary_path: str | None = None
        self._components: list[tuple[str, Any]] = []
        self._component_by_path: dict[str, Any] = {}
        self._current_context: TrainingContext | None = None
        self._registered_specialization = False
        self._runtime_input_preparer: Callable[[Any], Any] | None = None
        self.data_collator = DataCollatorForAudioTraining(field_schemas=self.spec.field_schemas, )

    @property
    def model_type(self) -> str:
        return self.spec.model_type

    @property
    def is_ready(self) -> bool:
        return self.primary_model is not None

    @property
    def current_context(self) -> TrainingContext | None:
        """The context being executed, or ``None`` outside a forward call."""
        return self._current_context

    @property
    def current_phase(self) -> TrainingPhaseSpec:
        if self._current_context is not None:
            return self._current_context.phase
        return self.spec.get_phase()

    def setup(self):
        """Load the wrapper for training and resolve all source components."""
        if self.is_ready:
            self._validate_loaded_training_graph()
            return self
        self.validate_support()

        load_for_training = getattr(self.model, "load_for_training", None)
        if callable(load_for_training):
            load_for_training()
        elif hasattr(self.model, "load"):
            self.model.load()

        component_paths = list(self.spec.component_paths)
        for phase in self.spec.phases:
            component_paths.extend(phase.component_paths)
            if phase.forward_component is not None:
                component_paths.append(phase.forward_component)
        component_paths = list(dict.fromkeys(component_paths))

        components = []
        for path in component_paths:
            candidate = self._resolve_path(path)
            if candidate is None:
                continue
            if self._has_trainable_parameters(candidate):
                self._component_by_path[path] = candidate
                components.append((path, candidate))

        for path in self.spec.module_paths:
            candidate = self._resolve_path(path)
            if not self._is_forward_module(candidate):
                continue
            self.primary_model = candidate
            self.primary_path = path
            self._component_by_path[path] = candidate
            components.insert(0, (path, candidate))
            break

        discovered = []
        if self.primary_model is None and self.spec.allow_module_discovery:
            discovered = self._discover_trainable_modules(self.model)
            for path, candidate in discovered:
                self._component_by_path.setdefault(path, candidate)
            for path, candidate in discovered:
                if self._is_forward_module(candidate):
                    self.primary_model = candidate
                    self.primary_path = path
                    break
            components.extend(discovered)

        if self.primary_model is None:
            # A phase can select a callable source component even when the
            # wrapper has no single root nn.Module.
            for path in component_paths:
                candidate = self._resolve_path(path)
                if self._is_forward_module(candidate):
                    self.primary_model = candidate
                    self.primary_path = path
                    self._component_by_path[path] = candidate
                    components.insert(0, (path, candidate))
                    break

        if self.primary_model is None:
            checked = ", ".join(self.spec.module_paths)
            discovery_hint = (
                " Bounded module discovery is disabled for this production "
                "profile; declare an exact path or explicitly opt in."
                if not self.spec.allow_module_discovery else "")
            raise TypeError(
                f"{self.model_type!r} loaded successfully but no trainable "
                f"callable module was found. Checked: {checked}."
                f"{discovery_hint}")

        self._components = self._deduplicate_components(components)
        if not self._components:
            self._components = [(self.primary_path or "model", self.primary_model)]
        self._validate_loaded_training_graph()
        return self

    def build_training_graph(self):
        """Construct and validate the recipe-owned trainable graph.

        This is the public graph-factory boundary for future recipes
        that need to attach discriminators, frozen tokenizers, EMA
        copies, or parameter-efficient adapters. The default graph is
        the set of exact module paths resolved by :meth:`setup`.
        """
        self.setup()
        self._restore_portable_recipe_state()
        return self

    def _restore_portable_recipe_state(self) -> None:
        """Hydrate recipe-owned state retained by a portable model load.

        Portable component weights are restored by the short-lived
        adapter used during :meth:`PreTrainedTTSModel.load`. Recipe
        state must instead be applied after the caller's concrete
        adapter has finished building auxiliary objects such as EMA
        shadows.
        """
        payload = getattr(
            self.model,
            "_pending_training_recipe_state",
            None,
        )
        if payload is None:
            return
        if not isinstance(payload, Mapping):
            raise TypeError("Pending portable training recipe state must be a mapping.")
        model_type = payload.get("model_type")
        if model_type != self.model_type:
            raise ValueError(
                "Portable training recipe state targets "
                f"{model_type!r}, not {self.model_type!r}.")
        recipe_id = payload.get("recipe_id")
        if recipe_id != self.recipe_id:
            raise ValueError(
                "Portable training recipe state was written for "
                f"{recipe_id!r}, but the active adapter is "
                f"{self.recipe_id!r}.")
        state = payload.get("state")
        if not isinstance(state, Mapping):
            raise TypeError(
                "Pending portable training recipe payload must contain a "
                "mapping under 'state'.")
        self.load_recipe_state_dict(state, strict=True)
        self.model._pending_training_recipe_state = None

    @property
    def recipe_id(self) -> str:
        """Stable identifier used to reject incompatible exact resumes."""
        return f"{type(self).__module__}.{type(self).__qualname__}"

    def artifact_manifest(self) -> dict[str, Any]:
        """Describe the recipe, external artifacts, and save semantics.

        Safetensors and source checkpoints are weight warm starts. Exact
        continuation is deliberately represented separately because it
        also needs optimizer, scheduler, scaler, RNG, callback, sampler,
        and recipe-owned state.
        """
        config = getattr(self.model, "config", None)
        return {
            "format_version": 1,
            "model_type": self.model_type,
            "task": self.spec.task.value,
            "family": self.spec.family_name,
            "support": self.spec.support.value,
            "recipe_id": self.recipe_id,
            "recipe_version": self.RECIPE_VERSION,
            "recipe_kind": self.spec.recipe_kind.value,
            "phases": [phase.name for phase in self.spec.phases],
            'base_model': getattr(config, "name_or_path", None),
            "training_default_model": (self.spec.training_default_model_name_or_path),
            "source_entrypoints": list(self.spec.source_entrypoints),
            "checkpoint_semantics": {
                "safetensors": "weight-warm-start",
                "voicehub_checkpoint": "exact-resume",
                "save_pretrained": self.native_export_semantics,
            },
            "training_strategy": "pluggable",
            "quantized_training": self.supports_quantized_training,
        }

    @classmethod
    def _manifest_value(cls, value: Any) -> Any:
        """Normalize recipe configuration into deterministic JSON values."""
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, Enum):
            return cls._manifest_value(value.value)
        if isinstance(value, Path):
            return str(value)
        if is_dataclass(value):
            return cls._manifest_value(asdict(value))
        if isinstance(value, Mapping):
            return {
                str(key): cls._manifest_value(item)
                for key, item in sorted(
                    value.items(),
                    key=lambda pair: str(pair[0]),
                )
            }
        if isinstance(value, (tuple, list)):
            return [cls._manifest_value(item) for item in value]
        if isinstance(value, (set, frozenset)):
            return sorted(
                (cls._manifest_value(item) for item in value),
                key=repr,
            )
        attributes = getattr(value, "__dict__", None)
        if isinstance(attributes, Mapping):
            return {
                "__class__": (f"{type(value).__module__}.{type(value).__qualname__}"),
                "attributes": cls._manifest_value(attributes),
            }
        return {
            "__class__": f"{type(value).__module__}.{type(value).__qualname__}",
        }

    def recipe_resume_configuration(self) -> Mapping[str, Any]:
        """Return model-specific settings that affect exact continuation.

        Configuration keys prefixed with ``training_`` are included by
        default. Specialized recipes can override this hook to add
        resolved defaults or other objective/schedule controls.
        """
        config = getattr(self.model, "config", None)
        if config is None:
            return {}
        to_dict = getattr(config, "to_dict", None)
        values = to_dict() if callable(to_dict) else getattr(config, "__dict__", {})
        if not isinstance(values, Mapping):
            return {}
        return {
            str(key): self._manifest_value(value)
            for key, value in sorted(values.items()) if str(key).startswith("training_")
        }

    def resume_signature(self) -> dict[str, Any]:
        """Describe recipe topology and controls required for exact resume."""
        phases = []
        for phase in self.spec.phases:
            phases.append({
                "name":
                phase.name,
                "kind":
                phase.kind.value,
                "component_paths":
                list(phase.component_paths),
                "optimizer_names":
                list(phase.optimizer_names),
                "forward_component":
                phase.forward_component,
                "forward_method":
                phase.forward_method,
                "label_names":
                list(phase.label_names),
                "prediction_keys":
                list(phase.prediction_keys),
                "loss_keys":
                list(phase.loss_keys),
                "loss_weights": [[name, weight] for name, weight in phase.loss_weights],
                "input_aliases": [[source, destination] for source, destination in phase.input_aliases],
                "required_inputs":
                list(phase.required_inputs),
                "frequency":
                phase.frequency,
                "offset":
                phase.offset,
                "fallback_objective":
                phase.fallback_objective,
                "detach_inputs":
                list(phase.detach_inputs),
                "frozen_component_paths":
                list(phase.frozen_component_paths, ),
                "optimizer_step_after_phase":
                phase.optimizer_step_after_phase,
            })
        return {
            "recipe_id": self.recipe_id,
            "recipe_version": self.RECIPE_VERSION,
            "model_type": self.model_type,
            "task": self.spec.task.value,
            "family": self.spec.family_name,
            "support": self.spec.support.value,
            "recipe_kind": self.spec.recipe_kind.value,
            "module_paths": list(self.spec.module_paths),
            "component_paths": list(self.spec.component_paths),
            "training_default_model": (self.spec.training_default_model_name_or_path),
            "default_phase": self.spec.default_phase,
            "separate_optimizers": self.spec.separate_optimizers,
            "field_schemas": self._manifest_value(self.spec.field_schemas, ),
            "phases": phases,
            "configuration": self._manifest_value(self.recipe_resume_configuration(), ),
        }

    def validate_support(self) -> None:
        """Validate capability and checkpoint variant without loading
        weights."""
        if self.spec.support is TrainingSupport.INFERENCE_ONLY:
            raise ValueError(
                f"{self.model_type!r} currently exposes an inference-only "
                "runtime. Register a specialized training adapter and profile "
                "instead of loading its fused or inference-optimized checkpoint.")
        if (self.spec.support is TrainingSupport.CUSTOM and not self.supports_custom_recipe and
                not self._registered_specialization):
            raise ValueError(
                f"{self.model_type!r} requires a specialized training adapter "
                "for its source-native recipe. Register one with "
                "AutoTrainingAdapter.register() before setup.")
        self._validate_configured_training_artifact()
        validate_runtime = getattr(self.model, "_validate_training_runtime", None)
        if callable(validate_runtime):
            validate_runtime()

    @staticmethod
    def _configuration_items(value: Any) -> Mapping[str, Any]:
        if isinstance(value, Mapping):
            return value
        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            try:
                output = to_dict()
            except (AttributeError, TypeError, ValueError):
                output = None
            if isinstance(output, Mapping):
                return output
        attributes = getattr(value, "__dict__", None)
        return attributes if isinstance(attributes, Mapping) else {}

    @classmethod
    def _quantization_setting(cls, value: Any) -> str | None:
        """Return the first explicit quantization control in a config."""
        pending = [(value, 0)]
        visited = set()
        boolean_keys = {
            "is_quantized",
            "is_loaded_in_4bit",
            "is_loaded_in_8bit",
            "load_in_4bit",
            "load_in_8bit",
        }
        object_keys = {
            "hf_quantizer",
            "quantization_config",
            "quantization_method",
        }
        nested_keys = {
            "additional_model_config",
            "model_kwargs",
        }
        while pending:
            current, depth = pending.pop()
            if id(current) in visited:
                continue
            visited.add(id(current))
            values = cls._configuration_items(current)
            for key in boolean_keys:
                if bool(values.get(key, False)):
                    return key
            for key in object_keys:
                setting = values.get(key)
                is_empty = (
                    setting is None or setting is False or isinstance(setting, str) and not setting or
                    isinstance(setting, Mapping) and not setting)
                if not is_empty:
                    return key
            if depth < 2:
                for key in nested_keys:
                    nested = values.get(key)
                    if nested is not None:
                        pending.append((nested, depth + 1))
        return None

    def _validate_configured_training_artifact(self) -> None:
        """Reject clearly serving-only artifacts before allocating weights."""
        config = getattr(self.model, "config", None)
        identifier = str(getattr(config, "name_or_path", "")).lower()
        serving_markers = (
            ".gguf",
            "-gguf",
            "/gguf",
            "llama.cpp",
            "llama_cpp",
            ".onnx",
            ".engine",
        )
        if any(marker in identifier for marker in serving_markers):
            raise ValueError(
                f"{self.model_type!r} fine-tuning requires a differentiable "
                "PyTorch/source checkpoint; the selected artifact "
                f"{identifier!r} is a serving-only format.")
        quantized_markers = (
            "int4",
            "int8",
            "4bit",
            "8bit",
            "gptq",
            "awq",
        )
        if (not self.supports_quantized_training and
                any(marker in identifier for marker in quantized_markers)):
            raise ValueError(
                f"{self.model_type!r} does not provide a quantization-aware "
                "training adapter. Select an unquantized checkpoint or "
                "register a PEFT/QLoRA-aware adapter.")
        setting = self._quantization_setting(config)
        if setting is not None and not self.supports_quantized_training:
            raise ValueError(
                f"{self.model_type!r} training configuration enables "
                f"{setting!r}, but this adapter supports full-precision "
                "fine-tuning only.")

    def _validate_loaded_training_graph(self) -> None:
        """Backstop artifact checks against the exact resolved components."""
        roots = [self.primary_model]
        roots.extend(component for _, component in self._components)
        objects = []
        seen = set()
        for root in roots:
            if root is None or id(root) in seen:
                continue
            seen.add(id(root))
            objects.append(root)

        serving_module_markers = (
            "onnxruntime",
            "llama_cpp",
            "tensorrt",
            "vllm",
        )
        compiled_markers = (
            "torch._dynamo",
            "optimizedmodule",
            "scriptmodule",
        )
        quantized_module_markers = (
            "bitsandbytes",
            "auto_gptq",
            "gptqmodel",
            "awq",
            "torchao",
            "quanto",
            "linear4bit",
            "linear8bitlt",
            "quantlinear",
        )
        inspected = []
        inspected_ids = set()
        for root in objects:
            modules = getattr(root, "modules", None)
            candidates = modules() if callable(modules) else (root, )
            for candidate in candidates:
                if id(candidate) in inspected_ids:
                    continue
                inspected_ids.add(id(candidate))
                if len(inspected) >= 16_384:
                    raise RuntimeError(
                        "Training graph validation exceeded 16,384 modules; "
                        "declare a narrower trainable component path.")
                inspected.append(candidate)
                qualified = (f"{type(candidate).__module__}."
                             f"{type(candidate).__qualname__}").lower()
                if any(marker in qualified for marker in serving_module_markers):
                    raise TypeError(
                        f"{self.model_type!r} resolved serving runtime "
                        f"{qualified!r} as a trainable component.")
                if (not self.supports_compiled_training and
                        any(marker in qualified for marker in compiled_markers)):
                    raise TypeError(
                        f"{self.model_type!r} resolved compiled/scripted "
                        f"component {qualified!r}; load its unfused source "
                        "module for training.")
                setting = self._quantization_setting(candidate)
                is_known_quantized = any(marker in qualified for marker in quantized_module_markers)
                if (not self.supports_quantized_training and (setting is not None or is_known_quantized)):
                    detail = setting or qualified
                    raise TypeError(
                        f"{self.model_type!r} resolved quantized training "
                        f"component {detail!r}, but its adapter is not "
                        "PEFT/QLoRA-aware.")

        for component_name, component in self._components:
            parameters = getattr(component, "named_parameters", None)
            if not callable(parameters):
                continue
            for parameter_name, parameter in parameters():
                if not getattr(parameter, "requires_grad", False):
                    continue
                dtype = getattr(parameter, "dtype", None)
                is_floating = bool(getattr(dtype, "is_floating_point", False))
                is_complex = bool(getattr(dtype, "is_complex", False))
                if dtype is not None and not (is_floating or is_complex):
                    raise TypeError(
                        f"Trainable parameter {component_name}."
                        f"{parameter_name} has non-differentiable dtype "
                        f"{dtype}.")

    def _resolve_path(self, path: str):
        return self._resolve_from(self.model, path)

    @staticmethod
    def _resolve_from(root, path: str):
        current = root
        for part in path.split("."):
            if isinstance(current, Mapping):
                if part not in current:
                    return None
                current = current[part]
            elif isinstance(current, (list, tuple)) and part.isdigit():
                index = int(part)
                if not 0 <= index < len(current):
                    return None
                current = current[index]
            else:
                current = getattr(current, part, None)
            if current is None:
                return None
        return current

    @staticmethod
    def _has_trainable_parameters(candidate) -> bool:
        if candidate is None or not hasattr(candidate, "parameters"):
            return False
        try:
            return any(getattr(parameter, "requires_grad", False) for parameter in candidate.parameters())
        except (AttributeError, TypeError):
            return False

    @classmethod
    def _is_forward_module(cls, candidate) -> bool:
        return cls._has_trainable_parameters(candidate) and (
            callable(candidate) or callable(getattr(candidate, "forward", None)))

    @classmethod
    def _is_trainable(cls, candidate) -> bool:
        """Backward-compatible alias for trainable callable detection."""
        return cls._is_forward_module(candidate)

    @classmethod
    def _discover_trainable_modules(
        cls,
        root,
        *,
        max_depth: int = 4,
        max_nodes: int = 512,
    ) -> list[tuple[str, Any]]:
        queue = deque([("model", root, 0)])
        visited = set()
        discovered = []
        while queue and len(visited) < max_nodes:
            path, value, depth = queue.popleft()
            identity = id(value)
            if identity in visited:
                continue
            visited.add(identity)
            if cls._has_trainable_parameters(value):
                discovered.append((path, value))
                if cls._is_forward_module(value):
                    continue
            if depth >= max_depth:
                continue
            for name, child in cls._iter_children(value):
                queue.append((f"{path}.{name}", child, depth + 1))
        discovered.sort(
            key=lambda item: cls._parameter_count(item[1]),
            reverse=True,
        )
        return discovered

    @staticmethod
    def _iter_children(value):
        if isinstance(value, Mapping):
            for key, child in value.items():
                yield str(key), child
            return
        if isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                yield str(index), child
            return
        values = getattr(value, "__dict__", None)
        if not isinstance(values, dict):
            return
        for name, child in values.items():
            if name.startswith("__"):
                continue
            if isinstance(child, (str, bytes, int, float, bool, type(None))):
                continue
            yield name, child

    @staticmethod
    def _parameter_count(module) -> int:
        try:
            return sum(
                parameter.numel() for parameter in module.parameters()
                if getattr(parameter, "requires_grad", False))
        except (AttributeError, TypeError):
            return 0

    @staticmethod
    def _deduplicate_components(components):
        output = []
        seen = set()
        for name, component in components:
            if id(component) in seen:
                continue
            seen.add(id(component))
            output.append((name, component))
        return output

    @staticmethod
    def _safe_component_name(path: str) -> str:
        return path.replace(".", "_")

    @staticmethod
    def _named_trainable_parameters(component):
        try:
            parameters = component.named_parameters(remove_duplicate=True)
        except TypeError:
            parameters = component.named_parameters()
        for name, parameter in parameters:
            if getattr(parameter, "requires_grad", False):
                yield name, parameter

    def _parameter_owner_components(self):
        # Prefer the most specific source path so a child module owns its
        # parameters instead of inheriting a broad parent prefix.
        routed = []
        for phase in self.spec.phases:
            for component_path, _ in phase.component_optimizer_routes:
                component = self._component_by_path.get(component_path)
                if component is None:
                    component = self._resolve_path(component_path)
                if self._has_trainable_parameters(component):
                    routed.append((component_path, component))
        candidates = (self._deduplicate_components(routed) if routed else self._components)
        indexed = list(enumerate(candidates))
        indexed.sort(key=lambda item: (
            -item[1][0].count("."),
            item[0],
        ))
        return [component for _, component in indexed]

    def named_parameters(self):
        """Yield every trainable parameter exactly once across components."""
        self.setup()
        seen_parameters = set()
        seen_names: dict[str, int] = {}
        for component_name, component in self._parameter_owner_components():
            prefix = self._safe_component_name(component_name)
            for name, parameter in self._named_trainable_parameters(component):
                identity = id(parameter)
                if identity in seen_parameters:
                    continue
                full_name = f"{prefix}.{name}"
                previous = seen_names.get(full_name)
                if previous is not None and previous != identity:
                    raise RuntimeError(f"Parameter name collision while resolving {full_name!r}.")
                seen_parameters.add(identity)
                seen_names[full_name] = identity
                yield full_name, parameter

    def parameters(self):
        for _, parameter in self.named_parameters():
            yield parameter

    def _phase_component_routes(
        self,
        phases: tuple[TrainingPhaseSpec, ...],
    ) -> list[tuple[str, str, Any]]:
        routes = []
        for phase in phases:
            for component_path, optimizer_name in phase.component_optimizer_routes:
                component = self._component_by_path.get(component_path)
                if component is None:
                    component = self._resolve_path(component_path)
                if not self._has_trainable_parameters(component):
                    continue
                routes.append((optimizer_name, component_path, component))
        return routes

    def named_parameter_groups(
        self,
        training_phase: str | TrainingPhaseSpec | None = None,
    ):
        """Return collision-free parameter groups routed to named optimizers.

        Shared parameters assigned to two optimizer names are rejected
        instead of being silently stepped twice.
        """
        self.setup()
        phases = ((self.select_training_phase(training_phase), )
                  if training_phase is not None else self.spec.phases)
        routes = self._phase_component_routes(phases)
        if routes:
            grouped: dict[str, list[tuple[str, Any]]] = {}
            owners: dict[int, str] = {}
            names_by_group: dict[str, dict[str, int]] = {}
            for optimizer_name, component_path, component in routes:
                parameters = grouped.setdefault(optimizer_name, [])
                seen_names = names_by_group.setdefault(optimizer_name, {})
                prefix = self._safe_component_name(component_path)
                for name, parameter in self._named_trainable_parameters(component):
                    identity = id(parameter)
                    previous_owner = owners.get(identity)
                    if previous_owner is not None:
                        if previous_owner != optimizer_name:
                            raise ValueError(
                                f"Parameter {component_path}.{name} is routed to both "
                                f"{previous_owner!r} and {optimizer_name!r}.")
                        continue
                    full_name = f"{prefix}.{name}"
                    previous_identity = seen_names.get(full_name)
                    if previous_identity is not None and previous_identity != identity:
                        raise ValueError(
                            f"Optimizer {optimizer_name!r} has two parameters named "
                            f"{full_name!r}.")
                    owners[identity] = optimizer_name
                    seen_names[full_name] = identity
                    parameters.append((full_name, parameter))
            return [(name, parameters) for name, parameters in grouped.items() if parameters]

        # Legacy profiles without optimizer routes retain component partitioning.
        seen = set()
        groups = []
        group_names = set()
        for component_name, component in reversed(self._components):
            parameters = []
            for name, parameter in self._named_trainable_parameters(component):
                if id(parameter) in seen:
                    continue
                seen.add(id(parameter))
                parameters.append((name, parameter))
            if parameters:
                group_name = self._safe_component_name(component_name)
                if group_name in group_names:
                    raise ValueError(f"Component paths collide on optimizer name {group_name!r}.")
                group_names.add(group_name)
                groups.append((group_name, parameters))
        groups.reverse()
        return groups

    def to(self, device):
        self.setup()
        for _, component in self._components:
            if hasattr(component, "to"):
                component.to(device)
        set_training_device = getattr(
            self.model,
            "_set_training_device",
            None,
        )
        if callable(set_training_device):
            set_training_device(str(device))
        return self

    def train(self, mode: bool = True):
        self.setup()
        for _, component in self._components:
            component.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def _gradient_checkpoint_targets(self):
        self.setup()
        targets = [self.model, self.primary_model]
        targets.extend(component for _, component in self._components)
        output = []
        seen = set()
        for target in targets:
            if target is None or id(target) in seen:
                continue
            seen.add(id(target))
            output.append(target)
        return output

    def _set_gradient_checkpointing(self, enabled: bool, **kwargs) -> None:
        delegated = 0
        method_name = ("gradient_checkpointing_enable" if enabled else "gradient_checkpointing_disable")
        for target in self._gradient_checkpoint_targets():
            method = getattr(target, method_name, None)
            if callable(method):
                method(**kwargs)
                delegated += 1
                continue
            setter = getattr(target, "set_gradient_checkpointing", None)
            if callable(setter):
                setter(enabled)
                delegated += 1
        if not delegated:
            action = "enable" if enabled else "disable"
            raise ValueError(
                f"{self.model_type!r} has no component that can {action} "
                "gradient checkpointing.")

    def gradient_checkpointing_enable(self, **kwargs) -> None:
        """Delegate checkpointing to every component that implements it."""
        self._set_gradient_checkpointing(True, **kwargs)

    def gradient_checkpointing_disable(self, **kwargs) -> None:
        """Disable delegated component checkpointing."""
        self._set_gradient_checkpointing(False, **kwargs)

    def state_dict(self):
        """Serialize a versioned adapter state with an exact topology."""
        self.setup()
        state_components = self._state_components()
        topology = tuple(name for name, _ in state_components)
        return {
            "__voicehub_training_adapter__": self.model_type,
            "__voicehub_training_adapter_version__": self.ADAPTER_STATE_VERSION,
            "topology": topology,
            "components": {
                name: component.state_dict()
                for name, component in state_components
            },
            "recipe_state": self.recipe_state_dict(),
        }

    def recipe_state_dict(self) -> Mapping[str, Any]:
        """Return model-recipe state not owned by a trainable component.

        EMA shadows, loss-balancer statistics, or source-native counters
        belong here. The default recipe has no additional state.
        """
        return {}

    def load_recipe_state_dict(
        self,
        state_dict: Mapping[str, Any],
        *,
        strict: bool = True,
    ) -> None:
        """Restore state returned by :meth:`recipe_state_dict`."""
        if not isinstance(state_dict, Mapping):
            raise TypeError("Training recipe state must be a mapping.")
        if strict and state_dict:
            unexpected = ", ".join(sorted(str(key) for key in state_dict))
            raise ValueError(
                "This training adapter does not own recipe state, but the "
                f"checkpoint contains: {unexpected}.")

    def _state_components(self):
        """Return the smallest component roots that cover adapter
        parameters."""
        indexed = list(enumerate(self._components))
        parameter_sets = {}
        for _, (name, component) in indexed:
            try:
                parameter_sets[name] = {id(parameter) for parameter in component.parameters()}
            except (AttributeError, TypeError):
                parameter_sets[name] = set()

        by_coverage = sorted(
            indexed,
            key=lambda item: (
                -len(parameter_sets[item[1][0]]),
                item[0],
            ),
        )
        covered = set()
        selected_indices = []
        for index, (name, _) in by_coverage:
            parameters = parameter_sets[name]
            if parameters and parameters <= covered:
                continue
            selected_indices.append(index)
            covered.update(parameters)
        selected = set(selected_indices)
        return [component for index, component in indexed if index in selected]

    @staticmethod
    def _load_component_state(component, state, *, strict: bool):
        try:
            return component.load_state_dict(state, strict=strict)
        except TypeError:
            return component.load_state_dict(state)

    def load_state_dict(
        self,
        state_dict,
        strict: bool = True,
        *,
        load_recipe_state: bool = True,
    ):
        """Load adapter state, rejecting incompatible versions/topologies.

        ``load_recipe_state=False`` is reserved for the portable wrapper
        lifecycle: component weights are restored immediately, while the
        recipe payload is retained until the caller-owned adapter has built
        its auxiliary graph.
        """
        self.setup()
        if not isinstance(state_dict, Mapping):
            raise TypeError("Training adapter state must be a mapping.")

        marker = state_dict.get("__voicehub_training_adapter__")
        if marker is None and "components" not in state_dict:
            # Preserve support for raw source-module checkpoints.
            return self._load_component_state(
                self.primary_model,
                state_dict,
                strict=strict,
            )
        if marker != self.model_type:
            raise ValueError(f"Adapter checkpoint targets {marker!r}, not {self.model_type!r}.")

        version = state_dict.get("__voicehub_training_adapter_version__")
        if version not in self.SUPPORTED_ADAPTER_STATE_VERSIONS:
            supported = ", ".join(str(item) for item in self.SUPPORTED_ADAPTER_STATE_VERSIONS)
            raise ValueError(
                f"Unsupported training adapter state version {version!r}; "
                f"supported versions are: {supported}.")
        component_states = state_dict.get("components")
        topology = state_dict.get("topology")
        if not isinstance(component_states, Mapping):
            raise TypeError("Adapter checkpoint 'components' must be a mapping.")
        if not isinstance(topology, (tuple, list)):
            raise TypeError("Adapter checkpoint 'topology' must be a sequence.")
        topology = tuple(topology)
        if topology != tuple(component_states):
            raise ValueError("Adapter checkpoint topology does not match its component payload.")

        available = dict(self._state_components())
        expected = tuple(available)
        missing = tuple(name for name in expected if name not in component_states)
        unexpected = tuple(name for name in topology if name not in available)
        if strict and (missing or unexpected or topology != expected):
            details = []
            if missing:
                details.append(f"missing={missing}")
            if unexpected:
                details.append(f"unexpected={unexpected}")
            if not details:
                details.append(f"expected_order={expected}, checkpoint_order={topology}")
            raise ValueError("Training adapter checkpoint topology mismatch: " + ", ".join(details))

        results = {}
        for name in topology:
            component = available.get(name)
            if component is not None:
                results[name] = self._load_component_state(
                    component,
                    component_states[name],
                    strict=strict,
                )
        if load_recipe_state:
            recipe_state = state_dict.get("recipe_state", {})
            self.load_recipe_state_dict(recipe_state, strict=strict)
        return results

    def select_training_phase(
        self,
        training_phase: str | TrainingPhaseSpec | None = None,
    ) -> TrainingPhaseSpec:
        """Resolve an explicit phase control value."""
        if training_phase is None:
            return self.spec.get_phase()
        if isinstance(training_phase, TrainingPhaseSpec):
            registered = self.spec.phase_map.get(training_phase.name)
            if registered != training_phase:
                raise ValueError(
                    f"Phase {training_phase.name!r} is not part of "
                    f"{self.model_type!r}'s training profile.")
            return training_phase
        if not isinstance(training_phase, str):
            raise TypeError("training_phase must be a phase name or TrainingPhaseSpec.")
        return self.spec.get_phase(training_phase)

    def select_evaluation_phase(
        self,
        training_phase: str | TrainingPhaseSpec | None = None,
    ) -> TrainingPhaseSpec:
        """Resolve the phase used for evaluation and prediction."""
        return self.select_training_phase(training_phase)

    def plan_training_phases(self, step: int) -> tuple[TrainingPhaseSpec, ...]:
        """Return every phase scheduled for a zero-based recipe step."""
        return tuple(phase for phase in self.spec.phases if phase.is_scheduled(step))

    def create_training_context(
        self,
        inputs: Mapping[str, Any],
        *,
        training_phase: str | TrainingPhaseSpec | None = None,
        step: int | None = None,
        epoch: float | None = None,
        is_training: bool = True,
        metadata: Mapping[str, Any] | None = None,
    ) -> TrainingContext:
        return TrainingContext(
            phase=self.select_training_phase(training_phase),
            inputs=inputs,
            step=step,
            epoch=epoch,
            is_training=is_training,
            metadata=metadata or {},
        )

    def prepare_training_inputs(
        self,
        inputs: Mapping[str, Any],
        context: TrainingContext,
    ) -> Mapping[str, Any]:
        """Model-specific hook for converting a batch to backend inputs."""
        prepare = getattr(self.model, "prepare_training_inputs", None)
        if callable(prepare):
            return prepare(dict(inputs), phase=context.phase.name)
        return inputs

    def prepare_batch(
        self,
        inputs: Mapping[str, Any],
        context: TrainingContext,
    ) -> Mapping[str, Any]:
        """Public batch-preparation boundary used by recipe integrations."""
        return self.prepare_training_inputs(inputs, context)

    def set_runtime_input_preparer(
        self,
        preparer: Callable[[Any], Any] | None,
    ) -> None:
        """Set the execution-strategy hook for model-created batch values.

        Dataloader tensors are moved before an adapter runs its model-
        specific preprocessing. Tokenizers, feature extractors, and
        audio processors can create new CPU tensors during that later
        step, so the active strategy must prepare the resulting batch
        once more before forward.
        """
        if preparer is not None and not callable(preparer):
            raise TypeError("Runtime input preparer must be callable or None.")
        self._runtime_input_preparer = preparer

    def prepare_runtime_inputs(
        self,
        inputs: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Move model-created inputs through the active execution strategy."""
        if not isinstance(inputs, Mapping):
            raise TypeError("Runtime training inputs must be a mapping.")
        values = dict(inputs)
        if self._runtime_input_preparer is None:
            return values
        prepared = self._runtime_input_preparer(values)
        if not isinstance(prepared, Mapping):
            raise TypeError("The training strategy input preparer must return a mapping.")
        return dict(prepared)

    def optimizer_plan(self) -> dict[str, tuple[str, ...]]:
        """Return declared component routes for each named optimizer."""
        plan: dict[str, list[str]] = {}
        for phase in self.spec.phases:
            for path, name in phase.component_optimizer_routes:
                paths = plan.setdefault(name, [])
                if path not in paths:
                    paths.append(path)
        return {name: tuple(paths) for name, paths in plan.items()}

    def create_optimizer(
        self,
        name: str,
        parameters: list[tuple[str, Any]],
        training_args,
    ):
        """Optionally create a source-native optimizer for one route.

        Returning ``None`` delegates to :class:`voicehub.Trainer`'s
        AdamW default. Specialized recipes can preserve upstream
        optimizer choices without taking ownership of the complete
        training loop.
        """
        del name, parameters, training_args
        return None

    def create_scheduler(
        self,
        name: str,
        optimizer,
        num_training_steps: int,
        training_args,
    ):
        """Optionally create a source-native scheduler for one route."""
        del name, optimizer, num_training_steps, training_args
        return None

    def on_before_optimizer_step(
        self,
        *,
        optimizer_names: tuple[str, ...] | None,
        step: int,
    ) -> None:
        """Run immediately before a routed optimizer update."""

    def on_optimizer_step(
        self,
        *,
        optimizer_names: tuple[str, ...] | None,
        step: int,
    ) -> None:
        """Run after a successful optimizer update.

        EMA and other update-coupled recipe state should be advanced
        here so skipped mixed-precision steps cannot mutate it.
        """

    def on_optimizer_step_skipped(
        self,
        *,
        optimizer_names: tuple[str, ...] | None,
        step: int,
    ) -> None:
        """Run when precision overflow prevents an optimizer update."""

    def save_pretrained(self, save_directory) -> None:
        """Optionally export source-native weights or inference artifacts.

        VoiceHub always writes its portable adapter checkpoint
        separately. Recipes can additionally emit Hugging Face
        safetensors, component weights, or a complete upstream layout
        from this hook. Declare the exact meaning through
        ``native_export_semantics``.
        """
        del save_directory

    def create_dataset(self, records, **kwargs):
        """Build a dependency-light dataset for a portable training recipe."""
        if not self.spec.is_turnkey:
            raise NotImplementedError(
                f"{self.model_type!r} requires its source-native dataset "
                "pipeline. Use the upstream training recipe or register a "
                "specialized adapter.")
        from voicehub.training.datasets import SpeechDataset

        if isinstance(records, SpeechDataset) and not kwargs:
            return records
        return SpeechDataset(records, **kwargs)

    def on_training_phase_start(self, context: TrainingContext) -> None:
        """Hook invoked immediately before one phase forward."""

    def on_training_phase_end(
        self,
        context: TrainingContext,
        output: SpeechTrainingOutput,
    ) -> SpeechTrainingOutput:
        """Hook invoked after a phase output has been normalized."""
        return output

    def evaluation_label_values(
        self,
        inputs: Mapping[str, Any],
        phase: TrainingPhaseSpec,
    ) -> tuple[Any, ...]:
        """Return references from an unprocessed evaluation batch.

        Most adapters receive model-ready targets, so the declared phase
        labels are sufficient. Raw speech recipes can override this hook
        to expose transcripts or other references before
        ``prepare_batch()`` converts them into backend tensors.
        """
        names = tuple(dict.fromkeys(phase.label_names + self.spec.label_names))
        for name in names:
            if name in inputs:
                return (inputs[name], )
        if self.spec.task is SpeechTask.AUTOMATIC_SPEECH_RECOGNITION:
            for name in ("text", "transcription", "transcript"):
                if name in inputs:
                    return (inputs[name], )
        return ()

    def prepare_evaluation_predictions(
        self,
        outputs: Any,
        context: TrainingContext,
        predictions: Any,
    ) -> Any:
        """Convert native outputs into values suitable for evaluation."""
        del outputs, context
        return predictions

    def compute_evaluation_metrics(
        self,
        predictions: Any,
        label_ids: Any,
    ) -> Mapping[str, Any]:
        """Return recipe-owned metrics after all evaluation batches."""
        del predictions, label_ids
        return {}

    def evaluation_scheduler_metric(
        self,
        metrics: Mapping[str, Any],
    ) -> float | None:
        """Select an evaluation metric for a recipe-owned scheduler.

        Step-based schedulers keep their existing optimizer-update
        cadence. Specialized validation schedulers opt in by returning
        one scalar metric after evaluation; the default intentionally
        returns ``None``.
        """
        del metrics
        return None

    def execute_training_plan(
        self,
        inputs: Mapping[str, Any],
        *,
        step: int,
        epoch: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> tuple[SpeechTrainingOutput, ...]:
        """Execute all phases due at ``step`` in declared order."""
        flattened = self.flatten_model_inputs(inputs)
        outputs = []
        for phase in self.plan_training_phases(step):
            context = self.create_training_context(
                flattened,
                training_phase=phase,
                step=step,
                epoch=epoch,
                metadata=metadata,
            )
            outputs.append(self.execute_training_phase(context))
        return tuple(outputs)

    def compute_step(
        self,
        context: TrainingContext,
    ) -> SpeechTrainingOutput:
        """Execute one native recipe step.

        Specialized adapters normally override
        ``execute_training_phase``; this named boundary lets execution
        strategies compose or replace a recipe without depending on that
        historical method name.
        """
        return self.execute_training_phase(context)

    def __call__(self, **inputs) -> SpeechTrainingOutput:
        """Execute one phase selected by the ``training_phase`` control
        field."""
        self.setup()
        forward_inputs = self.flatten_model_inputs(inputs)
        training_phase = forward_inputs.pop("training_phase", None)
        supplied_context = forward_inputs.pop("training_context", None)
        if supplied_context is not None:
            if not isinstance(supplied_context, TrainingContext):
                raise TypeError("training_context must be a TrainingContext.")
            if training_phase is not None:
                raise ValueError("Pass either training_phase or training_context, not both.")
            merged = dict(supplied_context.inputs)
            merged.update(forward_inputs)
            context = supplied_context.with_inputs(merged)
        else:
            context = self.create_training_context(
                forward_inputs,
                training_phase=training_phase,
            )
        return self.execute_training_phase(context)

    @staticmethod
    def flatten_model_inputs(inputs: Mapping[str, Any]) -> dict[str, Any]:
        """Merge the portable ``model_inputs`` namespace into one native
        batch."""
        if not isinstance(inputs, Mapping):
            raise TypeError("Training inputs must be a mapping.")
        outer = dict(inputs)
        nested = outer.pop("model_inputs", None)
        if nested is None:
            return outer
        if not isinstance(nested, Mapping):
            raise TypeError("model_inputs must be a mapping when provided.")
        collisions = tuple(sorted(set(nested).intersection(outer)))
        if collisions:
            raise ValueError("model_inputs duplicates top-level training keys: " + ", ".join(collisions))
        flattened = dict(nested)
        flattened.update(outer)
        return flattened

    # Retained for compatibility with adapters built against VoiceHub 0.3.
    _flatten_model_inputs = flatten_model_inputs

    def execute_training_phase(
        self,
        context: TrainingContext,
    ) -> SpeechTrainingOutput:
        """Prepare, invoke, and normalize one backend phase."""
        self.setup()
        if not isinstance(context, TrainingContext):
            raise TypeError("execute_training_phase requires a TrainingContext.")
        phase = self.select_training_phase(context.phase)

        forward_inputs = self._apply_input_aliases(
            dict(context.inputs),
            phase,
        )
        prepared_by_model = self.prepare_batch(
            forward_inputs,
            context.with_inputs(forward_inputs),
        )
        if not isinstance(prepared_by_model, Mapping):
            raise TypeError("prepare_batch() must return a mapping.")
        forward_inputs = self.prepare_runtime_inputs(prepared_by_model)
        forward_inputs = self._detach_phase_inputs(forward_inputs, phase)
        labels = self._find_labels(forward_inputs, phase)
        optional_inputs = (set(phase.label_names) if not context.is_training and labels is None else set())
        self._validate_required_inputs(
            forward_inputs,
            phase,
            optional_inputs=optional_inputs,
        )
        context = TrainingContext(
            phase=phase,
            inputs=forward_inputs,
            step=context.step,
            epoch=context.epoch,
            is_training=context.is_training,
            metadata=context.metadata,
        )

        target, forward = self._resolve_phase_callable(phase)
        self._map_labels_to_signature(
            forward,
            forward_inputs,
            labels,
            phase.label_names,
        )
        prepared = self._filter_forward_inputs(forward, forward_inputs)

        previous_context = self._current_context
        self._current_context = context
        try:
            self.on_training_phase_start(context)
            frozen_parameters = self._freeze_phase_components(phase)
            try:
                outputs = self._invoke_forward(target, forward, prepared, phase)
            finally:
                self._restore_frozen_parameters(frozen_parameters)
            losses = self._extract_losses(outputs, phase)
            predictions = self._extract_predictions(outputs, phase)
            if not context.is_training:
                predictions = self.prepare_evaluation_predictions(
                    outputs,
                    context,
                    predictions,
                )
            loss = self._aggregate_losses(losses, phase)
            if loss is None:
                if labels is None and not context.is_training:
                    losses = {}
                else:
                    loss = self.compute_phase_objective(
                        predictions,
                        labels,
                        context,
                    )
                    losses = {"loss": loss}
            metadata = self._get_value(outputs, "metadata")
            metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
            metadata.update({
                "model_type": self.model_type,
                "training_family": self.spec.family_name,
                "training_support": self.spec.support.value,
                "training_phase": phase.name,
                "optimizer_names": phase.optimizer_names,
            })
            output_class = (
                TTSTrainingOutput if self.spec.task is SpeechTask.TEXT_TO_SPEECH else SpeechTrainingOutput)
            normalized = output_class(
                loss=loss,
                logits=predictions,
                audio_values=self._get_value(outputs, "audio_values"),
                hidden_states=self._get_value(outputs, "hidden_states"),
                attentions=self._get_value(outputs, "attentions"),
                losses=losses,
                metadata=metadata,
                training_phase=phase.name,
                optimizer_names=phase.optimizer_names,
            )
            return self.on_training_phase_end(context, normalized)
        finally:
            self._current_context = previous_context

    def execute_prediction_phase(self, context: TrainingContext):
        """Run a label-free backend forward without inventing an objective.

        Specialized training adapters often require labels in
        ``execute_training_phase`` even though their underlying module can
        return logits without them. Prediction deliberately bypasses that
        supervised recipe while retaining phase selection, input aliases,
        model-specific batch preparation, and strategy execution.
        """
        self.setup()
        if not isinstance(context, TrainingContext):
            raise TypeError("execute_prediction_phase requires a TrainingContext.")
        phase = self.select_evaluation_phase(context.phase)
        forward_inputs = self._apply_input_aliases(
            dict(context.inputs),
            phase,
        )
        prepared_by_model = self.prepare_batch(
            forward_inputs,
            context.with_inputs(forward_inputs),
        )
        if not isinstance(prepared_by_model, Mapping):
            raise TypeError("prepare_batch() must return a mapping.")
        prepared_by_model = self.prepare_runtime_inputs(prepared_by_model)
        forward_inputs = self._detach_phase_inputs(
            dict(prepared_by_model),
            phase,
        )
        self._validate_required_inputs(
            forward_inputs,
            phase,
            optional_inputs=set(phase.label_names),
        )
        target, forward = self._resolve_phase_callable(phase)
        labels = self._find_labels(forward_inputs, phase)
        if labels is not None:
            self._map_labels_to_signature(
                forward,
                forward_inputs,
                labels,
                phase.label_names,
            )
        prepared = self._filter_forward_inputs(forward, forward_inputs)

        previous_context = self._current_context
        self._current_context = context.with_inputs(forward_inputs)
        try:
            return self._invoke_forward(
                target,
                forward,
                prepared,
                phase,
            )
        finally:
            self._current_context = previous_context

    @classmethod
    def _nested_detach(cls, value):
        detach = getattr(value, "detach", None)
        if callable(detach):
            return detach()
        if isinstance(value, Mapping):
            return {key: cls._nested_detach(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(cls._nested_detach(item) for item in value)
        if isinstance(value, list):
            return [cls._nested_detach(item) for item in value]
        return value

    @classmethod
    def _detach_at_path(cls, value, path: tuple[str, ...]):
        if not path:
            return cls._nested_detach(value)
        name, *remaining = path
        if isinstance(value, Mapping):
            if name not in value:
                return value
            copied = dict(value)
            copied[name] = cls._detach_at_path(copied[name], tuple(remaining))
            return copied
        if isinstance(value, list) and name.isdigit():
            index = int(name)
            if not 0 <= index < len(value):
                return value
            copied = list(value)
            copied[index] = cls._detach_at_path(copied[index], tuple(remaining))
            return copied
        if isinstance(value, tuple) and name.isdigit():
            index = int(name)
            if not 0 <= index < len(value):
                return value
            copied = list(value)
            copied[index] = cls._detach_at_path(copied[index], tuple(remaining))
            return tuple(copied)
        return value

    @classmethod
    def _detach_phase_inputs(cls, inputs, phase: TrainingPhaseSpec):
        detached = inputs
        for path in phase.detach_inputs:
            detached = cls._detach_at_path(detached, tuple(path.split(".")))
        return detached

    def _freeze_phase_components(self, phase: TrainingPhaseSpec):
        original_flags = {}
        for path in phase.frozen_component_paths:
            component = self._resolve_path(path)
            if component is None or not hasattr(component, "parameters"):
                continue
            try:
                parameters = component.parameters()
            except TypeError:
                continue
            for parameter in parameters:
                identity = id(parameter)
                if identity in original_flags:
                    continue
                original_flags[identity] = (
                    parameter,
                    bool(getattr(parameter, "requires_grad", False)),
                )
                if hasattr(parameter, "requires_grad_"):
                    parameter.requires_grad_(False)
                else:
                    parameter.requires_grad = False
        return tuple(original_flags.values())

    @staticmethod
    def _restore_frozen_parameters(parameters) -> None:
        for parameter, requires_grad in parameters:
            if hasattr(parameter, "requires_grad_"):
                parameter.requires_grad_(requires_grad)
            else:
                parameter.requires_grad = requires_grad

    @staticmethod
    def _apply_input_aliases(inputs, phase: TrainingPhaseSpec):
        for source, target in phase.input_aliases:
            if source not in inputs:
                continue
            if target in inputs and target != source:
                raise ValueError(
                    f"Both aliased input {source!r} and backend input "
                    f"{target!r} were provided.")
            if target != source:
                inputs[target] = inputs.pop(source)
        return inputs

    @staticmethod
    def _validate_required_inputs(
        inputs,
        phase: TrainingPhaseSpec,
        *,
        optional_inputs: set[str] | None = None,
    ) -> None:
        optional_inputs = optional_inputs or set()
        missing = tuple(
            name for name in phase.required_inputs if name not in inputs and name not in optional_inputs)
        if missing:
            raise ValueError(f"Training phase {phase.name!r} requires inputs: "
                             f"{', '.join(missing)}.")

    def _resolve_phase_callable(self, phase: TrainingPhaseSpec):
        candidate_paths = []
        if phase.forward_component is not None:
            candidate_paths.append(phase.forward_component)
        candidate_paths.extend(phase.component_paths)

        for path in dict.fromkeys(candidate_paths):
            target = self._resolve_path(path)
            forward = self._resolve_forward_method(target, phase.forward_method)
            if callable(forward):
                return target, forward

        if candidate_paths:
            attempted = ", ".join(candidate_paths)
            raise TypeError(
                f"Training phase {phase.name!r} could not resolve callable "
                f"{phase.forward_method!r} from its declared path(s): {attempted}.")
        if phase.kind is not TrainingPhaseKind.OBJECTIVE:
            raise TypeError(
                f"Training phase {phase.name!r} is recipe-owned and requires "
                "a specialized adapter to supply its component and forward pass.")
        target = self.primary_model
        forward = self._resolve_forward_method(target, phase.forward_method)
        if callable(forward):
            return target, forward
        attempted = ", ".join(candidate_paths) or self.primary_path or "primary model"
        raise TypeError(
            f"Training phase {phase.name!r} could not resolve callable "
            f"{phase.forward_method!r} from: {attempted}.")

    @classmethod
    def _resolve_forward_method(cls, target, method_path: str):
        if target is None:
            return None
        if method_path in ("forward", "__call__"):
            if callable(target):
                return target
            return getattr(target, method_path, None)
        return cls._resolve_from(target, method_path)

    @staticmethod
    def _invoke_forward(target, forward, inputs, phase):
        del target, phase
        return forward(**inputs)

    @staticmethod
    def _signature(callable_object):
        candidate = (
            callable_object.forward if hasattr(callable_object, "forward") and
            inspect.ismethod(getattr(callable_object, "forward", None)) else callable_object)
        try:
            return inspect.signature(candidate)
        except (TypeError, ValueError):
            return None

    def _find_labels(self, inputs, phase: TrainingPhaseSpec | None = None):
        names = phase.label_names if phase is not None else self.spec.label_names
        for name in names:
            if name in inputs:
                return inputs[name]
        return None

    @classmethod
    def _map_labels_to_signature(
        cls,
        module,
        inputs,
        labels,
        label_names: tuple[str, ...] | None = None,
    ) -> None:
        if labels is None:
            return
        label_names = label_names or ("labels", "targets", "target")
        signature = cls._signature(module)
        if signature is None:
            return
        parameters = signature.parameters
        if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
            return
        if any(name in inputs and name in parameters for name in label_names):
            return
        for name in label_names:
            if name in parameters:
                inputs[name] = labels
                return

    @classmethod
    def _filter_forward_inputs(cls, module, inputs):
        model_inputs = {key: value for key, value in inputs.items() if key not in cls.SOURCE_METADATA_FIELDS}
        signature = cls._signature(module)
        if signature is None:
            return model_inputs
        parameters = signature.parameters
        if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
            return model_inputs
        accepted = set(parameters)
        return {key: value for key, value in model_inputs.items() if key in accepted}

    @staticmethod
    def _get_value(outputs, key):
        if isinstance(outputs, Mapping):
            return outputs.get(key)
        return getattr(outputs, key, None)

    def _extract_predictions(
        self,
        outputs,
        phase: TrainingPhaseSpec | None = None,
    ):
        keys = phase.prediction_keys if phase is not None else self.spec.prediction_keys
        for key in keys:
            value = self._get_value(outputs, key)
            if value is not None:
                return value
        if isinstance(outputs, (tuple, list)):
            for value in outputs:
                if hasattr(value, "shape") and getattr(value, "ndim", 0) > 0:
                    return value
        if hasattr(outputs, "shape") and getattr(outputs, "ndim", 0) > 0:
            return outputs
        return None

    def _extract_losses(
        self,
        outputs,
        phase: TrainingPhaseSpec | None = None,
    ) -> dict[str, Any]:
        phase = phase or self.current_phase
        losses = {}
        nested = self._get_value(outputs, "losses")
        if nested is None:
            nested = self._get_value(outputs, "loss_dict")
        if isinstance(nested, Mapping):
            for key in phase.loss_keys:
                value = nested.get(key)
                if self._is_scalar(value):
                    losses[key] = value

        for key in phase.loss_keys:
            value = self._get_value(outputs, key)
            if self._is_scalar(value):
                losses.setdefault(key, value)
        if isinstance(outputs, (tuple, list)) and outputs:
            if self._is_scalar(outputs[0]):
                losses.setdefault("loss", outputs[0])
        if self._is_scalar(outputs):
            losses.setdefault("loss", outputs)
        return losses

    @staticmethod
    def _is_scalar(value) -> bool:
        return value is not None and (
            isinstance(value, (int, float)) or hasattr(value, "ndim") and value.ndim == 0)

    def _aggregate_losses(
        self,
        losses,
        phase: TrainingPhaseSpec | None = None,
    ):
        if not losses:
            return None
        phase = phase or self.current_phase
        weights = phase.loss_weights or self.spec.loss_weights
        if weights:
            weighted = [losses[name] * weight for name, weight in weights if name in losses]
            if weighted:
                return sum(weighted)
        if "loss" in losses:
            return losses["loss"]
        return sum(losses.values())

    def compute_phase_objective(
        self,
        predictions,
        labels,
        context: TrainingContext,
    ):
        """Compute an explicitly configured fallback objective.

        Native losses are always extracted first. A phase without
        ``fallback_objective`` must return a loss from its backend.
        Specialized adapters can override this hook for a true
        architecture-native recipe.
        """
        if context.phase.fallback_objective is None:
            raise ValueError(
                f"Training phase {context.phase.name!r} returned no native loss. "
                "This profile deliberately has no generic fallback objective; "
                "provide backend-native losses or a specialized adapter.")
        return self.compute_objective(predictions, labels)

    def compute_objective(self, predictions, labels):
        """Compute a family fallback when the phase explicitly enables one."""
        raise NotImplementedError

    @staticmethod
    def _require_predictions_and_labels(predictions, labels):
        if predictions is None:
            raise ValueError(
                "The source model returned neither a loss nor predictions. "
                "Return a loss from forward() or override the training adapter.")
        if labels is None:
            raise ValueError(
                "The batch has no target field. Provide one of the phase's "
                "label_names or return a native loss from the source model.")

    @staticmethod
    def _cross_entropy(predictions, labels, *, shift: bool):
        torch = import_optional(
            "torch",
            model_type="Trainer",
            install_extra="training",
        )
        if predictions.ndim < 2:
            raise ValueError("Cross-entropy training requires rank-2+ logits.")
        expected_shape = tuple(predictions.shape[:-1])
        if tuple(labels.shape) != expected_shape:
            raise ValueError(
                "Token cross entropy requires labels to match every logits "
                "dimension except the final vocabulary dimension. Expected "
                f"{expected_shape}, but received {tuple(labels.shape)}. "
                "Align or pack tokens explicitly in the model's dataset "
                "preprocessor; VoiceHub will not silently truncate them.")
        is_floating_point = getattr(labels, "is_floating_point", None)
        if callable(is_floating_point) and is_floating_point():
            raise TypeError("Token cross entropy requires integer class-index labels.")
        logits = predictions
        targets = labels.to(device=logits.device)
        if shift:
            if logits.ndim < 3:
                raise ValueError(
                    "Causal token training requires batched sequence logits "
                    "with shape (..., sequence, vocabulary).")
            if logits.shape[-2] < 2:
                raise ValueError("Causal token training requires at least two aligned tokens.")
            logits = logits[..., :-1, :].contiguous()
            targets = targets[..., 1:].contiguous()
        valid = targets.ne(-100)
        if not bool(valid.any().item()):
            raise ValueError("Token cross entropy received no supervised labels.")
        logits = torch.where(
            valid.unsqueeze(-1),
            logits,
            torch.zeros((), device=logits.device, dtype=logits.dtype),
        )
        targets = torch.where(
            valid,
            targets,
            torch.zeros((), device=targets.device, dtype=targets.dtype),
        )
        per_token = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1).long(),
            reduction="none",
        ).reshape(targets.shape)
        return per_token.masked_select(valid, ).mean()

    _REGRESSION_MASK_NAMES = (
        "loss_mask",
        "label_mask",
        "labels_mask",
        "target_mask",
        "mel_mask",
        "spectrogram_mask",
        "audio_mask",
        "velocity_mask",
        "latent_mask",
    )

    @classmethod
    def _find_explicit_regression_mask(
        cls,
        context: TrainingContext,
    ):
        names = list(cls._REGRESSION_MASK_NAMES)
        names.extend(f"{label_name}_mask" for label_name in context.phase.label_names)
        masks = [(name, context.inputs[name]) for name in dict.fromkeys(names)
                 if name in context.inputs and context.inputs[name] is not None]
        if len(masks) > 1:
            provided = ", ".join(name for name, _ in masks)
            raise ValueError(
                "Regression fallback received multiple explicit loss masks "
                f"({provided}). Provide exactly one target-aligned mask.")
        return masks[0][1] if masks else None

    @staticmethod
    def _regression_loss(
        predictions,
        labels,
        *,
        objective: str,
        mask=None,
    ):
        torch = import_optional(
            "torch",
            model_type="Trainer",
            install_extra="training",
        )
        if tuple(predictions.shape) != tuple(labels.shape):
            raise ValueError(
                "Regression fallbacks require predictions and labels with "
                "identical shapes; implicit broadcasting can train the wrong "
                f"timebase. Received {tuple(predictions.shape)} and "
                f"{tuple(labels.shape)}.")
        targets = labels.to(
            device=predictions.device,
            dtype=predictions.dtype,
        )
        normalized = objective.strip().lower().replace("-", "_")
        if normalized in ("mse", "velocity_mse", "flow_mse"):
            loss_function = torch.nn.functional.mse_loss
        elif normalized == "l1":
            loss_function = torch.nn.functional.l1_loss
        else:
            raise ValueError(f"Unsupported regression fallback objective {objective!r}.")
        if mask is None:
            return loss_function(predictions, targets)

        valid = BaseTrainingAdapter._expand_regression_mask(
            mask,
            targets,
            torch=torch,
        )
        if not bool(valid.any().item()):
            raise ValueError("Regression fallback received no valid target elements after "
                             "masking.")
        safe_predictions = torch.where(
            valid,
            predictions,
            torch.zeros(
                (),
                device=predictions.device,
                dtype=predictions.dtype,
            ),
        )
        safe_targets = torch.where(
            valid,
            targets,
            torch.zeros(
                (),
                device=targets.device,
                dtype=targets.dtype,
            ),
        )
        losses = loss_function(
            safe_predictions,
            safe_targets,
            reduction="none",
        )
        return losses.masked_select(valid).mean()

    @staticmethod
    def _expand_regression_mask(mask, target, *, torch):
        if not isinstance(mask, torch.Tensor):
            try:
                mask = torch.as_tensor(mask, device=target.device)
            except (TypeError, ValueError) as exc:
                raise TypeError("Regression loss masks must be tensor-like.") from exc
        else:
            mask = mask.to(device=target.device)
        mask = mask.bool()
        if mask.ndim > target.ndim:
            raise ValueError(f"Regression loss mask rank {mask.ndim} exceeds target rank "
                             f"{target.ndim}.")
        if tuple(mask.shape) == tuple(target.shape):
            return mask

        candidates = []
        if tuple(target.shape[:mask.ndim]) == tuple(mask.shape):
            candidate = mask
            while candidate.ndim < target.ndim:
                candidate = candidate.unsqueeze(-1)
            candidates.append(candidate)
        if (mask.ndim >= 2 and target.ndim > mask.ndim and mask.shape[0] == target.shape[0] and
                tuple(mask.shape[1:]) == tuple(target.shape[-(mask.ndim - 1):])):
            candidate = mask
            for _ in range(target.ndim - mask.ndim):
                candidate = candidate.unsqueeze(1)
            candidates.append(candidate)
        if not candidates:
            raise ValueError(
                f"Regression loss mask shape {tuple(mask.shape)} is not "
                f"aligned with target shape {tuple(target.shape)}. Expected a "
                "full element mask, a prefix mask such as (batch, time), or "
                "a batch-plus-trailing-time mask.")
        expanded = []
        for candidate in candidates:
            try:
                expanded.append(candidate.expand(target.shape))
            except RuntimeError:
                continue
        if not expanded:
            raise ValueError(
                f"Regression loss mask shape {tuple(mask.shape)} cannot "
                f"expand to target shape {tuple(target.shape)}.")
        if len(expanded) > 1 and not torch.equal(expanded[0], expanded[1]):
            raise ValueError(
                f"Regression loss mask shape {tuple(mask.shape)} is ambiguous "
                f"for target shape {tuple(target.shape)}. Provide an "
                "element-wise mask.")
        return expanded[0]


class CausalLMTrainingAdapter(BaseTrainingAdapter):
    """Autoregressive codec-token objective with shifted cross entropy."""

    def compute_objective(self, predictions, labels):
        self._require_predictions_and_labels(predictions, labels)
        fallback = self.current_phase.fallback_objective
        if fallback not in ("causal_cross_entropy", "cross_entropy", "ce"):
            raise ValueError(f"Unsupported causal-LM fallback objective {fallback!r}.")
        return self._cross_entropy(predictions, labels, shift=True)


class Seq2SeqTrainingAdapter(BaseTrainingAdapter):
    """Teacher-forced sequence objective without causal label shifting."""

    def compute_objective(self, predictions, labels):
        self._require_predictions_and_labels(predictions, labels)
        fallback = self.current_phase.fallback_objective
        if fallback not in ("cross_entropy", "ce"):
            raise ValueError(f"Unsupported sequence fallback objective {fallback!r}.")
        return self._cross_entropy(predictions, labels, shift=False)


class SpeechSeq2SeqTrainingAdapter(Seq2SeqTrainingAdapter):
    """Speech encoder-decoder adapter with an optional token CE fallback.

    Backend-native losses are still extracted before this fallback is
    considered. A profile should only enable ``cross_entropy`` when its
    decoder logits and labels already share the same token timebase.
    """


class UpstreamNativeTrainingAdapter(BaseTrainingAdapter):
    """Require the integrated source runtime to return its native objective.

    This adapter is the safe extension point for objectives whose
    alignment, blank handling, topology, or auxiliary terms cannot be
    inferred from a pair of generic tensors.
    """

    objective_name = "upstream-native"

    def compute_objective(self, predictions, labels):
        del predictions, labels
        raise ValueError(
            f"{self.objective_name} training requires a backend-native loss. "
            "The source model must return a scalar loss from forward(), or a "
            "specialized training adapter must implement the complete native "
            "objective.")


class CTCTrainingAdapter(UpstreamNativeTrainingAdapter):
    """CTC adapter that preserves backend blank and alignment semantics."""

    objective_name = "CTC"


class RNNTTrainingAdapter(UpstreamNativeTrainingAdapter):
    """RNN-T adapter that requires the backend's transducer objective."""

    objective_name = "RNN-T"


class TDTTrainingAdapter(UpstreamNativeTrainingAdapter):
    """Token-and-duration transducer adapter requiring its native objective."""

    objective_name = "TDT"


class AudioClassificationTrainingAdapter(BaseTrainingAdapter):
    """Audio classifier with explicit CE/BCE-with-logits fallbacks.

    Native losses are preferred by :class:`BaseTrainingAdapter`. The
    fallback is only used when a profile explicitly declares
    ``classification``, ``cross_entropy``, or ``binary_cross_entropy``.
    Class logits must use the final dimension. An optional
    ``loss_mask``, ``label_mask``, ``labels_mask``, ``frame_mask``, or
    ``valid_frames`` tensor excludes padded examples or frames from the
    fallback loss.
    """

    _MASK_NAMES = (
        "loss_mask",
        "label_mask",
        "labels_mask",
        "frame_mask",
        "valid_frames",
    )
    _AUTO_OBJECTIVES = ("auto", "classification")
    _CE_OBJECTIVES = ("ce", "cross_entropy")
    _BCE_OBJECTIVES = (
        "bce",
        "binary_cross_entropy",
        "binary_cross_entropy_with_logits",
    )

    def compute_phase_objective(
        self,
        predictions,
        labels,
        context: TrainingContext,
    ):
        if context.phase.fallback_objective is None:
            return super().compute_phase_objective(
                predictions,
                labels,
                context,
            )
        return self._classification_loss(
            predictions,
            labels,
            objective=context.phase.fallback_objective,
            mask=self._find_explicit_loss_mask(context),
        )

    def compute_objective(self, predictions, labels):
        return self._classification_loss(
            predictions,
            labels,
            objective=self.current_phase.fallback_objective,
            mask=None,
        )

    @classmethod
    def _find_explicit_loss_mask(cls, context: TrainingContext):
        names = list(cls._MASK_NAMES)
        names.extend(f"{label_name}_mask" for label_name in context.phase.label_names)
        masks = [(name, context.inputs[name]) for name in dict.fromkeys(names)
                 if name in context.inputs and context.inputs[name] is not None]
        if len(masks) > 1:
            provided = ", ".join(name for name, _ in masks)
            raise ValueError(
                "Classification fallback received multiple explicit loss "
                f"masks ({provided}). Provide exactly one.")
        return masks[0][1] if masks else None

    def _classification_loss(
        self,
        predictions,
        labels,
        *,
        objective,
        mask,
    ):
        self._require_predictions_and_labels(predictions, labels)
        if not isinstance(objective, str) or not objective:
            raise ValueError("Classification fallback requires an explicit objective.")
        objective = objective.strip().lower().replace("-", "_")
        if objective in self._AUTO_OBJECTIVES:
            objective = (
                "binary_cross_entropy" if self._looks_like_binary_or_multilabel(
                    predictions,
                    labels,
                ) else "cross_entropy")
        if objective in self._CE_OBJECTIVES:
            return self._classification_cross_entropy(
                predictions,
                labels,
                mask=mask,
            )
        if objective in self._BCE_OBJECTIVES:
            return self._classification_binary_cross_entropy(
                predictions,
                labels,
                mask=mask,
            )
        supported = ("classification, cross_entropy, or binary_cross_entropy")
        raise ValueError(
            f"Unsupported classification fallback objective {objective!r}. "
            f"Expected {supported}.")

    @staticmethod
    def _looks_like_binary_or_multilabel(predictions, labels) -> bool:
        prediction_shape = tuple(predictions.shape)
        label_shape = tuple(labels.shape)
        return (
            prediction_shape == label_shape or
            predictions.ndim == labels.ndim + 1 and predictions.shape[-1] == 1)

    @staticmethod
    def _classification_cross_entropy(predictions, labels, *, mask):
        torch = import_optional(
            "torch",
            model_type="Trainer",
            install_extra="training",
        )
        if predictions.ndim < 2:
            raise ValueError(
                "Classification cross entropy requires rank-2+ logits with "
                "classes on the final dimension.")
        expected_shape = tuple(predictions.shape[:-1])
        if tuple(labels.shape) != expected_shape:
            raise ValueError(
                "Classification cross entropy requires label shape "
                f"{expected_shape}, but received {tuple(labels.shape)}. "
                "Align frame labels to the model output timebase before "
                "collation.")

        is_floating_point = getattr(labels, "is_floating_point", None)
        if callable(is_floating_point) and is_floating_point():
            raise TypeError(
                "Classification cross entropy requires integer class-index "
                "labels. Use binary_cross_entropy for floating-point binary "
                "or multi-label targets.")
        targets = labels.to(device=predictions.device).long()
        valid = targets.ne(-100)
        if mask is not None:
            valid = valid & AudioClassificationTrainingAdapter._expanded_mask(
                mask,
                targets,
                torch=torch,
            )
        if not bool(valid.any().item()):
            raise ValueError("Classification fallback received no valid labels after "
                             "masking.")
        losses = torch.nn.functional.cross_entropy(
            predictions.reshape(-1, predictions.shape[-1]),
            targets.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).reshape(targets.shape)
        return losses.masked_select(valid).mean()

    @staticmethod
    def _classification_binary_cross_entropy(
        predictions,
        labels,
        *,
        mask,
    ):
        torch = import_optional(
            "torch",
            model_type="Trainer",
            install_extra="training",
        )
        logits = predictions
        targets = labels
        if logits.ndim == targets.ndim + 1 and logits.shape[-1] == 1:
            logits = logits.squeeze(-1)
        elif targets.ndim == logits.ndim + 1 and targets.shape[-1] == 1:
            targets = targets.squeeze(-1)
        if tuple(logits.shape) != tuple(targets.shape):
            raise ValueError(
                "Binary or multi-label classification requires logits and "
                "labels with equal shapes, allowing only a trailing singleton "
                f"class dimension. Received {tuple(predictions.shape)} and "
                f"{tuple(labels.shape)}.")

        targets = targets.to(device=logits.device, dtype=logits.dtype)
        valid = targets.ne(-100)
        if mask is not None:
            valid = valid & AudioClassificationTrainingAdapter._expanded_mask(
                mask,
                targets,
                torch=torch,
            )
        if not bool(valid.any().item()):
            raise ValueError("Classification fallback received no valid labels after "
                             "masking.")
        safe_targets = torch.where(
            valid,
            targets,
            torch.zeros((), device=targets.device, dtype=targets.dtype),
        )
        losses = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            safe_targets,
            reduction="none",
        )
        valid_targets = targets.masked_select(valid)
        targets_are_finite = bool(torch.isfinite(valid_targets).all().item())
        targets_are_bounded = not bool(((valid_targets < 0) | (valid_targets > 1)).any().item())
        if not targets_are_finite or not targets_are_bounded:
            raise ValueError(
                "Binary cross entropy requires finite target values in "
                "[0, 1] for every unmasked label.")
        return losses.masked_select(valid).mean()

    @staticmethod
    def _expanded_mask(mask, target, *, torch):
        if not isinstance(mask, torch.Tensor):
            try:
                mask = torch.as_tensor(mask, device=target.device)
            except (TypeError, ValueError) as exc:
                raise TypeError("Classification loss masks must be tensor-like.") from exc
        else:
            mask = mask.to(device=target.device)
        mask = mask.bool()
        while mask.ndim < target.ndim:
            mask = mask.unsqueeze(-1)
        try:
            return mask.expand(target.shape)
        except RuntimeError as exc:
            raise ValueError(
                f"Classification loss mask shape {tuple(mask.shape)} cannot "
                f"broadcast to label shape {tuple(target.shape)}.") from exc


class FrameClassificationTrainingAdapter(
        AudioClassificationTrainingAdapter, ):
    """Frame classifier using the classification fallback and explicit mask."""


class FlowMatchingTrainingAdapter(BaseTrainingAdapter):
    """Continuous flow objective with strict native-loss preference."""

    def compute_phase_objective(
        self,
        predictions,
        labels,
        context: TrainingContext,
    ):
        if context.phase.fallback_objective is None:
            return super().compute_phase_objective(
                predictions,
                labels,
                context,
            )
        return self._flow_regression_loss(
            predictions,
            labels,
            objective=context.phase.fallback_objective,
            mask=self._find_explicit_regression_mask(context),
        )

    def compute_objective(self, predictions, labels):
        return self._flow_regression_loss(
            predictions,
            labels,
            objective=self.current_phase.fallback_objective,
        )

    def _flow_regression_loss(
        self,
        predictions,
        labels,
        *,
        objective,
        mask=None,
    ):
        if objective not in ("mse", "velocity_mse", "flow_mse"):
            raise ValueError(
                "A plain regression loss is not a complete native flow-matching "
                "objective. Return the backend's native flow loss or explicitly "
                "configure an MSE velocity-target fallback.")
        self._require_predictions_and_labels(predictions, labels)
        return self._regression_loss(
            predictions,
            labels,
            objective=objective,
            mask=mask,
        )


class AcousticTrainingAdapter(BaseTrainingAdapter):
    """Mel, codec, or waveform reconstruction objective."""

    def compute_phase_objective(
        self,
        predictions,
        labels,
        context: TrainingContext,
    ):
        if context.phase.fallback_objective is None:
            return super().compute_phase_objective(
                predictions,
                labels,
                context,
            )
        return self._acoustic_regression_loss(
            predictions,
            labels,
            objective=context.phase.fallback_objective,
            mask=self._find_explicit_regression_mask(context),
        )

    def compute_objective(self, predictions, labels):
        return self._acoustic_regression_loss(
            predictions,
            labels,
            objective=self.current_phase.fallback_objective,
        )

    def _acoustic_regression_loss(
        self,
        predictions,
        labels,
        *,
        objective,
        mask=None,
    ):
        self._require_predictions_and_labels(predictions, labels)
        if objective not in ("l1", "mse"):
            raise ValueError(f"Unsupported acoustic fallback objective {objective!r}.")
        return self._regression_loss(
            predictions,
            labels,
            objective=objective,
            mask=mask,
        )


class CompositeTrainingAdapter(BaseTrainingAdapter):
    """Multi-component adapter that prefers phase-specific native losses."""

    def compute_phase_objective(
        self,
        predictions,
        labels,
        context: TrainingContext,
    ):
        if context.phase.fallback_objective is None:
            return super().compute_phase_objective(
                predictions,
                labels,
                context,
            )
        return self._composite_objective(
            predictions,
            labels,
            fallback=context.phase.fallback_objective,
            mask=self._find_explicit_regression_mask(context),
        )

    def compute_objective(self, predictions, labels):
        return self._composite_objective(
            predictions,
            labels,
            fallback=self.current_phase.fallback_objective,
        )

    def _composite_objective(
        self,
        predictions,
        labels,
        *,
        fallback,
        mask=None,
    ):
        self._require_predictions_and_labels(predictions, labels)
        if fallback == "causal_cross_entropy":
            return self._cross_entropy(predictions, labels, shift=True)
        if fallback in ("cross_entropy", "ce"):
            return self._cross_entropy(predictions, labels, shift=False)
        if fallback == "auto" and (predictions.ndim >= 2 and
                                   not getattr(labels, "is_floating_point", lambda: True)()):
            return self._cross_entropy(predictions, labels, shift=False)
        if fallback in ("auto", "mse"):
            return self._regression_loss(
                predictions,
                labels,
                objective="mse",
                mask=mask,
            )
        if fallback == "l1":
            return self._regression_loss(
                predictions,
                labels,
                objective="l1",
                mask=mask,
            )
        raise ValueError(f"Unsupported composite fallback objective {fallback!r}.")


class VITSTrainingAdapter(CompositeTrainingAdapter):
    """Phase-aware VITS/GAN adapter.

    The base phase executor supplies the important adversarial semantics:
    separately routed optimizers, detached fake inputs, and temporary
    generator/discriminator freezing. Native recipe losses remain mandatory
    unless a phase explicitly declares a reconstruction fallback.
    """
