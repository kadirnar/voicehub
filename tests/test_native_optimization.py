from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from importlib import import_module
from pathlib import Path
from threading import RLock
from types import MethodType, SimpleNamespace

from voicehub.models.base import BaseSpeechModel
from voicehub.optimization import (
    OptimizationApplicationError,
    OptimizationCapabilities,
    OptimizationCompatibilityError,
    OptimizationContext,
    OptimizationError,
    OptimizationMode,
    OptimizationPass,
    OptimizationPassManager,
    OptimizationPassRegistry,
    PassResult,
    register_optimization_pass,
    unregister_optimization_pass,
)
from voicehub.outputs import ASROutput, SpeechSegment, TTSOutput, VADOutput
from voicehub.registry import ModelSpec, list_model_specs, register_model_spec, unregister_model_spec
from voicehub.runtime import get_architecture_spec
from voicehub.training.adapters import BaseTrainingAdapter
from voicehub.training.arguments import TrainingArguments
from voicehub.training.contracts import TrainingPhaseSpec, TrainingSupport
from voicehub.training.specs import ModelTrainingSpec, TrainingFamily
from voicehub.training.strategy import TorchTrainingStrategy
from voicehub.training.trainer import Trainer
from voicehub.training.utils import CHECKPOINT_MANIFEST_NAME, MODEL_STATE_NAME, OPTIMIZATION_MANIFEST_NAME


class _AddPass(OptimizationPass):
    pass_id = "test.add"
    pass_version = "1"
    capabilities = OptimizationCapabilities(
        modes=(OptimizationMode.INFERENCE, OptimizationMode.TRAINING),
        streaming_safe=True,
        distributed_safe=True,
        reversible=True,
    )

    def __init__(self, amount=1):
        self.amount = amount

    def manifest_configuration(self):
        return {"amount": self.amount}

    def apply(self, model, context):
        return PassResult(
            model=model + self.amount,
            state={"amount": self.amount},
        )

    def restore(self, model, state, context):
        return model - state["amount"]


class _InferencePass(OptimizationPass):
    pass_id = "test.inference"
    pass_version = "1"
    capabilities = OptimizationCapabilities(modes=(OptimizationMode.INFERENCE, ), )

    def manifest_configuration(self):
        return {}

    def apply(self, model, context):
        return PassResult(model=model)


class _CompileAddPass(_AddPass):
    pass_id = "test.compile-add"
    optimization_kind = ".COMPILE"


class _SdpaAddPass(_AddPass):
    pass_id = "test.sdpa-add"
    optimization_kind = "sdpa"
    requires_architecture_support = True


class _RecordingCompilePass(_CompileAddPass):

    def __init__(self):
        super().__init__()
        self.apply_calls = 0

    def manifest_configuration(self):
        return {"amount": self.amount}

    def apply(self, model, context):
        self.apply_calls += 1
        return super().apply(model, context)


class _FailingPass(OptimizationPass):
    pass_id = "test.failure"
    pass_version = "1"
    capabilities = OptimizationCapabilities(modes=(OptimizationMode.INFERENCE, ), )

    def manifest_configuration(self):
        return {}

    def apply(self, model, context):
        raise RuntimeError("deliberate")


class _LifecycleModel(BaseSpeechModel):

    def __init__(self):
        super().__init__(device="cpu")
        self.model = None
        self._lifecycle_lock = RLock()

    def load(self):
        self._validate_optimization_transition("inference")
        if self.model is None:
            self.model = 1
        return self

    def load_for_training(self):
        self._validate_optimization_transition("training")
        if self.model is None:
            self.model = 1
        return self

    def __call__(self):
        return self.model


class _RegisteredLifecycleModel(_LifecycleModel):

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="dia")


class _RegistrySemanticRuntime:

    def __init__(self, output):
        self.output = output

    def semantic_output(self):
        return self.output

    @staticmethod
    def state_dict():
        return {"sentinel": 1}


def _normalized_output(task):
    if task.value == "text-to-speech":
        return TTSOutput(audio=(0.0, ), sample_rate=16_000)
    if task.value == "automatic-speech-recognition":
        return ASROutput(text="registry contract")
    return VADOutput(
        segments=(SpeechSegment(start=0.0, end=0.5, score=1.0), ),
        duration=1.0,
        sample_rate=16_000,
    )


class NativeOptimizationTests(unittest.TestCase):

    def test_every_registered_model_uses_the_shared_public_lifecycle(self):
        lifecycle_methods = (
            "available_optimization_passes",
            "apply_optimization_plan",
            "optimization_result",
            "optimization_manifest",
            "restore_optimization_plan",
        )
        specs = list_model_specs(task=None)
        self.assertTrue(specs)

        for spec in specs:
            with self.subTest(model_type=spec.model_type):
                model_class = getattr(
                    import_module(spec.module),
                    spec.class_name,
                )
                self.assertTrue(
                    issubclass(model_class, BaseSpeechModel),
                    f"Registered model {spec.model_type!r} does not inherit "
                    "VoiceHub's shared speech-model contract.",
                )
                for method_name in lifecycle_methods:
                    owner = next(base for base in model_class.__mro__ if method_name in base.__dict__)
                    self.assertIs(
                        owner,
                        BaseSpeechModel,
                        f"Registered model {spec.model_type!r} overrides public "
                        f"optimization lifecycle method {method_name!r} in "
                        f"{owner.__module__}.{owner.__name__}; extend the shared "
                        "capability protocols instead.",
                    )

    def test_every_public_pass_has_a_reported_registry_wide_lifecycle(self):
        pass_names = BaseSpeechModel.available_optimization_passes()
        specs = list_model_specs(task=None)
        self.assertTrue(pass_names)
        self.assertTrue(specs)

        for spec in specs:
            architecture = (get_architecture_spec(spec.architecture) if spec.architecture else None)
            dtype = "float32"
            if architecture is not None:
                self.assertTrue(
                    architecture.capabilities.supports_device("cpu"),
                    f"{spec.model_type!r} has no CPU-safe architecture contract.",
                )
                if not architecture.capabilities.supports_dtype(dtype):
                    dtype = architecture.capabilities.dtypes[0]

            model_class = getattr(import_module(spec.module), spec.class_name)
            for pass_name in pass_names:
                with self.subTest(
                        model_type=spec.model_type,
                        optimization_pass=pass_name,
                ):
                    model = object.__new__(model_class)
                    BaseSpeechModel.__init__(model, device="cpu")
                    model.config = SimpleNamespace(model_type=spec.model_type)
                    runtime = _RegistrySemanticRuntime(_normalized_output(spec.task))
                    model.model = runtime
                    model.load = MethodType(lambda instance: instance.model, model)

                    before_output = runtime.semantic_output()
                    before_state = runtime.state_dict()
                    result = model.apply_optimization_plan(
                        pass_name,
                        mode="inference",
                        context=OptimizationContext(
                            mode="inference",
                            device="cpu",
                            dtype=dtype,
                        ),
                    )
                    manifest = model.optimization_manifest(mode="inference")
                    entry = manifest["passes"][0]
                    outcome = entry["metadata"]["outcome"]

                    self.assertIs(result.model, runtime)
                    self.assertEqual(
                        result.context.architecture,
                        (None if architecture is None else architecture.architecture_id))
                    self.assertIn(
                        outcome,
                        {
                            "compiled",
                            "configured",
                            "eager-fallback",
                            "not-applicable",
                        },
                    )
                    self.assertNotEqual(outcome, "skipped")
                    if outcome in {"eager-fallback", "not-applicable"}:
                        self.assertTrue(entry["metadata"].get("reason"))
                    self.assertEqual(
                        json.loads(json.dumps(manifest, allow_nan=False, sort_keys=True)),
                        manifest,
                    )
                    self.assertIsInstance(runtime.semantic_output(), type(before_output))
                    self.assertEqual(runtime.semantic_output(), before_output)
                    self.assertEqual(runtime.state_dict(), before_state)

                    self.assertIs(
                        model.restore_optimization_plan(mode="inference"),
                        runtime,
                    )
                    self.assertIsNone(model.optimization_result(mode="inference"))
                    self.assertEqual(runtime.semantic_output(), before_output)
                    self.assertEqual(runtime.state_dict(), before_state)

    def test_plan_validates_every_pass_before_transforming(self):
        add = _AddPass()
        context = OptimizationContext(mode="training")

        with self.assertRaises(OptimizationCompatibilityError):
            OptimizationPassManager().apply(
                0,
                (add, _InferencePass()),
                context,
            )

        self.assertEqual(add.amount, 1)

    def test_reversible_result_restores_original_model(self):
        result = OptimizationPassManager().apply(
            3,
            (_AddPass(2), ),
            OptimizationContext(mode="inference", streaming=True),
        )

        self.assertEqual(result.model, 5)
        self.assertEqual(result.restore(), 3)
        self.assertEqual(
            result.manifest_metadata()[0]["pass"],
            "test.add",
        )

    def test_failure_reports_the_pass_and_rolls_back_prior_steps(self):
        with self.assertRaises(OptimizationApplicationError) as context:
            OptimizationPassManager().apply(
                0,
                (_AddPass(), _FailingPass()),
                OptimizationContext(mode="inference"),
            )

        self.assertEqual(context.exception.pass_id, "test.failure@1")
        self.assertEqual(context.exception.rollback_errors, ())

    def test_invalid_manifest_metadata_rolls_back_the_current_pass(self):

        class InvalidMetadataPass(OptimizationPass):
            pass_id = "test.invalid-metadata"
            pass_version = "1"
            capabilities = OptimizationCapabilities(
                modes=(OptimizationMode.INFERENCE, ),
                reversible=True,
            )

            def manifest_configuration(self):
                return {}

            def apply(self, model, context):
                previous = model["value"]
                model["value"] = previous + 1
                return PassResult(
                    model=model,
                    state={"previous": previous},
                    metadata={"not_json": object()},
                )

            def restore(self, model, state, context):
                model["value"] = state["previous"]
                return model

        model = {"value": 0}
        with self.assertRaises(OptimizationApplicationError):
            OptimizationPassManager().apply(
                model,
                (InvalidMetadataPass(), ),
                OptimizationContext(mode="inference"),
            )

        self.assertEqual(model, {"value": 0})

    def test_registry_resolves_factories_lazily(self):
        registry = OptimizationPassRegistry()
        created = []
        registry.register("add", lambda: created.append(True) or _AddPass())

        self.assertEqual(created, [])
        self.assertIsInstance(registry.create("add"), _AddPass)
        self.assertEqual(created, [True])

    def test_public_pass_registration_supports_decorators(self):
        name = "test-public-add"
        removed = None
        try:

            @register_optimization_pass(name)
            def create_pass():
                return _AddPass(3)

            self.assertIn(name, _LifecycleModel.available_optimization_passes())
            result = OptimizationPassManager().apply_plan(
                2,
                name,
                OptimizationContext(mode="inference"),
            )
        finally:
            removed = unregister_optimization_pass(name, missing_ok=True)

        self.assertEqual(result.model, 5)
        self.assertIs(create_pass, removed)

    def test_named_plan_resolution_and_manifest_are_deterministic(self):
        registry = OptimizationPassRegistry()
        registry.register("add-two", lambda: _AddPass(2))

        result = OptimizationPassManager().apply_plan(
            3,
            "add-two",
            OptimizationContext(mode="inference"),
            registry=registry,
        )

        self.assertEqual(result.model, 5)
        self.assertEqual(
            result.manifest(), {
                "format_version":
                3,
                "context": {
                    "mode": "inference",
                    "architecture": None,
                    "device": "cpu",
                    "dtype": "float32",
                    "streaming": False,
                    "distributed": False,
                    "persist_result": False,
                },
                "passes": [{
                    "pass": "test.add",
                    "kind": "test.add",
                    "version": "1",
                    "configuration": {
                        "amount": 2,
                    },
                    "capabilities": {
                        "modes": ["inference", "training"],
                        "devices": ["cpu", "cuda", "mps"],
                        "dtypes": ["float32", "float16", "bfloat16"],
                        "streaming_safe": True,
                        "distributed_safe": True,
                        "persistent": False,
                        "reversible": True,
                        "changes_parameter_names": False,
                        "changes_topology": False,
                        "portable_export": False,
                    },
                    "metadata": {},
                }],
            })

    def test_architecture_compatibility_is_checked_before_mutation(self):
        compatible = OptimizationPassManager().apply(
            1,
            (_CompileAddPass(), ),
            OptimizationContext(
                mode="inference",
                architecture="dia-tts",
            ),
        )

        self.assertEqual(compatible.model, 2)
        self.assertEqual(
            compatible.manifest()["context"]["architecture"],
            "dia",
        )
        self.assertEqual(
            compatible.manifest()["passes"][0]["kind"],
            "compile",
        )

        incompatible = _SdpaAddPass()
        with self.assertRaisesRegex(OptimizationCompatibilityError, "does not declare compatibility"):
            OptimizationPassManager().apply(
                1,
                (incompatible, ),
                OptimizationContext(
                    mode="inference",
                    architecture="dia",
                ),
            )
        self.assertEqual(incompatible.amount, 1)

    def test_new_pass_kinds_do_not_require_every_architecture_to_change(self):
        result = OptimizationPassManager().apply(
            1,
            (_AddPass(), ),
            OptimizationContext(
                mode="inference",
                architecture="dia",
            ),
        )

        self.assertEqual(result.model, 2)
        self.assertEqual(
            result.manifest()["passes"][0]["kind"],
            "test.add",
        )

    def test_compatible_architecture_metadata_does_not_register_a_pass(self):
        with self.assertRaisesRegex(KeyError, "Unknown optimization pass"):
            OptimizationPassManager().apply_plan(
                1,
                "compile",
                OptimizationContext(
                    mode="inference",
                    architecture="dia",
                ),
                registry=OptimizationPassRegistry(),
            )

    def test_architecture_runtime_preflight_finishes_before_mutation(self):
        cases = (
            (
                OptimizationContext(
                    mode="inference",
                    architecture="dia",
                    device="xpu",
                ),
                "device",
            ),
            (
                OptimizationContext(
                    mode="inference",
                    architecture="dia",
                    dtype="float64",
                ),
                "dtype",
            ),
            (
                OptimizationContext(
                    mode="training",
                    architecture="webrtc-vad",
                ),
                "training execution",
            ),
            (
                OptimizationContext(
                    mode="inference",
                    architecture="dia",
                    streaming=True,
                ),
                "streaming execution",
            ),
            (
                OptimizationContext(
                    mode="training",
                    architecture="openvoice-v2-converter",
                    distributed=True,
                ),
                "distributed training",
            ),
            (
                OptimizationContext(
                    mode="inference",
                    architecture="dia",
                    distributed=True,
                ),
                "distributed inference",
            ),
        )

        for context, message in cases:
            with self.subTest(message=message):
                optimization_pass = _RecordingCompilePass()
                with self.assertRaisesRegex(
                        OptimizationCompatibilityError,
                        message,
                ):
                    OptimizationPassManager().apply(
                        1,
                        (optimization_pass, ),
                        context,
                    )
                self.assertEqual(optimization_pass.apply_calls, 0)

    def test_registered_model_without_architecture_remains_agnostic(self):
        register_model_spec(
            ModelSpec(
                model_type="optimization-agnostic-test",
                module="voicehub.models.base",
                class_name="BaseSpeechModel",
                default_model_path="",
                architecture=None,
            ))
        try:

            class AgnosticLifecycleModel(_LifecycleModel):

                def __init__(self):
                    super().__init__()
                    self.config = SimpleNamespace(model_type="optimization-agnostic-test")

            result = AgnosticLifecycleModel().apply_optimization_plan(
                _AddPass(),
                mode="inference",
            )
        finally:
            unregister_model_spec("optimization-agnostic-test")

        self.assertIsNone(result.context.architecture)
        self.assertEqual(result.model, 2)

    def test_pass_manifest_is_an_immutable_strict_json_snapshot(self):
        configuration = {
            "nested": {
                "values": [1, 2],
            },
        }
        metadata = {
            "nested": {
                "result": ["stable"],
            },
        }

        class SnapshotPass(OptimizationPass):
            pass_id = "test.snapshot"
            pass_version = "1"
            capabilities = OptimizationCapabilities(modes=(OptimizationMode.INFERENCE, ), )

            def manifest_configuration(self):
                return configuration

            def apply(self, model, context):
                configuration["nested"]["values"].append(3)
                return PassResult(
                    model=model,
                    metadata=metadata,
                )

        result = OptimizationPassManager().apply(
            object(),
            (SnapshotPass(), ),
            OptimizationContext(mode="inference"),
        )
        expected = result.manifest()
        metadata["nested"]["result"].append("mutated")
        configuration["nested"]["values"].append(4)

        self.assertEqual(result.manifest(), expected)
        self.assertEqual(
            expected["passes"][0]["configuration"]["nested"]["values"],
            [1, 2],
        )
        self.assertEqual(
            json.loads(json.dumps(expected, sort_keys=True)),
            expected,
        )

    def test_manifest_configuration_rejects_non_string_keys_before_apply(self):

        class InvalidConfigurationPass(OptimizationPass):
            pass_id = "test.invalid-configuration"
            pass_version = "1"
            capabilities = OptimizationCapabilities(modes=(OptimizationMode.INFERENCE, ), )

            def __init__(self):
                self.applied = False

            def manifest_configuration(self):
                return {"nested": {1: "coercion-is-forbidden"}}

            def apply(self, model, context):
                self.applied = True
                return PassResult(model=model)

        optimization_pass = InvalidConfigurationPass()
        with self.assertRaisesRegex(TypeError, "non-string mapping key"):
            OptimizationPassManager().apply(
                object(),
                (optimization_pass, ),
                OptimizationContext(mode="inference"),
            )
        self.assertFalse(optimization_pass.applied)

    def test_optimization_manifest_rejects_runtime_credentials(self):
        credential = "credential-sentinel-value"

        class CredentialManifestPass(OptimizationPass):
            pass_id = "test.credential-manifest"
            pass_version = "1"
            capabilities = OptimizationCapabilities(
                modes=(OptimizationMode.INFERENCE, ),
                reversible=True,
            )

            def __init__(
                self,
                *,
                configuration=None,
                metadata=None,
                runtime_status=None,
            ):
                self.configuration = configuration or {}
                self.metadata = metadata or {}
                self.runtime_status = runtime_status
                self.apply_calls = 0
                self.restore_calls = 0

            def manifest_configuration(self):
                return self.configuration

            def apply(self, model, context):
                self.apply_calls += 1
                return PassResult(model=model, metadata=self.metadata)

            def restore(self, model, state, context):
                self.restore_calls += 1
                return model

            def runtime_manifest_status(self, result):
                return self.runtime_status

        unsafe_configuration = CredentialManifestPass(
            configuration={
                "provider_options": {
                    "api_key": credential,
                },
            }, )
        with self.assertRaisesRegex(ValueError, "provider_options.api_key") as raised:
            OptimizationPassManager().apply(
                object(),
                (unsafe_configuration, ),
                OptimizationContext(mode="inference"),
            )
        self.assertNotIn(credential, str(raised.exception))
        self.assertEqual(unsafe_configuration.apply_calls, 0)

        unsafe_metadata = CredentialManifestPass(
            configuration={"token_count": 3},
            metadata={
                "provider_options": {
                    "password": credential,
                },
            },
        )
        with self.assertRaises(OptimizationApplicationError) as raised:
            OptimizationPassManager().apply(
                object(),
                (unsafe_metadata, ),
                OptimizationContext(mode="inference"),
            )
        self.assertIn("provider_options.password", str(raised.exception))
        self.assertNotIn(credential, str(raised.exception))
        self.assertEqual(unsafe_metadata.apply_calls, 1)
        self.assertEqual(unsafe_metadata.restore_calls, 1)

        mutable_status = {"token_count": 5}
        safe_pass = CredentialManifestPass(
            configuration={"token_count": 3},
            metadata={"token_count": 4},
            runtime_status=mutable_status,
        )
        result = OptimizationPassManager().apply(
            object(),
            (safe_pass, ),
            OptimizationContext(mode="inference"),
        )
        self.assertEqual(
            result.manifest()["passes"][0]["runtime_status"]["token_count"],
            5,
        )
        mutable_status["provider_options"] = {"auth_token": credential}
        with self.assertRaisesRegex(ValueError, "provider_options.auth_token") as raised:
            result.manifest()
        self.assertNotIn(credential, str(raised.exception))

    def test_same_identity_with_different_configuration_has_distinct_manifest(self):
        first = OptimizationPassManager().apply(
            0,
            (_AddPass(1), ),
            OptimizationContext(mode="inference"),
        )
        second = OptimizationPassManager().apply(
            0,
            (_AddPass(2), ),
            OptimizationContext(mode="inference"),
        )

        self.assertNotEqual(first.manifest(), second.manifest())
        self.assertEqual(
            first.manifest()["passes"][0]["pass"],
            second.manifest()["passes"][0]["pass"],
        )
        self.assertEqual(
            first.manifest()["passes"][0]["version"],
            second.manifest()["passes"][0]["version"],
        )

    def test_reversible_declaration_requires_restore_override(self):

        class InvalidReversiblePass(OptimizationPass):
            pass_id = "test.invalid-reversible"
            pass_version = "1"
            capabilities = OptimizationCapabilities(
                modes=(OptimizationMode.INFERENCE, ),
                reversible=True,
            )

            def manifest_configuration(self):
                return {}

            def apply(self, model, context):
                return PassResult(model=model)

        with self.assertRaisesRegex(TypeError, "must override restore"):
            OptimizationPassManager().apply(
                object(),
                (InvalidReversiblePass(), ),
                OptimizationContext(mode="inference"),
            )

    def test_model_lifecycle_binds_the_registered_architecture(self):
        model = _RegisteredLifecycleModel()

        result = model.apply_optimization_plan(
            _CompileAddPass(),
            mode="inference",
        )

        self.assertEqual(result.context.architecture, "dia")
        self.assertEqual(result.model, 2)

        model.restore_optimization_plan(mode="inference")
        with self.assertRaisesRegex(ValueError, "does not match the model"):
            model.apply_optimization_plan(
                _CompileAddPass(),
                mode="inference",
                context=OptimizationContext(
                    mode="inference",
                    architecture="kokoro",
                ),
            )

    def test_model_lifecycle_requires_explicit_cross_mode_restore(self):
        model = _LifecycleModel()
        registry = OptimizationPassRegistry()
        registry.register("add-two", lambda: _AddPass(2))

        result = model.apply_optimization_plan(
            "add-two",
            mode="inference",
            registry=registry,
        )

        self.assertEqual(result.model, 3)
        self.assertEqual(
            model.optimization_manifest(mode="inference")["passes"][0]["pass"],
            "test.add",
        )
        with self.assertRaisesRegex(RuntimeError, "explicitly"):
            model.load_for_training()

        self.assertEqual(
            model.restore_optimization_plan(mode="inference"),
            1,
        )
        model.load_for_training()

    def test_optimization_contract_import_remains_torch_lazy(self):
        script = """
import importlib.abc
import sys

class BlockTensorRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in {"torch", "numpy"}:
            raise ModuleNotFoundError(fullname)
        return None

sys.meta_path.insert(0, BlockTensorRuntime())
from voicehub.optimization import OptimizationPassManager
print("torch" in sys.modules, "numpy" in sys.modules)
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.stdout.strip(), "False False")

    def test_trainer_requires_persistent_optimization_context(self):
        with self.assertRaisesRegex(ValueError, "persist_result=True"):
            Trainer(
                model=object(),
                optimization_plan=_AddPass(),
                optimization_context=OptimizationContext(
                    mode="training",
                    persist_result=False,
                ),
            )

    def test_trainer_rejects_nonpersistent_pass_before_apply(self):
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("PyTorch is required for Trainer optimization")

        class TinyModel(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(()))

            def forward(self, input_values, labels):
                predictions = input_values * self.weight
                return {
                    "loss": (predictions - labels).square().mean(),
                }

        class NonpersistentPass(OptimizationPass):
            pass_id = "test.nonpersistent-training"
            pass_version = "1"
            capabilities = OptimizationCapabilities(modes=(OptimizationMode.TRAINING, ), )

            def __init__(self):
                self.apply_calls = 0

            def manifest_configuration(self):
                return {}

            def apply(self, model, context):
                self.apply_calls += 1
                return PassResult(model=model)

        optimization_pass = NonpersistentPass()
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(
                model=TinyModel(),
                args=TrainingArguments(
                    output_dir=directory,
                    max_steps=1,
                    per_device_train_batch_size=1,
                    logging_strategy="no",
                    save_strategy="no",
                    use_cpu=True,
                ),
                train_dataset=[{
                    "input_values": torch.ones(1),
                    "labels": torch.ones(1),
                }],
                optimization_plan=optimization_pass,
            )
            with self.assertRaisesRegex(
                    OptimizationCompatibilityError,
                    "persistent checkpoint output",
            ):
                trainer.train()

        self.assertEqual(optimization_pass.apply_calls, 0)

    def test_trainer_applies_named_plan_before_optimizer_and_records_it(self):
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("PyTorch is required for Trainer optimization")

        class TinyModel(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(model_type="dia")
                self.projection = torch.nn.Linear(1, 1)

            def forward(self, input_values, labels):
                predictions = self.projection(input_values)
                return {
                    "loss": torch.nn.functional.mse_loss(
                        predictions,
                        labels,
                    ),
                    "logits": predictions,
                }

        class ScaledRuntime(torch.nn.Module):

            def __init__(self, runtime):
                super().__init__()
                self.runtime = runtime
                self.scale = torch.nn.Parameter(torch.ones(()))

            def forward(self, input_values, labels):
                return self.runtime(
                    input_values * self.scale,
                    labels,
                )

        class WrapPass(OptimizationPass):
            pass_id = "test.training-wrap"
            pass_version = "1"
            optimization_kind = "compile"
            capabilities = OptimizationCapabilities(
                modes=(OptimizationMode.TRAINING, ),
                distributed_safe=True,
                persistent=True,
                reversible=True,
                changes_parameter_names=True,
            )

            def manifest_configuration(self):
                return {}

            def apply(self, model, context):
                return PassResult(
                    model=ScaledRuntime(model),
                    state={"model": model},
                    metadata={"wrapper": "scaled"},
                )

            def restore(self, model, state, context):
                return state["model"]

        registry = OptimizationPassRegistry()
        registry.register("training-wrap", WrapPass)
        dataset = [{
            "input_values": torch.tensor([1.0]),
            "labels": torch.tensor([2.0]),
        }]

        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(
                model=TinyModel(),
                args=TrainingArguments(
                    output_dir=directory,
                    max_steps=1,
                    per_device_train_batch_size=1,
                    logging_strategy="no",
                    save_strategy="steps",
                    save_steps=1,
                    use_cpu=True,
                ),
                train_dataset=dataset,
                optimization_plan="training-wrap",
                optimization_pass_registry=registry,
            )
            trainer.train()

            result = trainer.optimization_result
            self.assertIsInstance(result.model, ScaledRuntime)
            optimizer_parameter_ids = {
                id(parameter)
                for group in trainer.optimizer.param_groups
                for parameter in group["params"]
            }
            self.assertIn(id(result.model.scale), optimizer_parameter_ids)

            checkpoint = Path(directory) / "checkpoint-1"
            optimization_manifest = json.loads(
                (checkpoint / OPTIMIZATION_MANIFEST_NAME).read_text(encoding="utf-8"))
            checkpoint_manifest = json.loads(
                (checkpoint / CHECKPOINT_MANIFEST_NAME).read_text(encoding="utf-8"))
            mismatched = Trainer(
                model=TinyModel(),
                args=TrainingArguments(
                    output_dir=directory,
                    max_steps=1,
                    per_device_train_batch_size=1,
                    logging_strategy="no",
                    save_strategy="no",
                    use_cpu=True,
                ),
                train_dataset=dataset,
            )
            with self.assertRaisesRegex(ValueError, "optimization plan"):
                mismatched.train(resume_from_checkpoint=str(checkpoint))

        self.assertEqual(
            optimization_manifest["passes"][0]["pass"],
            "test.training-wrap",
        )
        self.assertEqual(
            optimization_manifest["passes"][0]["kind"],
            "compile",
        )
        self.assertEqual(
            optimization_manifest["context"]["architecture"],
            "dia",
        )
        self.assertEqual(
            checkpoint_manifest["optimization_plan"],
            optimization_manifest,
        )

    def test_checkpoint_resume_rejects_changed_pass_configuration(self):
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("PyTorch is required for Trainer optimization")

        class TinyModel(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(model_type="dia")
                self.weight = torch.nn.Parameter(torch.ones(()))

            def forward(self, input_values, labels):
                predictions = input_values * self.weight
                return {
                    "loss": (predictions - labels).square().mean(),
                }

        class ConfiguredPass(OptimizationPass):
            pass_id = "test.configured-resume"
            pass_version = "1"
            optimization_kind = "compile"
            capabilities = OptimizationCapabilities(
                modes=(OptimizationMode.TRAINING, ),
                persistent=True,
            )

            def __init__(self, algorithm):
                self.algorithm = algorithm

            def manifest_configuration(self):
                return {"algorithm": self.algorithm}

            def apply(self, model, context):
                return PassResult(model=model)

        dataset = [{
            "input_values": torch.ones(1),
            "labels": torch.zeros(1),
        }]
        with tempfile.TemporaryDirectory() as directory:
            common_args = {
                "output_dir": directory,
                "max_steps": 1,
                "per_device_train_batch_size": 1,
                "logging_strategy": "no",
                "use_cpu": True,
            }
            Trainer(
                model=TinyModel(),
                args=TrainingArguments(
                    **common_args,
                    save_strategy="steps",
                    save_steps=1,
                ),
                train_dataset=dataset,
                optimization_plan=ConfiguredPass("baseline"),
            ).train()
            checkpoint = Path(directory) / "checkpoint-1"
            resumed = Trainer(
                model=TinyModel(),
                args=TrainingArguments(
                    **common_args,
                    save_strategy="no",
                ),
                train_dataset=dataset,
                optimization_plan=ConfiguredPass("different"),
            )

            with self.assertRaisesRegex(ValueError, "optimization plan"):
                resumed.train(resume_from_checkpoint=str(checkpoint))

    def test_separate_optimizer_recipe_rejects_unrouted_topology_pass(self):
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("PyTorch is required for Trainer optimization")

        class CompositeModel(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.generator = torch.nn.Linear(1, 1)
                self.discriminator = torch.nn.Linear(1, 1)

        class UnroutedPass(OptimizationPass):
            pass_id = "test.unrouted-topology"
            pass_version = "1"
            capabilities = OptimizationCapabilities(
                modes=(OptimizationMode.TRAINING, ),
                persistent=True,
                changes_topology=True,
            )

            def __init__(self):
                self.apply_calls = 0

            def manifest_configuration(self):
                return {}

            def apply(self, model, context):
                self.apply_calls += 1
                return PassResult(model=model)

        model = CompositeModel()
        spec = ModelTrainingSpec(
            model_type="optimization-routing-test",
            family=TrainingFamily.COMPOSITE,
            module_paths=("generator", ),
            component_paths=("generator", "discriminator"),
            support=TrainingSupport.PREPROCESSED,
            separate_optimizers=True,
            phases=(
                TrainingPhaseSpec(
                    name="generator",
                    component_paths=("generator", ),
                    optimizer_names=("generator", ),
                    forward_component="generator",
                ),
                TrainingPhaseSpec(
                    name="discriminator",
                    component_paths=("discriminator", ),
                    optimizer_names=("discriminator", ),
                    forward_component="discriminator",
                ),
            ),
        )
        adapter = BaseTrainingAdapter(model, spec)
        optimization_pass = UnroutedPass()
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(
                model=model,
                training_adapter=adapter,
                args=TrainingArguments(
                    output_dir=directory,
                    max_steps=1,
                    per_device_train_batch_size=1,
                    logging_strategy="no",
                    save_strategy="no",
                    use_cpu=True,
                ),
                train_dataset=[{
                    "input_values": torch.ones(1),
                    "labels": torch.zeros(1),
                }],
                optimization_plan=optimization_pass,
            )
            with self.assertRaisesRegex(
                    ValueError,
                    "complete optimizer-routing",
            ):
                trainer.train()

        self.assertEqual(optimization_pass.apply_calls, 0)

    def test_separate_optimizer_recipe_accepts_complete_post_transform_routes(self):
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("PyTorch is required for Trainer optimization")

        class CompositeModel(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.generator = torch.nn.Linear(1, 1)
                self.discriminator = torch.nn.Linear(1, 1)

        class RoutedRuntime:

            def __init__(self, adapter):
                self.adapter = adapter

            def named_parameters(self):
                for name, parameter in self.adapter.named_parameters():
                    yield f"optimized.{name}", parameter

            def parameters(self):
                for _, parameter in self.named_parameters():
                    yield parameter

            def train(self, mode=True):
                self.adapter.train(mode)

            def eval(self):
                self.adapter.eval()

            def __call__(self, *, training_context):
                return self.adapter.compute_step(training_context)

        class RoutedPass(OptimizationPass):
            pass_id = "test.routed-topology"
            pass_version = "1"
            capabilities = OptimizationCapabilities(
                modes=(OptimizationMode.TRAINING, ),
                persistent=True,
                changes_parameter_names=True,
            )

            def manifest_configuration(self):
                return {}

            def apply(self, model, context):
                return PassResult(model=RoutedRuntime(model))

            def route_optimizer_parameters(
                self,
                model,
                *,
                optimizer_names,
            ):
                original = dict(model.adapter.named_parameter_groups())
                return {
                    optimizer_name: [(f"optimized.{name}", parameter)
                                     for name, parameter in original[optimizer_name]]
                    for optimizer_name in optimizer_names
                }

        model = CompositeModel()
        spec = ModelTrainingSpec(
            model_type="optimization-routing-success-test",
            family=TrainingFamily.COMPOSITE,
            module_paths=("generator", ),
            component_paths=("generator", "discriminator"),
            support=TrainingSupport.PREPROCESSED,
            separate_optimizers=True,
            phases=(
                TrainingPhaseSpec(
                    name="generator",
                    component_paths=("generator", ),
                    optimizer_names=("generator", ),
                    forward_component="generator",
                ),
                TrainingPhaseSpec(
                    name="discriminator",
                    component_paths=("discriminator", ),
                    optimizer_names=("discriminator", ),
                    forward_component="discriminator",
                ),
            ),
        )
        adapter = BaseTrainingAdapter(model, spec)
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(
                model=model,
                training_adapter=adapter,
                args=TrainingArguments(
                    output_dir=directory,
                    max_steps=1,
                    per_device_train_batch_size=1,
                    logging_strategy="no",
                    save_strategy="no",
                    use_cpu=True,
                ),
                optimization_plan=RoutedPass(),
            )
            trainer._ensure_model_loaded()
            trainer._move_model_to_device()
            optimizer = trainer.create_optimizer()

        self.assertEqual(
            set(optimizer.optimizers),
            {"generator", "discriminator"},
        )

    def test_public_save_rejects_nonportable_topology_transform(self):
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("PyTorch is required for Trainer optimization")

        class TinyModel(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(()))

            def forward(self, input_values, labels):
                prediction = input_values * self.weight
                return {
                    "loss": (prediction - labels).square().mean(),
                }

        class RuntimeWrapper(torch.nn.Module):

            def __init__(self, canonical):
                super().__init__()
                self.canonical = canonical

            def forward(self, **inputs):
                return self.canonical(**inputs)

        class NonportablePass(OptimizationPass):
            pass_id = "test.nonportable-topology"
            pass_version = "1"
            capabilities = OptimizationCapabilities(
                modes=(OptimizationMode.TRAINING, ),
                persistent=True,
                changes_parameter_names=True,
            )

            def manifest_configuration(self):
                return {}

            def apply(self, model, context):
                return PassResult(model=RuntimeWrapper(model))

        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(
                model=TinyModel(),
                args=TrainingArguments(
                    output_dir=directory,
                    max_steps=1,
                    per_device_train_batch_size=1,
                    logging_strategy="no",
                    save_strategy="no",
                    use_cpu=True,
                ),
                optimization_plan=NonportablePass(),
            )
            trainer._ensure_model_loaded()
            trainer._move_model_to_device()
            portable_directory = Path(directory) / "portable"
            with self.assertRaisesRegex(
                    OptimizationError,
                    "does not declare a canonical portable export",
            ):
                trainer.save_model(portable_directory)
            self.assertFalse(portable_directory.exists())

    def test_portable_export_roundtrips_and_strategy_wraps_after_pass(self):
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("PyTorch is required for Trainer optimization")

        events = []

        class TinyModel(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.projection = torch.nn.Linear(1, 1)

            def forward(self, input_values, labels):
                prediction = self.projection(input_values)
                return {
                    "loss": torch.nn.functional.mse_loss(
                        prediction,
                        labels,
                    ),
                    "logits": prediction,
                }

        class TransparentRuntime(torch.nn.Module):

            def __init__(self, canonical):
                super().__init__()
                self.canonical = canonical

            def forward(self, **inputs):
                return self.canonical(**inputs)

        class PortablePass(OptimizationPass):
            pass_id = "test.portable-topology"
            pass_version = "1"
            capabilities = OptimizationCapabilities(
                modes=(OptimizationMode.TRAINING, ),
                persistent=True,
                changes_parameter_names=True,
                portable_export=True,
            )

            def manifest_configuration(self):
                return {"wrapper": "transparent"}

            def apply(self, model, context):
                events.append("pass")
                return PassResult(model=TransparentRuntime(model))

            def export_portable_state(self, model, context):
                return model.canonical.state_dict()

        class StrategyProxy:

            def __init__(self, runtime):
                self.runtime = runtime
                self.calls = []

            def __call__(self, **inputs):
                self.calls.append("forward")
                return self.runtime(**inputs)

            def train(self, mode=True):
                self.runtime.train(mode)

            def eval(self):
                self.runtime.eval()

        class ProxyStrategy(TorchTrainingStrategy):

            def __init__(self):
                super().__init__()
                self.proxy = None

            def prepare_device(self, model, *, device):
                events.append("device")
                return super().prepare_device(model, device=device)

            def prepare_model(self, model, *, device):
                del device
                events.append("strategy")
                if not isinstance(model, TransparentRuntime):
                    raise AssertionError("Strategy must receive the transformed graph.")
                self.proxy = StrategyProxy(model)
                return self.proxy

            def unwrap_model(self, model):
                return model.runtime if isinstance(
                    model,
                    StrategyProxy,
                ) else model

        strategy = ProxyStrategy()
        dataset = [{
            "input_values": torch.tensor([1.0]),
            "labels": torch.tensor([2.0]),
        }]
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(
                model=TinyModel(),
                args=TrainingArguments(
                    output_dir=directory,
                    max_steps=1,
                    per_device_train_batch_size=1,
                    per_device_eval_batch_size=1,
                    logging_strategy="no",
                    save_strategy="no",
                    eval_strategy="steps",
                    eval_steps=1,
                    use_cpu=True,
                ),
                train_dataset=dataset,
                eval_dataset=dataset,
                optimization_plan=PortablePass(),
                training_strategy=strategy,
            )
            trainer.train()
            portable_directory = trainer.save_model(Path(directory) / "portable")
            state = torch.load(
                portable_directory / MODEL_STATE_NAME,
                map_location="cpu",
                weights_only=True,
            )
            fresh = TinyModel()
            fresh.load_state_dict(state)
            optimized = trainer.optimization_result.model
            sample = torch.tensor([[0.25]])
            labels = torch.tensor([[0.0]])
            optimized_output = optimized(
                input_values=sample,
                labels=labels,
            )["logits"]
            fresh_output = fresh(
                input_values=sample,
                labels=labels,
            )["logits"]

        self.assertEqual(events[:3], ["device", "pass", "strategy"])
        self.assertGreaterEqual(len(strategy.proxy.calls), 2)
        self.assertTrue(torch.equal(optimized_output, fresh_output))


if __name__ == "__main__":
    unittest.main()
