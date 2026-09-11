from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from unittest import mock

import torch
from torch import nn

from voicehub.auto import AutoConfig, AutoModelForTextToSpeech
from voicehub.configuration import VoiceHubConfig
from voicehub.models.base import BaseTTSModel
from voicehub.models.tts import PreTrainedTTSModel
from voicehub.optimization import (
    OptimizationCapabilities,
    OptimizationCompatibilityError,
    OptimizationContext,
    OptimizationMode,
    OptimizationPass,
    OptimizationPassManager,
    OptimizationPassRegistry,
    PassResult,
    TorchCompileCapabilityReport,
    TorchCompilePass,
    TTSOptimizationCompatibilityError,
    TTSOptimizationConfig,
    get_tts_optimization_config,
    get_tts_optimization_support,
    list_tts_optimization_support,
    resolve_tts_optimization,
)
from voicehub.outputs import TTSOutput
from voicehub.registry import list_model_specs
from voicehub.tasks import SpeechTask
from voicehub.training.arguments import TrainingArguments
from voicehub.training.trainer import Trainer
from voicehub.training.utils import OPTIMIZATION_MANIFEST_NAME


def _context(
    mode: str,
    *,
    device: str = "cpu",
    dtype: str = "float32",
) -> OptimizationContext:
    return OptimizationContext(
        mode=mode,
        device=device,
        dtype=dtype,
        persist_result=mode == "training",
    )


class _TinyTTSModel(BaseTTSModel):

    def __init__(self):
        super().__init__(device="cpu")
        self.config = SimpleNamespace(model_type="dia")
        self.model = None
        self.load_calls = 0
        self.training_load_calls = 0

    def load(self):
        self.load_calls += 1
        if self.model is None:
            self.model = nn.Linear(2, 2)
        return self

    def load_for_training(self):
        self.training_load_calls += 1
        if self.model is None:
            self.model = nn.Linear(2, 2)
        return self

    def __call__(self, value):
        self.load()
        return self.model(value)

    @property
    def sample_rate(self) -> int:
        return 24_000


class _PluginCompilePass(OptimizationPass):
    pass_id = "test.plugin-compile"
    pass_version = "1"
    optimization_kind = "compile"
    capabilities = OptimizationCapabilities(
        modes=(OptimizationMode.INFERENCE, OptimizationMode.TRAINING),
        streaming_safe=True,
        distributed_safe=True,
        persistent=True,
        reversible=True,
    )

    def manifest_configuration(self):
        return {"provider": "test-plugin"}

    def apply(self, model, context):
        del context
        return PassResult(
            model=model,
            state={"model": model},
            metadata={"provider": "test-plugin"},
        )

    def restore(self, model, state, context):
        del model, context
        return state["model"]


class _NewOptimizationKindPass(_PluginCompilePass):
    pass_id = "test.new-optimization-kind"
    optimization_kind = "future-optimization"


class _NestedManifestPluginPass(_PluginCompilePass):
    pass_id = "test.nested-plugin-compile"

    def manifest_configuration(self):
        return MappingProxyType({
            "nested": MappingProxyType({
                "value": 1,
            }),
        })


class _InvalidManifestPluginPass(_PluginCompilePass):
    pass_id = "test.invalid-plugin-compile"

    def manifest_configuration(self):
        return {
            "nested": {
                1: "invalid",
            },
        }


class _CountingManifestPluginPass(_PluginCompilePass):
    pass_id = "test.counting-plugin-compile"

    def __init__(self):
        self.manifest_calls = 0

    def manifest_configuration(self):
        self.manifest_calls += 1
        return {
            "snapshot": self.manifest_calls,
        }


class _MarkerCompilePass(OptimizationPass):
    pass_id = "test.marker-compile"
    pass_version = "1"
    optimization_kind = "compile"
    capabilities = OptimizationCapabilities(
        modes=(OptimizationMode.INFERENCE, ),
        reversible=True,
    )

    def manifest_configuration(self):
        return {}

    def apply(self, model, context):
        del context
        model.voicehub_test_marker = "optimized"
        return PassResult(model=model, state={"model": model})

    def restore(self, model, state, context):
        del context
        delattr(model, "voicehub_test_marker")
        return state["model"]


class _TrainerModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="dia")
        self.projection = nn.Linear(2, 1)

    def forward(self, input_values, labels=None):
        prediction = self.projection(input_values)
        loss = prediction.square().mean()
        if labels is not None:
            loss = (prediction - labels).square().mean()
        return {"loss": loss, "logits": prediction}


class _TinyPretrainedConfig(VoiceHubConfig):
    model_type = "dia"


class _TinyPretrainedTTS(PreTrainedTTSModel):
    config_class = _TinyPretrainedConfig

    def __init__(self, config, **kwargs):
        self.load_calls = 0
        super().__init__(config, **kwargs)

    def _load_pretrained_model(self):
        self.load_calls += 1
        self.model = nn.Linear(2, 2)

    def _generate(self, text, **kwargs):
        del text, kwargs
        audio = self.model(torch.ones(1, 2)).detach().reshape(-1)
        return TTSOutput(audio=audio, sample_rate=self.sample_rate)


class _ArchitectureShapedTarget:

    def __init__(self, model_type: str):
        self.config = SimpleNamespace(model_type=model_type)

    def _generate(self, text, **kwargs):
        del text, kwargs


class UniversalTTSOptimizationTests(unittest.TestCase):

    def test_every_registered_tts_model_resolves_for_inference_and_training(self, ):
        specs = list_model_specs(task=SpeechTask.TEXT_TO_SPEECH)
        self.assertEqual(len(specs), 34)
        self.assertEqual(len({spec.model_type for spec in specs}), 34)

        support = list_tts_optimization_support()
        self.assertEqual(
            [item.model_type for item in support],
            [spec.model_type for spec in specs],
        )

        for mode in ("inference", "training"):
            for spec in specs:
                with self.subTest(mode=mode, model_type=spec.model_type):
                    plan = resolve_tts_optimization(
                        spec.model_type,
                        context=_context(mode),
                        mode=mode,
                    )
                    self.assertEqual(plan.model_type, spec.model_type)
                    expected_architecture = spec.architecture.replace(
                        "_",
                        "-",
                    )
                    self.assertEqual(
                        plan.architecture,
                        expected_architecture,
                    )
                    self.assertEqual(
                        plan.context.architecture,
                        expected_architecture,
                    )
                    self.assertEqual(
                        plan.context.persist_result,
                        mode == "training",
                    )
                    self.assertTrue(plan.support.compile)
                    self.assertIn(
                        "compile",
                        plan.support.optimization_kinds,
                    )
                    self.assertEqual(
                        [decision.feature for decision in plan.decisions[:2]],
                        ["kernels", "attention"],
                    )
                    self.assertEqual(
                        plan.decisions[-1].feature,
                        "compile",
                    )
                    if mode == "inference":
                        self.assertNotIn(
                            "torch.compile",
                            tuple(item.pass_id for item in plan.passes),
                        )
                        self.assertEqual(
                            plan.decisions[-1].selected,
                            "eager",
                        )
                        expected_reason = (
                            "real-checkpoint validation" if spec.model_type in {"neutts", "vits", "vui"} else
                            "retained real-checkpoint validation")
                        self.assertIn(expected_reason, plan.decisions[-1].reason)
                    else:
                        self.assertTrue(plan.passes)
                        self.assertEqual(
                            plan.passes[-1].compatibility_kind,
                            "compile",
                        )
                    manifest = plan.manifest()
                    self.assertEqual(
                        manifest,
                        json.loads(json.dumps(
                            manifest,
                            allow_nan=False,
                            sort_keys=True,
                        )),
                    )

    def test_every_tts_plan_defers_compile_target_discovery_to_runtime(self):
        for spec in list_model_specs(task=SpeechTask.TEXT_TO_SPEECH):
            with self.subTest(model_type=spec.model_type):
                target = _ArchitectureShapedTarget(spec.model_type)
                if spec.model_type in {"neutts", "vits", "vui"}:
                    plan = resolve_tts_optimization(
                        target,
                        context=_context("inference"),
                    )
                    self.assertNotIn(
                        "torch.compile",
                        tuple(item.pass_id for item in plan.passes),
                    )
                else:
                    plan = resolve_tts_optimization(
                        target,
                        TTSOptimizationConfig(
                            attn_implementation="native",
                            kernel_backend="native",
                            compile="required",
                        ),
                        context=_context("inference"),
                    )
                    configuration = (plan.passes[-1].manifest_configuration())
                    self.assertIsNone(configuration["execution_targets"])

    def test_default_specialization_matrix_is_semantics_driven(self):
        custom_kernel_models = {
            "conversationtts",
            "cosyvoice",
            "echo",
            "f5tts",
            "gptsovits",
            "inflecttts",
            "irodoritts",
            "melotts",
            "openvoice",
            "qwen3tts",
            "vibevoice",
            "vits",
        }
        flash_attention_models = set()
        specs = list_model_specs(task=SpeechTask.TEXT_TO_SPEECH)

        observed_custom = set()
        observed_flash = set()
        for spec in specs:
            plan = resolve_tts_optimization(
                spec.model_type,
                context=_context("inference"),
            )
            pass_ids = [item.pass_id for item in plan.passes]
            self.assertNotIn("torch.compile", pass_ids)
            self.assertEqual(plan.decisions[-1].selected, "eager")
            if "custom-kernels" in pass_ids:
                observed_custom.add(spec.model_type)
                self.assertEqual(
                    plan.decisions[0].selected,
                    "torch",
                )
            if "flash-attention-4" in pass_ids:
                observed_flash.add(spec.model_type)

        self.assertEqual(observed_custom, custom_kernel_models)
        self.assertEqual(observed_flash, flash_attention_models)

        vits = get_tts_optimization_support("vits")
        self.assertEqual(vits.attention_implementations, ("native", ))
        self.assertIn("triton", vits.kernel_backends)
        for model_type in flash_attention_models:
            with self.subTest(model_type=model_type):
                support = get_tts_optimization_support(model_type)
                self.assertIn(
                    "flash_attention_4",
                    support.attention_implementations,
                )
                self.assertIn("triton", support.kernel_backends)

    def test_explicit_incompatible_requests_fail_during_resolution(self):
        with self.assertRaisesRegex(
                TTSOptimizationCompatibilityError,
                "attention-backend protocol",
        ):
            resolve_tts_optimization(
                "vits",
                TTSOptimizationConfig(
                    attn_implementation="flash_attention_4",
                    kernel_backend="native",
                    compile=False,
                ),
                context=_context(
                    "inference",
                    device="cuda",
                    dtype="float16",
                ),
            )

        with self.assertRaisesRegex(
                TTSOptimizationCompatibilityError,
                "does not declare an SDPA-compatible",
        ):
            resolve_tts_optimization(
                "vits",
                TTSOptimizationConfig(
                    attn_implementation="sdpa",
                    kernel_backend="native",
                    compile=False,
                ),
                context=_context("inference"),
            )

        with self.assertRaisesRegex(
                TTSOptimizationCompatibilityError,
                "requires CUDA",
        ):
            resolve_tts_optimization(
                "qwen3tts",
                TTSOptimizationConfig(
                    attn_implementation="native",
                    kernel_backend="triton",
                    compile=False,
                ),
                context=_context("inference"),
            )

        with self.assertRaisesRegex(
                TTSOptimizationCompatibilityError,
                "device 'mps'",
        ):
            resolve_tts_optimization(
                "dia",
                TTSOptimizationConfig(
                    attn_implementation="native",
                    kernel_backend="native",
                    compile=True,
                ),
                context=_context("inference", device="mps"),
            )

    def test_vui_rejects_inference_compile_and_preserves_training_compile(self):
        for dynamic in (None, False, True):
            with (
                    self.subTest(dynamic=dynamic),
                    self.assertRaisesRegex(
                        TTSOptimizationCompatibilityError,
                        r"vui.*inference.*real-checkpoint",
                    ),
            ):
                get_tts_optimization_config(
                    "vui",
                    compile=True,
                    compile_config={
                        "backend": "eager",
                        "dynamic": dynamic,
                    },
                )

        automatic = resolve_tts_optimization(
            "vui",
            TTSOptimizationConfig(),
            context=_context("inference"),
        )
        self.assertEqual(automatic.passes, ())
        self.assertEqual(automatic.decisions[-1].selected, "eager")
        self.assertIn(
            "real-checkpoint validation",
            automatic.decisions[-1].reason,
        )

        training = resolve_tts_optimization(
            "vui",
            TTSOptimizationConfig(
                compile=True,
                compile_config={
                    "backend": "eager",
                    "dynamic": None,
                },
            ),
            mode="training",
            context=_context("training"),
        )
        self.assertEqual(
            training.passes[-1].config.dynamic,
            None,
        )
        self.assertEqual(training.decisions[-1].selected, "torch.compile:eager")

        disabled = get_tts_optimization_config(
            "vui",
            compile=False,
            compile_config={
                "dynamic": True,
            },
        )
        self.assertEqual(disabled.compile.value, "disabled")

    def test_mps_auto_policy_produces_an_inspectable_eager_fallback(self):
        plan = resolve_tts_optimization(
            "dia",
            TTSOptimizationConfig(),
            context=_context("inference", device="mps"),
        )

        self.assertEqual(plan.passes, ())
        decisions = {decision.feature: decision for decision in plan.decisions}
        self.assertEqual(decisions["attention"].selected, "native")
        self.assertEqual(decisions["kernels"].selected, "torch")
        self.assertEqual(decisions["compile"].selected, "eager")
        self.assertIn(
            "retained real-checkpoint validation",
            decisions["compile"].reason,
        )
        self.assertEqual(
            plan.manifest()["passes"],
            [],
        )

    def test_dtype_aliases_are_canonical_across_context_and_pass_checks(self):
        aliases = {
            "bf16": "bfloat16",
            "fp16": "float16",
            "half": "float16",
            "torch.float16": "float16",
            "fp32": "float32",
            "double": "float64",
        }
        for alias, expected in aliases.items():
            with self.subTest(alias=alias):
                context = _context(
                    "inference",
                    device="cuda",
                    dtype=alias,
                )
                self.assertEqual(context.dtype, expected)

        plan = resolve_tts_optimization(
            "qwen3tts",
            TTSOptimizationConfig(
                attn_implementation="flash_attention_4",
                kernel_backend="native",
                compile=False,
            ),
            context=_context(
                "inference",
                device="cuda",
                dtype="bf16",
            ),
        )
        self.assertEqual(plan.context.dtype, "bfloat16")
        self.assertEqual(
            [item.pass_id for item in plan.passes],
            ["flash-attention-4"],
        )
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            OptimizationCapabilities(
                modes=("inference", ),
                dtypes=(),
            )
        with self.assertRaisesRegex(TypeError, "iterable of strings"):
            OptimizationCapabilities(
                modes=("inference", ),
                dtypes="float32",
            )
        with self.assertRaisesRegex(ValueError, "duplicates"):
            OptimizationCapabilities(
                modes=("inference", ),
                dtypes=("fp16", "float16"),
            )

    def test_config_and_plan_manifests_round_trip_through_strict_json(self):
        config = TTSOptimizationConfig(
            attn_implementation="fa4",
            kernel_backend="cuda-extension",
            compile=True,
            compile_config={
                "backend": "eager",
                "fullgraph": True,
                "dynamic": False,
                "options": {
                    "trace.enabled": False
                },
            },
        )
        serialized = config.to_json_string()
        payload = json.loads(serialized)
        restored = TTSOptimizationConfig.from_dict(payload)

        self.assertEqual(restored, config)
        self.assertEqual(json.loads(restored.to_json_string()), payload)
        self.assertEqual(payload["compile"], "required")
        self.assertEqual(
            payload["compile_config"]["requirement"],
            "required",
        )
        self.assertEqual(
            payload["attn_implementation"],
            "flash_attention_4",
        )
        self.assertEqual(payload["kernel_backend"], "cuda_extension")

        first = resolve_tts_optimization(
            "qwen3tts",
            mode="training",
            context=_context("training"),
        ).manifest()
        second = resolve_tts_optimization(
            "qwen3tts",
            mode="training",
            context=_context("training"),
        ).manifest()
        self.assertEqual(first, second)
        self.assertEqual(
            first,
            json.loads(json.dumps(
                first,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )),
        )
        with self.assertRaisesRegex(ValueError, "strict JSON"):
            TTSOptimizationConfig(
                compile_config={
                    "options": {
                        "invalid": object(),
                    },
                }, )
        with self.assertRaisesRegex(ValueError, "strict JSON"):
            TTSOptimizationConfig(
                compile_config={
                    "options": {
                        "nested": {
                            1: "invalid",
                        },
                    },
                }, )
        with self.assertRaisesRegex(ValueError, "strict JSON"):
            TTSOptimizationConfig(
                compile_config={
                    "options": {
                        "invalid": float("nan"),
                    },
                }, )

    def test_resolved_plan_configuration_is_immutable(self):
        source_options = {
            "nested": {
                "value": 1,
            },
        }
        config = TTSOptimizationConfig(
            attn_implementation="native",
            kernel_backend="native",
            compile=True,
            compile_config={
                "backend": "eager",
                "options": source_options,
            },
        )
        plan = resolve_tts_optimization(
            "dia",
            config,
            context=_context("inference"),
        )
        before = plan.manifest()
        source_options["nested"]["value"] = 2

        with self.assertRaises(TypeError):
            config.compile_config.options["nested"]["value"] = 3

        after = plan.manifest()
        self.assertEqual(after, before)
        self.assertEqual(
            after["config"]["compile_config"]["options"]["nested"]["value"],
            1,
        )
        self.assertEqual(
            after["passes"][0]["configuration"]["options"]["nested"]["value"],
            1,
        )

    def test_resolution_does_not_import_optional_acceleration_packages(self):
        code = """
import json
import sys
from voicehub.optimization import (
    OptimizationContext,
    TTSOptimizationConfig,
)

plan = TTSOptimizationConfig().resolve(
    "qwen3tts",
    context=OptimizationContext(
        mode="inference",
        device="cpu",
        dtype="float32",
    ),
)
optional = sorted(
    name for name in sys.modules
    if (
        name == "triton"
        or name.startswith("triton.")
        or name == "flash_attn"
        or name.startswith("flash_attn.")
        or name == "flash_attn_interface"
        or name.startswith("flash_attn_interface.")
        or name == "torch.utils.cpp_extension"
    )
)
print(json.dumps({
    "passes": [item.pass_id for item in plan.passes],
    "optional": optional,
}))
"""
        completed = subprocess.run(
            (sys.executable, "-c", code),
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            json.loads(completed.stdout),
            {
                "passes": [
                    "custom-kernels",
                ],
                "optional": [],
            },
        )

    def test_plugin_registry_extends_a_registered_architecture(self):
        registry = OptimizationPassRegistry()
        registry.register("plugin-compile", _PluginCompilePass)
        config = TTSOptimizationConfig(
            attn_implementation="native",
            kernel_backend="native",
            compile=False,
            optimization_passes=("plugin-compile", ),
        )

        plan = resolve_tts_optimization(
            "dia",
            config,
            mode="training",
            context=_context("training"),
            registry=registry,
        )

        self.assertEqual(
            [item.qualified_id for item in plan.passes],
            ["test.plugin-compile@1"],
        )
        self.assertEqual(
            plan.decisions[2].feature,
            "pass:plugin-compile",
        )
        self.assertEqual(plan.decisions[-1].selected, "eager")
        self.assertEqual(
            plan.manifest()["passes"][0]["kind"],
            "compile",
        )

    def test_new_plugin_kind_needs_no_architecture_catalog_edits(self):
        registry = OptimizationPassRegistry()
        registry.register("future-optimization", _NewOptimizationKindPass)
        config = TTSOptimizationConfig(
            attn_implementation="native",
            kernel_backend="native",
            compile=False,
            optimization_passes=("future-optimization", ),
        )

        plan = resolve_tts_optimization(
            "dia",
            config,
            mode="training",
            context=_context("training"),
            registry=registry,
        )

        self.assertNotIn(
            "future-optimization",
            plan.support.optimization_kinds,
        )
        self.assertEqual(
            plan.manifest()["passes"][0]["kind"],
            "future-optimization",
        )

    def test_plugin_manifests_use_the_generic_strict_json_contract(self):
        valid_registry = OptimizationPassRegistry()
        valid_registry.register(
            "nested-plugin",
            _NestedManifestPluginPass,
        )
        config = TTSOptimizationConfig(
            attn_implementation="native",
            kernel_backend="native",
            compile=False,
            optimization_passes=("nested-plugin", ),
        )
        plan = resolve_tts_optimization(
            "dia",
            config,
            mode="training",
            context=_context("training"),
            registry=valid_registry,
        )
        self.assertEqual(
            plan.manifest()["passes"][0]["configuration"],
            {
                "nested": {
                    "value": 1,
                },
            },
        )

        invalid_registry = OptimizationPassRegistry()
        invalid_registry.register(
            "invalid-plugin",
            _InvalidManifestPluginPass,
        )
        invalid_config = TTSOptimizationConfig(
            attn_implementation="native",
            kernel_backend="native",
            compile=False,
            optimization_passes=("invalid-plugin", ),
        )
        with self.assertRaisesRegex(TypeError, "non-string mapping key"):
            resolve_tts_optimization(
                "dia",
                invalid_config,
                mode="training",
                context=_context("training"),
                registry=invalid_registry,
            )

    def test_resolved_pass_declarations_are_snapshotted_once(self):
        optimization_pass = _CountingManifestPluginPass()
        registry = OptimizationPassRegistry()
        registry.register(
            "counting-plugin",
            lambda: optimization_pass,
        )
        config = TTSOptimizationConfig(
            attn_implementation="native",
            kernel_backend="native",
            compile=False,
            optimization_passes=("counting-plugin", ),
        )
        plan = resolve_tts_optimization(
            "dia",
            config,
            mode="training",
            context=_context("training"),
            registry=registry,
        )

        first = plan.manifest()
        second = plan.manifest()
        manager = OptimizationPassManager()
        application = manager.apply(
            nn.Linear(2, 2),
            plan.passes,
            plan.context,
            declaration_snapshots=plan.pass_declaration_snapshots,
        )

        self.assertEqual(optimization_pass.manifest_calls, 1)
        self.assertEqual(first, second)
        self.assertEqual(
            first["passes"][0]["configuration"],
            {
                "snapshot": 1,
            },
        )
        self.assertEqual(
            application.manifest()["passes"][0]["configuration"],
            {
                "snapshot": 1,
            },
        )

    def test_config_helper_validates_explicit_architecture_choices(self):
        with self.assertRaisesRegex(
                TTSOptimizationCompatibilityError,
                "attention-backend protocol",
        ):
            get_tts_optimization_config(
                "vits",
                attn_implementation="flash_attention_4",
            )

    def test_wrapper_native_fallback_lifecycle_does_not_load_on_resolve(self):
        wrapper = _TinyTTSModel()
        config = TTSOptimizationConfig(
            attn_implementation="native",
            kernel_backend="native",
            compile=False,
        )

        plan = wrapper.resolve_optimization(config)
        self.assertEqual(wrapper.load_calls, 0)
        self.assertEqual(plan.passes, ())

        result = wrapper.optimize(config)
        runtime = wrapper.model
        self.assertFalse(result.optimized)
        self.assertIs(result.model, runtime)
        self.assertIs(
            wrapper.tts_optimization_result(mode="inference"),
            result,
        )
        manifest = wrapper.tts_optimization_manifest(mode="inference")
        self.assertIsNone(manifest["application"])
        self.assertEqual(
            manifest["resolution"]["passes"],
            [],
        )

        self.assertIs(
            wrapper.restore_tts_optimization(mode="inference"),
            runtime,
        )
        self.assertIsNone(wrapper.tts_optimization_result(mode="inference"), )

    def test_native_and_explicit_same_mode_lifecycles_are_mutually_exclusive(self):
        native_config = TTSOptimizationConfig(
            attn_implementation="native",
            kernel_backend="native",
            compile=False,
        )
        wrapper = _TinyTTSModel()
        wrapper.optimize(native_config)

        with self.assertRaisesRegex(RuntimeError, "universal.*policy"):
            wrapper.apply_optimization_plan(
                _MarkerCompilePass(),
                mode="inference",
            )
        self.assertFalse(hasattr(wrapper.model, "voicehub_test_marker"), )
        wrapper.restore_tts_optimization()

        explicit = wrapper.apply_optimization_plan(
            _MarkerCompilePass(),
            mode="inference",
        )
        self.assertEqual(
            wrapper.model.voicehub_test_marker,
            "optimized",
        )
        with self.assertRaisesRegex(RuntimeError, "explicit.*plan"):
            wrapper.optimize(native_config)
        self.assertIs(
            wrapper.optimization_result(mode="inference"),
            explicit,
        )
        wrapper.restore_optimization_plan(mode="inference")
        self.assertFalse(hasattr(wrapper.model, "voicehub_test_marker"), )

    def test_wrapper_applies_and_restores_a_required_compile_policy(self):
        wrapper = _TinyTTSModel()
        wrapper.load()
        runtime = wrapper.model
        original_keys = tuple(runtime.state_dict())
        report = TorchCompileCapabilityReport(
            available=True,
            backend="eager",
            backend_available=True,
            torch_version=torch.__version__,
            available_backends=("eager", ),
        )
        config = TTSOptimizationConfig(
            attn_implementation="native",
            kernel_backend="native",
            compile=True,
            compile_config={"backend": "eager"},
        )

        with (
                mock.patch(
                    "voicehub.optimization.torch_compile."
                    "inspect_torch_compile",
                    return_value=report,
                ),
                mock.patch.object(
                    torch,
                    "compile",
                    side_effect=lambda function, **kwargs: function,
                ) as compile_mock,
        ):
            result = wrapper.optimize(config)

        self.assertTrue(result.optimized)
        self.assertIs(wrapper.model, runtime)
        self.assertIn("forward", runtime.__dict__)
        self.assertEqual(tuple(runtime.state_dict()), original_keys)
        self.assertEqual(compile_mock.call_count, 1)
        self.assertEqual(
            result.plan.passes[-1].pass_id,
            "torch.compile",
        )
        self.assertEqual(
            result.manifest()["application"]["passes"][0]["kind"],
            "compile",
        )

        self.assertIs(wrapper.restore_tts_optimization(), runtime)
        self.assertNotIn("forward", runtime.__dict__)
        self.assertEqual(tuple(runtime.state_dict()), original_keys)

    def test_training_policy_targets_architecture_adapter_not_inference_runtime(self, ):

        class InferenceRuntime(nn.Module):

            def __init__(self):
                super().__init__()
                self.projection = nn.Linear(2, 2)

            def forward(self, value):
                return self.projection(value)

        class TrainingComposite:

            def __init__(self, wrapper):
                self.model = wrapper
                self.primary_model = nn.Linear(2, 1)
                self._components = [
                    ("training_graph", self.primary_model),
                ]
                self.build_calls = 0

            def build_training_graph(self):
                self.build_calls += 1
                return self

            def state_dict(self):
                return {
                    f"primary_model.{name}": value
                    for name, value in self.primary_model.state_dict().items()
                }

        class AdapterBackedTTS(_TinyPretrainedTTS):

            def __init__(self, config, **kwargs):
                self.training_composite = None
                super().__init__(config, **kwargs)

            def _load_pretrained_model(self):
                self.load_calls += 1
                self.model = InferenceRuntime()

            def get_training_adapter(self):
                if self.training_composite is None:
                    self.training_composite = TrainingComposite(self)
                return self.training_composite

        report = TorchCompileCapabilityReport(
            available=True,
            backend="eager",
            backend_available=True,
            torch_version=torch.__version__,
            available_backends=("eager", ),
        )
        compiled_owners = []

        def compile_method(function, **kwargs):
            del kwargs
            compiled_owners.append(function.__self__)
            return function

        wrapper = AdapterBackedTTS(
            _TinyPretrainedConfig(name_or_path=""),
            device="cpu",
        )
        config = TTSOptimizationConfig(
            attn_implementation="native",
            kernel_backend="native",
            compile=True,
            compile_config={"backend": "eager"},
        )

        with (
                mock.patch(
                    "voicehub.optimization.torch_compile."
                    "inspect_torch_compile",
                    return_value=report,
                ),
                mock.patch.object(
                    torch,
                    "compile",
                    side_effect=compile_method,
                ),
        ):
            result = wrapper.optimize(
                config,
                mode="training",
                context=_context("training"),
            )

        inference_runtime = wrapper.model
        adapter = wrapper.training_composite
        self.assertIsNotNone(adapter)
        self.assertIs(result.model, adapter)
        self.assertIs(result.application.model, adapter)
        self.assertEqual(adapter.build_calls, 1)
        self.assertEqual(compiled_owners, [adapter.primary_model])
        self.assertNotIn("forward", inference_runtime.__dict__)
        self.assertIn("forward", adapter.primary_model.__dict__)
        self.assertEqual(
            result.manifest()["application"]["passes"][0]["metadata"]["execution_targets"],
            ["component:primary_model"],
        )
        self.assertEqual(
            tuple(adapter.primary_model(torch.ones(1, 2)).shape),
            (1, 1),
        )

        self.assertIs(
            wrapper.restore_tts_optimization(mode="training"),
            adapter,
        )
        self.assertIs(wrapper.model, inference_runtime)
        self.assertNotIn("forward", adapter.primary_model.__dict__)
        self.assertIsNone(wrapper.optimization_result(mode="training"))
        self.assertIsNone(wrapper.tts_optimization_result(mode="training"))

        compiled_owners.clear()
        explicit_pass = TorchCompilePass(
            backend="eager",
            requirement="required",
        )
        with (
                mock.patch(
                    "voicehub.optimization.torch_compile."
                    "inspect_torch_compile",
                    return_value=report,
                ),
                mock.patch.object(
                    torch,
                    "compile",
                    side_effect=compile_method,
                ),
        ):
            explicit = wrapper.apply_optimization_plan(
                explicit_pass,
                mode="training",
                context=_context("training"),
            )
        self.assertIs(explicit.model, inference_runtime)
        self.assertEqual(compiled_owners, [inference_runtime])
        self.assertIn("forward", inference_runtime.__dict__)
        self.assertNotIn("forward", adapter.primary_model.__dict__)
        wrapper.restore_optimization_plan(mode="training")
        self.assertNotIn("forward", inference_runtime.__dict__)

    def test_trainer_keeps_native_fallback_local_and_reports_noop(self):
        model = _TrainerModel()
        source_config = {
            "attn_implementation": "native",
            "kernel_backend": "native",
            "compile": False,
        }
        unchanged = dict(source_config)
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(
                model=model,
                args=TrainingArguments(
                    output_dir=directory,
                    use_cpu=True,
                    report_to=[],
                ),
                optimization_config=source_config,
            )
            trainer._move_model_to_device()

        self.assertEqual(source_config, unchanged)
        self.assertIs(trainer.model, model)
        self.assertIs(trainer.model_wrapped, model)
        self.assertIsNone(trainer.optimization_result)
        self.assertIsNotNone(trainer.tts_optimization_plan)
        self.assertEqual(trainer.tts_optimization_plan.passes, ())
        manifest = trainer.optimization_manifest()
        self.assertEqual(manifest["kind"], "tts-optimization")
        self.assertIsNone(manifest["application"])
        self.assertEqual(manifest["resolution"]["passes"], [])
        self.assertEqual(
            manifest,
            json.loads(json.dumps(manifest, allow_nan=False)),
        )
        self.assertFalse(hasattr(model, "_tts_optimization_results_by_mode"), )

    def test_from_pretrained_schedules_without_leaking_config_fields(self):
        model = _TinyPretrainedTTS.from_pretrained(
            "",
            attn_implementation="native",
            lazy_load=True,
        )

        self.assertFalse(model.is_loaded)
        self.assertNotIn("optimization_config", model.config.__dict__)
        self.assertNotIn("attn_implementation", model.config.__dict__)
        pending = model._pending_tts_optimization_config
        self.assertEqual(pending.attn_implementation.value, "native")
        self.assertEqual(pending.kernel_backend.value, "native")
        self.assertEqual(pending.compile.value, "disabled")

        model.load()

        self.assertEqual(model.load_calls, 1)
        self.assertIsNone(model._pending_tts_optimization_config)
        result = model.tts_optimization_result(mode="inference")
        self.assertIsNotNone(result)
        self.assertFalse(result.optimized)
        self.assertEqual(result.plan.passes, ())

    def test_from_pretrained_schedules_direct_diffusion_cache_options(self):
        model = _TinyPretrainedTTS.from_pretrained(
            "",
            diffusion_cache="auto",
            diffusion_cache_config={
                "front_blocks": 2,
                "back_blocks": 1,
                "predictor": "taylorseer",
            },
            lazy_load=True,
        )

        self.assertFalse(model.is_loaded)
        self.assertNotIn("diffusion_cache", model.config.__dict__)
        self.assertNotIn("diffusion_cache_config", model.config.__dict__)
        pending = model._pending_tts_optimization_config
        self.assertEqual(pending.diffusion_cache.value, "auto")
        self.assertEqual(pending.diffusion_cache_config.front_blocks, 2)
        self.assertEqual(pending.diffusion_cache_config.back_blocks, 1)
        self.assertEqual(
            pending.diffusion_cache_config.predictor.value,
            "taylor",
        )
        self.assertEqual(pending.compile.value, "disabled")

    def test_auto_factory_forwards_direct_diffusion_cache_options(self):
        config = AutoConfig.for_model("f5tts")

        model = AutoModelForTextToSpeech.from_config(
            config,
            diffusion_cache=True,
            diffusion_cache_config={
                "front_blocks": 3,
                "residual_diff_threshold": 0.05,
            },
        )

        self.assertFalse(model.is_loaded)
        pending = model._pending_tts_optimization_config
        self.assertEqual(pending.diffusion_cache.value, "required")
        self.assertEqual(pending.diffusion_cache_config.front_blocks, 3)
        self.assertEqual(
            pending.diffusion_cache_config.residual_diff_threshold,
            0.05,
        )
        self.assertEqual(pending.compile.value, "disabled")

    def test_from_pretrained_schedules_direct_diffusion_sampling_options(self):
        model = _TinyPretrainedTTS.from_pretrained(
            "",
            diffusion_sampling="auto",
            diffusion_sampling_config={
                "target_steps": 20,
                "prediction_cache": "fora",
            },
            lazy_load=True,
        )

        self.assertFalse(model.is_loaded)
        self.assertNotIn("diffusion_sampling", model.config.__dict__)
        self.assertNotIn("diffusion_sampling_config", model.config.__dict__)
        pending = model._pending_tts_optimization_config
        self.assertEqual(pending.diffusion_sampling.value, "auto")
        self.assertEqual(
            pending.diffusion_sampling_config.target_steps,
            20,
        )
        self.assertEqual(
            pending.diffusion_sampling_config.prediction_cache.value,
            "fora",
        )
        self.assertEqual(pending.compile.value, "disabled")

    def test_auto_factory_forwards_direct_diffusion_sampling_options(self):
        config = AutoConfig.for_model("f5tts")

        model = AutoModelForTextToSpeech.from_config(
            config,
            diffusion_sampling=True,
            diffusion_sampling_config={
                "target_steps": 16,
                "solver": "stork2",
            },
        )

        self.assertFalse(model.is_loaded)
        pending = model._pending_tts_optimization_config
        self.assertEqual(pending.diffusion_sampling.value, "required")
        self.assertEqual(
            pending.diffusion_sampling_config.target_steps,
            16,
        )
        self.assertEqual(
            pending.diffusion_sampling_config.solver.value,
            "stork2",
        )
        self.assertEqual(pending.compile.value, "disabled")

    def test_scheduled_compile_discovers_and_runs_loaded_runtime_boundary(self):
        report = TorchCompileCapabilityReport(
            available=True,
            backend="eager",
            backend_available=True,
            torch_version=torch.__version__,
            available_backends=("eager", ),
        )
        compiled_calls = []

        def compile_method(function, **kwargs):
            del kwargs

            def compiled(*args, **call_kwargs):
                compiled_calls.append(function.__name__)
                return function(*args, **call_kwargs)

            return compiled

        model = _TinyPretrainedTTS.from_pretrained(
            "",
            optimization_config={
                "attn_implementation": "native",
                "kernel_backend": "native",
                "compile": True,
                "compile_config": {
                    "backend": "eager",
                },
            },
            device="cpu",
            lazy_load=True,
        )
        self.assertFalse(model.is_loaded)
        self.assertEqual(model.resolve_optimization().passes, ())
        self.assertEqual(
            model._pending_tts_optimization_config.compile.value,
            "required",
        )

        with (
                mock.patch(
                    "voicehub.optimization.torch_compile."
                    "inspect_torch_compile",
                    return_value=report,
                ),
                mock.patch.object(
                    torch,
                    "compile",
                    side_effect=compile_method,
                ),
        ):
            output = model.generate("hello")

        self.assertTrue(output.audio.numel())
        self.assertEqual(compiled_calls, ["forward"])
        self.assertNotIn("_generate", model.__dict__)
        self.assertIn("forward", model.model.__dict__)
        manifest = model.tts_optimization_manifest(mode="inference")
        self.assertEqual(
            manifest["application"]["passes"][0]["metadata"]["execution_targets"],
            ["forward"],
        )
        model.restore_tts_optimization()
        self.assertNotIn("forward", model.model.__dict__)

    def test_failed_pending_policy_can_be_cleared_without_reloading_weights(self):
        model = _TinyPretrainedTTS.from_pretrained(
            "",
            optimization_config={
                "attn_implementation": "native",
                "kernel_backend": "native",
                "compile": True,
                "compile_config": {
                    "backend": "voicehub-missing-test-backend",
                },
            },
            device="cpu",
            lazy_load=True,
        )

        with self.assertRaisesRegex(
                OptimizationCompatibilityError,
                "Required torch.compile",
        ):
            model.load()

        self.assertEqual(model.load_calls, 1)
        self.assertIsNotNone(model._pending_tts_optimization_config)
        cleared = model.clear_optimization_config()
        self.assertEqual(
            cleared.compile.value,
            "required",
        )

        model.load()

        self.assertEqual(model.load_calls, 1)
        self.assertTrue(model._inference_ready)
        self.assertIsNone(model.tts_optimization_result(mode="inference"))

    def test_pending_policy_rejects_other_optimization_entry_points_preload(self):
        model = _TinyPretrainedTTS.from_pretrained(
            "",
            optimization_config={
                "attn_implementation": "native",
                "kernel_backend": "native",
                "compile": False,
            },
            device="cpu",
            lazy_load=True,
        )
        pending = model._pending_tts_optimization_config

        with self.assertRaisesRegex(
                RuntimeError,
                "configuration is pending",
        ):
            model.apply_optimization_plan(
                _MarkerCompilePass(),
                mode="inference",
            )
        with self.assertRaisesRegex(
                RuntimeError,
                "configuration is pending",
        ):
            model.optimize(
                TTSOptimizationConfig(
                    attn_implementation="native",
                    kernel_backend="native",
                    compile=False,
                ))

        self.assertFalse(model.is_loaded)
        self.assertEqual(model.load_calls, 0)
        self.assertIs(
            model._pending_tts_optimization_config,
            pending,
        )
        self.assertIsNone(model.tts_optimization_result(mode="inference"), )
        self.assertIsNone(model.optimization_result(mode="inference"), )

    def test_trainer_checkpoint_records_and_compares_native_policy(self):
        dataset = [{
            "input_values": torch.ones(1, 2),
            "labels": torch.zeros(1, 1),
        }]
        with tempfile.TemporaryDirectory() as directory:
            common = {
                "output_dir": directory,
                "max_steps": 1,
                "per_device_train_batch_size": 1,
                "logging_strategy": "no",
                "use_cpu": True,
                "report_to": [],
            }
            trainer = Trainer(
                model=_TrainerModel(),
                args=TrainingArguments(
                    **common,
                    save_strategy="steps",
                    save_steps=1,
                ),
                train_dataset=dataset,
                optimization_config={
                    "attn_implementation": "native",
                    "kernel_backend": "native",
                    "compile": False,
                },
            )
            trainer.train()
            checkpoint = Path(directory) / "checkpoint-1"
            manifest = json.loads((checkpoint / OPTIMIZATION_MANIFEST_NAME).read_text(encoding="utf-8"))
            self.assertEqual(manifest["kind"], "tts-optimization")
            self.assertIsNone(manifest["application"])
            self.assertEqual(manifest["resolution"]["passes"], [])

            mismatched = Trainer(
                model=_TrainerModel(),
                args=TrainingArguments(
                    **common,
                    save_strategy="no",
                ),
                train_dataset=dataset,
                optimization_config={
                    "attn_implementation": "native",
                    "kernel_backend": "torch",
                    "compile": False,
                },
            )
            with self.assertRaisesRegex(
                    ValueError,
                    "optimization plan",
            ):
                mismatched.train(resume_from_checkpoint=str(checkpoint))

    def test_resume_identity_ignores_only_evolving_runtime_status(self):
        first = {
            "format_version":
            3,
            "passes": [{
                "pass": "plugin",
                "configuration": {
                    "passes": [{
                        "metadata": {
                            "algorithm": "a",
                        },
                    }],
                },
                "metadata": {
                    "provider": "stable",
                },
                "runtime_status": {
                    "outcome": "compiled",
                },
            }],
        }
        second = json.loads(json.dumps(first))
        second["passes"][0]["runtime_status"] = {
            "outcome": "eager-fallback",
        }
        self.assertEqual(
            Trainer._optimization_resume_identity(first),
            Trainer._optimization_resume_identity(second),
        )

        second["passes"][0]["configuration"]["passes"][0]["metadata"]["algorithm"] = "b"
        self.assertNotEqual(
            Trainer._optimization_resume_identity(first),
            Trainer._optimization_resume_identity(second),
        )


if __name__ == "__main__":
    unittest.main()
