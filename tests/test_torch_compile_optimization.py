from __future__ import annotations

import importlib.util
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from voicehub.optimization import (
    OPTIMIZATION_PASSES,
    CustomKernelPass,
    FlashAttention4Pass,
    OptimizationApplicationError,
    OptimizationCompatibilityError,
    OptimizationCompileTarget,
    OptimizationContext,
    OptimizationPassManager,
    TorchCompileCapabilityReport,
    TorchCompileConfig,
    TorchCompilePass,
    TorchCompileRuntimeError,
    TorchCompileUnavailableError,
    inspect_torch_compile,
)
from voicehub.training.arguments import TrainingArguments
from voicehub.training.trainer import Trainer

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch
    from torch import nn


@unittest.skipUnless(
    TORCH_AVAILABLE and callable(getattr(torch, "compile", None)),
    "torch.compile requires PyTorch 2 or newer",
)
class TorchCompileOptimizationTests(unittest.TestCase):

    @staticmethod
    def _context(mode: str = "inference", **overrides):
        values = {
            "mode": mode,
            "device": "cpu",
            "dtype": "float32",
            "persist_result": mode == "training",
        }
        values.update(overrides)
        return OptimizationContext(**values)

    def test_configuration_and_registered_default_are_explicit(self):
        config = TorchCompileConfig(
            backend="eager",
            mode="default",
            fullgraph=True,
            dynamic=True,
            requirement="required",
        )
        self.assertEqual(
            config.manifest(),
            {
                "backend": "eager",
                "mode": "default",
                "fullgraph": True,
                "dynamic": True,
                "options": None,
                "requirement": "required",
            },
        )
        self.assertIsInstance(
            OPTIMIZATION_PASSES.create("compile"),
            TorchCompilePass,
        )
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            TorchCompileConfig(mode="fastest")
        with self.assertRaisesRegex(TypeError, "dynamic"):
            TorchCompileConfig(dynamic=1)

    def test_capability_report_and_context_checks_fail_closed(self):
        report = inspect_torch_compile("eager")
        self.assertTrue(report.available, report.reason)
        self.assertIn("eager", report.available_backends)

        model = nn.Linear(2, 1)
        optimization_pass = TorchCompilePass(
            backend="eager",
            requirement="required",
        )
        with self.assertRaisesRegex(
                OptimizationCompatibilityError,
                "device 'mps'",
        ):
            OptimizationPassManager().apply(
                model,
                (optimization_pass, ),
                self._context(device="mps"),
            )
        with self.assertRaisesRegex(
                OptimizationCompatibilityError,
                "streaming",
        ):
            OptimizationPassManager().apply(
                model,
                (optimization_pass, ),
                self._context(streaming=True),
            )

        class StatelessCallable:

            def __call__(self, value):
                return value

        stateless = StatelessCallable()
        with self.assertRaisesRegex(
                OptimizationCompatibilityError,
                "state_dict",
        ):
            OptimizationPassManager().apply(
                stateless,
                (TorchCompilePass(
                    backend="eager",
                    requirement="required",
                ), ),
                self._context("training"),
            )
        fallback = OptimizationPassManager().apply(
            stateless,
            (TorchCompilePass(
                backend="eager",
                requirement="auto",
            ), ),
            self._context("training"),
        )
        self.assertIs(fallback.model, stateless)
        self.assertEqual(
            fallback.manifest_metadata()[0]["metadata"]["outcome"],
            "eager-fallback",
        )

    def test_module_compilation_preserves_identity_state_keys_and_gradients(self):
        model = nn.Sequential(
            nn.Linear(3, 4),
            nn.SiLU(),
            nn.Linear(4, 1),
        )
        original_keys = tuple(model.state_dict())
        result = OptimizationPassManager().apply(
            model,
            (TorchCompilePass(
                backend="eager",
                dynamic=True,
                requirement="required",
            ), ),
            self._context("training"),
        )

        self.assertIs(result.model, model)
        self.assertEqual(tuple(result.model.state_dict()), original_keys)
        self.assertFalse(any(name.startswith("_orig_mod.") for name in result.model.state_dict()))
        loss = result.model(torch.randn(2, 3)).square().mean()
        loss.backward()
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))
        self.assertEqual(
            tuple(result.portable_state_dict()),
            original_keys,
        )

        restored = result.restore()
        self.assertIs(restored, model)
        self.assertNotIn("forward", model.__dict__)
        self.assertEqual(tuple(restored.state_dict()), original_keys)

    def test_compile_settings_are_forwarded_without_global_configuration(self):
        model = nn.Linear(2, 1)

        def return_eager(function, **kwargs):
            del kwargs
            return function

        with patch.object(
                torch,
                "compile",
                side_effect=return_eager,
        ) as compile_mock:
            result = OptimizationPassManager().apply(
                model,
                (
                    TorchCompilePass(
                        backend="eager",
                        mode="reduce-overhead",
                        fullgraph=True,
                        dynamic=False,
                        requirement="required",
                    ), ),
                self._context(),
            )
            self.assertEqual(
                compile_mock.call_args.kwargs,
                {
                    "backend": "eager",
                    "mode": "reduce-overhead",
                    "fullgraph": True,
                    "dynamic": False,
                },
            )
            result.restore()

        options_model = nn.Linear(2, 1)
        with patch.object(
                torch,
                "compile",
                side_effect=return_eager,
        ) as compile_mock:
            result = OptimizationPassManager().apply(
                options_model,
                (
                    TorchCompilePass(
                        backend="eager",
                        fullgraph=False,
                        dynamic=True,
                        options={"trace.enabled": False},
                        requirement="required",
                    ), ),
                self._context(),
            )
            self.assertEqual(
                compile_mock.call_args.kwargs,
                {
                    "backend": "eager",
                    "fullgraph": False,
                    "dynamic": True,
                    "options": {
                        "trace.enabled": False,
                    },
                },
            )
            result.restore()

    def test_inherited_unimplemented_forward_uses_real_infer_boundary(self):

        class InferOnlyModule(nn.Module):

            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(2.0))

            def infer(self, value):
                return value * self.weight

        compile_calls = []

        def compile_method(function, **kwargs):
            del kwargs

            def compiled(*args, **call_kwargs):
                compile_calls.append(function.__name__)
                return function(*args, **call_kwargs)

            return compiled

        model = InferOnlyModule()
        with patch.object(
                torch,
                "compile",
                side_effect=compile_method,
        ):
            result = OptimizationPassManager().apply(
                model,
                (TorchCompilePass(
                    backend="eager",
                    requirement="required",
                ), ),
                self._context(),
            )

        self.assertNotIn("forward", model.__dict__)
        self.assertIn("infer", model.__dict__)
        torch.testing.assert_close(
            result.model.infer(torch.tensor(3.0)),
            torch.tensor(6.0),
        )
        self.assertEqual(compile_calls, ["infer"])
        self.assertEqual(
            result.manifest()["passes"][0]["metadata"]["execution_targets"],
            ["infer"],
        )
        result.restore()
        self.assertNotIn("infer", model.__dict__)

    def test_declared_targets_precede_forward_and_accept_generators(self):

        class Runtime(nn.Module):

            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(2.0))

            def forward(self, value):
                raise AssertionError("unused training boundary")

            def infer(self, value):
                return value * self.weight

            def optimization_compile_targets(self, mode):
                self.requested_mode = mode
                yield (" inference ", self, " infer ")

        compiled_names = []

        def compile_method(function, **kwargs):
            del kwargs

            def compiled(*args, **call_kwargs):
                compiled_names.append(function.__name__)
                return function(*args, **call_kwargs)

            return compiled

        runtime = Runtime()
        with patch.object(
                torch,
                "compile",
                side_effect=compile_method,
        ):
            result = OptimizationPassManager().apply(
                runtime,
                (TorchCompilePass(
                    backend="eager",
                    requirement="required",
                ), ),
                self._context(),
            )

        self.assertEqual(runtime.requested_mode, "inference")
        self.assertNotIn("forward", runtime.__dict__)
        self.assertIn("infer", runtime.__dict__)
        torch.testing.assert_close(
            runtime.infer(torch.tensor(3.0)),
            torch.tensor(6.0),
        )
        self.assertEqual(compiled_names, ["infer"])
        self.assertEqual(
            result.manifest_metadata()[0]["metadata"]["execution_targets"],
            ["inference"],
        )
        self.assertEqual(
            result.manifest_metadata()[0]["metadata"]["execution_target_bindings"],
            [{
                "label": "inference",
                "owner_type": (f"{type(runtime).__module__}."
                               f"{type(runtime).__qualname__}"),
                "attribute": "infer",
            }],
        )
        result.restore()
        self.assertNotIn("infer", runtime.__dict__)

    def test_vui_rejects_inference_compile_and_restores_training_compile(self):
        from voicehub.models.vui.model import Vui

        class TinyDecoder(nn.Module):

            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(2.0))

            def forward(self, value):
                return value * self.weight

        class TinyCodec(nn.Module):

            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(1.0))
                self.calls = 0

            def from_indices(self, value):
                self.calls += 1
                return value + self.weight

        runtime = object.__new__(Vui)
        nn.Module.__init__(runtime)
        runtime.decoder = TinyDecoder()
        runtime.codec = TinyCodec()
        original_keys = tuple(runtime.state_dict())
        inference_context = self._context(architecture="vui")

        for dynamic in (None, False, True):
            with (
                    self.subTest(dynamic=dynamic),
                    self.assertRaisesRegex(
                        OptimizationCompatibilityError,
                        r"vui.*inference.*real-checkpoint",
                    ),
            ):
                OptimizationPassManager().apply(
                    runtime,
                    (TorchCompilePass(
                        backend="eager",
                        dynamic=dynamic,
                        requirement="required",
                    ), ),
                    inference_context,
                )
        self.assertNotIn("forward", runtime.decoder.__dict__)
        self.assertNotIn("from_indices", runtime.codec.__dict__)

        fallback = OptimizationPassManager().apply(
            runtime,
            (TorchCompilePass(
                backend="eager",
                requirement="auto",
            ), ),
            inference_context,
        )
        self.assertEqual(
            fallback.manifest_metadata()[0]["metadata"]["outcome"],
            "eager-fallback",
        )
        self.assertIn(
            "real-checkpoint validation",
            fallback.manifest_metadata()[0]["metadata"]["reason"],
        )
        self.assertIs(fallback.restore(), runtime)

        training_context = self._context(
            "training",
            architecture="vui",
        )
        result = OptimizationPassManager().apply(
            runtime,
            (TorchCompilePass(
                backend="eager",
                requirement="required",
            ), ),
            training_context,
        )

        self.assertIn("forward", runtime.decoder.__dict__)
        self.assertNotIn("from_indices", runtime.codec.__dict__)
        self.assertEqual(
            result.manifest_metadata()[0]["metadata"]["execution_targets"],
            ["decoder.forward"],
        )
        torch.testing.assert_close(
            runtime.decoder(torch.tensor(3.0)),
            torch.tensor(6.0),
        )
        torch.testing.assert_close(
            runtime.codec.from_indices(torch.tensor(3.0)),
            torch.tensor(4.0),
        )
        self.assertEqual(runtime.codec.calls, 1)
        self.assertEqual(tuple(runtime.state_dict()), original_keys)

        self.assertIs(result.restore(), runtime)
        self.assertNotIn("forward", runtime.decoder.__dict__)
        self.assertNotIn("from_indices", runtime.codec.__dict__)
        self.assertEqual(tuple(runtime.state_dict()), original_keys)

    def test_runtime_binding_is_part_of_immutable_application_identity(self):

        class Runtime(nn.Module):

            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(()))
                self.selected = "first"

            def first(self, value):
                return value * self.weight

            def second(self, value):
                return value + self.weight

            def optimization_compile_targets(self, mode):
                del mode
                return (("stage", self, self.selected), )

        runtime = Runtime()
        first = OptimizationPassManager().apply(
            runtime,
            (TorchCompilePass(
                backend="eager",
                requirement="required",
            ), ),
            self._context(),
        )
        first_binding = first.manifest_metadata()[0]["metadata"]["execution_target_bindings"]
        first.restore()

        runtime.selected = "second"
        second = OptimizationPassManager().apply(
            runtime,
            (TorchCompilePass(
                backend="eager",
                requirement="required",
            ), ),
            self._context(),
        )
        second_binding = second.manifest_metadata()[0]["metadata"]["execution_target_bindings"]

        self.assertEqual(first_binding[0]["label"], "stage")
        self.assertEqual(second_binding[0]["label"], "stage")
        self.assertEqual(first_binding[0]["attribute"], "first")
        self.assertEqual(second_binding[0]["attribute"], "second")
        self.assertNotEqual(first_binding, second_binding)
        second.restore()

    def test_declared_empty_targets_are_authoritative(self):

        class UnsupportedRuntime(nn.Module):

            def forward(self, value):
                return value

            @staticmethod
            def optimization_compile_targets(mode):
                del mode
                return ()

        runtime = UnsupportedRuntime()
        result = OptimizationPassManager().apply(
            runtime,
            (TorchCompilePass(
                backend="eager",
                requirement="auto",
            ), ),
            self._context(),
        )
        self.assertNotIn("forward", runtime.__dict__)
        self.assertEqual(
            result.manifest_metadata()[0]["metadata"]["outcome"],
            "eager-fallback",
        )
        self.assertIn(
            "no compilable execution callable",
            result.manifest_metadata()[0]["metadata"]["reason"],
        )

        with self.assertRaisesRegex(
                OptimizationCompatibilityError,
                "no compilable execution callable",
        ):
            OptimizationPassManager().apply(
                UnsupportedRuntime(),
                (TorchCompilePass(
                    backend="eager",
                    requirement="required",
                ), ),
                self._context(),
            )

    def test_declared_and_explicit_targets_reject_ambiguous_declarations(self):

        class Runtime(nn.Module):

            def first(self, value):
                return value

            def second(self, value):
                return value

            def optimization_compile_targets(self, mode):
                del mode
                return (
                    ("same", self, "first"),
                    ("same", self, "second"),
                )

        with self.assertRaisesRegex(ValueError, "target label"):
            OptimizationPassManager().apply(
                Runtime(),
                (TorchCompilePass(
                    backend="eager",
                    requirement="required",
                ), ),
                self._context(),
            )

        first = Runtime()
        second = Runtime()
        with self.assertRaisesRegex(ValueError, "repeat a target label"):
            TorchCompilePass(
                execution_targets=(
                    OptimizationCompileTarget(
                        "same",
                        first,
                        "first",
                    ),
                    OptimizationCompileTarget(
                        "same",
                        second,
                        "second",
                    ),
                ), )

        class Unimplemented(nn.Module):
            pass

        unimplemented = Unimplemented()
        with self.assertRaisesRegex(TypeError, "unimplemented"):
            TorchCompilePass(
                execution_targets=(OptimizationCompileTarget(
                    "forward",
                    unimplemented,
                    "forward",
                ), ), )

    def test_f5_runtime_compiles_boundaries_used_by_infer(self):
        from voicehub.models.f5tts.native.runtime import NativeF5TTSRuntime

        class Flow(nn.Module):

            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(()))
                self.transformer = nn.Identity()
                self.config = SimpleNamespace(
                    sample_rate=24_000,
                    hop_length=2,
                )
                self.num_channels = 4

            @property
            def device(self):
                return self.weight.device

            def forward(self, value):
                return value * self.weight

            def sample(
                self,
                reference,
                token_ids,
                duration,
                **kwargs,
            ):
                del token_ids, kwargs
                self.transformer(reference)
                return (
                    torch.ones(
                        reference.shape[0],
                        duration,
                        self.num_channels,
                        device=reference.device,
                    ) * self.weight,
                    None,
                )

        class Vocoder(nn.Module):

            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(()))
                self.backbone = SimpleNamespace(input_channels=4)
                self.decode_calls = 0

            def decode(self, features):
                self.decode_calls += 1
                return features.mean(dim=1) * self.weight

        class Frontend:

            @staticmethod
            def normalize(value):
                return tuple(value)

            @staticmethod
            def encode_batch(values, *, device):
                return torch.ones(
                    len(values),
                    2,
                    device=device,
                    dtype=torch.long,
                )

        flow = Flow()
        vocoder = Vocoder()
        runtime = NativeF5TTSRuntime(
            flow_model=flow,
            vocoder=vocoder,
            frontend=Frontend(),
        )
        original_keys = tuple(runtime.state_dict())
        compile_calls = []

        def compile_method(function, **kwargs):
            del kwargs

            def compiled(*args, **call_kwargs):
                compile_calls.append(function.__name__)
                return function(*args, **call_kwargs)

            return compiled

        with (
                patch.object(
                    torch,
                    "compile",
                    side_effect=compile_method,
                ),
                patch.object(
                    NativeF5TTSRuntime,
                    "_prepare_reference",
                    return_value=(torch.ones(8), 0.1),
                ),
        ):
            inference_kwargs = {
                "ref_file": "unused.wav",
                "ref_text": (1, ),
                "gen_text": (2, ),
                "nfe_step": 1,
                "cross_fade_duration": 0.0,
            }
            expected_waveform, expected_rate, expected_spectrogram = (runtime.infer(**inference_kwargs))
            result = OptimizationPassManager().apply(
                runtime,
                (TorchCompilePass(
                    backend="eager",
                    requirement="required",
                ), ),
                OptimizationContext(
                    mode="inference",
                    architecture="f5tts",
                    device="cpu",
                    dtype="float32",
                ),
            )
            waveform, sample_rate, spectrogram = runtime.infer(**inference_kwargs)

        self.assertEqual(sample_rate, expected_rate)
        self.assertEqual(sample_rate, 24_000)
        self.assertTrue(waveform.numel())
        self.assertTrue(spectrogram.numel())
        torch.testing.assert_close(waveform, expected_waveform)
        torch.testing.assert_close(spectrogram, expected_spectrogram)
        self.assertEqual(compile_calls, ["forward"])
        self.assertEqual(vocoder.decode_calls, 2)
        self.assertEqual(
            result.manifest()["passes"][0]["metadata"]["execution_targets"],
            ["flow_model.transformer.forward"],
        )
        self.assertIn("forward", flow.transformer.__dict__)
        self.assertNotIn("sample", flow.__dict__)
        self.assertNotIn("decode", vocoder.__dict__)
        self.assertEqual(tuple(runtime.state_dict()), original_keys)
        result.restore()
        self.assertNotIn("forward", flow.transformer.__dict__)
        self.assertNotIn("sample", flow.__dict__)
        self.assertNotIn("decode", vocoder.__dict__)
        self.assertEqual(tuple(runtime.state_dict()), original_keys)

    def test_neutts_rejects_inference_compile_and_keeps_training_compile(self):
        from voicehub.models.neutts.native.modeling import NeuTTSRuntime

        class Backbone(nn.Module):

            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(2.0))

            def forward(self, value):
                return value * self.weight

        class Codec(nn.Module):

            def __init__(self):
                super().__init__()
                self.bias = nn.Parameter(torch.tensor(1.0))
                self.decode_calls = 0

            def decode_code(self, value):
                self.decode_calls += 1
                return value + self.bias

        runtime = object.__new__(NeuTTSRuntime)
        nn.Module.__init__(runtime)
        runtime.backbone = Backbone()
        runtime.codec = Codec()
        original_keys = tuple(runtime.state_dict())
        compile_calls = []

        def compile_method(function, **kwargs):
            del kwargs

            def compiled(*args, **call_kwargs):
                compile_calls.append(function.__name__)
                return function(*args, **call_kwargs)

            return compiled

        with patch.object(
                torch,
                "compile",
                side_effect=compile_method,
        ):
            with self.assertRaisesRegex(
                    OptimizationCompatibilityError,
                    r"neutts.*inference.*real-checkpoint",
            ):
                OptimizationPassManager().apply(
                    runtime,
                    (TorchCompilePass(
                        backend="eager",
                        requirement="required",
                    ), ),
                    self._context(architecture="neutts"),
                )
            fallback = OptimizationPassManager().apply(
                runtime,
                (TorchCompilePass(
                    backend="eager",
                    requirement="auto",
                ), ),
                self._context(architecture="neutts"),
            )
            result = OptimizationPassManager().apply(
                runtime,
                (TorchCompilePass(
                    backend="eager",
                    requirement="required",
                ), ),
                self._context("training", architecture="neutts"),
            )
            actual = runtime.codec.decode_code(runtime.backbone(runtime.backbone(torch.tensor(3.0))))

        torch.testing.assert_close(actual, torch.tensor(13.0))
        self.assertEqual(compile_calls, ["forward", "forward"])
        self.assertEqual(runtime.codec.decode_calls, 1)
        self.assertEqual(
            fallback.manifest_metadata()[0]["metadata"]["outcome"],
            "eager-fallback",
        )
        self.assertEqual(
            result.manifest()["passes"][0]["metadata"]["execution_targets"],
            ["backbone.forward"],
        )
        self.assertIn("forward", runtime.backbone.__dict__)
        self.assertNotIn("decode_code", runtime.codec.__dict__)
        self.assertEqual(tuple(runtime.state_dict()), original_keys)

        result.restore()

        self.assertNotIn("forward", runtime.backbone.__dict__)
        self.assertNotIn("decode_code", runtime.codec.__dict__)
        self.assertEqual(tuple(runtime.state_dict()), original_keys)

    def test_qwen_runtime_routes_selectors_and_compile_into_synthesis(self):
        from voicehub.kernels import KernelBackend
        from voicehub.models.qwen3tts.native.runtime import NativeQwen3TTSRuntime
        from voicehub.neural.backends import FlashAttention4Policy

        class Talker(nn.Module):

            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(()))
                self.kernel_backend = KernelBackend.AUTO
                self.flash_attention4_policy = (FlashAttention4Policy.DISABLED)

            def set_kernel_backend(self, backend):
                self.kernel_backend = KernelBackend.coerce(backend)

            def set_flash_attention4_policy(self, policy):
                self.flash_attention4_policy = (FlashAttention4Policy.coerce(policy))

            def generate_codes(self, **kwargs):
                del kwargs
                return torch.ones(1, 4, dtype=torch.long)

        class Model(nn.Module):

            def __init__(self):
                super().__init__()
                self.talker = Talker()

            @property
            def device(self):
                return self.talker.weight.device

            def forward(self, value):
                return value

        class Decoder(nn.Module):

            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(()))

            def chunked_decode(self, codes):
                return torch.ones(
                    codes.shape[0],
                    1,
                    8,
                    device=codes.device,
                ) * self.weight

        model = Model()
        decoder = Decoder()
        runtime = NativeQwen3TTSRuntime(
            artifacts=None,
            config=SimpleNamespace(),
            tokenizer_config=SimpleNamespace(output_sample_rate=24_000),
            tokenizer=None,
            processor=None,
            model=model,
            speech_decoder=decoder,
            generation_config={},
        )
        original_keys = tuple(runtime.state_dict())
        compile_calls = []

        def compile_method(function, **kwargs):
            del kwargs

            def compiled(*args, **call_kwargs):
                compile_calls.append(function.__name__)
                return function(*args, **call_kwargs)

            return compiled

        with (
                patch.object(
                    torch,
                    "compile",
                    side_effect=compile_method,
                ),
                patch.object(
                    NativeQwen3TTSRuntime,
                    "_generation_values",
                    return_value={},
                ),
                patch.object(
                    NativeQwen3TTSRuntime,
                    "_prompt",
                    return_value=(
                        torch.ones(1, 2, 2),
                        torch.ones(1, 2, dtype=torch.long),
                        torch.empty(1, 0, 2),
                    ),
                ),
        ):
            result = OptimizationPassManager().apply(
                runtime,
                (
                    CustomKernelPass(backend="torch"),
                    FlashAttention4Pass(policy="auto"),
                    TorchCompilePass(
                        backend="eager",
                        requirement="required",
                    ),
                ),
                OptimizationContext(
                    mode="inference",
                    architecture="qwen3-tts",
                    device="cpu",
                    dtype="float32",
                ),
            )
            waveforms, sample_rate = runtime._synthesize(
                "test",
                language="auto",
                speaker=None,
                instruction="",
                speaker_embedding=None,
                non_streaming_mode=True,
                seed=1,
            )

        self.assertEqual(sample_rate, 24_000)
        self.assertEqual(len(waveforms), 1)
        self.assertEqual(
            compile_calls,
            ["generate_codes", "chunked_decode"],
        )
        self.assertEqual(tuple(runtime.state_dict()), original_keys)
        self.assertEqual(model.talker.kernel_backend, KernelBackend.TORCH)
        self.assertEqual(
            model.talker.flash_attention4_policy,
            FlashAttention4Policy.AUTO,
        )
        result.restore()
        self.assertEqual(model.talker.kernel_backend, KernelBackend.AUTO)
        self.assertEqual(
            model.talker.flash_attention4_policy,
            FlashAttention4Policy.DISABLED,
        )
        self.assertNotIn("generate_codes", model.talker.__dict__)
        self.assertNotIn("chunked_decode", decoder.__dict__)

    def test_compile_config_is_strict_immutable_and_rejects_mode_with_options(self):
        source = {
            "nested": {
                "value": 1,
            },
            "sequence": [1, 2],
        }
        config = TorchCompileConfig(options=source)
        source["nested"]["value"] = 2
        source["sequence"].append(3)

        self.assertEqual(
            config.manifest()["options"],
            {
                "nested": {
                    "value": 1,
                },
                "sequence": [1, 2],
            },
        )
        with self.assertRaises(TypeError):
            config.options["nested"]["value"] = 3
        with self.assertRaisesRegex(ValueError, "strict JSON"):
            TorchCompileConfig(options={"invalid": object()})
        with self.assertRaisesRegex(ValueError, "strict JSON"):
            TorchCompileConfig(options={"invalid": float("nan")})
        with self.assertRaisesRegex(ValueError, "non-string mapping key"):
            TorchCompileConfig(options={"nested": {1: "invalid"}})
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            TorchCompileConfig(
                mode="reduce-overhead",
                options={"trace.enabled": False},
            )

    def test_auto_mode_falls_back_when_capability_is_unavailable(self):
        unavailable = TorchCompileCapabilityReport(
            available=False,
            backend="missing",
            backend_available=False,
            torch_version=torch.__version__,
            available_backends=("eager", ),
            reason="backend unavailable for test",
        )
        model = nn.Linear(2, 1)
        with patch(
                "voicehub.optimization.torch_compile.inspect_torch_compile",
                return_value=unavailable,
        ):
            result = OptimizationPassManager().apply(
                model,
                (TorchCompilePass(backend="missing", requirement="auto"), ),
                self._context(),
            )
            self.assertIs(result.model, model)
            self.assertEqual(
                result.manifest_metadata()[0]["metadata"]["outcome"],
                "eager-fallback",
            )
            with self.assertRaisesRegex(
                    OptimizationCompatibilityError,
                    "Required torch.compile",
            ):
                OptimizationPassManager().apply(
                    model,
                    (TorchCompilePass(
                        backend="missing",
                        requirement="required",
                    ), ),
                    self._context(),
                )

    def test_lazy_compiler_failure_falls_back_only_in_auto_mode(self):

        class CountingModule(nn.Module):

            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(2.0))
                self.eager_calls = 0

            def forward(self, value):
                self.eager_calls += 1
                return value * self.weight

        compiler_calls = []

        def failing_compiled(*args, **kwargs):
            del args, kwargs
            compiler_calls.append("called")
            raise torch._dynamo.exc.TorchDynamoException("deliberate compiler failure")

        auto_model = CountingModule()
        with patch.object(torch, "compile", return_value=failing_compiled):
            auto_result = OptimizationPassManager().apply(
                auto_model,
                (TorchCompilePass(backend="eager", requirement="auto"), ),
                self._context(),
            )
        torch.testing.assert_close(
            auto_result.model(torch.tensor(3.0)),
            torch.tensor(6.0),
        )
        torch.testing.assert_close(
            auto_result.model(torch.tensor(4.0)),
            torch.tensor(8.0),
        )
        self.assertEqual(compiler_calls, ["called"])
        self.assertEqual(auto_model.eager_calls, 2)
        self.assertTrue(auto_model.forward.using_eager)
        self.assertIn(
            "TorchDynamoException",
            auto_model.forward.fallback_reason,
        )
        fallback_manifest = auto_result.manifest()
        fallback_entry = fallback_manifest["passes"][0]
        self.assertEqual(
            fallback_entry["metadata"]["outcome"],
            "compiled",
        )
        fallback_status = fallback_entry["runtime_status"]
        self.assertEqual(
            fallback_status["outcome"],
            "eager-fallback",
        )
        self.assertEqual(
            fallback_status["fallbacks"][0]["execution_target"],
            "forward",
        )
        self.assertIn(
            "TorchDynamoException",
            fallback_status["fallbacks"][0]["reason"],
        )
        with self.assertRaises(AttributeError):
            auto_result.applied[0].result.metadata["execution_targets"].append("mutated")
        self.assertEqual(
            auto_result.manifest()["passes"][0],
            fallback_entry,
        )
        auto_result.restore()

        required_model = CountingModule()
        with patch.object(torch, "compile", return_value=failing_compiled):
            required_result = OptimizationPassManager().apply(
                required_model,
                (TorchCompilePass(
                    backend="eager",
                    requirement="required",
                ), ),
                self._context(),
            )
        with self.assertRaisesRegex(
                TorchCompileRuntimeError,
                "Required torch.compile execution failed",
        ):
            required_result.model(torch.tensor(3.0))
        self.assertEqual(required_model.eager_calls, 0)
        required_entry = required_result.manifest()["passes"][0]
        self.assertEqual(required_entry["metadata"]["outcome"], "compiled")
        required_status = required_entry["runtime_status"]
        self.assertEqual(required_status["outcome"], "compile-error")
        self.assertIn(
            "TorchDynamoException",
            required_status["errors"][0]["reason"],
        )
        required_result.restore()

    def test_state_key_mismatch_restores_once_and_preserves_required_error(self):

        class StateChangingModule(nn.Module):

            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(()))

            def forward(self, value):
                return value * self.weight

            def state_dict(self, *args, **kwargs):
                state = super().state_dict(*args, **kwargs)
                if "forward" in self.__dict__:
                    state["compiled-only"] = torch.ones(())
                return state

        model = StateChangingModule()
        with (
                patch.object(
                    torch,
                    "compile",
                    side_effect=lambda function, **kwargs: function,
                ),
                self.assertRaises(OptimizationApplicationError) as caught,
        ):
            OptimizationPassManager().apply(
                model,
                (TorchCompilePass(
                    backend="eager",
                    requirement="required",
                ), ),
                self._context(),
            )

        self.assertIsInstance(
            caught.exception.cause,
            TorchCompileUnavailableError,
        )
        self.assertIn(
            "changed canonical state-dict keys",
            str(caught.exception.cause),
        )
        self.assertNotIn("forward", model.__dict__)

    def test_training_adapter_components_compile_without_wrapping_adapter(self):

        class Adapter:

            def __init__(self):
                self.primary_model = nn.Linear(2, 2)
                self.auxiliary = nn.Linear(2, 1)
                self._components = [
                    ("model", self.primary_model),
                    ("model_duplicate", self.primary_model),
                    ("auxiliary", self.auxiliary),
                ]

            def state_dict(self):
                state = {}
                for name, component in (
                    ("model", self.primary_model),
                    ("auxiliary", self.auxiliary),
                ):
                    state.update({f"{name}.{key}": value for key, value in component.state_dict().items()})
                return state

        adapter = Adapter()
        original_keys = tuple(adapter.state_dict())
        result = OptimizationPassManager().apply(
            adapter,
            (TorchCompilePass(
                backend="eager",
                requirement="required",
            ), ),
            self._context("training"),
        )

        self.assertIs(result.model, adapter)
        self.assertEqual(tuple(adapter.state_dict()), original_keys)
        self.assertIn("forward", adapter.primary_model.__dict__)
        self.assertIn("forward", adapter.auxiliary.__dict__)
        self.assertEqual(
            result.manifest_metadata()[0]["metadata"]["execution_targets"],
            [
                "component:primary_model",
                "component:auxiliary",
            ],
        )
        output = adapter.auxiliary(adapter.primary_model(torch.ones(1, 2)))
        output.sum().backward()
        self.assertIsNotNone(adapter.primary_model.weight.grad)
        self.assertIsNotNone(adapter.auxiliary.weight.grad)

        self.assertIs(result.restore(), adapter)
        self.assertNotIn("forward", adapter.primary_model.__dict__)
        self.assertNotIn("forward", adapter.auxiliary.__dict__)

    def test_training_adapter_honors_component_target_provider(self):

        class Component(nn.Module):

            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(2.0))

            def compute(self, value):
                return value * self.weight

            def optimization_compile_targets(self, mode):
                self.requested_mode = mode
                return (OptimizationCompileTarget(
                    "compute",
                    self,
                    "compute",
                ), )

        class Adapter:

            def __init__(self):
                self.primary_model = Component()
                self._components = [
                    ("primary", self.primary_model),
                ]

            def state_dict(self):
                return {f"primary.{name}": value for name, value in self.primary_model.state_dict().items()}

        adapter = Adapter()
        result = OptimizationPassManager().apply(
            adapter,
            (TorchCompilePass(
                backend="eager",
                requirement="required",
            ), ),
            self._context("training"),
        )

        self.assertEqual(adapter.primary_model.requested_mode, "training")
        self.assertIn("compute", adapter.primary_model.__dict__)
        self.assertNotIn("forward", adapter.primary_model.__dict__)
        torch.testing.assert_close(
            adapter.primary_model.compute(torch.tensor(3.0)),
            torch.tensor(6.0),
        )
        self.assertEqual(
            result.manifest_metadata()[0]["metadata"]["execution_targets"],
            ["component:primary_model.compute"],
        )
        result.restore()
        self.assertNotIn("compute", adapter.primary_model.__dict__)

    def test_trainer_applies_compile_before_optimization_and_trains_on_cpu(self):

        class LossModel(nn.Module):

            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(()))

            def forward(self, input_values, labels):
                prediction = input_values * self.weight
                return {
                    "loss": (prediction - labels).square().mean(),
                }

        model = LossModel()
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(
                model=model,
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
                optimization_plan=TorchCompilePass(
                    backend="eager",
                    requirement="required",
                ),
            )
            output = trainer.train()

        self.assertEqual(output.global_step, 1)
        self.assertIs(trainer.optimization_result.model, model)
        self.assertEqual(tuple(model.state_dict()), ("weight", ))
        self.assertEqual(
            trainer.optimization_manifest()["passes"][0]["kind"],
            "compile",
        )
        optimizer_parameters = {
            id(parameter)
            for group in trainer.optimizer.param_groups
            for parameter in group["params"]
        }
        self.assertEqual(optimizer_parameters, {id(model.weight)})

    def test_auto_fallback_checkpoint_resumes_by_static_plan_identity(self):

        class LossModel(nn.Module):

            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(()))

            def forward(self, input_values, labels):
                prediction = input_values * self.weight
                return {
                    "loss": (prediction - labels).square().mean(),
                }

        def failing_compiled(*args, **kwargs):
            del args, kwargs
            raise torch._dynamo.exc.TorchDynamoException("deliberate lazy compiler failure")

        dataset = [{
            "input_values": torch.ones(1),
            "labels": torch.zeros(1),
        }]
        with tempfile.TemporaryDirectory() as directory:
            common = {
                "output_dir": directory,
                "max_steps": 1,
                "per_device_train_batch_size": 1,
                "logging_strategy": "no",
                "save_strategy": "steps",
                "save_steps": 1,
                "use_cpu": True,
                "report_to": [],
            }
            first = Trainer(
                model=LossModel(),
                args=TrainingArguments(**common),
                train_dataset=dataset,
                optimization_plan=TorchCompilePass(
                    backend="eager",
                    requirement="auto",
                ),
            )
            with patch.object(
                    torch,
                    "compile",
                    return_value=failing_compiled,
            ):
                first.train()

            self.assertEqual(
                first.optimization_manifest()["passes"][0]["runtime_status"]["outcome"],
                "eager-fallback",
            )
            checkpoint = f"{directory}/checkpoint-1"
            resumed = Trainer(
                model=LossModel(),
                args=TrainingArguments(**common),
                train_dataset=dataset,
                optimization_plan=TorchCompilePass(
                    backend="eager",
                    requirement="auto",
                ),
            )
            with patch.object(
                    torch,
                    "compile",
                    return_value=failing_compiled,
            ):
                output = resumed.train(resume_from_checkpoint=checkpoint, )

            self.assertEqual(output.global_step, 1)

    def test_callable_proxy_delegates_canonical_state_and_restores(self):

        class CallableRuntime:

            def __init__(self):
                self.weight = torch.tensor(3.0)

            def __call__(self, value):
                return value * self.weight

            def state_dict(self):
                return {"weight": self.weight}

        runtime = CallableRuntime()
        result = OptimizationPassManager().apply(
            runtime,
            (TorchCompilePass(
                backend="eager",
                requirement="required",
            ), ),
            self._context(),
        )
        self.assertIsNot(result.model, runtime)
        self.assertEqual(tuple(result.model.state_dict()), ("weight", ))
        torch.testing.assert_close(
            result.model(torch.tensor(2.0)),
            torch.tensor(6.0),
        )
        self.assertEqual(tuple(result.portable_state_dict()), ("weight", ))
        self.assertIs(result.restore(), runtime)


if __name__ == "__main__":
    unittest.main()
