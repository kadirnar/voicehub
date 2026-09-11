import ast
import copy
import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from voicehub.dependencies import resolve_import_path
from voicehub.errors import OptionalDependencyError
from voicehub.training.adapters import (
    BaseTrainingAdapter,
    CausalLMTrainingAdapter,
    FlowMatchingTrainingAdapter,
    VITSTrainingAdapter,
)
from voicehub.training.auto import AutoTrainingAdapter
from voicehub.training.contracts import (
    TrainingContext,
    TrainingPhaseKind,
    TrainingPhaseSpec,
    TrainingRecipeKind,
    TrainingSupport,
)
from voicehub.training.recipes import BUILTIN_MODEL_ADAPTERS, CodecCausalLMTrainingAdapter
from voicehub.training.specs import (
    MODEL_TRAINING_SPECS,
    ModelTrainingSpec,
    TrainingFamily,
    get_training_spec,
    list_training_specs,
    register_training_alias,
    register_training_spec,
    unregister_training_alias,
    unregister_training_spec,
)

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class TrainingContractValidationTests(unittest.TestCase):

    def test_support_and_phase_values_are_validated(self):
        self.assertIs(
            TrainingSupport.coerce("preprocessed"),
            TrainingSupport.PREPROCESSED,
        )
        with self.assertRaisesRegex(ValueError, "Unknown training support"):
            TrainingSupport.coerce("probably")
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            TrainingPhaseSpec(name="flow", frequency=0)
        with self.assertRaisesRegex(ValueError, "optimizer_names"):
            TrainingPhaseSpec(
                name="discriminator",
                kind=TrainingPhaseKind.DISCRIMINATOR,
            )
        with self.assertRaisesRegex(ValueError, "one name"):
            TrainingPhaseSpec(
                name="bad-routing",
                component_paths=("model.a", ),
                optimizer_names=("a", "b"),
            )
        with self.assertRaisesRegex(ValueError, "must declare optimizer_names"):
            TrainingPhaseSpec(
                name="unrouted-boundary",
                optimizer_step_after_phase=True,
            )

    def test_support_introspection_and_filtering(self):
        inference_only = list_training_specs(support=TrainingSupport.INFERENCE_ONLY)
        self.assertEqual(inference_only, ())
        self.assertTrue(all(not spec.is_turnkey for spec in inference_only))
        self.assertTrue(all(not spec.requires_custom_adapter for spec in inference_only))

        custom = list_training_specs(support="custom")
        self.assertTrue(custom)
        self.assertTrue(all(spec.requires_custom_adapter for spec in custom))
        self.assertTrue(all(not spec.is_turnkey for spec in custom))
        self.assertTrue(all(not spec.supports_training for spec in custom))
        self.assertTrue(all(spec.has_training_recipe for spec in custom))

    def test_all_builtin_models_still_have_profiles(self):
        self.assertEqual(len(MODEL_TRAINING_SPECS), 34)
        self.assertEqual(get_training_spec("f5-tts").model_type, "f5tts")
        self.assertIn(TrainingFamily.VITS, {spec.family for spec in list_training_specs()})
        self.assertEqual(
            get_training_spec("qwen3tts").training_default_model_name_or_path,
            "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        )
        self.assertEqual(
            {
                model_type: get_training_spec(model_type).dataset_factory
                for model_type in (
                    "orpheustts",
                    "llasa",
                    "neutts",
                    "outetts",
                )
            },
            {
                "orpheustts": ("voicehub.models.orpheustts.training:"
                               "build_training_dataset"),
                "llasa": ("voicehub.models.llasa.training:"
                          "build_training_dataset"),
                "neutts": ("voicehub.models.neutts.training:"
                           "build_training_dataset"),
                "outetts": ("voicehub.models.outetts.training:"
                            "build_training_dataset"),
            },
        )

    def test_lazy_training_factories_and_tokenizer_paths_are_validated(self):
        spec = ModelTrainingSpec(
            model_type="future-codec-tts",
            family=TrainingFamily.CAUSAL_LM,
            adapter_factory=(" voicehub.training.adapters:"
                             "CausalLMTrainingAdapter "),
            dataset_factory=(" voicehub.models.orpheustts.training:"
                             "build_training_dataset "),
            dataset_spec_factory=(" voicehub.training.asr_data_contracts:"
                                  "build_asr_whisper_dataset_spec "),
            tokenizer_paths=(
                " assets.tokenizer ",
                "model.tokenizer",
            ),
        )
        self.assertEqual(
            spec.adapter_factory,
            "voicehub.training.adapters:CausalLMTrainingAdapter",
        )
        self.assertEqual(
            spec.dataset_factory,
            "voicehub.models.orpheustts.training:build_training_dataset",
        )
        self.assertEqual(
            spec.dataset_spec_factory,
            "voicehub.training.asr_data_contracts:build_asr_whisper_dataset_spec",
        )
        self.assertEqual(
            spec.tokenizer_paths,
            ("assets.tokenizer", "model.tokenizer"),
        )

        with self.assertRaisesRegex(ValueError, "module:attribute"):
            ModelTrainingSpec(
                model_type="bad-dataset-factory",
                family=TrainingFamily.CAUSAL_LM,
                dataset_factory="voicehub.training.datasets",
            )
        with self.assertRaisesRegex(ValueError, "module:attribute"):
            ModelTrainingSpec(
                model_type="bad-adapter-factory",
                family=TrainingFamily.CAUSAL_LM,
                adapter_factory="voicehub.training.adapters",
            )
        with self.assertRaisesRegex(ValueError, "module:attribute"):
            ModelTrainingSpec(
                model_type="bad-dataset-spec-factory",
                family=TrainingFamily.CTC,
                dataset_spec_factory="voicehub.training.asr_data_contracts",
            )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            ModelTrainingSpec(
                model_type="duplicate-tokenizer-paths",
                family=TrainingFamily.CAUSAL_LM,
                tokenizer_paths=("tokenizer", "tokenizer"),
            )
        with self.assertRaisesRegex(ValueError, "dotted attribute paths"):
            ModelTrainingSpec(
                model_type="bad-tokenizer-path",
                family=TrainingFamily.CAUSAL_LM,
                tokenizer_paths=("model..tokenizer", ),
            )

    def test_framework_imports_remain_lazy(self):
        script = (
            "import sys;"
            "from voicehub.training.contracts import TrainingPhaseSpec;"
            "from voicehub.training.specs import list_training_specs;"
            "from voicehub.training.auto import AutoTrainingAdapter;"
            "print(len(list_training_specs()),"
            "'torch' in sys.modules,'numpy' in sys.modules,"
            "any(name in sys.modules for name in ("
            "'voicehub.models.orpheustts.training',"
            "'voicehub.models.conversationtts.training',"
            "'voicehub.models.f5tts.training',"
            "'voicehub.models.qwen3tts.training_adapter',"
            "'voicehub.models.llasa.training',"
            "'voicehub.models.neutts.training',"
            "'voicehub.models.outetts.training')))")
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.stdout.strip(), "34 False False False")

    def test_builtin_specialized_adapters_are_declarative_and_resolvable(self):
        specs = tuple(list_training_specs(task=None))
        declared = {
            spec.model_type: spec.adapter_factory
            for spec in specs if spec.adapter_factory is not None
        }
        self.assertEqual(len(specs), 70)
        self.assertEqual(len(declared), 66)
        self.assertEqual(set(BUILTIN_MODEL_ADAPTERS), set(declared))

        training_root = Path(__file__).parents[1] / "voicehub" / "training"
        auto_source = (training_root / "auto.py").read_text(encoding="utf-8")
        recipes_source = (training_root / "recipes.py").read_text(encoding="utf-8")
        self.assertNotIn("BUILTIN_MODEL_ADAPTERS", auto_source)
        self.assertNotIn("BUILTIN_MODEL_ADAPTERS = {", recipes_source)
        wrapper_names = [
            node.name for node in ast.parse(recipes_source).body
            if isinstance(node, ast.FunctionDef) and node.name.endswith("_adapter")
        ]
        self.assertEqual(wrapper_names, [])
        adapter_class_names = {
            node.name
            for node in ast.parse(recipes_source).body
            if isinstance(node, ast.ClassDef) and node.name.endswith("TrainingAdapter")
        }
        self.assertEqual(
            adapter_class_names,
            {
                "CodecCausalLMTrainingAdapter",
                "SourceRecipeTrainingAdapter",
            },
        )
        self.assertTrue(all(not path.startswith("voicehub.training.recipes:") for path in declared.values()))

        from voicehub.training import recipes

        compatibility_exports = {
            "ConversationTTSTrainingAdapter":
            ("voicehub.models.conversationtts.training:"
             "ConversationTTSTrainingAdapter"),
            "F5TTSTrainingAdapter": ("voicehub.models.f5tts.training:F5TTSTrainingAdapter"),
            "OrpheusTrainingAdapter": ("voicehub.models.orpheustts.training:OrpheusTrainingAdapter"),
            "Qwen3TTSTrainingAdapter":
            ("voicehub.models.qwen3tts.training_adapter:"
             "Qwen3TTSTrainingAdapter"),
        }
        for legacy_name, path in compatibility_exports.items():
            with self.subTest(legacy_name=legacy_name):
                self.assertIs(getattr(recipes, legacy_name), resolve_import_path(path))

        for model_type, path in declared.items():
            with self.subTest(model_type=model_type):
                factory = resolve_import_path(path)
                self.assertTrue(callable(factory))
                if isinstance(factory, type):
                    self.assertTrue(issubclass(factory, BaseTrainingAdapter))
                self.assertIs(BUILTIN_MODEL_ADAPTERS[model_type], factory)

    def test_phase_schedule_must_cover_every_recipe_step(self):
        with self.assertRaisesRegex(ValueError, "cover every recipe step"):
            ModelTrainingSpec(
                model_type="gapped-recipe",
                family=TrainingFamily.ACOUSTIC,
                phases=(TrainingPhaseSpec(
                    name="occasional",
                    frequency=2,
                    offset=0,
                ), ),
            )

    def test_source_audit_metadata_is_never_forwarded_as_model_kwargs(self):

        class Runtime:

            def forward(self, values, **kwargs):
                del values, kwargs

        filtered = BaseTrainingAdapter._filter_forward_inputs(
            Runtime().forward,
            {
                "values": [1.0],
                "language": "en",
                "speaker_id": "speaker-1",
                "id": "utterance-1",
                "session_id": "session-1",
                "consent": True,
                "license": "owned",
                "source": "studio",
                "metadata": {
                    "microphone": "one"
                },
            },
        )
        self.assertEqual(
            filtered,
            {
                "values": [1.0],
                "language": "en",
                "speaker_id": "speaker-1",
            },
        )

    def test_sequential_optimizer_boundary_is_part_of_resume_signature(self):
        phase = TrainingPhaseSpec(
            name="generator",
            kind=TrainingPhaseKind.GENERATOR,
            component_paths=("model.generator", ),
            optimizer_names=("generator", ),
            optimizer_step_after_phase=True,
        )
        spec = ModelTrainingSpec(
            model_type="dummy-resume-boundary",
            family=TrainingFamily.VITS,
            component_paths=("model.generator", ),
            phases=(phase, ),
        )

        signature = BaseTrainingAdapter(object(), spec).resume_signature()

        self.assertTrue(signature["phases"][0]["optimizer_step_after_phase"], )


class CodecDatasetFactoryContractTests(unittest.TestCase):

    class ReadyAdapter(CodecCausalLMTrainingAdapter):

        def setup(self):
            return self

    def test_extension_dataset_factory_needs_no_shared_model_branch(self):
        calls = []
        expected = object()

        def build_dataset(model, records, **kwargs):
            calls.append((model, records, kwargs))
            return expected

        model = SimpleNamespace()
        spec = ModelTrainingSpec(
            model_type="future-codec-dataset",
            family=TrainingFamily.CAUSAL_LM,
            dataset_factory="extension.training:build_dataset",
        )
        adapter = self.ReadyAdapter(model, spec)
        records = [{"text": "hello"}]

        with patch(
                "voicehub.training.recipes.resolve_import_path",
                return_value=build_dataset,
        ) as resolver:
            dataset = adapter.create_dataset(
                records,
                max_length=128,
            )

        self.assertIs(dataset, expected)
        resolver.assert_called_once_with("extension.training:build_dataset")
        self.assertEqual(calls, [(model, records, {"max_length": 128})])

    def test_dataset_factory_resolution_fails_actionably(self):
        spec = ModelTrainingSpec(
            model_type="missing-codec-dataset",
            family=TrainingFamily.CAUSAL_LM,
            dataset_factory="extension.training:missing",
        )
        adapter = self.ReadyAdapter(object(), spec)

        with patch(
                "voicehub.training.recipes.resolve_import_path",
                side_effect=AttributeError("missing"),
        ):
            with self.assertRaisesRegex(
                    ImportError,
                    "Could not resolve training dataset factory.*missing-codec-dataset",
            ):
                adapter.create_dataset([])
        with patch(
                "voicehub.training.recipes.resolve_import_path",
                return_value=object(),
        ):
            with self.assertRaisesRegex(TypeError, "must be callable"):
                adapter.create_dataset([])
        with patch(
                "voicehub.training.recipes.resolve_import_path",
                side_effect=ModuleNotFoundError("torch"),
        ):
            with self.assertRaisesRegex(
                    OptionalDependencyError,
                    r'pip install "voicehub\[training\]"',
            ):
                adapter.create_dataset([])

    def test_tokenizer_export_uses_declared_paths(self):

        class Saveable:

            def __init__(self):
                self.destinations = []

            def save_pretrained(self, destination, **kwargs):
                self.destinations.append((destination, kwargs))

        primary = Saveable()
        tokenizer = Saveable()
        model = SimpleNamespace(assets=SimpleNamespace(tokenizer=tokenizer))
        spec = ModelTrainingSpec(
            model_type="future-tokenizer-layout",
            family=TrainingFamily.CAUSAL_LM,
            tokenizer_paths=("assets.tokenizer", ),
        )
        adapter = self.ReadyAdapter(model, spec)
        adapter.primary_model = primary

        adapter.save_pretrained("export")

        self.assertEqual(
            primary.destinations,
            [(Path("export"), {
                "safe_serialization": True
            })],
        )
        self.assertEqual(tokenizer.destinations, [(Path("export"), {})])

    def test_shared_codec_adapter_has_no_registered_model_comparisons(self):
        source = (Path(__file__).parents[1] / "voicehub" / "training" /
                  "recipes.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        adapter_class = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "CodecCausalLMTrainingAdapter")
        compared_strings = {
            operand.value
            for node in ast.walk(adapter_class) if isinstance(node, ast.Compare)
            for operand in (node.left, *node.comparators)
            if isinstance(operand, ast.Constant) and isinstance(operand.value, str)
        }
        registered_models = {spec.model_type for spec in list_training_specs(task=None)}

        self.assertTrue(compared_strings.isdisjoint(registered_models), compared_strings)


class DynamicTrainingRegistryTests(unittest.TestCase):

    class FutureAdapter(BaseTrainingAdapter):
        supports_custom_recipe = True

        def compute_objective(self, predictions, labels):
            raise AssertionError("The dummy backend returns a native loss.")

    class Config:
        model_type = "future-voice-alias"

    class Wrapper:

        def __init__(self):
            self.config = DynamicTrainingRegistryTests.Config()
            self.runtime = None
            self.loaded = False

        def load_for_training(self):
            self.loaded = True
            self.runtime = DynamicTrainingRegistryTests._runtime()

    @staticmethod
    def _runtime():
        if not TORCH_AVAILABLE:
            return None
        import torch

        class Runtime(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))

            def forward(self, values):
                prediction = values * self.weight
                return {"loss": prediction.square().mean(), "logits": prediction}

        return Runtime()

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is an optional training extra")
    def test_future_family_spec_and_alias_registration(self):
        spec = ModelTrainingSpec(
            model_type="future-voice",
            family="future-objective",
            module_paths=("runtime", ),
            support=TrainingSupport.CUSTOM,
        )
        register_training_spec(spec, aliases=("future-voice-alias", ))
        AutoTrainingAdapter.register_family(
            "future-objective",
            self.FutureAdapter,
        )
        try:
            self.assertIs(get_training_spec("future-voice-alias"), spec)
            self.assertIn("future-voice", MODEL_TRAINING_SPECS)
            with self.assertRaisesRegex(ValueError, "already registered"):
                register_training_spec(spec)
            with self.assertRaisesRegex(ValueError, "already registered"):
                register_training_alias(
                    "future-voice-alias",
                    "future-voice",
                )

            wrapper = self.Wrapper()
            adapter = AutoTrainingAdapter.from_model(wrapper)
            self.assertIsInstance(adapter, self.FutureAdapter)
            adapter.setup()
            self.assertTrue(wrapper.loaded)
        finally:
            AutoTrainingAdapter.unregister_family(
                "future-objective",
                missing_ok=True,
            )
            unregister_training_spec(
                "future-voice",
                missing_ok=True,
            )

        with self.assertRaises(KeyError):
            get_training_spec("future-voice-alias")

    def test_alias_registration_can_be_managed_independently(self):
        spec = ModelTrainingSpec(
            model_type="future-alias-target",
            family="future-alias-family",
        )
        register_training_spec(spec)
        try:
            register_training_alias(
                "future_alias",
                "future-alias-target",
            )
            self.assertIs(get_training_spec("future_alias"), spec)
            self.assertEqual(
                unregister_training_alias("future_alias"),
                "future-alias-target",
            )
            with self.assertRaises(KeyError):
                get_training_spec("future_alias")
        finally:
            unregister_training_spec(
                "future-alias-target",
                missing_ok=True,
            )

    def test_adapter_override_can_be_removed_after_its_spec(self):
        spec = ModelTrainingSpec(
            model_type="future-teardown",
            family=TrainingFamily.ACOUSTIC,
        )
        register_training_spec(spec)
        AutoTrainingAdapter.register("future-teardown", self.FutureAdapter)
        unregister_training_spec("future-teardown")
        self.assertIs(
            AutoTrainingAdapter.unregister("future-teardown"),
            self.FutureAdapter,
        )

    def test_declarative_adapter_factory_requires_no_central_registration(self):
        model_type = "future-declarative-adapter"
        spec = ModelTrainingSpec(
            model_type=model_type,
            family="future-declarative-family",
            adapter_factory=("voicehub.training.adapters:"
                             "AcousticTrainingAdapter"),
        )
        register_training_spec(spec)
        try:
            wrapper = SimpleNamespace(config=SimpleNamespace(model_type=model_type), )
            adapter = AutoTrainingAdapter.from_model(wrapper)

            self.assertIsInstance(adapter, BaseTrainingAdapter)
            self.assertEqual(type(adapter).__name__, "AcousticTrainingAdapter")
            self.assertTrue(adapter._registered_specialization)
            self.assertNotIn(model_type, AutoTrainingAdapter._model_overrides)
        finally:
            unregister_training_spec(model_type, missing_ok=True)

    def test_declarative_adapter_factory_fails_with_actionable_path(self):
        spec = ModelTrainingSpec(
            model_type="future-missing-adapter",
            family="future-missing-family",
            adapter_factory="voicehub.training.adapters:MissingAdapter",
        )

        with self.assertRaisesRegex(
                ImportError,
                "Could not resolve training adapter factory.*MissingAdapter",
        ):
            AutoTrainingAdapter.from_model(object(), spec=spec)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is an optional training extra")
class PhaseExecutionTests(unittest.TestCase):

    class Config:
        model_type = "dummy-vits"

    class Wrapper:

        def __init__(self, runtime_factory):
            self.config = PhaseExecutionTests.Config()
            self.runtime_factory = runtime_factory
            self.runtime = None
            self.loaded = False
            self.prepared_phases = []

        def load_for_training(self):
            self.loaded = True
            self.runtime = self.runtime_factory()
            return self

        def prepare_training_inputs(self, inputs, *, phase):
            self.prepared_phases.append(phase)
            return inputs

    @staticmethod
    def _runtime():
        import torch

        class Runtime(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.generator = torch.nn.Linear(1, 1, bias=False)
                self.discriminator = torch.nn.Linear(1, 1, bias=False)
                self.duration_discriminator = torch.nn.Linear(
                    1,
                    1,
                    bias=False,
                )
                self.observations = {}
                self.checkpointing = False

            def generator_step(self, values, target):
                self.observations["generator_disc_frozen"] = (not self.discriminator.weight.requires_grad)
                prediction = self.generator(values)
                return {
                    "loss": (prediction - target).square().mean(),
                    "waveform": prediction,
                }

            def discriminator_step(self, fake, target):
                self.observations["discriminator_fake_detached"] = (not fake.requires_grad)
                self.observations["discriminator_gen_frozen"] = (not self.generator.weight.requires_grad)
                score = self.discriminator(fake)
                return {
                    "discriminator_loss": (score - target).square().mean(),
                    "score": score,
                }

            def duration_step(self, duration_prediction, target):
                self.observations["duration_input_detached"] = (not duration_prediction.requires_grad)
                score = self.duration_discriminator(duration_prediction)
                return {
                    "duration_discriminator_loss": (score - target).square().mean(),
                }

            def forward(self, values, target):
                return self.generator_step(values, target)

            def gradient_checkpointing_enable(self):
                self.checkpointing = True

            def gradient_checkpointing_disable(self):
                self.checkpointing = False

        return Runtime()

    @classmethod
    def _spec(cls):
        return ModelTrainingSpec(
            model_type="dummy-vits",
            family=TrainingFamily.VITS,
            module_paths=("runtime", ),
            component_paths=(
                "runtime.generator",
                "runtime.discriminator",
                "runtime.duration_discriminator",
            ),
            support=TrainingSupport.PREPROCESSED,
            recipe_kind=TrainingRecipeKind.ADVERSARIAL,
            separate_optimizers=True,
            phases=(
                TrainingPhaseSpec(
                    name="generator",
                    kind=TrainingPhaseKind.GENERATOR,
                    component_paths=("runtime.generator", ),
                    optimizer_names=("generator", ),
                    forward_component="runtime",
                    forward_method="generator_step",
                    required_inputs=("values", "target"),
                    prediction_keys=("waveform", ),
                    loss_keys=("loss", ),
                    frozen_component_paths=("runtime.discriminator", ),
                ),
                TrainingPhaseSpec(
                    name="discriminator",
                    kind=TrainingPhaseKind.DISCRIMINATOR,
                    component_paths=("runtime.discriminator", ),
                    optimizer_names=("discriminator", ),
                    forward_component="runtime",
                    forward_method="discriminator_step",
                    input_aliases=(("fake_audio", "fake"), ),
                    required_inputs=("fake", "target"),
                    loss_keys=("discriminator_loss", ),
                    detach_inputs=("fake", ),
                    frozen_component_paths=("runtime.generator", ),
                ),
                TrainingPhaseSpec(
                    name="duration_discriminator",
                    kind=TrainingPhaseKind.DURATION_DISCRIMINATOR,
                    component_paths=("runtime.duration_discriminator", ),
                    optimizer_names=("duration_discriminator", ),
                    forward_component="runtime",
                    forward_method="duration_step",
                    required_inputs=("duration_prediction", "target"),
                    loss_keys=("duration_discriminator_loss", ),
                    detach_inputs=("duration_prediction", ),
                    frozen_component_paths=("runtime.generator", ),
                    frequency=2,
                    offset=1,
                ),
            ),
            default_phase="generator",
        )

    def _adapter(self):
        wrapper = self.Wrapper(self._runtime)
        adapter = AutoTrainingAdapter.from_model(
            wrapper,
            spec=self._spec(),
        )
        self.assertIsInstance(adapter, VITSTrainingAdapter)
        return wrapper, adapter

    def test_adversarial_plan_routing_freeze_detach_and_cadence(self):
        import torch

        wrapper, adapter = self._adapter()
        fake_audio = torch.ones(2, 1, requires_grad=True)
        duration = torch.ones(2, 1, requires_grad=True)
        batch = {
            "model_inputs": {
                "values": torch.ones(2, 1),
                "target": torch.zeros(2, 1),
                "fake_audio": fake_audio,
                "duration_prediction": duration,
            },
        }

        first = adapter.execute_training_plan(batch, step=0)
        second = adapter.execute_training_plan(batch, step=1)
        self.assertEqual(
            [output.training_phase for output in first],
            ["generator", "discriminator"],
        )
        self.assertEqual(
            [output.training_phase for output in second],
            ["generator", "discriminator", "duration_discriminator"],
        )
        self.assertEqual(
            [output.optimizer_names for output in second],
            [
                ("generator", ),
                ("discriminator", ),
                ("duration_discriminator", ),
            ],
        )
        self.assertTrue(wrapper.runtime.observations["generator_disc_frozen"])
        self.assertTrue(wrapper.runtime.observations["discriminator_fake_detached"])
        self.assertTrue(wrapper.runtime.observations["discriminator_gen_frozen"])
        self.assertTrue(wrapper.runtime.observations["duration_input_detached"])
        self.assertTrue(wrapper.runtime.generator.weight.requires_grad)
        self.assertTrue(wrapper.runtime.discriminator.weight.requires_grad)
        self.assertTrue(fake_audio.requires_grad)
        self.assertEqual(
            wrapper.prepared_phases,
            [
                "generator",
                "discriminator",
                "generator",
                "discriminator",
                "duration_discriminator",
            ],
        )

    def test_phase_loss_keys_are_a_strict_allowlist(self):
        import torch

        class MultiLossRuntime(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))

            def forward(self, values):
                return {
                    "generator_loss": self.weight * values.mean(),
                    "discriminator_loss": self.weight * values.mean() * 2,
                }

        wrapper = self.Wrapper(MultiLossRuntime)
        spec = ModelTrainingSpec(
            model_type="dummy-loss-isolation",
            family=TrainingFamily.VITS,
            module_paths=("runtime", ),
            component_paths=("runtime", ),
            support=TrainingSupport.PREPROCESSED,
            phases=(
                TrainingPhaseSpec(
                    name="generator",
                    kind=TrainingPhaseKind.GENERATOR,
                    component_paths=("runtime", ),
                    optimizer_names=("generator", ),
                    loss_keys=("generator_loss", ),
                ), ),
        )
        adapter = VITSTrainingAdapter(wrapper, spec)
        output = adapter(values=torch.tensor([1.0]))

        self.assertEqual(tuple(output.losses), ("generator_loss", ))
        self.assertEqual(output.loss.item(), 1.0)

    def test_declared_phase_path_never_falls_back_to_primary_model(self):
        import torch

        wrapper = self.Wrapper(self._runtime)
        spec = ModelTrainingSpec(
            model_type="dummy-strict-route",
            family=TrainingFamily.ACOUSTIC,
            module_paths=("runtime", ),
            support=TrainingSupport.PREPROCESSED,
            phases=(
                TrainingPhaseSpec(
                    name="strict",
                    component_paths=("runtime.typo", ),
                    optimizer_names=("typo", ),
                    loss_keys=("loss", ),
                ), ),
        )
        adapter = AutoTrainingAdapter.from_model(wrapper, spec=spec)
        with self.assertRaisesRegex(TypeError, "declared path"):
            adapter(values=torch.ones(1, 1), target=torch.zeros(1, 1))

    def test_context_is_restored_when_phase_start_hook_raises(self):

        class FailingHookAdapter(VITSTrainingAdapter):

            def on_training_phase_start(self, context):
                raise RuntimeError("hook failed")

        wrapper = self.Wrapper(self._runtime)
        adapter = FailingHookAdapter(wrapper, self._spec())
        with self.assertRaisesRegex(RuntimeError, "hook failed"):
            adapter(
                values=__import__("torch").ones(1, 1),
                target=__import__("torch").zeros(1, 1),
            )
        self.assertIsNone(adapter.current_context)

    def test_parameter_groups_are_exact_and_collision_free(self):
        wrapper, adapter = self._adapter()
        groups = dict(adapter.named_parameter_groups())
        self.assertEqual(
            tuple(groups),
            ("generator", "discriminator", "duration_discriminator"),
        )
        parameter_ids = [id(parameter) for parameters in groups.values() for _, parameter in parameters]
        self.assertEqual(len(parameter_ids), len(set(parameter_ids)))
        self.assertEqual(len(parameter_ids), 3)
        self.assertTrue(wrapper.loaded)

    def test_shared_parameter_cannot_be_routed_to_two_optimizers(self):
        import torch

        class SharedRuntime(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.shared = torch.nn.Linear(1, 1)
                self.generator = self.shared
                self.discriminator = self.shared

            def forward(self, values, target):
                prediction = self.shared(values)
                return {"loss": (prediction - target).square().mean()}

        wrapper = self.Wrapper(SharedRuntime)
        spec = ModelTrainingSpec(
            model_type="dummy-shared-vits",
            family=TrainingFamily.VITS,
            module_paths=("runtime", ),
            support=TrainingSupport.PREPROCESSED,
            recipe_kind=TrainingRecipeKind.ADVERSARIAL,
            separate_optimizers=True,
            phases=(
                TrainingPhaseSpec(
                    name="generator",
                    kind=TrainingPhaseKind.GENERATOR,
                    component_paths=("runtime.generator", ),
                    optimizer_names=("generator", ),
                ),
                TrainingPhaseSpec(
                    name="discriminator",
                    kind=TrainingPhaseKind.DISCRIMINATOR,
                    component_paths=("runtime.discriminator", ),
                    optimizer_names=("discriminator", ),
                ),
            ),
        )
        adapter = VITSTrainingAdapter(wrapper, spec)
        with self.assertRaisesRegex(ValueError, "routed to both"):
            adapter.named_parameter_groups()

    def test_strict_versioned_state_uses_minimal_roots(self):
        _, adapter = self._adapter()
        state = adapter.state_dict()
        self.assertEqual(
            state["__voicehub_training_adapter_version__"],
            2,
        )
        self.assertEqual(state["topology"], ("runtime", ))
        self.assertEqual(tuple(state["components"]), ("runtime", ))

        _, restored = self._adapter()
        restored.load_state_dict(copy.deepcopy(state))

        legacy_state = copy.deepcopy(state)
        legacy_state["__voicehub_training_adapter_version__"] = 1
        legacy_state.pop("recipe_state")
        restored.load_state_dict(legacy_state)

        wrong_version = copy.deepcopy(state)
        wrong_version["__voicehub_training_adapter_version__"] = 999
        with self.assertRaisesRegex(ValueError, "version"):
            restored.load_state_dict(wrong_version)

        wrong_topology = copy.deepcopy(state)
        wrong_topology["topology"] = ("runtime.generator", )
        with self.assertRaisesRegex(ValueError, "topology"):
            restored.load_state_dict(wrong_topology)

    def test_gradient_checkpointing_delegates(self):
        wrapper, adapter = self._adapter()
        adapter.gradient_checkpointing_enable()
        self.assertTrue(wrapper.runtime.checkpointing)
        adapter.gradient_checkpointing_disable()
        self.assertFalse(wrapper.runtime.checkpointing)

    def test_model_inputs_collision_is_rejected(self):
        import torch

        _, adapter = self._adapter()
        with self.assertRaisesRegex(ValueError, "duplicates"):
            adapter.execute_training_plan(
                {
                    "values": torch.ones(1, 1),
                    "model_inputs": {
                        "values": torch.zeros(1, 1)
                    },
                },
                step=0,
            )


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is an optional training extra")
class SupportAndObjectiveBoundaryTests(unittest.TestCase):

    class Config:
        model_type = "dummy"

    class Wrapper:

        def __init__(self, runtime_factory):
            self.config = SupportAndObjectiveBoundaryTests.Config()
            self.runtime_factory = runtime_factory
            self.runtime = None
            self.loaded = False

        def load_for_training(self):
            self.loaded = True
            self.runtime = self.runtime_factory()

    @staticmethod
    def _regression_runtime():
        import torch

        class Runtime(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(1.0))

            def forward(self, values):
                return {"velocity": values * self.scale}

        return Runtime()

    def test_generic_adapter_gates_custom_and_inference_before_loading(self):
        for support in (
                TrainingSupport.CUSTOM,
                TrainingSupport.INFERENCE_ONLY,
        ):
            with self.subTest(support=support):
                wrapper = self.Wrapper(self._regression_runtime)
                spec = ModelTrainingSpec(
                    model_type=f"dummy-{support.value}",
                    family=TrainingFamily.FLOW_MATCHING,
                    module_paths=("runtime", ),
                    support=support,
                )
                adapter = FlowMatchingTrainingAdapter(wrapper, spec)
                with self.assertRaisesRegex(
                        ValueError,
                        "specialized|inference-only",
                ):
                    adapter.setup()
                self.assertFalse(wrapper.loaded)

    def test_serving_and_quantized_artifacts_are_rejected_before_loading(self):
        cases = (
            (
                {
                    "name_or_path": "publisher/model-q4.gguf"
                },
                "serving-only format",
            ),
            (
                {
                    "name_or_path": "publisher/model-AWQ"
                },
                "quantization-aware",
            ),
            (
                {
                    "name_or_path": "publisher/model",
                    "quantization_config": {
                        "load_in_4bit": True
                    },
                },
                "full-precision",
            ),
        )
        for values, message in cases:
            with self.subTest(values=values):
                wrapper = self.Wrapper(self._regression_runtime)
                wrapper.config = type("ArtifactConfig", (), {})()
                wrapper.config.model_type = "dummy-artifact"
                for name, value in values.items():
                    setattr(wrapper.config, name, value)
                spec = ModelTrainingSpec(
                    model_type="dummy-artifact",
                    family=TrainingFamily.FLOW_MATCHING,
                    module_paths=("runtime", ),
                    support=TrainingSupport.PREPROCESSED,
                )
                with self.assertRaisesRegex(ValueError, message):
                    FlowMatchingTrainingAdapter(wrapper, spec).setup()
                self.assertFalse(wrapper.loaded)

    def test_safetensors_identifier_is_allowed_for_training(self):
        wrapper = self.Wrapper(self._regression_runtime)
        wrapper.config = type(
            "ArtifactConfig",
            (),
            {
                "model_type": "dummy-safetensors",
                "name_or_path": "publisher/model.safetensors",
            },
        )()
        spec = ModelTrainingSpec(
            model_type="dummy-safetensors",
            family=TrainingFamily.FLOW_MATCHING,
            module_paths=("runtime", ),
            support=TrainingSupport.PREPROCESSED,
        )
        FlowMatchingTrainingAdapter(wrapper, spec).setup()
        self.assertTrue(wrapper.loaded)

    def test_loaded_quantized_module_is_rejected_as_a_backstop(self):
        runtime_factory = self._regression_runtime

        def quantized_runtime():
            runtime = runtime_factory()
            runtime.is_quantized = True
            return runtime

        wrapper = self.Wrapper(quantized_runtime)
        wrapper.config = type(
            "ArtifactConfig",
            (),
            {
                "model_type": "dummy-loaded-quantized",
                "name_or_path": "publisher/model",
            },
        )()
        spec = ModelTrainingSpec(
            model_type="dummy-loaded-quantized",
            family=TrainingFamily.FLOW_MATCHING,
            module_paths=("runtime", ),
            support=TrainingSupport.PREPROCESSED,
        )
        with self.assertRaisesRegex(TypeError, "quantized training"):
            FlowMatchingTrainingAdapter(wrapper, spec).setup()
        self.assertTrue(wrapper.loaded)

    def test_flow_mse_requires_an_explicit_velocity_target_contract(self):
        import torch

        wrapper = self.Wrapper(self._regression_runtime)
        native_only = ModelTrainingSpec(
            model_type="dummy-native-flow",
            family=TrainingFamily.FLOW_MATCHING,
            module_paths=("runtime", ),
            support=TrainingSupport.PREPROCESSED,
            phases=(
                TrainingPhaseSpec(
                    name="flow",
                    label_names=("velocity_target", ),
                    prediction_keys=("velocity", ),
                ), ),
        )
        adapter = FlowMatchingTrainingAdapter(wrapper, native_only)
        with self.assertRaisesRegex(ValueError, "no generic fallback"):
            adapter(
                values=torch.ones(2, 1),
                velocity_target=torch.zeros(2, 1),
            )

        wrapper = self.Wrapper(self._regression_runtime)
        velocity_target = ModelTrainingSpec(
            model_type="dummy-velocity-flow",
            family=TrainingFamily.FLOW_MATCHING,
            module_paths=("runtime", ),
            support=TrainingSupport.PREPROCESSED,
            phases=(
                TrainingPhaseSpec(
                    name="flow",
                    label_names=("velocity_target", ),
                    prediction_keys=("velocity", ),
                    required_inputs=("velocity_target", ),
                    fallback_objective="velocity_mse",
                ), ),
        )
        adapter = FlowMatchingTrainingAdapter(wrapper, velocity_target)
        output = adapter(
            values=torch.ones(2, 1),
            velocity_target=torch.zeros(2, 1),
        )
        self.assertGreater(output.loss.item(), 0)

    def test_token_cross_entropy_never_silently_truncates_timebases(self):
        import torch

        spec = ModelTrainingSpec(
            model_type="dummy-causal",
            family=TrainingFamily.CAUSAL_LM,
            fallback_objective="causal_cross_entropy",
        )
        adapter = CausalLMTrainingAdapter(object(), spec)
        with self.assertRaisesRegex(ValueError, "will not silently truncate"):
            adapter.compute_objective(
                torch.randn(2, 5, 11),
                torch.ones(2, 4, dtype=torch.long),
            )

    def test_regression_fallback_rejects_implicit_broadcasting(self):
        import torch

        spec = ModelTrainingSpec(
            model_type="dummy-flow-shapes",
            family=TrainingFamily.FLOW_MATCHING,
            fallback_objective="velocity_mse",
        )
        adapter = FlowMatchingTrainingAdapter(object(), spec)
        with self.assertRaisesRegex(ValueError, "implicit broadcasting"):
            adapter.compute_objective(
                torch.randn(2, 4, 1),
                torch.randn(2, 4),
            )

    def test_flow_mask_excludes_padded_target_frames(self):
        import torch

        phase = TrainingPhaseSpec(
            name="flow",
            label_names=("velocity_target", ),
            fallback_objective="velocity_mse",
        )
        spec = ModelTrainingSpec(
            model_type="dummy-flow-mask",
            family=TrainingFamily.FLOW_MATCHING,
            phases=(phase, ),
        )
        adapter = FlowMatchingTrainingAdapter(object(), spec)
        predictions = torch.tensor([
            [[1.0, 1.0], [2.0, 2.0], [100.0, 100.0]],
            [[3.0, 3.0], [100.0, 100.0], [100.0, 100.0]],
        ])
        targets = torch.zeros_like(predictions)
        mask = torch.tensor([
            [True, True, False],
            [True, False, False],
        ])
        context = TrainingContext(
            phase=phase,
            inputs={
                "velocity_target": targets,
                "velocity_target_mask": mask,
            },
        )

        actual = adapter.compute_phase_objective(
            predictions,
            targets,
            context,
        )
        expected = torch.tensor((1.0 + 1.0 + 4.0 + 4.0 + 9.0 + 9.0) / 6)
        self.assertTrue(torch.allclose(actual, expected))

    def test_module_discovery_requires_an_explicit_opt_in(self):
        import torch

        runtime_factory = self._regression_runtime

        class Nested:

            def __init__(self):
                self.backend = runtime_factory()

        wrapper = self.Wrapper(lambda: Nested())
        disabled = ModelTrainingSpec(
            model_type="dummy-no-discovery",
            family=TrainingFamily.FLOW_MATCHING,
            module_paths=("runtime.missing", ),
            support=TrainingSupport.PREPROCESSED,
        )
        with self.assertRaisesRegex(TypeError, "discovery is disabled"):
            FlowMatchingTrainingAdapter(wrapper, disabled).setup()

        wrapper = self.Wrapper(lambda: Nested())
        enabled = ModelTrainingSpec(
            model_type="dummy-discovery",
            family=TrainingFamily.FLOW_MATCHING,
            module_paths=("runtime.missing", ),
            support=TrainingSupport.PREPROCESSED,
            allow_module_discovery=True,
        )
        adapter = FlowMatchingTrainingAdapter(wrapper, enabled)
        adapter.setup()
        self.assertIsInstance(adapter.primary_model, torch.nn.Module)


if __name__ == "__main__":
    unittest.main()
