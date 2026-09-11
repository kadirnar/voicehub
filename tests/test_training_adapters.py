import importlib.util
import tempfile
import unittest
from unittest.mock import PropertyMock, patch

from voicehub import AutoModelForTextToSpeech, PreTrainedTTSModel, TTSOutput, VoiceHubConfig
from voicehub.training import (
    AutoTrainingAdapter,
    DataCollatorForTTSTraining,
    ModelTrainingSpec,
    TrainingFamily,
    TrainingPhaseKind,
    TrainingPhaseSpec,
    TrainingRecipeKind,
    TrainingSupport,
    VITSTrainingAdapter,
    get_training_spec,
    list_training_specs,
)
from voicehub.training.adapters import (
    AcousticTrainingAdapter,
    CausalLMTrainingAdapter,
    CompositeTrainingAdapter,
    FlowMatchingTrainingAdapter,
    Seq2SeqTrainingAdapter,
)
from voicehub.training.arguments import TrainingArguments
from voicehub.training.callbacks import TrainerCallback
from voicehub.training.optimization import OptimizerBundle
from voicehub.training.strategy import TorchTrainingStrategy
from voicehub.training.trainer import Trainer

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

EXPECTED_ADAPTERS = {
    TrainingFamily.CAUSAL_LM: CausalLMTrainingAdapter,
    TrainingFamily.SEQ2SEQ: Seq2SeqTrainingAdapter,
    TrainingFamily.FLOW_MATCHING: FlowMatchingTrainingAdapter,
    TrainingFamily.ACOUSTIC: AcousticTrainingAdapter,
    TrainingFamily.VITS: VITSTrainingAdapter,
    TrainingFamily.COMPOSITE: CompositeTrainingAdapter,
}


class TrainingProfileTests(unittest.TestCase):

    def test_every_registered_model_has_one_training_profile(self):
        registered = {spec.model_type for spec in AutoModelForTextToSpeech.available_models()}
        profiled = {spec.model_type for spec in list_training_specs()}
        self.assertEqual(profiled, registered)
        self.assertEqual(len(profiled), 34)

    def test_registry_connects_models_to_training_profiles(self):
        for model_spec in AutoModelForTextToSpeech.available_models():
            with self.subTest(model_type=model_spec.model_type):
                training_spec = model_spec.training
                self.assertEqual(training_spec.model_type, model_spec.model_type)
                self.assertEqual(training_spec.install_extra, "training")
                self.assertIsNone(model_spec.install_extra)

    def test_all_builtin_training_families_are_used(self):
        families = {spec.family for spec in list_training_specs(task=None)}
        represented = {
            TrainingFamily.CAUSAL_LM,
            TrainingFamily.SEQ2SEQ,
            TrainingFamily.FLOW_MATCHING,
            TrainingFamily.ACOUSTIC,
            TrainingFamily.VITS,
            TrainingFamily.COMPOSITE,
            TrainingFamily.AUDIO_CLASSIFICATION,
            TrainingFamily.UPSTREAM_NATIVE,
        }
        self.assertTrue(represented.issubset(families))

    def test_all_lazy_models_resolve_an_adapter_without_loading(self):
        for model_spec in AutoModelForTextToSpeech.available_models():
            with self.subTest(model_type=model_spec.model_type):
                model = AutoModelForTextToSpeech.from_pretrained(
                    model_type=model_spec.model_type,
                    device="cpu",
                )
                adapter = model.get_training_adapter()
                expected_class = EXPECTED_ADAPTERS[model_spec.training.family]
                self.assertIsInstance(adapter, expected_class)
                self.assertFalse(adapter.is_ready)
                self.assertFalse(model.is_loaded)

    def test_alias_resolves_training_profile(self):
        self.assertEqual(
            get_training_spec("f5-tts").model_type,
            "f5tts",
        )


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is an optional training extra")
class TrainingAdapterLoopTests(unittest.TestCase):

    class DummyForTextToSpeech(PreTrainedTTSModel):

        def __init__(self, config, module_factory):
            self.module_factory = module_factory
            super().__init__(config, device="cpu")

        def _load_pretrained_model(self) -> None:
            self.model = self.module_factory()

        def _generate(self, text: str, **kwargs) -> TTSOutput:
            return TTSOutput(audio=[0.0], sample_rate=24000)

    @staticmethod
    def _config(model_type):
        config = VoiceHubConfig(name_or_path="dummy")
        config.model_type = model_type
        return config

    @staticmethod
    def _token_model():
        import torch

        class TokenModel(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(8, 8)
                self.projection = torch.nn.Linear(8, 8)

            def forward(self, input_ids):
                return {"logits": self.projection(self.embedding(input_ids))}

        return TokenModel()

    @staticmethod
    def _regression_model():
        import torch

        class RegressionModel(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(0.5))

            def forward(self, input_values):
                return {"predictions": input_values * self.scale}

        return RegressionModel()

    @staticmethod
    def _composite_model():
        import torch

        class PhaseModel(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(0.5))

            def forward(self, input_values, labels):
                predictions = input_values * self.scale
                return {
                    "logits": predictions,
                    "loss": (predictions - labels).square().mean(),
                }

        return PhaseModel()

    @classmethod
    def _composite_runtime(cls):
        import torch

        class Runtime:

            def __init__(self):
                self.generator = cls._composite_model()
                self.model = self.generator
                self.discriminator = cls._composite_model()

        return Runtime()

    @staticmethod
    def _composite_spec():
        return ModelTrainingSpec(
            model_type="dummy-composite",
            family=TrainingFamily.COMPOSITE,
            module_paths=("model.generator", "model.discriminator"),
            component_paths=("model.generator", "model.discriminator"),
            support=TrainingSupport.PREPROCESSED,
            separate_optimizers=True,
            recipe_kind=TrainingRecipeKind.ADVERSARIAL,
            phases=(
                TrainingPhaseSpec(
                    name="generator",
                    kind=TrainingPhaseKind.GENERATOR,
                    component_paths=("model.generator", ),
                    optimizer_names=("generator", ),
                    forward_component="model.generator",
                    loss_keys=("loss", ),
                    frozen_component_paths=("model.discriminator", ),
                ),
                TrainingPhaseSpec(
                    name="discriminator",
                    kind=TrainingPhaseKind.DISCRIMINATOR,
                    component_paths=("model.discriminator", ),
                    optimizer_names=("discriminator", ),
                    forward_component="model.discriminator",
                    loss_keys=("loss", ),
                    frozen_component_paths=("model.generator", ),
                ),
            ),
        )

    def _train_once(self, model_type, module_factory, dataset, *, spec=None):
        model = self.DummyForTextToSpeech(
            self._config(model_type),
            module_factory,
        )
        training_adapter = (AutoTrainingAdapter.from_model(model, spec=spec) if spec is not None else None)
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(
                model=model,
                args=TrainingArguments(
                    output_dir=directory,
                    max_steps=1,
                    per_device_train_batch_size=2,
                    logging_strategy="no",
                    save_strategy="no",
                    use_cpu=True,
                ),
                train_dataset=dataset,
                training_adapter=training_adapter,
            )
            output = trainer.train()
        self.assertEqual(output.global_step, 1)
        self.assertIsNotNone(trainer.training_adapter)
        self.assertTrue(trainer.training_adapter.is_ready)
        return trainer

    def test_token_and_sequence_families_train(self):
        import torch

        dataset = [
            {
                "input_ids": torch.tensor([1, 2, 3, 4]),
                "labels": torch.tensor([1, 2, 3, 4]),
            },
            {
                "input_ids": torch.tensor([2, 3, 4]),
                "labels": torch.tensor([2, 3, 4]),
            },
        ]
        causal = self._train_once(
            "orpheustts",
            self._token_model,
            dataset,
        )
        sequence = self._train_once(
            "parlertts",
            self._token_model,
            dataset,
        )
        self.assertIsInstance(
            causal.training_adapter,
            CausalLMTrainingAdapter,
        )
        self.assertIsInstance(
            sequence.training_adapter,
            Seq2SeqTrainingAdapter,
        )

    def test_model_created_tensors_are_prepared_by_the_active_strategy(self):
        import torch

        class RawBatchModel(self.DummyForTextToSpeech):

            def prepare_training_inputs(self, inputs, *, phase):
                del phase
                batch_size = int(inputs["raw_marker"].shape[0])
                input_ids = torch.tensor(
                    [[1, 2, 3, 4]],
                    dtype=torch.long,
                ).expand(batch_size, -1)
                return {
                    "input_ids": input_ids,
                    "labels": input_ids.clone(),
                }

        class RecordingStrategy(TorchTrainingStrategy):

            def __init__(self):
                super().__init__()
                self.model_created_batches = 0
                self.requested_devices = []

            def prepare_input(self, value, *, device):
                self.requested_devices.append(device)
                if isinstance(value, dict) and "input_ids" in value:
                    self.model_created_batches += 1
                    self.assert_cpu_tensors(value)
                return value

            @staticmethod
            def assert_cpu_tensors(value):
                for item in value.values():
                    if torch.is_tensor(item):
                        if item.device.type != "cpu":
                            raise AssertionError("The test processor must create CPU tensors.")

            def prepare_device(self, model, *, device):
                # The explicit device hook owns placement before graph
                # optimization. Keep tensors on CPU in this CPU-only test
                # while recording the requested production device.
                self.requested_devices.append(device)
                return model

            def prepare_training_adapter(self, adapter, *, device):
                self.requested_devices.append(device)
                return adapter

        spec = ModelTrainingSpec(
            model_type="device-placement-test",
            family=TrainingFamily.CAUSAL_LM,
            module_paths=("model", ),
            component_paths=("model", ),
            label_names=("labels", ),
            support=TrainingSupport.PREPROCESSED,
        )
        model = RawBatchModel(
            self._config("device-placement-test"),
            self._token_model,
        )
        adapter = CausalLMTrainingAdapter(model, spec)
        strategy = RecordingStrategy()

        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(
                model=model,
                args=TrainingArguments(
                    output_dir=directory,
                    max_steps=1,
                    per_device_train_batch_size=2,
                    logging_strategy="no",
                    save_strategy="no",
                    use_cpu=False,
                    dataloader_pin_memory=False,
                ),
                train_dataset=[
                    {
                        "raw_marker": torch.tensor([1])
                    },
                    {
                        "raw_marker": torch.tensor([2])
                    },
                ],
                training_adapter=adapter,
                training_strategy=strategy,
            )
            with patch.object(
                    TrainingArguments,
                    "device",
                    new_callable=PropertyMock,
                    return_value="cuda",
            ):
                trainer.train()

        self.assertGreaterEqual(strategy.model_created_batches, 1)
        self.assertTrue(strategy.requested_devices)
        self.assertEqual(set(strategy.requested_devices), {"cuda"})

    def test_adapter_prediction_accepts_unlabeled_batches(self):
        import torch

        model = self.DummyForTextToSpeech(
            self._config("orpheustts"),
            self._token_model,
        )
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(
                model=model,
                args=TrainingArguments(
                    output_dir=directory,
                    per_device_eval_batch_size=2,
                    use_cpu=True,
                ),
            )
            output = trainer.predict([
                {
                    "input_ids": torch.tensor([1, 2, 3])
                },
                {
                    "input_ids": torch.tensor([2, 3, 4])
                },
            ])

        self.assertIsNone(output.label_ids)
        self.assertNotIn("test_loss", output.metrics)
        self.assertEqual(tuple(output.predictions.shape), (2, 3, 8))

    def test_adapter_device_move_notifies_wrapper_runtime(self):
        import torch

        class Runtime(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(()))

            def forward(self, input_ids):
                return {"logits": input_ids * self.weight}

        model = self.DummyForTextToSpeech(
            self._config("orpheustts"),
            Runtime,
        )
        devices = []
        model._set_training_device = devices.append
        adapter = model.get_training_adapter()
        adapter.to("cpu")

        self.assertEqual(devices, ["cpu"])

    def test_flow_and_acoustic_families_train(self):
        import torch

        dataset = [
            {
                "input_values": torch.tensor([0.2, 0.4, 0.6]),
                "labels": torch.tensor([0.4, 0.8, 1.2]),
            },
            {
                "input_values": torch.tensor([0.1, 0.3]),
                "labels": torch.tensor([0.2, 0.6]),
            },
        ]
        flow = self._train_once(
            "dummy-flow",
            self._regression_model,
            dataset,
            spec=ModelTrainingSpec(
                model_type="dummy-flow",
                family=TrainingFamily.FLOW_MATCHING,
                module_paths=("model", ),
                support=TrainingSupport.PREPROCESSED,
                fallback_objective="velocity_mse",
            ),
        )
        acoustic = self._train_once(
            "dummy-acoustic",
            self._regression_model,
            dataset,
            spec=ModelTrainingSpec(
                model_type="dummy-acoustic",
                family=TrainingFamily.ACOUSTIC,
                module_paths=("model", ),
                support=TrainingSupport.PREPROCESSED,
            ),
        )
        self.assertIsInstance(
            flow.training_adapter,
            FlowMatchingTrainingAdapter,
        )
        self.assertIsInstance(
            acoustic.training_adapter,
            AcousticTrainingAdapter,
        )

    def test_composite_family_uses_named_source_losses(self):
        import torch

        dataset = [
            {
                "input_values": torch.tensor([0.2, 0.4]),
                "labels": torch.tensor([0.4, 0.8]),
            },
            {
                "input_values": torch.tensor([0.1, 0.3]),
                "labels": torch.tensor([0.2, 0.6]),
            },
        ]
        trainer = self._train_once(
            "dummy-composite",
            self._composite_runtime,
            dataset,
            spec=self._composite_spec(),
        )
        self.assertIsInstance(
            trainer.training_adapter,
            CompositeTrainingAdapter,
        )
        self.assertIsInstance(trainer.optimizer, OptimizerBundle)
        self.assertEqual(len(trainer.optimizer.optimizers), 2)

    def test_vits_phase_boundaries_step_discriminator_before_generator(self):
        import torch

        class SequentialRuntime(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.discriminator = torch.nn.Linear(1, 1, bias=False)
                self.generator = torch.nn.Linear(1, 1, bias=False)
                torch.nn.init.zeros_(self.discriminator.weight)
                torch.nn.init.zeros_(self.generator.weight)
                self.generator_observed_discriminator = None

            def discriminator_step(self, input_values, labels):
                score = self.discriminator(input_values)
                return {
                    "discriminator_loss": (score - labels).square().mean(),
                }

            def generator_step(self, input_values, labels):
                del labels
                self.generator_observed_discriminator = float(self.discriminator.weight.detach().item())
                generated = self.generator(input_values)
                target = self.discriminator.weight.detach().expand_as(generated)
                return {
                    "generator_loss": (generated - target).square().mean(),
                }

        spec = ModelTrainingSpec(
            model_type="dummy-sequential-vits",
            family=TrainingFamily.VITS,
            module_paths=("model", ),
            component_paths=(
                "model.discriminator",
                "model.generator",
            ),
            support=TrainingSupport.PREPROCESSED,
            separate_optimizers=True,
            recipe_kind=TrainingRecipeKind.ADVERSARIAL,
            phases=(
                TrainingPhaseSpec(
                    name="discriminator",
                    kind=TrainingPhaseKind.DISCRIMINATOR,
                    component_paths=("model.discriminator", ),
                    optimizer_names=("discriminator", ),
                    forward_component="model",
                    forward_method="discriminator_step",
                    required_inputs=("input_values", "labels"),
                    loss_keys=("discriminator_loss", ),
                    frozen_component_paths=("model.generator", ),
                    optimizer_step_after_phase=True,
                ),
                TrainingPhaseSpec(
                    name="generator",
                    kind=TrainingPhaseKind.GENERATOR,
                    component_paths=("model.generator", ),
                    optimizer_names=("generator", ),
                    forward_component="model",
                    forward_method="generator_step",
                    required_inputs=("input_values", "labels"),
                    loss_keys=("generator_loss", ),
                    frozen_component_paths=("model.discriminator", ),
                    optimizer_step_after_phase=True,
                ),
            ),
        )
        dataset = [
            {
                "input_values": torch.ones(1),
                "labels": torch.ones(1),
            },
            {
                "input_values": torch.ones(1),
                "labels": torch.ones(1),
            },
        ]

        class PartialStepStrategy(TorchTrainingStrategy):

            def __init__(self, outcomes):
                super().__init__()
                self.outcomes = list(outcomes)
                self.calls = 0

            def optimizer_step(
                self,
                optimizer,
                *,
                scaler=None,
                optimizer_names=None,
            ):
                self.calls += 1
                outcome = self.outcomes.pop(0)
                if not outcome:
                    return False
                return super().optimizer_step(
                    optimizer,
                    scaler=scaler,
                    optimizer_names=optimizer_names,
                )

        def train_once(outcomes=None):
            model = self.DummyForTextToSpeech(
                self._config("dummy-sequential-vits"),
                SequentialRuntime,
            )
            adapter = VITSTrainingAdapter(model, spec)
            strategy = (TorchTrainingStrategy() if outcomes is None else PartialStepStrategy(outcomes))
            with tempfile.TemporaryDirectory() as directory:
                trainer = Trainer(
                    model=model,
                    args=TrainingArguments(
                        output_dir=directory,
                        max_steps=1,
                        per_device_train_batch_size=2,
                        logging_strategy="no",
                        save_strategy="no",
                        max_grad_norm=0,
                        use_cpu=True,
                    ),
                    train_dataset=dataset,
                    training_adapter=adapter,
                    training_strategy=strategy,
                    optimizer_cls_and_kwargs=(
                        torch.optim.SGD,
                        {
                            "lr": 0.5
                        },
                    ),
                )
                output = trainer.train()
            return output, model, strategy

        output, model, _ = train_once()

        self.assertEqual(output.global_step, 1)
        self.assertAlmostEqual(
            model.model.generator_observed_discriminator,
            1.0,
        )
        self.assertAlmostEqual(model.model.discriminator.weight.item(), 1.0)
        self.assertAlmostEqual(model.model.generator.weight.item(), 1.0)

        for outcomes in ([True, False], [False, True]):
            with self.subTest(outcomes=outcomes):
                output, partial_model, strategy = train_once(outcomes)
                self.assertEqual(output.global_step, 1)
                self.assertEqual(strategy.calls, 2)
                expected_discriminator = 1.0 if outcomes[0] else 0.0
                self.assertAlmostEqual(
                    partial_model.model.discriminator.weight.item(),
                    expected_discriminator,
                )

    def test_scheduler_sizing_counts_each_sequential_optimizer_boundary(self):
        import torch

        phases = tuple(
            TrainingPhaseSpec(
                name=f"phase-{index}",
                kind=TrainingPhaseKind.GENERATOR,
                component_paths=("model.scale", ),
                optimizer_names=("shared", ),
                optimizer_step_after_phase=True,
            ) for index in range(2))
        spec = ModelTrainingSpec(
            model_type="dummy-shared-boundaries",
            family=TrainingFamily.VITS,
            component_paths=("model.scale", ),
            support=TrainingSupport.PREPROCESSED,
            separate_optimizers=True,
            phases=phases,
        )
        model = self.DummyForTextToSpeech(
            self._config("dummy-shared-boundaries"),
            self._regression_model,
        )
        adapter = VITSTrainingAdapter(model, spec)
        trainer = Trainer(
            model=model,
            args=TrainingArguments(use_cpu=True),
            training_adapter=adapter,
        )

        self.assertEqual(
            trainer._optimizer_training_steps("shared", 3),
            6,
        )

    def test_sequential_boundaries_reject_unrouted_scheduled_phases(self):
        import torch

        spec = ModelTrainingSpec(
            model_type="dummy-unrouted-boundary",
            family=TrainingFamily.VITS,
            component_paths=("model.scale", ),
            support=TrainingSupport.PREPROCESSED,
            separate_optimizers=True,
            phases=(
                TrainingPhaseSpec(
                    name="unrouted",
                    component_paths=("model.scale", ),
                ),
                TrainingPhaseSpec(
                    name="generator",
                    kind=TrainingPhaseKind.GENERATOR,
                    component_paths=("model.scale", ),
                    optimizer_names=("generator", ),
                    optimizer_step_after_phase=True,
                ),
            ),
        )
        model = self.DummyForTextToSpeech(
            self._config("dummy-unrouted-boundary"),
            self._regression_model,
        )
        adapter = VITSTrainingAdapter(model, spec)
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
                    "labels": torch.ones(1),
                }],
                training_adapter=adapter,
            )
            with self.assertRaisesRegex(ValueError, "route every scheduled"):
                trainer.train()

    def test_variable_length_collator_pads_tokens_labels_and_audio(self):
        import torch

        collator = DataCollatorForTTSTraining()
        batch = collator([
            {
                "input_ids": torch.tensor([1, 2, 3]),
                "labels": torch.tensor([4, 5, 6]),
                "input_values": torch.tensor([0.1, 0.2, 0.3]),
            },
            {
                "input_ids": torch.tensor([7, 8]),
                "labels": torch.tensor([9, 10]),
                "input_values": torch.tensor([0.4, 0.5]),
            },
        ])
        self.assertEqual(batch["input_ids"].tolist(), [[1, 2, 3], [7, 8, 0]])
        self.assertEqual(
            batch["labels"].tolist(),
            [[4, 5, 6], [9, 10, -100]],
        )
        self.assertEqual(
            batch["attention_mask"].tolist(),
            [[True, True, True], [True, True, False]],
        )
        self.assertEqual(tuple(batch["input_values"].shape), (2, 3))

        spectrograms = collator([
            {
                "input_values": torch.ones(80, 3)
            },
            {
                "input_values": torch.ones(80, 2)
            },
        ])
        self.assertEqual(
            tuple(spectrograms["input_values"].shape),
            (2, 80, 3),
        )

    def test_composite_optimizer_and_scheduler_resume_together(self):
        import torch

        class StopAfterOne(TrainerCallback):

            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step == 1:
                    control.should_save = True
                    control.should_training_stop = True
                return control

        dataset = [
            {
                "input_values": torch.tensor([0.2, 0.4]),
                "labels": torch.tensor([0.4, 0.8]),
            },
            {
                "input_values": torch.tensor([0.1, 0.3]),
                "labels": torch.tensor([0.2, 0.6]),
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            first = Trainer(
                model=(
                    first_model := self.DummyForTextToSpeech(
                        self._config("dummy-composite"),
                        self._composite_runtime,
                    )),
                args=TrainingArguments(
                    output_dir=directory,
                    max_steps=2,
                    per_device_train_batch_size=2,
                    logging_strategy="no",
                    save_steps=1,
                    use_cpu=True,
                ),
                train_dataset=dataset,
                training_adapter=AutoTrainingAdapter.from_model(
                    first_model,
                    spec=self._composite_spec(),
                ),
                callbacks=[StopAfterOne],
            )
            first.train()

            resumed = Trainer(
                model=(
                    resumed_model := self.DummyForTextToSpeech(
                        self._config("dummy-composite"),
                        self._composite_runtime,
                    )),
                args=TrainingArguments(
                    output_dir=directory,
                    max_steps=2,
                    per_device_train_batch_size=2,
                    logging_strategy="no",
                    save_steps=1,
                    use_cpu=True,
                ),
                train_dataset=dataset,
                training_adapter=AutoTrainingAdapter.from_model(
                    resumed_model,
                    spec=self._composite_spec(),
                ),
            )
            output = resumed.train(resume_from_checkpoint=True)

        self.assertEqual(output.global_step, 2)
        self.assertIsInstance(resumed.optimizer, OptimizerBundle)

    def test_inflect_warm_start_requires_acknowledgement_before_loading(self):

        class Runtime:

            def __init__(self, nested):
                self.nested = nested

        model = self.DummyForTextToSpeech(
            self._config("inflecttts"),
            lambda: Runtime(self._regression_model()),
        )
        adapter = AutoTrainingAdapter.from_model(model)
        with self.assertRaisesRegex(ValueError, "enable_native_finetuning=True"):
            adapter.setup()
        self.assertFalse(model.is_loaded)

    def test_model_init_preserves_explicit_data_collator(self):
        import torch

        class NativeRegression(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(0.5))

            def forward(self, input_values, labels):
                predictions = input_values * self.scale
                return {
                    "loss": (predictions - labels).square().mean(),
                    "logits": predictions,
                }

        collator_calls = []

        def collator(features):
            collator_calls.append(len(features))
            return {
                "input_values": torch.stack([feature["input_values"] for feature in features]),
                "labels": torch.stack([feature["labels"] for feature in features]),
            }

        trainer = Trainer(
            model_init=NativeRegression,
            args=TrainingArguments(
                max_steps=1,
                per_device_train_batch_size=2,
                logging_strategy="no",
                save_strategy="no",
                use_cpu=True,
            ),
            data_collator=collator,
            train_dataset=[
                {
                    "input_values": torch.tensor([0.1, 0.2]),
                    "labels": torch.tensor([0.2, 0.4]),
                },
                {
                    "input_values": torch.tensor([0.3, 0.4]),
                    "labels": torch.tensor([0.6, 0.8]),
                },
            ],
        )
        trainer.train()
        self.assertIs(trainer.data_collator, collator)
        self.assertEqual(collator_calls, [2])


if __name__ == "__main__":
    unittest.main()
