import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from voicehub import PreTrainedTTSModel, TTSOutput, VoiceHubConfig
from voicehub.training import AutoTrainingAdapter, DataCollatorForTTSTraining
from voicehub.training.adapters import CausalLMTrainingAdapter, VITSTrainingAdapter
from voicehub.training.arguments import TrainingArguments
from voicehub.training.callbacks import EarlyStoppingCallback, TrainerCallback
from voicehub.training.contracts import TrainingPhaseKind, TrainingPhaseSpec, TrainingSupport
from voicehub.training.optimization import OptimizerBundle, SchedulerBundle
from voicehub.training.specs import ModelTrainingSpec, TrainingFamily
from voicehub.training.strategy import TorchTrainingStrategy, TrainingStrategy
from voicehub.training.trainer import Trainer
from voicehub.training.utils import (
    CHECKPOINT_COMPLETE_NAME,
    CHECKPOINT_FORMAT_VERSION,
    CHECKPOINT_MANIFEST_NAME,
    MODEL_STATE_NAME,
    NATIVE_EXPORT_DIR,
    SCALER_STATE_NAME,
    TRAINING_RECIPE_NAME,
    TRAINING_RUNTIME_STATE_NAME,
    get_last_checkpoint,
)

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is an optional training extra")
class TrainingRuntimeTests(unittest.TestCase):

    @staticmethod
    def _dataset(length=8):
        import torch

        return [{
            "input_values": torch.tensor([float(index + 1)]),
            "labels": torch.tensor([float((index + 1) * 2)]),
        } for index in range(length)]

    @staticmethod
    def _dropout_model():
        import torch

        class DropoutModel(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.projection = torch.nn.Linear(1, 1)
                self.dropout = torch.nn.Dropout(0.25)

            def forward(self, input_values, labels):
                predictions = self.projection(self.dropout(input_values))
                return {
                    "loss": torch.nn.functional.mse_loss(
                        predictions,
                        labels,
                    ),
                    "logits": predictions,
                }

        return DropoutModel()

    def test_named_optimizer_route_updates_only_selected_component(self):
        import torch

        class RoutedModel(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.first = torch.nn.Parameter(torch.tensor(0.0))
                self.second = torch.nn.Parameter(torch.tensor(0.0))

            def forward(self, input_values):
                loss = (self.first - 1).square() + (self.second - 1).square()
                return {
                    "loss": loss,
                    "logits": input_values,
                    "optimizer_names": ("first", ),
                }

        model = RoutedModel()
        optimizers = OptimizerBundle({
            "first": torch.optim.SGD([model.first], lr=0.5),
            "second": torch.optim.SGD([model.second], lr=0.5),
        })
        schedulers = SchedulerBundle({
            name: torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
            for name, optimizer in optimizers.optimizers.items()
        })
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(
                model=model,
                args=TrainingArguments(
                    output_dir=directory,
                    max_steps=1,
                    per_device_train_batch_size=1,
                    max_grad_norm=0,
                    logging_strategy="no",
                    save_strategy="no",
                    use_cpu=True,
                ),
                train_dataset=[{
                    "input_values": torch.tensor([0.0])
                }],
                optimizers=(optimizers, schedulers),
            )
            trainer.train()

        self.assertEqual(model.first.item(), 1.0)
        self.assertEqual(model.second.item(), 0.0)
        self.assertIsNone(model.second.grad)

    def test_dataset_owned_collator_is_used_by_default(self):
        import torch

        class DatasetWithCollator:

            def __init__(self):
                self.records = [
                    {
                        "value": 1.0,
                    },
                    {
                        "value": 2.0,
                    },
                ]
                self.collator_calls = 0

            def __len__(self):
                return len(self.records)

            def __getitem__(self, index):
                return self.records[index]

            def collate_fn(self, records):
                self.collator_calls += 1
                values = torch.tensor([record["value"] for record in records]).unsqueeze(-1)
                return {
                    "input_values": values,
                    "labels": values * 2,
                }

        dataset = DatasetWithCollator()
        with tempfile.TemporaryDirectory() as directory:
            Trainer(
                model=self._dropout_model(),
                args=TrainingArguments(
                    output_dir=directory,
                    max_steps=1,
                    per_device_train_batch_size=2,
                    logging_strategy="no",
                    save_strategy="no",
                    use_cpu=True,
                ),
                train_dataset=dataset,
            ).train()

        self.assertEqual(dataset.collator_calls, 1)

    def test_named_routes_survive_an_opaque_optimizer_proxy(self):
        import torch

        class RoutedModel(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.first = torch.nn.Parameter(torch.tensor(0.0))
                self.second = torch.nn.Parameter(torch.tensor(0.0))

            def forward(self, input_values):
                return {
                    "loss": (self.first - 1).square() + (self.second - 1).square(),
                    "logits": input_values,
                    "optimizer_names": ("first", ),
                }

        class OptimizerProxy:

            def __init__(self, bundle):
                self.bundle = bundle

            @property
            def param_groups(self):
                return self.bundle.param_groups

            def zero_grad(self, set_to_none=True):
                self.bundle.zero_grad(set_to_none=set_to_none)

            def state_dict(self):
                return self.bundle.state_dict()

            def load_state_dict(self, state_dict):
                self.bundle.load_state_dict(state_dict)

        class OpaqueOptimizerStrategy(TorchTrainingStrategy):

            def __init__(self):
                super().__init__()
                self.bundle = None
                self.routes = []

            def prepare_optimization(self, model, optimizer, scheduler):
                self.bundle = optimizer
                return model, OptimizerProxy(optimizer), scheduler

            def normalize_gradients(self, optimizer, microstep_counts):
                super().normalize_gradients(
                    self.bundle,
                    microstep_counts,
                )

            def optimizer_step(self, optimizer, **kwargs):
                names = kwargs.get("optimizer_names")
                self.routes.append(names)
                self.bundle.step(names=names)
                return True

        model = RoutedModel()
        optimizers = OptimizerBundle({
            "first": torch.optim.SGD([model.first], lr=0.5),
            "second": torch.optim.SGD([model.second], lr=0.5),
        })
        schedulers = SchedulerBundle({
            name: torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
            for name, optimizer in optimizers.optimizers.items()
        })
        strategy = OpaqueOptimizerStrategy()
        with tempfile.TemporaryDirectory() as directory:
            Trainer(
                model=model,
                args=TrainingArguments(
                    output_dir=directory,
                    max_steps=1,
                    per_device_train_batch_size=1,
                    max_grad_norm=0,
                    logging_strategy="no",
                    save_strategy="no",
                    use_cpu=True,
                ),
                train_dataset=[{
                    "input_values": torch.tensor([0.0])
                }],
                optimizers=(optimizers, schedulers),
                training_strategy=strategy,
            ).train()

        self.assertEqual(strategy.routes, [("first", )])
        self.assertEqual(model.first.item(), 1.0)
        self.assertEqual(model.second.item(), 0.0)

    def test_gradient_clipping_uses_only_selected_optimizer_parameters(self):
        import torch

        class RoutedModel(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.first = torch.nn.Parameter(torch.tensor(0.0))
                self.second = torch.nn.Parameter(torch.tensor(0.0))

            def forward(self, input_values):
                return {
                    "loss": ((self.first - 1).square() + (self.second - 1).square()),
                    "logits": input_values,
                    "optimizer_names": ("first", ),
                }

        model = RoutedModel()
        optimizers = OptimizerBundle({
            "first": torch.optim.SGD([model.first], lr=0.5),
            "second": torch.optim.SGD([model.second], lr=0.5),
        })
        schedulers = SchedulerBundle({
            name: torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
            for name, optimizer in optimizers.optimizers.items()
        })
        with tempfile.TemporaryDirectory() as directory:
            Trainer(
                model=model,
                args=TrainingArguments(
                    output_dir=directory,
                    max_steps=1,
                    per_device_train_batch_size=1,
                    max_grad_norm=1.0,
                    logging_strategy="no",
                    save_strategy="no",
                    use_cpu=True,
                ),
                train_dataset=[{
                    "input_values": torch.tensor([0.0])
                }],
                optimizers=(optimizers, schedulers),
            ).train()

        self.assertAlmostEqual(model.first.item(), 0.5, places=5)
        self.assertEqual(model.second.item(), 0.0)

    def test_mixed_precision_overflow_skips_all_named_optimizers(self):
        import torch

        first = torch.nn.Parameter(torch.tensor(1.0))
        second = torch.nn.Parameter(torch.tensor(1.0))
        first.grad = torch.tensor(1.0)
        second.grad = torch.tensor(float("inf"))
        optimizers = OptimizerBundle({
            "first": torch.optim.SGD([first], lr=1.0),
            "second": torch.optim.SGD([second], lr=1.0),
        })

        class FakeScaler:

            def __init__(self):
                self.scale = 2.0
                self.optimizers = []
                self.step_calls = 0

            def get_scale(self):
                return self.scale

            def unscale_(self, optimizer):
                self.optimizers.append(optimizer)

            def step(self, optimizer):
                self.step_calls += 1
                optimizer.step()

            def update(self):
                gradients = [
                    parameter.grad for optimizer in self.optimizers for group in optimizer.param_groups
                    for parameter in group["params"] if parameter.grad is not None
                ]
                if any(not torch.isfinite(value).all() for value in gradients):
                    self.scale /= 2

        scaler = FakeScaler()
        did_step = TorchTrainingStrategy().optimizer_step(
            optimizers,
            scaler=scaler,
            optimizer_names=("first", "second"),
        )

        self.assertFalse(did_step)
        self.assertEqual(scaler.step_calls, 0)
        self.assertEqual(first.item(), 1.0)
        self.assertEqual(second.item(), 1.0)

    def test_gradient_accumulation_averages_each_optimizer_by_active_batches(self):
        import torch

        class AlternatingModel(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.generator = torch.nn.Parameter(torch.tensor(0.0))
                self.discriminator = torch.nn.Parameter(torch.tensor(0.0))

            def forward(self, input_values, training_phase):
                del input_values
                parameter = getattr(self, training_phase)
                return {
                    "loss": (parameter - 1).square(),
                    "optimizer_names": (training_phase, ),
                }

        model = AlternatingModel()
        optimizers = OptimizerBundle({
            "generator": torch.optim.SGD([model.generator], lr=0.5),
            "discriminator": torch.optim.SGD([model.discriminator], lr=0.5),
        })
        schedulers = SchedulerBundle({
            name: torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
            for name, optimizer in optimizers.optimizers.items()
        })
        dataset = [{
            "input_values": torch.tensor([0.0]),
            "training_phase": phase,
        } for phase in (
            "generator",
            "discriminator",
            "generator",
            "discriminator",
        )]
        with tempfile.TemporaryDirectory() as directory:
            Trainer(
                model=model,
                data_collator=DataCollatorForTTSTraining(),
                args=TrainingArguments(
                    output_dir=directory,
                    max_steps=1,
                    gradient_accumulation_steps=4,
                    per_device_train_batch_size=1,
                    max_grad_norm=0,
                    logging_strategy="no",
                    save_strategy="no",
                    use_cpu=True,
                ),
                train_dataset=dataset,
                optimizers=(optimizers, schedulers),
            ).train()

        self.assertEqual(model.generator.item(), 1.0)
        self.assertEqual(model.discriminator.item(), 1.0)

    def test_trainer_requests_training_runtime_before_ordinary_load(self):
        import torch

        class NativeLossModel(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(0.5))

            def forward(self, input_values, labels):
                predictions = input_values * self.scale
                return {
                    "loss": (predictions - labels).square().mean(),
                    "logits": predictions,
                }

        class TrainingAwareWrapper(PreTrainedTTSModel):

            def __init__(self, config):
                self.load_modes = []
                super().__init__(config, device="cpu")

            def _load_pretrained_model(self):
                self.load_modes.append(self.is_training_load)
                self.model = NativeLossModel()

            def _generate(self, text, **kwargs):
                return TTSOutput(audio=[0.0], sample_rate=24000)

        config = VoiceHubConfig(name_or_path="dummy")
        config.model_type = "orpheustts"
        model = TrainingAwareWrapper(config)
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(
                model=model,
                args=TrainingArguments(
                    output_dir=directory,
                    max_steps=1,
                    per_device_train_batch_size=2,
                    max_grad_norm=0,
                    logging_strategy="no",
                    save_strategy="no",
                    use_cpu=True,
                ),
                train_dataset=self._dataset(2),
            )
            trainer.train()

        self.assertEqual(model.load_modes, [True])

    def test_explicit_custom_adapter_is_an_authoritative_recipe(self):
        import torch

        class NativeLossModel(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(0.5))

            def forward(self, input_values, labels):
                predictions = input_values * self.scale
                return {
                    "loss": (predictions - labels).square().mean(),
                    "logits": predictions,
                }

        class CustomWrapper(PreTrainedTTSModel):

            def _load_pretrained_model(self):
                self.model = NativeLossModel()

            def _generate(self, text, **kwargs):
                return TTSOutput(audio=[0.0], sample_rate=24000)

        spec = ModelTrainingSpec(
            model_type="dummy-custom-recipe",
            family=TrainingFamily.CAUSAL_LM,
            module_paths=("model", ),
            support=TrainingSupport.CUSTOM,
        )
        config = VoiceHubConfig(name_or_path="dummy")
        config.model_type = spec.model_type
        model = CustomWrapper(config, device="cpu")

        class ExplicitCustomAdapter(CausalLMTrainingAdapter):
            supports_custom_recipe = True

        adapter = ExplicitCustomAdapter(
            model,
            spec,
        )
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(
                model=model,
                training_adapter=adapter,
                args=TrainingArguments(
                    output_dir=directory,
                    max_steps=1,
                    per_device_train_batch_size=2,
                    max_grad_norm=0,
                    logging_strategy="no",
                    save_strategy="no",
                    use_cpu=True,
                ),
                train_dataset=self._dataset(2),
            )
            output = trainer.train()

        self.assertEqual(output.global_step, 1)

    def test_generic_adapter_cannot_bypass_custom_recipe_gate(self):

        class CustomWrapper(PreTrainedTTSModel):

            def _load_pretrained_model(self):
                raise AssertionError("The support gate must run before loading.")

            def _generate(self, text, **kwargs):
                return TTSOutput(audio=[0.0], sample_rate=24000)

        spec = ModelTrainingSpec(
            model_type="dummy-gated-recipe",
            family=TrainingFamily.CAUSAL_LM,
            module_paths=("model", ),
            support=TrainingSupport.CUSTOM,
        )
        config = VoiceHubConfig(name_or_path="dummy")
        config.model_type = spec.model_type
        model = CustomWrapper(config, device="cpu")
        adapter = CausalLMTrainingAdapter(model, spec)
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(
                model=model,
                training_adapter=adapter,
                args=TrainingArguments(
                    output_dir=directory,
                    max_steps=1,
                    use_cpu=True,
                ),
                train_dataset=self._dataset(1),
            )

            with self.assertRaisesRegex(ValueError, "specialized training adapter"):
                trainer.train()

    def test_variable_length_evaluation_pads_across_batches(self):
        import torch

        class IdentityModel(torch.nn.Module):

            def forward(self, input_values, labels):
                return {
                    "loss": torch.nn.functional.mse_loss(
                        input_values,
                        labels,
                    ),
                    "logits": input_values,
                }

        dataset = [
            {
                "input_values": torch.tensor([1.0, 2.0]),
                "labels": torch.tensor([1.0, 2.0]),
            },
            {
                "input_values": torch.tensor([3.0, 4.0, 5.0]),
                "labels": torch.tensor([3.0, 4.0, 5.0]),
            },
        ]
        trainer = Trainer(
            model=IdentityModel(),
            args=TrainingArguments(
                per_device_eval_batch_size=1,
                use_cpu=True,
            ),
        )
        output = trainer.predict(dataset)
        self.assertEqual(output.predictions.shape, (2, 3))
        self.assertEqual(output.predictions[0].tolist(), [1.0, 2.0, 0.0])
        self.assertEqual(output.label_ids[0].tolist(), [1.0, 2.0, -100.0])

    def test_evaluation_loss_is_weighted_by_observed_samples(self):
        import torch

        class BatchMeanModel(torch.nn.Module):

            def forward(self, input_values, labels):
                return {
                    "loss": input_values.mean(),
                    "logits": input_values,
                }

        dataset = [
            {
                "input_values": torch.tensor([1.0]),
                "labels": torch.tensor([0.0]),
            },
            {
                "input_values": torch.tensor([1.0]),
                "labels": torch.tensor([0.0]),
            },
            {
                "input_values": torch.tensor([9.0]),
                "labels": torch.tensor([0.0]),
            },
        ]
        trainer = Trainer(
            model=BatchMeanModel(),
            args=TrainingArguments(
                per_device_eval_batch_size=2,
                use_cpu=True,
            ),
        )
        metrics = trainer.evaluate(dataset)
        self.assertAlmostEqual(metrics["eval_loss"], 11.0 / 3.0)

    def test_loss_only_output_does_not_become_a_prediction(self):
        import torch

        class LossOnlyModel(torch.nn.Module):

            def forward(self, input_values, labels):
                return {
                    "loss": (input_values - labels).square().mean(),
                }

        trainer = Trainer(
            model=LossOnlyModel(),
            args=TrainingArguments(
                per_device_eval_batch_size=1,
                use_cpu=True,
            ),
        )
        output = trainer.predict(self._dataset(1))

        self.assertIsNone(output.predictions)
        self.assertIsNotNone(output.label_ids)
        self.assertIn("test_loss", output.metrics)

    def test_checkpoint_is_versioned_and_marked_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(
                model=self._dropout_model(),
                args=TrainingArguments(
                    output_dir=directory,
                    max_steps=1,
                    per_device_train_batch_size=2,
                    logging_strategy="no",
                    save_steps=1,
                    use_cpu=True,
                ),
                train_dataset=self._dataset(2),
            )
            trainer.train()
            checkpoint = Path(directory) / "checkpoint-1"

            self.assertTrue((checkpoint / CHECKPOINT_COMPLETE_NAME).is_file())
            self.assertTrue((checkpoint / CHECKPOINT_MANIFEST_NAME).is_file())
            self.assertTrue((checkpoint / TRAINING_RUNTIME_STATE_NAME).is_file())
            self.assertFalse((checkpoint / SCALER_STATE_NAME).exists())
            manifest = json.loads((checkpoint / CHECKPOINT_MANIFEST_NAME).read_text(encoding="utf-8", ))
            self.assertEqual(
                manifest["format_version"],
                CHECKPOINT_FORMAT_VERSION,
            )
            self.assertIn("resume_signature", manifest)

    def test_resume_fingerprints_reject_credentials_and_keep_safe_metrics(self):

        class FingerprintedObject:

            def __init__(self, fingerprint):
                self.fingerprint = fingerprint

            def resume_fingerprint(self):
                return self.fingerprint

        safe_fingerprint = {
            "dataset_id": "public-corpus",
            "token_count": 512,
        }
        self.assertEqual(
            Trainer._resume_fingerprint(
                FingerprintedObject(safe_fingerprint),
                owner="train_dataset",
            ),
            safe_fingerprint,
        )

        credential = "must-not-appear-in-errors"
        for owner in (
                "train_dataset",
                'data_collator',
                "callback 'example.StatefulCallback'",
                "optimizer 'main'",
                "scheduler 'main'",
        ):
            with self.subTest(owner=owner):
                with self.assertRaisesRegex(
                        ValueError,
                        rf"{owner}\.resume_fingerprint.*metadata\.api_key",
                ) as captured:
                    Trainer._resume_fingerprint(
                        FingerprintedObject({
                            "metadata": {
                                "api_key": credential,
                            },
                        }),
                        owner=owner,
                    )
                self.assertNotIn(credential, str(captured.exception))

    def test_checkpoint_save_rejects_a_subclass_secret_atomically(self):

        class CredentialManifestTrainer(Trainer):

            def _checkpoint_manifest(self, checkpoint):
                manifest = super()._checkpoint_manifest(checkpoint)
                manifest["metadata"] = {
                    "credentials": {
                        "api_key": "must-not-be-persisted",
                    },
                }
                return manifest

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                    ValueError,
                    r"Trainer checkpoint manifest.*metadata\.credentials",
            ):
                CredentialManifestTrainer(
                    model=self._dropout_model(),
                    args=TrainingArguments(
                        output_dir=directory,
                        max_steps=1,
                        per_device_train_batch_size=2,
                        logging_strategy="no",
                        save_steps=1,
                        use_cpu=True,
                    ),
                    train_dataset=self._dataset(2),
                ).train()

            output = Path(directory)
            self.assertFalse((output / "checkpoint-1").exists())
            self.assertEqual(
                list(output.glob(".checkpoint-1.incomplete-*")),
                [],
            )

    def test_checkpoint_binary_state_rejects_credentials_atomically(self):
        import torch

        credential = "must-not-appear-in-errors"

        class CredentialStateCallback(TrainerCallback):

            def state_dict(self):
                return {
                    "metrics": {
                        "token_count": 512,
                    },
                    "metadata": {
                        "api_key": credential,
                    },
                }

        class CredentialOptimizer(torch.optim.SGD):

            def state_dict(self):
                state = super().state_dict()
                state["metadata"] = {
                    "api_key": credential,
                }
                return state

        for owner, callback, use_credential_optimizer in (
            ("Trainer runtime checkpoint state", CredentialStateCallback(), False),
            ("Trainer optimizer checkpoint state", None, True),
        ):
            with self.subTest(owner=owner), tempfile.TemporaryDirectory() as directory:
                model = self._dropout_model()
                optimizers = (None, None)
                if use_credential_optimizer:
                    optimizer = CredentialOptimizer(model.parameters(), lr=1e-3)
                    scheduler = torch.optim.lr_scheduler.LambdaLR(
                        optimizer,
                        lambda _: 1.0,
                    )
                    optimizers = (optimizer, scheduler)
                with self.assertRaisesRegex(
                        ValueError,
                        rf"{owner}.*metadata\.api_key",
                ) as captured:
                    Trainer(
                        model=model,
                        args=TrainingArguments(
                            output_dir=directory,
                            max_steps=1,
                            per_device_train_batch_size=2,
                            logging_strategy="no",
                            save_steps=1,
                            use_cpu=True,
                        ),
                        train_dataset=self._dataset(2),
                        callbacks=([callback] if callback is not None else None),
                        optimizers=optimizers,
                    ).train()

                self.assertNotIn(credential, str(captured.exception))
                output = Path(directory)
                self.assertFalse((output / "checkpoint-1").exists())
                self.assertEqual(
                    list(output.glob(".checkpoint-1.incomplete-*")),
                    [],
                )

    def test_safe_callback_checkpoint_state_round_trips(self):

        class SafeStateCallback(TrainerCallback):

            def __init__(self, token_count):
                self.token_count = token_count

            def state_dict(self):
                return {
                    "metrics": {
                        "token_count": self.token_count,
                    },
                }

            def load_state_dict(self, state_dict):
                self.token_count = int(state_dict["metrics"]["token_count"])

        with tempfile.TemporaryDirectory() as directory:
            Trainer(
                model=self._dropout_model(),
                args=TrainingArguments(
                    output_dir=directory,
                    max_steps=1,
                    per_device_train_batch_size=2,
                    logging_strategy="no",
                    save_steps=1,
                    use_cpu=True,
                ),
                train_dataset=self._dataset(2),
                callbacks=[SafeStateCallback(512)],
            ).train()

            restored = SafeStateCallback(0)
            Trainer(
                model=self._dropout_model(),
                args=TrainingArguments(
                    output_dir=directory,
                    max_steps=1,
                    per_device_train_batch_size=2,
                    logging_strategy="no",
                    save_strategy="no",
                    use_cpu=True,
                ),
                train_dataset=self._dataset(2),
                callbacks=[restored],
            ).train(resume_from_checkpoint=True)

            self.assertEqual(restored.token_count, 512)

    def test_checkpoint_binary_state_rejects_credentials_before_restoration(self):
        credential = "must-not-appear-in-errors"

        class CredentialCheckpointTrainer(Trainer):

            def _validate_checkpoint_manifest(self, checkpoint):
                return None

            def _torch_load(self, path):
                if path.name == "optimizer.pt":
                    return {
                        "metadata": {
                            "api_key": credential,
                        },
                    }
                return super()._torch_load(path)

        with tempfile.TemporaryDirectory() as directory:
            Trainer(
                model=self._dropout_model(),
                args=TrainingArguments(
                    output_dir=directory,
                    max_steps=1,
                    per_device_train_batch_size=2,
                    logging_strategy="no",
                    save_steps=1,
                    use_cpu=True,
                ),
                train_dataset=self._dataset(2),
            ).train()
            checkpoint = Path(directory) / "checkpoint-1"
            resumed = CredentialCheckpointTrainer(
                model=self._dropout_model(),
                args=TrainingArguments(
                    output_dir=directory,
                    max_steps=1,
                    per_device_train_batch_size=2,
                    logging_strategy="no",
                    save_strategy="no",
                    use_cpu=True,
                ),
                train_dataset=self._dataset(2),
            )
            restored_models = []
            resumed._load_model = restored_models.append

            with self.assertRaisesRegex(
                    ValueError,
                    r"Trainer optimizer checkpoint state.*metadata\.api_key",
            ) as captured:
                resumed._load_checkpoint(checkpoint)

            self.assertNotIn(credential, str(captured.exception))
            self.assertEqual(restored_models, [])

    def test_checkpoint_model_state_rejects_credentials_before_restoration(self):
        import torch

        credential = "must-not-appear-in-errors"

        class TrackingModel(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.projection = torch.nn.Linear(1, 1)
                self.dropout = torch.nn.Dropout(0.25)
                self.load_state_dict_calls = 0

            def forward(self, input_values, labels):
                predictions = self.projection(self.dropout(input_values))
                return {
                    "loss": torch.nn.functional.mse_loss(
                        predictions,
                        labels,
                    ),
                    "logits": predictions,
                }

            def load_state_dict(self, state_dict, *args, **kwargs):
                self.load_state_dict_calls += 1
                return super().load_state_dict(state_dict, *args, **kwargs)

        class CredentialModelStateTrainer(Trainer):

            def _validate_checkpoint_manifest(self, checkpoint):
                return None

            def _torch_load(self, path):
                if path.name == MODEL_STATE_NAME:
                    return {
                        "metadata": {
                            "api_key": credential,
                        },
                    }
                return super()._torch_load(path)

        with tempfile.TemporaryDirectory() as directory:
            Trainer(
                model=self._dropout_model(),
                args=TrainingArguments(
                    output_dir=directory,
                    max_steps=1,
                    per_device_train_batch_size=2,
                    logging_strategy="no",
                    save_steps=1,
                    use_cpu=True,
                ),
                train_dataset=self._dataset(2),
            ).train()
            checkpoint = Path(directory) / "checkpoint-1"

            unsafe_model = TrackingModel()
            resumed = CredentialModelStateTrainer(
                model=unsafe_model,
                args=TrainingArguments(
                    output_dir=directory,
                    max_steps=1,
                    per_device_train_batch_size=2,
                    logging_strategy="no",
                    save_strategy="no",
                    use_cpu=True,
                ),
                train_dataset=self._dataset(2),
            )
            with self.assertRaisesRegex(
                    ValueError,
                    r"Trainer model artifact state.*metadata\.api_key",
            ) as captured:
                resumed._load_checkpoint(checkpoint)

            self.assertNotIn(credential, str(captured.exception))
            self.assertEqual(unsafe_model.load_state_dict_calls, 0)

            safe_model = TrackingModel()
            Trainer(
                model=safe_model,
                args=TrainingArguments(
                    output_dir=directory,
                    use_cpu=True,
                ),
            )._load_model(checkpoint)

            self.assertEqual(safe_model.load_state_dict_calls, 1)

    def test_checkpoint_load_rejects_credentials_before_restoration(self):
        credential = "must-not-appear-in-errors"
        with tempfile.TemporaryDirectory() as directory:
            Trainer(
                model=self._dropout_model(),
                args=TrainingArguments(
                    output_dir=directory,
                    max_steps=1,
                    per_device_train_batch_size=2,
                    logging_strategy="no",
                    save_steps=1,
                    use_cpu=True,
                ),
                train_dataset=self._dataset(2),
            ).train()
            checkpoint = Path(directory) / "checkpoint-1"
            manifest_path = checkpoint / CHECKPOINT_MANIFEST_NAME
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["metadata"] = {
                "api_key": credential,
            }
            manifest_path.write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )

            resumed = Trainer(
                model=self._dropout_model(),
                args=TrainingArguments(
                    output_dir=directory,
                    max_steps=1,
                    per_device_train_batch_size=2,
                    logging_strategy="no",
                    save_strategy="no",
                    use_cpu=True,
                ),
                train_dataset=self._dataset(2),
            )
            restored_models = []
            resumed._load_model = restored_models.append
            with self.assertRaisesRegex(
                    ValueError,
                    r"Trainer checkpoint manifest.*metadata\.api_key",
            ) as captured:
                resumed._load_checkpoint(checkpoint)

            self.assertNotIn(credential, str(captured.exception))
            self.assertEqual(restored_models, [])

    def test_exact_resume_rejects_a_changed_schedule(self):

        class StopAfterOne(TrainerCallback):

            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step == 1:
                    control.should_save = True
                    control.should_training_stop = True
                return control

        with tempfile.TemporaryDirectory() as directory:
            Trainer(
                model=self._dropout_model(),
                args=TrainingArguments(
                    output_dir=directory,
                    max_steps=2,
                    per_device_train_batch_size=2,
                    logging_strategy="no",
                    save_strategy="no",
                    use_cpu=True,
                ),
                train_dataset=self._dataset(),
                callbacks=[StopAfterOne],
            ).train()

            resumed = Trainer(
                model=self._dropout_model(),
                args=TrainingArguments(
                    output_dir=directory,
                    max_steps=3,
                    per_device_train_batch_size=2,
                    logging_strategy="no",
                    save_strategy="no",
                    use_cpu=True,
                ),
                train_dataset=self._dataset(),
            )
            with self.assertRaisesRegex(
                    ValueError,
                    r"schedule\.max_steps",
            ):
                resumed.train(resume_from_checkpoint=True)

    def test_exact_resume_uses_optional_dataset_fingerprint(self):
        import torch

        class FingerprintedDataset(torch.utils.data.Dataset):

            def __init__(self, identifier):
                self.identifier = identifier

            def __len__(self):
                return 4

            def __getitem__(self, index):
                value = torch.tensor([float(index + 1)])
                return {
                    "input_values": value,
                    "labels": value * 2,
                }

            def resume_fingerprint(self):
                return {
                    "dataset_id": self.identifier,
                    "revision": 1,
                }

        class StopAfterOne(TrainerCallback):

            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step == 1:
                    control.should_save = True
                    control.should_training_stop = True
                return control

        with tempfile.TemporaryDirectory() as directory:
            arguments = {
                "output_dir": directory,
                "max_steps": 2,
                "per_device_train_batch_size": 2,
                "logging_strategy": "no",
                "save_strategy": "no",
                "use_cpu": True,
            }
            Trainer(
                model=self._dropout_model(),
                args=TrainingArguments(**arguments),
                train_dataset=FingerprintedDataset("revision-a"),
                callbacks=[StopAfterOne],
            ).train()

            resumed = Trainer(
                model=self._dropout_model(),
                args=TrainingArguments(**arguments),
                train_dataset=FingerprintedDataset("revision-b"),
            )
            with self.assertRaisesRegex(
                    ValueError,
                    r"dataset\.fingerprint",
            ):
                resumed.train(resume_from_checkpoint=True)

    def test_exact_resume_rejects_changed_collator_configuration(self):

        class StopAfterOne(TrainerCallback):

            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step == 1:
                    control.should_save = True
                    control.should_training_stop = True
                return control

        with tempfile.TemporaryDirectory() as directory:
            arguments = TrainingArguments(
                output_dir=directory,
                max_steps=2,
                per_device_train_batch_size=2,
                logging_strategy="no",
                save_strategy="no",
                use_cpu=True,
            )
            Trainer(
                model=self._dropout_model(),
                args=arguments,
                train_dataset=self._dataset(),
                data_collator=DataCollatorForTTSTraining(label_pad_token_id=-100, ),
                callbacks=[StopAfterOne],
            ).train()

            resumed = Trainer(
                model=self._dropout_model(),
                args=arguments,
                train_dataset=self._dataset(),
                data_collator=DataCollatorForTTSTraining(label_pad_token_id=0, ),
                callbacks=[StopAfterOne],
            )
            with self.assertRaisesRegex(
                    ValueError,
                    r"collator\.fingerprint\.label_pad_token_id",
            ):
                resumed.train(resume_from_checkpoint=True)

    def test_exact_resume_rejects_changed_stateful_callback_configuration(self, ):

        class StopAfterOne(TrainerCallback):

            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step == 1:
                    control.should_save = True
                    control.should_training_stop = True
                return control

        with tempfile.TemporaryDirectory() as directory:
            arguments = TrainingArguments(
                output_dir=directory,
                max_steps=2,
                per_device_train_batch_size=2,
                per_device_eval_batch_size=2,
                eval_strategy="steps",
                eval_steps=1,
                save_strategy="steps",
                save_steps=1,
                load_best_model_at_end=True,
                logging_strategy="no",
                use_cpu=True,
            )
            Trainer(
                model=self._dropout_model(),
                args=arguments,
                train_dataset=self._dataset(),
                eval_dataset=self._dataset(2),
                callbacks=[
                    StopAfterOne,
                    EarlyStoppingCallback(
                        early_stopping_patience=2,
                        early_stopping_threshold=0.0,
                    ),
                ],
            ).train()

            resumed = Trainer(
                model=self._dropout_model(),
                args=arguments,
                train_dataset=self._dataset(),
                eval_dataset=self._dataset(2),
                callbacks=[
                    StopAfterOne,
                    EarlyStoppingCallback(
                        early_stopping_patience=9,
                        early_stopping_threshold=0.5,
                    ),
                ],
            )
            with self.assertRaisesRegex(
                    ValueError,
                    r"stateful_callbacks",
            ):
                resumed.train(resume_from_checkpoint=True)

    def test_native_export_cannot_overwrite_voicehub_configuration(self):
        import torch

        class ExportConfig(VoiceHubConfig):
            model_type = "orpheustts"

        class ExportModel(PreTrainedTTSModel):
            config_class = ExportConfig

            def _load_pretrained_model(self):

                class Runtime(torch.nn.Module):

                    def __init__(self):
                        super().__init__()
                        self.weight = torch.nn.Parameter(torch.ones(()))

                    def forward(self, input_values, labels):
                        prediction = input_values * self.weight
                        return {
                            "loss": (prediction - labels).square().mean(),
                            "logits": prediction,
                        }

                self.model = Runtime()

            def _generate(self, text, **kwargs):
                return TTSOutput(audio=[0.0], sample_rate=24000)

        class SourceExporter(CausalLMTrainingAdapter):
            native_export_semantics = "inference-export"

            def save_pretrained(self, save_directory):
                destination = Path(save_directory)
                destination.mkdir(parents=True, exist_ok=True)
                (destination / "config.json").write_text(
                    json.dumps({
                        "model_type": "upstream-native",
                    }),
                    encoding="utf-8",
                )

        model = ExportModel(
            ExportConfig(name_or_path="source/model"),
            device="cpu",
            lazy_load=False,
        )
        spec = ModelTrainingSpec(
            model_type="orpheustts",
            family=TrainingFamily.CAUSAL_LM,
            module_paths=("model", ),
            component_paths=("model", ),
        )
        adapter = SourceExporter(model, spec)
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "artifact"
            trainer = Trainer(
                model=model,
                args=TrainingArguments(
                    output_dir=directory,
                    use_cpu=True,
                ),
                training_adapter=adapter,
            )
            trainer.save_model(destination)

            voicehub_config = json.loads((destination / "config.json").read_text(encoding="utf-8"))
            native_config = json.loads(
                (destination / NATIVE_EXPORT_DIR / "config.json").read_text(encoding="utf-8"))
            recipe = json.loads((destination / TRAINING_RECIPE_NAME).read_text(encoding="utf-8", ))

        self.assertEqual(voicehub_config["model_type"], "orpheustts")
        self.assertEqual(native_config["model_type"], "upstream-native")
        self.assertEqual(
            recipe["native_export_path"],
            NATIVE_EXPORT_DIR,
        )

    def test_training_recipe_manifest_rejects_secrets_before_artifact_mutation(self):
        import torch

        credential = "must-not-appear-in-errors"

        class TrackingModel(torch.nn.Linear):

            def __init__(self):
                super().__init__(1, 1)
                self.state_dict_calls = 0

            def state_dict(self, *args, **kwargs):
                self.state_dict_calls += 1
                return super().state_dict(*args, **kwargs)

        class RecipeAdapter(CausalLMTrainingAdapter):

            def __init__(self, model, spec, metadata):
                super().__init__(model, spec)
                self.metadata = metadata

            def artifact_manifest(self):
                manifest = super().artifact_manifest()
                manifest["run_metadata"] = self.metadata
                return manifest

        spec = ModelTrainingSpec(
            model_type="orpheustts",
            family=TrainingFamily.CAUSAL_LM,
            module_paths=("model", ),
        )
        with tempfile.TemporaryDirectory() as directory:
            unsafe_model = TrackingModel()
            unsafe_destination = Path(directory) / "unsafe-artifact"
            unsafe_trainer = Trainer(
                model=unsafe_model,
                args=TrainingArguments(
                    output_dir=directory,
                    use_cpu=True,
                ),
                training_adapter=RecipeAdapter(
                    unsafe_model,
                    spec,
                    {
                        "provider_options": {
                            "api_key": credential,
                        },
                    },
                ),
            )
            with self.assertRaisesRegex(ValueError, "runtime secrets") as captured:
                unsafe_trainer.save_model(
                    unsafe_destination,
                    include_native_export=False,
                )
            self.assertNotIn(credential, str(captured.exception))
            self.assertEqual(unsafe_model.state_dict_calls, 0)
            self.assertFalse(unsafe_destination.exists())

            safe_model = TrackingModel()
            safe_destination = Path(directory) / "safe-artifact"
            safe_trainer = Trainer(
                model=safe_model,
                args=TrainingArguments(
                    output_dir=directory,
                    use_cpu=True,
                ),
                training_adapter=RecipeAdapter(
                    safe_model,
                    spec,
                    {
                        "token_count": 512,
                        "dataset_id": "speech/train",
                    },
                ),
            )
            safe_trainer.save_model(
                safe_destination,
                include_native_export=False,
            )
            recipe = json.loads((safe_destination / TRAINING_RECIPE_NAME).read_text(encoding="utf-8", ))

        self.assertEqual(
            recipe["run_metadata"],
            {
                "token_count": 512,
                "dataset_id": "speech/train",
            },
        )

    def test_model_artifact_state_rejects_secrets_at_both_write_boundaries(self):
        import torch

        credential = "must-not-appear-in-errors"

        class ArtifactStateModel(torch.nn.Linear):

            def __init__(self, metadata, *, mutate_on_save=False):
                super().__init__(1, 1)
                self.metadata = metadata
                self.mutate_on_save = mutate_on_save
                self.artifact_state = None
                self.save_pretrained_calls = 0

            def state_dict(self, *args, **kwargs):
                self.artifact_state = super().state_dict(*args, **kwargs)
                self.artifact_state["run_metadata"] = self.metadata
                return self.artifact_state

            def save_pretrained(self, _directory):
                self.save_pretrained_calls += 1
                if self.mutate_on_save:
                    self.artifact_state["run_metadata"] = {
                        "api_key": credential,
                    }

        with tempfile.TemporaryDirectory() as directory:
            unsafe_destination = Path(directory) / "unsafe-artifact"
            unsafe_model = ArtifactStateModel({
                "api_key": credential,
            })
            unsafe_trainer = Trainer(
                model=unsafe_model,
                args=TrainingArguments(
                    output_dir=directory,
                    use_cpu=True,
                ),
            )
            with self.assertRaisesRegex(ValueError, "Trainer model artifact state") as captured:
                unsafe_trainer.save_model(unsafe_destination)
            self.assertNotIn(credential, str(captured.exception))
            self.assertEqual(unsafe_model.save_pretrained_calls, 0)
            self.assertFalse(unsafe_destination.exists())

            mutated_destination = Path(directory) / "mutated-artifact"
            mutated_model = ArtifactStateModel(
                {
                    "token_count": 512,
                },
                mutate_on_save=True,
            )
            mutated_trainer = Trainer(
                model=mutated_model,
                args=TrainingArguments(
                    output_dir=directory,
                    use_cpu=True,
                ),
            )
            with self.assertRaisesRegex(ValueError, "Trainer model artifact state") as captured:
                mutated_trainer.save_model(mutated_destination)
            self.assertNotIn(credential, str(captured.exception))
            self.assertEqual(mutated_model.save_pretrained_calls, 1)
            self.assertFalse((mutated_destination / MODEL_STATE_NAME).exists())

            safe_destination = Path(directory) / "safe-artifact"
            safe_model = ArtifactStateModel({
                "dataset_id": "speech/train",
                "token_count": 512,
            })
            safe_trainer = Trainer(
                model=safe_model,
                args=TrainingArguments(
                    output_dir=directory,
                    use_cpu=True,
                ),
            )
            safe_trainer.save_model(safe_destination)
            saved_state = torch.load(
                safe_destination / MODEL_STATE_NAME,
                weights_only=False,
            )

        self.assertEqual(
            saved_state["run_metadata"],
            {
                "dataset_id": "speech/train",
                "token_count": 512,
            },
        )

    def test_checkpoint_discovery_skips_corrupt_latest_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            Trainer(
                model=self._dropout_model(),
                args=TrainingArguments(
                    output_dir=directory,
                    max_steps=2,
                    per_device_train_batch_size=1,
                    logging_strategy="no",
                    save_steps=1,
                    use_cpu=True,
                ),
                train_dataset=self._dataset(2),
            ).train()
            corrupt = Path(directory) / "checkpoint-2" / MODEL_STATE_NAME
            corrupt.write_bytes(b"corrupt")

            self.assertEqual(
                Path(get_last_checkpoint(directory)).name,
                "checkpoint-1",
            )

    def test_checkpoint_discovery_skips_ambiguous_or_non_finite_manifest(self):
        documents = (
            (
                "duplicate",
                '{"format_version": 1, "global_step": 2, "global_step": 2, '
                '"required_files": []}',
            ),
            (
                "non-finite",
                '{"format_version": 1, "global_step": 2, "required_files": [], '
                '"metadata": {"loss": 1e10000}}',
            ),
        )

        for name, document in documents:
            with self.subTest(document=name):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    first = root / "checkpoint-1"
                    first.mkdir()
                    (first / CHECKPOINT_COMPLETE_NAME).write_text(
                        "complete\n",
                        encoding="utf-8",
                    )
                    (first / CHECKPOINT_MANIFEST_NAME).write_text(
                        json.dumps({
                            "format_version": 1,
                            "global_step": 1,
                            "required_files": [],
                        }),
                        encoding="utf-8",
                    )
                    second = root / "checkpoint-2"
                    second.mkdir()
                    (second / CHECKPOINT_COMPLETE_NAME).write_text(
                        "complete\n",
                        encoding="utf-8",
                    )
                    (second / CHECKPOINT_MANIFEST_NAME).write_text(
                        document,
                        encoding="utf-8",
                    )

                    self.assertEqual(
                        Path(get_last_checkpoint(root)).name,
                        "checkpoint-1",
                    )

    def test_checkpoint_json_loaders_reject_before_model_or_runtime_restore(self):
        documents = (
            (
                "manifest-duplicate",
                '{"format_version": 1, "format_version": 1, '
                '"global_step": 2, "required_files": []}',
                {},
                "Duplicate JSON object key 'format_version'",
            ),
            (
                "manifest-overflow",
                '{"format_version": 1, "global_step": 2, "required_files": [], '
                '"metadata": {"loss": 1e10000}}',
                {},
                "$.metadata.loss",
            ),
            (
                "optimization-duplicate",
                json.dumps({
                    "format_version": 1,
                    "global_step": 2,
                    "required_files": ["optimization_manifest.json"],
                    "optimization_plan": {
                        "format_version": 1,
                    },
                }),
                {
                    "optimization_manifest.json": '{"format_version": 1, "format_version": 1}',
                },
                "Duplicate JSON object key 'format_version'",
            ),
            (
                "trainer-state-duplicate",
                json.dumps({
                    "format_version": 1,
                    "global_step": 2,
                    "required_files": ["trainer_state.json"],
                }),
                {
                    "trainer_state.json": '{"global_step": 2, "global_step": 2}',
                },
                "Duplicate JSON object key 'global_step'",
            ),
            (
                "trainer-state-overflow",
                json.dumps({
                    "format_version": 1,
                    "global_step": 2,
                    "required_files": ["trainer_state.json"],
                }),
                {
                    "trainer_state.json": '{"global_step": 2, "log_history": [{"loss": 1e10000}]}',
                },
                "$.log_history[0].loss",
            ),
        )

        for name, manifest, artifacts, expected_detail in documents:
            with self.subTest(document=name):
                with tempfile.TemporaryDirectory() as directory:
                    checkpoint = Path(directory) / "checkpoint-2"
                    checkpoint.mkdir()
                    (checkpoint / CHECKPOINT_COMPLETE_NAME).write_text(
                        "complete\n",
                        encoding="utf-8",
                    )
                    (checkpoint / CHECKPOINT_MANIFEST_NAME).write_text(
                        manifest,
                        encoding="utf-8",
                    )
                    for filename, document in artifacts.items():
                        (checkpoint / filename).write_text(
                            document,
                            encoding="utf-8",
                        )

                    trainer = object.__new__(Trainer)
                    restored_models = []
                    trainer._load_model = restored_models.append
                    with self.assertRaises(ValueError) as captured:
                        trainer._load_checkpoint(checkpoint)

                rendered_error = str(captured.exception)
                self.assertIn(str(checkpoint), rendered_error)
                self.assertIn(expected_detail, rendered_error)
                self.assertEqual(restored_models, [])

    def test_pretrained_round_trip_restores_trainer_model_state(self):
        import torch

        class RoundTripConfig(VoiceHubConfig):
            model_type = "orpheustts"

        class RoundTripModel(PreTrainedTTSModel):
            config_class = RoundTripConfig

            def _load_pretrained_model(self):

                class Runtime(torch.nn.Module):

                    def __init__(self):
                        super().__init__()
                        self.weight = torch.nn.Parameter(torch.zeros(1, 1))

                    def forward(self, input_values, labels):
                        predictions = input_values @ self.weight
                        return {
                            "loss": (predictions - labels).square().mean(),
                            "logits": predictions,
                        }

                self.model = Runtime()

            def _generate(self, text, **kwargs):
                return TTSOutput(audio=[0.0], sample_rate=24000)

        config = RoundTripConfig(name_or_path="dummy-source")
        model = RoundTripModel(config, device="cpu")
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(
                model=model,
                args=TrainingArguments(
                    output_dir=directory,
                    max_steps=1,
                    per_device_train_batch_size=2,
                    learning_rate=0.1,
                    max_grad_norm=0,
                    logging_strategy="no",
                    save_strategy="no",
                    use_cpu=True,
                ),
                train_dataset=self._dataset(2),
            )
            trainer.train()
            artifact = Path(directory) / "artifact"
            trainer.save_model(artifact)
            expected = model.model.weight.detach().clone()

            restored = RoundTripModel.from_pretrained(
                str(artifact),
                device="cpu",
                lazy_load=False,
            )

        torch.testing.assert_close(
            restored.model.weight,
            expected,
            rtol=0,
            atol=0,
        )

    def test_portable_artifact_hands_recipe_state_to_trainer_adapter(self):
        import torch

        class StatefulConfig(VoiceHubConfig):
            model_type = "orpheustts"

        class StatefulModel(PreTrainedTTSModel):
            config_class = StatefulConfig

            def _load_pretrained_model(self):

                class Runtime(torch.nn.Module):

                    def __init__(self):
                        super().__init__()
                        self.weight = torch.nn.Parameter(torch.ones(1, 1))

                    def forward(self, input_values, labels):
                        prediction = input_values @ self.weight
                        return {
                            "loss": (prediction - labels).square().mean(),
                            "logits": prediction,
                        }

                self.model = Runtime()

            def _generate(self, text, **kwargs):
                return TTSOutput(audio=[0.0], sample_rate=24000)

        recipe_loads = []

        class StatefulAdapter(CausalLMTrainingAdapter):

            def __init__(self, model, spec):
                super().__init__(model, spec)
                self.recipe_value = torch.tensor(-1)

            def recipe_state_dict(self):
                return {"value": self.recipe_value.detach().clone()}

            def load_recipe_state_dict(self, state_dict, *, strict=True):
                if strict and set(state_dict) != {"value"}:
                    raise ValueError("Stateful recipe requires exactly 'value'.")
                recipe_loads.append(self)
                self.recipe_value = state_dict["value"].detach().clone()

        AutoTrainingAdapter.register("orpheustts", StatefulAdapter)
        try:
            with tempfile.TemporaryDirectory() as directory:
                arguments = TrainingArguments(
                    output_dir=directory,
                    max_steps=1,
                    per_device_train_batch_size=2,
                    learning_rate=0.1,
                    logging_strategy="no",
                    save_strategy="no",
                    use_cpu=True,
                )
                model = StatefulModel(
                    StatefulConfig(name_or_path="dummy-source"),
                    device="cpu",
                )
                trainer = Trainer(
                    model=model,
                    args=arguments,
                    train_dataset=self._dataset(2),
                )
                trainer.train()
                trainer.training_adapter.recipe_value = torch.tensor(37)
                artifact = Path(directory) / "artifact"
                trainer.save_model(artifact)

                restored = StatefulModel.from_pretrained(
                    str(artifact),
                    device="cpu",
                    lazy_load=False,
                )
                self.assertIsNotNone(restored._pending_training_recipe_state)
                self.assertEqual(recipe_loads, [])

                resumed = Trainer(model=restored, args=arguments)
                resumed._ensure_model_loaded()

                self.assertEqual(
                    resumed.training_adapter.recipe_value.item(),
                    37,
                )
                self.assertEqual(
                    recipe_loads,
                    [resumed.training_adapter],
                )
                self.assertIsNone(restored._pending_training_recipe_state)
        finally:
            AutoTrainingAdapter.unregister("orpheustts")

    def test_best_model_is_saved_even_between_regular_save_steps(self):
        import torch

        scores = iter((1.0, 2.0))

        class Model(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))

            def forward(self, input_values, labels):
                prediction = input_values * self.weight
                return {
                    "loss": (prediction - labels).square().mean(),
                    "logits": prediction,
                }

        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(
                model=Model(),
                args=TrainingArguments(
                    output_dir=directory,
                    max_steps=2,
                    per_device_train_batch_size=1,
                    per_device_eval_batch_size=1,
                    eval_strategy="steps",
                    eval_steps=1,
                    save_strategy="steps",
                    save_steps=2,
                    load_best_model_at_end=True,
                    metric_for_best_model="score",
                    greater_is_better=False,
                    logging_strategy="no",
                    use_cpu=True,
                ),
                train_dataset=self._dataset(2),
                eval_dataset=self._dataset(1),
                compute_metrics=lambda _: {"score": next(scores)},
            )
            trainer.train()

            self.assertEqual(
                Path(trainer.state.best_model_checkpoint).name,
                "checkpoint-1",
            )
            self.assertTrue((Path(directory) / "checkpoint-1" / CHECKPOINT_COMPLETE_NAME).is_file())

    def test_interrupted_resume_matches_uninterrupted_training(self):
        import torch

        class StopAndSaveAtStep(TrainerCallback):

            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step == 3:
                    control.should_save = True
                    control.should_training_stop = True
                return control

        torch.manual_seed(91)
        initial = self._dropout_model().state_dict()

        full_model = self._dropout_model()
        full_model.load_state_dict(initial)
        interrupted_model = self._dropout_model()
        interrupted_model.load_state_dict(initial)

        common = {
            "max_steps": 6,
            "per_device_train_batch_size": 2,
            "learning_rate": 1e-3,
            "seed": 123,
            "data_seed": 456,
            "logging_strategy": "no",
            "save_strategy": "no",
            "use_cpu": True,
        }
        with tempfile.TemporaryDirectory() as full_directory, tempfile.TemporaryDirectory(
        ) as resumed_directory:
            full = Trainer(
                model=full_model,
                args=TrainingArguments(
                    output_dir=full_directory,
                    **common,
                ),
                train_dataset=self._dataset(),
            )
            full.train()

            interrupted = Trainer(
                model=interrupted_model,
                args=TrainingArguments(
                    output_dir=resumed_directory,
                    **common,
                ),
                train_dataset=self._dataset(),
                callbacks=[StopAndSaveAtStep],
            )
            interrupted.train()

            resumed_model = self._dropout_model()
            resumed = Trainer(
                model=resumed_model,
                args=TrainingArguments(
                    output_dir=resumed_directory,
                    **common,
                ),
                train_dataset=self._dataset(),
            )
            resumed.train(resume_from_checkpoint=True)

        for name, expected in full_model.state_dict().items():
            torch.testing.assert_close(
                resumed_model.state_dict()[name],
                expected,
                rtol=0,
                atol=0,
            )
        self.assertEqual(resumed.state.global_step, full.state.global_step)
        self.assertEqual(
            resumed.get_learning_rates(),
            full.get_learning_rates(),
        )

    def test_epoch_end_callback_state_is_included_in_boundary_checkpoint(self):

        class EpochCounter(TrainerCallback):

            def __init__(self, *, stop_after_first=False):
                self.epoch_ends = 0
                self.stop_after_first = stop_after_first

            def on_epoch_end(self, args, state, control, **kwargs):
                self.epoch_ends += 1
                if self.stop_after_first and self.epoch_ends == 1:
                    control.should_training_stop = True
                return control

            def state_dict(self):
                return {"epoch_ends": self.epoch_ends}

            def load_state_dict(self, state_dict):
                self.epoch_ends = int(state_dict["epoch_ends"])

        with tempfile.TemporaryDirectory() as directory:
            first_counter = EpochCounter(stop_after_first=True)
            Trainer(
                model=self._dropout_model(),
                args=TrainingArguments(
                    output_dir=directory,
                    max_steps=3,
                    per_device_train_batch_size=1,
                    logging_strategy="no",
                    save_strategy="steps",
                    save_steps=2,
                    use_cpu=True,
                ),
                train_dataset=self._dataset(2),
                callbacks=[first_counter],
            ).train()
            self.assertEqual(first_counter.epoch_ends, 1)

            resumed_counter = EpochCounter()
            Trainer(
                model=self._dropout_model(),
                args=TrainingArguments(
                    output_dir=directory,
                    max_steps=3,
                    per_device_train_batch_size=1,
                    logging_strategy="no",
                    save_strategy="steps",
                    save_steps=2,
                    use_cpu=True,
                ),
                train_dataset=self._dataset(2),
                callbacks=[resumed_counter],
            ).train(resume_from_checkpoint=True)

        self.assertEqual(resumed_counter.epoch_ends, 2)

    def test_resume_restores_rng_after_skipping_stochastic_samples(self):
        import torch

        class StochasticDataset(torch.utils.data.Dataset):

            def __len__(self):
                return 8

            def __getitem__(self, index):
                scale = 1.0 + torch.rand(())
                value = torch.tensor([float(index + 1)]) * scale
                return {
                    "input_values": value,
                    "labels": value * 2,
                }

        class StopAndSaveAtStep(TrainerCallback):

            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step == 3:
                    control.should_save = True
                    control.should_training_stop = True
                return control

        torch.manual_seed(17)
        initial = self._dropout_model().state_dict()
        full_model = self._dropout_model()
        full_model.load_state_dict(initial)
        interrupted_model = self._dropout_model()
        interrupted_model.load_state_dict(initial)
        common = {
            "max_steps": 6,
            "per_device_train_batch_size": 2,
            "learning_rate": 1e-3,
            "seed": 222,
            "data_seed": 333,
            "logging_strategy": "no",
            "save_strategy": "no",
            "use_cpu": True,
        }
        with tempfile.TemporaryDirectory() as full_directory, tempfile.TemporaryDirectory(
        ) as resumed_directory:
            full = Trainer(
                model=full_model,
                args=TrainingArguments(
                    output_dir=full_directory,
                    **common,
                ),
                train_dataset=StochasticDataset(),
            )
            full.train()
            Trainer(
                model=interrupted_model,
                args=TrainingArguments(
                    output_dir=resumed_directory,
                    **common,
                ),
                train_dataset=StochasticDataset(),
                callbacks=[StopAndSaveAtStep],
            ).train()
            resumed_model = self._dropout_model()
            resumed = Trainer(
                model=resumed_model,
                args=TrainingArguments(
                    output_dir=resumed_directory,
                    **common,
                ),
                train_dataset=StochasticDataset(),
            )
            resumed.train(resume_from_checkpoint=True)

        for name, expected in full_model.state_dict().items():
            torch.testing.assert_close(
                resumed_model.state_dict()[name],
                expected,
                rtol=0,
                atol=0,
            )

    def test_resume_cursor_includes_skipped_optimizer_windows(self):
        import torch

        class SkipFirstStepStrategy(TorchTrainingStrategy):

            def __init__(self):
                super().__init__()
                self.optimizer_calls = 0

            def optimizer_step(self, optimizer, **kwargs):
                self.optimizer_calls += 1
                if self.optimizer_calls == 1:
                    return False
                return super().optimizer_step(optimizer, **kwargs)

            def state_dict(self):
                return {"optimizer_calls": self.optimizer_calls}

            def load_state_dict(self, state_dict):
                self.optimizer_calls = int(state_dict["optimizer_calls"])

        class StopAndSaveAtStep(TrainerCallback):

            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step == 1:
                    control.should_save = True
                    control.should_training_stop = True
                return control

        torch.manual_seed(707)
        initial = self._dropout_model().state_dict()
        full_model = self._dropout_model()
        full_model.load_state_dict(initial)
        interrupted_model = self._dropout_model()
        interrupted_model.load_state_dict(initial)
        common = {
            "max_steps": 2,
            "per_device_train_batch_size": 1,
            "learning_rate": 1e-3,
            "seed": 808,
            "data_seed": 909,
            "logging_strategy": "no",
            "save_strategy": "no",
            "use_cpu": True,
        }

        with tempfile.TemporaryDirectory() as full_directory, tempfile.TemporaryDirectory(
        ) as resumed_directory:
            full = Trainer(
                model=full_model,
                args=TrainingArguments(
                    output_dir=full_directory,
                    **common,
                ),
                train_dataset=self._dataset(5),
                training_strategy=SkipFirstStepStrategy(),
            )
            full.train()

            interrupted = Trainer(
                model=interrupted_model,
                args=TrainingArguments(
                    output_dir=resumed_directory,
                    **common,
                ),
                train_dataset=self._dataset(5),
                callbacks=[StopAndSaveAtStep],
                training_strategy=SkipFirstStepStrategy(),
            )
            interrupted.train()
            self.assertEqual(interrupted.state.train_batch_cursor, 2)

            resumed_model = self._dropout_model()
            resumed = Trainer(
                model=resumed_model,
                args=TrainingArguments(
                    output_dir=resumed_directory,
                    **common,
                ),
                train_dataset=self._dataset(5),
                training_strategy=SkipFirstStepStrategy(),
            )
            resumed.train(resume_from_checkpoint=True)

        for name, expected in full_model.state_dict().items():
            torch.testing.assert_close(
                resumed_model.state_dict()[name],
                expected,
                rtol=0,
                atol=0,
            )

    def test_evaluation_preserves_optimization_prepared_model_proxy(self):
        import torch

        class Model(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))

            def forward(self, input_values, labels):
                prediction = input_values * self.weight
                return {
                    "loss": (prediction - labels).square().mean(),
                    "logits": prediction,
                }

        class Proxy:

            def __init__(self, model, name, calls):
                self.model = model
                self.name = name
                self.calls = calls

            def __call__(self, **inputs):
                self.calls.append(self.name)
                return self.model(**inputs)

            def train(self, mode=True):
                self.model.train(mode)

            def eval(self):
                self.model.eval()

        class ProxyStrategy(TorchTrainingStrategy):

            def __init__(self):
                super().__init__()
                self.calls = []
                self.prepare_calls = 0

            def prepare_model(self, model, *, device):
                self.prepare_calls += 1
                model.to(device)
                return Proxy(model, "prepared", self.calls)

            def prepare_optimization(self, model, optimizer, scheduler):
                return (
                    Proxy(model.model, "optimized", self.calls),
                    optimizer,
                    scheduler,
                )

            def unwrap_model(self, model):
                return model.model

        strategy = ProxyStrategy()
        with tempfile.TemporaryDirectory() as directory:
            Trainer(
                model=Model(),
                args=TrainingArguments(
                    output_dir=directory,
                    max_steps=2,
                    per_device_train_batch_size=1,
                    per_device_eval_batch_size=1,
                    eval_strategy="steps",
                    eval_steps=1,
                    save_strategy="no",
                    logging_strategy="no",
                    use_cpu=True,
                ),
                train_dataset=self._dataset(2),
                eval_dataset=self._dataset(1),
                training_strategy=strategy,
            ).train()

        self.assertEqual(strategy.prepare_calls, 1)
        self.assertEqual(strategy.calls, ["optimized"] * 4)

    def test_custom_training_strategy_owns_execution(self):
        import torch

        class RecordingStrategy(TrainingStrategy):
            name = "recording"

            def __init__(self):
                self.events = []

            def backward(self, loss, *, scaler=None):
                self.events.append("backward")
                loss.backward()

            def optimizer_step(self, optimizer, **kwargs):
                self.events.append("optimizer_step")
                optimizer.step()

            def scheduler_step(self, scheduler, **kwargs):
                self.events.append("scheduler_step")
                scheduler.step()

            def zero_grad(self, optimizer, **kwargs):
                self.events.append("zero_grad")
                optimizer.zero_grad(set_to_none=True)

        strategy = RecordingStrategy()
        model = torch.nn.Linear(1, 1)

        class LossWrapper(torch.nn.Module):

            def __init__(self, projection):
                super().__init__()
                self.projection = projection

            def forward(self, input_values, labels):
                values = self.projection(input_values)
                return {
                    "loss": torch.nn.functional.mse_loss(values, labels),
                    "logits": values,
                }

        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(
                model=LossWrapper(model),
                args=TrainingArguments(
                    output_dir=directory,
                    max_steps=1,
                    per_device_train_batch_size=2,
                    max_grad_norm=0,
                    logging_strategy="no",
                    save_strategy="no",
                    use_cpu=True,
                ),
                train_dataset=self._dataset(2),
                training_strategy=strategy,
            )
            trainer.train()

        self.assertIn("backward", strategy.events)
        self.assertIn("optimizer_step", strategy.events)
        self.assertIn("scheduler_step", strategy.events)
        self.assertGreaterEqual(strategy.events.count("zero_grad"), 2)

    def test_strategy_proxy_preserves_multi_phase_execution(self):
        import torch

        class Runtime(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.generator = torch.nn.Linear(1, 1)
                self.discriminator = torch.nn.Linear(1, 1)

            def generator_step(self, values, target):
                prediction = self.generator(values)
                return {"loss": (prediction - target).square().mean()}

            def discriminator_step(self, values, target):
                prediction = self.discriminator(values.detach())
                return {"discriminator_loss": (prediction - target).square().mean()}

            def forward(self, values, target):
                return self.generator_step(values, target)

        class Wrapper:

            class Config:
                model_type = "proxy-vits"

            def __init__(self):
                self.config = self.Config()
                self.runtime = None

            def load_for_training(self):
                self.runtime = Runtime()

        spec = ModelTrainingSpec(
            model_type="proxy-vits",
            family=TrainingFamily.VITS,
            module_paths=("runtime", ),
            component_paths=("runtime.generator", "runtime.discriminator"),
            support=TrainingSupport.PREPROCESSED,
            separate_optimizers=True,
            phases=(
                TrainingPhaseSpec(
                    name="generator",
                    kind=TrainingPhaseKind.GENERATOR,
                    component_paths=("runtime.generator", ),
                    optimizer_names=("generator", ),
                    forward_component="runtime",
                    forward_method="generator_step",
                    loss_keys=("loss", ),
                ),
                TrainingPhaseSpec(
                    name="discriminator",
                    kind=TrainingPhaseKind.DISCRIMINATOR,
                    component_paths=("runtime.discriminator", ),
                    optimizer_names=("discriminator", ),
                    forward_component="runtime",
                    forward_method="discriminator_step",
                    loss_keys=("discriminator_loss", ),
                ),
            ),
        )
        wrapper = Wrapper()
        adapter = VITSTrainingAdapter(wrapper, spec)

        class Proxy:

            def __init__(self, wrapped):
                self.wrapped = wrapped

            def __call__(self, *, training_context):
                return self.wrapped.execute_training_phase(training_context)

            def train(self, mode=True):
                self.wrapped.train(mode)

            def eval(self):
                self.wrapped.eval()

        class ProxyStrategy(TorchTrainingStrategy):

            def __init__(self):
                super().__init__()
                self.phases = []

            def prepare_training_adapter(self, adapter, *, device):
                adapter.to(device)
                proxy = Proxy(adapter)
                original = proxy.__call__

                def record(*, training_context):
                    self.phases.append(training_context.phase.name)
                    return original(training_context=training_context)

                proxy.__call__ = record
                return proxy

            def execute_training_phase(self, model, adapter, context):
                self.phases.append(context.phase.name)
                return model(training_context=context)

            def unwrap_model(self, model):
                return model.wrapped

        strategy = ProxyStrategy()
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(
                model=wrapper,
                training_adapter=adapter,
                training_strategy=strategy,
                args=TrainingArguments(
                    output_dir=directory,
                    max_steps=1,
                    per_device_train_batch_size=1,
                    max_grad_norm=0,
                    logging_strategy="no",
                    save_strategy="no",
                    use_cpu=True,
                ),
                train_dataset=[{
                    "model_inputs": {
                        "values": torch.ones(1),
                        "target": torch.zeros(1),
                    },
                }],
            )
            trainer.train()

        self.assertEqual(strategy.phases, ["generator", "discriminator"])

    def test_skipped_optimizer_step_does_not_advance_scheduler_or_global_step(self):
        import torch

        class OverflowStrategy(TorchTrainingStrategy):

            def __init__(self):
                super().__init__()
                self.scheduler_steps = 0

            def optimizer_step(self, optimizer, **kwargs):
                optimizer.zero_grad(set_to_none=True)
                return False

            def scheduler_step(self, scheduler, **kwargs):
                self.scheduler_steps += 1
                scheduler.step()

        strategy = OverflowStrategy()
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(
                model=self._dropout_model(),
                args=TrainingArguments(
                    output_dir=directory,
                    max_steps=1,
                    num_train_epochs=1,
                    per_device_train_batch_size=2,
                    logging_strategy="no",
                    save_strategy="no",
                    use_cpu=True,
                ),
                train_dataset=self._dataset(2),
                training_strategy=strategy,
            )
            output = trainer.train()

        self.assertEqual(output.global_step, 0)
        self.assertEqual(strategy.scheduler_steps, 0)

    def test_iterable_dataset_flushes_partial_accumulation_window(self):
        import torch

        class FiniteIterable(torch.utils.data.IterableDataset):

            def __iter__(self):
                yield {
                    "input_values": torch.tensor([1.0]),
                    "labels": torch.tensor([2.0]),
                }

        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(
                model=self._dropout_model(),
                args=TrainingArguments(
                    output_dir=directory,
                    max_steps=1,
                    gradient_accumulation_steps=4,
                    per_device_train_batch_size=1,
                    logging_strategy="no",
                    save_strategy="no",
                    use_cpu=True,
                ),
                train_dataset=FiniteIterable(),
            )
            output = trainer.train()

        self.assertEqual(output.global_step, 1)

    def test_model_init_is_seeded_before_construction(self):
        import torch

        created_weights = []

        def model_init():
            model = self._dropout_model()
            created_weights.append(model.projection.weight.detach().clone())
            return model

        for _ in range(2):
            with tempfile.TemporaryDirectory() as directory:
                Trainer(
                    model_init=model_init,
                    args=TrainingArguments(
                        output_dir=directory,
                        max_steps=1,
                        per_device_train_batch_size=2,
                        logging_strategy="no",
                        save_strategy="no",
                        seed=909,
                        use_cpu=True,
                    ),
                    train_dataset=self._dataset(2),
                ).train()

        torch.testing.assert_close(
            created_weights[0],
            created_weights[1],
            rtol=0,
            atol=0,
        )

    def test_optimizer_bundle_load_is_topology_strict(self):
        import torch

        parameter = torch.nn.Parameter(torch.tensor(0.0))
        bundle = OptimizerBundle({
            "generator": torch.optim.SGD([parameter], lr=0.1),
        })
        with self.assertRaisesRegex(ValueError, "topology"):
            bundle.load_state_dict({"discriminator": {}})


if __name__ == "__main__":
    unittest.main()
