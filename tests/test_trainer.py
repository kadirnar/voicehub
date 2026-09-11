import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path

from voicehub import SpeechTask
from voicehub.outputs import SpeechTrainingOutput, TTSTrainingOutput
from voicehub.training import ASRDataset, ModelTrainingSpec, TrainingFamily, TrainingSupport
from voicehub.training.adapters import BaseTrainingAdapter
from voicehub.training.arguments import TrainingArguments
from voicehub.training.asr_datasets import EpochGroupedBatchSampler
from voicehub.training.callbacks import EarlyStoppingCallback, TrainerCallback, TrainerControl, TrainerState
from voicehub.training.data_collator import DefaultDataCollator
from voicehub.training.trainer import Trainer
from voicehub.training.utils import EpochRandomSampler, IntervalStrategy

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class EventRecorder(TrainerCallback):

    def __init__(self):
        self.events = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        self.events.append(("log", state.global_step))
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        self.events.append(("evaluate", state.global_step))
        return control

    def on_save(self, args, state, control, **kwargs):
        self.events.append(("save", state.global_step))
        return control


@dataclass
class UnsafeTrainerState(TrainerState):
    """Bypass construction validation to exercise the final write boundary."""

    provider_options: dict[str, object] = field(default_factory=dict)

    def __post_init__(self):
        pass


@dataclass
class UnsafeTrainingArguments(TrainingArguments):
    """Bypass construction validation to exercise final write checks."""

    provider_options: dict[str, object] = field(default_factory=dict)

    def __post_init__(self):
        pass


class TrainerApiTests(unittest.TestCase):

    def test_training_arguments_round_trip_and_alias(self):
        arguments = TrainingArguments(
            output_dir="trainer-output",
            evaluation_strategy="steps",
            eval_steps=2,
            save_strategy="steps",
            save_steps=2,
            load_best_model_at_end=True,
        )
        self.assertIs(arguments.eval_strategy, IntervalStrategy.STEPS)
        self.assertEqual(arguments.metric_for_best_model, "loss")
        self.assertFalse(arguments.greater_is_better)

        with tempfile.TemporaryDirectory() as directory:
            path = arguments.save_json(Path(directory) / "training_args.json")
            restored = TrainingArguments.from_json_file(path)
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(restored.to_dict(), arguments.to_dict())
        self.assertNotIn("evaluation_strategy", payload)

    def test_training_arguments_reject_runtime_secrets_at_every_artifact_boundary(self):
        credential = "must-not-appear-in-errors"

        @dataclass
        class ExtendedTrainingArguments(TrainingArguments):
            provider_options: dict[str, object] = field(default_factory=dict)

        with self.assertRaisesRegex(ValueError, "runtime secrets") as captured:
            ExtendedTrainingArguments(
                provider_options={
                    "credentials": {
                        "api_key": credential,
                    },
                }, )
        self.assertNotIn(credential, str(captured.exception))

        unsafe = UnsafeTrainingArguments(
            provider_options={
                "metrics": {
                    "token_count": 512,
                },
            }, )
        unsafe.provider_options["authorization"] = credential
        for operation in (
                unsafe.to_dict,
                unsafe.to_json_string,
        ):
            with self.subTest(operation=operation.__name__):
                with self.assertRaisesRegex(ValueError, "runtime secrets") as captured:
                    operation()
                self.assertNotIn(credential, str(captured.exception))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training_args.json"
            with self.assertRaisesRegex(ValueError, "runtime secrets") as captured:
                unsafe.save_json(path)
            self.assertNotIn(credential, str(captured.exception))
            self.assertFalse(path.exists())

    def test_training_arguments_loader_rejects_secrets_before_construction(self):
        construction_events = []
        credential = "must-not-appear-in-errors"

        @dataclass
        class TrackingTrainingArguments(TrainingArguments):
            provider_options: dict[str, object] = field(default_factory=dict)

            def __post_init__(self):
                construction_events.append("constructed")
                super().__post_init__()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training_args.json"
            path.write_text(
                json.dumps({
                    "provider_options": {
                        "api_key": credential,
                    },
                }),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "runtime secrets") as captured:
                TrackingTrainingArguments.from_json_file(path)

        self.assertEqual(construction_events, [])
        self.assertNotIn(credential, str(captured.exception))

    def test_training_arguments_safe_subclass_fields_round_trip(self):

        @dataclass
        class ExtendedTrainingArguments(TrainingArguments):
            run_metadata: dict[str, object] = field(default_factory=dict)

        arguments = ExtendedTrainingArguments(
            run_metadata={
                "token_count": 512,
                "dataset_id": "speech/train",
            }, )
        with tempfile.TemporaryDirectory() as directory:
            path = arguments.save_json(Path(directory) / "training_args.json")
            restored = ExtendedTrainingArguments.from_json_file(path)
        self.assertEqual(restored.to_dict(), arguments.to_dict())

    def test_training_json_loaders_reject_ambiguous_data_before_construction(self):
        construction_events = []

        @dataclass
        class TrackingTrainingArguments(TrainingArguments):

            def __post_init__(self):
                construction_events.append("training-arguments")
                super().__post_init__()

        @dataclass
        class TrackingTrainerState(TrainerState):

            def __post_init__(self):
                construction_events.append("trainer-state")
                super().__post_init__()

        documents = (
            (
                "training-arguments-duplicate",
                TrackingTrainingArguments.from_json_file,
                '{"output_dir": "discarded-secret", "output_dir": "trainer-output"}',
                "Duplicate JSON object key 'output_dir'",
            ),
            (
                "training-arguments-overflow",
                TrackingTrainingArguments.from_json_file,
                '{"learning_rate": 1e10000}',
                "$.learning_rate",
            ),
            (
                "trainer-state-duplicate",
                TrackingTrainerState.load_from_json,
                '{"log_history": [{"loss": 0.4, "loss": 0.3}]}',
                "Duplicate JSON object key 'loss'",
            ),
            (
                "trainer-state-overflow",
                TrackingTrainerState.load_from_json,
                '{"log_history": [{"loss": 1e10000}]}',
                "$.log_history[0].loss",
            ),
        )

        for name, loader, document, expected_detail in documents:
            with self.subTest(document=name):
                construction_events.clear()
                with tempfile.TemporaryDirectory() as directory:
                    source = Path(directory) / f"{name}.json"
                    source.write_text(document, encoding="utf-8")

                    with self.assertRaises(ValueError) as captured:
                        loader(source)

                rendered_error = str(captured.exception)
                self.assertIn(str(source), rendered_error)
                self.assertIn(expected_detail, rendered_error)
                self.assertNotIn("discarded-secret", rendered_error)
                self.assertEqual(construction_events, [])

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is an optional training extra")
    def test_trainer_export_rejects_unsafe_arguments_before_creating_destination(self):
        import torch

        credential = "must-not-appear-in-errors"
        arguments = UnsafeTrainingArguments(
            provider_options={
                "api_key": credential,
            },
            use_cpu=True,
        )
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "artifact"
            trainer = Trainer(
                model=torch.nn.Linear(1, 1),
                args=arguments,
            )
            with self.assertRaisesRegex(ValueError, "runtime secrets") as captured:
                trainer.save_model(destination)
            self.assertNotIn(credential, str(captured.exception))
            self.assertFalse(destination.exists())

    def test_training_arguments_validate_incompatible_values(self):
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            TrainingArguments(per_device_train_batch_size=0)
        with self.assertRaisesRegex(ValueError, "At most one"):
            TrainingArguments(fp16=True, bf16=True)
        with self.assertRaisesRegex(ValueError, "must match"):
            TrainingArguments(
                eval_strategy="steps",
                save_strategy="epoch",
                load_best_model_at_end=True,
            )

    def test_training_arguments_reject_invalid_numeric_types_and_ranges(self):
        invalid = (
            ({
                "gradient_accumulation_steps": True
            }, ValueError),
            ({
                "eval_accumulation_steps": 1.5
            }, ValueError),
            ({
                "max_steps": -2
            }, ValueError),
            ({
                "max_steps": 0
            }, ValueError),
            ({
                "num_train_epochs": float("nan")
            }, TypeError),
            ({
                "learning_rate": float("inf")
            }, ValueError),
            ({
                "adam_epsilon": 0.0
            }, ValueError),
            ({
                "adam_beta2": 1.0
            }, ValueError),
            ({
                "eval_steps": 1.5
            }, TypeError),
            ({
                "save_total_limit": True
            }, TypeError),
            ({
                "warmup_ratio": float("nan")
            }, ValueError),
            ({
                "label_names": ["labels", "labels"]
            }, ValueError),
        )
        for values, error in invalid:
            with self.subTest(values=values):
                with self.assertRaises(error):
                    TrainingArguments(**values)

    def test_trainer_state_round_trip(self):
        state = TrainerState(
            epoch=1.5,
            global_step=12,
            max_steps=20,
            log_history=[{
                "loss": 0.4,
                "step": 12
            }, {
                "token_count": 512,
                "step": 12,
            }],
        )
        with tempfile.TemporaryDirectory() as directory:
            path = state.save_to_json(Path(directory) / "trainer_state.json")
            restored = TrainerState.load_from_json(path)
        self.assertEqual(restored, state)

    def test_trainer_state_rejects_nested_runtime_secrets_at_construction(self):
        with self.assertRaisesRegex(ValueError, "runtime secrets"):
            TrainerState(
                log_history=[{
                    "provider_options": {
                        "authorization": "Bearer must-not-be-persisted",
                    },
                }])

    def test_trainer_state_loader_rejects_nested_runtime_secrets(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trainer_state.json"
            path.write_text(
                json.dumps({
                    "log_history": [{
                        "provider_options": {
                            "api_key": "must-not-be-persisted",
                        },
                    }],
                }),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "runtime secrets"):
                TrainerState.load_from_json(path)

    def test_trainer_state_rejects_mutated_secrets_before_writing(self):
        state = TrainerState()
        state.log_history.append({
            "provider_options": {
                "api_key": "must-not-be-persisted",
            },
        })

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact" / "trainer_state.json"
            with self.assertRaisesRegex(ValueError, "runtime secrets"):
                state.save_to_json(path)

            self.assertFalse(path.parent.exists())

    def test_trainer_state_writer_validates_final_subclass_payload(self):
        state = UnsafeTrainerState(provider_options={
            "api_key": "must-not-be-persisted",
        })

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact" / "trainer_state.json"
            with self.assertRaisesRegex(ValueError, "runtime secrets"):
                state.save_to_json(path)

            self.assertFalse(path.parent.exists())

    def test_trainer_log_rejects_runtime_secrets_before_state_or_callbacks(self):
        recorder = EventRecorder()
        trainer = Trainer(
            model=object(),
            args=TrainingArguments(disable_tqdm=True),
            callbacks=[recorder],
        )

        with self.assertRaisesRegex(ValueError, "runtime secrets"):
            trainer.log({
                "provider_options": {
                    "api_key": "must-not-be-persisted",
                },
            })

        self.assertEqual(trainer.state.log_history, [])
        self.assertEqual(recorder.events, [])

    def test_training_output_uses_loss_first_mapping_contract(self):
        output = TTSTrainingOutput(
            loss=0.25,
            logits=[1.0],
            metadata={"architecture": "dummy"},
        )
        self.assertEqual(output[0], 0.25)
        self.assertEqual(output["logits"], [1.0])
        self.assertEqual(
            output.keys(),
            ("loss", "logits", "metadata"),
        )

    def test_control_resets_recurring_actions(self):
        control = TrainerControl(
            should_save=True,
            should_evaluate=True,
            should_log=True,
        )
        control._new_step()
        self.assertFalse(control.should_save)
        self.assertFalse(control.should_evaluate)
        self.assertFalse(control.should_log)

    def test_early_stopping_uses_best_metric_and_patience(self):
        arguments = TrainingArguments(
            eval_strategy="steps",
            eval_steps=1,
            save_steps=1,
            load_best_model_at_end=True,
        )
        state = TrainerState(best_metric=0.5)
        control = TrainerControl()
        callback = EarlyStoppingCallback(early_stopping_patience=2)
        callback.on_train_begin(arguments, state, control)
        callback.on_evaluate(
            arguments,
            state,
            control,
            metrics={"eval_loss": 0.6},
        )
        self.assertFalse(control.should_training_stop)
        callback.on_evaluate(
            arguments,
            state,
            control,
            metrics={"eval_loss": 0.7},
        )
        self.assertTrue(control.should_training_stop)

    def test_public_trainer_import_remains_framework_lazy(self):
        script = (
            "import sys;from voicehub.training import Trainer, TrainingArguments;print('torch' in sys.modules, 'numpy' in sys.modules)"
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.stdout.strip(), "False False")

    def test_checkpoint_rotation_keeps_best_and_latest(self):
        with tempfile.TemporaryDirectory() as directory:
            arguments = TrainingArguments(
                output_dir=directory,
                save_total_limit=1,
            )
            trainer = Trainer(model=object(), args=arguments)
            best = Path(directory) / "checkpoint-1"
            latest = Path(directory) / "checkpoint-2"
            best.mkdir()
            latest.mkdir()
            trainer.state.best_model_checkpoint = str(best)
            trainer._rotate_checkpoints()
            remaining = sorted(path.name for path in Path(directory).glob("checkpoint-*"))
        self.assertEqual(remaining, ["checkpoint-1", "checkpoint-2"])

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is an optional training extra")
    def test_default_collator_stacks_tensors_and_preserves_text(self):
        import numpy
        import torch

        collator = DefaultDataCollator()
        batch = collator([
            {
                "input_values": torch.tensor([1.0]),
                "audio_values": numpy.asarray([0.1, 0.2]),
                "text": "one",
                "label": 1.0,
            },
            {
                "input_values": torch.tensor([2.0]),
                "audio_values": numpy.asarray([0.3, 0.4]),
                "text": "two",
                "label": 2.0,
            },
        ])
        self.assertEqual(tuple(batch["input_values"].shape), (2, 1))
        self.assertEqual(tuple(batch["audio_values"].shape), (2, 2))
        self.assertEqual(batch["text"], ["one", "two"])
        self.assertEqual(batch["labels"].tolist(), [1.0, 2.0])

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is an optional training extra")
    def test_each_dataset_can_supply_its_own_collator(self):
        import torch

        class Dataset(list):

            def __init__(self, name):
                super().__init__([{"value": torch.tensor([1.0])}])
                self.name = name

            def collate_fn(self, features):
                return {
                    "dataset": self.name,
                    "value": torch.stack([feature["value"] for feature in features]),
                }

        train_dataset = Dataset("train")
        eval_dataset = Dataset("eval")
        trainer = Trainer(
            model=torch.nn.Linear(1, 1),
            args=TrainingArguments(
                per_device_train_batch_size=1,
                per_device_eval_batch_size=1,
                use_cpu=True,
            ),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        train_batch = next(iter(trainer.get_train_dataloader()))
        eval_batch = next(iter(trainer.get_eval_dataloader()))

        self.assertEqual(train_batch["dataset"], "train")
        self.assertEqual(eval_batch["dataset"], "eval")


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is an optional training extra")
class TrainerLoopTests(unittest.TestCase):

    @staticmethod
    def _model():
        import torch

        class TinyTTSModel(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.projection = torch.nn.Linear(1, 1)

            def forward(self, input_values, labels=None):
                logits = self.projection(input_values)
                loss = None
                if labels is not None:
                    loss = torch.nn.functional.mse_loss(logits, labels)
                return TTSTrainingOutput(loss=loss, logits=logits)

        return TinyTTSModel()

    @staticmethod
    def _dataset():
        import torch

        return [{
            "input_values": torch.tensor([float(index)]),
            "labels": torch.tensor([float(index * 2)]),
            "metadata": f"sample-{index}",
        } for index in range(8)]

    def test_training_loss_validation_rejects_invalid_graphs(self):
        import torch

        parameter = torch.nn.Parameter(torch.tensor(1.0))
        valid_loss = parameter.square()
        self.assertIs(
            Trainer._validate_training_loss(valid_loss),
            valid_loss,
        )
        with self.assertRaisesRegex(TypeError, "PyTorch tensor"):
            Trainer._validate_training_loss(1.0)
        with self.assertRaisesRegex(ValueError, "scalar tensor"):
            Trainer._validate_training_loss(parameter.expand(2))
        with self.assertRaisesRegex(ValueError, "detached"):
            Trainer._validate_training_loss(torch.tensor(1.0))
        with self.assertRaisesRegex(FloatingPointError, "NaN or infinite"):
            Trainer._validate_training_loss(parameter * torch.tensor(float("nan")))

    def test_train_evaluate_predict_callbacks_and_checkpoints(self):
        recorder = EventRecorder()
        with tempfile.TemporaryDirectory() as directory:
            arguments = TrainingArguments(
                output_dir=directory,
                max_steps=4,
                per_device_train_batch_size=2,
                per_device_eval_batch_size=2,
                learning_rate=0.01,
                logging_steps=1,
                eval_strategy="steps",
                eval_steps=2,
                save_strategy="steps",
                save_steps=2,
                save_total_limit=2,
                load_best_model_at_end=True,
                use_cpu=True,
            )
            trainer = Trainer(
                model=self._model(),
                args=arguments,
                train_dataset=self._dataset(),
                eval_dataset=self._dataset(),
                compute_metrics=lambda prediction:
                {"mae": abs(prediction.predictions - prediction.label_ids).mean()},
                callbacks=[recorder],
            )
            train_output = trainer.train()
            prediction_output = trainer.predict(self._dataset())

            checkpoints = sorted(path.name for path in Path(directory).glob("checkpoint-*"))
            best_checkpoint = trainer.state.best_model_checkpoint

        self.assertEqual(train_output.global_step, 4)
        self.assertGreaterEqual(train_output.training_loss, 0.0)
        self.assertEqual(checkpoints, ["checkpoint-2", "checkpoint-4"])
        self.assertIsNotNone(best_checkpoint)
        self.assertEqual(prediction_output.predictions.shape, (8, 1))
        self.assertIn("test_mae", prediction_output.metrics)
        self.assertIn(("evaluate", 2), recorder.events)
        self.assertIn(("save", 4), recorder.events)

    def test_resume_from_last_checkpoint(self):

        class StopAtTwo(TrainerCallback):

            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step == 2:
                    control.should_save = True
                    control.should_training_stop = True
                return control

        with tempfile.TemporaryDirectory() as directory:
            first = Trainer(
                model=self._model(),
                args=TrainingArguments(
                    output_dir=directory,
                    max_steps=4,
                    per_device_train_batch_size=2,
                    logging_strategy="no",
                    save_steps=1,
                    save_total_limit=4,
                    use_cpu=True,
                ),
                train_dataset=self._dataset(),
                callbacks=[StopAtTwo],
            )
            first.train()

            resumed = Trainer(
                model=self._model(),
                args=TrainingArguments(
                    output_dir=directory,
                    max_steps=4,
                    per_device_train_batch_size=2,
                    logging_strategy="no",
                    save_steps=1,
                    save_total_limit=4,
                    use_cpu=True,
                ),
                train_dataset=self._dataset(),
            )
            output = resumed.train(resume_from_checkpoint=True)

        self.assertEqual(output.global_step, 4)

    def test_custom_compute_loss_function(self):
        import torch

        class LogitsOnly(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.projection = torch.nn.Linear(1, 1)

            def forward(self, input_values):
                return {"logits": self.projection(input_values)}

        received_items = []

        def compute_loss(outputs, labels, num_items_in_batch):
            received_items.append(num_items_in_batch)
            return torch.nn.functional.mse_loss(outputs["logits"], labels)

        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(
                model=LogitsOnly(),
                args=TrainingArguments(
                    output_dir=directory,
                    max_steps=1,
                    per_device_train_batch_size=2,
                    logging_strategy="no",
                    save_strategy="no",
                    use_cpu=True,
                ),
                train_dataset=self._dataset(),
                compute_loss_func=compute_loss,
            )
            output = trainer.train()

        self.assertEqual(output.global_step, 1)
        self.assertEqual(received_items, [2])

    def test_incomplete_accumulation_group_keeps_full_loss_scale(self):
        import torch

        class ScalarModel(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([[0.0]]))

            def forward(self, input_values, labels):
                logits = input_values @ self.weight
                loss = torch.nn.functional.mse_loss(logits, labels)
                return TTSTrainingOutput(loss=loss, logits=logits)

        model = ScalarModel()
        dataset = [{
            "input_values": torch.tensor([1.0]),
            "labels": torch.tensor([1.0]),
        }]
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(
                model=model,
                args=TrainingArguments(
                    output_dir=directory,
                    num_train_epochs=1,
                    per_device_train_batch_size=1,
                    gradient_accumulation_steps=4,
                    max_grad_norm=0,
                    logging_strategy="no",
                    save_strategy="no",
                    use_cpu=True,
                ),
                train_dataset=dataset,
                optimizer_cls_and_kwargs=(torch.optim.SGD, {
                    "lr": 1.0
                }),
            )
            trainer.train()

        self.assertAlmostEqual(model.weight.item(), 2.0)

    def test_predict_keeps_first_tuple_value_without_labels(self):
        import torch

        class TupleModel(torch.nn.Module):

            def forward(self, input_values):
                return (input_values * 3, )

        dataset = [{"input_values": torch.tensor([1.0])}, {"input_values": torch.tensor([2.0])}]
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(
                model=TupleModel(),
                args=TrainingArguments(
                    output_dir=directory,
                    per_device_eval_batch_size=2,
                    use_cpu=True,
                ),
            )
            output = trainer.predict(dataset)

        self.assertEqual(output.predictions.tolist(), [[3.0], [6.0]])

    def test_raw_transcripts_are_labels_only_for_asr_adapter_evaluation(self):
        import torch

        class NativeLossModel(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(0.5))

            def forward(self, input_values, labels=None):
                logits = input_values * self.scale
                loss = (None if labels is None else torch.nn.functional.mse_loss(logits, labels))
                return SpeechTrainingOutput(loss=loss, logits=logits)

        class RawSpeechWrapper:

            def __init__(self):
                self.model = NativeLossModel()

            def load_for_training(self):
                return self

            @staticmethod
            def prepare_training_inputs(inputs, phase=None):
                del phase
                transcripts = inputs["text"]
                batch_size = (len(transcripts) if isinstance(transcripts, list) else 1)
                return {
                    "input_values": torch.ones(batch_size, 1),
                    "labels": torch.zeros(batch_size, 1),
                }

        class RecordingAdapter(BaseTrainingAdapter):

            def __init__(self, model, spec):
                super().__init__(model, spec)
                self.supervised_calls = 0
                self.prediction_calls = 0

            def execute_training_phase(self, context):
                self.supervised_calls += 1
                return super().execute_training_phase(context)

            def execute_prediction_phase(self, context):
                self.prediction_calls += 1
                return super().execute_prediction_phase(context)

        records = [
            {
                "audio": "first.wav",
                "text": "first transcript",
            },
            {
                "audio": "second.wav",
                "text": "second transcript",
            },
        ]

        def predict_for(task):
            wrapper = RawSpeechWrapper()
            spec = ModelTrainingSpec(
                model_type=f"dummy-{task.value}",
                family=TrainingFamily.SPEECH_SEQ2SEQ,
                module_paths=("model", ),
                support=TrainingSupport.NATIVE,
                task=task,
            )
            adapter = RecordingAdapter(wrapper, spec)
            trainer = Trainer(
                model=wrapper,
                args=TrainingArguments(
                    per_device_eval_batch_size=2,
                    use_cpu=True,
                ),
                data_collator=DefaultDataCollator(),
                training_adapter=adapter,
            )
            return trainer.predict(records), adapter

        asr_output, asr_adapter = predict_for(SpeechTask.AUTOMATIC_SPEECH_RECOGNITION, )
        self.assertEqual(asr_adapter.supervised_calls, 1)
        self.assertEqual(asr_adapter.prediction_calls, 0)
        self.assertAlmostEqual(asr_output.metrics["test_loss"], 0.25)
        self.assertEqual(
            asr_output.label_ids,
            ["first transcript", "second transcript"],
        )

        tts_output, tts_adapter = predict_for(SpeechTask.TEXT_TO_SPEECH)
        self.assertEqual(tts_adapter.supervised_calls, 0)
        self.assertEqual(tts_adapter.prediction_calls, 1)
        self.assertIsNone(tts_output.label_ids)

    def test_trainer_uses_resume_safe_homogeneous_asr_batch_samplers(self):
        import torch

        def cohere_records():
            return [{
                "id": f"{language}-{punctuation}-{index}",
                "audio": f"{language}-{punctuation}-{index}.wav",
                "text": "Cohere transcript.",
                "language": language,
                "punctuation": punctuation,
            } for language, punctuation in (
                ("en", True),
                ("en", False),
                ("tr", True),
            ) for index in range(4)]

        def seamless_records():
            return [{
                "id": f"{language}-{index}",
                "audio": f"{language}-{index}.wav",
                "text": "Seamless transcript.",
                "target_language": language,
            } for language in ("eng", "tur") for index in range(4)]

        def trainer_for(dataset):
            return Trainer(
                model=torch.nn.Linear(1, 1),
                args=TrainingArguments(
                    per_device_train_batch_size=3,
                    per_device_eval_batch_size=3,
                    seed=17,
                    data_seed=29,
                    use_cpu=True,
                ),
                train_dataset=dataset,
                eval_dataset=dataset,
            )

        def batch_ids(dataloader):
            return [tuple(batch["id"]) for batch in dataloader]

        for model_type, records in (
            ("asr_cohere", cohere_records()),
            ("asr_seamless_m4t_v2", seamless_records()),
        ):
            with self.subTest(model_type=model_type):
                dataset = ASRDataset(records, model_type=model_type)
                expected_groups = {
                    record["id"]: dataset.batch_group_key(record)
                    for record in dataset._records
                }
                trainer = trainer_for(dataset)
                train_dataloader = trainer.get_train_dataloader()

                self.assertIsInstance(
                    trainer._train_sampler,
                    EpochGroupedBatchSampler,
                )
                trainer._train_sampler.set_epoch(5)
                epoch_five_batches = batch_ids(train_dataloader)
                for ids in epoch_five_batches:
                    self.assertEqual(
                        len({expected_groups[item]
                             for item in ids}),
                        1,
                    )

                checkpoint_state = trainer._runtime_checkpoint_state()
                self.assertEqual(checkpoint_state["sampler"]["epoch"], 5)

                resumed = trainer_for(dataset)
                resumed_dataloader = resumed.get_train_dataloader()
                resumed._load_runtime_checkpoint_state({
                    "sampler": checkpoint_state["sampler"],
                })
                self.assertEqual(
                    batch_ids(resumed_dataloader),
                    epoch_five_batches,
                )

                first_eval_batches = batch_ids(trainer.get_eval_dataloader(), )
                second_eval_batches = batch_ids(trainer.get_eval_dataloader(), )
                self.assertEqual(first_eval_batches, second_eval_batches)
                for ids in first_eval_batches:
                    self.assertEqual(
                        len({expected_groups[item]
                             for item in ids}),
                        1,
                    )

        ordinary = trainer_for([{
            "id": f"ordinary-{index}",
        } for index in range(4)])
        ordinary.get_train_dataloader()
        self.assertIsInstance(ordinary._train_sampler, EpochRandomSampler)


if __name__ == "__main__":
    unittest.main()
