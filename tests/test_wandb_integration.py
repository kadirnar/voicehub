import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

from voicehub.errors import OptionalDependencyError
from voicehub.training.arguments import TrainingArguments
from voicehub.training.callbacks import TrainerCallback
from voicehub.training.integrations import WandbCallback
from voicehub.training.trainer import Trainer
from voicehub.training.utils import CHECKPOINT_COMPLETE_NAME, MODEL_STATE_NAME, TRAINING_ARGS_NAME

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class _FakeConfig(dict):

    def __init__(self):
        super().__init__()
        self.allow_val_change = None

    def update(self, values, allow_val_change=None):
        self.allow_val_change = allow_val_change
        super().update(values)


class _FakeArtifact:

    def __init__(self, name, *, type, metadata):
        self.name = name
        self.type = type
        self.metadata = metadata
        self.directories = []

    def add_dir(self, path):
        self.directories.append(path)


class _FakeRun:

    def __init__(self, run_id="run-123", name="readable run"):
        self.id = run_id
        self.name = name
        self.config = _FakeConfig()
        self.logged = []
        self.artifacts = []
        self.finish_calls = 0
        self.events = []

    def log(self, values):
        self.logged.append(values)
        self.events.append("log")

    def log_artifact(self, artifact, *, aliases):
        self.artifacts.append((artifact, aliases))
        self.events.append("artifact")

    def finish(self):
        self.finish_calls += 1
        self.events.append("finish")


def _fake_wandb(*, existing_run=None):
    module = ModuleType("wandb")
    module.run = existing_run
    module.init_calls = []
    module.metric_calls = []
    module.created_runs = []
    module.Artifact = _FakeArtifact

    def init(**kwargs):
        module.init_calls.append(kwargs)
        run = _FakeRun(run_id=kwargs.get("id", "run-123"))
        module.created_runs.append(run)
        module.run = run
        return run

    def define_metric(*args, **kwargs):
        module.metric_calls.append((args, kwargs))

    module.init = init
    module.define_metric = define_metric
    return module


class WandbArgumentsTests(unittest.TestCase):

    def test_wandb_arguments_normalize_and_round_trip(self):
        arguments = TrainingArguments(
            report_to="wandb",
            run_name="baseline",
            wandb_project="voicehub-tests",
            wandb_entity="speech-team",
            wandb_group="asr",
            wandb_tags=["speech", "speech", "ctc"],
            wandb_notes="A deterministic run",
            wandb_mode="OFFLINE",
            wandb_log_model=True,
        )
        self.assertEqual(arguments.report_to, ["wandb"])
        self.assertEqual(arguments.wandb_tags, ["speech", "ctc"])
        self.assertEqual(arguments.wandb_mode, "offline")
        self.assertEqual(arguments.wandb_log_model, "end")

        with tempfile.TemporaryDirectory() as directory:
            path = arguments.save_json(Path(directory) / "args.json")
            restored = TrainingArguments.from_json_file(path)
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(restored.to_dict(), arguments.to_dict())
        self.assertEqual(payload["wandb_log_model"], "end")

    def test_report_to_special_values_and_validation(self):
        self.assertEqual(TrainingArguments(report_to="none").report_to, [])
        self.assertEqual(TrainingArguments(report_to="all").report_to, ["wandb"])
        self.assertEqual(
            TrainingArguments(report_to=["wandb", "WANDB"]).report_to,
            ["wandb"],
        )
        invalid_values = (
            {
                "report_to": ["wandb", "none"]
            },
            {
                "report_to": ["tensorboard"]
            },
            {
                "report_to": [1]
            },
            {
                "wandb_mode": "sometimes"
            },
            {
                "wandb_tags": ["valid", ""]
            },
            {
                "wandb_log_model": "every-step"
            },
        )
        for values in invalid_values:
            with self.subTest(values=values):
                with self.assertRaises((TypeError, ValueError)):
                    TrainingArguments(**values)


class WandbCallbackTests(unittest.TestCase):

    def test_default_trainer_does_not_import_wandb(self):
        script = (
            "import sys;from voicehub.training.trainer import Trainer;Trainer(model=object());print('wandb' in sys.modules)"
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.stdout.strip(), "False")

    def test_report_to_registers_callback_without_eager_import(self):
        with patch("voicehub.training.integrations.import_optional") as import_dependency:
            with tempfile.TemporaryDirectory() as directory:
                trainer = Trainer(
                    model=object(),
                    args=TrainingArguments(
                        output_dir=directory,
                        report_to=["wandb"],
                    ),
                )
        callbacks = trainer.callback_handler.callbacks
        self.assertEqual(
            sum(isinstance(callback, WandbCallback) for callback in callbacks),
            1,
        )
        import_dependency.assert_not_called()

    def test_owned_run_receives_config_metrics_and_is_finished(self):
        fake_wandb = _fake_wandb()
        patched_import = patch(
            "voicehub.training.integrations.import_optional",
            return_value=fake_wandb,
        )
        with tempfile.TemporaryDirectory() as directory, patched_import:
            trainer = Trainer(
                model=object(),
                args=TrainingArguments(
                    output_dir=directory,
                    report_to="wandb",
                    run_name="readable run",
                    wandb_project="voicehub-project",
                    wandb_entity="speech-team",
                    wandb_group="ctc",
                    wandb_tags=["asr", "baseline"],
                    wandb_notes="integration test",
                    wandb_mode="offline",
                ),
            )
            trainer.state.global_step = 7
            trainer.callback_handler.on_train_begin(
                trainer.args,
                trainer.state,
                trainer.control,
            )
            trainer.log({
                "loss": 0.25,
                "eval_wer": 0.1,
                "train_runtime": 3.0,
            })
            trainer.callback_handler.on_train_end(
                trainer.args,
                trainer.state,
                trainer.control,
            )
            trainer.callback_handler.on_train_end(
                trainer.args,
                trainer.state,
                trainer.control,
            )

        init_call = fake_wandb.init_calls[0]
        self.assertEqual(init_call["project"], "voicehub-project")
        self.assertEqual(init_call["entity"], "speech-team")
        self.assertEqual(init_call["group"], "ctc")
        self.assertEqual(init_call["tags"], ["asr", "baseline"])
        self.assertEqual(init_call["mode"], "offline")
        self.assertEqual(init_call["job_type"], "train")
        self.assertEqual(init_call["config"]["training"]["report_to"], ["wandb"])
        run = fake_wandb.created_runs[0]
        self.assertEqual(run.logged[0]["train/loss"], 0.25)
        self.assertEqual(run.logged[0]["eval/wer"], 0.1)
        self.assertEqual(run.logged[0]["train/runtime"], 3.0)
        self.assertEqual(run.logged[0]["train/global_step"], 7)
        self.assertEqual(run.finish_calls, 1)

    def test_existing_run_is_reused_and_not_finished(self):
        existing_run = _FakeRun(run_id="external")
        fake_wandb = _fake_wandb(existing_run=existing_run)
        with patch(
                "voicehub.training.integrations.import_optional",
                return_value=fake_wandb,
        ):
            trainer = Trainer(
                model=object(),
                args=TrainingArguments(report_to="wandb"),
            )
            trainer.callback_handler.on_train_begin(
                trainer.args,
                trainer.state,
                trainer.control,
            )
            trainer.log({"loss": 1.0})
            trainer.callback_handler.on_train_end(
                trainer.args,
                trainer.state,
                trainer.control,
            )

        self.assertEqual(fake_wandb.init_calls, [])
        self.assertEqual(existing_run.config.allow_val_change, True)
        self.assertEqual(existing_run.logged[0]["train/loss"], 1.0)
        self.assertEqual(existing_run.finish_calls, 0)

    def test_existing_run_destination_is_captured_for_exact_resume(self):
        existing_run = _FakeRun(run_id="external")
        existing_run.project = "user-managed-project"
        existing_run.entity = "speech-team"
        fake_wandb = _fake_wandb(existing_run=existing_run)
        callback = WandbCallback()
        with patch(
                "voicehub.training.integrations.import_optional",
                return_value=fake_wandb,
        ):
            trainer = Trainer(
                model=object(),
                args=TrainingArguments(report_to="none"),
                callbacks=[callback],
            )
            trainer.callback_handler.on_train_begin(
                trainer.args,
                trainer.state,
                trainer.control,
            )

        fingerprint = callback.resume_fingerprint()
        self.assertEqual(fingerprint["project"], "user-managed-project")
        self.assertEqual(fingerprint["entity"], "speech-team")

    def test_non_primary_process_never_imports_or_logs(self):
        callback = WandbCallback()
        arguments = TrainingArguments(report_to="wandb")
        trainer = Trainer(model=object(), args=arguments, callbacks=[callback])
        trainer.state.is_world_process_zero = False
        with patch(
                "voicehub.training.integrations.import_optional",
                side_effect=AssertionError("must remain lazy"),
        ):
            trainer.callback_handler.on_train_begin(
                trainer.args,
                trainer.state,
                trainer.control,
            )
            trainer.log({"loss": 1.0})
            trainer.callback_handler.on_train_end(
                trainer.args,
                trainer.state,
                trainer.control,
            )

    def test_prediction_metrics_are_logged_with_the_test_namespace(self):
        fake_wandb = _fake_wandb()
        with patch(
                "voicehub.training.integrations.import_optional",
                return_value=fake_wandb,
        ):
            trainer = Trainer(
                model=object(),
                args=TrainingArguments(report_to="wandb"),
            )
            trainer.state.global_step = 3
            trainer.callback_handler.on_train_begin(
                trainer.args,
                trainer.state,
                trainer.control,
            )
            trainer.callback_handler.on_predict(
                trainer.args,
                trainer.state,
                trainer.control,
                {"test_wer": 0.08},
            )

        self.assertEqual(
            fake_wandb.created_runs[0].logged,
            [{
                "test/wer": 0.08,
                "train/global_step": 3,
            }],
        )

    def test_logs_before_training_do_not_create_an_unowned_run(self):
        trainer = Trainer(
            model=object(),
            args=TrainingArguments(report_to="wandb"),
        )
        with patch(
                "voicehub.training.integrations.import_optional",
                side_effect=AssertionError("pre-training logs must remain lazy"),
        ):
            trainer.log({"eval_loss": 0.5})
            trainer.callback_handler.on_predict(
                trainer.args,
                trainer.state,
                trainer.control,
                {"test_wer": 0.1},
            )

    def test_missing_sdk_has_actionable_optional_dependency_error(self):
        trainer = Trainer(
            model=object(),
            args=TrainingArguments(report_to="wandb"),
        )
        error = OptionalDependencyError("install voicehub[training]")
        with patch(
                "voicehub.training.integrations.import_optional",
                side_effect=error,
        ):
            with self.assertRaisesRegex(
                    OptionalDependencyError,
                    r"voicehub\[training\]",
            ):
                trainer.callback_handler.on_train_begin(
                    trainer.args,
                    trainer.state,
                    trainer.control,
                )

    def test_owned_run_is_finished_when_training_fails(self):
        fake_wandb = _fake_wandb()
        with patch(
                "voicehub.training.integrations.import_optional",
                return_value=fake_wandb,
        ):
            trainer = Trainer(
                model=object(),
                args=TrainingArguments(report_to="wandb"),
            )

            def fail_training(checkpoint):
                trainer.callback_handler.on_train_begin(
                    trainer.args,
                    trainer.state,
                    trainer.control,
                )
                raise RuntimeError("training failed")

            with patch.object(
                    trainer,
                    "_train_impl",
                    side_effect=fail_training,
            ):
                with self.assertRaisesRegex(RuntimeError, "training failed"):
                    trainer.train()

        self.assertEqual(fake_wandb.created_runs[0].finish_calls, 1)

    def test_preflight_failure_does_not_disable_reporting_for_retry(self):
        callback = WandbCallback()
        trainer = Trainer(
            model=object(),
            args=TrainingArguments(report_to="none"),
            callbacks=[callback],
        )
        with patch.object(
                trainer,
                "_train_impl",
                side_effect=ValueError("preflight failed"),
        ):
            with self.assertRaisesRegex(ValueError, "preflight failed"):
                trainer.train()

        self.assertFalse(callback._finished)
        fake_wandb = _fake_wandb()
        with patch(
                "voicehub.training.integrations.import_optional",
                return_value=fake_wandb,
        ):
            trainer.callback_handler.on_train_begin(
                trainer.args,
                trainer.state,
                trainer.control,
            )

        self.assertEqual(len(fake_wandb.init_calls), 1)

    def test_failed_finish_can_be_retried_during_error_cleanup(self):
        fake_wandb = _fake_wandb()
        callback = WandbCallback()
        trainer = Trainer(
            model=object(),
            args=TrainingArguments(report_to="none"),
            callbacks=[callback],
        )
        with patch(
                "voicehub.training.integrations.import_optional",
                return_value=fake_wandb,
        ):
            trainer.callback_handler.on_train_begin(
                trainer.args,
                trainer.state,
                trainer.control,
            )
            run = fake_wandb.created_runs[0]
            run.finish = Mock(side_effect=[RuntimeError("finish failed"), None])
            with self.assertRaisesRegex(RuntimeError, "finish failed"):
                trainer.callback_handler.on_train_end(
                    trainer.args,
                    trainer.state,
                    trainer.control,
                )
            trainer.callback_handler.on_train_error(
                trainer.args,
                trainer.state,
                trainer.control,
                RuntimeError("finish failed"),
            )

        self.assertEqual(run.finish.call_count, 2)
        self.assertTrue(callback._finished)

    def test_partially_initialized_run_is_owned_and_closed(self):
        fake_wandb = _fake_wandb()
        partial_run = _FakeRun(run_id="partial")

        def fail_init(**kwargs):
            fake_wandb.run = partial_run
            raise RuntimeError("initialization failed")

        fake_wandb.init = fail_init
        callback = WandbCallback()
        trainer = Trainer(
            model=object(),
            args=TrainingArguments(report_to="none"),
            callbacks=[callback],
        )
        with patch(
                "voicehub.training.integrations.import_optional",
                return_value=fake_wandb,
        ):
            with self.assertRaisesRegex(RuntimeError, "initialization failed"):
                trainer.callback_handler.on_train_begin(
                    trainer.args,
                    trainer.state,
                    trainer.control,
                )
            trainer.callback_handler.on_train_error(
                trainer.args,
                trainer.state,
                trainer.control,
                RuntimeError("initialization failed"),
            )

        self.assertEqual(partial_run.finish_calls, 1)
        self.assertTrue(callback._finished)

    def test_partial_init_cleanup_failure_preserves_ownership_for_retry(self):
        fake_wandb = _fake_wandb()
        partial_run = _FakeRun(run_id="partial")
        finish_attempts = 0

        def flaky_finish():
            nonlocal finish_attempts
            finish_attempts += 1
            if finish_attempts == 1:
                raise RuntimeError("cleanup failed")
            fake_wandb.run = None

        partial_run.finish = Mock(side_effect=flaky_finish)

        def flaky_init(**kwargs):
            fake_wandb.init_calls.append(kwargs)
            if len(fake_wandb.init_calls) == 1:
                fake_wandb.run = partial_run
                raise RuntimeError("initialization failed")
            run = _FakeRun(run_id="retry-run")
            fake_wandb.created_runs.append(run)
            fake_wandb.run = run
            return run

        fake_wandb.init = flaky_init
        callback = WandbCallback()
        trainer = Trainer(
            model=object(),
            args=TrainingArguments(report_to="none"),
            callbacks=[callback],
        )
        with patch(
                "voicehub.training.integrations.import_optional",
                return_value=fake_wandb,
        ):
            with self.assertRaisesRegex(RuntimeError, "initialization failed"):
                trainer.callback_handler.on_train_begin(
                    trainer.args,
                    trainer.state,
                    trainer.control,
                )
            with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
                trainer.callback_handler.on_train_error(
                    trainer.args,
                    trainer.state,
                    trainer.control,
                    RuntimeError("initialization failed"),
                )
            self.assertTrue(callback._owns_run)
            callback.load_state_dict({"run_id": "restored-run"})
            trainer.callback_handler.on_train_begin(
                trainer.args,
                trainer.state,
                trainer.control,
            )

        self.assertEqual(partial_run.finish.call_count, 2)
        self.assertIs(callback._run, fake_wandb.created_runs[0])
        self.assertTrue(callback._owns_run)

    def test_preflight_failure_after_success_keeps_next_run_independent(self):
        fake_wandb = _fake_wandb()
        callback = WandbCallback()
        trainer = Trainer(
            model=object(),
            args=TrainingArguments(report_to="none"),
            callbacks=[callback],
        )
        with patch(
                "voicehub.training.integrations.import_optional",
                return_value=fake_wandb,
        ):
            trainer.callback_handler.on_train_begin(
                trainer.args,
                trainer.state,
                trainer.control,
            )
            first_run = fake_wandb.created_runs[0]
            first_finish = first_run.finish

            def finish_and_clear_global_run():
                first_finish()
                fake_wandb.run = None

            first_run.finish = finish_and_clear_global_run
            trainer.callback_handler.on_train_end(
                trainer.args,
                trainer.state,
                trainer.control,
            )
            with patch.object(
                    trainer,
                    "_train_impl",
                    side_effect=ValueError("preflight failed"),
            ):
                with self.assertRaisesRegex(ValueError, "preflight failed"):
                    trainer.train()
            trainer.callback_handler.on_train_begin(
                trainer.args,
                trainer.state,
                trainer.control,
            )

        self.assertEqual(len(fake_wandb.init_calls), 2)
        self.assertNotIn("id", fake_wandb.init_calls[1])
        self.assertNotIn("resume", fake_wandb.init_calls[1])

    def test_independent_training_lifecycles_create_distinct_runs(self):
        fake_wandb = _fake_wandb()
        callback = WandbCallback()
        trainer = Trainer(
            model=object(),
            args=TrainingArguments(report_to="none"),
            callbacks=[callback],
        )
        with patch(
                "voicehub.training.integrations.import_optional",
                return_value=fake_wandb,
        ):
            trainer.callback_handler.on_train_begin(
                trainer.args,
                trainer.state,
                trainer.control,
            )
            first_run = fake_wandb.created_runs[0]
            first_finish = first_run.finish

            def finish_and_clear_global_run():
                first_finish()
                fake_wandb.run = None

            first_run.finish = finish_and_clear_global_run
            trainer.callback_handler.on_train_end(
                trainer.args,
                trainer.state,
                trainer.control,
            )
            trainer.callback_handler.on_train_begin(
                trainer.args,
                trainer.state,
                trainer.control,
            )

        self.assertEqual(len(fake_wandb.init_calls), 2)
        self.assertNotIn("id", fake_wandb.init_calls[1])
        self.assertNotIn("resume", fake_wandb.init_calls[1])

    def test_error_retry_resumes_the_owned_run(self):
        fake_wandb = _fake_wandb()
        callback = WandbCallback()
        trainer = Trainer(
            model=object(),
            args=TrainingArguments(report_to="none"),
            callbacks=[callback],
        )
        with patch(
                "voicehub.training.integrations.import_optional",
                return_value=fake_wandb,
        ):
            trainer.callback_handler.on_train_begin(
                trainer.args,
                trainer.state,
                trainer.control,
            )
            first_run = fake_wandb.created_runs[0]
            first_finish = first_run.finish

            def finish_and_clear_global_run():
                first_finish()
                fake_wandb.run = None

            first_run.finish = finish_and_clear_global_run
            trainer.callback_handler.on_train_error(
                trainer.args,
                trainer.state,
                trainer.control,
                RuntimeError("training failed"),
            )
            trainer.callback_handler.on_train_begin(
                trainer.args,
                trainer.state,
                trainer.control,
            )

        self.assertEqual(fake_wandb.init_calls[1]["id"], first_run.id)
        self.assertEqual(fake_wandb.init_calls[1]["resume"], "allow")

    def test_resume_fingerprint_tracks_live_reporting_arguments(self):
        callback = WandbCallback()
        trainer = Trainer(
            model=object(),
            args=TrainingArguments(
                report_to="none",
                wandb_project="project-a",
            ),
            callbacks=[callback],
        )

        trainer.args.wandb_project = "project-b"

        self.assertEqual(
            callback.resume_fingerprint()["project"],
            "project-b",
        )

    def test_environment_resolved_project_is_part_of_resume_fingerprint(self):
        callback = WandbCallback()
        with patch.dict(os.environ, {"WANDB_PROJECT": "project-a"}, clear=False):
            Trainer(
                model=object(),
                args=TrainingArguments(report_to="none"),
                callbacks=[callback],
            )
            first_fingerprint = callback.resume_fingerprint()
        with patch.dict(os.environ, {"WANDB_PROJECT": "project-b"}, clear=False):
            second_fingerprint = callback.resume_fingerprint()

        self.assertEqual(first_fingerprint["project"], "project-a")
        self.assertEqual(second_fingerprint["project"], "project-b")
        self.assertNotEqual(first_fingerprint, second_fingerprint)

    def test_restored_run_id_is_used_for_resumption(self):
        fake_wandb = _fake_wandb()
        with patch(
                "voicehub.training.integrations.import_optional",
                return_value=fake_wandb,
        ):
            callback = WandbCallback()
            trainer = Trainer(
                model=object(),
                args=TrainingArguments(report_to="none"),
                callbacks=[callback],
            )
            callback.load_state_dict({"run_id": "resumed-run"})
            trainer.callback_handler.on_train_begin(
                trainer.args,
                trainer.state,
                trainer.control,
            )

        self.assertEqual(fake_wandb.init_calls[0]["id"], "resumed-run")
        self.assertEqual(fake_wandb.init_calls[0]["resume"], "allow")
        self.assertEqual(callback.state_dict(), {"run_id": "resumed-run"})

    def test_checkpoint_artifact_is_logged_only_after_completion(self):
        fake_wandb = _fake_wandb()
        patched_import = patch(
            "voicehub.training.integrations.import_optional",
            return_value=fake_wandb,
        )
        with tempfile.TemporaryDirectory() as directory, patched_import:
            checkpoint = Path(directory) / "checkpoint-4"
            checkpoint.mkdir()
            arguments = TrainingArguments(
                output_dir=directory,
                report_to="wandb",
                wandb_log_model="checkpoint",
            )
            trainer = Trainer(model=object(), args=arguments)
            trainer.state.global_step = 4
            trainer.callback_handler.on_train_begin(
                trainer.args,
                trainer.state,
                trainer.control,
            )
            with self.assertRaisesRegex(RuntimeError, "completed"):
                trainer.callback_handler.on_checkpoint_saved(
                    trainer.args,
                    trainer.state,
                    trainer.control,
                    checkpoint,
                )
            (checkpoint / CHECKPOINT_COMPLETE_NAME).write_text(
                "complete\n",
                encoding="utf-8",
            )
            trainer.callback_handler.on_checkpoint_saved(
                trainer.args,
                trainer.state,
                trainer.control,
                checkpoint,
            )

        artifact, aliases = fake_wandb.created_runs[0].artifacts[0]
        self.assertEqual(artifact.name, "readable-run-checkpoint")
        self.assertEqual(artifact.metadata, {"global_step": 4})
        self.assertEqual(artifact.directories, [str(checkpoint)])
        self.assertEqual(aliases, ["step-4", "latest"])

    def test_end_mode_requests_and_logs_only_the_final_model_directory(self):
        fake_wandb = _fake_wandb()
        patched_import = patch(
            "voicehub.training.integrations.import_optional",
            return_value=fake_wandb,
        )
        with tempfile.TemporaryDirectory() as directory, patched_import:
            final_model = Path(directory) / "final-model"
            final_model.mkdir()
            (final_model / "model.safetensors").touch()
            trainer = Trainer(
                model=object(),
                args=TrainingArguments(
                    output_dir=directory,
                    report_to="wandb",
                    wandb_log_model="end",
                ),
            )
            self.assertTrue(trainer.callback_handler.requires_final_model(
                trainer.args,
                trainer.state,
            ))
            trainer.callback_handler.on_train_begin(
                trainer.args,
                trainer.state,
                trainer.control,
            )
            trainer.callback_handler.on_final_model_saved(
                trainer.args,
                trainer.state,
                trainer.control,
                final_model,
            )
            trainer.callback_handler.on_train_end(
                trainer.args,
                trainer.state,
                trainer.control,
            )

        run = fake_wandb.created_runs[0]
        artifact, aliases = run.artifacts[0]
        self.assertEqual(artifact.directories, [str(final_model)])
        self.assertEqual(aliases, ["final", "latest"])
        self.assertEqual(run.finish_calls, 1)

    @unittest.skipUnless(
        TORCH_AVAILABLE,
        "PyTorch is an optional training dependency",
    )
    def test_trainer_saves_final_portable_model_before_upload_and_finish(self):
        import torch

        class TinyModel(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.projection = torch.nn.Linear(1, 1)

            def forward(self, input_values, labels):
                logits = self.projection(input_values)
                return {
                    "loss": torch.nn.functional.mse_loss(logits, labels),
                    "logits": logits,
                }

        fake_wandb = _fake_wandb()
        patched_import = patch(
            "voicehub.training.integrations.import_optional",
            return_value=fake_wandb,
        )
        with tempfile.TemporaryDirectory() as directory, patched_import:
            trainer = Trainer(
                model=TinyModel(),
                args=TrainingArguments(
                    output_dir=directory,
                    max_steps=1,
                    per_device_train_batch_size=1,
                    logging_strategy="no",
                    save_strategy="no",
                    use_cpu=True,
                    report_to="wandb",
                    wandb_log_model="end",
                ),
                train_dataset=[{
                    "input_values": torch.tensor([1.0]),
                    "labels": torch.tensor([2.0]),
                }],
            )
            trainer.train()
            final_model = Path(directory) / "final-model"
            self.assertTrue((final_model / MODEL_STATE_NAME).is_file())
            self.assertTrue((final_model / TRAINING_ARGS_NAME).is_file())

        run = fake_wandb.created_runs[0]
        self.assertEqual(run.events[-3:], ["log", "artifact", "finish"])
        artifact, _ = run.artifacts[0]
        self.assertEqual(artifact.directories, [str(final_model)])
        self.assertIn("train/runtime", run.logged[-1])

    @unittest.skipUnless(
        TORCH_AVAILABLE,
        "PyTorch is an optional training dependency",
    )
    def test_final_artifact_failures_emit_only_the_error_terminal_event(self):
        import torch

        class TinyModel(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.projection = torch.nn.Linear(1, 1)

            def forward(self, input_values, labels):
                logits = self.projection(input_values)
                return {
                    "loss": torch.nn.functional.mse_loss(logits, labels),
                    "logits": logits,
                }

        class TerminalObserver(TrainerCallback):

            def __init__(self):
                self.events = []

            def on_train_end(self, args, state, control, **kwargs):
                self.events.append("end")
                return control

            def on_train_error(self, args, state, control, **kwargs):
                self.events.append("error")
                return control

        for failure_point in ("save", "upload"):
            with self.subTest(failure_point=failure_point):
                fake_wandb = _fake_wandb()
                observer = TerminalObserver()
                with tempfile.TemporaryDirectory() as directory, patch(
                        "voicehub.training.integrations.import_optional",
                        return_value=fake_wandb,
                ):
                    trainer = Trainer(
                        model=TinyModel(),
                        args=TrainingArguments(
                            output_dir=directory,
                            max_steps=1,
                            per_device_train_batch_size=1,
                            logging_strategy="no",
                            save_strategy="no",
                            use_cpu=True,
                            report_to="wandb",
                            wandb_log_model="end",
                        ),
                        train_dataset=[{
                            "input_values": torch.tensor([1.0]),
                            "labels": torch.tensor([2.0]),
                        }],
                        callbacks=[observer],
                    )
                    if failure_point == "save":
                        failure = patch.object(
                            trainer,
                            "_save_final_model",
                            side_effect=RuntimeError("final save failed"),
                        )
                    else:
                        failure = patch.object(
                            WandbCallback,
                            "_log_artifact",
                            side_effect=RuntimeError("artifact upload failed"),
                        )
                    with failure, self.assertRaisesRegex(
                            RuntimeError,
                            "failed",
                    ):
                        trainer.train()

                self.assertEqual(observer.events, ["error"])
                self.assertEqual(fake_wandb.created_runs[0].finish_calls, 1)

    @unittest.skipUnless(
        TORCH_AVAILABLE,
        "PyTorch is an optional training dependency",
    )
    def test_final_artifact_replaces_stale_directory_before_upload(self):
        import torch

        class TinyModel(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.projection = torch.nn.Linear(1, 1)

            def forward(self, input_values, labels):
                logits = self.projection(input_values)
                return {
                    "loss": torch.nn.functional.mse_loss(logits, labels),
                    "logits": logits,
                }

        fake_wandb = _fake_wandb()
        with tempfile.TemporaryDirectory() as directory, patch(
                "voicehub.training.integrations.import_optional",
                return_value=fake_wandb,
        ):
            final_model = Path(directory) / "final-model"
            final_model.mkdir()
            stale_file = final_model / "must-not-upload.txt"
            stale_file.write_text("old or sensitive data", encoding="utf-8")
            trainer = Trainer(
                model=TinyModel(),
                args=TrainingArguments(
                    output_dir=directory,
                    overwrite_output_dir=True,
                    max_steps=1,
                    per_device_train_batch_size=1,
                    logging_strategy="no",
                    save_strategy="no",
                    use_cpu=True,
                    report_to="wandb",
                    wandb_log_model="end",
                ),
                train_dataset=[{
                    "input_values": torch.tensor([1.0]),
                    "labels": torch.tensor([2.0]),
                }],
            )
            trainer.train()

            self.assertFalse(stale_file.exists())
            artifact, _ = fake_wandb.created_runs[0].artifacts[0]
            self.assertEqual(artifact.directories, [str(final_model)])

    def test_artifact_names_are_bounded_and_collision_resistant(self):
        run = _FakeRun(name=f"{'a' * 200}alpha")
        other_run = _FakeRun(name=f"{'a' * 200}beta")
        arguments = TrainingArguments(run_name="ignored")

        first = WandbCallback._artifact_name(run, arguments, "checkpoint")
        second = WandbCallback._artifact_name(other_run, arguments, "checkpoint")

        self.assertEqual(len(first), 128)
        self.assertEqual(len(second), 128)
        self.assertTrue(first.endswith("-checkpoint"))
        self.assertNotEqual(first, second)

    @unittest.skipUnless(
        TORCH_AVAILABLE,
        "PyTorch is an optional training dependency",
    )
    def test_strict_legacy_train_end_callback_remains_compatible(self):
        import torch

        class StrictEndCallback(TrainerCallback):

            def __init__(self):
                self.called = False

            def on_train_end(
                self,
                args,
                state,
                control,
                model=None,
                processing_class=None,
                optimizer=None,
                lr_scheduler=None,
            ):
                self.called = True
                return control

        class TinyModel(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.projection = torch.nn.Linear(1, 1)

            def forward(self, input_values, labels):
                logits = self.projection(input_values)
                return {
                    "loss": torch.nn.functional.mse_loss(logits, labels),
                    "logits": logits,
                }

        callback = StrictEndCallback()
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
                    "input_values": torch.tensor([1.0]),
                    "labels": torch.tensor([2.0]),
                }],
                callbacks=[callback],
            )
            trainer.train()

        self.assertTrue(callback.called)

    def test_post_save_event_observes_completed_checkpoint(self):

        class CompletionObserver(TrainerCallback):

            def __init__(self):
                self.completed = False

            def on_checkpoint_saved(
                self,
                args,
                state,
                control,
                checkpoint_path=None,
                **kwargs,
            ):
                self.completed = (Path(checkpoint_path) / CHECKPOINT_COMPLETE_NAME).is_file()
                return control

        observer = CompletionObserver()
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(
                model=object(),
                args=TrainingArguments(output_dir=directory),
                callbacks=[observer],
            )
            checkpoint = Path(directory) / "checkpoint-1"

            def save_checkpoint():
                checkpoint.mkdir()
                (checkpoint / CHECKPOINT_COMPLETE_NAME).touch()
                return checkpoint

            trainer.control.should_save = True
            with patch.object(
                    trainer,
                    "_save_checkpoint",
                    side_effect=save_checkpoint,
            ) as save:
                trainer._maybe_log_save_evaluate(0.0)

        save.assert_called_once_with()
        self.assertTrue(observer.completed)
