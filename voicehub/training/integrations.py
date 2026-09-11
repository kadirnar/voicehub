"""Optional experiment-tracking integrations for :class:`voicehub.Trainer`.

Integration modules are imported only when their callback starts a run.
This keeps VoiceHub importable in lightweight inference and data-
preparation environments.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any

from voicehub.dependencies import import_optional
from voicehub.training.callbacks import TrainerCallback
from voicehub.training.utils import CHECKPOINT_COMPLETE_NAME

_ARTIFACT_NAME_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")
_MAX_ARTIFACT_NAME_LENGTH = 128


def get_reporting_integration_callbacks(report_to: list[str], ) -> list[type[TrainerCallback]]:
    """Resolve normalized reporting names to lazy callback classes."""
    callbacks: list[type[TrainerCallback]] = []
    for integration in report_to:
        if integration == "wandb":
            callbacks.append(WandbCallback)
            continue
        raise ValueError(f"Unsupported reporting integration: {integration!r}.")
    return callbacks


class WandbCallback(TrainerCallback):
    """Log VoiceHub training runs to Weights & Biases.

    The SDK is loaded on the world-primary process when the first run
    event is received. Existing user-managed W&B runs are reused and
    never finished by this callback.
    """

    def __init__(self) -> None:
        self._wandb = None
        self._run = None
        self._owns_run = False
        self._initialized = False
        self._finished = False
        self._run_id: str | None = None
        self._clear_run_id_on_next_begin = False
        self._needs_cleanup_before_retry = False
        self._args = None
        self._configuration: dict[str, Any] = {}

    @staticmethod
    def _environment_value(name: str) -> str | None:
        value = os.environ.get(name)
        if value is None:
            return None
        return value.strip() or None

    @classmethod
    def _configured_values(cls, args) -> dict[str, Any]:
        tags = list(args.wandb_tags)
        environment_tags = cls._environment_value("WANDB_TAGS")
        if not tags and environment_tags is not None:
            tags = [tag.strip() for tag in environment_tags.split(",") if tag.strip()]
        mode = args.wandb_mode or cls._environment_value("WANDB_MODE")
        if mode is not None:
            mode = mode.lower()
        return {
            "run_name": args.run_name or cls._environment_value("WANDB_NAME"),
            "project": (args.wandb_project or cls._environment_value("WANDB_PROJECT") or "voicehub"),
            "entity": args.wandb_entity or cls._environment_value("WANDB_ENTITY"),
            "group": args.wandb_group or cls._environment_value("WANDB_RUN_GROUP"),
            "tags": tags,
            "notes": args.wandb_notes or cls._environment_value("WANDB_NOTES"),
            "mode": mode,
            "log_model": args.wandb_log_model,
            "base_url": (cls._environment_value("WANDB_BASE_URL") or cls._environment_value("WANDB_HOST")),
        }

    def on_init_end(self, args, state, control, **kwargs):
        self._args = args
        self._configuration = self._configured_values(args)
        return control

    def resume_fingerprint(self) -> dict[str, Any]:
        if self._initialized and not self._finished:
            return dict(self._configuration)
        if self._args is not None:
            return self._configured_values(self._args)
        return dict(self._configuration)

    def state_dict(self) -> dict[str, Any]:
        if self._run_id is None:
            return {}
        return {"run_id": self._run_id}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        run_id = state_dict.get("run_id")
        if run_id is not None and (not isinstance(run_id, str) or not run_id):
            raise ValueError("A restored W&B run ID must be a non-empty string.")
        self._run_id = run_id
        self._clear_run_id_on_next_begin = False

    @staticmethod
    def _model_metadata(model) -> dict[str, Any]:
        if model is None:
            return {}
        model_type = type(model)
        metadata: dict[str, Any] = {
            "class": f"{model_type.__module__}.{model_type.__qualname__}",
        }
        config = getattr(model, "config", None)
        configured_model_type = getattr(config, "model_type", None)
        if configured_model_type:
            metadata["model_type"] = str(configured_model_type)
        name_or_path = getattr(config, "name_or_path", None)
        if name_or_path:
            metadata["name_or_path"] = str(name_or_path)
        return metadata

    @staticmethod
    def _update_run_config(run, values: dict[str, Any]) -> None:
        config = getattr(run, "config", None)
        update = getattr(config, "update", None)
        if not callable(update):
            return
        try:
            update(values, allow_val_change=True)
        except TypeError:
            update(values)

    def _setup(self, args, state, *, model=None) -> None:
        if self._initialized or self._finished or not state.is_world_process_zero:
            return

        wandb = import_optional(
            "wandb",
            model_type="Weights & Biases reporting",
            install_extra="training",
        )
        self._wandb = wandb
        existing_run = getattr(wandb, "run", None)
        run_config = {
            "training": args.to_dict(),
            "model": self._model_metadata(model),
        }
        if existing_run is not None:
            self._run = existing_run
            self._owns_run = False
            for name in ("project", "entity"):
                value = getattr(existing_run, name, None)
                if isinstance(value, str) and value.strip():
                    self._configuration[name] = value.strip()
            self._update_run_config(existing_run, run_config)
        else:
            configuration = self._configuration
            init_kwargs: dict[str, Any] = {
                "project": configuration["project"],
                "config": run_config,
                "dir": args.output_dir,
                "job_type": "train",
            }
            for key, configuration_key in (
                ("entity", "entity"),
                ("group", "group"),
                ("name", "run_name"),
                ("notes", "notes"),
                ("mode", "mode"),
            ):
                value = configuration[configuration_key]
                if value is not None:
                    init_kwargs[key] = value
            if configuration["tags"]:
                init_kwargs["tags"] = list(configuration["tags"])
            if self._run_id is not None:
                init_kwargs["id"] = self._run_id
                init_kwargs["resume"] = "allow"

            try:
                run = wandb.init(**init_kwargs)
            except BaseException:
                partially_initialized_run = getattr(wandb, "run", None)
                if partially_initialized_run is not None:
                    self._run = partially_initialized_run
                    self._owns_run = True
                raise
            self._run = run if run is not None else getattr(wandb, "run", None)
            if self._run is None:
                raise RuntimeError("wandb.init() did not return or register a run.")
            self._owns_run = True

        resolved_run_id = getattr(self._run, "id", None)
        if resolved_run_id:
            self._run_id = str(resolved_run_id)
        define_metric = getattr(wandb, "define_metric", None)
        if callable(define_metric):
            define_metric("train/global_step")
            define_metric("*", step_metric="train/global_step")
        self._initialized = True

    @staticmethod
    def _rewrite_logs(logs: dict[str, Any]) -> dict[str, Any]:
        rewritten = {}
        for name, value in logs.items():
            if name.startswith("eval_"):
                key = f"eval/{name[5:]}"
            elif name.startswith("test_"):
                key = f"test/{name[5:]}"
            elif name.startswith("train_"):
                key = f"train/{name[6:]}"
            else:
                key = f"train/{name}"
            rewritten[key] = value
        return rewritten

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if (self._needs_cleanup_before_retry and self._owns_run and self._run is not None and
                not self._finished):
            self._finish_owned_run()

        configuration = self._configured_values(args)
        if (self._needs_cleanup_before_retry and self._run_id is not None and self._configuration and
                configuration != self._configuration):
            raise ValueError(
                "Cannot retry a W&B run after its reporting destination or "
                "configuration changed.")
        self._args = args
        self._configuration = configuration
        if self._clear_run_id_on_next_begin:
            self._run_id = None
            self._clear_run_id_on_next_begin = False
        if self._finished:
            self._wandb = None
            self._run = None
            self._owns_run = False
            self._initialized = False
            self._finished = False
        self._needs_cleanup_before_retry = False
        self._setup(args, state, model=model)
        return control

    def on_log(self, args, state, control, logs=None, model=None, **kwargs):
        if (self._finished or not self._initialized or not state.is_world_process_zero):
            return control
        if self._run is None or not logs:
            return control
        payload = self._rewrite_logs(logs)
        payload["train/global_step"] = state.global_step
        self._run.log(payload)
        return control

    def on_predict(self, args, state, control, metrics=None, model=None, **kwargs):
        """Log metrics produced by :meth:`Trainer.predict`."""
        return self.on_log(
            args,
            state,
            control,
            logs=metrics,
            model=model,
        )

    @staticmethod
    def _artifact_name(run, args, suffix: str) -> str:
        run_name = (getattr(run, "name", None) or args.run_name or getattr(run, "id", None) or "voicehub")
        raw_name = str(run_name)
        normalized = _ARTIFACT_NAME_PATTERN.sub("-", raw_name).strip("-.") or "voicehub"
        suffix = f"-{suffix}"
        available = _MAX_ARTIFACT_NAME_LENGTH - len(suffix)
        if len(normalized) > available:
            digest = hashlib.sha256(raw_name.encode("utf-8")).hexdigest()[:12]
            prefix_length = available - len(digest) - 1
            prefix = normalized[:prefix_length].rstrip("-.")
            normalized = f"{prefix}-{digest}"
        return f"{normalized}{suffix}"

    def _log_artifact(
        self,
        args,
        state,
        path: Path,
        *,
        suffix: str,
        aliases: list[str],
    ) -> None:
        artifact = self._wandb.Artifact(
            self._artifact_name(self._run, args, suffix),
            type="model",
            metadata={"global_step": state.global_step},
        )
        artifact.add_dir(str(path))
        self._run.log_artifact(artifact, aliases=aliases)

    def on_checkpoint_saved(
        self,
        args,
        state,
        control,
        checkpoint_path=None,
        model=None,
        **kwargs,
    ):
        if (args.wandb_log_model != "checkpoint" or not self._initialized or
                not state.is_world_process_zero or self._finished):
            return control
        if checkpoint_path is None:
            raise RuntimeError("W&B checkpoint logging requires a completed checkpoint path.")
        checkpoint = Path(checkpoint_path)
        if not (checkpoint / CHECKPOINT_COMPLETE_NAME).is_file():
            raise RuntimeError(
                "W&B checkpoint logging requires a completed VoiceHub "
                f"checkpoint: {checkpoint}.")
        self._log_artifact(
            args,
            state,
            checkpoint,
            suffix="checkpoint",
            aliases=[f"step-{state.global_step}", "latest"],
        )
        return control

    def requires_final_model(self, args, state) -> bool:
        return (args.wandb_log_model == "end" and state.is_world_process_zero)

    def on_final_model_saved(
        self,
        args,
        state,
        control,
        final_model_path=None,
        model=None,
        **kwargs,
    ):
        if self._finished or not self._initialized or not state.is_world_process_zero:
            return control
        if args.wandb_log_model != "end" or final_model_path is None:
            return control
        model_path = Path(final_model_path)
        if not model_path.is_dir():
            raise RuntimeError(f"Final model directory was not found: {model_path}.")
        self._log_artifact(
            args,
            state,
            model_path,
            suffix="model",
            aliases=["final", "latest"],
        )
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self._finish_owned_run()
        self._clear_run_id_on_next_begin = True
        self._needs_cleanup_before_retry = False
        return control

    def on_train_error(self, args, state, control, **kwargs):
        if (self._finished and self._clear_run_id_on_next_begin):
            return control
        if not self._initialized and self._run is None:
            return control
        self._clear_run_id_on_next_begin = False
        self._needs_cleanup_before_retry = True
        self._finish_owned_run()
        return control

    def _finish_owned_run(self) -> None:
        if self._finished:
            return
        if self._owns_run and self._run is not None:
            finish = getattr(self._run, "finish", None)
            if callable(finish):
                finish()
            else:
                module_finish = getattr(self._wandb, "finish", None)
                if callable(module_finish):
                    module_finish()
            self._finished = True
        elif self._initialized:
            self._finished = True
