"""Transformers-style configuration for VoiceHub training loops."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from importlib import import_module
from numbers import Real
from pathlib import Path
from typing import Any

from voicehub.hub import read_json_file
from voicehub.serialization_utils import reject_serialized_secrets
from voicehub.training.utils import IntervalStrategy, SchedulerType, write_json


@dataclass
class TrainingArguments:
    """Arguments controlling a single-process PyTorch training run.

    The names intentionally follow ``transformers.TrainingArguments`` so
    TTS recipes can move to VoiceHub without inventing a second
    vocabulary.
    """

    output_dir: str = "trainer_output"
    overwrite_output_dir: bool = False
    do_train: bool = False
    do_eval: bool = False
    eval_strategy: IntervalStrategy | str = IntervalStrategy.NO
    prediction_loss_only: bool = False

    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    eval_accumulation_steps: int | None = None

    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    adamw_fused: bool = False
    adamw_torch_compile: bool = False
    max_grad_norm: float = 1.0
    num_train_epochs: float = 3.0
    max_steps: int = -1
    lr_scheduler_type: SchedulerType | str = SchedulerType.LINEAR
    warmup_ratio: float = 0.0
    warmup_steps: int = 0
    lr_scheduler_gamma: float = 1.0

    logging_strategy: IntervalStrategy | str = IntervalStrategy.STEPS
    logging_steps: int = 500
    logging_first_step: bool = False
    eval_steps: int | None = None
    save_strategy: IntervalStrategy | str = IntervalStrategy.STEPS
    save_steps: int = 500
    save_total_limit: int | None = None

    seed: int = 42
    data_seed: int | None = None
    dataloader_drop_last: bool = False
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = True
    remove_unused_columns: bool = True
    label_names: list[str] = field(default_factory=lambda: ["labels"])

    load_best_model_at_end: bool = False
    metric_for_best_model: str | None = None
    greater_is_better: bool | None = None
    gradient_checkpointing: bool = False
    fp16: bool = False
    bf16: bool = False
    use_cpu: bool = False
    disable_tqdm: bool = True
    report_to: list[str] | str = field(default_factory=list)
    run_name: str | None = None
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_group: str | None = None
    wandb_tags: list[str] = field(default_factory=list)
    wandb_notes: str | None = None
    wandb_mode: str | None = None
    wandb_log_model: str | bool = False

    # Compatibility alias retained for recipes written against older
    # Transformers releases.
    evaluation_strategy: IntervalStrategy | str | None = field(
        default=None,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.evaluation_strategy is not None:
            if self.eval_strategy != IntervalStrategy.NO:
                raise ValueError("Use either `eval_strategy` or `evaluation_strategy`, not both.")
            self.eval_strategy = self.evaluation_strategy

        self.eval_strategy = IntervalStrategy(self.eval_strategy)
        self.logging_strategy = IntervalStrategy(self.logging_strategy)
        self.save_strategy = IntervalStrategy(self.save_strategy)
        self.lr_scheduler_type = SchedulerType(self.lr_scheduler_type)
        self.output_dir = str(Path(self.output_dir).expanduser())
        self.report_to = self._normalize_report_to(self.report_to)
        self.wandb_log_model = self._normalize_wandb_log_model(self.wandb_log_model, )

        positive_values = {
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
        }
        for name, value in positive_values.items():
            if (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
                raise ValueError(f"`{name}` must be greater than zero.")
        if (self.eval_accumulation_steps is not None and
            (isinstance(self.eval_accumulation_steps, bool) or
             not isinstance(self.eval_accumulation_steps, int) or self.eval_accumulation_steps <= 0)):
            raise ValueError("`eval_accumulation_steps` must be a positive integer or "
                             "None.")
        if isinstance(self.max_steps, bool) or not isinstance(self.max_steps, int):
            raise TypeError("`max_steps` must be an integer.")
        if self.max_steps < -1 or self.max_steps == 0:
            raise ValueError("`max_steps` must be -1 or a positive integer.")
        if (isinstance(self.num_train_epochs, bool) or not isinstance(self.num_train_epochs, Real) or
                not math.isfinite(float(self.num_train_epochs))):
            raise TypeError("`num_train_epochs` must be a finite number.")
        if self.num_train_epochs <= 0 and self.max_steps <= 0:
            raise ValueError("Set a positive `num_train_epochs` or `max_steps`.")
        optimizer_values = {
            "learning_rate": (self.learning_rate, 0.0, False),
            "weight_decay": (self.weight_decay, 0.0, False),
            "adam_epsilon": (self.adam_epsilon, 0.0, True),
            "lr_scheduler_gamma": (self.lr_scheduler_gamma, 0.0, True),
            "max_grad_norm": (self.max_grad_norm, 0.0, False),
        }
        for name, (value, minimum, strict) in optimizer_values.items():
            valid = (
                not isinstance(value, bool) and isinstance(value, Real) and math.isfinite(float(value)) and
                (value > minimum if strict else value >= minimum))
            if not valid:
                comparison = "greater than" if strict else "at least"
                raise ValueError(f"`{name}` must be finite and {comparison} {minimum}.")
        for name, value in (
            ("adam_beta1", self.adam_beta1),
            ("adam_beta2", self.adam_beta2),
        ):
            if (isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)) or
                    not 0.0 <= value < 1.0):
                raise ValueError(f"`{name}` must be in the interval [0, 1).")
        for name, value in (
            ("logging_steps", self.logging_steps),
            ("save_steps", self.save_steps),
            ("warmup_steps", self.warmup_steps),
            ("dataloader_num_workers", self.dataloader_num_workers),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"`{name}` must be an integer.")
        for name, value in (
            ("eval_steps", self.eval_steps),
            ("save_total_limit", self.save_total_limit),
        ):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
                raise TypeError(f"`{name}` must be an integer or None.")
        for name, value in (
            ("seed", self.seed),
            ("data_seed", self.data_seed),
        ):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
                raise TypeError(f"`{name}` must be an integer or None.")
        if self.logging_strategy is IntervalStrategy.STEPS and self.logging_steps <= 0:
            raise ValueError("`logging_steps` must be positive for step logging.")
        if self.save_strategy is IntervalStrategy.STEPS and self.save_steps <= 0:
            raise ValueError("`save_steps` must be positive for step checkpointing.")
        if self.eval_strategy is IntervalStrategy.STEPS:
            self.eval_steps = self.eval_steps or self.logging_steps
            if self.eval_steps <= 0:
                raise ValueError("`eval_steps` must be positive for step evaluation.")
        if self.save_total_limit is not None and self.save_total_limit <= 0:
            raise ValueError("`save_total_limit` must be greater than zero.")
        if (isinstance(self.warmup_ratio, bool) or not isinstance(self.warmup_ratio, Real) or
                not math.isfinite(float(self.warmup_ratio)) or not 0.0 <= self.warmup_ratio <= 1.0):
            raise ValueError("`warmup_ratio` must be between zero and one.")
        if self.warmup_steps < 0:
            raise ValueError("`warmup_steps` cannot be negative.")
        if self.fp16 and self.bf16:
            raise ValueError("At most one of `fp16` and `bf16` can be enabled.")
        if not isinstance(self.adamw_fused, bool):
            raise TypeError("`adamw_fused` must be a boolean.")
        if not isinstance(self.adamw_torch_compile, bool):
            raise TypeError("`adamw_torch_compile` must be a boolean.")
        if (not isinstance(self.label_names, list) or not self.label_names or
                any(not isinstance(name, str) or not name.strip() for name in self.label_names) or
                len(set(self.label_names)) != len(self.label_names)):
            raise ValueError("`label_names` must be a non-empty list of unique names.")
        for name in (
                "run_name",
                "wandb_project",
                "wandb_entity",
                "wandb_group",
                "wandb_notes",
        ):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"`{name}` must be a non-empty string or None.")
        if (not isinstance(self.wandb_tags, list) or
                any(not isinstance(tag, str) or not tag.strip() for tag in self.wandb_tags)):
            raise ValueError("`wandb_tags` must be a list of non-empty strings.")
        self.wandb_tags = list(dict.fromkeys(tag.strip() for tag in self.wandb_tags))
        if self.wandb_mode is not None:
            if not isinstance(self.wandb_mode, str):
                raise TypeError("`wandb_mode` must be a string or None.")
            self.wandb_mode = self.wandb_mode.strip().lower()
            if self.wandb_mode not in {"online", "offline", "disabled"}:
                raise ValueError("`wandb_mode` must be 'online', 'offline', 'disabled', "
                                 "or None.")

        if self.load_best_model_at_end:
            if self.eval_strategy is IntervalStrategy.NO:
                raise ValueError("`load_best_model_at_end` requires an evaluation strategy.")
            if self.eval_strategy is not self.save_strategy:
                raise ValueError("Save and evaluation strategies must match when loading the best model.")
            if self.eval_strategy is IntervalStrategy.STEPS:
                if self.save_steps % int(self.eval_steps or 1) != 0:
                    raise ValueError(
                        "`save_steps` must be a multiple of `eval_steps` when "
                        "loading the best model.")

        if self.metric_for_best_model is None and self.load_best_model_at_end:
            self.metric_for_best_model = "loss"
        if self.greater_is_better is None and self.metric_for_best_model is not None:
            self.greater_is_better = not self.metric_for_best_model.endswith("loss")

        TrainingArguments._validated_serialization_payload(self)

    @staticmethod
    def _normalize_report_to(value: list[str] | str) -> list[str]:
        if isinstance(value, str):
            integrations = [value]
        elif isinstance(value, list):
            integrations = value
        else:
            raise TypeError("`report_to` must be a string or list of strings.")
        if any(not isinstance(name, str) or not name.strip() for name in integrations):
            raise ValueError("`report_to` entries must be non-empty strings.")

        normalized = [name.strip().lower() for name in integrations]
        if "none" in normalized:
            if len(normalized) != 1:
                raise ValueError("`report_to='none'` cannot be combined with integrations.")
            return []
        if "all" in normalized:
            if len(normalized) != 1:
                raise ValueError("`report_to='all'` cannot be combined with integrations.")
            normalized = ["wandb"]
        unsupported = sorted(set(normalized) - {"wandb"})
        if unsupported:
            names = ", ".join(unsupported)
            raise ValueError(
                f"Unsupported reporting integration(s): {names}. "
                "Supported integrations: wandb.")
        return list(dict.fromkeys(normalized))

    @staticmethod
    def _normalize_wandb_log_model(value: str | bool) -> str:
        if isinstance(value, bool):
            return "end" if value else "false"
        if not isinstance(value, str):
            raise TypeError("`wandb_log_model` must be a boolean or string.")
        normalized = value.strip().lower()
        aliases = {
            "true": "end",
            "false": "false",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in {"false", "checkpoint", "end"}:
            raise ValueError("`wandb_log_model` must be false, 'checkpoint', or 'end'.")
        return normalized

    @property
    def train_batch_size(self) -> int:
        """Effective per-process training batch size."""
        return self.per_device_train_batch_size

    @property
    def eval_batch_size(self) -> int:
        """Effective per-process evaluation batch size."""
        return self.per_device_eval_batch_size

    @property
    def device(self) -> str:
        """Select CPU, CUDA, or MPS without importing PyTorch at package
        import."""
        if self.use_cpu:
            return "cpu"
        try:
            torch = import_module("torch")
        except ModuleNotFoundError:
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"

    def get_warmup_steps(self, num_training_steps: int) -> int:
        """Return explicit warmup steps or derive them from
        ``warmup_ratio``."""
        if self.warmup_steps > 0:
            return self.warmup_steps
        return int(num_training_steps * self.warmup_ratio)

    def _serialization_payload(self) -> dict[str, Any]:
        values = asdict(self)
        values.pop("evaluation_strategy", None)
        for key in (
                "eval_strategy",
                "logging_strategy",
                "save_strategy",
                "lr_scheduler_type",
        ):
            value = values[key]
            values[key] = value.value if isinstance(value, Enum) else value
        return values

    def _validated_serialization_payload(self) -> dict[str, Any]:
        values = TrainingArguments._serialization_payload(self)
        reject_serialized_secrets(values, owner=self.__class__.__name__)
        return values

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return TrainingArguments._validated_serialization_payload(self)

    def to_json_string(self) -> str:
        """Serialize arguments as stable, readable JSON."""
        return json.dumps(
            TrainingArguments._validated_serialization_payload(self),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n"

    def save_json(self, json_file: str | Path) -> Path:
        """Save arguments for checkpoint reproducibility."""
        payload = TrainingArguments._validated_serialization_payload(self)
        return write_json(json_file, payload)

    @classmethod
    def from_json_file(cls, json_file: str | Path) -> TrainingArguments:
        """Restore arguments saved by :meth:`save_json`."""
        payload = read_json_file(Path(json_file).expanduser())
        reject_serialized_secrets(payload, owner=cls.__name__)
        return cls(**payload)
