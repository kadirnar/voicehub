"""Framework-light utility types shared by the VoiceHub trainer."""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Iterator, Sized
from enum import Enum
from importlib import import_module
from pathlib import Path
from typing import Any, NamedTuple

from voicehub.hub import read_json_file, write_json_file

PREFIX_CHECKPOINT_DIR = "checkpoint"
TRAINER_STATE_NAME = "trainer_state.json"
TRAINING_ARGS_NAME = "training_args.json"
MODEL_STATE_NAME = "model_state.pt"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
RNG_STATE_NAME = "rng_state.pth"
SCALER_STATE_NAME = "scaler.pt"
TRAINING_RUNTIME_STATE_NAME = "training_runtime.pt"
TRAINING_RECIPE_NAME = "training_recipe.json"
OPTIMIZATION_MANIFEST_NAME = "optimization_manifest.json"
NATIVE_EXPORT_DIR = "native_export"
CHECKPOINT_MANIFEST_NAME = "checkpoint_manifest.json"
CHECKPOINT_COMPLETE_NAME = ".complete"
CHECKPOINT_FORMAT_VERSION = 3
FORMAT_V2_REQUIRED_FILES = (
    MODEL_STATE_NAME,
    OPTIMIZER_NAME,
    SCHEDULER_NAME,
    TRAINER_STATE_NAME,
    TRAINING_ARGS_NAME,
    RNG_STATE_NAME,
    TRAINING_RUNTIME_STATE_NAME,
)
LEGACY_RESUME_FILES = (
    MODEL_STATE_NAME,
    OPTIMIZER_NAME,
    SCHEDULER_NAME,
    TRAINER_STATE_NAME,
    RNG_STATE_NAME,
)


class ExplicitEnum(str, Enum):
    """String enum that reports accepted values in invalid-value errors."""

    @classmethod
    def _missing_(cls, value):
        choices = ", ".join(repr(member.value) for member in cls)
        raise ValueError(f"{value!r} is not valid for {cls.__name__}; choose {choices}.")

    def __str__(self) -> str:
        return self.value


class IntervalStrategy(ExplicitEnum):
    """When a recurring Trainer action should run."""

    NO = "no"
    STEPS = "steps"
    EPOCH = "epoch"


class SchedulerType(ExplicitEnum):
    """Learning-rate schedules implemented without external trainer
    packages."""

    LINEAR = "linear"
    COSINE = "cosine"
    CONSTANT = "constant"
    EXPONENTIAL = "exponential"


class TrainOutput(NamedTuple):
    """Return value of :meth:`voicehub.Trainer.train`."""

    global_step: int
    training_loss: float
    metrics: dict[str, float]


class PredictionOutput(NamedTuple):
    """Predictions, references, and metrics returned by ``Trainer.predict``."""

    predictions: Any
    label_ids: Any
    metrics: dict[str, float]


class EvalPrediction:
    """Container passed to a user-provided ``compute_metrics`` function."""

    def __init__(
        self,
        predictions: Any,
        label_ids: Any,
        inputs: Any | None = None,
    ):
        self.predictions = predictions
        self.label_ids = label_ids
        self.inputs = inputs

    def __iter__(self):
        if self.inputs is None:
            return iter((self.predictions, self.label_ids))
        return iter((self.predictions, self.label_ids, self.inputs))

    def __getitem__(self, index: int):
        return tuple(self)[index]


class EpochRandomSampler:
    """Deterministic epoch-addressable sampler without an eager Torch import.

    A seed plus epoch reproduces the exact permutation after resume,
    unlike a mutable generator whose state has already advanced when a
    DataLoader creates its iterator.
    """

    def __init__(
        self,
        data_source: Sized,
        *,
        seed: int,
        shuffle: bool = True,
    ):
        self.data_source = data_source
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        length = len(self.data_source)
        if not self.shuffle:
            return iter(range(length))
        torch = import_module("torch")
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(length, generator=generator).tolist())

    def __len__(self) -> int:
        return len(self.data_source)

    def state_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "shuffle": self.shuffle,
            "epoch": self.epoch,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        checkpoint_seed = int(state_dict["seed"])
        if checkpoint_seed != self.seed:
            raise ValueError(
                "Sampler seed in the checkpoint does not match the current "
                f"training run ({checkpoint_seed} != {self.seed}).")
        if bool(state_dict["shuffle"]) != self.shuffle:
            raise ValueError("Sampler shuffle mode differs from the checkpoint.")
        self.epoch = int(state_dict["epoch"])


def set_seed(seed: int) -> None:
    """Seed Python and the native PyTorch compute backend."""
    random.seed(seed)
    try:
        torch = import_module("torch")
    except ModuleNotFoundError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_last_checkpoint(folder: str | Path) -> str | None:
    """Return the checkpoint with the greatest numeric global step."""
    output_dir = Path(folder).expanduser()
    if not output_dir.is_dir():
        return None

    checkpoints: list[tuple[int, Path]] = []
    for path in output_dir.glob(f"{PREFIX_CHECKPOINT_DIR}-*"):
        if not path.is_dir():
            continue
        try:
            step = int(path.name.rsplit("-", 1)[1])
        except (IndexError, ValueError):
            continue
        manifest = path / CHECKPOINT_MANIFEST_NAME
        complete = path / CHECKPOINT_COMPLETE_NAME
        if manifest.is_file():
            if not complete.is_file():
                continue
            try:
                payload = read_json_file(manifest)
            except (OSError, TypeError, ValueError):
                continue
            format_version = payload.get("format_version")
            manifest_step = payload.get("global_step")
            if (isinstance(format_version, bool) or not isinstance(format_version, int) or
                    format_version <= 0 or format_version > CHECKPOINT_FORMAT_VERSION or
                    isinstance(manifest_step, bool) or not isinstance(manifest_step, int) or
                    manifest_step != step):
                continue
            required = payload.get("required_files")
            if (not isinstance(required, list) or
                    any(not isinstance(name, str) or Path(name).name != name or name in ("", ".", "..")
                        for name in required) or any(not (path / name).is_file() for name in required)):
                continue
            if (format_version >= 2 and not set(FORMAT_V2_REQUIRED_FILES).issubset(required)):
                continue
            integrity = payload.get("file_integrity")
            if format_version >= 2 and not isinstance(integrity, dict):
                continue
            if format_version >= 3 and not isinstance(
                    payload.get("resume_signature"),
                    dict,
            ):
                continue
            if integrity is not None:
                if not isinstance(integrity, dict):
                    continue
                invalid_size = False
                for name in required:
                    record = integrity.get(name)
                    if (not isinstance(record, dict) or record.get("size") != (path / name).stat().st_size):
                        invalid_size = True
                        break
                    expected = record.get("sha256")
                    if not isinstance(expected, str):
                        invalid_size = True
                        break
                    digest = hashlib.sha256()
                    with (path / name).open("rb") as handle:
                        for chunk in iter(
                                lambda: handle.read(1024 * 1024),
                                b"",
                        ):
                            digest.update(chunk)
                    if digest.hexdigest() != expected:
                        invalid_size = True
                        break
                if invalid_size:
                    continue
        elif any(not (path / name).is_file() for name in LEGACY_RESUME_FILES):
            continue
        checkpoints.append((step, path))
    if not checkpoints:
        return None
    return str(max(checkpoints, key=lambda item: item[0])[1])


def get_scheduler_lambda(
    scheduler_type: SchedulerType | str,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    exponential_gamma: float = 1.0,
    num_train_epochs: float = 1.0,
):
    """Build the scalar schedule used by ``torch.optim.lr_scheduler``."""
    schedule = SchedulerType(scheduler_type)
    total_steps = max(1, num_training_steps)
    warmup_steps = max(0, num_warmup_steps)

    def warmup(current_step: int) -> float:
        if warmup_steps and current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0

    if schedule is SchedulerType.CONSTANT:
        return warmup

    if schedule is SchedulerType.EXPONENTIAL:
        if (not isinstance(exponential_gamma, (int, float)) or isinstance(exponential_gamma, bool) or
                not math.isfinite(float(exponential_gamma)) or float(exponential_gamma) <= 0):
            raise ValueError("`exponential_gamma` must be finite and positive.")
        if (not isinstance(num_train_epochs, (int, float)) or isinstance(num_train_epochs, bool) or
                not math.isfinite(float(num_train_epochs)) or float(num_train_epochs) <= 0):
            raise ValueError("`num_train_epochs` must be finite and positive.")
        decay_steps = max(1, total_steps - warmup_steps)
        steps_per_epoch = decay_steps / float(num_train_epochs)

        def exponential(current_step: int) -> float:
            if current_step < warmup_steps:
                return warmup(current_step)
            elapsed = max(0, current_step - warmup_steps)
            return float(exponential_gamma)**(elapsed / steps_per_epoch)

        return exponential

    def progress(current_step: int) -> float:
        if current_step < warmup_steps:
            return warmup(current_step)
        denominator = max(1, total_steps - warmup_steps)
        return min(1.0, max(0.0, (current_step - warmup_steps) / denominator))

    if schedule is SchedulerType.COSINE:

        def cosine(current_step: int) -> float:
            if current_step < warmup_steps:
                return warmup(current_step)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress(current_step))))

        return cosine

    def linear(current_step: int) -> float:
        if current_step < warmup_steps:
            return warmup(current_step)
        return max(0.0, 1.0 - progress(current_step))

    return linear


def denumpify_detensorize(metrics: dict[str, Any]) -> dict[str, Any]:
    """Convert scalar tensor-like metric values into JSON-compatible values."""
    normalized = {}
    for key, value in metrics.items():
        if hasattr(value, "item") and callable(value.item):
            try:
                value = value.item()
            except (TypeError, ValueError):
                pass
        normalized[key] = value
    return normalized


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write a deterministic UTF-8 JSON document."""
    output_path = Path(path).expanduser()
    write_json_file(output_path, payload)
    return output_path
