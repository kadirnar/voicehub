"""Small, dependency-lazy helpers shared by source-integrated backends."""

from __future__ import annotations

import os
import random
import secrets
import sys
from contextlib import contextmanager
from importlib import import_module
from numbers import Integral
from pathlib import Path
from threading import RLock
from typing import Any

from voicehub.dependencies import import_optional
from voicehub.hub_transport import download_hugging_face_snapshot
from voicehub.outputs import TTSOutput
from voicehub.path_utils import is_explicit_local_path

_TORCH_DTYPE_ALIASES = {
    "bf16": "bfloat16",
    "double": "float64",
    "fp16": "float16",
    "fp32": "float32",
    "fp64": "float64",
    "half": "float16",
}
_RANDOM_STATE_LOCK = RLock()
_TORCH_SEED_MIN = -(2**63)
_TORCH_SEED_MAX = 2**64 - 1


def _mutable_torch_backend_flags(torch: Any) -> list[tuple[Any, str]]:
    """Return backend flags commonly mutated by upstream seed helpers."""
    backends = getattr(torch, "backends", None)
    cuda = getattr(backends, "cuda", None)
    cudnn = getattr(backends, "cudnn", None)
    candidates = (
        (getattr(cuda, "matmul", None), "allow_tf32"),
        (cudnn, "allow_tf32"),
        (cudnn, "benchmark"),
        (cudnn, "deterministic"),
        (cudnn, "enabled"),
    )
    return [(owner, name) for owner, name in candidates if owner is not None and hasattr(owner, name)]


def validate_seed(
    seed: int | None,
    *,
    option_name: str = "seed",
) -> int | None:
    """Validate a seed against the range supported by Torch generators."""
    if seed is None:
        return None
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise TypeError(f"`{option_name}` must be an integer or None.")
    seed = int(seed)
    if not _TORCH_SEED_MIN <= seed <= _TORCH_SEED_MAX:
        raise ValueError(
            f"`{option_name}` must be in Torch's supported range "
            f"[{_TORCH_SEED_MIN}, {_TORCH_SEED_MAX}].")
    return seed


@contextmanager
def preserve_inference_state(
    *,
    device: str,
    model_type: str,
):
    """Preserve process-global RNG and mutable Torch backend state."""
    numpy = sys.modules.get("numpy")
    numpy_random = getattr(numpy, "random", None)
    try:
        torch = import_module("torch")
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        torch = None

    if torch is None:
        with _RANDOM_STATE_LOCK:
            python_state = random.getstate()
            numpy_state = (numpy_random.get_state() if numpy_random is not None else None)
            hash_seed = os.environ.get("PYTHONHASHSEED")
            try:
                yield None, None, (), None
            finally:
                if hash_seed is None:
                    os.environ.pop("PYTHONHASHSEED", None)
                else:
                    os.environ["PYTHONHASHSEED"] = hash_seed
                random.setstate(python_state)
                if numpy_random is not None and numpy_state is not None:
                    numpy_random.set_state(numpy_state)
        return

    resolved_device = torch.device(device)
    cuda_devices = []
    if resolved_device.type == "cuda":
        cuda_devices.append(
            resolved_device.index if resolved_device.index is not None else torch.cuda.current_device())

    with _RANDOM_STATE_LOCK:
        python_state = random.getstate()
        numpy_state = (numpy_random.get_state() if numpy_random is not None else None)
        hash_seed = os.environ.get("PYTHONHASHSEED")
        backend_flags = [(owner, name, getattr(owner, name))
                         for owner, name in _mutable_torch_backend_flags(torch)]
        mps_state = None
        if (resolved_device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "get_rng_state")):
            mps_state = torch.mps.get_rng_state()
        xpu_device = None
        xpu_state = None
        if (resolved_device.type == "xpu" and hasattr(torch, "xpu") and hasattr(torch.xpu, "get_rng_state")):
            xpu_device = (
                resolved_device.index if resolved_device.index is not None else torch.xpu.current_device())
            xpu_state = torch.xpu.get_rng_state(xpu_device)
        try:
            with torch.random.fork_rng(devices=cuda_devices):
                yield torch, resolved_device, cuda_devices, xpu_device
        finally:
            if mps_state is not None:
                torch.mps.set_rng_state(mps_state)
            if xpu_state is not None:
                torch.xpu.set_rng_state(xpu_state, xpu_device)
            for owner, name, value in backend_flags:
                setattr(owner, name, value)
            if hash_seed is None:
                os.environ.pop("PYTHONHASHSEED", None)
            else:
                os.environ["PYTHONHASHSEED"] = hash_seed
            random.setstate(python_state)
            if numpy_random is not None and numpy_state is not None:
                numpy_random.set_state(numpy_state)


@contextmanager
def seeded_inference(
    seed: int | None,
    *,
    device: str,
    model_type: str,
):
    """Temporarily seed stochastic backends without leaking RNG state.

    Seeded calls are serialized because Python, NumPy, and Torch expose
    process-global RNGs. An omitted seed receives fresh system entropy
    so unseeded requests remain random while still leaving caller RNGs
    untouched.
    """
    seed = validate_seed(seed)
    if seed is None:
        seed = secrets.randbits(63)
    import_optional(
        "torch",
        model_type=model_type,
    )

    with preserve_inference_state(
            device=device,
            model_type=model_type,
    ) as (torch, resolved_device, cuda_devices, xpu_device):
        numpy = sys.modules.get("numpy")
        numpy_random = getattr(numpy, "random", None)
        random.seed(seed)
        if numpy_random is not None:
            numpy_random.seed(seed % (2**32))
        torch.random.default_generator.manual_seed(seed)
        if cuda_devices:
            with torch.cuda.device(cuda_devices[0]):
                torch.cuda.manual_seed(seed)
        elif (resolved_device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed")):
            torch.mps.manual_seed(seed)
        elif (xpu_device is not None and hasattr(torch.xpu, "manual_seed")):
            with torch.xpu.device(xpu_device):
                torch.xpu.manual_seed(seed)
        yield seed


def finish_audio_output(
    audio: Any,
    sample_rate: int,
    *,
    output_file: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> TTSOutput:
    """Build and optionally persist the normalized VoiceHub output."""
    output = TTSOutput(
        audio=audio,
        sample_rate=sample_rate,
        metadata={} if metadata is None else metadata,
    )
    if output_file is not None:
        output.save(output_file)
    return output


def validate_local_file(
    value: Any,
    *,
    option_name: str,
    required: bool = False,
) -> Path | None:
    """Validate a local file option without importing a model backend."""
    if value is None:
        if required:
            raise ValueError(f"`{option_name}` must point to a local file.")
        return None
    if (not isinstance(value, (str, os.PathLike)) or not str(value).strip()):
        qualifier = "" if required else " or None"
        raise ValueError(f"`{option_name}` must point to a local file{qualifier}.")
    path = Path(value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"`{option_name}` was not found: {path}.")
    return path.resolve()


def resolve_model_directory(
    name_or_path: str | Path,
    *,
    model_type: str,
    install_extra: str | None = None,
    **download_kwargs: Any,
) -> Path:
    """Resolve a local directory or download one complete Hub snapshot."""
    del install_extra
    if not isinstance(name_or_path, (str, Path)):
        raise TypeError("`name_or_path` must be a string or pathlib.Path.")
    if isinstance(name_or_path, str) and not name_or_path.strip():
        raise ValueError("`name_or_path` must be a non-empty path or Hub ID.")
    source = Path(name_or_path).expanduser()
    if source.is_dir():
        return source.resolve()
    if source.exists():
        raise NotADirectoryError(f"Expected a model directory, received file: {source}.")
    if is_explicit_local_path(name_or_path):
        raise FileNotFoundError(f"Model directory was not found: {source}.")
    snapshot = download_hugging_face_snapshot(
        str(name_or_path),
        **{
            key: value
            for key, value in download_kwargs.items() if value is not None
        },
    )
    if not snapshot.is_dir():
        raise RuntimeError(f"Downloaded {model_type!r} snapshot is not a directory: {snapshot}.")
    return snapshot


def resolve_torch_dtype(torch: Any, dtype_name: str, device: str) -> Any:
    """Map a configured dtype name while keeping CPU defaults safe."""
    if not isinstance(dtype_name, str) or not dtype_name.strip():
        raise ValueError("`dtype_name` must be a non-empty torch dtype name.")
    normalized = dtype_name.strip().lower()
    if normalized.startswith("torch."):
        normalized = normalized.removeprefix("torch.")
    normalized = _TORCH_DTYPE_ALIASES.get(normalized, normalized)
    try:
        dtype = getattr(torch, normalized)
    except AttributeError as exc:
        raise ValueError(f"Unknown torch dtype: {dtype_name!r}.") from exc
    dtype_type = getattr(torch, "dtype", None)
    if isinstance(dtype_type, type) and not isinstance(dtype, dtype_type):
        raise ValueError(f"Unknown torch dtype: {dtype_name!r}.")
    is_floating_point = getattr(dtype, "is_floating_point", None)
    if callable(is_floating_point):
        is_floating_point = is_floating_point()
    if is_floating_point is None:
        is_floating_point = (
            normalized in {"bfloat16", "float16", "float32", "float64"} or normalized.startswith("float8_"))
    if not is_floating_point:
        raise ValueError(
            "`dtype_name` must identify a floating-point torch dtype; "
            f"received {dtype_name!r}.")
    device_type = getattr(device, "type", None)
    if device_type is None:
        device_type = str(device).split(":", 1)[0].lower()
    if device_type == "cpu" and dtype in {torch.float16, torch.bfloat16}:
        return torch.float32
    return dtype
