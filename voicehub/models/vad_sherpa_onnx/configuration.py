"""Configuration for the native Sherpa-compatible Silero/TEN provider."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from numbers import Real
from pathlib import Path, PurePosixPath
from typing import Any

from voicehub.configuration import VoiceHubConfig, reject_serialized_secrets
from voicehub.inference_configuration import VADInferenceConfig

_MODEL_FAMILIES = frozenset({"silero", "ten"})
_PROVIDERS = frozenset({"cpu", "cuda"})


def _positive_number(value: Any, *, name: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, Real) or not isfinite(float(value)) or
            float(value) <= 0):
        raise ValueError(f"`{name}` must be finite and greater than zero.")
    return float(value)


def _relative_asset_name(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("`model_filename` must be a non-empty string.")
    normalized = value.strip().replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("`model_filename` must be a safe relative path.")
    if path.suffix.lower() not in {".onnx", ".safetensors", ".jit"}:
        raise ValueError(
            "`model_filename` must identify ONNX source weights, "
            "Safetensors, or the reviewed Silero JIT weight container.")
    return str(path)


class SherpaONNXVADConfig(VoiceHubConfig):
    """Configure native execution with Sherpa-compatible segmentation.

    The historical model type and class names are retained for
    serialized configuration compatibility. Neither family imports or
    executes ``sherpa_onnx`` or ``onnxruntime``.
    """

    model_type = "vad_sherpa_onnx"
    architecture_family = "frame-classification"

    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        model_family: str = "silero",
        model_filename: str | None = None,
        subfolder: str = "",
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        num_threads: int = 1,
        provider: str = "cpu",
        debug: bool = False,
        buffer_size_s: float = 60.0,
        window_size_samples: int | None = None,
        training_max_duration_s: float = 8.0,
        training_positive_weight: float = 1.0,
        training_train_encoder: bool = False,
        training_noise_loss: float = 0.5,
        training_label_threshold: float = 0.5,
        inference_config: VADInferenceConfig | Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        reject_serialized_secrets(
            {
                "inference_config": inference_config,
                "kwargs": kwargs,
            },
            owner="SherpaONNXVADConfig",
        )
        if isinstance(inference_config, VADInferenceConfig):
            inference_values = inference_config.to_dict()
        elif inference_config is None:
            inference_values = {}
        elif isinstance(inference_config, Mapping):
            inference_values = dict(inference_config)
        else:
            raise TypeError("`inference_config` must be a VADInferenceConfig, mapping, or None.")
        for name in VADInferenceConfig._COMMON_FIELDS:
            if name in kwargs:
                inference_values[name] = kwargs.pop(name)

        if sample_rate != 16_000:
            raise ValueError("Native Sherpa compatibility requires 16 kHz audio.")
        if (not isinstance(model_family, str) or model_family.strip().lower() not in _MODEL_FAMILIES):
            raise ValueError("`model_family` must be either 'silero' or 'ten'.")
        model_family = model_family.strip().lower()
        if model_filename is None:
            model_filename = ("silero_vad.onnx" if model_family == "silero" else "ten-vad.onnx")
        model_filename = _relative_asset_name(model_filename)
        if not isinstance(subfolder, str):
            raise TypeError("`subfolder` must be a string.")
        subfolder = subfolder.strip().replace("\\", "/")
        if subfolder:
            path = PurePosixPath(subfolder)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("`subfolder` must be a safe relative path.")
            subfolder = str(path)
        if revision is not None and (not isinstance(revision, str) or not revision.strip()):
            raise ValueError("`revision` must be a non-empty string or None.")
        if cache_dir is not None and not isinstance(cache_dir, (str, Path)):
            raise TypeError("`cache_dir` must be a string, Path, or None.")
        if not isinstance(local_files_only, bool):
            raise TypeError("`local_files_only` must be a boolean.")
        if (isinstance(num_threads, bool) or not isinstance(num_threads, int) or num_threads <= 0):
            raise ValueError("`num_threads` must be a positive integer.")
        if (not isinstance(provider, str) or provider.strip().lower() not in _PROVIDERS):
            raise ValueError("Native `provider` must be 'cpu' or 'cuda'.")
        if not isinstance(debug, bool):
            raise TypeError("`debug` must be a boolean.")
        if not isinstance(training_train_encoder, bool):
            raise TypeError("`training_train_encoder` must be a boolean.")
        buffer_size_s = _positive_number(buffer_size_s, name="buffer_size_s")
        training_max_duration_s = _positive_number(
            training_max_duration_s,
            name="training_max_duration_s",
        )
        training_positive_weight = _positive_number(
            training_positive_weight,
            name="training_positive_weight",
        )
        for name, value in (
            ("training_noise_loss", training_noise_loss),
            ("training_label_threshold", training_label_threshold),
        ):
            if (isinstance(value, bool) or not isinstance(value, Real) or not isfinite(float(value)) or
                    not 0 <= float(value) <= 1):
                raise ValueError(f"`{name}` must be finite and in [0, 1].")
        if window_size_samples is None:
            window_size_samples = 512 if model_family == "silero" else 256
        if (isinstance(window_size_samples, bool) or not isinstance(window_size_samples, int) or
                window_size_samples <= 0):
            raise ValueError("`window_size_samples` must be a positive integer.")
        if model_family == "silero" and window_size_samples != 512:
            raise ValueError("The verified native 16 kHz Silero graph requires a "
                             "512-sample shift.")
        if model_family == "ten" and window_size_samples > 768:
            raise ValueError("TEN VAD windows cannot exceed 768 samples.")

        super().__init__(
            sample_rate=sample_rate,
            model_family=model_family,
            model_filename=model_filename,
            subfolder=subfolder,
            revision=None if revision is None else revision.strip(),
            cache_dir=(None if cache_dir is None else str(Path(cache_dir).expanduser())),
            local_files_only=local_files_only,
            num_threads=num_threads,
            provider=provider.strip().lower(),
            debug=debug,
            buffer_size_s=buffer_size_s,
            window_size_samples=window_size_samples,
            training_max_duration_s=training_max_duration_s,
            training_positive_weight=training_positive_weight,
            training_train_encoder=training_train_encoder,
            training_noise_loss=float(training_noise_loss),
            training_label_threshold=float(training_label_threshold),
            inference_config=VADInferenceConfig.from_dict(inference_values).to_dict(),
            **kwargs,
        )


__all__ = ["SherpaONNXVADConfig"]
