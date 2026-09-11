"""Strict native checkpoint lifecycle for the Fluac-based Vui family."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from voicehub.checkpointing import SafeTensorReader, save_safetensors
from voicehub.configuration import reject_serialized_secrets
from voicehub.hub import read_json_file, write_json_file

VUI_NATIVE_FORMAT = "voicehub-vui"
VUI_NATIVE_FORMAT_VERSION = 1
VUI_NATIVE_MODEL_FILENAME = "model.safetensors"
VUI_NATIVE_CODEC_FILENAME = "codec.safetensors"
VUI_NATIVE_CONFIG_FILENAME = "config.json"


@dataclass(frozen=True, slots=True)
class NativeVuiArtifact:
    """Resolved files and graph configurations for one native Vui export."""

    root: Path
    model_checkpoint: Path
    codec_checkpoint: Path
    model_config: dict[str, Any]
    codec_config: dict[str, Any]


def _serialize_config(value: Any, *, label: str) -> dict[str, Any]:
    serializer = getattr(value, "to_dict", None)
    if not callable(serializer):
        serializer = getattr(value, "model_dump", None)
    if not callable(serializer):
        serializer = getattr(value, "dict", None)
    if callable(serializer):
        serialized = serializer()
    elif isinstance(value, Mapping):
        serialized = dict(value)
    else:
        raise TypeError(f"Vui {label} configuration is not serializable.")
    if not isinstance(serialized, Mapping):
        raise TypeError(f"Vui {label} configuration must serialize to a mapping.")
    return dict(serialized)


def _clean_state_name(name: str) -> str:
    """Remove process-local compilation and distributed prefixes."""
    return name.replace("_orig_mod.", "").replace("module.", "")


def _component_state_dict(model: Any, *, codec: bool) -> dict[str, Any]:
    state_dict = getattr(model, "state_dict", None)
    if not callable(state_dict):
        raise TypeError("A native Vui export requires a PyTorch model.")
    selected: dict[str, Any] = {}
    for original_name, tensor in state_dict().items():
        name = _clean_state_name(original_name)
        is_codec = name.startswith("codec.")
        if is_codec != codec:
            continue
        if codec:
            name = name.removeprefix("codec.")
        selected[name] = tensor.detach().cpu().contiguous()
    component = "codec" if codec else "model"
    if not selected:
        raise ValueError(f"Native Vui {component} state cannot be empty.")
    return selected


def _safe_filename(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Native Vui `{field}` must be a non-empty filename.")
    if Path(value).name != value or not value.endswith(".safetensors"):
        raise ValueError(f"Native Vui `{field}` must be a local .safetensors filename.")
    return value


def export_vui_pretrained(
    model: Any,
    save_directory: str | Path,
    *,
    wrapper_config: Any | None = None,
) -> NativeVuiArtifact:
    """Write a complete, fresh-inference-reloadable native Vui artifact."""
    destination = Path(save_directory).expanduser()
    model_config = _serialize_config(
        getattr(model, "config", None),
        label="model",
    )
    codec = getattr(model, "codec", None)
    if codec is None:
        raise TypeError("A native Vui export requires its frozen Fluac codec.")
    codec_config = _serialize_config(
        getattr(codec, "config", None),
        label="codec",
    )
    if wrapper_config is None:
        config_values: dict[str, Any] = {}
    else:
        config_values = _serialize_config(
            wrapper_config,
            label="wrapper",
        )
    config_values.update({
        "checkpoint_filename": VUI_NATIVE_MODEL_FILENAME,
        "codec_filename": VUI_NATIVE_CODEC_FILENAME,
        "model_type": "vui",
        "native_artifact_format": VUI_NATIVE_FORMAT,
        "native_artifact_format_version": VUI_NATIVE_FORMAT_VERSION,
        "native_codec_config": codec_config,
        "native_model_config": model_config,
        "sample_rate": int(codec_config["sample_rate"]),
        "verify_official_integrity": False,
    })
    reject_serialized_secrets(
        config_values,
        owner="Vui native artifact configuration",
    )

    destination.mkdir(parents=True, exist_ok=True)
    model_checkpoint = destination / VUI_NATIVE_MODEL_FILENAME
    codec_checkpoint = destination / VUI_NATIVE_CODEC_FILENAME
    common_metadata = {
        "format": VUI_NATIVE_FORMAT,
        "format_version": str(VUI_NATIVE_FORMAT_VERSION),
    }
    save_safetensors(
        _component_state_dict(model, codec=False),
        model_checkpoint,
        metadata={
            **common_metadata,
            "component": "model",
        },
    )
    save_safetensors(
        _component_state_dict(model, codec=True),
        codec_checkpoint,
        metadata={
            **common_metadata,
            "component": "codec",
        },
    )
    write_json_file(
        destination / VUI_NATIVE_CONFIG_FILENAME,
        config_values,
    )
    return resolve_native_vui_artifact(destination)


def resolve_native_vui_artifact(source: str | Path, ) -> NativeVuiArtifact:
    """Validate and resolve one VoiceHub-native Vui Safetensors directory."""
    path = Path(source).expanduser()
    root = path if path.is_dir() else path.parent
    config_path = root / VUI_NATIVE_CONFIG_FILENAME
    if not config_path.is_file():
        raise FileNotFoundError(
            "Native Vui Safetensors require a sibling config.json containing "
            "the Vui and Fluac graph configurations.")
    values = read_json_file(config_path)
    if values.get("model_type") != "vui":
        raise ValueError("Native Vui config.json must declare `model_type=\"vui\"`.")
    if values.get("native_artifact_format") != VUI_NATIVE_FORMAT:
        raise ValueError("Native Vui config.json has an unsupported or missing artifact format.")
    if (values.get("native_artifact_format_version") != VUI_NATIVE_FORMAT_VERSION):
        raise ValueError("Native Vui config.json has an unsupported artifact format version.")

    model_filename = _safe_filename(
        values.get("checkpoint_filename"),
        field="checkpoint_filename",
    )
    codec_filename = _safe_filename(
        values.get("codec_filename"),
        field="codec_filename",
    )
    model_config = values.get("native_model_config")
    codec_config = values.get("native_codec_config")
    if not isinstance(model_config, Mapping):
        raise ValueError("Native Vui config.json is missing `native_model_config`.")
    if not isinstance(codec_config, Mapping):
        raise ValueError("Native Vui config.json is missing `native_codec_config`.")

    model_checkpoint = (root / model_filename).resolve()
    codec_checkpoint = (root / codec_filename).resolve()
    if not model_checkpoint.is_file():
        raise FileNotFoundError(f"Native Vui model checkpoint was not found: {model_checkpoint}.")
    if not codec_checkpoint.is_file():
        raise FileNotFoundError(f"Native Vui codec checkpoint was not found: {codec_checkpoint}.")
    if path.is_file() and path.resolve() not in {
            model_checkpoint,
            codec_checkpoint,
    }:
        raise ValueError(f"{path.name!r} is not declared by the native Vui config.json.")
    return NativeVuiArtifact(
        root=root.resolve(),
        model_checkpoint=model_checkpoint,
        codec_checkpoint=codec_checkpoint,
        model_config=dict(model_config),
        codec_config=dict(codec_config),
    )


def load_vui_safetensors(
    checkpoint_path: str | Path,
    *,
    component: str,
) -> dict[str, Any]:
    """Load one validated model or codec component from a native export."""
    if component not in {"model", "codec"}:
        raise ValueError("Vui Safetensors component must be `model` or `codec`.")
    with SafeTensorReader(checkpoint_path) as reader:
        expected_metadata = {
            "component": component,
            "format": VUI_NATIVE_FORMAT,
            "format_version": str(VUI_NATIVE_FORMAT_VERSION),
        }
        if reader.metadata != expected_metadata:
            raise ValueError(
                f"Vui {component} Safetensors metadata is incompatible: "
                f"expected {expected_metadata!r}, found {reader.metadata!r}.")
        return reader.state_dict()


__all__ = [
    "VUI_NATIVE_CODEC_FILENAME",
    "VUI_NATIVE_CONFIG_FILENAME",
    "VUI_NATIVE_FORMAT",
    "VUI_NATIVE_FORMAT_VERSION",
    "VUI_NATIVE_MODEL_FILENAME",
    "NativeVuiArtifact",
    "export_vui_pretrained",
    "load_vui_safetensors",
    "resolve_native_vui_artifact",
]
