"""Strict GPT-SoVITS checkpoint conversion and Safetensors I/O."""

from __future__ import annotations

import hashlib
import io
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from voicehub.checkpointing import SafeTensorReader, save_safetensors
from voicehub.hub import read_json_file, resolve_pretrained_file, write_json_file
from voicehub.models.gptsovits.native.configuration import (
    GPTSoVITSS1Config,
    GPTSoVITSS2Config,
    normalize_gptsovits_variant,
    s1_variant_for_s2,
)
from voicehub.models.gptsovits.native.metadata import (
    GPT_SOVITS_LICENSE,
    GPT_SOVITS_REPOSITORY,
    GPT_SOVITS_REVISION,
    GPT_SOVITS_SOURCE_REVISION,
    GPT_SOVITS_VARIANTS,
    LEGACY_NATIVE_FORMAT,
    LEGACY_NATIVE_FORMAT_VERSION,
    NATIVE_CONFIG_FILENAME,
    NATIVE_FORMAT,
    NATIVE_FORMAT_VERSION,
    NATIVE_S1_FILENAME,
    NATIVE_S2_DISCRIMINATOR_FILENAME,
    NATIVE_S2_GENERATOR_FILENAME,
    S1_FILENAME,
    S1_INVENTORY,
    S1_SHA256,
    S1_SUBFOLDER,
    S1_TENSORS,
    S1_VALUES,
    S2_DISCRIMINATOR_FILENAME,
    S2_DISCRIMINATOR_INVENTORY,
    S2_DISCRIMINATOR_SHA256,
    S2_DISCRIMINATOR_SUBFOLDER,
    S2_DISCRIMINATOR_TENSORS,
    S2_DISCRIMINATOR_VALUES,
    S2_GENERATOR_FILENAME,
    S2_GENERATOR_INVENTORY,
    S2_GENERATOR_SHA256,
    S2_GENERATOR_SUBFOLDER,
    S2_GENERATOR_TENSORS,
    S2_GENERATOR_VALUES,
)


@dataclass(frozen=True, slots=True)
class GPTSoVITSArtifacts:
    """Coherent S1/S2 artifact set."""

    source: str | Path
    revision: str | None
    s1_path: Path
    s2_generator_path: Path
    s2_discriminator_path: Path | None
    s1_config: GPTSoVITSS1Config
    s2_config: GPTSoVITSS2Config
    legacy_pytorch: bool
    official_release: bool
    integrity: GPTSoVITSArtifactIntegrity


@dataclass(frozen=True, slots=True)
class GPTSoVITSCheckpointReport:
    """Strict inventory summary for one stage."""

    component: str
    tensor_count: int
    parameter_count: int
    inventory_fingerprint: str


@dataclass(frozen=True, slots=True)
class GPTSoVITSComponentIntegrity:
    """Digest and topology expected for one staged artifact file."""

    sha256: str
    inventory_fingerprint: str
    tensor_count: int
    parameter_count: int


@dataclass(frozen=True, slots=True)
class GPTSoVITSArtifactIntegrity:
    """Optional per-component integrity declared by an artifact set."""

    s1: GPTSoVITSComponentIntegrity | None = None
    s2_generator: GPTSoVITSComponentIntegrity | None = None
    s2_discriminator: GPTSoVITSComponentIntegrity | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _component_integrity(
    payload: Any,
    *,
    component: str,
    filename: str,
) -> GPTSoVITSComponentIntegrity:
    if not isinstance(payload, Mapping):
        raise ValueError(f"Native GPT-SoVITS {component} integrity must be a mapping.")
    sha256 = payload.get("sha256")
    inventory = payload.get("inventory_fingerprint")
    tensor_count = payload.get("tensor_count")
    parameter_count = payload.get("parameter_count")
    if payload.get("filename") != filename:
        raise ValueError(f"Native GPT-SoVITS {component} filename is incompatible.")
    if (not isinstance(sha256, str) or len(sha256) != 64 or
            any(character not in "0123456789abcdef" for character in sha256)):
        raise ValueError(f"Native GPT-SoVITS {component} SHA-256 is invalid.")
    if (not isinstance(inventory, str) or len(inventory) != 64 or
            any(character not in "0123456789abcdef" for character in inventory)):
        raise ValueError(f"Native GPT-SoVITS {component} inventory fingerprint is invalid.")
    for name, value in (
        ("tensor_count", tensor_count),
        ("parameter_count", parameter_count),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"Native GPT-SoVITS {component} {name} must be positive.")
    return GPTSoVITSComponentIntegrity(
        sha256=sha256,
        inventory_fingerprint=inventory,
        tensor_count=tensor_count,
        parameter_count=parameter_count,
    )


def _validate_file_digest(
    path: Path,
    integrity: GPTSoVITSComponentIntegrity,
    *,
    component: str,
) -> None:
    actual = _sha256(path)
    if actual != integrity.sha256:
        raise ValueError(f"Native GPT-SoVITS {component} SHA-256 mismatch: {actual}.")


def tensor_inventory_fingerprint(tensors: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(tensors):
        tensor = tensors[name]
        if not isinstance(name, str) or not name:
            raise ValueError("Checkpoint tensor names must be non-empty.")
        if not isinstance(tensor, Tensor):
            raise TypeError(f"Checkpoint value {name!r} is not a tensor.")
        dtype = str(tensor.dtype).removeprefix("torch.")
        shape = ",".join(str(item) for item in tensor.shape)
        digest.update(f"{name}|{dtype}|{shape}\n".encode())
    return digest.hexdigest()


def _resolve_legacy_file(
    source: str | Path,
    filename: str,
    subfolder: str,
    **kwargs: Any,
) -> Path:
    source_path = Path(source).expanduser()
    if source_path.is_dir():
        candidates = [source_path / filename]
        if subfolder:
            candidates.insert(0, source_path / subfolder / filename)
        matches = [candidate for candidate in candidates if candidate.is_file()]
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected exactly one GPT-SoVITS checkpoint named {filename} "
                f"under {source_path}.")
        return matches[0].resolve()
    return resolve_pretrained_file(
        source,
        filename,
        subfolder=subfolder,
        **kwargs,
    ).resolve()


def resolve_gptsovits_artifacts(
    source: str | Path,
    *,
    variant: str | None = None,
    revision: str | None = None,
    cache_dir: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
    require_discriminator: bool = False,
) -> GPTSoVITSArtifacts:
    """Resolve one native artifact directory or an exact public release set."""
    requested_variant = (None if variant is None else normalize_gptsovits_variant(variant))
    source_path = Path(source).expanduser()
    native_config = source_path / NATIVE_CONFIG_FILENAME
    if source_path.is_dir() and native_config.is_file():
        payload = read_json_file(native_config)
        artifact_format = payload.get("format")
        format_version = payload.get("format_version")
        current_format = (artifact_format == NATIVE_FORMAT and format_version == NATIVE_FORMAT_VERSION)
        legacy_format = (
            artifact_format == LEGACY_NATIVE_FORMAT and format_version == LEGACY_NATIVE_FORMAT_VERSION)
        if not current_format and not legacy_format:
            raise ValueError("Unexpected GPT-SoVITS native artifact format.")
        artifact_variant = ("v2" if legacy_format else normalize_gptsovits_variant(payload.get("variant")))
        if (requested_variant is not None and artifact_variant != requested_variant):
            raise ValueError(
                f"Native GPT-SoVITS artifact variant {artifact_variant!r} "
                f"does not match requested variant {requested_variant!r}.")
        s1_path = source_path / NATIVE_S1_FILENAME
        generator_path = source_path / NATIVE_S2_GENERATOR_FILENAME
        discriminator_path = source_path / NATIVE_S2_DISCRIMINATOR_FILENAME
        for component, path in (
            ("S1", s1_path),
            ("S2 generator", generator_path),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"Native GPT-SoVITS {component} file is missing: {path}.")
        if require_discriminator and not discriminator_path.is_file():
            raise FileNotFoundError("Native GPT-SoVITS S2 discriminator file is missing.")
        files = payload.get("files")
        if not isinstance(files, Mapping):
            raise ValueError("Native GPT-SoVITS artifact config is missing file integrity.")
        s1_integrity = _component_integrity(
            files.get("s1"),
            component="S1",
            filename=NATIVE_S1_FILENAME,
        )
        generator_integrity = _component_integrity(
            files.get("s2_generator"),
            component="S2 generator",
            filename=NATIVE_S2_GENERATOR_FILENAME,
        )
        discriminator_integrity = (
            _component_integrity(
                files.get("s2_discriminator"),
                component="S2 discriminator",
                filename=NATIVE_S2_DISCRIMINATOR_FILENAME,
            ) if discriminator_path.is_file() else None)
        _validate_file_digest(s1_path, s1_integrity, component="S1")
        _validate_file_digest(
            generator_path,
            generator_integrity,
            component="S2 generator",
        )
        if discriminator_integrity is not None:
            _validate_file_digest(
                discriminator_path,
                discriminator_integrity,
                component="S2 discriminator",
            )
        s1_config = GPTSoVITSS1Config.from_dict(payload["s1"])
        s2_config = GPTSoVITSS2Config.from_dict(payload["s2"])
        if s2_config.version != artifact_variant:
            raise ValueError("Native GPT-SoVITS S2 config has a conflicting variant.")
        if s1_config.version != s1_variant_for_s2(artifact_variant):
            raise ValueError("Native GPT-SoVITS S1/S2 variants are incompatible.")
        return GPTSoVITSArtifacts(
            source=source_path,
            revision=payload.get("source_revision"),
            s1_path=s1_path.resolve(),
            s2_generator_path=generator_path.resolve(),
            s2_discriminator_path=(discriminator_path.resolve() if discriminator_path.is_file() else None),
            s1_config=s1_config,
            s2_config=s2_config,
            legacy_pytorch=False,
            official_release=False,
            integrity=GPTSoVITSArtifactIntegrity(
                s1=s1_integrity,
                s2_generator=generator_integrity,
                s2_discriminator=discriminator_integrity,
            ),
        )

    canonical_variant = requested_variant or "v2"
    release = GPT_SOVITS_VARIANTS[canonical_variant]
    official = str(source) == GPT_SOVITS_REPOSITORY
    resolved_revision = revision
    if official:
        if revision is None:
            resolved_revision = GPT_SOVITS_REVISION
        elif revision != GPT_SOVITS_REVISION:
            raise ValueError(
                "The public GPT-SoVITS provider is audited only at immutable "
                f"revision {GPT_SOVITS_REVISION}.")
    kwargs = {
        "revision": resolved_revision,
        "cache_dir": cache_dir,
        "token": token,
        "local_files_only": local_files_only,
    }
    s1_path = _resolve_legacy_file(
        source,
        release.s1.filename,
        release.s1.subfolder,
        **kwargs,
    )
    generator_path = _resolve_legacy_file(
        source,
        release.s2_generator.filename,
        release.s2_generator.subfolder,
        **kwargs,
    )
    discriminator_path = None
    if require_discriminator or source_path.is_dir():
        try:
            discriminator_path = _resolve_legacy_file(
                source,
                release.s2_discriminator.filename,
                release.s2_discriminator.subfolder,
                **kwargs,
            )
        except FileNotFoundError:
            if require_discriminator:
                raise
    return GPTSoVITSArtifacts(
        source=source,
        revision=resolved_revision,
        s1_path=s1_path,
        s2_generator_path=generator_path,
        s2_discriminator_path=discriminator_path,
        s1_config=GPTSoVITSS1Config.for_variant(canonical_variant),
        s2_config=GPTSoVITSS2Config.for_variant(canonical_variant),
        legacy_pytorch=True,
        official_release=official,
        integrity=(
            GPTSoVITSArtifactIntegrity(
                s1=GPTSoVITSComponentIntegrity(
                    sha256=release.s1.sha256,
                    inventory_fingerprint=release.s1.inventory_fingerprint,
                    tensor_count=release.s1.tensor_count,
                    parameter_count=release.s1.parameter_count,
                ),
                s2_generator=GPTSoVITSComponentIntegrity(
                    sha256=release.s2_generator.sha256,
                    inventory_fingerprint=release.s2_generator.inventory_fingerprint,
                    tensor_count=release.s2_generator.tensor_count,
                    parameter_count=release.s2_generator.parameter_count,
                ),
                s2_discriminator=GPTSoVITSComponentIntegrity(
                    sha256=release.s2_discriminator.sha256,
                    inventory_fingerprint=release.s2_discriminator.inventory_fingerprint,
                    tensor_count=release.s2_discriminator.tensor_count,
                    parameter_count=release.s2_discriminator.parameter_count,
                ),
            ) if official else GPTSoVITSArtifactIntegrity()),
    )


class _LegacyHParams:
    """Data-only compatibility target for historical ``utils.HParams``."""

    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)

    def keys(self):
        return self.__dict__.keys()

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


def _plain_mapping(payload: Any, *, component: str) -> dict[str, Any]:
    if isinstance(payload, Mapping):
        items = payload.items()
    elif isinstance(payload, _LegacyHParams):
        items = payload.__dict__.items()
    else:
        raise ValueError(f"GPT-SoVITS {component} checkpoint config must be a mapping.")
    normalized = {}
    for key, value in items:
        if not isinstance(key, str):
            raise ValueError(f"GPT-SoVITS {component} config keys must be strings.")
        if isinstance(value, (Mapping, _LegacyHParams)):
            value = _plain_mapping(value, component=component)
        elif isinstance(value, tuple):
            value = list(value)
        normalized[key] = value
    return normalized


def _legacy_state(
    path: Path,
    *,
    component: str,
    expected_sha256: str | None,
    trust_pickle_checkpoint: bool,
    variant: str,
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    if trust_pickle_checkpoint is not True:
        raise ValueError(
            f"The public GPT-SoVITS {component} checkpoint is a PyTorch "
            "pickle container. Review its origin and pass "
            "`trust_pickle_checkpoint=True` once, then export Safetensors.")
    if expected_sha256 is not None:
        actual = _sha256(path)
        if actual != expected_sha256:
            raise ValueError(f"Official GPT-SoVITS {component} SHA-256 mismatch: {actual}.")
    source: Path | io.BytesIO = path
    with path.open("rb") as stream:
        header = stream.read(2)
    tagged_variants = {
        b"05": "v2Pro",
        b"06": "v2ProPlus",
    }
    if header in tagged_variants:
        tagged_variant = tagged_variants[header]
        if tagged_variant != variant:
            raise ValueError(
                f"GPT-SoVITS checkpoint header declares {tagged_variant}, "
                f"not requested variant {variant}.")
        with path.open("rb") as stream:
            stream.read(2)
            source = io.BytesIO(b"PK" + stream.read())
    elif header != b"PK":
        raise ValueError(
            "Unsupported GPT-SoVITS checkpoint header. V3/V4 and LoRA "
            "containers require a different native graph.")
    try:
        with torch.serialization.safe_globals([
            (_LegacyHParams, "utils.HParams"),
        ]):
            payload = torch.load(
                source,
                map_location="cpu",
                weights_only=True,
            )
    except TypeError as error:  # pragma: no cover
        raise RuntimeError("GPT-SoVITS conversion requires PyTorch weights-only loading.") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"GPT-SoVITS {component} checkpoint must contain a mapping.")
    state = payload.get("weight")
    config = payload.get("config")
    if not isinstance(state, Mapping):
        raise ValueError(f"GPT-SoVITS {component} checkpoint is missing weight/config mappings.")
    if any(not isinstance(name, str) or not isinstance(value, Tensor) for name, value in state.items()):
        raise TypeError(f"GPT-SoVITS {component} weights must map names to tensors.")
    return dict(state), _plain_mapping(config, component=component)


def _safe_state(path: Path) -> dict[str, Tensor]:
    with SafeTensorReader(path) as reader:
        return reader.state_dict()


def _load_component(
    module: nn.Module,
    state: Mapping[str, Tensor],
    *,
    component: str,
    expected_inventory: str | None,
    expected_tensors: int | None,
    expected_values: int | None,
) -> GPTSoVITSCheckpointReport:
    expected = module.state_dict()
    missing = tuple(sorted(set(expected) - set(state)))
    unexpected = tuple(sorted(set(state) - set(expected)))
    mismatches = []
    for name in sorted(set(state) & set(expected)):
        actual_shape = tuple(state[name].shape)
        expected_shape = tuple(expected[name].shape)
        if actual_shape != expected_shape:
            mismatches.append((name, actual_shape, expected_shape))
    mismatches = tuple(mismatches)
    if missing or unexpected or mismatches:
        raise ValueError(
            f"GPT-SoVITS {component} inventory is incompatible "
            f"(missing={missing[:8]!r}, unexpected={unexpected[:8]!r}, "
            f"shape_mismatches={mismatches[:8]!r}).")
    fingerprint = tensor_inventory_fingerprint(state)
    tensor_count = len(state)
    parameter_count = sum(tensor.numel() for tensor in state.values())
    if expected_inventory is not None and fingerprint != expected_inventory:
        raise ValueError(f"GPT-SoVITS {component} inventory fingerprint mismatch: "
                         f"{fingerprint}.")
    if expected_tensors is not None and tensor_count != expected_tensors:
        raise ValueError(f"GPT-SoVITS {component} tensor count mismatch.")
    if expected_values is not None and parameter_count != expected_values:
        raise ValueError(f"GPT-SoVITS {component} value count mismatch.")
    module.load_state_dict(state, strict=True)
    return GPTSoVITSCheckpointReport(
        component=component,
        tensor_count=tensor_count,
        parameter_count=parameter_count,
        inventory_fingerprint=fingerprint,
    )


def load_gptsovits_checkpoints(
    *,
    s1: nn.Module,
    s2_generator: nn.Module,
    artifacts: GPTSoVITSArtifacts,
    s2_discriminator: nn.Module | None = None,
    trust_pickle_checkpoint: bool = False,
) -> dict[str, GPTSoVITSCheckpointReport]:
    """Strictly load every requested stage after topology and digest
    validation."""
    reports = {}
    variant = artifacts.s2_config.version
    if artifacts.legacy_pytorch:
        s1_integrity = artifacts.integrity.s1
        generator_integrity = artifacts.integrity.s2_generator
        s1_state, s1_payload_config = _legacy_state(
            artifacts.s1_path,
            component="S1",
            expected_sha256=(None if s1_integrity is None else s1_integrity.sha256),
            trust_pickle_checkpoint=trust_pickle_checkpoint,
            variant=variant,
        )
        GPTSoVITSS1Config.from_upstream(
            s1_payload_config,
            variant=variant,
        )
        generator_state, s2_payload_config = _legacy_state(
            artifacts.s2_generator_path,
            component="S2 generator",
            expected_sha256=(None if generator_integrity is None else generator_integrity.sha256),
            trust_pickle_checkpoint=trust_pickle_checkpoint,
            variant=variant,
        )
        GPTSoVITSS2Config.from_upstream(
            s2_payload_config,
            variant=variant,
        )
    else:
        s1_state = _safe_state(artifacts.s1_path)
        generator_state = _safe_state(artifacts.s2_generator_path)
    s1_integrity = artifacts.integrity.s1
    generator_integrity = artifacts.integrity.s2_generator
    reports["s1"] = _load_component(
        s1,
        s1_state,
        component="S1",
        expected_inventory=(None if s1_integrity is None else s1_integrity.inventory_fingerprint),
        expected_tensors=(None if s1_integrity is None else s1_integrity.tensor_count),
        expected_values=(None if s1_integrity is None else s1_integrity.parameter_count),
    )
    reports["s2_generator"] = _load_component(
        s2_generator,
        generator_state,
        component="S2 generator",
        expected_inventory=(
            None if generator_integrity is None else generator_integrity.inventory_fingerprint),
        expected_tensors=(None if generator_integrity is None else generator_integrity.tensor_count),
        expected_values=(None if generator_integrity is None else generator_integrity.parameter_count),
    )
    if s2_discriminator is not None:
        path = artifacts.s2_discriminator_path
        if path is None:
            raise FileNotFoundError("GPT-SoVITS S2 discriminator was requested but is absent.")
        if artifacts.legacy_pytorch:
            discriminator_state, discriminator_config = _legacy_state(
                path,
                component="S2 discriminator",
                expected_sha256=(
                    None if artifacts.integrity.s2_discriminator is None else
                    artifacts.integrity.s2_discriminator.sha256),
                trust_pickle_checkpoint=trust_pickle_checkpoint,
                variant=variant,
            )
            # Historical V1 discriminator metadata set freeze_quantizer=False,
            # although that component has no quantizer. Validate the graph-
            # relevant discriminator fields through the generator contract.
            if variant == "v1":
                discriminator_config["model"]["freeze_quantizer"] = True
            GPTSoVITSS2Config.from_upstream(
                discriminator_config,
                variant=variant,
            )
        else:
            discriminator_state = _safe_state(path)
        discriminator_integrity = artifacts.integrity.s2_discriminator
        reports["s2_discriminator"] = _load_component(
            s2_discriminator,
            discriminator_state,
            component="S2 discriminator",
            expected_inventory=(
                None if discriminator_integrity is None else discriminator_integrity.inventory_fingerprint),
            expected_tensors=(
                None if discriminator_integrity is None else discriminator_integrity.tensor_count),
            expected_values=(
                None if discriminator_integrity is None else discriminator_integrity.parameter_count),
        )
    return reports


def load_gptsovits_discriminator(
    discriminator: nn.Module,
    artifacts: GPTSoVITSArtifacts,
    *,
    trust_pickle_checkpoint: bool = False,
) -> GPTSoVITSCheckpointReport:
    """Load only the S2 discriminator without re-reading S1 and S2-G."""
    path = artifacts.s2_discriminator_path
    if path is None:
        raise FileNotFoundError("GPT-SoVITS S2 discriminator was requested but is absent.")
    if artifacts.legacy_pytorch:
        state, payload_config = _legacy_state(
            path,
            component="S2 discriminator",
            expected_sha256=(
                None if artifacts.integrity.s2_discriminator is None else
                artifacts.integrity.s2_discriminator.sha256),
            trust_pickle_checkpoint=trust_pickle_checkpoint,
            variant=artifacts.s2_config.version,
        )
        if artifacts.s2_config.version == "v1":
            payload_config["model"]["freeze_quantizer"] = True
        GPTSoVITSS2Config.from_upstream(
            payload_config,
            variant=artifacts.s2_config.version,
        )
    else:
        state = _safe_state(path)
    integrity = artifacts.integrity.s2_discriminator
    return _load_component(
        discriminator,
        state,
        component="S2 discriminator",
        expected_inventory=(None if integrity is None else integrity.inventory_fingerprint),
        expected_tensors=(None if integrity is None else integrity.tensor_count),
        expected_values=(None if integrity is None else integrity.parameter_count),
    )


def export_gptsovits_checkpoint(
    directory: str | Path,
    *,
    s1: nn.Module,
    s2_generator: nn.Module,
    s1_config: GPTSoVITSS1Config,
    s2_config: GPTSoVITSS2Config,
    s2_discriminator: nn.Module | None = None,
    source_revision: str | None = None,
) -> Path:
    """Export a fresh-reloadable, pickle-free staged artifact."""
    if s1_config.version != s1_variant_for_s2(s2_config.version):
        raise ValueError("Cannot export checkpoint-incompatible GPT-SoVITS S1/S2 variants.")
    destination = Path(directory).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    components = {
        NATIVE_S1_FILENAME: ("s1", s1),
        NATIVE_S2_GENERATOR_FILENAME: ("s2_generator", s2_generator),
    }
    if s2_discriminator is not None:
        components[NATIVE_S2_DISCRIMINATOR_FILENAME] = (
            "s2_discriminator",
            s2_discriminator,
        )
    file_integrity = {}
    for filename, (component, module) in components.items():
        state = {name: tensor.detach().cpu().contiguous() for name, tensor in module.state_dict().items()}
        fingerprint = tensor_inventory_fingerprint(state)
        path = destination / filename
        save_safetensors(
            state,
            path,
            metadata={
                "format": NATIVE_FORMAT,
                "format_version": str(NATIVE_FORMAT_VERSION),
                "component": component,
                "license": GPT_SOVITS_LICENSE,
                "tensor_inventory_fingerprint": fingerprint,
            },
        )
        file_integrity[component] = {
            "filename": filename,
            "sha256": _sha256(path),
            "inventory_fingerprint": fingerprint,
            "tensor_count": len(state),
            "parameter_count": sum(tensor.numel() for tensor in state.values()),
        }
    write_json_file(
        destination / NATIVE_CONFIG_FILENAME,
        {
            "format": NATIVE_FORMAT,
            "format_version": NATIVE_FORMAT_VERSION,
            "architecture": "gpt-sovits-classic-s2",
            "variant": s2_config.version,
            "license": GPT_SOVITS_LICENSE,
            "source_repository": GPT_SOVITS_REPOSITORY,
            "source_revision": source_revision or GPT_SOVITS_REVISION,
            "s1": s1_config.to_dict(),
            "s2": s2_config.to_dict(),
            "files": file_integrity,
        },
    )
    return destination


def convert_gptsovits_legacy_checkpoints(
    source: str | Path,
    destination: str | Path,
    *,
    variant: str = "v2",
    revision: str | None = None,
    cache_dir: str | None = None,
    local_files_only: bool = False,
    trust_pickle_checkpoint: bool = False,
) -> Path:
    """Convert one reviewed public classic S1/G/D set into Safetensors."""
    from voicehub.models.gptsovits.native.modeling import build_s2_discriminator, build_s2_generator
    from voicehub.models.gptsovits.native.semantic import GPTSoVITSSemanticModel

    artifacts = resolve_gptsovits_artifacts(
        source,
        variant=variant,
        revision=revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        require_discriminator=True,
    )
    if not artifacts.legacy_pytorch:
        raise ValueError("GPT-SoVITS conversion expects legacy public checkpoints.")
    # The reviewed public classic-S2 weights are FP16. Build targets in
    # the same dtype so conversion does not silently double artifact size or
    # rewrite the released numeric representation.
    s1 = GPTSoVITSSemanticModel(artifacts.s1_config).half()
    generator = build_s2_generator(artifacts.s2_config).half()
    discriminator = build_s2_discriminator(artifacts.s2_config).half()
    load_gptsovits_checkpoints(
        s1=s1,
        s2_generator=generator,
        s2_discriminator=discriminator,
        artifacts=artifacts,
        trust_pickle_checkpoint=trust_pickle_checkpoint,
    )
    return export_gptsovits_checkpoint(
        destination,
        s1=s1,
        s2_generator=generator,
        s2_discriminator=discriminator,
        s1_config=artifacts.s1_config,
        s2_config=artifacts.s2_config,
        source_revision=artifacts.revision,
    )


__all__ = [
    "GPT_SOVITS_LICENSE",
    "GPT_SOVITS_REPOSITORY",
    "GPT_SOVITS_REVISION",
    "GPT_SOVITS_SOURCE_REVISION",
    "GPTSoVITSArtifacts",
    "GPTSoVITSArtifactIntegrity",
    "GPTSoVITSCheckpointReport",
    "GPTSoVITSComponentIntegrity",
    "NATIVE_CONFIG_FILENAME",
    "NATIVE_FORMAT",
    "NATIVE_FORMAT_VERSION",
    "NATIVE_S1_FILENAME",
    "NATIVE_S2_DISCRIMINATOR_FILENAME",
    "NATIVE_S2_GENERATOR_FILENAME",
    "S1_FILENAME",
    "S1_INVENTORY",
    "S1_SHA256",
    "S1_TENSORS",
    "S1_VALUES",
    "S2_DISCRIMINATOR_FILENAME",
    "S2_DISCRIMINATOR_INVENTORY",
    "S2_DISCRIMINATOR_SHA256",
    "S2_DISCRIMINATOR_TENSORS",
    "S2_DISCRIMINATOR_VALUES",
    "S2_GENERATOR_FILENAME",
    "S2_GENERATOR_INVENTORY",
    "S2_GENERATOR_SHA256",
    "S2_GENERATOR_TENSORS",
    "S2_GENERATOR_VALUES",
    "convert_gptsovits_legacy_checkpoints",
    "export_gptsovits_checkpoint",
    "load_gptsovits_discriminator",
    "load_gptsovits_checkpoints",
    "resolve_gptsovits_artifacts",
    "tensor_inventory_fingerprint",
]
