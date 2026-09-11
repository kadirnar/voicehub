"""Strict conversion for the audited WeNet GigaSpeech U2++ checkpoint."""

from __future__ import annotations

import hashlib
import shutil
import tarfile
import tempfile
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from voicehub.checkpointing import CheckpointAdapter, CopyTensor, SafeTensorReader, TensorPlan, save_safetensors
from voicehub.hub import write_json_file
from voicehub.models.asr_wenet.native.configuration import WeNetU2PPConfig
from voicehub.models.asr_wenet.native.metadata import (
    GIGASPEECH_ARCHIVE_SHA256,
    GIGASPEECH_CMVN_SHA256,
    GIGASPEECH_CONFIG_SHA256,
    GIGASPEECH_STATE_VALUES,
    GIGASPEECH_TENSOR_COUNT,
    GIGASPEECH_TENSOR_FINGERPRINT,
    GIGASPEECH_TOKENIZER_SHA256,
    GIGASPEECH_UNITS_SHA256,
    GIGASPEECH_WEIGHTS_SHA256,
)
from voicehub.models.asr_wenet.native.tokenization import WeNetGigaSpeechTokenizer
from voicehub.processing import load_global_cmvn

NATIVE_WENET_FORMAT = "voicehub-wenet-gigaspeech-u2pp-v1"
NATIVE_WENET_FILENAME = "model.safetensors"
WENET_TOKENIZER_FILENAME = "tokenizer.model"
WENET_UNITS_FILENAME = "units.txt"
_SOURCE_FILES = {
    "final.pt": (GIGASPEECH_WEIGHTS_SHA256, 600 * 1024 * 1024),
    "global_cmvn": (GIGASPEECH_CMVN_SHA256, 1024 * 1024),
    "train.yaml": (GIGASPEECH_CONFIG_SHA256, 1024 * 1024),
    "train_xl_unigram5000.model": (
        GIGASPEECH_TOKENIZER_SHA256,
        16 * 1024 * 1024,
    ),
    "units.txt": (GIGASPEECH_UNITS_SHA256, 4 * 1024 * 1024),
}
WeNetConfigLike = WeNetU2PPConfig | Mapping[str, Any]


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_inventory_fingerprint(tensors: Mapping[str, Any]) -> str:
    """Hash sorted ``name|portable-dtype|dimxdim`` inventory rows."""
    rows = []
    for name, tensor in sorted(tensors.items()):
        dtype = {
            "torch.float32": "F32",
            "torch.float64": "F64",
            "torch.float16": "F16",
            "torch.bfloat16": "BF16",
            "torch.int64": "I64",
            "torch.int32": "I32",
            "torch.int16": "I16",
            "torch.int8": "I8",
            "torch.uint8": "U8",
            "torch.bool": "BOOL",
        }.get(str(tensor.dtype), str(tensor.dtype))
        shape = "x".join(str(item) for item in tensor.shape)
        rows.append(f"{name}|{dtype}|{shape}")
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def native_wenet_tensor_shapes(config: WeNetConfigLike | None = None, ) -> dict[str, tuple[int, ...]]:
    import torch

    from voicehub.models.asr_wenet.native.modeling import WeNetU2PPForASR

    with torch.device("meta"):
        model = WeNetU2PPForASR(WeNetU2PPConfig.coerce(config))
    return {name: tuple(tensor.shape) for name, tensor in model.state_dict().items()}


class WeNetU2PPSafeTensorsCheckpointAdapter(CheckpointAdapter):
    """Identity-map a validated native U2++ Safetensors checkpoint."""

    architecture_id = "wenet-gigaspeech-u2pp"
    adapter_id = "voicehub-wenet-u2pp-safetensors"
    adapter_version = "1"

    def probe(
        self,
        files: tuple[Path, ...],
        config: Mapping[str, Any],
    ) -> bool:
        try:
            WeNetU2PPConfig.from_dict(config)
        except (TypeError, ValueError):
            return False
        return any(path.suffix == ".safetensors" for path in files)

    def tensor_plan(self, config: Mapping[str, Any]) -> TensorPlan:
        shapes = native_wenet_tensor_shapes(config)
        return TensorPlan(rules=tuple(CopyTensor(name, name) for name in sorted(shapes)))


def _verify_source_directory(root: Path) -> dict[str, Path]:
    result = {}
    for name, (expected_sha, maximum_size) in _SOURCE_FILES.items():
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"WeNet source directory is missing {name!r}: {root}.")
        size = path.stat().st_size
        if size <= 0 or size > maximum_size:
            raise ValueError(f"WeNet source file {name!r} has an unsafe size.")
        actual_sha = file_sha256(path)
        if actual_sha != expected_sha:
            raise ValueError(f"WeNet source file {name!r} digest mismatch: {actual_sha}.")
        result[name] = path
    return result


@contextmanager
def _verified_source_files(source: Path) -> Iterator[dict[str, Path]]:
    if source.is_dir():
        yield _verify_source_directory(source)
        return
    if not source.is_file():
        raise FileNotFoundError(f"WeNet checkpoint source was not found: {source}.")
    if file_sha256(source) != GIGASPEECH_ARCHIVE_SHA256:
        raise ValueError(
            "Native WeNet conversion is limited to the hash-pinned "
            "20210728 GigaSpeech U2++ checkpoint archive.")
    with tempfile.TemporaryDirectory(prefix="voicehub-wenet-convert-") as raw:
        temporary = Path(raw)
        try:
            archive = tarfile.open(source, mode="r:gz")
        except (tarfile.TarError, OSError) as error:
            raise ValueError("Invalid WeNet GigaSpeech archive.") from error
        with archive:
            selected: dict[str, tarfile.TarInfo] = {}
            for member in archive.getmembers():
                if member.issym() or member.islnk() or member.isdev():
                    raise ValueError("WeNet conversion refuses archive links and devices.")
                basename = Path(member.name).name
                if basename not in _SOURCE_FILES:
                    continue
                if not member.isfile() or basename in selected:
                    raise ValueError(f"Invalid or duplicate WeNet member {basename!r}.")
                maximum = _SOURCE_FILES[basename][1]
                if member.size <= 0 or member.size > maximum:
                    raise ValueError(f"WeNet member {basename!r} has an unsafe size.")
                selected[basename] = member
            missing = set(_SOURCE_FILES) - set(selected)
            if missing:
                raise ValueError(f"WeNet archive is missing {sorted(missing)!r}.")
            for name, member in selected.items():
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError(f"Could not read WeNet member {name!r}.")
                destination = temporary / name
                digest = hashlib.sha256()
                total = 0
                with destination.open("wb") as output:
                    while chunk := stream.read(1024 * 1024):
                        total += len(chunk)
                        if total > member.size:
                            raise ValueError(f"WeNet member {name!r} exceeds declared size.")
                        digest.update(chunk)
                        output.write(chunk)
                if total != member.size:
                    raise ValueError(f"WeNet member {name!r} has an inconsistent size.")
                if digest.hexdigest() != _SOURCE_FILES[name][0]:
                    raise ValueError(f"WeNet member {name!r} digest mismatch.")
        yield _verify_source_directory(temporary)


def _load_restricted_state(
    path: Path,
    *,
    trust_pickle_checkpoint: bool,
) -> Mapping[str, Any]:
    if not trust_pickle_checkpoint:
        raise ValueError(
            "The official WeNet `final.pt` uses Python's pickle container. "
            "Set `trust_pickle_checkpoint=True` only after verifying the "
            "documented SHA-256; conversion still uses PyTorch's restricted "
            "`weights_only=True` reader.")
    import torch

    try:
        state = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError("Could not read the restricted WeNet tensor state.") from error
    if not isinstance(state, Mapping) or not state:
        raise ValueError("WeNet checkpoint must be a non-empty tensor mapping.")
    if any(not isinstance(name, str) or not isinstance(tensor, torch.Tensor)
           for name, tensor in state.items()):
        raise TypeError("WeNet state must map string names to tensors only.")
    return state


def _validate_state(
    state: Mapping[str, Any],
    config: WeNetU2PPConfig,
) -> str:
    expected_shapes = native_wenet_tensor_shapes(config)
    expected_names = set(expected_shapes)
    source_names = set(state)
    if source_names != expected_names:
        raise ValueError(
            "WeNet tensor namespace does not match the audited U2++ graph "
            f"(missing={sorted(expected_names - source_names)}, "
            f"unexpected={sorted(source_names - expected_names)}).")
    mismatches = {
        name: (tuple(state[name].shape), expected_shapes[name])
        for name in expected_names if tuple(state[name].shape) != expected_shapes[name]
    }
    if mismatches:
        raise ValueError(f"WeNet U2++ tensor shape mismatch: {mismatches}.")
    fingerprint = tensor_inventory_fingerprint(state)
    if fingerprint != GIGASPEECH_TENSOR_FINGERPRINT:
        raise ValueError("WeNet tensor inventory fingerprint mismatch.")
    if len(state) != GIGASPEECH_TENSOR_COUNT:
        raise ValueError("WeNet tensor count mismatch.")
    if sum(tensor.numel() for tensor in state.values()) != GIGASPEECH_STATE_VALUES:
        raise ValueError("WeNet stored-value count mismatch.")
    return fingerprint


def convert_wenet_gigaspeech_checkpoint(
    source: str | Path,
    destination: str | Path,
    *,
    trust_pickle_checkpoint: bool = False,
) -> Path:
    """Convert the exact public archive/directory to native Safetensors."""
    source_path = Path(source).expanduser().resolve()
    output = Path(destination).expanduser().resolve()
    with _verified_source_files(source_path) as files:
        state = _load_restricted_state(
            files["final.pt"],
            trust_pickle_checkpoint=trust_pickle_checkpoint,
        )
        config = WeNetU2PPConfig()
        fingerprint = _validate_state(state, config)
        normalization = load_global_cmvn(
            files["global_cmvn"],
            format="auto",
            expected_dimension=config.input_dim,
        )
        import torch

        if not torch.allclose(
                normalization.mean,
                state["encoder.global_cmvn.mean"],
                atol=3e-6,
                rtol=1e-6,
        ) or not torch.allclose(
                normalization.inverse_std,
                state["encoder.global_cmvn.istd"],
                atol=1e-7,
                rtol=1e-6,
        ):
            raise ValueError("WeNet CMVN statistics disagree with checkpoint buffers.")
        tokenizer = WeNetGigaSpeechTokenizer.from_files(
            files["train_xl_unigram5000.model"],
            files["units.txt"],
        )
        if tokenizer.vocabulary_size != config.vocab_size:
            raise ValueError("WeNet tokenizer vocabulary does not match the model graph.")
        output.mkdir(parents=True, exist_ok=True)
        checkpoint = output / NATIVE_WENET_FILENAME
        save_safetensors(
            {name: state[name].detach().cpu().contiguous()
             for name in sorted(state)},
            checkpoint,
            metadata={
                "architecture": "wenet-gigaspeech-u2pp",
                "format": NATIVE_WENET_FORMAT,
                "source_sha256": GIGASPEECH_WEIGHTS_SHA256,
                "tensor_fingerprint": fingerprint,
            },
        )
        shutil.copyfile(
            files["train_xl_unigram5000.model"],
            output / WENET_TOKENIZER_FILENAME,
        )
        shutil.copyfile(
            files["units.txt"],
            output / WENET_UNITS_FILENAME,
        )
        values = config.to_dict()
        values.update({
            'architectures': ["WeNetU2PPForASR"],
            "checkpoint_format": NATIVE_WENET_FORMAT,
            "model_type": "asr_wenet",
            "source_checkpoint_name": source_path.name,
            "source_checkpoint_sha256": GIGASPEECH_WEIGHTS_SHA256,
            "source_tensor_fingerprint": fingerprint,
            "voicehub_provider": "asr_wenet",
        })
        write_json_file(output / "config.json", values)

    from voicehub.models.asr_wenet.native.modeling import WeNetU2PPForASR

    with SafeTensorReader(checkpoint) as reader:
        WeNetU2PPSafeTensorsCheckpointAdapter().load_streaming(
            WeNetU2PPForASR(config),
            reader,
            values,
            strict=True,
        )
    return output


__all__ = [
    "NATIVE_WENET_FILENAME",
    "NATIVE_WENET_FORMAT",
    "WENET_TOKENIZER_FILENAME",
    "WENET_UNITS_FILENAME",
    "WeNetU2PPSafeTensorsCheckpointAdapter",
    "convert_wenet_gigaspeech_checkpoint",
    "file_sha256",
    "native_wenet_tensor_shapes",
    "tensor_inventory_fingerprint",
]
