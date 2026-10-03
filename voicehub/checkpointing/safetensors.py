"""Dependency-free Safetensors I/O backed by PyTorch.

This module implements the documented Safetensors file layout directly
so native VoiceHub architectures do not need to import the
``safetensors`` package.  The reader is intentionally strict: untrusted
headers are bounded, duplicate JSON keys are rejected, tensor ranges may
not overlap, and every payload size must agree with its declared dtype
and shape.

Only tensor bytes are materialized by
:meth:`SafeTensorReader.get_tensor`. Opening a large checkpoint
therefore has constant memory cost.
"""

from __future__ import annotations

import ctypes
import json
import mmap
import os
import struct
import sys
import tempfile
from collections.abc import Iterable, Mapping
from contextlib import AbstractContextManager, ExitStack
from dataclasses import dataclass
from math import prod
from pathlib import Path
from typing import Any, BinaryIO

from voicehub.checkpointing.errors import CheckpointFormatError
from voicehub.json_utils import parse_json_object

_HEADER_LENGTH_BYTES = 8
_DEFAULT_MAX_HEADER_BYTES = 100 * 1024 * 1024
_DEFAULT_MAX_TENSORS = 1_000_000
_DTYPE_SIZES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "I16": 2,
    "U16": 2,
    "I32": 4,
    "U32": 4,
    "I64": 8,
    "U64": 8,
    "F16": 2,
    "BF16": 2,
    "F32": 4,
    "F64": 8,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
}


def _torch():
    try:
        import torch
    except ModuleNotFoundError as error:  # pragma: no cover - package invariant
        raise RuntimeError(
            "Native checkpoint loading requires PyTorch, VoiceHub's compute "
            "runtime.") from error
    return torch


def _read_exact(stream: BinaryIO, size: int, *, context: str) -> bytes:
    value = stream.read(size)
    if len(value) != size:
        raise CheckpointFormatError(
            f"Safetensors file ended while reading {context}: expected "
            f"{size} bytes, found {len(value)}.")
    return value


def _checked_shape(value: Any, *, tensor_name: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise CheckpointFormatError(f"Tensor {tensor_name!r} has a non-list `shape`.")
    dimensions: list[int] = []
    for dimension in value:
        if isinstance(dimension, bool) or not isinstance(dimension, int):
            raise CheckpointFormatError(f"Tensor {tensor_name!r} has a non-integer dimension.")
        if dimension < 0:
            raise CheckpointFormatError(f"Tensor {tensor_name!r} has a negative dimension.")
        dimensions.append(dimension)
    return tuple(dimensions)


@dataclass(frozen=True)
class TensorRecord:
    """Validated metadata for one tensor payload."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    start: int
    end: int

    @property
    def number_of_elements(self) -> int:
        return prod(self.shape, start=1)

    @property
    def number_of_bytes(self) -> int:
        return self.end - self.start


def _parse_record(name: str, value: Any) -> TensorRecord:
    if not isinstance(name, str) or not name:
        raise CheckpointFormatError("Safetensors tensor names must be non-empty.")
    if not isinstance(value, Mapping):
        raise CheckpointFormatError(f"Tensor {name!r} header must be a JSON object.")
    expected_fields = {"dtype", "shape", "data_offsets"}
    unknown = set(value) - expected_fields
    missing = expected_fields - set(value)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing {sorted(missing)!r}")
        if unknown:
            details.append(f"unknown {sorted(unknown)!r}")
        raise CheckpointFormatError(f"Tensor {name!r} has invalid header fields ({'; '.join(details)}).")
    dtype = value["dtype"]
    if not isinstance(dtype, str) or dtype not in _DTYPE_SIZES:
        raise CheckpointFormatError(f"Tensor {name!r} uses unsupported dtype {dtype!r}.")
    shape = _checked_shape(value["shape"], tensor_name=name)
    offsets = value["data_offsets"]
    if (not isinstance(offsets, list) or len(offsets) != 2 or
            any(isinstance(item, bool) or not isinstance(item, int) for item in offsets)):
        raise CheckpointFormatError(f"Tensor {name!r} must declare two integer `data_offsets`.")
    start, end = offsets
    if start < 0 or end < start:
        raise CheckpointFormatError(f"Tensor {name!r} has invalid byte offsets {offsets!r}.")
    expected_size = prod(shape, start=1) * _DTYPE_SIZES[dtype]
    if end - start != expected_size:
        raise CheckpointFormatError(
            f"Tensor {name!r} declares {end - start} bytes, but dtype "
            f"{dtype} and shape {shape!r} require {expected_size}.")
    return TensorRecord(
        name=name,
        dtype=dtype,
        shape=shape,
        start=start,
        end=end,
    )


def _torch_dtype(dtype_name: str):
    torch = _torch()
    names = {
        "BOOL": "bool",
        "U8": "uint8",
        "I8": "int8",
        "I16": "int16",
        "U16": "uint16",
        "I32": "int32",
        "U32": "uint32",
        "I64": "int64",
        "U64": "uint64",
        "F16": "float16",
        "BF16": "bfloat16",
        "F32": "float32",
        "F64": "float64",
        "F8_E4M3": "float8_e4m3fn",
        "F8_E5M2": "float8_e5m2",
    }
    attribute = names[dtype_name]
    result = getattr(torch, attribute, None)
    if result is None:
        raise CheckpointFormatError(
            f"This PyTorch version cannot materialize Safetensors dtype "
            f"{dtype_name!r}.")
    return result


def _safetensors_dtype(dtype: Any) -> str:
    torch = _torch()
    mapping = {
        torch.bool: "BOOL",
        torch.uint8: "U8",
        torch.int8: "I8",
        torch.int16: "I16",
        torch.int32: "I32",
        torch.int64: "I64",
        torch.float16: "F16",
        torch.bfloat16: "BF16",
        torch.float32: "F32",
        torch.float64: "F64",
    }
    for attribute, safetensors_name in (
        ("uint16", "U16"),
        ("uint32", "U32"),
        ("uint64", "U64"),
        ("float8_e4m3fn", "F8_E4M3"),
        ("float8_e5m2", "F8_E5M2"),
    ):
        candidate = getattr(torch, attribute, None)
        if candidate is not None:
            mapping[candidate] = safetensors_name
    try:
        return mapping[dtype]
    except KeyError as error:
        raise CheckpointFormatError(f"Cannot serialize unsupported PyTorch dtype {dtype!r}.") from error


class SafeTensorReader(AbstractContextManager["SafeTensorReader"]):
    """Validate and lazily read one Safetensors file."""

    def __init__(
        self,
        path: str | Path,
        *,
        max_header_bytes: int = _DEFAULT_MAX_HEADER_BYTES,
        max_tensors: int = _DEFAULT_MAX_TENSORS,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"Safetensors file was not found: {self.path}")
        if max_header_bytes <= 0 or max_tensors <= 0:
            raise ValueError("Safetensors parser limits must be positive.")
        self._stream = self.path.open("rb")
        try:
            self._records, self.metadata, self._data_start = self._read_header(
                max_header_bytes=max_header_bytes,
                max_tensors=max_tensors,
            )
        except BaseException:
            self._stream.close()
            raise

    def _read_header(
        self,
        *,
        max_header_bytes: int,
        max_tensors: int,
    ) -> tuple[dict[str, TensorRecord], dict[str, str], int]:
        encoded_length = _read_exact(
            self._stream,
            _HEADER_LENGTH_BYTES,
            context="header length",
        )
        (header_length, ) = struct.unpack("<Q", encoded_length)
        if header_length == 0 or header_length > max_header_bytes:
            raise CheckpointFormatError(
                f"Safetensors header length {header_length} is outside the "
                f"allowed range 1..{max_header_bytes}.")
        encoded_header = _read_exact(
            self._stream,
            header_length,
            context="JSON header",
        )
        try:
            header = parse_json_object(
                encoded_header,
                source=f"Safetensors header {self.path}",
            )
        except ValueError as error:
            raise CheckpointFormatError(str(error)) from error

        raw_metadata = header.pop("__metadata__", {})
        if not isinstance(raw_metadata, dict) or any(not isinstance(key, str) or not isinstance(value, str)
                                                     for key, value in raw_metadata.items()):
            raise CheckpointFormatError("Safetensors `__metadata__` must map strings to strings.")
        if len(header) > max_tensors:
            raise CheckpointFormatError(
                f"Safetensors file declares {len(header)} tensors; the limit "
                f"is {max_tensors}.")
        records = {name: _parse_record(name, value) for name, value in header.items()}
        file_size = self.path.stat().st_size
        data_start = _HEADER_LENGTH_BYTES + header_length
        payload_size = file_size - data_start
        previous_end = 0
        for record in sorted(records.values(), key=lambda item: item.start):
            if record.start < previous_end:
                raise CheckpointFormatError(f"Tensor {record.name!r} overlaps an earlier tensor payload.")
            if record.end > payload_size:
                raise CheckpointFormatError(f"Tensor {record.name!r} ends beyond the checkpoint payload.")
            previous_end = record.end
        return records, dict(raw_metadata), data_start

    def keys(self) -> tuple[str, ...]:
        """Return tensor names in deterministic lexical order."""
        return tuple(sorted(self._records))

    def __contains__(self, name: object) -> bool:
        return name in self._records

    def __len__(self) -> int:
        return len(self._records)

    def record(self, name: str) -> TensorRecord:
        """Return validated metadata without reading tensor bytes."""
        try:
            return self._records[name]
        except KeyError as error:
            raise KeyError(f"Tensor {name!r} is not present in {self.path.name!r}.") from error

    def tensor_shape(self, name: str) -> tuple[int, ...]:
        """Return a declared tensor shape without reading its payload."""
        return self.record(name).shape

    def get_tensor(
        self,
        name: str,
        *,
        device: Any = "cpu",
        dtype: Any | None = None,
    ):
        """Read one tensor and optionally transfer or cast it."""
        if self._stream.closed:
            raise RuntimeError("Cannot read from a closed SafeTensorReader.")
        if sys.byteorder != "little":  # pragma: no cover - uncommon platform
            raise CheckpointFormatError("Safetensors loading on big-endian hosts is not supported.")
        torch = _torch()
        record = self.record(name)
        if record.number_of_elements == 0:
            tensor = torch.empty(record.shape, dtype=_torch_dtype(record.dtype))
        else:
            self._stream.seek(self._data_start + record.start)
            # Read straight into the tensor's own storage so the payload is
            # copied from the file once, not through intermediate buffers.
            tensor = torch.empty(record.shape, dtype=_torch_dtype(record.dtype))
            destination = (ctypes.c_char * record.number_of_bytes).from_address(tensor.data_ptr())
            found = self._stream.readinto(destination)
            if found != record.number_of_bytes:
                raise CheckpointFormatError(
                    f"Safetensors file ended while reading tensor {name!r}: "
                    f"expected {record.number_of_bytes} bytes, found {found}.")
        if dtype is not None or str(device) != "cpu":
            tensor = tensor.to(device=device, dtype=dtype or tensor.dtype)
        return tensor

    def state_dict(
        self,
        *,
        names: Iterable[str] | None = None,
        device: Any = "cpu",
        dtype: Any | None = None,
    ) -> dict[str, Any]:
        """Materialize selected tensors as a state-dict."""
        selected = self.keys() if names is None else tuple(names)
        if len(selected) != len(set(selected)):
            raise ValueError("Requested tensor names must be unique.")
        return {name: self.get_tensor(name, device=device, dtype=dtype) for name in selected}

    def close(self) -> None:
        self._stream.close()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def _materialize_tensor(tensor: Any) -> Any:
    torch = _torch()
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(
            f"Safetensors values must be torch.Tensor instances, found "
            f"{type(tensor).__name__}.")
    if tensor.layout != torch.strided:
        raise CheckpointFormatError(f"Safetensors cannot serialize tensor layout {tensor.layout}.")
    if tensor.device.type == "meta":
        raise CheckpointFormatError("Safetensors cannot serialize meta tensors.")
    materialized = tensor.detach().to(device="cpu").contiguous()
    if sys.byteorder != "little":  # pragma: no cover - uncommon platform
        raise CheckpointFormatError("Safetensors writing on big-endian hosts is not supported.")
    return materialized


def save_safetensors(
    tensors: Mapping[str, Any],
    path: str | Path,
    *,
    metadata: Mapping[str, str] | None = None,
) -> Path:
    """Atomically write a deterministic Safetensors checkpoint."""
    if not isinstance(tensors, Mapping) or not tensors:
        raise ValueError("`tensors` must be a non-empty mapping.")
    names = tuple(sorted(tensors))
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("Safetensors tensor names must be non-empty strings.")
    if "__metadata__" in tensors:
        raise ValueError("`__metadata__` is reserved by the Safetensors format.")
    normalized_metadata = dict(metadata or {})
    if any(not isinstance(key, str) or not isinstance(value, str)
           for key, value in normalized_metadata.items()):
        raise TypeError("Safetensors metadata must map strings to strings.")

    header: dict[str, Any] = {}
    offset = 0
    for name in names:
        tensor = _materialize_tensor(tensors[name])
        number_of_bytes = tensor.numel() * tensor.element_size()
        next_offset = offset + number_of_bytes
        header[name] = {
            "dtype": _safetensors_dtype(tensor.dtype),
            "shape": list(tensor.shape),
            "data_offsets": [offset, next_offset],
        }
        offset = next_offset
    if normalized_metadata:
        header["__metadata__"] = normalized_metadata

    encoded_header = json.dumps(
        header,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    padding = (-len(encoded_header)) % 8
    encoded_header += b" " * padding

    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w+b",
                dir=output_path.parent,
                prefix=f".{output_path.name}.",
                suffix=".tmp",
                delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(struct.pack("<Q", len(encoded_header)))
            stream.write(encoded_header)
            data_start = stream.tell()
            stream.truncate(data_start + offset)
            stream.flush()
            with mmap.mmap(
                    stream.fileno(),
                    length=0,
                    access=mmap.ACCESS_WRITE,
            ) as mapping:
                torch = _torch()
                for name in names:
                    tensor = _materialize_tensor(tensors[name])
                    record = header[name]
                    start, end = record["data_offsets"]
                    number_of_bytes = end - start
                    if number_of_bytes == 0:
                        continue
                    destination = torch.frombuffer(
                        mapping,
                        dtype=torch.uint8,
                        count=number_of_bytes,
                        offset=data_start + start,
                    )
                    source = tensor.reshape(-1).view(torch.uint8)
                    destination.copy_(source)
                mapping.flush()
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, output_path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return output_path


@dataclass(frozen=True)
class SafeTensorIndex:
    """Validated mapping from tensor names to Safetensors shard files."""

    path: Path
    weight_map: Mapping[str, str]
    metadata: Mapping[str, Any]

    @classmethod
    def from_file(cls, path: str | Path) -> SafeTensorIndex:
        # Preserve the logical index location. Hugging Face snapshots expose
        # the index and its shards as sibling symlinks into an extensionless
        # content-addressed blob store. Resolving the index first would make
        # relative shard names resolve beside the blob instead of beside the
        # snapshot links.
        index_path = Path(path).expanduser().absolute()
        if not index_path.is_file():
            raise FileNotFoundError(f"Safetensors index file was not found: {index_path}")
        try:
            value = parse_json_object(
                index_path.read_bytes(),
                source=f"Safetensors index {index_path}",
            )
        except (OSError, ValueError) as error:
            raise CheckpointFormatError(
                f"Could not parse Safetensors index {index_path}: {error}.") from error
        weight_map = value.get("weight_map")
        metadata = value.get("metadata", {})
        if (not isinstance(weight_map, dict) or not weight_map or
                any(not isinstance(name, str) or not name or not isinstance(shard, str) or not shard
                    for name, shard in weight_map.items())):
            raise CheckpointFormatError(
                "Safetensors index `weight_map` must map tensor names to "
                "non-empty shard paths.")
        if not isinstance(metadata, dict):
            raise CheckpointFormatError("Safetensors index `metadata` must be a JSON object.")
        for shard in set(weight_map.values()):
            shard_path = Path(shard)
            if (shard_path.is_absolute() or ".." in shard_path.parts or len(shard_path.parts) != 1):
                raise CheckpointFormatError(f"Unsafe Safetensors shard path {shard!r}.")
            logical_shard = index_path.parent / shard_path
            if not logical_shard.is_file():
                raise FileNotFoundError(f"Safetensors shard was not found: {logical_shard}")
        return cls(
            path=index_path,
            weight_map=dict(weight_map),
            metadata=dict(metadata),
        )

    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(self.weight_map))

    def shard_path(self, tensor_name: str) -> Path:
        try:
            shard = self.weight_map[tensor_name]
        except KeyError as error:
            raise KeyError(f"Tensor {tensor_name!r} is not present in {self.path.name!r}.") from error
        return self.path.parent / shard


class ShardedSafeTensorReader(AbstractContextManager["ShardedSafeTensorReader"]):
    """Expose a sharded Safetensors index through the lazy tensor protocol.

    Shards are opened only when their first tensor is requested and
    remain open for the lifetime of the reader.  This lets strict
    checkpoint adapters inspect multi-gigabyte models without first
    materializing the complete state dictionary.
    """

    def __init__(
        self,
        index_path: str | Path,
        *,
        max_header_bytes: int = _DEFAULT_MAX_HEADER_BYTES,
        max_tensors_per_shard: int = _DEFAULT_MAX_TENSORS,
    ) -> None:
        self.index = SafeTensorIndex.from_file(index_path)
        if max_header_bytes <= 0 or max_tensors_per_shard <= 0:
            raise ValueError("Safetensors parser limits must be positive.")
        self._max_header_bytes = max_header_bytes
        self._max_tensors_per_shard = max_tensors_per_shard
        self._readers: dict[Path, SafeTensorReader] = {}
        self._closed = False

    def keys(self) -> tuple[str, ...]:
        """Return indexed tensor names in deterministic lexical order."""
        return self.index.keys()

    def __contains__(self, name: object) -> bool:
        return name in self.index.weight_map

    def __len__(self) -> int:
        return len(self.index.weight_map)

    def _reader(self, path: Path) -> SafeTensorReader:
        if self._closed:
            raise RuntimeError("Cannot read from a closed ShardedSafeTensorReader.")
        reader = self._readers.get(path)
        if reader is None:
            reader = SafeTensorReader(
                path,
                max_header_bytes=self._max_header_bytes,
                max_tensors=self._max_tensors_per_shard,
            )
            self._readers[path] = reader
        return reader

    def get_tensor(
        self,
        name: str,
        *,
        device: Any = "cpu",
        dtype: Any | None = None,
    ):
        """Materialize one indexed tensor from its declared shard."""
        shard_path = self.index.shard_path(name)
        reader = self._reader(shard_path)
        if name not in reader:
            raise CheckpointFormatError(
                f"Index maps tensor {name!r} to {shard_path.name!r}, but the "
                "shard does not contain it.")
        return reader.get_tensor(name, device=device, dtype=dtype)

    def tensor_shape(self, name: str) -> tuple[int, ...]:
        """Return an indexed tensor shape while reading only its shard
        header."""
        shard_path = self.index.shard_path(name)
        reader = self._reader(shard_path)
        if name not in reader:
            raise CheckpointFormatError(
                f"Index maps tensor {name!r} to {shard_path.name!r}, but the "
                "shard does not contain it.")
        return reader.tensor_shape(name)

    def close(self) -> None:
        """Close every shard opened by this reader."""
        if self._closed:
            return
        self._closed = True
        errors: list[BaseException] = []
        for reader in self._readers.values():
            try:
                reader.close()
            except BaseException as error:  # pragma: no cover - defensive
                errors.append(error)
        self._readers.clear()
        if errors:
            raise errors[0]

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def load_sharded_safetensors(
    index_path: str | Path,
    *,
    names: Iterable[str] | None = None,
    device: Any = "cpu",
    dtype: Any | None = None,
) -> dict[str, Any]:
    """Load selected tensors while opening each referenced shard only once."""
    index = SafeTensorIndex.from_file(index_path)
    selected = index.keys() if names is None else tuple(names)
    if len(selected) != len(set(selected)):
        raise ValueError("Requested tensor names must be unique.")
    unknown = sorted(set(selected) - set(index.weight_map))
    if unknown:
        raise KeyError(f"Unknown tensors in shard request: {unknown!r}.")
    loaded: dict[str, Any] = {}
    with ExitStack() as stack:
        readers: dict[Path, SafeTensorReader] = {}
        for name in selected:
            shard_path = index.shard_path(name)
            reader = readers.get(shard_path)
            if reader is None:
                reader = stack.enter_context(SafeTensorReader(shard_path))
                readers[shard_path] = reader
            if name not in reader:
                raise CheckpointFormatError(
                    f"Index maps tensor {name!r} to {shard_path.name!r}, but "
                    "the shard does not contain it.")
            loaded[name] = reader.get_tensor(
                name,
                device=device,
                dtype=dtype,
            )
    return loaded
