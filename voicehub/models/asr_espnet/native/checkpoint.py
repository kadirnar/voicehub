"""Strict one-time conversion of the audited ESPnet pickle artifacts."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicehub.checkpointing import CheckpointAdapter, CopyTensor, SafeTensorReader, TensorPlan, save_safetensors
from voicehub.hub import write_json_file
from voicehub.models.asr_espnet.native.configuration import ESPnetLibriSpeechTransformerConfig
from voicehub.models.asr_espnet.native.metadata import (
    ESPNET_ASR_SHA256,
    ESPNET_ASR_SIZE,
    ESPNET_ASR_STATE_VALUES,
    ESPNET_ASR_TENSOR_COUNT,
    ESPNET_ASR_TENSOR_FINGERPRINT,
    ESPNET_CHECKPOINT_LICENSE,
    ESPNET_CONFIG_SHA256,
    ESPNET_CONFIG_SIZE,
    ESPNET_LM_NATIVE_TENSOR_FINGERPRINT,
    ESPNET_LM_SHA256,
    ESPNET_LM_SIZE,
    ESPNET_LM_SOURCE_TENSOR_FINGERPRINT,
    ESPNET_LM_STATE_VALUES,
    ESPNET_LM_TENSOR_COUNT,
    ESPNET_REVISION,
    ESPNET_SOURCE_REVISION,
    ESPNET_TOKEN_LIST_SHA256,
    ESPNET_TOKEN_LIST_SIZE,
    ESPNET_TOKENIZER_SHA256,
    ESPNET_TOKENIZER_SIZE,
)

NATIVE_ESPNET_FORMAT = "voicehub-espnet-librispeech-transformer-e18-v1"
NATIVE_ESPNET_FILENAME = "model.safetensors"
NATIVE_ESPNET_LM_FILENAME = "language_model.safetensors"
NATIVE_ESPNET_TOKENIZER = "tokenizer.model"
NATIVE_ESPNET_TOKENS = "tokens.txt"
_ConfigLike = ESPnetLibriSpeechTransformerConfig | Mapping[str, Any] | None


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_dtype(tensor: Any, *, legacy: bool) -> str:
    if legacy:
        return {
            "torch.float32": "FloatStorage",
            "torch.float64": "DoubleStorage",
            "torch.float16": "HalfStorage",
            "torch.int64": "LongStorage",
            "torch.int32": "IntStorage",
        }.get(str(tensor.dtype), str(tensor.dtype))
    return {
        "torch.bfloat16": "BF16",
        "torch.bool": "BOOL",
        "torch.float16": "F16",
        "torch.float32": "F32",
        "torch.float64": "F64",
        "torch.int16": "I16",
        "torch.int32": "I32",
        "torch.int64": "I64",
        "torch.int8": "I8",
        "torch.uint8": "U8",
    }.get(str(tensor.dtype), str(tensor.dtype))


def tensor_inventory_fingerprint(
    tensors: Mapping[str, Any],
    *,
    legacy_storage_names: bool = True,
) -> str:
    """Hash insertion-ordered names, dtypes, and shapes without values."""
    rows = [(
        f"{name}|{_portable_dtype(tensor, legacy=legacy_storage_names)}|"
        f"{'x'.join(str(item) for item in tensor.shape)}") for name, tensor in tensors.items()]
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def native_espnet_asr_tensor_shapes(config: _ConfigLike = None, ) -> dict[str, tuple[int, ...]]:
    import torch

    from voicehub.models.asr_espnet.native.modeling import ESPnetLibriSpeechTransformerForASR

    resolved = ESPnetLibriSpeechTransformerConfig.coerce(config)
    with torch.device("meta"):
        model = ESPnetLibriSpeechTransformerForASR(resolved)
    return {name: tuple(tensor.shape) for name, tensor in model.state_dict().items()}


def native_espnet_lm_tensor_shapes(config: _ConfigLike = None, ) -> dict[str, tuple[int, ...]]:
    import torch

    from voicehub.models.asr_espnet.native.modeling import ESPnetSequentialRNNLanguageModel

    resolved = ESPnetLibriSpeechTransformerConfig.coerce(config)
    with torch.device("meta"):
        model = ESPnetSequentialRNNLanguageModel(resolved)
    return {name: tuple(tensor.shape) for name, tensor in model.state_dict().items()}


class ESPnetASRSafeTensorsCheckpointAdapter(CheckpointAdapter):
    """Identity-map a complete VoiceHub-native acoustic checkpoint."""

    architecture_id = "espnet-librispeech-transformer-e18"
    adapter_id = "voicehub-espnet-transformer-safetensors"
    adapter_version = "1"

    def probe(
        self,
        files: tuple[Path, ...],
        config: Mapping[str, Any],
    ) -> bool:
        try:
            ESPnetLibriSpeechTransformerConfig.from_dict(config)
        except (TypeError, ValueError):
            return False
        return any(path.name == NATIVE_ESPNET_FILENAME for path in files)

    def tensor_plan(self, config: Mapping[str, Any]) -> TensorPlan:
        shapes = native_espnet_asr_tensor_shapes(config)
        return TensorPlan(rules=tuple(CopyTensor(name, name) for name in shapes))


def _verify_file(
    path: str | Path,
    *,
    label: str,
    expected_sha256: str | None,
    expected_size: int | None,
) -> tuple[Path, str]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"ESPnet {label} was not found: {resolved}.")
    if expected_size is not None and resolved.stat().st_size != expected_size:
        raise ValueError(
            f"ESPnet {label} size mismatch: expected {expected_size}, "
            f"found {resolved.stat().st_size}.")
    digest = file_sha256(resolved)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"ESPnet {label} SHA-256 mismatch: expected "
                         f"{expected_sha256}, found {digest}.")
    return resolved, digest


def _restricted_state_dict(
    path: Path,
    *,
    label: str,
    trust_pickle_checkpoint: bool,
) -> Mapping[str, Any]:
    if trust_pickle_checkpoint is not True:
        raise ValueError(
            "The published ESPnet checkpoints use Python's pickle container. "
            "Review the pinned artifact digests and pass "
            "`trust_pickle_checkpoint=True` for one-time conversion. "
            "Steady-state inference and training use Safetensors only.")
    import torch

    try:
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError(f"Could not read the restricted ESPnet {label} tensor state.") from error
    if not isinstance(payload, Mapping) or not payload:
        raise TypeError(f"ESPnet {label} must be a non-empty state dictionary.")
    if any(not isinstance(name, str) or not isinstance(tensor, torch.Tensor)
           for name, tensor in payload.items()):
        raise TypeError(f"ESPnet {label} must map string names to tensors only.")
    return payload


def extract_espnet_token_list(config_yaml: str | Path) -> tuple[str, ...]:
    """Extract only ``token_list`` from the pinned YAML without PyYAML."""
    payload = Path(config_yaml).read_bytes()
    if len(payload) > 4 * 1024 * 1024:
        raise ValueError("ESPnet config YAML exceeds the 4 MiB safety limit.")
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("ESPnet config YAML is not valid UTF-8.") from error
    collecting = False
    tokens: list[str] = []
    for line in lines:
        if not collecting:
            if line.strip() == "token_list:" and not line.startswith((" ", "\t")):
                collecting = True
            continue
        if not line.startswith("- "):
            break
        scalar = line[2:]
        if scalar.startswith('"'):
            try:
                value = json.loads(scalar)
            except json.JSONDecodeError as error:
                raise ValueError("ESPnet token list contains an invalid quoted scalar.") from error
        elif scalar.startswith("'") and scalar.endswith("'"):
            value = scalar[1:-1].replace("''", "'")
        else:
            value = scalar
        if not isinstance(value, str) or not value:
            raise ValueError("ESPnet token list contains an invalid entry.")
        tokens.append(value)
    if not tokens:
        raise ValueError("ESPnet config YAML contains no top-level token list.")
    if len(tokens) != len(set(tokens)):
        raise ValueError("ESPnet config YAML contains duplicate tokens.")
    return tuple(tokens)


def _token_payload(tokens: tuple[str, ...]) -> bytes:
    return ("\n".join(tokens) + "\n").encode("utf-8")


def _validate_shapes(
    state: Mapping[str, Any],
    expected_shapes: Mapping[str, tuple[int, ...]],
    *,
    label: str,
) -> None:
    expected = set(expected_shapes)
    actual = set(state)
    if actual != expected:
        raise ValueError(
            f"ESPnet {label} tensor namespace mismatch "
            f"(missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}).")
    mismatches = {
        name: (tuple(state[name].shape), expected_shapes[name])
        for name in expected_shapes if tuple(state[name].shape) != expected_shapes[name]
    }
    if mismatches:
        raise ValueError(f"ESPnet {label} tensor shape mismatch: {mismatches}.")


def _validate_official_asr_state(state: Mapping[str, Any]) -> str:
    _validate_shapes(
        state,
        native_espnet_asr_tensor_shapes(),
        label="ASR",
    )
    if len(state) != ESPNET_ASR_TENSOR_COUNT:
        raise ValueError("ESPnet ASR tensor count mismatch.")
    if sum(tensor.numel() for tensor in state.values()) != ESPNET_ASR_STATE_VALUES:
        raise ValueError("ESPnet ASR stored-value count mismatch.")
    fingerprint = tensor_inventory_fingerprint(state)
    if fingerprint != ESPNET_ASR_TENSOR_FINGERPRINT:
        raise ValueError("ESPnet ASR tensor inventory fingerprint mismatch.")
    return fingerprint


def _convert_language_model_state(
    state: Mapping[str, Any],
    *,
    official: bool,
    config: ESPnetLibriSpeechTransformerConfig,
) -> tuple[dict[str, Any], str]:
    source_fingerprint = tensor_inventory_fingerprint(state)
    if official:
        if len(state) != ESPNET_LM_TENSOR_COUNT:
            raise ValueError("ESPnet language-model tensor count mismatch.")
        if (sum(tensor.numel() for tensor in state.values()) != ESPNET_LM_STATE_VALUES):
            raise ValueError("ESPnet language-model stored-value count mismatch.")
        if source_fingerprint != ESPNET_LM_SOURCE_TENSOR_FINGERPRINT:
            raise ValueError("ESPnet language-model tensor inventory fingerprint mismatch.")
    if any(not name.startswith("lm.") for name in state):
        raise ValueError("ESPnet language-model tensors must use the `lm.` source prefix.")
    native = {name[3:]: tensor for name, tensor in state.items()}
    _validate_shapes(
        native,
        native_espnet_lm_tensor_shapes(config),
        label="language-model",
    )
    if official:
        native_fingerprint = tensor_inventory_fingerprint(native)
        if native_fingerprint != ESPNET_LM_NATIVE_TENSOR_FINGERPRINT:
            raise RuntimeError("Internal ESPnet language-model mapping drifted.")
    return native, source_fingerprint


def convert_espnet_librispeech_checkpoints(
    *,
    asr_checkpoint: str | Path,
    language_model_checkpoint: str | Path,
    tokenizer_model: str | Path,
    config_yaml: str | Path,
    destination: str | Path,
    config: _ConfigLike = None,
    trust_pickle_checkpoint: bool = False,
    expected_asr_sha256: str | None = None,
    expected_asr_size: int | None = None,
    expected_lm_sha256: str | None = None,
    expected_lm_size: int | None = None,
    expected_tokenizer_sha256: str | None = None,
    expected_tokenizer_size: int | None = None,
    expected_config_sha256: str | None = None,
    expected_config_size: int | None = None,
) -> Path:
    """Convert the two reviewed pickle states into strict native artifacts."""
    if trust_pickle_checkpoint is not True:
        raise ValueError(
            "The published ESPnet checkpoints use Python's pickle container. "
            "Review the pinned artifact digests and pass "
            "`trust_pickle_checkpoint=True` for one-time conversion. "
            "Steady-state inference and training use Safetensors only.")
    resolved = ESPnetLibriSpeechTransformerConfig.coerce(config)
    official = resolved.variant == "librispeech-transformer-e18"
    if official:
        supplied_expectations = {
            "expected_asr_sha256": (
                expected_asr_sha256,
                ESPNET_ASR_SHA256,
            ),
            "expected_asr_size": (
                expected_asr_size,
                ESPNET_ASR_SIZE,
            ),
            "expected_lm_sha256": (
                expected_lm_sha256,
                ESPNET_LM_SHA256,
            ),
            "expected_lm_size": (
                expected_lm_size,
                ESPNET_LM_SIZE,
            ),
            "expected_tokenizer_sha256": (
                expected_tokenizer_sha256,
                ESPNET_TOKENIZER_SHA256,
            ),
            "expected_tokenizer_size": (
                expected_tokenizer_size,
                ESPNET_TOKENIZER_SIZE,
            ),
            "expected_config_sha256": (
                expected_config_sha256,
                ESPNET_CONFIG_SHA256,
            ),
            "expected_config_size": (
                expected_config_size,
                ESPNET_CONFIG_SIZE,
            ),
        }
        conflicts = [
            name for name, (supplied, pinned) in supplied_expectations.items()
            if supplied is not None and supplied != pinned
        ]
        if conflicts:
            raise ValueError(
                "Official ESPnet conversion expectations are immutable: " + ", ".join(conflicts) + ".")
        expected_asr_sha256 = ESPNET_ASR_SHA256
        expected_asr_size = ESPNET_ASR_SIZE
        expected_lm_sha256 = ESPNET_LM_SHA256
        expected_lm_size = ESPNET_LM_SIZE
        expected_tokenizer_sha256 = ESPNET_TOKENIZER_SHA256
        expected_tokenizer_size = ESPNET_TOKENIZER_SIZE
        expected_config_sha256 = ESPNET_CONFIG_SHA256
        expected_config_size = ESPNET_CONFIG_SIZE
    files = {}
    digests = {}
    for key, source, label, sha, size in (
        (
            "asr",
            asr_checkpoint,
            "ASR checkpoint",
            expected_asr_sha256,
            expected_asr_size,
        ),
        (
            "lm",
            language_model_checkpoint,
            "language-model checkpoint",
            expected_lm_sha256,
            expected_lm_size,
        ),
        (
            "tokenizer",
            tokenizer_model,
            "tokenizer",
            expected_tokenizer_sha256,
            expected_tokenizer_size,
        ),
        (
            "config",
            config_yaml,
            "config YAML",
            expected_config_sha256,
            expected_config_size,
        ),
    ):
        files[key], digests[key] = _verify_file(
            source,
            label=label,
            expected_sha256=sha,
            expected_size=size,
        )
    tokens = extract_espnet_token_list(files["config"])
    token_payload = _token_payload(tokens)
    if len(tokens) != resolved.vocabulary_size:
        raise ValueError("ESPnet config vocabulary does not match the graph.")
    if official and (len(token_payload) != ESPNET_TOKEN_LIST_SIZE or
                     hashlib.sha256(token_payload).hexdigest() != ESPNET_TOKEN_LIST_SHA256):
        raise ValueError("ESPnet extracted token-list fingerprint mismatch.")

    output = Path(destination).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
            prefix="voicehub-espnet-convert-",
            dir=output.parent,
    ) as temporary:
        staging = Path(temporary)
        asr_state = _restricted_state_dict(
            files["asr"],
            label="ASR",
            trust_pickle_checkpoint=trust_pickle_checkpoint,
        )
        if official:
            asr_fingerprint = _validate_official_asr_state(asr_state)
        else:
            _validate_shapes(
                asr_state,
                native_espnet_asr_tensor_shapes(resolved),
                label="ASR",
            )
            asr_fingerprint = tensor_inventory_fingerprint(asr_state)
        save_safetensors(
            {
                name: tensor.detach().cpu().contiguous()
                for name, tensor in asr_state.items()
            },
            staging / NATIVE_ESPNET_FILENAME,
            metadata={
                "architecture": "espnet-librispeech-transformer-e18",
                "format": NATIVE_ESPNET_FORMAT,
                "model_license": ESPNET_CHECKPOINT_LICENSE,
                "source_revision": ESPNET_REVISION,
                "source_tensor_fingerprint": asr_fingerprint,
            },
        )
        del asr_state

        lm_state = _restricted_state_dict(
            files["lm"],
            label="language model",
            trust_pickle_checkpoint=trust_pickle_checkpoint,
        )
        native_lm, lm_fingerprint = _convert_language_model_state(
            lm_state,
            official=official,
            config=resolved,
        )
        save_safetensors(
            {
                name: tensor.detach().cpu().contiguous()
                for name, tensor in native_lm.items()
            },
            staging / NATIVE_ESPNET_LM_FILENAME,
            metadata={
                "architecture": "espnet-sequential-rnn-lm",
                "format": NATIVE_ESPNET_FORMAT,
                "model_license": ESPNET_CHECKPOINT_LICENSE,
                "source_revision": ESPNET_REVISION,
                "source_tensor_fingerprint": lm_fingerprint,
            },
        )
        del lm_state, native_lm

        shutil.copy2(
            files["tokenizer"],
            staging / NATIVE_ESPNET_TOKENIZER,
        )
        (staging / NATIVE_ESPNET_TOKENS).write_bytes(token_payload)
        values = resolved.to_dict()
        values.update({
            'architectures': [
                "ESPnetASRForSpeechRecognition",
                "ESPnetLibriSpeechTransformerForASR",
            ],
            "checkpoint_format":
            NATIVE_ESPNET_FORMAT,
            "model_type":
            "asr_espnet",
            "source_artifact_revision":
            ESPNET_REVISION,
            "source_asr_sha256":
            digests["asr"],
            "source_asr_tensor_fingerprint":
            asr_fingerprint,
            "source_config_sha256":
            digests["config"],
            "source_lm_sha256":
            digests["lm"],
            "source_lm_tensor_fingerprint":
            lm_fingerprint,
            "source_revision":
            ESPNET_SOURCE_REVISION,
            "source_tokenizer_sha256":
            digests["tokenizer"],
            "voicehub_provider":
            "asr_espnet",
        })
        write_json_file(staging / "config.json", values)
        for name in (
                NATIVE_ESPNET_FILENAME,
                NATIVE_ESPNET_LM_FILENAME,
                NATIVE_ESPNET_TOKENIZER,
                NATIVE_ESPNET_TOKENS,
                "config.json",
        ):
            (staging / name).replace(output / name)
    return output


def official_espnet_conversion_kwargs() -> dict[str, str | int]:
    """Return all immutable validation arguments for the public release."""
    return {
        "expected_asr_sha256": ESPNET_ASR_SHA256,
        "expected_asr_size": ESPNET_ASR_SIZE,
        "expected_lm_sha256": ESPNET_LM_SHA256,
        "expected_lm_size": ESPNET_LM_SIZE,
        "expected_tokenizer_sha256": ESPNET_TOKENIZER_SHA256,
        "expected_tokenizer_size": ESPNET_TOKENIZER_SIZE,
        "expected_config_sha256": ESPNET_CONFIG_SHA256,
        "expected_config_size": ESPNET_CONFIG_SIZE,
    }


def _validate_safe_inventory(
    path: Path,
    expected: Mapping[str, tuple[int, ...]],
    *,
    label: str,
) -> None:
    with SafeTensorReader(path) as reader:
        names = set(reader.keys())
        if names != set(expected):
            raise ValueError(
                f"ESPnet native {label} namespace mismatch "
                f"(missing={sorted(set(expected) - names)}, "
                f"unexpected={sorted(names - set(expected))}).")
        mismatches = {
            name: (reader.tensor_shape(name), expected[name])
            for name in expected if reader.tensor_shape(name) != expected[name]
        }
        if mismatches:
            raise ValueError(f"ESPnet native {label} tensor shape mismatch: {mismatches}.")


def load_native_espnet_models(
    *,
    checkpoint: str | Path,
    language_model_checkpoint: str | Path,
    config: ESPnetLibriSpeechTransformerConfig | Mapping[str, Any],
    device: str = "cpu",
    dtype: Any = None,
):
    """Allocate and strictly reload fresh native ASR and LM graphs."""
    import torch

    from voicehub.models.asr_espnet.native.modeling import (
        ESPnetLibriSpeechTransformerForASR,
        ESPnetSequentialRNNLanguageModel,
    )

    resolved = ESPnetLibriSpeechTransformerConfig.coerce(config)
    asr_path = Path(checkpoint).expanduser().resolve()
    lm_path = Path(language_model_checkpoint).expanduser().resolve()
    _validate_safe_inventory(
        asr_path,
        native_espnet_asr_tensor_shapes(resolved),
        label="ASR",
    )
    _validate_safe_inventory(
        lm_path,
        native_espnet_lm_tensor_shapes(resolved),
        label="language-model",
    )
    model = ESPnetLibriSpeechTransformerForASR(resolved)
    adapter = ESPnetASRSafeTensorsCheckpointAdapter()
    with SafeTensorReader(asr_path) as reader:
        declared = reader.metadata.get("format")
        if declared != NATIVE_ESPNET_FORMAT:
            raise ValueError(f"ESPnet Safetensors declares unsupported format {declared!r}.")
        adapter.load_streaming(
            model,
            reader,
            resolved.to_dict(),
            strict=True,
        )
    language_model = ESPnetSequentialRNNLanguageModel(resolved)
    with SafeTensorReader(lm_path) as reader:
        declared = reader.metadata.get("format")
        if declared != NATIVE_ESPNET_FORMAT:
            raise ValueError(
                "ESPnet language-model Safetensors declares unsupported "
                f"format {declared!r}.")
        state = {name: reader.get_tensor(name) for name in reader.keys()}
    language_model.load_state_dict(state, strict=True)
    target_dtype = torch.float32 if dtype is None else dtype
    return (
        model.to(device=device, dtype=target_dtype),
        language_model.to(device=device, dtype=target_dtype),
    )


__all__ = [
    "ESPnetASRSafeTensorsCheckpointAdapter",
    "NATIVE_ESPNET_FILENAME",
    "NATIVE_ESPNET_FORMAT",
    "NATIVE_ESPNET_LM_FILENAME",
    "NATIVE_ESPNET_TOKENIZER",
    "NATIVE_ESPNET_TOKENS",
    "convert_espnet_librispeech_checkpoints",
    "extract_espnet_token_list",
    "file_sha256",
    "load_native_espnet_models",
    "native_espnet_asr_tensor_shapes",
    "native_espnet_lm_tensor_shapes",
    "official_espnet_conversion_kwargs",
    "tensor_inventory_fingerprint",
]
