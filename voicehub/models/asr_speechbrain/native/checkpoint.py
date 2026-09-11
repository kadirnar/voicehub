"""Strict conversion of SpeechBrain CRDNN ASR releases to Safetensors."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicehub.checkpointing import CheckpointAdapter, CopyTensor, SafeTensorReader, TensorPlan, save_safetensors
from voicehub.hub import write_json_file
from voicehub.models.asr_speechbrain.native.configuration import SpeechBrainCRDNNASRConfig
from voicehub.models.asr_speechbrain.native.metadata import (
    SPEECHBRAIN_ASR_HPARAMS_SHA256,
    SPEECHBRAIN_ASR_LM_SHA256,
    SPEECHBRAIN_ASR_LM_TENSOR_FINGERPRINT,
    SPEECHBRAIN_ASR_MODEL_SHA256,
    SPEECHBRAIN_ASR_NORMALIZER_SHA256,
    SPEECHBRAIN_ASR_NORMALIZER_TENSOR_FINGERPRINT,
    SPEECHBRAIN_ASR_REVISION,
    SPEECHBRAIN_ASR_SOURCE_REVISION,
    SPEECHBRAIN_ASR_TENSOR_FINGERPRINT,
    SPEECHBRAIN_ASR_TOKENIZER_SHA256,
)

NATIVE_SPEECHBRAIN_ASR_FORMAT = "voicehub-speechbrain-crdnn-asr-v1"
NATIVE_SPEECHBRAIN_ASR_FILENAME = "model.safetensors"
NATIVE_SPEECHBRAIN_ASR_TOKENIZER = "tokenizer.model"
_ConfigInput = SpeechBrainCRDNNASRConfig | Mapping[str, Any] | None


def _asr_mapping(config: SpeechBrainCRDNNASRConfig) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for block in range(len(config.cnn_channels)):
        for convolution in ("conv_1", "conv_2"):
            for suffix in ("weight", "bias"):
                mapping[
                    f"0.CNN.block_{block}.{convolution}.conv.{suffix}"] = f"encoder.cnn_blocks.{block}.{convolution}.{suffix}"
        for normalization in ("norm_1", "norm_2"):
            for suffix in ("weight", "bias"):
                mapping[
                    f"0.CNN.block_{block}.{normalization}.norm.{suffix}"] = f"encoder.cnn_blocks.{block}.{normalization}.{suffix}"
    for layer in range(config.rnn_layers):
        directions = ("", "_reverse") if config.rnn_bidirectional else ("", )
        for direction in directions:
            for prefix in (
                    "weight_ih",
                    "weight_hh",
                    "bias_ih",
                    "bias_hh",
            ):
                source = f"{prefix}_l{layer}{direction}"
                mapping[f"0.RNN.rnn.{source}"] = f"encoder.rnn.{source}"
    for block in range(config.dnn_blocks):
        for suffix in ("weight", "bias"):
            mapping[f"0.DNN.block_{block}.linear.w.{suffix}"] = f"encoder.dnn_blocks.{block}.linear.{suffix}"
        for suffix in (
                "weight",
                "bias",
                "running_mean",
                "running_var",
                "num_batches_tracked",
        ):
            mapping[f"0.DNN.block_{block}.norm.norm.{suffix}"] = f"encoder.dnn_blocks.{block}.norm.{suffix}"
    mapping.update({
        "1.Embedding.weight": "embedding.weight",
        "2.proj.weight": "decoder.output_projection.weight",
        "2.proj.bias": "decoder.output_projection.bias",
        "2.attn.mlp_enc.weight": "decoder.attention.encoder_projection.weight",
        "2.attn.mlp_enc.bias": "decoder.attention.encoder_projection.bias",
        "2.attn.mlp_dec.weight": "decoder.attention.decoder_projection.weight",
        "2.attn.mlp_dec.bias": "decoder.attention.decoder_projection.bias",
        "2.attn.mlp_attn.weight": "decoder.attention.score_projection.weight",
        "2.attn.conv_loc.weight": "decoder.attention.location_convolution.weight",
        "2.attn.mlp_loc.weight": "decoder.attention.location_projection.weight",
        "2.attn.mlp_loc.bias": "decoder.attention.location_projection.bias",
        "2.attn.mlp_out.weight": "decoder.attention.output_projection.weight",
        "2.attn.mlp_out.bias": "decoder.attention.output_projection.bias",
        "2.rnn.rnn_cells.0.weight_ih": "decoder.rnn_cells.0.weight_ih",
        "2.rnn.rnn_cells.0.weight_hh": "decoder.rnn_cells.0.weight_hh",
        "2.rnn.rnn_cells.0.bias_ih": "decoder.rnn_cells.0.bias_ih",
        "2.rnn.rnn_cells.0.bias_hh": "decoder.rnn_cells.0.bias_hh",
        "3.w.weight": "ctc_linear.weight",
        "3.w.bias": "ctc_linear.bias",
        "4.w.weight": "sequence_linear.weight",
        "4.w.bias": "sequence_linear.bias",
    })
    return mapping


def _lm_mapping(config: SpeechBrainCRDNNASRConfig) -> dict[str, str]:
    mapping = {
        "embedding.Embedding.weight": "language_model.embedding.weight",
        "out.w.weight": "language_model.output.weight",
        "out.w.bias": "language_model.output.bias",
    }
    for layer in range(config.lm_rnn_layers):
        for prefix in (
                "weight_ih",
                "weight_hh",
                "bias_ih",
                "bias_hh",
        ):
            source = f"{prefix}_l{layer}"
            mapping[f"rnn.rnn.{source}"] = f"language_model.rnn.{source}"
    # The released RNNLM has one DNN block. Its upstream sequential container
    # deliberately reuses the unindexed names ``linear`` and ``norm``.
    if config.lm_dnn_blocks != 1:
        raise ValueError("Official SpeechBrain RNNLM conversion requires one DNN block.")
    mapping.update({
        "dnn.linear.w.weight": "language_model.dnn_blocks.0.0.weight",
        "dnn.linear.w.bias": "language_model.dnn_blocks.0.0.bias",
        "dnn.norm.norm.weight": "language_model.dnn_blocks.0.1.weight",
        "dnn.norm.norm.bias": "language_model.dnn_blocks.0.1.bias",
    })
    return mapping


def speechbrain_asr_source_tensor_mapping(config: _ConfigInput = None) -> dict[str, str]:
    """Return the complete acoustic-checkpoint namespace mapping."""
    resolved = SpeechBrainCRDNNASRConfig.coerce(config or SpeechBrainCRDNNASRConfig(), )
    return _asr_mapping(resolved)


def speechbrain_lm_source_tensor_mapping(config: _ConfigInput = None) -> dict[str, str]:
    """Return the complete RNNLM-checkpoint namespace mapping."""
    resolved = SpeechBrainCRDNNASRConfig.coerce(config or SpeechBrainCRDNNASRConfig(), )
    return _lm_mapping(resolved)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_digest(
    path: Path,
    expected: str | None,
    *,
    label: str,
) -> str:
    actual = _file_sha256(path)
    if expected is not None:
        if (not isinstance(expected, str) or len(expected) != 64 or
                any(character not in "0123456789abcdefABCDEF" for character in expected)):
            raise ValueError(f"Expected {label} SHA-256 must be a hex digest.")
        if actual.lower() != expected.lower():
            raise ValueError(
                f"SpeechBrain ASR {label} SHA-256 mismatch: expected "
                f"{expected}, found {actual}.")
    return actual


def tensor_inventory_fingerprint(tensors: Mapping[str, Any]) -> str:
    """Hash tensor names, dtypes, and shapes without reading values."""
    rows = []
    for name, tensor in sorted(tensors.items()):
        dtype = {
            "torch.float32": "F32",
            "torch.float64": "F64",
            "torch.float16": "F16",
            "torch.bfloat16": "BF16",
            "torch.int64": "I64",
        }.get(str(tensor.dtype), str(tensor.dtype))
        shape = "x".join(str(item) for item in tensor.shape)
        rows.append(f"{name}|{dtype}|{shape}")
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def native_speechbrain_asr_tensor_shapes(config: _ConfigInput = None) -> dict[str, tuple[int, ...]]:
    """Build expected native shapes without allocating checkpoint storage."""
    import torch

    from voicehub.models.asr_speechbrain.native.modeling import SpeechBrainCRDNNForASR

    resolved = SpeechBrainCRDNNASRConfig.coerce(config or SpeechBrainCRDNNASRConfig(), )
    with torch.device("meta"):
        model = SpeechBrainCRDNNForASR(resolved)
    return {name: tuple(tensor.shape) for name, tensor in model.state_dict().items()}


def _restricted_state_dict(
    path: Path,
    *,
    label: str,
) -> Mapping[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or not payload:
        raise TypeError(f"SpeechBrain {label} must contain a non-empty state dict.")
    if any(not isinstance(name, str) or not isinstance(tensor, torch.Tensor)
           for name, tensor in payload.items()):
        raise TypeError(f"SpeechBrain {label} entries must map names to tensors.")
    return payload


def _normalizer_tensors(path: Path) -> Mapping[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    expected = {
        "count",
        "glob_mean",
        "glob_std",
        "spk_dict_mean",
        "spk_dict_std",
        "spk_dict_count",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise ValueError("SpeechBrain normalizer checkpoint namespace mismatch.")
    if any(payload[name] != {} for name in (
            "spk_dict_mean",
            "spk_dict_std",
            "spk_dict_count",
    )):
        raise ValueError("The LibriSpeech global normalizer must not carry speaker state.")
    count = payload["count"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("SpeechBrain normalizer count must be a positive integer.")
    mean = payload["glob_mean"]
    std = payload["glob_std"]
    if (not isinstance(mean, torch.Tensor) or not isinstance(std, torch.Tensor)):
        raise TypeError("SpeechBrain global normalizer statistics must be tensors.")
    return {
        "count": torch.tensor(count, dtype=torch.long),
        "glob_mean": mean,
        "glob_std": std,
    }


class SpeechBrainASRSafeTensorsCheckpointAdapter(CheckpointAdapter):
    """Identity-load a complete VoiceHub-native CRDNN ASR artifact."""

    architecture_id = "speechbrain-crdnn-asr"
    adapter_id = "voicehub-speechbrain-crdnn-asr-safetensors"
    adapter_version = "1"

    def probe(
        self,
        files: tuple[Path, ...],
        config: Mapping[str, Any],
    ) -> bool:
        try:
            SpeechBrainCRDNNASRConfig.from_dict(config)
        except (TypeError, ValueError):
            return False
        return any(path.suffix == ".safetensors" for path in files)

    def tensor_plan(self, config: Mapping[str, Any]) -> TensorPlan:
        shapes = native_speechbrain_asr_tensor_shapes(config)
        return TensorPlan(rules=tuple(CopyTensor(name, name) for name in sorted(shapes)), )


def convert_speechbrain_asr_checkpoints(
    *,
    asr_checkpoint: str | Path,
    lm_checkpoint: str | Path,
    normalizer_checkpoint: str | Path,
    tokenizer_model: str | Path,
    destination: str | Path,
    hyperparams_file: str | Path | None = None,
    config: SpeechBrainCRDNNASRConfig | Mapping[str, Any] | None = None,
    trust_pickle_checkpoint: bool = False,
    expected_asr_sha256: str | None = None,
    expected_lm_sha256: str | None = None,
    expected_normalizer_sha256: str | None = None,
    expected_tokenizer_sha256: str | None = None,
    expected_hyperparams_sha256: str | None = None,
) -> Path:
    """Convert the three reviewed pickle files into one strict artifact."""
    if trust_pickle_checkpoint is not True:
        raise ValueError(
            "SpeechBrain publishes ASR, RNNLM, and normalizer state in "
            "pickle-based checkpoints. Review their origin and pass "
            "`trust_pickle_checkpoint=True` for one-time restricted "
            "conversion.")
    resolved = SpeechBrainCRDNNASRConfig.coerce(config or SpeechBrainCRDNNASRConfig(), )
    sources = {
        "ASR checkpoint": Path(asr_checkpoint).expanduser().resolve(),
        "LM checkpoint": Path(lm_checkpoint).expanduser().resolve(),
        "normalizer checkpoint": (Path(normalizer_checkpoint).expanduser().resolve()),
        "tokenizer": Path(tokenizer_model).expanduser().resolve(),
    }
    for label, path in sources.items():
        if not path.is_file():
            raise FileNotFoundError(f"SpeechBrain {label} was not found: {path}.")
    digests = {
        "asr":
        _validate_digest(
            sources["ASR checkpoint"],
            expected_asr_sha256,
            label="ASR checkpoint",
        ),
        "lm":
        _validate_digest(
            sources["LM checkpoint"],
            expected_lm_sha256,
            label="LM checkpoint",
        ),
        "normalizer":
        _validate_digest(
            sources["normalizer checkpoint"],
            expected_normalizer_sha256,
            label="normalizer checkpoint",
        ),
        "tokenizer":
        _validate_digest(
            sources["tokenizer"],
            expected_tokenizer_sha256,
            label="tokenizer",
        ),
    }
    hyperparams_sha = None
    if hyperparams_file is not None:
        hyperparams = Path(hyperparams_file).expanduser().resolve()
        if not hyperparams.is_file():
            raise FileNotFoundError(f"SpeechBrain hyperparams file was not found: {hyperparams}.")
        hyperparams_sha = _validate_digest(
            hyperparams,
            expected_hyperparams_sha256,
            label="hyperparams",
        )
    elif expected_hyperparams_sha256 is not None:
        raise ValueError("`hyperparams_file` is required when verifying its SHA-256.")

    asr_state = _restricted_state_dict(
        sources["ASR checkpoint"],
        label="ASR checkpoint",
    )
    lm_state = _restricted_state_dict(
        sources["LM checkpoint"],
        label="LM checkpoint",
    )
    normalizer = _normalizer_tensors(sources["normalizer checkpoint"], )
    asr_mapping = _asr_mapping(resolved)
    lm_mapping = _lm_mapping(resolved)
    if set(asr_state) != set(asr_mapping):
        raise ValueError(
            "SpeechBrain ASR checkpoint tensor namespace mismatch "
            f"(missing={sorted(set(asr_mapping) - set(asr_state))}, "
            f"unexpected={sorted(set(asr_state) - set(asr_mapping))}).")
    if set(lm_state) != set(lm_mapping):
        raise ValueError(
            "SpeechBrain LM checkpoint tensor namespace mismatch "
            f"(missing={sorted(set(lm_mapping) - set(lm_state))}, "
            f"unexpected={sorted(set(lm_state) - set(lm_mapping))}).")
    fingerprints = {
        "asr": tensor_inventory_fingerprint(asr_state),
        "lm": tensor_inventory_fingerprint(lm_state),
        "normalizer": tensor_inventory_fingerprint(normalizer),
    }
    if resolved.variant == "librispeech-bpe-1000":
        expected_fingerprints = {
            "asr": SPEECHBRAIN_ASR_TENSOR_FINGERPRINT,
            "lm": SPEECHBRAIN_ASR_LM_TENSOR_FINGERPRINT,
            "normalizer": (SPEECHBRAIN_ASR_NORMALIZER_TENSOR_FINGERPRINT),
        }
        if fingerprints != expected_fingerprints:
            raise ValueError(
                "SpeechBrain checkpoint inventory fingerprint mismatch: "
                f"expected {expected_fingerprints}, found {fingerprints}.")

    state = {native: asr_state[source].detach().cpu().contiguous() for source, native in asr_mapping.items()}
    state.update({
        native: lm_state[source].detach().cpu().contiguous()
        for source, native in lm_mapping.items()
    })
    state.update({
        "frontend.normalizer.glob_mean": normalizer["glob_mean"].detach().cpu().contiguous(),
        "frontend.normalizer.glob_std": normalizer["glob_std"].detach().cpu().contiguous(),
        "frontend.normalizer.count": normalizer["count"].detach().cpu().contiguous(),
    })
    shapes = native_speechbrain_asr_tensor_shapes(resolved)
    if set(state) != set(shapes):
        raise RuntimeError(
            "Internal SpeechBrain ASR conversion is incomplete "
            f"(missing={sorted(set(shapes) - set(state))}, "
            f"unexpected={sorted(set(state) - set(shapes))}).")
    mismatches = {
        name: (tuple(state[name].shape), shapes[name])
        for name in shapes if tuple(state[name].shape) != shapes[name]
    }
    if mismatches:
        raise ValueError(f"SpeechBrain ASR tensor shape mismatch: {mismatches}.")

    output = Path(destination).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    safe_path = output / NATIVE_SPEECHBRAIN_ASR_FILENAME
    metadata = {
        "format": NATIVE_SPEECHBRAIN_ASR_FORMAT,
        "architecture": "speechbrain-crdnn-asr",
        "source_artifact_revision": SPEECHBRAIN_ASR_REVISION,
        "source_training_revision": SPEECHBRAIN_ASR_SOURCE_REVISION,
        "source_asr_sha256": digests["asr"],
        "source_lm_sha256": digests["lm"],
        "source_normalizer_sha256": digests["normalizer"],
        "source_tokenizer_sha256": digests["tokenizer"],
        "source_asr_tensor_fingerprint": fingerprints["asr"],
        "source_lm_tensor_fingerprint": fingerprints["lm"],
        "source_normalizer_tensor_fingerprint": fingerprints["normalizer"],
    }
    save_safetensors(state, safe_path, metadata=metadata)
    tokenizer_output = output / NATIVE_SPEECHBRAIN_ASR_TOKENIZER
    tokenizer_output.write_bytes(sources["tokenizer"].read_bytes())
    values = resolved.to_dict()
    values.update({
        "checkpoint_format": NATIVE_SPEECHBRAIN_ASR_FORMAT,
        "source_artifact_revision": SPEECHBRAIN_ASR_REVISION,
        "source_training_revision": SPEECHBRAIN_ASR_SOURCE_REVISION,
        "source_checkpoint_sha256": digests["asr"],
        "source_lm_sha256": digests["lm"],
        "source_normalizer_sha256": digests["normalizer"],
        "source_tokenizer_sha256": digests["tokenizer"],
        "source_hyperparams_sha256": hyperparams_sha,
        "source_tensor_fingerprint": fingerprints["asr"],
    })
    write_json_file(output / "config.json", values)

    # Reopen the steady-state format and verify every tensor can be streamed
    # into the exact graph before declaring conversion complete.
    from voicehub.models.asr_speechbrain.native.modeling import SpeechBrainCRDNNForASR

    model = SpeechBrainCRDNNForASR(resolved)
    adapter = SpeechBrainASRSafeTensorsCheckpointAdapter()
    with SafeTensorReader(safe_path) as reader:
        adapter.load_streaming(
            model,
            reader,
            values,
            strict=True,
        )
    del model
    return output


def official_speechbrain_asr_conversion_kwargs() -> dict[str, str]:
    """Return immutable digest arguments for the public release."""
    return {
        "expected_asr_sha256": SPEECHBRAIN_ASR_MODEL_SHA256,
        "expected_lm_sha256": SPEECHBRAIN_ASR_LM_SHA256,
        "expected_normalizer_sha256": SPEECHBRAIN_ASR_NORMALIZER_SHA256,
        "expected_tokenizer_sha256": SPEECHBRAIN_ASR_TOKENIZER_SHA256,
        "expected_hyperparams_sha256": SPEECHBRAIN_ASR_HPARAMS_SHA256,
    }


__all__ = [
    "NATIVE_SPEECHBRAIN_ASR_FILENAME",
    "NATIVE_SPEECHBRAIN_ASR_FORMAT",
    "NATIVE_SPEECHBRAIN_ASR_TOKENIZER",
    "SpeechBrainASRSafeTensorsCheckpointAdapter",
    "convert_speechbrain_asr_checkpoints",
    "native_speechbrain_asr_tensor_shapes",
    "official_speechbrain_asr_conversion_kwargs",
    "speechbrain_asr_source_tensor_mapping",
    "speechbrain_lm_source_tensor_mapping",
    "tensor_inventory_fingerprint",
]
