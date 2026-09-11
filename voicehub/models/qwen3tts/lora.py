"""Native, PEFT-free LoRA support for the Qwen3-TTS SFT graph."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch import Tensor, nn

from voicehub.checkpointing import SafeTensorReader, save_safetensors
from voicehub.hub import read_json_file, write_json_file
from voicehub.models.qwen3tts.lora_config import QWEN3_TTS_ATTENTION_LORA_TARGETS, normalize_qwen3_tts_lora_targets
from voicehub.optimization import LoRAConfig, LoRAInjection, inject_lora

QWEN3_TTS_LORA_ADAPTER_FORMAT = "voicehub-qwen3-tts-lora"
QWEN3_TTS_LORA_ADAPTER_VERSION = 1
QWEN3_TTS_LORA_CONFIG_NAME = "adapter_config.json"
QWEN3_TTS_LORA_WEIGHTS_NAME = "adapter_model.safetensors"
_TARGET_SPEAKER_EMBEDDING_KEY = "_voicehub.target_speaker_embedding"


class Qwen3TTSLoRAInjection(LoRAInjection):
    """LoRA handle that also restores Qwen3-TTS-wide trainability flags."""

    def __init__(
        self,
        injection: LoRAInjection,
        original_trainability: tuple[tuple[nn.Parameter, bool], ...],
    ) -> None:
        super().__init__(
            injection.model,
            injection.modules,
            injection.config,
        )
        self._original_trainability = original_trainability

    def restore(self) -> nn.Module:
        model = super().restore()
        for parameter, requires_grad in self._original_trainability:
            parameter.requires_grad_(requires_grad)
        return model


def _target_patterns(targets: tuple[str, ...]) -> tuple[str, ...]:
    patterns: list[str] = []
    for target in targets:
        component = ("self_attn" if target in QWEN3_TTS_ATTENTION_LORA_TARGETS else "mlp")
        patterns.extend((
            f"talker.model.layers.*.{component}.{target}",
            f"talker.code_predictor.model.layers.*.{component}.{target}",
        ))
    return tuple(patterns)


def _expected_target_names(
    model: nn.Module,
    targets: tuple[str, ...],
) -> tuple[str, ...]:
    names: list[str] = []
    for stack in (
            "talker.model",
            "talker.code_predictor.model",
    ):
        try:
            layers = model.get_submodule(f"{stack}.layers")
        except (AttributeError, ValueError) as error:
            raise ValueError(
                "Qwen3-TTS LoRA requires both the talker and residual "
                f"code-predictor decoder stacks; missing {stack!r}.") from error
        layer_names = tuple(name for name, _ in layers.named_children())
        if not layer_names:
            raise ValueError(f"Qwen3-TTS LoRA decoder stack {stack!r} has no layers.")
        for layer_name in layer_names:
            for target in targets:
                component = ("self_attn" if target in QWEN3_TTS_ATTENTION_LORA_TARGETS else "mlp")
                module_name = (f"{stack}.layers.{layer_name}.{component}.{target}")
                try:
                    module = model.get_submodule(module_name)
                except (AttributeError, ValueError) as error:
                    raise ValueError(
                        "Qwen3-TTS LoRA topology is incomplete; missing "
                        f"projection {module_name!r}.") from error
                if not isinstance(module, nn.Linear):
                    raise TypeError(
                        "Qwen3-TTS LoRA projection "
                        f"{module_name!r} must be torch.nn.Linear, received "
                        f"{type(module).__name__}.")
                names.append(module_name)
    return tuple(sorted(names))


def build_qwen3_tts_lora_config(
    *,
    rank: int,
    alpha: float,
    dropout: float,
    target_modules: tuple[str, ...],
    seed: int,
) -> LoRAConfig:
    """Translate public projection names into both trainable decoder stacks."""
    targets = normalize_qwen3_tts_lora_targets(target_modules)
    return LoRAConfig(
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        target_modules=_target_patterns(targets),
        freeze_base=True,
        seed=seed,
    )


def inject_qwen3_tts_lora(
    model: nn.Module,
    *,
    rank: int,
    alpha: float,
    dropout: float,
    target_modules: tuple[str, ...],
    seed: int,
) -> Qwen3TTSLoRAInjection:
    """Inject exact talker/code-predictor adapters and freeze every base
    weight."""
    if not isinstance(model, nn.Module):
        raise TypeError("Qwen3-TTS LoRA requires a torch.nn.Module.")
    targets = normalize_qwen3_tts_lora_targets(target_modules)
    expected_names = _expected_target_names(model, targets)
    config = build_qwen3_tts_lora_config(
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        target_modules=targets,
        seed=seed,
    )
    original_trainability = tuple((parameter, parameter.requires_grad) for parameter in model.parameters())
    generic_injection = inject_lora(model, config)
    injection = Qwen3TTSLoRAInjection(
        generic_injection,
        original_trainability,
    )
    try:
        if injection.module_names != expected_names:
            raise RuntimeError(
                "Qwen3-TTS LoRA injection produced an unexpected topology: "
                f"expected {expected_names!r}, received "
                f"{injection.module_names!r}.")

        # Generic injection freezes the selected Linear bases. Qwen3-TTS LoRA
        # is deliberately stricter: all other embeddings, heads, norms, and
        # the speaker encoder remain frozen, so optimizer discovery sees only
        # the low-rank matrices.
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in injection.parameters():
            parameter.requires_grad_(True)
    except BaseException:
        injection.restore()
        raise
    return injection


def merged_qwen3_tts_state_dict(
        model: nn.Module,
        injection: LoRAInjection,
        *,
        drop_prefixes: tuple[str, ...] = (),
) -> dict[str, Tensor]:
    """Build a clean merged state without mutating live training weights."""
    if injection.model is not model:
        raise ValueError("The LoRA injection does not belong to this Qwen3-TTS model.")
    if any(module.merged for module in injection.modules.values()):
        raise RuntimeError("Qwen3-TTS export requires unmerged LoRA modules.")
    output: dict[str, Tensor] = {}
    module_names = tuple(sorted(
        injection.module_names,
        key=lambda name: (-name.count("."), name),
    ))
    for name, tensor in model.state_dict().items():
        if any(name.startswith(prefix) for prefix in drop_prefixes):
            continue
        destination_name = name
        skip = False
        for module_name in module_names:
            if name in {
                    f"{module_name}.lora_a",
                    f"{module_name}.lora_b",
            }:
                skip = True
                break
            base_prefix = f"{module_name}.base."
            if name.startswith(base_prefix):
                destination_name = f"{module_name}.{name[len(base_prefix):]}"
                break
        if skip:
            continue
        if destination_name in output:
            raise RuntimeError(
                "Qwen3-TTS merged LoRA export produced duplicate key "
                f"{destination_name!r}.")
        output[destination_name] = tensor.detach().clone()

    for module_name in injection.module_names:
        wrapper = injection.modules[module_name]
        weight_name = f"{module_name}.weight"
        if weight_name not in output:
            raise RuntimeError(
                "Qwen3-TTS merged LoRA export could not locate base weight "
                f"{weight_name!r}.")
        output[weight_name] = (
            wrapper.base.weight.detach().clone() + wrapper.adapter_delta().detach().to(
                device=wrapper.base.weight.device,
                dtype=wrapper.base.weight.dtype,
            ))
    return output


@dataclass(frozen=True)
class Qwen3TTSLoRAAdapterLoadResult:
    """Non-parameter state restored with an adapter-only checkpoint."""

    target_speaker_embedding: Tensor | None
    speaker_name: str | None
    speaker_id: int | None
    base_model: str | None


def save_qwen3_tts_lora_adapter(
    injection: LoRAInjection,
    directory: str | Path,
    *,
    target_modules: tuple[str, ...],
    base_model: str | Path | None,
    target_speaker_embedding: Tensor | None,
    speaker_name: str,
    speaker_id: int,
) -> Path:
    """Write adapter-only tensors plus enough metadata for strict
    reattachment."""
    targets = normalize_qwen3_tts_lora_targets(target_modules)
    if isinstance(base_model, Path):
        base_model = str(base_model)
    if base_model is not None:
        if not isinstance(base_model, str) or not base_model.strip():
            raise ValueError("Qwen3-TTS adapter base model must be non-empty or None.")
        base_model = base_model.strip()
    if not isinstance(speaker_name, str) or not speaker_name.strip():
        raise ValueError("Qwen3-TTS adapter speaker name must be non-empty.")
    if (isinstance(speaker_id, bool) or not isinstance(speaker_id, int) or speaker_id < 0):
        raise ValueError("Qwen3-TTS adapter speaker id must be a non-negative integer.")
    state = injection.adapter_state_dict()
    if target_speaker_embedding is not None:
        if (not isinstance(target_speaker_embedding, Tensor) or target_speaker_embedding.ndim != 1 or
                not target_speaker_embedding.dtype.is_floating_point):
            raise ValueError("Qwen3-TTS target speaker embedding must be a floating-point "
                             "vector.")
        state[_TARGET_SPEAKER_EMBEDDING_KEY] = (target_speaker_embedding.detach().clone())

    destination = Path(directory).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    save_safetensors(
        state,
        destination / QWEN3_TTS_LORA_WEIGHTS_NAME,
        metadata={
            "format": QWEN3_TTS_LORA_ADAPTER_FORMAT,
            "format_version": str(QWEN3_TTS_LORA_ADAPTER_VERSION),
        },
    )
    config = injection.config
    write_json_file(
        destination / QWEN3_TTS_LORA_CONFIG_NAME,
        {
            "format": QWEN3_TTS_LORA_ADAPTER_FORMAT,
            "format_version": QWEN3_TTS_LORA_ADAPTER_VERSION,
            'base_model': base_model,
            "rank": config.rank,
            "alpha": config.alpha,
            "dropout": config.dropout,
            "seed": config.seed,
            "target_modules": list(targets),
            "injected_module_names": list(injection.module_names),
            "speaker_name": speaker_name.strip(),
            "speaker_id": speaker_id,
            "has_target_speaker_embedding": (target_speaker_embedding is not None),
            "merge_semantics": "clone-and-merge-without-live-weight-mutation",
        },
    )
    return destination


def _adapter_config_signature(injection: LoRAInjection) -> dict[str, Any]:
    config = injection.config
    return {
        "rank": config.rank,
        "alpha": config.alpha,
        "dropout": config.dropout,
        "seed": config.seed,
        "expanded_target_modules": list(config.target_modules),
        "injected_module_names": list(injection.module_names),
    }


def load_qwen3_tts_lora_adapter(
    injection: LoRAInjection,
    directory: str | Path,
    *,
    target_modules: tuple[str, ...],
    expected_base_model: str | Path | None = None,
    expected_speaker_name: str | None = None,
    expected_speaker_id: int | None = None,
    expected_speaker_embedding_size: int | None = None,
) -> Qwen3TTSLoRAAdapterLoadResult:
    """Strictly restore a VoiceHub Qwen3-TTS adapter into an injected graph."""
    targets = normalize_qwen3_tts_lora_targets(target_modules)
    source = Path(directory).expanduser()
    manifest = read_json_file(source / QWEN3_TTS_LORA_CONFIG_NAME)
    if not isinstance(manifest, Mapping):
        raise TypeError("Qwen3-TTS LoRA adapter manifest must be a JSON object.")
    if manifest.get("format") != QWEN3_TTS_LORA_ADAPTER_FORMAT:
        raise ValueError("Not a VoiceHub Qwen3-TTS LoRA adapter directory.")
    format_version = manifest.get("format_version")
    if (isinstance(format_version, bool) or not isinstance(format_version, int) or
            format_version != QWEN3_TTS_LORA_ADAPTER_VERSION):
        raise ValueError("Unsupported Qwen3-TTS LoRA adapter format version "
                         f"{format_version!r}.")
    expected_manifest_keys = {
        "alpha",
        'base_model',
        "dropout",
        "format",
        "format_version",
        "has_target_speaker_embedding",
        "injected_module_names",
        "merge_semantics",
        "rank",
        "seed",
        "speaker_id",
        "speaker_name",
        "target_modules",
    }
    if set(manifest) != expected_manifest_keys:
        raise ValueError(
            "Qwen3-TTS LoRA adapter manifest fields mismatch: "
            f"expected {sorted(expected_manifest_keys)!r}, "
            f"received {sorted(manifest)!r}.")
    if manifest.get("merge_semantics") != ("clone-and-merge-without-live-weight-mutation"):
        raise ValueError("Qwen3-TTS LoRA adapter has unknown merge semantics.")

    # Constructing the generic config gives adapter manifests the same strict
    # scalar validation as the live injection (including bool-vs-int checks).
    stored_config = LoRAConfig(
        rank=manifest.get("rank"),
        alpha=manifest.get("alpha"),
        dropout=manifest.get("dropout"),
        target_modules=_target_patterns(targets),
        freeze_base=True,
        seed=manifest.get("seed"),
    )
    expected = _adapter_config_signature(injection)
    received = {
        "rank": stored_config.rank,
        "alpha": stored_config.alpha,
        "dropout": stored_config.dropout,
        "seed": stored_config.seed,
        "expanded_target_modules": list(stored_config.target_modules),
        "injected_module_names": manifest.get("injected_module_names"),
    }
    if received != expected:
        raise ValueError(
            "Qwen3-TTS LoRA adapter topology/configuration mismatch: "
            f"expected {expected!r}, received {received!r}.")
    if manifest.get("target_modules") != list(targets):
        raise ValueError(
            "Qwen3-TTS LoRA adapter public target topology does not match "
            "the active training configuration.")
    base_model = manifest.get('base_model')
    if base_model is not None and (not isinstance(base_model, str) or not base_model.strip()):
        raise TypeError("Qwen3-TTS LoRA adapter `base_model` must be a non-empty "
                        "string or null.")
    if isinstance(expected_base_model, Path):
        expected_base_model = str(expected_base_model)
    if (expected_base_model is not None and base_model != expected_base_model):
        raise ValueError(
            f"Qwen3-TTS LoRA adapter expects base model {base_model!r}, "
            f"not {expected_base_model!r}.")
    speaker_name = manifest.get("speaker_name")
    if speaker_name is not None and (not isinstance(speaker_name, str) or not speaker_name.strip()):
        raise ValueError("Qwen3-TTS LoRA adapter `speaker_name` must be non-empty or null.")
    if (expected_speaker_name is not None and speaker_name != expected_speaker_name):
        raise ValueError(
            f"Qwen3-TTS LoRA adapter expects speaker {speaker_name!r}, "
            f"not {expected_speaker_name!r}.")
    speaker_id = manifest.get("speaker_id")
    if speaker_id is not None and (isinstance(speaker_id, bool) or not isinstance(speaker_id, int) or
                                   speaker_id < 0):
        raise ValueError("Qwen3-TTS LoRA adapter `speaker_id` must be non-negative or null.")
    if (expected_speaker_id is not None and speaker_id != expected_speaker_id):
        raise ValueError(
            f"Qwen3-TTS LoRA adapter expects speaker id {speaker_id!r}, "
            f"not {expected_speaker_id!r}.")
    expects_embedding = manifest.get("has_target_speaker_embedding")
    if not isinstance(expects_embedding, bool):
        raise TypeError("Qwen3-TTS LoRA adapter `has_target_speaker_embedding` must "
                        "be a boolean.")

    with SafeTensorReader(source / QWEN3_TTS_LORA_WEIGHTS_NAME) as reader:
        if reader.metadata.get("format") != QWEN3_TTS_LORA_ADAPTER_FORMAT:
            raise ValueError("Qwen3-TTS LoRA weights have incompatible Safetensors metadata.")
        if reader.metadata.get("format_version") != str(QWEN3_TTS_LORA_ADAPTER_VERSION):
            raise ValueError("Qwen3-TTS LoRA weights have an incompatible format version.")
        state = reader.state_dict()
    target_speaker_embedding = state.pop(
        _TARGET_SPEAKER_EMBEDDING_KEY,
        None,
    )
    if expects_embedding != (target_speaker_embedding is not None):
        raise ValueError(
            "Qwen3-TTS LoRA adapter speaker-embedding manifest does not "
            "match its tensor payload.")
    if target_speaker_embedding is not None:
        if (target_speaker_embedding.ndim != 1 or not target_speaker_embedding.dtype.is_floating_point):
            raise ValueError(
                "Qwen3-TTS LoRA adapter target speaker embedding must be a "
                "floating-point vector.")
        if (expected_speaker_embedding_size is not None and
                target_speaker_embedding.numel() != expected_speaker_embedding_size):
            raise ValueError(
                "Qwen3-TTS LoRA adapter speaker embedding has "
                f"{target_speaker_embedding.numel()} values; expected "
                f"{expected_speaker_embedding_size}.")
    injection.load_adapter_state_dict(state, strict=True)

    return Qwen3TTSLoRAAdapterLoadResult(
        target_speaker_embedding=target_speaker_embedding,
        speaker_name=speaker_name,
        speaker_id=speaker_id,
        base_model=base_model,
    )


__all__ = [
    "QWEN3_TTS_LORA_ADAPTER_FORMAT",
    "QWEN3_TTS_LORA_ADAPTER_VERSION",
    "QWEN3_TTS_LORA_CONFIG_NAME",
    "QWEN3_TTS_LORA_WEIGHTS_NAME",
    "Qwen3TTSLoRAInjection",
    "Qwen3TTSLoRAAdapterLoadResult",
    "build_qwen3_tts_lora_config",
    "inject_qwen3_tts_lora",
    "load_qwen3_tts_lora_adapter",
    "merged_qwen3_tts_state_dict",
    "save_qwen3_tts_lora_adapter",
]
