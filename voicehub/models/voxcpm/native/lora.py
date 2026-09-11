"""Published VoxCPM2 LoRA topology with pickle-free adapter artifacts."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional

from voicehub.checkpointing import SafeTensorReader, save_safetensors
from voicehub.checkpointing.errors import CheckpointCompatibilityError
from voicehub.models.voxcpm.native.modeling import VoxCPM2Model


@dataclass(frozen=True, slots=True)
class VoxCPMLoRAConfig:
    """The exact adapter switches exposed by the official fine-tuner."""

    enable_lm: bool = True
    enable_dit: bool = True
    enable_proj: bool = True
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules_lm: tuple[str, ...] = (
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",
    )
    target_modules_dit: tuple[str, ...] = (
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",
    )
    target_proj_modules: tuple[str, ...] = (
        "enc_to_lm_proj",
        "lm_to_dit_proj",
        "res_to_dit_proj",
        "fusion_concat_proj",
    )

    def __post_init__(self) -> None:
        for name in ("enable_lm", "enable_dit", "enable_proj"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"`{name}` must be a boolean.")
        if not any((self.enable_lm, self.enable_dit, self.enable_proj)):
            raise ValueError("At least one VoxCPM LoRA component must be enabled.")
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank <= 0:
            raise ValueError("VoxCPM LoRA rank must be a positive integer.")
        alpha = float(self.alpha)
        dropout = float(self.dropout)
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("VoxCPM LoRA alpha must be finite and positive.")
        if not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("VoxCPM LoRA dropout must be in [0, 1).")
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "dropout", dropout)
        for name in (
                "target_modules_lm",
                "target_modules_dit",
                "target_proj_modules",
        ):
            values = tuple(getattr(self, name))
            if not values or any(not isinstance(value, str) or not value for value in values):
                raise ValueError(f"`{name}` must contain non-empty module names.")
            object.__setattr__(self, name, values)

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> VoxCPMLoRAConfig:
        if not isinstance(values, Mapping):
            raise TypeError("VoxCPM LoRA configuration must be a mapping.")
        normalized = dict(values)
        if "r" in normalized and "rank" not in normalized:
            normalized["rank"] = normalized.pop("r")
        return cls(**normalized)


class VoxCPMLoRALinear(nn.Module):
    """LoRA linear preserving the base checkpoint's weight/bias keys."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.rank = rank
        self.alpha = alpha
        self.base_scaling = alpha / rank
        self.register_buffer(
            "scaling",
            torch.tensor(
                self.base_scaling,
                device=base.weight.device,
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.weight = base.weight
        self.bias = base.bias
        self.lora_A = nn.Parameter(
            torch.empty(
                rank,
                self.in_features,
                device=base.weight.device,
                dtype=base.weight.dtype,
            ))
        self.lora_B = nn.Parameter(
            torch.zeros(
                self.out_features,
                rank,
                device=base.weight.device,
                dtype=base.weight.dtype,
            ))
        if base.weight.device.type != "meta":
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()

    def forward(self, inputs: Tensor) -> Tensor:
        base = functional.linear(inputs, self.weight, self.bias)
        adapter = functional.linear(
            functional.linear(inputs, self.lora_A),
            self.lora_B,
        )
        return base + self.dropout(adapter) * self.scaling

    def set_enabled(self, enabled: bool) -> None:
        self.scaling.fill_(self.base_scaling if enabled else 0.0)


def _replace_named_linears(
    root: nn.Module,
    targets: Sequence[str],
    config: VoxCPMLoRAConfig,
) -> int:
    target_set = frozenset(targets)
    replacements = []
    for name, module in root.named_modules():
        if isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in target_set:
            replacements.append((name, module))
    for name, module in replacements:
        parts = name.split(".")
        parent = root
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(
            parent,
            parts[-1],
            VoxCPMLoRALinear(
                module,
                rank=config.rank,
                alpha=config.alpha,
                dropout=config.dropout,
            ),
        )
    return len(replacements)


def inject_voxcpm_lora(
    model: VoxCPM2Model,
    config: VoxCPMLoRAConfig,
    *,
    freeze_base: bool = True,
) -> tuple[str, ...]:
    if not isinstance(model, VoxCPM2Model):
        raise TypeError("VoxCPM LoRA targets must be a VoxCPM2Model.")
    if not isinstance(config, VoxCPMLoRAConfig):
        raise TypeError("`config` must be a VoxCPMLoRAConfig.")
    if any(isinstance(module, VoxCPMLoRALinear) for module in model.modules()):
        raise RuntimeError("VoxCPM LoRA has already been injected.")
    count = 0
    if config.enable_lm:
        count += _replace_named_linears(
            model.base_lm,
            config.target_modules_lm,
            config,
        )
        count += _replace_named_linears(
            model.residual_lm,
            config.target_modules_lm,
            config,
        )
    if config.enable_dit:
        count += _replace_named_linears(
            model.feat_decoder.estimator,
            config.target_modules_dit,
            config,
        )
    if config.enable_proj:
        for name in config.target_proj_modules:
            module = getattr(model, name, None)
            if not isinstance(module, nn.Linear):
                raise ValueError(f"VoxCPM projection {name!r} is not a linear layer.")
            setattr(
                model,
                name,
                VoxCPMLoRALinear(
                    module,
                    rank=config.rank,
                    alpha=config.alpha,
                    dropout=config.dropout,
                ),
            )
            count += 1
    if not count:
        raise ValueError("VoxCPM LoRA configuration selected no modules.")
    if freeze_base:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = (name.endswith(".lora_A") or name.endswith(".lora_B"))
    return tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)


def voxcpm_lora_state_dict(model: VoxCPM2Model) -> dict[str, Tensor]:
    state = {
        name: value
        for name, value in model.state_dict().items() if name.endswith(".lora_A") or name.endswith(".lora_B")
    }
    if not state:
        raise RuntimeError("VoxCPM model has no injected LoRA parameters.")
    return state


def merged_voxcpm_state_dict(model: VoxCPM2Model) -> dict[str, Tensor]:
    """Return the standard VoxCPM2 namespace with every LoRA delta applied."""
    if not isinstance(model, VoxCPM2Model):
        raise TypeError("VoxCPM LoRA merge requires a VoxCPM2Model.")
    modules = {name: module for name, module in model.named_modules() if isinstance(module, VoxCPMLoRALinear)}
    if not modules:
        raise RuntimeError("VoxCPM model has no injected LoRA parameters.")
    state = {
        name: value.detach()
        for name, value in model.state_dict().items()
        if not name.endswith(".lora_A") and not name.endswith(".lora_B")
    }
    for name, module in modules.items():
        weight_name = f"{name}.weight"
        delta = torch.matmul(
            module.lora_B.float(),
            module.lora_A.float(),
        ).mul(module.base_scaling)
        state[weight_name] = (module.weight.detach().float() + delta).to(dtype=module.weight.dtype)
    return state


def export_voxcpm_lora(
    model: VoxCPM2Model,
    directory: str | Path,
    config: VoxCPMLoRAConfig,
) -> Path:
    destination = Path(directory).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    save_safetensors(
        {
            name: value.detach()
            for name, value in voxcpm_lora_state_dict(model).items()
        },
        destination / "lora_weights.safetensors",
        metadata={
            "format": "voicehub-native-voxcpm2-lora-v1",
            "architecture": "voxcpm2",
            "training_objective": "source-cfm-plus-stop-ce",
        },
    )
    (destination / "lora_config.json").write_text(
        json.dumps(
            {
                "format": "voicehub-native-voxcpm2-lora-v1",
                "lora_config": asdict(config),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    return destination.resolve()


def read_voxcpm_lora_config(path: str | Path) -> VoxCPMLoRAConfig:
    """Read and validate the configuration stored beside a LoRA artifact."""
    source = Path(path).expanduser().resolve()
    if source.is_file():
        source = source.with_name("lora_config.json")
    else:
        source = source / "lora_config.json"
    if not source.is_file():
        raise FileNotFoundError(f"VoxCPM LoRA configuration was not found: {source}.")
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid VoxCPM LoRA configuration: {source}.") from error
    if not isinstance(document, Mapping):
        raise TypeError("VoxCPM LoRA configuration root must be a mapping.")
    if document.get("format") != "voicehub-native-voxcpm2-lora-v1":
        raise ValueError("Unsupported VoxCPM LoRA artifact format.")
    values = document.get("lora_config")
    if not isinstance(values, Mapping):
        raise TypeError("VoxCPM LoRA artifact requires a `lora_config` mapping.")
    return VoxCPMLoRAConfig.from_mapping(values)


def load_voxcpm_lora(
    model: VoxCPM2Model,
    path: str | Path,
    config: VoxCPMLoRAConfig,
) -> None:
    if not isinstance(config, VoxCPMLoRAConfig):
        raise TypeError("`config` must be a VoxCPMLoRAConfig.")
    source = Path(path).expanduser().resolve()
    if source.is_dir():
        source = source / "lora_weights.safetensors"
    expected = voxcpm_lora_state_dict(model)
    with SafeTensorReader(source) as reader:
        actual_names = set(reader.keys())
        expected_names = set(expected)
        if actual_names != expected_names:
            raise CheckpointCompatibilityError(
                "VoxCPM LoRA namespace mismatch: "
                f"missing={sorted(expected_names - actual_names)!r}, "
                f"unexpected={sorted(actual_names - expected_names)!r}.")
        values = {}
        for name, target in expected.items():
            if reader.tensor_shape(name) != tuple(target.shape):
                raise CheckpointCompatibilityError(f"VoxCPM LoRA tensor {name!r} has an incompatible shape.")
            values[name] = reader.get_tensor(name).to(
                device=target.device,
                dtype=target.dtype,
            )
    result = model.load_state_dict(values, strict=False)
    unexpected = [name for name in result.unexpected_keys if name in expected_names]
    if unexpected:
        raise CheckpointCompatibilityError(f"Unexpected VoxCPM LoRA keys: {unexpected!r}.")


__all__ = [
    "VoxCPMLoRAConfig",
    "VoxCPMLoRALinear",
    "export_voxcpm_lora",
    "inject_voxcpm_lora",
    "load_voxcpm_lora",
    "merged_voxcpm_state_dict",
    "read_voxcpm_lora_config",
    "voxcpm_lora_state_dict",
]
