"""Checkpoint-safe caching for legacy VITS-family weight normalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn.utils.weight_norm import WeightNorm


def _is_compiling() -> bool:
    compiler = getattr(torch, "compiler", None)
    predicate = getattr(compiler, "is_compiling", None)
    return bool(predicate()) if callable(predicate) else False


class _CachedWeightNormHook(WeightNorm):
    """Retain a legacy hook while reusing its eval-only materialization."""

    def __init__(self, original: WeightNorm) -> None:
        super().__init__(original.name, original.dim)
        self.original = original
        self._signature: tuple[Any, ...] | None = None
        self._weight: Tensor | None = None

    def _parameter_signature(self, module: nn.Module) -> tuple[Any, ...]:
        magnitude = getattr(module, self.name + "_g")
        direction = getattr(module, self.name + "_v")
        return (
            id(magnitude),
            id(direction),
            magnitude._version,
            direction._version,
            magnitude.device,
            direction.device,
            magnitude.dtype,
            direction.dtype,
        )

    def clear(self) -> None:
        self._signature = None
        self._weight = None

    def materialize(self, module: nn.Module) -> int:
        self.original(module, ())
        weight = getattr(module, self.name)
        self._signature = self._parameter_signature(module)
        self._weight = weight
        return weight.numel() * weight.element_size()

    def __call__(
        self,
        module: nn.Module,
        inputs: tuple[Any, ...],
    ) -> None:
        # Let Dynamo capture the original tensor expression instead of this
        # Python cache policy. Inductor can then fuse or schedule it normally.
        if _is_compiling():
            self.original(module, inputs)
            return
        if module.training or torch.is_grad_enabled():
            self.original(module, inputs)
            self.clear()
            return
        weight = getattr(module, self.name, None)
        signature = self._parameter_signature(module)
        if self._weight is weight and self._signature == signature:
            return
        self.materialize(module)


@dataclass(slots=True)
class LegacyWeightNormInferenceCache:
    """Reversible collection of topology-preserving cached legacy hooks."""

    replacements: tuple[tuple[nn.Module, int, WeightNorm, _CachedWeightNormHook], ...]

    @property
    def module_count(self) -> int:
        return len(self.replacements)

    @property
    def bytes(self) -> int:
        return sum(
            0 if cached._weight is None else cached._weight.numel() * cached._weight.element_size()
            for _, _, _, cached in self.replacements)

    def clear(self) -> None:
        for _, _, _, cached in self.replacements:
            cached.clear()

    def materialize(self) -> int:
        total = 0
        with torch.inference_mode():
            for module, _, _, cached in self.replacements:
                total += cached.materialize(module)
        return total

    def restore(self) -> None:
        for module, key, original, cached in self.replacements:
            current = module._forward_pre_hooks.get(key)
            if current is cached:
                module._forward_pre_hooks[key] = original
        self.clear()


def enable_legacy_weight_norm_inference_cache(
    root: nn.Module,
    *,
    materialize: bool = True,
) -> LegacyWeightNormInferenceCache:
    """Cache legacy ``weight_g``/``weight_v`` hooks without changing state.

    Training, grad-enabled evaluation, parameter mutation, checkpoint
    loading, and device/dtype conversion all fall back to a fresh
    materialization.
    """
    if not isinstance(root, nn.Module):
        raise TypeError("`root` must be a torch.nn.Module.")
    replacements = []
    for module in root.modules():
        for key, hook in tuple(module._forward_pre_hooks.items()):
            if isinstance(hook, _CachedWeightNormHook):
                continue
            if not isinstance(hook, WeightNorm):
                continue
            cached = _CachedWeightNormHook(hook)
            module._forward_pre_hooks[key] = cached
            replacements.append((module, key, hook, cached))
    handle = LegacyWeightNormInferenceCache(tuple(replacements))
    if materialize:
        handle.materialize()
    return handle


__all__ = [
    "LegacyWeightNormInferenceCache",
    "enable_legacy_weight_norm_inference_cache",
]
