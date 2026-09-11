"""Shared source-native fine-tuning adapter contracts.

Model-specific objectives live beside their integrations and compose
these contracts with VoiceHub's common optimization, checkpoint, and
strategy layers.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicehub.dependencies import resolve_import_path
from voicehub.errors import OptionalDependencyError
from voicehub.outputs import TTSTrainingOutput
from voicehub.training.adapters import BaseTrainingAdapter, CausalLMTrainingAdapter
from voicehub.training.contracts import TrainingContext


class SourceRecipeTrainingAdapter(BaseTrainingAdapter):
    """Base class for model-author training recipes integrated by VoiceHub."""

    supports_custom_recipe = True

    def _training_output(
        self,
        context: TrainingContext,
        *,
        loss,
        losses: Mapping[str, Any] | None = None,
        logits=None,
        metadata: Mapping[str, Any] | None = None,
    ) -> TTSTrainingOutput:
        output_metadata = {
            "model_type": self.model_type,
            "training_family": self.spec.family_name,
            "training_support": self.spec.support.value,
            "training_phase": context.phase.name,
            "optimizer_names": context.phase.optimizer_names,
            "source_native_recipe": True,
        }
        output_metadata.update(dict(metadata or {}))
        return TTSTrainingOutput(
            loss=loss,
            logits=logits,
            losses=dict(losses or {"loss": loss}),
            metadata=output_metadata,
            training_phase=context.phase.name,
            optimizer_names=context.phase.optimizer_names,
        )


class CodecCausalLMTrainingAdapter(
        CausalLMTrainingAdapter,
        SourceRecipeTrainingAdapter,
):
    """Causal codec LMs with frozen audio tokenizers and native HF loss."""

    supports_custom_recipe = True

    def setup(self):
        super().setup()
        codec = getattr(self.model, "codec", None)
        if codec is None:
            runtime = getattr(self.model, "model", None)
            codec = getattr(runtime, "codec", None)
            if codec is None:
                codec = getattr(runtime, "audio_codec", None)
        if codec is not None:
            if hasattr(codec, "eval"):
                codec.eval()
            if hasattr(codec, "parameters"):
                for parameter in codec.parameters():
                    parameter.requires_grad_(False)
        return self

    def create_dataset(self, records, **kwargs):
        self.setup()
        factory_path = self.spec.dataset_factory
        if factory_path is not None:
            try:
                factory = resolve_import_path(factory_path)
            except ModuleNotFoundError as exc:
                raise OptionalDependencyError(
                    f"{self.model_type!r} training dataset factory "
                    f"{factory_path!r} requires unavailable training "
                    "dependencies. Install them with `pip install "
                    '"voicehub[training]"` and retry.') from exc
            except (ImportError, AttributeError, TypeError, ValueError) as exc:
                raise ImportError(
                    f"Could not resolve training dataset factory "
                    f"{factory_path!r} for {self.model_type!r}: {exc}") from exc
            if not callable(factory):
                raise TypeError(
                    f"Training dataset factory {factory_path!r} for "
                    f"{self.model_type!r} must be callable.")
            return factory(self.model, records, **kwargs)
        return super().create_dataset(records, **kwargs)

    def save_pretrained(self, save_directory) -> None:
        self.setup()
        destination = Path(save_directory)
        if hasattr(self.primary_model, "save_pretrained"):
            self.primary_model.save_pretrained(
                destination,
                safe_serialization=True,
            )
        tokenizer = None
        for path in self.spec.tokenizer_paths:
            tokenizer = self._resolve_path(path)
            if tokenizer is not None:
                break
        if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(destination)


_COMPATIBILITY_EXPORTS = {
    "ConversationTTSTrainingAdapter":
    ("voicehub.models.conversationtts.training:"
     "ConversationTTSTrainingAdapter"),
    "F5TTSTrainingAdapter": ("voicehub.models.f5tts.training:F5TTSTrainingAdapter"),
    "OrpheusTrainingAdapter": ("voicehub.models.orpheustts.training:OrpheusTrainingAdapter"),
    "Qwen3TTSTrainingAdapter": ("voicehub.models.qwen3tts.training_adapter:"
                                "Qwen3TTSTrainingAdapter"),
}

__all__ = [
    "BUILTIN_MODEL_ADAPTERS",
    "CodecCausalLMTrainingAdapter",
    "SourceRecipeTrainingAdapter",
    *sorted(_COMPATIBILITY_EXPORTS),
]


def __getattr__(name: str) -> Any:
    """Resolve historical model-adapter imports without owning their code."""
    try:
        reference = _COMPATIBILITY_EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = resolve_import_path(reference)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_COMPATIBILITY_EXPORTS))


class _BuiltInModelAdapters(Mapping[str, Any]):
    """Read-only compatibility view over declarative adapter factories.

    Adapter modules resolve only when a value is requested. New
    integrations declare their factory on ``ModelTrainingSpec`` and
    never edit this module.
    """

    @staticmethod
    def _specs():
        from voicehub.training.specs import list_training_specs

        return tuple(spec for spec in list_training_specs(task=None) if spec.adapter_factory is not None)

    def __getitem__(self, model_type: str):
        from voicehub.training.specs import get_training_spec

        try:
            spec = get_training_spec(model_type)
        except KeyError:
            raise KeyError(model_type) from None
        if spec.model_type != model_type or spec.adapter_factory is None:
            raise KeyError(model_type)
        factory = resolve_import_path(spec.adapter_factory)
        if not callable(factory):
            raise TypeError(
                f"Training adapter factory {spec.adapter_factory!r} for "
                f"{model_type!r} must be callable.")
        return factory

    def __iter__(self):
        return iter(spec.model_type for spec in self._specs())

    def __len__(self) -> int:
        return len(self._specs())


BUILTIN_MODEL_ADAPTERS: Mapping[str, Any] = _BuiltInModelAdapters()
