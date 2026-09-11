"""Model-local F5-TTS source-native training adapter."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicehub.training.adapters import FlowMatchingTrainingAdapter
from voicehub.training.contracts import TrainingContext
from voicehub.training.ema import ExponentialMovingAverage
from voicehub.training.recipes import SourceRecipeTrainingAdapter


class F5TTSTrainingAdapter(
        FlowMatchingTrainingAdapter,
        SourceRecipeTrainingAdapter,
):
    """Native conditional-flow objective with update-coupled EMA."""

    supports_custom_recipe = True
    native_export_semantics = "inference-export"

    def __init__(self, model, spec):
        super().__init__(model, spec)
        self._ema: ExponentialMovingAverage | None = None

    def _use_ema(self) -> bool:
        config = getattr(self.model, "config", None)
        enabled = getattr(config, "use_ema", True)
        if not isinstance(enabled, bool):
            raise TypeError("F5-TTS `use_ema` must be a boolean.")
        return enabled

    def setup(self):
        super().setup()
        if not self._use_ema():
            self._ema = None
        elif self._ema is None:
            config = getattr(self.model, "config", None)
            self._ema = ExponentialMovingAverage(
                self.primary_model,
                decay=float(getattr(config, "ema_decay", 0.9999)),
                update_after_step=int(getattr(config, "ema_update_after_step", 0)),
                update_every=int(getattr(config, "ema_update_every", 1)),
            )
        return self

    def recipe_resume_configuration(self):
        configuration = dict(super().recipe_resume_configuration())
        config = getattr(self.model, "config", None)
        configuration.update({
            "resolved_use_ema":
            self._use_ema(),
            "resolved_ema_decay":
            float(getattr(config, "ema_decay", 0.9999), ),
            "resolved_ema_update_after_step":
            int(getattr(config, "ema_update_after_step", 0), ),
            "resolved_ema_update_every":
            int(getattr(config, "ema_update_every", 1), ),
        })
        return configuration

    def prepare_training_inputs(
        self,
        inputs: Mapping[str, Any],
        context: TrainingContext,
    ) -> Mapping[str, Any]:
        del context
        prepared = dict(inputs)
        aliases = {
            "mel": "inp",
            "mel_spec": "inp",
            "input_values": "inp",
            "input_ids": "text",
            "mel_lengths": "lens",
            "lengths": "lens",
        }
        for source, target in aliases.items():
            if source in prepared and target not in prepared:
                value = prepared.pop(source)
                if (source in ("mel", "mel_spec") and getattr(value, "ndim", None) == 3):
                    value = value.permute(0, 2, 1)
                prepared[target] = value
        allowed = ("inp", "text", "lens", "noise_scheduler")
        return {name: prepared[name] for name in allowed if name in prepared}

    def on_optimizer_step(
        self,
        *,
        optimizer_names: tuple[str, ...] | None,
        step: int,
    ) -> None:
        del optimizer_names
        self.setup()
        if self._ema is not None:
            self._ema.update(step=step)

    def recipe_state_dict(self) -> Mapping[str, Any]:
        self.setup()
        if self._ema is None:
            return {}
        return {"ema": self._ema.state_dict()}

    def load_recipe_state_dict(
        self,
        state_dict: Mapping[str, Any],
        *,
        strict: bool = True,
    ) -> None:
        self.setup()
        if not isinstance(state_dict, Mapping):
            raise TypeError("F5-TTS recipe state must be a mapping.")
        if self._ema is None:
            if strict and state_dict:
                raise ValueError("F5-TTS recipe state cannot contain EMA data when "
                                 "`use_ema=False`.")
            return
        if not state_dict:
            return
        if strict and set(state_dict) != {"ema"}:
            raise ValueError("F5-TTS recipe state must contain only 'ema'.")
        if "ema" in state_dict:
            self._ema.load_state_dict(state_dict["ema"], strict=strict)

    def save_pretrained(self, save_directory) -> None:
        """Export EMA weights when enabled, otherwise explicit raw weights."""
        self.setup()
        from voicehub.models.f5tts.native.checkpoint import export_f5tts_checkpoint

        destination = Path(save_directory)
        destination.mkdir(parents=True, exist_ok=True)
        state = self.primary_model.state_dict()
        prefix = ""
        export_state = state
        if self._ema is not None:
            ema_state = self._ema.state_dict()["shadow"]
            export_state = {name: ema_state.get(name, value) for name, value in state.items()}
            prefix = "ema_model."
        export_f5tts_checkpoint(
            self.primary_model,
            destination / "model.safetensors",
            prefix=prefix,
            state_override=export_state,
        )
        runtime = getattr(self.model, "model", None)
        frontend = getattr(runtime, "frontend", None)
        vocabulary = getattr(frontend, "vocabulary", None)
        if vocabulary is None or not callable(getattr(vocabulary, "save", None)):
            raise TypeError("Native F5-TTS export requires the loaded vocabulary.")
        vocabulary.save(destination / "vocab.txt")
        self.model.config.save_pretrained(destination)


__all__ = ["F5TTSTrainingAdapter"]
