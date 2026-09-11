"""Component-native CosyVoice LM, flow, and HiFT fine-tuning."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from voicehub.optimization.protocols import OptimizationCompileTarget
from voicehub.outputs import TTSTrainingOutput
from voicehub.training.adapters import CompositeTrainingAdapter
from voicehub.training.contracts import TrainingContext
from voicehub.training.datasets import SpeechDataset


@dataclass(frozen=True)
class CosyVoiceTrainingCollator:
    """Preserve raw records until the runtime-owned tokenizer is available."""

    def __call__(
        self,
        features: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        if not features:
            return {}
        if any(not isinstance(record, Mapping) for record in features):
            raise TypeError("Every CosyVoice record must be a mapping.")
        return {"records": [dict(record) for record in features]}

    def resume_fingerprint(self) -> dict[str, Any]:
        return {
            "architecture": "cosyvoice3",
            "checkpoint_format": "safetensors",
            "frontend": "frozen-native-or-preencoded-speech-tokens",
            "type": "cosyvoice-native-records-v1",
        }


class CosyVoiceTrainingAdapter(CompositeTrainingAdapter):
    """Execute exactly one source CosyVoice component objective per job."""

    supports_custom_recipe = True
    native_export_semantics = ("inference-ready-voicehub-native-cosyvoice-safetensors")
    _PHASE_ALIASES = {
        "language_model": "llm",
        "vocoder_generator": "hifigan_generator",
        "vocoder_discriminator": "hifigan_discriminator",
    }

    def __init__(self, model, spec) -> None:
        super().__init__(model, spec)
        self.data_collator = CosyVoiceTrainingCollator()

    @property
    def selected_component(self) -> str:
        selected = str(self.model.config.training_component).strip().lower().replace("-", "_")
        return "llm" if selected == "language_model" else selected

    def select_training_phase(
        self,
        training_phase: str | Any | None = None,
    ):
        if isinstance(training_phase, str):
            training_phase = self._PHASE_ALIASES.get(
                training_phase,
                training_phase,
            )
        phase = (
            self.spec.get_phase(self.selected_component)
            if training_phase is None else super().select_training_phase(training_phase))
        if phase.name != self.selected_component:
            raise ValueError(
                "This CosyVoice runtime was configured for "
                f"{self.selected_component!r}, not {phase.name!r}. Start a "
                "separate job with the other `training_component`.")
        return phase

    def plan_training_phases(self, step: int):
        del step
        return (self.select_training_phase(None), )

    def setup(self) -> CosyVoiceTrainingAdapter:
        self.model.load_for_training()
        runtime = getattr(self.model, "native_runtime", None)
        if runtime is None:
            raise ValueError("CosyVoice training requires the native runtime.")
        runtime.prepare_for_training(self.selected_component)
        super().setup()
        return self

    def optimization_module_roots(self):
        """Include the frozen raw-audio tokenizer in training policies."""
        self.setup()
        return self.model.native_runtime.optimization_module_roots()

    def optimization_compile_targets(
        self,
        mode: str,
    ) -> tuple[OptimizationCompileTarget, ...]:
        """Compile the selected objective and frozen preprocessing boundary."""
        if mode != "training":
            raise ValueError(f"Unsupported optimization mode {mode!r}.")
        self.setup()
        runtime = self.model.native_runtime
        component = self.selected_component
        if component in {"llm", "language_model"}:
            owner = runtime.model.llm
        elif component == "flow":
            owner = runtime.model.flow
        elif component in {"hift", "vocoder"}:
            owner = runtime.model.hift
        elif component in {
                "hifigan_generator",
                "hifigan_discriminator",
        }:
            owner = runtime.model.hifigan
            if owner is None:
                raise RuntimeError("CosyVoice HiFiGAN training graph is not attached.")
        else:
            raise ValueError(f"Unsupported CosyVoice component {component!r}.")
        targets = [
            OptimizationCompileTarget(
                f"cosyvoice.training.{component}.forward",
                owner,
                "forward",
            ),
        ]
        if runtime.speech_tokenizer is not None:
            targets.append(
                OptimizationCompileTarget(
                    "cosyvoice.training.speech_tokenizer.forward",
                    runtime.speech_tokenizer,
                    "forward",
                ))
        return tuple(targets)

    def create_dataset(self, records: Any, **kwargs: Any) -> SpeechDataset:
        return SpeechDataset(records, required_fields=(), **kwargs)

    def prepare_training_inputs(
        self,
        inputs: Mapping[str, Any],
        context: TrainingContext,
    ) -> Mapping[str, Any]:
        return self.model.prepare_training_inputs(
            inputs,
            phase=self.selected_component,
        )

    def execute_training_phase(
        self,
        context: TrainingContext,
    ) -> TTSTrainingOutput:
        self.setup()
        phase = self.select_training_phase(context.phase)
        prepared = self.prepare_training_inputs(
            context.inputs,
            context,
        )
        component = phase.name
        if component in {"hifigan_generator", "hifigan_discriminator"}:
            output = self.model.model(
                component=component,
                **dict(prepared),
            )
            logits = None
            losses = dict(output.losses)
            metadata = {"vocoder_phase": output.phase}
        elif component in {"llm", "language_model"}:
            output = self.model.model(
                component="llm",
                **dict(prepared),
            )
            logits = output.logits
            losses = {"language_model_loss": output.loss}
            metadata = {"accuracy": output.accuracy}
        elif component == "flow":
            output = self.model.model(
                component="flow",
                **dict(prepared),
            )
            logits = output.path
            losses = {"flow_matching_loss": output.loss}
            metadata = {}
        else:
            raise ValueError(f"Unsupported CosyVoice component {component!r}.")
        metadata.update({
            "architecture": "cosyvoice3",
            "checkpoint_format": "safetensors",
            "native_objective": True,
            "selected_component": component,
            "speech_tokenizer_frozen": True,
        })
        return TTSTrainingOutput(
            loss=output.loss,
            logits=logits,
            losses=losses,
            metadata=metadata,
            training_phase=phase.name,
            optimizer_names=phase.optimizer_names,
        )

    def recipe_resume_configuration(self) -> Mapping[str, Any]:
        configuration = dict(super().recipe_resume_configuration())
        configuration.update({
            "architecture": "cosyvoice3",
            "checkpoint_format": "safetensors",
            "selected_component": self.selected_component,
            "speech_tokenizer": "frozen-native-or-preencoded",
            "source_objectives": {
                "flow": "rectified-conditional-flow-matching",
                "hifigan_discriminator": "least-squares-adversarial",
                "hifigan_generator": ("adversarial+feature+spectral+pitch"),
                "llm": "length-normalized-speech-token-cross-entropy",
            },
        })
        return configuration

    def save_pretrained(self, save_directory: str | Path) -> None:
        self.setup()
        self.model.export_native_pretrained(save_directory)


__all__ = [
    "CosyVoiceTrainingAdapter",
    "CosyVoiceTrainingCollator",
]
