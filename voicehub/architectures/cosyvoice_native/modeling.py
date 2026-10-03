"""Composite native CosyVoice graph and component-specific objectives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from voicehub.architectures.cosyvoice_native.configuration import CosyVoiceArchitectureConfig
from voicehub.architectures.cosyvoice_native.flow import CosyVoiceFlowMatchingModel, CosyVoiceFlowOutput
from voicehub.architectures.cosyvoice_native.language_model import CosyVoiceLanguageModel, CosyVoiceLanguageOutput
from voicehub.architectures.cosyvoice_native.vocoder import (
    CosyVoiceHiFTDiscriminator,
    CosyVoiceHiFTGenerator,
    CosyVoiceHiFTTrainingModel,
    CosyVoiceHiFTTrainingOutput,
)
from voicehub.optimization.protocols import OptimizationCompileTarget

#: FSQ silence/breath speech tokens of the CosyVoice 3 tokenizer. The
#: source ``CosyVoice3Model.llm_job`` forwards at most this many consecutive
#: ones to the flow decoder (the LM itself still conditions on all of them).
SILENT_SPEECH_TOKENS = frozenset({1, 2, 28, 29, 55, 248, 494, 2241, 2242, 2322, 2323})
MAX_CONSECUTIVE_SILENT_TOKENS = 5


def suppress_long_silences(
    speech_tokens: Tensor,
    *,
    silent_tokens: frozenset[int] = SILENT_SPEECH_TOKENS,
    max_consecutive: int = MAX_CONSECUTIVE_SILENT_TOKENS,
) -> Tensor:
    """Drop silent tokens beyond ``max_consecutive`` in a run, as the source
    does."""
    if speech_tokens.ndim != 2 or speech_tokens.shape[0] != 1:
        raise ValueError("`speech_tokens` must have shape [1, sequence].")
    kept: list[int] = []
    run = 0
    for token in speech_tokens[0].tolist():
        if token in silent_tokens:
            run += 1
            if run > max_consecutive:
                continue
        else:
            run = 0
        kept.append(token)
    return speech_tokens.new_tensor([kept])


@dataclass(frozen=True)
class CosyVoiceSynthesisOutput:
    """End-to-end speech tokens, mel features, and waveform."""

    waveform: Tensor
    speech_tokens: Tensor
    speech_features: Tensor
    sample_rate: int


class CosyVoiceNativeModel(nn.Module):
    """One graph with explicit LM, flow, and HiFT component boundaries.

    Optimization is deliberately component-selective: CosyVoice's source
    recipes do not pretend that three heterogeneous losses share one forward.
    Frontend text tokenization and any speech-token codec remain outside this
    graph and are frozen by the runtime/trainer boundary.
    """

    def __init__(
        self,
        config: CosyVoiceArchitectureConfig,
        *,
        initialize: bool = True,
        device=None,
        dtype=None,
        build_discriminator: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(config, CosyVoiceArchitectureConfig):
            raise TypeError("`config` must be CosyVoiceArchitectureConfig.")
        self.config = config
        self.llm = CosyVoiceLanguageModel(
            config.language,
            initialize=initialize,
            device=device,
            dtype=dtype,
        )
        self.flow = CosyVoiceFlowMatchingModel(
            config.flow,
            device=device,
            dtype=dtype,
        )
        self.hift = CosyVoiceHiFTGenerator(config.hift)
        self.hift.configure_source_noise(
            speech_vocab_size=config.flow.speech_vocab_size,
            speaker_embedding_dim=config.flow.speaker_embedding_dim,
        )
        self.hifigan: CosyVoiceHiFTTrainingModel | None = None
        if build_discriminator:
            self.attach_discriminator(tiny=(config.hift.base_channels < 64), )

    def optimization_compile_targets(
        self,
        mode: str,
    ) -> tuple[OptimizationCompileTarget, ...]:
        """Expose repeated stages instead of the Python sampling loops."""
        if mode == "training":
            return (OptimizationCompileTarget(
                "cosyvoice.forward",
                self,
                "forward",
            ), )
        if mode != "inference":
            raise ValueError(f"Unsupported optimization mode {mode!r}.")
        return (
            OptimizationCompileTarget(
                "cosyvoice.llm.transformer.forward",
                self.llm.llm.model.model,
                "forward",
            ),
            OptimizationCompileTarget(
                "cosyvoice.flow.estimator.forward",
                self.flow.decoder.estimator,
                "forward",
            ),
            OptimizationCompileTarget(
                "cosyvoice.hift.forward",
                self.hift,
                "forward",
            ),
        )

    def attach_discriminator(self, *, tiny: bool = False) -> None:
        """Attach the training-only adversarial graph exactly once."""
        if self.hifigan is None:
            self.hifigan = CosyVoiceHiFTTrainingModel(
                self.hift,
                CosyVoiceHiFTDiscriminator(tiny=tiny),
            )

    def trainable_component(self, name: str) -> nn.Module:
        normalized = str(name).strip().lower().replace("-", "_")
        aliases = {
            "language_model": "llm",
            "vocoder": "hift",
            "hifigan_generator": "hift",
            "hifigan_discriminator": "hifigan",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized == "hifigan" and self.hifigan is None:
            raise ValueError("HiFT discriminator graph is not attached.")
        try:
            component = getattr(self, normalized)
        except AttributeError as error:
            raise ValueError("CosyVoice component must be llm, flow, hift, or hifigan.") from error
        if component is None:
            raise ValueError(f"CosyVoice component {normalized!r} is unavailable.")
        return component

    def freeze_except(self, component: str) -> nn.Module:
        """Freeze unrelated components for source-style selected training."""
        selected = self.trainable_component(component)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for parameter in selected.parameters():
            parameter.requires_grad_(True)
        return selected

    def forward(
        self,
        *,
        component: str,
        **inputs: Any,
    ) -> (
            CosyVoiceLanguageOutput
            | CosyVoiceFlowOutput
            | CosyVoiceHiFTTrainingOutput
            | tuple[Tensor, Tensor]):
        normalized = str(component).strip().lower().replace("-", "_")
        if normalized in {"llm", "language_model"}:
            return self.llm(**inputs)
        if normalized == "flow":
            return self.flow(**inputs)
        if normalized in {"hift", "vocoder"}:
            return self.hift(**inputs)
        if normalized in {
                "hifigan",
                "hifigan_generator",
                "hifigan_discriminator",
        }:
            if self.hifigan is None:
                raise ValueError("Attach the training-only HiFT discriminator first.")
            phase = (
                "discriminator" if normalized.endswith("discriminator") else inputs.pop("phase", "generator"))
            return self.hifigan(phase=phase, **inputs)
        raise ValueError("Unknown CosyVoice training component.")

    @torch.inference_mode()
    def synthesize(
        self,
        text_tokens: Tensor,
        instruction_tokens: Tensor,
        speaker_embedding: Tensor,
        *,
        prompt_speech_tokens: Tensor | None = None,
        prompt_features: Tensor | None = None,
        min_new_tokens: int = 0,
        max_new_tokens: int | None = None,
        top_k: int = 25,
        top_p: float = 0.8,
        temperature: float = 1.0,
        flow_steps: int = 10,
        generator: torch.Generator | None = None,
    ) -> CosyVoiceSynthesisOutput:
        # Like the source zero-shot frontend, a speech prompt conditions both
        # the language model (tokens) and the flow (tokens plus mel features).
        has_prompt_tokens = (prompt_speech_tokens is not None and prompt_speech_tokens.numel() > 0)
        if has_prompt_tokens != (prompt_features is not None):
            raise ValueError("`prompt_speech_tokens` and `prompt_features` must be supplied together.")
        speech_tokens = self.llm.generate(
            text_tokens,
            instruction_tokens=instruction_tokens,
            prompt_speech_tokens=prompt_speech_tokens,
            min_new_tokens=min_new_tokens,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            generator=generator,
        )
        if speech_tokens.shape[1] == 0:
            raise RuntimeError("CosyVoice language model emitted no speech tokens.")
        speech_tokens = suppress_long_silences(speech_tokens)
        speech_lengths = speech_tokens.new_tensor([speech_tokens.shape[1]])
        # The source flow stage always uses its fixed noise at temperature
        # one; the LM sampling temperature and seed do not reach it.
        features = self.flow.generate(
            speech_tokens,
            speech_lengths,
            speaker_embedding,
            prompt_speech_tokens=prompt_speech_tokens,
            prompt_features=prompt_features,
            steps=flow_steps,
        )
        waveform, _ = self.hift(features)
        return CosyVoiceSynthesisOutput(
            waveform=waveform,
            speech_tokens=speech_tokens,
            speech_features=features,
            sample_rate=self.config.sample_rate,
        )


__all__ = [
    "CosyVoiceNativeModel",
    "CosyVoiceSynthesisOutput",
    "MAX_CONSECUTIVE_SILENT_TOKENS",
    "SILENT_SPEECH_TOKENS",
    "suppress_long_silences",
]
