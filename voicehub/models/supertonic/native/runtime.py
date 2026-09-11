"""Native differentiable Supertonic synthesis and fine-tuning runtime."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional

from voicehub.checkpointing import ONNXModel
from voicehub.hub import read_json_file
from voicehub.models.supertonic.native.configuration import SupertonicArchitectureConfig
from voicehub.models.supertonic.native.frontend import SupertonicStyle, SupertonicUnicodeProcessor, length_mask
from voicehub.neural.onnx import NativeONNXGraph
from voicehub.optimization.diffusion_cache import DiffusionCacheMixin
from voicehub.optimization.diffusion_sampling import (
    DiffusionGuidanceStrategy,
    DiffusionPredictionCacheMethod,
    DiffusionSamplingCompatibilityError,
    DiffusionSamplingMixin,
    DiffusionScheduleStrategy,
    DiffusionSolverStrategy,
    coerce_diffusion_sampling_config,
)
from voicehub.optimization.protocols import OptimizationCompileTarget

_SENTENCE_BOUNDARY = re.compile(
    r"(?<!Mr\.)(?<!Mrs\.)(?<!Ms\.)(?<!Dr\.)(?<!Prof\.)"
    r"(?<!Sr\.)(?<!Jr\.)(?<!Ph\.D\.)(?<!etc\.)(?<!e\.g\.)"
    r"(?<!i\.e\.)(?<!vs\.)(?<!Inc\.)(?<!Ltd\.)(?<!Co\.)"
    r"(?<!Corp\.)(?<!St\.)(?<!Ave\.)(?<!Blvd\.)"
    r"(?<!\b[A-Z]\.)(?<=[.!?])\s+")


def chunk_text(text: str, *, maximum_characters: int = 300) -> tuple[str, ...]:
    """Split paragraphs and sentences without truncating model input."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Supertonic text must be non-empty.")
    if (isinstance(maximum_characters, bool) or not isinstance(maximum_characters, int) or
            maximum_characters <= 0):
        raise ValueError("`maximum_characters` must be a positive integer.")
    chunks: list[str] = []
    paragraphs = (paragraph.strip() for paragraph in re.split(r"\n\s*\n+", text.strip()))
    for paragraph in paragraphs:
        if not paragraph:
            continue
        current = ""
        for sentence in _SENTENCE_BOUNDARY.split(paragraph):
            candidate = f"{current} {sentence}".strip()
            if not current or len(candidate) <= maximum_characters:
                current = candidate
                continue
            chunks.append(current)
            current = sentence
        if current:
            chunks.append(current)
    return tuple(chunks)


@dataclass(slots=True)
class SupertonicFineTuningOutput:
    """Loss components from the released inference graph."""

    loss: Tensor
    losses: dict[str, Tensor]
    duration: Tensor
    next_latent: Tensor | None = None
    waveform: Tensor | None = None


class NativeSupertonicRuntime(
        DiffusionCacheMixin,
        DiffusionSamplingMixin,
        nn.Module,
):
    """Own all four released graphs as trainable PyTorch modules.

    The public release omits the audio/style encoders and original
    optimizer recipe. VoiceHub therefore supports exact inference and
    explicitly reconstructed fine-tuning of the published duration,
    text-to-latent, vector-update, and vocoder graphs from precomputed
    style/latent targets.
    """

    diffusion_sampling_capabilities = frozenset({"discrete-step-count"})

    def __init__(
        self,
        *,
        architecture: SupertonicArchitectureConfig,
        processor: SupertonicUnicodeProcessor,
        duration_predictor: ONNXModel,
        text_encoder: ONNXModel,
        vector_estimator: ONNXModel,
        vocoder: ONNXModel,
    ) -> None:
        super().__init__()
        if not isinstance(architecture, SupertonicArchitectureConfig):
            raise TypeError("`architecture` must be SupertonicArchitectureConfig.")
        if not isinstance(processor, SupertonicUnicodeProcessor):
            raise TypeError("`processor` must be SupertonicUnicodeProcessor.")
        self.architecture = architecture
        self.processor = processor
        self.duration_predictor = NativeONNXGraph(duration_predictor)
        self.text_encoder = NativeONNXGraph(text_encoder)
        self.vector_estimator = NativeONNXGraph(vector_estimator)
        self.vocoder = NativeONNXGraph(vocoder)
        self._initialize_diffusion_cache()
        self._initialize_diffusion_sampling()
        self._validate_graph_contracts()

    def enable_diffusion_sampling(self, config=None):
        resolved = coerce_diffusion_sampling_config(config)
        if resolved.schedule is not DiffusionScheduleStrategy.NATIVE:
            raise DiffusionSamplingCompatibilityError(
                "Supertonic uses a discrete current_step/total_step "
                "estimator; non-native continuous schedules are not "
                "supported.")
        if resolved.guidance is not DiffusionGuidanceStrategy.NATIVE:
            raise DiffusionSamplingCompatibilityError(
                "Supertonic has no classifier-free-guidance branch to "
                "prune or adapt.")
        if (resolved.prediction_cache is not DiffusionPredictionCacheMethod.DISABLED):
            raise DiffusionSamplingCompatibilityError(
                "Supertonic returns the next absolute latent rather than "
                "a denoiser or velocity prediction; reusing that output "
                "would stall the released recurrence.")
        if resolved.solver is not DiffusionSolverStrategy.NATIVE:
            raise DiffusionSamplingCompatibilityError(
                "Supertonic does not expose the direct continuous velocity "
                "required by STORK-2.")
        return super().enable_diffusion_sampling(resolved)

    def optimization_compile_targets(
        self,
        mode: str,
    ) -> tuple[OptimizationCompileTarget, ...]:
        """Expose published graph boundaries reached by each runtime mode."""
        if mode == "training":
            return (OptimizationCompileTarget(
                "fine_tuning_loss",
                self,
                "fine_tuning_loss",
            ), )
        if mode != "inference":
            raise ValueError(f"Unsupported optimization mode {mode!r}.")
        return tuple(
            OptimizationCompileTarget(
                f"{name}.forward",
                getattr(self, name),
                "forward",
            ) for name in (
                "duration_predictor",
                "text_encoder",
                "vector_estimator",
                "vocoder",
            ))

    @property
    def sample_rate(self) -> int:
        return self.architecture.sample_rate

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def _validate_graph_contracts(self) -> None:
        expected = {
            "duration_predictor": (
                ("text_ids", "style_dp", "text_mask"),
                ("duration", ),
            ),
            "text_encoder": (
                ("text_ids", "style_ttl", "text_mask"),
                ("text_emb", ),
            ),
            "vector_estimator": (
                (
                    "noisy_latent",
                    "text_emb",
                    "style_ttl",
                    "latent_mask",
                    "text_mask",
                    "current_step",
                    "total_step",
                ),
                ("denoised_latent", ),
            ),
            "vocoder": (
                ("latent", ),
                ("wav_tts", ),
            ),
        }
        for name, (inputs, outputs) in expected.items():
            graph = getattr(self, name)
            if graph.input_names != inputs or graph.output_names != outputs:
                raise ValueError(f"Supertonic {name} I/O contract differs from the "
                                 "reviewed release.")

    def prepare_for_training(self) -> None:
        self.train()

    def prepare_for_inference(self) -> None:
        self.eval()

    @staticmethod
    def _style_batch(
        style: SupertonicStyle,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> SupertonicStyle:
        resolved = style.to(device=device, dtype=dtype)
        if resolved.ttl.shape[0] == batch_size:
            return resolved
        if resolved.ttl.shape[0] != 1:
            raise ValueError("Supertonic style batch must be one or match the text batch.")
        return SupertonicStyle(
            ttl=resolved.ttl.expand(batch_size, -1, -1),
            duration=resolved.duration.expand(batch_size, -1, -1),
        )

    def _latent_mask(self, duration: Tensor) -> tuple[Tensor, int]:
        waveform_lengths = (duration * self.sample_rate).to(dtype=torch.int64)
        hop = self.architecture.latent_hop_length
        latent_lengths = torch.div(
            waveform_lengths + hop - 1,
            hop,
            rounding_mode="floor",
        )
        maximum = int(latent_lengths.max().item())
        return length_mask(latent_lengths, maximum), maximum

    def _text_features(
        self,
        texts: tuple[str, ...],
        languages: tuple[str, ...],
        style: SupertonicStyle,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        text_ids, text_mask = self.processor.encode(
            texts,
            languages,
            device=self.device,
        )
        style = self._style_batch(
            style,
            len(texts),
            device=self.device,
            dtype=self.dtype,
        )
        text_mask = text_mask.to(dtype=self.dtype)
        duration = self.duration_predictor(
            text_ids=text_ids,
            style_dp=style.duration,
            text_mask=text_mask,
        )
        text_embedding = self.text_encoder(
            text_ids=text_ids,
            style_ttl=style.ttl,
            text_mask=text_mask,
        )
        return duration, text_embedding, text_mask, style.ttl

    def _cached_vector_step(
        self,
        noisy_latent: Tensor,
        *,
        text_embedding: Tensor,
        style_ttl: Tensor,
        latent_mask: Tensor,
        text_mask: Tensor,
        current_step: Tensor,
        total_step: Tensor,
    ) -> Tensor:
        """Run or predict the released estimator's latent residual.

        Supertonic's ONNX export flattens the estimator graph, so it has no
        stable ``ModuleList`` boundary. A timestep-tagged current latent is
        the probe and the cached quantity is ``next_latent - current_latent``;
        caching the absolute output would stall the recurrence.
        """
        step_marker = (current_step / total_step.clamp_min(1)).to(
            device=noisy_latent.device,
            dtype=noisy_latent.dtype,
        )
        while step_marker.ndim < noisy_latent.ndim:
            step_marker = step_marker.unsqueeze(-1)
        latent_channels = noisy_latent.shape[1]
        cache_state = torch.cat(
            (noisy_latent, noisy_latent + step_marker),
            dim=1,
        )

        def apply_stage(stage: str, value: Tensor) -> Tensor:
            if stage == "probe":
                return value
            if stage == "estimator":
                output = self.vector_estimator(
                    noisy_latent=value[:, :latent_channels],
                    text_emb=text_embedding,
                    style_ttl=style_ttl,
                    latent_mask=latent_mask,
                    text_mask=text_mask,
                    current_step=current_step,
                    total_step=total_step,
                )
                return torch.cat(
                    (output, value[:, latent_channels:]),
                    dim=1,
                )
            return value[:, :latent_channels]

        return self._run_diffusion_blocks(
            ("probe", "estimator", "tail"),
            cache_state,
            apply_stage,
            cache_lane="vector-estimator",
            valid_mask=latent_mask,
        )

    def infer_batch(
        self,
        texts: tuple[str, ...] | list[str],
        languages: tuple[str, ...] | list[str],
        style: SupertonicStyle,
        *,
        total_steps: int = 5,
        speed: float = 1.05,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Synthesize a batch and return padded audio plus durations."""
        texts = tuple(texts)
        languages = tuple(languages)
        if (isinstance(total_steps, bool) or not isinstance(total_steps, int) or total_steps <= 0):
            raise ValueError("`total_steps` must be a positive integer.")
        if (isinstance(speed, bool) or not isinstance(speed, (int, float)) or not math.isfinite(speed) or
                speed <= 0):
            raise ValueError("`speed` must be a finite positive number.")
        self.reset_diffusion_sampling()
        sampling_controller = self.diffusion_sampling_controller
        duration, text_embedding, text_mask, style_ttl = (self._text_features(texts, languages, style))
        duration = duration / float(speed)
        if (not torch.isfinite(duration).all() or (duration <= 0).any()):
            raise RuntimeError("Supertonic duration predictor returned invalid values.")
        latent_mask, latent_length = self._latent_mask(duration)
        latent = torch.randn(
            len(texts),
            self.architecture.latent_channels,
            latent_length,
            device=self.device,
            dtype=self.dtype,
            generator=generator,
        )
        latent = latent * latent_mask.to(dtype=self.dtype)
        effective_steps = total_steps
        if sampling_controller is not None:
            native_steps = torch.arange(
                total_steps + 1,
                device=self.device,
                dtype=torch.float32,
            )
            prepared_steps = sampling_controller.prepare_schedule(native_steps)
            effective_steps = int(prepared_steps.numel() - 1)
        total = torch.full(
            (len(texts), ),
            float(effective_steps),
            device=self.device,
            dtype=self.dtype,
        )
        self.reset_diffusion_cache()
        with self.diffusion_cache_session():
            for step in range(effective_steps):
                current = torch.full(
                    (len(texts), ),
                    float(step),
                    device=self.device,
                    dtype=self.dtype,
                )
                latent = self._cached_vector_step(
                    latent,
                    text_embedding=text_embedding,
                    style_ttl=style_ttl,
                    latent_mask=latent_mask.to(dtype=self.dtype),
                    text_mask=text_mask,
                    current_step=current,
                    total_step=total,
                )
        waveform = self.vocoder(latent=latent)
        return waveform, duration

    @torch.no_grad()
    def synthesize(
        self,
        text: str,
        language: str,
        style: SupertonicStyle,
        *,
        total_steps: int = 5,
        speed: float = 1.05,
        silence_duration: float = 0.3,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Synthesize one text, trimming every padded chunk before joining."""
        if (isinstance(silence_duration, bool) or not isinstance(silence_duration, (int, float)) or
                not math.isfinite(silence_duration) or silence_duration < 0):
            raise ValueError("`silence_duration` must be finite and non-negative.")
        maximum = 120 if language in {"ko", "ja"} else 300
        chunks = chunk_text(text, maximum_characters=maximum)
        pieces: list[Tensor] = []
        durations: list[Tensor] = []
        silence = torch.zeros(
            int(silence_duration * self.sample_rate),
            device=self.device,
            dtype=self.dtype,
        )
        for index, chunk in enumerate(chunks):
            waveform, duration = self.infer_batch(
                (chunk, ),
                (language, ),
                style,
                total_steps=total_steps,
                speed=speed,
                generator=generator,
            )
            sample_count = min(
                waveform.shape[-1],
                max(1, round(float(duration[0].item()) * self.sample_rate)),
            )
            if index and silence.numel():
                pieces.append(silence)
            pieces.append(waveform[0, :sample_count])
            durations.append(duration[0])
        total_duration = torch.stack(durations).sum()
        if len(durations) > 1:
            total_duration = (total_duration + (len(durations) - 1) * silence_duration)
        return torch.cat(pieces).unsqueeze(0), total_duration.unsqueeze(0)

    def fine_tuning_loss(
        self,
        *,
        text_ids: Tensor,
        text_mask: Tensor,
        style_ttl: Tensor,
        style_dp: Tensor,
        target_duration: Tensor | None = None,
        target_latent: Tensor | None = None,
        source_noise: Tensor | None = None,
        latent_mask: Tensor | None = None,
        current_step: Tensor | None = None,
        total_steps: int = 5,
        target_audio: Tensor | None = None,
        duration_weight: float = 1.0,
        flow_weight: float = 1.0,
        vocoder_weight: float = 1.0,
    ) -> SupertonicFineTuningOutput:
        """Fine-tune published graphs from precomputed latent supervision.

        This objective is reconstructed from the released iterative
        graph; it is not presented as Supertone's unpublished original
        training recipe.
        """
        if text_ids.ndim != 2 or text_ids.dtype != torch.int64:
            raise ValueError("`text_ids` must have shape [batch, text].")
        batch_size = text_ids.shape[0]
        if text_mask.shape != (batch_size, 1, text_ids.shape[1]):
            raise ValueError("`text_mask` must have shape [batch, 1, text].")
        style = self._style_batch(
            SupertonicStyle(style_ttl, style_dp),
            batch_size,
            device=text_ids.device,
            dtype=self.dtype,
        )
        text_mask = text_mask.to(device=self.device, dtype=self.dtype)
        text_ids = text_ids.to(device=self.device)
        predicted_duration = self.duration_predictor(
            text_ids=text_ids,
            style_dp=style.duration,
            text_mask=text_mask,
        )
        losses: dict[str, Tensor] = {}
        if target_duration is not None:
            target_duration = target_duration.to(
                device=self.device,
                dtype=self.dtype,
            ).reshape(-1)
            if target_duration.shape != predicted_duration.shape:
                raise ValueError("`target_duration` must have shape [batch].")
            losses["duration_loss"] = functional.l1_loss(
                predicted_duration,
                target_duration,
            ) * float(duration_weight)

        next_latent: Tensor | None = None
        waveform: Tensor | None = None
        if target_latent is not None:
            target_latent = target_latent.to(
                device=self.device,
                dtype=self.dtype,
            )
            if (target_latent.ndim != 3 or target_latent.shape[0] != batch_size or
                    target_latent.shape[1] != self.architecture.latent_channels):
                raise ValueError(
                    "`target_latent` must have shape "
                    f"[batch, {self.architecture.latent_channels}, time].")
            if source_noise is None:
                source_noise = torch.randn_like(target_latent)
            else:
                source_noise = source_noise.to(
                    device=self.device,
                    dtype=self.dtype,
                )
                if source_noise.shape != target_latent.shape:
                    raise ValueError("`source_noise` and `target_latent` shapes differ.")
            if latent_mask is None:
                latent_mask = torch.ones(
                    batch_size,
                    1,
                    target_latent.shape[-1],
                    device=self.device,
                    dtype=self.dtype,
                )
            else:
                latent_mask = latent_mask.to(
                    device=self.device,
                    dtype=self.dtype,
                )
                if latent_mask.shape != (
                        batch_size,
                        1,
                        target_latent.shape[-1],
                ):
                    raise ValueError("`latent_mask` must have shape [batch, 1, time].")
            if (isinstance(total_steps, bool) or not isinstance(total_steps, int) or total_steps <= 0):
                raise ValueError("`total_steps` must be a positive integer.")
            if current_step is None:
                current_step = torch.randint(
                    0,
                    total_steps,
                    (batch_size, ),
                    device=self.device,
                ).to(dtype=self.dtype)
            else:
                current_step = current_step.to(
                    device=self.device,
                    dtype=self.dtype,
                ).reshape(-1)
                if current_step.shape != (batch_size, ):
                    raise ValueError("`current_step` must have shape [batch].")
            if ((current_step < 0).any() or (current_step >= total_steps).any()):
                raise ValueError("`current_step` must lie in [0, total_steps).")
            total = torch.full(
                (batch_size, ),
                float(total_steps),
                device=self.device,
                dtype=self.dtype,
            )
            ratio = (current_step / total).reshape(batch_size, 1, 1)
            next_ratio = ((current_step + 1.0) / total).reshape(batch_size, 1, 1)
            current_latent = (source_noise + ratio * (target_latent - source_noise)) * latent_mask
            expected_next = (source_noise + next_ratio * (target_latent - source_noise)) * latent_mask
            text_embedding = self.text_encoder(
                text_ids=text_ids,
                style_ttl=style.ttl,
                text_mask=text_mask,
            )
            next_latent = self.vector_estimator(
                noisy_latent=current_latent,
                text_emb=text_embedding,
                style_ttl=style.ttl,
                latent_mask=latent_mask,
                text_mask=text_mask,
                current_step=current_step,
                total_step=total,
            )
            denominator = latent_mask.sum().clamp_min(1.0)
            losses["flow_step_loss"] = ((next_latent - expected_next).square() * latent_mask).sum() / (
                denominator * target_latent.shape[1]) * float(flow_weight)

            if target_audio is not None:
                waveform = self.vocoder(latent=target_latent)
                target_audio = target_audio.to(
                    device=self.device,
                    dtype=self.dtype,
                )
                if target_audio.ndim == 1:
                    target_audio = target_audio.unsqueeze(0)
                if target_audio.ndim != 2 or target_audio.shape[0] != batch_size:
                    raise ValueError("`target_audio` must have shape [batch, samples].")
                common = min(waveform.shape[-1], target_audio.shape[-1])
                if common <= 0:
                    raise ValueError("Vocoder targets cannot be empty.")
                losses["vocoder_l1_loss"] = functional.l1_loss(
                    waveform[..., :common],
                    target_audio[..., :common],
                ) * float(vocoder_weight)
        elif target_audio is not None:
            raise ValueError("`target_audio` requires its matching `target_latent`.")

        if not losses:
            raise ValueError(
                "Supertonic fine-tuning requires `target_duration`, "
                "`target_latent`, or both.")
        loss = torch.stack(tuple(losses.values())).sum()
        return SupertonicFineTuningOutput(
            loss=loss,
            losses=losses,
            duration=predicted_duration,
            next_latent=next_latent,
            waveform=waveform,
        )


def load_native_supertonic_runtime(
    artifacts: Any,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> NativeSupertonicRuntime:
    """Build the reviewed graph and apply an optional native weight overlay.

    ``artifacts`` is intentionally duck-typed at this low-level boundary
    to avoid importing the network-aware resolver during architecture
    discovery. Every field is still validated before graph construction.
    """
    graph_models = getattr(artifacts, "graph_models", None)
    expected_roles = {
        "duration_predictor",
        "text_encoder",
        "vector_estimator",
        "vocoder",
    }
    if not isinstance(graph_models, Mapping) or set(graph_models) != expected_roles:
        raise TypeError("Supertonic artifacts must provide all four reviewed graph models.")
    architecture_path = getattr(artifacts, "architecture_config", None)
    indexer_path = getattr(artifacts, "unicode_indexer", None)
    architecture = SupertonicArchitectureConfig.from_mapping(read_json_file(architecture_path))
    processor = SupertonicUnicodeProcessor.from_file(indexer_path)
    runtime = NativeSupertonicRuntime(
        architecture=architecture,
        processor=processor,
        duration_predictor=graph_models["duration_predictor"],
        text_encoder=graph_models["text_encoder"],
        vector_estimator=graph_models["vector_estimator"],
        vocoder=graph_models["vocoder"],
    )
    native_weights = getattr(artifacts, "native_weights", None)
    if native_weights:
        from voicehub.models.supertonic.native.checkpoint import load_supertonic_native_weights

        load_supertonic_native_weights(runtime, native_weights)
    runtime.to(device=device, dtype=dtype)
    return runtime


__all__ = [
    "NativeSupertonicRuntime",
    "SupertonicFineTuningOutput",
    "chunk_text",
    "load_native_supertonic_runtime",
]
