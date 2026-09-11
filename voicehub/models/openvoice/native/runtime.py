"""Native OpenVoice V2 loading, inference, training, and export lifecycle."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from voicehub.hub import write_json_file
from voicehub.models.openvoice.native.artifacts import OpenVoiceArtifacts, resolve_openvoice_artifacts
from voicehub.models.openvoice.native.checkpoint import load_openvoice_checkpoint, save_openvoice_checkpoint
from voicehub.models.openvoice.native.modeling import OpenVoiceConverterOutput, OpenVoiceToneColorConverter
from voicehub.models.openvoice.native.processing import OpenVoiceAudioProcessor
from voicehub.models.vits.native.weight_norm import (
    LegacyWeightNormInferenceCache,
    enable_legacy_weight_norm_inference_cache,
)
from voicehub.optimization.protocols import OptimizationCompileTarget, OptimizationModuleRoot


class OpenVoiceRuntime:
    """Loaded converter graph and its exact audio processor."""

    def __init__(
        self,
        *,
        model: OpenVoiceToneColorConverter,
        processor: OpenVoiceAudioProcessor,
        artifacts: OpenVoiceArtifacts,
    ) -> None:
        if not isinstance(model, OpenVoiceToneColorConverter):
            raise TypeError("`model` must be OpenVoiceToneColorConverter.")
        if not isinstance(processor, OpenVoiceAudioProcessor):
            raise TypeError("`processor` must be OpenVoiceAudioProcessor.")
        self.model = model
        self.processor = processor
        self.artifacts = artifacts
        self.config = model.config
        self._weight_norm_cache: LegacyWeightNormInferenceCache | None = None
        if not model.training:
            self._weight_norm_cache = (enable_legacy_weight_norm_inference_cache(model))

    def train(self, mode: bool = True) -> OpenVoiceRuntime:
        self.model.train(mode)
        if mode:
            if self._weight_norm_cache is not None:
                self._weight_norm_cache.restore()
                self._weight_norm_cache = None
        elif self._weight_norm_cache is None:
            self._weight_norm_cache = (enable_legacy_weight_norm_inference_cache(self.model))
        return self

    def eval(self) -> OpenVoiceRuntime:
        return self.train(False)

    def optimization_compile_targets(
        self,
        mode: str,
    ) -> tuple[OptimizationCompileTarget, ...]:
        """Compile the tensor graph while leaving audio preparation eager."""
        if mode not in {"inference", "training"}:
            raise ValueError(f"Unsupported optimization mode {mode!r}.")
        return (OptimizationCompileTarget(
            "model.forward",
            self.model,
            "forward",
        ), )

    def optimization_module_roots(self) -> tuple[OptimizationModuleRoot, ...]:
        return (OptimizationModuleRoot("model", self.model), )

    def state_dict(self):
        return self.model.state_dict()

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    @torch.inference_mode()
    def extract_speaker_embedding(
        self,
        reference_waveform: Any,
        *,
        segment_seconds: float = 10.0,
    ) -> Tensor:
        """Extract and average embeddings without an external VAD runtime."""
        waveforms = self.processor.waveforms(reference_waveform)
        segments = []
        for waveform in waveforms:
            segments.extend(
                self.processor.equal_reference_segments(
                    waveform,
                    segment_seconds=segment_seconds,
                ))
        embeddings = []
        for segment in segments:
            batch = self.processor.spectrogram(
                segment,
                device=self.device,
                dtype=self.dtype,
            )
            embeddings.append(self.model.extract_speaker_embedding(batch.values))
        if not embeddings:
            raise ValueError("OpenVoice reference processing produced no speech segments.")
        return torch.stack(embeddings).mean(dim=0)

    def convert(
        self,
        source_waveform: Any,
        *,
        source_embedding: Tensor | None,
        target_embedding: Tensor,
        tau: float = 0.3,
    ) -> Tensor:
        """Convert one waveform and return finite CPU float32 audio."""
        batch = self.processor.spectrogram(
            source_waveform,
            device=self.device,
            dtype=self.dtype,
        )
        if batch.values.shape[0] != 1:
            raise ValueError("One OpenVoice conversion request accepts one source waveform.")
        with torch.inference_mode():
            if source_embedding is None:
                source_embedding = self.model.extract_speaker_embedding(batch.values)
            output = self.model(
                batch.values,
                batch.lengths,
                source_embedding,
                target_embedding,
                tau=tau,
            )
        waveform = output.waveform[0, 0].detach().float().cpu().contiguous()
        if waveform.numel() == 0:
            raise RuntimeError("OpenVoice produced an empty waveform.")
        if not bool(torch.isfinite(waveform).all()):
            raise RuntimeError("OpenVoice produced NaN or infinite audio.")
        return waveform

    def training_forward(
        self,
        *,
        source_waveform: Any,
        target_waveform: Any,
        source_reference: Any | None = None,
        target_reference: Any | None = None,
        source_embedding: Tensor | None = None,
        target_embedding: Tensor | None = None,
        tau: float = 0.3,
        reduction: str = "mean",
    ) -> OpenVoiceConverterOutput:
        """Run the reconstructed paired-conversion fine-tuning objective."""
        prepared = self.prepare_training_batch(
            source_waveform=source_waveform,
            target_waveform=target_waveform,
            source_reference=source_reference,
            target_reference=target_reference,
            source_embedding=source_embedding,
            target_embedding=target_embedding,
            tau=tau,
            reduction=reduction,
        )
        return self.model(**prepared)

    def prepare_training_batch(
        self,
        *,
        source_waveform: Any,
        target_waveform: Any,
        source_reference: Any | None = None,
        target_reference: Any | None = None,
        source_embedding: Tensor | None = None,
        target_embedding: Tensor | None = None,
        tau: float = 0.3,
        reduction: str = "mean",
    ) -> dict[str, Any]:
        """Prepare raw paired audio without executing the trainable graph."""
        source = self.processor.spectrogram(
            source_waveform,
            device=self.device,
            dtype=self.dtype,
        )
        targets = self.processor.waveform_batch(
            target_waveform,
            device=self.device,
            dtype=self.dtype,
        )
        if targets.values.shape[0] != source.values.shape[0]:
            raise ValueError("OpenVoice source and target training batches must align.")
        prepared: dict[str, Any] = {
            "source_spectrogram": source.values,
            "source_lengths": source.lengths,
            "source_embedding": source_embedding,
            "target_embedding": target_embedding,
            "target_waveform": targets.values,
            "target_lengths": targets.lengths,
            "tau": tau,
            "reduction": reduction,
        }
        if source_embedding is None:
            if source_reference is None:
                source_reference_batch = source
            else:
                source_reference_batch = self.processor.spectrogram(
                    source_reference,
                    device=self.device,
                    dtype=self.dtype,
                )
            prepared.update(
                source_reference_spectrogram=source_reference_batch.values,
                source_reference_lengths=source_reference_batch.lengths,
            )
        if target_embedding is None:
            target_reference_batch = self.processor.spectrogram(
                target_waveform if target_reference is None else target_reference,
                device=self.device,
                dtype=self.dtype,
            )
            prepared.update(
                target_reference_spectrogram=target_reference_batch.values,
                target_reference_lengths=target_reference_batch.lengths,
            )
        return prepared

    def save_pretrained(self, destination: str | Path) -> Path:
        """Atomically export a new local native artifact directory."""
        destination = Path(destination).expanduser().resolve()
        if destination.exists():
            if not destination.is_dir() or any(destination.iterdir()):
                raise FileExistsError(
                    "OpenVoice export destination must be absent or an empty "
                    f"directory: {destination}.")
            destination.rmdir()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
                prefix=".voicehub-openvoice-",
                dir=destination.parent,
        ) as temporary:
            staging = Path(temporary)
            write_json_file(staging / "config.json", self.config.to_dict())
            save_openvoice_checkpoint(
                self.model,
                staging / "model.safetensors",
            )
            staging.replace(destination)
        return destination


def load_openvoice_runtime(
    source: str | Path,
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
    trust_pickle_checkpoint: bool = False,
    for_training: bool = False,
) -> OpenVoiceRuntime:
    """Build and strictly load one native converter runtime."""
    artifacts = resolve_openvoice_artifacts(
        source,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    model = OpenVoiceToneColorConverter(artifacts.config)
    load_openvoice_checkpoint(
        model,
        artifacts,
        device=device,
        dtype=dtype,
        trust_pickle_checkpoint=trust_pickle_checkpoint,
    )
    model.train(for_training)
    return OpenVoiceRuntime(
        model=model,
        processor=OpenVoiceAudioProcessor(artifacts.config),
        artifacts=artifacts,
    )


__all__ = [
    "OpenVoiceRuntime",
    "load_openvoice_runtime",
]
