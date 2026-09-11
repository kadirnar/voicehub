"""Source-faithful GPT-only fine-tuning for native XTTS v2."""

from __future__ import annotations

from pathlib import Path

import torch

from voicehub.optimization.protocols import OptimizationCompileTarget
from voicehub.outputs import TTSTrainingOutput
from voicehub.training.adapters import CompositeTrainingAdapter


class XTTSTrainingAdapter(CompositeTrainingAdapter):
    supports_custom_recipe = True
    native_export_semantics = "voicehub-native-xtts2-safetensors"

    def setup(self):
        super().setup()
        runtime = getattr(self.model, "native_runtime", None)
        if runtime is None:
            raise ValueError("XTTS fine-tuning requires the native runtime.")
        for parameter in runtime.parameters():
            parameter.requires_grad_(False)
        for parameter in runtime.gpt.parameters():
            parameter.requires_grad_(True)
        runtime.hifigan_decoder.eval()
        self.primary_model = runtime.gpt
        return self

    def optimization_module_roots(self, ) -> tuple[tuple[str, torch.nn.Module], ...]:
        """Expose trainable GPT and frozen waveform-token preparation."""
        self.setup()
        roots = [("xtts.gpt", self.primary_model)]
        boundary = getattr(
            self.model,
            "training_audio_encoder",
            None,
        )
        if boundary is not None:
            boundary.requires_grad_(False)
            boundary.eval()
            roots.append(("xtts.training_audio_encoder", boundary))
        return tuple(roots)

    def optimization_compile_targets(
        self,
        mode: str,
    ) -> tuple[OptimizationCompileTarget, ...]:
        """Compile every execution boundary reached by XTTS training."""
        if mode != "training":
            raise ValueError(f"Unsupported XTTS adapter optimization mode {mode!r}.")
        roots = self.optimization_module_roots()
        return tuple(
            OptimizationCompileTarget(
                f"{label}.forward",
                module,
                "forward",
            ) for label, module in roots)

    def prepare_training_inputs(self, inputs, context):
        del context
        inputs = dict(inputs)
        required = {
            "text_inputs",
            "text_lengths",
            "audio_codes",
            "wav_lengths",
        }
        if "audio_codes" not in inputs:
            waveform_names = [
                name for name in ("wav", "audio_values") if name in inputs and inputs[name] is not None
            ]
            if len(waveform_names) > 1:
                raise ValueError(
                    "XTTS raw-waveform fine-tuning accepts one of `wav` or "
                    "`audio_values`, not both.")
            if waveform_names:
                boundary = getattr(
                    self.model,
                    "training_audio_encoder",
                    None,
                )
                if boundary is None:
                    raise RuntimeError(
                        "XTTS waveform batches require the separately converted "
                        "`dvae.safetensors` and `mel_stats.safetensors` artifacts. "
                        "Configure `training_dvae_checkpoint` and "
                        "`training_mel_stats_checkpoint`; legacy pickle is never "
                        "loaded by the trainer.")
                waveform = inputs[waveform_names[0]]
                if not isinstance(waveform, torch.Tensor):
                    raise TypeError("XTTS waveform batches must be collated into a PyTorch tensor.")
                sample_rate = inputs.get(
                    "audio_sample_rate",
                    getattr(
                        getattr(
                            getattr(self.model, "_runtime_config", None),
                            "audio",
                            None,
                        ),
                        "sample_rate",
                        22_050,
                    ),
                )
                inputs["audio_codes"] = boundary(
                    waveform,
                    sample_rate=sample_rate,
                )
                if "wav_lengths" in inputs:
                    waveform_lengths = torch.as_tensor(
                        inputs["wav_lengths"],
                        dtype=torch.long,
                        device=waveform.device,
                    )
                else:
                    batch_size = (1 if waveform.ndim == 1 else waveform.shape[0])
                    waveform_lengths = torch.full(
                        (batch_size, ),
                        waveform.shape[-1],
                        dtype=torch.long,
                        device=waveform.device,
                    )
                target_rate = boundary.dvae.config.sample_rate
                if sample_rate != target_rate:
                    waveform_lengths = torch.round(
                        waveform_lengths.to(dtype=torch.float64) * target_rate /
                        sample_rate).to(dtype=torch.long)
                inputs["wav_lengths"] = waveform_lengths
        if not required <= set(inputs):
            raise ValueError(
                "Native XTTS fine-tuning requires precomputed text/audio tokens, "
                "or text tokens and a waveform plus the separately loaded frozen "
                "native DVAE boundary; waveform lengths are required or derived.", )
        allowed = required | {"cond_mels", "cond_idxs", "cond_lens", "cond_latents"}
        return {name: value for name, value in inputs.items() if name in allowed}

    def execute_training_phase(self, context):
        self.setup()
        prepared = self.prepare_batch(context.inputs, context)
        raw_text, raw_mel, logits = self.primary_model(**prepared)
        text_loss = raw_text * self.model.config.training_text_loss_weight
        mel_loss = raw_mel * self.model.config.training_mel_loss_weight
        loss = text_loss + mel_loss
        return TTSTrainingOutput(
            loss=loss,
            logits=logits,
            losses={
                "loss": loss,
                "loss_text_ce": text_loss,
                "loss_mel_ce": mel_loss,
                "raw_text_ce": raw_text,
                "raw_mel_ce": raw_mel,
            },
            metadata={
                "backend": "voicehub-native",
                "objective": "source-text-ce-plus-acoustic-token-ce",
                "gpt_trainable": True,
                "speaker_encoder_frozen": True,
                "vocoder_frozen": True,
                "dvae_boundary": "native-frozen-or-precomputed-audio-codes",
            },
            training_phase=context.phase.name,
            optimizer_names=context.phase.optimizer_names,
        )

    def save_pretrained(self, save_directory: str | Path) -> None:
        self.setup()
        self.model.export_native_pretrained(save_directory)


__all__ = ["XTTSTrainingAdapter"]
