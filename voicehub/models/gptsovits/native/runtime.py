"""Native prepared-input GPT-SoVITS classic-S2 inference runtime."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from voicehub.models.gptsovits.native.checkpoint import load_gptsovits_checkpoints, resolve_gptsovits_artifacts
from voicehub.models.gptsovits.native.configuration import SUPPORTED_GPT_SOVITS_VARIANTS, normalize_gptsovits_variant
from voicehub.models.gptsovits.native.frontend import reject_raw_text, validate_prepared_inference
from voicehub.models.gptsovits.native.modeling import GPTSoVITSSynthesizer, build_s2_generator
from voicehub.models.gptsovits.native.semantic import GPTSoVITSSemanticModel
from voicehub.models.vits.native.weight_norm import (
    LegacyWeightNormInferenceCache,
    enable_legacy_weight_norm_inference_cache,
)
from voicehub.optimization.protocols import OptimizationCompileTarget


class GPTSoVITSRuntime(nn.Module):
    """Compose the exact S1 semantic and S2 decoder checkpoints."""

    def __init__(
        self,
        s1: GPTSoVITSSemanticModel,
        s2: GPTSoVITSSynthesizer,
    ) -> None:
        super().__init__()
        if not isinstance(s1, GPTSoVITSSemanticModel):
            raise TypeError("GPT-SoVITS runtime requires the native S1 model.")
        if not isinstance(s2, GPTSoVITSSynthesizer):
            raise TypeError("GPT-SoVITS runtime requires the native S2 model.")
        self.s1 = s1
        self.s2 = s2
        self.sample_rate = s2.config.sample_rate
        self._weight_norm_cache: LegacyWeightNormInferenceCache | None = None
        if not s2.training:
            self._weight_norm_cache = (enable_legacy_weight_norm_inference_cache(s2))

    def train(self, mode: bool = True) -> GPTSoVITSRuntime:
        super().train(mode)
        if mode:
            if self._weight_norm_cache is not None:
                self._weight_norm_cache.restore()
                self._weight_norm_cache = None
        elif self._weight_norm_cache is None:
            self._weight_norm_cache = (enable_legacy_weight_norm_inference_cache(self.s2))
        return self

    def optimization_compile_targets(
        self,
        mode: str,
    ) -> tuple[OptimizationCompileTarget, ...]:
        """Expose prepared synthesis; staged training owns another graph."""
        if mode == "training":
            return ()
        if mode != "inference":
            raise ValueError(f"Unsupported optimization mode {mode!r}.")
        return (OptimizationCompileTarget(
            "gptsovits.synthesize_prepared",
            self,
            "synthesize_prepared",
        ), )

    @property
    def t2s_model(self) -> GPTSoVITSSemanticModel:
        return self.s1

    @property
    def vits_model(self) -> GPTSoVITSSynthesizer:
        return self.s2

    @torch.inference_mode()
    def synthesize_prepared(
        self,
        *,
        s1_phoneme_ids: Any,
        s1_bert_features: Any,
        s2_phoneme_ids: Any,
        prompt_semantic_ids: Any | None,
        reference_spectrogram: Any,
        speaker_embedding: Any | None = None,
        semantic_codes: Any | None = None,
        top_k: int = 15,
        top_p: float = 1.0,
        temperature: float = 1.0,
        repetition_penalty: float = 1.35,
        maximum_new_tokens: int | None = None,
        noise_scale: float = 0.5,
        speed: float = 1.0,
    ) -> tuple[int, Tensor]:
        reference = next(self.parameters())
        prepared = validate_prepared_inference(
            s1_phoneme_ids=s1_phoneme_ids,
            s1_bert_features=s1_bert_features,
            s2_phoneme_ids=s2_phoneme_ids,
            prompt_semantic_ids=prompt_semantic_ids,
            reference_spectrogram=reference_spectrogram,
            speaker_embedding=speaker_embedding,
            semantic_codes=semantic_codes,
            s1_config=self.s1.config,
            s2_config=self.s2.config,
            device=reference.device,
            dtype=reference.dtype,
        )
        codes = prepared["semantic_codes"]
        if codes is None:
            generated = self.s1.generate(
                phoneme_ids=prepared["s1_phoneme_ids"],
                phoneme_lengths=torch.tensor(
                    [prepared["s1_phoneme_ids"].shape[1]],
                    device=reference.device,
                ),
                bert_features=prepared["s1_bert_features"],
                prompt_semantic_ids=prepared["prompt_semantic_ids"],
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                maximum_new_tokens=maximum_new_tokens,
            )
            codes = generated.unsqueeze(1)
        waveform = self.s2.decode(
            codes,
            prepared["s2_phoneme_ids"],
            prepared["reference_spectrogram"],
            speaker_embedding=prepared["speaker_embedding"],
            noise_scale=noise_scale,
            speed=speed,
        )
        return self.sample_rate, waveform[0, 0].float()

    def synthesize(self, text: str, **prepared: Any) -> tuple[int, Tensor]:
        if not prepared:
            reject_raw_text(text)
        return self.synthesize_prepared(**prepared)


class TTS_Config:
    """Small compatibility shell for VoiceHub's stable public wrapper."""

    def __init__(self, source: Mapping[str, Any] | str) -> None:
        if isinstance(source, str):
            raise ValueError(
                "Native GPT-SoVITS no longer accepts upstream YAML. Supply a "
                "VoiceHub native artifact root and explicit prepared inputs.")
        if not isinstance(source, Mapping):
            raise TypeError("GPT-SoVITS runtime config must be a mapping.")
        known_sections = (
            "v1",
            "v2",
            "v2Pro",
            "v2ProPlus",
            "v3",
            "v4",
            "v3lora",
            "v4lora",
            "LoRA",
        )
        variant_keys = [name for name in known_sections if name in source]
        unsupported = sorted(set(variant_keys) - set(SUPPORTED_GPT_SOVITS_VARIANTS))
        if unsupported:
            raise ValueError(
                "GPT-SoVITS V3/V4/LoRA require the separate native "
                "flow-matching "
                f"runtime; unsupported classic-S2 sections: {unsupported}.")
        supported_sections = [name for name in variant_keys if name in SUPPORTED_GPT_SOVITS_VARIANTS]
        if len(supported_sections) > 1:
            raise ValueError(
                "Select exactly one GPT-SoVITS runtime variant; received "
                f"{supported_sections}.")
        if "custom" in source and supported_sections:
            raise ValueError("GPT-SoVITS `custom` cannot be combined with a version section.")
        if "custom" in source:
            selected = source["custom"]
            section_variant = None
        elif supported_sections:
            section_variant = supported_sections[0]
            selected = source[section_variant]
        else:
            section_variant = None
            selected = source
        if not isinstance(selected, Mapping):
            raise TypeError("The selected GPT-SoVITS runtime config must be a mapping.")
        version = normalize_gptsovits_variant(selected.get("version", section_variant or "v2"))
        if section_variant is not None and version != section_variant:
            raise ValueError(
                f"GPT-SoVITS section {section_variant!r} conflicts with "
                f"declared version {version!r}.")
        artifact_root = selected.get("artifact_root")
        if not isinstance(artifact_root, (str, Path)) or not str(artifact_root):
            raise ValueError(
                "Native GPT-SoVITS requires `artifact_root`; separate arbitrary "
                "S1/S2 paths cannot establish a coherent audited release.")
        self.artifact_root = str(artifact_root)
        self.variant = version
        self.revision = selected.get("revision")
        self.cache_dir = selected.get("cache_dir")
        self.local_files_only = bool(selected.get("local_files_only", False))
        self.trust_pickle_checkpoint = bool(selected.get("trust_pickle_checkpoint", False))
        self.device = selected.get("device", "cpu")
        self.is_half = bool(selected.get("is_half", False))

    def update_configs(self) -> None:
        if str(self.device).split(":", 1)[0].lower() == "cpu":
            self.is_half = False


class TTS(GPTSoVITSRuntime):
    """Compatibility constructor which still resolves only native
    components."""

    def __init__(self, config: TTS_Config) -> None:
        artifacts = resolve_gptsovits_artifacts(
            config.artifact_root,
            variant=config.variant,
            revision=config.revision,
            cache_dir=config.cache_dir,
            local_files_only=config.local_files_only,
        )
        s1 = GPTSoVITSSemanticModel(artifacts.s1_config)
        s2 = build_s2_generator(artifacts.s2_config)
        load_gptsovits_checkpoints(
            s1=s1,
            s2_generator=s2,
            artifacts=artifacts,
            trust_pickle_checkpoint=config.trust_pickle_checkpoint,
        )
        dtype = torch.float16 if config.is_half else torch.float32
        s1.to(device=config.device, dtype=dtype).eval()
        s2.to(device=config.device, dtype=dtype).eval()
        super().__init__(s1, s2)
        self.artifacts = artifacts

    def run(self, request: Mapping[str, Any]):
        if not isinstance(request, Mapping):
            raise TypeError("GPT-SoVITS request must be a mapping.")
        unsupported = {
            "ref_audio_path",
            "prompt_text",
            "prompt_lang",
            "text_lang",
        }
        if not all(name in request for name in (
                "s1_phoneme_ids",
                "s1_bert_features",
                "s2_phoneme_ids",
                "reference_spectrogram",
        )):
            present = sorted(name for name in unsupported if request.get(name))
            raise ValueError(
                "Native GPT-SoVITS requires prepared IDs/features/spectrograms; "
                f"raw upstream fields cannot be reproduced exactly: {present}.")
        sample_rate, waveform = self.synthesize_prepared(
            s1_phoneme_ids=request["s1_phoneme_ids"],
            s1_bert_features=request["s1_bert_features"],
            s2_phoneme_ids=request["s2_phoneme_ids"],
            prompt_semantic_ids=request.get("prompt_semantic_ids"),
            reference_spectrogram=request["reference_spectrogram"],
            speaker_embedding=request.get("speaker_embedding"),
            semantic_codes=request.get("semantic_codes"),
            top_k=request.get("top_k", 15),
            top_p=request.get("top_p", 1.0),
            temperature=request.get("temperature", 1.0),
            repetition_penalty=request.get("repetition_penalty", 1.35),
            maximum_new_tokens=request.get("maximum_new_tokens"),
            noise_scale=request.get("noise_scale", 0.5),
            speed=request.get("speed_factor", 1.0),
        )
        yield sample_rate, waveform


__all__ = ["GPTSoVITSRuntime", "TTS", "TTS_Config"]
