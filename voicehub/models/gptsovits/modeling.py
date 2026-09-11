"""VoiceHub-native GPT-SoVITS classic-S2 inference and fine-tuning."""

from __future__ import annotations

import math
import secrets
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from voicehub.dependencies import import_optional
from voicehub.models._shared import finish_audio_output, seeded_inference
from voicehub.models.gptsovits.configuration import GPTSoVITSConfig
from voicehub.models.gptsovits.native.checkpoint import (
    GPT_SOVITS_REPOSITORY,
    export_gptsovits_checkpoint,
    load_gptsovits_discriminator,
    resolve_gptsovits_artifacts,
)
from voicehub.models.gptsovits.native.modeling import build_s2_discriminator
from voicehub.models.gptsovits.native.runtime import GPTSoVITSRuntime
from voicehub.models.gptsovits.native.training import GPTSoVITSStagedTrainingModel
from voicehub.models.tts import PreTrainedTTSModel
from voicehub.outputs import TTSOutput


class GPTSoVITSForTextToSpeech(PreTrainedTTSModel):
    """Prepared-input synthesis with source-faithful staged objectives."""

    config_class = GPTSoVITSConfig
    default_model_name_or_path = GPT_SOVITS_REPOSITORY
    _KNOWN_VERSION_CONFIG_KEYS = frozenset({
        "custom",
        "v1",
        "v2",
        "v2Pro",
        "v2ProPlus",
        "v3",
        "v4",
        "v3lora",
        "v4lora",
        "LoRA",
    })

    def __init__(
        self,
        config: GPTSoVITSConfig | str | None = None,
        *,
        model_path: str | None = None,
        device: str = "auto",
        lazy_load: bool = True,
        **config_overrides: Any,
    ) -> None:
        config = self._coerce_config(
            config,
            model_path=model_path,
            **config_overrides,
        )
        super().__init__(config, device=device, lazy_load=lazy_load)
        self.training_model: GPTSoVITSStagedTrainingModel | None = None
        self._resolved_artifacts = None
        self._checkpoint_reports = None

    @classmethod
    def _normalize_runtime_config(
        cls,
        config_source: Mapping[str, Any],
    ) -> dict[str, Any]:
        normalized = dict(config_source)
        if cls._KNOWN_VERSION_CONFIG_KEYS.intersection(normalized):
            return normalized
        return {"custom": dict(normalized)}

    def _resolve_runtime_config(self) -> dict[str, Any]:
        source = self.config.runtime_config
        if source is not None:
            normalized = self._normalize_runtime_config(source)
            section_names = [name for name in self._KNOWN_VERSION_CONFIG_KEYS if name in normalized]
            selected_name = (
                "custom"
                if "custom" in section_names else section_names[0] if len(section_names) == 1 else None)
            selected = (normalized.get(selected_name) if selected_name is not None else normalized)
            if isinstance(selected, Mapping):
                selected = dict(selected)
                if selected_name is not None:
                    normalized[selected_name] = selected
                selected.setdefault(
                    "version",
                    (self.config.variant if selected_name in {None, "custom"} else selected_name),
                )
                optional_defaults = {
                    "revision": self.config.revision,
                    "cache_dir": self.config.cache_dir,
                }
                for name, value in optional_defaults.items():
                    if value is not None:
                        selected.setdefault(name, value)
                if self.config.local_files_only:
                    selected.setdefault("local_files_only", True)
                if self.config.trust_pickle_checkpoint:
                    selected.setdefault("trust_pickle_checkpoint", True)
            return normalized
        model_path = self.config.name_or_path or self.default_model_name_or_path
        return {
            "custom": {
                "artifact_root": model_path,
                "version": self.config.variant,
                "revision": self.config.revision,
                "cache_dir": self.config.cache_dir,
                "local_files_only": self.config.local_files_only,
                "trust_pickle_checkpoint": self.config.trust_pickle_checkpoint,
            },
        }

    def _load_pretrained_model(self) -> None:
        runtime = import_optional(
            "voicehub.models.gptsovits.native.runtime",
            model_type="gptsovits",
            install_extra=None,
        )
        runtime_config = runtime.TTS_Config(self._resolve_runtime_config())
        runtime_config.device = self.device
        if str(self.device).split(":", 1)[0].lower() == "cpu":
            runtime_config.is_half = False
        runtime_config.update_configs()
        self.model = runtime.TTS(runtime_config)
        self.config.variant = self._runtime_variant()
        sample_rate = getattr(self.model, "sample_rate", None)
        if sample_rate is not None:
            self.config.sample_rate = sample_rate

    def _runtime_variant(self) -> str:
        """Read the native variant while retaining legacy-shell
        compatibility."""
        s2 = getattr(self.model, "s2", None)
        native_config = getattr(s2, "config", None)
        version = getattr(native_config, "version", None)
        return self.config.variant if version is None else str(version)

    def _prepare_for_inference(self) -> None:
        if self.model is None:
            return
        evaluate = getattr(self.model, "eval", None)
        if callable(evaluate):
            evaluate()
        # Preserve compatibility with older shells while remaining harmless
        # for the native runtime's property-backed S1/S2 components.
        for component_name in (
                "t2s_model",
                "vits_model",
                "cnhuhbert_model",
                "bert_model",
                "vocoder",
        ):
            component = getattr(self.model, component_name, None)
            component_eval = getattr(component, "eval", None)
            if callable(component_eval):
                component_eval()

    def _validate_training_runtime(self) -> None:
        if not self.config.enable_native_finetuning:
            raise ValueError(
                "Set `enable_native_finetuning=True` to train the audited "
                "S1 semantic and S2 VITS/GAN stages from explicitly prepared "
                "phoneme IDs, BERT features, SSL features, spectrograms, and "
                "32 kHz waveforms.")

    def _prepare_for_training(self) -> None:
        if not isinstance(self.model, GPTSoVITSRuntime):
            raise TypeError("GPT-SoVITS training requires the native runtime.")
        discriminator = None
        if self.config.training_enable_s2_discriminator:
            loaded_artifacts = getattr(self.model, "artifacts", None)
            if loaded_artifacts is None:
                raise RuntimeError(
                    "The native GPT-SoVITS runtime did not retain its "
                    "coherent artifact set.")
            artifacts = resolve_gptsovits_artifacts(
                loaded_artifacts.source,
                variant=self.model.s2.config.version,
                revision=loaded_artifacts.revision,
                cache_dir=self.config.cache_dir,
                local_files_only=self.config.local_files_only,
                require_discriminator=True,
            )
            discriminator = build_s2_discriminator(self.model.s2.config, ).to(self.device)
            report = load_gptsovits_discriminator(
                discriminator,
                artifacts=artifacts,
                trust_pickle_checkpoint=self.config.trust_pickle_checkpoint,
            )
            self._checkpoint_reports = {"s2_discriminator": report}
            self._resolved_artifacts = artifacts
        self.training_model = GPTSoVITSStagedTrainingModel(
            s1=self.model.s1,
            s2_generator=self.model.s2,
            s2_discriminator=discriminator,
            enable_s2_discriminator=(self.config.training_enable_s2_discriminator),
        )
        self.training_model.train()
        self.model.train()

    def prepare_training_inputs(
        self,
        inputs: dict[str, Any],
        *,
        phase: str,
    ) -> dict[str, Any]:
        prepared = dict(inputs)
        prepared["phase"] = phase
        if phase == "s1":
            if "phoneme_lengths" not in prepared and "phoneme_attention_mask" in prepared:
                prepared["phoneme_lengths"] = (prepared["phoneme_attention_mask"].long().sum(-1))
            if "semantic_lengths" not in prepared and "semantic_attention_mask" in prepared:
                prepared["semantic_lengths"] = (prepared["semantic_attention_mask"].long().sum(-1))
        else:
            if "phoneme_lengths" not in prepared and "phoneme_attention_mask" in prepared:
                prepared["phoneme_lengths"] = (prepared["phoneme_attention_mask"].long().sum(-1))
            if ("spectrogram_lengths" not in prepared and "spectrogram_attention_mask" in prepared):
                prepared["spectrogram_lengths"] = (prepared["spectrogram_attention_mask"].long().sum(-1))
        return prepared

    def _validate_generation_inputs(self, model_inputs: dict[str, Any]) -> None:
        prepared_names = {
            "s1_phoneme_ids",
            "s1_bert_features",
            "s2_phoneme_ids",
            "reference_spectrogram",
            "speaker_embedding",
            "semantic_codes",
            "prompt_semantic_ids",
        }
        prepared = any(model_inputs.get(name) is not None for name in prepared_names)
        if prepared:
            missing = [
                name for name in (
                    "s1_phoneme_ids",
                    "s1_bert_features",
                    "s2_phoneme_ids",
                    "reference_spectrogram",
                ) if model_inputs.get(name) is None
            ]
            if missing:
                raise ValueError("Prepared GPT-SoVITS inference requires: " + ", ".join(missing))
        else:
            for name, description in (
                ("text_language", "the synthesis-text language"),
                ("prompt_language", "the reference-transcript language"),
            ):
                value = model_inputs.get(name)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"`{name}` must specify {description}.")
            speaker_audio_path = model_inputs.get("speaker_audio_path")
            if (not isinstance(speaker_audio_path, (str, Path)) or not str(speaker_audio_path).strip()):
                raise ValueError("`speaker_audio_path` must point to local reference audio.")
            reference = Path(speaker_audio_path).expanduser()
            if not reference.is_file():
                raise FileNotFoundError(f"GPT-SoVITS reference audio was not found: {reference}.")
        speed = model_inputs.get("speed", 1.0)
        if (not isinstance(speed, (int, float)) or isinstance(speed, bool) or not math.isfinite(speed) or
                speed <= 0):
            raise ValueError("`speed` must be a finite positive number.")
        temperature = model_inputs.get("temperature", 1.0)
        if (not isinstance(temperature, (int, float)) or isinstance(temperature, bool) or
                not math.isfinite(temperature) or temperature < 0):
            raise ValueError("`temperature` must be finite and non-negative.")
        top_k = model_inputs.get("top_k", 15)
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("`top_k` must be a positive integer.")
        top_p = model_inputs.get("top_p", 1.0)
        if (not isinstance(top_p, (int, float)) or isinstance(top_p, bool) or not math.isfinite(top_p) or
                not 0 <= top_p <= 1):
            raise ValueError("`top_p` must be in [0, 1].")
        seed = model_inputs.get("seed", -1)
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError("`seed` must be an integer.")
        if seed != -1 and not 0 <= seed < 2**32:
            raise ValueError("`seed` must be -1 or an integer in [0, 2**32).")

    @staticmethod
    def _build_request(
        text: str,
        *,
        text_language: str,
        speaker_audio_path: str,
        prompt_language: str,
        prompt_text: str,
        speed: float,
        seed: int,
        batch_size: int,
        text_split_method: str,
        parallel_inference: bool,
        top_k: int,
        top_p: float,
        temperature: float,
        s1_phoneme_ids: Any = None,
        s1_bert_features: Any = None,
        s2_phoneme_ids: Any = None,
        prompt_semantic_ids: Any = None,
        reference_spectrogram: Any = None,
        speaker_embedding: Any = None,
        semantic_codes: Any = None,
        repetition_penalty: float = 1.35,
        maximum_new_tokens: int | None = None,
        noise_scale: float = 0.5,
    ) -> dict[str, Any]:
        return {
            "text": text,
            "text_lang": text_language,
            "ref_audio_path": str(Path(speaker_audio_path).expanduser()),
            "prompt_text": prompt_text,
            "prompt_lang": prompt_language,
            "speed_factor": speed,
            "seed": seed,
            "batch_size": batch_size,
            "text_split_method": text_split_method,
            "parallel_infer": parallel_inference,
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "streaming_mode": False,
            "return_fragment": False,
            "s1_phoneme_ids": s1_phoneme_ids,
            "s1_bert_features": s1_bert_features,
            "s2_phoneme_ids": s2_phoneme_ids,
            "prompt_semantic_ids": prompt_semantic_ids,
            "reference_spectrogram": reference_spectrogram,
            "speaker_embedding": speaker_embedding,
            "semantic_codes": semantic_codes,
            "repetition_penalty": repetition_penalty,
            "maximum_new_tokens": maximum_new_tokens,
            "noise_scale": noise_scale,
        }

    def _generate(
        self,
        text: str,
        *,
        text_language: str = "",
        speaker_audio_path: str = "",
        prompt_language: str = "",
        prompt_text: str = "",
        output_file: str | None = None,
        speed: float = 1.0,
        seed: int = -1,
        batch_size: int = 1,
        text_split_method: str = "cut5",
        parallel_inference: bool = True,
        top_k: int = 15,
        top_p: float = 1.0,
        temperature: float = 1.0,
        s1_phoneme_ids: Any = None,
        s1_bert_features: Any = None,
        s2_phoneme_ids: Any = None,
        prompt_semantic_ids: Any = None,
        reference_spectrogram: Any = None,
        speaker_embedding: Any = None,
        semantic_codes: Any = None,
        repetition_penalty: float = 1.35,
        maximum_new_tokens: int | None = None,
        noise_scale: float = 0.5,
    ) -> TTSOutput:
        effective_seed = secrets.randbelow(2**32) if seed == -1 else seed
        request = self._build_request(
            text,
            text_language=text_language,
            speaker_audio_path=speaker_audio_path,
            prompt_language=prompt_language,
            prompt_text=prompt_text,
            speed=speed,
            seed=effective_seed,
            batch_size=batch_size,
            text_split_method=text_split_method,
            parallel_inference=parallel_inference,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            s1_phoneme_ids=s1_phoneme_ids,
            s1_bert_features=s1_bert_features,
            s2_phoneme_ids=s2_phoneme_ids,
            prompt_semantic_ids=prompt_semantic_ids,
            reference_spectrogram=reference_spectrogram,
            speaker_embedding=speaker_embedding,
            semantic_codes=semantic_codes,
            repetition_penalty=repetition_penalty,
            maximum_new_tokens=maximum_new_tokens,
            noise_scale=noise_scale,
        )
        with seeded_inference(
                effective_seed,
                device=self.device,
                model_type="gptsovits",
        ):
            results = list(self.model.run(request))
        if not results:
            raise RuntimeError("GPT-SoVITS returned no audio.")
        malformed = [result for result in results if not isinstance(result, tuple) or len(result) != 2]
        if malformed:
            raise RuntimeError(
                "GPT-SoVITS returned malformed audio; expected "
                "(sample_rate, waveform) pairs.")
        sample_rates = {int(sample_rate) for sample_rate, _ in results}
        if len(sample_rates) != 1:
            raise RuntimeError("GPT-SoVITS returned chunks with different sample rates.")
        self.config.sample_rate = sample_rates.pop()
        try:
            chunks = [torch.as_tensor(chunk).reshape(-1) for _, chunk in results]
        except (TypeError, ValueError, RuntimeError) as error:
            raise RuntimeError("GPT-SoVITS returned non-numeric audio.") from error
        if any(chunk.numel() == 0 for chunk in chunks):
            raise RuntimeError("GPT-SoVITS returned an empty audio chunk.")
        return finish_audio_output(
            torch.cat(chunks),
            self.sample_rate,
            output_file=output_file,
            metadata={
                "seed": effective_seed,
                "version": self._runtime_variant(),
                "frontend": ("prepared-native" if s1_phoneme_ids is not None else "unsupported-raw"),
            },
        )

    def _save_pretrained(self, save_directory: Path) -> None:
        self.load()
        if not isinstance(self.model, GPTSoVITSRuntime):
            raise TypeError("GPT-SoVITS export requires the native runtime.")
        discriminator = (
            None if self.training_model is None or self.training_model.s2 is None else
            self.training_model.s2.discriminator)
        export_gptsovits_checkpoint(
            save_directory,
            s1=self.model.s1,
            s2_generator=self.model.s2,
            s2_discriminator=discriminator,
            s1_config=self.model.s1.config,
            s2_config=self.model.s2.config,
            source_revision=(None if self._resolved_artifacts is None else self._resolved_artifacts.revision),
        )


GPTSoVITSTTS = GPTSoVITSForTextToSpeech

__all__ = [
    "GPTSoVITSConfig",
    "GPTSoVITSForTextToSpeech",
    "GPTSoVITSTTS",
]
