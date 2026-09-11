"""Public VoiceHub-native OpenVoice V2 inference and training wrapper."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicehub.models._shared import finish_audio_output, resolve_torch_dtype, seeded_inference
from voicehub.models.openvoice.configuration import OpenVoiceConfig
from voicehub.models.tts import PreTrainedTTSModel
from voicehub.outputs import TTSOutput
from voicehub.processing.waveform import load_native_audio, resample_waveform


class OpenVoiceForTextToSpeech(PreTrainedTTSModel):
    """Compose native base speech with exact V2 tone-color conversion."""

    config_class = OpenVoiceConfig
    default_model_name_or_path = "myshell-ai/OpenVoiceV2"
    architecture_family = "tone-color-converter"
    native_checkpoint_format = "voicehub-openvoice-v2-v1"

    def __init__(
        self,
        config: OpenVoiceConfig | str | Path | None = None,
        *,
        model_path: str | Path | None = None,
        device: str = "auto",
        lazy_load: bool = True,
        token: str | bool | None = None,
        **config_overrides: Any,
    ) -> None:
        if token is not None and not isinstance(token, (str, bool)):
            raise TypeError("`token` must be a string, boolean, or None.")
        if isinstance(token, str) and not token.strip():
            raise ValueError("String `token` values must be non-empty.")
        self._hub_token = token
        self.runtime: Any | None = None
        self._base_model: Any | None = None
        config = self._coerce_config(
            config,
            model_path=model_path,
            **config_overrides,
        )
        super().__init__(
            config,
            device=device,
            lazy_load=lazy_load,
        )

    def _load_pretrained_model(self) -> None:
        import torch

        from voicehub.models.openvoice.native.runtime import load_openvoice_runtime

        dtype = resolve_torch_dtype(
            torch,
            self.config.dtype,
            self.device,
        )
        self.runtime = load_openvoice_runtime(
            self.config.name_or_path or self.default_model_name_or_path,
            revision=self.config.revision,
            cache_dir=self.config.cache_dir,
            token=self._hub_token,
            local_files_only=self.config.local_files_only,
            device=self.device,
            dtype=dtype,
            trust_pickle_checkpoint=self.config.trust_pickle_checkpoint,
            for_training=self.is_training_load,
        )
        self.model = self.runtime.model
        self.config.sample_rate = self.runtime.config.sample_rate

    @staticmethod
    def _finite_positive(name: str, value: Any) -> float:
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                not math.isfinite(float(value)) or value <= 0):
            raise ValueError(f"`{name}` must be finite and positive.")
        return float(value)

    @staticmethod
    def _validate_embedding(value: Any, *, name: str) -> None:
        if value is None:
            return
        try:
            import torch
        except ModuleNotFoundError:  # pragma: no cover - base runtime dependency
            return
        if isinstance(value, (str, Path)):
            if not Path(value).expanduser().is_file():
                raise FileNotFoundError(f"OpenVoice {name} file was not found: {value}.")
            return
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"`{name}` must be a tensor, local weights-only path, or None.")

    def _validate_generation_inputs(
        self,
        model_inputs: dict[str, Any],
    ) -> None:
        target_embedding = model_inputs.get("target_embedding")
        speaker_audio = model_inputs.get("speaker_audio_path")
        if target_embedding is None and speaker_audio is None:
            raise ValueError("OpenVoice requires `speaker_audio_path` or "
                             "`target_embedding`.")
        if speaker_audio is not None and isinstance(speaker_audio, (str, Path)):
            if not Path(speaker_audio).expanduser().is_file():
                raise FileNotFoundError("OpenVoice reference audio was not found: "
                                        f"{speaker_audio}.")
        self._validate_embedding(
            model_inputs.get("source_embedding"),
            name="source_embedding",
        )
        self._validate_embedding(
            target_embedding,
            name="target_embedding",
        )
        self._finite_positive("speed", model_inputs.get("speed", 1.0))
        tau = model_inputs.get("tau", 0.3)
        if (isinstance(tau, bool) or not isinstance(tau, (int, float)) or not math.isfinite(float(tau)) or
                not 0.0 <= float(tau) <= 1.0):
            raise ValueError("`tau` must be finite and in [0, 1].")
        if model_inputs.get("vad", False) is not False:
            raise ValueError(
                "Native OpenVoice does not silently call the upstream "
                "external VAD. Provide trimmed speech or run a VoiceHub VAD "
                "before generation, then pass `vad=False`.")
        requested_watermark = model_inputs.get("watermark")
        if requested_watermark is None:
            requested_watermark = self.config.watermark
        if requested_watermark is not None:
            raise ValueError(
                "OpenVoice watermarking is an explicit postprocessor. "
                "Apply VoiceHub's native WavMark strategy separately.")
        if model_inputs.get("base_audio") is None:
            if self.config.base_model_name_or_path is None:
                raise ValueError(
                    "OpenVoice is a tone-color converter. Supply `base_audio` "
                    "or configure `base_model_name_or_path` with the exact "
                    "native MeloTTS linguistic features.")
            required = (
                "input_ids",
                "tone_ids",
                "language_ids",
                "bert_features",
                "ja_bert_features",
            )
            missing = [name for name in required if model_inputs.get(name) is None]
            if missing:
                raise ValueError("Native MeloTTS base synthesis is missing: " + ", ".join(missing) + ".")

    @staticmethod
    def _load_embedding(
        value: Any,
        *,
        name: str,
        device: str,
    ):
        import torch

        if isinstance(value, (str, Path)):
            try:
                payload = torch.load(
                    Path(value).expanduser(),
                    map_location="cpu",
                    weights_only=True,
                )
            except TypeError as error:  # pragma: no cover - old PyTorch
                raise RuntimeError(
                    "OpenVoice embedding loading requires "
                    "`torch.load(..., weights_only=True)`.") from error
            if isinstance(payload, Mapping):
                candidates = [
                    payload[key] for key in ("embedding", "speaker_embedding", "se") if key in payload
                ]
                if len(candidates) != 1:
                    raise ValueError(
                        f"OpenVoice {name} mapping must expose exactly one "
                        "recognized embedding.")
                payload = candidates[0]
            value = payload
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"OpenVoice {name} must resolve to a tensor.")
        if value.ndim == 1 and value.shape[0] == 256:
            value = value[None, :, None]
        elif value.ndim == 2 and value.shape == (256, 1):
            value = value.unsqueeze(0)
        elif value.ndim == 2 and value.shape[1] == 256:
            value = value.unsqueeze(-1)
        if value.ndim != 3 or value.shape[1:] != (256, 1):
            raise ValueError(f"OpenVoice {name} must have shape [batch, 256, 1].")
        return value.to(device=device)

    def _native_base_audio(
        self,
        text: str,
        *,
        language: str,
        speaker: str | int | None,
        speed: float,
        input_ids: Any,
        tone_ids: Any,
        language_ids: Any,
        bert_features: Any,
        ja_bert_features: Any,
        seed: int | None,
    ):
        from voicehub.models.melotts import MeloTTSConfig, MeloTTSForTextToSpeech

        if self._base_model is None:
            config = MeloTTSConfig(
                name_or_path=self.config.base_model_name_or_path,
                language=language,
                trust_pickle_checkpoint=self.config.trust_pickle_checkpoint,
            )
            self._base_model = MeloTTSForTextToSpeech(
                config,
                device=self.device,
                lazy_load=True,
            )
        return self._base_model.generate(
            text,
            input_ids=input_ids,
            tone_ids=tone_ids,
            language_ids=language_ids,
            bert_features=bert_features,
            ja_bert_features=ja_bert_features,
            speaker=speaker,
            speed=speed,
            seed=seed,
        )

    def _generate(
        self,
        text: str,
        *,
        speaker_audio_path: Any = None,
        speaker_audio_sampling_rate: int | None = None,
        base_audio: Any = None,
        base_audio_sampling_rate: int | None = None,
        source_embedding: Any = None,
        target_embedding: Any = None,
        output_file: str | None = None,
        language: str = "EN",
        speaker: str | int | None = None,
        speed: float = 1.0,
        tau: float = 0.3,
        vad: bool = False,
        watermark: str | None = None,
        input_ids: Any = None,
        tone_ids: Any = None,
        language_ids: Any = None,
        bert_features: Any = None,
        ja_bert_features: Any = None,
        seed: int | None = None,
    ) -> TTSOutput:
        del vad, watermark
        provided_base_audio = base_audio is not None
        with seeded_inference(
                seed,
                device=self.device,
                model_type="openvoice",
        ) as effective_seed:
            if base_audio is None:
                generated = self._native_base_audio(
                    text,
                    language=language,
                    speaker=speaker,
                    speed=speed,
                    input_ids=input_ids,
                    tone_ids=tone_ids,
                    language_ids=language_ids,
                    bert_features=bert_features,
                    ja_bert_features=ja_bert_features,
                    seed=effective_seed,
                )
                base_audio = generated.audio
                base_audio_sampling_rate = generated.sample_rate
            base = load_native_audio(
                base_audio,
                sampling_rate=base_audio_sampling_rate,
                target_sampling_rate=self.sample_rate,
            )
            if source_embedding is None:
                source_embedding = self.runtime.extract_speaker_embedding(
                    base.waveform,
                    segment_seconds=self.config.reference_segment_seconds,
                )
            else:
                source_embedding = self._load_embedding(
                    source_embedding,
                    name="source_embedding",
                    device=self.device,
                )
            if target_embedding is None:
                target = load_native_audio(
                    speaker_audio_path,
                    sampling_rate=speaker_audio_sampling_rate,
                    target_sampling_rate=self.sample_rate,
                )
                target_embedding = self.runtime.extract_speaker_embedding(
                    target.waveform,
                    segment_seconds=self.config.reference_segment_seconds,
                )
            else:
                target_embedding = self._load_embedding(
                    target_embedding,
                    name="target_embedding",
                    device=self.device,
                )
            waveform = self.runtime.convert(
                base.waveform,
                source_embedding=source_embedding,
                target_embedding=target_embedding,
                tau=tau,
            )
        return finish_audio_output(
            waveform,
            self.sample_rate,
            output_file=output_file,
            metadata={
                "architecture": "openvoice-v2-converter",
                "base_source": ("provided-audio" if provided_base_audio else "native-melotts"),
                "language": language,
                "reference_segmentation": "equal-no-external-vad",
                "seed": effective_seed,
                "requested_seed": seed,
                "tau": float(tau),
            },
        )

    def _validate_training_runtime(self) -> None:
        if not self.config.enable_reconstructed_finetuning:
            raise ValueError(
                "The OpenVoice repository does not publish its converter "
                "training recipe. Set `enable_reconstructed_finetuning=True` "
                "to opt into VoiceHub's paired waveform reconstruction "
                "objective.")

    def _training_waveforms(
        self,
        value: Any,
        *,
        sampling_rate: int | None,
        name: str,
    ) -> Any:
        """Normalize raw training audio while preserving variable lengths."""
        if isinstance(value, (str, Path, Mapping)):
            return load_native_audio(
                value,
                sampling_rate=sampling_rate,
                target_sampling_rate=self.sample_rate,
            ).waveform
        rows = self.runtime.processor.waveforms(value)
        source_rate = self.sample_rate if sampling_rate is None else sampling_rate
        try:
            return tuple(
                resample_waveform(
                    row,
                    source_rate,
                    self.sample_rate,
                ).to(device=self.device, dtype=self.runtime.dtype) for row in rows)
        except (TypeError, ValueError, RuntimeError) as error:
            raise type(error)(f"Invalid OpenVoice {name}: {error}") from error

    def prepare_training_inputs(
        self,
        inputs: dict[str, Any],
        *,
        phase: str,
    ) -> dict[str, Any]:
        del phase
        if self.runtime is None:
            raise RuntimeError("OpenVoice runtime is not loaded for training.")
        source = inputs.get("source_audio", inputs.get("audio"))
        target = inputs.get("target_audio", inputs.get("target_waveform"))
        if source is None or target is None:
            raise ValueError(
                "OpenVoice fine-tuning requires `source_audio` and "
                "`target_audio` paired by linguistic content.")
        source = self._training_waveforms(
            source,
            sampling_rate=inputs.get(
                "source_sampling_rate",
                inputs.get("sampling_rate"),
            ),
            name="source audio",
        )
        target = self._training_waveforms(
            target,
            sampling_rate=inputs.get(
                "target_sampling_rate",
                inputs.get("sampling_rate"),
            ),
            name="target audio",
        )
        source_reference = inputs.get("source_reference_audio")
        if source_reference is not None:
            source_reference = self._training_waveforms(
                source_reference,
                sampling_rate=inputs.get(
                    "source_reference_sampling_rate",
                    inputs.get(
                        "source_sampling_rate",
                        inputs.get("sampling_rate"),
                    ),
                ),
                name="source reference audio",
            )
        target_reference = inputs.get("target_reference_audio")
        if target_reference is not None:
            target_reference = self._training_waveforms(
                target_reference,
                sampling_rate=inputs.get(
                    "target_reference_sampling_rate",
                    inputs.get(
                        "target_sampling_rate",
                        inputs.get("sampling_rate"),
                    ),
                ),
                name="target reference audio",
            )
        source_embedding = inputs.get("source_embedding")
        if source_embedding is not None:
            source_embedding = self._load_embedding(
                source_embedding,
                name="source_embedding",
                device=self.device,
            )
        target_embedding = inputs.get("target_embedding")
        if target_embedding is not None:
            target_embedding = self._load_embedding(
                target_embedding,
                name="target_embedding",
                device=self.device,
            )
        return self.runtime.prepare_training_batch(
            source_waveform=source,
            target_waveform=target,
            source_reference=source_reference,
            target_reference=target_reference,
            source_embedding=source_embedding,
            target_embedding=target_embedding,
            tau=inputs.get("tau", 0.3),
            reduction=inputs.get("reduction", "mean"),
        )

    def _save_pretrained(self, save_directory: Path) -> None:
        self.load()
        self.runtime.save_pretrained(save_directory)


OpenVoiceTTS = OpenVoiceForTextToSpeech

__all__ = [
    "OpenVoiceForTextToSpeech",
    "OpenVoiceTTS",
]
