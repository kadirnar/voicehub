"""Public MeloTTS API backed by VoiceHub's native VITS2 graph."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicehub.models._shared import finish_audio_output, resolve_model_directory, resolve_torch_dtype, seeded_inference
from voicehub.models.melotts.configuration import MeloTTSConfig
from voicehub.models.tts import PreTrainedTTSModel
from voicehub.outputs import TTSOutput


class MeloTTSForTextToSpeech(PreTrainedTTSModel):
    """Multilingual MeloTTS synthesis without a provider runtime."""

    config_class = MeloTTSConfig
    default_model_name_or_path = "EN"

    def __init__(
        self,
        config: MeloTTSConfig | str | Path | None = None,
        *,
        model_path: str | Path | None = None,
        device: str = "auto",
        lazy_load: bool = True,
        **config_overrides: Any,
    ) -> None:
        explicit_model_source = (
            model_path is not None or isinstance(config, (str, Path)) or
            (isinstance(config, MeloTTSConfig) and bool(config.name_or_path)))
        config = self._coerce_config(
            config,
            model_path=model_path,
            **config_overrides,
        )
        if (explicit_model_source and not self._looks_like_checkpoint_source(config.name_or_path)):
            config.language = config.name_or_path.upper()
        elif not explicit_model_source:
            config.name_or_path = config.language
        super().__init__(config, device=device, lazy_load=lazy_load)
        self.architecture_config = None
        self.training_model = None

    @staticmethod
    def _looks_like_checkpoint_source(name_or_path: str) -> bool:
        source = Path(name_or_path).expanduser()
        return (source.exists() or "/" in name_or_path or "\\" in name_or_path)

    def _resolve_checkpoint_paths(self) -> tuple[str | None, str | None]:
        """Retain the historical local-path helper without provider imports."""
        config_path = self.config.config_path
        checkpoint_path = self.config.checkpoint_path
        if config_path is not None and checkpoint_path is not None:
            return (
                str(Path(config_path).expanduser().resolve()),
                str(Path(checkpoint_path).expanduser().resolve()),
            )
        if not self._looks_like_checkpoint_source(self.config.name_or_path):
            return (
                None if config_path is None else str(config_path),
                None if checkpoint_path is None else str(checkpoint_path),
            )

        source = Path(self.config.name_or_path).expanduser()
        if source.is_file():
            model_directory = source.parent.resolve()
            if source.suffix.lower() == ".json":
                config_path = config_path or str(source.resolve())
            else:
                checkpoint_path = checkpoint_path or str(source.resolve())
        else:
            model_directory = resolve_model_directory(
                self.config.name_or_path,
                model_type="melotts",
            )
        if config_path is None:
            candidate = model_directory / "config.json"
            if not candidate.is_file():
                raise FileNotFoundError(f"MeloTTS configuration was not found: {candidate}.")
            config_path = str(candidate.resolve())
        if checkpoint_path is None:
            if self.config.checkpoint_filename is not None:
                candidate = (model_directory / self.config.checkpoint_filename)
            else:
                safe = model_directory / "model.safetensors"
                candidate = (safe if safe.is_file() else model_directory / "checkpoint.pth")
            if not candidate.is_file():
                raise FileNotFoundError(f"MeloTTS checkpoint was not found: {candidate}.")
            checkpoint_path = str(candidate.resolve())
        return str(config_path), str(checkpoint_path)

    def _load_pretrained_model(self) -> None:
        import torch

        from voicehub.models.melotts.native.runtime import MeloTTSRuntime

        config_path, checkpoint_path = self._resolve_checkpoint_paths()
        dtype = resolve_torch_dtype(
            torch,
            self.config.dtype,
            self.device,
        )
        self.model = MeloTTSRuntime(
            self.config.name_or_path or self.config.language,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            checkpoint_filename=self.config.checkpoint_filename,
            revision=self.config.revision,
            device=self.device,
            dtype=dtype,
            trust_pickle_checkpoint=self.config.trust_pickle_checkpoint,
        )
        self.architecture_config = self.model.config
        self.config.sample_rate = self.model.sample_rate

    @property
    def speakers(self) -> tuple[str, ...]:
        self.load()
        return self.model.speakers

    @staticmethod
    def _resolve_speaker_id(
        speaker_ids: Mapping[str, int],
        speaker: str | int | None,
    ) -> int:
        """Compatibility helper used without loading a checkpoint."""
        if not speaker_ids:
            raise RuntimeError("The loaded MeloTTS checkpoint defines no speakers.")
        if speaker is None:
            return int(next(iter(speaker_ids.values())))
        if isinstance(speaker, int) and not isinstance(speaker, bool):
            if speaker not in speaker_ids.values():
                available = ", ".join(str(value) for value in speaker_ids.values())
                raise ValueError(f"Unknown speaker ID {speaker}. Available IDs: {available}.")
            return speaker
        if not isinstance(speaker, str):
            raise TypeError("`speaker` must be a speaker name, integer ID, or None.")
        try:
            return int(speaker_ids[speaker])
        except KeyError as error:
            available = ", ".join(speaker_ids)
            raise ValueError(f"Unknown speaker {speaker!r}. Available speakers: {available}.") from error

    def _validate_training_runtime(self) -> None:
        if not self.config.enable_native_finetuning:
            raise ValueError(
                "Set `enable_native_finetuning=True` to fine-tune MeloTTS "
                "with explicit phones, tones, language IDs, BERT features, "
                "spectrograms, speaker IDs, and waveforms.")

    def _prepare_for_training(self) -> None:
        from voicehub.models.melotts.native.runtime import MeloTTSRuntime
        from voicehub.models.melotts.native.training import MeloTTSTrainingModel

        if not isinstance(self.model, MeloTTSRuntime):
            raise TypeError("MeloTTS training requires the native runtime.")
        self.model.train()
        self.training_model = MeloTTSTrainingModel(
            self.model.model,
            self.model.config,
            enable_discriminators=(self.config.training_enable_discriminators),
        )

    def prepare_training_inputs(
        self,
        inputs: dict[str, Any],
        *,
        phase: str,
    ) -> dict[str, Any]:
        prepared = dict(inputs)
        prepared["phase"] = phase
        return prepared

    def _validate_generation_inputs(
        self,
        model_inputs: dict[str, Any],
    ) -> None:
        speaker = model_inputs.get("speaker")
        if speaker is not None and (isinstance(speaker, bool) or not isinstance(speaker, (str, int))):
            raise TypeError("`speaker` must be a speaker name, integer ID, or None.")
        if isinstance(speaker, str) and not speaker.strip():
            raise ValueError("`speaker` must not be an empty string.")
        for name, default in (
            ("speed", 1.0),
            ("sdp_ratio", 0.2),
            ("noise_scale", 0.6),
            ("noise_scale_w", 0.8),
        ):
            value = model_inputs.get(name, default)
            if (not isinstance(value, (int, float)) or isinstance(value, bool)):
                raise TypeError(f"`{name}` must be numeric.")
            if not math.isfinite(value):
                raise ValueError(f"`{name}` must be finite.")
            if name == "speed" and value <= 0:
                raise ValueError("`speed` must be greater than zero.")
            if name != "speed" and value < 0:
                raise ValueError(f"`{name}` must be non-negative.")
            if name == "sdp_ratio" and value > 1:
                raise ValueError("`sdp_ratio` must be in [0, 1].")
        max_frames = model_inputs.get("max_frames", 4_096)
        if max_frames is not None and (isinstance(max_frames, bool) or not isinstance(max_frames, int) or
                                       max_frames < 1):
            raise ValueError("`max_frames` must be a positive integer or None.")
        seed = model_inputs.get("seed")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise TypeError("`seed` must be an integer or None.")

        from voicehub.models.melotts.native.frontend import validate_feature_mapping

        validate_feature_mapping(model_inputs)

    def _generate(
        self,
        text: str,
        *,
        output_file: str | None = None,
        input_ids: Any = None,
        tone_ids: Any = None,
        language_ids: Any = None,
        bert_features: Any = None,
        ja_bert_features: Any = None,
        speaker: str | int | None = None,
        speed: float = 1.0,
        sdp_ratio: float = 0.2,
        noise_scale: float = 0.6,
        noise_scale_w: float = 0.8,
        max_frames: int | None = 4_096,
        seed: int | None = None,
    ) -> TTSOutput:
        del text
        with seeded_inference(
                seed,
                device=self.device,
                model_type="melotts",
        ) as effective_seed:
            audio = self.model.generate(
                input_ids=input_ids,
                tone_ids=tone_ids,
                language_ids=language_ids,
                bert_features=bert_features,
                ja_bert_features=ja_bert_features,
                speaker=speaker,
                speed=speed,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                max_frames=max_frames,
            )
        speaker_id = self.model.frontend.resolve_speaker(speaker)
        return finish_audio_output(
            audio,
            self.sample_rate,
            output_file=output_file,
            metadata={
                "speaker_id": speaker_id,
                "speed": speed,
                "seed": effective_seed,
                "requested_seed": seed,
                "frontend": "precomputed-linguistic-features",
            },
        )

    def create_native_training_model(
        self,
        *,
        enable_discriminators: bool = True,
        loss_weights: Any = None,
    ):
        self.load()
        from voicehub.models.melotts.native.training import MeloTTSTrainingModel

        self.training_model = MeloTTSTrainingModel(
            self.model.model,
            self.model.config,
            enable_discriminators=enable_discriminators,
            loss_weights=loss_weights,
        )
        return self.training_model

    def _save_pretrained(self, save_directory: Path) -> None:
        self.load()
        from voicehub.models.melotts.native.checkpoint import save_melotts_pretrained

        save_melotts_pretrained(
            self.model.model,
            self.model.config,
            save_directory,
        )


MeloTTS = MeloTTSForTextToSpeech

__all__ = [
    "MeloTTS",
    "MeloTTSConfig",
    "MeloTTSForTextToSpeech",
]
