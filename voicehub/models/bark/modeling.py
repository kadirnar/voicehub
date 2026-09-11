"""VoiceHub-native Bark inference and stage-specific fine-tuning."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from numbers import Integral, Real
from pathlib import Path
from typing import Any

from voicehub.checkpointing import SafeTensorReader
from voicehub.dependencies import import_optional
from voicehub.models._shared import finish_audio_output, resolve_torch_dtype, seeded_inference
from voicehub.models.tts import PreTrainedTTSModel
from voicehub.outputs import TTSOutput

from .configuration import BarkConfig


def _build_bark_training_model(torch: Any, model: Any):
    """Compatibility factory returning VoiceHub's native training graph."""
    del torch
    from voicehub.models.bark.native.training import BarkTrainingModel

    return BarkTrainingModel.from_model(model)


class BarkForTextToSpeech(PreTrainedTTSModel):
    """Generate speech and fine-tune Bark without Transformers."""

    config_class = BarkConfig
    default_model_name_or_path = "suno/bark-small"
    passthrough_generation_options = frozenset({
        "coarse_do_sample",
        "coarse_temperature",
        "coarse_top_k",
        "coarse_top_p",
        "fine_temperature",
        "min_eos_p",
        "semantic_do_sample",
        "semantic_temperature",
        "semantic_top_k",
        "semantic_top_p",
    })
    # Kept as descriptive compatibility attributes; they are never imported.
    transformers_model_class = "BarkModel"
    transformers_processor_class = "BarkProcessor"

    def __init__(
        self,
        config: BarkConfig | str | Path | None = None,
        *,
        model_path: str | Path | None = None,
        device: str = "auto",
        lazy_load: bool = True,
        token: str | bool | None = None,
        **config_overrides: Any,
    ) -> None:
        trust_official_pickle = config_overrides.pop(
            "trust_official_pickle",
            False,
        )
        config = self._coerce_config(
            config,
            model_path=model_path,
            **config_overrides,
        )
        if not isinstance(config, BarkConfig):
            raise TypeError("Bark requires a BarkConfig.")
        config.validate()
        if token is not None and (not isinstance(token, (str, bool)) or
                                  isinstance(token, str) and not token.strip()):
            raise ValueError("`token` must be a non-empty string, boolean, or None.")
        if not isinstance(trust_official_pickle, bool):
            raise TypeError("`trust_official_pickle` must be a boolean.")
        self._token = token.strip() if isinstance(token, str) else token
        self._trust_official_pickle = trust_official_pickle
        self.native_config = None
        self.native_generation_config = None
        self.transformers_processor = None
        self.training_model = None
        self._torch = None
        self._resolved_artifacts = None
        super().__init__(config, device=device, lazy_load=lazy_load)

    @property
    def training_processor(self):
        return self.transformers_processor

    def _hub_kwargs(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in {
                "revision": self.config.revision,
                "cache_dir": self.config.cache_dir,
                "local_files_only": self.config.local_files_only,
                "token": self._token,
            }.items() if value is not None
        }

    @staticmethod
    def _local_weight_file(name_or_path: str | Path) -> Path | None:
        path = Path(name_or_path).expanduser()
        if path.is_file() and path.suffix.lower() in {".bin", ".safetensors"}:
            return path.resolve()
        return None

    def _model_source(self) -> str:
        source = self.config.name_or_path or self.default_model_name_or_path
        weight = self._local_weight_file(source)
        return str(weight.parent) if weight is not None else str(source)

    def _config_source(self) -> str:
        return (
            str(self.config.config_name_or_path)
            if self.config.config_name_or_path is not None else self._model_source())

    def _processor_source(self) -> str:
        return (
            str(self.config.processor_name_or_path)
            if self.config.processor_name_or_path is not None else self._config_source())

    def _direct_state_dict(self) -> Mapping[str, Any] | None:
        weight = self._local_weight_file(self.config.name_or_path)
        if weight is None:
            return None
        if weight.suffix.lower() == ".safetensors":
            with SafeTensorReader(weight) as reader:
                return reader.state_dict(device="cpu")
        if not self._trust_official_pickle:
            raise PermissionError("Direct Bark `.bin` loading requires "
                                  "`trust_official_pickle=True`.")
        torch = import_optional("torch", model_type="bark", install_extra=None)
        try:
            state = torch.load(
                weight,
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise ValueError("Could not read Bark legacy tensor state.") from error
        if not isinstance(state, Mapping):
            raise TypeError("Bark legacy checkpoint did not contain a state mapping.")
        return state

    def _load_pretrained_model(self) -> None:
        import json
        from dataclasses import replace

        from voicehub.models.bark.native.artifacts import resolve_bark_artifacts
        from voicehub.models.bark.native.checkpoint import load_bark_safetensors, load_official_bark_checkpoint
        from voicehub.models.bark.native.configuration import BarkArchitectureConfig, BarkGenerationConfig
        from voicehub.models.bark.native.modeling import BarkModel
        from voicehub.models.bark.native.processing import BarkProcessor

        torch = import_optional("torch", model_type="bark", install_extra=None)
        artifact_source = self._config_source()
        artifacts = resolve_bark_artifacts(
            artifact_source,
            revision=self.config.revision,
            cache_dir=self.config.cache_dir,
            token=self._token,
            local_files_only=self.config.local_files_only,
            allow_legacy_checkpoint=self._trust_official_pickle,
            verify_integrity=self.config.verify_official_integrity,
        )
        direct = self._local_weight_file(self.config.name_or_path)
        if direct is not None:
            artifacts = replace(
                artifacts,
                checkpoint=direct,
                legacy_checkpoint=direct.suffix.lower() == ".bin",
            )
        if artifacts.checkpoint is None:
            raise PermissionError(
                "The pinned `suno/bark-small` release contains only a legacy "
                "pickle archive. Convert it once with "
                "`convert_official_bark_checkpoint(...)`, or explicitly pass "
                "`trust_official_pickle=True` for the digest-pinned source.")
        try:
            raw_config = (
                self.config.native_model_config or json.loads(artifacts.config.read_text(encoding="utf-8")))
            raw_generation = (
                self.config.native_generation_config or
                json.loads(artifacts.generation_config.read_text(encoding="utf-8")))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Bark configuration artifacts are invalid.") from error
        self.native_config = BarkArchitectureConfig.from_dict(raw_config)
        self.native_generation_config = BarkGenerationConfig.from_dict(raw_generation)
        if self.native_generation_config.sample_rate != self.config.sample_rate:
            self.config.sample_rate = self.native_generation_config.sample_rate

        model = BarkModel(
            self.native_config,
            generation_config=self.native_generation_config,
        )
        if artifacts.legacy_checkpoint:
            load_official_bark_checkpoint(
                model,
                artifacts.checkpoint,
                trust_official_pickle=self._trust_official_pickle,
            )
        else:
            target_dtype = (
                None if self.config.torch_dtype is None else resolve_torch_dtype(
                    torch,
                    self.config.torch_dtype,
                    self.device,
                ))
            model = model.to(
                device=self.device,
                dtype=target_dtype,
            )
            load_bark_safetensors(model, artifacts.checkpoint)
        if artifacts.legacy_checkpoint:
            target_dtype = (
                None if self.config.torch_dtype is None else resolve_torch_dtype(
                    torch,
                    self.config.torch_dtype,
                    self.device,
                ))
            model = model.to(device=self.device, dtype=target_dtype)

        local_root = (Path(artifacts.source) if Path(artifacts.source).is_dir() else None)
        speaker_source = (
            str(local_root) if local_root is not None and (local_root / "speaker_embeddings").is_dir() else
            (artifacts.source if artifacts.official_snapshot else None))
        processor = BarkProcessor.from_files(
            artifacts.vocabulary,
            speaker_index=artifacts.speaker_index,
            speaker_source=speaker_source,
            revision=artifacts.revision,
            cache_dir=self.config.cache_dir,
            token=self._token,
            local_files_only=self.config.local_files_only,
        )
        self._torch = torch
        self.model = model
        self.processor = processor
        self.transformers_processor = processor
        self._resolved_artifacts = artifacts

    def _prepare_for_training(self) -> None:
        from voicehub.models.bark.native.training import BarkTrainingModel

        self.model.train()
        self.model.codec_model.eval()
        self.model.codec_model.requires_grad_(False)
        if self.training_model is None:
            self.training_model = BarkTrainingModel.from_model(self.model)
        self.training_model.train()

    def _prepare_for_inference(self) -> None:
        self.model.eval()
        if self.training_model is not None:
            self.training_model.eval()

    @staticmethod
    def _phase_inputs(
        inputs: Mapping[str, Any],
        *,
        prefix: str,
    ) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for source, target in {
                f"{prefix}_input_ids": "input_ids",
                f"{prefix}_attention_mask": "attention_mask",
                f"{prefix}_labels": "labels",
        }.items():
            if source in inputs:
                output[target] = inputs[source]
        for name in ("input_ids", "attention_mask", "labels"):
            if name in inputs and name not in output:
                output[name] = inputs[name]
        if prefix == "fine" and "codebook_idx" in inputs:
            output["codebook_idx"] = inputs["codebook_idx"]
        return output

    def prepare_training_inputs(
        self,
        inputs: dict[str, Any],
        *,
        phase: str,
    ) -> dict[str, Any]:
        if phase not in {"semantic", "coarse", "fine"}:
            raise ValueError(f"Unknown Bark training phase {phase!r}.")
        prepared = self._phase_inputs(inputs, prefix=phase)
        missing = [name for name in ("input_ids", "labels") if name not in prepared]
        if phase == "fine" and "codebook_idx" not in prepared:
            missing.append("codebook_idx")
        if missing:
            raise ValueError(
                f"Bark {phase!r} fine-tuning requires precomputed "
                f"stage tokens; missing: {', '.join(missing)}.")
        return prepared

    def _processor_inputs(self, text: str, **kwargs: Any) -> dict[str, Any]:
        processor = self.transformers_processor or self.processor
        encoded = processor(
            text=text,
            return_tensors="pt",
            **kwargs,
        )
        if not isinstance(encoded, Mapping):
            raise TypeError("BarkProcessor must return a mapping.")
        return dict(self._move_to_device(encoded, self.device))

    @classmethod
    def _move_to_device(cls, value: Any, device: str):
        move = getattr(value, "to", None)
        if callable(move):
            moved = move(device)
            return value if moved is None else moved
        if isinstance(value, Mapping):
            return {key: cls._move_to_device(item, device) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(cls._move_to_device(item, device) for item in value)
        if isinstance(value, list):
            return [cls._move_to_device(item, device) for item in value]
        return value

    def _normalize_waveform(
        self,
        audio: Any,
        *,
        output_length: Any | None = None,
    ):
        if (isinstance(audio, (tuple, list)) and not hasattr(audio, "shape") and audio and
                not isinstance(audio[0], Real)):
            audio = audio[0]
        waveform = self._torch.as_tensor(
            audio,
            dtype=self._torch.float32,
            device="cpu",
        ).detach()
        if waveform.ndim == 2:
            if waveform.shape[0] != 1:
                raise RuntimeError("Bark returned more than one utterance for one request.")
            waveform = waveform[0]
        waveform = waveform.squeeze()
        if waveform.ndim == 0:
            waveform = waveform.reshape(1)
        if waveform.ndim != 1:
            raise RuntimeError(f"Bark returned non-mono audio with shape {waveform.shape}.")
        if output_length is not None:
            if hasattr(output_length, "detach"):
                output_length = output_length.detach().cpu()
            if isinstance(output_length, (tuple, list)):
                output_length = output_length[0]
            if hasattr(output_length, "item"):
                output_length = output_length.item()
            if (isinstance(output_length, bool) or not isinstance(output_length, Integral) or
                    output_length <= 0):
                raise RuntimeError("Bark returned an invalid waveform length.")
            waveform = waveform[:int(output_length)]
        if waveform.numel() == 0:
            raise RuntimeError("Bark returned empty audio.")
        if not bool(self._torch.isfinite(waveform).all().item()):
            raise RuntimeError("Bark returned NaN or infinite audio.")
        return waveform

    def _generate(
        self,
        text: str,
        *,
        voice_preset: str | Mapping[str, Any] | None = None,
        output_file: str | Path | None = None,
        seed: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_new_tokens: int | None = None,
        **generation_options: Any,
    ) -> TTSOutput:
        processor_options = {}
        if voice_preset is not None:
            processor_options["voice_preset"] = voice_preset
        inputs = self._processor_inputs(text, **processor_options)
        if "return_output_lengths" in generation_options:
            raise ValueError("`return_output_lengths` is managed by VoiceHub.")
        if temperature is not None:
            if not isinstance(temperature,
                              (int, float)) or not isfinite(float(temperature)) or temperature <= 0:
                raise ValueError("Bark temperature must be finite and positive.")
            generation_options["temperature"] = float(temperature)
        if top_p is not None:
            if (not isinstance(top_p, (int, float)) or not isfinite(float(top_p)) or not 0 < top_p <= 1):
                raise ValueError("Bark top-p must be in (0, 1].")
            generation_options["top_p"] = float(top_p)
        if max_new_tokens is not None:
            if (isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or
                    max_new_tokens <= 0):
                raise ValueError("Bark max new tokens must be positive.")
            generation_options["semantic_max_new_tokens"] = max_new_tokens
        generation_options["return_output_lengths"] = True
        with seeded_inference(
                seed,
                device=self.device,
                model_type=self.config.model_type,
        ) as effective_seed:
            with self._torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    **generation_options,
                )
        if not isinstance(generated, tuple) or len(generated) < 2:
            raise RuntimeError("Bark did not return VoiceHub-managed waveform lengths.")
        waveform = self._normalize_waveform(
            generated[0],
            output_length=generated[1],
        )
        preset_name = (
            voice_preset if isinstance(voice_preset, str) else "custom" if voice_preset is not None else None)
        return finish_audio_output(
            waveform,
            self.sample_rate,
            output_file=output_file,
            metadata={
                "model_type": self.config.model_type,
                "seed": effective_seed,
                "voice_preset": preset_name,
                "runtime": "voicehub-native",
            },
        )

    def _save_pretrained(self, save_directory: Path) -> None:
        import shutil

        from voicehub.hub import write_json_file
        from voicehub.models.bark.native.checkpoint import save_bark_safetensors

        if self.model is None or self.native_config is None:
            self.load_for_training()
        save_directory.mkdir(parents=True, exist_ok=True)
        save_bark_safetensors(
            self.model,
            save_directory / "model.safetensors",
        )
        write_json_file(
            save_directory / "config.json",
            self.native_config.to_dict(),
        )
        write_json_file(
            save_directory / "generation_config.json",
            self.native_generation_config.to_dict(),
        )
        self.processor.save_pretrained(save_directory)
        artifacts = self._resolved_artifacts
        if artifacts is not None:
            shutil.copy2(artifacts.tokenizer, save_directory / "tokenizer.json")
            shutil.copy2(
                artifacts.speaker_index,
                save_directory / "speaker_embeddings_path.json",
            )
        elif not (save_directory / "tokenizer.json").is_file():
            raise RuntimeError("Bark export requires the original declarative tokenizer.json "
                               "artifact.")


__all__ = [
    "BarkConfig",
    "BarkForTextToSpeech",
    "_build_bark_training_model",
]
