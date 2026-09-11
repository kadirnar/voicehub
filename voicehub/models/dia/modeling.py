"""VoiceHub-native Dia inference and lifecycle integration.

The public wrapper intentionally contains no provider runtime imports.
Dia's text encoder, autoregressive decoder, delay protocol, strict
Safetensors adapter, and DAC processor all live under
:mod:`voicehub.runtime`.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Any

from voicehub.models._shared import finish_audio_output, seeded_inference, validate_local_file
from voicehub.models.dia.native.metadata import NARI_DIA_CHECKPOINT_REVISION
from voicehub.models.tts import PreTrainedTTSModel
from voicehub.outputs import TTSOutput
from voicehub.training.utils import NATIVE_EXPORT_DIR

from .configuration import DiaConfig

_NATIVE_CHECKPOINT = "nari-labs/Dia-1.6B-0626"
_LEGACY_CHECKPOINT = "nari-labs/Dia-1.6B"
_GENERATION_ALIASES = (
    ("max_tokens", "max_new_tokens"),
    ("cfg_scale", "guidance_scale"),
    ("cfg_filter_top_k", "top_k"),
)


class DiaForTextToSpeech(PreTrainedTTSModel):
    """Dialogue synthesis and full fine-tuning through native PyTorch Dia."""

    config_class = DiaConfig
    default_model_name_or_path = _NATIVE_CHECKPOINT
    training_default_model_name_or_path = _NATIVE_CHECKPOINT
    passthrough_generation_options = frozenset({
        "audio",
        "audio_prompt",
        "audio_prompt_path",
        "cfg_filter_top_k",
        "cfg_scale",
        "do_sample",
        "guidance_scale",
        "max_new_tokens",
        "max_tokens",
        "output_file",
        "seed",
        "temperature",
        "top_k",
        "top_p",
        "use_cfg_filter",
        "verbose",
    })

    def __init__(
        self,
        config: DiaConfig | str | Path | None = None,
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
        config = self._coerce_config(
            config,
            model_path=model_path,
            **config_overrides,
        )
        config.validate()
        self._hub_token = token
        self._dia_runtime = None
        self._loaded_backend: str | None = None
        super().__init__(config, device=device, lazy_load=lazy_load)

    @staticmethod
    def _is_legacy_checkpoint(name_or_path: str | Path) -> bool:
        normalized = str(name_or_path).rstrip("/").lower()
        if normalized == _LEGACY_CHECKPOINT.lower():
            return True
        source = Path(name_or_path).expanduser()
        config_path = (
            source if source.is_file() and source.name == "config.json" else source / "config.json")
        if not config_path.is_file():
            return False
        try:
            values = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return False
        return (
            isinstance(values, Mapping) and "data" in values and "model" in values and
            values.get("model_type") != "dia")

    def _select_backend(self, *, for_training: bool) -> str:
        del for_training
        self._validate_supported_checkpoint()
        return "native"

    def _validate_supported_checkpoint(self) -> None:
        if self._is_legacy_checkpoint(self.config.name_or_path):
            raise ValueError(
                "The original `nari-labs/Dia-1.6B` checkpoint uses a "
                "different pickle/JAX layout. Use "
                "`nari-labs/Dia-1.6B-0626` for strict native inference and "
                "fine-tuning.")

    def _validate_training_runtime(self) -> None:
        self._validate_supported_checkpoint()

    def _runtime_source(self) -> str | Path:
        source = Path(self.config.name_or_path).expanduser()
        native_export = source / NATIVE_EXPORT_DIR
        if source.is_dir() and (native_export / "config.json").is_file():
            return native_export
        return self.config.name_or_path

    def _load_pretrained_model(self) -> None:
        self._select_backend(for_training=self.is_training_load)
        from voicehub.models.dia.native.runtime import load_dia_runtime

        source = self._runtime_source()
        source_path = Path(source).expanduser()
        is_local = source_path.exists()
        revision = self.config.revision
        if revision is None and not is_local and str(source) == _NATIVE_CHECKPOINT:
            revision = NARI_DIA_CHECKPOINT_REVISION
        runtime = load_dia_runtime(
            source,
            device=self.device,
            compute_dtype=self.config.compute_dtype,
            revision=revision,
            cache_dir=self.config.cache_dir,
            token=self._hub_token,
            local_files_only=self.config.local_files_only,
            for_training=self.is_training_load,
        )
        self.model = runtime.model
        self._dia_runtime = runtime
        self._loaded_backend = "native"
        self.config.sample_rate = runtime.sample_rate

    @property
    def training_backend(self):
        runtime = self._dia_runtime
        if runtime is not None and runtime.model is self.model:
            return runtime
        return None

    def _prepare_for_training(self) -> None:
        runtime = self.training_backend
        if runtime is None:
            raise RuntimeError("Dia's native training runtime is not loaded.")
        runtime.prepare_for_training()

    def _prepare_for_inference(self) -> None:
        runtime = self.training_backend
        if runtime is None:
            raise RuntimeError("Dia's native inference runtime is not loaded.")
        runtime.prepare_for_inference()

    def prepare_training_inputs(
        self,
        inputs: dict[str, Any],
        *,
        phase: str,
    ) -> dict[str, Any]:
        del phase
        runtime = self.training_backend
        if runtime is None:
            raise RuntimeError("Dia training inputs require load_for_training() first.")
        return runtime.prepare_inputs(inputs)

    def _validate_generation_inputs(self, model_inputs: dict[str, Any]) -> None:
        for source, target in _GENERATION_ALIASES:
            if model_inputs.get(source) is not None and model_inputs.get(target) is not None:
                raise ValueError(f"Pass either {source!r} or {target!r}, not both.")

        prompt_names = ("audio", "audio_prompt", "audio_prompt_path")
        supplied = [name for name in prompt_names if model_inputs.get(name) is not None]
        if len(supplied) > 1:
            raise ValueError("Pass only one of 'audio', 'audio_prompt', or "
                             "'audio_prompt_path'.")
        if supplied:
            name = supplied[0]
            prompt = model_inputs[name]
            if isinstance(prompt, Mapping) and "path" in prompt:
                prompt = prompt["path"]
            if isinstance(prompt, (str, PathLike)):
                normalized = validate_local_file(prompt, option_name=name)
                original = model_inputs[name]
                if isinstance(original, Mapping):
                    original = dict(original)
                    original["path"] = str(normalized)
                    model_inputs[name] = original
                else:
                    model_inputs[name] = str(normalized)

        for name in ("max_tokens", "max_new_tokens", "cfg_filter_top_k", "top_k"):
            value = model_inputs.get(name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"`{name}` must be a positive integer.")
        for name in ("cfg_scale", "guidance_scale"):
            value = model_inputs.get(name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"`{name}` must be a finite number.")
            if not math.isfinite(float(value)) or value < 0:
                raise ValueError(f"`{name}` must be finite and non-negative.")
        do_sample = model_inputs.get("do_sample")
        if do_sample is not None and not isinstance(do_sample, bool):
            raise TypeError("`do_sample` must be a boolean.")
        top_p = model_inputs.get("top_p")
        if top_p is not None and float(top_p) <= 0:
            raise ValueError("Dia `top_p` must be greater than zero.")

    @staticmethod
    def _rename_generation_options(options: dict[str, Any]) -> None:
        for source, target in _GENERATION_ALIASES:
            if source not in options:
                continue
            if target in options:
                raise ValueError(f"Pass either {source!r} or {target!r}, not both.")
            options[target] = options.pop(source)

    @staticmethod
    def _pop_audio_prompt(options: dict[str, Any]) -> Any:
        prompt = None
        for name in ("audio", "audio_prompt", "audio_prompt_path"):
            value = options.pop(name, None)
            if value is None:
                continue
            if prompt is not None:
                raise ValueError("Pass only one of 'audio', 'audio_prompt', or "
                                 "'audio_prompt_path'.")
            prompt = value
        return prompt

    def _generate_native(
        self,
        text: str,
        generation_options: dict[str, Any],
    ) -> Any:
        runtime = self.training_backend
        if runtime is None:
            raise RuntimeError("Dia's native runtime is not loaded.")
        options = dict(generation_options)
        self._rename_generation_options(options)
        options.pop("use_cfg_filter", None)
        options.pop("verbose", None)
        audio = self._pop_audio_prompt(options)
        inputs = runtime.processor(
            text=[text],
            audio=audio,
            generation=True,
            padding=True,
            return_tensors="pt",
        )
        prompt_length = (
            runtime.processor.get_audio_prompt_len(inputs["decoder_attention_mask"])
            if audio is not None else None)
        generated = self.model.generate(
            **inputs.to(self.device),
            **options,
        )
        decoded = runtime.processor.batch_decode(
            generated,
            audio_prompt_len=prompt_length,
        )
        if not decoded:
            raise RuntimeError("Native Dia decoding returned no audio.")
        return decoded[0]

    def _generate(
        self,
        text: str,
        *,
        output_file: str | None = None,
        **generation_options: Any,
    ) -> TTSOutput:
        options = dict(generation_options)
        requested_seed = options.pop("seed", None)
        with seeded_inference(
                requested_seed,
                device=self.device,
                model_type="dia",
        ) as effective_seed:
            audio = self._generate_native(text, options)
        revision = (self._dia_runtime.artifacts.revision if self._dia_runtime is not None else None)
        return finish_audio_output(
            audio,
            self.sample_rate,
            output_file=output_file,
            metadata={
                "backend": "voicehub-native",
                "checkpoint_revision": revision,
                "requested_seed": requested_seed,
                "seed": effective_seed,
            },
        )

    def _save_pretrained(self, save_directory: Path) -> None:
        if self._dia_runtime is None:
            self.load()
        if self._dia_runtime is None:
            raise RuntimeError("Dia's native runtime is not available for export.")
        self._dia_runtime.save_pretrained(save_directory)


DiaVoiceHubConfig = DiaConfig
DiaTTS = DiaForTextToSpeech

__all__ = [
    "DiaConfig",
    "DiaForTextToSpeech",
    "DiaTTS",
    "DiaVoiceHubConfig",
]
