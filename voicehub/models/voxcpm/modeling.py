"""Public VoxCPM2 API backed entirely by VoiceHub-native code."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from voicehub.models._shared import finish_audio_output, resolve_torch_dtype, validate_local_file
from voicehub.models.tts import PreTrainedTTSModel
from voicehub.models.voxcpm.configuration import VoxCPMConfig
from voicehub.outputs import TTSOutput

_PREPARED_TRAINING_KEYS = frozenset({
    "audio_feats",
    "audio_mask",
    "labels",
    "loss_mask",
    "position_ids",
    "text_mask",
    "text_tokens",
})


def _batch_item(value: Any, index: int, count: int) -> Any:
    if isinstance(value, Mapping):
        return {name: _batch_item(item, index, count) for name, item in value.items()}
    if isinstance(value, (str, bytes, bytearray)):
        return value
    shape = getattr(value, "shape", ())
    if shape and int(shape[0]) == count:
        return value[index]
    if isinstance(value, Sequence) and len(value) == count:
        return value[index]
    return value


def _training_records(inputs: Mapping[str, Any]) -> list[dict[str, Any]]:
    supplied = inputs.get("records")
    if supplied is not None:
        if (isinstance(supplied, (str, bytes, Mapping)) or not isinstance(supplied, Sequence) or
                not supplied):
            raise ValueError("VoxCPM `records` must be a non-empty sequence.")
        if any(not isinstance(record, Mapping) for record in supplied):
            raise TypeError("Every VoxCPM training record must be a mapping.")
        return [dict(record) for record in supplied]
    text = inputs.get("text")
    if isinstance(text, str):
        return [dict(inputs)]
    if (isinstance(text, Sequence) and not isinstance(text, (str, bytes, bytearray)) and text):
        count = len(text)
        return [{
            name: _batch_item(value, index, count)
            for name, value in inputs.items() if name != "training_phase"
        } for index in range(count)]
    raise ValueError("Raw VoxCPM training inputs require `text` or an explicit `records` sequence.")


class VoxCPMForTextToSpeech(PreTrainedTTSModel):
    """Multilingual VoxCPM2 synthesis and source-faithful fine-tuning."""

    config_class = VoxCPMConfig
    default_model_name_or_path = "openbmb/VoxCPM2"

    def __init__(
        self,
        config: VoxCPMConfig | str | None = None,
        *,
        model_path: str | None = None,
        device: str = "auto",
        lazy_load: bool = True,
        token: str | bool | None = None,
        **config_overrides: Any,
    ) -> None:
        config = self._coerce_config(
            config,
            model_path=model_path,
            **config_overrides,
        )
        self._runtime = None
        self._hub_token = token
        self._active_lora_config = None
        super().__init__(config, device=device, lazy_load=lazy_load)

    @property
    def native_runtime(self):
        return self._runtime

    def _activate_lora(self, config, *, weights: str | Path | None = None) -> None:
        from voicehub.models.voxcpm.native.lora import inject_voxcpm_lora, load_voxcpm_lora

        if self._active_lora_config is not None:
            if self._active_lora_config != config:
                raise ValueError("Loaded and requested VoxCPM LoRA configurations differ.")
        else:
            inject_voxcpm_lora(self._runtime.model, config)
            self._active_lora_config = config
        if weights is not None:
            load_voxcpm_lora(self._runtime.model, weights, config)

    def _load_pretrained_model(self) -> None:
        from voicehub.models.voxcpm.native.runtime import VoxCPM2Runtime, load_voxcpm2_runtime

        if self._runtime is not None:
            if not isinstance(self._runtime, VoxCPM2Runtime):
                raise TypeError("Injected VoxCPM runtime must be a VoxCPM2Runtime.")
            runtime = self._runtime
        else:
            import torch

            dtype = resolve_torch_dtype(
                torch,
                self.config.torch_dtype,
                self.device,
            )
            runtime = load_voxcpm2_runtime(
                self.config.name_or_path,
                revision=self.config.revision,
                codec_path=self.config.codec_path,
                device=self.device,
                dtype=dtype,
                trust_legacy_codec=self.config.trust_legacy_codec,
                cache_dir=self.config.cache_dir,
                token=self._hub_token,
                local_files_only=self.config.local_files_only,
                verify_integrity=self.config.verify_integrity,
                verify_checkpoint_integrity=(self.config.verify_checkpoint_integrity),
            )
            self._runtime = runtime
        self.model = runtime.model
        self.config.sample_rate = runtime.sample_rate
        if self.config.lora_path is not None:
            from voicehub.models.voxcpm.native.lora import read_voxcpm_lora_config

            lora_config = read_voxcpm_lora_config(self.config.lora_path)
            self._activate_lora(
                lora_config,
                weights=self.config.lora_path,
            )

    def _validate_training_runtime(self) -> None:
        if self.config.use_safetensors is False:
            raise ValueError("VoxCPM fine-tuning requires Safetensors.")

    def _prepare_for_training(self) -> None:
        if self._runtime is None:
            raise RuntimeError("VoxCPM native runtime was not loaded.")
        self._runtime.prepare_for_training()
        lora_values = self.config.training_lora_config
        if lora_values is not None:
            from voicehub.models.voxcpm.native.lora import VoxCPMLoRAConfig

            self._activate_lora(VoxCPMLoRAConfig.from_mapping(lora_values))
        self.model = self._runtime.model

    def _prepare_for_inference(self) -> None:
        if self._runtime is None:
            raise RuntimeError("VoxCPM native runtime was not loaded.")
        if self.model is not self._runtime.model and callable(getattr(self.model, "generate_features", None)):
            self._runtime.model = self.model
        self._runtime.prepare_for_inference()

    def _set_training_device(self, device: str) -> None:
        super()._set_training_device(device)
        if self._runtime is None:
            return
        self._runtime.codec.to(device=device)

    def _validate_generation_inputs(self, model_inputs: dict[str, Any]) -> None:
        prompt_audio = validate_local_file(
            model_inputs.get("prompt_audio_path"),
            option_name="prompt_audio_path",
        )
        reference_audio = validate_local_file(
            model_inputs.get("speaker_audio_path"),
            option_name="speaker_audio_path",
        )
        prompt_text = model_inputs.get("reference_text")
        if prompt_text is not None and (not isinstance(prompt_text, str) or not prompt_text.strip()):
            raise ValueError("`reference_text` must be a non-empty string or None.")
        if (prompt_audio is None) != (prompt_text is None):
            raise ValueError("`prompt_audio_path` and `reference_text` must be provided together.")
        if prompt_audio is not None:
            model_inputs["prompt_audio_path"] = str(prompt_audio)
        if reference_audio is not None:
            model_inputs["speaker_audio_path"] = str(reference_audio)
        guidance = model_inputs.get("cfg_value", 2.0)
        if (isinstance(guidance, bool) or not isinstance(guidance, (int, float)) or
                not math.isfinite(float(guidance)) or guidance < 0):
            raise ValueError("`cfg_value` must be finite and non-negative.")
        steps = model_inputs.get("inference_timesteps", 10)
        if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
            raise ValueError("`inference_timesteps` must be a positive integer.")
        minimum = model_inputs.get("min_len", 2)
        maximum = model_inputs.get("max_len", 2_000)
        for name, value in (("min_len", minimum), ("max_len", maximum)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"`{name}` must be a positive integer.")
        if minimum >= maximum:
            raise ValueError("`min_len` must be below `max_len`.")
        seed = model_inputs.get("seed")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
            raise ValueError("`seed` must be a non-negative integer or None.")
        unsupported = [
            name for name in ("denoise", "normalize", "retry_badcase")
            if model_inputs.get(name, False) is not False
        ]
        if unsupported:
            raise ValueError(
                "Native VoxCPM keeps external audio postprocessing outside "
                "the model boundary; unsupported enabled option(s): " + ", ".join(unsupported) + ".")

    def _generate(
        self,
        text: str,
        *,
        output_file: str | None = None,
        speaker_audio_path: str | None = None,
        reference_text: str | None = None,
        prompt_audio_path: str | None = None,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
        seed: int | None = None,
        min_len: int = 2,
        max_len: int = 2_000,
        normalize: bool = False,
        denoise: bool = False,
        retry_badcase: bool = False,
    ) -> TTSOutput:
        del normalize, denoise, retry_badcase
        if self._runtime is None:
            raise RuntimeError("VoxCPM native runtime was not loaded.")
        audio = self._runtime.generate(
            text,
            prompt_audio=prompt_audio_path,
            prompt_text="" if reference_text is None else reference_text,
            reference_audio=speaker_audio_path,
            min_length=min_len,
            max_length=max_len,
            diffusion_steps=inference_timesteps,
            guidance=cfg_value,
            seed=seed,
        ).detach().float().cpu()
        if audio.ndim == 2 and audio.shape[0] == 1:
            audio = audio[0]
        return finish_audio_output(
            audio,
            self.sample_rate,
            output_file=output_file,
            metadata={
                "backend": "voicehub-native",
                "architecture": "voxcpm2",
                "checkpoint_format": "safetensors",
                "seed": seed,
                "requested_seed": seed,
                "reference_audio_isolation": speaker_audio_path is not None,
                "audio_continuation": prompt_audio_path is not None,
            },
        )

    def prepare_training_inputs(
        self,
        inputs: dict[str, Any],
        *,
        phase: str,
    ) -> dict[str, Any]:
        del phase
        if not isinstance(inputs, Mapping):
            raise TypeError("VoxCPM training inputs must be a mapping.")
        if _PREPARED_TRAINING_KEYS <= set(inputs):
            return {name: inputs[name] for name in _PREPARED_TRAINING_KEYS}
        if self._runtime is None:
            raise RuntimeError("VoxCPM training preparation requires a loaded runtime.")
        return self._runtime.prepare_training_inputs(_training_records(inputs))

    def _save_pretrained(self, save_directory: Path) -> None:
        """Export a flat, pickle-free runtime with LoRA merged when active."""
        self.load()
        if self._runtime is None:
            raise RuntimeError("Only VoiceHub-native VoxCPM runtimes can be exported.")
        merged_state = None
        if self._active_lora_config is not None:
            from voicehub.models.voxcpm.native.lora import export_voxcpm_lora, merged_voxcpm_state_dict

            merged_state = merged_voxcpm_state_dict(self._runtime.model)
        self._runtime.save_pretrained_with_state(
            save_directory,
            model_state_override=merged_state,
        )
        if self._active_lora_config is not None:
            export_voxcpm_lora(
                self._runtime.model,
                save_directory / "lora_adapter",
                self._active_lora_config,
            )


VoxCPMTTS = VoxCPMForTextToSpeech

__all__ = ["VoxCPMForTextToSpeech", "VoxCPMTTS"]
