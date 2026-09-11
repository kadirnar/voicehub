"""Public TTS wrapper backed entirely by VoiceHub-native Higgs Audio v2."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import Any

from voicehub.models._shared import finish_audio_output, resolve_torch_dtype, validate_local_file
from voicehub.models.higgstts.configuration import HiggsTTSConfig
from voicehub.models.higgstts.native.metadata import HIGGS_AUDIO_V2_REPOSITORY
from voicehub.models.tts import PreTrainedTTSModel
from voicehub.outputs import TTSOutput

_PREPARED_TRAINING_KEYS = frozenset({
    "attention_mask",
    "audio_input_ids",
    "audio_input_ids_mask",
    "audio_labels",
    "input_ids",
    "labels",
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
            raise ValueError("Higgs `records` must be a non-empty sequence.")
        if any(not isinstance(record, Mapping) for record in supplied):
            raise TypeError("Every Higgs training record must be a mapping.")
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
    raise ValueError("Raw Higgs training inputs require `text` or an explicit "
                     "`records` sequence.")


class HiggsTTSForTextToSpeech(PreTrainedTTSModel):
    """Expressive synthesis and full SFT on the audited native graph."""

    config_class = HiggsTTSConfig
    default_model_name_or_path = HIGGS_AUDIO_V2_REPOSITORY

    def __init__(
        self,
        config: HiggsTTSConfig | str | Path | None = None,
        *,
        model_path: str | Path | None = None,
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
        super().__init__(config, device=device, lazy_load=lazy_load)

    @property
    def native_runtime(self):
        """Return the loaded native runtime, or ``None`` before loading."""
        return self._runtime

    def _load_pretrained_model(self) -> None:
        from voicehub.models.higgstts.native.runtime import HiggsAudioV2Runtime, load_higgs_audio_v2_runtime

        if self._runtime is not None:
            if not isinstance(self._runtime, HiggsAudioV2Runtime):
                raise TypeError("Injected Higgs runtime must be a HiggsAudioV2Runtime.")
            runtime = self._runtime
        else:
            import torch

            dtype = resolve_torch_dtype(
                torch,
                self.config.torch_dtype,
                self.device,
            )
            runtime = load_higgs_audio_v2_runtime(
                self.config.name_or_path,
                revision=self.config.revision,
                codec_source=self.config.audio_tokenizer_name_or_path,
                codec_revision=self.config.codec_revision,
                device=self.device,
                dtype=dtype,
                cache_dir=self.config.cache_dir,
                token=self._hub_token,
                local_files_only=self.config.local_files_only,
                verify_integrity=self.config.verify_integrity,
                verify_checkpoint_integrity=(self.config.verify_checkpoint_integrity),
            )
            self._runtime = runtime
        self.model = runtime.model
        self.config.sample_rate = runtime.sample_rate

    def _validate_training_runtime(self) -> None:
        if self.config.use_safetensors is False:
            raise ValueError("Higgs fine-tuning requires Safetensors.")

    def _prepare_for_training(self) -> None:
        if self._runtime is None:
            raise RuntimeError("Higgs native runtime was not loaded.")
        self._runtime.prepare_for_training()
        self.model = self._runtime.model

    def _prepare_for_inference(self) -> None:
        if self._runtime is None:
            raise RuntimeError("Higgs native runtime was not loaded.")
        if self.model is not self._runtime.model:
            self._runtime.model = self.model
        self._runtime.prepare_for_inference()

    def _set_training_device(self, device: str) -> None:
        super()._set_training_device(device)
        if self._runtime is not None:
            self._runtime.audio_tokenizer.to(device=device)

    def _validate_generation_inputs(
        self,
        model_inputs: dict[str, Any],
    ) -> None:
        speaker_audio = validate_local_file(
            model_inputs.get("speaker_audio_path"),
            option_name="speaker_audio_path",
        )
        reference_audio = model_inputs.get("reference_audio")
        if speaker_audio is not None and reference_audio is not None:
            raise ValueError("Pass either `speaker_audio_path` or `reference_audio`, "
                             "not both.")
        if speaker_audio is not None:
            model_inputs["speaker_audio_path"] = str(speaker_audio)
        reference_text = model_inputs.get("reference_text")
        has_reference = speaker_audio is not None or reference_audio is not None
        if has_reference and (not isinstance(reference_text, str) or not reference_text.strip()):
            raise ValueError("Higgs voice cloning requires a non-empty `reference_text`.")
        if not has_reference and reference_text is not None:
            raise ValueError("`reference_text` requires reference audio.")
        if not isinstance(model_inputs.get("force_audio_gen", True), bool):
            raise TypeError("`force_audio_gen` must be a boolean.")
        if model_inputs.get("force_audio_gen", True) is False:
            raise ValueError(
                "The VoiceHub TTS API requires audio generation; "
                "`force_audio_gen=False` is unsupported.")
        system_prompt = model_inputs.get("system_prompt")
        if system_prompt is not None and (not isinstance(system_prompt, str) or not system_prompt.strip()):
            raise ValueError("`system_prompt` must be a non-empty string or None.")
        scene_prompt = model_inputs.get("scene_prompt")
        if scene_prompt is not None and (not isinstance(scene_prompt, str) or not scene_prompt.strip()):
            raise ValueError("`scene_prompt` must be a non-empty string or None.")
        top_k = model_inputs.get("top_k", 50)
        if top_k is not None and (isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0):
            raise ValueError("`top_k` must be a positive integer or None.")
        temperature = model_inputs.get("temperature", 1.0)
        top_p = model_inputs.get("top_p", 0.95)
        if (isinstance(temperature, bool) or not isinstance(temperature, Real) or
                not math.isfinite(float(temperature)) or float(temperature) < 0.0):
            raise ValueError("`temperature` must be a finite non-negative number.")
        if (isinstance(top_p, bool) or not isinstance(top_p, Real) or not math.isfinite(float(top_p)) or
                not 0.0 <= float(top_p) <= 1.0 or (float(temperature) > 0.0 and float(top_p) == 0.0)):
            interval = "[0, 1]" if float(temperature) == 0.0 else "(0, 1]"
            raise ValueError(f"`top_p` must be in the interval {interval}.")
        for name, default in (
            ("ras_win_len", 7),
            ("ras_win_max_num_repeat", 2),
        ):
            value = model_inputs.get(name, default)
            if name == "ras_win_len" and value is None:
                continue
            if (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
                raise ValueError(
                    f"`{name}` must be a positive integer" + (" or None." if name == "ras_win_len" else "."))

    def _generate(
        self,
        text: str,
        *,
        output_file: str | None = None,
        speaker_audio_path: str | None = None,
        reference_audio: Any | None = None,
        reference_sampling_rate: int | None = None,
        reference_text: str | None = None,
        system_prompt: str | None = None,
        scene_prompt: str | None = None,
        max_new_tokens: int = 1_024,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int | None = 50,
        seed: int | None = None,
        force_audio_gen: bool = True,
        ras_win_len: int | None = 7,
        ras_win_max_num_repeat: int = 2,
    ) -> TTSOutput:
        del force_audio_gen
        if self._runtime is None:
            raise RuntimeError("Higgs native runtime was not loaded.")
        reference = (reference_audio if reference_audio is not None else speaker_audio_path)
        result = self._runtime.generate(
            text,
            reference_audio=reference,
            reference_sampling_rate=reference_sampling_rate,
            reference_text=reference_text,
            system_prompt=system_prompt or self.config.system_prompt,
            scene_prompt=(self.config.scene_prompt if scene_prompt is None else scene_prompt),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            ras_window=ras_win_len,
            ras_max_repeats=ras_win_max_num_repeat,
            seed=seed,
        )
        audio = result.waveform.detach().float().cpu()
        while audio.ndim > 1 and audio.shape[0] == 1:
            audio = audio[0]
        return finish_audio_output(
            audio,
            result.sample_rate,
            output_file=output_file,
            metadata={
                "architecture": "higgs_audio_v2",
                "backend": "voicehub-native",
                "checkpoint_format": "safetensors",
                "generated_steps": result.generated_steps,
                "reference_audio": reference is not None,
                "requested_seed": seed,
                "seed": seed,
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
            raise TypeError("Higgs training inputs must be a mapping.")
        if _PREPARED_TRAINING_KEYS <= set(inputs):
            return {name: inputs[name] for name in _PREPARED_TRAINING_KEYS}
        if self._runtime is None:
            raise RuntimeError("Higgs training preparation requires a loaded runtime.")
        batch = self._runtime.prepare_training_inputs(_training_records(inputs))
        return batch.model_inputs()

    def _save_pretrained(self, save_directory: Path) -> None:
        self.load()
        if self._runtime is None:
            raise RuntimeError("Only VoiceHub-native Higgs runtimes can be exported.")
        self._runtime.save_pretrained(save_directory)


HiggsTTS = HiggsTTSForTextToSpeech

__all__ = [
    "HiggsTTS",
    "HiggsTTSForTextToSpeech",
]
