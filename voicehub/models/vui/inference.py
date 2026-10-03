"""Vui integration using the architecture source included in VoiceHub."""

from __future__ import annotations

import math
from typing import Any

from voicehub.configuration_utils import VoiceHubConfig
from voicehub.dependencies import import_optional
from voicehub.modeling_outputs import TTSOutput
from voicehub.modeling_utils import PreTrainedTTSModel
from voicehub.models._shared import finish_audio_output, seeded_inference
from voicehub.models.vui.artifacts import (
    VUI_CODEC_FILENAME,
    VUI_MODEL_FILENAME,
    VUI_REVISION,
    VuiArtifacts,
    resolve_vui_artifacts,
)


class VuiConfig(VoiceHubConfig):
    """Configuration for Vui checkpoint loading."""

    model_type = "vui"

    def __init__(
        self,
        *,
        sample_rate: int = 22_050,
        checkpoint_filename: str | None = None,
        codec_filename: str = VUI_CODEC_FILENAME,
        native_artifact_format: str | None = None,
        native_artifact_format_version: int | None = None,
        native_model_config: dict[str, Any] | None = None,
        native_codec_config: dict[str, Any] | None = None,
        revision: str = VUI_REVISION,
        cache_dir: str | None = None,
        local_files_only: bool = False,
        verify_official_integrity: bool = True,
        vad_backend: str = "pyannote",
        **kwargs,
    ):
        from voicehub.models.vui.vad import VAD_BACKENDS

        if vad_backend not in VAD_BACKENDS:
            raise ValueError(f"`vad_backend` must be one of {VAD_BACKENDS}, received {vad_backend!r}.")
        super().__init__(
            sample_rate=sample_rate,
            checkpoint_filename=checkpoint_filename,
            codec_filename=codec_filename,
            native_artifact_format=native_artifact_format,
            native_artifact_format_version=native_artifact_format_version,
            native_model_config=native_model_config,
            native_codec_config=native_codec_config,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            verify_official_integrity=verify_official_integrity,
            vad_backend=vad_backend,
            **kwargs,
        )


class VuiForTextToSpeech(PreTrainedTTSModel):
    """Vui synthesis with locally maintained source."""

    config_class = VuiConfig
    default_model_name_or_path = VUI_MODEL_FILENAME
    passthrough_generation_options = frozenset({
        "max_chunk_retries",
        "max_secs",
        "prompt_codes",
        "temperature",
        "top_k",
        "top_p",
    })

    def __init__(
        self,
        config: VuiConfig | str | None = None,
        *,
        model_path: str | None = None,
        device: str = "auto",
        lazy_load: bool = True,
        token: str | bool | None = None,
        **config_overrides,
    ):
        config = self._coerce_config(
            config,
            model_path=model_path,
            **config_overrides,
        )
        self._hub_token = token
        self.artifacts: VuiArtifacts | None = None
        self.vad_pipeline = None
        super().__init__(config, device=device, lazy_load=lazy_load)

    def _load_pretrained_model(self) -> None:
        from voicehub.models.vui.model import Vui

        self.artifacts = resolve_vui_artifacts(
            self.config.name_or_path,
            model_filename=self.config.checkpoint_filename,
            codec_filename=self.config.codec_filename,
            revision=self.config.revision,
            cache_dir=self.config.cache_dir,
            token=self._hub_token,
            local_files_only=self.config.local_files_only,
            verify_official_integrity=self.config.verify_official_integrity,
        )
        self.model = Vui.from_pretrained(
            checkpoint_path=self.artifacts.model_checkpoint,
            codec_path=self.artifacts.codec_checkpoint,
            model_config=self.config.native_model_config,
            codec_config=self.config.native_codec_config,
        ).to(self.device)
        self.model.eval()
        self.config.sample_rate = int(self.model.codec.config.sample_rate)
        self.vad_pipeline = self._load_vad_pipeline()

    def _load_vad_pipeline(self):
        """Create the trimming detector; upstream uses pyannote VAD.

        Its (gated) segmentation checkpoint is resolved on first use, as
        in upstream's ``render``.
        """
        from voicehub.models.vui.vad import create_vad_pipeline

        return create_vad_pipeline(
            getattr(self.config, "vad_backend", "pyannote"),
            device=str(self.device),
            token=self._hub_token,
            local_files_only=self.config.local_files_only,
        )

    def _prepare_for_training(self) -> None:
        """Restore the uncached autoregressive graph used for token
        training."""
        decoder = getattr(self.model, "decoder", None)
        if decoder is not None and hasattr(decoder, "deallocate_kv_cache"):
            decoder.deallocate_kv_cache()
        codec = getattr(self.model, "codec", None)
        if codec is not None:
            codec.eval()
            for parameter in codec.parameters():
                parameter.requires_grad_(False)
        self.model.train()

    def _prepare_for_inference(self) -> None:
        decoder = getattr(self.model, "decoder", None)
        if decoder is not None and hasattr(decoder, "deallocate_kv_cache"):
            decoder.deallocate_kv_cache()
        self.model.eval()

    def _validate_generation_inputs(self, model_inputs: dict[str, Any]) -> None:
        temperature = model_inputs.get("temperature", 0.5)
        if (not isinstance(temperature, (int, float)) or isinstance(temperature, bool) or
                not math.isfinite(temperature) or temperature <= 0):
            raise ValueError("`temperature` must be a finite positive number.")
        top_k = model_inputs.get("top_k", 100)
        if top_k is not None and (not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0):
            raise ValueError("`top_k` must be a positive integer or None.")
        top_p = model_inputs.get("top_p")
        if top_p is not None and (not isinstance(top_p, (int, float)) or isinstance(top_p, bool) or
                                  not math.isfinite(top_p) or not 0 < top_p <= 1):
            raise ValueError("`top_p` must be finite and in the interval (0, 1] or None.")
        max_secs = model_inputs.get("max_secs", 100)
        if (not isinstance(max_secs, int) or isinstance(max_secs, bool) or max_secs <= 0):
            raise ValueError("`max_secs` must be a positive integer.")
        max_chunk_retries = model_inputs.get("max_chunk_retries", 3)
        if (not isinstance(max_chunk_retries, int) or isinstance(max_chunk_retries, bool) or
                max_chunk_retries <= 0):
            raise ValueError("`max_chunk_retries` must be a positive integer.")

    def _generate(
        self,
        text: str,
        *,
        output_file: str | None = None,
        seed: int | None = None,
        **generation_options,
    ) -> TTSOutput:
        from voicehub.models.vui.tts import render

        torch = import_optional(
            "torch",
            model_type="vui",
            install_extra=None,
        )
        with seeded_inference(
                seed,
                device=self.device,
                model_type="vui",
        ) as effective_seed:
            with torch.inference_mode():
                waveform = render(
                    self.model,
                    text,
                    vad_pipeline=self.vad_pipeline,
                    **generation_options,
                )
        if waveform is None or waveform.numel() == 0:
            raise RuntimeError("Vui returned an empty audio waveform.")
        audio = waveform[0] if waveform.ndim > 1 else waveform
        return finish_audio_output(
            audio,
            self.sample_rate,
            output_file=output_file,
            metadata={
                "seed": effective_seed,
                "requested_seed": seed,
            },
        )


VuiTTS = VuiForTextToSpeech
