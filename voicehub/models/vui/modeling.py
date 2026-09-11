"""Vui integration using the architecture source included in VoiceHub."""

from __future__ import annotations

import math
from typing import Any

from voicehub.dependencies import import_optional
from voicehub.models._shared import finish_audio_output, seeded_inference
from voicehub.models.tts import PreTrainedTTSModel
from voicehub.models.vui.artifacts import VUI_MODEL_FILENAME, VuiArtifacts, resolve_vui_artifacts
from voicehub.outputs import TTSOutput

from .configuration import VuiConfig


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
