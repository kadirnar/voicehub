"""Chatterbox integration using source included in VoiceHub."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from voicehub.models._shared import finish_audio_output, seeded_inference, validate_local_file
from voicehub.models.tts import PreTrainedTTSModel
from voicehub.outputs import TTSOutput

from .configuration import ChatterboxConfig


class ChatterboxForTextToSpeech(PreTrainedTTSModel):
    """Zero-shot voice cloning without the ``chatterbox-tts`` package."""

    config_class = ChatterboxConfig
    default_model_name_or_path = "ResembleAI/chatterbox"
    passthrough_generation_options = frozenset({
        "cfg_weight",
        "exaggeration",
        "max_new_tokens",
        "min_p",
        "repetition_penalty",
        "temperature",
        "top_p",
    })

    def __init__(
        self,
        config: ChatterboxConfig | str | None = None,
        *,
        model_path: str | None = None,
        device: str = "auto",
        lazy_load: bool = True,
        **config_overrides,
    ):
        config = self._coerce_config(
            config,
            model_path=model_path,
            **config_overrides,
        )
        super().__init__(config, device=device, lazy_load=lazy_load)

    def _load_pretrained_model(self) -> None:
        from voicehub.models.chatterbox.tts import ChatterboxTTS

        source = Path(self.config.name_or_path).expanduser()
        if source.is_dir():
            model = ChatterboxTTS.from_local(source.resolve(), self.device)
        elif source.exists():
            raise ValueError("Chatterbox expects a checkpoint directory, but received "
                             f"a file: {source}.")
        else:
            from voicehub.models.chatterbox.checkpoint import CHECKPOINT_REVISION

            model = ChatterboxTTS.from_pretrained(
                device=self.device,
                repo_id=self.config.name_or_path,
                revision=(self.config.checkpoint_revision or CHECKPOINT_REVISION),
            )
        if not callable(getattr(model, "generate", None)):
            raise TypeError("The loaded Chatterbox runtime does not implement generate().")
        sample_rate = int(getattr(model, "sr", 0))
        if sample_rate <= 0:
            raise ValueError("The loaded Chatterbox runtime reported an invalid sample rate.")
        self.model = model
        self.device = str(getattr(model, "device", self.device))
        self.config.sample_rate = sample_rate

    def _prepare_for_inference(self) -> None:
        """Put every trainable Chatterbox submodule in serving mode."""
        if self.model is None:
            return
        for name in ("t3", "s3gen", "ve"):
            component = getattr(self.model, name, None)
            if component is not None and hasattr(component, "eval"):
                component.eval()

    def _set_training_device(self, device: str) -> None:
        """Keep the plain upstream runtime aligned with Trainer placement."""
        super()._set_training_device(device)
        if self.model is None:
            return
        self.model.device = str(device)
        for name in ("t3", "s3gen", "ve", "conds"):
            component = getattr(self.model, name, None)
            move = getattr(component, "to", None)
            if callable(move):
                moved = move(device)
                if moved is not None:
                    setattr(self.model, name, moved)

    def _validate_generation_inputs(self, model_inputs: dict[str, Any]) -> None:
        super()._validate_generation_inputs(model_inputs)
        speaker_prompt = model_inputs.get("speaker_audio_path")
        audio_prompt = model_inputs.get("audio_prompt_path")
        if speaker_prompt is not None and audio_prompt is not None:
            raise ValueError("Pass either `speaker_audio_path` or `audio_prompt_path`, "
                             "not both.")
        speaker_path = validate_local_file(
            speaker_prompt,
            option_name="speaker_audio_path",
        )
        audio_path = validate_local_file(
            audio_prompt,
            option_name="audio_prompt_path",
        )
        if speaker_path is not None:
            model_inputs["speaker_audio_path"] = str(speaker_path)
        if audio_path is not None:
            model_inputs["audio_prompt_path"] = str(audio_path)

        max_new_tokens = model_inputs.get("max_new_tokens", 1000)
        if (isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens <= 0):
            raise ValueError("`max_new_tokens` must be a positive integer.")

        constraints = {
            "repetition_penalty": (0.0, None, False),
            "min_p": (0.0, 1.0, True),
            "top_p": (0.0, 1.0, False),
            "exaggeration": (0.0, None, True),
            "cfg_weight": (0.0, None, True),
            "temperature": (0.0, None, False),
        }
        for name, (minimum, maximum, inclusive_minimum) in constraints.items():
            value = model_inputs.get(name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"`{name}` must be a finite number.")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"`{name}` must be finite.")
            valid_minimum = (value >= minimum if inclusive_minimum else value > minimum)
            if not valid_minimum or (maximum is not None and value > maximum):
                right = f", {maximum}]" if maximum is not None else ", infinity)"
                left = "[" if inclusive_minimum else "("
                raise ValueError(f"`{name}` must be in the interval {left}{minimum}{right}.")

    def _generate(
        self,
        text: str,
        *,
        output_file: str | None = None,
        speaker_audio_path: str | None = None,
        audio_prompt_path: str | None = None,
        seed: int | None = None,
        max_new_tokens: int = 1000,
        **generation_options,
    ) -> TTSOutput:
        if self.model is None:
            raise RuntimeError("Chatterbox must be loaded before generation.")
        prompt_path = speaker_audio_path or audio_prompt_path
        with seeded_inference(
                seed,
                device=self.device,
                model_type="chatterbox",
        ) as effective_seed:
            waveform = self.model.generate(
                text,
                audio_prompt_path=prompt_path,
                max_new_tokens=max_new_tokens,
                **generation_options,
            )
        return finish_audio_output(
            waveform,
            self.sample_rate,
            output_file=output_file,
            metadata={
                "voice_cloned": prompt_path is not None,
                "max_new_tokens": max_new_tokens,
                "seed": effective_seed,
                "requested_seed": seed,
            },
        )


ChatterboxInference = ChatterboxForTextToSpeech
