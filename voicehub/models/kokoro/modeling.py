"""VoiceHub-native Kokoro inference and preprocessed fine-tuning lifecycle."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from voicehub.hub import write_json_file
from voicehub.models._shared import finish_audio_output, seeded_inference
from voicehub.models.kokoro.configuration import KOKORO_SAMPLE_RATE, KokoroConfig
from voicehub.models.tts import PreTrainedTTSModel
from voicehub.outputs import TTSOutput


class KokoroForTextToSpeech(PreTrainedTTSModel):
    """Kokoro synthesis without Transformers, Misaki, NumPy, or Hub clients."""

    config_class = KokoroConfig
    default_model_name_or_path = "hexgrad/Kokoro-82M"
    architecture_family = "kokoro"
    passthrough_generation_options = frozenset()

    def __init__(
        self,
        config: KokoroConfig | str | Path | None = None,
        *,
        model_path: str | Path | None = None,
        device: str = "auto",
        lazy_load: bool = True,
        token: str | bool | None = None,
        **config_overrides: Any,
    ) -> None:
        text_frontend = config_overrides.pop("text_frontend", None)
        if token is not None and not isinstance(token, (str, bool)):
            raise TypeError("`token` must be a string, boolean, or None.")
        if isinstance(token, str) and not token.strip():
            raise ValueError("`token` cannot be an empty string.")
        if text_frontend is not None and not callable(text_frontend):
            raise TypeError("`text_frontend` must be callable or None.")
        self._hub_token = token.strip() if isinstance(token, str) else token
        self._text_frontend = text_frontend
        self._requested_device = device
        self.pipeline: Any | None = None
        self.artifacts: Any | None = None
        self.native_config: Any | None = None
        self.training_model: Any | None = None
        config = self._coerce_config(
            config,
            model_path=model_path,
            **config_overrides,
        )
        config.validate()
        super().__init__(config, device=device, lazy_load=lazy_load)

    def _runtime_device(self) -> str:
        """Return a Kokoro-compatible device for the released decoder.

        PyTorch selects MPS for ``device="auto"`` on Apple Silicon, but
        the released iSTFT decoder still contains operations without
        complete MPS coverage. Keep an explicitly requested MPS device
        visible to callers; only the automatic selection falls back to
        CPU unless the user has opted into PyTorch's MPS fallback
        behavior.
        """
        import os

        device_type = str(self.device).split(":", 1)[0].lower()
        if (self._requested_device == "auto" and device_type == "mps" and
                os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "1"):
            self.device = "cpu"
        return self.device

    def _model_dtype(self):
        import torch

        configured = self.config.torch_dtype
        if configured == "auto":
            return (torch.float16 if torch.device(self.device).type == "cuda" else torch.float32)
        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[configured]
        if torch.device(self.device).type == "cpu" and dtype == torch.float16:
            raise ValueError("Native Kokoro does not support float16 execution on CPU.")
        return dtype

    def _load_pretrained_model(self) -> None:
        from voicehub.models.kokoro.pipeline import KPipeline

        runtime_device = self._runtime_device()
        source = self.config.name_or_path or self.default_model_name_or_path
        pipeline = KPipeline(
            lang_code=self.config.language_code,
            repo_id=source,
            model=True,
            frontend=self._text_frontend,
            device=runtime_device,
            checkpoint_filename=self.config.checkpoint_filename,
            revision=self.config.revision,
            cache_dir=self.config.cache_dir,
            token=self._hub_token,
            local_files_only=self.config.local_files_only,
            allow_legacy_checkpoint_conversion=(self.config.allow_legacy_checkpoint_conversion),
        )
        model = pipeline.model
        if model is None:  # pragma: no cover - constructor invariant
            raise RuntimeError("Kokoro pipeline did not construct a model.")
        dtype = self._model_dtype()
        model.to(device=runtime_device, dtype=dtype)
        self.pipeline = pipeline
        self.artifacts = pipeline.artifacts
        self.native_config = model.native_config
        self.model = model
        self.config.revision = pipeline.revision
        self.config.sample_rate = KOKORO_SAMPLE_RATE

    def _validate_training_runtime(self) -> None:
        if not self.config.enable_preprocessed_training:
            raise ValueError(
                "Kokoro preprocessed decoder fine-tuning is disabled. Set "
                "`enable_preprocessed_training=True` only with phoneme IDs, "
                "style vectors, duration/alignment, and prepared acoustic "
                "targets. The repository does not release the complete "
                "raw-audio StyleTTS2 recipe.")

    def _prepare_for_training(self) -> None:
        self.model.train()
        if self.training_model is None:
            from voicehub.models.kokoro.training import KokoroPreprocessedTrainingModel

            self.training_model = KokoroPreprocessedTrainingModel(
                self.model,
                duration_loss_weight=(self.config.training_duration_loss_weight),
                f0_loss_weight=self.config.training_f0_loss_weight,
                energy_loss_weight=self.config.training_energy_loss_weight,
                waveform_loss_weight=(self.config.training_waveform_loss_weight),
                spectral_loss_weight=(self.config.training_spectral_loss_weight),
            )
        self.training_model.train()

    def _prepare_for_inference(self) -> None:
        self.model.eval()
        if self.training_model is not None:
            self.training_model.eval()

    def _validate_generation_inputs(
        self,
        model_inputs: dict[str, Any],
    ) -> None:
        language_code = self.config.language_code
        if not isinstance(language_code, str) or not language_code.strip():
            raise ValueError("`language_code` must be a non-empty Kokoro language code.")
        voice = model_inputs.get("voice", "af_heart")
        try:
            import torch
        except ModuleNotFoundError:  # pragma: no cover - package invariant
            torch = None
        if not (isinstance(voice, str) and voice.strip() or torch is not None and torch.is_tensor(voice)):
            raise ValueError("`voice` must be a non-empty name or Kokoro style tensor.")
        speed = model_inputs.get("speed", 1.0)
        if (not isinstance(speed, (int, float)) or isinstance(speed, bool) or not math.isfinite(speed) or
                speed <= 0):
            raise ValueError("`speed` must be a finite positive number.")
        split_pattern = model_inputs.get("split_pattern", r"\n+")
        if split_pattern is not None:
            if not isinstance(split_pattern, str) or not split_pattern:
                raise ValueError("`split_pattern` must be a non-empty regex or None.")
            try:
                re.compile(split_pattern)
            except re.error as exc:
                raise ValueError(f"Invalid `split_pattern`: {exc}.") from exc
        phonemes = model_inputs.get("phonemes")
        if phonemes is not None and (not isinstance(phonemes, str) or not phonemes):
            raise ValueError("`phonemes` must be a non-empty string or None.")

    def _generate(
        self,
        text: str,
        *,
        voice: Any = "af_heart",
        speed: float = 1.0,
        split_pattern: str | None = r"\n+",
        phonemes: str | None = None,
        output_file: str | None = None,
        seed: int | None = None,
    ) -> TTSOutput:
        chunks: list[Any] = []
        segments: list[str] = []
        emitted_phonemes: list[str] = []
        frontend_ids: list[str] = []
        with seeded_inference(
                seed,
                device=self.device,
                model_type="kokoro",
        ) as effective_seed:
            for result in self.pipeline(
                    text,
                    voice=voice,
                    speed=speed,
                    split_pattern=split_pattern,
                    phonemes=phonemes,
            ):
                if result.audio is not None:
                    chunks.append(result.audio.reshape(-1))
                    segments.append(result.graphemes)
                    emitted_phonemes.append(result.phonemes)
                    if result.frontend_id is not None:
                        frontend_ids.append(result.frontend_id)
        if not chunks:
            raise RuntimeError("Kokoro returned no audio.")
        import torch

        return finish_audio_output(
            torch.cat(chunks),
            self.sample_rate,
            output_file=output_file,
            metadata={
                "segments": tuple(segments),
                "phonemes": tuple(emitted_phonemes),
                "frontend_ids": tuple(frontend_ids),
                "source_equivalent_g2p": phonemes is not None,
                "voice": (voice if isinstance(voice, str) else "<style-tensor>"),
                "speed": speed,
                "seed": effective_seed,
                "requested_seed": seed,
            },
        )

    @staticmethod
    def _batch_strings(value: Any, *, name: str) -> tuple[str, ...]:
        if isinstance(value, str):
            values = (value, )
        elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
            values = tuple(value)
        else:
            raise TypeError(f"Kokoro `{name}` must be a string or sequence.")
        if not values or any(not isinstance(item, str) or not item for item in values):
            raise ValueError(f"Kokoro `{name}` values must be non-empty.")
        return values

    def _tokenize_training_phonemes(
        self,
        phonemes: Any,
    ) -> dict[str, Any]:
        import torch

        values = self._batch_strings(phonemes, name="phonemes")
        encoded = [[0, *self.model.tokenize_phonemes(value), 0] for value in values]
        lengths = torch.tensor(
            [len(item) for item in encoded],
            device=self.device,
            dtype=torch.long,
        )
        input_ids = torch.zeros(
            (len(encoded), int(lengths.max().item())),
            device=self.device,
            dtype=torch.long,
        )
        for index, item in enumerate(encoded):
            input_ids[index, :len(item)] = torch.tensor(
                item,
                device=self.device,
                dtype=torch.long,
            )
        return {"input_ids": input_ids, "input_lengths": lengths}

    def prepare_training_inputs(
        self,
        inputs: dict[str, Any],
        *,
        phase: str,
    ) -> dict[str, Any]:
        """Prepare only phoneme/style/acoustic batches; never invent G2P."""
        if phase not in {"duration", "acoustic"}:
            raise ValueError(f"Unknown Kokoro training phase {phase!r}.")
        import torch

        prepared = dict(inputs)
        if "input_ids" not in prepared:
            phonemes = prepared.pop("phonemes", None)
            if phonemes is None:
                if "text" in prepared:
                    raise ValueError(
                        "Kokoro training does not apply the fallback raw-text "
                        "frontend. Precompute author-compatible phonemes and "
                        "pass `phonemes` or `input_ids`.")
                raise ValueError("Kokoro training requires `input_ids` or `phonemes`.")
            prepared.update(self._tokenize_training_phonemes(phonemes))
        else:
            input_ids = prepared["input_ids"]
            if not torch.is_tensor(input_ids):
                input_ids = torch.as_tensor(input_ids)
            if input_ids.ndim == 1:
                input_ids = input_ids.unsqueeze(0)
            if input_ids.ndim != 2:
                raise ValueError("Kokoro `input_ids` must have shape [batch, text].")
            if (input_ids.dtype == torch.bool or input_ids.is_floating_point() or input_ids.is_complex()):
                raise TypeError("Kokoro `input_ids` must use an integer dtype.")
            prepared["input_ids"] = input_ids.long()
            if "input_lengths" not in prepared:
                prepared["input_lengths"] = torch.full(
                    (input_ids.shape[0], ),
                    input_ids.shape[1],
                    dtype=torch.long,
                    device=input_ids.device,
                )
        if "ref_s" not in prepared:
            voice = prepared.pop("voice", None)
            if voice is None:
                raise ValueError(
                    "Kokoro training requires a prepared `ref_s` style "
                    "vector or a loaded `voice` name.")
            voices = (self._batch_strings(voice, name="voice") if not torch.is_tensor(voice) else (voice, ))
            if len(voices) not in {1, prepared["input_ids"].shape[0]}:
                raise ValueError("Kokoro voice count must be one or match batch size.")
            if len(voices) == 1:
                voices = voices * prepared["input_ids"].shape[0]
            phoneme_counts = (prepared["input_ids"] != 0).sum(dim=1)
            styles = []
            for index, item in enumerate(voices):
                pack = self.pipeline.load_voice(item)
                length = int(phoneme_counts[index].item())
                if length < 1 or length > pack.shape[0]:
                    raise ValueError("Kokoro phoneme-token count is outside the voice pack.")
                styles.append(pack[length - 1, 0])
            prepared["ref_s"] = torch.stack(styles)
        if "durations" not in prepared:
            raise ValueError("Kokoro training requires integer `durations`.")
        if phase == "acoustic" and "audio_values" not in prepared:
            audio = prepared.pop("audio", prepared.pop("labels", None))
            if audio is None:
                raise ValueError("Kokoro acoustic training requires `audio_values`.")
            prepared["audio_values"] = audio
        return prepared

    def _native_export_config(self) -> dict[str, Any]:
        if self.native_config is None:
            raise RuntimeError("Kokoro must be loaded before export.")
        values = self.native_config.to_dict()
        values.update({
            "model_type": "kokoro",
            'architectures': ["KokoroForTextToSpeech"],
            "sample_rate": KOKORO_SAMPLE_RATE,
            "language_code": self.config.language_code,
            "torch_dtype": self.config.torch_dtype,
            "enable_preprocessed_training": (self.config.enable_preprocessed_training),
            "training_duration_loss_weight": (self.config.training_duration_loss_weight),
            "training_f0_loss_weight": (self.config.training_f0_loss_weight),
            "training_energy_loss_weight": (self.config.training_energy_loss_weight),
            "training_waveform_loss_weight": (self.config.training_waveform_loss_weight),
            "training_spectral_loss_weight": (self.config.training_spectral_loss_weight),
        })
        return values

    def _save_pretrained(self, save_directory: Path) -> None:
        from voicehub.models.kokoro.native.checkpoint import save_native_kokoro_checkpoint, save_native_kokoro_voice

        save_directory.mkdir(parents=True, exist_ok=True)
        write_json_file(
            save_directory / "config.json",
            self._native_export_config(),
        )
        save_native_kokoro_checkpoint(
            self.model,
            save_directory / "model.safetensors",
        )
        if self.pipeline is not None and self.pipeline.voices:
            voice_directory = save_directory / "voices"
            voice_directory.mkdir(parents=True, exist_ok=True)
            for name, voice in sorted(self.pipeline.voices.items()):
                if "," in name or "/" in name or "\\" in name:
                    continue
                save_native_kokoro_voice(
                    voice,
                    voice_directory / f"{name}.safetensors",
                )


KokoroTTS = KokoroForTextToSpeech

__all__ = [
    "KOKORO_SAMPLE_RATE",
    "KokoroConfig",
    "KokoroForTextToSpeech",
    "KokoroTTS",
]
