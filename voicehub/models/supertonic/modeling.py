"""VoiceHub-native Supertonic 3 inference and fine-tuning lifecycle."""

from __future__ import annotations

import math
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from voicehub.hub import write_json_file
from voicehub.models._shared import finish_audio_output, resolve_torch_dtype, seeded_inference
from voicehub.models.supertonic.configuration import SUPERTONIC_SAMPLE_RATE, SupertonicConfig
from voicehub.models.supertonic.native.artifacts import (
    SupertonicArtifacts,
    resolve_supertonic_artifacts,
    resolve_supertonic_style,
)
from voicehub.models.supertonic.native.checkpoint import save_supertonic_native_weights
from voicehub.models.supertonic.native.frontend import AVAILABLE_LANGUAGES, SupertonicStyle, length_mask
from voicehub.models.supertonic.native.metadata import (
    SUPERTONIC_CHECKPOINT_REPOSITORY,
    SUPERTONIC_CHECKPOINT_REVISION,
    SUPERTONIC_GRAPH_FILES,
    SUPERTONIC_SOURCE_REPOSITORY,
    SUPERTONIC_SOURCE_REVISION,
)
from voicehub.models.supertonic.native.runtime import NativeSupertonicRuntime, load_native_supertonic_runtime
from voicehub.models.tts import PreTrainedTTSModel
from voicehub.outputs import TTSOutput

SUPPORTED_LANGUAGES = AVAILABLE_LANGUAGES


def _strings(value: Any, *, name: str) -> tuple[str, ...]:
    is_sequence = isinstance(value, Sequence)
    is_binary_value = isinstance(value, (bytes, bytearray))
    if isinstance(value, str):
        result = (value, )
    elif is_sequence and not is_binary_value:
        result = tuple(value)
    else:
        raise TypeError(f"Supertonic `{name}` must be a string or sequence.")
    if not result or any(not isinstance(item, str) or not item.strip() for item in result):
        raise ValueError(f"Supertonic `{name}` values must be non-empty strings.")
    return tuple(item.strip() for item in result)


class SupertonicForTextToSpeech(PreTrainedTTSModel):
    """Run all published Supertonic graphs with differentiable PyTorch ops."""

    config_class = SupertonicConfig
    default_model_name_or_path = SUPERTONIC_CHECKPOINT_REPOSITORY
    architecture_family = "supertonic"
    passthrough_generation_options = frozenset()

    def __init__(
        self,
        config: SupertonicConfig | str | Path | None = None,
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
            raise ValueError("`token` cannot be empty.")
        self._hub_token = token.strip() if isinstance(token, str) else token
        self.artifacts: SupertonicArtifacts | None = None
        self.native_config = None
        self._style_cache: dict[Path, SupertonicStyle] = {}
        config = self._coerce_config(
            config,
            model_path=model_path,
            **config_overrides,
        )
        if not isinstance(config, SupertonicConfig):
            raise TypeError("Supertonic requires a SupertonicConfig.")
        config.validate()
        super().__init__(config, device=device, lazy_load=lazy_load)

    def _model_dtype(self) -> torch.dtype:
        if self.config.torch_dtype == "auto":
            return torch.float32
        return resolve_torch_dtype(
            torch,
            self.config.torch_dtype,
            self.device,
        )

    def _load_pretrained_model(self) -> None:
        source = (self.config.name_or_path or self.default_model_name_or_path)
        artifacts = resolve_supertonic_artifacts(
            source,
            revision=self.config.revision,
            cache_dir=self.config.cache_dir,
            token=self._hub_token,
            local_files_only=self.config.local_files_only,
            verify_integrity=self.config.verify_integrity,
        )
        runtime = load_native_supertonic_runtime(
            artifacts,
            device=self.device,
            dtype=self._model_dtype(),
        )
        self.artifacts = artifacts.without_materialized_graphs()
        self.native_config = runtime.architecture
        self.model = runtime
        self.config.revision = artifacts.revision
        self.config.sample_rate = runtime.sample_rate

    def _validate_training_runtime(self) -> None:
        if not self.config.enable_preprocessed_training:
            raise ValueError(
                "Supertonic's released artifact omits the audio/style "
                "encoders and original optimizer recipe. Set "
                "`enable_preprocessed_training=True` only when supplying "
                "style tensors, target durations, and/or target latents for "
                "the explicitly reconstructed published-graph objective.")

    def _prepare_for_training(self) -> None:
        if not isinstance(self.model, NativeSupertonicRuntime):
            raise TypeError("Supertonic did not load its native runtime.")
        self.model.prepare_for_training()

    def _prepare_for_inference(self) -> None:
        if not isinstance(self.model, NativeSupertonicRuntime):
            raise TypeError("Supertonic did not load its native runtime.")
        self.model.prepare_for_inference()

    def _style(self, voice: str | Path) -> SupertonicStyle:
        if self.artifacts is None:
            raise RuntimeError("Supertonic artifacts are not loaded.")
        path = resolve_supertonic_style(
            self.artifacts,
            voice,
            cache_dir=self.config.cache_dir,
            token=self._hub_token,
            local_files_only=self.config.local_files_only,
            verify_integrity=self.config.verify_integrity,
        )
        try:
            return self._style_cache[path]
        except KeyError:
            style = SupertonicStyle.from_file(path)
            self._style_cache[path] = style
            return style

    def _validate_generation_inputs(
        self,
        model_inputs: dict[str, Any],
    ) -> None:
        voice = model_inputs.get("voice") or self.config.voice
        if not isinstance(voice, (str, Path)) or not str(voice).strip():
            raise ValueError("`voice` must be a non-empty voice ID or style JSON path.")
        language = model_inputs.get("language") or self.config.language
        normalized = (language.strip().lower() if isinstance(language, str) else language)
        if normalized not in SUPPORTED_LANGUAGES:
            supported = ", ".join(sorted(SUPPORTED_LANGUAGES))
            raise ValueError(f"Unsupported Supertonic language {language!r}. "
                             f"Supported: {supported}.")
        total_steps = model_inputs.get("total_steps", 5)
        if (isinstance(total_steps, bool) or not isinstance(total_steps, int) or total_steps <= 0):
            raise ValueError("`total_steps` must be a positive integer.")
        for name, default, allow_zero in (
            ("speed", 1.05, False),
            ("silence_duration", 0.3, True),
        ):
            value = model_inputs.get(name, default)
            if (isinstance(value, bool) or not isinstance(value,
                                                          (int, float)) or not math.isfinite(float(value)) or
                    float(value) < 0 or (not allow_zero and float(value) == 0)):
                qualifier = "non-negative" if allow_zero else "positive"
                raise ValueError(f"`{name}` must be a finite {qualifier} number.")

    def _trim_waveform(self, audio: Any, duration: Any):
        """Retain the public padding helper for direct batched inference."""
        if audio is None or duration is None:
            raise RuntimeError("Supertonic returned no audio waveform.")
        waveform = audio[0] if getattr(audio, "ndim", 1) > 1 else audio
        first = duration[0] if hasattr(duration, "__getitem__") else duration
        seconds = float(first.item() if hasattr(first, "item") else first)
        if not math.isfinite(seconds) or seconds <= 0:
            raise RuntimeError(f"Supertonic returned an invalid audio duration: {seconds}.")
        sample_count = min(
            len(waveform),
            max(0, round(self.sample_rate * seconds)),
        )
        if sample_count == 0:
            raise RuntimeError("Supertonic returned an empty audio waveform.")
        return waveform[:sample_count]

    def _generate(
        self,
        text: str,
        *,
        output_file: str | None = None,
        voice: str | Path | None = None,
        language: str | None = None,
        total_steps: int = 5,
        speed: float = 1.05,
        silence_duration: float = 0.3,
        seed: int | None = None,
    ) -> TTSOutput:
        selected_voice = voice or self.config.voice
        selected_language = (language or self.config.language).strip().lower()
        style = self._style(selected_voice)
        with seeded_inference(
                seed,
                device=self.device,
                model_type="supertonic",
        ) as effective_seed:
            audio, duration = self.model.synthesize(
                text,
                selected_language,
                style,
                total_steps=total_steps,
                speed=speed,
                silence_duration=silence_duration,
            )
        return finish_audio_output(
            audio[0].detach().cpu(),
            self.sample_rate,
            output_file=output_file,
            metadata={
                "voice": str(selected_voice),
                "language": selected_language,
                "duration_seconds": float(duration[0].item()),
                "seed": effective_seed,
                "requested_seed": seed,
                "runtime": "voicehub-native-pytorch",
            },
        )

    def prepare_training_inputs(
        self,
        inputs: dict[str, Any],
        *,
        phase: str,
    ) -> dict[str, Any]:
        """Prepare text only; acoustic/style supervision remains explicit."""
        if phase != "published_graph":
            raise ValueError(f"Unknown Supertonic training phase {phase!r}.")
        if not isinstance(self.model, NativeSupertonicRuntime):
            raise RuntimeError("Load Supertonic for training before preparing a batch.")
        prepared = dict(inputs)
        if "text_ids" not in prepared:
            texts = _strings(prepared.pop("text", None), name="text")
            languages = _strings(
                prepared.pop("language", self.config.language),
                name="language",
            )
            if len(languages) == 1 and len(texts) > 1:
                languages = languages * len(texts)
            if len(texts) != len(languages):
                raise ValueError("Supertonic text and language batch sizes differ.")
            text_ids, text_mask = self.model.processor.encode(
                texts,
                languages,
                device=self.model.device,
            )
            prepared["text_ids"] = text_ids
            prepared["text_mask"] = text_mask
        elif "text_mask" not in prepared:
            lengths = prepared.pop("text_lengths", None)
            if lengths is None:
                raise ValueError(
                    "Pre-tokenized Supertonic batches require `text_mask` "
                    "or integer `text_lengths`.")
            ids = torch.as_tensor(prepared["text_ids"])
            prepared["text_mask"] = length_mask(
                torch.as_tensor(lengths, dtype=torch.int64),
                ids.shape[1],
            )

        style = prepared.pop("style", None)
        if style is not None:
            if not isinstance(style, SupertonicStyle):
                raise TypeError("Supertonic `style` must be a SupertonicStyle instance.")
            if "style_ttl" in prepared or "style_dp" in prepared:
                raise ValueError("Pass either `style` or explicit style tensors, not both.")
            prepared["style_ttl"] = style.ttl
            prepared["style_dp"] = style.duration
        if "style_ttl" not in prepared or "style_dp" not in prepared:
            raise ValueError("Supertonic fine-tuning requires `style_ttl` and `style_dp`.")

        aliases = {
            "duration": "target_duration",
            "duration_seconds": "target_duration",
            "latent": "target_latent",
            "latents": "target_latent",
            "audio": "target_audio",
            "audio_values": "target_audio",
        }
        for source, target in aliases.items():
            if source not in prepared:
                continue
            if target in prepared:
                raise ValueError(f"Both `{source}` and `{target}` were provided.")
            prepared[target] = prepared.pop(source)
        prepared.setdefault(
            "duration_weight",
            self.config.training_duration_loss_weight,
        )
        prepared.setdefault(
            "flow_weight",
            self.config.training_flow_loss_weight,
        )
        prepared.setdefault(
            "vocoder_weight",
            self.config.training_vocoder_loss_weight,
        )
        return prepared

    def _save_pretrained(self, save_directory: Path) -> None:
        if (not isinstance(self.model, NativeSupertonicRuntime) or self.artifacts is None):
            self.load()
        save_directory.mkdir(parents=True, exist_ok=True)
        config_values = self.config.to_dict()
        config_values.update({
            'architectures': ["SupertonicForTextToSpeech"],
            "sample_rate": SUPERTONIC_SAMPLE_RATE,
            "voicehub_checkpoint_format": "native-supertonic-v1",
        })
        write_json_file(save_directory / "config.json", config_values)

        graph_directory = save_directory / "onnx"
        graph_directory.mkdir(parents=True, exist_ok=True)
        for role, filename in SUPERTONIC_GRAPH_FILES.items():
            shutil.copy2(
                self.artifacts.graphs[role],
                graph_directory / filename,
            )
        shutil.copy2(
            self.artifacts.architecture_config,
            graph_directory / "tts.json",
        )
        shutil.copy2(
            self.artifacts.unicode_indexer,
            graph_directory / "unicode_indexer.json",
        )
        weight_paths = save_supertonic_native_weights(
            self.model,
            save_directory,
        )

        style_directory = save_directory / "voice_styles"
        style_directory.mkdir(parents=True, exist_ok=True)
        default_style = resolve_supertonic_style(
            self.artifacts,
            self.config.voice,
            cache_dir=self.config.cache_dir,
            token=self._hub_token,
            local_files_only=self.config.local_files_only,
            verify_integrity=self.config.verify_integrity,
        )
        shutil.copy2(
            default_style,
            style_directory / f"{Path(self.config.voice).stem}.json",
        )
        for path in sorted(self._style_cache, key=str):
            destination = style_directory / path.name
            if not destination.exists():
                shutil.copy2(path, destination)

        write_json_file(
            save_directory / "voicehub_supertonic.json",
            {
                "format_version": 1,
                "architecture": "supertonic-3",
                "architecture_source": {
                    "repository": SUPERTONIC_SOURCE_REPOSITORY,
                    "revision": SUPERTONIC_SOURCE_REVISION,
                },
                "base_checkpoint": {
                    "repository": str(self.artifacts.source),
                    "revision": self.artifacts.revision,
                    "official_repository": SUPERTONIC_CHECKPOINT_REPOSITORY,
                    "official_revision": SUPERTONIC_CHECKPOINT_REVISION,
                },
                "graphs": {
                    role: f"onnx/{filename}"
                    for role, filename in SUPERTONIC_GRAPH_FILES.items()
                },
                "native_weights": {
                    role: str(path.relative_to(save_directory))
                    for role, path in weight_paths.items()
                },
                "training_scope": ("reconstructed-published-graph-precomputed-latents"),
                "author_recipe_published": False,
            },
        )


SupertonicTTS = SupertonicForTextToSpeech

__all__ = [
    "SUPPORTED_LANGUAGES",
    "SupertonicConfig",
    "SupertonicForTextToSpeech",
    "SupertonicTTS",
]
