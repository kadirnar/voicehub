"""VoiceHub-native XTTS v2 inference wrapper."""

from __future__ import annotations

import hashlib
import math
import shutil
from collections.abc import Sequence
from numbers import Integral, Real
from pathlib import Path

import torch

from voicehub.models._shared import finish_audio_output, resolve_model_directory, resolve_torch_dtype, seeded_inference
from voicehub.models.tts import PreTrainedTTSModel
from voicehub.models.xtts.configuration import XTTSConfig
from voicehub.models.xtts.native import XTTS2Config, XTTS2Model, XTTS2Tokenizer, load_xtts2_checkpoint
from voicehub.models.xtts.native.configuration import XTTS2_LANGUAGES
from voicehub.models.xtts.native.metadata import (
    XTTS2_CHECKPOINT_REPOSITORY,
    XTTS2_CHECKPOINT_REVISION,
    XTTS2_CONFIG_SHA256,
    XTTS2_VOCAB_SHA256,
)
from voicehub.outputs import TTSOutput
from voicehub.tokenization.assets import read_bounded_asset
from voicehub.training.utils import NATIVE_EXPORT_DIR


def _verify_asset_digest(
    path: Path,
    *,
    expected: str,
    max_bytes: int,
) -> None:
    digest = hashlib.sha256(read_bounded_asset(path, max_bytes=max_bytes)).hexdigest()
    if digest != expected:
        raise ValueError(
            f"XTTS published asset digest mismatch for {path.name}: "
            f"expected {expected}, found {digest}.")


class XTTSForTextToSpeech(PreTrainedTTSModel):
    config_class = XTTSConfig
    default_model_name_or_path = "coqui/XTTS-v2"

    def __init__(
        self,
        config: XTTSConfig | str | None = None,
        *,
        model_path: str | None = None,
        device: str = "auto",
        lazy_load: bool = True,
        **config_overrides,
    ) -> None:
        config = self._coerce_config(config, model_path=model_path, **config_overrides)
        self._runtime_config = None
        self._tokenizer = None
        self._model_directory = None
        self._training_audio_encoder = None
        super().__init__(config, device=device, lazy_load=lazy_load)

    @staticmethod
    def _checkpoint_sample_rate(runtime_config) -> int:
        sample_rate = getattr(
            getattr(runtime_config, "audio", None),
            "output_sample_rate",
            None,
        )
        if (isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0):
            raise ValueError(
                "XTTS checkpoint configuration must define a positive "
                "`audio.output_sample_rate`.")
        return sample_rate

    @property
    def native_runtime(self):
        return self.model

    @property
    def training_audio_encoder(self):
        """Separately loaded frozen DVAE boundary, if configured."""
        return self._training_audio_encoder

    @staticmethod
    def _bundled_training_audio_artifacts(directory: str | Path | None, ) -> tuple[Path, Path] | None:
        if directory is None:
            return None
        from voicehub.models.xtts.native.dvae_checkpoint import (
            NATIVE_XTTS2_DVAE_FILENAME,
            NATIVE_XTTS2_DVAE_MEL_STATS_FILENAME,
        )

        source = Path(directory).expanduser()
        candidates = (
            source / NATIVE_EXPORT_DIR,
            source,
        )
        seen = set()
        for candidate in candidates:
            identity = str(candidate)
            if identity in seen or not candidate.is_dir():
                continue
            seen.add(identity)
            dvae = candidate / NATIVE_XTTS2_DVAE_FILENAME
            mel_stats = candidate / NATIVE_XTTS2_DVAE_MEL_STATS_FILENAME
            if dvae.is_file() and mel_stats.is_file():
                return dvae.resolve(), mel_stats.resolve()
        return None

    def _training_audio_artifacts(self, ) -> tuple[Path, Path] | None:
        for directory in (
                self._model_directory,
                self.config.name_or_path,
        ):
            bundled = self._bundled_training_audio_artifacts(directory)
            if bundled is not None:
                return bundled
        if (self.config.training_dvae_checkpoint is None or
                self.config.training_mel_stats_checkpoint is None):
            return None
        return (
            Path(self.config.training_dvae_checkpoint).expanduser(),
            Path(self.config.training_mel_stats_checkpoint).expanduser(),
        )

    def configure_training_audio_encoder(
        self,
        dvae_checkpoint: str | Path,
        mel_stats_checkpoint: str | Path,
    ):
        """Attach safe waveform-to-code preparation without changing model
        state."""
        if self.model is None or self._runtime_config is None:
            raise RuntimeError("Load the XTTS runtime before configuring its training DVAE.")
        from voicehub.models.xtts.native.dvae import XTTS2DVAEConfig
        from voicehub.models.xtts.native.dvae_checkpoint import load_xtts2_training_audio_encoder

        args = self._runtime_config.model_args
        dvae_config = XTTS2DVAEConfig(
            sample_rate=args.input_sample_rate,
            num_tokens=args.gpt_num_audio_tokens - 2,
        )
        if dvae_config.code_stride_samples != args.gpt_code_stride_len:
            raise ValueError(
                "XTTS GPT/DVAE stride mismatch: GPT expects "
                f"{args.gpt_code_stride_len}, while the native DVAE produces "
                f"{dvae_config.code_stride_samples} samples per code.")
        model_dtype = next(self.model.parameters()).dtype
        self._training_audio_encoder = load_xtts2_training_audio_encoder(
            dvae_checkpoint,
            mel_stats_checkpoint,
            config=dvae_config,
            device=self.device,
            dtype=model_dtype,
        )
        return self._training_audio_encoder

    def _load_pretrained_model(self) -> None:
        directory = resolve_model_directory(
            self.config.name_or_path,
            model_type="xtts",
            revision=self.config.revision,
            cache_dir=self.config.cache_dir,
            local_files_only=self.config.local_files_only,
        )
        native_directory = directory / NATIVE_EXPORT_DIR
        if all((native_directory / filename).is_file()
               for filename in ("config.json", "vocab.json", "model.safetensors")):
            directory = native_directory
        config_path = directory / "config.json"
        vocabulary_path = directory / "vocab.json"
        checkpoint_path = directory / "model.safetensors"
        for path in (config_path, vocabulary_path):
            if not path.is_file():
                raise FileNotFoundError(f"XTTS v2 artifact was not found: {path}.")
        if not checkpoint_path.is_file():
            legacy = directory / "model.pth"
            detail = f" Found legacy {legacy.name}." if legacy.is_file() else ""
            raise PermissionError(
                "Native XTTS v2 requires model.safetensors." + detail +
                " Run the explicit trusted conversion utility once; runtime "
                "loading never deserializes pickle.", )
        is_published_checkpoint = (
            str(self.config.name_or_path).strip() == XTTS2_CHECKPOINT_REPOSITORY and
            self.config.revision == XTTS2_CHECKPOINT_REVISION)
        if is_published_checkpoint:
            _verify_asset_digest(
                config_path,
                expected=XTTS2_CONFIG_SHA256,
                max_bytes=1024 * 1024,
            )
            _verify_asset_digest(
                vocabulary_path,
                expected=XTTS2_VOCAB_SHA256,
                max_bytes=4 * 1024 * 1024,
            )
        runtime_config = XTTS2Config.from_json(config_path)
        tokenizer = XTTS2Tokenizer.from_file(vocabulary_path)
        args = runtime_config.model_args
        if len(tokenizer) != args.gpt_number_text_tokens:
            raise ValueError(
                "XTTS vocabulary/config mismatch: "
                f"{len(tokenizer)} != {args.gpt_number_text_tokens}.", )
        with torch.device("meta"):
            model = XTTS2Model(
                runtime_config,
                start_text_token=tokenizer.start_id,
                stop_text_token=tokenizer.stop_id,
            )
        dtype = resolve_torch_dtype(
            torch,
            self.config.torch_dtype,
            self.device,
        )
        load_xtts2_checkpoint(
            model,
            checkpoint_path,
            device=self.device,
            dtype=dtype,
        )
        self.model = model.to(device=self.device)
        self._runtime_config = runtime_config
        self._tokenizer = tokenizer
        self._model_directory = directory
        self.config.sample_rate = self._checkpoint_sample_rate(runtime_config)

    def _prepare_for_training(self) -> None:
        if self._training_audio_encoder is None:
            training_artifacts = self._training_audio_artifacts()
            if training_artifacts is not None:
                self.configure_training_audio_encoder(*training_artifacts)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        for parameter in self.model.gpt.parameters():
            parameter.requires_grad_(True)
        self.model.train()
        self.model.hifigan_decoder.eval()

    def _prepare_for_inference(self) -> None:
        self.model.eval()

    def _validate_generation_inputs(self, model_inputs) -> None:
        reference = model_inputs.get("speaker_audio_path")
        if isinstance(reference, (str, Path)):
            references = (reference, )
        elif (isinstance(reference, Sequence) and not isinstance(reference, (bytes, bytearray))):
            references = tuple(reference)
        else:
            references = ()
        if (not references or any(not isinstance(item, (str, Path)) or not Path(item).expanduser().is_file()
                                  for item in references)):
            raise FileNotFoundError(
                "XTTS reference audio was not found; pass an existing path "
                "or a non-empty sequence of existing `speaker_audio_path`s.")
        language = model_inputs.get("language") or self.config.language
        if not isinstance(language, str) or not language.strip():
            raise ValueError("XTTS `language` must be a non-empty language code.")
        language = language.strip().lower()
        if language == "zh":
            language = "zh-cn"
        supported = (XTTS2_LANGUAGES if self._runtime_config is None else self._runtime_config.languages)
        if language not in supported:
            raise ValueError(f"Unsupported XTTS language: {language!r}.")
        text_is_normalized = model_inputs.get("text_is_normalized", False)
        if not isinstance(text_is_normalized, bool):
            raise TypeError("`text_is_normalized` must be a boolean.")
        do_sample = model_inputs.get("do_sample", True)
        if not isinstance(do_sample, bool):
            raise TypeError("`do_sample` must be a boolean.")
        top_k = model_inputs.get("top_k")
        if top_k is not None and (isinstance(top_k, bool) or not isinstance(top_k, Integral) or top_k < 0):
            raise ValueError("XTTS `top_k` must be a non-negative integer.")
        for name in ("temperature", "repetition_penalty"):
            value = model_inputs.get(name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, Real) or
                                      not math.isfinite(value) or value <= 0):
                raise ValueError(f"XTTS `{name}` must be finite and greater than zero.")
        top_p = model_inputs.get("top_p")
        if top_p is not None and (isinstance(top_p, bool) or not isinstance(top_p, Real) or
                                  not math.isfinite(top_p) or not 0 < top_p <= 1):
            raise ValueError("XTTS `top_p` must be finite and in the interval (0, 1].")

    def _generate(
        self,
        text: str,
        *,
        output_file: str | None = None,
        speaker_audio_path: str | Path | Sequence[str | Path],
        language: str | None = None,
        text_is_normalized: bool = False,
        seed: int | None = None,
        speed: float = 1.0,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        do_sample: bool = True,
        repetition_penalty: float | None = None,
        max_new_tokens: int | None = None,
    ) -> TTSOutput:
        language = (language or self.config.language).lower()
        if language == "zh":
            language = "zh-cn"
        if language not in self._runtime_config.languages:
            raise ValueError(f"Unsupported XTTS language: {language!r}.")
        tokens = torch.tensor(
            [self._tokenizer.encode(
                text,
                language=language,
                preprocessed=text_is_normalized,
            )],
            device=self.device,
            dtype=torch.long,
        )
        if tokens.shape[1] > self.model.gpt.max_text_tokens - 2:
            raise ValueError(
                "XTTS text exceeds the checkpoint's autoregressive context "
                f"limit ({tokens.shape[1]} tokens).")
        conditioning, speaker = self.model.conditioning_latents(
            speaker_audio_path,
            max_ref_length=self._runtime_config.max_ref_len,
            gpt_cond_length=self._runtime_config.gpt_cond_len,
            chunk_length=self._runtime_config.gpt_cond_chunk_len,
            sound_norm_refs=self._runtime_config.sound_norm_refs,
        )
        temperature = (self._runtime_config.temperature if temperature is None else float(temperature))
        top_k = self._runtime_config.top_k if top_k is None else int(top_k)
        top_p = self._runtime_config.top_p if top_p is None else float(top_p)
        repetition_penalty = (
            self._runtime_config.repetition_penalty
            if repetition_penalty is None else float(repetition_penalty))
        with seeded_inference(
                seed,
                device=self.device,
                model_type="xtts",
        ) as used_seed:
            waveform = self.model.synthesize_tokens(
                tokens,
                conditioning,
                speaker,
                speed=speed,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                max_new_tokens=max_new_tokens,
            ).squeeze().detach().float().cpu()
        return finish_audio_output(
            waveform,
            self.sample_rate,
            output_file=output_file,
            metadata={
                "backend": "voicehub-native",
                "checkpoint_format": "safetensors",
                "language": language,
                "text_is_normalized": text_is_normalized,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
                "speed": speed,
                "requested_seed": seed,
                "seed": used_seed,
            },
        )

    def save_pretrained(
        self,
        save_directory: str | Path,
        *,
        include_native_export: bool = True,
    ) -> Path:
        output_directory = super().save_pretrained(
            save_directory,
            include_native_export=include_native_export,
        )
        if (include_native_export and self._training_audio_encoder is not None and
                self._bundled_training_audio_artifacts(output_directory) is not None):
            portable_values = self.config.to_dict()
            portable_values["training_dvae_checkpoint"] = None
            portable_values["training_mel_stats_checkpoint"] = None
            self.config_class.from_dict(portable_values).save_pretrained(output_directory, )
        return output_directory

    def _save_pretrained(self, save_directory: Path) -> None:
        if self.model is None:
            self.load()
        if self._model_directory is None:
            raise RuntimeError("XTTS source artifacts are unavailable for export.")
        save_directory.mkdir(parents=True, exist_ok=True)
        from voicehub.models.xtts.native.checkpoint import save_xtts2_checkpoint

        save_xtts2_checkpoint(
            self.native_runtime,
            save_directory / "model.safetensors",
        )
        for filename in ("config.json", "vocab.json"):
            source = self._model_directory / filename
            if not source.is_file():
                raise FileNotFoundError(f"XTTS export source artifact was not found: {source}.")
            shutil.copy2(source, save_directory / filename)
        if self._training_audio_encoder is not None:
            from voicehub.models.xtts.native.dvae_checkpoint import save_xtts2_training_audio_encoder

            save_xtts2_training_audio_encoder(
                self._training_audio_encoder,
                save_directory,
            )


XTTS = XTTSForTextToSpeech

__all__ = ["XTTS", "XTTSForTextToSpeech"]
