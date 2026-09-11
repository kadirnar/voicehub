"""Inference lifecycle for the native StyleTTS 2 architecture."""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from voicehub.models.styletts2.native.checkpoint import load_styletts2_checkpoint, read_legacy_styletts2_checkpoint
from voicehub.models.styletts2.native.configuration import StyleTTS2ArchitectureConfig, load_styletts2_config
from voicehub.models.styletts2.native.frontend import (
    NativeStyleTTS2Frontend,
    StyleTTS2MelSpectrogram,
    load_style_reference,
)
from voicehub.models.styletts2.native.modeling import DEPLOYABLE_STYLETTS2_COMPONENTS, build_styletts2_model
from voicehub.models.styletts2.native.sampling import StyleTTS2DiffusionSampler
from voicehub.models.styletts2.source.styletts2.Modules.diffusion.sampler import ADPM2Sampler, KarrasSchedule
from voicehub.optimization.protocols import OptimizationCompileTarget, OptimizationModuleRoot


class StyleTTS2Runtime:
    """Strict native inference over the released multispeaker graph."""

    _CRITICAL_CHECKPOINT_MODULES = frozenset(DEPLOYABLE_STYLETTS2_COMPONENTS)

    def __init__(
        self,
        *,
        checkpoint_path: str,
        config_path: str | None = None,
        assets_directory: str | None = None,
        device: str | torch.device = "cpu",
        language: str = "en-us",
        trust_pickle_checkpoint: bool = False,
        dtype: torch.dtype | None = None,
    ) -> None:
        if assets_directory is not None:
            asset_root = Path(assets_directory).expanduser()
            if not asset_root.is_dir():
                raise NotADirectoryError(
                    "StyleTTS 2 `assets_directory` must be a directory when "
                    "provided.")
        if language != "en-us":
            raise ValueError(
                "The pinned StyleTTS 2 phoneme inventory was released for "
                "English (`en-us`). VoiceHub accepts explicit phonemes only.")
        self.device = torch.device(device)
        self.config = load_styletts2_config(config_path)
        self.sample_rate = self.config.sample_rate
        self.frontend = NativeStyleTTS2Frontend()
        self.to_mel = StyleTTS2MelSpectrogram(
            sample_rate=self.config.sample_rate,
            n_fft=self.config.n_fft,
            win_length=self.config.win_length,
            hop_length=self.config.hop_length,
            n_mels=self.config.n_mels,
        )
        self.model = build_styletts2_model(self.config)

        checkpoint = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"StyleTTS 2 checkpoint was not found: {checkpoint}.")
        if checkpoint.suffix.lower() == ".safetensors":
            load_styletts2_checkpoint(
                self.model,
                checkpoint,
                device=self.device,
                dtype=dtype,
            )
            if dtype is None:
                self.model.to(device=self.device)
            else:
                self.model.to(device=self.device, dtype=dtype)
        else:
            state = read_legacy_styletts2_checkpoint(
                self.model,
                checkpoint,
                trust_pickle_checkpoint=trust_pickle_checkpoint,
            )
            incompatible = self.model.load_state_dict(state, strict=True)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise RuntimeError("StyleTTS 2 legacy checkpoint failed strict assignment.")
            if dtype is None:
                self.model.to(device=self.device)
            else:
                self.model.to(device=self.device, dtype=dtype)
        self.to_mel.to(device=self.device)
        self.eval()
        self.sampler = StyleTTS2DiffusionSampler(
            self.model.diffusion.diffusion,
            sampler=ADPM2Sampler(),
            sigma_schedule=KarrasSchedule(
                sigma_min=0.0001,
                sigma_max=3.0,
                rho=9.0,
            ),
            clamp=False,
        )

    def optimization_module_roots(self, ) -> tuple[OptimizationModuleRoot, ...]:
        """Expose the model and the separate sampler policy module."""
        return (
            OptimizationModuleRoot("model", self.model),
            OptimizationModuleRoot("sampler", self.sampler),
        )

    def optimization_compile_targets(
        self,
        mode: str,
    ) -> tuple[OptimizationCompileTarget, ...]:
        """Expose the diffusion and waveform graphs reached by synthesis."""
        if mode == "training":
            # Fine-tuning executes through StyleTTS2TrainingModel, whose
            # phase-aware forward is discovered by the training adapter.
            return ()
        if mode != "inference":
            raise ValueError(f"Unsupported optimization mode {mode!r}.")
        return (
            OptimizationCompileTarget(
                "sampler.forward",
                self.sampler,
                "forward",
            ),
            OptimizationCompileTarget(
                "decoder.forward",
                self.model.decoder,
                "forward",
            ),
        )

    def train(self, mode: bool = True) -> StyleTTS2Runtime:
        train = getattr(self.model, "train", None)
        if callable(train):
            train(mode)
        else:
            for module in self.model.values():
                nested = getattr(module, "train", None)
                if callable(nested):
                    nested(mode)
        return self

    def eval(self) -> StyleTTS2Runtime:
        evaluate = getattr(self.model, "eval", None)
        if callable(evaluate):
            evaluate()
        else:
            for module in self.model.values():
                nested = getattr(module, "eval", None)
                if callable(nested):
                    nested()
        return self

    @staticmethod
    def _resolve_asset(
        value: Any,
        search_roots: tuple[Path, ...],
    ) -> Path:
        """Backward-compatible deterministic path resolution helper."""
        if not isinstance(value, (str, Path)) or not str(value).strip():
            raise TypeError("StyleTTS 2 asset paths must be non-empty strings.")
        path = Path(value).expanduser()
        if path.is_absolute():
            return path.resolve()
        candidates = tuple(root / path for root in search_roots)
        return next(
            (candidate.resolve() for candidate in candidates if candidate.exists()),
            candidates[0].resolve(),
        )

    @staticmethod
    def _normalize_state_dict(state: Mapping[str, Any], ) -> OrderedDict[str, Any]:
        return OrderedDict((
            name[7:] if name.startswith("module.") else name,
            value,
        ) for name, value in state.items())

    @classmethod
    def _load_module_checkpoint(
        cls,
        module_name: str,
        module: Any,
        state: Mapping[str, Any],
    ) -> set[str]:
        """Compatibility helper retained for legacy callers and tests."""
        module_parameters = {name for name, _ in module.named_parameters()}
        if not module_parameters:
            module_parameters = set(module.state_dict())
        candidate = state
        matching = module_parameters.intersection(candidate)
        try:
            module.load_state_dict(candidate)
        except RuntimeError:
            candidate = cls._normalize_state_dict(state)
            matching = module_parameters.intersection(candidate)
            if not matching:
                raise RuntimeError(
                    f"StyleTTS 2 checkpoint component {module_name!r} has "
                    "no parameter keys matching the runtime module.")
            incompatible = module.load_state_dict(candidate, strict=False)
            matching.difference_update(getattr(incompatible, "unexpected_keys", ()))
        if not matching:
            raise RuntimeError(
                f"StyleTTS 2 checkpoint component {module_name!r} loaded no "
                "matching parameter keys.")
        return matching

    @staticmethod
    def _length_to_mask(lengths: Tensor) -> Tensor:
        positions = torch.arange(
            int(lengths.max()),
            device=lengths.device,
        ).unsqueeze(0)
        return positions + 1 > lengths.unsqueeze(1)

    def _tokens(
        self,
        text: str,
        *,
        input_ids: Tensor | Sequence[int] | Sequence[Sequence[int]] | None,
        text_is_phonemes: bool,
    ) -> Tensor:
        if input_ids is not None:
            if text_is_phonemes:
                raise ValueError("Pass either `input_ids` or explicit phoneme text, not both.")
            return self.frontend.normalize_input_ids(
                input_ids,
                device=self.device,
            )
        return self.frontend.encode_phonemes(
            text,
            explicit=text_is_phonemes,
            device=self.device,
        )

    def _reference_style(self, audio: Any) -> Tensor:
        waveform = load_style_reference(
            audio,
            sample_rate=self.sample_rate,
        ).to(device=self.device)
        reference_parameter = next(self.model.style_encoder.parameters())
        waveform = waveform.to(dtype=reference_parameter.dtype)
        mel = self.to_mel(waveform)
        mel = (torch.log(1e-5 + mel.unsqueeze(0)) + 4.0) / 4.0
        mel = mel.unsqueeze(1)
        with torch.no_grad():
            reference = self.model.style_encoder(mel)
            prosody = self.model.predictor_encoder(mel)
        return torch.cat([reference, prosody], dim=1)

    def generate(
        self,
        text: str,
        *,
        speaker_audio_path: Any | None = None,
        alpha: float = 0.3,
        beta: float = 0.7,
        diffusion_steps: int = 5,
        embedding_scale: float = 1.0,
        seed: int | None = None,
        input_ids: Tensor | Sequence[int] | Sequence[Sequence[int]] | None = None,
        text_is_phonemes: bool = False,
    ) -> Tensor:
        self._validate_request(
            text=text,
            speaker_audio_path=speaker_audio_path,
            alpha=alpha,
            beta=beta,
            diffusion_steps=diffusion_steps,
            embedding_scale=embedding_scale,
            seed=seed,
            input_ids=input_ids,
            text_is_phonemes=text_is_phonemes,
        )
        tokens = self._tokens(
            text,
            input_ids=input_ids,
            text_is_phonemes=text_is_phonemes,
        )
        reference_style = (
            self._reference_style(speaker_audio_path) if speaker_audio_path is not None else None)

        with torch.inference_mode():
            input_lengths = torch.tensor(
                [tokens.shape[-1]],
                dtype=torch.long,
                device=self.device,
            )
            text_mask = self._length_to_mask(input_lengths)
            text_encoding = self.model.text_encoder(
                tokens,
                input_lengths,
                text_mask,
            )
            bert_duration = self.model.bert(
                tokens,
                attention_mask=(~text_mask).int(),
            )
            duration_encoding = self.model.bert_encoder(bert_duration).transpose(-1, -2)
            noise = torch.randn(
                (1, 1, self.config.style_dim * 2),
                device=self.device,
                dtype=text_encoding.dtype,
            )

            if not self.config.multispeaker:
                style_prediction = self.sampler(
                    noise,
                    embedding=bert_duration[0].unsqueeze(0),
                    num_steps=diffusion_steps,
                    embedding_scale=embedding_scale,
                ).squeeze(0)
            else:
                style_prediction = self.sampler(
                    noise=noise,
                    embedding=bert_duration,
                    embedding_scale=embedding_scale,
                    features=reference_style,
                    num_steps=diffusion_steps,
                ).squeeze(1)
            if not bool(torch.isfinite(style_prediction).all()):
                raise RuntimeError(
                    "StyleTTS 2 diffusion produced NaN or infinite style "
                    "values; verify checkpoint/config compatibility.")

            style = style_prediction[:, self.config.style_dim:]
            reference = style_prediction[:, :self.config.style_dim]
            if self.config.multispeaker:
                reference = (alpha * reference + (1.0 - alpha) * reference_style[:, :self.config.style_dim])
                style = (beta * style + (1.0 - beta) * reference_style[:, self.config.style_dim:])

            predictor_encoding = self.model.predictor.text_encoder(
                duration_encoding,
                style,
                input_lengths,
                text_mask,
            )
            duration_hidden, _ = self.model.predictor.lstm(predictor_encoding)
            duration = self.model.predictor.duration_proj(duration_hidden)
            duration = torch.sigmoid(duration).sum(dim=-1)
            if not bool(torch.isfinite(duration).all()):
                raise RuntimeError("StyleTTS 2 duration prediction contains NaN or infinity.")
            predicted_duration = torch.round(duration).clamp(min=1).reshape(-1)
            if reference_style is None:
                predicted_duration[-1] += 5

            alignment = torch.zeros(
                tokens.shape[-1],
                int(predicted_duration.sum().item()),
                device=self.device,
                dtype=text_encoding.dtype,
            )
            frame = 0
            for token_index, token_duration in enumerate(predicted_duration):
                next_frame = frame + int(token_duration.item())
                alignment[token_index, frame:next_frame] = 1
                frame = next_frame
            alignment = alignment.unsqueeze(0)

            prosody_encoding = (predictor_encoding.transpose(-1, -2) @ alignment)
            text_decoder_encoding = text_encoding @ alignment
            if self.config.decoder.type == "hifigan":
                prosody_encoding = self._shift(prosody_encoding)
                text_decoder_encoding = self._shift(text_decoder_encoding)
            f0, noise_prediction = self.model.predictor.F0Ntrain(
                prosody_encoding,
                style,
            )
            output = self.model.decoder(
                text_decoder_encoding,
                f0,
                noise_prediction,
                reference.squeeze().unsqueeze(0),
            )
        audio = output.squeeze().detach().to(device="cpu", dtype=torch.float32)
        if not bool(torch.isfinite(audio).all()):
            raise RuntimeError("StyleTTS 2 decoder returned NaN or infinite samples.")
        if self.config.decoder.type != "hifigan":
            if audio.numel() == 0:
                raise RuntimeError("StyleTTS 2 returned an empty waveform.")
            return audio.contiguous()
        if audio.numel() <= 50:
            raise RuntimeError("StyleTTS 2 returned fewer samples than the HiFi-GAN trim size.")
        return audio[:-50].contiguous()

    def _validate_request(
        self,
        *,
        text: str,
        speaker_audio_path: Any | None,
        alpha: float,
        beta: float,
        diffusion_steps: int,
        embedding_scale: float,
        seed: int | None,
        input_ids: Any = None,
        text_is_phonemes: bool = False,
    ) -> None:
        if input_ids is None and (not isinstance(text, str) or not text.strip()):
            raise ValueError("`text` must be a non-empty phoneme string.")
        if not isinstance(text_is_phonemes, bool):
            raise TypeError("`text_is_phonemes` must be a boolean.")
        if input_ids is None and text_is_phonemes is not True:
            raise ValueError(
                "StyleTTS 2 requires explicit phonemes; set "
                "`text_is_phonemes=True` or pass `input_ids`.")
        if self.config.multispeaker and speaker_audio_path is None:
            raise ValueError(
                "The released multispeaker StyleTTS 2 checkpoint requires "
                "`speaker_audio_path` (or another explicit VoiceHub audio "
                "input) for style conditioning.")
        if not self.config.multispeaker and speaker_audio_path is not None:
            raise ValueError(
                "The released single-speaker StyleTTS 2 iSTFTNet checkpoint "
                "does not consume reference audio.")
        if isinstance(speaker_audio_path, (str, Path)):
            if not Path(speaker_audio_path).expanduser().is_file():
                raise FileNotFoundError(
                    "StyleTTS 2 reference audio was not found: "
                    f"{speaker_audio_path}.")
        for name, value in (("alpha", alpha), ("beta", beta)):
            if (not isinstance(value, (int, float)) or isinstance(value, bool)):
                raise TypeError(f"`{name}` must be numeric.")
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"`{name}` must be in the interval [0, 1].")
        if (not isinstance(diffusion_steps, int) or isinstance(diffusion_steps, bool) or diffusion_steps < 2):
            raise ValueError("`diffusion_steps` must be an integer >= 2.")
        if (not isinstance(embedding_scale, (int, float)) or isinstance(embedding_scale, bool) or
                not math.isfinite(embedding_scale) or embedding_scale <= 0):
            raise ValueError("`embedding_scale` must be a finite positive number.")
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
            raise TypeError("`seed` must be an integer or None.")

    @staticmethod
    def _shift(encoding: Tensor) -> Tensor:
        shifted = torch.zeros_like(encoding)
        shifted[:, :, 0] = encoding[:, :, 0]
        shifted[:, :, 1:] = encoding[:, :, :-1]
        return shifted


__all__ = ["StyleTTS2Runtime"]
