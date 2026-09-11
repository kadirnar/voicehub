"""VoiceHub-native VITS and MMS-TTS inference and generator training."""

from __future__ import annotations

import math
import secrets
from collections.abc import Mapping, Sequence
from dataclasses import replace
from numbers import Integral, Real
from pathlib import Path
from typing import Any

from voicehub.audio import load_audio
from voicehub.hub import read_json_file, write_json_file
from voicehub.models._shared import finish_audio_output, validate_seed
from voicehub.models.tts import PreTrainedTTSModel
from voicehub.models.vits.configuration import VitsConfig
from voicehub.outputs import TTSOutput


def _output_value(output: Any, name: str) -> Any:
    if isinstance(output, Mapping):
        return output.get(name)
    return getattr(output, name, None)


def _build_vits_training_model(
    torch: Any,
    model: Any,
    *,
    waveform_loss_weight: float = 1.0,
    spectral_loss_weight: float = 0.1,
    duration_loss_weight: float = 1.0,
    kl_loss_weight: float = 1.0,
):
    """Build the explicitly partial native VITS generator objective.

    This facade uses the posterior encoder, monotonic alignment search,
    duration objective, flow, and decoder from the native training
    graph. It intentionally requires a checkpoint-compatible linear
    spectrogram rather than guessing an acoustic frontend. It is a
    generator warm-start path, not the full discriminator/mel/feature-
    matching VITS recipe.
    """

    class NativeVitsGeneratorTrainingModel(torch.nn.Module):

        def __init__(self, native_model: Any) -> None:
            super().__init__()
            self.native_model = native_model
            self.waveform_loss_weight = float(waveform_loss_weight)
            self.spectral_loss_weight = float(spectral_loss_weight)
            self.duration_loss_weight = float(duration_loss_weight)
            self.kl_loss_weight = float(kl_loss_weight)

        @staticmethod
        def _waveform_batch(value: Any, reference: Any) -> Any:
            target = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
            target = target.to(
                device=reference.device,
                dtype=reference.dtype,
            )
            if target.ndim == 1:
                target = target.unsqueeze(0)
            elif target.ndim == 3 and target.shape[1] == 1:
                target = target[:, 0]
            if target.ndim != 2:
                raise ValueError(
                    "VITS waveform targets must have shape [batch, samples] "
                    f"or [batch, 1, samples]; received {tuple(target.shape)}.")
            if target.shape[0] != reference.shape[0]:
                raise ValueError(
                    "VITS waveform target batch size does not match the text "
                    "and spectrogram batch.")
            if target.shape[1] < 1:
                raise ValueError("VITS waveform targets cannot be empty.")
            if not torch.isfinite(target).all():
                raise ValueError("VITS waveform targets cannot contain NaN or infinite values.")
            return target

        @staticmethod
        def _lengths(
            value: Any,
            *,
            batch_size: int,
            maximum: int,
            device: Any,
            name: str,
        ) -> Any:
            if value is None:
                return torch.full(
                    (batch_size, ),
                    maximum,
                    dtype=torch.long,
                    device=device,
                )
            lengths = (value if isinstance(value, torch.Tensor) else torch.as_tensor(value))
            lengths = lengths.to(device=device)
            if lengths.ndim == 0:
                lengths = lengths.expand(batch_size)
            if tuple(lengths.shape) != (batch_size, ):
                raise ValueError(f"`{name}` must have shape [batch].")
            if lengths.dtype == torch.bool or lengths.is_floating_point():
                raise TypeError(f"`{name}` must use an integer dtype.")
            lengths = lengths.long()
            if ((lengths < 1) | (lengths > maximum)).any():
                raise ValueError(f"`{name}` values must be in the interval [1, {maximum}].")
            return lengths

        @staticmethod
        def _spectral_loss(
            prediction: Any,
            target: Any,
            lengths: Any,
        ) -> Any:
            losses = []
            for batch_index in range(prediction.shape[0]):
                length = int(lengths[batch_index].item())
                predicted_item = prediction[batch_index, :length].float()
                target_item = target[batch_index, :length].float()
                for n_fft in (256, 512, 1024):
                    if length < n_fft:
                        continue
                    window = torch.hann_window(
                        n_fft,
                        device=prediction.device,
                        dtype=torch.float32,
                    )
                    predicted_stft = torch.stft(
                        predicted_item,
                        n_fft=n_fft,
                        hop_length=n_fft // 4,
                        win_length=n_fft,
                        window=window,
                        return_complex=True,
                    )
                    target_stft = torch.stft(
                        target_item,
                        n_fft=n_fft,
                        hop_length=n_fft // 4,
                        win_length=n_fft,
                        window=window,
                        return_complex=True,
                    )
                    losses.append(
                        torch.nn.functional.l1_loss(
                            torch.log1p(predicted_stft.abs()),
                            torch.log1p(target_stft.abs()),
                        ))
            if not losses:
                return prediction.new_zeros((), dtype=torch.float32)
            return torch.stack(losses).mean()

        def forward(
            self,
            input_ids: Any,
            *,
            spectrogram: Any = None,
            audio_values: Any = None,
            labels: Any = None,
            attention_mask: Any = None,
            spectrogram_attention_mask: Any = None,
            durations: Any = None,
            speaker_id: Any = None,
            audio_lengths: Any = None,
            generator: Any = None,
            **kwargs: Any,
        ) -> dict[str, Any]:
            del kwargs
            if spectrogram is None:
                raise ValueError(
                    "Native VITS generator training requires `spectrogram` "
                    "with shape [batch, spectrogram_bins, frames]. Generate it "
                    "with the checkpoint's explicit FFT/hop/window recipe.")
            target = audio_values if audio_values is not None else labels
            if target is None:
                raise ValueError(
                    "Native VITS generator training requires `audio_values` "
                    "waveform targets aligned with the spectrogram.")
            output = self.native_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                speaker_id=speaker_id,
                spectrogram=spectrogram,
                spectrogram_attention_mask=spectrogram_attention_mask,
                durations=durations,
                generator=generator,
            )
            waveform = _output_value(output, "waveform")
            sequence_lengths = _output_value(output, "sequence_lengths")
            duration_loss = _output_value(output, "duration_loss")
            if waveform is None or sequence_lengths is None or duration_loss is None:
                raise RuntimeError(
                    "The native VITS training graph returned an incomplete "
                    "generator output.")
            target = self._waveform_batch(target, waveform)
            common_length = min(waveform.shape[1], target.shape[1])
            if common_length < 1:
                raise ValueError("VITS waveform targets cannot be empty.")
            prediction = waveform[:, :common_length]
            target = target[:, :common_length]
            generated_lengths = self._lengths(
                sequence_lengths,
                batch_size=prediction.shape[0],
                maximum=waveform.shape[1],
                device=prediction.device,
                name="sequence_lengths",
            ).clamp_max(common_length)
            target_lengths = self._lengths(
                audio_lengths,
                batch_size=target.shape[0],
                maximum=target.shape[1],
                device=target.device,
                name="audio_lengths",
            )
            lengths = torch.minimum(generated_lengths, target_lengths)
            sample_mask = (
                torch.arange(common_length, device=prediction.device).unsqueeze(0).lt(lengths.unsqueeze(1)))
            element_loss = torch.nn.functional.smooth_l1_loss(
                prediction,
                target,
                reduction="none",
            )
            waveform_loss = (element_loss *
                             sample_mask.to(dtype=element_loss.dtype)).sum() / sample_mask.sum().clamp_min(1)
            spectral_loss = self._spectral_loss(
                prediction,
                target,
                lengths,
            )

            from voicehub.models.vits.native.losses import vits_kl_loss

            kl_loss = vits_kl_loss(
                _output_value(output, "prior_latents"),
                _output_value(output, "posterior_log_variances"),
                _output_value(output, "expanded_prior_means"),
                _output_value(output, "expanded_prior_log_variances"),
                _output_value(output, "spectrogram_mask"),
            )
            loss = (
                self.waveform_loss_weight * waveform_loss + self.spectral_loss_weight * spectral_loss +
                self.duration_loss_weight * duration_loss.float() + self.kl_loss_weight * kl_loss)
            return {
                "loss": loss,
                "waveform": waveform,
                "audio_values": waveform,
                "losses": {
                    "waveform_loss": waveform_loss,
                    "spectral_loss": spectral_loss,
                    "duration_loss": duration_loss,
                    "kl_loss": kl_loss,
                },
                "native_output": output,
            }

    return NativeVitsGeneratorTrainingModel(model)


def _build_vits_adversarial_training_model(
    model: Any,
    *,
    acoustic_config: Mapping[str, Any],
    mel_loss_weight: float,
    kl_loss_weight: float,
):
    """Build VoiceHub's source-compatible two-optimizer VITS recipe."""
    from voicehub.models.vits.native.training import VitsAdversarialTrainingModel

    return VitsAdversarialTrainingModel(
        model,
        acoustic_config,
        mel_weight=mel_loss_weight,
        kl_weight=kl_loss_weight,
    )


class VitsForTextToSpeech(PreTrainedTTSModel):
    """Load VITS and Meta MMS-TTS checkpoints without Transformers."""

    config_class = VitsConfig
    default_model_name_or_path = "facebook/mms-tts-eng"
    architecture_family = "vits"
    passthrough_generation_options = frozenset()

    def __init__(
        self,
        config: VitsConfig | str | Path | None = None,
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
            raise ValueError("`token` must be a non-empty string, boolean, or None.")
        self._hub_token = token.strip() if isinstance(token, str) else token
        # Frontend implementations are live runtime objects and must never
        # enter the serializable model configuration.
        self._text_normalizer = config_overrides.pop("text_normalizer", None)
        self._text_romanizer = config_overrides.pop("text_romanizer", None)
        self._text_phonemizer = config_overrides.pop("text_phonemizer", None)
        self.artifacts: Any | None = None
        self.native_config: Any | None = None
        self.tokenizer: Any | None = None
        self.training_model: Any | None = None
        self._torch: Any | None = None
        config = self._coerce_config(
            config,
            model_path=model_path,
            **config_overrides,
        )
        config.validate()
        super().__init__(
            config,
            device=device,
            lazy_load=lazy_load,
        )

    @property
    def training_processor(self) -> Any | None:
        """Return the declarative tokenizer paired with the generator."""
        return self.tokenizer

    @property
    def transformers_processor(self) -> Any | None:
        """Compatibility alias for code migrating from the former provider."""
        return self.tokenizer

    @transformers_processor.setter
    def transformers_processor(self, value: Any | None) -> None:
        self.tokenizer = value

    def _hub_kwargs(self) -> dict[str, Any]:
        """Return runtime-only Hub transport options."""
        return {
            key: value
            for key, value in {
                "revision": self.config.revision,
                "cache_dir": self.config.cache_dir,
                "local_files_only": self.config.local_files_only,
                "token": self._hub_token,
            }.items() if value is not None
        }

    def _model_dtype(self) -> Any:
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
            raise ValueError(
                "Native VITS does not support float16 execution on CPU; "
                "use float32 or bfloat16.")
        return dtype

    @staticmethod
    def _validate_architecture(values: Mapping[str, Any]) -> None:
        model_type = str(values.get("model_type", "")).strip().lower()
        if model_type != "vits":
            raise ValueError(
                "Native VITS requires a VITS checkpoint; received model type "
                f"{model_type or '<missing>'!r}.")
        architectures = values.get('architectures', ())
        if isinstance(architectures, str):
            architectures = (architectures, )
        if not isinstance(architectures, Sequence):
            raise TypeError("VITS `architectures` must be a sequence.")
        names = tuple(str(name) for name in architectures)
        if names and not any(name in {"VitsModel", "VitsForTextToSpeech"} for name in names):
            raise ValueError(
                "Native VITS requires a VitsModel checkpoint architecture; "
                f"received: {', '.join(names)}.")

    def _load_pretrained_model(self) -> None:
        from voicehub.checkpointing import SafeTensorReader, ShardedSafeTensorReader
        from voicehub.models.vits.artifacts import resolve_vits_artifacts
        from voicehub.models.vits.native.checkpoint import HuggingFaceVitsCheckpointAdapter, NativeVitsCheckpointAdapter
        from voicehub.models.vits.native.configuration import VitsConfig as NativeVitsConfig
        from voicehub.models.vits.native.frontend import VitsTokenizer
        from voicehub.models.vits.native.modeling import VitsModel

        source = self.config.name_or_path or self.default_model_name_or_path
        artifacts = resolve_vits_artifacts(
            source,
            checkpoint_filename=self.config.checkpoint_filename,
            vocabulary_filename=self.config.vocabulary_filename,
            tokenizer_config_filename=self.config.tokenizer_config_filename,
            cache_dir=self.config.cache_dir,
            revision=self.config.revision,
            token=self._hub_token,
            local_files_only=self.config.local_files_only,
        )
        architecture_values = read_json_file(artifacts.config)
        self._validate_architecture(architecture_values)
        native_config = NativeVitsConfig.from_dict(architecture_values)
        tokenizer = VitsTokenizer.from_files(
            artifacts.vocabulary,
            tokenizer_config_file=artifacts.tokenizer_config,
            normalizer=self._text_normalizer,
            romanizer=self._text_romanizer,
            phonemizer=self._text_phonemizer,
        )
        if tokenizer.vocab_size != native_config.vocab_size:
            raise ValueError(
                "VITS tokenizer/model vocabulary mismatch: tokenizer has "
                f"{tokenizer.vocab_size} IDs, model expects "
                f"{native_config.vocab_size}.")
        if (native_config.pad_token_id is not None and tokenizer.pad_token_id != native_config.pad_token_id):
            raise ValueError(
                "VITS tokenizer/model pad-token mismatch: tokenizer uses "
                f"{tokenizer.pad_token_id}, model expects "
                f"{native_config.pad_token_id}.")

        model = VitsModel(native_config)
        reader_type = (ShardedSafeTensorReader if artifacts.is_sharded else SafeTensorReader)
        checkpoint_format = architecture_values.get("voicehub_checkpoint_format")
        adapter = (
            NativeVitsCheckpointAdapter()
            if checkpoint_format == "native-vits-v1" else HuggingFaceVitsCheckpointAdapter())
        with reader_type(artifacts.checkpoint) as reader:
            adapter.load_streaming(
                model,
                reader,
                architecture_values,
                strict=True,
            )
        model.to(device=self.device, dtype=self._model_dtype())

        self._torch = __import__("torch")
        self.artifacts = artifacts
        self.native_config = native_config
        self.tokenizer = tokenizer
        self.model = model
        self.config.sample_rate = native_config.sampling_rate

    def _prepare_for_training(self) -> None:
        if not self.config.enable_native_generator_training:
            raise ValueError(
                "Native VITS training is opt-in. Set "
                "`enable_native_adversarial_training=True` with an explicit "
                "`training_acoustic_config` for the full two-optimizer GAN "
                "recipe, or set `enable_native_generator_training=True` for "
                "the preprocessed compatibility warm start.")
        self.model.train()
        if self.training_model is None:
            if self.config.enable_native_adversarial_training:
                if self.config.training_acoustic_config is None:
                    raise ValueError(
                        "Full VITS adversarial fine-tuning requires an "
                        "explicit `training_acoustic_config`. MMS-TTS "
                        "checkpoint metadata does not publish its FFT, hop, "
                        "window, mel, or segment training settings.")
                self.training_model = _build_vits_adversarial_training_model(
                    self.model,
                    acoustic_config=self.config.training_acoustic_config,
                    mel_loss_weight=self.config.training_mel_loss_weight,
                    kl_loss_weight=self.config.training_kl_loss_weight,
                )
                self.training_model.to(
                    device=self.device,
                    dtype=self._model_dtype(),
                )
            else:
                self.training_model = _build_vits_training_model(
                    self._torch,
                    self.model,
                    waveform_loss_weight=self.config.training_waveform_loss_weight,
                    spectral_loss_weight=self.config.training_spectral_loss_weight,
                    duration_loss_weight=self.config.training_duration_loss_weight,
                    kl_loss_weight=self.config.training_kl_loss_weight,
                )
        self.training_model.train()

    def _prepare_for_inference(self) -> None:
        self.model.eval()
        self.model.cache_weight_norm_for_inference()
        if self.training_model is not None:
            self.training_model.eval()

    def _validate_training_runtime(self) -> None:
        if not self.config.enable_native_generator_training:
            raise ValueError(
                "Native VITS training is disabled. Enable the complete "
                "adversarial recipe with explicit acoustic settings, or "
                "enable the legacy generator warm start with precomputed "
                "linear spectrograms.")
        if (self.config.enable_native_adversarial_training and self.config.training_acoustic_config is None):
            raise ValueError(
                "Full VITS adversarial fine-tuning requires an explicit "
                "`training_acoustic_config`; it cannot be inferred from "
                "MMS-TTS checkpoint metadata.")

    def _request_tokenizer(self, normalize: bool | None) -> Any:
        if self.tokenizer is None:
            raise RuntimeError("VITS tokenizer is not loaded.")
        if normalize is None or normalize == self.tokenizer.config.normalize:
            return self.tokenizer
        if not isinstance(normalize, bool):
            raise TypeError("`normalize` must be a boolean or None.")
        from voicehub.models.vits.native.frontend import VitsTokenizer

        return VitsTokenizer(
            self.tokenizer.vocabulary,
            config=replace(self.tokenizer.config, normalize=normalize),
            normalizer=self._text_normalizer,
            romanizer=self._text_romanizer,
            phonemizer=self._text_phonemizer,
        )

    def _tokenize(
        self,
        texts: str | Sequence[str],
        *,
        normalize: bool | None,
    ) -> dict[str, Any]:
        if isinstance(texts, str):
            values = (texts, )
        elif isinstance(texts, Sequence) and not isinstance(texts, (bytes, bytearray)):
            values = tuple(texts)
        else:
            raise TypeError("VITS text must be a string or sequence of strings.")
        if not values or any(not isinstance(text, str) or not text.strip() for text in values):
            raise ValueError("VITS text items must be non-empty strings.")
        tokenizer = self._request_tokenizer(normalize)
        encoded = tokenizer.encode_batch(values, padding=True)
        if not encoded.input_ids or not encoded.input_ids[0]:
            raise ValueError("VITS frontend processing produced no checkpoint tokens.")
        return {
            "input_ids":
            self._torch.tensor(
                encoded.input_ids,
                dtype=self._torch.long,
                device=self.device,
            ),
            "attention_mask":
            self._torch.tensor(
                encoded.attention_mask,
                dtype=self._torch.bool,
                device=self.device,
            ),
        }

    def _validate_generation_inputs(self, model_inputs: dict[str, Any]) -> None:
        speaker_id = model_inputs.get("speaker_id")
        invalid_speaker = (
            isinstance(speaker_id, bool) or not isinstance(speaker_id, Integral) or speaker_id < 0)
        if speaker_id is not None and invalid_speaker:
            raise ValueError("`speaker_id` must be a non-negative integer or None.")
        normalize = model_inputs.get("normalize")
        if normalize is not None and not isinstance(normalize, bool):
            raise TypeError("`normalize` must be a boolean or None.")

    @staticmethod
    def _positive_real(
        value: Any,
        *,
        name: str,
        allow_zero: bool,
    ) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"`{name}` must be a real number.")
        normalized = float(value)
        valid = normalized >= 0.0 if allow_zero else normalized > 0.0
        if not math.isfinite(normalized) or not valid:
            qualifier = "non-negative" if allow_zero else "greater than zero"
            raise ValueError(f"`{name}` must be finite and {qualifier}.")
        return normalized

    def _generate(
        self,
        text: str,
        *,
        speaker_id: int | None = None,
        speaking_rate: float | None = None,
        speed: float | None = None,
        noise_scale: float | None = None,
        noise_scale_duration: float | None = None,
        normalize: bool | None = None,
        max_output_frames: int | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        output_file: str | Path | None = None,
        seed: int | None = None,
    ) -> TTSOutput:
        from voicehub.models.vits.native.modeling import VitsSamplingConfig

        if speaking_rate is not None and speed is not None:
            raise ValueError("Pass `speaking_rate` or `speed`, not both.")
        resolved_rate = (
            speaking_rate
            if speaking_rate is not None else speed if speed is not None else self.config.speaking_rate)
        resolved_rate = self._positive_real(
            resolved_rate,
            name="speaking_rate",
            allow_zero=False,
        )
        resolved_noise = (
            self.native_config.noise_scale if noise_scale is None and self.config.noise_scale is None else
            self.config.noise_scale if noise_scale is None else noise_scale)
        resolved_duration_noise = (
            self.native_config.noise_scale_duration
            if noise_scale_duration is None and self.config.noise_scale_duration is None else
            self.config.noise_scale_duration if noise_scale_duration is None else noise_scale_duration)
        resolved_noise = self._positive_real(
            resolved_noise,
            name="noise_scale",
            allow_zero=True,
        )
        resolved_duration_noise = self._positive_real(
            resolved_duration_noise,
            name="noise_scale_duration",
            allow_zero=True,
        )
        if max_output_frames is None:
            max_output_frames = self.config.max_output_frames
        if (isinstance(max_output_frames, bool) or not isinstance(max_output_frames, Integral) or
                max_output_frames <= 0):
            raise ValueError("`max_output_frames` must be a positive integer.")
        if not isinstance(output_attentions, bool):
            raise TypeError("`output_attentions` must be a boolean.")
        if not isinstance(output_hidden_states, bool):
            raise TypeError("`output_hidden_states` must be a boolean.")
        if self.native_config.num_speakers == 1 and speaker_id is not None:
            raise ValueError("`speaker_id` is invalid for this single-speaker VITS checkpoint.")
        if (speaker_id is not None and speaker_id >= self.native_config.num_speakers):
            raise ValueError(
                f"`speaker_id` must be smaller than "
                f"{self.native_config.num_speakers} for this checkpoint.")
        requested_seed = validate_seed(seed)
        effective_seed = (secrets.randbits(63) if requested_seed is None else requested_seed % (2**63))
        sampling = VitsSamplingConfig(
            speaking_rate=resolved_rate,
            noise_scale=resolved_noise,
            noise_scale_duration=resolved_duration_noise,
            seed=effective_seed,
            max_output_frames=int(max_output_frames),
        )
        inputs = self._tokenize(text, normalize=normalize)
        with self._torch.inference_mode():
            generated = self.model.synthesize(
                **inputs,
                speaker_id=speaker_id,
                sampling=sampling,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
        waveform = generated.waveform
        if waveform.ndim != 2 or waveform.shape[0] != 1:
            raise RuntimeError(
                "Native VITS returned an invalid waveform batch shape "
                f"{tuple(waveform.shape)}.")
        sequence_lengths = generated.sequence_lengths
        if sequence_lengths.numel() != 1:
            raise RuntimeError("Native VITS returned an invalid sequence-length batch.")
        output_length = int(sequence_lengths[0].item())
        if output_length < 1 or output_length > waveform.shape[1]:
            raise RuntimeError("Native VITS returned an invalid waveform sequence length.")
        audio = waveform[0, :output_length].detach().float().cpu().contiguous()
        return finish_audio_output(
            audio,
            self.sample_rate,
            output_file=output_file,
            metadata={
                "backend": "voicehub-native",
                "architecture": "vits",
                "checkpoint_family": "mms-tts/vits",
                "speaker_id": speaker_id,
                "speaking_rate": resolved_rate,
                "noise_scale": resolved_noise,
                "noise_scale_duration": resolved_duration_noise,
                "seed": effective_seed,
                "requested_seed": requested_seed,
                "max_output_frames": int(max_output_frames),
                "frontend_language": self.tokenizer.config.language,
            },
        )

    @staticmethod
    def _sampling_rate_value(value: Any) -> int | None:
        if value is None:
            return None
        if hasattr(value, "detach"):
            values = value.detach().reshape(-1)
            if values.numel() == 0:
                return None
            first = int(values[0].item())
            if values.numel() > 1 and not bool((values == first).all().item()):
                raise ValueError("Every VITS waveform in a batch must share one sampling rate.")
            return first
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            values = tuple(value)
            if not values:
                return None
            first = int(values[0])
            if any(int(item) != first for item in values[1:]):
                raise ValueError("Every VITS waveform in a batch must share one sampling rate.")
            return first
        return int(value)

    def _materialize_training_audio(
        self,
        audio: Any,
        *,
        sampling_rate: Any,
    ) -> Any:
        source_rate = self._sampling_rate_value(sampling_rate)
        if isinstance(audio, self._torch.Tensor):
            if source_rate is not None and source_rate != self.sample_rate:
                raise ValueError(
                    "Batched VITS waveform tensors must already be resampled "
                    f"to {self.sample_rate} Hz.")
            return audio
        if isinstance(audio, (str, Path, Mapping)):
            loaded = load_audio(
                audio,
                sampling_rate=source_rate,
                target_sampling_rate=self.sample_rate,
            )
            return self._torch.as_tensor(
                loaded.waveform,
                dtype=self._torch.float32,
            ).unsqueeze(0)
        return self._torch.as_tensor(audio, dtype=self._torch.float32)

    def prepare_training_inputs(
        self,
        inputs: dict[str, Any],
        *,
        phase: str,
    ) -> dict[str, Any]:
        """Prepare a legacy warm-start or full adversarial VITS batch."""
        if phase not in {
                "discriminator",
                "generator",
                "waveform_reconstruction",
        }:
            raise ValueError(f"Unknown VITS training phase {phase!r}.")
        prepared = dict(inputs)
        if "input_ids" not in prepared:
            text = prepared.pop("text", None)
            if text is None:
                raise ValueError("VITS generator training requires `input_ids` or raw `text`.")
            prepared.update(self._tokenize(text, normalize=None))
        else:
            input_ids = prepared["input_ids"]
            if not isinstance(input_ids, self._torch.Tensor):
                input_ids = self._torch.as_tensor(input_ids)
            if input_ids.ndim == 1:
                input_ids = input_ids.unsqueeze(0)
            if input_ids.ndim != 2:
                raise ValueError("`input_ids` must have shape [batch, text].")
            prepared["input_ids"] = input_ids.long()
            if "attention_mask" not in prepared:
                prepared["attention_mask"] = self._torch.ones_like(
                    input_ids,
                    dtype=self._torch.bool,
                )

        spectrogram = prepared.get("spectrogram")
        if (spectrogram is None and not self.config.enable_native_adversarial_training):
            raise ValueError(
                "Native VITS generator training requires precomputed "
                "`spectrogram` inputs. Raw waveform alone is insufficient "
                "because FFT, hop, window, and mel conventions are "
                "checkpoint-specific.")
        if spectrogram is not None:
            if not isinstance(spectrogram, self._torch.Tensor):
                spectrogram = self._torch.as_tensor(spectrogram)
            if spectrogram.ndim == 2:
                spectrogram = spectrogram.unsqueeze(0)
            if spectrogram.ndim != 3:
                raise ValueError("`spectrogram` must have shape "
                                 "[batch, spectrogram_bins, frames].")
            prepared["spectrogram"] = spectrogram.float()
            if "spectrogram_attention_mask" not in prepared:
                lengths = prepared.pop("spectrogram_lengths", None)
                if lengths is None:
                    prepared["spectrogram_attention_mask"] = self._torch.ones(
                        (spectrogram.shape[0], spectrogram.shape[2]),
                        dtype=self._torch.bool,
                        device=spectrogram.device,
                    )
                else:
                    lengths = self._torch.as_tensor(
                        lengths,
                        device=spectrogram.device,
                    ).reshape(-1)
                    if lengths.numel() != spectrogram.shape[0]:
                        raise ValueError("`spectrogram_lengths` must contain one value "
                                         "per batch item.")
                    prepared["spectrogram_attention_mask"] = (
                        self._torch.arange(
                            spectrogram.shape[2],
                            device=spectrogram.device,
                        ).unsqueeze(0).lt(lengths.unsqueeze(1)))

        audio = prepared.pop(
            "audio",
            prepared.get("audio_values", prepared.get("labels")),
        )
        if audio is None:
            raise ValueError(
                "Native VITS generator training requires `audio_values` or "
                "raw `audio` aligned with the supplied spectrogram.")
        prepared["audio_values"] = self._materialize_training_audio(
            audio,
            sampling_rate=prepared.pop(
                "sampling_rate",
                prepared.pop("sample_rate", None),
            ),
        )
        if prepared["audio_values"].ndim == 1:
            prepared["audio_values"] = prepared["audio_values"].unsqueeze(0)
        prepared.pop("labels", None)
        return prepared

    def _save_pretrained(self, save_directory: Path) -> None:
        from voicehub.checkpointing import save_safetensors

        if self.model is None or self.native_config is None or self.tokenizer is None:
            self.load()
        save_directory.mkdir(parents=True, exist_ok=True)
        save_safetensors(
            self.model.state_dict(),
            save_directory / "model.safetensors",
            metadata={"format": "voicehub-native-vits-v1"},
        )
        config_values = self.native_config.to_dict()
        config_values.update({
            'architectures': ["VitsModel"],
            "model_type": "vits",
            "voicehub_checkpoint_format": "native-vits-v1",
            "voicehub_provider": "vits",
        })
        write_json_file(save_directory / "config.json", config_values)
        write_json_file(
            save_directory / self.config.vocabulary_filename,
            dict(self.tokenizer.vocabulary),
        )
        write_json_file(
            save_directory / self.config.tokenizer_config_filename,
            self.tokenizer.config.to_dict(),
        )


MmsTTSForTextToSpeech = VitsForTextToSpeech
VitsTTS = VitsForTextToSpeech

__all__ = [
    "MmsTTSForTextToSpeech",
    "VitsConfig",
    "VitsForTextToSpeech",
    "VitsTTS",
    "_build_vits_adversarial_training_model",
    "_build_vits_training_model",
]
