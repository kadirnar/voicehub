"""VoiceHub-native NeuTTS inference with the safe NeuCodec conversion."""

from __future__ import annotations

import math
from numbers import Real
from pathlib import Path
from typing import Any

from voicehub.dependencies import import_optional
from voicehub.models._shared import finish_audio_output, resolve_torch_dtype, seeded_inference
from voicehub.models.neutts.configuration import NeuTTSConfig
from voicehub.models.tts import PreTrainedTTSModel
from voicehub.outputs import TTSOutput


class NeuTTSForTextToSpeech(PreTrainedTTSModel):
    """Safetensors NeuTTS synthesis with no provider runtime dependencies."""

    config_class = NeuTTSConfig
    default_model_name_or_path = "neuphonic/neutts-2e"

    def __init__(
        self,
        config: NeuTTSConfig | str | None = None,
        *,
        model_path: str | None = None,
        device: str = "auto",
        lazy_load: bool = True,
        token: str | bool | None = None,
        **config_overrides: Any,
    ) -> None:
        if token is not None and not isinstance(token, (str, bool)):
            raise TypeError("`token` must be a string, boolean, or None.")
        if isinstance(token, str) and not token.strip():
            raise ValueError("String `token` values must be non-empty.")
        config = self._coerce_config(
            config,
            model_path=model_path,
            **config_overrides,
        )
        config.validate()
        self._hub_token = token
        self.artifacts: Any | None = None
        self.codec_artifacts: Any | None = None
        self.native_config: Any | None = None
        self.tokenizer = None
        self.codec = None
        self._torch = None
        super().__init__(config, device=device, lazy_load=lazy_load)

    def _language_model_dtype(
        self,
        torch,
        architecture_values: dict[str, Any],
    ):
        if self.config.torch_dtype != "auto":
            return resolve_torch_dtype(
                torch,
                self.config.torch_dtype,
                self.device,
            )
        device_type = torch.device(self.device).type
        declared = str(architecture_values.get(
            "dtype",
            architecture_values.get("torch_dtype", "float32"),
        )).lower()
        if device_type == "cuda":
            if (declared in {"bfloat16", "bf16"} and torch.cuda.is_bf16_supported()):
                return torch.bfloat16
            return torch.float16
        if device_type == "mps":
            return torch.float16
        # CPU generation is most portable in float32, including checkpoints
        # published in BF16.
        return torch.float32

    def _load_pretrained_model(self) -> None:
        from voicehub.checkpointing import SafeTensorReader
        from voicehub.hub import read_json_file
        from voicehub.models.causal_lm.native.checkpoint import open_causal_lm_tensor_source
        from voicehub.models.neutts.native.artifacts import resolve_neucodec_artifacts, resolve_neutts_artifacts
        from voicehub.models.neutts.native.checkpoint import NeuCodecCheckpointAdapter, NeuTTSCheckpointAdapter
        from voicehub.models.neutts.native.configuration import NeuCodecConfig, NeuTTSBackboneConfig
        from voicehub.models.neutts.native.modeling import NeuTTSBackbone, NeuTTSRuntime
        from voicehub.models.neutts.native.neucodec import NeuCodecModel
        from voicehub.models.neutts.native.tokenization import NeuTTSTokenizer

        torch = import_optional(
            "torch",
            model_type="neutts",
            install_extra=None,
        )
        source = self.config.name_or_path or self.default_model_name_or_path
        artifacts = resolve_neutts_artifacts(
            source,
            checkpoint_filename=self.config.checkpoint_filename,
            revision=self.config.revision,
            cache_dir=self.config.cache_dir,
            token=self._hub_token,
            local_files_only=self.config.local_files_only,
        )
        architecture_values = read_json_file(artifacts.config)
        native_config = NeuTTSBackboneConfig.from_dict(architecture_values)
        if self._loading_for_training:
            self._validate_native_training_config(native_config)
        architectures = architecture_values.get('architectures', ())
        if isinstance(architectures, str):
            architectures = (architectures, )
        expected_architecture = native_config.causal_lm.huggingface_architecture_name
        if architectures and expected_architecture not in architectures:
            raise ValueError(
                "NeuTTS checkpoint architecture metadata conflicts with "
                f"`model_type`: expected {expected_architecture!r}.")
        tokenizer = NeuTTSTokenizer.from_tokenizer_json(
            artifacts.tokenizer,
            tokenizer_config_path=artifacts.tokenizer_config,
            bos_token_id=native_config.causal_lm.bos_token_id,
            eos_token_id=(
                native_config.causal_lm.eos_token_ids[0] if native_config.causal_lm.eos_token_ids else None),
            pad_token_id=native_config.causal_lm.pad_token_id,
            expected_vocabulary_size=native_config.causal_lm.vocab_size,
        )
        dtype = self._language_model_dtype(torch, architecture_values)
        backbone = NeuTTSBackbone(
            native_config,
            initialize=False,
            device=self.device,
            dtype=dtype,
        )
        with open_causal_lm_tensor_source(artifacts.checkpoint) as reader:
            NeuTTSCheckpointAdapter().load_streaming(
                backbone,
                reader,
                native_config.causal_lm.to_dict(),
                strict=True,
            )
        if native_config.causal_lm.tie_word_embeddings:
            backbone.tie_weights()

        local_codec = artifacts.root / "neucodec"
        codec_source = (str(local_codec) if local_codec.is_dir() else self.config.codec_name_or_path)
        codec_artifacts = resolve_neucodec_artifacts(
            codec_source,
            checkpoint_filename=self.config.codec_checkpoint_filename,
            revision=self.config.codec_revision,
            cache_dir=self.config.cache_dir,
            token=self._hub_token,
            local_files_only=self.config.local_files_only,
        )
        codec_values = read_json_file(codec_artifacts.config)
        codec_config = NeuCodecConfig.from_dict(codec_values)
        if codec_config.output_sampling_rate != self.config.sample_rate:
            raise ValueError(
                "NeuTTS and NeuCodec output sample rates disagree: "
                f"{self.config.sample_rate} != "
                f"{codec_config.output_sampling_rate}.")
        codec = NeuCodecModel(
            codec_config,
            initialize=False,
            device=self.device,
        )
        if codec_artifacts.preprocessor_config is not None:
            codec.feature_extractor.validate_preprocessor_config(
                read_json_file(codec_artifacts.preprocessor_config))
        codec._reset_derived_buffers()
        with SafeTensorReader(codec_artifacts.checkpoint) as reader:
            NeuCodecCheckpointAdapter.for_model(codec).load_streaming(
                codec,
                reader,
                codec_values,
                strict=True,
            )

        runtime = NeuTTSRuntime(
            backbone=backbone,
            tokenizer=tokenizer,
            codec=codec,
            max_context=self.config.max_total_tokens,
            min_new_tokens=self.config.min_new_tokens,
            language=self.config.language,
        )
        self.artifacts = artifacts
        self.codec_artifacts = codec_artifacts
        self.native_config = native_config
        self.tokenizer = tokenizer
        self.codec = codec
        self.model = runtime.eval()
        self._torch = torch

    def _validate_generation_inputs(self, model_inputs: dict[str, Any]) -> None:
        speaker_audio_path = model_inputs.get("speaker_audio_path")
        reference_codes = model_inputs.get("reference_codes")
        reference_text = model_inputs.get("reference_text")
        has_path = (isinstance(speaker_audio_path, (str, Path)) and bool(str(speaker_audio_path).strip()))
        has_codes = reference_codes is not None
        if has_path == has_codes:
            raise ValueError(
                "NeuTTS requires exactly one of `speaker_audio_path` or "
                "`reference_codes`, together with a non-empty "
                "`reference_text`.")
        if speaker_audio_path is not None and not has_path:
            raise ValueError("`speaker_audio_path` must be a non-empty path or None.")
        if has_path:
            reference_path = Path(speaker_audio_path).expanduser()
            if not reference_path.is_file():
                raise FileNotFoundError(f"NeuTTS reference audio was not found: {reference_path}.")
        if not isinstance(reference_text, str) or not reference_text.strip():
            raise ValueError(
                "NeuTTS requires exactly one of `speaker_audio_path` or "
                "`reference_codes`, together with a non-empty "
                "`reference_text`.")
        temperature = model_inputs.get("temperature")
        if temperature is None:
            temperature = self.config.temperature
        if (isinstance(temperature, bool) or not isinstance(temperature, Real) or
                not math.isfinite(temperature) or temperature <= 0):
            raise ValueError("`temperature` must be a finite positive number.")
        top_k = model_inputs.get("top_k")
        if top_k is None:
            top_k = self.config.top_k
        if (isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0):
            raise ValueError("`top_k` must be a positive integer.")
        max_new_tokens = model_inputs.get("max_new_tokens")
        if max_new_tokens is None:
            max_new_tokens = self.config.max_new_tokens
        if (isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens <= 0):
            raise ValueError("`max_new_tokens` must be a positive integer.")
        emotion = model_inputs.get("emotion")
        if emotion is not None and (not isinstance(emotion, str) or not emotion.strip()):
            raise ValueError("`emotion` must be a non-empty string or None.")
        phonemizer = model_inputs.get("phonemizer")
        if (phonemizer is not None and not callable(phonemizer) and
                not callable(getattr(phonemizer, "phonemize", None))):
            raise TypeError("`phonemizer` must be callable or expose phonemize().")

    def _validate_training_runtime(self) -> None:
        source = str(self.config.name_or_path or self.default_model_name_or_path).lower()
        if source.endswith((".gguf", ".onnx", ".bin", ".pt", ".pth")) or ("-gguf" in source):
            raise ValueError(
                "NeuTTS fine-tuning requires a differentiable Safetensors "
                "backbone; GGUF/ONNX are inference formats and pickle "
                "checkpoints are rejected.")
        if source == "neuphonic/neutts-2e" or "neutts-nano" in source:
            raise ValueError(
                "The pinned upstream project publishes a source-faithful "
                "fine-tuning recipe only for NeuTTS-Air. VoiceHub refuses to "
                "present an unverified Nano/2E objective as equivalent. Use "
                "`neuphonic/neutts-air` or a local Air-derived artifact.")

    @staticmethod
    def _validate_native_training_config(native_config: Any) -> None:
        causal_lm = getattr(native_config, "causal_lm", None)
        model_type = getattr(causal_lm, "model_type", None)
        input_format = getattr(native_config, "input_format", None)
        if model_type != "qwen2" or input_format != "phonemes":
            raise ValueError(
                "The pinned NeuTTS fine-tuning recipe is verified only for "
                "the phoneme-based Qwen2 NeuTTS-Air graph. This local "
                "artifact declares a different architecture.")

    def _prepare_for_inference(self) -> None:
        if self.model is not None and hasattr(self.model, "eval"):
            self.model.eval()
        backbone = getattr(self.model, "backbone", None)
        if backbone is not None and hasattr(backbone, "eval"):
            backbone.eval()
        codec = getattr(self.model, "codec", self.codec)
        if codec is not None and hasattr(codec, "eval"):
            codec.eval()
        model_config = getattr(backbone, "config", None)
        if model_config is not None and hasattr(model_config, "use_cache"):
            try:
                model_config.use_cache = True
            except (AttributeError, TypeError):
                pass

    def _prepare_for_training(self) -> None:
        backbone = getattr(self.model, "backbone", None)
        codec = getattr(self.model, "codec", self.codec)
        if backbone is None:
            raise RuntimeError("Native NeuTTS runtime has no language backbone.")
        backbone.train()
        if codec is not None:
            codec.eval()
            for parameter in codec.parameters():
                parameter.requires_grad_(False)

    def _generate(
        self,
        text: str,
        *,
        output_file: str | None = None,
        speaker_audio_path: str | None = None,
        reference_codes: Any | None = None,
        reference_text: str | None = None,
        text_phonemes: str | None = None,
        reference_phonemes: str | None = None,
        phonemizer: Any | None = None,
        emotion: str | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        max_new_tokens: int | None = None,
        seed: int | None = None,
    ) -> TTSOutput:
        if reference_codes is None:
            reference_codes = self.model.encode_reference(str(Path(speaker_audio_path).expanduser()))
        requested_seed = self.config.seed if seed is None else seed
        sampling_temperature = (self.config.temperature if temperature is None else float(temperature))
        sampling_top_k = self.config.top_k if top_k is None else top_k
        requested_tokens = (self.config.max_new_tokens if max_new_tokens is None else max_new_tokens)
        with seeded_inference(
                requested_seed,
                device=self.device,
                model_type="neutts",
        ) as fallback_seed:
            audio = self.model.infer(
                text,
                reference_codes,
                reference_text,
                emotion=emotion,
                temperature=sampling_temperature,
                top_k=sampling_top_k,
                seed=fallback_seed,
                max_new_tokens=requested_tokens,
                text_phonemes=text_phonemes,
                reference_phonemes=reference_phonemes,
                phonemizer=phonemizer,
            )
            effective_seed = getattr(self.model, "last_seed", None)
            if effective_seed is None:
                effective_seed = fallback_seed
        return finish_audio_output(
            audio,
            self.sample_rate,
            output_file=output_file,
            metadata={
                "emotion": emotion,
                "seed": effective_seed,
                "requested_seed": requested_seed,
                "voice_cloned": True,
                "native_runtime": True,
                "input_format": getattr(
                    self.model,
                    "input_format",
                    None,
                ),
            },
        )

    def _save_pretrained(self, save_directory: Path) -> None:
        if self.model is None:
            self.load()
        self.model.save_pretrained(save_directory)


NeuTTSModel = NeuTTSForTextToSpeech

__all__ = [
    "NeuTTSConfig",
    "NeuTTSForTextToSpeech",
    "NeuTTSModel",
]
