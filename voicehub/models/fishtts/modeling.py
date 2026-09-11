"""VoiceHub-native Fish Speech S2 inference and lifecycle integration."""

from __future__ import annotations

import math
from numbers import Real
from pathlib import Path
from typing import Any

from voicehub.dependencies import import_optional
from voicehub.models._shared import finish_audio_output, resolve_torch_dtype, seeded_inference
from voicehub.models.fishtts.configuration import FishTTSConfig
from voicehub.models.fishtts.native.metadata import (
    FISH_S2_CHECKPOINT,
    FISH_S2_CHECKPOINT_REVISION,
    FISH_S2_CONFIG_SHA256,
    FISH_S2_HEADER_FINGERPRINT,
    FISH_S2_INDEX_SHA256,
    FISH_S2_PARAMETER_COUNT,
    FISH_S2_SHARDS,
    FISH_S2_TENSOR_COUNT,
    FISH_S2_TOKENIZER_CONFIG_SHA256,
    FISH_S2_TOKENIZER_SHA256,
)
from voicehub.models.tts import PreTrainedTTSModel
from voicehub.outputs import TTSOutput


class FishTTSForTextToSpeech(PreTrainedTTSModel):
    """Dual-AR synthesis and semantic fine-tuning without provider runtimes."""

    config_class = FishTTSConfig
    default_model_name_or_path = FISH_S2_CHECKPOINT
    training_default_model_name_or_path = FISH_S2_CHECKPOINT
    passthrough_generation_options = frozenset({
        "chunk_length",
        "iterative_prompt",
        "max_new_tokens",
        "num_samples",
        "output_file",
        "reference_codes",
        "reference_text",
        "repetition_penalty",
        "seed",
        "speaker_audio_path",
        "temperature",
        "top_k",
        "top_p",
    })

    def __init__(
        self,
        config: FishTTSConfig | str | Path | None = None,
        *,
        model_path: str | Path | None = None,
        device: str = "auto",
        lazy_load: bool = True,
        token: str | bool | None = None,
        **config_overrides: Any,
    ) -> None:
        trust_official_codec_pickle = config_overrides.pop(
            "trust_official_codec_pickle",
            False,
        )
        if token is not None and (not isinstance(token, (str, bool)) or
                                  isinstance(token, str) and not token.strip()):
            raise ValueError("`token` must be a non-empty string, boolean, or None.")
        if not isinstance(trust_official_codec_pickle, bool):
            raise TypeError("`trust_official_codec_pickle` must be a boolean.")
        config = self._coerce_config(
            config,
            model_path=model_path,
            **config_overrides,
        )
        if not isinstance(config, FishTTSConfig):
            raise TypeError("Fish S2 requires a FishTTSConfig.")
        config.validate()
        self._hub_token = token.strip() if isinstance(token, str) else token
        # This acknowledgement is intentionally not serialized.
        self._trust_official_codec_pickle = trust_official_codec_pickle
        self.artifacts = None
        self.codec_artifacts = None
        self.native_config = None
        self.tokenizer = None
        self.codec = None
        self._codec = None
        self._runtime = None
        self._torch = None
        self._loaded_for_training = False
        super().__init__(config, device=device, lazy_load=lazy_load)

    def _resolved_dtype(self, torch, architecture_values: dict[str, Any]):
        if self.config.torch_dtype != "auto":
            return resolve_torch_dtype(
                torch,
                self.config.torch_dtype,
                self.device,
            )
        device_type = torch.device(self.device).type
        if device_type == "cuda":
            return (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
        if device_type == "mps":
            return torch.float16
        return torch.float32

    def _codec_conversion_directory(self) -> Path:
        configured = self.config.codec_conversion_directory
        if configured is not None:
            return Path(configured).expanduser()
        if self.config.cache_dir is not None:
            root = Path(self.config.cache_dir).expanduser()
        else:
            root = Path.home() / ".cache" / "voicehub"
        return (root / "fishtts" / ("s2-pro-codec-"
                                    "74fc41c5a7151c6f350af8bd7e5d6e3a"))

    def _resolve_codec_artifacts(self, semantic_artifacts):
        from voicehub.models.fishtts.native.artifacts import (
            resolve_fish_codec_artifacts,
            resolve_official_fish_legacy_codec,
        )
        from voicehub.models.fishtts.native.checkpoint import convert_legacy_fish_codec

        embedded = semantic_artifacts.root / "codec"
        if ((embedded / "config.json").is_file() and (embedded / "model.safetensors").is_file()):
            return resolve_fish_codec_artifacts(embedded)
        if self.config.codec_name_or_path is not None:
            return resolve_fish_codec_artifacts(
                self.config.codec_name_or_path,
                revision=self.config.codec_revision,
                cache_dir=self.config.cache_dir,
                token=self._hub_token,
                local_files_only=self.config.local_files_only,
            )
        conversion = self._codec_conversion_directory()
        if ((conversion / "config.json").is_file() and (conversion / "model.safetensors").is_file()):
            return resolve_fish_codec_artifacts(conversion)
        if not self._trust_official_codec_pickle:
            raise PermissionError(
                "Fish Audio publishes S2-Pro's ModifiedDAC only as "
                "`codec.pth`. VoiceHub never loads that pickle during "
                "steady-state inference. Either pass `codec_name_or_path` "
                "pointing to an audited Safetensors conversion, call "
                "`convert_legacy_fish_codec(...)` explicitly, or opt into "
                "the pinned one-time conversion with "
                "`trust_official_codec_pickle=True`.")
        local_legacy = semantic_artifacts.root / "codec.pth"
        if local_legacy.is_file():
            legacy = local_legacy
        elif (semantic_artifacts.source == FISH_S2_CHECKPOINT and
              semantic_artifacts.revision == FISH_S2_CHECKPOINT_REVISION):
            legacy = resolve_official_fish_legacy_codec(
                cache_dir=self.config.cache_dir,
                token=self._hub_token,
                local_files_only=self.config.local_files_only,
            )
        else:
            raise PermissionError(
                "Automatic Fish codec conversion is restricted to the "
                "immutable official S2-Pro artifact. Convert this custom "
                "codec explicitly and pass its safe directory.")
        convert_legacy_fish_codec(
            legacy,
            conversion,
            trust_legacy_pickle=True,
            verify_official_integrity=True,
        )
        return resolve_fish_codec_artifacts(conversion)

    def _verify_official_metadata(self, artifacts, report) -> None:
        if not (artifacts.source == FISH_S2_CHECKPOINT and artifacts.revision == FISH_S2_CHECKPOINT_REVISION):
            return
        expected = (
            FISH_S2_TENSOR_COUNT,
            FISH_S2_PARAMETER_COUNT,
            FISH_S2_HEADER_FINGERPRINT,
        )
        actual = (
            report.tensor_count,
            report.parameter_count,
            report.header_fingerprint,
        )
        if actual != expected:
            raise ValueError(
                "Official Fish S2 semantic inventory mismatch: "
                f"found={actual!r}, expected={expected!r}.")
        if not self.config.verify_official_integrity:
            return
        from voicehub.models.fishtts.native.checkpoint import verify_file_integrity

        verify_file_integrity(
            artifacts.config,
            expected_sha256=FISH_S2_CONFIG_SHA256,
        )
        if artifacts.checkpoint.name.endswith(".safetensors.index.json"):
            verify_file_integrity(
                artifacts.checkpoint,
                expected_sha256=FISH_S2_INDEX_SHA256,
            )
        verify_file_integrity(
            artifacts.tokenizer,
            expected_sha256=FISH_S2_TOKENIZER_SHA256,
        )
        if artifacts.tokenizer_config is None:
            raise FileNotFoundError("The pinned official Fish S2 tokenizer config is missing.")
        verify_file_integrity(
            artifacts.tokenizer_config,
            expected_sha256=FISH_S2_TOKENIZER_CONFIG_SHA256,
        )
        if self.config.verify_full_shard_hashes:
            for filename, metadata in FISH_S2_SHARDS.items():
                verify_file_integrity(
                    artifacts.root / filename,
                    expected_size=int(metadata["size"]),
                    expected_sha256=str(metadata["sha256"]),
                )

    def _attach_codec_runtime(self, *, torch, dtype) -> None:
        if self.model is None or self.artifacts is None or self.tokenizer is None:
            raise RuntimeError("Fish semantic artifacts must be loaded before its codec.")
        from voicehub.hub import read_json_file
        from voicehub.models.fishtts.native.checkpoint import load_fish_codec_checkpoint
        from voicehub.models.fishtts.native.codec import FishModifiedDAC
        from voicehub.models.fishtts.native.configuration import FishCodecConfig
        from voicehub.models.fishtts.native.runtime import FishS2Runtime

        codec_artifacts = self._resolve_codec_artifacts(self.artifacts)
        codec_values = read_json_file(codec_artifacts.config)
        codec_config = FishCodecConfig.from_dict(codec_values)
        native_config = self.native_config
        if (codec_config.sample_rate != native_config.sample_rate or
                codec_config.num_codebooks != native_config.num_codebooks or
                codec_config.semantic_codebook_size != native_config.codebook_size):
            raise ValueError("Fish ModifiedDAC and semantic protocol are incompatible.")
        with torch.device("meta"):
            codec = FishModifiedDAC(codec_config, initialize=False)
        load_fish_codec_checkpoint(
            codec,
            codec_artifacts.checkpoint,
            device=self.device,
            dtype=dtype,
        )
        codec.eval()
        for parameter in codec.parameters():
            parameter.requires_grad_(False)
        self.codec_artifacts = codec_artifacts
        self.codec = codec
        self._codec = codec
        self._runtime = FishS2Runtime(
            semantic_model=self.model,
            tokenizer=self.tokenizer,
            codec=codec,
        )

    def ensure_codec_loaded(self):
        """Attach the frozen safe codec without replacing semantic weights."""
        with self._lifecycle_lock:
            if self.model is None:
                self.load_for_training()
            if self._runtime is None:
                if self._torch is None:
                    raise RuntimeError("Fish PyTorch runtime is not loaded.")
                dtype = next(self.model.parameters()).dtype
                self._attach_codec_runtime(
                    torch=self._torch,
                    dtype=dtype,
                )
            return self._codec

    def _load_pretrained_model(self) -> None:
        from voicehub.hub import read_json_file
        from voicehub.models.fishtts.native.artifacts import resolve_fish_semantic_artifacts
        from voicehub.models.fishtts.native.checkpoint import load_fish_semantic_checkpoint
        from voicehub.models.fishtts.native.configuration import FishS2Config
        from voicehub.models.fishtts.native.modeling import FishS2ForConditionalGeneration
        from voicehub.models.fishtts.native.tokenization import FishTokenizer

        torch = import_optional(
            "torch",
            model_type="fishtts",
            install_extra=None,
        )
        source = self.config.name_or_path or self.default_model_name_or_path
        artifacts = resolve_fish_semantic_artifacts(
            source,
            revision=self.config.revision,
            cache_dir=self.config.cache_dir,
            token=self._hub_token,
            local_files_only=self.config.local_files_only,
        )
        architecture_values = read_json_file(artifacts.config)
        native_config = FishS2Config.from_dict(architecture_values)
        if native_config.sample_rate != self.config.sample_rate:
            raise ValueError("Fish wrapper and semantic checkpoint sample rates disagree.")
        tokenizer = FishTokenizer.from_tokenizer_json(
            artifacts.tokenizer,
            config=native_config,
            tokenizer_config_path=artifacts.tokenizer_config,
        )
        dtype = self._resolved_dtype(torch, architecture_values)
        with torch.device("meta"):
            semantic_model = FishS2ForConditionalGeneration(
                native_config,
                initialize=False,
            )
        semantic_report = load_fish_semantic_checkpoint(
            semantic_model,
            artifacts.checkpoint,
            device=self.device,
            dtype=dtype,
        )
        self._verify_official_metadata(artifacts, semantic_report)

        self.artifacts = artifacts
        self.native_config = native_config
        self.tokenizer = tokenizer
        self._torch = torch
        self.model = semantic_model
        self._loaded_for_training = self.is_training_load
        if not self.is_training_load:
            self._attach_codec_runtime(torch=torch, dtype=dtype)

    def _validate_training_runtime(self) -> None:
        source = str(self.config.name_or_path or self.default_model_name_or_path).lower()
        if source.endswith((".gguf", ".onnx", ".bin", ".pt", ".pth")) or "int4" in source or "int8" in source:
            raise ValueError(
                "Fish fine-tuning requires differentiable Safetensors, not "
                "GGUF/ONNX/quantized/pickle semantic weights.")

    def _prepare_for_training(self) -> None:
        if self.model is None:
            raise RuntimeError("Fish semantic model is not loaded.")
        if self._runtime is not None:
            self._runtime.prepare_for_training()
        else:
            self.model.clear_caches()
            self.model.train()
        self._loaded_for_training = True

    def _prepare_for_inference(self) -> None:
        if self._runtime is None:
            if self._torch is None:
                raise RuntimeError("Fish PyTorch runtime is not loaded.")
            dtype = next(self.model.parameters()).dtype
            self._attach_codec_runtime(torch=self._torch, dtype=dtype)
        self._runtime.prepare_for_inference()
        self._loaded_for_training = False

    def _validate_generation_inputs(self, model_inputs: dict[str, Any]) -> None:
        speaker_audio_path = model_inputs.get("speaker_audio_path")
        reference_codes = model_inputs.get("reference_codes")
        reference_text = model_inputs.get("reference_text")
        has_audio = speaker_audio_path is not None
        has_codes = reference_codes is not None
        if has_audio and has_codes:
            raise ValueError("Pass either `speaker_audio_path` or `reference_codes`, "
                             "not both.")
        if reference_text is not None and not isinstance(reference_text, str):
            raise TypeError("`reference_text` must be a string or None.")
        if (has_audio or has_codes) and (not isinstance(reference_text, str) or not reference_text.strip()):
            raise ValueError("Fish voice cloning requires a non-empty `reference_text`.")
        if reference_text and not (has_audio or has_codes):
            raise ValueError("`reference_text` requires reference audio or codes.")
        if has_audio:
            if (not isinstance(speaker_audio_path, (str, Path)) or not str(speaker_audio_path).strip()):
                raise ValueError("`speaker_audio_path` must be a non-empty path or None.")
            path = Path(speaker_audio_path).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"Fish reference audio was not found: {path}.")
            model_inputs["speaker_audio_path"] = str(path)
        for name in ("max_new_tokens", "chunk_length", "top_k"):
            value = model_inputs.get(name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"`{name}` must be a positive integer.")
        for name, lower, upper in (
            ("temperature", 0.0, 2.0),
            ("top_p", 0.0, 1.0),
        ):
            value = model_inputs.get(name)
            if value is None:
                continue
            if (isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)) or
                    not lower < float(value) <= upper or name == "temperature" and float(value) == upper):
                bracket = "(0, 2)" if name == "temperature" else "(0, 1]"
                raise ValueError(f"`{name}` must be finite and in {bracket}.")
        repetition = model_inputs.get("repetition_penalty")
        if repetition is not None and (isinstance(repetition, bool) or not isinstance(repetition, Real) or
                                       not math.isfinite(float(repetition)) or float(repetition) <= 0):
            raise ValueError("`repetition_penalty` must be a finite positive number.")
        num_samples = model_inputs.get("num_samples", 1)
        if num_samples != 1:
            raise ValueError("Fish waveform generation requires `num_samples=1`.")
        iterative = model_inputs.get("iterative_prompt")
        if iterative is not None and not isinstance(iterative, bool):
            raise TypeError("`iterative_prompt` must be a boolean.")

    def _generate(
        self,
        text: str,
        *,
        output_file: str | None = None,
        speaker_audio_path: str | None = None,
        reference_codes: Any | None = None,
        reference_text: str | None = None,
        max_new_tokens: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float = 1.1,
        temperature: float | None = None,
        chunk_length: int | None = None,
        iterative_prompt: bool = True,
        num_samples: int = 1,
        seed: int | None = None,
    ) -> TTSOutput:
        del repetition_penalty, num_samples
        if self._runtime is None or self._torch is None:
            raise RuntimeError("Fish runtime is not loaded.")
        if reference_codes is None and speaker_audio_path is not None:
            reference_codes = self._runtime.encode_reference(speaker_audio_path)
        requested_seed = self.config.seed if seed is None else seed
        maximum = (self.config.max_new_tokens if max_new_tokens is None else max_new_tokens)
        nucleus = self.config.top_p if top_p is None else float(top_p)
        candidates = self.config.top_k if top_k is None else top_k
        sampling_temperature = (self.config.temperature if temperature is None else float(temperature))
        chunk_bytes = (self.config.chunk_length if chunk_length is None else chunk_length)
        if not iterative_prompt:
            chunk_bytes = max(
                chunk_bytes,
                len(text.encode("utf-8")),
            )
        with seeded_inference(
                requested_seed,
                device=self.device,
                model_type="fishtts",
        ) as effective_seed:
            audio = self._runtime.infer(
                text,
                reference_text=reference_text,
                reference_codes=reference_codes,
                maximum_chunk_bytes=chunk_bytes,
                max_new_tokens=maximum,
                temperature=sampling_temperature,
                top_p=nucleus,
                top_k=candidates,
                seed=effective_seed,
            )
        return finish_audio_output(
            audio.detach().cpu(),
            self.sample_rate,
            output_file=output_file,
            metadata={
                "native_runtime": True,
                "requested_seed": requested_seed,
                "seed": effective_seed,
                "voice_cloned": reference_codes is not None,
            },
        )

    def _save_pretrained(self, save_directory: Path) -> None:
        self.ensure_codec_loaded()
        self._runtime.save_pretrained(save_directory)


FishTTS = FishTTSForTextToSpeech

__all__ = [
    "FishTTS",
    "FishTTSConfig",
    "FishTTSForTextToSpeech",
]
