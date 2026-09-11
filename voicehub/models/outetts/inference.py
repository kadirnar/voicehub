"""VoiceHub-native OuteTTS V3 inference."""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
from pathlib import Path
from typing import Any

from voicehub.dependencies import import_optional
from voicehub.models._shared import finish_audio_output, resolve_torch_dtype, seeded_inference
from voicehub.models.outetts.configuration import OuteTTSConfig
from voicehub.models.tts import PreTrainedTTSModel
from voicehub.outputs import TTSOutput


class OuteTTSForTextToSpeech(PreTrainedTTSModel):
    """OuteTTS synthesis using only VoiceHub, Python, and PyTorch."""

    config_class = OuteTTSConfig
    default_model_name_or_path = "OuteAI/Llama-OuteTTS-1.0-1B"
    _BACKENDS = (
        "HF",
        "NATIVE",
        "LLAMACPP",
        "EXL2",
        "EXL2ASYNC",
        "VLLM",
        "LLAMACPP_SERVER",
        "LLAMACPP_ASYNC_SERVER",
    )
    _NATIVE_BACKENDS = frozenset({"HF", "NATIVE"})
    _INTERFACE_VERSIONS = ("V1", "V2", "V3")
    _GENERATION_TYPES = (
        "REGULAR",
        "CHUNKED",
        "GUIDED_WORDS",
        "STREAM",
        "BATCH",
    )
    _BATCH_BACKENDS = frozenset({
        "EXL2ASYNC",
        "VLLM",
        "LLAMACPP_ASYNC_SERVER",
    })
    _GUIDED_WORDS_BACKENDS = frozenset({
        "LLAMACPP",
        "LLAMACPP_SERVER",
    })
    _SAMPLER_OPTIONS = frozenset({
        "temperature",
        "repetition_penalty",
        "repetition_range",
        "top_k",
        "top_p",
        "min_p",
        "mirostat_tau",
        "mirostat_eta",
        "mirostat",
    })

    def __init__(
        self,
        config: OuteTTSConfig | str | None = None,
        *,
        model_path: str | None = None,
        device: str = "auto",
        lazy_load: bool = True,
        token: str | bool | None = None,
        **config_overrides,
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
        self._hub_token = token
        self.tokenizer = None
        self.codec = None
        self.artifacts = None
        self.codec_artifacts = None
        self.native_config = None
        self._torch = None
        super().__init__(config, device=device, lazy_load=lazy_load)

    @staticmethod
    def _enum_member(enum_type, value: str, *, option_name: str):
        """Retain the historical enum helper without importing OuteTTS."""
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"`{option_name}` must be a non-empty string.")
        compact = (value.strip().upper().replace("-", "").replace("_", "").replace(".", ""))
        for member in enum_type:
            candidates = (member.name, str(member.value))
            if any(candidate.upper().replace("-", "").replace("_", "").replace(".", "") == compact
                   for candidate in candidates):
                return member
        supported = ", ".join(member.name.lower() for member in enum_type)
        raise ValueError(f"Unsupported OuteTTS {option_name} {value!r}. "
                         f"Choose one of: {supported}.")

    @staticmethod
    def _canonical_choice(
        value: str,
        choices: tuple[str, ...],
        *,
        option_name: str,
    ) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"`{option_name}` must be a non-empty string.")
        compact = (value.strip().upper().replace("-", "").replace("_", "").replace(".", ""))
        for choice in choices:
            if choice.replace("_", "") == compact:
                return choice
            if choice.startswith("V") and choice[1:] == compact:
                return choice
        supported = ", ".join(choice.lower() for choice in choices)
        raise ValueError(f"Unsupported OuteTTS {option_name} {value!r}. "
                         f"Choose one of: {supported}.")

    def _interface_version(self) -> str:
        return self._canonical_choice(
            self.config.interface_version,
            self._INTERFACE_VERSIONS,
            option_name="interface_version",
        )

    def _backend(self) -> str:
        return self._canonical_choice(
            self.config.backend,
            self._BACKENDS,
            option_name="backend",
        )

    def _generation_type(self, value: str | None) -> str:
        if value is None:
            return ("BATCH" if self._backend() in self._BATCH_BACKENDS else "CHUNKED")
        return self._canonical_choice(
            value,
            self._GENERATION_TYPES,
            option_name="generation_type",
        )

    def _configured_max_seq_length(self) -> int:
        value = self.config.max_seq_length
        if value is None:
            return 8_192
        if (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
            raise ValueError("OuteTTS `max_seq_length` must be a positive integer or None.")
        return value

    def _validate_backend_generation_pair(
        self,
        *,
        backend: str,
        interface_version: str,
        generation_type: str,
    ) -> None:
        if generation_type == "STREAM":
            raise ValueError(
                "OuteTTS streaming returns an iterator and is not supported "
                "by the waveform-returning VoiceHub `generate()` API.")
        if backend in self._BATCH_BACKENDS and generation_type != "BATCH":
            raise ValueError(
                f"OuteTTS backend {backend.lower()!r} only supports batch "
                "generation. Set `generation_type='batch'`.")
        if generation_type == "BATCH" and backend not in self._BATCH_BACKENDS:
            supported = ", ".join(name.lower() for name in sorted(self._BATCH_BACKENDS))
            raise ValueError("OuteTTS batch generation requires an asynchronous backend: "
                             f"{supported}.")
        if generation_type == "GUIDED_WORDS":
            if interface_version != "V3":
                raise ValueError("OuteTTS guided-words generation requires "
                                 "`interface_version='V3'`.")
            if backend not in self._GUIDED_WORDS_BACKENDS:
                supported = ", ".join(name.lower() for name in sorted(self._GUIDED_WORDS_BACKENDS))
                raise ValueError(
                    "OuteTTS guided-words generation requires a llama.cpp "
                    f"backend: {supported}.")
        if backend not in self._NATIVE_BACKENDS:
            raise ValueError(
                f"OuteTTS backend {backend.lower()!r} depends on an external "
                "runtime. VoiceHub-native OuteTTS supports `backend='native'` "
                "(the legacy `hf` spelling remains an alias).")
        if interface_version != "V3":
            raise ValueError("VoiceHub-native OuteTTS supports the audited V3 protocol "
                             "only.")
        if generation_type not in {"REGULAR", "CHUNKED"}:
            raise ValueError("VoiceHub-native OuteTTS supports regular and chunked "
                             "generation only.")

    def _validate_sampler(self, sampler: dict | None) -> None:
        if sampler is None:
            return
        if not isinstance(sampler, dict):
            raise TypeError("`sampler` must be a dictionary or None.")
        unsupported = sorted(set(sampler) - self._SAMPLER_OPTIONS)
        if unsupported:
            raise ValueError("Unsupported OuteTTS sampler option(s): "
                             f"{', '.join(unsupported)}.")
        if sampler.get("mirostat") is True:
            raise ValueError("Native OuteTTS does not implement the provider-specific "
                             "Mirostat sampler.")
        top_k = sampler.get("top_k")
        if top_k is not None and (isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0):
            raise ValueError("OuteTTS sampler `top_k` must be non-negative.")
        repetition_range = sampler.get("repetition_range")
        if repetition_range is not None and (isinstance(repetition_range, bool) or
                                             not isinstance(repetition_range, int) or repetition_range <= 0):
            raise ValueError("OuteTTS sampler `repetition_range` must be positive.")
        mirostat = sampler.get("mirostat")
        if mirostat is not None and not isinstance(mirostat, bool):
            raise TypeError("OuteTTS sampler `mirostat` must be a boolean.")
        for name in ("top_p", "min_p"):
            value = sampler.get(name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, Real) or
                                      not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError(f"OuteTTS sampler `{name}` must be finite and in [0, 1].")
        temperature = sampler.get("temperature")
        if temperature is not None and (isinstance(temperature, bool) or not isinstance(temperature, Real) or
                                        not math.isfinite(temperature) or temperature < 0):
            raise ValueError("OuteTTS sampler `temperature` must be finite and "
                             "non-negative.")
        for name in (
                "repetition_penalty",
                "mirostat_tau",
                "mirostat_eta",
        ):
            value = sampler.get(name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, Real) or
                                      not math.isfinite(value) or value <= 0):
                raise ValueError(f"OuteTTS sampler `{name}` must be finite and positive.")

    @staticmethod
    def _validate_profile_mapping(value: Any) -> None:
        if value is not None and not isinstance(value, Mapping):
            raise TypeError("`speaker_profile` must be a V3 profile mapping or None.")

    def _validate_generation_inputs(self, model_inputs: dict) -> None:
        interface_version = self._interface_version()
        backend = self._backend()
        generation_type = self._generation_type(model_inputs.get("generation_type"))

        speaker_audio_path = model_inputs.get("speaker_audio_path")
        speaker_profile_path = model_inputs.get("speaker_profile_path")
        speaker_profile = model_inputs.get("speaker_profile")
        supplied = sum(
            value is not None for value in (
                speaker_audio_path,
                speaker_profile_path,
                speaker_profile,
            ))
        if supplied > 1:
            raise ValueError(
                "Pass only one of `speaker_audio_path`, "
                "`speaker_profile_path`, or `speaker_profile`.")
        if (interface_version in {"V1", "V2"} and speaker_audio_path is None and
                speaker_profile_path is None and speaker_profile is None):
            raise ValueError(
                f"OuteTTS {interface_version} has no bundled default speaker. "
                "Provide `speaker_audio_path` or `speaker_profile_path`.")
        self._validate_backend_generation_pair(
            backend=backend,
            interface_version=interface_version,
            generation_type=generation_type,
        )
        self._validate_profile_mapping(speaker_profile)
        for name, value in (
            ("speaker_audio_path", speaker_audio_path),
            ("speaker_profile_path", speaker_profile_path),
        ):
            if value is None:
                continue
            if (not isinstance(value, (str, Path)) or not str(value).strip()):
                raise ValueError(f"`{name}` must be a non-empty local path or None.")
            path = Path(value).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"OuteTTS {name.replace('_', ' ')} was not found: {path}.")
        if speaker_audio_path is not None:
            raise ValueError(
                "Native OuteTTS does not approximate word timestamps from raw "
                "speaker audio. Create an author-compatible V3 speaker profile "
                "and pass `speaker_profile_path`.")
        speaker = model_inputs.get("speaker", "EN-FEMALE-1-NEUTRAL")
        if not isinstance(speaker, str) or not speaker.strip():
            raise ValueError("`speaker` must be a non-empty speaker name.")
        max_seq_length = self._configured_max_seq_length()
        max_length = model_inputs.get("max_length")
        if max_length is None:
            max_length = max_seq_length
        if (isinstance(max_length, bool) or not isinstance(max_length, int) or max_length <= 0):
            raise ValueError("`max_length` must be a positive integer.")
        if max_length > max_seq_length:
            raise ValueError(
                f"OuteTTS `max_length` ({max_length}) exceeds the configured "
                f"`max_seq_length` ({max_seq_length}).")
        self._validate_sampler(model_inputs.get("sampler"))

    def _validate_native_configuration(self) -> None:
        options = self.config.additional_model_config
        prohibited = {
            "load_in_4bit",
            "load_in_8bit",
            "quantization_config",
            "attn_implementation",
            "trust_remote_code",
        }
        configured = sorted(
            name for name in prohibited if name in options and options[name] not in (None, False))
        if configured:
            raise ValueError(
                "Native OuteTTS rejects external/quantized model options: " + ", ".join(configured))

    def _validate_training_runtime(self) -> None:
        backend = self._backend()
        if backend not in self._NATIVE_BACKENDS:
            raise ValueError(
                "OuteTTS fine-tuning requires the differentiable native "
                f"backend; {self.config.backend!r} is unsupported.")
        if self._interface_version() != "V3":
            raise ValueError("OuteTTS fine-tuning supports the audited V3 objective only.")
        self._validate_native_configuration()

    def _load_pretrained_model(self) -> None:
        self._validate_native_configuration()
        torch = import_optional(
            "torch",
            model_type="outetts",
            install_extra=None,
        )
        from voicehub.models.outetts.native.runtime import load_outetts_runtime

        dtype = None
        if self.config.torch_dtype != "auto":
            dtype = resolve_torch_dtype(
                torch,
                self.config.torch_dtype,
                self.device,
            )
        runtime = load_outetts_runtime(
            self.config.name_or_path or self.default_model_name_or_path,
            tokenizer_source=self.config.tokenizer_path,
            revision=self.config.revision,
            codec_source=self.config.codec_name_or_path,
            codec_revision=self.config.codec_revision,
            cache_dir=self.config.cache_dir,
            token=self._hub_token,
            local_files_only=self.config.local_files_only,
            device=self.device,
            dtype=dtype,
        )
        self.model = runtime
        self.tokenizer = runtime.tokenizer
        self.codec = runtime.codec
        self.artifacts = runtime.artifacts
        self.codec_artifacts = runtime.codec_artifacts
        self.native_config = runtime.language_model.config
        self.config.sample_rate = runtime.sample_rate
        self._torch = torch

    def _prepare_for_training(self) -> None:
        self.model.language_model.train()
        self.model.codec.eval()
        for parameter in self.model.codec.parameters():
            parameter.requires_grad_(False)

    def _prepare_for_inference(self) -> None:
        """Restore serving mode while preserving optimizer parameter
        identity."""
        candidate = self.model
        seen: set[int] = set()
        modules = []
        language_model = None
        for _ in range(5):
            if candidate is None or id(candidate) in seen:
                break
            seen.add(id(candidate))
            if hasattr(candidate, "eval"):
                modules.append(candidate)
                language_model = candidate
            nested = getattr(candidate, "language_model", None)
            if nested is None:
                nested = getattr(candidate, "model", None)
            if nested is None or nested is candidate:
                break
            candidate = nested
        for module in modules:
            module.eval()
        model_config = getattr(language_model, "config", None)
        if model_config is not None and hasattr(model_config, "use_cache"):
            try:
                model_config.use_cache = True
            except (AttributeError, TypeError):
                pass
        codec = getattr(self.model, "codec", self.codec)
        if codec is not None and hasattr(codec, "eval"):
            codec.eval()

    @staticmethod
    def _normalize_output_file(output_file: str | None) -> str | None:
        if output_file is None:
            return None
        output_path = Path(output_file)
        if output_path.suffix:
            return str(output_path)
        return f"{output_path}.wav"

    def _generate(
        self,
        text: str,
        *,
        output_file: str | None = None,
        speaker: str = "EN-FEMALE-1-NEUTRAL",
        speaker_audio_path: str | None = None,
        speaker_profile_path: str | None = None,
        speaker_profile: Mapping[str, Any] | None = None,
        generation_type: str | None = None,
        max_length: int | None = None,
        sampler: dict | None = None,
        seed: int | None = None,
    ) -> TTSOutput:
        del speaker_audio_path
        resolved_generation_type = self._generation_type(generation_type)
        if max_length is None:
            max_length = self._configured_max_seq_length()
        profile = self.model.resolve_speaker(
            speaker=speaker,
            speaker_profile=speaker_profile,
            speaker_profile_path=speaker_profile_path,
        )
        resolved_sampler = dict(sampler or {})
        with seeded_inference(
                seed,
                device=self.device,
                model_type="outetts",
        ) as effective_seed:
            with self._torch.inference_mode():
                audio = self.model.generate(
                    text,
                    speaker=profile,
                    generation_type=resolved_generation_type,
                    max_length=max_length,
                    sampler=resolved_sampler,
                    seed=effective_seed,
                )
        return finish_audio_output(
            audio,
            self.sample_rate,
            output_file=self._normalize_output_file(output_file),
            metadata={
                "speaker": speaker,
                "backend": "native",
                "interface_version": "v3",
                "generation_type": resolved_generation_type.lower(),
                "seed": effective_seed,
                "requested_seed": seed,
            },
        )

    def _save_pretrained(self, save_directory: Path) -> None:
        if self.model is None:
            self.load()
        self.model.save_pretrained(save_directory)


OuteTTS = OuteTTSForTextToSpeech

__all__ = [
    "OuteTTS",
    "OuteTTSConfig",
    "OuteTTSForTextToSpeech",
]
