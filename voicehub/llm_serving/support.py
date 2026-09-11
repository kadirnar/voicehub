"""Fail-closed capability registry for LLM-backed TTS serving."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any

from voicehub.errors import LLMBackendCompatibilityError, UnknownModelError
from voicehub.llm_serving.configuration import LLMBackend, LLMBackendTransport
from voicehub.registry import get_model_spec, normalize_model_type

_SPEECH_DIRECT_OPTIONS = (
    "duration_tokens",
    "initial_codec_chunk_frames",
    "language",
    "max_new_tokens",
    "non_streaming_mode",
    "repetition_penalty",
    "seed",
    "speed",
    "stage_params",
    "temperature",
    "token_count",
    "top_k",
    "top_p",
    "x_vector_only_mode",
)
_SPEECH_NATIVE_ONLY_OPTIONS = frozenset({
    "ambient_sound",
    "audio_repetition_penalty",
    "audio_temperature",
    "audio_top_k",
    "audio_top_p",
    "cfg_value",
    "chunk_length",
    "class_temperature",
    "denoise",
    "duration",
    "flow_steps",
    "force_audio_gen",
    "guidance_scale",
    "inference_timesteps",
    "iterative_prompt",
    "layer_penalty_factor",
    "max_len",
    "min_len",
    "min_new_tokens",
    "normalize",
    "normalize_text",
    "num_samples",
    "num_step",
    "num_steps",
    "position_temperature",
    "postprocess_output",
    "preprocess_prompt",
    "quality",
    "ras_win_len",
    "ras_win_max_num_repeat",
    "reference_codes",
    "reference_sampling_rate",
    "retry_badcase",
    "scene_prompt",
    "sound_event",
    "speaker_audio_codes",
    "system_prompt",
    "t_shift",
    "text_temperature",
    "text_top_k",
    "text_top_p",
    "time_shift",
    "use_kv_cache",
})
_SPEECH_REQUEST_OPTIONS = frozenset({
    "instruct",
    "instruction",
    "instructions",
    "mode",
    "output_file",
    "prompt_audio_path",
    "prompt_features",
    "prompt_speech_tokens",
    "ref_audio",
    "ref_text",
    "reference_audio",
    "reference_text",
    "speaker",
    "speaker_audio",
    "speaker_audio_path",
    "speaker_embedding",
    "task_type",
    "voice",
})
_SPEECH_DEFAULT_OPTIONS = frozenset({
    "duration_tokens",
    "initial_codec_chunk_frames",
    "instruct",
    "instruction",
    "instructions",
    "language",
    "max_new_tokens",
    "mode",
    "non_streaming_mode",
    "repetition_penalty",
    "seed",
    "speaker",
    "speed",
    "stage_params",
    "task_type",
    "temperature",
    "token_count",
    "top_k",
    "top_p",
    "voice",
    "x_vector_only_mode",
})
_SPEECH_INPUT_OPTIONS = (
    frozenset(_SPEECH_DIRECT_OPTIONS) | _SPEECH_NATIVE_ONLY_OPTIONS | _SPEECH_REQUEST_OPTIONS)


@dataclass(frozen=True, slots=True)
class LLMBackendSupport:
    """One verified model/backend protocol pairing."""

    model_type: str
    backend: LLMBackend
    transports: tuple[LLMBackendTransport, ...]
    default_transport: LLMBackendTransport
    engine: str
    checkpoint_family: str
    notes: str = ""
    task_type_without_reference: str | None = None
    task_type_with_reference: str | None = None
    task_type_aliases: tuple[tuple[str, str], ...] = ()
    reference_format: str = "flat"
    speech_string_options: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        model_type = normalize_model_type(self.model_type)
        backend = LLMBackend.coerce(self.backend)
        transports = tuple(LLMBackendTransport.coerce(item) for item in self.transports)
        default_transport = LLMBackendTransport.coerce(self.default_transport)
        if backend is LLMBackend.NATIVE:
            raise ValueError("LLM backend support records require an external backend.")
        if not transports:
            raise ValueError("`transports` must contain at least one transport.")
        if any(item is LLMBackendTransport.AUTO for item in transports):
            raise ValueError("`transports` must contain concrete transports, not `auto`.")
        if len(set(transports)) != len(transports):
            raise ValueError("`transports` must not contain duplicates.")
        if default_transport not in transports:
            raise ValueError("`default_transport` must be listed in `transports`.")
        strings = {}
        for field_name in ("engine", "checkpoint_family"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"`{field_name}` must be a non-empty string.")
            strings[field_name] = value.strip()
        if not isinstance(self.notes, str):
            raise TypeError("`notes` must be a string.")
        task_types = {}
        for field_name in (
                "task_type_without_reference",
                "task_type_with_reference",
        ):
            value = getattr(self, field_name)
            if value is not None:
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"`{field_name}` must be a non-empty string or None.")
                value = value.strip()
            task_types[field_name] = value
        aliases = []
        seen_aliases = set()
        for item in tuple(self.task_type_aliases):
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise ValueError("`task_type_aliases` entries must be two-item pairs.")
            alias, task_type = item
            if not isinstance(alias, str) or not alias.strip():
                raise ValueError("`task_type_aliases` keys must be non-empty strings.")
            if not isinstance(task_type, str) or not task_type.strip():
                raise ValueError("`task_type_aliases` values must be non-empty strings.")
            alias = alias.strip().lower().replace("-", "_")
            if alias == "auto":
                raise ValueError("`auto` is reserved for the reference-aware default task type.")
            if alias in seen_aliases:
                raise ValueError(f"`task_type_aliases` contains duplicate key {alias!r}.")
            seen_aliases.add(alias)
            aliases.append((alias, task_type.strip()))
        if not isinstance(self.reference_format, str):
            raise TypeError("`reference_format` must be a string.")
        reference_format = self.reference_format.strip().lower().replace("_", "-")
        if reference_format not in {"flat", "references"}:
            raise ValueError("`reference_format` must be `flat` or `references`.")
        options = tuple(self.speech_string_options)
        if any(not isinstance(name, str) or not name.strip() for name in options):
            raise ValueError("`speech_string_options` must contain non-empty strings.")
        options = tuple(name.strip() for name in options)
        if len(set(options)) != len(options):
            raise ValueError("`speech_string_options` must not contain duplicates.")
        if any(not name.isidentifier() for name in options):
            raise ValueError("`speech_string_options` must contain identifier field names.")
        reserved = {
            "input",
            "instructions",
            "model",
            "ref_audio",
            "ref_text",
            "references",
            "response_format",
            "stream",
            "task_type",
            "text",
            "voice",
        } | frozenset(_SPEECH_DIRECT_OPTIONS) | _SPEECH_REQUEST_OPTIONS
        conflicts = sorted(set(options) & reserved)
        if conflicts:
            raise ValueError(
                "`speech_string_options` cannot override request-owned "
                "field(s): " + ", ".join(conflicts) + ".")
        has_speech_metadata = (
            any(value is not None for value in task_types.values()) or bool(aliases) or
            reference_format != "flat" or bool(options))
        if has_speech_metadata and LLMBackendTransport.SPEECH not in transports:
            raise ValueError("Speech request metadata requires the `speech` transport.")
        object.__setattr__(self, "model_type", model_type)
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "transports", transports)
        object.__setattr__(self, "default_transport", default_transport)
        object.__setattr__(self, "engine", strings["engine"])
        object.__setattr__(self, "checkpoint_family", strings["checkpoint_family"])
        object.__setattr__(self, "notes", self.notes.strip())
        for field_name, value in task_types.items():
            object.__setattr__(self, field_name, value)
        object.__setattr__(self, "task_type_aliases", tuple(aliases))
        object.__setattr__(self, "reference_format", reference_format)
        object.__setattr__(self, "speech_string_options", options)

    @property
    def speech_input_options(self) -> tuple[str, ...]:
        """Return every recognized wrapper input for speech transport.

        Recognition is not a support claim. Names in
        :attr:`speech_native_only_options` fail explicitly unless the
        support record promotes them through ``speech_string_options``.
        """
        if LLMBackendTransport.SPEECH not in self.transports:
            return ()
        return tuple(sorted(_SPEECH_INPUT_OPTIONS | frozenset(self.speech_string_options)))

    @property
    def speech_default_options(self) -> tuple[str, ...]:
        """Return model-generation defaults safe to forward to this pairing."""
        if LLMBackendTransport.SPEECH not in self.transports:
            return ()
        return tuple(sorted(_SPEECH_DEFAULT_OPTIONS | frozenset(self.speech_string_options)))

    @property
    def speech_native_only_options(self) -> tuple[str, ...]:
        """Return recognized inputs that this pairing cannot preserve."""
        if LLMBackendTransport.SPEECH not in self.transports:
            return ()
        return tuple(sorted(_SPEECH_NATIVE_ONLY_OPTIONS - frozenset(self.speech_string_options)))

    @property
    def speech_direct_options(self) -> tuple[str, ...]:
        """Return typed VoiceHub options serialized as backend fields."""
        if LLMBackendTransport.SPEECH not in self.transports:
            return ()
        return _SPEECH_DIRECT_OPTIONS

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable capability metadata."""
        return {
            "model_type": self.model_type,
            "backend": self.backend.value,
            "transports": [item.value for item in self.transports],
            "default_transport": self.default_transport.value,
            "engine": self.engine,
            "checkpoint_family": self.checkpoint_family,
            "notes": self.notes,
            "task_type_without_reference": self.task_type_without_reference,
            "task_type_with_reference": self.task_type_with_reference,
            "task_type_aliases": dict(self.task_type_aliases),
            "reference_format": self.reference_format,
            "speech_string_options": list(self.speech_string_options),
            "speech_input_options": list(self.speech_input_options),
            "speech_default_options": list(self.speech_default_options),
            "speech_native_only_options": list(self.speech_native_only_options),
        }


def _support(
        model_type: str,
        backend: LLMBackend,
        transport: LLMBackendTransport,
        *,
        engine: str,
        checkpoint_family: str,
        notes: str = "",
        task_type_without_reference: str | None = None,
        task_type_with_reference: str | None = None,
        task_type_aliases: tuple[tuple[str, str], ...] = (),
        reference_format: str = "flat",
        speech_string_options: tuple[str, ...] = (),
) -> LLMBackendSupport:
    return LLMBackendSupport(
        model_type=model_type,
        backend=backend,
        transports=(transport, ),
        default_transport=transport,
        engine=engine,
        checkpoint_family=checkpoint_family,
        notes=notes,
        task_type_without_reference=task_type_without_reference,
        task_type_with_reference=task_type_with_reference,
        task_type_aliases=task_type_aliases,
        reference_format=reference_format,
        speech_string_options=speech_string_options,
    )


_QWEN_TASK_TYPE_ALIASES = (
    ("base", "Base"),
    ("voice_clone", "Base"),
    ("customvoice", "CustomVoice"),
    ("custom_voice", "CustomVoice"),
    ("voicedesign", "VoiceDesign"),
    ("voice_design", "VoiceDesign"),
)

_SUPPORT = (
    _support(
        "orpheustts",
        LLMBackend.VLLM,
        LLMBackendTransport.TOKENS,
        engine="vLLM OpenAI completions",
        checkpoint_family="dense Llama causal LM",
        notes="VoiceHub retains the tokenizer and SNAC decoder.",
    ),
    _support(
        "orpheustts",
        LLMBackend.SGLANG,
        LLMBackendTransport.TOKENS,
        engine="SGLang token-in/token-out server",
        checkpoint_family="dense Llama causal LM",
        notes="VoiceHub retains the tokenizer and SNAC decoder.",
    ),
    _support(
        "llasa",
        LLMBackend.VLLM,
        LLMBackendTransport.TOKENS,
        engine="vLLM OpenAI completions",
        checkpoint_family="LLaSA dense Llama causal LM",
        notes="VoiceHub retains XCodec2, including reference-prefix handling.",
    ),
    _support(
        "llasa",
        LLMBackend.SGLANG,
        LLMBackendTransport.TOKENS,
        engine="SGLang token-in/token-out server",
        checkpoint_family="LLaSA dense Llama causal LM",
        notes="VoiceHub retains XCodec2, including reference-prefix handling.",
    ),
    _support(
        "qwen3tts",
        LLMBackend.VLLM,
        LLMBackendTransport.SPEECH,
        engine="vLLM-Omni",
        checkpoint_family="Qwen3-TTS Base, CustomVoice, or VoiceDesign",
        task_type_without_reference="CustomVoice",
        task_type_with_reference="Base",
        task_type_aliases=_QWEN_TASK_TYPE_ALIASES,
    ),
    _support(
        "qwen3tts",
        LLMBackend.SGLANG,
        LLMBackendTransport.SPEECH,
        engine="SGLang-Omni",
        checkpoint_family="Qwen3-TTS Base, CustomVoice, or VoiceDesign",
        task_type_without_reference="CustomVoice",
        task_type_with_reference="Base",
        task_type_aliases=_QWEN_TASK_TYPE_ALIASES,
    ),
    _support(
        "fishtts",
        LLMBackend.VLLM,
        LLMBackendTransport.SPEECH,
        engine="vLLM-Omni",
        checkpoint_family="fishaudio/s2-pro",
    ),
    _support(
        "fishtts",
        LLMBackend.SGLANG,
        LLMBackendTransport.SPEECH,
        engine="SGLang-Omni",
        checkpoint_family="fishaudio/s2-pro",
        reference_format="references",
    ),
    _support(
        "mosstts",
        LLMBackend.VLLM,
        LLMBackendTransport.SPEECH,
        engine="vLLM-Omni",
        checkpoint_family="supported MOSS-TTS pipeline",
        notes="The served checkpoint and vLLM-Omni deployment recipe must match.",
        speech_string_options=("ambient_sound", ),
    ),
    _support(
        "mosstts",
        LLMBackend.SGLANG,
        LLMBackendTransport.SPEECH,
        engine="SGLang-Omni",
        checkpoint_family="MOSS-TTS v1.5 delay or local pipeline",
    ),
    _support(
        "cosyvoice",
        LLMBackend.VLLM,
        LLMBackendTransport.SPEECH,
        engine="vLLM-Omni",
        checkpoint_family="FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
    ),
    _support(
        "voxcpm",
        LLMBackend.VLLM,
        LLMBackendTransport.SPEECH,
        engine="vLLM-Omni",
        checkpoint_family="openbmb/VoxCPM2",
    ),
    _support(
        "omnivoice",
        LLMBackend.VLLM,
        LLMBackendTransport.SPEECH,
        engine="vLLM-Omni",
        checkpoint_family="k2-fsa/OmniVoice",
    ),
    _support(
        "higgstts",
        LLMBackend.VLLM,
        LLMBackendTransport.SPEECH,
        engine="vLLM-Omni",
        checkpoint_family="Higgs Audio v2 3B",
        notes="SGLang-Omni's Higgs v3 pipeline is not compatible with this wrapper.",
    ),
)

_SUPPORT_BY_KEY = {(item.model_type, item.backend): item for item in _SUPPORT}
_BUILTIN_SUPPORT_KEYS = frozenset(_SUPPORT_BY_KEY)
_SUPPORT_LOCK = RLock()

_DEFAULT_UNSUPPORTED_REASON = "No verified engine adapter exists for this architecture."


def _architecture_backend_blocker(model_type: str) -> str:
    """Read an architecture-owned external-serving limitation lazily."""
    try:
        model_spec = get_model_spec(model_type)
    except UnknownModelError:
        return _DEFAULT_UNSUPPORTED_REASON
    architecture = model_spec.native_architecture
    if architecture is None:
        return _DEFAULT_UNSUPPORTED_REASON
    reason = architecture.metadata.get("external_llm_backend_blocker")
    if reason is None:
        return _DEFAULT_UNSUPPORTED_REASON
    if not isinstance(reason, str) or not reason.strip():
        raise TypeError(
            f"Architecture {architecture.architecture_id!r} must declare "
            "metadata.external_llm_backend_blocker as a non-empty string.")
    return reason.strip()


def list_llm_backend_support(
    *,
    backend: str | LLMBackend | None = None,
    model_type: str | None = None,
) -> tuple[LLMBackendSupport, ...]:
    """List verified pairings without importing either serving engine."""
    resolved_backend = None if backend is None else LLMBackend.coerce(backend)
    resolved_model = None if model_type is None else normalize_model_type(model_type)
    with _SUPPORT_LOCK:
        support = tuple(_SUPPORT_BY_KEY.values())
    return tuple(
        item for item in support if (resolved_backend is None or item.backend is resolved_backend) and
        (resolved_model is None or item.model_type == resolved_model))


def register_llm_backend_support(
    support: LLMBackendSupport,
    *,
    exist_ok: bool = False,
) -> None:
    """Register one process-local model/backend capability record."""
    if not isinstance(support, LLMBackendSupport):
        raise TypeError("`support` must be an LLMBackendSupport instance.")
    if not isinstance(exist_ok, bool):
        raise TypeError("`exist_ok` must be a boolean.")
    key = (support.model_type, support.backend)
    with _SUPPORT_LOCK:
        existing = _SUPPORT_BY_KEY.get(key)
        if existing is not None:
            if exist_ok and existing == support:
                return
            raise ValueError(
                f"LLM backend support is already registered for "
                f"{support.model_type!r} and {support.backend.value!r}.")
        _SUPPORT_BY_KEY[key] = support


def unregister_llm_backend_support(
    model_type: str,
    backend: str | LLMBackend,
    *,
    missing_ok: bool = False,
) -> LLMBackendSupport | None:
    """Remove one process-local capability record and return it."""
    if not isinstance(missing_ok, bool):
        raise TypeError("`missing_ok` must be a boolean.")
    key = (
        normalize_model_type(model_type),
        LLMBackend.coerce(backend),
    )
    if key in _BUILTIN_SUPPORT_KEYS:
        raise ValueError("Built-in LLM backend support records cannot be unregistered.")
    with _SUPPORT_LOCK:
        support = _SUPPORT_BY_KEY.pop(key, None)
    if support is None and not missing_ok:
        raise KeyError(f"No LLM backend support is registered for {key[0]!r} and "
                       f"{key[1].value!r}.")
    return support


def get_llm_backend_support(
    model_type: str,
    backend: str | LLMBackend,
    *,
    transport: str | LLMBackendTransport = LLMBackendTransport.AUTO,
) -> tuple[LLMBackendSupport, LLMBackendTransport]:
    """Resolve one pairing and its concrete transport, or fail clearly."""
    canonical_model = normalize_model_type(model_type)
    resolved_backend = LLMBackend.coerce(backend)
    resolved_transport = LLMBackendTransport.coerce(transport)
    if resolved_backend is LLMBackend.NATIVE:
        raise LLMBackendCompatibilityError(
            "The native VoiceHub runtime does not use an external LLM "
            "backend support record.")
    with _SUPPORT_LOCK:
        support = _SUPPORT_BY_KEY.get((canonical_model, resolved_backend))
        model_support = tuple(item for item in _SUPPORT_BY_KEY.values() if item.model_type == canonical_model)
    if support is None:
        reason = _architecture_backend_blocker(canonical_model)
        available = ", ".join(item.backend.value for item in model_support) or "native"
        raise LLMBackendCompatibilityError(
            f"{resolved_backend.value} does not support VoiceHub model "
            f"{canonical_model!r}: {reason} Available backend(s): {available}.")
    if resolved_transport is LLMBackendTransport.AUTO:
        resolved_transport = support.default_transport
    if resolved_transport not in support.transports:
        supported = ", ".join(item.value for item in support.transports)
        raise LLMBackendCompatibilityError(
            f"{resolved_backend.value} supports {canonical_model!r} through "
            f"{supported}, not {resolved_transport.value}.")
    return support, resolved_transport


__all__ = [
    "LLMBackendSupport",
    "get_llm_backend_support",
    "list_llm_backend_support",
    "register_llm_backend_support",
    "unregister_llm_backend_support",
]
