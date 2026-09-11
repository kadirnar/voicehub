"""Configurations for ASR providers with native checkpoint runtimes."""

from collections.abc import Mapping
from math import isfinite
from numbers import Integral, Real
from pathlib import Path

from voicehub.configuration import VoiceHubConfig
from voicehub.models.asr_whisper_native.configuration import WhisperASRConfig

_SECRET_OPTIONS = frozenset({
    "access_token",
    "api_key",
    "auth_token",
    "fetch_config",
    "hf_token",
    "token",
    "use_auth_token",
})


def _secret_keys(value) -> set[str]:
    found = set()
    if isinstance(value, Mapping):
        for name, nested in value.items():
            normalized = str(name).strip().lower()
            if normalized in _SECRET_OPTIONS:
                found.add(normalized)
            found.update(_secret_keys(nested))
    elif isinstance(value, (tuple, list)):
        for nested in value:
            found.update(_secret_keys(nested))
    return found


def _non_empty_string(value, *, name: str, allow_none: bool = False):
    if allow_none and value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        suffix = " or None" if allow_none else ""
        raise ValueError(f"`{name}` must be a non-empty string{suffix}.")
    return value.strip()


class _NativeASRConfig(VoiceHubConfig):

    def __init__(
            self,
            *,
            sample_rate: int = 16_000,
            model_kwargs: Mapping | None = None,
            inference_config=None,
            _managed_model_options=(),
            **kwargs,
    ):
        if (isinstance(sample_rate, bool) or not isinstance(sample_rate, Integral) or sample_rate <= 0):
            raise ValueError("ASR `sample_rate` must be a positive integer.")
        if model_kwargs is not None and not isinstance(model_kwargs, Mapping):
            raise TypeError("`model_kwargs` must be a mapping or None.")
        model_kwargs = dict(model_kwargs or {})
        if (_secret_keys(kwargs) or _secret_keys(model_kwargs) or _secret_keys(inference_config)):
            raise ValueError(
                "Authentication credentials are runtime-only values and "
                "cannot be stored in an ASR configuration.")
        collisions = sorted(set(model_kwargs) & set(_managed_model_options))
        if collisions:
            formatted = ", ".join(f"`{name}`" for name in collisions)
            raise ValueError(f"`model_kwargs` cannot override managed option(s): {formatted}.")
        super().__init__(
            sample_rate=int(sample_rate),
            inference_config=inference_config or {},
            model_kwargs=model_kwargs,
            **kwargs,
        )


class FasterWhisperConfig(WhisperASRConfig):
    """Compatibility configuration for native Whisper execution.

    ``compute_type`` is retained as an input alias for callers migrating
    from CTranslate2. Quantized CTranslate2 modes are rejected
    explicitly until a VoiceHub quantization pass can provide equivalent
    checkpoint semantics.
    """

    model_type = "asr_faster_whisper"

    def __init__(
        self,
        *,
        compute_type: str = "default",
        cpu_threads: int = 0,
        num_workers: int = 1,
        model_kwargs: Mapping | None = None,
        torch_dtype: str | None = None,
        **kwargs,
    ):
        compute_type = _non_empty_string(
            compute_type,
            name="compute_type",
        ).lower()
        dtype_by_compute_type = {
            "auto": "auto",
            "default": "auto",
            "bfloat16": "bfloat16",
            "float16": "float16",
            "float32": "float32",
        }
        try:
            resolved_dtype = dtype_by_compute_type[compute_type]
        except KeyError as error:
            raise ValueError(
                "Native faster-whisper compatibility supports compute_type "
                "'default', 'auto', 'float32', 'float16', or 'bfloat16'. "
                "CTranslate2 quantized modes require a future VoiceHub "
                "quantization pass.") from error
        if torch_dtype is not None:
            if not isinstance(torch_dtype, str):
                raise TypeError("`torch_dtype` must be a string or None.")
            normalized_dtype = torch_dtype.strip().lower()
            if (compute_type not in {"default", "auto"} and normalized_dtype != resolved_dtype):
                raise ValueError("`compute_type` and `torch_dtype` select different "
                                 "numeric dtypes.")
            resolved_dtype = normalized_dtype
        if (isinstance(cpu_threads, bool) or not isinstance(cpu_threads, Integral) or cpu_threads < 0):
            raise ValueError("`cpu_threads` must be a non-negative integer.")
        if (isinstance(num_workers, bool) or not isinstance(num_workers, Integral) or num_workers <= 0):
            raise ValueError("`num_workers` must be a positive integer.")
        if cpu_threads != 0 or num_workers != 1:
            raise ValueError(
                "`cpu_threads` and `num_workers` were CTranslate2 runtime "
                "controls and cannot be applied silently to the native graph.")
        if model_kwargs is not None and not isinstance(model_kwargs, Mapping):
            raise TypeError("`model_kwargs` must be a mapping or None.")
        model_kwargs = dict(model_kwargs or {})
        if model_kwargs:
            names = ", ".join(sorted(model_kwargs))
            raise ValueError(
                "Native faster-whisper compatibility does not accept "
                f"CTranslate2 `model_kwargs`: {names}.")
        super().__init__(
            torch_dtype=resolved_dtype,
            **kwargs,
        )
        self.compute_type = compute_type
        self.cpu_threads = 0
        self.num_workers = 1
        self.model_kwargs = {}


class WhisperXConfig(WhisperASRConfig):
    """Configure the native Whisper plus CTC forced-alignment pipeline.

    ``compute_type`` remains a compatibility alias for older WhisperX
    configurations.  It selects the dtype of VoiceHub's native Whisper
    graph; CTranslate2 execution and arbitrary upstream loader kwargs
    are deliberately not retained.
    """

    model_type = "asr_whisperx"

    def __init__(
        self,
        *,
        compute_type: str = "default",
        align_output: bool = False,
        alignment_model_path: str | Path | None = None,
        alignment_revision: str | None = None,
        alignment_cache_dir: str | Path | None = None,
        alignment_local_files_only: bool = False,
        alignment_torch_dtype: str = "auto",
        model_kwargs: Mapping | None = None,
        torch_dtype: str | None = None,
        **kwargs,
    ):
        compute_type = _non_empty_string(
            compute_type,
            name="compute_type",
        ).lower()
        dtype_by_compute_type = {
            "auto": "auto",
            "default": "auto",
            "bfloat16": "bfloat16",
            "float16": "float16",
            "float32": "float32",
        }
        try:
            resolved_dtype = dtype_by_compute_type[compute_type]
        except KeyError as error:
            raise ValueError(
                "Native WhisperX compatibility supports compute_type "
                "'default', 'auto', 'float32', 'float16', or 'bfloat16'. "
                "CTranslate2 quantized modes belong to a VoiceHub "
                "optimization strategy.") from error
        if torch_dtype is not None:
            if not isinstance(torch_dtype, str):
                raise TypeError("`torch_dtype` must be a string or None.")
            normalized_dtype = torch_dtype.strip().lower()
            aliases = {
                "bf16": "bfloat16",
                "bfloat16": "bfloat16",
                "float": "float32",
                "float16": "float16",
                "float32": "float32",
                "fp16": "float16",
                "fp32": "float32",
                "half": "float16",
                "auto": "auto",
            }
            try:
                normalized_dtype = aliases[normalized_dtype]
            except KeyError as error:
                raise ValueError("`torch_dtype` must be auto, float32, float16, or "
                                 "bfloat16.") from error
            if (compute_type not in {"default", "auto"} and normalized_dtype != resolved_dtype):
                raise ValueError("`compute_type` and `torch_dtype` select different "
                                 "numeric dtypes.")
            resolved_dtype = normalized_dtype
        if not isinstance(align_output, bool):
            raise TypeError("`align_output` must be a boolean.")
        if alignment_model_path is not None:
            if not isinstance(alignment_model_path, (str, Path)):
                raise TypeError("`alignment_model_path` must be path-like or None.")
            alignment_model_path = str(alignment_model_path).strip()
            if not alignment_model_path:
                raise ValueError("`alignment_model_path` must be non-empty or None.")
        if alignment_revision is not None:
            if (not isinstance(alignment_revision, str) or not alignment_revision.strip()):
                raise ValueError("`alignment_revision` must be a non-empty string or None.")
            alignment_revision = alignment_revision.strip()
        if alignment_cache_dir is not None:
            if not isinstance(alignment_cache_dir, (str, Path)):
                raise TypeError("`alignment_cache_dir` must be path-like or None.")
            alignment_cache_dir = str(Path(alignment_cache_dir).expanduser())
        if not isinstance(alignment_local_files_only, bool):
            raise TypeError("`alignment_local_files_only` must be a boolean.")
        if not isinstance(alignment_torch_dtype, str):
            raise TypeError("`alignment_torch_dtype` must be a string.")
        if model_kwargs is not None and not isinstance(model_kwargs, Mapping):
            raise TypeError("`model_kwargs` must be a mapping or None.")
        model_kwargs = dict(model_kwargs or {})
        if model_kwargs:
            names = ", ".join(sorted(str(name) for name in model_kwargs))
            raise ValueError(
                "Native WhisperX does not delegate upstream `model_kwargs`; "
                f"unsupported option(s): {names}.")
        super().__init__(
            torch_dtype=resolved_dtype,
            **kwargs,
        )
        self.compute_type = compute_type
        self.align_output = align_output
        self.alignment_model_path = alignment_model_path
        self.alignment_revision = alignment_revision
        self.alignment_cache_dir = alignment_cache_dir
        self.alignment_local_files_only = alignment_local_files_only
        self.alignment_torch_dtype = alignment_torch_dtype
        self.model_kwargs = {}


class OpenAIWhisperConfig(WhisperASRConfig):
    """Compatibility configuration for the VoiceHub-native Whisper graph."""

    model_type = "asr_openai_whisper"

    def __init__(
        self,
        *,
        model_kwargs: Mapping | None = None,
        **kwargs,
    ):
        if model_kwargs is not None and not isinstance(model_kwargs, Mapping):
            raise TypeError("`model_kwargs` must be a mapping or None.")
        model_kwargs = dict(model_kwargs or {})
        if model_kwargs:
            names = ", ".join(sorted(model_kwargs))
            raise ValueError(
                "OpenAI Whisper now uses VoiceHub's owned graph; upstream "
                f"`model_kwargs` are not applicable: {names}.")
        super().__init__(**kwargs)
        self.model_kwargs = {}


class NeMoASRConfig(_NativeASRConfig):
    """Configure the VoiceHub-native QuartzNet/Jasper CTC runtime.

    ``model_class`` remains accepted for source compatibility. The
    native provider supports only ``ASRModel`` and ``EncDecCTCModel``
    because other NeMo classes can represent unrelated RNN-T, TDT,
    Conformer, or encoder-decoder graphs.
    """

    model_type = "asr_nemo"

    def __init__(
        self,
        *,
        model_class: str = "ASRModel",
        checkpoint_filename: str = "model.safetensors",
        torch_dtype: str = "auto",
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        model_kwargs: Mapping | None = None,
        sample_rate: int = 16_000,
        **kwargs,
    ):
        model_class = _non_empty_string(model_class, name="model_class")
        if model_class not in {"ASRModel", "EncDecCTCModel"}:
            raise ValueError(
                "Native NeMo ASR supports the verified character-CTC "
                "`ASRModel`/`EncDecCTCModel` boundary only.")
        checkpoint_filename = _non_empty_string(
            checkpoint_filename,
            name="checkpoint_filename",
        )
        if not checkpoint_filename.endswith(".safetensors"):
            raise ValueError("`checkpoint_filename` must identify a Safetensors file.")
        if not isinstance(torch_dtype, str) or not torch_dtype.strip():
            raise ValueError("`torch_dtype` must be a non-empty string.")
        dtype_aliases = {
            "auto": "auto",
            "bf16": "bfloat16",
            "bfloat16": "bfloat16",
            "float": "float32",
            "float16": "float16",
            "float32": "float32",
            "fp16": "float16",
            "fp32": "float32",
            "half": "float16",
        }
        try:
            torch_dtype = dtype_aliases[torch_dtype.strip().lower()]
        except KeyError as error:
            raise ValueError("`torch_dtype` must be auto, float32, float16, or bfloat16.") from error
        revision = _non_empty_string(
            revision,
            name="revision",
            allow_none=True,
        )
        if cache_dir is not None and not isinstance(cache_dir, (str, Path)):
            raise TypeError("`cache_dir` must be a string, Path, or None.")
        if not isinstance(local_files_only, bool):
            raise TypeError("`local_files_only` must be a boolean.")
        if model_kwargs is not None and not isinstance(model_kwargs, Mapping):
            raise TypeError("`model_kwargs` must be a mapping or None.")
        model_kwargs = dict(model_kwargs or {})
        if model_kwargs:
            managed = {
                "checkpoint_path",
                "map_location",
                "model_name",
                "restore_path",
            }
            collisions = sorted(set(model_kwargs) & managed)
            if collisions:
                formatted = ", ".join(f"`{name}`" for name in collisions)
                raise ValueError("`model_kwargs` cannot override managed option(s): "
                                 f"{formatted}.")
            names = ", ".join(sorted(str(name) for name in model_kwargs))
            raise ValueError(
                "Native NeMo ASR does not delegate arbitrary upstream "
                f"`model_kwargs`; unsupported option(s): {names}.")
        if sample_rate != 16_000:
            raise ValueError("The verified QuartzNet15x5 checkpoint requires 16 kHz audio.")
        super().__init__(
            model_class=model_class,
            checkpoint_filename=checkpoint_filename,
            torch_dtype=torch_dtype,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            model_kwargs={},
            sample_rate=sample_rate,
            _managed_model_options={
                "checkpoint_path",
                "map_location",
                "model_name",
                "restore_path",
            },
            **kwargs,
        )


class SpeechBrainASRConfig(_NativeASRConfig):
    """Configure VoiceHub's owned CRDNN, attention decoder, and RNNLM.

    The public checkpoint was originally distributed as three pickle
    state dictionaries and a SentencePiece model.  VoiceHub converts
    those files once behind an explicit trust boundary, then loads only
    a coherent native Safetensors artifact.  Former SpeechBrain and
    HyperPyYAML options remain in the signature so legacy configurations
    fail precisely instead of being forwarded to an external runtime.
    """

    model_type = "asr_speechbrain"
    architecture_family = "speech-seq2seq"

    def __init__(
        self,
        *,
        checkpoint_filename: str = "model.safetensors",
        tokenizer_filename: str = "tokenizer.model",
        torch_dtype: str = "float32",
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        training_max_duration_s: float = 60.0,
        training_uppercase_transcripts: bool = True,
        # Compatibility fields from the delegated SpeechBrain provider.
        hparams_file: str = "hyperparams.yaml",
        savedir: str | Path | None = None,
        overrides: Mapping | None = None,
        model_kwargs: Mapping | None = None,
        sample_rate: int = 16_000,
        **kwargs,
    ):
        if hparams_file != "hyperparams.yaml":
            raise ValueError(
                "Native SpeechBrain ASR does not execute HyperPyYAML; "
                "`hparams_file` must remain 'hyperparams.yaml'.")
        if overrides is not None and not isinstance(overrides, Mapping):
            raise TypeError("`overrides` must be a mapping or None.")
        if overrides:
            raise ValueError("Native SpeechBrain ASR does not execute HyperPyYAML "
                             "`overrides`.")
        if model_kwargs is not None and not isinstance(model_kwargs, Mapping):
            raise TypeError("`model_kwargs` must be a mapping or None.")
        if model_kwargs:
            names = ", ".join(sorted(str(name) for name in model_kwargs))
            raise ValueError(
                "Native SpeechBrain ASR does not delegate arbitrary loader "
                f"options; unsupported `model_kwargs`: {names}.")
        checkpoint_filename = _non_empty_string(
            checkpoint_filename,
            name="checkpoint_filename",
        )
        if checkpoint_filename != "model.safetensors":
            raise ValueError(
                "Native SpeechBrain ASR requires the coherent "
                "`model.safetensors` checkpoint.")
        tokenizer_filename = _non_empty_string(
            tokenizer_filename,
            name="tokenizer_filename",
        )
        if tokenizer_filename != "tokenizer.model":
            raise ValueError(
                "Native SpeechBrain ASR requires `tokenizer.model` beside "
                "the Safetensors checkpoint.")
        if not isinstance(torch_dtype, str):
            raise TypeError("`torch_dtype` must be a string.")
        dtype_aliases = {
            "auto": "float32",
            "float": "float32",
            "float32": "float32",
            "fp32": "float32",
        }
        try:
            torch_dtype = dtype_aliases[torch_dtype.strip().lower()]
        except KeyError as error:
            raise ValueError("The audited SpeechBrain CRDNN graph supports float32 only.") from error
        revision = _non_empty_string(
            revision,
            name="revision",
            allow_none=True,
        )
        if cache_dir is not None and not isinstance(cache_dir, (str, Path)):
            raise TypeError("`cache_dir` must be a string, Path, or None.")
        if savedir is not None and not isinstance(savedir, (str, Path)):
            raise TypeError("`savedir` must be a string, Path, or None.")
        if savedir is not None:
            normalized_savedir = Path(savedir).expanduser().as_posix()
            if (cache_dir is not None and Path(cache_dir).expanduser().as_posix() != normalized_savedir):
                raise ValueError(
                    "`savedir` is a deprecated alias for `cache_dir`; pass "
                    "only one cache location.")
            cache_dir = normalized_savedir
        if cache_dir is not None:
            cache_dir = Path(cache_dir).expanduser().as_posix()
        if not isinstance(local_files_only, bool):
            raise TypeError("`local_files_only` must be a boolean.")
        if sample_rate != 16_000:
            raise ValueError("The audited SpeechBrain CRDNN checkpoint requires 16 kHz "
                             "audio.")
        if (isinstance(training_max_duration_s, bool) or not isinstance(training_max_duration_s, Real) or
                not isfinite(training_max_duration_s) or training_max_duration_s <= 0):
            raise ValueError("`training_max_duration_s` must be finite and positive.")
        if not isinstance(training_uppercase_transcripts, bool):
            raise TypeError("`training_uppercase_transcripts` must be a boolean.")
        if not training_uppercase_transcripts:
            raise ValueError(
                "The published LibriSpeech tokenizer was trained on uppercase "
                "transcripts; disabling uppercase normalization would produce "
                "out-of-vocabulary text.")
        super().__init__(
            checkpoint_filename=checkpoint_filename,
            tokenizer_filename=tokenizer_filename,
            torch_dtype=torch_dtype,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            training_max_duration_s=float(training_max_duration_s),
            training_uppercase_transcripts=True,
            hparams_file="hyperparams.yaml",
            savedir=None if savedir is None else Path(savedir).expanduser().as_posix(),
            overrides={},
            model_kwargs={},
            sample_rate=sample_rate,
            _managed_model_options={
                "fetch_config",
                "hparams_file",
                "map_location",
                "overrides",
                "run_opts",
                "savedir",
                "source",
                "use_auth_token",
            },
            **kwargs,
        )


class FunASRConfig(_NativeASRConfig):
    model_type = "asr_funasr"

    def __init__(
        self,
        *,
        vad_model: str | None = None,
        punc_model: str | None = None,
        spk_model: str | None = None,
        generate_kwargs: Mapping | None = None,
        **kwargs,
    ):
        vad_model = _non_empty_string(
            vad_model,
            name="vad_model",
            allow_none=True,
        )
        punc_model = _non_empty_string(
            punc_model,
            name="punc_model",
            allow_none=True,
        )
        spk_model = _non_empty_string(
            spk_model,
            name="spk_model",
            allow_none=True,
        )
        if generate_kwargs is not None and not isinstance(
                generate_kwargs,
                Mapping,
        ):
            raise TypeError("`generate_kwargs` must be a mapping or None.")
        generate_kwargs = dict(generate_kwargs or {})
        if _secret_keys(generate_kwargs):
            raise ValueError("`generate_kwargs` cannot contain authentication credentials.")
        collisions = sorted(set(generate_kwargs) & {
            "hotword",
            "input",
            "language",
        })
        if collisions:
            raise ValueError(
                "`generate_kwargs` cannot override VoiceHub-managed option(s): "
                f"{', '.join(collisions)}.")
        super().__init__(
            vad_model=vad_model,
            punc_model=punc_model,
            spk_model=spk_model,
            generate_kwargs=generate_kwargs,
            _managed_model_options={
                "device",
                "model",
                "punc_model",
                "spk_model",
                "vad_model",
            },
            **kwargs,
        )


class ESPnetASRConfig(_NativeASRConfig):
    """Configure the exact native LibriSpeech Transformer e18 runtime."""

    model_type = "asr_espnet"
    architecture_family = "speech-seq2seq"

    def __init__(
        self,
        *,
        beam_size: int = 10,
        ctc_weight: float = 0.3,
        language_model_weight: float = 0.6,
        checkpoint_filename: str = "model.safetensors",
        language_model_filename: str = "language_model.safetensors",
        tokenizer_filename: str = "tokenizer.model",
        tokens_filename: str = "tokens.txt",
        torch_dtype: str = "float32",
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        training_max_duration_s: float = 60.0,
        model_kwargs: Mapping | None = None,
        sample_rate: int = 16_000,
        **kwargs,
    ):
        if (isinstance(beam_size, bool) or not isinstance(beam_size, Integral) or beam_size <= 0):
            raise ValueError("`beam_size` must be a positive integer.")
        if (isinstance(ctc_weight, bool) or not isinstance(ctc_weight, Real) or not isfinite(ctc_weight) or
                not 0.0 <= ctc_weight <= 1.0):
            raise ValueError("`ctc_weight` must be finite and between 0 and 1.")
        if (isinstance(language_model_weight, bool) or not isinstance(language_model_weight, Real) or
                not isfinite(language_model_weight) or not 0.0 <= language_model_weight <= 1.0):
            raise ValueError("`language_model_weight` must be finite and between 0 and 1.")
        for name, value, expected in (
            ("checkpoint_filename", checkpoint_filename, "model.safetensors"),
            (
                "language_model_filename",
                language_model_filename,
                "language_model.safetensors",
            ),
            ("tokenizer_filename", tokenizer_filename, "tokenizer.model"),
            ("tokens_filename", tokens_filename, "tokens.txt"),
        ):
            value = _non_empty_string(value, name=name)
            if value != expected:
                raise ValueError(f"Native ESPnet requires `{name}={expected!r}`.")
        if not isinstance(torch_dtype, str):
            raise TypeError("`torch_dtype` must be a string.")
        if torch_dtype.strip().lower() not in {
                "auto",
                "float",
                "float32",
                "fp32",
        }:
            raise ValueError("The audited ESPnet Transformer checkpoint supports float32 only.")
        revision = _non_empty_string(
            revision,
            name="revision",
            allow_none=True,
        )
        if cache_dir is not None and not isinstance(cache_dir, (str, Path)):
            raise TypeError("`cache_dir` must be a string, Path, or None.")
        if cache_dir is not None:
            cache_dir = str(Path(cache_dir).expanduser())
        if not isinstance(local_files_only, bool):
            raise TypeError("`local_files_only` must be a boolean.")
        if (isinstance(training_max_duration_s, bool) or not isinstance(training_max_duration_s, Real) or
                not isfinite(training_max_duration_s) or training_max_duration_s <= 0):
            raise ValueError("`training_max_duration_s` must be finite and positive.")
        if model_kwargs is not None and not isinstance(model_kwargs, Mapping):
            raise TypeError("`model_kwargs` must be a mapping or None.")
        if model_kwargs:
            names = ", ".join(sorted(str(name) for name in model_kwargs))
            raise ValueError(
                "Native ESPnet does not delegate arbitrary loader options; "
                f"unsupported `model_kwargs`: {names}.")
        if sample_rate != 16_000:
            raise ValueError("The audited ESPnet LibriSpeech checkpoint requires 16 kHz audio.")
        super().__init__(
            beam_size=int(beam_size),
            ctc_weight=float(ctc_weight),
            language_model_weight=float(language_model_weight),
            checkpoint_filename="model.safetensors",
            language_model_filename="language_model.safetensors",
            tokenizer_filename="tokenizer.model",
            tokens_filename="tokens.txt",
            torch_dtype="float32",
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            training_max_duration_s=float(training_max_duration_s),
            model_kwargs={},
            sample_rate=sample_rate,
            _managed_model_options={
                "beam_size",
                "ctc_weight",
                "device",
                "language_model_weight",
                "model_tag",
            },
            **kwargs,
        )


class WeNetASRConfig(_NativeASRConfig):
    """Configure the exact native GigaSpeech U2++ Conformer runtime."""

    model_type = "asr_wenet"

    def __init__(
        self,
        *,
        language: str | None = None,
        checkpoint_filename: str = "model.safetensors",
        tokenizer_filename: str = "tokenizer.model",
        units_filename: str = "units.txt",
        decoding_strategy: str = "attention_rescoring",
        beam_size: int = 5,
        ctc_weight: float = 0.3,
        reverse_weight: float = 0.3,
        torch_dtype: str = "float32",
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        model_kwargs: Mapping | None = None,
        sample_rate: int = 16_000,
        **kwargs,
    ):
        language = _non_empty_string(
            language,
            name="language",
            allow_none=True,
        )
        for name, value, suffix in (
            ("checkpoint_filename", checkpoint_filename, ".safetensors"),
            ("tokenizer_filename", tokenizer_filename, ".model"),
            ("units_filename", units_filename, ".txt"),
        ):
            value = _non_empty_string(value, name=name)
            if not value.endswith(suffix):
                raise ValueError(f"`{name}` must end with {suffix!r}.")
        checkpoint_filename = checkpoint_filename.strip()
        tokenizer_filename = tokenizer_filename.strip()
        units_filename = units_filename.strip()
        decoding_strategy = _non_empty_string(
            decoding_strategy,
            name="decoding_strategy",
        ).lower()
        if decoding_strategy not in {
                "attention_rescoring",
                "ctc_prefix_beam_search",
                "ctc_greedy_search",
        }:
            raise ValueError(
                "`decoding_strategy` must be attention_rescoring, "
                "ctc_prefix_beam_search, or ctc_greedy_search.")
        if (isinstance(beam_size, bool) or not isinstance(beam_size, Integral) or beam_size <= 0):
            raise ValueError("`beam_size` must be a positive integer.")
        for name, value in (
            ("ctc_weight", ctc_weight),
            ("reverse_weight", reverse_weight),
        ):
            if (isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value) or
                    not 0.0 <= float(value) <= 1.0):
                raise ValueError(f"`{name}` must be finite and in [0, 1].")
        if not isinstance(torch_dtype, str):
            raise TypeError("`torch_dtype` must be a string.")
        aliases = {
            "auto": "auto",
            "bf16": "bfloat16",
            "bfloat16": "bfloat16",
            "float16": "float16",
            "float32": "float32",
            "fp16": "float16",
            "fp32": "float32",
        }
        try:
            torch_dtype = aliases[torch_dtype.strip().lower()]
        except KeyError as error:
            raise ValueError("`torch_dtype` must be auto, float32, float16, or bfloat16.") from error
        revision = _non_empty_string(
            revision,
            name="revision",
            allow_none=True,
        )
        if cache_dir is not None and not isinstance(cache_dir, (str, Path)):
            raise TypeError("`cache_dir` must be path-like or None.")
        if not isinstance(local_files_only, bool):
            raise TypeError("`local_files_only` must be a boolean.")
        if model_kwargs is not None and not isinstance(model_kwargs, Mapping):
            raise TypeError("`model_kwargs` must be a mapping or None.")
        model_kwargs = dict(model_kwargs or {})
        if model_kwargs:
            names = ", ".join(sorted(str(name) for name in model_kwargs))
            raise ValueError(
                "Native WeNet does not delegate upstream `model_kwargs`; "
                f"unsupported option(s): {names}.")
        if sample_rate != 16_000:
            raise ValueError("The audited GigaSpeech U2++ checkpoint requires 16 kHz audio.")
        super().__init__(
            language=language,
            checkpoint_filename=checkpoint_filename,
            tokenizer_filename=tokenizer_filename,
            units_filename=units_filename,
            decoding_strategy=decoding_strategy,
            beam_size=int(beam_size),
            ctc_weight=float(ctc_weight),
            reverse_weight=float(reverse_weight),
            torch_dtype=torch_dtype,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            model_kwargs={},
            sample_rate=sample_rate,
            _managed_model_options={
                "beam",
                "context_path",
                "device",
                "gpu",
                "model_dir",
                "resample_rate",
            },
            **kwargs,
        )
