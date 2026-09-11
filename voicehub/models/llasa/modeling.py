"""VoiceHub-native LLaSA inference with a frozen XCodec2 audio codec."""

from __future__ import annotations

import math
import secrets
from contextlib import nullcontext
from numbers import Real
from pathlib import Path
from typing import Any

from voicehub.dependencies import import_optional
from voicehub.models._shared import finish_audio_output, resolve_torch_dtype, seeded_inference, validate_seed
from voicehub.models.llasa.configuration import LlasaConfig
from voicehub.models.llasa.tokenization_llasa import (
    EOT_TOKEN,
    LLASA_SPEECH_CODEBOOK_SIZE,
    LLASA_SPEECH_TOKEN_OFFSET,
    SPEECH_GENERATION_END,
    SPEECH_GENERATION_START,
    TEXT_UNDERSTANDING_END,
    TEXT_UNDERSTANDING_START,
)
from voicehub.models.tts import PreTrainedTTSModel
from voicehub.outputs import TTSOutput
from voicehub.processing.waveform import load_native_audio


class LlasaForTextToSpeech(PreTrainedTTSModel):
    """Multilingual LLaSA synthesis and reference-conditioned cloning.

    The language model, tokenizer, XCodec2 semantic/acoustic encoders,
    quantizer, and waveform decoder are all VoiceHub-owned PyTorch
    runtime components. Hub repositories provide declarative assets and
    Safetensors weights only.
    """

    config_class = LlasaConfig
    default_model_name_or_path = "HKUSTAudio/Llasa-1B-Multilingual"
    _TEXT_START = TEXT_UNDERSTANDING_START
    _TEXT_END = TEXT_UNDERSTANDING_END
    _SPEECH_START = SPEECH_GENERATION_START
    _SPEECH_END = SPEECH_GENERATION_END
    _SPEECH_TOKEN_OFFSET = LLASA_SPEECH_TOKEN_OFFSET
    _SPEECH_CODEBOOK_SIZE = LLASA_SPEECH_CODEBOOK_SIZE

    def __init__(
        self,
        config: LlasaConfig | str | None = None,
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
        self._hub_token = token
        self.artifacts: Any | None = None
        self.codec_artifacts: Any | None = None
        self.native_config: Any | None = None
        self.tokenizer = None
        self.codec = None
        self._torch = None
        super().__init__(config, device=device, lazy_load=lazy_load)

    def _language_model_dtype(self, torch, architecture_values: dict[str, Any]):
        if self.config.torch_dtype != "auto":
            return resolve_torch_dtype(
                torch,
                self.config.torch_dtype,
                self.device,
            )
        device_type = torch.device(self.device).type
        if device_type == "cuda":
            declared = str(architecture_values.get("torch_dtype", "bfloat16")).lower()
            if declared in {"bfloat16", "bf16"} and torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        if device_type == "mps":
            return torch.float16
        return torch.float32

    def _load_pretrained_model(self) -> None:
        from voicehub.checkpointing import SafeTensorReader
        from voicehub.hub import read_json_file
        from voicehub.models.causal_lm.native.configuration import CausalLMConfig
        from voicehub.models.llasa.artifacts import resolve_llasa_artifacts, resolve_xcodec2_artifacts
        from voicehub.models.llasa.checkpoint import XCodec2CheckpointAdapter
        from voicehub.models.llasa.tokenization_llasa import LlasaTokenizer
        from voicehub.models.llasa.xcodec2 import XCodec2Config, XCodec2Model

        torch = import_optional(
            "torch",
            model_type="llasa",
            install_extra=None,
        )
        source = self.config.name_or_path or self.default_model_name_or_path
        artifacts = resolve_llasa_artifacts(
            source,
            checkpoint_filename=self.config.checkpoint_filename,
            revision=self.config.revision,
            cache_dir=self.config.cache_dir,
            token=self._hub_token,
            local_files_only=self.config.local_files_only,
            include_checkpoint=not self.uses_llm_token_backend,
        )
        architecture_values = read_json_file(artifacts.config)
        native_config = CausalLMConfig.from_dict(architecture_values)
        if native_config.model_type != "llama":
            raise ValueError(
                "LLaSA requires a dense Llama causal-LM checkpoint; "
                f"received {native_config.model_type!r}.")
        architectures = architecture_values.get('architectures', ())
        if isinstance(architectures, str):
            architectures = (architectures, )
        if architectures and "LlamaForCausalLM" not in architectures:
            raise ValueError("LLaSA checkpoint does not declare LlamaForCausalLM.")
        tokenizer = LlasaTokenizer.from_tokenizer_json(
            artifacts.tokenizer,
            tokenizer_config_path=artifacts.tokenizer_config,
        )
        if tokenizer.token_id_space_size != native_config.vocab_size:
            raise ValueError(
                "LLaSA tokenizer/model vocabulary mismatch: tokenizer ID "
                f"space ends at {tokenizer.token_id_space_size}, model expects "
                f"{native_config.vocab_size}.")

        if self.uses_llm_token_backend:
            model = self._create_remote_causal_lm_proxy()
        else:
            from voicehub.models.causal_lm.native.checkpoint import (
                HuggingFaceCausalLMCheckpointAdapter,
                open_causal_lm_tensor_source,
            )
            from voicehub.models.causal_lm.native.modeling import LlamaForCausalLM

            if artifacts.checkpoint is None:
                raise RuntimeError("Native LLaSA resolution returned no checkpoint.")
            dtype = self._language_model_dtype(torch, architecture_values)
            model = LlamaForCausalLM(
                native_config,
                initialize=False,
                device=self.device,
                dtype=dtype,
            )
            with open_causal_lm_tensor_source(artifacts.checkpoint) as reader:
                HuggingFaceCausalLMCheckpointAdapter().load_streaming(
                    model,
                    reader,
                    architecture_values,
                    strict=True,
                )
            if native_config.tie_word_embeddings:
                model.tie_weights()

        local_codec = artifacts.root / "xcodec2"
        codec_source = (str(local_codec) if local_codec.is_dir() else self.config.codec_name_or_path)
        codec_artifacts = resolve_xcodec2_artifacts(
            codec_source,
            checkpoint_filename=self.config.codec_checkpoint_filename,
            revision=self.config.codec_revision,
            cache_dir=self.config.cache_dir,
            token=self._hub_token,
            local_files_only=self.config.local_files_only,
        )
        codec_values = read_json_file(codec_artifacts.config)
        codec_config = XCodec2Config.from_dict(codec_values)
        if codec_config.sampling_rate != self.config.sample_rate:
            raise ValueError(
                "LLaSA and XCodec2 sample rates disagree: "
                f"{self.config.sample_rate} != {codec_config.sampling_rate}.")
        with torch.device(self.device):
            codec = XCodec2Model(codec_config, initialize=False)
        if codec_artifacts.preprocessor_config is not None:
            codec.feature_extractor.validate_preprocessor_config(
                read_json_file(codec_artifacts.preprocessor_config))
        codec._reset_derived_buffers()
        with SafeTensorReader(codec_artifacts.checkpoint) as reader:
            XCodec2CheckpointAdapter.for_model(codec).load_streaming(
                codec,
                reader,
                codec_values,
                strict=True,
            )

        self.artifacts = artifacts
        self.codec_artifacts = codec_artifacts
        self.native_config = native_config
        self.tokenizer = tokenizer
        self.model = model.eval()
        self.codec = codec.eval()
        self._torch = torch

    def _prepare_for_inference(self) -> None:
        """Restore serving mode without replacing fine-tuned parameters."""
        if self.model is not None and hasattr(self.model, "eval"):
            self.model.eval()
        if self.codec is not None and hasattr(self.codec, "eval"):
            self.codec.eval()
        model_config = getattr(self.model, "config", None)
        if model_config is not None and hasattr(model_config, "use_cache"):
            try:
                model_config.use_cache = True
            except (AttributeError, TypeError):
                # Native causal-LM configs are immutable and already preserve
                # their checkpoint-declared cache setting.
                pass

    def _validate_generation_inputs(self, model_inputs: dict) -> None:
        speaker_audio_path = model_inputs.get("speaker_audio_path")
        reference_text = model_inputs.get("reference_text", "")
        if speaker_audio_path is not None and (not isinstance(speaker_audio_path, (str, Path)) or
                                               not str(speaker_audio_path).strip()):
            raise ValueError("`speaker_audio_path` must be a non-empty local path or None.")
        if not isinstance(reference_text, str):
            raise TypeError("`reference_text` must be a string.")
        if (speaker_audio_path is not None) != bool(reference_text.strip()):
            raise ValueError(
                "LLaSA voice cloning requires `speaker_audio_path` and a "
                "non-empty `reference_text` together.")
        if speaker_audio_path is not None:
            reference_path = Path(speaker_audio_path).expanduser()
            if not reference_path.is_file():
                raise FileNotFoundError(f"LLaSA reference audio was not found: {reference_path}.")
        max_new_tokens = model_inputs.get("max_new_tokens")
        if max_new_tokens is None:
            max_new_tokens = self.config.max_new_tokens
        if (isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens <= 0):
            raise ValueError("`max_new_tokens` must be a positive integer.")
        temperature = model_inputs.get("temperature")
        if temperature is None:
            temperature = self.config.temperature
        if (isinstance(temperature, bool) or not isinstance(temperature, Real) or
                not math.isfinite(temperature) or temperature <= 0):
            raise ValueError("LLaSA sampling requires `temperature` greater than zero.")
        top_p = model_inputs.get("top_p")
        if top_p is None:
            top_p = self.config.top_p
        if (isinstance(top_p, bool) or not isinstance(top_p, Real) or not math.isfinite(top_p) or
                not 0 < top_p <= 1):
            raise ValueError("LLaSA sampling requires `top_p` in the interval (0, 1].")

    @staticmethod
    def _ids_to_speech_tokens(speech_ids) -> list[str]:
        return [f"<|s_{int(speech_id)}|>" for speech_id in speech_ids]

    @classmethod
    def _extract_speech_ids(cls, tokens: list[str]) -> list[int]:
        speech_ids = []
        for token in tokens:
            if token.startswith("<|s_") and token.endswith("|>"):
                value = token[4:-2]
                try:
                    speech_id = int(value)
                except ValueError as error:
                    raise RuntimeError("LLaSA generated a malformed speech token: "
                                       f"{token!r}.") from error
                if not 0 <= speech_id < cls._SPEECH_CODEBOOK_SIZE:
                    raise RuntimeError("LLaSA generated an out-of-range speech token: "
                                       f"{token!r}.")
                speech_ids.append(speech_id)
        if not speech_ids:
            raise RuntimeError("LLaSA generated no XCodec2 speech tokens.")
        return speech_ids

    def _load_reference(self, audio_path: str):
        reference = load_native_audio(
            audio_path,
            target_sampling_rate=self.sample_rate,
        )
        if reference.waveform.numel() == 0:
            raise ValueError("LLaSA reference audio contains no samples.")
        return reference.waveform.unsqueeze(0)

    def _encode_reference(self, audio_path: str) -> tuple[list[int], int]:
        reference = self._load_reference(audio_path)
        codec_device = next(self.codec.parameters()).device
        with self._torch.inference_mode():
            encoded = self.codec.encode_code(
                input_waveform=reference.to(codec_device),
                sample_rate=self.sample_rate,
            )
        prefix_ids = [int(value) for value in encoded[0, 0].detach().cpu().tolist()]
        return prefix_ids, len(prefix_ids) * int(self.codec.hop_length)

    def _build_generation_prompt(
        self,
        text: str,
        *,
        reference_text: str,
        prefix_ids: list[int],
    ):
        formatted_text = (self._TEXT_START + reference_text + text + self._TEXT_END)
        prefix = "".join(self._ids_to_speech_tokens(prefix_ids))
        conversation = [
            {
                "role": "user",
                "content": "Convert the text to speech:" + formatted_text,
            },
            {
                "role": "assistant",
                "content": self._SPEECH_START + prefix,
            },
        ]
        return self.tokenizer.apply_chat_template(
            conversation,
            tokenize=True,
            return_tensors="pt",
            continue_final_message=True,
        ).to(self.device)

    def _generate(
        self,
        text: str,
        *,
        speaker_audio_path: str | None = None,
        reference_text: str = "",
        output_file: str | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        seed: int | None = None,
    ) -> TTSOutput:
        """Synthesize text, optionally from an aligned reference utterance."""
        from voicehub.generation import GenerationConfig

        requested_tokens = (self.config.max_new_tokens if max_new_tokens is None else max_new_tokens)
        sampling_temperature = (self.config.temperature if temperature is None else float(temperature))
        nucleus_probability = (self.config.top_p if top_p is None else float(top_p))
        seed_context = (
            nullcontext(validate_seed(seed) if seed is not None else secrets.randbits(63))
            if self.uses_llm_token_backend else seeded_inference(
                seed,
                device=self.device,
                model_type="llasa",
            ))
        with seed_context as effective_seed:
            prefix_ids: list[int] = []
            prompt_samples = 0
            if speaker_audio_path:
                prefix_ids, prompt_samples = self._encode_reference(speaker_audio_path)
            input_ids = self._build_generation_prompt(
                text,
                reference_text=reference_text,
                prefix_ids=prefix_ids,
            )
            available = self.config.max_total_tokens - input_ids.shape[1]
            if available <= 0:
                raise ValueError(
                    "The LLaSA prompt reaches `max_total_tokens`; shorten the "
                    "text/reference or raise the explicit limit.")
            effective_tokens = min(requested_tokens, available)
            attention_mask = self._torch.ones_like(
                input_ids,
                dtype=self._torch.bool,
            )
            speech_end_id = self.tokenizer.convert_tokens_to_ids(self._SPEECH_END)
            with self._torch.inference_mode():
                generation = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=GenerationConfig(
                        max_new_tokens=effective_tokens,
                        do_sample=True,
                        temperature=sampling_temperature,
                        top_p=nucleus_probability,
                        eos_token_id=speech_end_id,
                        pad_token_id=self.tokenizer.convert_tokens_to_ids(EOT_TOKEN),
                        seed=effective_seed,
                        use_cache=True,
                    ),
                )
                generated_ids = generation.sequences[
                    0,
                    input_ids.shape[1]:,
                ]
                token_strings = self.tokenizer.convert_ids_to_tokens(generated_ids.detach().cpu().tolist())
                speech_ids = prefix_ids + self._extract_speech_ids(token_strings)
                codec_device = next(self.codec.parameters()).device
                codec_tokens = self._torch.tensor(
                    speech_ids,
                    dtype=self._torch.long,
                    device=codec_device,
                ).view(1, 1, -1)
                waveform = self.codec.decode_code(codec_tokens)[0, 0]
            if prompt_samples:
                waveform = waveform[prompt_samples:]

        waveform = waveform.detach().cpu()
        return finish_audio_output(
            waveform,
            self.sample_rate,
            output_file=output_file,
            metadata={
                "model_type": self.config.model_type,
                "backend": self.llm_backend.value,
                "engine_transport": ("tokens" if self.uses_llm_token_backend else "native"),
                "seed": effective_seed,
                "requested_seed": seed,
                "voice_cloned": bool(speaker_audio_path),
                "audio_tokens": len(speech_ids) - len(prefix_ids),
                "prompt_audio_tokens": len(prefix_ids),
                "max_new_tokens": effective_tokens,
            },
        )

    def _save_pretrained(self, save_directory: Path) -> None:
        """Export a self-contained LM/tokenizer/XCodec2 Safetensors runtime."""
        if self.model is None or self.codec is None or self.tokenizer is None:
            self.load()
        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)
        self.codec.save_pretrained(save_directory / "xcodec2")


LlasaTTS = LlasaForTextToSpeech

__all__ = [
    "LlasaConfig",
    "LlasaForTextToSpeech",
    "LlasaTTS",
]
