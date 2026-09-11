"""VoiceHub-native Orpheus causal-LM and SNAC inference."""

from __future__ import annotations

import math
import secrets
from contextlib import nullcontext
from numbers import Real
from pathlib import Path
from typing import Any

from voicehub.dependencies import import_optional
from voicehub.models._shared import finish_audio_output, resolve_torch_dtype, seeded_inference, validate_seed
from voicehub.models.orpheustts.configuration import OrpheusTTSConfig
from voicehub.models.orpheustts.protocol import (
    AUDIO_TOKEN_OFFSET,
    END_HUMAN_TOKEN_ID,
    END_SPEECH_TOKEN_ID,
    END_TEXT_TOKEN_ID,
    SNAC_CODEBOOK_SIZE,
    SNAC_FRAME_WIDTH,
    START_AI_TOKEN_ID,
    START_HUMAN_TOKEN_ID,
    START_SPEECH_TOKEN_ID,
    deinterleave_snac_codes,
    normalize_orpheus_audio_tokens,
)
from voicehub.models.tts import PreTrainedTTSModel
from voicehub.outputs import TTSOutput


class OrpheusTTSForTextToSpeech(PreTrainedTTSModel):
    """Expressive speech generation with VoiceHub-owned runtime graphs."""

    config_class = OrpheusTTSConfig
    default_model_name_or_path = "canopylabs/orpheus-3b-0.1-ft"
    _START_HUMAN_TOKEN_ID = START_HUMAN_TOKEN_ID
    _END_TEXT_TOKEN_ID = END_TEXT_TOKEN_ID
    _END_HUMAN_TOKEN_ID = END_HUMAN_TOKEN_ID
    _START_AI_TOKEN_ID = START_AI_TOKEN_ID
    _START_SPEECH_TOKEN_ID = START_SPEECH_TOKEN_ID
    _END_SPEECH_TOKEN_ID = END_SPEECH_TOKEN_ID
    _AUDIO_TOKEN_OFFSET = AUDIO_TOKEN_OFFSET
    _SNAC_CODEBOOK_SIZE = SNAC_CODEBOOK_SIZE
    _SNAC_FRAME_WIDTH = SNAC_FRAME_WIDTH

    def __init__(
        self,
        config: OrpheusTTSConfig | str | None = None,
        *,
        model_path: str | None = None,
        device: str = "auto",
        lazy_load: bool = True,
        token: str | bool | None = None,
        **config_overrides,
    ):
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

    def _load_pretrained_model(self) -> None:
        from voicehub.hub import read_json_file
        from voicehub.models.causal_lm.native.configuration import CausalLMConfig
        from voicehub.models.orpheustts.artifacts import resolve_orpheus_artifacts
        from voicehub.models.orpheustts.source.snac import SNAC
        from voicehub.models.orpheustts.tokenization_orpheustts import OrpheusTokenizer

        torch = import_optional(
            "torch",
            model_type="orpheustts",
            install_extra=None,
        )
        if self.config.torch_dtype == "auto":
            device_type = torch.device(self.device).type
            dtype = (torch.float16 if device_type in {"cuda", "mps"} else torch.float32)
        else:
            dtype = resolve_torch_dtype(
                torch,
                self.config.torch_dtype,
                self.device,
            )
        source = self.config.name_or_path or self.default_model_name_or_path
        artifacts = resolve_orpheus_artifacts(
            source,
            checkpoint_filename=self.config.checkpoint_filename,
            tokenizer_filename=self.config.tokenizer_filename,
            cache_dir=self.config.cache_dir,
            revision=self.config.revision,
            token=self._hub_token,
            local_files_only=self.config.local_files_only,
            include_checkpoint=not self.uses_llm_token_backend,
        )
        architecture_values = read_json_file(artifacts.config)
        native_config = CausalLMConfig.from_dict(architecture_values)
        if native_config.model_type != "llama":
            raise ValueError(
                "Orpheus requires a dense Llama causal-LM checkpoint; "
                f"received {native_config.model_type!r}.")
        architectures = architecture_values.get('architectures', ())
        if isinstance(architectures, str):
            architectures = (architectures, )
        if architectures and "LlamaForCausalLM" not in architectures:
            raise ValueError("Orpheus checkpoint does not declare LlamaForCausalLM.")
        tokenizer = OrpheusTokenizer.from_tokenizer_json(
            artifacts.tokenizer,
            tokenizer_config_path=artifacts.tokenizer_config,
        )
        if tokenizer.token_id_space_size != native_config.vocab_size:
            raise ValueError(
                "Orpheus tokenizer/model vocabulary mismatch: tokenizer ID "
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
                raise RuntimeError("Native Orpheus resolution returned no checkpoint.")
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

        local_codec = artifacts.root / "snac"
        codec_source = (str(local_codec) if local_codec.is_dir() else self.config.codec_name_or_path)
        codec = SNAC.from_pretrained(
            codec_source,
            checkpoint_filename=self.config.codec_checkpoint_filename,
            cache_dir=self.config.cache_dir,
            revision=self.config.codec_revision,
            token=self._hub_token,
            local_files_only=self.config.local_files_only,
        ).eval().to("cpu")
        sample_rate = int(getattr(codec, "sampling_rate", 0))
        if sample_rate <= 0:
            raise RuntimeError("Orpheus SNAC codec reported an invalid sample rate: "
                               f"{sample_rate}.")
        self.config.sample_rate = sample_rate
        self.artifacts = artifacts
        self.codec_artifacts = getattr(codec, "_voicehub_artifacts", None)
        self.native_config = native_config
        self.codec = codec
        self.model = model.eval()
        self.tokenizer = tokenizer
        self._torch = torch

    def _prepare_for_inference(self) -> None:
        """Restore deterministic serving state without replacing trained
        weights."""
        if self.model is not None and hasattr(self.model, "eval"):
            self.model.eval()
        if self.codec is not None and hasattr(self.codec, "eval"):
            self.codec.eval()
        model_config = getattr(self.model, "config", None)
        if model_config is not None and hasattr(model_config, "use_cache"):
            try:
                model_config.use_cache = True
            except (AttributeError, TypeError):
                # Native causal-LM configs are immutable and already retain
                # their checkpoint-declared cache policy.
                pass

    def _validate_generation_inputs(self, model_inputs: dict) -> None:
        voice = model_inputs.get("voice")
        if not isinstance(voice, str) or not voice.strip():
            raise ValueError("Orpheus generation requires a non-empty `voice`.")
        temperature = model_inputs.get("temperature", 0.6)
        if (isinstance(temperature, bool) or not isinstance(temperature, Real) or
                not math.isfinite(temperature) or temperature <= 0):
            raise ValueError("Orpheus sampling requires `temperature` greater than zero.")
        top_p = model_inputs.get("top_p", 0.8)
        if (isinstance(top_p, bool) or not isinstance(top_p, Real) or not math.isfinite(top_p) or
                not 0 < top_p <= 1):
            raise ValueError("Orpheus sampling requires `top_p` in the interval (0, 1].")
        repetition_penalty = model_inputs.get("repetition_penalty", 1.3)
        if (isinstance(repetition_penalty, bool) or not isinstance(repetition_penalty, Real) or
                not math.isfinite(repetition_penalty) or repetition_penalty <= 0):
            raise ValueError("`repetition_penalty` must be a finite positive number.")
        max_new_tokens = model_inputs.get("max_new_tokens", 1200)
        if (isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens <= 0):
            raise ValueError("`max_new_tokens` must be a positive integer.")

    def _prepare_inputs(self, text: str, voice: str):
        torch = self._torch
        text_ids = self.tokenizer.encode(
            f"{voice}: {text}",
            add_special_tokens=True,
        )
        formatted = torch.tensor(
            [[
                self._START_HUMAN_TOKEN_ID,
                *text_ids.input_ids,
                self._END_TEXT_TOKEN_ID,
                self._END_HUMAN_TOKEN_ID,
                self._START_AI_TOKEN_ID,
                self._START_SPEECH_TOKEN_ID,
            ]],
            dtype=torch.int64,
            device=self.device,
        )
        return (
            formatted,
            torch.ones_like(formatted),
        )

    @classmethod
    def _extract_audio_codes(cls, generated_ids) -> list[int]:
        """Extract one complete Orpheus speech span from generated tokens."""
        tokens = [int(token) for token in generated_ids[0].detach().cpu().tolist()]
        try:
            speech_start = len(tokens) - 1 - tokens[::-1].index(cls._START_SPEECH_TOKEN_ID)
        except ValueError as exc:
            raise RuntimeError(
                "Orpheus returned no speech-start token; generation did not "
                "produce a decodable audio span.") from exc

        speech_tokens = tokens[speech_start + 1:]
        if cls._END_SPEECH_TOKEN_ID in speech_tokens:
            speech_tokens = speech_tokens[:speech_tokens.index(cls._END_SPEECH_TOKEN_ID)]
        audio_tokens = [token for token in speech_tokens if token >= cls._AUDIO_TOKEN_OFFSET]
        complete_length = (len(audio_tokens) - len(audio_tokens) % cls._SNAC_FRAME_WIDTH)
        if complete_length == 0:
            raise RuntimeError("Orpheus returned no complete SNAC frames.")
        audio_tokens = audio_tokens[:complete_length]
        try:
            return normalize_orpheus_audio_tokens(audio_tokens)
        except (TypeError, ValueError) as error:
            raise RuntimeError(str(error)) from error

    def _decode_codes(self, codes: list[int]):
        torch = self._torch
        try:
            layer_1, layer_2, layer_3 = deinterleave_snac_codes(codes)
        except (TypeError, ValueError) as error:
            raise RuntimeError(str(error)) from error
        codec_device = next(self.codec.parameters()).device
        return self.codec.decode([
            torch.tensor(layer_1, dtype=torch.long, device=codec_device).unsqueeze(0),
            torch.tensor(layer_2, dtype=torch.long, device=codec_device).unsqueeze(0),
            torch.tensor(layer_3, dtype=torch.long, device=codec_device).unsqueeze(0),
        ])

    def _generate(
        self,
        text: str,
        *,
        voice: str,
        output_file: str | None = None,
        max_new_tokens: int = 1200,
        temperature: float = 0.6,
        top_p: float = 0.8,
        repetition_penalty: float = 1.3,
        seed: int | None = None,
    ) -> TTSOutput:
        input_ids, attention_mask = self._prepare_inputs(text, voice)
        from voicehub.generation import GenerationConfig

        seed_context = (
            nullcontext(validate_seed(seed) if seed is not None else secrets.randbits(63))
            if self.uses_llm_token_backend else seeded_inference(
                seed,
                device=self.device,
                model_type="orpheustts",
            ))
        with seed_context as effective_seed:
            with self._torch.inference_mode():
                generation = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=GenerationConfig(
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=temperature,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        eos_token_id=self._END_SPEECH_TOKEN_ID,
                        pad_token_id=(
                            self.native_config.pad_token_id
                            if self.native_config.pad_token_id is not None else self._END_SPEECH_TOKEN_ID),
                        seed=effective_seed,
                        use_cache=True,
                    ),
                )
                codes = self._extract_audio_codes(generation.sequences)
                audio = self._decode_codes(codes)
        return finish_audio_output(
            audio,
            self.sample_rate,
            output_file=output_file,
            metadata={
                "voice": voice,
                "audio_tokens": len(codes),
                "backend": self.llm_backend.value,
                "engine_transport": ("tokens" if self.uses_llm_token_backend else "native"),
                "seed": effective_seed,
                "requested_seed": seed,
            },
        )

    def _save_pretrained(self, save_directory: Path) -> None:
        """Export a self-contained native LM/tokenizer/SNAC artifact."""
        if self.model is None or self.codec is None or self.tokenizer is None:
            self.load()
        from voicehub.checkpointing import save_safetensors
        from voicehub.hub import read_json_file, write_json_file

        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)
        codec_directory = save_directory / "snac"
        codec_directory.mkdir(parents=True, exist_ok=True)
        if self.codec_artifacts is not None:
            codec_config = read_json_file(self.codec_artifacts.config)
        else:
            codec_config = {
                "sampling_rate": self.codec.sampling_rate,
                "encoder_dim": self.codec.encoder_dim,
                "encoder_rates": list(self.codec.encoder_rates),
                "latent_dim": self.codec.latent_dim,
                "decoder_dim": self.codec.decoder_dim,
                "decoder_rates": list(self.codec.decoder_rates),
                "attn_window_size": self.codec.attn_window_size,
                "codebook_size": self.codec.codebook_size,
                "codebook_dim": self.codec.codebook_dim,
                "vq_strides": list(self.codec.vq_strides),
                "noise": self.codec.noise,
                "depthwise": self.codec.depthwise,
            }
        write_json_file(codec_directory / "config.json", codec_config)
        save_safetensors(
            self.codec.state_dict(),
            codec_directory / "model.safetensors",
            metadata={
                "format": "pt",
                "architecture": "snac",
                "producer": "voicehub",
            },
        )


OrpheusTTS = OrpheusTTSForTextToSpeech
