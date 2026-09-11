"""Native NeuTTS Llama/Qwen backbone and end-to-end synthesis runtime."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from voicehub.generation import GenerationConfig
from voicehub.generation.engine import AutoregressiveGenerator, GenerationStepInput, GenerationStepOutput
from voicehub.models.causal_lm.native.modeling import CausalLMForCausalLM
from voicehub.models.neutts.native.configuration import NeuTTSBackboneConfig
from voicehub.models.neutts.native.neucodec import NeuCodecModel
from voicehub.models.neutts.native.tokenization import (
    SPEECH_CODEBOOK_SIZE,
    SPEECH_GENERATION_END,
    SPEECH_GENERATION_START,
    SUPPORTED_EMOTIONS,
    TEXT_PROMPT_END,
    TEXT_PROMPT_START,
    NeuTTSTokenizer,
    normalize_neutts_text,
)
from voicehub.neural.rotary import RotaryEmbedding
from voicehub.optimization.protocols import OptimizationCompileTarget
from voicehub.processing.waveform import load_native_audio


class LinearScalingRotaryEmbedding(RotaryEmbedding):
    """NeuTTS-Nano's published linear position interpolation."""

    def __init__(
        self,
        dimension: int,
        *,
        base: float,
        factor: float,
        device: Any = None,
    ) -> None:
        if (isinstance(factor, bool) or not isinstance(factor, (int, float)) or
                not math.isfinite(float(factor)) or factor < 1.0):
            raise ValueError("Linear RoPE factor must be finite and at least one.")
        super().__init__(dimension, base=base, device=device)
        self.factor = float(factor)
        self.inverse_frequency.div_(self.factor)


class NeuTTSBackbone(CausalLMForCausalLM):
    """Dense native LM with the local NeuTTS-Nano RoPE specialization."""

    def __init__(
        self,
        config: NeuTTSBackboneConfig,
        *,
        initialize: bool = True,
        device: Any = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        if not isinstance(config, NeuTTSBackboneConfig):
            raise TypeError("`config` must be a NeuTTSBackboneConfig.")
        self.neutts_config = config
        super().__init__(
            config.causal_lm,
            initialize=initialize,
            device=device,
            dtype=dtype,
        )
        factor = config.linear_rope_factor
        if factor is not None:
            for layer in self.model.layers:
                layer.self_attn.rotary = LinearScalingRotaryEmbedding(
                    self.config.head_dim,
                    base=self.config.rope_theta,
                    factor=factor,
                    device=device,
                )

    def save_pretrained(self, directory: str | Path) -> Path:
        """Export exact upstream config semantics plus safe native weights."""
        from voicehub.checkpointing import save_safetensors

        target = Path(directory).expanduser()
        target.mkdir(parents=True, exist_ok=True)
        (target / "config.json").write_text(
            json.dumps(
                self.neutts_config.to_dict(),
                indent=2,
                sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )
        state = dict(self.state_dict())
        if self.config.tie_word_embeddings:
            state.pop("lm_head.weight", None)
        save_safetensors(
            state,
            target / "model.safetensors",
            metadata={
                "format": "pt",
                "architecture": self.config.model_type,
                "producer": "voicehub",
                "family": "neutts",
            },
        )
        return target.resolve()


class _MinimumTokensBeforeEos:
    """Mask the speech terminator until the published minimum is reached."""

    def __init__(
        self,
        *,
        prompt_length: int,
        minimum_tokens: int,
        eos_token_id: int,
    ) -> None:
        self.prompt_length = prompt_length
        self.minimum_tokens = minimum_tokens
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: Tensor, logits: Tensor) -> Tensor:
        generated = input_ids.shape[-1] - self.prompt_length
        if generated < self.minimum_tokens:
            logits[:, self.eos_token_id] = torch.finfo(logits.dtype).min
        return logits


class NeuTTSRuntime(nn.Module):
    """End-to-end prompt, generation, reference encoding, and decoding."""

    def __init__(
        self,
        *,
        backbone: NeuTTSBackbone,
        tokenizer: NeuTTSTokenizer,
        codec: NeuCodecModel,
        max_context: int = 2_048,
        min_new_tokens: int = 50,
        language: str | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(backbone, NeuTTSBackbone):
            raise TypeError("`backbone` must be a NeuTTSBackbone.")
        if not isinstance(tokenizer, NeuTTSTokenizer):
            raise TypeError("`tokenizer` must be a NeuTTSTokenizer.")
        if not isinstance(codec, NeuCodecModel):
            raise TypeError("`codec` must be a NeuCodecModel.")
        if (isinstance(max_context, bool) or not isinstance(max_context, int) or max_context <= 0):
            raise ValueError("`max_context` must be a positive integer.")
        if (isinstance(min_new_tokens, bool) or not isinstance(min_new_tokens, int) or min_new_tokens <= 0):
            raise ValueError("`min_new_tokens` must be a positive integer.")
        self.backbone = backbone
        self.codec = codec
        self.tokenizer = tokenizer
        self.native_config = backbone.neutts_config
        self.input_format = self.native_config.input_format
        self.max_context = max_context
        self.min_new_tokens = min_new_tokens
        declared_languages = self.native_config.supported_languages
        self.language = (language or (declared_languages[0] if declared_languages else None))
        self.last_seed: int | None = None

    @property
    def sample_rate(self) -> int:
        return self.codec.output_sampling_rate

    @property
    def device(self) -> torch.device:
        return next(self.backbone.parameters()).device

    def forward(self, *args: Any, **kwargs: Any):
        """Delegate differentiable language-model training to the backbone."""
        return self.backbone(*args, **kwargs)

    def optimization_compile_targets(
        self,
        mode: str,
    ) -> tuple[OptimizationCompileTarget, ...]:
        """Expose only the quality-safe NeuTTS training boundary.

        Real-checkpoint inference validation changed generated audio and
        transcript content after compiling the autoregressive backbone.
        Training keeps the full-sequence backbone boundary because it
        does not perform stochastic token generation or waveform
        reconstruction.
        """
        if mode == "inference":
            return ()
        if mode == "training":
            return (OptimizationCompileTarget(
                "backbone.forward",
                self.backbone,
                "forward",
            ), )
        raise ValueError("NeuTTS compile targets require 'inference' or 'training' mode.")

    @staticmethod
    def _normalize_codes(reference_codes: Any) -> list[int]:
        if isinstance(reference_codes, Tensor):
            values = reference_codes.detach().cpu().reshape(-1).tolist()
        else:
            if isinstance(reference_codes, (str, bytes)):
                raise TypeError("`reference_codes` must be a tensor or integer sequence.")
            try:
                values = list(reference_codes)
            except TypeError as error:
                raise TypeError("`reference_codes` must be a tensor or integer sequence.") from error
        if not values:
            raise ValueError("NeuTTS reference codes cannot be empty.")
        output = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError("NeuTTS reference codes must be integers.")
            if not 0 <= value < SPEECH_CODEBOOK_SIZE:
                raise ValueError(
                    f"NeuTTS reference code {value} is outside "
                    f"[0, {SPEECH_CODEBOOK_SIZE - 1}].")
            output.append(value)
        return output

    def encode_reference(
        self,
        audio: Any,
        *,
        sampling_rate: int | None = None,
    ) -> Tensor:
        reference = load_native_audio(
            audio,
            sampling_rate=sampling_rate,
            target_sampling_rate=self.codec.input_sampling_rate,
        )
        codec_device = next(self.codec.parameters()).device
        with torch.inference_mode():
            encoded = self.codec.encode_code(
                reference.waveform.to(codec_device).unsqueeze(0),
                sample_rate=self.codec.input_sampling_rate,
            )
        return encoded[0, 0].detach()

    @staticmethod
    def _clean_phonemes(value: Any, *, name: str) -> str:
        if not isinstance(value, str):
            raise TypeError(f"`{name}` must be a string.")
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError(f"`{name}` cannot be empty.")
        if "<|" in normalized or "|>" in normalized:
            raise ValueError(f"`{name}` cannot contain NeuTTS control-token delimiters.")
        return normalized

    @classmethod
    def _resolve_phonemes(
        cls,
        text: str,
        *,
        explicit: str | None,
        phonemizer: Callable[[str], str] | Any | None,
        name: str,
    ) -> str:
        if explicit is not None:
            return cls._clean_phonemes(explicit, name=name)
        if phonemizer is None:
            raise ValueError(
                f"Phoneme-based NeuTTS requires `{name}` or an explicitly "
                "injected `phonemizer`. VoiceHub does not silently depend on "
                "eSpeak or approximate graphemes as phonemes.")
        if callable(phonemizer):
            value = phonemizer(text)
        else:
            method = getattr(phonemizer, "phonemize", None)
            if not callable(method):
                raise TypeError("`phonemizer` must be callable or expose phonemize().")
            value = method([text])
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                value = value[0] if value else ""
        return cls._clean_phonemes(value, name=name)

    def _validate_emotion(self, emotion: str | None) -> str | None:
        if emotion is None:
            return None
        if not isinstance(emotion, str) or not emotion.strip():
            raise ValueError("`emotion` must be a non-empty string or None.")
        normalized = emotion.strip().lower()
        if normalized == "neutral":
            return None
        if self.input_format == "phonemes":
            raise ValueError("Emotion tokens are available only for BPE NeuTTS variants.")
        supported = (self.native_config.supported_emotions or SUPPORTED_EMOTIONS)
        if normalized not in supported:
            raise ValueError(
                f"Unknown NeuTTS emotion {normalized!r}; supported values: " + ", ".join(supported) + ".")
        token = f"<|{normalized.upper()}|>"
        try:
            self.tokenizer.convert_tokens_to_ids(token)
        except KeyError as error:
            raise ValueError(f"Emotion token {token!r} is absent from the tokenizer.") from error
        return normalized

    def build_prompt(
        self,
        text: str,
        *,
        reference_codes: Any,
        reference_text: str,
        text_phonemes: str | None = None,
        reference_phonemes: str | None = None,
        phonemizer: Callable[[str], str] | Any | None = None,
        emotion: str | None = None,
    ) -> Tensor:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("NeuTTS text must be a non-empty string.")
        if not isinstance(reference_text, str) or not reference_text.strip():
            raise ValueError("NeuTTS requires a non-empty reference transcript.")
        codes = self._normalize_codes(reference_codes)
        emotion = self._validate_emotion(emotion)
        if self.input_format == "phonemes":
            reference_content = self._resolve_phonemes(
                reference_text,
                explicit=reference_phonemes,
                phonemizer=phonemizer,
                name="reference_phonemes",
            )
            input_content = self._resolve_phonemes(
                text,
                explicit=text_phonemes,
                phonemizer=phonemizer,
                name="text_phonemes",
            )
            content = f"{reference_content} {input_content}"
            prefix = (
                "user: Convert the text to speech:"
                f"{TEXT_PROMPT_START}{content}{TEXT_PROMPT_END}\n"
                f"assistant:{SPEECH_GENERATION_START}")
        else:
            reference_content = normalize_neutts_text(reference_text)
            input_content = normalize_neutts_text(text)
            content = (
                f"{reference_content} {input_content}" if emotion is None else
                (reference_content + f"<|{emotion.upper()}|>" + input_content))
            prefix = (f"{TEXT_PROMPT_START}{content}{TEXT_PROMPT_END}"
                      f"{SPEECH_GENERATION_START}")
        prefix_ids = self.tokenizer.encode(
            prefix,
            add_special_tokens=True,
        ).input_ids
        code_ids = tuple(self.tokenizer.speech_code_to_token_id(code) for code in codes)
        ids = (*prefix_ids, *code_ids)
        if len(ids) >= self.max_context:
            raise ValueError(
                "NeuTTS prompt reaches the model context limit; shorten the "
                "reference/text or use fewer reference codes.")
        return torch.tensor(
            ids,
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(0)

    def _generate_tokens(
        self,
        input_ids: Tensor,
        *,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        seed: int | None,
    ) -> Tensor:
        speech_end_id = self.tokenizer.convert_tokens_to_ids(SPEECH_GENERATION_END)

        def decoder_step(step: GenerationStepInput) -> GenerationStepOutput:
            output = self.backbone(
                step.token_ids,
                attention_mask=None,
                past_key_values=step.cache,
                use_cache=step.use_cache,
            )
            return GenerationStepOutput(
                logits=output.logits,
                cache=output.past_key_values,
            )

        available = self.max_context - input_ids.shape[-1]
        if available <= 0:
            raise ValueError("NeuTTS prompt leaves no room for generation.")
        effective_tokens = min(max_new_tokens, available)
        minimum_tokens = min(self.min_new_tokens, effective_tokens)
        generation = AutoregressiveGenerator().generate(
            decoder_step,
            input_ids,
            GenerationConfig(
                max_new_tokens=effective_tokens,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                eos_token_id=speech_end_id,
                pad_token_id=(
                    self.tokenizer.pad_token_id
                    if self.tokenizer.pad_token_id is not None else speech_end_id),
                seed=seed,
                use_cache=True,
            ),
            logits_processors=(
                _MinimumTokensBeforeEos(
                    prompt_length=input_ids.shape[-1],
                    minimum_tokens=minimum_tokens,
                    eos_token_id=speech_end_id,
                ), ),
        )
        return generation.sequences[:, input_ids.shape[-1]:]

    def infer(
        self,
        text: str,
        reference_codes: Any,
        reference_text: str,
        *,
        emotion: str | None = None,
        temperature: float = 1.0,
        top_k: int = 50,
        seed: int | None = None,
        max_new_tokens: int = 2_000,
        text_phonemes: str | None = None,
        reference_phonemes: str | None = None,
        phonemizer: Callable[[str], str] | Any | None = None,
    ) -> Tensor:
        input_ids = self.build_prompt(
            text,
            reference_codes=reference_codes,
            reference_text=reference_text,
            text_phonemes=text_phonemes,
            reference_phonemes=reference_phonemes,
            phonemizer=phonemizer,
            emotion=emotion,
        )
        with torch.inference_mode():
            generated = self._generate_tokens(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                seed=seed,
            )
        speech_end_id = self.tokenizer.convert_tokens_to_ids(SPEECH_GENERATION_END)
        codes = []
        for token_id in generated[0].detach().cpu().tolist():
            if token_id == speech_end_id:
                break
            try:
                codes.append(self.tokenizer.token_id_to_speech_code(token_id))
            except ValueError:
                # Upstream decoding extracts speech tokens and ignores text
                # vocabulary intrusions. Keep that behavior while avoiding a
                # string/regex round trip.
                continue
        if not codes:
            raise RuntimeError("NeuTTS generated no valid NeuCodec tokens.")
        codec_device = next(self.codec.parameters()).device
        codec_tokens = torch.tensor(
            codes,
            dtype=torch.long,
            device=codec_device,
        ).view(1, 1, -1)
        with torch.inference_mode():
            waveform = self.codec.decode_code(codec_tokens)[0, 0]
        self.last_seed = seed
        return waveform.detach()

    def save_pretrained(self, directory: str | Path) -> Path:
        target = self.backbone.save_pretrained(directory)
        self.tokenizer.save_pretrained(target)
        self.codec.save_pretrained(target / "neucodec")
        return target


__all__ = [
    "LinearScalingRotaryEmbedding",
    "NeuTTSBackbone",
    "NeuTTSRuntime",
]
