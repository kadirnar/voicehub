"""End-to-end native Fish S2 prompt, generation, and codec runtime."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from voicehub.models.fishtts.native.modeling import FishS2ForConditionalGeneration
from voicehub.models.fishtts.native.prompting import (
    FishConversationTurn,
    append_generated_turn,
    build_fish_prompt,
    group_speaker_turns,
    split_speaker_turns,
)
from voicehub.models.fishtts.native.sampling import generate_fish_codes
from voicehub.models.fishtts.native.tokenization import FishTokenizer
from voicehub.processing.waveform import load_native_audio


class FishS2Runtime(nn.Module):
    """Composes the trainable semantic model with frozen ModifiedDAC."""

    def __init__(
        self,
        *,
        semantic_model: FishS2ForConditionalGeneration,
        tokenizer: FishTokenizer,
        codec: nn.Module,
    ) -> None:
        super().__init__()
        if not isinstance(
                semantic_model,
                FishS2ForConditionalGeneration,
        ):
            raise TypeError("`semantic_model` must be a native Fish S2 model.")
        if not isinstance(tokenizer, FishTokenizer):
            raise TypeError("`tokenizer` must be a FishTokenizer.")
        for method in ("encode", "from_indices"):
            if not callable(getattr(codec, method, None)):
                raise TypeError(f"Fish codec must expose a callable {method}().")
        sample_rate = getattr(codec, "sample_rate", None)
        if sample_rate != semantic_model.config.sample_rate:
            raise ValueError("Fish semantic model and codec sample rates disagree.")
        self.semantic_model = semantic_model
        self.tokenizer = tokenizer
        self.codec = codec
        self.last_seed: int | None = None

    @property
    def sample_rate(self) -> int:
        return int(self.codec.sample_rate)

    @property
    def device(self) -> torch.device:
        return next(self.semantic_model.parameters()).device

    def forward(self, *args: Any, **kwargs: Any):
        """Delegate differentiable fine-tuning to the semantic graph."""
        return self.semantic_model(*args, **kwargs)

    @torch.inference_mode()
    def encode_reference(
        self,
        audio: Any,
        *,
        sampling_rate: int | None = None,
    ) -> Tensor:
        loaded = load_native_audio(
            audio,
            sampling_rate=sampling_rate,
            target_sampling_rate=self.sample_rate,
        )
        parameter = next(self.codec.parameters())
        waveform = loaded.waveform.to(
            device=parameter.device,
            dtype=parameter.dtype,
        ).view(1, 1, -1)
        lengths = torch.tensor(
            [waveform.shape[-1]],
            device=waveform.device,
            dtype=torch.long,
        )
        codes, code_lengths = self.codec.encode(
            waveform,
            audio_lengths=lengths,
        )
        length = int(code_lengths[0].item())
        return codes[0, :, :length].detach()

    @torch.inference_mode()
    def decode_codes(self, codes: Tensor) -> Tensor:
        values = torch.as_tensor(codes)
        if (values.dtype == torch.bool or values.is_floating_point() or values.is_complex()):
            raise TypeError("Fish codes must use an integer dtype.")
        expected = self.semantic_model.config.num_codebooks
        if values.ndim != 2 or values.shape[0] != expected:
            raise ValueError(f"Fish codes must have shape [{expected}, time].")
        if values.shape[1] == 0:
            raise ValueError("Fish code sequences cannot be empty.")
        codebook_size = self.semantic_model.config.codebook_size
        if (int(values.min().item()) < 0 or int(values.max().item()) >= codebook_size):
            raise ValueError("Fish codes contain an out-of-range ID.")
        values = values.to(
            dtype=torch.long,
            device=self.device,
        )
        audio = self.codec.from_indices(values.unsqueeze(0))
        if (not isinstance(audio, Tensor) or audio.ndim != 3 or audio.shape[:2] != (1, 1)):
            raise RuntimeError("Fish codec returned an invalid waveform tensor.")
        return audio[0, 0].float()

    @torch.inference_mode()
    def infer(
        self,
        text: str,
        *,
        reference_text: str | None = None,
        reference_codes: Tensor | None = None,
        maximum_chunk_bytes: int = 512,
        max_new_tokens: int = 1_024,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 30,
        seed: int | None = None,
    ) -> Tensor:
        chunks = group_speaker_turns(
            split_speaker_turns(text),
            maximum_utf8_bytes=maximum_chunk_bytes,
        )
        history: tuple[FishConversationTurn, ...] = ()
        audio: list[Tensor] = []
        if seed is None:
            seed = torch.seed()
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("Fish seed must be an integer or None.")
        self.last_seed = int(seed)
        torch.manual_seed(self.last_seed)
        for chunk in chunks:
            prompt = build_fish_prompt(
                chunk,
                self.tokenizer,
                reference_text=reference_text,
                reference_codes=reference_codes,
                history=history,
            ).to(self.device)
            codes = generate_fish_codes(
                self.semantic_model,
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            audio.append(self.decode_codes(codes))
            history = append_generated_turn(
                history,
                text=chunk,
                codes=codes,
            )
        if not audio:
            raise RuntimeError("Fish runtime produced no audio.")
        return torch.cat(audio)

    def prepare_for_training(self) -> None:
        self.semantic_model.clear_caches()
        self.semantic_model.train()
        self.codec.eval()
        for parameter in self.codec.parameters():
            parameter.requires_grad_(False)

    def prepare_for_inference(self) -> None:
        self.semantic_model.eval()
        self.codec.eval()

    def save_pretrained(self, directory: str | Any) -> Any:
        from voicehub.models.fishtts.native.checkpoint import save_fish_runtime_pretrained

        return save_fish_runtime_pretrained(self, directory)


__all__ = ["FishS2Runtime"]
