"""Native Kokoro decoder graph and phoneme-level synthesis."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from voicehub.hub import resolve_pretrained_file
from voicehub.models.kokoro.native.configuration import KokoroArchitectureConfig
from voicehub.optimization.protocols import OptimizationCompileTarget

from .istftnet import Decoder
from .modules import CustomAlbert, ProsodyPredictor, TextEncoder

_LOGGER = logging.getLogger(__name__)


class KModel(nn.Module):
    """Released Kokoro PL-BERT, prosody, text, and iSTFTNet graph.

    Construction is intentionally separate from checkpoint I/O. Provider
    loaders resolve a coherent artifact set and then use the strict
    native checkpoint adapter. This keeps the architecture usable for
    tiny tests, fine-tuning, and future optimization passes without
    network side effects.
    """

    MODEL_NAMES = {
        "hexgrad/Kokoro-82M": "kokoro-v1_0.pth",
        "hexgrad/Kokoro-82M-v1.1-zh": "kokoro-v1_1-zh.pth",
    }

    @staticmethod
    def _local_snapshot_file(
        repo_id: str | Path,
        filename: str,
    ) -> str | None:
        root = Path(repo_id).expanduser()
        if not root.is_dir():
            return None
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(f"Kokoro local snapshot is missing {filename!r}: {path}.")
        return str(path.resolve())

    @classmethod
    def _model_file(
        cls,
        repo_id: str | Path,
        *,
        revision: str | None = None,
        cache_dir: str | None = None,
        token: str | bool | None = None,
        local_files_only: bool = False,
    ) -> str:
        root = Path(repo_id).expanduser()
        if root.is_dir():
            native_names = (
                "model.safetensors",
                "kokoro-v1_0.safetensors",
                "pytorch_model.safetensors",
            )
            for filename in (*native_names, *cls.MODEL_NAMES.values()):
                candidate = root / filename
                if candidate.is_file():
                    return str(candidate.resolve())
            candidates = sorted(
                (
                    *root.glob("*.safetensors"),
                    *root.glob("*.pth"),
                ),
                key=lambda item: item.name,
            )
            if len(candidates) == 1:
                return str(candidates[0].resolve())
            available = ", ".join(path.name for path in candidates) or "none"
            raise FileNotFoundError(
                "Kokoro local snapshot must contain one native Safetensors "
                "or released checkpoint at its root; found: "
                f"{available}.")
        try:
            filename = cls.MODEL_NAMES[str(repo_id)]
        except KeyError as exc:
            supported = ", ".join(cls.MODEL_NAMES)
            raise ValueError(
                f"Unknown Kokoro repository {str(repo_id)!r}. Supported Hub "
                f"repositories: {supported}; local artifact directories are "
                "also accepted.") from exc
        return str(
            resolve_pretrained_file(
                repo_id,
                filename,
                revision=revision,
                cache_dir=cache_dir,
                token=token,
                local_files_only=local_files_only,
            ))

    def __init__(
        self,
        config: KokoroArchitectureConfig | dict[str, Any],
        *,
        disable_complex: bool = False,
    ) -> None:
        super().__init__()
        self.native_config = KokoroArchitectureConfig.coerce(config)
        config = self.native_config
        self.vocab = dict(config.vocab)
        self.bert = CustomAlbert(config.plbert)
        self.bert_encoder = nn.Linear(
            config.plbert.hidden_size,
            config.hidden_dim,
        )
        self.context_length = config.plbert.max_position_embeddings
        self.predictor = ProsodyPredictor(
            style_dim=config.style_dim,
            d_hid=config.hidden_dim,
            nlayers=config.n_layer,
            max_dur=config.max_dur,
            dropout=config.dropout,
        )
        self.text_encoder = TextEncoder(
            channels=config.hidden_dim,
            kernel_size=config.text_encoder_kernel_size,
            depth=config.n_layer,
            n_symbols=config.n_token,
        )
        self.decoder = Decoder(
            dim_in=config.hidden_dim,
            style_dim=config.style_dim,
            dim_out=config.n_mels,
            disable_complex=disable_complex,
            **config.istftnet.to_dict(),
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @dataclass
    class Output:
        """One synthesis result."""

        audio: torch.Tensor
        pred_dur: torch.Tensor | None = None

    @staticmethod
    def _lengths(
        input_ids: torch.Tensor,
        input_lengths: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, sequence_length = input_ids.shape
        if input_lengths is None:
            return torch.full(
                (batch_size, ),
                sequence_length,
                dtype=torch.long,
                device=input_ids.device,
            )
        if not torch.is_tensor(input_lengths):
            raise TypeError("`input_lengths` must be a tensor or None.")
        if tuple(input_lengths.shape) != (batch_size, ):
            raise ValueError("`input_lengths` must have shape [batch].")
        if (input_lengths.dtype == torch.bool or input_lengths.is_floating_point() or
                input_lengths.is_complex()):
            raise TypeError("`input_lengths` must use an integer dtype.")
        input_lengths = input_lengths.to(
            device=input_ids.device,
            dtype=torch.long,
        )
        if bool(((input_lengths < 1) | (input_lengths > sequence_length)).any()):
            raise ValueError("`input_lengths` values must be within the token sequence.")
        return input_lengths

    @staticmethod
    def _text_mask(
        input_lengths: torch.Tensor,
        sequence_length: int,
    ) -> torch.Tensor:
        positions = torch.arange(
            sequence_length,
            device=input_lengths.device,
        )
        return positions.unsqueeze(0) >= input_lengths.unsqueeze(1)

    @staticmethod
    def alignment_from_durations(
        durations: torch.Tensor,
        *,
        text_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build a dense text-to-frame alignment from integer durations."""
        if not torch.is_tensor(durations) or durations.ndim != 2:
            raise ValueError("`durations` must have shape [batch, text].")
        if (durations.dtype == torch.bool or durations.is_floating_point() or durations.is_complex()):
            raise TypeError("`durations` must use an integer dtype.")
        durations = durations.long()
        if bool((durations < 0).any()):
            raise ValueError("`durations` cannot contain negative values.")
        if text_mask is not None:
            if text_mask.shape != durations.shape:
                raise ValueError("`text_mask` must match `durations`.")
            durations = durations.masked_fill(text_mask, 0)
        frame_lengths = durations.sum(dim=1)
        if bool((frame_lengths < 1).any()):
            raise ValueError("Every Kokoro item must align to at least one acoustic frame.")
        maximum_frames = int(frame_lengths.max().item())
        alignment = torch.zeros(
            durations.shape[0],
            durations.shape[1],
            maximum_frames,
            device=durations.device,
            dtype=torch.float32,
        )
        for batch_index in range(durations.shape[0]):
            cursor = 0
            for token_index in range(durations.shape[1]):
                count = int(durations[batch_index, token_index].item())
                if count:
                    alignment[
                        batch_index,
                        token_index,
                        cursor:cursor + count,
                    ] = 1.0
                    cursor += count
        return alignment

    def encode_text(
        self,
        input_ids: torch.Tensor,
        *,
        input_lengths: torch.Tensor | None = None,
        ref_s: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Run both text encoders and expose duration-training activations."""
        if not torch.is_tensor(input_ids) or input_ids.ndim != 2:
            raise ValueError("`input_ids` must have shape [batch, text].")
        if (input_ids.dtype == torch.bool or input_ids.is_floating_point() or input_ids.is_complex()):
            raise TypeError("`input_ids` must use an integer dtype.")
        if input_ids.shape[1] > self.context_length:
            raise ValueError("Kokoro token sequence exceeds its "
                             f"{self.context_length}-token context.")
        if not torch.is_tensor(ref_s) or ref_s.ndim != 2:
            raise ValueError("`ref_s` must have shape [batch, 256].")
        expected_style = self.native_config.style_dim * 2
        if tuple(ref_s.shape) != (input_ids.shape[0], expected_style):
            raise ValueError("`ref_s` must have shape "
                             f"[batch, {expected_style}].")
        input_ids = input_ids.to(device=self.device, dtype=torch.long)
        ref_s = ref_s.to(device=self.device, dtype=self.dtype)
        lengths = self._lengths(input_ids, input_lengths)
        text_mask = self._text_mask(lengths, input_ids.shape[1])
        bert_hidden = self.bert(
            input_ids,
            attention_mask=(~text_mask).to(dtype=torch.long),
        )
        duration_features = self.bert_encoder(bert_hidden).transpose(-1, -2)
        predictor_style = ref_s[:, self.native_config.style_dim:]
        duration_encoding = self.predictor.text_encoder(
            duration_features,
            predictor_style,
            lengths,
            text_mask,
        )
        duration_hidden, _ = self.predictor.lstm(duration_encoding)
        duration_logits = self.predictor.duration_proj(duration_hidden)
        text_encoding = self.text_encoder(input_ids, lengths, text_mask)
        return {
            "duration_encoding": duration_encoding,
            "duration_logits": duration_logits,
            "input_lengths": lengths,
            "predictor_style": predictor_style,
            "decoder_style": ref_s[:, :self.native_config.style_dim],
            "text_encoding": text_encoding,
            "text_mask": text_mask,
        }

    def decode_aligned(
        self,
        encoded: dict[str, torch.Tensor],
        alignment: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Predict prosody and waveform for a supplied alignment."""
        duration_encoding = encoded["duration_encoding"]
        text_encoding = encoded["text_encoding"]
        if not torch.is_tensor(alignment) or alignment.ndim != 3:
            raise ValueError("`alignment` must have shape [batch, text, acoustic_frames].")
        expected_prefix = (
            duration_encoding.shape[0],
            duration_encoding.shape[1],
        )
        if tuple(alignment.shape[:2]) != expected_prefix:
            raise ValueError("Kokoro alignment batch/text dimensions must match input IDs.")
        alignment = alignment.to(
            device=self.device,
            dtype=duration_encoding.dtype,
        )
        if not bool(torch.isfinite(alignment).all()):
            raise ValueError("Kokoro alignment must contain finite values.")
        if bool((alignment < 0).any()):
            raise ValueError("Kokoro alignment cannot contain negative values.")
        aligned_duration = duration_encoding.transpose(-1, -2) @ alignment
        f0, energy = self.predictor.F0Ntrain(
            aligned_duration,
            encoded["predictor_style"],
        )
        aligned_text = text_encoding @ alignment
        waveform = self.decoder(
            aligned_text,
            f0,
            energy,
            encoded["decoder_style"],
        )
        return {
            "waveform": waveform,
            "f0": f0,
            "energy": energy,
            "alignment": alignment,
        }

    def forward_preprocessed(
        self,
        input_ids: torch.Tensor,
        *,
        ref_s: torch.Tensor,
        input_lengths: torch.Tensor | None = None,
        alignment: torch.Tensor | None = None,
        durations: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Differentiable decoder forward for prepared training batches."""
        encoded = self.encode_text(
            input_ids,
            input_lengths=input_lengths,
            ref_s=ref_s,
        )
        if alignment is None:
            if durations is None:
                raise ValueError(
                    "Kokoro preprocessed forward requires `alignment` or "
                    "integer `durations`.")
            alignment = self.alignment_from_durations(
                durations.to(device=self.device),
                text_mask=encoded["text_mask"],
            )
        decoded = self.decode_aligned(encoded, alignment)
        return {**encoded, **decoded}

    def optimization_compile_targets(
        self,
        mode: str,
    ) -> tuple[OptimizationCompileTarget, ...]:
        """Keep text parsing outside the compiled Kokoro tensor graph."""
        if mode == "inference":
            return (
                OptimizationCompileTarget(
                    "decoder.forward_with_tokens",
                    self,
                    "forward_with_tokens",
                ), )
        if mode == "training":
            return (
                OptimizationCompileTarget(
                    "decoder.forward_preprocessed",
                    self,
                    "forward_preprocessed",
                ), )
        raise ValueError("Kokoro compile targets require 'inference' or 'training' mode.")

    @torch.no_grad()
    def forward_with_tokens(
        self,
        input_ids: torch.Tensor,
        ref_s: torch.Tensor,
        speed: float = 1.0,
        *,
        input_lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Synthesize a prepared phoneme-token batch."""
        if (isinstance(speed, bool) or not isinstance(speed, (int, float)) or
                not 0 < float(speed) < float("inf")):
            raise ValueError("`speed` must be finite and positive.")
        encoded = self.encode_text(
            input_ids,
            input_lengths=input_lengths,
            ref_s=ref_s,
        )
        predicted = (torch.sigmoid(encoded["duration_logits"]).sum(dim=-1) / float(speed))
        durations = torch.round(predicted).clamp(min=1).long()
        durations = durations.masked_fill(encoded["text_mask"], 0)
        alignment = self.alignment_from_durations(
            durations,
            text_mask=encoded["text_mask"],
        )
        decoded = self.decode_aligned(encoded, alignment)
        return decoded["waveform"], durations

    def tokenize_phonemes(self, phonemes: str) -> list[int]:
        """Map released Kokoro phoneme symbols to IDs."""
        if not isinstance(phonemes, str) or not phonemes:
            raise ValueError("`phonemes` must be a non-empty string.")
        unknown = sorted(set(phonemes) - set(self.vocab))
        if unknown:
            display = ", ".join(repr(item) for item in unknown)
            raise ValueError("Kokoro phonemes contain unsupported symbols: "
                             f"{display}.")
        input_ids = [self.vocab[symbol] for symbol in phonemes]
        if not input_ids:
            raise ValueError("The phoneme sequence contains no symbols in Kokoro's "
                             "released vocabulary.")
        if len(input_ids) + 2 > self.context_length:
            raise ValueError(
                "Kokoro phoneme sequence exceeds its "
                f"{self.context_length - 2}-symbol payload limit.")
        return input_ids

    def forward(
        self,
        phonemes: str,
        ref_s: torch.Tensor,
        speed: float = 1.0,
        return_output: bool = False,
    ) -> Output | torch.Tensor:
        input_ids = self.tokenize_phonemes(phonemes)
        _LOGGER.debug("Mapped %d phonemes to Kokoro IDs.", len(input_ids))
        tokens = torch.tensor(
            [[0, *input_ids, 0]],
            device=self.device,
            dtype=torch.long,
        )
        style = ref_s.to(device=self.device, dtype=self.dtype)
        if style.ndim == 1:
            style = style.unsqueeze(0)
        audio, pred_dur = self.forward_with_tokens(tokens, style, speed)
        audio = audio.squeeze().detach().cpu()
        pred_dur = pred_dur.squeeze(0).detach().cpu()
        result = self.Output(audio=audio, pred_dur=pred_dur)
        return result if return_output else result.audio


class KModelForONNX(nn.Module):
    """Tensor-only compatibility wrapper around :class:`KModel`."""

    def __init__(self, kmodel: KModel) -> None:
        super().__init__()
        self.kmodel = kmodel

    def forward(
        self,
        input_ids: torch.Tensor,
        ref_s: torch.Tensor,
        speed: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.kmodel.forward_with_tokens(input_ids, ref_s, speed)


__all__ = ["KModel", "KModelForONNX"]
