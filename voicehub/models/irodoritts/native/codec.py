"""Digest-gated Semantic-DACVAE runtime used by Irodori-TTS."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .codec_graph import DACVAE
from .metadata import IRODORI_CODEC_CHECKPOINT


def sha256_file(path: str | Path) -> str:
    resolved = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_digest_gated_codec_payload(
    path: str | Path,
    *,
    expected_sha256: str = IRODORI_CODEC_CHECKPOINT["lfs_sha256"],
) -> Mapping[str, Any]:
    """Load the one audited legacy codec archive without unsafe unpickling."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Irodori codec checkpoint was not found: {resolved}.")
    normalized_digest = str(expected_sha256).strip().lower()
    if len(normalized_digest) != 64 or any(character not in "0123456789abcdef"
                                           for character in normalized_digest):
        raise ValueError("Codec `expected_sha256` must be a lowercase SHA-256 digest.")
    actual_digest = sha256_file(resolved)
    if actual_digest != normalized_digest:
        raise ValueError(
            "Refusing to unpickle an unverified Irodori codec checkpoint: "
            f"SHA-256 {actual_digest} != {normalized_digest}.")
    try:
        payload = torch.load(
            resolved,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError as error:
        raise RuntimeError(
            "Irodori codec loading requires a PyTorch release with "
            "`torch.load(weights_only=True)`.") from error
    if not isinstance(payload, Mapping):
        raise ValueError("Irodori codec checkpoint must contain a mapping.")
    state_dict = payload.get("state_dict")
    metadata = payload.get("metadata")
    if not isinstance(state_dict, Mapping) or not isinstance(metadata, Mapping):
        raise ValueError("Irodori codec checkpoint requires state_dict and metadata mappings.")
    constructor = metadata.get("kwargs")
    if not isinstance(constructor, Mapping):
        raise ValueError("Irodori codec checkpoint is missing constructor metadata.")
    expected_constructor = IRODORI_CODEC_CHECKPOINT["constructor"]
    normalized_constructor = {
        name: list(value) if isinstance(value, tuple) else value
        for name, value in constructor.items()
    }
    if normalized_constructor != expected_constructor:
        raise ValueError("Irodori codec constructor metadata differs from the audited release.")
    return payload


class IrodoriDACVAECodec:
    """Native wrapper around the exact published Semantic-DACVAE graph."""

    sample_rate = 48_000
    hop_length = 1_920
    latent_dim = 32

    def __init__(
        self,
        model: DACVAE,
        *,
        normalize_db: float | None = -16.0,
    ) -> None:
        if not isinstance(model, DACVAE):
            raise TypeError("Irodori codec model must be the native DACVAE graph.")
        if model.sample_rate != self.sample_rate or model.hop_length != self.hop_length:
            raise ValueError("Irodori codec graph has incompatible timing metadata.")
        self.model = model
        self.normalize_db = normalize_db
        self.model.decoder.alpha = 0.0

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        expected_sha256: str = IRODORI_CODEC_CHECKPOINT["lfs_sha256"],
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        normalize_db: float | None = -16.0,
    ) -> IrodoriDACVAECodec:
        if dtype not in {torch.float32, torch.bfloat16}:
            raise ValueError("Irodori codec execution supports fp32 or bf16.")
        payload = load_digest_gated_codec_payload(
            path,
            expected_sha256=expected_sha256,
        )
        model = DACVAE(**IRODORI_CODEC_CHECKPOINT["constructor"])
        incompatible = model.load_state_dict(payload["state_dict"], strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise ValueError("Irodori codec checkpoint did not load strictly.")
        model.to(device=device, dtype=dtype).eval()
        return cls(model, normalize_db=normalize_db)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    @staticmethod
    def _normalize_peak_and_rms(
        waveform: torch.Tensor,
        target_db: float | None,
    ) -> torch.Tensor:
        """Deterministic PyTorch-only loudness boundary.

        The official recipe targets -16 dB before encoding.  VoiceHub
        uses a bounded RMS equivalent here; no accuracy-parity claim is
        made for an unavailable external perceptual loudness
        implementation.
        """
        waveform = waveform.float()
        peak = waveform.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
        waveform = waveform / peak.clamp_min(1.0)
        if target_db is None:
            return waveform
        target_rms = 10.0**(float(target_db) / 20.0)
        rms = waveform.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)
        gain = (target_rms / rms).clamp(max=1.0 / peak.clamp_min(1e-8))
        return (waveform * gain).clamp(-1.0, 1.0)

    def encode_waveform(
        self,
        waveform: torch.Tensor,
        *,
        normalize_db: float | None = None,
        deterministic: bool = True,
    ) -> torch.Tensor:
        if not isinstance(waveform, torch.Tensor):
            raise TypeError("Irodori codec waveform must be a torch.Tensor.")
        if waveform.ndim == 1:
            waveform = waveform[None, None, :]
        elif waveform.ndim == 2:
            waveform = waveform[:, None, :] if waveform.shape[0] > 1 else waveform[None]
        if waveform.ndim != 3:
            raise ValueError("Irodori codec waveform must have shape (B, C, T).")
        if waveform.numel() == 0 or waveform.shape[-1] == 0:
            raise ValueError("Irodori codec waveform cannot be empty.")
        if not waveform.is_floating_point():
            waveform = waveform.float()
        if not torch.isfinite(waveform).all().item():
            raise ValueError("Irodori codec waveform must contain only finite values.")
        if not isinstance(deterministic, bool):
            raise TypeError("`deterministic` must be a boolean.")
        if waveform.shape[1] != 1:
            waveform = waveform.mean(dim=1, keepdim=True)
        target_db = self.normalize_db if normalize_db is None else normalize_db
        if target_db is not None and (isinstance(target_db, bool) or not isinstance(target_db,
                                                                                    (int, float)) or
                                      not math.isfinite(float(target_db))):
            raise ValueError("Irodori normalization dB must be finite or None.")
        waveform = self._normalize_peak_and_rms(waveform, target_db)
        waveform = waveform.to(device=self.device, dtype=self.dtype)
        length = waveform.shape[-1]
        remainder = length % self.hop_length
        if remainder:
            amount = self.hop_length - remainder
            mode = "reflect" if length > amount else "replicate"
            waveform = F.pad(waveform, (0, amount), mode=mode)
        encoded = self.model.encoder(waveform)
        mean, scale = self.model.quantizer.in_proj(encoded).chunk(2, dim=1)
        if deterministic:
            latent = mean
        else:
            standard_deviation = F.softplus(scale) + 1e-4
            latent = mean + torch.randn_like(mean) * standard_deviation
        return latent.transpose(1, 2).contiguous()

    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        if not isinstance(latent, torch.Tensor) or latent.ndim != 3:
            raise ValueError("Irodori latents must have shape (B, T, 32).")
        if latent.shape[-1] != self.latent_dim:
            raise ValueError(f"Irodori latent width must be {self.latent_dim}, got {latent.shape[-1]}.")
        if latent.numel() == 0 or not latent.is_floating_point():
            raise ValueError("Irodori latents must be non-empty floating-point tensors.")
        if not torch.isfinite(latent).all().item():
            raise ValueError("Irodori latents must contain only finite values.")
        latent = latent.to(device=self.device, dtype=self.dtype).transpose(1, 2)
        decoded = self.model.decode(latent)
        if decoded.ndim != 3 or decoded.shape[1] != 1:
            raise RuntimeError("Irodori codec returned a non-mono waveform.")
        return decoded[:, 0].float()


__all__ = [
    "IrodoriDACVAECodec",
    "load_digest_gated_codec_payload",
    "sha256_file",
]
