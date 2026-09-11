"""Native Irodori-TTS data preparation and flow-matching objectives."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from voicehub.checkpointing import SafeTensorReader
from voicehub.processing.waveform import resample_waveform

from .codec import IrodoriDACVAECodec
from .configuration import IrodoriModelConfig
from .duration import build_duration_features
from .flow_matching import rf_interpolate, rf_velocity_target, sample_logit_normal_t, sample_stratified_logit_normal_t
from .modeling import TextToLatentRFDiT
from .normalization import normalize_text
from .runtime import patchify_latent
from .tokenization import IrodoriTokenizer


def load_preencoded_latent(path: str | Path) -> torch.Tensor:
    resolved = Path(path).expanduser().resolve()
    if resolved.suffix.lower() != ".safetensors":
        raise ValueError("Irodori pre-encoded training data must use Safetensors.")
    with SafeTensorReader(resolved) as reader:
        if set(reader.keys()) != {"latent"}:
            raise ValueError("Irodori latent artifact must contain only the `latent` tensor.")
        latent = reader.get_tensor("latent")
    if latent.ndim not in {2, 3} or not latent.is_floating_point():
        raise ValueError("Irodori latent artifact must contain a rank-two/three float tensor.")
    if not torch.isfinite(latent).all():
        raise ValueError("Irodori latent artifact contains non-finite values.")
    return latent.float()


def _as_text_batch(value: Any) -> list[str]:
    if value is None:
        raise ValueError("Irodori training requires `text`.")
    if isinstance(value, str):
        values = [value]
    else:
        try:
            values = list(value)
        except TypeError as error:
            raise TypeError("Irodori training `text` must be a string or a sequence of strings.") from error
    if not values:
        raise ValueError("Irodori training text batch cannot be empty.")
    normalized = []
    for text in values:
        if not isinstance(text, str):
            raise TypeError("Irodori training text values must be strings.")
        text = normalize_text(text).strip()
        if not text:
            raise ValueError("Irodori training text became empty after normalization.")
        normalized.append(text)
    return normalized


def _as_latent_batch(
    value: Any,
    *,
    field_name: str,
) -> list[torch.Tensor | str | Path]:
    if isinstance(value, (str, Path)):
        return [value]
    if isinstance(value, torch.Tensor):
        if value.ndim == 2:
            return [value]
        if value.ndim == 3:
            return list(value.unbind(dim=0))
        raise ValueError(f"Irodori `{field_name}` tensor must have shape (T, D) or (B, T, D).")
    try:
        values = list(value)
    except TypeError as error:
        raise TypeError(f"Irodori `{field_name}` must contain tensors or Safetensors paths.") from error
    if not values:
        raise ValueError(f"Irodori `{field_name}` batch cannot be empty.")
    if any(not isinstance(item, (torch.Tensor, str, Path)) for item in values):
        raise TypeError(f"Irodori `{field_name}` must contain only tensors or Safetensors paths.")
    return values


def _pad_latents(
    latents: Sequence[torch.Tensor],
    *,
    latent_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not latents:
        raise ValueError("Irodori latent batch cannot be empty.")
    normalized = []
    for latent in latents:
        if not isinstance(latent, torch.Tensor):
            raise TypeError("Irodori latents must be torch.Tensor values.")
        if latent.ndim == 3 and latent.shape[0] == 1:
            latent = latent[0]
        if latent.ndim != 2:
            raise ValueError("Each Irodori training latent must have shape (T, D).")
        if latent.shape[0] == 0 or latent.shape[1] == 0:
            raise ValueError("Irodori training latents cannot be empty.")
        if not latent.is_floating_point():
            raise ValueError("Irodori training latents must be floating-point.")
        if not torch.isfinite(latent).all().item():
            raise ValueError("Irodori training latents must contain only finite values.")
        if latent.shape[-1] != latent_dim:
            if latent.shape[0] == latent_dim:
                latent = latent.transpose(0, 1)
            else:
                raise ValueError("Irodori training latent width is incompatible.")
        normalized.append(latent.float())
    maximum = max(latent.shape[0] for latent in normalized)
    values = torch.zeros(len(normalized), maximum, latent_dim, dtype=torch.float32)
    mask = torch.zeros(len(normalized), maximum, dtype=torch.bool)
    for index, latent in enumerate(normalized):
        values[index, :latent.shape[0]] = latent
        mask[index, :latent.shape[0]] = True
    return values, mask


def _materialize_latent_batch(values: Sequence[torch.Tensor | str | Path], ) -> list[torch.Tensor]:
    latents: list[torch.Tensor] = []
    for value in values:
        latent = (load_preencoded_latent(value) if isinstance(value, (str, Path)) else value.detach())
        if latent.ndim == 3:
            latents.extend(latent.unbind(dim=0))
        else:
            latents.append(latent)
    return latents


class IrodoriBatchProcessor:
    """Prepare raw waveform or pre-encoded latent batches without providers."""

    def __init__(
        self,
        *,
        config: IrodoriModelConfig,
        tokenizer: IrodoriTokenizer,
        codec: IrodoriDACVAECodec,
        device: str | torch.device,
        max_text_length: int = 256,
        max_caption_length: int = 512,
    ) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.codec = codec
        self.device = torch.device(device)
        self.max_text_length = int(max_text_length)
        self.max_caption_length = int(max_caption_length)

    def _target_latents(
        self,
        batch: Mapping[str, Any],
        *,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        provided = batch.get("target_latent", batch.get("latent"))
        if provided is not None:
            provided = _as_latent_batch(
                provided,
                field_name="target_latent",
            )
            latents = _materialize_latent_batch(provided)
            if len(latents) != batch_size:
                raise ValueError("Irodori target latent count must match text count.")
            return _pad_latents(latents, latent_dim=self.config.latent_dim)
        waveforms = batch.get("waveform", batch.get("audio"))
        if waveforms is None:
            raise ValueError("Irodori training requires waveform/audio or target_latent.")
        if isinstance(waveforms, torch.Tensor):
            if waveforms.ndim == 1:
                waveforms = [waveforms]
            elif waveforms.ndim == 2:
                waveforms = list(waveforms)
            else:
                raise ValueError("Irodori raw waveform batch must have shape (B, T).")
        else:
            waveforms = list(waveforms)
        if len(waveforms) != batch_size:
            raise ValueError("Irodori waveform count must match text count.")
        sample_rates = batch.get("sample_rate", self.codec.sample_rate)
        if isinstance(sample_rates, int):
            sample_rates = [sample_rates] * batch_size
        else:
            sample_rates = list(sample_rates)
        if len(sample_rates) != batch_size:
            raise ValueError("Irodori sample-rate count must match text count.")
        latents = []
        with torch.no_grad():
            for waveform, sample_rate in zip(waveforms, sample_rates):
                if not isinstance(waveform, torch.Tensor):
                    raise TypeError("Irodori raw waveforms must be torch.Tensor values.")
                if waveform.numel() == 0 or not torch.isfinite(waveform).all().item():
                    raise ValueError("Irodori raw waveforms must be non-empty and finite.")
                if (isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0):
                    raise ValueError("Irodori sample rates must be positive integers.")
                waveform = waveform.detach().float().flatten()
                if sample_rate != self.codec.sample_rate:
                    waveform = resample_waveform(
                        waveform,
                        sample_rate,
                        self.codec.sample_rate,
                    )
                latents.append(self.codec.encode_waveform(waveform)[0].cpu())
        return _pad_latents(latents, latent_dim=self.config.latent_dim)

    def _reference(
        self,
        batch: Mapping[str, Any],
        *,
        target_latent: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
        batch_size = target_latent.shape[0]
        if not self.config.use_speaker_condition_resolved:
            return None, None, torch.zeros(batch_size, dtype=torch.bool)
        provided = batch.get("reference_latent")
        use_target_as_reference = batch.get(
            "use_target_as_reference",
            provided is None,
        )
        if not isinstance(use_target_as_reference, bool):
            raise TypeError("`use_target_as_reference` must be a boolean.")
        used_target_as_reference = provided is None and use_target_as_reference
        if used_target_as_reference:
            provided = [target_latent[index, target_mask[index]] for index in range(batch_size)]
        if provided is None:
            steps = max(1, self.config.speaker_patch_size)
            return (
                torch.zeros(
                    batch_size,
                    steps,
                    self.config.patched_latent_dim,
                    dtype=torch.float32,
                ),
                torch.zeros(batch_size, steps, dtype=torch.bool),
                torch.zeros(batch_size, dtype=torch.bool),
            )
        provided = _as_latent_batch(
            provided,
            field_name="reference_latent",
        )
        latents = _materialize_latent_batch(provided)
        if len(latents) != batch_size:
            raise ValueError("Irodori reference latent count must match text count.")
        values, mask = _pad_latents(latents, latent_dim=self.config.latent_dim)
        patched = patchify_latent(values, self.config.latent_patch_size)
        patched_mask = mask[:, :patched.shape[1] * self.config.latent_patch_size]
        patched_mask = patched_mask.reshape(
            mask.shape[0],
            patched.shape[1],
            self.config.latent_patch_size,
        ).all(dim=-1)
        raw_has_speaker = batch.get("has_speaker")
        if raw_has_speaker is None:
            has_speaker = torch.full(
                (batch_size, ),
                not used_target_as_reference,
                dtype=torch.bool,
            )
        elif isinstance(raw_has_speaker, bool):
            has_speaker = torch.full(
                (batch_size, ),
                raw_has_speaker,
                dtype=torch.bool,
            )
        elif isinstance(raw_has_speaker, torch.Tensor):
            has_speaker = raw_has_speaker.detach().to(dtype=torch.bool).flatten()
        else:
            try:
                has_speaker = torch.tensor(
                    list(raw_has_speaker),
                    dtype=torch.bool,
                ).flatten()
            except (TypeError, ValueError) as error:
                raise TypeError("`has_speaker` must be a boolean or a sequence of booleans.") from error
        if has_speaker.shape != (batch_size, ):
            raise ValueError("Irodori `has_speaker` count must match text count.")
        return patched, patched_mask, has_speaker

    def __call__(self, batch: Mapping[str, Any]) -> dict[str, torch.Tensor | None]:
        if not isinstance(batch, Mapping):
            raise TypeError("Irodori training batch must be a mapping.")
        texts = _as_text_batch(batch.get("text"))
        batch_size = len(texts)
        token_rows, token_masks = self.tokenizer.encode_batch(
            texts,
            max_length=self.max_text_length,
        )
        target_latent, target_mask = self._target_latents(batch, batch_size=batch_size)
        patched_target = patchify_latent(target_latent, self.config.latent_patch_size)
        patched_mask = target_mask[:, :patched_target.shape[1] * self.config.latent_patch_size].reshape(
            batch_size,
            patched_target.shape[1],
            self.config.latent_patch_size,
        ).all(dim=-1)
        reference, reference_mask, has_speaker = self._reference(
            batch,
            target_latent=target_latent,
            target_mask=target_mask,
        )
        caption_ids = caption_mask = None
        captions: list[str] | None = None
        if self.config.use_caption_condition:
            raw_captions = batch.get("caption", [""] * batch_size)
            if isinstance(raw_captions, str):
                raw_captions = [raw_captions]
            captions = [str(value).strip() for value in raw_captions]
            if len(captions) != batch_size:
                raise ValueError("Irodori caption count must match text count.")
            rows = []
            masks = []
            for caption in captions:
                if caption:
                    ids = self.tokenizer.encode(
                        caption,
                        max_length=self.max_caption_length,
                    )
                    rows.append(ids)
                    masks.append((True, ) * len(ids))
                else:
                    rows.append((self.tokenizer.bos_token_id, ))
                    masks.append((False, ))
            maximum = max(len(row) for row in rows)
            caption_ids = torch.tensor(
                [row + (self.tokenizer.pad_token_id, ) * (maximum - len(row)) for row in rows],
                dtype=torch.long,
            )
            caption_mask = torch.tensor(
                [mask + (False, ) * (maximum - len(mask)) for mask in masks],
                dtype=torch.bool,
            )
        lengths = patched_mask.sum(dim=1)
        duration_features = build_duration_features(
            texts,
            token_counts=torch.tensor([sum(mask) for mask in token_masks]),
            max_text_len=self.max_text_length,
            has_speaker=has_speaker,
        )
        result = {
            "target_latent":
            patched_target,
            "latent_mask":
            patched_mask,
            "text_input_ids":
            torch.tensor(token_rows, dtype=torch.long),
            "text_mask":
            torch.tensor(token_masks, dtype=torch.bool),
            "ref_latent":
            reference,
            "ref_mask":
            reference_mask,
            "caption_input_ids":
            caption_ids,
            "caption_mask":
            caption_mask,
            "duration_features":
            duration_features,
            "duration_target":
            torch.log1p(lengths.float()),
            "duration_has_speaker":
            has_speaker,
            "duration_has_caption": (
                None if captions is None else torch.tensor([bool(value) for value in captions],
                                                           dtype=torch.bool)),
        }
        return {name: None if value is None else value.to(self.device) for name, value in result.items()}


def masked_flow_matching_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    mode: str = "utterance_mean",
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("Irodori flow prediction and target shapes differ.")
    if mask.shape != prediction.shape[:2]:
        raise ValueError("Irodori latent mask has an incompatible shape.")
    squared = (prediction.float() - target.float()).square()
    visible = mask.unsqueeze(-1).expand_as(squared)
    if not visible.any():
        raise ValueError("Irodori flow batch contains no visible latent frames.")
    if mode == "utterance_mean":
        numerator = (squared * visible).sum(dim=(1, 2))
        denominator = visible.sum(dim=(1, 2)).clamp_min(1)
        return (numerator / denominator).mean()
    if mode == "echo":
        return squared.masked_select(visible).mean()
    raise ValueError("Irodori RF loss mode must be 'utterance_mean' or 'echo'.")


def irodori_training_step(
    model: TextToLatentRFDiT,
    batch: Mapping[str, torch.Tensor | None],
    *,
    objective: str = "joint",
    rf_loss_mode: str = "utterance_mean",
    duration_loss_weight: float = 0.1,
    duration_huber_delta: float = 0.1,
    text_condition_dropout: float = 0.1,
    speaker_condition_dropout: float = 0.1,
    caption_condition_dropout: float = 0.1,
    timestep_logit_mean: float = 0.0,
    timestep_logit_std: float = 1.0,
    stratified_timesteps: bool = True,
) -> dict[str, torch.Tensor]:
    if not isinstance(model, TextToLatentRFDiT):
        raise TypeError("Irodori training requires the native RF-DiT graph.")
    objective = str(objective).strip().lower()
    if objective not in {"flow", "duration", "joint"}:
        raise ValueError("Irodori objective must be flow, duration, or joint.")
    target = batch.get("target_latent")
    mask = batch.get("latent_mask")
    if not isinstance(target, torch.Tensor) or not isinstance(mask, torch.Tensor):
        raise ValueError("Irodori training batch requires target_latent and latent_mask.")
    batch_size = target.shape[0]
    drop = lambda probability: torch.rand(batch_size, device=target.device) < float(probability)
    common = {
        "text_input_ids": batch["text_input_ids"],
        "text_mask": batch["text_mask"],
        "ref_latent": batch.get("ref_latent"),
        "ref_mask": batch.get("ref_mask"),
        "caption_input_ids": batch.get("caption_input_ids"),
        "caption_mask": batch.get("caption_mask"),
        "duration_features": batch.get("duration_features"),
        "duration_has_speaker": batch.get("duration_has_speaker"),
        "duration_has_caption": batch.get("duration_has_caption"),
    }
    losses: dict[str, torch.Tensor] = {}
    if objective == "duration":
        prediction = model(
            None,
            None,
            duration_only=True,
            **common,
        )
        duration_target = batch.get("duration_target")
        if not isinstance(duration_target, torch.Tensor):
            raise ValueError("Irodori duration objective requires duration_target.")
        duration_loss = F.huber_loss(
            prediction.float(),
            duration_target.float(),
            delta=float(duration_huber_delta),
        )
        return {
            "loss": duration_loss,
            "duration_loss": duration_loss,
            "duration_prediction": prediction,
        }
    timestep_sampler = (sample_stratified_logit_normal_t if stratified_timesteps else sample_logit_normal_t)
    timestep = timestep_sampler(
        batch_size,
        target.device,
        mean=timestep_logit_mean,
        std=timestep_logit_std,
    ).to(dtype=target.dtype)
    noise = torch.randn_like(target)
    x_t = rf_interpolate(target, noise, timestep)
    velocity_target = rf_velocity_target(target, noise)
    outputs = model(
        x_t,
        timestep,
        latent_mask=mask,
        text_condition_dropout=drop(text_condition_dropout),
        speaker_condition_dropout=drop(speaker_condition_dropout),
        caption_condition_dropout=(
            drop(caption_condition_dropout) if model.cfg.use_caption_condition else None),
        **common,
    )
    if isinstance(outputs, tuple):
        velocity, duration_prediction = outputs
    else:
        velocity, duration_prediction = outputs, None
    flow_loss = masked_flow_matching_loss(
        velocity,
        velocity_target,
        mask,
        mode=rf_loss_mode,
    )
    losses["flow_loss"] = flow_loss
    total = flow_loss
    if objective == "joint":
        if duration_prediction is None:
            if model.cfg.use_duration_predictor:
                raise RuntimeError("Irodori joint objective returned no duration prediction.")
        else:
            duration_target = batch.get("duration_target")
            if not isinstance(duration_target, torch.Tensor):
                raise ValueError("Irodori joint objective requires duration_target.")
            duration_loss = F.huber_loss(
                duration_prediction.float(),
                duration_target.float(),
                delta=float(duration_huber_delta),
            )
            losses["duration_loss"] = duration_loss
            total = total + float(duration_loss_weight) * duration_loss
    losses.update({
        "loss": total,
        "velocity": velocity,
        "velocity_target": velocity_target,
    })
    return losses


__all__ = [
    "IrodoriBatchProcessor",
    "irodori_training_step",
    "load_preencoded_latent",
    "masked_flow_matching_loss",
]
