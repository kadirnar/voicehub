"""Native inference lifecycle for Irodori-TTS."""

from __future__ import annotations

import math
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from voicehub.checkpointing import SafeTensorReader
from voicehub.optimization.protocols import OptimizationCompileTarget, OptimizationModuleRoot
from voicehub.processing.waveform import load_pcm_wave, resample_waveform

from .checkpoint import load_irodori_safetensors
from .codec import IrodoriDACVAECodec
from .conditioning import (
    load_speaker_inversion_payload,
    normalize_speaker_embedding_tensor,
    speaker_inversion_batch_tensors,
)
from .configuration import IrodoriModelConfig
from .duration import build_duration_features
from .flow_matching import sample_euler_rf_cfg
from .modeling import TextToLatentRFDiT
from .normalization import normalize_text
from .tokenization import IrodoriTokenizer


def resolve_runtime_device(value: str | torch.device) -> torch.device:
    if isinstance(value, torch.device):
        return value
    normalized = str(value).strip().lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested for Irodori but is unavailable.")
    return device


def resolve_runtime_dtype(precision: str) -> torch.dtype:
    normalized = str(precision).strip().lower()
    if normalized == "fp32":
        return torch.float32
    if normalized == "bf16":
        return torch.bfloat16
    raise ValueError("Irodori precision must be 'fp32' or 'bf16'.")


def patchify_latent(latent: torch.Tensor, patch_size: int) -> torch.Tensor:
    if patch_size <= 1:
        return latent
    usable = latent.shape[1] // patch_size * patch_size
    if usable == 0:
        raise ValueError("Irodori latent is shorter than one model patch.")
    return latent[:, :usable].reshape(
        latent.shape[0],
        usable // patch_size,
        latent.shape[2] * patch_size,
    )


def unpatchify_latent(
    latent: torch.Tensor,
    patch_size: int,
    latent_dim: int,
) -> torch.Tensor:
    if patch_size <= 1:
        return latent
    return latent.reshape(
        latent.shape[0],
        latent.shape[1] * patch_size,
        latent_dim,
    )


def find_flattening_point(
    latent: torch.Tensor,
    *,
    window_size: int = 20,
    std_threshold: float = 0.05,
    mean_threshold: float = 0.1,
) -> int:
    if latent.ndim != 2:
        raise ValueError("Irodori tail trimming expects a (time, latent) tensor.")
    if window_size <= 0:
        return latent.shape[0]
    padding = torch.zeros(
        window_size,
        latent.shape[1],
        device=latent.device,
        dtype=latent.dtype,
    )
    padded = torch.cat((latent, padding), dim=0)
    for index in range(padded.shape[0] - window_size):
        window = padded[index:index + window_size]
        if (window.std(unbiased=False) < std_threshold and window.mean().abs() < mean_threshold):
            return index
    return latent.shape[0]


@dataclass(frozen=True)
class RuntimeKey:
    checkpoint: str
    checkpoint_model_id: str | None = None
    checkpoint_revision: str | None = None
    model_device: str = "cpu"
    codec_repo: str = ""
    tokenizer_directory: str | None = None
    model_precision: str = "fp32"
    codec_device: str = "cpu"
    codec_precision: str = "fp32"
    compile_model: bool = False
    compile_dynamic: bool = False


@dataclass
class SamplingRequest:
    text: str
    caption: str | None = None
    ref_wav: str | None = None
    ref_latent: str | torch.Tensor | None = None
    ref_embed: str | None = None
    no_ref: bool = False
    ref_normalize_db: float | None = -16.0
    ref_ensure_max: bool = True
    num_candidates: int = 1
    decode_mode: str = "sequential"
    seconds: float | None = None
    duration_scale: float = 1.0
    min_seconds: float = 0.5
    max_seconds: float = 30.0
    max_ref_seconds: float | None = 30.0
    max_text_len: int | None = None
    max_caption_len: int | None = None
    num_steps: int = 40
    cfg_scale_text: float = 3.0
    cfg_scale_caption: float = 3.0
    cfg_scale_speaker: float = 5.0
    cfg_guidance_mode: str = "independent"
    cfg_scale: float | None = None
    cfg_min_t: float = 0.5
    cfg_max_t: float = 1.0
    truncation_factor: float | None = None
    rescale_k: float | None = None
    rescale_sigma: float | None = None
    context_kv_cache: bool = True
    speaker_kv_scale: float | None = None
    speaker_kv_min_t: float | None = None
    speaker_kv_max_layers: int | None = None
    speaker_uncond_mode: str = "mask"
    seed: int | None = None
    t_schedule_mode: str = "linear"
    sway_coeff: float = -1.0
    trim_tail: bool = True
    tail_window_size: int = 20
    tail_std_threshold: float = 0.05
    tail_mean_threshold: float = 0.1
    watermark: bool = False
    lora_adapter: str | None = None


@dataclass
class SamplingResult:
    audio: torch.Tensor
    audios: list[torch.Tensor]
    sample_rate: int
    used_seed: int
    stage_timings: list[tuple[str, float]] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)


class InferenceRuntime:
    """Own the native model, tokenizer, codec, and deterministic protocol."""

    def __init__(
        self,
        *,
        model: TextToLatentRFDiT,
        model_cfg: IrodoriModelConfig,
        tokenizer: IrodoriTokenizer,
        codec: IrodoriDACVAECodec,
        model_device: str | torch.device,
    ) -> None:
        self.model = model
        self.model_cfg = model_cfg
        self.tokenizer = tokenizer
        self.caption_tokenizer = tokenizer if model_cfg.use_caption_condition else None
        self.codec = codec
        self.model_device = resolve_runtime_device(model_device)
        self.codec_device = codec.device
        self.default_text_max_len = 256
        self.default_caption_max_len = 512 if model_cfg.use_caption_condition else 256

    def optimization_compile_targets(
        self,
        mode: str,
    ) -> tuple[OptimizationCompileTarget, ...]:
        """Expose the RF-DiT boundary used by sampling or fine-tuning."""
        if mode == "training":
            return (OptimizationCompileTarget(
                "model.forward",
                self.model,
                "forward",
            ), )
        if mode != "inference":
            raise ValueError(f"Unsupported optimization mode {mode!r}.")
        return (
            OptimizationCompileTarget(
                "model.forward_with_encoded_conditions",
                self.model,
                "forward_with_encoded_conditions",
            ),
            OptimizationCompileTarget(
                "codec.decode_latent",
                self.codec,
                "decode_latent",
            ),
        )

    def optimization_module_roots(self, ) -> tuple[OptimizationModuleRoot, ...]:
        """Expose the native DiT and codec without reparenting either graph."""
        codec_root = (self.codec if isinstance(self.codec, torch.nn.Module) else self.codec.model)
        return (
            OptimizationModuleRoot("model", self.model),
            OptimizationModuleRoot("codec", codec_root),
        )

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return stable runtime-owned checkpoint keys for selector safety."""
        codec_root = (self.codec if isinstance(self.codec, torch.nn.Module) else self.codec.model)
        state = {f"model.{name}": value for name, value in self.model.state_dict().items()}
        state.update({f"codec.{name}": value for name, value in codec_root.state_dict().items()})
        return state

    @classmethod
    def from_key(cls, key: RuntimeKey) -> InferenceRuntime:
        model_device = resolve_runtime_device(key.model_device)
        codec_device = resolve_runtime_device(key.codec_device)
        model, model_cfg = load_irodori_safetensors(
            key.checkpoint,
            device=model_device,
            dtype=resolve_runtime_dtype(key.model_precision),
            model_id=key.checkpoint_model_id,
            revision=key.checkpoint_revision,
        )
        tokenizer_directory = Path(key.tokenizer_directory or Path(key.checkpoint).parent).expanduser()
        tokenizer = IrodoriTokenizer.from_files(
            tokenizer_directory / "tokenizer.json",
            tokenizer_config=(
                tokenizer_directory / "tokenizer_config.json" if
                (tokenizer_directory / "tokenizer_config.json").is_file() else None),
            expected_vocabulary_size=model_cfg.text_vocab_size,
        )
        codec_path = Path(key.codec_repo).expanduser()
        if codec_path.is_dir():
            codec_path = codec_path / "weights.pth"
        codec = IrodoriDACVAECodec.from_checkpoint(
            codec_path,
            device=codec_device,
            dtype=resolve_runtime_dtype(key.codec_precision),
        )
        if model_cfg.latent_dim != codec.latent_dim:
            raise ValueError("Irodori model and codec latent dimensions differ.")
        if key.compile_model:
            if not hasattr(torch, "compile"):
                raise RuntimeError("Irodori compilation requires torch.compile.")
            model = torch.compile(model, dynamic=bool(key.compile_dynamic))
        model.eval()
        return cls(
            model=model,
            model_cfg=model_cfg,
            tokenizer=tokenizer,
            codec=codec,
            model_device=model_device,
        )

    def _batch_tokens(
        self,
        text: str,
        *,
        batch_size: int,
        max_length: int,
        allow_empty: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not text and allow_empty:
            ids = torch.full(
                (batch_size, 1),
                self.tokenizer.bos_token_id,
                dtype=torch.long,
                device=self.model_device,
            )
            return ids, torch.zeros_like(ids, dtype=torch.bool)
        rows, masks = self.tokenizer.encode_batch(
            [text] * batch_size,
            max_length=max_length,
        )
        return (
            torch.tensor(rows, dtype=torch.long, device=self.model_device),
            torch.tensor(masks, dtype=torch.bool, device=self.model_device),
        )

    def _load_preencoded_latent(self, value: str | torch.Tensor) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            latent = value.detach()
        else:
            path = Path(value).expanduser().resolve()
            if path.suffix.lower() != ".safetensors":
                raise ValueError("Pre-encoded Irodori latents must use Safetensors.")
            with SafeTensorReader(path) as reader:
                keys = set(reader.keys())
                if keys != {"latent"}:
                    raise ValueError("Irodori latent artifact must contain only `latent`.")
                latent = reader.get_tensor("latent")
        if latent.ndim == 2:
            latent = latent.unsqueeze(0)
        if latent.ndim != 3:
            raise ValueError("Irodori latent must have shape (B, T, D) or (T, D).")
        if latent.shape[-1] != self.model_cfg.latent_dim:
            if latent.shape[1] == self.model_cfg.latent_dim:
                latent = latent.transpose(1, 2)
            else:
                raise ValueError("Irodori pre-encoded latent has an incompatible width.")
        if not latent.is_floating_point() or not torch.isfinite(latent).all():
            raise ValueError("Irodori pre-encoded latent must be finite and floating-point.")
        return latent.float().contiguous()

    def _reference(
        self,
        request: SamplingRequest,
        *,
        batch_size: int,
    ) -> tuple[
            torch.Tensor | None,
            torch.Tensor | None,
            torch.Tensor | None,
            torch.Tensor | None,
    ]:
        if not self.model_cfg.use_speaker_condition_resolved:
            if any((request.ref_wav, request.ref_latent, request.ref_embed)):
                raise ValueError("This Irodori checkpoint has no speaker-conditioning branch.")
            return None, None, None, None
        if request.ref_embed is not None:
            embedding = load_speaker_inversion_payload(request.ref_embed)["speaker_embedding"]
            embedding = normalize_speaker_embedding_tensor(
                embedding,
                speaker_dim=self.model_cfg.speaker_dim,
            )
            state, mask = speaker_inversion_batch_tensors(
                embedding,
                batch_size,
                self.model_device,
                self.model.dtype,
            )
            return None, None, state, mask
        if request.no_ref:
            steps = max(1, self.model_cfg.speaker_patch_size)
            latent = torch.zeros(
                batch_size,
                steps,
                self.model_cfg.patched_latent_dim,
                device=self.model_device,
                dtype=self.model.dtype,
            )
            mask = torch.zeros(
                batch_size,
                steps,
                dtype=torch.bool,
                device=self.model_device,
            )
            return latent, mask, None, None
        if request.ref_latent is not None:
            latent = self._load_preencoded_latent(request.ref_latent)
        elif request.ref_wav is not None:
            waveform, sample_rate = load_pcm_wave(request.ref_wav)
            if request.max_ref_seconds is not None:
                waveform = waveform[:max(1, int(float(request.max_ref_seconds) * sample_rate))]
            if sample_rate != self.codec.sample_rate:
                waveform = resample_waveform(
                    waveform,
                    sample_rate,
                    self.codec.sample_rate,
                )
            latent = self.codec.encode_waveform(
                waveform,
                normalize_db=request.ref_normalize_db,
            ).cpu()
        else:
            raise ValueError("Supply a reference or set `no_reference=True`.")
        patched = patchify_latent(latent, self.model_cfg.latent_patch_size).to(
            device=self.model_device,
            dtype=self.model.dtype,
        )
        if patched.shape[0] == 1 and batch_size > 1:
            patched = patched.expand(batch_size, -1, -1)
        if patched.shape[0] != batch_size:
            raise ValueError("Reference latent batch does not match candidate count.")
        mask = torch.ones(
            patched.shape[:2],
            dtype=torch.bool,
            device=self.model_device,
        )
        return patched, mask, None, None

    @torch.inference_mode()
    def synthesize(self, request: SamplingRequest) -> SamplingResult:
        if not isinstance(request, SamplingRequest):
            raise TypeError("Irodori synthesis requires SamplingRequest.")
        if request.watermark:
            raise ValueError(
                "Native Irodori does not claim SilentCipher parity; "
                "`watermark=True` is rejected.")
        if request.lora_adapter is not None:
            raise ValueError(
                "Provider PEFT adapters are not accepted. Export a merged native "
                "Safetensors checkpoint before inference.")
        text = normalize_text(request.text).strip()
        if not text:
            raise ValueError("Irodori text became empty after normalization.")
        batch_size = int(request.num_candidates)
        if batch_size <= 0:
            raise ValueError("`num_candidates` must be positive.")
        max_text_len = request.max_text_len or self.default_text_max_len
        max_caption_len = request.max_caption_len or self.default_caption_max_len
        timings = []
        started = time.perf_counter()
        text_ids, text_mask = self._batch_tokens(
            text,
            batch_size=batch_size,
            max_length=max_text_len,
        )
        caption_ids = caption_mask = None
        caption = "" if request.caption is None else request.caption.strip()
        if self.model_cfg.use_caption_condition:
            caption_ids, caption_mask = self._batch_tokens(
                caption,
                batch_size=batch_size,
                max_length=max_caption_len,
                allow_empty=True,
            )
        elif caption:
            raise ValueError("This Irodori checkpoint has no caption-conditioning branch.")
        timings.append(("tokenize", time.perf_counter() - started))
        reference_started = time.perf_counter()
        ref_latent, ref_mask, speaker_state, speaker_mask = self._reference(
            request,
            batch_size=batch_size,
        )
        timings.append(("reference", time.perf_counter() - reference_started))

        if request.seconds is not None:
            seconds = min(
                float(request.max_seconds),
                max(float(request.min_seconds), float(request.seconds)),
            )
            latent_steps = math.ceil(seconds * self.codec.sample_rate / self.codec.hop_length)
        elif self.model_cfg.use_duration_predictor:
            (
                text_state,
                encoded_text_mask,
                encoded_speaker,
                encoded_speaker_mask,
                encoded_caption,
                encoded_caption_mask,
            ) = self.model.encode_conditions(
                text_input_ids=text_ids,
                text_mask=text_mask,
                ref_latent=ref_latent,
                ref_mask=ref_mask,
                caption_input_ids=caption_ids,
                caption_mask=caption_mask,
                speaker_state_override=speaker_state,
                speaker_mask_override=speaker_mask,
            )
            has_speaker = (
                torch.zeros(batch_size, dtype=torch.bool, device=self.model_device)
                if encoded_speaker_mask is None else encoded_speaker_mask.any(dim=1))
            features = build_duration_features(
                [text] * batch_size,
                token_counts=text_mask.sum(dim=1),
                max_text_len=max_text_len,
                has_speaker=has_speaker,
            ).to(self.model_device)
            prediction = self.model.predict_duration_log_frames(
                text_state=text_state,
                text_mask=encoded_text_mask,
                speaker_state=encoded_speaker,
                speaker_mask=encoded_speaker_mask,
                caption_state=encoded_caption,
                caption_mask=encoded_caption_mask,
                duration_features=features,
                has_speaker=has_speaker,
                has_caption=(
                    torch.full(
                        (batch_size, ),
                        bool(caption),
                        dtype=torch.bool,
                        device=self.model_device,
                    ) if self.model_cfg.use_caption_condition else None),
            )
            latent_steps = max(
                1,
                round(torch.expm1(prediction.float()).mean().item() * float(request.duration_scale)),
            )
            minimum = math.ceil(request.min_seconds * self.codec.sample_rate / self.codec.hop_length)
            maximum = math.floor(request.max_seconds * self.codec.sample_rate / self.codec.hop_length)
            latent_steps = min(maximum, max(minimum, latent_steps))
        else:
            latent_steps = math.ceil(30.0 * self.codec.sample_rate / self.codec.hop_length)
        target_samples = latent_steps * self.codec.hop_length
        patched_steps = math.ceil(latent_steps / self.model_cfg.latent_patch_size)
        used_seed = secrets.randbits(63) if request.seed is None else int(request.seed)
        sampled_started = time.perf_counter()
        sampled = sample_euler_rf_cfg(
            model=self.model,
            text_input_ids=text_ids,
            text_mask=text_mask,
            ref_latent=ref_latent,
            ref_mask=ref_mask,
            sequence_length=patched_steps,
            caption_input_ids=caption_ids,
            caption_mask=caption_mask,
            speaker_state_override=speaker_state,
            speaker_mask_override=speaker_mask,
            speaker_uncond_mode=request.speaker_uncond_mode,
            num_steps=request.num_steps,
            cfg_scale_text=request.cfg_scale_text,
            cfg_scale_caption=request.cfg_scale_caption,
            cfg_scale_speaker=request.cfg_scale_speaker,
            cfg_guidance_mode=request.cfg_guidance_mode,
            cfg_min_t=request.cfg_min_t,
            cfg_max_t=request.cfg_max_t,
            seed=used_seed,
            cfg_scale=request.cfg_scale,
            truncation_factor=request.truncation_factor,
            rescale_k=request.rescale_k,
            rescale_sigma=request.rescale_sigma,
            use_context_kv_cache=request.context_kv_cache,
            speaker_kv_scale=request.speaker_kv_scale,
            speaker_kv_max_layers=request.speaker_kv_max_layers,
            speaker_kv_min_t=request.speaker_kv_min_t,
            t_schedule_mode=request.t_schedule_mode,
            sway_coeff=request.sway_coeff,
        )
        timings.append(("sample", time.perf_counter() - sampled_started))
        latents = unpatchify_latent(
            sampled,
            self.model_cfg.latent_patch_size,
            self.model_cfg.latent_dim,
        )[:, :latent_steps]
        decoded_started = time.perf_counter()
        decoded = self.codec.decode_latent(latents).cpu()
        audios = []
        for index, waveform in enumerate(decoded):
            maximum = target_samples
            if request.trim_tail:
                endpoint = find_flattening_point(
                    latents[index],
                    window_size=request.tail_window_size,
                    std_threshold=request.tail_std_threshold,
                    mean_threshold=request.tail_mean_threshold,
                )
                if endpoint > 0:
                    maximum = min(maximum, endpoint * self.codec.hop_length)
            audios.append(waveform[:maximum].contiguous())
        timings.append(("decode", time.perf_counter() - decoded_started))
        return SamplingResult(
            audio=audios[0],
            audios=audios,
            sample_rate=self.codec.sample_rate,
            used_seed=used_seed,
            stage_timings=timings,
            messages=[],
        )


__all__ = [
    "InferenceRuntime",
    "RuntimeKey",
    "SamplingRequest",
    "SamplingResult",
    "find_flattening_point",
    "patchify_latent",
    "resolve_runtime_device",
    "resolve_runtime_dtype",
    "unpatchify_latent",
]
