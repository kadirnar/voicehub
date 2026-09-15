from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Tuple

import torch

from voicehub.checkpointing import SafeTensorReader
from voicehub.hub import resolve_pretrained_file
from voicehub.models.echo.autoencoder import DAC, build_ae
from voicehub.models.echo.model import EchoDiT
from voicehub.optimization.diffusion_sampling import DiffusionStepContext
from voicehub.processing.waveform import load_native_audio, save_pcm_wave


def _resolve_checkpoint_file(
    name_or_path: str | Path,
    filename: str,
    *,
    token: str | None = None,
) -> Path:
    """Resolve one Echo artifact through VoiceHub's native Hub transport."""
    return resolve_pretrained_file(
        name_or_path,
        filename,
        token=token,
    )


def _load_safetensors(
        path: str | Path,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        excluded_prefixes: tuple[str, ...] = (),
        excluded_fragments: tuple[str, ...] = (),
) -> dict[str, torch.Tensor]:
    """Materialize a validated Echo state dictionary without third-party
    I/O."""
    with SafeTensorReader(path) as reader:
        names = tuple(
            name for name in reader.keys() if not name.startswith(excluded_prefixes) and not any(
                fragment in name for fragment in excluded_fragments))
        return {
            name: reader.get_tensor(
                name,
                device=device,
                dtype=dtype,
            )
            for name in names
        }


def _assign_validated_state(
        model: torch.nn.Module,
        state: dict[str, torch.Tensor],
        *,
        allowed_missing_prefixes: tuple[str, ...] = (),
        allowed_missing_fragments: tuple[str, ...] = (),
) -> None:
    """Assign a safe state only when its complete tensor inventory is known."""
    expected = set(model.state_dict())
    # Released Fish quantizers use torch's legacy weight_norm spelling.
    # Translate only aliases whose modern parametrization is in this graph;
    # retain strict inventory checks and reject ambiguous duplicate tensors.
    normalized = {}
    for name, value in state.items():
        destination = name
        for old_suffix, new_suffix in (
            (".weight_g", ".parametrizations.weight.original0"),
            (".weight_v", ".parametrizations.weight.original1"),
        ):
            if name.endswith(old_suffix):
                candidate = name[:-len(old_suffix)] + new_suffix
                if candidate in expected:
                    destination = candidate
                break
        if destination in normalized:
            raise RuntimeError(f"Echo checkpoint contains duplicate weight aliases for {destination}.")
        normalized[destination] = value
    state = normalized
    supplied = set(state)
    allowed_missing = {
        name
        for name in expected if name.startswith(allowed_missing_prefixes) or any(
            fragment in name for fragment in allowed_missing_fragments)
    }
    missing = expected - supplied
    unexpected = supplied - expected
    if missing != allowed_missing or unexpected:
        details = []
        unapproved_missing = sorted(missing - allowed_missing)
        unexpectedly_present = sorted(unexpected)
        expected_but_present = sorted(allowed_missing - missing)
        if unapproved_missing:
            details.append("missing: " + ", ".join(unapproved_missing[:8]))
        if unexpectedly_present:
            details.append("unexpected: " + ", ".join(unexpectedly_present[:8]))
        if expected_but_present:
            details.append(
                "excluded tensors were unexpectedly supplied: " + ", ".join(expected_but_present[:8]))
        raise RuntimeError(
            "Echo checkpoint tensor inventory does not match the pinned "
            f"architecture ({'; '.join(details)}).")
    incompatible = model.load_state_dict(
        state,
        strict=False,
        assign=True,
    )
    if (set(incompatible.missing_keys) != allowed_missing or incompatible.unexpected_keys):
        raise RuntimeError(
            "PyTorch reported a different Echo checkpoint inventory after "
            "validated assignment.")


def _discard_blockwise_only_modules(model: EchoDiT) -> None:
    """Remove weights that the non-blockwise sampler intentionally omits."""
    del model.latent_encoder
    del model.latent_norm
    for block in model.blocks:
        del block.attention.wk_latent
        del block.attention.wv_latent
    model.blockwise_generation_available = False


def load_model_from_hf(
        repo_id: str | Path = "jordand/echo-tts-base",
        device: str = "cuda",
        dtype: torch.dtype | None = torch.bfloat16,
        compile: bool = False,
        token: str | None = None,
        delete_blockwise_modules: bool = False) -> EchoDiT:
    with torch.device("meta"):
        model = EchoDiT(
            latent_size=80,
            model_size=2048,
            num_layers=24,
            num_heads=16,
            intermediate_size=5888,
            norm_eps=1e-5,
            text_vocab_size=256,
            text_model_size=1280,
            text_num_layers=14,
            text_num_heads=10,
            text_intermediate_size=3328,
            speaker_patch_size=4,
            speaker_model_size=1280,
            speaker_num_layers=14,
            speaker_num_heads=10,
            speaker_intermediate_size=3328,
            timestep_embed_size=512,
            adaln_rank=256,
        )
    w_path = _resolve_checkpoint_file(
        repo_id,
        "pytorch_model.safetensors",
        token=token,
    )
    state = _load_safetensors(
        w_path,
        device=device,
        dtype=dtype,
        excluded_prefixes=(("latent_encoder.", "latent_norm") if delete_blockwise_modules else ()),
        excluded_fragments=((".wk_latent", ".wv_latent") if delete_blockwise_modules else ()),
    )

    missing_prefixes = (("latent_encoder.", "latent_norm") if delete_blockwise_modules else ())
    missing_fragments = ((".wk_latent", ".wv_latent") if delete_blockwise_modules else ())
    _assign_validated_state(
        model,
        state,
        allowed_missing_prefixes=missing_prefixes,
        allowed_missing_fragments=missing_fragments,
    )
    if delete_blockwise_modules:
        _discard_blockwise_only_modules(model)
    model = model.eval()

    if compile:
        model = compile_model(model)

    return model


def compile_model(model: EchoDiT) -> EchoDiT:
    model = torch.compile(model)
    model.get_kv_cache_text = torch.compile(model.get_kv_cache_text)
    model.get_kv_cache_speaker = torch.compile(model.get_kv_cache_speaker)
    if getattr(model, "blockwise_generation_available", True):
        model.get_kv_cache_latent = torch.compile(model.get_kv_cache_latent)
    return model


def load_fish_ae_from_hf(
        repo_id: str | Path = "jordand/fish-s1-dac-min",
        device: str = "cuda",
        dtype: torch.dtype | None = torch.float32,
        compile: bool = False,
        token: str | None = None) -> DAC:

    with torch.device("meta"):
        fish_ae = build_ae()

    w_path = _resolve_checkpoint_file(
        repo_id,
        "pytorch_model.safetensors",
        token=token,
    )
    state = _load_safetensors(
        w_path,
        device=device,
        dtype=dtype,
    )
    _assign_validated_state(fish_ae, state)

    fish_ae = fish_ae.eval().to(device)

    if compile:
        fish_ae = compile_fish_ae(fish_ae)

    return fish_ae


def compile_fish_ae(fish_ae: DAC) -> DAC:
    fish_ae.quantizer.upsample = torch.compile(fish_ae.quantizer.upsample)
    fish_ae.quantizer.downsample = torch.compile(fish_ae.quantizer.downsample)
    fish_ae.quantizer.pre_module = torch.compile(fish_ae.quantizer.pre_module)
    fish_ae.quantizer.post_module = torch.compile(fish_ae.quantizer.post_module)
    return fish_ae


@dataclass
class PCAState:
    pca_components: torch.Tensor
    pca_mean: torch.Tensor
    latent_scale: float


def load_pca_state_from_hf(
        repo_id: str | Path = "jordand/echo-tts-base",
        device: str = "cuda",
        filename: str = "pca_state.safetensors",
        token: str | None = None) -> PCAState:
    p_path = _resolve_checkpoint_file(
        repo_id,
        filename,
        token=token,
    )
    t = _load_safetensors(p_path, device=device)
    return PCAState(
        pca_components=t["pca_components"],
        pca_mean=t["pca_mean"],
        latent_scale=float(t["latent_scale"].item()),
    )


# ________


def load_audio(path: str, max_duration: int = 300) -> torch.Tensor:
    if isinstance(max_duration, bool) or not isinstance(max_duration, int):
        raise TypeError("`max_duration` must be an integer number of seconds.")
    if max_duration <= 0:
        raise ValueError("`max_duration` must be positive.")
    materialized = load_native_audio(
        path,
        target_sampling_rate=44_100,
    )
    audio = materialized.waveform[:max_duration * 44_100].unsqueeze(0)
    denominator = audio.abs().max().clamp_min(1.0)
    return audio / denominator


def tokenizer_encode(
        text: str,
        append_bos: bool = True,
        normalize: bool = True,
        return_normalized_text: bool = False) -> torch.Tensor | Tuple[torch.Tensor, str]:

    if normalize:
        text = text.replace("…", "...")
        text = text.replace('’', "'")
        text = text.replace('”', '"')
        text = text.replace('”', '"')
        text = text.replace("\n", " ")
        text = text.replace(":", ",")
        text = text.replace(";", ",")
        text = text.replace("—", ", ")
        if not text.startswith("[") and not text.startswith("(") and 'S1' not in text and 'S2' not in text:
            text = "[S1] " + text

    b = list(text.encode("utf-8"))
    if append_bos:
        b.insert(0, 0)

    if return_normalized_text:
        return torch.tensor(b), text

    return torch.tensor(b)


def get_text_input_ids_and_mask(
    text_arr: List[str],
    max_length: int | None,
    device: str | None = None,
    normalize: bool = True,
    return_normalized_text: bool = False,
    pad_to_max: bool = True
) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, List[str]]:
    encoded_texts = [
        tokenizer_encode(text, normalize=normalize, return_normalized_text=True) for text in text_arr
    ]

    if max_length is None:
        max_length = max(len(enc) for enc, _ in encoded_texts)

    tokens = torch.zeros((len(text_arr), max_length), dtype=torch.int32)
    mask = torch.zeros((len(text_arr), max_length), dtype=torch.bool)

    for i, (encoded, _) in enumerate(encoded_texts):
        length = min(len(encoded), max_length)
        tokens[i, :length] = encoded[:length]
        mask[i, :length] = 1

    if not pad_to_max and max_length is not None:
        tokens, mask = tokens[:, :max_length], mask[:, :max_length]

    if device is not None:
        tokens, mask = tokens.to(device), mask.to(device)

    if return_normalized_text:
        return tokens, mask, [text for _, text in encoded_texts]
    return tokens, mask


# ________


@torch.inference_mode()
def ae_encode(fish_ae: DAC, pca_state: PCAState, audio: torch.Tensor) -> torch.Tensor:
    assert audio.ndim == 3 and audio.shape[1] == 1  # (b, 1, length)
    z_q = fish_ae.encode_zq(audio).float()
    z_q = (z_q.transpose(1, 2) - pca_state.pca_mean) @ pca_state.pca_components.T
    z_q = z_q * pca_state.latent_scale
    return z_q


@torch.inference_mode()
def ae_decode(fish_ae: DAC, pca_state: PCAState, z_q: torch.Tensor) -> torch.Tensor:
    z_q = (z_q / pca_state.latent_scale) @ pca_state.pca_components + pca_state.pca_mean
    return fish_ae.decode_zq(z_q.transpose(1, 2).to(fish_ae.dtype)).float()


@torch.inference_mode()
def ae_reconstruct(fish_ae: DAC, pca_state: PCAState, audio: torch.Tensor) -> torch.Tensor:
    assert audio.ndim == 3 and audio.shape[1] == 1  # (b, 1, length)
    z_q = ae_encode(fish_ae, pca_state, audio.to(fish_ae.dtype))
    return ae_decode(fish_ae, pca_state, z_q)


# ________


@torch.inference_mode()
def get_speaker_latent_and_mask(
    fish_ae: DAC,
    pca_state: PCAState,
    audio: torch.Tensor,  # (1, length)
    max_speaker_latent_length: int = 6400,  # pretrained max length
    audio_chunk_size: int = 640 *
    2048,  # (~30 seconds, 1/10 max speaker condition size; max chunk seen in training)
    pad_to_max: bool = False,
    divis_by_patch_size: int | None = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # gets speaker latent and mask from audio, computes in chunks and concatenates (similar to training setup)

    AE_DOWNSAMPLE_FACTOR = 2048
    max_audio_len_length = max_speaker_latent_length * AE_DOWNSAMPLE_FACTOR

    assert audio.ndim == 2 and audio.shape[0] == 1  # (1, length)
    audio = audio[:, :max_audio_len_length]

    latent_arr = []

    for i in range(0, audio.shape[1], audio_chunk_size):
        audio_chunk = audio[:, i:i + audio_chunk_size]
        if audio_chunk.shape[1] < audio_chunk_size:
            audio_chunk = torch.nn.functional.pad(audio_chunk, (0, audio_chunk_size - audio_chunk.shape[1]))

        latent_chunk = ae_encode(fish_ae, pca_state, audio_chunk.unsqueeze(0))
        latent_arr.append(latent_chunk)

    speaker_latent = torch.cat(latent_arr, dim=1)

    actual_latent_length = audio.shape[1] // AE_DOWNSAMPLE_FACTOR
    speaker_mask = (
        torch.arange(speaker_latent.shape[1], device=speaker_latent.device)
        < actual_latent_length).unsqueeze(0)

    if pad_to_max and speaker_latent.shape[1] < max_speaker_latent_length:
        speaker_latent = torch.nn.functional.pad(
            speaker_latent, (0, 0, 0, max_speaker_latent_length - speaker_latent.shape[1]))
        speaker_mask = torch.nn.functional.pad(
            speaker_mask, (0, max_speaker_latent_length - speaker_mask.shape[1]))
    elif not pad_to_max:
        speaker_latent = speaker_latent[:, :actual_latent_length]
        speaker_mask = speaker_mask[:, :actual_latent_length]

    if divis_by_patch_size is not None:
        speaker_latent = speaker_latent[:, :speaker_latent.shape[1] // divis_by_patch_size *
                                        divis_by_patch_size]
        speaker_mask = speaker_mask[:, :speaker_mask.shape[1] // divis_by_patch_size * divis_by_patch_size]

    return speaker_latent, speaker_mask


# ________


def find_flattening_point(data, target_value=0.0, window_size=20, std_threshold=0.05):
    # simple heuristic to find end of latent generations; slow and can be improved
    # (data is (length, 80))
    padded_data = torch.cat(
        [data, torch.zeros(window_size, *data.shape[1:], device=data.device, dtype=data.dtype)])
    for i in range(len(padded_data) - window_size):
        window = padded_data[i:i + window_size]
        if window.std() < std_threshold and abs(window.mean() - target_value) < 0.1:
            return i
    return len(data)


def crop_audio_to_flattening_point(audio: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
    # (audio is (..., length), latent is (length, 80))
    flattening_point = find_flattening_point(latent)
    return audio[..., :flattening_point * 2048]


SampleFn = Callable[[EchoDiT, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int], torch.Tensor]


@torch.inference_mode()
def sample_pipeline(
    model: EchoDiT,
    fish_ae: DAC,
    pca_state: PCAState,
    sample_fn: SampleFn,
    text_prompt: str,
    speaker_audio: torch.Tensor | None,
    rng_seed: int,
    pad_to_max_speaker_latent_length: int | None = None,
    pad_to_max_text_length: int | None = None,
    normalize_text: bool = True,
) -> Tuple[torch.Tensor, str]:

    MAX_SPEAKER_LATENT_LENGTH = 6400  # max seen during training, though maybe can go higher?
    MAX_TEXT_LENGTH = 768

    device, dtype = model.device, model.dtype

    text_input_ids, text_mask, normalized_text = get_text_input_ids_and_mask(
        [text_prompt],
        max_length=min(pad_to_max_text_length or MAX_TEXT_LENGTH, MAX_TEXT_LENGTH),
        device=device,
        normalize=normalize_text,
        return_normalized_text=True,
        pad_to_max=(pad_to_max_text_length is not None))

    if speaker_audio is None:
        speaker_latent = torch.zeros((1, pad_to_max_speaker_latent_length or 4, 80),
                                     device=device,
                                     dtype=dtype)
        speaker_mask = torch.zeros((1, pad_to_max_speaker_latent_length or 4),
                                   device=device,
                                   dtype=torch.bool)
    else:
        speaker_latent, speaker_mask = get_speaker_latent_and_mask(
            fish_ae,
            pca_state,
            speaker_audio.to(fish_ae.dtype).to(device),
            max_speaker_latent_length=pad_to_max_speaker_latent_length or MAX_SPEAKER_LATENT_LENGTH,
            pad_to_max=(pad_to_max_speaker_latent_length is not None))

    latent_out = sample_fn(model, speaker_latent, speaker_mask, text_input_ids, text_mask, rng_seed)

    audio_out = ae_decode(fish_ae, pca_state, latent_out)

    audio_out = crop_audio_to_flattening_point(audio_out, latent_out[0])

    return audio_out, normalized_text[0]


# ________

KVCache = List[Tuple[torch.Tensor, torch.Tensor]]


def _concat_kv_caches(*caches: KVCache) -> KVCache:
    # helper that concatenates multiple KV caches along the batch dimension
    num_layers = len(caches[0])
    result = []
    for i in range(num_layers):
        k = torch.cat([c[i][0] for c in caches], dim=0)
        v = torch.cat([c[i][1] for c in caches], dim=0)
        result.append((k, v))
    return result


def _multiply_kv_cache(cache: KVCache, scale: float, max_layers: int | None = None) -> None:
    # helper that multiplies KV cache values in-place, for kv speaker scaling
    num_layers = len(cache) if max_layers is None else min(max_layers, len(cache))
    for i in range(num_layers):
        k, v = cache[i]
        k.mul_(scale)
        v.mul_(scale)


def _temporal_score_rescale(
        v_pred: torch.Tensor, x_t: torch.Tensor, t: float, rescale_k: float,
        rescale_sigma: float) -> torch.Tensor:
    # for https://arxiv.org/pdf/2510.01184
    if t < 1:
        snr = (1 - t)**2 / (t**2)
        ratio = (snr * rescale_sigma**2 + 1) / (snr * rescale_sigma**2 / rescale_k + 1)
        return 1 / (1 - t) * (ratio * ((1 - t) * v_pred + x_t) - x_t)
    return v_pred


@torch.inference_mode()
def sample_euler_cfg_independent_guidances(
    model: EchoDiT,
    speaker_latent: torch.Tensor,
    speaker_mask: torch.Tensor,
    text_input_ids: torch.Tensor,
    text_mask: torch.Tensor,
    rng_seed: int,
    num_steps: int,
    cfg_scale_text: float,
    cfg_scale_speaker: float,
    cfg_min_t: float,
    cfg_max_t: float,
    truncation_factor: float | None,
    rescale_k: float | None,
    rescale_sigma: float | None,
    speaker_kv_scale: float | None,
    speaker_kv_max_layers: int | None,
    speaker_kv_min_t: float | None,
    sequence_length: int | None = None,
) -> torch.Tensor:

    if sequence_length is None:
        sequence_length = 640  # max sequence length during training
    model.reset_diffusion_cache()
    sampling_controller = model.diffusion_sampling_controller
    if sampling_controller is not None:
        sampling_controller.reset()

    INIT_SCALE = 0.999  # so that we can apply rescale to first step

    device, dtype = model.device, model.dtype
    batch_size = text_input_ids.shape[0]

    rng = torch.Generator(device=device).manual_seed(rng_seed)

    t_schedule = torch.linspace(1., 0., num_steps + 1, device=device) * INIT_SCALE
    if sampling_controller is not None:
        t_schedule = sampling_controller.prepare_schedule(t_schedule)

    text_mask_uncond = torch.zeros_like(text_mask)
    speaker_mask_uncond = torch.zeros_like(speaker_mask)

    kv_text_cond = model.get_kv_cache_text(text_input_ids, text_mask)
    kv_speaker_cond = model.get_kv_cache_speaker(speaker_latent.to(dtype))

    if speaker_kv_scale is not None:
        _multiply_kv_cache(kv_speaker_cond, speaker_kv_scale, speaker_kv_max_layers)

    # masks prevent decoder from attending to unconds:
    kv_text_full = _concat_kv_caches(kv_text_cond, kv_text_cond, kv_text_cond)
    kv_speaker_full = _concat_kv_caches(kv_speaker_cond, kv_speaker_cond, kv_speaker_cond)

    full_text_mask = torch.cat([text_mask, text_mask_uncond, text_mask], dim=0)
    full_speaker_mask = torch.cat([speaker_mask, speaker_mask, speaker_mask_uncond], dim=0)

    x_t = torch.randn((batch_size, sequence_length, 80), device=device, dtype=torch.float32, generator=rng)
    if truncation_factor is not None:
        x_t = x_t * truncation_factor

    prepared_steps = len(t_schedule) - 1
    for i in range(prepared_steps):
        t, t_next = t_schedule[i], t_schedule[i + 1]

        native_cfg = bool(((t >= cfg_min_t) * (t <= cfg_max_t)).item())
        guidance_context = DiffusionStepContext(
            index=i,
            total_steps=prepared_steps,
            timestep=t,
            next_timestep=t_next,
            lane="packed-cfg",
            solver="euler",
        )
        has_cfg = (
            native_cfg if sampling_controller is None else sampling_controller.should_use_guidance(
                guidance_context,
                native=native_cfg,
            ))
        context = (
            guidance_context if has_cfg else DiffusionStepContext(
                index=i,
                total_steps=prepared_steps,
                timestep=t,
                next_timestep=t_next,
                lane="conditional",
                solver="euler",
            ))

        def compute_velocity() -> torch.Tensor:
            if has_cfg:
                v_cond, v_uncond_text, v_uncond_speaker = model(
                    x=torch.cat([x_t, x_t, x_t], dim=0).to(dtype),
                    t=(torch.ones((batch_size * 3, ), device=device) * t).to(dtype),
                    text_mask=full_text_mask,
                    speaker_mask=full_speaker_mask,
                    kv_cache_text=kv_text_full,
                    kv_cache_speaker=kv_speaker_full,
                    diffusion_cache_lane="packed-cfg",
                ).float().chunk(
                    3, dim=0)
                if sampling_controller is not None:
                    sampling_controller.observe_guidance(
                        context,
                        torch.cat((v_cond, v_cond), dim=0),
                        torch.cat(
                            (v_uncond_text, v_uncond_speaker),
                            dim=0,
                        ),
                    )
                return (
                    v_cond + cfg_scale_text * (v_cond - v_uncond_text) + cfg_scale_speaker *
                    (v_cond - v_uncond_speaker))
            return model(
                x=x_t.to(dtype),
                t=(torch.ones((batch_size, ), device=device) * t).to(dtype),
                text_mask=text_mask,
                speaker_mask=speaker_mask,
                kv_cache_text=kv_text_cond,
                kv_cache_speaker=kv_speaker_cond,
                diffusion_cache_lane="conditional",
            ).float()

        v_pred = (
            compute_velocity() if sampling_controller is None else sampling_controller.evaluate(
                context,
                x_t,
                compute_velocity,
            ))

        # optional temporal score rescaling: https://arxiv.org/pdf/2510.01184
        if rescale_k is not None and rescale_sigma is not None:
            v_pred = _temporal_score_rescale(v_pred, x_t, t, rescale_k, rescale_sigma)

        # optional kv speaker scaling
        if speaker_kv_scale is not None and t_next < speaker_kv_min_t and t >= speaker_kv_min_t:
            _multiply_kv_cache(kv_speaker_cond, 1. / speaker_kv_scale, speaker_kv_max_layers)
            kv_speaker_full = _concat_kv_caches(kv_speaker_cond, kv_speaker_cond, kv_speaker_cond)
            model.reset_diffusion_cache()
            if sampling_controller is not None:
                sampling_controller.reset()

        x_t = x_t + v_pred * (t_next - t)

    return x_t


# ___________________________________________________________
# simple example

if __name__ == "__main__":
    model = load_model_from_hf(delete_blockwise_modules=True)
    fish_ae = load_fish_ae_from_hf()
    pca_state = load_pca_state_from_hf()

    speaker_audio_path = "/path/to/speaker/audio.wav"
    speaker_audio = load_audio(speaker_audio_path).cuda()
    speaker_latent, speaker_mask = get_speaker_latent_and_mask(fish_ae, pca_state, speaker_audio)

    text = "[S1] Alright, I'm going to demo this new model called Echo TTS. Hopefully this works, I'm super excited to try this and see what it can do."
    text_input_ids, text_mask = get_text_input_ids_and_mask([text], max_length=None, device="cuda")

    latent_out = sample_euler_cfg_independent_guidances(
        model=model,
        speaker_latent=speaker_latent,
        speaker_mask=speaker_mask,
        text_input_ids=text_input_ids,
        text_mask=text_mask,
        rng_seed=0,
        num_steps=40,
        cfg_scale_text=3.0,
        cfg_scale_speaker=8.0,
        cfg_min_t=0.5,
        cfg_max_t=1.0,
        truncation_factor=0.8,
        rescale_k=None,
        rescale_sigma=None,
        speaker_kv_scale=None,
        speaker_kv_max_layers=None,
        speaker_kv_min_t=None,
        # Shorter lengths generate prefixes, not necessarily full generations.
        sequence_length=640,
    )
    audio_out = ae_decode(fish_ae, pca_state, latent_out)
    audio_out = crop_audio_to_flattening_point(audio_out, latent_out[0])
    save_pcm_wave("output.wav", audio_out[0].cpu(), 44_100)
