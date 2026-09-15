from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from voicehub.kernels.diffusion import DiffusionModulationKernelOptimizable
from voicehub.optimization.diffusion_cache import DiffusionCacheMixin
from voicehub.optimization.diffusion_sampling import DiffusionSamplingMixin


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta**(torch.arange(0, dim, 2)[:(dim // 2)] / dim))
    t = torch.arange(end)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.complex(torch.cos(freqs), torch.sin(freqs))
    return freqs_cis


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> torch.Tensor:
    x_ = torch.view_as_complex(x.float().reshape(*x.shape[:3], -1, 2))
    x_ = x_ * freqs_cis[..., None, :]
    x_ = torch.view_as_real(x_).reshape(x.shape)
    return x_.type_as(x)


def get_timestep_embedding(
    timestep: torch.Tensor,
    embed_size: int,
) -> torch.Tensor:
    assert embed_size % 2 == 0

    half = embed_size // 2

    freqs = 1000 * torch.exp(
        -torch.log(torch.tensor(10000.0)) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            timestep.device)

    args = timestep[..., None] * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    return embedding.to(timestep.dtype)


class LowRankAdaLN(DiffusionModulationKernelOptimizable, nn.Module):

    def __init__(self, model_size: int, rank: int, eps: float):
        super().__init__()
        self.eps = eps

        self.shift_down = nn.Linear(model_size, rank, bias=False)
        self.scale_down = nn.Linear(model_size, rank, bias=False)
        self.gate_down = nn.Linear(model_size, rank, bias=False)

        self.shift_up = nn.Linear(rank, model_size, bias=True)
        self.scale_up = nn.Linear(rank, model_size, bias=True)
        self.gate_up = nn.Linear(rank, model_size, bias=True)
        self._initialize_diffusion_kernel_backend()

    def forward(
        self,
        x: torch.Tensor,
        cond_embed: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        shift, scale, gate = cond_embed.chunk(3, dim=-1)

        shift = self.shift_up(self.shift_down(F.silu(shift))) + shift
        scale = self.scale_up(self.scale_down(F.silu(scale))) + scale
        gate = self.gate_up(self.gate_down(F.silu(gate))) + gate

        x_dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(torch.pow(x.float(), 2).mean(dim=-1, keepdim=True) + self.eps)
        if scale.dtype in (torch.float16, torch.bfloat16):
            # Upstream adds one before promoting the gain to float32.
            # Preserve that low-precision rounding while supplying the
            # shared modulation kernel with three float32 tensors.
            scale = (scale + 1).float() - 1
        x = self._diffusion_modulate(
            x,
            shift.to(x.dtype),
            scale.to(x.dtype),
        )

        gate = torch.tanh(gate)

        return x.to(x_dtype), gate


class RMSNorm(nn.Module):  # could also just use torch rmsnorm

    def __init__(self, model_size: int | Tuple[int, int], eps: float):
        super().__init__()
        self.eps = eps

        if isinstance(model_size, int):
            model_size = (model_size, )
        self.weight = nn.Parameter(torch.ones(model_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(torch.pow(x.float(), 2).mean(dim=-1, keepdim=True) + self.eps)
        x = x * self.weight
        return x.to(x_dtype)


class SelfAttention(nn.Module):

    def __init__(self, model_size: int, num_heads: int, is_causal: bool, norm_eps: float):
        super().__init__()
        self.num_heads = num_heads
        self.is_causal = is_causal

        self.wq = nn.Linear(model_size, model_size, bias=False)
        self.wk = nn.Linear(model_size, model_size, bias=False)
        self.wv = nn.Linear(model_size, model_size, bias=False)
        self.wo = nn.Linear(model_size, model_size, bias=False)
        self.gate = nn.Linear(model_size, model_size, bias=False)

        assert model_size % num_heads == 0
        self.q_norm = RMSNorm((num_heads, model_size // num_heads), eps=norm_eps)
        self.k_norm = RMSNorm((num_heads, model_size // num_heads), eps=norm_eps)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None, freqs_cis: torch.Tensor) -> torch.Tensor:

        batch_size, seq_len = x.shape[:2]

        xq = self.wq(x).reshape(batch_size, seq_len, self.num_heads, -1)
        xk = self.wk(x).reshape(batch_size, seq_len, self.num_heads, -1)
        xv = self.wv(x).reshape(batch_size, seq_len, self.num_heads, -1)

        gate = self.gate(x)

        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        xq = apply_rotary_emb(xq, freqs_cis[:seq_len])
        xk = apply_rotary_emb(xk, freqs_cis[:seq_len])

        if mask is not None:
            assert mask.ndim == 2  # (b, s)
            mask = mask[:, None, None]

        output = F.scaled_dot_product_attention(
            query=xq.transpose(1, 2),
            key=xk.transpose(1, 2),
            value=xv.transpose(1, 2),
            attn_mask=mask,
            is_causal=self.is_causal).transpose(1, 2)

        output = output.reshape(batch_size, seq_len, -1)
        output = output * torch.sigmoid(gate)

        output = self.wo(output)

        return output


class JointAttention(nn.Module):

    def __init__(
            self, model_size: int, num_heads: int, text_model_size: int, speaker_model_size: int,
            speaker_patch_size: int, norm_eps: float):
        super().__init__()
        self.speaker_patch_size = speaker_patch_size
        self.num_heads = num_heads

        self.wq = nn.Linear(model_size, model_size, bias=False)
        self.wk = nn.Linear(model_size, model_size, bias=False)
        self.wv = nn.Linear(model_size, model_size, bias=False)

        self.wk_text = nn.Linear(text_model_size, model_size, bias=False)
        self.wv_text = nn.Linear(text_model_size, model_size, bias=False)

        self.wk_speaker = nn.Linear(speaker_model_size, model_size, bias=False)
        self.wv_speaker = nn.Linear(speaker_model_size, model_size, bias=False)

        self.wk_latent = nn.Linear(speaker_model_size, model_size, bias=False)
        self.wv_latent = nn.Linear(speaker_model_size, model_size, bias=False)

        assert model_size % num_heads == 0
        self.head_dim = model_size // num_heads
        self.q_norm = RMSNorm((num_heads, self.head_dim), eps=norm_eps)
        self.k_norm = RMSNorm((num_heads, self.head_dim), eps=norm_eps)

        self.gate = nn.Linear(model_size, model_size, bias=False)

        self.wo = nn.Linear(model_size, model_size, bias=False)

    def _apply_rotary_half(self, y: torch.Tensor, fc: torch.Tensor) -> torch.Tensor:
        y1, y2 = y.chunk(2, dim=-2)
        y1 = apply_rotary_emb(y1, fc)
        return torch.cat([y1, y2], dim=-2)

    def forward(
            self, x: torch.Tensor, text_mask: torch.Tensor, speaker_mask: torch.Tensor,
            freqs_cis: torch.Tensor, kv_cache_text: Tuple[torch.Tensor, torch.Tensor],
            kv_cache_speaker: Tuple[torch.Tensor, torch.Tensor], start_pos: int | None,
            kv_cache_latent: Tuple[torch.Tensor, torch.Tensor] | None) -> torch.Tensor:
        batch_size, seq_len = x.shape[:2]

        xq = self.wq(x).reshape(batch_size, seq_len, self.num_heads, -1)
        xk_self = self.wk(x).reshape(batch_size, seq_len, self.num_heads, -1)
        xv_self = self.wv(x).reshape(batch_size, seq_len, self.num_heads, -1)

        xq = self.q_norm(xq)
        xk_self = self.k_norm(xk_self)

        gate = self.gate(x)

        if start_pos is None:
            start_pos = 0

        freqs_q = freqs_cis[start_pos:start_pos + seq_len]

        xq = self._apply_rotary_half(xq, freqs_q)
        xk_self = self._apply_rotary_half(xk_self, freqs_q)

        xk_text, xv_text = kv_cache_text
        xk_speaker, xv_speaker = kv_cache_speaker

        if kv_cache_latent is None or kv_cache_latent[0].shape[1] == 0:
            xk_latent = torch.zeros((batch_size, 0, self.num_heads, xq.shape[-1]),
                                    device=x.device,
                                    dtype=x.dtype)
            xv_latent = torch.zeros((batch_size, 0, self.num_heads, xq.shape[-1]),
                                    device=x.device,
                                    dtype=x.dtype)
            latent_mask = torch.zeros((batch_size, 0), dtype=torch.bool, device=x.device)
        else:
            xk_latent, xv_latent = kv_cache_latent
            latent_positions = torch.arange(
                xk_latent.shape[1], device=x.device, dtype=torch.long) * self.speaker_patch_size
            latent_mask = (latent_positions[None, :] < start_pos).expand(batch_size, xk_latent.shape[1])

        xk = torch.cat([xk_self, xk_latent, xk_text, xk_speaker], dim=1)
        xv = torch.cat([xv_self, xv_latent, xv_text, xv_speaker], dim=1)

        self_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=x.device)

        mask = torch.cat([self_mask, latent_mask, text_mask, speaker_mask], dim=1)
        mask = mask[:, None, None]

        output = F.scaled_dot_product_attention(
            query=xq.transpose(1, 2),
            key=xk.transpose(1, 2),
            value=xv.transpose(1, 2),
            attn_mask=mask,
            is_causal=False).transpose(1, 2)

        output = output.reshape(batch_size, seq_len, -1)
        output = output * torch.sigmoid(gate)

        output = self.wo(output)

        return output

    def get_kv_cache_text(self, text_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = text_state.shape[0]
        xk = self.wk_text(text_state).reshape(batch_size, text_state.shape[1], self.num_heads, -1)
        xv = self.wv_text(text_state).reshape(batch_size, text_state.shape[1], self.num_heads, -1)
        xk = self.k_norm(xk)
        return xk, xv

    def get_kv_cache_speaker(self, speaker_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = speaker_state.shape[0]
        xk = self.wk_speaker(speaker_state).reshape(batch_size, speaker_state.shape[1], self.num_heads, -1)
        xv = self.wv_speaker(speaker_state).reshape(batch_size, speaker_state.shape[1], self.num_heads, -1)
        xk = self.k_norm(xk)
        return xk, xv

    def get_kv_cache_latent(self, latent_state: torch.Tensor,
                            freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = latent_state.shape[0]
        seq_len = latent_state.shape[1]
        xk = self.wk_latent(latent_state).reshape(batch_size, seq_len, self.num_heads, -1)
        xv = self.wv_latent(latent_state).reshape(batch_size, seq_len, self.num_heads, -1)
        xk = self.k_norm(xk)

        xk = self._apply_rotary_half(xk, freqs_cis)

        return xk, xv


class MLP(nn.Module):

    def __init__(self, model_size: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Linear(model_size, intermediate_size, bias=False)
        self.w3 = nn.Linear(model_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, model_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class EncoderTransformerBlock(nn.Module):

    def __init__(
            self, model_size: int, num_heads: int, intermediate_size: int, is_causal: bool, norm_eps: float):
        super().__init__()
        self.attention = SelfAttention(
            model_size=model_size, num_heads=num_heads, is_causal=is_causal, norm_eps=norm_eps)
        self.mlp = MLP(model_size=model_size, intermediate_size=intermediate_size)

        self.attention_norm = RMSNorm(model_size, norm_eps)
        self.mlp_norm = RMSNorm(model_size, norm_eps)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None, freqs_cis: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x), mask, freqs_cis)
        x = x + self.mlp(self.mlp_norm(x))

        return x


class TransformerBlock(nn.Module):

    def __init__(
        self,
        model_size: int,
        num_heads: int,
        intermediate_size: int,
        norm_eps: float,
        text_model_size: int,
        speaker_model_size: int,
        speaker_patch_size: int,
        adaln_rank: int,
    ):
        super().__init__()
        self.attention = JointAttention(
            model_size=model_size,
            num_heads=num_heads,
            text_model_size=text_model_size,
            speaker_model_size=speaker_model_size,
            speaker_patch_size=speaker_patch_size,
            norm_eps=norm_eps)

        self.mlp = MLP(model_size=model_size, intermediate_size=intermediate_size)

        self.attention_adaln = LowRankAdaLN(model_size=model_size, rank=adaln_rank, eps=norm_eps)
        self.mlp_adaln = LowRankAdaLN(model_size=model_size, rank=adaln_rank, eps=norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cond_embed: torch.Tensor,
        text_mask: torch.Tensor,
        speaker_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        kv_cache_text: Tuple[torch.Tensor, torch.Tensor],
        kv_cache_speaker: Tuple[torch.Tensor, torch.Tensor],
        start_pos: int | None,
        kv_cache_latent: Tuple[torch.Tensor, torch.Tensor] | None,
    ) -> torch.Tensor:

        x_norm, attention_gate = self.attention_adaln(x, cond_embed)
        x = x + attention_gate * self.attention(
            x_norm, text_mask, speaker_mask, freqs_cis, kv_cache_text, kv_cache_speaker, start_pos,
            kv_cache_latent)

        x_norm, mlp_gate = self.mlp_adaln(x, cond_embed)
        x = x + mlp_gate * self.mlp(x_norm)

        return x


class TextEncoder(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        model_size: int,
        num_layers: int,
        num_heads: int,
        intermediate_size: int,
        norm_eps: float,
    ):
        super().__init__()
        self.text_embedding = nn.Embedding(vocab_size, model_size)

        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            block = EncoderTransformerBlock(
                model_size=model_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                is_causal=False,
                norm_eps=norm_eps)
            self.blocks.append(block)

        self.head_dim = model_size // num_heads

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.text_embedding(input_ids)

        freqs_cis = precompute_freqs_cis(self.head_dim, input_ids.shape[1]).to(x.device)  # could cache

        for block in self.blocks:
            x = block(x, mask, freqs_cis)

        return x


class SpeakerEncoder(nn.Module):

    def __init__(
        self,
        latent_size: int,
        patch_size: int,
        model_size: int,
        num_layers: int,
        num_heads: int,
        intermediate_size: int,
        norm_eps: float,
    ):
        super().__init__()
        self.patch_size = patch_size

        self.in_proj = nn.Linear(latent_size * patch_size, model_size, bias=True)

        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            block = EncoderTransformerBlock(
                model_size=model_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                is_causal=True,
                norm_eps=norm_eps)
            self.blocks.append(block)

        self.head_dim = model_size // num_heads

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        x = latent.reshape(
            *latent.shape[:-2], latent.shape[-2] // self.patch_size, latent.shape[-1] * self.patch_size)

        x = self.in_proj(x)
        x = x / 6.  # this helped with initial activation dynamics in early ablations, could also bake into in_proj

        freqs_cis = precompute_freqs_cis(self.head_dim, x.shape[1]).to(x.device)  # could cache

        for block in self.blocks:
            x = block(x, None, freqs_cis)

        return x


class EchoDiT(DiffusionCacheMixin, DiffusionSamplingMixin, nn.Module):

    def __init__(
        self,
        latent_size: int,
        #
        model_size: int,
        num_layers: int,
        num_heads: int,
        intermediate_size: int,
        norm_eps: float,
        #
        text_vocab_size: int,
        text_model_size: int,
        text_num_layers: int,
        text_num_heads: int,
        text_intermediate_size: int,
        #
        speaker_patch_size: int,
        speaker_model_size: int,
        speaker_num_layers: int,
        speaker_num_heads: int,
        speaker_intermediate_size: int,
        #
        timestep_embed_size: int,
        adaln_rank: int,
    ):
        super().__init__()
        self.speaker_patch_size = speaker_patch_size
        self.timestep_embed_size = timestep_embed_size
        self.blockwise_generation_available = True

        self.text_encoder = TextEncoder(
            vocab_size=text_vocab_size,
            model_size=text_model_size,
            num_layers=text_num_layers,
            num_heads=text_num_heads,
            intermediate_size=text_intermediate_size,
            norm_eps=norm_eps,
        )
        self.speaker_encoder = SpeakerEncoder(
            latent_size=latent_size,
            patch_size=speaker_patch_size,
            model_size=speaker_model_size,
            num_layers=speaker_num_layers,
            num_heads=speaker_num_heads,
            intermediate_size=speaker_intermediate_size,
            norm_eps=norm_eps,
        )
        self.latent_encoder = SpeakerEncoder(
            latent_size=latent_size,
            patch_size=speaker_patch_size,
            model_size=speaker_model_size,
            num_layers=speaker_num_layers,
            num_heads=speaker_num_heads,
            intermediate_size=speaker_intermediate_size,
            norm_eps=norm_eps,
        )
        self.text_norm = RMSNorm(text_model_size, norm_eps)
        self.speaker_norm = RMSNorm(speaker_model_size, norm_eps)
        self.latent_norm = RMSNorm(speaker_model_size, norm_eps)

        self.cond_module = nn.Sequential(
            nn.Linear(timestep_embed_size, model_size, bias=False),
            nn.SiLU(),
            nn.Linear(model_size, model_size, bias=False),
            nn.SiLU(),
            nn.Linear(model_size, model_size * 3, bias=False),
        )

        self.in_proj = nn.Linear(latent_size, model_size, bias=True)

        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            block = TransformerBlock(
                model_size=model_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                norm_eps=norm_eps,
                text_model_size=text_model_size,
                speaker_model_size=speaker_model_size,
                speaker_patch_size=speaker_patch_size,
                adaln_rank=adaln_rank,
            )
            self.blocks.append(block)

        self.out_norm = RMSNorm(model_size, norm_eps)
        self.out_proj = nn.Linear(model_size, latent_size, bias=True)

        self.head_dim = model_size // num_heads
        self._initialize_diffusion_cache()
        self._initialize_diffusion_sampling()

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        text_mask: torch.Tensor,
        speaker_mask: torch.Tensor,
        kv_cache_text: List[Tuple[torch.Tensor, torch.Tensor]],
        kv_cache_speaker: List[Tuple[torch.Tensor, torch.Tensor]],
        start_pos: int | None = None,
        kv_cache_latent: List[Tuple[torch.Tensor, torch.Tensor]] | None = None,
        diffusion_cache_lane: str = "default",
    ) -> torch.Tensor:

        if start_pos is None:
            start_pos = 0

        max_pos = start_pos + x.shape[1]
        freqs_cis = precompute_freqs_cis(self.head_dim, max_pos).to(x.device)  # could cache

        speaker_mask = speaker_mask[..., ::self.speaker_patch_size]

        cond_embed = self.cond_module(get_timestep_embedding(t, self.timestep_embed_size))
        cond_embed = cond_embed[:, None]

        x = self.in_proj(x)

        block_indices = {id(block): index for index, block in enumerate(self.blocks)}

        def apply_block(block: nn.Module, value: torch.Tensor) -> torch.Tensor:
            index = block_indices[id(block)]
            return block(
                x=value,
                cond_embed=cond_embed,
                text_mask=text_mask,
                speaker_mask=speaker_mask,
                freqs_cis=freqs_cis,
                kv_cache_text=kv_cache_text[index],
                kv_cache_speaker=kv_cache_speaker[index],
                start_pos=start_pos,
                kv_cache_latent=(kv_cache_latent[index] if kv_cache_latent is not None else None),
            )

        x = self._run_diffusion_blocks(
            self.blocks,
            x,
            apply_block,
            cache_lane=diffusion_cache_lane,
        )
        x = self.out_norm(x)
        x = self.out_proj(x)

        return x.float()

    def get_kv_cache_text(
        self,
        text_input_ids: torch.Tensor,
        text_mask: torch.Tensor | None,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        text_state = self.text_encoder(text_input_ids, text_mask)
        text_state = self.text_norm(text_state)
        return [block.attention.get_kv_cache_text(text_state) for block in self.blocks]

    def get_kv_cache_speaker(
        self,
        speaker_latent: torch.Tensor,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        speaker_state = self.speaker_encoder(speaker_latent)
        speaker_state = self.speaker_norm(speaker_state)
        return [block.attention.get_kv_cache_speaker(speaker_state) for block in self.blocks]

    def get_kv_cache_latent(
        self,
        prefix_latent: torch.Tensor,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        if not self.blockwise_generation_available:
            raise RuntimeError("This Echo runtime was loaded without blockwise-only "
                               "checkpoint tensors.")
        latent_state = self.latent_encoder(prefix_latent)
        latent_state = self.latent_norm(latent_state)

        seq_len = latent_state.shape[1]
        max_pos = seq_len * self.speaker_patch_size
        freqs_cis = precompute_freqs_cis(self.head_dim, max_pos).to(latent_state.device)  # could cache
        positions = torch.arange(seq_len, device=latent_state.device) * self.speaker_patch_size
        freqs_latent = freqs_cis[positions]

        return [block.attention.get_kv_cache_latent(latent_state, freqs_latent) for block in self.blocks]

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype
