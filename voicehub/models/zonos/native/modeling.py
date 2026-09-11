"""VoiceHub-owned PyTorch implementation of Zonos v0.1 Transformer.

The module preserves the exact published Safetensors namespace while
keeping training and inference as explicit execution modes.  It depends
only on PyTorch and other VoiceHub modules; the vendored upstream tree
is provenance, not an executable runtime dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from voicehub.models.zonos.native.configuration import (
    ZonosArchitectureConfig,
    ZonosBackboneConfig,
    ZonosPrefixConditionerConfig,
)
from voicehub.models.zonos.native.pattern import add_endpoint_and_delay
from voicehub.optimization.protocols import OptimizationCompileTarget


@dataclass
class ZonosInferenceCache:
    """Mutable KV-cache state for incremental generation."""

    max_sequence_length: int
    max_batch_size: int
    sequence_offset: int = 0
    batch_offset: int = 0
    key_values: dict[int, Tensor] = field(default_factory=dict)
    lengths_per_sample: Tensor | None = None


@dataclass(frozen=True)
class ZonosForCausalLMOutput:
    """Teacher-forced acoustic-language-model output."""

    logits: Tensor
    labels: Tensor
    loss: Tensor | None = None


def precompute_rotary_frequencies(
    sequence_length: int,
    head_dim: int,
    *,
    device: torch.device | str,
    base: float = 10_000.0,
) -> Tensor:
    if sequence_length <= 0 or head_dim <= 0 or head_dim % 2:
        raise ValueError("Rotary sequence length and even head dimension must be positive.")
    frequencies = 1.0 / (
        base**(torch.arange(
            0,
            head_dim,
            2,
            dtype=torch.float32,
            device=device,
        ) / head_dim))
    positions = torch.arange(
        sequence_length,
        dtype=torch.float32,
        device=device,
    )
    angles = torch.outer(positions, frequencies)
    return torch.stack((angles.cos(), angles.sin()), dim=-1)


def apply_rotary_embedding(values: Tensor, frequencies: Tensor) -> Tensor:
    """Apply the interleaved rotary layout used by the published graph."""
    if values.shape[-1] % 2:
        raise ValueError("Zonos rotary head dimension must be even.")
    paired = values.float().reshape(*values.shape[:-1], -1, 2)
    if frequencies.ndim == 3:
        frequencies = frequencies.unsqueeze(0)
    if frequencies.ndim != 4:
        raise ValueError("Zonos rotary frequencies must have shape "
                         "[batch, time, head_dim / 2, 2].")
    frequencies = frequencies.reshape(
        -1,
        paired.shape[1],
        1,
        paired.shape[3],
        2,
    )
    rotated = torch.stack(
        (
            paired[..., 0] * frequencies[..., 0] - paired[..., 1] * frequencies[..., 1],
            paired[..., 1] * frequencies[..., 0] + paired[..., 0] * frequencies[..., 1],
        ),
        dim=-1,
    )
    return rotated.flatten(3).to(dtype=values.dtype)


class ZonosConditioner(nn.Module):
    """Base module preserving the source conditioner state layout."""

    def __init__(
        self,
        output_dim: int,
        name: str,
        *,
        cond_dim: int | None = None,
        projection: str = "none",
        uncond_type: str = "none",
        **_: Any,
    ) -> None:
        super().__init__()
        self.name = name
        self.output_dim = output_dim
        self.cond_dim = output_dim if cond_dim is None else cond_dim
        if projection == "linear":
            self.project = nn.Linear(self.cond_dim, output_dim)
        elif projection == "mlp":
            self.project = nn.Sequential(
                nn.Linear(self.cond_dim, output_dim),
                nn.SiLU(),
                nn.Linear(output_dim, output_dim),
            )
        elif projection == "none":
            self.project = nn.Identity()
        else:  # guarded by configuration validation
            raise ValueError(f"Unknown Zonos projection {projection!r}.")
        if uncond_type == "learned":
            self.uncond_vector = nn.Parameter(torch.zeros(output_dim))
        elif uncond_type == "none":
            self.register_parameter("uncond_vector", None)
        else:
            raise ValueError(f"Unknown Zonos unconditional type {uncond_type!r}.")

    def apply_condition(self, value: Any) -> Tensor:
        raise NotImplementedError

    def forward(self, value: Any | None) -> Tensor:
        if value is None:
            if self.uncond_vector is None:
                raise ValueError(f"Zonos conditioning value {self.name!r} is required.")
            # Do not use ``.data`` here. The learned unconditional vector is
            # part of the published checkpoint and must remain trainable.
            return self.uncond_vector.view(1, 1, -1)
        return self.project(self.apply_condition(value))


class ZonosPhonemeConditioner(ZonosConditioner):
    """Embedding layer for already-tokenized eSpeak phoneme IDs."""

    def __init__(self, output_dim: int, **kwargs: Any) -> None:
        super().__init__(output_dim, **kwargs)
        self.phoneme_embedder = nn.Embedding(189, output_dim)

    def apply_condition(self, value: Tensor) -> Tensor:
        if not isinstance(value, Tensor) or value.ndim != 2:
            raise ValueError(
                "Zonos `espeak` conditioning must contain phoneme IDs with "
                "shape [batch, time].")
        if value.dtype == torch.bool or value.is_floating_point():
            raise TypeError("Zonos phoneme IDs must use an integer dtype.")
        if bool(((value < 0) | (value >= self.phoneme_embedder.num_embeddings)).any()):
            raise ValueError("Zonos phoneme IDs are outside the released vocabulary.")
        return self.phoneme_embedder(value.to(device=self.phoneme_embedder.weight.device, dtype=torch.long), )


class ZonosFourierConditioner(ZonosConditioner):

    def __init__(
        self,
        output_dim: int,
        *,
        input_dim: int = 1,
        std: float = 1.0,
        min_val: float = 0.0,
        max_val: float = 1.0,
        **kwargs: Any,
    ) -> None:
        if output_dim % 2:
            raise ValueError("Zonos Fourier conditioner dimension must be even.")
        super().__init__(output_dim, **kwargs)
        self.register_buffer(
            "weight",
            torch.randn(output_dim // 2, input_dim) * std,
        )
        self.input_dim = input_dim
        self.min_val = min_val
        self.max_val = max_val

    def apply_condition(self, value: Tensor) -> Tensor:
        if not isinstance(value, Tensor) or value.ndim != 3:
            raise ValueError(f"Zonos `{self.name}` conditioning must have shape "
                             "[batch, time, feature].")
        if value.shape[-1] != self.input_dim:
            raise ValueError(
                f"Zonos `{self.name}` expects {self.input_dim} features, "
                f"received {value.shape[-1]}.")
        normalized = (value.to(device=self.weight.device, dtype=self.weight.dtype) -
                      self.min_val) / (self.max_val - self.min_val)
        frequencies = 2 * torch.pi * normalized @ self.weight.T
        return torch.cat((frequencies.cos(), frequencies.sin()), dim=-1)


class ZonosIntegerConditioner(ZonosConditioner):

    def __init__(
        self,
        output_dim: int,
        *,
        min_val: int = 0,
        max_val: int = 512,
        **kwargs: Any,
    ) -> None:
        super().__init__(output_dim, **kwargs)
        self.min_val = min_val
        self.max_val = max_val
        self.int_embedder = nn.Embedding(
            max_val - min_val + 1,
            output_dim,
        )

    def apply_condition(self, value: Tensor) -> Tensor:
        if not isinstance(value, Tensor) or value.ndim != 3:
            raise ValueError(f"Zonos `{self.name}` conditioning must have shape "
                             "[batch, time, 1].")
        if value.shape[-1] != 1:
            raise ValueError(f"Zonos `{self.name}` expects one integer feature.")
        if value.dtype == torch.bool or value.is_floating_point():
            raise TypeError(f"Zonos `{self.name}` conditioning must use an integer dtype.")
        indices = value.squeeze(-1).to(
            device=self.int_embedder.weight.device,
            dtype=torch.long,
        ) - self.min_val
        if bool(((indices < 0) | (indices >= self.int_embedder.num_embeddings)).any()):
            raise ValueError(f"Zonos `{self.name}` values must be in "
                             f"[{self.min_val}, {self.max_val}].")
        return self.int_embedder(indices)


class ZonosPassthroughConditioner(ZonosConditioner):

    def apply_condition(self, value: Tensor) -> Tensor:
        if not isinstance(value, Tensor) or value.ndim != 3:
            raise ValueError(f"Zonos `{self.name}` conditioning must have shape "
                             "[batch, time, feature].")
        if value.shape[-1] != self.cond_dim:
            raise ValueError(
                f"Zonos `{self.name}` expects feature dimension "
                f"{self.cond_dim}, received {value.shape[-1]}.")
        parameter = next(self.project.parameters(), None)
        if parameter is None:
            return value
        return value.to(device=parameter.device, dtype=parameter.dtype)


_CONDITIONERS = {
    "EspeakPhonemeConditioner": ZonosPhonemeConditioner,
    "FourierConditioner": ZonosFourierConditioner,
    "IntegerConditioner": ZonosIntegerConditioner,
    "PassthroughConditioner": ZonosPassthroughConditioner,
}


class ZonosPrefixConditioner(ZonosConditioner):
    """Ordered prefix graph with the exact upstream module namespace."""

    def __init__(
        self,
        config: ZonosPrefixConditionerConfig,
        output_dim: int,
    ) -> None:
        super().__init__(
            output_dim,
            "prefix",
            projection=config.projection,
        )
        self.conditioners = nn.ModuleList([
            _CONDITIONERS[str(values["type"])](
                output_dim,
                **dict(values),
            ) for values in config.conditioners
        ])
        self.norm = nn.LayerNorm(output_dim)
        self.required_keys = frozenset(
            conditioner.name for conditioner in self.conditioners if conditioner.uncond_vector is None)

    def apply_condition(self, value: Any) -> Tensor:  # pragma: no cover
        raise NotImplementedError("Prefix conditioner owns a mapping forward.")

    def forward(self, values: dict[str, Any]) -> Tensor:
        if not isinstance(values, dict):
            raise TypeError("Zonos prefix conditioning must be a dictionary.")
        missing = self.required_keys - set(values)
        if missing:
            raise ValueError(f"Missing required Zonos conditioning keys: {sorted(missing)!r}.")
        conditioned = [conditioner(values.get(conditioner.name)) for conditioner in self.conditioners]
        batch_size = max(item.shape[0] for item in conditioned)
        if any(item.shape[0] not in {1, batch_size} for item in conditioned):
            raise ValueError("Zonos conditioner batch dimensions are not broadcastable.")
        conditioned = [item.expand(batch_size, -1, -1) for item in conditioned]
        return self.norm(self.project(torch.cat(conditioned, dim=-2)))


class ZonosAttention(nn.Module):

    def __init__(
        self,
        config: ZonosBackboneConfig,
        layer_index: int,
    ) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.num_heads_kv = config.num_heads_kv
        self.head_dim = config.head_dim
        self.layer_index = layer_index
        projected = (self.num_heads + 2 * self.num_heads_kv) * self.head_dim
        self.in_proj = nn.Linear(
            config.d_model,
            projected,
            bias=False,
        )
        self.out_proj = nn.Linear(
            self.num_heads * self.head_dim,
            config.d_model,
            bias=False,
        )

    def allocate_cache(
        self,
        *,
        batch_size: int,
        sequence_length: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> Tensor:
        return torch.empty(
            batch_size,
            sequence_length,
            2,
            self.num_heads_kv,
            self.head_dim,
            dtype=dtype,
            device=device,
        )

    def _project(self, hidden_states: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch_size, sequence_length, _ = hidden_states.shape
        query_size = self.num_heads * self.head_dim
        kv_size = self.num_heads_kv * self.head_dim
        query, key, value = self.in_proj(hidden_states).split(
            (query_size, kv_size, kv_size),
            dim=-1,
        )
        return (
            query.view(
                batch_size,
                sequence_length,
                self.num_heads,
                self.head_dim,
            ),
            key.view(
                batch_size,
                sequence_length,
                self.num_heads_kv,
                self.head_dim,
            ),
            value.view(
                batch_size,
                sequence_length,
                self.num_heads_kv,
                self.head_dim,
            ),
        )

    def _attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        causal: bool,
    ) -> Tensor:
        query, key, value = (item.transpose(1, 2) for item in (query, key, value))
        try:
            attended = F.scaled_dot_product_attention(
                query,
                key,
                value,
                is_causal=causal,
                enable_gqa=True,
            )
        except TypeError:  # pragma: no cover - compatibility with old PyTorch
            groups = self.num_heads // self.num_heads_kv
            attended = F.scaled_dot_product_attention(
                query,
                key.repeat_interleave(groups, dim=1),
                value.repeat_interleave(groups, dim=1),
                is_causal=causal,
            )
        batch_size, _, sequence_length, _ = attended.shape
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size,
            sequence_length,
            self.num_heads * self.head_dim,
        )
        return self.out_proj(attended)

    def forward_training(
        self,
        hidden_states: Tensor,
        frequencies: Tensor,
    ) -> Tensor:
        query, key, value = self._project(hidden_states)
        query = apply_rotary_embedding(query, frequencies)
        key = apply_rotary_embedding(key, frequencies)
        return self._attention(
            query,
            key,
            value,
            causal=True,
        )

    def forward(
        self,
        hidden_states: Tensor,
        cache: ZonosInferenceCache,
        frequencies: Tensor,
    ) -> Tensor:
        query, key, value = self._project(hidden_states)
        query = apply_rotary_embedding(query, frequencies)
        key = apply_rotary_embedding(key, frequencies)
        try:
            storage = cache.key_values[self.layer_index]
        except KeyError as error:
            raise RuntimeError(f"Missing Zonos KV cache for layer {self.layer_index}.") from error
        batch_start = cache.batch_offset
        batch_end = batch_start + key.shape[0]
        sequence_start = cache.sequence_offset
        sequence_end = sequence_start + key.shape[1]
        if batch_end > storage.shape[0] or sequence_end > storage.shape[1]:
            raise RuntimeError("Zonos generation exceeded its allocated KV-cache capacity.")
        storage[
            batch_start:batch_end,
            sequence_start:sequence_end,
            0,
        ] = key
        storage[
            batch_start:batch_end,
            sequence_start:sequence_end,
            1,
        ] = value
        active = storage[
            batch_start:batch_end,
            :sequence_end,
        ]
        cached_key, cached_value = active.unbind(dim=2)
        return self._attention(
            query,
            cached_key,
            cached_value,
            causal=hidden_states.shape[1] > 1,
        )


class ZonosFeedForward(nn.Module):

    def __init__(self, config: ZonosBackboneConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(
            config.d_model,
            2 * config.attn_mlp_d_intermediate,
            bias=False,
        )
        self.fc2 = nn.Linear(
            config.attn_mlp_d_intermediate,
            config.d_model,
            bias=False,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        values, gate = self.fc1(hidden_states).chunk(2, dim=-1)
        return self.fc2(values * F.silu(gate))


class ZonosTransformerBlock(nn.Module):

    def __init__(
        self,
        config: ZonosBackboneConfig,
        layer_index: int,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(
            config.d_model,
            eps=config.norm_epsilon,
        )
        self.mixer = ZonosAttention(config, layer_index)
        self.norm2 = nn.LayerNorm(
            config.d_model,
            eps=config.norm_epsilon,
        )
        self.mlp = ZonosFeedForward(config)

    def forward_training(
        self,
        hidden_states: Tensor,
        frequencies: Tensor,
    ) -> Tensor:
        hidden_states = hidden_states + self.mixer.forward_training(
            self.norm(hidden_states),
            frequencies,
        )
        return hidden_states + self.mlp(self.norm2(hidden_states))

    def forward(
        self,
        hidden_states: Tensor,
        cache: ZonosInferenceCache,
        frequencies: Tensor,
    ) -> Tensor:
        hidden_states = hidden_states + self.mixer(
            self.norm(hidden_states),
            cache,
            frequencies,
        )
        return hidden_states + self.mlp(self.norm2(hidden_states))


class ZonosTransformerBackbone(nn.Module):
    """Dense decoder-only Transformer used by the default release."""

    def __init__(self, config: ZonosBackboneConfig) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(ZonosTransformerBlock(config, index) for index in range(config.n_layer))
        self.norm_f = nn.LayerNorm(
            config.d_model,
            eps=config.norm_epsilon,
        )

    def allocate_inference_cache(
        self,
        *,
        batch_size: int,
        sequence_length: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> dict[int, Tensor]:
        return {
            index:
            layer.mixer.allocate_cache(
                batch_size=batch_size,
                sequence_length=sequence_length,
                dtype=dtype,
                device=device,
            )
            for index, layer in enumerate(self.layers)
        }

    def forward_training(self, hidden_states: Tensor) -> Tensor:
        frequencies = precompute_rotary_frequencies(
            hidden_states.shape[1],
            self.config.head_dim,
            device=hidden_states.device,
        ).unsqueeze(0).expand(hidden_states.shape[0], -1, -1, -1)
        for layer in self.layers:
            hidden_states = layer.forward_training(
                hidden_states,
                frequencies,
            )
        return self.norm_f(hidden_states)

    def forward(
        self,
        hidden_states: Tensor,
        cache: ZonosInferenceCache,
    ) -> Tensor:
        if cache.lengths_per_sample is None:
            raise RuntimeError("Zonos inference cache has no per-sample positions.")
        positions = (
            torch.arange(
                hidden_states.shape[1],
                device=hidden_states.device,
            )[None, :] + cache.lengths_per_sample.to(hidden_states.device)[:, None])
        if int(positions.max().item()) >= cache.max_sequence_length:
            raise RuntimeError("Zonos rotary position exceeds the allocated sequence length.")
        all_frequencies = precompute_rotary_frequencies(
            cache.max_sequence_length,
            self.config.head_dim,
            device=hidden_states.device,
        )
        frequencies = all_frequencies[positions]
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                cache,
                frequencies,
            )
        return self.norm_f(hidden_states)


class ZonosForCausalLM(nn.Module):
    """Published nine-codebook Zonos v0.1 Transformer graph."""

    def __init__(
        self,
        config: ZonosArchitectureConfig | dict[str, Any],
    ) -> None:
        super().__init__()
        if isinstance(config, dict):
            config = ZonosArchitectureConfig.from_dict(config)
        if not isinstance(config, ZonosArchitectureConfig):
            raise TypeError("`config` must be a ZonosArchitectureConfig or mapping.")
        self.config = config
        dimension = config.backbone.d_model
        self.eos_token_id = config.eos_token_id
        self.masked_token_id = config.masked_token_id
        self.backbone = ZonosTransformerBackbone(config.backbone)
        self.prefix_conditioner = ZonosPrefixConditioner(
            config.prefix_conditioner,
            dimension,
        )
        self.embeddings = nn.ModuleList(
            [nn.Embedding(config.input_vocab_size, dimension) for _ in range(config.num_codebooks)])
        self.heads = nn.ModuleList([
            nn.Linear(
                dimension,
                config.output_vocab_size,
                bias=False,
            ) for _ in range(config.num_codebooks)
        ])

    def optimization_compile_targets(
        self,
        mode: str,
    ) -> tuple[OptimizationCompileTarget, ...]:
        """Expose teacher-forced or incremental execution boundaries."""
        if mode == "training":
            return (OptimizationCompileTarget(
                "zonos.forward",
                self,
                "forward",
            ), )
        if mode != "inference":
            raise ValueError(f"Unsupported optimization mode {mode!r}.")
        return (
            OptimizationCompileTarget(
                "zonos.prefill",
                self,
                "prefill",
            ),
            OptimizationCompileTarget(
                "zonos.decode_step",
                self,
                "decode_step",
            ),
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def embed_codes(self, codes: Tensor) -> Tensor:
        if not isinstance(codes, Tensor) or codes.ndim != 3:
            raise ValueError("Zonos delayed codes must have shape "
                             "[batch, codebook, time].")
        if codes.shape[1] != self.config.num_codebooks:
            raise ValueError(
                f"Zonos expects {self.config.num_codebooks} codebooks, "
                f"received {codes.shape[1]}.")
        if codes.dtype == torch.bool or codes.is_floating_point():
            raise TypeError("Zonos delayed codes must use an integer dtype.")
        if bool(((codes < 0) | (codes >= self.config.input_vocab_size)).any()):
            raise ValueError("Zonos delayed codes contain invalid token IDs.")
        embedded = self.embeddings[0](codes[:, 0].long())
        for index in range(1, self.config.num_codebooks):
            embedded = embedded + self.embeddings[index](codes[:, index].long(), )
        return embedded

    def apply_heads(self, hidden_states: Tensor) -> Tensor:
        return torch.stack(
            [head(hidden_states) for head in self.heads],
            dim=1,
        )

    def prepare_conditioning(
        self,
        conditional: dict[str, Any],
        unconditional: dict[str, Any] | None = None,
    ) -> Tensor:
        if unconditional is None:
            unconditional = {key: conditional[key] for key in self.prefix_conditioner.required_keys}
        conditioned = self.prefix_conditioner(conditional)
        unconditioned = self.prefix_conditioner(unconditional)
        if conditioned.shape != unconditioned.shape:
            raise ValueError("Conditional and unconditional Zonos prefixes must have "
                             "matching shapes.")
        return torch.cat((conditioned, unconditioned), dim=0)

    def teacher_forced_logits(
        self,
        prefix_conditioning: Tensor,
        audio_codes: Tensor,
        *,
        audio_code_lengths: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if (not isinstance(prefix_conditioning, Tensor) or prefix_conditioning.ndim != 3):
            raise ValueError(
                "Zonos prefix conditioning must have shape "
                "[batch, prefix_time, hidden_size].")
        if prefix_conditioning.shape[-1] != self.config.backbone.d_model:
            raise ValueError("Zonos prefix hidden size does not match the architecture.")
        if not isinstance(audio_codes, Tensor) or audio_codes.ndim != 3:
            raise ValueError("Zonos audio codes must have shape "
                             "[batch, codebook, time].")
        if audio_codes.shape[:2] != (
                prefix_conditioning.shape[0],
                self.config.num_codebooks,
        ):
            raise ValueError("Zonos prefix and audio-code batch/codebook dimensions do "
                             "not match.")
        model_codes, labels = add_endpoint_and_delay(
            audio_codes.to(device=self.device),
            lengths=audio_code_lengths,
            eos_token_id=self.eos_token_id,
            mask_token_id=self.masked_token_id,
        )
        prefix = prefix_conditioning.to(
            device=self.device,
            dtype=self.dtype,
        )
        hidden_states = torch.cat(
            (prefix, self.embed_codes(model_codes)),
            dim=1,
        )
        hidden_states = self.backbone.forward_training(hidden_states)
        codec_hidden_states = hidden_states[:, prefix.shape[1]:]
        return self.apply_heads(codec_hidden_states).float(), labels

    def forward(
        self,
        prefix_conditioning: Tensor,
        audio_codes: Tensor,
        *,
        audio_code_lengths: Tensor | None = None,
        compute_loss: bool = True,
    ) -> ZonosForCausalLMOutput:
        logits, labels = self.teacher_forced_logits(
            prefix_conditioning,
            audio_codes,
            audio_code_lengths=audio_code_lengths,
        )
        loss = None
        if compute_loss:
            supervised = labels.ne(self.masked_token_id)
            if not bool(supervised.any()):
                raise ValueError("Zonos training batch contains no supervised codec token.")
            targets = labels.masked_fill(~supervised, -100)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                ignore_index=-100,
            )
        return ZonosForCausalLMOutput(
            logits=logits,
            labels=labels,
            loss=loss,
        )

    def setup_cache(
        self,
        *,
        batch_size: int,
        max_sequence_length: int,
    ) -> ZonosInferenceCache:
        if batch_size <= 0 or max_sequence_length <= 0:
            raise ValueError("Zonos cache batch and sequence dimensions must be positive.")
        aligned_length = (
            max_sequence_length if max_sequence_length % 8 == 0 else max_sequence_length + 8 -
            max_sequence_length % 8)
        return ZonosInferenceCache(
            max_sequence_length=aligned_length,
            max_batch_size=batch_size,
            key_values=self.backbone.allocate_inference_cache(
                batch_size=batch_size,
                sequence_length=aligned_length,
                dtype=self.dtype,
                device=self.device,
            ),
            lengths_per_sample=torch.zeros(
                batch_size,
                dtype=torch.int32,
                device=self.device,
            ),
        )

    def compute_generation_logits(
        self,
        hidden_states: Tensor,
        cache: ZonosInferenceCache,
        *,
        cfg_scale: float,
    ) -> Tensor:
        hidden_states = self.backbone(hidden_states, cache)[:, -1:, :]
        logits = self.apply_heads(hidden_states).squeeze(2).float()
        if cfg_scale != 1.0:
            conditional, unconditional = logits.chunk(2)
            logits = (unconditional + (conditional - unconditional) * cfg_scale)
        return logits

    def prefill(
        self,
        prefix_conditioning: Tensor,
        delayed_codes: Tensor,
        cache: ZonosInferenceCache,
        *,
        cfg_scale: float,
    ) -> Tensor:
        if cfg_scale != 1.0:
            delayed_codes = delayed_codes.expand(
                prefix_conditioning.shape[0],
                -1,
                -1,
            )
        hidden_states = torch.cat(
            (prefix_conditioning, self.embed_codes(delayed_codes)),
            dim=1,
        )
        return self.compute_generation_logits(
            hidden_states,
            cache,
            cfg_scale=cfg_scale,
        )

    def decode_step(
        self,
        input_ids: Tensor,
        cache: ZonosInferenceCache,
        *,
        cfg_scale: float,
    ) -> Tensor:
        hidden_states = self.embed_codes(input_ids)
        if cfg_scale != 1.0:
            hidden_states = hidden_states.repeat(2, 1, 1)
        return self.compute_generation_logits(
            hidden_states,
            cache,
            cfg_scale=cfg_scale,
        )


__all__ = [
    "ZonosForCausalLM",
    "ZonosForCausalLMOutput",
    "ZonosInferenceCache",
    "ZonosPrefixConditioner",
    "ZonosTransformerBackbone",
    "apply_rotary_embedding",
    "precompute_rotary_frequencies",
]
