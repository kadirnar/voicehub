"""Native PyTorch FastConformer and Token-and-Duration Transducer graph."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from voicehub.models.asr_parakeet_tdt.native.configuration import ParakeetEncoderConfig, ParakeetTDTConfig
from voicehub.models.asr_parakeet_tdt.native.loss import tdt_loss


def _activation(name: str):
    if name == "relu":
        return F.relu
    if name == "silu":
        return F.silu
    raise ValueError(f"Unsupported Parakeet activation {name!r}.")


@dataclass
class ParakeetEncoderOutput:
    """FastConformer output and its post-subsampling validity mask."""

    last_hidden_state: torch.Tensor
    attention_mask: torch.Tensor | None = None
    pooler_output: torch.Tensor | None = None


@dataclass
class ParakeetTDTOutput:
    """Training or inference output from the native TDT graph."""

    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    last_hidden_state: torch.Tensor | None = None
    pooler_output: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None
    decoder_cache: ParakeetDecoderCache | None = None


@dataclass
class ParakeetGenerateOutput:
    """Greedy TDT token/duration emissions, including the initial blank."""

    sequences: torch.Tensor
    durations: torch.Tensor


class RelativePositionalEncoding(nn.Module):
    """Transformer-XL style relative sinusoidal positions."""

    def __init__(self, config: ParakeetEncoderConfig) -> None:
        super().__init__()
        self.config = config
        inverse_frequency = 1.0 / (
            10_000.0**(torch.arange(0, config.hidden_size, 2, dtype=torch.float32) / config.hidden_size))
        self.register_buffer(
            "inv_freq",
            inverse_frequency,
            persistent=False,
        )

    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        sequence_length = hidden_states.shape[1]
        positions = torch.arange(
            sequence_length - 1,
            -sequence_length,
            -1,
            device=hidden_states.device,
            dtype=torch.float32,
        )
        frequencies = torch.outer(
            positions,
            self.inv_freq.to(device=hidden_states.device),
        )
        sine = frequencies.sin()
        cosine = frequencies.cos()
        positional = torch.stack((sine, cosine), dim=-1).reshape(
            2 * sequence_length - 1,
            -1,
        )
        return positional.unsqueeze(0).expand(
            hidden_states.shape[0],
            -1,
            -1,
        ).to(dtype=hidden_states.dtype)


class FeedForwardModule(nn.Module):

    def __init__(self, config: ParakeetEncoderConfig) -> None:
        super().__init__()
        self.linear1 = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=config.attention_bias,
        )
        self.linear2 = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.activation = _activation(config.hidden_act)
        self.activation_dropout = config.activation_dropout

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.activation(self.linear1(hidden_states))
        hidden_states = F.dropout(
            hidden_states,
            p=self.activation_dropout,
            training=self.training,
        )
        return self.linear2(hidden_states)


class ConvolutionModule(nn.Module):
    """Conformer GLU/depthwise convolution branch."""

    def __init__(self, config: ParakeetEncoderConfig) -> None:
        super().__init__()
        channels = config.hidden_size
        padding = (config.conv_kernel_size - 1) // 2
        self.pointwise_conv1 = nn.Conv1d(
            channels,
            channels * 2,
            kernel_size=1,
            bias=config.convolution_bias,
        )
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=config.conv_kernel_size,
            padding=padding,
            groups=channels,
            bias=config.convolution_bias,
        )
        self.norm = nn.BatchNorm1d(channels)
        self.pointwise_conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
            bias=config.convolution_bias,
        )
        self.activation = _activation(config.hidden_act)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = F.glu(self.pointwise_conv1(hidden_states), dim=1)
        if attention_mask is not None:
            invalid_columns = ~attention_mask.any(dim=2)
            hidden_states = hidden_states.masked_fill(invalid_columns, 0.0)
        hidden_states = self.depthwise_conv(hidden_states)
        hidden_states = self.norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.pointwise_conv2(hidden_states)
        return hidden_states.transpose(1, 2)


def _repeat_key_values(hidden_states: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return hidden_states
    batch, heads, sequence, width = hidden_states.shape
    expanded = hidden_states[:, :, None].expand(
        batch,
        heads,
        repeats,
        sequence,
        width,
    )
    return expanded.reshape(batch, heads * repeats, sequence, width)


class RelativeSelfAttention(nn.Module):
    """Multi-head self attention with Transformer-XL relative logits."""

    def __init__(self, config: ParakeetEncoderConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.key_value_groups = (config.num_attention_heads // config.num_key_value_heads)
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.relative_k_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=False,
        )
        self.bias_u = nn.Parameter(torch.zeros(config.num_attention_heads, self.head_dim))
        self.bias_v = nn.Parameter(torch.zeros(config.num_attention_heads, self.head_dim))

    @staticmethod
    def _relative_shift(scores: torch.Tensor) -> torch.Tensor:
        batch, heads, query_length, position_length = scores.shape
        scores = F.pad(scores, (1, 0))
        scores = scores.view(batch, heads, -1, query_length)
        scores = scores[:, :, 1:].view(
            batch,
            heads,
            query_length,
            position_length,
        )
        return scores

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, sequence_length, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(
            batch,
            sequence_length,
            self.config.num_attention_heads,
            self.head_dim,
        ).transpose(1, 2)
        key = self.k_proj(hidden_states).view(
            batch,
            sequence_length,
            self.config.num_key_value_heads,
            self.head_dim,
        ).transpose(1, 2)
        value = self.v_proj(hidden_states).view(
            batch,
            sequence_length,
            self.config.num_key_value_heads,
            self.head_dim,
        ).transpose(1, 2)
        key = _repeat_key_values(key, self.key_value_groups)
        value = _repeat_key_values(value, self.key_value_groups)

        query_content = query + self.bias_u[None, :, None, :]
        query_position = query + self.bias_v[None, :, None, :]
        relative_key = self.relative_k_proj(position_embeddings).view(
            batch,
            -1,
            self.config.num_attention_heads,
            self.head_dim,
        )
        content_scores = torch.matmul(
            query_content,
            key.transpose(2, 3),
        )
        position_scores = torch.matmul(
            query_position,
            relative_key.permute(0, 2, 3, 1),
        )
        position_scores = self._relative_shift(position_scores)[..., :sequence_length]
        scores = (content_scores + position_scores) * self.scaling
        valid_queries = None
        if attention_mask is not None:
            scores = scores.masked_fill(~attention_mask, float("-inf"))
            valid_queries = attention_mask.any(dim=-1, keepdim=True)
            # PyTorch SDPA, used by the audited upstream runtime, returns
            # zeros for a fully masked query. Plain softmax would return NaN.
            scores = torch.where(valid_queries, scores, torch.zeros_like(scores))
        weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        if valid_queries is not None:
            weights = torch.where(
                valid_queries,
                weights,
                torch.zeros_like(weights),
            )
        weights = F.dropout(
            weights,
            p=self.attention_dropout,
            training=self.training,
        )
        output = torch.matmul(weights, value)
        output = output.transpose(1, 2).reshape(batch, sequence_length, -1)
        return self.o_proj(output), weights


class SubsamplingConv2D(nn.Module):
    """Three-stage depthwise-separable 8x temporal/frequency subsampling."""

    def __init__(self, config: ParakeetEncoderConfig) -> None:
        super().__init__()
        self.kernel_size = config.subsampling_conv_kernel_size
        self.stride = config.subsampling_conv_stride
        self.channels = config.subsampling_conv_channels
        self.padding = (self.kernel_size - 1) // 2
        self.num_layers = int(math.log2(config.subsampling_factor))
        layers: list[nn.Module] = [
            nn.Conv2d(
                1,
                self.channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
            ),
            nn.ReLU(),
        ]
        for _ in range(self.num_layers - 1):
            layers.extend((
                nn.Conv2d(
                    self.channels,
                    self.channels,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    groups=self.channels,
                ),
                nn.Conv2d(self.channels, self.channels, kernel_size=1),
                nn.ReLU(),
            ))
        self.layers = nn.ModuleList(layers)
        output_frequency = config.num_mel_bins // (self.stride**self.num_layers)
        self.linear = nn.Linear(
            self.channels * output_frequency,
            config.hidden_size,
        )

    @staticmethod
    def _output_length(
        lengths: torch.Tensor,
        convolution: nn.Conv2d,
    ) -> torch.Tensor:
        if convolution.stride == (1, 1):
            return lengths
        padding = convolution.padding[0] * 2
        kernel = convolution.kernel_size[0]
        stride = convolution.stride[0]
        return (lengths + padding - kernel) // stride + 1

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = input_features.unsqueeze(1)
        lengths = (attention_mask.sum(-1).to(torch.long) if attention_mask is not None else None)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
            if isinstance(layer, nn.Conv2d) and lengths is not None:
                lengths = self._output_length(lengths, layer)
                valid = (
                    torch.arange(
                        hidden_states.shape[2],
                        device=hidden_states.device,
                    )[None, :] < lengths[:, None])
                hidden_states = hidden_states * valid[:, None, :, None]
        hidden_states = hidden_states.transpose(1, 2).reshape(
            hidden_states.shape[0],
            hidden_states.shape[2],
            -1,
        )
        return self.linear(hidden_states)


class EncoderBlock(nn.Module):

    def __init__(self, config: ParakeetEncoderConfig, layer_idx: int) -> None:
        super().__init__()
        self.feed_forward1 = FeedForwardModule(config)
        self.self_attn = RelativeSelfAttention(config, layer_idx)
        self.conv = ConvolutionModule(config)
        self.feed_forward2 = FeedForwardModule(config)
        self.norm_feed_forward1 = nn.LayerNorm(config.hidden_size)
        self.norm_self_att = nn.LayerNorm(config.hidden_size)
        self.norm_conv = nn.LayerNorm(config.hidden_size)
        self.norm_feed_forward2 = nn.LayerNorm(config.hidden_size)
        self.norm_out = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = hidden_states + 0.5 * self.feed_forward1(self.norm_feed_forward1(hidden_states))
        attention_output, _ = self.self_attn(
            self.norm_self_att(hidden_states),
            position_embeddings,
            attention_mask,
        )
        hidden_states = hidden_states + attention_output
        hidden_states = hidden_states + self.conv(
            self.norm_conv(hidden_states),
            attention_mask,
        )
        hidden_states = hidden_states + 0.5 * self.feed_forward2(self.norm_feed_forward2(hidden_states))
        return self.norm_out(hidden_states)


class ParakeetEncoder(nn.Module):
    """FastConformer acoustic encoder."""

    def __init__(self, config: ParakeetEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False
        self.dropout = config.dropout
        self.dropout_positions = config.dropout_positions
        self.layerdrop = config.layerdrop
        self.input_scale = math.sqrt(config.hidden_size) if config.scale_input else 1.0
        self.subsampling = SubsamplingConv2D(config)
        self.encode_positions = RelativePositionalEncoding(config)
        self.layers = nn.ModuleList(EncoderBlock(config, index) for index in range(config.num_hidden_layers))

    def _get_subsampling_output_length(
        self,
        input_lengths: torch.Tensor,
    ) -> torch.Tensor:
        config = self.config
        padding = (config.subsampling_conv_kernel_size - 1) // 2 * 2
        add_padding = padding - config.subsampling_conv_kernel_size
        lengths = input_lengths.float()
        for _ in range(int(math.log2(config.subsampling_factor))):
            lengths = torch.floor((lengths + add_padding) / config.subsampling_conv_stride + 1.0)
        return lengths.to(torch.int)

    def _get_output_attention_mask(
        self,
        attention_mask: torch.Tensor,
        *,
        target_length: int,
    ) -> torch.Tensor:
        lengths = self._get_subsampling_output_length(attention_mask.sum(-1))
        positions = torch.arange(target_length, device=attention_mask.device)
        return positions[None, :] < lengths[:, None]

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> ParakeetEncoderOutput:
        if input_features.ndim != 3:
            raise ValueError("Parakeet input features must have shape [batch, frames, mel].")
        if input_features.shape[-1] != self.config.num_mel_bins:
            raise ValueError(
                f"Parakeet expects {self.config.num_mel_bins} mel bins; "
                f"received {input_features.shape[-1]}.")
        if attention_mask is not None:
            if attention_mask.shape != input_features.shape[:2]:
                raise ValueError("Parakeet attention mask must match batch and frame dimensions.")
            attention_mask = attention_mask.bool()
        hidden_states = self.subsampling(input_features, attention_mask)
        hidden_states = hidden_states * self.input_scale
        positions = self.encode_positions(hidden_states)
        hidden_states = F.dropout(
            hidden_states,
            p=self.dropout,
            training=self.training,
        )
        positions = F.dropout(
            positions,
            p=self.dropout_positions,
            training=self.training,
        )
        output_mask = None
        square_mask = None
        if attention_mask is not None:
            output_mask = self._get_output_attention_mask(
                attention_mask,
                target_length=hidden_states.shape[1],
            )
            square_mask = output_mask[:, None, None, :] & output_mask[:, None, :, None]
        for layer in self.layers:
            drop_layer = (
                self.training and self.layerdrop and torch.rand(
                    (), device=hidden_states.device) < self.layerdrop)
            if drop_layer:
                continue
            if self.gradient_checkpointing and self.training:
                hidden_states = checkpoint(
                    layer,
                    hidden_states,
                    square_mask,
                    positions,
                    use_reentrant=False,
                )
            else:
                hidden_states = layer(hidden_states, square_mask, positions)
        return ParakeetEncoderOutput(
            last_hidden_state=hidden_states,
            attention_mask=(output_mask.to(torch.int) if output_mask is not None else None),
        )


class ParakeetDecoderCache:
    """Mutable LSTM state used by blank-skipping greedy decoding."""

    def __init__(self, config: ParakeetTDTConfig) -> None:
        self.config = config
        self.output: torch.Tensor | None = None
        self.hidden_state: torch.Tensor | None = None
        self.cell_state: torch.Tensor | None = None

    @property
    def is_initialized(self) -> bool:
        return self.output is not None

    def initialize(self, reference: torch.Tensor) -> None:
        batch = reference.shape[0]
        common = {
            "device": reference.device,
            "dtype": reference.dtype,
        }
        self.output = torch.zeros(
            batch,
            1,
            self.config.decoder_hidden_size,
            **common,
        )
        self.hidden_state = torch.zeros(
            self.config.num_decoder_layers,
            batch,
            self.config.decoder_hidden_size,
            **common,
        )
        self.cell_state = torch.zeros_like(self.hidden_state)

    def update(
        self,
        output: torch.Tensor,
        hidden_state: torch.Tensor,
        cell_state: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
    ) -> None:
        if not self.is_initialized:
            self.initialize(output)
        assert self.output is not None
        assert self.hidden_state is not None
        assert self.cell_state is not None
        if mask is None:
            self.output.copy_(output)
            self.hidden_state.copy_(hidden_state)
            self.cell_state.copy_(cell_state)
            return
        self.output = torch.where(mask[:, None, None], output, self.output)
        state_mask = mask[None, :, None]
        self.hidden_state = torch.where(
            state_mask,
            hidden_state,
            self.hidden_state,
        )
        self.cell_state = torch.where(
            state_mask,
            cell_state,
            self.cell_state,
        )


class ParakeetTDTDecoder(nn.Module):
    """LSTM prediction network."""

    def __init__(self, config: ParakeetTDTConfig) -> None:
        super().__init__()
        self.blank_token_id = config.blank_token_id
        self.embedding = nn.Embedding(
            config.vocab_size,
            config.decoder_hidden_size,
        )
        self.lstm = nn.LSTM(
            input_size=config.decoder_hidden_size,
            hidden_size=config.decoder_hidden_size,
            num_layers=config.num_decoder_layers,
            batch_first=True,
        )
        self.decoder_projector = nn.Linear(
            config.decoder_hidden_size,
            config.decoder_hidden_size,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        cache: ParakeetDecoderCache | None = None,
    ) -> torch.Tensor:
        if input_ids is None or input_ids.ndim != 2:
            raise ValueError("Parakeet decoder IDs must have shape [batch, sequence].")
        blank_mask = input_ids[:, -1] == self.blank_token_id
        if cache is not None and cache.is_initialized and torch.all(blank_mask):
            assert cache.output is not None
            return cache.output
        embeddings = self.embedding(input_ids)
        was_initialized = cache is not None and cache.is_initialized
        state = None
        if cache is not None:
            if not cache.is_initialized:
                cache.initialize(embeddings)
            assert cache.hidden_state is not None
            assert cache.cell_state is not None
            state = (cache.hidden_state, cache.cell_state)
        output, (hidden_state, cell_state) = self.lstm(embeddings, state)
        output = self.decoder_projector(output)
        if cache is not None:
            cache.update(
                output,
                hidden_state,
                cell_state,
                mask=(~blank_mask if was_initialized else None),
            )
            assert cache.output is not None
            return cache.output
        return output


class ParakeetTDTJointNetwork(nn.Module):
    """Joint token and duration classifier."""

    def __init__(self, config: ParakeetTDTConfig) -> None:
        super().__init__()
        self.activation = _activation(config.hidden_act)
        self.head = nn.Linear(
            config.decoder_hidden_size,
            config.vocab_size + len(config.durations),
        )

    def forward(
        self,
        *,
        decoder_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.head(self.activation(encoder_hidden_states + decoder_hidden_states))


class ParakeetForTDT(nn.Module):
    """Complete native Parakeet Token-and-Duration Transducer."""

    def __init__(
        self,
        config: ParakeetTDTConfig | dict[str, Any],
        *,
        initialize: bool = True,
    ) -> None:
        super().__init__()
        self.config = ParakeetTDTConfig.coerce(config)
        self.encoder = ParakeetEncoder(self.config.encoder_config)
        self.encoder_projector = nn.Linear(
            self.config.encoder_config.hidden_size,
            self.config.decoder_hidden_size,
        )
        self.decoder = ParakeetTDTDecoder(self.config)
        self.joint = ParakeetTDTJointNetwork(self.config)
        self.max_symbols_per_step = self.config.max_symbols_per_step
        if initialize:
            self.apply(self._initialize_module)

    def _initialize_module(self, module: nn.Module) -> None:
        standard_deviation = self.config.encoder_config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            nn.init.normal_(module.weight, mean=0.0, std=standard_deviation)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=standard_deviation)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, RelativeSelfAttention):
            nn.init.normal_(
                module.bias_u,
                mean=0.0,
                std=standard_deviation,
            )
            nn.init.normal_(
                module.bias_v,
                mean=0.0,
                std=standard_deviation,
            )

    def gradient_checkpointing_enable(self) -> None:
        self.encoder.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self.encoder.gradient_checkpointing = False

    def get_audio_features(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> ParakeetEncoderOutput:
        output = self.encoder(input_features, attention_mask)
        output.pooler_output = self.encoder_projector(output.last_hidden_state)
        return output

    def forward(
        self,
        input_features: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        decoder_input_ids: torch.Tensor | None = None,
        decoder_cache: ParakeetDecoderCache | None = None,
        use_decoder_cache: bool = False,
        encoder_outputs: ParakeetEncoderOutput | None = None,
        labels: torch.Tensor | None = None,
        *,
        sigma: float = 0.0,
        reduction: str = "mean",
    ) -> ParakeetTDTOutput:
        if encoder_outputs is None:
            if input_features is None:
                raise ValueError("Parakeet requires input features or encoder outputs.")
            encoder_outputs = self.get_audio_features(
                input_features,
                attention_mask,
            )
        if encoder_outputs.pooler_output is None:
            encoder_outputs.pooler_output = self.encoder_projector(encoder_outputs.last_hidden_state)
        if decoder_input_ids is None:
            raise ValueError("Parakeet requires `decoder_input_ids`.")
        if use_decoder_cache and decoder_cache is None:
            decoder_cache = ParakeetDecoderCache(self.config)
        decoder_hidden = self.decoder(decoder_input_ids, cache=decoder_cache)
        logits = self.joint(
            encoder_hidden_states=encoder_outputs.pooler_output[:, :, None, :],
            decoder_hidden_states=decoder_hidden[:, None, :, :],
        )
        if decoder_hidden.shape[1] == 1:
            logits = logits.squeeze(2)
        loss = None
        if labels is not None:
            if logits.ndim != 4:
                raise ValueError(
                    "TDT training requires full decoder sequences, not cached "
                    "single-step decoding.")
            if encoder_outputs.attention_mask is None:
                raise ValueError("TDT training requires an encoder attention mask.")
            if labels.ndim != 2:
                raise ValueError("TDT labels must have shape [batch, labels].")
            if decoder_input_ids.shape != (
                    labels.shape[0],
                    labels.shape[1] + 1,
            ):
                raise ValueError(
                    "TDT decoder inputs must contain one blank prefix followed "
                    "by every padded label.")
            if torch.any(decoder_input_ids[:, 0] != self.config.blank_token_id) or not torch.equal(
                    decoder_input_ids[:, 1:], labels):
                raise ValueError("TDT decoder inputs must equal `[blank] + labels` exactly.")
            label_mask = labels != self.config.pad_token_id
            label_lengths = label_mask.sum(-1)
            expected_mask = (
                torch.arange(labels.shape[1], device=labels.device)[None, :] < label_lengths[:, None])
            if not torch.equal(label_mask, expected_mask):
                raise ValueError(
                    "TDT labels must use contiguous right padding; pad IDs "
                    "cannot appear inside supervised targets.")
            loss = tdt_loss(
                logits[..., :self.config.vocab_size],
                logits[..., self.config.vocab_size:],
                labels,
                encoder_outputs.attention_mask.sum(-1),
                label_lengths,
                self.config.blank_token_id,
                self.config.durations,
                sigma=sigma,
                reduction=reduction,
            )
        return ParakeetTDTOutput(
            loss=loss,
            logits=logits,
            last_hidden_state=encoder_outputs.last_hidden_state,
            pooler_output=encoder_outputs.pooler_output,
            attention_mask=encoder_outputs.attention_mask,
            decoder_cache=decoder_cache,
        )

    @torch.no_grad()
    def generate(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        maximum_steps: int | None = None,
    ) -> ParakeetGenerateOutput:
        """Greedily decode tokens and TDT frame durations.

        The implementation intentionally supports greedy decoding only.
        Beam search requires a separate, explicitly validated transducer
        decoder.
        """
        encoder_output = self.get_audio_features(input_features, attention_mask)
        projected = encoder_output.pooler_output
        assert projected is not None
        batch, encoded_frames, _ = projected.shape
        if encoder_output.attention_mask is None:
            valid_lengths = torch.full(
                (batch, ),
                encoded_frames,
                dtype=torch.long,
                device=projected.device,
            )
        else:
            valid_lengths = encoder_output.attention_mask.sum(-1).long()
        if maximum_steps is None:
            maximum_steps = self.max_symbols_per_step * encoded_frames
        if isinstance(maximum_steps, bool) or maximum_steps < 1:
            raise ValueError("`maximum_steps` must be a positive integer.")

        frame_indices = torch.zeros(
            batch,
            dtype=torch.long,
            device=projected.device,
        )
        previous_tokens = torch.full(
            (batch, 1),
            self.config.blank_token_id,
            dtype=torch.long,
            device=projected.device,
        )
        sequence_steps = [previous_tokens[:, 0]]
        duration_steps = [torch.zeros_like(previous_tokens[:, 0])]
        cache = ParakeetDecoderCache(self.config)
        finished = frame_indices >= valid_lengths

        for _ in range(maximum_steps):
            if bool(torch.all(finished)):
                break
            safe_indices = frame_indices.clamp(max=encoded_frames - 1)
            frame = projected[
                torch.arange(batch, device=projected.device),
                safe_indices,
            ][:, None, :]
            decoder_hidden = self.decoder(previous_tokens, cache=cache)
            logits = self.joint(
                encoder_hidden_states=frame,
                decoder_hidden_states=decoder_hidden,
            )[:, -1]
            next_tokens = logits[:, :self.config.vocab_size].argmax(-1)
            duration_indices = logits[:, self.config.vocab_size:].argmax(-1)
            configured_durations = torch.tensor(
                self.config.durations,
                device=duration_indices.device,
                dtype=torch.long,
            )
            step_durations = configured_durations[duration_indices]
            step_durations = torch.where(
                (next_tokens == self.config.blank_token_id)
                & (step_durations == 0),
                torch.ones_like(step_durations),
                step_durations,
            )
            next_tokens = torch.where(
                finished,
                torch.full_like(next_tokens, self.config.pad_token_id),
                next_tokens,
            )
            step_durations = torch.where(
                finished,
                torch.zeros_like(step_durations),
                step_durations,
            )
            sequence_steps.append(next_tokens)
            duration_steps.append(step_durations)
            frame_indices = frame_indices + step_durations
            finished = frame_indices >= valid_lengths
            previous_tokens = next_tokens[:, None]
        else:
            if not bool(torch.all(finished)):
                raise RuntimeError(
                    "Parakeet TDT decoding reached its safety bound before "
                    "exhausting the encoder. The checkpoint may be incompatible.")
        return ParakeetGenerateOutput(
            sequences=torch.stack(sequence_steps, dim=1),
            durations=torch.stack(duration_steps, dim=1),
        )


__all__ = [
    "ParakeetDecoderCache",
    "ParakeetEncoder",
    "ParakeetEncoderOutput",
    "ParakeetForTDT",
    "ParakeetGenerateOutput",
    "ParakeetTDTOutput",
]
