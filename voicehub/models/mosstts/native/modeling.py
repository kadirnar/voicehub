"""VoiceHub-native models for every official MOSS-TTS graph family.

The module mirrors the published Safetensors namespaces but owns its runtime:
there are no Transformers, Accelerate, PEFT, or vendor-runtime imports.  The
four model variants deliberately expose one common training contract while
retaining their architecture-specific logits and generation schedules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import torch
from torch import Tensor, nn
from torch.nn import functional

from voicehub.models.causal_lm.native.modeling import CausalLMModel
from voicehub.models.mosstts.native.configuration import MossGPT2Config, MossTTSConfig
from voicehub.models.mosstts.native.local_transformer import MossGPT2Model, MossQwenDepthModel
from voicehub.models.mosstts.native.sampling import sample_token
from voicehub.neural.cache import DynamicKVCache
from voicehub.neural.normalization import RMSNorm
from voicehub.optimization.protocols import OptimizationCompileTarget

MossCache: TypeAlias = DynamicKVCache | tuple[tuple[Tensor, Tensor], ...]


@dataclass(frozen=True)
class MossTTSOutput:
    """Unified native MOSS-TTS forward output."""

    loss: Tensor | None
    logits: tuple[Tensor, ...]
    last_hidden_state: Tensor
    past_key_values: MossCache | None = None
    channel_losses: tuple[Tensor, ...] | None = None


def _validate_multichannel_ids(
    input_ids: Tensor,
    *,
    channels: int,
    name: str = "input_ids",
) -> None:
    if not isinstance(input_ids, Tensor) or input_ids.ndim != 3:
        raise ValueError(f"`{name}` must have shape [batch, sequence, channels].")
    if input_ids.shape[-1] != channels:
        raise ValueError(f"`{name}` requires {channels} channels; found {input_ids.shape[-1]}.")
    if (input_ids.dtype == torch.bool or input_ids.is_floating_point() or input_ids.is_complex()):
        raise TypeError(f"`{name}` must use an integer dtype.")


def _loss_weights(
    count: int,
    value: tuple[float, ...] | list[float] | None,
) -> Tensor:
    if value is None:
        values = [1.0] * count
    else:
        values = [float(item) for item in value]
        if len(values) == 2 and count > 1:
            text_weight, total_audio_weight = values
            values = [
                text_weight,
                *([total_audio_weight / (count - 1)] * (count - 1)),
            ]
    if len(values) != count:
        raise ValueError("Channel weights must have two entries or one entry per output head.")
    tensor = torch.tensor(values, dtype=torch.float32)
    if not bool(torch.isfinite(tensor).all()) or bool((tensor < 0).any()):
        raise ValueError("Channel weights must be finite and non-negative.")
    if float(tensor.sum()) <= 0:
        raise ValueError("Channel weights must sum to a positive value.")
    return tensor


def _channelwise_cross_entropy(
    logits: tuple[Tensor, ...],
    labels: Tensor,
    *,
    weights: tuple[float, ...] | list[float] | None,
) -> tuple[Tensor, tuple[Tensor, ...]]:
    if labels.ndim != 3 or labels.shape[-1] != len(logits):
        raise ValueError("Labels must match the multichannel output layout.")
    resolved = _loss_weights(len(logits), weights).to(device=labels.device)
    losses: list[Tensor] = []
    active_weights: list[Tensor] = []
    for index, channel_logits in enumerate(logits):
        targets = labels[..., index]
        if tuple(channel_logits.shape[:2]) != tuple(targets.shape):
            raise ValueError("Channel labels must match logits batch and sequence.")
        if bool((targets != -100).any()):
            loss = functional.cross_entropy(
                channel_logits.float().reshape(-1, channel_logits.shape[-1]),
                targets.reshape(-1),
                ignore_index=-100,
            )
            active_weights.append(resolved[index])
        else:
            loss = channel_logits.float().sum() * 0.0
            active_weights.append(resolved[index] * 0.0)
        losses.append(loss)
    denominator = torch.stack(active_weights).sum()
    if float(denominator.detach()) <= 0:
        raise ValueError("MOSS-TTS received a batch with all labels ignored.")
    total = sum(weight * loss for weight, loss in zip(active_weights, losses)) / denominator
    return total, tuple(losses)


def _find_last_equal(input_ids: Tensor, value: int) -> Tensor:
    matches = input_ids.eq(value)
    if not bool(matches.any(dim=1).all()):
        raise ValueError(f"Every sequence must contain token ID {value}.")
    positions = torch.arange(
        input_ids.shape[1],
        device=input_ids.device,
        dtype=torch.long,
    )
    return positions.unsqueeze(0).masked_fill(~matches, -1).max(dim=1).values


class MossGatedMLP(nn.Module):
    """SwiGLU bridge with the names used by the older Local release."""

    def __init__(
        self,
        input_size: int,
        intermediate_size: int,
        output_size: int,
        *,
        initialize: bool,
        initializer_range: float,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.gate_proj = nn.Linear(
            input_size,
            intermediate_size,
            bias=False,
            **factory_kwargs,
        )
        self.up_proj = nn.Linear(
            input_size,
            intermediate_size,
            bias=False,
            **factory_kwargs,
        )
        self.down_proj = nn.Linear(
            intermediate_size,
            output_size,
            bias=False,
            **factory_kwargs,
        )
        if initialize:
            for module in (self.gate_proj, self.up_proj, self.down_proj):
                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=initializer_range,
                )

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.down_proj(functional.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class MossDelayModel(nn.Module):
    """Delay-pattern Qwen3 model used by MOSS-TTS and MOSS-TTS-v1.5."""

    def __init__(
        self,
        config: MossTTSConfig,
        *,
        initialize: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if config.variant != "delay":
            raise ValueError("MossDelayModel requires `variant='delay'`.")
        self.config = config
        factory_kwargs = {"device": device, "dtype": dtype}
        self.language_model = CausalLMModel(
            config.language_config,
            initialize=initialize,
            **factory_kwargs,
        )
        self.emb_ext = nn.ModuleList([
            nn.Embedding(
                config.audio_vocab_size + 1,
                config.language_config.hidden_size,
                **factory_kwargs,
            ) for _ in range(config.n_vq)
        ])
        self.lm_heads = nn.ModuleList([
            nn.Linear(
                config.language_config.hidden_size,
                config.language_config.vocab_size,
                bias=False,
                **factory_kwargs,
            ),
            *[
                nn.Linear(
                    config.language_config.hidden_size,
                    config.audio_vocab_size + 1,
                    bias=False,
                    **factory_kwargs,
                ) for _ in range(config.n_vq)
            ],
        ])
        if initialize:
            for module in (*self.emb_ext, *self.lm_heads):
                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=config.initializer_range,
                )

    def optimization_compile_targets(
        self,
        mode: str,
    ) -> tuple[OptimizationCompileTarget, ...]:
        """Expose the uniform MOSS execution boundary for this variant."""
        if mode == "training":
            attribute = "forward"
        elif mode == "inference":
            attribute = "generate"
        else:
            raise ValueError(f"Unsupported optimization mode {mode!r}.")
        return (OptimizationCompileTarget(
            f"mosstts.{attribute}",
            self,
            attribute,
        ), )

    def _input_embeddings(self, input_ids: Tensor) -> Tensor:
        hidden_states = self.language_model.embed_tokens(input_ids[..., 0])
        for index, embedding in enumerate(self.emb_ext):
            hidden_states = hidden_states + embedding(input_ids[..., index + 1])
        return hidden_states

    def forward(
        self,
        input_ids: Tensor,
        *,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        past_key_values: DynamicKVCache | None = None,
        labels: Tensor | None = None,
        use_cache: bool = False,
        channelwise_loss_weight: tuple[float, ...] | list[float] | None = None,
    ) -> MossTTSOutput:
        _validate_multichannel_ids(input_ids, channels=self.config.channels)
        outputs = self.language_model(
            inputs_embeds=self._input_embeddings(input_ids),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        logits: list[Tensor] = []
        for index, head in enumerate(self.lm_heads):
            channel_logits = head(outputs.last_hidden_state)
            if index:
                invalid = torch.full_like(channel_logits[..., -1:], -torch.inf)
                channel_logits = torch.cat(
                    [channel_logits[..., :-1], invalid],
                    dim=-1,
                )
            logits.append(channel_logits)
        loss = None
        losses = None
        if labels is not None:
            _validate_multichannel_ids(
                labels,
                channels=self.config.channels,
                name="labels",
            )
            loss, losses = _channelwise_cross_entropy(
                tuple(logits),
                labels,
                weights=channelwise_loss_weight,
            )
        return MossTTSOutput(
            loss=loss,
            logits=tuple(logits),
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            channel_losses=losses,
        )

    @torch.inference_mode()
    def generate(
        self,
        input_ids: Tensor,
        *,
        attention_mask: Tensor | None = None,
        max_new_tokens: int = 1_000,
        text_temperature: float = 1.5,
        text_top_p: float = 1.0,
        text_top_k: int = 50,
        audio_temperature: float = 1.7,
        audio_top_p: float = 0.8,
        audio_top_k: int = 25,
        audio_repetition_penalty: float = 1.0,
    ) -> list[tuple[int, Tensor]]:
        """Generate with the source-audited diagonal delay schedule."""
        _validate_multichannel_ids(input_ids, channels=self.config.channels)
        if isinstance(max_new_tokens, bool) or max_new_tokens <= 0:
            raise ValueError("`max_new_tokens` must be a positive integer.")
        if input_ids.shape[1] == 0:
            raise ValueError("MOSS-TTS generation requires a non-empty prompt.")
        batch_size, prompt_length, _ = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones(
                batch_size,
                prompt_length,
                dtype=torch.bool,
                device=input_ids.device,
            )
        if attention_mask.shape != input_ids.shape[:2]:
            raise ValueError("Attention mask must match the prompt batch and length.")
        current_ids = input_ids
        current_mask = attention_mask.to(dtype=torch.bool)
        generated = input_ids.clone()
        cache = None
        stopped = torch.zeros(
            batch_size,
            dtype=torch.bool,
            device=input_ids.device,
        )
        audio_lengths = torch.zeros(
            batch_size,
            dtype=torch.long,
            device=input_ids.device,
        )
        maximum = torch.iinfo(torch.long).max
        delayed_lengths = torch.full_like(audio_lengths, maximum)
        text_ids = input_ids[..., 0]
        audio_start_indices = _find_last_equal(
            text_ids,
            self.config.audio_start_token_id,
        )
        continuation = (
            text_ids[:, -1].eq(self.config.audio_start_token_id)
            | text_ids[:, -1].eq(self.config.audio_assistant_slot_token_id))
        audio_start_mask = continuation & audio_start_indices.ne(-1)
        audio_lengths[audio_start_mask] = (prompt_length - audio_start_indices[audio_start_mask])
        in_audio = audio_start_mask.clone()
        excluded_before_audio = torch.tensor(
            [
                self.config.pad_token_id,
                self.config.audio_assistant_slot_token_id,
                self.config.audio_assistant_delay_slot_token_id,
                self.config.audio_end_token_id,
            ],
            device=input_ids.device,
        )
        audio_text_mask = torch.ones(
            self.config.language_config.vocab_size,
            dtype=torch.bool,
            device=input_ids.device,
        )
        audio_text_mask[[
            self.config.audio_assistant_slot_token_id,
            self.config.audio_assistant_delay_slot_token_id,
        ]] = False

        for step in range(max_new_tokens):
            output = self(
                current_ids,
                attention_mask=current_mask,
                past_key_values=cache,
                use_cache=True,
            )
            cache = output.past_key_values
            text_logits = output.logits[0][:, -1].clone()
            next_text = torch.full(
                (batch_size, ),
                self.config.pad_token_id,
                dtype=torch.long,
                device=input_ids.device,
            )
            next_text[~stopped & (delayed_lengths < self.config.n_vq)] = (
                self.config.audio_assistant_delay_slot_token_id)
            audio_eos = ~stopped & delayed_lengths.eq(self.config.n_vq)
            next_text[audio_eos] = self.config.audio_end_token_id
            in_audio[audio_eos] = False
            sampling_text = ~stopped & (delayed_lengths > self.config.n_vq)
            text_logits[~in_audio] = text_logits[~in_audio].index_fill(
                -1,
                excluded_before_audio,
                -torch.inf,
            )
            text_logits[in_audio] = text_logits[in_audio].masked_fill(
                audio_text_mask,
                -torch.inf,
            )
            if step == 0:
                text_logits[
                    ...,
                    self.config.audio_assistant_delay_slot_token_id,
                ] = -torch.inf
            if step <= self.config.n_vq:
                text_logits[..., self.config.im_end_token_id] = -torch.inf
            if bool(sampling_text.any()):
                next_text[sampling_text] = sample_token(
                    text_logits[sampling_text],
                    temperature=text_temperature,
                    top_k=text_top_k,
                    top_p=text_top_p,
                )
            in_audio |= next_text.eq(self.config.audio_start_token_id)
            stopped |= next_text.eq(self.config.im_end_token_id)

            next_audio = torch.full(
                (batch_size, self.config.n_vq),
                self.config.audio_pad_token_id,
                dtype=torch.long,
                device=input_ids.device,
            )
            codebook = torch.arange(
                self.config.n_vq,
                device=input_ids.device,
            ).expand(batch_size, -1)
            pre_audio = audio_lengths.unsqueeze(1) > codebook
            post_audio = codebook > delayed_lengths.unsqueeze(1) - 1
            post_audio[delayed_lengths.eq(maximum)] = True
            sample_audio = pre_audio & post_audio
            for channel in range(self.config.n_vq):
                active = sample_audio[:, channel]
                if not bool(active.any()):
                    continue
                channel_logits = output.logits[channel + 1][:, -1][active]
                next_audio[active, channel] = sample_token(
                    channel_logits,
                    temperature=audio_temperature,
                    top_k=audio_top_k,
                    top_p=audio_top_p,
                    repetition_penalty=audio_repetition_penalty,
                    previous_token_ids=generated[active, :, channel + 1],
                )
            audio_lengths[next_text.eq(self.config.audio_start_token_id)
                          | next_text.eq(self.config.audio_assistant_slot_token_id)
                          | next_text.eq(self.config.audio_assistant_delay_slot_token_id)] += 1
            audio_lengths[next_text.eq(self.config.audio_end_token_id)] = 0
            starts_delay = (
                delayed_lengths.eq(maximum)
                & next_text.eq(self.config.audio_assistant_delay_slot_token_id))
            delayed_lengths[starts_delay] = 0
            delayed_lengths[delayed_lengths.ne(maximum)] += 1
            delayed_lengths[delayed_lengths > self.config.n_vq] = maximum
            current_ids = torch.cat(
                [next_text[:, None, None], next_audio[:, None]],
                dim=-1,
            )
            current_mask = torch.cat(
                [current_mask, (~stopped).unsqueeze(1)],
                dim=1,
            )
            generated = torch.cat([generated, current_ids], dim=1)
            if bool(stopped.all()):
                break

        starts = _find_last_equal(
            input_ids[..., 0],
            self.config.im_start_token_id,
        ) + 3
        outputs: list[tuple[int, Tensor]] = []
        for start, sequence in zip(starts.tolist(), generated):
            outputs.append((
                prompt_length - int(start),
                sequence[int(start):],
            ))
        return outputs


class MossOldLocalBackbone(nn.Module):
    """Multichannel embedding front-end and Qwen3 global decoder."""

    def __init__(
        self,
        config: MossTTSConfig,
        *,
        initialize: bool,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.embedding_list = nn.ModuleList([
            nn.Embedding(
                config.language_config.vocab_size,
                config.language_config.hidden_size,
                config.pad_token_id,
                **factory_kwargs,
            ),
            *[
                nn.Embedding(
                    config.audio_vocab_size + 1,
                    config.language_config.hidden_size,
                    config.audio_pad_token_id,
                    **factory_kwargs,
                ) for _ in range(config.n_vq)
            ],
        ])
        self.language_model = CausalLMModel(
            config.language_config,
            initialize=initialize,
            **factory_kwargs,
        )
        if initialize:
            for embedding in self.embedding_list:
                nn.init.normal_(
                    embedding.weight,
                    mean=0.0,
                    std=config.initializer_range,
                )

    def embeddings(
        self,
        input_ids: Tensor,
        *,
        n_vq_for_inference: int | None = None,
    ) -> Tensor:
        channels = len(self.embedding_list)
        _validate_multichannel_ids(input_ids, channels=channels)
        active = channels - 1 if n_vq_for_inference is None else n_vq_for_inference
        if not 1 <= active <= channels - 1:
            raise ValueError("Requested RVQ inference depth is outside the graph.")
        hidden = torch.zeros(
            *input_ids.shape[:2],
            self.language_model.config.hidden_size,
            dtype=self.embedding_list[0].weight.dtype,
            device=input_ids.device,
        )
        for index in range(active + 1):
            hidden = hidden + self.embedding_list[index](input_ids[..., index])
        return hidden


class MossOldLocalModel(nn.Module):
    """The position-free Qwen depth model in Local Transformer 1.7B."""

    def __init__(
        self,
        config: MossTTSConfig,
        *,
        initialize: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if config.variant != "local":
            raise ValueError("MossOldLocalModel requires `variant='local'`.")
        if not hasattr(config.local_config, "hidden_size"):
            raise TypeError("Old Local requires a Qwen local configuration.")
        self.config = config
        local_config = config.local_config
        self.model = MossOldLocalBackbone(
            config,
            initialize=initialize,
            device=device,
            dtype=dtype,
        )
        self.local_transformer = MossQwenDepthModel(
            local_config,
            initialize=initialize,
            device=device,
            dtype=dtype,
        )
        intermediate = config.additional_mlp_ffn_hidden_size
        if intermediate is None:
            raise ValueError("Old Local config is missing its bridge MLP size.")
        self.speech_embedding_to_local_mlp = MossGatedMLP(
            config.language_config.hidden_size,
            intermediate,
            local_config.hidden_size,
            initialize=initialize,
            initializer_range=config.initializer_range,
            device=device,
            dtype=dtype,
        )
        self.local_to_speech_embedding_mlps = nn.ModuleList([
            MossGatedMLP(
                local_config.hidden_size,
                intermediate,
                config.language_config.hidden_size,
                initialize=initialize,
                initializer_range=config.initializer_range,
                device=device,
                dtype=dtype,
            ) for _ in range(config.channels)
        ])
        self.layer_norm_before_lm_heads = nn.ModuleList([
            RMSNorm(
                config.language_config.hidden_size,
                epsilon=1e-6,
                device=device,
                dtype=dtype,
            ) for _ in range(config.channels)
        ])
        self.lm_heads = nn.ModuleList([
            nn.Linear(
                config.language_config.hidden_size,
                config.language_config.vocab_size,
                bias=False,
                device=device,
                dtype=dtype,
            ),
            *[
                nn.Linear(
                    config.language_config.hidden_size,
                    config.audio_vocab_size + 1,
                    bias=False,
                    device=device,
                    dtype=dtype,
                ) for _ in range(config.n_vq)
            ],
        ])
        if initialize:
            for head in self.lm_heads:
                nn.init.normal_(
                    head.weight,
                    mean=0.0,
                    std=config.initializer_range,
                )

    def optimization_compile_targets(
        self,
        mode: str,
    ) -> tuple[OptimizationCompileTarget, ...]:
        """Expose the uniform MOSS execution boundary for this variant."""
        if mode == "training":
            attribute = "forward"
        elif mode == "inference":
            attribute = "generate"
        else:
            raise ValueError(f"Unsupported optimization mode {mode!r}.")
        return (OptimizationCompileTarget(
            f"mosstts.{attribute}",
            self,
            attribute,
        ), )

    def _project_local(
        self,
        global_hidden: Tensor,
        teacher_labels: Tensor,
    ) -> tuple[Tensor, ...]:
        batch_size, sequence_length, hidden_size = global_hidden.shape
        teacher: list[Tensor] = [global_hidden]
        # The official objective predicts text then all RVQ channels from a
        # depth sequence teacher-forced by text + RVQ[0:n-1].
        for index in range(self.config.n_vq):
            targets = teacher_labels[..., index]
            valid = targets.ne(-100)
            safe = targets.masked_fill(~valid, 0)
            embedded = self.model.embedding_list[index](safe)
            teacher.append(embedded * valid.unsqueeze(-1))
        local_inputs = torch.stack(teacher, dim=2)
        local_inputs = self.speech_embedding_to_local_mlp(local_inputs)
        local_inputs = local_inputs.reshape(
            batch_size * sequence_length,
            self.config.channels,
            -1,
        )
        local_hidden = self.local_transformer(
            inputs_embeds=local_inputs,
            use_cache=False,
            apply_rope=False,
        ).last_hidden_state
        result: list[Tensor] = []
        for index, (bridge, norm, head) in enumerate(zip(
                self.local_to_speech_embedding_mlps,
                self.layer_norm_before_lm_heads,
                self.lm_heads,
        )):
            channel_hidden = norm(bridge(local_hidden[:, index]))
            channel_hidden = channel_hidden.view(
                batch_size,
                sequence_length,
                hidden_size,
            )
            result.append(head(channel_hidden))
        return tuple(result)

    def forward(
        self,
        input_ids: Tensor,
        *,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        past_key_values: DynamicKVCache | None = None,
        labels: Tensor | None = None,
        use_cache: bool = False,
        channelwise_loss_weight: tuple[float, ...] | list[float] | None = None,
        n_vq_for_inference: int | None = None,
    ) -> MossTTSOutput:
        _validate_multichannel_ids(input_ids, channels=self.config.channels)
        global_output = self.model.language_model(
            inputs_embeds=self.model.embeddings(
                input_ids,
                n_vq_for_inference=n_vq_for_inference,
            ),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        if labels is None:
            return MossTTSOutput(
                loss=None,
                logits=(),
                last_hidden_state=global_output.last_hidden_state,
                past_key_values=global_output.past_key_values,
            )
        _validate_multichannel_ids(
            labels,
            channels=self.config.channels,
            name="labels",
        )
        logits = self._project_local(
            global_output.last_hidden_state,
            labels,
        )
        loss, losses = _channelwise_cross_entropy(
            logits,
            labels,
            weights=channelwise_loss_weight,
        )
        return MossTTSOutput(
            loss=loss,
            logits=logits,
            last_hidden_state=global_output.last_hidden_state,
            past_key_values=global_output.past_key_values,
            channel_losses=losses,
        )

    @torch.inference_mode()
    def generate(
        self,
        input_ids: Tensor,
        *,
        attention_mask: Tensor | None = None,
        max_new_tokens: int = 4_096,
        text_temperature: float = 1.5,
        text_top_p: float = 1.0,
        text_top_k: int = 50,
        audio_temperature: float = 1.0,
        audio_top_p: float = 0.95,
        audio_top_k: int = 50,
        audio_repetition_penalty: float = 1.1,
        n_vq_for_inference: int | None = None,
    ) -> list[tuple[int, Tensor]]:
        _validate_multichannel_ids(input_ids, channels=self.config.channels)
        depth = self.config.n_vq if n_vq_for_inference is None else int(n_vq_for_inference)
        if not 1 <= depth <= self.config.n_vq:
            raise ValueError("Inference RVQ depth is outside the trained graph.")
        if max_new_tokens <= 0:
            raise ValueError("`max_new_tokens` must be positive.")
        batch_size, prompt_length, _ = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones(
                batch_size,
                prompt_length,
                dtype=torch.bool,
                device=input_ids.device,
            )
        generated = input_ids.clone()
        current = input_ids
        mask = attention_mask.to(torch.bool)
        cache = None
        finished = torch.zeros(
            batch_size,
            dtype=torch.bool,
            device=input_ids.device,
        )
        for _ in range(max_new_tokens):
            output = self(
                current,
                attention_mask=mask,
                past_key_values=cache,
                use_cache=True,
                n_vq_for_inference=depth,
            )
            cache = output.past_key_values
            local_input = self.speech_embedding_to_local_mlp(output.last_hidden_state[:, -1]).unsqueeze(1)
            next_channels: list[Tensor] = []
            for channel in range(depth + 1):
                local_hidden = self.local_transformer(
                    inputs_embeds=local_input,
                    use_cache=False,
                    apply_rope=False,
                ).last_hidden_state[:, -1]
                channel_logits = self.lm_heads[channel](
                    self.layer_norm_before_lm_heads[channel](
                        self.local_to_speech_embedding_mlps[channel](local_hidden)))
                if channel:
                    channel_logits[..., self.config.audio_pad_token_id] = -torch.inf
                    previous = generated[..., channel]
                    token = sample_token(
                        channel_logits,
                        temperature=audio_temperature,
                        top_k=audio_top_k,
                        top_p=audio_top_p,
                        repetition_penalty=audio_repetition_penalty,
                        previous_token_ids=previous,
                    )
                else:
                    token = sample_token(
                        channel_logits,
                        temperature=text_temperature,
                        top_k=text_top_k,
                        top_p=text_top_p,
                    )
                next_channels.append(token)
                if channel < depth:
                    next_local = self.model.embedding_list[channel](token)
                    next_local = self.speech_embedding_to_local_mlp(next_local)
                    local_input = torch.cat(
                        [local_input, next_local.unsqueeze(1)],
                        dim=1,
                    )
            for _ in range(depth + 1, self.config.channels):
                next_channels.append(
                    torch.full(
                        (batch_size, ),
                        self.config.audio_pad_token_id,
                        dtype=torch.long,
                        device=input_ids.device,
                    ))
            row = torch.stack(next_channels, dim=-1)
            row[finished, 0] = self.config.audio_end_token_id
            row[finished, 1:] = self.config.audio_pad_token_id
            generated = torch.cat([generated, row[:, None]], dim=1)
            current = row[:, None]
            finished |= row[:, 0].eq(self.config.audio_end_token_id)
            mask = torch.cat([mask, (~finished).unsqueeze(1)], dim=1)
            if bool(finished.all()):
                break
        starts = _find_last_equal(
            input_ids[..., 0],
            self.config.audio_start_token_id,
        )
        return [(
            prompt_length - int(start) - 1,
            sequence[int(start):],
        ) for start, sequence in zip(starts.tolist(), generated)]


class MossLocalV15Model(nn.Module):
    """Local Transformer v1.5 with fixed 12-codebook depth."""

    def __init__(
        self,
        config: MossTTSConfig,
        *,
        initialize: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if config.variant != "local_v1_5":
            raise ValueError("MossLocalV15Model requires `variant='local_v1_5'`.")
        if not isinstance(config.local_config, MossGPT2Config):
            raise TypeError("Local v1.5 requires MossGPT2Config.")
        self.config = config
        self.transformer = CausalLMModel(
            config.language_config,
            initialize=initialize,
            device=device,
            dtype=dtype,
        )
        self.local_transformer = MossGPT2Model(
            config.local_config,
            initialize=initialize,
            device=device,
            dtype=dtype,
        )
        self.audio_embeddings = nn.ModuleList([
            nn.Embedding(
                size,
                config.language_config.hidden_size,
                device=device,
                dtype=dtype,
            ) for size in config.audio_codebook_sizes
        ])
        self.text_lm_head = nn.Linear(
            config.language_config.hidden_size,
            config.language_config.vocab_size,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.audio_lm_heads = nn.ModuleList([
            nn.Linear(
                config.language_config.hidden_size,
                size,
                bias=False,
                device=device,
                dtype=dtype,
            ) for size in config.audio_codebook_sizes
        ])
        self.local_text_lm_head = nn.Linear(
            config.language_config.hidden_size,
            2,
            bias=False,
            device=device,
            dtype=dtype,
        )
        if initialize:
            for module in (
                    *self.audio_embeddings,
                    self.text_lm_head,
                    *self.audio_lm_heads,
                    self.local_text_lm_head,
            ):
                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=config.initializer_range,
                )

    def optimization_compile_targets(
        self,
        mode: str,
    ) -> tuple[OptimizationCompileTarget, ...]:
        """Expose the uniform MOSS execution boundary for this variant."""
        if mode == "training":
            attribute = "forward"
        elif mode == "inference":
            attribute = "generate"
        else:
            raise ValueError(f"Unsupported optimization mode {mode!r}.")
        return (OptimizationCompileTarget(
            f"mosstts.{attribute}",
            self,
            attribute,
        ), )

    def tie_weights(self) -> None:
        """Restore the source-declared embedding/head aliases after loading."""
        self.text_lm_head.weight = self.transformer.embed_tokens.weight
        for embedding, head in zip(self.audio_embeddings, self.audio_lm_heads):
            head.weight = embedding.weight

    def initialize_binary_head(self) -> None:
        candidate_ids = torch.tensor(
            [
                self.config.audio_assistant_slot_token_id,
                self.config.audio_end_token_id,
            ],
            device=self.text_lm_head.weight.device,
        )
        with torch.no_grad():
            self.local_text_lm_head.weight.copy_(self.text_lm_head.weight.index_select(0, candidate_ids))

    def _input_embeddings(self, input_ids: Tensor) -> Tensor:
        _validate_multichannel_ids(input_ids, channels=self.config.channels)
        hidden = self.transformer.embed_tokens(input_ids[..., 0])
        for index, embedding in enumerate(self.audio_embeddings):
            ids = input_ids[..., index + 1]
            valid = ids.ne(self.config.audio_pad_token_id)
            safe = ids.masked_fill(~valid, 0)
            hidden = hidden + embedding(safe) * valid.unsqueeze(-1)
        return hidden

    def _local_objective(
        self,
        hidden_states: Tensor,
        labels: Tensor,
        *,
        channelwise_loss_weight: tuple[float, ...] | list[float] | None,
    ) -> tuple[Tensor, tuple[Tensor, ...], tuple[Tensor, ...]]:
        _validate_multichannel_ids(
            labels,
            channels=self.config.channels,
            name="labels",
        )
        batch_size, sequence_length, hidden_size = hidden_states.shape
        flat_hidden = hidden_states.reshape(-1, hidden_size)
        flat_labels = labels.reshape(-1, self.config.channels)
        local_inputs = torch.zeros(
            flat_hidden.shape[0],
            self.config.n_vq,
            hidden_size,
            dtype=self.local_transformer.ln_f.weight.dtype,
            device=hidden_states.device,
        )
        local_inputs[:, 0] = flat_hidden.to(local_inputs.dtype)
        audio_targets = flat_labels[:, 1:]
        for index in range(self.config.n_vq - 1):
            targets = audio_targets[:, index]
            embedding = self.audio_embeddings[index]
            valid = (targets >= 0) & (targets < embedding.num_embeddings)
            safe = targets.masked_fill(~valid, 0)
            local_inputs[:, index + 1] = (embedding(safe).to(local_inputs.dtype) * valid.unsqueeze(-1))
        local_hidden = self.local_transformer(
            local_inputs,
            use_cache=False,
        ).last_hidden_state

        text_targets = flat_labels[:, 0]
        binary_targets = torch.full_like(text_targets, -100)
        binary_targets[text_targets.eq(self.config.audio_assistant_slot_token_id)] = 0
        binary_targets[text_targets.eq(self.config.audio_end_token_id)] = 1
        logits: list[Tensor] = [
            self.local_text_lm_head(local_hidden[:, 0]),
        ]
        target_channels: list[Tensor] = [binary_targets]
        for index, head in enumerate(self.audio_lm_heads):
            logits.append(head(local_hidden[:, index]))
            target_channels.append(audio_targets[:, index])
        weights = _loss_weights(
            self.config.channels,
            channelwise_loss_weight,
        ).to(device=hidden_states.device)
        losses: list[Tensor] = []
        active: list[Tensor] = []
        for index, (channel_logits, targets) in enumerate(zip(logits, target_channels)):
            if bool(targets.ne(-100).any()):
                loss = functional.cross_entropy(
                    channel_logits.float(),
                    targets,
                    ignore_index=-100,
                )
                active.append(weights[index])
            else:
                loss = channel_logits.float().sum() * 0.0
                active.append(weights[index] * 0.0)
            losses.append(loss)
        denominator = torch.stack(active).sum()
        if float(denominator.detach()) <= 0:
            raise ValueError("Local v1.5 received a batch with all labels ignored.")
        total = sum(weight * loss for weight, loss in zip(active, losses)) / denominator
        shaped_logits = tuple(item.view(batch_size, sequence_length, -1) for item in logits)
        return total, tuple(losses), shaped_logits

    def forward(
            self,
            input_ids: Tensor,
            *,
            attention_mask: Tensor | None = None,
            position_ids: Tensor | None = None,
            past_key_values: DynamicKVCache | None = None,
            labels: Tensor | None = None,
            use_cache: bool = False,
            channelwise_loss_weight: tuple[float, ...] | list[float] | None = (
                1.0,
                32.0,
            ),
    ) -> MossTTSOutput:
        global_output = self.transformer(
            inputs_embeds=self._input_embeddings(input_ids),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        if labels is None:
            return MossTTSOutput(
                loss=None,
                logits=(),
                last_hidden_state=global_output.last_hidden_state,
                past_key_values=global_output.past_key_values,
            )
        loss, losses, logits = self._local_objective(
            global_output.last_hidden_state,
            labels,
            channelwise_loss_weight=channelwise_loss_weight,
        )
        return MossTTSOutput(
            loss=loss,
            logits=logits,
            last_hidden_state=global_output.last_hidden_state,
            past_key_values=global_output.past_key_values,
            channel_losses=losses,
        )

    @torch.inference_mode()
    def generate(
        self,
        input_ids: Tensor,
        *,
        attention_mask: Tensor | None = None,
        max_new_frames: int = 4_096,
        text_temperature: float = 1.0,
        text_top_p: float = 1.0,
        text_top_k: int = 50,
        audio_temperature: float = 1.0,
        audio_top_p: float = 0.95,
        audio_top_k: int = 50,
        audio_repetition_penalty: float = 1.0,
        use_kv_cache: bool = True,
        n_vq_for_inference: int | None = None,
    ) -> list[tuple[int, Tensor]]:
        if (n_vq_for_inference is not None and int(n_vq_for_inference) != self.config.n_vq):
            raise ValueError(f"Local v1.5 is fixed at {self.config.n_vq} RVQ channels.")
        _validate_multichannel_ids(input_ids, channels=self.config.channels)
        if max_new_frames <= 0:
            raise ValueError("`max_new_frames` must be positive.")
        batch_size, prompt_length, _ = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones(
                batch_size,
                prompt_length,
                dtype=torch.bool,
                device=input_ids.device,
            )
        generated = input_ids.clone()
        current = input_ids
        mask = attention_mask.to(torch.bool)
        cache = None
        finished = torch.zeros(
            batch_size,
            dtype=torch.bool,
            device=input_ids.device,
        )
        generated_frames: list[Tensor] = []
        candidate_ids = torch.tensor(
            [
                self.config.audio_assistant_slot_token_id,
                self.config.audio_end_token_id,
            ],
            device=input_ids.device,
        )
        for _ in range(max_new_frames):
            global_output = self.transformer(
                inputs_embeds=self._input_embeddings(current),
                attention_mask=mask,
                past_key_values=cache,
                use_cache=use_kv_cache,
            )
            prefix = global_output.last_hidden_state[:, -1:].to(self.local_transformer.ln_f.weight.dtype)
            local_output = self.local_transformer(
                prefix,
                use_cache=True,
            )
            local_hidden = local_output.last_hidden_state[:, -1]
            text_index = sample_token(
                self.local_text_lm_head(local_hidden),
                temperature=text_temperature,
                top_k=text_top_k,
                top_p=text_top_p,
            )
            text_token = candidate_ids[text_index]
            continuing = (text_token.eq(self.config.audio_assistant_slot_token_id) & ~finished)
            finished |= text_token.eq(self.config.audio_end_token_id)
            if not bool(continuing.any()):
                break
            frame: list[Tensor] = []
            local_cache = local_output.past_key_values
            for index, head in enumerate(self.audio_lm_heads):
                history = (None if not generated_frames else torch.stack(generated_frames, dim=1)[..., index])
                token = sample_token(
                    head(local_hidden),
                    temperature=audio_temperature,
                    top_k=audio_top_k,
                    top_p=audio_top_p,
                    repetition_penalty=audio_repetition_penalty,
                    previous_token_ids=history,
                )
                frame.append(token)
                if index + 1 < self.config.n_vq:
                    local_output = self.local_transformer(
                        self.audio_embeddings[index](token).unsqueeze(1),
                        past_key_values=local_cache,
                        use_cache=True,
                    )
                    local_cache = local_output.past_key_values
                    local_hidden = local_output.last_hidden_state[:, -1]
            frame_tensor = torch.stack(frame, dim=-1)
            frame_tensor = frame_tensor.masked_fill(
                ~continuing.unsqueeze(-1),
                self.config.audio_pad_token_id,
            )
            generated_frames.append(frame_tensor)
            row = torch.full(
                (batch_size, 1, self.config.channels),
                self.config.audio_pad_token_id,
                dtype=torch.long,
                device=input_ids.device,
            )
            row[..., 0] = self.config.audio_assistant_slot_token_id
            row[..., 1:] = frame_tensor.unsqueeze(1)
            row[~continuing, 0, 0] = self.config.pad_token_id
            generated = torch.cat([generated, row], dim=1)
            current = row if use_kv_cache else generated
            cache = global_output.past_key_values if use_kv_cache else None
            mask = torch.cat([mask, continuing.unsqueeze(1)], dim=1)
        starts = _find_last_equal(
            input_ids[..., 0],
            self.config.audio_start_token_id,
        )
        return [(
            prompt_length - int(start) - 1,
            sequence[int(start):],
        ) for start, sequence in zip(starts.tolist(), generated)]


class MossRealtimeLocalForCausalLM(nn.Module):
    """Realtime RVQ depth decoder under the published checkpoint namespace."""

    def __init__(
        self,
        config: MossTTSConfig,
        *,
        initialize: bool,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if not hasattr(config.local_config, "num_hidden_layers"):
            raise TypeError("Realtime config requires its local Qwen graph.")
        self.model = MossQwenDepthModel(
            config.local_config,
            audio_codebooks=config.n_vq - 1,
            audio_vocab_size=config.audio_vocab_size,
            audio_pad_token_id=config.audio_pad_token_id,
            initialize=initialize,
            device=device,
            dtype=dtype,
        )
        self.local_lm_heads = nn.ModuleList([
            nn.Linear(
                config.local_config.hidden_size,
                config.audio_vocab_size,
                bias=False,
                device=device,
                dtype=dtype,
            ) for _ in range(config.n_vq)
        ])
        if initialize:
            for head in self.local_lm_heads:
                nn.init.normal_(
                    head.weight,
                    mean=0.0,
                    std=config.initializer_range,
                )

    def forward(
        self,
        input_ids: Tensor,
        *,
        backbone_last_hidden_state: Tensor,
        use_cache: bool = False,
    ) -> tuple[tuple[Tensor, ...], MossCache | None, Tensor]:
        output = self.model(
            input_ids=input_ids,
            first_hidden_state=backbone_last_hidden_state,
            use_cache=use_cache,
            apply_rope=True,
        )
        if output.last_hidden_state.shape[1] != len(self.local_lm_heads):
            raise ValueError("Realtime local sequence must contain one position per RVQ head.")
        logits = tuple(
            head(output.last_hidden_state[:, index]) for index, head in enumerate(self.local_lm_heads))
        return logits, output.past_key_values, output.last_hidden_state


class MossRealtimeModel(nn.Module):
    """Native training graph for MOSS-TTS-Realtime.

    The official high-level streamer couples an LLM transport, mutable
    queues, and chunked codec state.  That orchestration is
    intentionally not inferred here; :meth:`generate_audio_frame`
    exposes the audited model-level depth step, while :meth:`generate`
    fails closed.
    """

    def __init__(
        self,
        config: MossTTSConfig,
        *,
        initialize: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if config.variant != "realtime":
            raise ValueError("MossRealtimeModel requires `variant='realtime'`.")
        self.config = config
        self.embed_tokens = nn.ModuleList([
            nn.Embedding(
                config.language_config.vocab_size,
                config.language_config.hidden_size,
                config.pad_token_id,
                device=device,
                dtype=dtype,
            ),
            *[
                nn.Embedding(
                    config.audio_vocab_size,
                    config.language_config.hidden_size,
                    config.audio_pad_token_id,
                    device=device,
                    dtype=dtype,
                ) for _ in range(config.n_vq)
            ],
        ])
        self.language_model = CausalLMModel(
            config.language_config,
            initialize=initialize,
            device=device,
            dtype=dtype,
        )
        self.local_transformer = MossRealtimeLocalForCausalLM(
            config,
            initialize=initialize,
            device=device,
            dtype=dtype,
        )
        if initialize:
            for embedding in self.embed_tokens:
                nn.init.normal_(
                    embedding.weight,
                    mean=0.0,
                    std=config.initializer_range,
                )

    def optimization_compile_targets(
        self,
        mode: str,
    ) -> tuple[OptimizationCompileTarget, ...]:
        """Expose the uniform MOSS execution boundary for this variant."""
        if mode == "training":
            attribute = "forward"
        elif mode == "inference":
            attribute = "generate"
        else:
            raise ValueError(f"Unsupported optimization mode {mode!r}.")
        return (OptimizationCompileTarget(
            f"mosstts.{attribute}",
            self,
            attribute,
        ), )

    def _input_embeddings(self, input_ids: Tensor) -> Tensor:
        _validate_multichannel_ids(input_ids, channels=self.config.channels)
        hidden = self.embed_tokens[0](input_ids[..., 0])
        for index in range(self.config.n_vq):
            hidden = hidden + self.embed_tokens[index + 1](input_ids[..., index + 1])
        return hidden

    def forward(
        self,
        input_ids: Tensor,
        *,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        past_key_values: DynamicKVCache | None = None,
        labels: Tensor | None = None,
        use_cache: bool = False,
        channelwise_loss_weight: tuple[float, ...] | list[float] | None = None,
    ) -> MossTTSOutput:
        global_output = self.language_model(
            inputs_embeds=self._input_embeddings(input_ids),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        if labels is None:
            return MossTTSOutput(
                loss=None,
                logits=(),
                last_hidden_state=global_output.last_hidden_state,
                past_key_values=global_output.past_key_values,
            )
        _validate_multichannel_ids(
            labels,
            channels=self.config.channels,
            name="labels",
        )
        audio_labels = labels[..., 1:]
        train_mask = ~audio_labels.eq(-100).all(dim=-1)
        if not bool(train_mask.any()):
            raise ValueError("Realtime training received no assistant audio labels.")
        local_ids = audio_labels[train_mask][..., :self.config.n_vq - 1].clone()
        local_ids[local_ids.eq(-100)] = self.config.audio_pad_token_id
        local_ids = functional.pad(local_ids, (1, 0), value=0)
        batch_indices, time_indices = train_mask.nonzero(as_tuple=True)
        hidden_positions = (time_indices - 1).clamp_min(0)
        backbone_hidden = global_output.last_hidden_state[
            batch_indices,
            hidden_positions,
        ].unsqueeze(1)
        logits, local_cache, local_hidden = self.local_transformer(
            local_ids,
            backbone_last_hidden_state=backbone_hidden,
            use_cache=False,
        )
        local_targets = audio_labels[train_mask]
        loss, losses = _channelwise_cross_entropy(
            tuple(item.unsqueeze(1) for item in logits),
            local_targets.unsqueeze(1),
            weights=channelwise_loss_weight,
        )
        return MossTTSOutput(
            loss=loss,
            logits=logits,
            last_hidden_state=global_output.last_hidden_state,
            past_key_values=local_cache,
            channel_losses=losses,
        )

    @torch.inference_mode()
    def generate_audio_frame(
        self,
        backbone_hidden_state: Tensor,
        *,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0,
        repetition_window: int | None = None,
        previous_audio_tokens: Tensor | None = None,
    ) -> Tensor:
        """Decode one 16-codebook frame from an audited backbone state."""
        if backbone_hidden_state.ndim == 2:
            backbone_hidden_state = backbone_hidden_state.unsqueeze(1)
        if (backbone_hidden_state.ndim != 3 or backbone_hidden_state.shape[1] != 1 or
                backbone_hidden_state.shape[2] != self.config.language_config.hidden_size):
            raise ValueError("Realtime frame decoding requires [batch, 1, hidden] input.")
        batch_size = backbone_hidden_state.shape[0]
        tokens: list[Tensor] = []
        local_inputs = torch.zeros(
            batch_size,
            self.config.n_vq,
            dtype=torch.long,
            device=backbone_hidden_state.device,
        )
        # Recompute the tiny depth graph after each sampled codebook.  This is
        # deterministic with the official full-depth forward and avoids
        # inventing mutable streaming cache behavior.
        for index in range(self.config.n_vq):
            logits, _, _ = self.local_transformer(
                local_inputs,
                backbone_last_hidden_state=backbone_hidden_state,
                use_cache=False,
            )
            previous = None
            if previous_audio_tokens is not None:
                if (previous_audio_tokens.ndim != 3 or previous_audio_tokens.shape[0] != batch_size or
                        previous_audio_tokens.shape[2] != self.config.n_vq):
                    raise ValueError("Realtime audio history must have shape "
                                     "[batch, frames, n_vq].")
                previous = previous_audio_tokens[..., index]
                if repetition_window is not None:
                    if (isinstance(repetition_window, bool) or not isinstance(repetition_window, int) or
                            repetition_window <= 0):
                        raise ValueError("`repetition_window` must be a positive integer "
                                         "or None.")
                    previous = previous[:, -repetition_window:]
            token = sample_token(
                logits[index],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                previous_token_ids=previous,
            )
            tokens.append(token)
            if index + 1 < self.config.n_vq:
                local_inputs[:, index + 1] = token
        return torch.stack(tokens, dim=-1)

    @torch.inference_mode()
    def generate(
        self,
        input_ids: Tensor,
        *,
        attention_mask: Tensor,
        text_ids: Tensor,
        text_cursor: int,
        max_new_tokens: int = 1_000,
        temperature: float = 0.8,
        top_p: float = 0.6,
        top_k: int = 30,
        repetition_penalty: float = 1.1,
        repetition_window: int | None = 50,
    ) -> Tensor:
        """Run the official buffered Realtime autoregressive schedule."""
        _validate_multichannel_ids(input_ids, channels=self.config.channels)
        if input_ids.shape[0] != 1:
            raise ValueError("Native buffered Realtime generation currently requires "
                             "batch size one.")
        if attention_mask.shape != input_ids.shape[:2]:
            raise ValueError("Realtime attention mask has an invalid shape.")
        if text_ids.ndim != 1:
            raise ValueError("Realtime text IDs must be rank one.")
        if (isinstance(text_cursor, bool) or not isinstance(text_cursor, int) or
                not 0 <= text_cursor <= text_ids.numel()):
            raise ValueError("Realtime text cursor is outside the text IDs.")
        if (isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens <= 0):
            raise ValueError("`max_new_tokens` must be a positive integer.")
        if self.config.text_pad_token_id is None:
            raise ValueError("Realtime config has no text-padding token.")

        output = self(
            input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
        past_key_values = output.past_key_values
        frame = self.generate_audio_frame(
            output.last_hidden_state[:, -1:],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            repetition_window=repetition_window,
        )
        frames = [frame]
        finished = frame[:, 0].eq(1_026)

        for _ in range(max_new_tokens - 1):
            if bool(finished.all()):
                break
            text_token = (
                text_ids[text_cursor] if text_cursor < text_ids.numel() else text_ids.new_tensor(
                    self.config.text_pad_token_id))
            text_cursor += 1
            step = torch.full(
                (1, 1, self.config.channels),
                self.config.audio_pad_token_id,
                dtype=torch.long,
                device=input_ids.device,
            )
            step[:, 0, 0] = text_token
            step[:, 0, 1:] = frame
            attention_mask = torch.cat([
                attention_mask,
                (~finished).unsqueeze(1),
            ], dim=1)
            output = self(
                step,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = output.past_key_values
            history = torch.stack(frames, dim=1)
            frame = self.generate_audio_frame(
                output.last_hidden_state[:, -1:],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                repetition_window=repetition_window,
                previous_audio_tokens=history,
            )
            frames.append(frame)
            finished |= frame[:, 0].eq(1_026)
        return torch.stack(frames, dim=1)


MossTTSModel: TypeAlias = (MossDelayModel | MossOldLocalModel | MossLocalV15Model | MossRealtimeModel)


def build_mosstts_model(
    config: MossTTSConfig,
    *,
    initialize: bool = True,
    device=None,
    dtype=None,
) -> MossTTSModel:
    """Build exactly the graph declared by a validated configuration."""
    constructors = {
        "delay": MossDelayModel,
        "local": MossOldLocalModel,
        "local_v1_5": MossLocalV15Model,
        "realtime": MossRealtimeModel,
    }
    return constructors[config.variant](
        config,
        initialize=initialize,
        device=device,
        dtype=dtype,
    )


__all__ = [
    "MossDelayModel",
    "MossLocalV15Model",
    "MossOldLocalModel",
    "MossRealtimeModel",
    "MossTTSModel",
    "MossTTSOutput",
    "build_mosstts_model",
]
