"""Native PyTorch graphs for VibeVoice ASR and TTS checkpoints."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional

from voicehub.models.causal_lm.native import CausalLMModel, Qwen2Config
from voicehub.models.vibevoice.native.asr_codec import VibeVoiceASRConvCache, VibeVoiceASRTokenizerEncoder
from voicehub.models.vibevoice.native.codec import (
    VibeVoiceAcousticTokenizer,
    VibeVoiceCodecCache,
    VibeVoiceSemanticTokenizer,
)
from voicehub.models.vibevoice.native.configuration import VibeVoiceASRConfig, VibeVoiceTTSConfig
from voicehub.models.vibevoice.native.diffusion import VibeVoiceDiffusionHead, VibeVoiceDPMSolver
from voicehub.neural.cache import DynamicKVCache
from voicehub.neural.normalization import RMSNorm
from voicehub.optimization.diffusion_sampling import DiffusionStepContext
from voicehub.optimization.protocols import OptimizationCompileTarget


@dataclass(frozen=True)
class VibeVoiceASROutput:
    logits: Tensor
    loss: Tensor | None = None
    past_key_values: DynamicKVCache | None = None
    audio_hidden_states: Tensor | None = None
    hidden_states: tuple[Tensor, ...] | None = None
    attentions: tuple[Tensor, ...] | None = None


@dataclass(frozen=True)
class VibeVoiceTTSOutput:
    logits: Tensor
    diffusion_loss: Tensor | None = None
    loss: Tensor | None = None
    speech_token_num: int = 0
    past_key_values: DynamicKVCache | None = None
    hidden_states: tuple[Tensor, ...] | None = None
    attentions: tuple[Tensor, ...] | None = None


@dataclass(frozen=True)
class VibeVoiceRealtimeOutput:
    last_hidden_state: Tensor
    logits: Tensor | None = None
    past_key_values: DynamicKVCache | None = None
    hidden_states: tuple[Tensor, ...] | None = None
    attentions: tuple[Tensor, ...] | None = None


class VibeVoiceSpeechConnector(nn.Module):
    """Two-layer speech projector used by both TTS releases."""

    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        *,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(
            input_dimension,
            output_dimension,
            device=device,
            dtype=dtype,
        )
        self.norm = RMSNorm(
            output_dimension,
            epsilon=1e-6,
            device=device,
            dtype=dtype,
        )
        self.fc2 = nn.Linear(
            output_dimension,
            output_dimension,
            device=device,
            dtype=dtype,
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.fc2(self.norm(self.fc1(features)))


class VibeVoiceASRMultiModalProjector(nn.Module):

    def __init__(
        self,
        config: VibeVoiceASRConfig,
        *,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        text_size = config.text_config.hidden_size
        acoustic_size = config.acoustic_tokenizer_encoder_config.hidden_size
        semantic_size = config.semantic_tokenizer_encoder_config.hidden_size
        self.acoustic_linear_1 = nn.Linear(
            acoustic_size,
            text_size,
            device=device,
            dtype=dtype,
        )
        self.acoustic_norm = RMSNorm(
            text_size,
            epsilon=1e-6,
            device=device,
            dtype=dtype,
        )
        self.acoustic_linear_2 = nn.Linear(
            text_size,
            text_size,
            device=device,
            dtype=dtype,
        )
        self.semantic_linear_1 = nn.Linear(
            semantic_size,
            text_size,
            device=device,
            dtype=dtype,
        )
        self.semantic_norm = RMSNorm(
            text_size,
            epsilon=1e-6,
            device=device,
            dtype=dtype,
        )
        self.semantic_linear_2 = nn.Linear(
            text_size,
            text_size,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        acoustic_latents: Tensor,
        semantic_latents: Tensor,
    ) -> Tensor:
        if acoustic_latents.shape[:2] != semantic_latents.shape[:2]:
            raise ValueError("ASR acoustic and semantic latent timelines disagree.")
        acoustic = self.acoustic_linear_2(self.acoustic_norm(self.acoustic_linear_1(acoustic_latents)))
        semantic = self.semantic_linear_2(self.semantic_norm(self.semantic_linear_1(semantic_latents)))
        return acoustic + semantic


class VibeVoiceASRModel(nn.Module):
    """Speech encoders, projector, and bare Qwen2 decoder."""

    def __init__(
        self,
        config: VibeVoiceASRConfig,
        *,
        initialize: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.config = config
        self.acoustic_tokenizer_encoder = VibeVoiceASRTokenizerEncoder(
            config.acoustic_tokenizer_encoder_config,
            initialize=initialize,
            device=device,
            dtype=dtype,
        )
        self.semantic_tokenizer_encoder = VibeVoiceASRTokenizerEncoder(
            config.semantic_tokenizer_encoder_config,
            initialize=initialize,
            device=device,
            dtype=dtype,
        )
        self.multi_modal_projector = VibeVoiceASRMultiModalProjector(
            config,
            device=device,
            dtype=dtype,
        )
        self.language_model = CausalLMModel(
            config.text_config,
            initialize=initialize,
            device=device,
            dtype=dtype,
        )
        if initialize:
            self.multi_modal_projector.apply(self._initialize_projector)

    def _initialize_projector(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.text_config.initializer_range,
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def get_audio_features(
        self,
        input_values: Tensor,
        *,
        padding_mask: Tensor | None = None,
        chunk_size: int | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        if not isinstance(input_values, Tensor):
            raise TypeError("ASR `input_values` must be a PyTorch tensor.")
        if input_values.ndim == 2:
            input_values = input_values.unsqueeze(1)
        if input_values.ndim != 3:
            raise ValueError("ASR input must have shape [batch, samples] or "
                             "[batch, channels, samples].")
        chunk_size = (self.config.acoustic_tokenizer_chunk_size if chunk_size is None else chunk_size)
        hop_length = self.config.acoustic_tokenizer_encoder_config.hop_length
        if (isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0 or
                chunk_size % hop_length):
            raise ValueError(f"ASR chunk size must be a positive multiple of {hop_length}.")
        acoustic_cache = VibeVoiceASRConvCache()
        semantic_cache = VibeVoiceASRConvCache()
        acoustic_chunks: list[Tensor] = []
        semantic_chunks: list[Tensor] = []
        # The published ASR recipe treats both speech encoders as fixed
        # feature extractors. Projector and LM gradients remain enabled.
        with torch.no_grad():
            for chunk in torch.split(input_values, chunk_size, dim=-1):
                acoustic_chunks.append(
                    self.acoustic_tokenizer_encoder(
                        chunk,
                        padding_cache=acoustic_cache,
                        use_cache=True,
                    ).latents)
                semantic_chunks.append(
                    self.semantic_tokenizer_encoder(
                        chunk,
                        padding_cache=semantic_cache,
                        use_cache=True,
                    ).latents)
            acoustic = torch.cat(acoustic_chunks, dim=1)
            semantic = torch.cat(semantic_chunks, dim=1)
            batch_noise = torch.randn(
                (acoustic.shape[0], ),
                generator=generator,
                device=acoustic.device,
                dtype=acoustic.dtype,
            )
            latent_noise = torch.randn(
                acoustic.shape,
                generator=generator,
                device=acoustic.device,
                dtype=acoustic.dtype,
            )
            acoustic = acoustic + (
                self.config.acoustic_tokenizer_encoder_config.vae_std * batch_noise[:, None, None] *
                latent_noise)
        projected = self.multi_modal_projector(acoustic, semantic)
        if padding_mask is None:
            flattened = projected.reshape(-1, projected.shape[-1])
        else:
            if (not isinstance(padding_mask, Tensor) or padding_mask.ndim != 2 or
                    padding_mask.shape[0] != input_values.shape[0] or
                    padding_mask.shape[1] != input_values.shape[-1]):
                raise ValueError("ASR padding mask must align with waveform samples.")
            token_counts = torch.ceil(padding_mask.sum(dim=-1) / hop_length).to(torch.long)
            valid = torch.arange(
                projected.shape[1],
                device=projected.device,
            )[None] < token_counts[:, None].to(projected.device)
            flattened = projected[valid]
        return acoustic, flattened

    def forward(
        self,
        input_ids: Tensor,
        *,
        attention_mask: Tensor | None = None,
        input_values: Tensor | None = None,
        padding_mask: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
        past_key_values: DynamicKVCache | None = None,
        use_cache: bool | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        generator: torch.Generator | None = None,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.language_model.embed_tokens(input_ids)
        if input_values is not None:
            _, audio_features = self.get_audio_features(
                input_values,
                padding_mask=padding_mask,
                generator=generator,
            )
            mask = input_ids.eq(self.config.audio_token_id)
            placeholders = int(mask.sum().item())
            if placeholders != audio_features.shape[0]:
                raise ValueError(
                    "ASR prompt has "
                    f"{placeholders} audio placeholders for "
                    f"{audio_features.shape[0]} encoded frames.")
            inputs_embeds = inputs_embeds.clone()
            inputs_embeds[mask] = audio_features.to(
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
            )
        return self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )


class VibeVoiceASRForConditionalGeneration(nn.Module):
    """Exact ASR checkpoint graph with shifted causal-token objective."""

    def __init__(
        self,
        config: VibeVoiceASRConfig,
        *,
        initialize: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if not isinstance(config, VibeVoiceASRConfig):
            raise TypeError("VibeVoice ASR model requires a VibeVoiceASRConfig.")
        self.config = config
        self.model = VibeVoiceASRModel(
            config,
            initialize=initialize,
            device=device,
            dtype=dtype,
        )
        self.lm_head = nn.Linear(
            config.text_config.hidden_size,
            config.text_config.vocab_size,
            bias=False,
            device=device,
            dtype=dtype,
        )
        if initialize:
            nn.init.normal_(
                self.lm_head.weight,
                mean=0.0,
                std=config.text_config.initializer_range,
            )

    def gradient_checkpointing_enable(self) -> None:
        self.model.language_model.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self) -> None:
        self.model.language_model.gradient_checkpointing_disable()

    def forward(
        self,
        input_ids: Tensor,
        *,
        attention_mask: Tensor | None = None,
        input_values: Tensor | None = None,
        padding_mask: Tensor | None = None,
        labels: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
        past_key_values: DynamicKVCache | None = None,
        use_cache: bool | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        logits_to_keep: int = 0,
        generator: torch.Generator | None = None,
    ) -> VibeVoiceASROutput:
        if (isinstance(logits_to_keep, bool) or not isinstance(logits_to_keep, int) or logits_to_keep < 0):
            raise ValueError("`logits_to_keep` must be a non-negative integer.")
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            input_values=input_values,
            padding_mask=padding_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            generator=generator,
        )
        hidden_states = outputs.last_hidden_state
        selected = (hidden_states if logits_to_keep == 0 else hidden_states[:, -logits_to_keep:])
        logits = self.lm_head(selected)
        loss = None
        if labels is not None:
            if logits_to_keep:
                raise ValueError("Training loss requires logits for the complete sequence.")
            if labels.shape != input_ids.shape:
                raise ValueError("ASR labels must match input token IDs.")
            loss = functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return VibeVoiceASROutput(
            logits=logits,
            loss=loss,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        *,
        attention_mask: Tensor,
        input_values: Tensor,
        padding_mask: Tensor,
        max_new_tokens: int = 32_768,
        eos_token_id: int = 151_643,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        if (isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens <= 0):
            raise ValueError("`max_new_tokens` must be a positive integer.")
        sequences = input_ids
        cache: DynamicKVCache | None = None
        current_ids = input_ids
        current_audio: Tensor | None = input_values
        current_padding: Tensor | None = padding_mask
        mask = attention_mask
        for _ in range(max_new_tokens):
            output = self(
                current_ids,
                attention_mask=mask,
                input_values=current_audio,
                padding_mask=current_padding,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
                generator=generator,
            )
            cache = output.past_key_values
            token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
            sequences = torch.cat((sequences, token), dim=-1)
            if bool(token.eq(eos_token_id).all()):
                break
            current_ids = token
            current_audio = None
            current_padding = None
            mask = torch.cat(
                (mask, mask.new_ones(mask.shape[0], 1)),
                dim=-1,
            )
        return sequences


class VibeVoiceTTSModel(nn.Module):
    """Non-streaming 1.5B checkpoint body."""

    def __init__(
        self,
        config: VibeVoiceTTSConfig,
        *,
        initialize: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if config.is_streaming:
            raise ValueError("Use VibeVoiceRealtimeModel for streaming configs.")
        assert config.semantic_tokenizer_config is not None
        self.config = config
        self.register_buffer(
            "speech_scaling_factor",
            torch.tensor(float("nan"), device=device, dtype=dtype or torch.float32),
        )
        self.register_buffer(
            "speech_bias_factor",
            torch.tensor(float("nan"), device=device, dtype=dtype or torch.float32),
        )
        self.language_model = CausalLMModel(
            config.decoder_config,
            initialize=initialize,
            device=device,
            dtype=dtype,
        )
        self.acoustic_tokenizer = VibeVoiceAcousticTokenizer(
            config.acoustic_tokenizer_config,
            initialize=initialize,
            device=device,
            dtype=dtype,
        )
        self.semantic_tokenizer = VibeVoiceSemanticTokenizer(
            config.semantic_tokenizer_config,
            initialize=initialize,
            device=device,
            dtype=dtype,
        )
        self.acoustic_connector = VibeVoiceSpeechConnector(
            config.acoustic_vae_dim,
            config.decoder_config.hidden_size,
            device=device,
            dtype=dtype,
        )
        self.semantic_connector = VibeVoiceSpeechConnector(
            int(config.semantic_vae_dim),
            config.decoder_config.hidden_size,
            device=device,
            dtype=dtype,
        )
        self.prediction_head = VibeVoiceDiffusionHead(
            config.diffusion_head_config,
            initialize=initialize,
            device=device,
            dtype=dtype,
        )
        self.noise_scheduler = VibeVoiceDPMSolver(config.diffusion_head_config)
        if initialize:
            self.acoustic_connector.apply(self._initialize_module)
            self.semantic_connector.apply(self._initialize_module)

    def _initialize_module(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.decoder_config.initializer_range,
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)


class VibeVoiceForConditionalGeneration(nn.Module):
    """Published 1.5B training graph and portable Safetensors namespace."""

    def __init__(
        self,
        config: VibeVoiceTTSConfig,
        *,
        initialize: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if not isinstance(config, VibeVoiceTTSConfig) or config.is_streaming:
            raise TypeError("VibeVoice 1.5B model requires a non-streaming config.")
        self.config = config
        self.model = VibeVoiceTTSModel(
            config,
            initialize=initialize,
            device=device,
            dtype=dtype,
        )

    def optimization_compile_targets(
        self,
        mode: str,
    ) -> tuple[OptimizationCompileTarget, ...]:
        """Expose training while rejecting unsupported public inference."""
        if mode == "inference":
            return ()
        if mode != "training":
            raise ValueError(f"Unsupported optimization mode {mode!r}.")
        return (OptimizationCompileTarget(
            "vibevoice.forward",
            self,
            "forward",
        ), )

    @property
    def output_embedding(self) -> Tensor:
        # The published checkpoint ties output projection to input embedding
        # and therefore stores no duplicate ``lm_head`` tensor.
        return self.model.language_model.embed_tokens.weight

    def gradient_checkpointing_enable(self) -> None:
        self.model.language_model.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self) -> None:
        self.model.language_model.gradient_checkpointing_disable()

    def _speech_features(
        self,
        speech_tensors: Tensor,
        speech_masks: Tensor,
        *,
        speech_type: str,
        generator: torch.Generator | None,
    ) -> tuple[Tensor, Tensor]:
        with torch.no_grad():
            if speech_type == "audio":
                encoded = self.model.acoustic_tokenizer.encode(speech_tensors.unsqueeze(1))
                acoustic, _ = encoded.sample(
                    self.model.acoustic_tokenizer.std_dist_type,
                    generator=generator,
                )
            elif speech_type == "vae":
                acoustic = speech_tensors.reshape(
                    speech_tensors.shape[0],
                    -1,
                    self.config.acoustic_vae_dim,
                )
                deviation = torch.randn(
                    (acoustic.shape[0], ),
                    generator=generator,
                    device=acoustic.device,
                    dtype=acoustic.dtype,
                ) * (self.config.acoustic_tokenizer_config.fix_std / 0.8)
                noise = torch.randn(
                    acoustic.shape,
                    generator=generator,
                    device=acoustic.device,
                    dtype=acoustic.dtype,
                )
                acoustic = acoustic + deviation[:, None, None] * noise
            else:
                raise ValueError("VibeVoice speech type must be 'audio' or 'vae'.")
            if torch.isnan(self.model.speech_scaling_factor) or torch.isnan(self.model.speech_bias_factor):
                selected = acoustic[speech_masks]
                if selected.numel() < 2:
                    raise ValueError(
                        "Speech scaling initialization requires at least two "
                        "valid latent values.")
                self.model.speech_scaling_factor.copy_(selected.flatten().std().reciprocal())
                self.model.speech_bias_factor.copy_(-selected.flatten().mean())
            normalized = (acoustic + self.model.speech_bias_factor) * self.model.speech_scaling_factor
        return normalized, self.model.acoustic_connector(normalized)

    def forward(
        self,
        input_ids: Tensor,
        *,
        attention_mask: Tensor,
        speech_tensors: Tensor | None = None,
        speech_masks: Tensor | None = None,
        speeches_loss_input: Tensor | None = None,
        speech_semantic_tensors: Tensor | None = None,
        acoustic_input_mask: Tensor | None = None,
        acoustic_loss_mask: Tensor | None = None,
        labels: Tensor | None = None,
        speech_type: str = "audio",
        ddpm_batch_mul: int = 1,
        use_cache: bool = False,
        past_key_values: DynamicKVCache | None = None,
        generator: torch.Generator | None = None,
    ) -> VibeVoiceTTSOutput:
        if (isinstance(ddpm_batch_mul, bool) or not isinstance(ddpm_batch_mul, int) or ddpm_batch_mul <= 0):
            raise ValueError("`ddpm_batch_mul` must be a positive integer.")
        hidden_input = self.model.language_model.embed_tokens(input_ids)
        speech_features: Tensor | None = None
        speech_token_count = 0
        if speech_tensors is not None:
            if (speech_masks is None or acoustic_input_mask is None or speech_masks.ndim != 2 or
                    acoustic_input_mask.shape != input_ids.shape):
                raise ValueError("VibeVoice speech tensors require aligned masks.")
            all_features, all_connected = self._speech_features(
                speech_tensors.to(hidden_input.dtype),
                speech_masks.bool(),
                speech_type=speech_type,
                generator=generator,
            )
            if speech_semantic_tensors is not None:
                semantic = self.model.semantic_connector(speech_semantic_tensors.to(hidden_input.dtype))
                all_connected = all_connected + semantic
            if int(acoustic_input_mask.sum().item()) != int(speech_masks.sum().item()):
                raise ValueError("Acoustic placeholders and valid speech latents disagree.")
            hidden_input = hidden_input.clone()
            hidden_input[acoustic_input_mask.bool()] = all_connected[speech_masks.bool()]
            if speeches_loss_input is not None:
                target_mask = speeches_loss_input.bool() & speech_masks.bool()
                speech_features = all_features[target_mask]
                speech_token_count = int(speech_features.shape[0])
        decoder = self.model.language_model(
            inputs_embeds=hidden_input,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        logits = functional.linear(
            decoder.last_hidden_state,
            self.output_embedding,
        )
        causal_loss = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("VibeVoice labels must match token IDs.")
            causal_loss = functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        diffusion_loss: Tensor | None = None
        if (speech_tensors is not None and acoustic_loss_mask is not None and bool(acoustic_loss_mask.any())):
            if speech_features is None:
                raise ValueError("Diffusion targets require `speeches_loss_input`.")
            condition_mask = torch.zeros_like(
                acoustic_loss_mask,
                dtype=torch.bool,
            )
            condition_mask[:, :-1] = acoustic_loss_mask[:, 1:].bool()
            condition = decoder.last_hidden_state[condition_mask]
            if condition.shape[0] != speech_features.shape[0]:
                raise ValueError("VibeVoice causal conditions and diffusion targets disagree.")
            repeated_targets = speech_features.repeat_interleave(
                ddpm_batch_mul,
                dim=0,
            )
            repeated_condition = condition.repeat_interleave(
                ddpm_batch_mul,
                dim=0,
            )
            noise = torch.randn(
                repeated_targets.shape,
                generator=generator,
                device=repeated_targets.device,
                dtype=repeated_targets.dtype,
            )
            timesteps = torch.randint(
                self.config.diffusion_head_config.ddpm_num_steps,
                (repeated_targets.shape[0], ),
                generator=generator,
                device=repeated_targets.device,
            )
            noisy = self.model.noise_scheduler.add_noise(
                repeated_targets,
                noise,
                timesteps,
            )
            prediction = self.model.prediction_head(
                noisy,
                timesteps.to(noisy.dtype),
                repeated_condition,
            )
            target = self.model.noise_scheduler.get_velocity(
                repeated_targets,
                noise,
                timesteps,
            )
            diffusion_loss = functional.mse_loss(
                prediction.float(),
                target.float(),
                reduction="mean",
            )
        elif self.training:
            # Keep distributed reducer participation deterministic for a
            # text-only batch without changing the objective.
            diffusion_loss = (
                sum(parameter.sum() for parameter in self.model.prediction_head.parameters()) * 0.0)
            diffusion_loss = (
                diffusion_loss +
                sum(parameter.sum() for parameter in self.model.acoustic_connector.parameters()) * 0.0)
            diffusion_loss = (
                diffusion_loss +
                sum(parameter.sum() for parameter in self.model.semantic_connector.parameters()) * 0.0)
        combined_loss = None
        if causal_loss is not None and diffusion_loss is not None:
            combined_loss = causal_loss + diffusion_loss
        elif causal_loss is not None:
            combined_loss = causal_loss
        elif diffusion_loss is not None:
            combined_loss = diffusion_loss
        return VibeVoiceTTSOutput(
            logits=logits,
            diffusion_loss=diffusion_loss,
            loss=combined_loss,
            speech_token_num=speech_token_count,
            past_key_values=decoder.past_key_values,
            hidden_states=decoder.hidden_states,
            attentions=decoder.attentions,
        )


class VibeVoiceBinaryClassifier(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        *,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(
            hidden_size,
            hidden_size,
            device=device,
            dtype=dtype,
        )
        self.fc2 = nn.Linear(
            hidden_size,
            1,
            device=device,
            dtype=dtype,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.fc2(functional.relu(self.fc1(hidden_states)))


class VibeVoiceRealtimeModel(nn.Module):
    """Split lower/upper decoder body from the realtime 0.5B release."""

    def __init__(
        self,
        config: VibeVoiceTTSConfig,
        *,
        initialize: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if not config.is_streaming:
            raise ValueError("Realtime graph requires `vibevoice_streaming`.")
        assert config.tts_backbone_num_hidden_layers is not None
        self.config = config
        lower_values = config.decoder_config.to_dict()
        lower_values["num_hidden_layers"] = (
            config.decoder_config.num_hidden_layers - config.tts_backbone_num_hidden_layers)
        lower_values.pop('architectures', None)
        lower_config = Qwen2Config.from_dict(lower_values)
        upper_values = config.decoder_config.to_dict()
        upper_values["num_hidden_layers"] = config.tts_backbone_num_hidden_layers
        upper_values.pop('architectures', None)
        upper_config = Qwen2Config.from_dict(upper_values)
        self.language_model = CausalLMModel(
            lower_config,
            initialize=initialize,
            device=device,
            dtype=dtype,
        )
        # The official checkpoint omits this deliberately unused lower norm.
        self.language_model.norm = nn.Identity()
        self.tts_language_model = CausalLMModel(
            upper_config,
            initialize=initialize,
            device=device,
            dtype=dtype,
        )
        self.tts_input_types = nn.Embedding(
            2,
            config.decoder_config.hidden_size,
            device=device,
            dtype=dtype,
        )
        self.acoustic_tokenizer = VibeVoiceAcousticTokenizer(
            config.acoustic_tokenizer_config,
            decoder_only=True,
            initialize=initialize,
            device=device,
            dtype=dtype,
        )
        self.acoustic_connector = VibeVoiceSpeechConnector(
            config.acoustic_vae_dim,
            config.decoder_config.hidden_size,
            device=device,
            dtype=dtype,
        )
        self.register_buffer(
            "speech_scaling_factor",
            torch.tensor(float("nan"), device=device, dtype=dtype or torch.float32),
        )
        self.register_buffer(
            "speech_bias_factor",
            torch.tensor(float("nan"), device=device, dtype=dtype or torch.float32),
        )
        self.prediction_head = VibeVoiceDiffusionHead(
            config.diffusion_head_config,
            initialize=initialize,
            device=device,
            dtype=dtype,
        )
        self.noise_scheduler = VibeVoiceDPMSolver(config.diffusion_head_config)
        if initialize:
            self.acoustic_connector.apply(self._initialize_module)
            nn.init.normal_(
                self.tts_input_types.weight,
                mean=0.0,
                std=config.decoder_config.initializer_range,
            )

    def _initialize_module(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.decoder_config.initializer_range,
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def forward(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError(
            "Realtime VibeVoice has no unified forward. Use the explicit "
            "lower text and upper TTS stages.")


class VibeVoiceRealtimeForConditionalGeneration(nn.Module):
    """Published realtime graph with explicit, testable stages.

    The architecture exposes the exact low-level inference operations. A
    high-level streaming API is intentionally not advertised until chunk
    and cache parity are validated against the official runtime.
    """

    def __init__(
        self,
        config: VibeVoiceTTSConfig,
        *,
        initialize: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if not isinstance(config, VibeVoiceTTSConfig) or not config.is_streaming:
            raise TypeError("Realtime model requires a streaming VibeVoice config.")
        self.config = config
        self.model = VibeVoiceRealtimeModel(
            config,
            initialize=initialize,
            device=device,
            dtype=dtype,
        )
        self.tts_eos_classifier = VibeVoiceBinaryClassifier(
            config.decoder_config.hidden_size,
            device=device,
            dtype=dtype,
        )
        if initialize:
            self.tts_eos_classifier.apply(self.model._initialize_module)

    def optimization_compile_targets(
        self,
        mode: str,
    ) -> tuple[OptimizationCompileTarget, ...]:
        """Fail closed until the staged realtime API is publicly supported."""
        if mode not in {"inference", "training"}:
            raise ValueError(f"Unsupported optimization mode {mode!r}.")
        return ()

    def forward_lm(
        self,
        input_ids: Tensor | None = None,
        *,
        inputs_embeds: Tensor | None = None,
        attention_mask: Tensor | None = None,
        past_key_values: DynamicKVCache | None = None,
        use_cache: bool = True,
    ) -> VibeVoiceRealtimeOutput:
        output = self.model.language_model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        return VibeVoiceRealtimeOutput(
            last_hidden_state=output.last_hidden_state,
            past_key_values=output.past_key_values,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )

    def forward_tts_lm(
        self,
        input_ids: Tensor,
        *,
        lm_last_hidden_state: Tensor,
        tts_text_masks: Tensor,
        attention_mask: Tensor | None = None,
        past_key_values: DynamicKVCache | None = None,
        use_cache: bool = True,
    ) -> VibeVoiceRealtimeOutput:
        inputs_embeds = self.model.language_model.embed_tokens(input_ids)
        if (lm_last_hidden_state.ndim != 3 or lm_last_hidden_state.shape[0] != inputs_embeds.shape[0] or
                lm_last_hidden_state.shape[1] > inputs_embeds.shape[1] or
                lm_last_hidden_state.shape[2] != inputs_embeds.shape[2]):
            raise ValueError("Lower-LM hidden states cannot be spliced into TTS input.")
        inputs_embeds = inputs_embeds.clone()
        inputs_embeds[:, -lm_last_hidden_state.shape[1]:] = lm_last_hidden_state
        type_embeddings = self.model.tts_input_types(tts_text_masks.long())
        inputs_embeds = inputs_embeds + type_embeddings
        output = self.model.tts_language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        logits = self.tts_eos_classifier(output.last_hidden_state[:, -1])
        return VibeVoiceRealtimeOutput(
            last_hidden_state=output.last_hidden_state,
            logits=logits,
            past_key_values=output.past_key_values,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )

    @torch.no_grad()
    def sample_speech_latents(
        self,
        condition: Tensor,
        negative_condition: Tensor,
        *,
        guidance_scale: float = 3.0,
        inference_steps: int | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        if (isinstance(guidance_scale, bool) or not isinstance(guidance_scale, (int, float)) or
                not math.isfinite(guidance_scale) or guidance_scale <= 0):
            raise ValueError("Guidance scale must be finite and positive.")
        if condition.shape != negative_condition.shape or condition.ndim != 2:
            raise ValueError("Positive and negative TTS conditions must align.")
        prediction_head = self.model.prediction_head
        prediction_head.reset_diffusion_cache()
        prediction_head.reset_diffusion_sampling()
        combined_condition = torch.cat((condition, negative_condition), dim=0)
        with (
                prediction_head.diffusion_cache_session(),
                prediction_head.diffusion_sampling_session(),
        ):
            speech = torch.randn(
                (
                    combined_condition.shape[0],
                    self.config.acoustic_vae_dim,
                ),
                generator=generator,
                device=combined_condition.device,
                dtype=combined_condition.dtype,
            )
            native_schedule = (self.model.noise_scheduler.inference_timestep_schedule(inference_steps))
            controller = prediction_head.diffusion_sampling_controller
            prepared_schedule = (
                native_schedule if controller is None else controller.prepare_schedule(native_schedule))
            prepared_steps = prepared_schedule.numel() - 1
            # Rebuilding through the scheduler is essential: DPM-Solver++(2M)
            # history belongs to this exact sigma grid and this request.
            self.model.noise_scheduler.set_timesteps(
                prepared_steps,
                device=speech.device,
                timestep_schedule=prepared_schedule,
            )
            total_steps = self.model.noise_scheduler.timesteps.numel()
            for index, timestep in enumerate(self.model.noise_scheduler.timesteps):
                half = speech[:speech.shape[0] // 2]
                next_timestep = (
                    self.model.noise_scheduler.timesteps[index + 1] if index +
                    1 < total_steps else timestep.new_tensor(-1))
                guidance_context = DiffusionStepContext(
                    index=index,
                    total_steps=total_steps,
                    timestep=timestep,
                    next_timestep=next_timestep,
                    lane="guidance",
                    solver="dpm-solver++(2m)",
                )
                native_guidance = float(guidance_scale) != 1.0
                use_guidance = (
                    True if controller is None else controller.should_use_guidance(
                        guidance_context,
                        native=native_guidance,
                    ))
                evaluation_context = DiffusionStepContext(
                    index=index,
                    total_steps=total_steps,
                    timestep=timestep,
                    next_timestep=next_timestep,
                    lane="guided" if use_guidance else "conditional",
                    solver="dpm-solver++(2m)",
                )

                def evaluate_prediction() -> Tensor:
                    model_timestep = timestep.to(half.dtype)
                    if not use_guidance:
                        return prediction_head(
                            half,
                            model_timestep.expand(half.shape[0]),
                            condition,
                            diffusion_cache_lane="conditional",
                        )
                    duplicated = torch.cat((half, half), dim=0)
                    packed_prediction = prediction_head(
                        duplicated,
                        model_timestep.expand(duplicated.shape[0]),
                        combined_condition,
                        diffusion_cache_lane="packed-cfg",
                    )
                    positive, negative = packed_prediction.chunk(2, dim=0)
                    if controller is not None:
                        controller.observe_guidance(
                            guidance_context,
                            positive,
                            negative,
                        )
                    return (negative + float(guidance_scale) * (positive - negative))

                guided = (
                    evaluate_prediction() if controller is None else controller.evaluate(
                        evaluation_context,
                        half,
                        evaluate_prediction,
                    ))
                prediction = torch.cat((guided, guided), dim=0)
                # A model prediction may be reused, but no DPM++ update is
                # skipped: each prepared sigma transition consumes one value.
                speech = self.model.noise_scheduler.step(
                    prediction,
                    timestep,
                    speech,
                ).prev_sample
            return speech[:speech.shape[0] // 2]

    @torch.no_grad()
    def decode_speech_latents(
        self,
        latents: Tensor,
        *,
        cache: VibeVoiceCodecCache | None = None,
        sample_indices: Tensor | None = None,
        use_cache: bool = False,
    ) -> Tensor:
        normalized = (latents / self.model.speech_scaling_factor - self.model.speech_bias_factor)
        return self.model.acoustic_tokenizer.decode(
            normalized.unsqueeze(1) if normalized.ndim == 2 else normalized,
            cache=cache,
            sample_indices=sample_indices,
            use_cache=use_cache,
        )

    def forward(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError(
            "Realtime VibeVoice uses staged inference. Call `forward_lm`, "
            "`forward_tts_lm`, `sample_speech_latents`, and "
            "`decode_speech_latents` explicitly.")


__all__ = [
    "VibeVoiceASRForConditionalGeneration",
    "VibeVoiceASRModel",
    "VibeVoiceASROutput",
    "VibeVoiceForConditionalGeneration",
    "VibeVoiceRealtimeForConditionalGeneration",
    "VibeVoiceRealtimeModel",
    "VibeVoiceRealtimeOutput",
    "VibeVoiceSpeechConnector",
    "VibeVoiceTTSModel",
    "VibeVoiceTTSOutput",
]
