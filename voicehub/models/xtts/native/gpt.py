"""VoiceHub-owned XTTS v2 autoregressive model and source objectives."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from voicehub.models.xtts.native.conditioning import ConditioningEncoder, PerceiverResampler
from voicehub.models.xtts.native.transformer import GPT2Model


class LearnedPositionEmbeddings(nn.Module):

    def __init__(self, sequence_length: int, model_dim: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(sequence_length, model_dim)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.seq_len = sequence_length
        self.relative = False

    def forward(self, value: Tensor) -> Tensor:
        return self.emb(torch.arange(value.shape[1], device=value.device))

    def get_fixed_embedding(self, index: int, device: torch.device) -> Tensor:
        return self.emb(torch.tensor([index], device=device)).unsqueeze(0)


class XTTS2GPT(nn.Module):
    """Namespace-compatible replacement for Coqui's ``GPT`` module."""

    def __init__(
        self,
        *,
        start_text_token: int,
        stop_text_token: int,
        layers: int = 30,
        model_dim: int = 1_024,
        heads: int = 16,
        max_text_tokens: int = 402,
        max_mel_tokens: int = 605,
        max_prompt_tokens: int = 70,
        code_stride_len: int = 1_024,
        number_text_tokens: int = 6_681,
        num_audio_tokens: int = 1_026,
        start_audio_token: int = 1_024,
        stop_audio_token: int = 1_025,
        use_perceiver_resampler: bool = True,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.label_smoothing = label_smoothing
        self.number_text_tokens = number_text_tokens
        self.start_text_token = start_text_token
        self.stop_text_token = stop_text_token
        self.num_audio_tokens = num_audio_tokens
        self.start_audio_token = start_audio_token
        self.stop_audio_token = stop_audio_token
        self.start_prompt_token = start_audio_token
        self.stop_prompt_token = stop_audio_token
        self.layers = layers
        self.heads = heads
        self.model_dim = model_dim
        self.max_conditioning_inputs = 1
        self.max_gen_mel_tokens = max_mel_tokens - 3
        self.max_mel_tokens = max_mel_tokens + 3
        self.max_text_tokens = max_text_tokens + 2
        self.max_prompt_tokens = max_prompt_tokens
        self.code_stride_len = code_stride_len
        self.conditioning_encoder = ConditioningEncoder(
            80,
            model_dim,
            num_attn_heads=heads,
        )
        self.conditioning_dropout = nn.Dropout1d(0.1)
        self.average_conditioning_embeddings = False
        self.use_perceiver_resampler = use_perceiver_resampler
        self.perceiver_cond_length_compression = 256
        self.text_embedding = nn.Embedding(number_text_tokens, model_dim)
        self.mel_embedding = nn.Embedding(num_audio_tokens, model_dim)
        self.gpt = GPT2Model(layers, model_dim, heads)
        self.mel_pos_embedding = LearnedPositionEmbeddings(self.max_mel_tokens, model_dim)
        self.text_pos_embedding = LearnedPositionEmbeddings(self.max_text_tokens, model_dim)
        self.mel_layer_pos_embedding = None
        self.text_layer_pos_embedding = None
        self.mel_solo_embedding = 0
        self.text_solo_embedding = 0
        self.final_norm = nn.LayerNorm(model_dim)
        self.text_head = nn.Linear(model_dim, number_text_tokens)
        self.mel_head = nn.Linear(model_dim, num_audio_tokens)
        if use_perceiver_resampler:
            self.conditioning_perceiver = PerceiverResampler(dim=model_dim)
        else:
            self.prompt_embedding = nn.Embedding(num_audio_tokens, model_dim)
            self.prompt_pos_embedding = LearnedPositionEmbeddings(216, model_dim)

    def get_style_emb(self, cond_input: Tensor) -> Tensor:
        if cond_input.ndim == 4:
            cond_input = cond_input.squeeze(1)
        value = self.conditioning_encoder(cond_input)
        if self.use_perceiver_resampler:
            value = self.conditioning_perceiver(value.transpose(1, 2)).transpose(1, 2)
        return value

    @staticmethod
    def _inputs_and_targets(
        value: Tensor,
        start_token: int,
        stop_token: int,
    ) -> tuple[Tensor, Tensor]:
        return (
            F.pad(value, (1, 0), value=start_token),
            F.pad(value, (0, 1), value=stop_token),
        )

    def _logits(
        self,
        text_inputs: Tensor,
        audio_inputs: Tensor,
        prompt: Tensor,
        attention_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        text_emb = self.text_embedding(text_inputs) + self.text_pos_embedding(text_inputs)
        audio_emb = self.mel_embedding(audio_inputs) + self.mel_pos_embedding(audio_inputs)
        joined = torch.cat((prompt, text_emb, audio_emb), dim=1)
        encoded = self.gpt(
            joined,
            attention_mask=attention_mask,
        ).last_hidden_state[:, prompt.shape[1]:]
        encoded = self.final_norm(encoded)
        text_hidden = encoded[:, :text_emb.shape[1]]
        audio_hidden = encoded[:, -audio_emb.shape[1]:]
        return (
            self.text_head(text_hidden).transpose(1, 2),
            self.mel_head(audio_hidden).transpose(1, 2),
            audio_hidden,
        )

    def forward(
        self,
        text_inputs: Tensor,
        text_lengths: Tensor,
        audio_codes: Tensor,
        wav_lengths: Tensor,
        cond_mels: Tensor | None = None,
        cond_idxs: Tensor | None = None,
        cond_lens: Tensor | None = None,
        cond_latents: Tensor | None = None,
        return_attentions: bool = False,
        return_latent: bool = False,
    ):
        """Compute the published text CE and acoustic-token CE objectives."""
        if return_attentions:
            raise ValueError("Native XTTS v2 does not expose attention tensors.")
        text_count = int(text_lengths.max().item())
        code_lengths = torch.ceil(wav_lengths / self.code_stride_len).long() + 3
        audio_count = int(code_lengths.max().item())
        if audio_count > audio_codes.shape[-1]:
            audio_codes = F.pad(audio_codes, (0, audio_count - audio_codes.shape[-1]))
        text = F.pad(
            text_inputs[:, :text_count],
            (0, 1),
            value=self.stop_text_token,
        )
        audio = F.pad(
            audio_codes[:, :audio_count],
            (0, 1),
            value=self.stop_audio_token,
        )
        for index, length in enumerate(code_lengths - 3):
            audio[index, int(length.item()):] = self.stop_audio_token
        text, text_targets = self._inputs_and_targets(
            text,
            self.start_text_token,
            self.stop_text_token,
        )
        audio, audio_targets = self._inputs_and_targets(
            audio,
            self.start_audio_token,
            self.stop_audio_token,
        )
        if cond_latents is None:
            if cond_mels is None:
                raise ValueError("XTTS v2 forward requires `cond_mels` or `cond_latents`.")
            cond_latents = self.get_style_emb(cond_mels).transpose(1, 2)
        mask = None
        if not return_latent:
            conditioning_mask = torch.ones(
                (text.shape[0], cond_latents.shape[1]),
                dtype=torch.bool,
                device=text.device,
            )
            text_mask = torch.ones(
                text.shape,
                dtype=torch.bool,
                device=text.device,
            )
            audio_mask = torch.ones(
                audio.shape,
                dtype=torch.bool,
                device=text.device,
            )
            for index, length in enumerate(text_lengths):
                text_mask[index, int(length.item()) + 1:] = False
            for index, length in enumerate(code_lengths):
                audio_mask[index, int(length.item()) + 1:] = False
            mask = torch.cat(
                (conditioning_mask, text_mask, audio_mask),
                dim=1,
            )
        text_logits, audio_logits, audio_hidden = self._logits(
            text,
            audio,
            cond_latents,
            mask,
        )
        if return_latent:
            return audio_hidden[:, :-1 if self.training else -5]
        for index, length in enumerate(text_lengths):
            text_targets[index, int(length.item()) + 1:] = -1
        for index, length in enumerate(code_lengths):
            audio_targets[index, int(length.item()) + 1:] = -1
        if cond_idxs is not None:
            for index, interval in enumerate(cond_idxs):
                divisor = 256 if self.use_perceiver_resampler else self.code_stride_len
                start, end = (interval // divisor).tolist()
                audio_targets[index, start:end] = -1
        text_loss = F.cross_entropy(
            text_logits,
            text_targets.long(),
            ignore_index=-1,
            label_smoothing=self.label_smoothing,
        )
        audio_loss = F.cross_entropy(
            audio_logits,
            audio_targets.long(),
            ignore_index=-1,
            label_smoothing=self.label_smoothing,
        )
        return text_loss, audio_loss, audio_logits

    def autoregressive_step(
        self,
        prefix: Tensor,
        generated: Tensor,
    ) -> Tensor:
        audio_emb = self.mel_embedding(generated) + self.mel_pos_embedding(generated)
        encoded = self.gpt(torch.cat((prefix, audio_emb), dim=1)).last_hidden_state
        return self.mel_head(self.final_norm(encoded[:, -1]))

    @torch.inference_mode()
    def generate(
        self,
        cond_latents: Tensor,
        text_inputs: Tensor,
        *,
        max_new_tokens: int | None = None,
        temperature: float = 0.75,
        top_k: int = 50,
        top_p: float = 0.85,
        do_sample: bool = True,
        repetition_penalty: float = 1.0,
        **_: object,
    ) -> Tensor:
        invalid_max_new_tokens = (
            max_new_tokens is not None and
            (isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens <= 0))
        if invalid_max_new_tokens:
            raise ValueError("`max_new_tokens` must be a positive integer or None.")
        if (isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or
                not math.isfinite(temperature) or temperature <= 0):
            raise ValueError("`temperature` must be finite and greater than zero.")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
            raise ValueError("`top_k` must be a non-negative integer.")
        if (isinstance(top_p, bool) or not isinstance(top_p, (int, float)) or not math.isfinite(top_p) or
                not 0 < top_p <= 1):
            raise ValueError("`top_p` must be finite and in the interval (0, 1].")
        if not isinstance(do_sample, bool):
            raise TypeError("`do_sample` must be a boolean.")
        if (isinstance(repetition_penalty, bool) or not isinstance(repetition_penalty, (int, float)) or
                not math.isfinite(repetition_penalty) or repetition_penalty <= 0):
            raise ValueError("`repetition_penalty` must be finite and greater than zero.")
        text = F.pad(text_inputs, (1, 1), value=self.stop_text_token)
        text[:, 0] = self.start_text_token
        prefix = torch.cat((
            cond_latents,
            self.text_embedding(text) + self.text_pos_embedding(text),
        ), dim=1)
        generated = torch.full(
            (text.shape[0], 1),
            self.start_audio_token,
            dtype=torch.long,
            device=text.device,
        )
        unfinished = torch.ones(
            text.shape[0],
            dtype=torch.bool,
            device=text.device,
        )
        limit = min(
            self.max_gen_mel_tokens,
            self.max_gen_mel_tokens if max_new_tokens is None else max_new_tokens,
        )
        for _index in range(limit):
            logits = self.autoregressive_step(prefix, generated) / temperature
            if repetition_penalty != 1.0:
                repeated_scores = logits.gather(1, generated)
                repeated_scores = torch.where(
                    repeated_scores < 0,
                    repeated_scores * repetition_penalty,
                    repeated_scores / repetition_penalty,
                )
                logits.scatter_(1, generated, repeated_scores)
            logits = _filter_logits(logits, top_k=top_k, top_p=top_p)
            token = (
                torch.multinomial(F.softmax(logits, dim=-1), 1) if do_sample else logits.argmax(
                    dim=-1, keepdim=True))
            token = torch.where(
                unfinished[:, None],
                token,
                torch.full_like(token, self.stop_audio_token),
            )
            generated = torch.cat((generated, token), dim=1)
            unfinished &= token.squeeze(1) != self.stop_audio_token
            if not bool(unfinished.any()):
                break
        return generated[:, 1:]


def _filter_logits(logits: Tensor, *, top_k: int, top_p: float) -> Tensor:
    if 0 < top_k < logits.shape[-1]:
        threshold = logits.topk(top_k).values[:, -1:]
        logits = logits.masked_fill(logits < threshold, -torch.inf)
    if 0.0 < top_p < 1.0:
        sorted_logits, indices = logits.sort(descending=True)
        cumulative = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cumulative > top_p
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        mask = torch.zeros_like(remove).scatter(1, indices, remove)
        logits = logits.masked_fill(mask, -torch.inf)
    return logits


__all__ = ["LearnedPositionEmbeddings", "XTTS2GPT"]
