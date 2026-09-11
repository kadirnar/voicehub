import logging
from dataclasses import replace
from typing import List, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from voicehub.generation.logits import process_logits
from voicehub.models.causal_lm.native.configuration import LlamaConfig
from voicehub.models.causal_lm.native.modeling import CausalLMModel
from voicehub.models.chatterbox.models.t3.llama_configs import LLAMA_CONFIGS
from voicehub.models.chatterbox.models.t3.modules.cond_enc import T3Cond, T3CondEnc
from voicehub.models.chatterbox.models.t3.modules.learned_pos_emb import LearnedPositionEmbeddings
from voicehub.models.chatterbox.models.t3.modules.t3_config import T3Config
from voicehub.models.chatterbox.models.utils import AttrDict

logger = logging.getLogger(__name__)
IGNORE_INDEX = -100


def _ensure_BOT_EOT(text_tokens: Tensor, hp):
    """Validate that every sequence in the batch contains the required start
    and stop text tokens."""
    if not isinstance(text_tokens, Tensor) or text_tokens.ndim != 2:
        raise ValueError("Chatterbox text tokens must have shape [batch, time].")
    has_start = text_tokens.eq(hp.start_text_token).any(dim=1)
    has_stop = text_tokens.eq(hp.stop_text_token).any(dim=1)
    if not bool(has_start.all()):
        raise ValueError("Every Chatterbox text sequence requires a start text token.")
    if not bool(has_stop.all()):
        raise ValueError("Every Chatterbox text sequence requires a stop text token.")


class T3(nn.Module):
    """Token-To-Token (T3) model backed by VoiceHub's native Llama runtime.

    * tokenization, including start / stop tokens are always added externally to this class
    * conditioning data like CLAP, emotion, etc are all in a separate file for more modularity
    * careful! this class assumes relative positional encoding -- with absolute PE, we would at
        least want to reset the position to 0 when speech tokens begin, and optionally use a
        different PE embedding space for speech.
    """

    def __init__(self, hp=None):
        super().__init__()
        self.hp = hp or T3Config()
        hp = self.hp
        self.cfg = LlamaConfig.from_dict(LLAMA_CONFIGS[hp.llama_config_name])
        self.tfmr = CausalLMModel(self.cfg)
        self.dim = self.cfg.hidden_size
        self.deepspeed_patch_applied = False

        # conditioning / embedding
        self.cond_enc = T3CondEnc(hp)
        self.text_emb = nn.Embedding(hp.text_tokens_dict_size, self.dim)
        self.speech_emb = nn.Embedding(hp.speech_tokens_dict_size, self.dim)

        # custom position embedding
        if hp.input_pos_emb == "learned":
            max_text_seq_len = hp.max_text_tokens + 2
            self.text_pos_emb = LearnedPositionEmbeddings(max_text_seq_len, self.dim)

            max_mel_seq_len = hp.max_speech_tokens + 2 + 2
            self.speech_pos_emb = LearnedPositionEmbeddings(max_mel_seq_len, self.dim)

        # logit projection
        self.text_head = nn.Linear(self.cfg.hidden_size, hp.text_tokens_dict_size, bias=False)
        self.speech_head = nn.Linear(self.cfg.hidden_size, hp.speech_tokens_dict_size, bias=False)
        self.compiled = False

    @property
    def device(self):
        return self.speech_head.weight.device

    def get_input_embeddings(self) -> nn.Embedding:
        """Return T3's public text embedding rather than the dummy LM table."""
        return self.text_emb

    def gradient_checkpointing_enable(
        self,
        gradient_checkpointing_kwargs=None,
    ) -> None:
        """Enable native non-reentrant checkpointing in the Llama backbone."""
        options = dict(gradient_checkpointing_kwargs or {})
        if options.get("use_reentrant") not in (None, False):
            raise ValueError("Native T3 gradient checkpointing requires "
                             "use_reentrant=False.")
        unsupported = set(options) - {"use_reentrant"}
        if unsupported:
            raise ValueError(
                "Unsupported T3 gradient checkpointing options: " + ", ".join(sorted(unsupported)))
        self.tfmr.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self) -> None:
        self.tfmr.gradient_checkpointing_disable()

    def prepare_conditioning(self, t3_cond: T3Cond):
        """Token cond data needs to be embedded, so that needs to be here
        instead of in `T3CondEnc`."""
        if t3_cond.cond_prompt_speech_tokens is not None and t3_cond.cond_prompt_speech_emb is None:
            prompt_embedding = self.speech_emb(t3_cond.cond_prompt_speech_tokens)
            if self.hp.input_pos_emb == "learned":
                prompt_embedding = prompt_embedding + self.speech_pos_emb(t3_cond.cond_prompt_speech_tokens)
            t3_cond = replace(
                t3_cond,
                cond_prompt_speech_emb=prompt_embedding,
            )
        return self.cond_enc(t3_cond)  # (B, len_cond, dim)

    def prepare_input_embeds(
        self,
        *,
        t3_cond: T3Cond,
        text_tokens: torch.LongTensor,
        speech_tokens: torch.LongTensor,
        cfg_weight: float = 0.0,
    ):
        # prepare input embeddings (skip backbone tranformer embeddings)
        cond_emb = self.prepare_conditioning(t3_cond)  # (B, len_cond, dim)
        text_emb = self.text_emb(text_tokens)  # (B, len_text, dim)
        if cfg_weight > 0.0:
            if text_emb.shape[0] != 2:
                raise ValueError("Classifier-free guidance requires a batch of two.")
            text_emb = text_emb.clone()
            text_emb[1].zero_()  # CFG uncond

        speech_emb = self.speech_emb(speech_tokens)  # (B, len_speech, dim)
        if self.hp.input_pos_emb == "learned":
            text_emb = text_emb + self.text_pos_emb(text_tokens)
            speech_emb = speech_emb + self.speech_pos_emb(speech_tokens)
        len_cond = cond_emb.size(1)

        if cond_emb.size(0) != text_emb.size(0):
            cond_emb = cond_emb.expand(text_emb.size(0), -1, -1)

        # concat
        embeds = torch.stack([torch.cat((ce, te, se))
                              for ce, te, se in zip(cond_emb, text_emb, speech_emb)])  # (B, length, dim)
        return embeds, len_cond

    def forward(
        self,
        *,
        t3_cond: T3Cond,
        text_tokens: torch.LongTensor,
        text_token_lens: torch.LongTensor,
        speech_tokens: torch.LongTensor,
        speech_token_lens: torch.LongTensor,
        training=False,
    ):
        _ensure_BOT_EOT(text_tokens, self.hp)

        # prepare custom input embeds
        embeds, len_cond = self.prepare_input_embeds(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            speech_tokens=speech_tokens,
        )
        device = embeds.device
        text_positions = torch.arange(
            text_tokens.shape[1],
            device=device,
        ).unsqueeze(0)
        speech_positions = torch.arange(
            speech_tokens.shape[1],
            device=device,
        ).unsqueeze(0)
        attention_mask = torch.cat(
            (
                torch.ones(
                    text_tokens.shape[0],
                    len_cond,
                    dtype=torch.bool,
                    device=device,
                ),
                text_positions < text_token_lens.to(device=device).unsqueeze(1),
                speech_positions < speech_token_lens.to(device=device).unsqueeze(1),
            ),
            dim=1,
        )

        # backbone tranformer forward
        tfmr_out = self.tfmr.forward(
            input_ids=None,
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            output_hidden_states=False,
            use_cache=(not training),
        )
        hidden_states = tfmr_out.last_hidden_state

        # post-processing: splice out text and speech parts of hidden states
        len_text = text_tokens.size(1)
        len_speech = speech_tokens.size(1)
        B, _, dim = hidden_states.shape
        device, dtype = hidden_states.device, hidden_states.dtype
        text_latents = torch.zeros(B, len_text, dim, dtype=dtype, device=device)
        speech_latents = torch.zeros(B, len_speech, dim, dtype=dtype, device=device)
        ttl, stl = text_token_lens, speech_token_lens
        for i in range(B):
            text_end = len_cond + ttl[i].item()
            speech_start = len_cond + text_tokens.size(1)
            speech_end = speech_start + stl[i].item()
            text_latents[i, :ttl[i]] = hidden_states[i, len_cond:text_end]
            speech_latents[i, :stl[i]] = hidden_states[i, speech_start:speech_end]

        # logit projection
        text_logits = self.text_head(text_latents)
        speech_logits = self.speech_head(speech_latents)

        return AttrDict(
            text_logits=text_logits,
            text_latents=text_latents,
            speech_logits=speech_logits,
            speech_latents=speech_latents,
            hidden_states=hidden_states,
        )

    def loss(
        self,
        *,
        t3_cond: T3Cond,
        text_tokens: torch.LongTensor,
        text_token_lens: torch.LongTensor,
        speech_tokens: torch.LongTensor,
        speech_token_lens: torch.LongTensor,
        labels_text: Optional[torch.LongTensor] = None,
        labels_speech: Optional[torch.LongTensor] = None,
        prompt_lens: Optional[torch.LongTensor] = None,
    ):
        """Compute causal next-token losses for the released T3 objective.

        The original v0.1.2 helper passed ``[batch, time, vocabulary]``
        logits directly to :func:`cross_entropy` and supervised each token
        with itself.  Production fine-tuning implementations instead shift
        both streams by one position.  ``prompt_lens`` optionally excludes a
        leading in-utterance prompt from the speech objective, matching the
        published community data pipeline.

        Explicit labels may have the same width as their token stream (and
        are shifted here) or may already contain the ``time - 1`` causal
        targets.  Use ``-100`` to mask individual targets.
        """
        self._validate_training_stream(
            text_tokens,
            text_token_lens,
            name="text",
        )
        self._validate_training_stream(
            speech_tokens,
            speech_token_lens,
            name="speech",
        )

        out = self.forward(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            text_token_lens=text_token_lens,
            speech_tokens=speech_tokens,
            speech_token_lens=speech_token_lens,
            training=True,
        )  # (B, seq, vocab_size)

        loss_text = self._causal_cross_entropy(
            out.text_logits,
            text_tokens,
            text_token_lens,
            labels=labels_text,
            name="text",
        )
        loss_speech = self._causal_cross_entropy(
            out.speech_logits,
            speech_tokens,
            speech_token_lens,
            labels=labels_speech,
            prompt_lens=prompt_lens,
            name="speech",
        )

        return loss_text, loss_speech

    @staticmethod
    def _validate_training_stream(
        tokens: Tensor,
        lengths: Tensor,
        *,
        name: str,
    ) -> None:
        if tokens.ndim != 2:
            raise ValueError(f"Chatterbox T3 {name}_tokens must have shape [batch, time].")
        if lengths.ndim != 1 or lengths.shape[0] != tokens.shape[0]:
            raise ValueError(f"Chatterbox T3 {name}_token_lens must have shape [batch].")
        if torch.is_floating_point(lengths) and not torch.equal(
                lengths,
                lengths.round(),
        ):
            raise ValueError(f"Chatterbox T3 {name} lengths must be integers.")
        normalized = lengths.to(device=tokens.device, dtype=torch.long)
        if bool((normalized < 1).any()) or bool((normalized > tokens.shape[1]).any()):
            raise ValueError(f"Chatterbox T3 {name} lengths must be between 1 and "
                             f"{tokens.shape[1]}.")

    @staticmethod
    def _causal_cross_entropy(
        logits: Tensor,
        tokens: Tensor,
        lengths: Tensor,
        *,
        labels: Optional[Tensor] = None,
        prompt_lens: Optional[Tensor] = None,
        name: str,
    ) -> Tensor:
        if logits.ndim != 3 or logits.shape[:2] != tokens.shape:
            raise ValueError(
                f"Chatterbox T3 {name} logits must match the token batch "
                "and time dimensions.")
        if tokens.shape[1] < 2:
            raise ValueError(
                f"Chatterbox T3 {name} streams need at least two tokens for "
                "causal supervision.")

        causal_logits = logits[:, :-1, :]
        if labels is None:
            targets = tokens[:, 1:]
        else:
            labels = labels.to(device=logits.device, dtype=torch.long)
            if labels.shape == tokens.shape:
                targets = labels[:, 1:]
            elif labels.shape == tokens[:, 1:].shape:
                targets = labels
            else:
                raise ValueError(
                    f"Chatterbox T3 labels_{name} must have shape "
                    f"{tuple(tokens.shape)!r} or "
                    f"{tuple(tokens[:, 1:].shape)!r}.")

        targets = targets.to(device=logits.device, dtype=torch.long)
        normalized_lengths = lengths.to(
            device=logits.device,
            dtype=torch.long,
        )
        positions = torch.arange(
            targets.shape[1],
            device=logits.device,
        ).unsqueeze(0)
        padding_mask = positions >= (normalized_lengths - 1).unsqueeze(1)
        targets = targets.masked_fill(padding_mask, IGNORE_INDEX)

        if prompt_lens is not None:
            prompt_lens = torch.as_tensor(
                prompt_lens,
                device=logits.device,
            )
            if prompt_lens.ndim == 0:
                prompt_lens = prompt_lens.expand(tokens.shape[0])
            if prompt_lens.ndim != 1 or prompt_lens.shape[0] != tokens.shape[0]:
                raise ValueError("Chatterbox T3 prompt_lens must be a scalar or have "
                                 "shape [batch].")
            if torch.is_floating_point(prompt_lens) and not torch.equal(
                    prompt_lens,
                    prompt_lens.round(),
            ):
                raise ValueError("Chatterbox T3 prompt lengths must be integers.")
            prompt_lens = prompt_lens.to(dtype=torch.long)
            if bool((prompt_lens < 0).any()) or bool((prompt_lens > normalized_lengths - 1).any()):
                raise ValueError(
                    "Chatterbox T3 prompt lengths must be non-negative and "
                    "cannot exceed the available causal speech targets.")
            targets = targets.masked_fill(
                positions < prompt_lens.unsqueeze(1),
                IGNORE_INDEX,
            )

        supervised = targets.ne(IGNORE_INDEX)
        if not bool(supervised.any()):
            raise ValueError(f"Chatterbox T3 {name} loss has no supervised causal targets.")
        supervised_targets = targets[supervised]
        if bool((supervised_targets < 0).any()) or bool((supervised_targets >= logits.shape[-1]).any()):
            raise ValueError(
                f"Chatterbox T3 labels_{name} contain token IDs outside "
                f"[0, {logits.shape[-1] - 1}].")
        return F.cross_entropy(
            causal_logits.reshape(-1, causal_logits.shape[-1]),
            targets.reshape(-1),
            ignore_index=IGNORE_INDEX,
        )

    def training_objective(self, **batch) -> dict[str, Tensor]:
        """Return the two published T3 losses and their summed objective."""
        text_loss, speech_loss = self.loss(**batch)
        return {
            "loss": text_loss + speech_loss,
            "text_loss": text_loss,
            "speech_token_loss": speech_loss,
        }

    @torch.inference_mode()
    def inference(
        self,
        *,
        t3_cond: T3Cond,
        text_tokens: Tensor,
        initial_speech_tokens: Optional[Tensor] = None,

        # misc conditioning
        prepend_prompt_speech_tokens: Optional[Tensor] = None,

        # HF generate args
        num_return_sequences=1,
        max_new_tokens=None,
        stop_on_eos=True,
        do_sample=True,
        temperature=0.8,
        min_p=0.05,
        top_p=1.00,
        length_penalty=1.0,
        repetition_penalty=1.2,
        cfg_weight=0,
    ):
        """
        Args:
            text_tokens: a 1D (unbatched) or 2D (batched) tensor.
        """
        # Validate / sanitize inputs
        if prepend_prompt_speech_tokens is not None:
            raise NotImplementedError("prepend_prompt_speech_tokens is not implemented.")
        text_tokens = torch.atleast_2d(text_tokens).to(
            dtype=torch.long,
            device=self.device,
        )
        _ensure_BOT_EOT(text_tokens, self.hp)
        if cfg_weight > 0 and text_tokens.shape[0] == 1:
            text_tokens = text_tokens.expand(2, -1)
        expected_batch = 2 if cfg_weight > 0 else 1
        if text_tokens.shape[0] != expected_batch:
            raise ValueError(
                "Native Chatterbox inference expects one text sequence, "
                "or the conditional/unconditional pair used by CFG.")

        # Default initial speech to a single start-of-speech token
        if initial_speech_tokens is None:
            initial_speech_tokens = self.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])
        else:
            initial_speech_tokens = torch.atleast_2d(initial_speech_tokens).to(
                device=self.device, dtype=torch.long)
            if (cfg_weight > 0 and initial_speech_tokens.shape[0] == 1):
                initial_speech_tokens = initial_speech_tokens.expand(2, -1)
            if initial_speech_tokens.shape[0] != expected_batch:
                raise ValueError("initial_speech_tokens must match the inference batch.")

        # Prepare custom input embeds
        embeds, len_cond = self.prepare_input_embeds(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            speech_tokens=initial_speech_tokens,
            cfg_weight=cfg_weight,
        )

        if num_return_sequences != 1:
            raise ValueError("Native Chatterbox currently supports one return sequence.")
        if length_penalty != 1.0:
            raise ValueError("Native Chatterbox sampling does not use beam length penalties.")

        device = embeds.device

        bos_token = torch.tensor([[self.hp.start_speech_token]], dtype=torch.long, device=device)
        bos_embed = self.speech_emb(bos_token)  # shape: (B, 1, embed_dim)
        bos_embed = bos_embed + self.speech_pos_emb.get_fixed_embedding(0)

        # batch_size=2 for CFG
        bos_embed = torch.cat([bos_embed, bos_embed])

        # Combine condition and BOS token for the initial input if cfg_weight > 0
        if cfg_weight > 0:
            inputs_embeds = torch.cat([embeds, bos_embed], dim=1)
        else:
            inputs_embeds = embeds

        # Track generated token ids; start with the BOS token.
        generated_ids = bos_token.clone()
        predicted = []  # To store the predicted tokens

        # ---- Initial Forward Pass (no kv_cache yet) ----
        output = self.tfmr(
            inputs_embeds=inputs_embeds,
            past_key_values=None,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        # Initialize kv_cache with the full context.
        past = output.past_key_values

        # ---- Generation Loop using kv_cache ----
        generation_limit = (self.hp.max_speech_tokens if max_new_tokens is None else int(max_new_tokens))
        if generation_limit <= 0:
            raise ValueError("max_new_tokens must be greater than zero.")
        for i in range(generation_limit):
            logits = self.speech_head(output.last_hidden_state[:, -1, :])

            # CFG
            if cfg_weight > 0.0:
                logits_cond = logits[0:1]
                logits_uncond = logits[1:2]
                logits = logits_cond + cfg_weight * (logits_cond - logits_uncond)

            logits = process_logits(
                logits,
                generated_ids,
                do_sample=do_sample,
                temperature=temperature,
                min_p=min_p,
                top_p=top_p,
                repetition_penalty=float(repetition_penalty),
            )
            if do_sample:
                probabilities = torch.softmax(logits.float(), dim=-1)
                next_token = torch.multinomial(
                    probabilities,
                    num_samples=1,
                )
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            predicted.append(next_token)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            # Check for EOS token.
            if stop_on_eos and bool(next_token.eq(self.hp.stop_speech_token).all()):
                break

            # Get embedding for the new token.
            next_token_embed = self.speech_emb(next_token)
            next_token_embed = next_token_embed + self.speech_pos_emb.get_fixed_embedding(i + 1)

            #  For CFG
            if cfg_weight > 0.0:
                next_token_embed = torch.cat([next_token_embed, next_token_embed])

            # Forward pass with only the new token and the cached past.
            output = self.tfmr(
                inputs_embeds=next_token_embed,
                past_key_values=past,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False,
            )
            # Update the kv_cache.
            past = output.past_key_values

        # Concatenate all predicted tokens along the sequence dimension.
        predicted_tokens = torch.cat(predicted, dim=1)  # shape: (B, num_tokens)
        return predicted_tokens
