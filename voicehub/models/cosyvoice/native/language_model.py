"""Native Qwen2 speech-token language model for CosyVoice 3."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional
from torch.nn.utils.rnn import pad_sequence

from voicehub.models.causal_lm.native.modeling import Qwen2ForCausalLM
from voicehub.models.cosyvoice.native.configuration import CosyVoiceLanguageConfig
from voicehub.neural.cache import DynamicKVCache

IGNORE_INDEX = -100


@dataclass(frozen=True)
class CosyVoiceLanguageOutput:
    """Speech logits and source-aligned training diagnostics."""

    logits: Tensor
    loss: Tensor | None = None
    accuracy: Tensor | None = None
    labels: Tensor | None = None
    attention_mask: Tensor | None = None


class Qwen2Encoder(nn.Module):
    """Source-compatible owner of the native Qwen2 causal LM."""

    def __init__(
        self,
        config: CosyVoiceLanguageConfig,
        *,
        initialize: bool,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.model = Qwen2ForCausalLM(
            config.qwen_config(),
            initialize=initialize,
            device=device,
            dtype=dtype,
        )


def _validate_token_batch(
    name: str,
    tokens: Tensor,
    lengths: Tensor,
    *,
    vocabulary_size: int,
) -> None:
    if not isinstance(tokens, Tensor) or tokens.ndim != 2:
        raise ValueError(f"`{name}` must have shape [batch, sequence].")
    if tokens.dtype == torch.bool or tokens.is_floating_point() or tokens.is_complex():
        raise TypeError(f"`{name}` must use an integer dtype.")
    if not isinstance(lengths, Tensor) or lengths.ndim != 1:
        raise ValueError(f"`{name}_len` must have shape [batch].")
    if lengths.shape[0] != tokens.shape[0]:
        raise ValueError(f"`{name}` and `{name}_len` batch sizes differ.")
    if (lengths <= 0).any() or (lengths > tokens.shape[1]).any():
        raise ValueError(f"`{name}_len` contains an invalid sequence length.")
    valid = torch.arange(tokens.shape[1], device=tokens.device)[None] < lengths[:, None]
    values = tokens[valid]
    if values.numel() and ((values < 0).any() or (values >= vocabulary_size).any()):
        raise ValueError(f"`{name}` contains an out-of-vocabulary token.")


class CosyVoiceLanguageModel(nn.Module):
    """CosyVoice 3's Qwen2 backbone with a dedicated speech vocabulary.

    Text tokens use the Qwen input embedding. Speech/control tokens use a
    separate table and projection. The training sequence and label placement
    follow the author graph: ``SOS, instruction, text, TASK, speech`` predicts
    ``speech, EOS`` while every conditioning position is ignored.
    """

    def __init__(
        self,
        config: CosyVoiceLanguageConfig,
        *,
        initialize: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if not isinstance(config, CosyVoiceLanguageConfig):
            raise TypeError("`config` must be CosyVoiceLanguageConfig.")
        self.config = config
        self.llm = Qwen2Encoder(
            config,
            initialize=initialize,
            device=device,
            dtype=dtype,
        )
        self.speech_embedding = nn.Embedding(
            config.output_vocab_size,
            config.hidden_size,
            device=device,
            dtype=dtype,
        )
        self.llm_decoder = nn.Linear(
            config.hidden_size,
            config.output_vocab_size,
            bias=False,
            device=device,
            dtype=dtype,
        )
        if initialize:
            nn.init.normal_(
                self.speech_embedding.weight,
                mean=0.0,
                std=config.initializer_range,
            )
            nn.init.normal_(
                self.llm_decoder.weight,
                mean=0.0,
                std=config.initializer_range,
            )

    @property
    def stop_token_ids(self) -> tuple[int, ...]:
        return tuple(range(
            self.config.speech_vocab_size,
            self.config.output_vocab_size,
        ))

    def _build_training_sequence(
        self,
        text_tokens: Tensor,
        text_lengths: Tensor,
        speech_tokens: Tensor,
        speech_lengths: Tensor,
        instruction_tokens: Tensor | None,
        instruction_lengths: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch_size = text_tokens.shape[0]
        text_embeddings = self.llm.model.model.embed_tokens(text_tokens)
        speech_embeddings = self.speech_embedding(speech_tokens)
        if instruction_tokens is None:
            instruction_tokens = text_tokens.new_zeros((batch_size, 0))
            instruction_lengths = text_lengths.new_zeros((batch_size, ))
            instruction_embeddings = text_embeddings.new_zeros((batch_size, 0, self.config.hidden_size))
        else:
            if instruction_lengths is None:
                raise ValueError("`instruction_lengths` is required with instruction tokens.")
            _validate_token_batch(
                "instruction_tokens",
                instruction_tokens,
                instruction_lengths,
                vocabulary_size=self.config.text_vocab_size,
            )
            instruction_embeddings = self.llm.model.model.embed_tokens(instruction_tokens)

        sos = self.speech_embedding.weight[self.config.sos_token_id]
        task = self.speech_embedding.weight[self.config.task_token_id]
        sequences: list[Tensor] = []
        labels: list[Tensor] = []
        for index in range(batch_size):
            text_length = int(text_lengths[index].item())
            speech_length = int(speech_lengths[index].item())
            instruction_length = int(instruction_lengths[index].item())
            sequences.append(
                torch.cat(
                    (
                        sos[None],
                        instruction_embeddings[index, :instruction_length],
                        text_embeddings[index, :text_length],
                        task[None],
                        speech_embeddings[index, :speech_length],
                    ),
                    dim=0,
                ))
            labels.append(
                torch.cat((
                    speech_tokens.new_full(
                        (1 + instruction_length + text_length, ),
                        IGNORE_INDEX,
                    ),
                    speech_tokens[index, :speech_length],
                    speech_tokens.new_tensor([self.config.eos_token_id]),
                )))
        lengths = text_lengths.new_tensor(
            [sequence.shape[0] for sequence in sequences],
            dtype=torch.long,
        )
        inputs = pad_sequence(sequences, batch_first=True)
        targets = pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )
        attention_mask = (torch.arange(inputs.shape[1], device=inputs.device)[None] < lengths[:, None])
        return inputs, targets, attention_mask

    def forward(
        self,
        *,
        text_tokens: Tensor,
        text_lengths: Tensor,
        speech_tokens: Tensor,
        speech_lengths: Tensor,
        instruction_tokens: Tensor | None = None,
        instruction_lengths: Tensor | None = None,
    ) -> CosyVoiceLanguageOutput:
        _validate_token_batch(
            "text_tokens",
            text_tokens,
            text_lengths,
            vocabulary_size=self.config.text_vocab_size,
        )
        _validate_token_batch(
            "speech_tokens",
            speech_tokens,
            speech_lengths,
            vocabulary_size=self.config.speech_vocab_size,
        )
        if text_tokens.shape[0] != speech_tokens.shape[0]:
            raise ValueError("Text and speech token batches differ.")
        inputs, labels, attention_mask = self._build_training_sequence(
            text_tokens,
            text_lengths,
            speech_tokens,
            speech_lengths,
            instruction_tokens,
            instruction_lengths,
        )
        hidden = self.llm.model.model(
            inputs_embeds=inputs,
            attention_mask=attention_mask,
            use_cache=False,
        ).last_hidden_state
        logits = self.llm_decoder(hidden).float()
        flat_labels = labels.reshape(-1)
        loss = functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            flat_labels,
            ignore_index=IGNORE_INDEX,
            label_smoothing=self.config.label_smoothing,
            reduction="sum",
        )
        count = (flat_labels != IGNORE_INDEX).sum().clamp_min(1)
        if self.config.length_normalized_loss:
            loss = loss / count
        else:
            loss = loss / labels.shape[0]
        with torch.no_grad():
            predicted = logits.argmax(dim=-1)
            valid = labels != IGNORE_INDEX
            accuracy = ((predicted == labels) & valid).sum().float() / valid.sum().clamp_min(1)
        return CosyVoiceLanguageOutput(
            logits=logits,
            loss=loss,
            accuracy=accuracy,
            labels=labels,
            attention_mask=attention_mask,
        )

    @torch.inference_mode()
    def generate(
        self,
        text_tokens: Tensor,
        *,
        instruction_tokens: Tensor | None = None,
        prompt_speech_tokens: Tensor | None = None,
        min_new_tokens: int = 0,
        max_new_tokens: int = 1_024,
        top_k: int = 25,
        top_p: float = 0.8,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Autoregressively generate speech IDs through VoiceHub's Qwen
        graph."""
        if not isinstance(text_tokens, Tensor) or text_tokens.ndim != 2:
            raise ValueError("`text_tokens` must have shape [batch, sequence].")
        if text_tokens.shape[0] != 1 or text_tokens.shape[1] == 0:
            raise ValueError("CosyVoice generation currently requires one non-empty prompt.")
        if max_new_tokens <= 0 or min_new_tokens < 0 or min_new_tokens > max_new_tokens:
            raise ValueError("Invalid generation length bounds.")
        if top_k <= 0 or not 0 < top_p <= 1 or temperature <= 0:
            raise ValueError("Sampling controls must be positive and `top_p` at most one.")
        pieces = [
            self.speech_embedding.weight[self.config.sos_token_id].reshape(1, 1, -1),
        ]
        if instruction_tokens is not None:
            pieces.append(self.llm.model.model.embed_tokens(instruction_tokens))
        pieces.extend((
            self.llm.model.model.embed_tokens(text_tokens),
            self.speech_embedding.weight[self.config.task_token_id].reshape(1, 1, -1),
        ))
        if prompt_speech_tokens is not None and prompt_speech_tokens.numel():
            pieces.append(self.speech_embedding(prompt_speech_tokens))
        step_input = torch.cat(pieces, dim=1)
        cache: DynamicKVCache | None = None
        generated: list[Tensor] = []
        for step in range(max_new_tokens):
            hidden_output = self.llm.model.model(
                inputs_embeds=step_input,
                attention_mask=None,
                past_key_values=cache,
                use_cache=True,
            )
            cache = hidden_output.past_key_values
            logits = self.llm_decoder(hidden_output.last_hidden_state[:, -1]).float()
            logits = logits / temperature
            if step < min_new_tokens:
                logits[:, self.config.speech_vocab_size:] = -torch.inf
            top_count = min(top_k, logits.shape[-1])
            top_values, top_indices = torch.topk(logits, top_count, dim=-1)
            probabilities = functional.softmax(top_values, dim=-1)
            ordered_probabilities, order = torch.sort(
                probabilities,
                dim=-1,
                descending=True,
            )
            cumulative = ordered_probabilities.cumsum(dim=-1)
            remove = cumulative - ordered_probabilities >= top_p
            ordered_probabilities = ordered_probabilities.masked_fill(remove, 0)
            ordered_probabilities = ordered_probabilities / ordered_probabilities.sum(
                dim=-1,
                keepdim=True,
            )
            sampled_order = torch.multinomial(
                ordered_probabilities,
                1,
                generator=generator,
            )
            sampled = top_indices.gather(
                1,
                order.gather(1, sampled_order),
            )
            token_id = int(sampled.item())
            if token_id >= self.config.speech_vocab_size:
                break
            generated.append(sampled.squeeze(0))
            step_input = self.speech_embedding(sampled)
        if not generated:
            return text_tokens.new_empty((1, 0))
        return torch.stack(generated, dim=1)


__all__ = [
    "CosyVoiceLanguageModel",
    "CosyVoiceLanguageOutput",
    "IGNORE_INDEX",
]
