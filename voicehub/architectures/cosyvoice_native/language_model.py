"""Native Qwen2 speech-token language model for CosyVoice 3."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional
from torch.nn.utils.rnn import pad_sequence

from voicehub.architectures.causal_lm.modeling import Qwen2ForCausalLM
from voicehub.architectures.cosyvoice_native.configuration import CosyVoiceLanguageConfig
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


def nucleus_keep_count(ordered_probabilities: Tensor, *, top_p: float, top_k: int) -> int:
    """Length of the source nucleus prefix for descending probabilities.

    The source keeps the next candidate while the float32 running sum of the
    already-kept full-vocabulary probabilities is below ``top_p`` and fewer
    than ``top_k`` are kept. It is not top-p over a top-k-renormalized
    distribution.
    """
    head = ordered_probabilities.detach().to(device="cpu", dtype=torch.float32).flatten()
    cumulative = 0.0
    kept = 0
    for value in head:
        if not (cumulative < top_p and kept < top_k):
            break
        cumulative = cumulative + value
        kept += 1
    return kept


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

    @staticmethod
    def _nucleus_sample(
        scores: Tensor,
        *,
        top_k: int,
        top_p: float,
        generator: torch.Generator | None,
    ) -> int:
        """Source ``nucleus_sampling`` over one row of log-probabilities.

        The kept prefix is chosen on the *full-vocabulary* distribution
        (not after top-k renormalization) with the source's sequential
        float32 accumulation, and the draw uses ``multinomial`` on the kept
        probabilities in descending stable order so a generator seeded like
        the source consumes randomness identically.
        """
        probabilities = scores.softmax(dim=0)
        ordered, order = probabilities.sort(descending=True, stable=True)
        kept = nucleus_keep_count(ordered[:top_k], top_p=top_p, top_k=top_k)
        sampled = ordered[:kept].multinomial(
            1,
            replacement=True,
            generator=generator,
        )
        return int(order[sampled].item())

    def _sample_token(
        self,
        scores: Tensor,
        decoded: list[int],
        *,
        top_k: int,
        top_p: float,
        repetition_window: int,
        repetition_threshold: float,
        generator: torch.Generator | None,
    ) -> int:
        """Source repetition-aware sampling (RAS) for one decoding step."""
        token_id = self._nucleus_sample(
            scores,
            top_k=top_k,
            top_p=top_p,
            generator=generator,
        )
        if repetition_window > 0:
            window = decoded[-repetition_window:]
            repeats = sum(1 for value in window if value == token_id)
            if repeats >= repetition_window * repetition_threshold:
                scores = scores.clone()
                scores[token_id] = -torch.inf
                token_id = int(
                    scores.softmax(dim=0).multinomial(
                        1,
                        replacement=True,
                        generator=generator,
                    ).item())
        return token_id

    @torch.inference_mode()
    def generate(
        self,
        text_tokens: Tensor,
        *,
        instruction_tokens: Tensor | None = None,
        prompt_speech_tokens: Tensor | None = None,
        min_new_tokens: int = 0,
        max_new_tokens: int | None = None,
        top_k: int = 25,
        top_p: float = 0.8,
        temperature: float = 1.0,
        repetition_window: int = 10,
        repetition_threshold: float = 0.1,
        max_token_text_ratio: float = 20.0,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Autoregressively generate speech IDs through VoiceHub's Qwen
        graph.

        Decoding follows the source ``CosyVoice3LM.inference`` recipe:
        ``SOS, instruction/prompt text, text, TASK, prompt speech`` is the
        prefix, every step samples from the full speech/control vocabulary
        with repetition-aware nucleus sampling (top-p 0.8, top-k 25, window
        10, tau 0.1), and the default length cap is ``20 x`` the number of
        synthesis text tokens.
        """
        if not isinstance(text_tokens, Tensor) or text_tokens.ndim != 2:
            raise ValueError("`text_tokens` must have shape [batch, sequence].")
        if text_tokens.shape[0] != 1 or text_tokens.shape[1] == 0:
            raise ValueError("CosyVoice generation currently requires one non-empty prompt.")
        if max_new_tokens is None:
            max_new_tokens = max(1, int(text_tokens.shape[1] * max_token_text_ratio))
        if max_new_tokens <= 0 or min_new_tokens < 0 or min_new_tokens > max_new_tokens:
            raise ValueError("Invalid generation length bounds.")
        if top_k <= 0 or not 0 < top_p <= 1 or temperature <= 0:
            raise ValueError("Sampling controls must be positive and `top_p` at most one.")
        if repetition_window < 0 or repetition_threshold < 0:
            raise ValueError("Repetition-aware sampling controls must be non-negative.")
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
        generated: list[int] = []
        for step in range(max_new_tokens):
            hidden_output = self.llm.model.model(
                inputs_embeds=step_input,
                attention_mask=None,
                past_key_values=cache,
                use_cache=True,
            )
            cache = hidden_output.past_key_values
            logits = self.llm_decoder(hidden_output.last_hidden_state[:, -1]).float()
            if temperature != 1.0:
                logits = logits / temperature
            scores = logits.log_softmax(dim=-1)[0]
            if step < min_new_tokens:
                scores[self.config.speech_vocab_size:] = -torch.inf
            token_id = self._sample_token(
                scores,
                generated,
                top_k=top_k,
                top_p=top_p,
                repetition_window=repetition_window,
                repetition_threshold=repetition_threshold,
                generator=generator,
            )
            if token_id >= self.config.speech_vocab_size:
                break
            generated.append(token_id)
            step_input = self.speech_embedding.weight[token_id].reshape(1, 1, -1)
        if not generated:
            return text_tokens.new_empty((1, 0))
        return torch.tensor(
            [generated],
            dtype=torch.long,
            device=text_tokens.device,
        )


__all__ = [
    "CosyVoiceLanguageModel",
    "CosyVoiceLanguageOutput",
    "IGNORE_INDEX",
    "nucleus_keep_count",
]
